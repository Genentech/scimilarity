import json
import operator
import os
import time
from collections import defaultdict
from typing import Optional, Union, List, Set, Tuple

import anndata
import hnswlib
import numpy as np
import pandas as pd
import pegasusio as pgio
from tqdm import tqdm

from scimilarity.cell_embedding import CellEmbedding
from scimilarity.ontologies import import_cell_ontology, get_id_mapper
from scimilarity.utils import check_dataset, lognorm_counts, align_dataset
from scimilarity.zarr_dataset import ZarrDataset


class CellAnnotation(CellEmbedding):
    """A class that annotates cells using a cell embedding and then kNN search."""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        parameters: Optional[dict] = None,
        filenames: Optional[dict] = None,
        residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        model_path: str
            Path to the directory containing model files.
        use_gpu: bool, default: False
            Use GPU instead of CPU.
        parameters: dict, optional, default: None
            Use a dictionary of custom model parameters instead of infering from model files.
        filenames: dict, optional, default: None
            Use a dictionary of custom filenames for model files instead default.
        residual: bool, default: False
            Use residual connections.

        Examples
        --------
        >>> ca = CellAnnotation(model_path="/opt/data/model")
        """

        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            parameters=parameters,
            filenames=filenames,
            residual=residual,
        )

        if filenames is None:
            filenames = {}

        self.filenames["knn"] = os.path.join(
            model_path, filenames.get("knn", "labelled_kNN.bin")
        )
        self.filenames["celltype_labels"] = os.path.join(
            model_path, filenames.get("celltype_labels", "reference_labels.tsv")
        )

        # get knn
        self.load_knn_index(self.filenames["knn"])

        # get int2label
        with open(self.filenames["celltype_labels"], "r") as fh:
            self.idx2label = {i: line.strip() for i, line in enumerate(fh)}

        self.safelist = None
        self.blocklist = None

    def build_kNN(
        self,
        input_data: Union[anndata.AnnData, pgio.MultimodalData, pgio.UnimodalData, str],
        knn_filename: str = "labelled_kNN.bin",
        celltype_labels_filename: str = "reference_labels.tsv",
        obs_field: str = "celltype_name",
        ef_construction: int = 1000,
        M: int = 80,
        target_labels: Optional[List[str]] = None,
    ):
        """Build and save a kNN index from a h5ad data file or directory of aligned.zarr stores.

        Parameters
        ----------
        input_data: Union[anndata.AnnData, pegasusio.MultimodalData, pegasusio.UnimodalData, str],
            If a string, the filename of h5ad data file or directory containing zarr stores.
            Otherwise, the annotated data matrix with rows for cells and columns for genes.
        knn_filename: str, default: "labelled_kNN.bin"
            Filename of the kNN index.
        celltype_labels_filename: str, default: "reference_labels.tsv"
            Filename of the cell type reference labels.
        obs_field: str, default: "celltype_name"
            The obs key name of celltype labels.
        ef_construction: int, default: 1000
            The size of the dynamic list for the nearest neighbors.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        M: int, default: 80
            The number of bi-directional links created for every new element during construction.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        target_labels: Optional[List[str]], default: None
            Optional list of cell type names to filter the data.

        Examples
        --------
        >>> ca.build_knn(filename="/opt/data/train/train.h5ad")
        """

        if isinstance(input_data, str) and os.path.isdir(input_data):
            data_list = [
                f for f in os.listdir(input_data) if f.endswith(".aligned.zarr")
            ]
            embeddings_list = []
            labels = []
            for filename in tqdm(data_list):
                dataset = ZarrDataset(os.path.join(input_data, filename))
                obs = pd.DataFrame({obs_field: dataset.get_obs(obs_field)})
                obs.index = obs.index.astype(str)
                data = anndata.AnnData(
                    X=dataset.X_copy,
                    obs=obs,
                    var=pd.DataFrame(index=dataset.var_index),
                    dtype=np.float32,
                )

                if target_labels is not None:
                    data = data[data.obs["celltype_name"].isin(target_labels)].copy()
                if len(data.obs) == 0:
                    continue

                embeddings_list.append(
                    self.get_embeddings(align_dataset(data, self.gene_order).X)
                )
                labels.extend(data.obs[obs_field].tolist())
            embeddings = np.concatenate(embeddings_list)
        else:
            if isinstance(input_data, str) and os.path.isfile(input_data):
                data = pgio.read_input(input_data)
            else:
                data = input_data

            if target_labels is not None:
                data = data[data.obs["celltype_name"].isin(target_labels)].copy()

            name2id = {
                value: key
                for key, value in get_id_mapper(import_cell_ontology()).items()
            }
            valid_terms_idx = data.obs[obs_field].isin(name2id.keys())
            if valid_terms_idx.any():
                data = data[valid_terms_idx].copy()
            else:
                raise RuntimeError("No celltype labels have valid ontology cell ids.")

            embeddings = self.get_embeddings(align_dataset(data, self.gene_order).X)
            labels = data.obs[obs_field].tolist()

        # save knn
        n_cells, n_dims = embeddings.shape
        self.knn = hnswlib.Index(space="cosine", dim=n_dims)
        self.knn.init_index(max_elements=n_cells, ef_construction=ef_construction, M=M)
        self.knn.set_ef(ef_construction)
        self.knn.add_items(embeddings, range(len(embeddings)))

        knn_fullpath = os.path.join(self.model_path, knn_filename)
        if os.path.isfile(knn_fullpath):  # backup existing
            os.rename(knn_fullpath, knn_fullpath + ".bak")
        self.knn.save_index(knn_fullpath)

        # save labels
        celltype_labels_fullpath = os.path.join(
            self.model_path, celltype_labels_filename
        )
        if os.path.isfile(celltype_labels_fullpath):  # backup existing
            os.rename(
                celltype_labels_fullpath,
                celltype_labels_fullpath + ".bak",
            )
        with open(celltype_labels_fullpath, "w") as f:
            f.write("\n".join(labels))

        # load new int2label
        with open(celltype_labels_fullpath, "r") as fh:
            self.idx2label = {i: line.strip() for i, line in enumerate(fh)}

    def reset_kNN(self):
        """Reset the kNN such that nothing is marked deleted.

        Examples
        --------
        >>> ca.reset_kNN()
        """

        # hnswlib does not have a marked status, so we need to unmark all
        for i in self.idx2label:
            try:  # throws an expection if not already marked
                self.knn.unmark_deleted(i)
            except:
                pass

    def blocklist_celltypes(self, labels: Union[List[str], Set[str]]):
        """Blocklist celltypes.

        Parameters
        ----------
        labels: List[str], Set[str]
            A list or set containing blocklist labels.

        Notes
        -----
        Blocking a celltype will persist for this instance of the class
        and subsequent predictions will have this blocklist.
        Blocklists and safelists are mutually exclusive, setting one will clear the other.

        Examples
        --------
        >>> ca.blocklist_celltypes(["T cell"])
        """

        self.blocklist = set(labels) if isinstance(labels, list) else labels
        self.safelist = None
        self.reset_kNN()
        for i in [
            idx for idx in self.idx2label if self.idx2label[idx] in self.blocklist
        ]:
            self.knn.mark_deleted(i)

    def safelist_celltypes(self, labels: Union[List[str], Set[str]]):
        """Safelist celltypes.

        Parameters
        ----------
        labels: List[str], Set[str]
            A list or set containing safelist labels.

        Notes
        -----
        Safelisting a celltype will persist for this instance of the class
        and subsequent predictions will have this safelist.
        Blocklists and safelists are mutually exclusive, setting one will clear the other.

        Examples
        --------
        >>> ca.safelist_celltypes(["CD4-positive, alpha-beta T cell"])
        """

        self.blocklist = None
        self.safelist = set(labels) if isinstance(labels, list) else labels
        for i in range(len(self.idx2label)):  # mark all
            try:  # throws an exception if already marked
                self.knn.mark_deleted(i)
            except:
                pass
        for i in [
            idx for idx in self.idx2label if self.idx2label[idx] in self.safelist
        ]:
            self.knn.unmark_deleted(i)

    def get_predictions_kNN(
        self,
        embeddings: np.ndarray,
        k: int = 50,
        ef: int = 100,
        weighting: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Get predictions from kNN search results.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a numpy array.
        k: int, default: 50
            The number of nearest neighbors.
        ef: int, default: 100
            The size of the dynamic list for the nearest neighbors.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        weighting: bool, default: False
            Use distance weighting when getting the consensus prediction.

        Returns
        -------
        predictions: pandas.Series
            A pandas series containing celltype label predictions.
        nn_idxs: numpy.ndarray
            A 2D numpy array of nearest neighbor indices [num_cells x k].
        nn_dists: numpy.ndarray
            A 2D numpy array of nearest neighbor distances [num_cells x k].
        stats: pandas.DataFrame
            Prediction statistics columns:
            "hits" is a json string with the count for every class in k cells.
            "min_dist" is the minimum distance.
            "max_dist" is the maximum distance
            "vs2nd" is sum of best / (best + 2nd best).
            "vsAll" is sum of best / (all hits).
            "hits_weighted" is a json string with the weighted count for every class in k cells.
            "vs2nd_weighted" is weighted sum of best / (best + 2nd best).
            "vsAll_weighted" is weighted sum of best / (all hits).

        Examples
        --------
        >>> ca = CellAnnotation(model_path="/opt/data/model")
        >>> embedding = ca.get_embeddings(align_dataset(data, ca.gene_order).X)
        >>> predictions, nn_idxs, nn_dists, nn_stats = ca.get_predictions_kNN(embeddings)
        """

        start_time = time.time()
        nn_idxs, nn_dists = self.get_nearest_neighbors(
            embeddings=embeddings, k=k, ef=ef
        )
        end_time = time.time()
        print(
            f"Get nearest neighbors finished in: {float(end_time - start_time) / 60} min"
        )
        stats = {
            "hits": [],
            "hits_weighted": [],
            "min_dist": [],
            "max_dist": [],
            "vs2nd": [],
            "vsAll": [],
            "vs2nd_weighted": [],
            "vsAll_weighted": [],
        }
        if k == 1:
            predictions = pd.Series(nn_idxs.flatten()).map(self.idx2label)
        else:
            predictions = []
            for nns, d_nns in tqdm(zip(nn_idxs, nn_dists), total=nn_idxs.shape[0]):
                # count celltype in nearest neighbors (optionally with distance weights)
                celltype = defaultdict(float)
                celltype_weighted = defaultdict(float)
                for neighbor, dist in zip(nns, d_nns):
                    celltype[self.idx2label[neighbor]] += 1
                    celltype_weighted[self.idx2label[neighbor]] += 1 / dist
                # predict based on consensus max occurrence
                if weighting:
                    predictions.append(
                        max(celltype_weighted.items(), key=operator.itemgetter(1))[0]
                    )
                else:
                    predictions.append(
                        max(celltype.items(), key=operator.itemgetter(1))[0]
                    )
                # compute prediction stats
                stats["hits"].append(json.dumps(celltype))
                stats["hits_weighted"].append(json.dumps(celltype_weighted))
                stats["min_dist"].append(np.min(d_nns))
                stats["max_dist"].append(np.max(d_nns))

                hits = sorted(celltype.values(), reverse=True)
                hits_weighted = sorted(celltype_weighted.values(), reverse=True)
                if len(hits) > 1:
                    stats["vs2nd"].append(hits[0] / (hits[0] + hits[1]))
                    stats["vsAll"].append(hits[0] / sum(hits))
                    stats["vs2nd_weighted"].append(
                        hits_weighted[0] / (hits_weighted[0] + hits_weighted[1])
                    )
                    stats["vsAll_weighted"].append(
                        hits_weighted[0] / sum(hits_weighted)
                    )
                else:
                    stats["vs2nd"].append(1.0)
                    stats["vsAll"].append(1.0)
                    stats["vs2nd_weighted"].append(1.0)
                    stats["vsAll_weighted"].append(1.0)
        return (
            pd.Series(predictions),
            nn_idxs,
            nn_dists,
            pd.DataFrame(stats),
        )

    def annotate_dataset(
        self,
        dataset: Union[anndata.AnnData, pgio.MultimodalData, pgio.UnimodalData, str],
        return_type: Optional[str] = None,
        skip_preprocessing: bool = False,
    ) -> Union[anndata.AnnData, pgio.UnimodalData]:
        """Read a dataset, check validity, preprocess, and then annotate with celltype predictions.

        Parameters
        ----------
        dataset: Union[pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData, str]
            If a string, the filename of the h5ad file.
            Otherwise, the annotated data matrix with rows for cells and columns for genes.
        return_type: {"AnnData", "UnimodalData"}, optional
            Data return type string. If None, then it will return the same type as the input dataset.
            If a string was given for the dataset, defaults to UnimodalData as the return type.
        skip_preprocessing: bool, default: False
            Whether to skip preprocessing steps.

        Returns
        -------
        Union["AnnData", "UnimodalData"]
            A data object where the normalized data is in matrix/layer "lognorm",
            celltype predictions are in obs["celltype_hint"],
            and embeddings are in obs["X_triplet"].

        Examples
        --------
        >>> ca = CellAnnotation(model_path="/opt/data/model")
        >>> data = annotate_dataset("/opt/individual_anndatas/GSE124898/GSM3558026/GSM3558026.h5ad")
        """

        valid_return_types = {"AnnData", "UnimodalData"}
        if return_type is not None and return_type not in valid_return_types:
            raise ValueError(
                f"Unknown return_type {return_type}. Options are {valid_return_types}."
            )

        if isinstance(dataset, str):
            data = pgio.read_input(dataset)
            if return_type is None:
                return_type = "UnimodalData"
        else:
            data = dataset

        if isinstance(data, anndata.AnnData):
            return_type = "AnnData"
        else:
            return_type = "UnimodalData"

        if skip_preprocessing:
            normalized_data = data
        else:
            check_dataset(data, self.gene_order, gene_overlap_threshold=10000)
            normalized_data = lognorm_counts(data)

        embeddings = self.get_embeddings(
            align_dataset(normalized_data, self.gene_order).X
        )
        normalized_data.obsm["X_triplet"] = embeddings

        predictions, _, _, nn_stats = self.get_predictions_kNN(embeddings)
        normalized_data.obs["celltype_hint"] = predictions.values
        normalized_data.obs["min_dist"] = nn_stats["min_dist"].values
        normalized_data.obs["celltype_hits"] = nn_stats["hits"].values
        normalized_data.obs["celltype_hits_weighted"] = nn_stats["hits_weighted"].values
        normalized_data.obs["celltype_hint_stat"] = nn_stats["vsAll"].values
        normalized_data.obs["celltype_hint_weighted_stat"] = nn_stats[
            "vsAll_weighted"
        ].values

        if return_type == "AnnData" and not isinstance(dataset, anndata.AnnData):
            return normalized_data.to_anndata()
        return normalized_data
