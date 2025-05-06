from typing import Optional, Union, List, Set, Tuple

from .cell_search_knn import CellSearchKNN


class CellAnnotation(CellSearchKNN):
    """A class that annotates cells using a cell embedding and then knn search.

    Parameters
    ----------
    model_path: str
        Path to the directory containing model files.
    use_gpu: bool, default: False
        Use GPU instead of CPU.
    filenames: dict, optional, default: None
        Use a dictionary of custom filenames for files instead default.

    Examples
    --------
    >>> ca = CellAnnotation(model_path="/opt/data/model")
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        filenames: Optional[dict] = None,
    ):
        import os

        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            knn_type="hnswlib",
        )

        self.annotation_path = os.path.join(model_path, "annotation")
        os.makedirs(self.annotation_path, exist_ok=True)

        if filenames is None:
            filenames = {}

        self.filenames["knn"] = os.path.join(
            self.annotation_path, filenames.get("knn", "labelled_kNN.bin")
        )
        self.filenames["celltype_labels"] = os.path.join(
            self.annotation_path,
            filenames.get("celltype_labels", "reference_labels.tsv"),
        )

        # get knn
        self.load_knn_index(self.filenames["knn"])

        # get int2label and int2study
        self.idx2label = {}
        self.idx2study = {}
        if self.knn is not None:
            with open(self.filenames["celltype_labels"], "r") as fh:
                for i, line in enumerate(fh):
                    token = line.strip().split("\t")
                    self.idx2label[i] = token[0]
                    if len(token) > 1:
                        self.idx2study[i] = token[1]

        self.safelist = None
        self.blocklist = None

    @property
    def classes() -> set:
        """Get the set of all viable prediction classes."""

        return set(self.label2int.keys())

    def reset_knn(self):
        """Reset the knn such that nothing is marked deleted.

        Examples
        --------
        >>> ca.reset_knn()
        """

        self.blocklist = None
        self.safelist = None

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
        Blocking a celltype will persist for this instance of the class and subsequent predictions will have this blocklist.
        Blocklists and safelists are mutually exclusive, setting one will clear the other.

        Examples
        --------
        >>> ca.blocklist_celltypes(["T cell"])
        """

        self.reset_knn()
        self.blocklist = set(labels)
        self.safelist = None

        for i, celltype_name in self.idx2label.items():
            if celltype_name in self.blocklist:
                self.knn.mark_deleted(i)  # mark blocklist

    def safelist_celltypes(self, labels: Union[List[str], Set[str]]):
        """Safelist celltypes.

        Parameters
        ----------
        labels: List[str], Set[str]
            A list or set containing safelist labels.

        Notes
        -----
        Safelisting a celltype will persist for this instance of the class and subsequent predictions will have this safelist.
        Blocklists and safelists are mutually exclusive, setting one will clear the other.

        Examples
        --------
        >>> ca.safelist_celltypes(["CD4-positive, alpha-beta T cell"])
        """

        self.blocklist = None
        self.safelist = set(labels)

        for i in self.idx2label:  # mark all
            try:  # throws an exception if already marked
                self.knn.mark_deleted(i)
            except:
                pass
        for i, celltype_name in self.idx2label.items():
            if celltype_name in self.safelist:
                self.knn.unmark_deleted(i)  # unmark safelist

    def get_predictions_knn(
        self,
        embeddings: "numpy.ndarray",
        k: int = 50,
        ef: int = 100,
        weighting: bool = False,
        disable_progress: bool = False,
    ) -> Tuple["numpy.ndarray", "numpy.ndarray", "numpy.ndarray", "pandas.DataFrame"]:
        """Get predictions from knn search results.

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
        disable_progress: bool, default: False
            Disable tqdm progress bar

        Returns
        -------
        predictions: pandas.Series
            A pandas series containing celltype label predictions.
        nn_idxs: numpy.ndarray
            A 2D numpy array of nearest neighbor indices [num_cells x k].
        nn_dists: numpy.ndarray
            A 2D numpy array of nearest neighbor distances [num_cells x k].
        stats: pandas.DataFrame
            Prediction statistics dataframe with columns:
            "hits" is a json string with the count for every class in k cells.
            "min_dist" is the minimum distance.
            "max_dist" is the maximum distance
            "vs2nd" is sum(best) / sum(best + 2nd best).
            "vsAll" is sum(best) / sum(all hits).
            "hits_weighted" is a json string with the weighted count for every class in k cells.
            "vs2nd_weighted" is weighted sum(best) / sum(best + 2nd best).
            "vsAll_weighted" is weighted sum(best) / sum(all hits).

        Examples
        --------
        >>> ca = CellAnnotation(model_path="/opt/data/model")
        >>> embeddings = ca.get_embeddings(align_dataset(data, ca.gene_order).X)
        >>> predictions, nn_idxs, nn_dists, stats = ca.get_predictions_knn(embeddings)
        """

        from collections import defaultdict
        import json
        import operator
        import numpy as np
        import pandas as pd
        import time
        from tqdm import tqdm

        start_time = time.time()
        nn_idxs, nn_dists = self.get_nearest_neighbors(
            embeddings=embeddings, k=k, ef=ef
        )
        end_time = time.time()
        if not disable_progress:
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
            for nns, d_nns in tqdm(
                zip(nn_idxs, nn_dists), total=nn_idxs.shape[0], disable=disable_progress
            ):
                # count celltype in nearest neighbors (optionally with distance weights)
                celltype = defaultdict(float)
                celltype_weighted = defaultdict(float)
                for neighbor, dist in zip(nns, d_nns):
                    celltype[self.idx2label[neighbor]] += 1.0
                    celltype_weighted[self.idx2label[neighbor]] += 1.0 / float(
                        max(dist, 1e-6)
                    )
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
                hits_weighted = [
                    max(x, 1e-6)
                    for x in sorted(celltype_weighted.values(), reverse=True)
                ]
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
        data: "anndata.AnnData",
    ) -> "anndata.AnnData":
        """Annotate dataset with celltype predictions.

        Parameters
        ----------
        data: anndata.AnnData
            The annotated data matrix with rows for cells and columns for genes.
            This function assumes the data has been log normalized (i.e. via lognorm_counts) accordingly.

        Returns
        -------
        anndata.AnnData
            A data object where:
                - celltype predictions are in obs["celltype_hint"]
                - embeddings are in obs["X_scimilarity"].

        Examples
        --------
        >>> ca = CellAnnotation(model_path="/opt/data/model")
        >>> data = annotate_dataset(data)
        """

        from .utils import align_dataset

        embeddings = self.get_embeddings(align_dataset(data, self.gene_order).X)
        data.obsm["X_scimilarity"] = embeddings

        predictions, _, _, nn_stats = self.get_predictions_knn(embeddings)
        data.obs["celltype_hint"] = predictions.values
        data.obs["min_dist"] = nn_stats["min_dist"].values
        data.obs["celltype_hits"] = nn_stats["hits"].values
        data.obs["celltype_hits_weighted"] = nn_stats["hits_weighted"].values
        data.obs["celltype_hint_stat"] = nn_stats["vsAll"].values
        data.obs["celltype_hint_weighted_stat"] = nn_stats["vsAll_weighted"].values

        return data
