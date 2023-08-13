import os
from typing import Dict, List, Optional, Tuple, Union, Set

import anndata
import numpy as np
import pandas as pd
import pegasusio as pgio
import tiledb

from scimilarity.cell_embedding import CellEmbedding
from scimilarity.utils import get_cluster_centroids
from scimilarity.visualizations import aggregate_counts, assign_size, circ_dict2data, draw_circles


class CellQuery(CellEmbedding):
    """A class that searches for similar cells using a cell embedding and then a kNN search."""

    def __init__(
        self,
        model_path: str,
        cellsearch_path: str,
        use_gpu: bool = False,
        parameters: Optional[dict] = None,
        filenames: Optional[dict] = None,
        metadata_tiledb_uri: str = "cell_metadata",
        embedding_tiledb_uri: str = "cell_embedding",
        residual: bool = False,
        load_knn: bool = True,
    ):
        """Constructor.

        Parameters
        ----------
        model_path: str
            Path to the directory containing model files.
        cellsearch_path: str
            Path to the directory containing cell search files.
        use_gpu: bool, default: False
            Use GPU instead of CPU.
        parameters: dict, optional, default: None
            Use a dictionary of custom model parameters instead of infering from model files.
        filenames: dict, optional, default: None
            Use a dictionary of custom filenames for model files instead default.
        metadata_tiledb_uri: str, default: "cell_metadata"
            Relative path to the directory containing the tiledb cell metadata storage.
        embedding_tiledb_uri: str, default: "cell_embedding"
            Relative path to the directory containing the tiledb cell embedding storage.
        residual: bool, default: False
            Use residual connections.
        load_knn: bool, default: True
            Load the knn index. Set to False if knn search is not needed.

        Examples
        --------
        >>> cq = CellQuery(model_path="/opt/data/model")
        """

        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            parameters=parameters,
            filenames=filenames,
            residual=residual,
        )
        self.cellsearch_path = cellsearch_path

        if filenames is None:
            filenames = {}

        self.filenames["knn"] = os.path.join(
            cellsearch_path, filenames.get("knn", "full_kNN.bin")
        )
        self.filenames["cell_metadata"] = os.path.join(
            cellsearch_path, filenames.get("cell_metadata", "full_kNN_meta.csv")
        )

        # get knn
        if load_knn:
            self.load_knn_index(self.filenames["knn"])
        self.block_list = set()

        # get cell metadata: create tiledb storage if it does not exist
        # NOTE: process for creating this file is not hardened, no guarantee index column is unique
        metadata_tiledb_uri = os.path.join(cellsearch_path, metadata_tiledb_uri)
        if not os.path.isdir(metadata_tiledb_uri):
            print(f"Configuring tiledb dataframe: {metadata_tiledb_uri}")
            cell_metadata = (
                pd.read_csv(
                    self.filenames["cell_metadata"],
                    header=0,
                )
                .fillna("NA")
                .astype(str)
                .reset_index(drop=True)
            )
            cell_metadata = cell_metadata.rename(columns={"Unnamed: 0": "index"})
            convert_dict = {
                "index": int,
                "nn_dist": float,
                "fm_signature_score": float,
                "total_counts": float,
                "n_genes_by_counts": float,
                "total_counts_mt": float,
                "pct_counts_mt": float,
            }
            cell_metadata = cell_metadata.astype(convert_dict)
            tiledb.from_pandas(metadata_tiledb_uri, cell_metadata)
        self.cell_metadata = tiledb.open_dataframe(metadata_tiledb_uri)

        # get cell embeddings: create tiledb storage if it does not exist
        embedding_tiledb_uri = os.path.join(cellsearch_path, embedding_tiledb_uri)
        if not os.path.isdir(embedding_tiledb_uri):
            if os.path.isfile(os.path.join(cellsearch_path, "schub_ood_embedding.npy")):
                npy_list = [
                    "ood_embedding.npy",
                    "schub_ood_embedding.npy",
                    "train_embedding.npy",
                    "test_embedding.npy",
                ]
            else:
                npy_list = [
                    "ood_embedding.npy",
                    "train_embedding.npy",
                    "test_embedding.npy",
                ]
            data_list = [os.path.join(cellsearch_path, f) for f in npy_list]
            self.create_tiledb_array(embedding_tiledb_uri, data_list)
            self.optimize_tiledb_array(embedding_tiledb_uri)
        self.cell_embedding = tiledb.open(embedding_tiledb_uri)

        self.study_sample_cells = self.cell_metadata.groupby(["study", "sample"]).size()
        self.study_cells = self.cell_metadata.groupby("study").size()
        self.study_sample_index = self.cell_metadata.groupby(
            ["study", "sample", "train_type"]
        )["index"].min()
        self.study_index = self.cell_metadata.groupby(["study", "train_type"])[
            "index"
        ].min()

    def create_tiledb_array(
        self, tiledb_array_uri: str, data_list: List[str], batch_size: int = 10000
    ):
        """Create TileDB Array

        Parameters
        ----------
        tiledb_array_uri: str
            URI for the TileDB array.
        data_list: List[str]
            List of data values.
        batch_size: int, default: 10000
            Batch size for the tiles.
        """
        print(f"Configuring tiledb array: {tiledb_array_uri}")

        xdimtype = np.int32
        ydimtype = np.int32
        value_type = np.float32

        xdim = tiledb.Dim(
            name="x",
            domain=(0, self.cell_metadata.shape[0] - 1),
            tile=batch_size,
            dtype=xdimtype,
        )
        ydim = tiledb.Dim(
            name="y",
            domain=(0, self.latent_dim - 1),
            tile=self.latent_dim,
            dtype=ydimtype,
        )
        dom = tiledb.Domain(xdim, ydim)

        attr = tiledb.Attr(
            name="vals",
            dtype=value_type,
            filters=tiledb.FilterList([tiledb.GzipFilter()]),
        )

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            cell_order="row-major",
            tile_order="row-major",
            attrs=[attr],
        )
        tiledb.Array.create(tiledb_array_uri, schema)

        tdbfile = tiledb.open(tiledb_array_uri, "w")
        previous_shape = None
        for f in data_list:
            if previous_shape is None:
                paging_idx = 0
            else:
                paging_idx += previous_shape[0]

            arr = np.load(f)
            previous_shape = arr.shape

            tbd_slice = slice(paging_idx, paging_idx + arr.shape[0])
            tdbfile[tbd_slice, 0 : self.latent_dim] = arr
        tdbfile.close()

    def optimize_tiledb_array(self, tiledb_array_uri: str, verbose: bool = True):
        """Optimize TileDB Array

        Parameters
        ----------
        tiledb_array_uri: str
            URI for the TileDB array.
        verbose: bool
            Boolean indicating whether to use verbose printing.
        """
        if verbose:
            print(f"Optimizing {tiledb_array_uri}")

        frags = tiledb.array_fragments(tiledb_array_uri)
        if verbose:
            print("Fragments before consolidation: {}".format(len(frags)))

        cfg = tiledb.Config()
        cfg["sm.consolidation.step_min_frags"] = 1
        cfg["sm.consolidation.step_max_frags"] = 200
        tiledb.consolidate(tiledb_array_uri, config=cfg)
        tiledb.vacuum(tiledb_array_uri)

        frags = tiledb.array_fragments(tiledb_array_uri)
        if verbose:
            print("Fragments after consolidation: {}".format(len(frags)))

    def get_precomputed_embeddings(self, idx: List[int]) -> np.ndarray:
        """Fast get of embeddings from the cell_embedding tiledb array.

        Parameters
        ----------
        idx: List[int]
            A list of rows, which corresponds to cell indices.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array for the listed rows (cells).

        Examples
        --------
        >>> array = cq.get_tiledb_array([0, 1, 100])
        """
        return self.cell_embedding.query(attrs=["vals"], coords=True).multi_index[idx][
            "vals"
        ]

    def compile_sample_metadata(self, nn_idxs: np.ndarray) -> pd.DataFrame:
        """Compile sample metadata for nearest neighbors.

        Parameters
        ----------
        nn_idx: numpy.ndarray
            A 2D numpy arrary of nearest neighbor indices [num_cells x k].

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing sample metadata for nearest neighbors.

        Examples
        --------
        >>> embeddings = cq.get_embeddings(align_dataset(data, cq.gene_order).X)
        >>> nn_idxs, nn_dists = cq.get_nearest_neighbors(embeddings, k=50)
        >>> sample_metadata = cq.compile_sample_metadata(nn_idxs)
        """

        levels = ["tissue", "disease", "study", "sample"]
        df = pd.concat(
            [
                self.cell_metadata.loc[hits]
                .groupby(levels)
                .size()
                .reset_index(name="cells")
                for hits in nn_idxs
            ],
            axis=0,
        ).reset_index(drop=True)

        fraction = []
        for i, row in df.iterrows():
            total_cells = self.study_sample_cells.loc[(row["study"], row["sample"])]
            fraction.append(row["cells"] / total_cells)
        df["fraction"] = fraction
        return df

    def visualize_sample_metadata(
        self,
        sample_metadata: pd.DataFrame,
        fig_size: Tuple[int, int] = (10, 10),
        filename: Optional[str] = None,
    ):
        """Visualize sample metadata as circle plots for tissue and disease.

        Parameters
        ----------
        sample_metadata: pandas.DataFrame
            A pandas dataframe containing sample metadata for nearest neighbors.
        figsize: Tuple[int, int], default: (10, 10)
            Figure size, width x height
        filename: str, optional
            Filename to save the figure.

        Examples
        --------
        >>> cq.visualize_sample_metadata(sample_metadata)
        """

        levels = ["tissue", "disease"]

        circ_dict = aggregate_counts(sample_metadata, levels)
        circ_dict = assign_size(
            circ_dict, sample_metadata, levels, size_column="cells", name_column="study"
        )
        circ_data = circ_dict2data(circ_dict)
        draw_circles(circ_data, fig_size=fig_size, filename=filename)

    def groupby_studies(self, sample_metadata: pd.DataFrame) -> pd.DataFrame:
        """Performs a groupby studies operation on sample metadata.

        Parameters
        ----------
        sample_metadata: pandas.DataFrame
            A pandas dataframe containing sample metadata for nearest neighbors.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing sample metadata groupby studies with cell counts.

        Examples
        --------
        >>> cq.groupby_studies(sample_metadata)
        """

        levels = ["tissue", "disease", "study"]
        df = (
            sample_metadata[levels + ["cells"]]
            .groupby(levels)["cells"]
            .sum()
            .reset_index(name="cells")
        )
        fraction = []
        for i, row in df.iterrows():
            total_cells = self.study_cells.loc[row["study"]]
            fraction.append(row["cells"] / total_cells)
        df["fraction"] = fraction
        return df

    def annotate_cell_index(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Annotate a metadata dataframe with the cell index in sample datasets.

        Parameters
        ----------
        metadata: pandas.DataFrame
            A pandas dataframe containing columns: study, sample, and index.
            Where index is the cell query index (i.e. from cq.cell_metadata).
        aggregated: bool, default: False
            Whether the training and test datasets are aggregated.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the cell_index column which is the cell index
            per sample dataset.

        Examples
        --------
        >>> metadata = cq.annotate_cell_index(metadata)
        """
        cell_index = []
        for _, row in metadata.iterrows():
            study = row["study"]
            sample = row["sample"]

            if "train_type" not in row:
                raise RuntimeError("Required column: 'train_type'")
            train_type = row["train_type"]

            if train_type == "ood" or train_type == "schub_ood":
                index_start = self.study_sample_index.loc[study, sample, train_type]
            elif train_type == "train" or train_type == "test":
                index_start = self.study_index.loc[study, train_type]
            else:
                raise RuntimeError(f"{train_type}: Unknown train type.")

            cell_index.append(row["index"] - int(index_start))
        metadata["cell_index"] = cell_index
        return metadata

    def search(
        self,
        embeddings: np.ndarray,
        k: int = 1000,
        ef: int = None,
        max_dist: float = None,
        exclude_studies: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], pd.DataFrame]:
        """Performs a cell query search against the kNN.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a numpy array.
        k: int, default: 1000
            The number of nearest neighbors.
        ef: int, default: None
            The size of the dynamic list for the nearest neighbors. Defaults to k if None.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        max_dist: float, optional
            Assume k=1000000, then filter for cells that are within the max distance to the
            query. Overwrites the k parameter.
        exclude_studies: List[str], optional
            A list of studies to exclude from the search, given as a list of str study names.
            WARNING: If you do not use max_dist, you will potentially get less than k hits as
            the study exclusion is performed after the search.

        Returns
        -------
        nn_idxs: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor indices.
            One entry for every cell (row) in embeddings
        nn_dists: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor distances.
            One entry for every cell (row) in embeddings
        metadata: pandas.DataFrame
            A pandas dataframe containing cell metadata for nearest neighbors.

        Examples
        --------
        >>> nn_idxs, nn_dists, metadata = cq.search(embedding)
        """

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        nn_idxs, nn_dists = self.get_nearest_neighbors(embeddings, k=k, ef=ef)
        if max_dist is not None:
            new_nn_idxs = []
            new_nn_dists = []
            for row in range(nn_idxs.shape[0]):
                hits = nn_dists[row] <= max_dist
                new_nn_idxs.append(nn_idxs[row, hits])
                new_nn_dists.append(nn_dists[row, hits])
            nn_idxs = new_nn_idxs
            nn_dists = new_nn_dists
        else:
            nn_idxs = [row for row in nn_idxs]
            nn_dists = [row for row in nn_dists]

        if exclude_studies:
            study_index = self.cell_metadata["study"].values
            new_nn_idxs = []
            new_nn_dists = []
            for row in range(len(nn_idxs)):
                hits = [
                    True if study_index[x] not in exclude_studies else False
                    for x in nn_idxs[row]
                ]
                new_nn_idxs.append(nn_idxs[row][hits])
                new_nn_dists.append(nn_dists[row][hits])
            nn_idxs = new_nn_idxs
            nn_dists = new_nn_dists

        metadata = pd.concat(
            [self.cell_metadata.loc[hits].reset_index(drop=True) for hits in nn_idxs],
            axis=0,
        ).reset_index(drop=True)

        return nn_idxs, nn_dists, metadata

    def search_centroids(
        self,
        data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
        cluster_key: str,
        cluster_label: Optional[str] = None,
        k: int = 1000,
        ef: int = None,
        skip_null: bool = True,
        max_dist: float = None,
        exclude_studies: Optional[List[str]] = None,
    ) -> Tuple[
        np.ndarray,
        list,
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        pd.DataFrame,
    ]:
        """Performs a cell query search for cluster centroids against the kNN.

        Parameters
        ----------
        data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
            Annotated data matrix with rows for cells and columns for genes.
        cluster_key: str
            The obs column key that contains cluster labels.
        cluster_label: optional, str
            The cluster label of interest. If None, then get the centroids of
            all clusters, otherwise get only the centroid for the cluster
            of interest
        k: int, default: 1000
            The number of nearest neighbors.
        ef: int, default: None
            The size of the dynamic list for the nearest neighbors. Defaults to k if None.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        skip_null: bool, default: True
            Whether to skip cells with null/nan cluster labels.
        max_dist: float, optional
            Assume k=1000000, then filter for cells that are within the max distance to the
            query. Overwrites the k parameter.
        exclude_studies: List[str], optional
            A list of studies to exclude from the search, given as a list of str study names.
            WARNING: If you do not use max_dist, you will potentially get less than k hits as
            the study exclusion is performed after the search.

        Returns
        -------
        centroid_embeddings: numpy.ndarray
            A 2D numpy array of the log normalized (1e4) cluster centroid embeddings.
        cluster_idx: list
            A list of cluster labels corresponding to the order returned in centroids.
        nn_idxs: Dict[str, numpy.ndarray]
            A 2D numpy array of nearest neighbor indices [num_cells x k].
        nn_dists: Dict[str, numpy.ndarray]
            A 2D numpy array of nearest neighbor distances [num_cells x k].
        all_metadata: pandas.DataFrame
            A pandas dataframe containing cell metadata for nearest neighbors
            for all centroids.

        Examples
        --------
        >>> centroid_embeddings, cluster_idx, nn_idx, nn_dists, all_metadata = cq.search_centroids(data, "leidan")
        """

        centroids, cluster_idx = get_cluster_centroids(
            data, self.gene_order, cluster_key, cluster_label, skip_null=skip_null
        )

        centroid_embeddings = self.get_embeddings(centroids)

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        nn_idxs = {}
        nn_dists = {}
        metadata = {}
        for row in range(centroid_embeddings.shape[0]):
            (
                nn_idxs[cluster_idx[row]],
                nn_dists[cluster_idx[row]],
                metadata[cluster_idx[row]],
            ) = self.search(
                centroid_embeddings[row],
                k=k,
                ef=ef,
                max_dist=max_dist,
                exclude_studies=exclude_studies,
            )
            metadata[cluster_idx[row]]["centroid"] = cluster_idx[row]
            metadata[cluster_idx[row]]["nn_dist"] = nn_dists[cluster_idx[row]][0]

        all_metadata = pd.concat(metadata.values())
        all_metadata = all_metadata.set_index("index", drop=False)

        return (
            centroid_embeddings,
            cluster_idx,
            nn_idxs,
            nn_dists,
            all_metadata,
        )
