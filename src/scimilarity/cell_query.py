from typing import Dict, List, Optional, Tuple, Union, Set

from .cell_search_knn import CellSearchKNN


class CellQuery(CellSearchKNN):
    """A class that searches for similar cells using a cell embedding.

    Parameters
    ----------
    model_path: str
        Path to the model directory.
    use_gpu: bool, default: False
        Use GPU instead of CPU.
    filenames: dict, optional, default: None
        Use a dictionary of custom filenames for model files instead default.
    metadata_tiledb_uri: str, default: "cell_metadata"
        Relative path to the directory containing the tiledb cell metadata storage.
    embedding_tiledb_uri: str, default: "cell_embedding"
        Relative path to the directory containing the tiledb cell embedding storage.
    knn_type: str, default: "hnswlib"
        What type of knn to use, options are ["hnswlib", "tiledb_vector_search"]
    load_knn: bool, default: True
        Load the knn index. Set to False if knn is not needed.

    Examples
    --------
    >>> cq = CellQuery(model_path="/opt/data/model")
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        filenames: Optional[dict] = None,
        metadata_tiledb_uri: str = "cell_metadata",
        embedding_tiledb_uri: str = "cell_embedding",
        knn_type: str = "hnswlib",
        load_knn: bool = True,
    ):
        import os
        import numpy as np
        import pandas as pd
        import tiledb

        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            knn_type=knn_type,
        )

        self.cellsearch_path = os.path.join(model_path, "cellsearch")
        os.makedirs(self.cellsearch_path, exist_ok=True)

        if filenames is None:
            filenames = {}

        self.filenames["knn"] = os.path.join(
            self.cellsearch_path, filenames.get("knn", "full_kNN.bin")
        )
        self.filenames["cell_metadata"] = os.path.join(
            self.cellsearch_path, filenames.get("cell_metadata", "full_kNN_meta.csv")
        )
        self.filenames["cell_embeddings"] = os.path.join(
            self.cellsearch_path,
            filenames.get("cell_embeddings", "full_kNN_embedding.npy"),
        )
        self.knn_type = knn_type

        # get knn
        if load_knn:
            self.load_knn_index(self.filenames["knn"])
        self.block_list = set()

        # get cell metadata: create tiledb storage if it does not exist
        # NOTE: process for creating this file is not hardened, no guarantee index column is unique
        metadata_tiledb_uri = os.path.join(self.cellsearch_path, metadata_tiledb_uri)
        self.cell_metadata = tiledb.open(metadata_tiledb_uri, "r").df[:]

        # cell embeddings
        self.embedding_tiledb_uri = os.path.join(
            self.cellsearch_path, embedding_tiledb_uri
        )

        self.study_sample_index = (
            self.cell_metadata.groupby(["study", "sample", "data_type"], observed=True)[
                "index"
            ]
            .min()
            .sort_values()
        )

    def get_precomputed_embeddings(
        self, idx: Union[slice, List[int]]
    ) -> "numpy.ndarray":
        """Fast get of embeddings from the cell_embedding tiledb array.

        Parameters
        ----------
        idx: slice, List[int]
            Cell indices.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array for the listed cells.

        Examples
        --------
        >>> array = cq.get_precomputed_embeddings([0, 1, 100])
        """

        from .utils import embedding_from_tiledb

        return embedding_from_tiledb(idx, self.embedding_tiledb_uri)

    def annotate_cell_index(self, metadata: "pandas.DataFrame") -> "pandas.DataFrame":
        """Annotate a metadata dataframe with the cell index in datasets at the SAMPLE level.
           The cell index is the cell number, not related to the obs.index.

        Parameters
        ----------
        metadata: pandas.DataFrame
            A pandas dataframe containing columns: study, sample, and index.
            Where index is the cell query index (i.e. from cq.cell_metadata).

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the "cell_index" column which is the cell index
            per sample dataset.

        Examples
        --------
        >>> metadata = cq.annotate_cell_index(metadata)
        """

        cell_index = []
        for i, row in metadata.iterrows():
            study = row["study"]
            sample = row["sample"]
            if "data_type" not in row:
                raise RuntimeError("Required column: 'data_type'")
            data_type = row["data_type"]
            index_start = self.study_sample_index.loc[study, sample, data_type]
            cell_index.append(row["index"] - int(index_start))
        metadata["cell_index"] = cell_index
        return metadata

    def compile_sample_metadata(
        self,
        nn_idxs: "numpy.ndarray",
        levels: list = ["study", "sample", "tissue", "disease"],
    ) -> "pandas.DataFrame":
        """Compile sample metadata for nearest neighbors.

        Parameters
        ----------
        nn_idx: numpy.ndarray
            A 2D numpy arrary of nearest neighbor indices [num_cells x k].
        levels: list, default: ["study", "sample", "tissue", "disease"]
            Levels for aggregation. Requires "study" and "sample" in order to
            calculate fraction of cells that are similar to the query in the
            relevant studies and samples.

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

        import pandas as pd

        df = pd.concat(
            [
                self.cell_metadata.loc[hits]
                .groupby(levels, observed=True)
                .size()
                .reset_index(name="cells")
                for hits in nn_idxs
            ],
            axis=0,
        ).reset_index(drop=True)

        if "study" in levels and "sample" in levels:
            study_sample_cells = self.cell_metadata.groupby(
                ["study", "sample"], observed=True
            ).size()

            fraction = []
            total = []
            for i, row in df.iterrows():
                total_cells = study_sample_cells.loc[(row["study"], row["sample"])]
                fraction.append(row["cells"] / total_cells)
                total.append(total_cells)
            df["fraction"] = fraction
            df["total"] = total
        return df

    def search_nearest(
        self,
        embeddings: "numpy.ndarray",
        k: int = 10000,
        ef: int = None,
        max_dist: Optional[float] = None,
    ) -> Tuple[List["numpy.ndarray"], List["numpy.ndarray"], "pandas.DataFrame"]:
        """Performs a nearest neighbors search against the knn.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a numpy array.
        k: int, default: 10000
            The number of nearest neighbors.
        ef: int, default: None
            The size of the dynamic list for the nearest neighbors. Defaults to k if None.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        max_dist: float, optional
            Assume k=1000000, then filter for cells that are within the max distance to the
            query. Overwrites the k parameter.

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
        >>> nn_idxs, nn_dists, metadata = cq.search_nearest(embeddings)
        """

        import pandas as pd

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

        metadata = []
        for i in range(len(nn_idxs)):
            hits = nn_idxs[i]
            df = self.cell_metadata.loc[hits].reset_index(drop=True)
            df["embedding_idx"] = i
            df["query_nn_dist"] = nn_dists[i]
            metadata.append(df)
        metadata = pd.concat(metadata).reset_index(drop=True)

        return nn_idxs, nn_dists, metadata

    def search_centroid_nearest(
        self,
        adata: "anndata.AnnData",
        centroid_key: str,
        k: int = 10000,
        ef: int = None,
        max_dist: Optional[float] = None,
        qc: bool = True,
        qc_params: dict = {"k_clusters": 10},
        random_seed: int = 4,
    ) -> Tuple[
        "numpy.ndarray",
        List["numpy.ndarray"],
        List["numpy.ndarray"],
        "pandas.DataFrame",
        dict,
    ]:
        """Performs a nearest neighbors search for a centroid constructed from marked cells.

        Parameters
        ----------
        adata: anndata.AnnData
            Annotated data matrix with rows for cells and columns for genes.
            Requires a layers["counts"].
        centroid_key: str
            The obs column key that marks cells to centroid as 1, otherwise 0.
        k: int, default: 10000
            The number of nearest neighbors.
        ef: int, default: None
            The size of the dynamic list for the nearest neighbors. Defaults to k if None.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        max_dist: float, optional
            Assume k=1000000, then filter for cells that are within the max distance to the
            query. Overwrites the k parameter.
        qc: bool, default: True
            Whether to perform QC on the query
        qc_params: dict, default: {'k_clusters': 10}
            Parameters for the QC:
            k_clusters: the number of clusters in kmeans clustering
        random_seed: int, default: 1
            Random seed for k-means clustering

        Returns
        -------
        centroid_embedding: numpy.ndarray
            A 2D numpy array of the centroid embedding.
        nn_idxs: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor indices.
            One entry for every cell (row) in embeddings
        nn_dists: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor distances.
            One entry for every cell (row) in embeddings
        metadata: pandas.DataFrame
            A pandas dataframe containing cell metadata for nearest neighbors.
        qc_stats: dict
            A dictionary of stats for QC.

        Examples
        --------
        >>> cells_used_in_query = adata.obs["celltype_name"] == "macrophage"
        >>> adata.obs["used_in_query"] = cells_used_in_query.astype(int)
        >>> centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats = cq.search_centroid_nearest(adata, 'used_in_query')
        """

        import numpy as np
        from scipy.cluster.vq import kmeans
        from .utils import get_centroid, get_dist2centroid

        cells = adata[adata.obs[centroid_key] == 1].copy()
        centroid = get_centroid(cells.layers["counts"])
        centroid_embedding = self.get_embeddings(centroid)

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        nn_idxs, nn_dists, metadata = self.search_nearest(
            centroid_embedding,
            k=k,
            ef=ef,
            max_dist=max_dist,
        )

        qc_stats = {}
        if qc:
            cells_embedding = self.get_embeddings(cells.X)
            k_clusters = qc_params.get("k_clusters", 10)
            cluster_centroids = kmeans(cells_embedding, k_clusters, seed=random_seed)[0]

            cell_nn_idxs, _, _ = self.search_nearest(cluster_centroids, k=100)
            query_overlap = []
            for i in range(len(cell_nn_idxs)):
                overlap = [x for x in cell_nn_idxs[i] if x in nn_idxs[0]]
                query_overlap.append(len(overlap))
            qc_stats["query_coherence"] = np.mean(query_overlap)

        return centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats

    def search_cluster_centroids_nearest(
        self,
        adata: "anndata.AnnData",
        cluster_key: str,
        cluster_label: Optional[str] = None,
        k: int = 10000,
        ef: int = None,
        skip_null: bool = True,
        max_dist: Optional[float] = None,
    ) -> Tuple[
        "numpy.ndarray",
        list,
        Dict[str, "numpy.ndarray"],
        Dict[str, "numpy.ndarray"],
        "pandas.DataFrame",
    ]:
        """Performs a nearest neighbors search for cluster centroids against the knn.

        Parameters
        ----------
        adata: anndata.AnnData
            Annotated data matrix with rows for cells and columns for genes.
            Requires a layers["counts"].
        cluster_key: str
            The obs column key that contains cluster labels.
        cluster_label: str, optional, default: None
            The cluster label of interest. If None, then get the centroids of
            all clusters, otherwise get only the centroid for the cluster
            of interest
        k: int, default: 10000
            The number of nearest neighbors.
        ef: int, default: None
            The size of the dynamic list for the nearest neighbors. Defaults to k if None.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        skip_null: bool, default: True
            Whether to skip cells with null/nan cluster labels.
        max_dist: float, optional
            Assume k=1000000, then filter for cells that are within the max distance to the
            query. Overwrites the k parameter.

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
        >>> centroid_embeddings, cluster_idx, nn_idx, nn_dists, all_metadata = cq.search_cluster_centroids_nearest(adata, "leidan")
        """

        from .utils import get_cluster_centroids

        centroids, cluster_idx = get_cluster_centroids(
            adata, self.gene_order, cluster_key, cluster_label, skip_null=skip_null
        )

        centroid_embeddings = self.get_embeddings(centroids)

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        nn_idxs, nn_dists, metadata = self.search_nearest(
            centroid_embeddings,
            k=k,
            ef=ef,
            max_dist=max_dist,
        )

        metadata["centroid"] = metadata["embedding_idx"].map(
            {i: x for i, x in enumerate(cluster_idx)}
        )

        nn_idxs_dict = {}
        nn_dists_dict = {}
        for i in range(len(cluster_idx)):
            nn_idxs_dict[cluster_idx[i]] = [nn_idxs[i]]
            nn_dists_dict[cluster_idx[i]] = [nn_dists[i]]

        return (
            centroid_embeddings,
            cluster_idx,
            nn_idxs_dict,
            nn_dists_dict,
            metadata,
        )

    def search_exhaustive(
        self,
        embeddings: "numpy.ndarray",
        max_dist: float = 0.03,
        metadata_filter: Optional[dict] = None,
        buffer_size: int = 100000,
    ) -> Tuple[List["numpy.ndarray"], List["numpy.ndarray"], "pandas.DataFrame"]:
        """Performs an exhaustive search.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a numpy array.
        max_dist: float, default: 0.03
            Filter for cells that are within the max distance to the query.
        metadata_filter: dict, optional, default: None
            A dictionary where keys represent column names and values
            represent valid terms in the columns.
        buffer_size: int, default: 100000
            Batch size for processing cells.

        Returns
        -------
        nn_idxs: List[numpy.ndarray]
            A list of 2D numpy array of cell indices.
            One entry for every cell (row) in embeddings
        nn_dists: List[numpy.ndarray]
            A list of 2D numpy array of cell distances.
            One entry for every cell (row) in embeddings
        metadata: pandas.DataFrame
            A pandas dataframe containing cell metadata.

        Examples
        --------
        >>> nn_idxs, nn_dists, metadata = cq.search_exhaustive(embeddings)
        """

        import numpy as np
        import pandas as pd
        from scipy.spatial.distance import cdist

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        nn_idxs = [[] for _ in range(embeddings.shape[0])]
        nn_dists = [[] for _ in range(embeddings.shape[0])]
        n_cells = self.cell_metadata.shape[0]
        for i in range(0, n_cells, buffer_size):
            nn_idx = np.array(list(range(i, min(i + buffer_size, n_cells))))
            nn_dist = cdist(
                embeddings, self.get_precomputed_embeddings(nn_idx), metric="cosine"
            )

            for row in range(nn_dist.shape[0]):
                hits = nn_dist[row] <= max_dist
                metadata_hits = []
                if metadata_filter is not None:
                    for k, v in metadata_filter.items():
                        metadata_hits.append(
                            (self.cell_metadata.loc[nn_idx, k] == v).values
                        )
                    if len(metadata_hits) > 1:
                        metadata_hits = np.all(np.vstack(metadata_hits), axis=0)
                    hits = np.all(np.vstack([hits, metadata_hits]), axis=0)
                nn_idxs[row].append(nn_idx[hits])
                nn_dists[row].append(nn_dist[row, hits])

        # sort by lowest distance
        for row in range(len(nn_idxs)):
            nn_idxs[row] = np.hstack(nn_idxs[row])
            nn_dists[row] = np.hstack(nn_dists[row])
            sorted_indices = np.argsort(nn_dists[row])
            nn_idxs[row] = nn_idxs[row][sorted_indices]
            nn_dists[row] = nn_dists[row][sorted_indices]

        metadata = []
        for row in range(len(nn_idxs)):
            hits = nn_idxs[row]
            df = self.cell_metadata.loc[hits].reset_index(drop=True)
            df["embedding_idx"] = row
            df["query_nn_dist"] = nn_dists[row]
            metadata.append(df)
        metadata = pd.concat(metadata).reset_index(drop=True)

        return nn_idxs, nn_dists, metadata

    def search_centroid_exhaustive(
        self,
        adata: "anndata.AnnData",
        centroid_key: str,
        max_dist: float = 0.03,
        metadata_filter: Optional[dict] = None,
        qc: bool = True,
        qc_params: dict = {"k_clusters": 10},
        buffer_size: int = 100000,
        random_seed: int = 4,
    ) -> Tuple[
        "numpy.ndarray",
        List["numpy.ndarray"],
        List["numpy.ndarray"],
        "pandas.DataFrame",
        dict,
    ]:
        """Performs a nearest neighbors search for a centroid constructed from marked cells.

        Parameters
        ----------
        adata: anndata.AnnData
            Annotated data matrix with rows for cells and columns for genes.
            Requires a layers["counts"].
        centroid_key: str
            The obs column key that marks cells to centroid as 1, otherwise 0.
        max_dist: float, default: 0.03
            Filter for cells that are within the max distance to the query.
        metadata_filter: dict, optional, default: None
            A dictionary where keys represent column names and values
            represent valid terms in the columns.
        qc: bool, default: True
            Whether to perform QC on the query
        qc_params: dict, default: {'k_clusters': 10}
            Parameters for the QC:
            k_clusters: the number of clusters in kmeans clustering
        buffer_size: int, default: 100000
            Batch size for processing cells.
        random_seed: int, default: 1
            Random seed for k-means clustering

        Returns
        -------
        centroid_embedding: numpy.ndarray
            A 2D numpy array of the centroid embedding.
        nn_idxs: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor indices.
            One entry for every cell (row) in embeddings
        nn_dists: List[numpy.ndarray]
            A list of 2D numpy array of nearest neighbor distances.
            One entry for every cell (row) in embeddings
        metadata: pandas.DataFrame
            A pandas dataframe containing cell metadata for nearest neighbors.
        qc_stats: dict
            A dictionary of stats for QC.

        Examples
        --------
        >>> cells_used_in_query = adata.obs["celltype_name"] == "macrophage"
        >>> adata.obs["used_in_query"] = cells_used_in_query.astype(int)
        >>> centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats = cq.search_centroid_exhaustive(adata, 'used_in_query')
        """

        import numpy as np
        from scipy.cluster.vq import kmeans
        from .utils import get_centroid, get_dist2centroid

        cells = adata[adata.obs[centroid_key] == 1].copy()
        centroid = get_centroid(cells.layers["counts"])
        centroid_embedding = self.get_embeddings(centroid)

        nn_idxs, nn_dists, metadata = self.search_exhaustive(
            centroid_embedding,
            max_dist=max_dist,
            metadata_filter=metadata_filter,
            buffer_size=buffer_size,
        )

        qc_stats = {}
        if qc:
            cells_embedding = self.get_embeddings(cells.X)
            k_clusters = qc_params.get("k_clusters", 10)
            cluster_centroids = kmeans(cells_embedding, k_clusters, seed=random_seed)[0]

            cell_nn_idxs, _, _ = self.search_nearest(cluster_centroids, k=100)
            query_overlap = []
            for i in range(len(cell_nn_idxs)):
                overlap = [x for x in cell_nn_idxs[i] if x in nn_idxs[0]]
                query_overlap.append(len(overlap))
            qc_stats["query_coherence"] = np.mean(query_overlap)

        return centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats

    def search_cluster_centroids_exhaustive(
        self,
        adata: "anndata.AnnData",
        cluster_key: str,
        cluster_label: Optional[str] = None,
        max_dist: float = 0.03,
        metadata_filter: Optional[dict] = None,
        buffer_size: int = 100000,
        skip_null: bool = True,
    ) -> Tuple[
        "numpy.ndarray",
        list,
        Dict[str, "numpy.ndarray"],
        Dict[str, "numpy.ndarray"],
        "pandas.DataFrame",
    ]:
        """Performs a nearest neighbors search for cluster centroids against the knn.

        Parameters
        ----------
        adata: anndata.AnnData
            Annotated data matrix with rows for cells and columns for genes.
            Requires a layers["counts"].
        cluster_key: str
            The obs column key that contains cluster labels.
        cluster_label: str, optional, default: None
            The cluster label of interest. If None, then get the centroids of
            all clusters, otherwise get only the centroid for the cluster
            of interest
        max_dist: float, default: 0.03
            Filter for cells that are within the max distance to the query.
        metadata_filter: dict, optional, default: None
            A dictionary where keys represent column names and values
            represent valid terms in the columns.
        buffer_size: int, default: 100000
            Batch size for processing cells.
        skip_null: bool, default: True
            Whether to skip cells with null/nan cluster labels.

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
        >>> centroid_embeddings, cluster_idx, nn_idx, nn_dists, all_metadata = cq.search_cluster_centroids_exhaustive(adata, "leidan")
        """

        from .utils import get_cluster_centroids

        centroids, cluster_idx = get_cluster_centroids(
            adata, self.gene_order, cluster_key, cluster_label, skip_null=skip_null
        )

        centroid_embeddings = self.get_embeddings(centroids)

        nn_idxs, nn_dists, metadata = self.search_exhaustive(
            centroid_embeddings,
            max_dist=max_dist,
            metadata_filter=metadata_filter,
            buffer_size=buffer_size,
        )

        metadata["centroid"] = metadata["embedding_idx"].map(
            {i: x for i, x in enumerate(cluster_idx)}
        )

        nn_idxs_dict = {}
        nn_dists_dict = {}
        for i in range(len(cluster_idx)):
            nn_idxs_dict[cluster_idx[i]] = [nn_idxs[i]]
            nn_dists_dict[cluster_idx[i]] = [nn_dists[i]]

        return (
            centroid_embeddings,
            cluster_idx,
            nn_idxs_dict,
            nn_dists_dict,
            metadata,
        )
