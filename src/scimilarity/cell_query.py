from typing import Dict, List, Optional, Tuple, Union, Set

from scimilarity.cell_embedding import CellEmbedding


class CellQuery(CellEmbedding):
    """A class that searches for similar cells using a cell embedding and then a kNN search."""

    def __init__(
        self,
        model_path: str,
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
            Path to the model directory.
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

        import os
        import numpy as np
        import pandas as pd
        import tiledb
        from scimilarity.utils import write_tiledb_array, optimize_tiledb_array

        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
            parameters=parameters,
            filenames=filenames,
            residual=residual,
        )
        self.cellsearch_path = os.path.join(model_path, "cellsearch")

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

        # get knn
        if load_knn:
            self.load_knn_index(self.filenames["knn"])
        self.block_list = set()

        # get cell metadata: create tiledb storage if it does not exist
        # NOTE: process for creating this file is not hardened, no guarantee index column is unique
        metadata_tiledb_uri = os.path.join(self.cellsearch_path, metadata_tiledb_uri)
        if not os.path.isdir(metadata_tiledb_uri):
            print(f"Configuring tiledb dataframe: {metadata_tiledb_uri}")
            cell_metadata = (
                pd.read_csv(
                    self.filenames["cell_metadata"],
                    header=0,
                    dtype=str,
                )
                .fillna("NA")
                .reset_index(drop=True)
            )
            cell_metadata = cell_metadata.rename(columns={"Unnamed: 0": "index"})
            convert_dict = {
                "index": int,
                "prediction_nn_dist": float,
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
        embedding_tiledb_uri = os.path.join(self.cellsearch_path, embedding_tiledb_uri)
        if not os.path.isdir(embedding_tiledb_uri):
            cell_embeddings = np.load(
                os.path.join(cellsearch_path, self.filenames["cell_embeddings"])
            )
            write_tiledb_array(embedding_tiledb_uri, cell_embeddings)
            optimize_tiledb_array(embedding_tiledb_uri)
        self.cell_embedding = tiledb.open(embedding_tiledb_uri)

        self.study_sample_index = (
            self.cell_metadata.groupby(["study", "sample", "data_type"], observed=True)[
                "index"
            ]
            .min()
            .sort_values()
        )

    def get_precomputed_embeddings(self, idx: List[int]) -> "numpy.ndarray":
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
        >>> array = cq.get_precomputed_embeddings([0, 1, 100])
        """
        return self.cell_embedding.query(attrs=["vals"], coords=True).multi_index[idx][
            "vals"
        ]

    def annotate_cell_index(self, metadata: "pandas.DataFrame") -> "pandas.DataFrame":
        """Annotate a metadata dataframe with the cell index in sample datasets.

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

    def compile_sample_metadata(self, nn_idxs: "numpy.ndarray") -> "pandas.DataFrame":
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

        import pandas as pd

        levels = ["tissue", "disease", "study", "sample"]
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

        study_sample_cells = self.cell_metadata.groupby(
            ["study", "sample"], observed=True
        ).size()

        fraction = []
        for i, row in df.iterrows():
            total_cells = study_sample_cells.loc[(row["study"], row["sample"])]
            fraction.append(row["cells"] / total_cells)
        df["fraction"] = fraction
        return df

    def groupby_studies(
        self, sample_metadata: "pandas.DataFrame"
    ) -> "pandas.DataFrame":
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
            .groupby(levels, observed=True)["cells"]
            .sum()
            .reset_index(name="cells")
        )

        study_cells = self.cell_metadata.groupby("study", observed=True).size()

        fraction = []
        for i, row in df.iterrows():
            total_cells = study_cells.loc[row["study"]]
            fraction.append(row["cells"] / total_cells)
        df["fraction"] = fraction
        return df

    def search(
        self,
        embeddings: "numpy.ndarray",
        k: int = 10000,
        ef: int = None,
        max_dist: float = None,
    ) -> Tuple[List["numpy.ndarray"], List["numpy.ndarray"], "pandas.DataFrame"]:
        """Performs a cell query search against the kNN.

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
        >>> nn_idxs, nn_dists, metadata = cq.search(embedding)
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

    def search_centroid(
        self,
        data: "anndata.AnnData",
        centroid_key: str,
        k: int = 10000,
        ef: int = None,
        max_dist: float = None,
        qc: bool = True,
        qc_params: dict = {"k_clusters": 10},
    ) -> Tuple[
        "numpy.ndarray",
        List["numpy.ndarray"],
        List["numpy.ndarray"],
        "pandas.DataFrame",
        dict,
    ]:
        """Performs a cell query search for a centroid constructed from marked cells.

        Parameters
        ----------
        data: anndata.AnnData
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
        >>> centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats = cq.search_centroid(adata, 'used_in_query')
        """

        import numpy as np
        from scipy.cluster.vq import kmeans
        from scipy.spatial.distance import cdist, pdist, squareform
        from scimilarity.utils import get_centroid, get_dist2centroid

        cells = data[data.obs[centroid_key] == 1].copy()
        centroid = get_centroid(cells.layers["counts"])
        centroid_embedding = self.get_embeddings(centroid)

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        (nn_idxs, nn_dists, metadata) = self.search(
            centroid_embedding,
            k=k,
            ef=ef,
            max_dist=max_dist,
        )

        qc_stats = {}
        if qc:
            cells_embedding = self.get_embeddings(cells.X)
            k_clusters = qc_params.get("k_clusters", 10)
            cluster_centroids = kmeans(cells_embedding, k_clusters)[0]

            cell_nn_idxs, _, _ = self.search(cluster_centroids, k=100)
            query_overlap = []
            for i in range(len(cell_nn_idxs)):
                overlap = [x for x in cell_nn_idxs[i] if x in nn_idxs[0]]
                query_overlap.append(len(overlap))
            qc_stats["query_stability"] = np.mean(query_overlap)

        return centroid_embedding, nn_idxs, nn_dists, metadata, qc_stats

    def search_cluster_centroids(
        self,
        data: "anndata.AnnData",
        cluster_key: str,
        cluster_label: Optional[str] = None,
        k: int = 10000,
        ef: int = None,
        skip_null: bool = True,
        max_dist: float = None,
    ) -> Tuple[
        "numpy.ndarray",
        list,
        Dict[str, "numpy.ndarray"],
        Dict[str, "numpy.ndarray"],
        "pandas.DataFrame",
    ]:
        """Performs a cell query search for cluster centroids against the kNN.

        Parameters
        ----------
        data: anndata.AnnData
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
        >>> centroid_embeddings, cluster_idx, nn_idx, nn_dists, all_metadata = cq.search_cluster_centroids(data, "leidan")
        """

        from scimilarity.utils import get_cluster_centroids

        centroids, cluster_idx = get_cluster_centroids(
            data, self.gene_order, cluster_key, cluster_label, skip_null=skip_null
        )

        centroid_embeddings = self.get_embeddings(centroids)

        if max_dist is not None:
            k = 1000000

        if ef is None:
            ef = k

        (nn_idxs, nn_dists, metadata) = self.search(
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
