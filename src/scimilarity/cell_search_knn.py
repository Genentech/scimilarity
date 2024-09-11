from typing import Optional, Tuple, Union

from .cell_embedding import CellEmbedding


class CellSearchKNN(CellEmbedding):
    """A class for searching similar cells using cell embeddings kNN."""

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
            The kNN filenames also need to be specified here.
        residual: bool, default: False
            Use residual connections.

        Examples
        --------
        >>> filenames = {"knn": "knn.bin"}
        >>> cs = CellSearch(model_path="/opt/data/model", filenames=filesnames)
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

        self.knn = None
        self.safelist = None
        self.blocklist = None

    def load_knn_index(self, knn_file: str):
        """Load the kNN index file

        Parameters
        ----------
        knn_file: str
            Filename of the kNN index.
        """

        import hnswlib
        import os

        if os.path.isfile(knn_file):
            self.knn = hnswlib.Index(space="cosine", dim=self.model.latent_dim)
            self.knn.load_index(knn_file)
        else:
            print(f"Warning: No KNN index found at {knn_file}")
            self.knn = None

    def get_nearest_neighbors(
        self, embeddings: "numpy.ndarray", k: int = 50, ef: int = 100
    ) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
        """Get nearest neighbors.
        Used by classes that inherit from CellEmbedding and have an instantiated kNN.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a numpy array.
        k: int, default: 50
            The number of nearest neighbors.
        ef: int, default: 100
            The size of the dynamic list for the nearest neighbors.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

        Returns
        -------
        nn_idxs: numpy.ndarray
            A 2D numpy array of nearest neighbor indices [num_cells x k].
        nn_dists: numpy.ndarray
            A 2D numpy array of nearest neighbor distances [num_cells x k].

        Examples
        --------
        >>> nn_idxs, nn_dists = get_nearest_neighbors(embeddings)
        """

        if self.knn is None:
            raise RuntimeError("kNN is not initialized.")
        self.knn.set_ef(ef)
        return self.knn.knn_query(embeddings, k=k)
