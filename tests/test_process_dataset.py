"""
Tests.

Uses tests/data/test.h5ad as the source dataset.
"""

import os

import anndata
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix

from scimilarity.utils import align_dataset, lognorm_counts
from scimilarity.zarr_dataset import ZARR_V3, ZarrDataset


def _open_zarr(path, mode="r"):
    """Open a zarr group, handling zarr 2 vs 3 API differences."""
    if ZARR_V3:
        return zarr.open_group(path, mode=mode)
    store = zarr.DirectoryStore(path)
    return zarr.open_group(store, mode=mode, chunk_store=store)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
H5AD_PATH = os.path.join(DATA_DIR, "test.h5ad")
GENE_ORDER_PATH = os.path.join(DATA_DIR, "gene_order.tsv")


@pytest.fixture
def adata():
    """Load the test h5ad."""
    return anndata.read_h5ad(H5AD_PATH)


@pytest.fixture
def zarr_path(adata, tmp_path):
    """Write adata to zarr via anndata and return the path."""
    p = str(tmp_path / "test.zarr")
    adata.write_zarr(p)
    return p


@pytest.fixture
def adata_with_categoricals(adata):
    """Return a copy of adata with categorical obs columns added.

    The upstream test.h5ad has only numeric obs; we add categoricals so
    we can test the zarr.Group isinstance path used by ZarrDataset.
    """
    rng = np.random.default_rng(42)
    ad = adata.copy()
    ad.obs["celltype_name"] = pd.Categorical(
        rng.choice(["T cell", "B cell", "macrophage"], ad.shape[0])
    )
    ad.obs["study"] = pd.Categorical(
        rng.choice(["study_A", "study_B"], ad.shape[0])
    )
    return ad


@pytest.fixture
def zarr_path_with_categoricals(adata_with_categoricals, tmp_path):
    """Write the categorical-augmented adata to zarr."""
    p = str(tmp_path / "test_cat.zarr")
    adata_with_categoricals.write_zarr(p)
    return p


# ── zarr core API compatibility ─────────────────────────────────────────


class TestZarrAPIs:
    """Test that the zarr APIs used by scimilarity resolve correctly.

    These are the APIs that changed between zarr 2 and zarr 3.
    """

    def test_open_group(self, zarr_path):
        root = _open_zarr(zarr_path)
        assert "X" in root
        assert "obs" in root
        assert "var" in root

    def test_zarr_array_isinstance(self, zarr_path):
        """zarr.Array must resolve (was zarr.core.Array in zarr 2)."""
        root = _open_zarr(zarr_path)
        arr = root["X"]["data"]
        assert isinstance(arr, zarr.Array)

    def test_zarr_group_isinstance(self, zarr_path):
        """zarr.Group must resolve (was zarr.hierarchy.Group in zarr 2)."""
        root = _open_zarr(zarr_path)
        assert isinstance(root, zarr.Group)
        assert isinstance(root["X"], zarr.Group)

    def test_group_attrs(self, zarr_path):
        root = _open_zarr(zarr_path)
        assert root["X"].attrs["encoding-type"] == "csr_matrix"
        shape = root["X"].attrs["shape"]
        assert len(shape) == 2
        assert shape[0] > 0 and shape[1] > 0

    def test_array_slicing(self, zarr_path):
        """Verify zarr array slice + materialize with [...] work."""
        root = _open_zarr(zarr_path)
        data = root["X"]["data"]
        # Materialize full array
        full = data[...]
        assert isinstance(full, np.ndarray)
        assert full.shape[0] == data.shape[0]
        # Slice
        partial = data[0:5]
        assert partial.shape[0] == min(5, data.shape[0])


# ── ZarrDataset integration ─────────────────────────────────────────────


class TestZarrDataset:
    """Test ZarrDataset reading anndata-written zarr files."""

    def test_init(self, zarr_path):
        zd = ZarrDataset(zarr_path)
        assert zd.root is not None

    def test_shape(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        assert zd.shape == list(adata.shape)

    def test_dataset_info(self, zarr_path):
        zd = ZarrDataset(zarr_path)
        info = zd.dataset_info
        assert "obs" in info
        assert "var" in info
        assert "shape" in info

    def test_var_index(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        var_idx = zd.var_index
        assert var_idx is not None
        assert len(var_idx) == adata.shape[1]
        assert list(var_idx) == list(adata.var.index)

    def test_obs_index(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        obs_idx = zd.obs_index
        assert obs_idx is not None
        assert len(obs_idx) == adata.shape[0]

    def test_get_obs_numeric(self, adata, zarr_path):
        """Read a numeric obs column."""
        zd = ZarrDataset(zarr_path)
        values = zd.get_obs("n_genes")
        assert values is not None
        assert len(values) == adata.shape[0]
        np.testing.assert_array_equal(values, adata.obs["n_genes"].values)

    def test_get_obs_all_columns(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        for col in adata.obs.columns:
            values = zd.get_obs(col)
            assert values is not None, f"get_obs({col!r}) returned None"
            assert len(values) == adata.shape[0]

    def test_obs_dataframe(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        obs = zd.obs
        assert obs is not None
        assert obs.shape[0] == adata.shape[0]

    def test_var_dataframe(self, adata, zarr_path):
        zd = ZarrDataset(zarr_path)
        var = zd.var
        assert var is not None
        assert var.shape[0] == adata.shape[1]

    def test_get_obs_missing_column(self, zarr_path):
        zd = ZarrDataset(zarr_path)
        assert zd.get_obs("nonexistent_column") is None


# ── Categorical obs via zarr.Group ──────────────────────────────────────


class TestCategoricalObs:
    """Test reading categorical obs columns — the critical zarr.Group isinstance path."""

    def test_get_obs_categorical(
        self, adata_with_categoricals, zarr_path_with_categoricals
    ):
        """Read a categorical obs column — uses zarr.Group isinstance check."""
        zd = ZarrDataset(zarr_path_with_categoricals)
        celltypes = zd.get_obs("celltype_name")
        assert celltypes is not None
        assert len(celltypes) == adata_with_categoricals.shape[0]
        assert set(celltypes) == set(adata_with_categoricals.obs["celltype_name"])

    def test_categorical_stored_as_zarr_group(self, zarr_path_with_categoricals):
        """Categorical obs should be a zarr.Group (not zarr.hierarchy.Group)."""
        root = _open_zarr(zarr_path_with_categoricals)
        ct_group = root["obs"]["celltype_name"]
        assert isinstance(ct_group, zarr.Group)
        assert "categories" in ct_group
        assert "codes" in ct_group

    def test_all_categorical_columns(
        self, adata_with_categoricals, zarr_path_with_categoricals
    ):
        zd = ZarrDataset(zarr_path_with_categoricals)
        for col in ["celltype_name", "study"]:
            values = zd.get_obs(col)
            assert values is not None
            assert len(values) == adata_with_categoricals.shape[0]


# ── Sparse matrix round-trip ────────────────────────────────────────────


class TestSparseMatrixRoundTrip:
    """Test reading sparse matrices through ZarrDataset (the critical zarr 3 path)."""

    def test_get_X_backed(self, adata, zarr_path):
        """Get X with zarr-backed arrays (default, not in-memory)."""
        zd = ZarrDataset(zarr_path)
        X = zd.get_X(in_mem=False)
        assert X is not None
        assert X.shape == adata.shape
        # Components should be zarr arrays, not numpy
        assert isinstance(X.data, zarr.Array)
        assert isinstance(X.indices, zarr.Array)
        assert isinstance(X.indptr, zarr.Array)

    def test_get_X_in_memory(self, adata, zarr_path):
        """Get X materialized in memory."""
        zd = ZarrDataset(zarr_path)
        X = zd.get_X(in_mem=True)
        assert X is not None
        assert isinstance(X.data, np.ndarray)
        assert isinstance(X.indices, np.ndarray)
        assert isinstance(X.indptr, np.ndarray)
        # Values should match original
        orig = adata.X.toarray()
        result = X.toarray()
        np.testing.assert_array_almost_equal(result, orig)

    def test_get_cell(self, adata, zarr_path):
        """Get a single cell row — exercises row_slice_csr."""
        zd = ZarrDataset(zarr_path)
        cell = zd.get_cell(0)
        assert cell is not None
        assert cell.shape == (1, adata.shape[1])
        expected = adata.X[0].toarray()
        np.testing.assert_array_almost_equal(cell.toarray(), expected)

    def test_get_cell_multiple(self, adata, zarr_path):
        """Get several cells and verify values."""
        zd = ZarrDataset(zarr_path)
        for idx in [0, 1, adata.shape[0] - 1]:
            cell = zd.get_cell(idx)
            expected = adata.X[idx].toarray()
            np.testing.assert_array_almost_equal(cell.toarray(), expected)

    def test_get_counts_layer(self, adata, zarr_path):
        """Read the counts layer."""
        zd = ZarrDataset(zarr_path)
        counts = zd.get_counts(in_mem=True)
        assert counts is not None
        expected = adata.layers["counts"].toarray()
        np.testing.assert_array_almost_equal(counts.toarray(), expected)

    def test_backed_array_isinstance_zarr_array(self, zarr_path):
        """The isinstance check in cell_embedding.py: zarr.Array (not zarr.core.Array)."""
        zd = ZarrDataset(zarr_path)
        X = zd.get_X(in_mem=False)
        assert isinstance(X, csr_matrix)
        assert isinstance(X.data, zarr.Array)
        assert isinstance(X.indices, zarr.Array)
        assert isinstance(X.indptr, zarr.Array)

    def test_backed_materialize(self, zarr_path):
        """Materialize zarr-backed sparse arrays with [...] syntax."""
        zd = ZarrDataset(zarr_path)
        X = zd.get_X(in_mem=False)
        # This is the pattern from cell_embedding.py
        data = X.data[...]
        indices = X.indices[...]
        indptr = X.indptr[...]
        assert isinstance(data, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(indptr, np.ndarray)


# ── Write round-trip via ZarrDataset ────────────────────────────────────


class TestZarrDatasetWrite:
    """Test writing and reading via ZarrDataset and anndata.write_zarr."""

    def test_set_X_roundtrip(self, adata, tmp_path):
        """Write a CSR matrix via ZarrDataset and read it back."""
        p = str(tmp_path / "write_test.zarr")
        os.makedirs(p)
        zd = ZarrDataset(p, mode="w")
        zd.set_X(adata.X)
        # Read back
        zd2 = ZarrDataset(p)
        X = zd2.get_X(in_mem=True)
        np.testing.assert_array_almost_equal(X.toarray(), adata.X.toarray())

    def test_anndata_write_zarr_annotation_roundtrip(
        self, adata_with_categoricals, tmp_path
    ):
        """Write via anndata.write_zarr, read annotations back via ZarrDataset."""
        p = str(tmp_path / "anno_test.zarr")
        adata_with_categoricals.write_zarr(p)
        zd = ZarrDataset(p)
        obs_idx = zd.obs_index
        assert len(obs_idx) == adata_with_categoricals.shape[0]
        celltypes = zd.get_obs("celltype_name")
        assert set(celltypes) == set(adata_with_categoricals.obs["celltype_name"])

    def test_anndata_write_zarr_X_roundtrip(self, adata, tmp_path):
        """Write via anndata.write_zarr, read X back via ZarrDataset."""
        p = str(tmp_path / "x_test.zarr")
        adata.write_zarr(p)
        zd = ZarrDataset(p)
        X = zd.get_X(in_mem=True)
        np.testing.assert_array_almost_equal(X.toarray(), adata.X.toarray())


# ── Dataset processing (from scimilarity_gred) ────────────────────────


class TestProcessDataset:
    """Test align_dataset and lognorm_counts from scimilarity.utils."""

    def test_process_anndata(self):
        data = anndata.read_h5ad(H5AD_PATH)
        with open(GENE_ORDER_PATH, "r") as fh:
            gene_order = [line.strip() for line in fh]
        data = align_dataset(data, gene_order, gene_overlap_threshold=100)
        data = lognorm_counts(data)
        assert len(data.var) == len(gene_order)
