"""Tests for tiledb_data_models."""

import numpy as np
import pandas as pd
import pytest
import tiledb

from scimilarity.tiledb_data_models import scCollator


# cells x genes; the last two genes and the last cell have no counts, so the
# counts array's non-empty domain ends before the full gene and cell range
COUNTS = np.array(
    [
        [1, 0, 3, 0, 0],
        [0, 2, 0, 0, 0],
        [4, 0, 0, 0, 0],
        [0, 0, 5, 0, 0],
        [6, 7, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint32,
)


@pytest.fixture
def counts_uri(tmp_path):
    """Sparse counts array with the same schema as a CellArr counts assay."""
    uri = str(tmp_path / "counts")
    max_idx = np.iinfo(np.uint32).max - 1
    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max_idx), dtype=np.uint32),
        tiledb.Dim(name="gene_index", domain=(0, max_idx), dtype=np.uint32),
    )
    schema = tiledb.ArraySchema(
        domain=dom, sparse=True, attrs=[tiledb.Attr(name="data", dtype=np.uint32)]
    )
    tiledb.Array.create(uri, schema)
    rows, cols = np.nonzero(COUNTS)
    with tiledb.open(uri, "w") as tdb:
        tdb[rows.astype(np.uint32), cols.astype(np.uint32)] = COUNTS[rows, cols]
    return uri


def _batch(cell_idx):
    return [
        pd.DataFrame({"study": [f"s{i}"], "label_int": [i % 2]}, index=[i])
        for i in cell_idx
    ]


@pytest.mark.parametrize("gene_indices", [[0, 1, 2, 3, 4], [4, 2, 0], [3, 4]])
def test_collator_genes_beyond_nonempty_domain(counts_uri, gene_indices):
    """Genes after the last gene with counts are returned as zeros, not an IndexError."""
    collator = scCollator(counts_uri, gene_indices, "study", lognorm=False)
    cell_idx = [4, 0, 2]
    X, labels, studies = collator(_batch(cell_idx))

    np.testing.assert_array_equal(X.numpy(), COUNTS[np.ix_(cell_idx, gene_indices)])
    np.testing.assert_array_equal(labels.numpy(), [0, 0, 0])
    assert list(studies) == ["s4", "s0", "s2"]


def test_collator_cells_beyond_nonempty_domain(counts_uri):
    """A cell after the last cell with counts yields an all-zero row."""
    collator = scCollator(counts_uri, [0, 1, 2], "study", lognorm=False)
    X, _, _ = collator(_batch([5, 1]))
    np.testing.assert_array_equal(X.numpy(), COUNTS[np.ix_([5, 1], [0, 1, 2])])


def test_collator_lognorm(counts_uri):
    """Log normalization is unchanged: log1p of counts scaled to target_sum per cell."""
    gene_indices = [0, 1, 2, 3]
    collator = scCollator(counts_uri, gene_indices, "study", target_sum=1e4)
    cell_idx = [3, 0, 5]
    X, _, _ = collator(_batch(cell_idx))

    raw = COUNTS[np.ix_(cell_idx, gene_indices)].astype(np.float64)
    sums = raw.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    np.testing.assert_allclose(X.numpy(), np.log1p(raw / sums * 1e4), rtol=1e-5)
