from typing import Optional, Union, Tuple, List


def lognorm_counts(
    data: "anndata.AnnData",
) -> "anndata.AnnData":
    """Log normalize the gene expression raw counts (per 10k).

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.

    Returns
    -------
    anndata.AnnData
        A data object with normalized data that is ready to be used in further processes.

    Examples
    --------
    >>> data = lognorm_counts(data)
    """

    import numpy as np
    import scanpy as sc

    if "counts" not in data.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    data.X = data.layers["counts"].copy()

    # check for nan in expression data, zero
    if isinstance(data.X, np.ndarray) and np.isnan(data.X).any():
        import warnings

        warnings.warn(
            "NANs detected in counts. NANs will be zeroed before normalization in X.",
            UserWarning,
        )
        data.X = np.nan_to_num(data.X, nan=0.0)

    # log norm
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    del data.uns["log1p"]

    return data


def align_dataset(
    data: "anndata.AnnData",
    target_gene_order: list,
    keep_obsm: bool = True,
    gene_overlap_threshold: int = 5000,
) -> "anndata.AnnData":
    """Align the gene space to the target gene order.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: list
        A list containing the gene space.
    keep_obsm: bool, default: True
        Retain the original data's obsm matrices in output.
    gene_overlap_threshold: int, default: 5000
        The minimum number of genes in common between data and target_gene_order to be valid.

    Returns
    -------
    anndata.AnnData
        A data object with aligned gene space ready to be used for embedding cells.

    Examples
    --------
    >>> ca = CellAnnotation(model_path="/opt/data/model")
    >>> data = align_dataset(data, ca.gene_order)
    """

    import anndata
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    # raise an error if not enough genes from target_gene_order exists
    gene_overlap = sum(data.var.index.isin(target_gene_order))
    if gene_overlap < gene_overlap_threshold:
        raise RuntimeError(
            f"Dataset incompatible. Gene overlap of {gene_overlap} less than {gene_overlap_threshold}. Check that var.index uses gene symbols."
        )

    # check if X is dense, convert to csr_matrix if so
    if isinstance(data.X, np.ndarray):
        data.X = csr_matrix(data.X)

    # check for negatives in expression data
    if np.min(data.X) < 0:
        raise RuntimeError(f"Dataset contains negative values in expression matrix X.")

    # check if counts is dense, convert to csr_matrix if so
    if "counts" in data.layers and isinstance(data.layers["counts"], np.ndarray):
        data.layers["counts"] = csr_matrix(data.layers["counts"])

    # check for negatives in count data
    if "counts" in data.layers and np.min(data.layers["counts"]) < 0:
        raise RuntimeError(f"Dataset contains negative values in layers['counts'].")

    # return data if already aligned
    if data.var.index.values.tolist() == target_gene_order:
        return data

    orig_genes = data.var.index.values  # record original gene list before alignment
    shell = anndata.AnnData(
        X=csr_matrix((0, len(target_gene_order))),
        var=pd.DataFrame(index=target_gene_order),
    )
    shell = anndata.concat(
        (shell, data[:, data.var.index.isin(shell.var.index)]), join="outer"
    )
    shell.uns["orig_genes"] = orig_genes
    if not keep_obsm and hasattr(data, "obsm"):
        delattr(shell, "obsm")

    if data.var.shape[0] == 0:
        raise RuntimeError(f"Empty gene space detected.")

    return shell


def filter_cells(
    data: "anndata.AnnData",
    min_genes: int = 400,
    mito_prefix: Optional[str] = None,
    mito_percent: float = 30.0,
) -> "anndata.AnnData":
    """QC filter cells in the dataset from gene expression raw counts.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    min_genes: int, default: 400
        The minimum number of expressed genes in order not to be filtered out.
    mito_prefix: str, optional, default: None
        The prefix to represent mitochondria genes. Typically "MT-" or "mt-".
        If None, it will try to infer whether it is either "MT-" or "mt-".
    mito_percent: float, default: 30.0
        The maximum percent allowed expressed mitochondria genes in order not to be filtered out.

    Returns
    -------
    anndata.AnnData
        A data object with cells filtered out based on QC metrics that is ready to be used
        in further processes.

    Examples
    --------
    >>> data = filter_cells(data)
    """

    import scanpy as sc

    if "counts" not in data.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    # determine between "MT-" and "mt-"
    if not mito_prefix:
        mito_prefix = "MT-"
        if any(data.var.index.str.startswith("mt-")) is True:
            mito_prefix = "mt-"

    # filter
    data.var["mt"] = data.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(
        data,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
        layer="counts",
    )
    data = data[data.obs["pct_counts_mt"] < mito_percent].copy()
    cell_subset, _ = sc.pp.filter_cells(data, min_genes=min_genes, inplace=False)
    data = data[cell_subset].copy()

    return data


def consolidate_duplicate_symbols(
    adata: "anndata.AnnData",
) -> "anndata.AnnData":
    """Consolidate duplicate gene symbols with sum.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.

    Returns
    -------
    anndata.AnnData
        AnnData object with duplicate gene symbols consolidated.

    Examples
    --------
    >>> adata = consolidate_duplicate_symbols(adata)
    """

    import anndata
    from collections import Counter
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    if "counts" not in adata.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    gene_count = Counter(adata.var.index.values)
    dup_genes = {k for k in gene_count if gene_count[k] > 1}
    if len(dup_genes) == 0:
        return adata

    dup_genes_data = []
    for k in sorted(dup_genes):
        idx = [i for i, x in enumerate(adata.var.index.values) if x == k]
        counts = csr_matrix(
            np.array(adata.layers["counts"][:, idx].sum(axis=1)).flatten()[:, None]
        )
        gene_data = anndata.AnnData(
            X=csr_matrix(counts.shape),
            var=pd.DataFrame(index=[k]),
        )
        gene_data.layers["counts"] = counts
        dup_genes_data.append(gene_data)

    obs = adata.obs.copy()
    dup_genes_data = anndata.concat(dup_genes_data, axis=1)
    dup_genes_data.obs = obs.reset_index(drop=True)
    dup_genes_data.obs.index = dup_genes_data.obs.index.astype(str)

    adata.obs = obs.reset_index(drop=True)
    adata.obs.index = adata.obs.index.astype(str)
    adata = anndata.concat(
        [adata[:, ~adata.var.index.isin(dup_genes)].copy(), dup_genes_data], axis=1
    )
    adata.obs = obs.copy()

    return adata


def get_centroid(
    counts: Union["scipy.sparse.csr_matrix", "numpy.ndarray"]
) -> "numpy.ndarray":
    """Get the centroid for a raw counts matrix.

    Parameters
    ----------
    counts: scipy.sparse.csr_matrix, numpy.ndarray
        Raw gene expression counts.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of the log normalized (1e4) for the centroid.

    Examples
    --------
    >>> centroid = get_centroid(data.get_matrix("counts"))
    >>> centroid = get_centroid(data.layers["counts"])
    """

    import numpy as np
    from scipy.sparse import csr_matrix

    if isinstance(counts, np.ndarray):
        counts = csr_matrix(counts)

    sum_counts = counts.sum(axis=0).A
    normalization_factor = counts.sum()
    centroid = np.log(1 + 1e4 * sum_counts / normalization_factor)

    return centroid


def get_dist2centroid(
    centroid_embedding: "numpy.ndarray",
    X: Union["scipy.sparse.csr_matrix", "numpy.ndarray"],
) -> "numpy.ndarray":
    """Get the centroid for a raw counts matrix in sparse csr_matrix format.

    Parameters
    ----------
    centroid_embedding: numpy.ndarray
        The embedding of the centroid.
    X: scipy.sparse.csr_matrix, numpy.ndarray
        The embedding of SCimilarity log normalized gene expression values or
        SCimilarity log normalized gene expression values.
    embed: bool, default: False
        Whether to embed X.

    Returns
    -------
    float
        The mean distance of cells in X to the centroid embedding.

    Examples
    --------
    >>> distances = cq.get_dist2centroid(centroid_embedding, X)
    """

    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix

    if isinstance(X, csr_matrix):
        X = X.A
    distances = cdist(centroid_embedding.reshape(1, -1), X, metric="cosine").flatten()

    return distances


def get_cluster_centroids(
    data: "anndata.AnnData",
    target_gene_order: "numpy.ndarray",
    cluster_key: str,
    cluster_label: Optional[str] = None,
    skip_null: bool = True,
) -> Tuple["numpy.ndarray", list]:
    """Get centroids of clusters based on raw read counts.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: numpy.ndarray
        An array containing the gene space.
    cluster_key: str
        The obs column name that contains cluster labels.
    cluster_label: str, optional, default: None
        The cluster label of interest. If None, then get the centroids of
        all clusters, otherwise get only the centroid for the cluster
        of interest
    skip_null: bool, default: True
        Whether to skip cells with null/nan cluster labels.

    Returns
    -------
    centroids: numpy.ndarray
        A 2D numpy array of the log normalized (1e4) cluster centroids.
    cluster_idx: list
        A list of cluster labels corresponding to the order returned in centroids.

    Examples
    --------
    >>> centroids, cluster_idx = get_cluster_centroids(data, gene_order, "cluster_label")
    """

    import numpy as np

    if "counts" not in data.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    centroids = []
    cluster_idx = []

    aligned_data = align_dataset(data, target_gene_order)
    if skip_null:
        aligned_data = aligned_data[aligned_data.obs[cluster_key].notnull()].copy()
    aligned_data.obs[cluster_key] = aligned_data.obs[cluster_key].astype(str)

    for i in set(aligned_data.obs[cluster_key]):
        if cluster_label is not None:
            i = cluster_label

        cluster_idx.append(i)
        centroids.append(
            get_centroid(
                aligned_data[aligned_data.obs[cluster_key] == i].copy().layers["counts"]
            )
        )

        if cluster_label is not None:
            break

    centroids = np.vstack(centroids)

    if np.isnan(centroids).any():
        raise RuntimeError(f"NaN detected in centroids.")

    return centroids, cluster_idx


def write_array_to_tiledb(
    tdb: "tiledb.libtiledb.DenseArrayImpl",
    arr: "numpy.ndarray",
    value_type: type,
    row_start: int = 0,
    batch_size: int = 100000,
):
    """Write numpy array to TileDB.

    Parameters
    ----------
    tdb: tiledb.libtiledb.DenseArrayImpl
        TileDB array.
    arr: numpy.ndarray
        Dense 2D numpy array.
    value_type: type
        The type of the value, typically np.float32.
    row_start: int, default: 0
        The starting row in the TileDB array.
    batch_size: int, default: 100000
        Batch size for the tiles.
    """

    for i in range(0, arr.shape[0], batch_size):
        j = min(i + batch_size, arr.shape[0])
        arr_slice = slice(i, j)
        tdb_slice = slice(i + row_start, j + row_start)
        tdbfile[tdb_slice, 0 : arr.shape[1]] = arr[arr_slice, :].astype(value_type)


def write_csr_to_tiledb(
    tdb: "tiledb.libtiledb.SparseArrayImpl",
    matrix: "scipy.sparse.csr_matrix",
    value_type: type,
    row_start: int = 0,
    batch_size: int = 25000,
):
    """Write csr_matrix to TileDB.

    Parameters
    ----------
    tdb: tiledb.libtiledb.SparseArrayImpl
        TileDB array.
    arr: numpy.ndarray
        Dense numpy array.
    value_type: type
        The type of the value, typically np.float32.
    row_start: int, default: 0
        The starting row in the TileDB array.
    batch_size: int, default: 100000
        Batch size for the tiles.
    """
    indptrs = matrix.indptr
    indices = matrix.indices
    data = matrix.data

    x = []
    y = []
    vals = []
    for i, indptr in enumerate(indptrs):
        if i != 0 and (i % batch_size == 0 or i == len(indptrs) - 1):
            tdb[x, y] = vals
            x = []
            y = []
            vals = []

        stop = None
        if i != len(indptrs) - 1:
            stop = indptrs[i + 1]

        val_slice = data[slice(indptr, stop)].astype(value_type)
        ind_slice = indices[slice(indptr, stop)]

        x.extend([row_start + i] * len(ind_slice))
        y.extend(ind_slice)
        vals.extend(val_slice)


def optimize_tiledb_array(
    tiledb_array_uri: str,
    config: Optional["tiledb.ctx.Config"] = None,
    verbose: bool = True,
):
    """Optimize TileDB Array.

    Parameters
    ----------
    tiledb_array_uri: str
        URI for the TileDB array.
    verbose: bool
        Boolean indicating whether to use verbose printing.
    """

    import tiledb

    if verbose:
        print(f"Optimizing {tiledb_array_uri}")

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments before consolidation: {}".format(len(frags)))

    if config is None:
        cfg = tiledb.Config(
            {
                "sm.consolidation.steps": 500000,
                "sm.consolidation.step_min_frags": 2,
                "sm.consolidation.step_max_frags": 10,
                "sm.consolidation.buffer_size": 1000000000,  # 1G
                "sm.mem.total_budget": 200000000000,  # 200G
            }
        )
    else:
        cfg = config
    tiledb.consolidate(tiledb_array_uri, config=cfg)
    tiledb.vacuum(tiledb_array_uri)

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments after consolidation: {}".format(len(frags)))


def query_tiledb_df(
    tdb: "tiledb.libtiledb.DenseArrayImpl",
    query_condition: str,
    attrs: Optional[list] = None,
) -> "pandas.DataFrame":
    """Query TileDB DataFrame.

    Parameters
    ----------
    tdb: tiledb.libtiledb.DenseArrayImpl
        TileDB dataframe.
    query_condition: str
        Query condition.
    attrs: list, optional, default: None
        Columns to return in results
    """

    import re
    import numpy as np

    if attrs is not None:
        query = tdb.query(cond=query_condition, attrs=attrs)
    else:
        query = tdb.query(cond=query_condition)
    result = query.df[:]
    re_null = re.compile(pattern="\x00")  # replace null strings with nan
    result = result.replace(regex=re_null, value=np.nan)
    result = result.dropna()

    return result


def adata_from_tiledb(
    cell_idx: Union[list, "numpy.ndarray"],
    tiledb_base_path: str,
    gene_order: Optional[List[str]] = None,
    SAMPLEURI: str = "sample_metadata",
    GENEURI: str = "gene_annotation",
    CELLURI: str = "cell_metadata",
    COUNTSURI: str = "counts",
    config: Optional["tiledb.ctx.Config"] = None,
    lognorm: bool = True,
    target_sum: float = 1e4,
):
    """Constructs an AnnData object from cells in tiledb.

    Parameters
    ----------
    cell_idx: Union[list, "numpy.ndarray"]
        Cell indices in the tiledb.
    tiledb_base_path: str
        Base path of tiledb store
    gene_order: List[str], optional, default: None
        Gene order
    SAMPLEURI: str, default:"sample_metadata"
        Sub path of sample metadata store
    GENEURI: str, default:"gene_annotation"
        Sub path of gene annotation store
    CELLURI: str, default:"cell_metadata"
        Sub path of cell metadata store
    COUNTSURI: str, default:"counts"
        Sub path of count matrix store
    config: tiledb.ctx.Config, optional, default: None
        Custom tiledb config
    lognorm: bool, default: True
        Whether to return log normalized expression instead of raw counts.
    target_sum: float, default: 1e4
        Target sum for log normalization.

    Returns
    -------
    anndata.AnnData
        A data object where counts are in layers["counts"] and X is the lognorm expression

    Examples
    --------
    >>> adata = adata_from_tiledb(cell_idx, gene_order, tiledb_base_path)
    """

    import anndata
    import numpy as np
    import os
    import pandas as pd
    from scipy.sparse import coo_matrix, diags
    import tiledb

    if config is None:
        cfg = tiledb.Config()
        cfg["sm.mem.total_budget"] = 50000000000  # 50G
    else:
        cfg = config

    gene_tdb = tiledb.open(os.path.join(tiledb_base_path, GENEURI), "r", config=cfg)
    genes = (
        gene_tdb.query(attrs=["cellarr_gene_index"])
        .df[:]["cellarr_gene_index"]
        .tolist()
    )
    gene_tdb.close()
    if gene_order is not None:
        gene_indices = []
        for x in gene_order:
            try:
                gene_indices.append(genes.index(x))
            except:
                log.info(f"Gene not found: {x}")
                pass
    else:
        gene_order = genes
        gene_indices = list(range(0, len(genes)))

    # sorted indices are needed for tiledb, keep original indices to unsort later
    cell_idx = np.array(cell_idx)
    sorted_idx = np.argsort(cell_idx)
    original_idx = np.argsort(sorted_idx)
    sorted_cell_idx = cell_idx[sorted_idx]

    cell_tdb = tiledb.open(os.path.join(tiledb_base_path, CELLURI), "r", config=cfg)
    obs = cell_tdb.df[sorted_cell_idx]
    cell_tdb.close()

    matrix_tdb = tiledb.open(os.path.join(tiledb_base_path, COUNTSURI), "r", config=cfg)
    attr = matrix_tdb.schema.attr(0).name
    matrix_shape = (
        matrix_tdb.nonempty_domain()[0][1] + 1,
        matrix_tdb.nonempty_domain()[1][1] + 1,
    )
    results = matrix_tdb.multi_index[sorted_cell_idx, :]
    matrix_tdb.close()

    counts = coo_matrix(
        (results[attr], (results["cell_index"], results["gene_index"])),
        shape=matrix_shape,
    ).tocsr()
    counts = counts[sorted_cell_idx, :]
    counts = counts[:, gene_indices]
    counts = counts[original_idx, :]

    X = counts.astype(np.float32)

    if lognorm:
        # normalize to target sum
        row_sums = np.ravel(X.sum(axis=1))  # row sums as a 1D array
        # avoid division by zero by setting zero sums to one (they will remain zero after normalization)
        row_sums[row_sums == 0] = 1
        # create a sparse diagonal matrix with the inverse of the row sums
        inv_row_sums = diags(1 / row_sums).tocsr()
        # normalize the rows to sum to 1
        normalized_matrix = inv_row_sums.dot(X)
        # scale the rows sum to target_sum
        target_sum = 1e4
        X = normalized_matrix.multiply(target_sum)
        X = X.log1p()

    obs = obs.reindex(cell_idx)
    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_order),
    )
    adata.layers["counts"] = counts.astype(np.int32)

    return adata


def embedding_from_tiledb(
    cell_idx: Union[list, "numpy.ndarray"],
    embedding_tdb_uri: str,
    config: Optional["tiledb.ctx.Config"] = None,
):
    """Get embeddings from a precomputed tiledb.

    Parameters
    ----------
    cell_idx: Union[list, "numpy.ndarray"]
        Cell indices in the tiledb.
    embedding_tdb_uri: str
        Path of tiledb store
    config: tiledb.ctx.Config, optional, default: None
        Custom tiledb config

    Returns
    -------
    numpy.ndarrary
        Array of embeddings

    Examples
    --------
    >>> embedding = embedding_from_tiledb(cell_idx, embedding_tdb_uri)
    """

    import numpy as np
    import tiledb

    if config is None:
        cfg = tiledb.Config()
        cfg["sm.mem.total_budget"] = 50000000000  # 50G
    else:
        cfg = config

    # sorted indices are needed for tiledb, keep original indices to unsort later
    cell_idx = np.array(cell_idx)
    sorted_idx = np.argsort(cell_idx)
    original_idx = np.argsort(sorted_idx)
    sorted_cell_idx = cell_idx[sorted_idx]

    embedding_tdb = tiledb.open(embedding_tdb_uri, "r", config=cfg)
    attr = embedding_tdb.schema.attr(0).name
    embedding = embedding_tdb.query(attrs=[attr], coords=True).multi_index[
        sorted_cell_idx
    ][attr]
    embedding_tdb.close()
    embedding = embedding[original_idx, :]

    return embedding


def pseudobulk_anndata(
    adata: "anndata.AnnData",
    groupby_labels: Union[str, list],
    qc_filters: Optional[dict] = None,
    min_num_cells: int = 1,
    only_orig_genes: bool = False,
):
    """Pseudobulk an AnnData and return a new AnnData.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    groupby_labels: Union[str, list]
        List of labels to groupby prior to pseudobulking.
        For example: ["sample", "tissue", "disease", "celltype_name"]
        will groupby these columns and perform pseudobulking based on these groups.
    qc_filters: dict, optional, default: None
        Dictionary containing cell filters to perform prior to pseudobulking:
            "mito_percent": max percent of reads in mitochondrial genes
            "min_counts": min read count for cell
            "min_genes": min number of genes with reads for cell
            "max_nn_dist": max nearest neighbor distance to a reference label for predicted labels.
    min_num_cells: int, default: 1
        The minimum number of cells in a pseudobulk in order to be considered.
    only_orig_genes: bool, default: False
        Account for an aligned gene space and mask non original genes to the dataset with NaN as
        their pseudobulk. Assumes the original gene list is in adata.uns["orig_genes"].

    Returns
    -------
    anndata.AnnData
        A data object where pseudobulk counts are in layers["counts"] and detection rate is in
        layers["detection"]

    Examples
    --------
    >>> groupby_labels = ["sample", "tissue_raw", "celltype_name"]
    >>> qc_filters = {"mito_percent": 20.0, "min_counts": 1000, "min_genes": 500, "max_nn_dist": 0.03, "max_nn_dist_col": "min_dist"}
    >>> pseudobulk = pseudobulk_anndata(adata, groupby_labels, qc_filters=qc_filters, only_orig_genes=True)
    """

    import anndata
    from collections import Counter
    import numpy as np
    import pandas as pd
    import scanpy as sc

    if "counts" not in adata.layers:
        raise ValueError(f"Raw counts matrix not found in layers['counts'].")

    if qc_filters is not None:
        # determine prefix for mitochondrial genes
        mito_prefix = "MT-"
        if any(adata.var.index.str.startswith("mt-")) is True:
            mito_prefix = "mt-"

        mito_percent = qc_filters.get("mito_percent", 100.0)
        min_counts = qc_filters.get("min_counts", None)
        min_genes = qc_filters.get("min_genes", None)
        max_nn_dist = qc_filters.get("max_nn_dist", 0.03)
        max_nn_dist_col = qc_filters.get("max_nn_dist_col", "nn_dist")

        adata = adata.copy()
        adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
            layer="counts",
        )
        adata = adata[adata.obs["pct_counts_mt"] <= mito_percent].copy()
        if min_counts is not None:
            adata = adata[adata.obs["total_counts"] >= min_counts].copy()
        if min_genes is not None:
            adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
        if max_nn_dist_col in adata.obs.columns:
            adata = adata[adata.obs[max_nn_dist_col] <= max_nn_dist].copy()

    df_sample = (
        adata.obs.groupby(groupby_labels, observed=True)
        .size()
        .reset_index(name="cells")
    )
    pseudobulk = sc.get.aggregate(
        adata, by=groupby_labels, func=["sum", "count_nonzero"], layer="counts"
    )
    pseudobulk.obs = pseudobulk.obs.reset_index(drop=True)
    pseudobulk.obs = pd.merge(pseudobulk.obs, df_sample, on=groupby_labels)

    pseudobulk.layers["counts"] = pseudobulk.layers["sum"].copy()
    del pseudobulk.layers["sum"]
    pseudobulk.layers["detection"] = (
        pseudobulk.layers["count_nonzero"] / pseudobulk.obs["cells"].values[:, None]
    )
    del pseudobulk.layers["count_nonzero"]

    if min_num_cells > 1:
        pseudobulk = pseudobulk[pseudobulk.obs["cells"] >= min_num_cells].copy()
    if only_orig_genes and "uns" in dir(adata) and "orig_genes" in adata.uns:
        orig_genes = set(adata.uns["orig_genes"])
        not_orig_genes_idx = [
            i for i, x in enumerate(adata.var.index.tolist()) if x not in orig_genes
        ]
        pseudobulk.layers["counts"][:, not_orig_genes_idx] = np.nan
        pseudobulk.layers["detection"][:, not_orig_genes_idx] = np.nan

    return pseudobulk


def subset_by_unique_values(
    df: "pandas.DataFrame",
    group_columns: Union[List[str], str],
    value_column: str,
    n: int,
) -> "pandas.DataFrame":
    """Subset a pandas dataframe to only include rows where there are at least
    n unique values from value_column, for each grouping of group_column.

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe.
    group_columns: Union[List[str], str]
        Columns to group by.
    value_column: str
        Column value from which to check the number of instances.
    n: int
        Minimum number of values to be included.

    Returns
    -------
    pandas.DataFrame
        A subsetted dataframe.

    Examples
    --------
    >>> df = subset_by_unique_values(df, "disease", "sample", 10)
    """

    groups = df.groupby(group_columns)[value_column].transform("nunique") >= n

    return df[groups]


def subset_by_frequency(
    df: "pd.DataFrame",
    group_columns: Union[List[str], str],
    n: int,
) -> "pd.DataFrame":
    """Subset the DataFrame to only columns where the group appears at least n times.

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe
    group_columns: Union[List[str], str]
        Columns to group by.
    n: int
        Minimum number of values to be included.

    Returns
    -------
    pandas.DataFrame
        A subsetted dataframe.

    Examples
    --------
    >>> df = subset_by_frequency(df, ["disease", "prediction"], 10)
    """

    freq = df.groupby(group_columns).size()
    hits = freq[freq >= n].index

    return df.set_index(group_columns).loc[hits].reset_index(drop=False)


def categorize_and_sort_by_score(
    df: "pandas.DataFrame",
    name_column: str,
    score_column: str,
    ascending: bool = False,
    topn: Optional[int] = None,
) -> "pandas.DataFrame":
    """Transform column into category, sort, and choose top n

    Parameters
    ----------
    df: "pandas.DataFrame"
        Pandas dataframe.
    name_column: str
        Name of column to sort.
    score_column: str
        Name of score column to sort name_column by.
    ascending: bool
        Sort ascending
    topn: Optional[int], default: None
        Subset to the top n diseases.

    Returns
    -------
    pandas.DataFrame
        A sorted dataframe that is optionally subsetted to top n.

    Examples
    --------
    >>> df = categorize_and_sort_by_score(df, "disease", "Hit Percentage", topn=10)
    """

    mean_scores = (
        df.groupby(name_column)[score_column].mean().sort_values(ascending=ascending)
    )
    df[name_column] = df[name_column].astype("category")
    df[name_column] = df[name_column].cat.set_categories(
        mean_scores.index, ordered=True
    )

    if topn is not None:
        top_values = mean_scores.head(topn).index
        df = df[df[name_column].isin(top_values)]
        # remove unused cats from df
        df[name_column] = df[name_column].cat.remove_unused_categories()

    return df.sort_values(name_column, ascending=ascending)


def clean_tissues(tissues: "pandas.Series") -> "pandas.Series":
    """Mapper to clean tissue names.

    Parameters
    ----------
    tissues: pandas.Series
        A pandas Series containing tissue names.

    Returns
    -------
    pandas.Series
        A pandas Series containing cleaned tissue names.

    Examples
    --------
    >>> data.obs["tissue_simple"] = clean_tissues(data.obs["tissue"]).fillna("other tissue")
    """

    tissue_mapper = {
        "adipose": {
            "omentum",
            "adipose tissue",
            "Fat",
            "omental fat pad",
            "white adipose tissue",
            "subcutaneous adipose tissue",
            "visceral fat",
        },
        "adrenal gland": {"adrenal gland"},
        "airway": {
            "trachea",
            "trachea;bronchus",
            "Trachea",
            "bronchus",
            "nasopharynx",
            "respiratory tract epithelium",
            "bronchiole",
            "inferior nasal concha",
            "nose",
            "nasal turbinal",
            "respiratory airway",
            "trachea;respiratory airway",
            "bronchial epithelial cell",
            "tonsil",
            "dental pulp",
            "gingiva",
            "olfactory epithelium",
            "periodontium",
            "nasal cavity",
        },
        "biliary system": {
            "bile duct",
            "mucosa of gallbladder",
        },
        "bladder": {
            "urinary bladder",
            "Bladder",
            "bladder",
            "urothelium",
            "ureter",
            "urine",
        },
        "blood": {
            "blood",
            "umbilical cord blood",
            "peripheral blood",
            "Blood",
            "venous blood",
        },
        "bone": {
            "bone",
            "bone tissue",
            "head of femur",
            "bone spine",
        },
        "bone marrow": {
            "bone marrow",
            "Bone_Marrow",
        },
        "brain": {
            "brain",
            "cortex",
            "prefrontal cortex",
            "occipital cortex",
            "cerebrospinal fluid",
            "midbrain",
            "spinal cord",
            "superior frontal gyrus",
            "entorhinal cortex",
            "White Matter brain tissue",
            "Entorhinal Cortex",
            "cerebral hemisphere",
            "brain white matter",
            "cerebellum",
            "hypothalamus",
            "dorsal root ganglion",
            "Brodmann (1909) area 9",
            "choroid plexus",
            "striatum",
            "dorsolateral prefrontal cortex",
            "putamen",
            "middle temporal gyrus",
            "frontal cortex",
            "substantia nigra",
            "primary somatosensory cortex",
            "temporal cortex",
            "primary visual cortex",
            "central nervous system",
        },
        "breast": {
            "breast",
            "Mammary",
            "mammary gland",
            "upper outer quadrant of breast",
        },
        "ear": {"tympanic membrane"},
        # "embryo": {
        #     "amniotic fluid",
        #     "embryo",
        #     "blastocyst",
        #     "yolk sac",
        #     "ureteric bud",
        #     "placenta",
        # },
        "esophagus": {
            "esophagus",
            "esophagusmucosa",
            "esophagusmuscularis",
            "esophagus mucosa",
            "esophagus muscularis mucosa",
            "epithelium of esophagus",
        },
        "eye": {
            "eye",
            "uvea",
            "corneal epithelium",
            "retina",
            "Eye",
            "sclera",
            "lacrimal gland",
            "macula lutea proper",
            "peripheral region of retina",
            "fovea centralis",
            "pigment epithelium of eye",
            "CD31_choroid",
            "retinal pigment epithelium",
            "Optic Nerve",
            "Peripapillary Sclera",
            "Retinal Pigment Epithelium/Choroid, peripheral retina",
            "Sclera",
            "Optic Nerve Head",
            "Retinal Pigment Epithelium/Choroid",
            "Retinal Pigment Epithelium/Choroid, macula",
            "chorioretinal region",
            "retinal neural layer",
        },
        "stomach": {"stomach"},
        "gut": {
            "colon",
            "ascending colon",
            "sigmoid colon",
            "large intestine",
            "small intestine",
            "intestine",
            "Small_Intestine",
            "Large_Intestine",
            "ileum",
            "right colon",
            "left colon",
            "transverse colon",
            "digestive tract",
            "caecum",
            "jejunum",
            "jejunum ",
            "descending colon",
            "rectum",
            "colonic mucosa",
            "mucosa of descending colon",
        },
        "heart": {
            "heart",
            "aorta",
            "cardiac muscle of left ventricle",
            "Heart",
            "heart left ventricle",
            "pulmonary artery",
            "cardiac ventricle",
            "heart right ventricle",
            "left cardiac atrium",
            "right cardiac atrium",
            "apex of heart",
            "interventricular septum",
        },
        "joint": {
            "synovial fluid",
            "cartilage tissue",
            "portion of cartilage tissue in tibia",
            "layer of synovial tissue",
            "synovial membrane of synovial joint",
        },
        "kidney": {
            "adult mammalian kidney",
            "kidney",
            "Kidney",
            "inner medulla of kidney",
            "outer cortex of kidney",
            "renal medulla",
            "cortex of kidney",
            "renal pelvis",
            "kidney blood vessel",
            "renal papilla",
        },
        "liver": {
            "liver",
            "Liver",
            "caudate lobe of liver",
            "right lobe of liver",
            "left lobe of liver",
        },
        "lung": {
            "lung",
            "alveolar system",
            "lung parenchyma",
            "BAL",
            "Lung",
            "Parenchymal lung tissue",
            "Distal",
            "Proximal",
            "Intermediate",
            "lower lobe of lung",
            "upper lobe of lung",
            "upper lobe of left lung",
            "upper lobe of right lung",
            "lower lobe of right lung",
            "lower lobe of left lung",
            "left lung",
            "right lung",
            "lingula of left lung",
        },
        "lymph node": {
            "lymph node",
            "axillary lymph node",
            "Lymph_Node",
            "craniocervical lymph node",
            "thoracic lymph node",
            "mesenteric lymph node",
        },
        "male reproduction": {
            "male reproductive gland",
            "testis",
            "prostate gland",
            "epididymis epithelium",
            "Prostate",
            "prostate",
            "peripheral zone of prostate",
            "transition zone of prostate;urethra",
            "transition zone of prostate",
        },
        "female reproduction": {
            "ovary",
            "tertiary ovarian follicle",
            "ovarian follicle",
            "fimbria of uterine tube",
            "ampulla of uterine tube",
            "isthmus of fallopian tube",
            "fallopian tube",
            "uterus",
            "Uterus",
        },
        "muscle": {
            "psoas muscle",
            "muscle tissue",
            "gastrocnemius",
        },
        "pancreas": {
            "pancreas",
            "Pancreas",
            "islet of Langerhans",
            "exocrine pancreas",
        },
        "peritoneum": {
            "peritoneum",
            "pleural effusion",
        },
        "skin": {
            "skin of body",
            "skin epidermis",
            "skin of prepuce of penis",
            "scrotum skin",
            "Skin",
            "skin",
            "skin of leg",
            "zone of skin",
        },
        "spleen": {
            "spleen",
            "Spleen",
        },
        "thymus": {
            "thymus",
            "Thymus",
        },
        "vasculature": {
            "vasculature",
            "mesenteric artery",
            "umbilical vein",
            "Vasculature",
            "carotid artery segment",
            "posterior vena cava",
        },
    }
    term2simple = {}
    for tissue_simplified, children in tissue_mapper.items():
        for child in children:
            term2simple[child.lower()] = tissue_simplified

    return tissues.str.lower().map(term2simple)


def clean_diseases(diseases: "pandas.Series") -> "pandas.Series":
    """Mapper to clean disease names.

    Parameters
    ----------
    diseases: pandas.Series
        A pandas Series containing disease names.

    Returns
    -------
    pandas.Series
        A pandas Series containing cleaned disease names.

    Examples
    --------
    >>> data.obs["disease_simple"] = clean_diseases(data.obs["disease"]).fillna("healthy")
    """

    disease_mapper = {
        "healthy": {"healthy", "", "NA"},
        "Alzheimer's": {"Alzheimer's disease"},
        "COVID-19": {"COVID-19"},
        "ILD": {
            "pulmonary fibrosis",
            "idiopathic pulmonary fibrosis",
            "interstitial lung disease",
            "systemic scleroderma;interstitial lung disease",
            "fibrosis",
            "hypersensitivity pneumonitis",
            "Idiopathic pulmonary arterial hypertension",
        },
        "cancer": {
            "head and neck squamous cell carcinoma",
            "renal cell adenocarcinoma",
            "hepatocellular carcinoma",
            "B-cell acute lymphoblastic leukemia",
            "glioma",
            "ovarian serous carcinoma",
            "neuroblastoma",
            "pancreatic carcinoma",
            "melanoma",
            "multiple myeloma",
            "Gastrointestinal stromal tumor",
            "adenocarcinoma",
            "pancreatic ductal adenocarcinoma",
            "chronic lymphocytic leukemia",
            "Uveal Melanoma",
            "Myelofibrosis",
            "acute myeloid leukemia",
            "acute lymphoblastic leukemia",
            "precursor B-cell acute lymphoblastic leukemia",
            "T-cell acute lymphoblastic leukemia",
            "chronic myelogenous leukemia",
            "B-cell lymphoma",
            "precursor T-cell lymphoblastic leukemia-lymphoma",
            "human papilloma virus infection;head and neck squamous cell carcinoma",
            "squamous cell carcinoma",
            "Tonsillar Squamous Cell Carcinoma",
            "invasive breast ductal carcinoma",
            "basal cell carcinoma",
            "brain glioblastoma;non-small cell lung carcinoma",
            "renal cell carcinoma",
            "non-small cell lung carcinoma",
            "colorectal cancer",
            "esophageal carcinoma",
            "liver neoplasm;Uveal Melanoma",
            "glioblastoma multiforme",
            "Ewing sarcoma",
            "medulloblastoma",
            "brain glioblastoma",
            "breast neoplasm",
            "lung adenocarcinoma",
            "lung cancer",
            "nasopharyngeal neoplasm",
            "small cell lung carcinoma",
            "breast cancer",
            "prostate cancer",
            "gastric cancer",
            "gastric carcinoma",
            "bladder carcinoma",
            "urinary bladder cancer",
            "Pleuropulmonary blastoma",
            "cutaneous squamous cell carcinoma",
            "Merkel cell skin cancer",
            "urothelial neoplasm",
            "alveolar rhabdomyosarcoma",
            "myeloid neoplasm",
            "Sezary's disease",
            "essential thrombocythemia",
        },
        "MS": {"multiple sclerosis"},
        "dengue": {"dengue disease"},
        "HIV": {
            "HIV enteropathy",
            "HIV infection",
        },
        "IBD": {
            "Crohn's disease",
            "ulcerative colitis",
        },
        "SLE": {"systemic lupus erythematosus"},
        "scleroderma": {"scleroderma"},
        "LCH": {"Langerhans Cell Histiocytosis"},
        "NAFLD": {"non-alcoholic fatty liver disease", "non-alcoholic steatohepatitis"},
        "Kawasaki disease": {"mucocutaneous lymph node syndrome"},
        "eczema": {"atopic eczema"},
        "sepsis": {"septic shock"},
        "obesity": {"obesity"},
        "DRESS": {"drug hypersensitivity syndrome"},
        "hidradenitis suppurativa": {"hidradenitis suppurativa"},
        "diabetes": {
            "type II diabetes mellitus",
            "type 2 diabetes mellitus",
            "diabetes mellitus",
            "Wolfram syndrome",
        },
        "biliary atresia": {"Biliary atresia"},
        "monoclonal gammopathy": {"monoclonal gammopathy"},
        "psoriatic arthritis": {"psoriatic arthritis"},
        "RA": {"rheumatoid arthritis"},
        "osteoarthritis": {"osteoarthritis"},
        "periodontitis": {"periodontitis"},
        "LAM": {"Lymphangioleiomyomatosis"},
        "Parkinson's": {
            "Parkinson's disease",
            "Parkinson's Disease",
        },
        "cardiomyopathy": {
            "cardiomyopathy",
            "arrhythmogenic right ventricular cardiomyopathy",
            "dilated cardiomyopathy",
        },
    }

    term2simple = {}
    for disease_simplified, children in disease_mapper.items():
        for child in children:
            term2simple[child.lower()] = disease_simplified

    return diseases.str.lower().map(term2simple)
