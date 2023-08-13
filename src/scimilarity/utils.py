from typing import Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import pegasusio as pgio
import scanpy as sc
from numba import njit
from scipy.sparse import csr_matrix


def check_dataset(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    target_gene_order: np.ndarray,
    gene_overlap_threshold: int = 10000,
):
    """Check dataset to see if it able to be processed.

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: numpy.ndarray
        An array containing the gene space.
    gene_overlap_threshold: int, default: 10000
        The minimum number of genes in common between data and target_gene_order to be valid.

    Examples
    --------
    >>> ca = CellAnnotation(model_path="/opt/data/model")
    >>> check_dataset(data, ca.gene_order, gene_overlap_threshold=10000)
    """

    if type(data) not in [anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData]:
        raise ValueError(f"Unknown data type {type(data)}.")

    # check gene overlap
    n_genes = sum(data.var.index.isin(target_gene_order))

    if n_genes < gene_overlap_threshold:
        raise RuntimeError(
            f"Dataset incompatible: gene overlap less than {gene_overlap_threshold}"
        )

    # check if count matrix exists
    counts_exist = False
    if isinstance(data, pgio.MultimodalData):
        data = data.get_data(data.list_data()[0])
    if isinstance(data, pgio.UnimodalData):
        counts_exist = "counts" in data.list_keys() or "raw.X" in data.list_keys()
    if isinstance(data, anndata.AnnData):
        counts_exist = "counts" in data.layers

    if not counts_exist:
        raise RuntimeError("Dataset incompatible: no counts matrix found")


def lognorm_counts(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    clip_threshold: Optional[float] = None,
    clip_threshold_percentile: Optional[float] = None,
) -> Union[anndata.AnnData, pgio.UnimodalData]:
    """Log normalize the gene expression raw counts (per 10k).

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    clip_threshold: float, optional
        Clip the data to the given max value.
    clip_threshold_percentile: float, optional
        Clip the data to the value at the given data percentile.

    Returns
    -------
    pegasusio.UnimodalData, anndata.AnnData
        A data object with normalized data that is ready to be used in further processes.

    Examples
    --------
    >>> data = lognorm_counts(data)
    """

    if type(data) not in [anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData]:
        raise ValueError(f"Unknown data type {type(data)}.")

    return_unimodaldata = False
    if isinstance(data, pgio.MultimodalData):
        data = data.get_data(data.list_data()[0])
    if isinstance(data, pgio.UnimodalData):
        return_unimodaldata = True
        data = data.to_anndata()

    if "counts" not in data.layers and "raw.X" not in data.layers:
        raise ValueError(f"Raw counts matrix not found.")

    if "raw.X" in data.layers:
        data.layers["counts"] = data.layers["raw.X"].copy()
        del data.layers["raw.X"]
    data.X = data.layers["counts"].copy()

    # winsorize data
    if clip_threshold_percentile:
        clip_threshold = np.percentile(data.X.data, clip_threshold_percentile)
    if clip_threshold:
        data.X[data.X > clip_threshold] = clip_threshold

    # log norm
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)

    if return_unimodaldata:
        data = pgio.UnimodalData(data)
    return data


def filter_cells(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    min_genes: int = 400,
    mito_prefix: str = None,
    mito_percent: float = 30.0,
) -> Union[anndata.AnnData, pgio.MultimodalData]:
    """QC filter the dataset from gene expression raw counts.

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    min_genes: int, default: 400
        The minimum number of expressed genes in order not to be filtered out.
    mito_prefix: str, optional
        The prefix to represent mitochondria genes. Typically "MT-" or "mt-".
        If None, it will try to infer whether it is either "MT-" or "mt-".
    mito_percent: float, default: 30.0
        The maximum percent allowed expressed mitochondria genes in order not to be filtered out.

    Returns
    -------
    pegasusio.MultimodalData, anndata.AnnData
        A data object with cells filtered out based on QC metrics that is ready to be used
        in further processes.

    Examples
    --------
    >>> data = filter_cells(data)
    """

    if type(data) not in [anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData]:
        raise ValueError(f"Unknown data type {type(data)}.")

    return_unimodaldata = False
    if isinstance(data, pgio.MultimodalData):
        data = data.get_data(data.list_data()[0])
    if isinstance(data, pgio.UnimodalData):
        return_unimodaldata = True
        data = data.to_anndata()

    if "counts" not in data.layers and "raw.X" not in data.layers:
        raise ValueError(f"Raw counts matrix not found.")

    if "raw.X" in data.layers:
        data.layers["counts"] = data.layers["raw.X"].copy()
        del data.layers["raw.X"]

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

    if return_unimodaldata:
        data = pgio.UnimodalData(data)
    return data


def process_data(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    resolution: float = 1.3,
) -> Union[anndata.AnnData, pgio.UnimodalData]:
    """Process the dataset: hvf selection, pca, umap, clustering

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    n_top_genes: int, default: 2000
        The number of highly variable genes to select.
    batch_key: str, optional
        The obs key which holds batch information for highly variable gene selection.
    resolution: float, default: 1.3
        The leiden clustering resolution.

    Returns
    -------
    pegasusio.UnimodalData, anndata.AnnData
        A data object where highly variable genes are in obs["highly_variable_features"],
        pca data is in obsm["X_pca"], umap data is in obsm["X_umap"], and clustering
        data is in obs["leiden_labels"].

    Examples
    --------
    >>> data = filter_cells(data)
    """

    return_unimodaldata = False
    if isinstance(data, pgio.MultimodalData):
        data = data.get_data(data.list_data()[0])
    if isinstance(data, pgio.UnimodalData):
        return_unimodaldata = True
        data = data.to_anndata()

    # pca
    sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, batch_key=batch_key)
    sc.tl.pca(data)

    # umap
    sc.pp.neighbors(data, use_rep="X_pca")
    sc.tl.umap(data)

    # clustering
    sc.tl.leiden(data, resolution=resolution)

    if return_unimodaldata:
        data = pgio.UnimodalData(data)
    return data


def switch_gene_symbols(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    var_key: str,
) -> Union[anndata.AnnData, pgio.UnimodalData]:
    """Switch to a different set of gene symbols, contained in data.var

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    var_key: str
        The var key which holds the symbol information.

    Returns
    -------
    pegasusio.UnimodalData, anndata.AnnData
        A data object where the var index is set to those in var_key, with
        nulls and duplicates removed.

    Examples
    --------
    >>> data = switch_gen_symbols(data, "symbol")
    """
    data.var = data.var.set_index(var_key, drop=False)
    return data[:, ~(data.var.index.isnull() | data.var.index.duplicated())].copy()


@njit(fastmath=True, cache=True)
def select_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    indexer: np.ndarray,
    new_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset a scipy.sparse.csr_matrix based on index.

    Parameters
    ----------
    data: numpy.ndarray
        Data array of the matrix.
    indices: numpy.ndarray
        Index array of the matrix.
    indptr: numpy.ndarray
        Index pointer array of the matrix.
    indexer: numpy.ndarray
        The subset index array.
    new_size: int
        The size of the new matrix.

    Returns
    -------
    numpy.ndarray
        The new data array.
    numpy.ndarray
        The new index array.
    numpy.ndarray
        The new index pointer array.

    Examples
    --------
    >>> data = filter_cells(data)
    """

    data_new = np.zeros_like(data[0:new_size])
    indices_new = np.zeros_like(indices[0:new_size])
    indptr_new = np.zeros_like(indptr)

    cnt = 0
    for i in range(indptr.size - 1):
        indptr_new[i] = cnt
        for j in range(indptr[i], indptr[i + 1]):
            new_idx = indexer[indices[j]]
            if new_idx >= 0:
                data_new[cnt] = data[j]
                indices_new[cnt] = new_idx
                cnt += 1
    indptr_new[indptr.size - 1] = cnt

    return data_new, indices_new, indptr_new


def align_dataset(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    target_gene_order: np.ndarray,
    keep_obsm: bool = False,
    gene_overlap_threshold: int = 5000,
) -> Union[anndata.AnnData, pgio.UnimodalData]:
    """Align the gene space to the target gene order.

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: numpy.ndarray
        An array containing the gene space.
    keep_obsm: bool, default: False
        Retain the original data's obsm matrices in output.
    gene_overlap_threshold: int, default 5000
        The minimum number of genes in common between data and target_gene_order to be valid.

    Returns
    -------
    pegasusio.UnimodalData, anndata.AnnData
        A data object with aligned gene space ready to be used for embedding cells.

    Examples
    --------
    >>> ca = CellAnnotation(model_path="/opt/data/model")
    >>> align_dataset(data, ca.gene_order)
    """

    if isinstance(data, pgio.MultimodalData):
        data = data.get_data(data.list_data()[0])

    # raise an error if not enough genes from target_gene_order exists
    if sum(data.var.index.isin(target_gene_order)) < gene_overlap_threshold:
        raise RuntimeError(
            f"Dataset incompatible: gene overlap less than {gene_overlap_threshold}. Check that var.index uses gene symbols."
        )

    # check for negatives in expression data
    if np.min(data.X) < 0:
        raise RuntimeError(f"Dataset contains negative values in expression matrix X.")

    # return data if already aligned
    if data.var.index.values.tolist() == target_gene_order:
        return data

    shell = None
    if isinstance(data, pgio.UnimodalData):
        mat = data.X
        obs_field = data.obs
        var_field = pd.DataFrame(index=target_gene_order)

        indexer = var_field.index.get_indexer(data.var_names)
        new_size = (indexer[mat.indices] >= 0).sum()
        data_new, indices_new, indptr_new = select_csr(
            mat.data, mat.indices, mat.indptr, indexer, new_size
        )
        data_matrix = csr_matrix(
            (data_new, indices_new, indptr_new),
            shape=(mat.shape[0], len(target_gene_order)),
        )
        data_matrix.sort_indices()

        obs_field.index.name = "barcodekey"
        var_field.index.name = "featurekey"
        shell = pgio.UnimodalData(
            barcode_metadata=obs_field,
            feature_metadata=var_field,
            matrices={"X": data_matrix},
        )

        if "counts" in data.list_keys():
            mat = data.get_matrix("counts")
            data_new, indices_new, indptr_new = select_csr(
                mat.data, mat.indices, mat.indptr, indexer, new_size
            )
            shell.add_matrix(
                "counts",
                csr_matrix(
                    (data_new, indices_new, indptr_new),
                    shape=(mat.shape[0], len(target_gene_order)),
                ),
            )
        if "raw.X" in data.list_keys():
            mat = data.get_matrix("raw.X")
            data_new, indices_new, indptr_new = select_csr(
                mat.data, mat.indices, mat.indptr, indexer, new_size
            )
            shell.add_matrix(
                "raw.X",
                csr_matrix(
                    (data_new, indices_new, indptr_new),
                    shape=(mat.shape[0], len(target_gene_order)),
                ),
            )
        if keep_obsm and hasattr(data, "obsm"):
            shell.obsm = data.obsm

    if isinstance(data, anndata.AnnData):
        shell = anndata.AnnData(
            X=csr_matrix((0, len(target_gene_order))),
            var=pd.DataFrame(index=target_gene_order),
            dtype=np.float32,
        )
        shell = anndata.concat(
            (shell, data[:, data.var.index.isin(shell.var.index)]), join="outer"
        )
        if not keep_obsm and hasattr(data, "obsm"):
            delattr(shell, "obsm")

    if data.var.shape[0] == 0:
        raise RuntimeError(f"Empty gene space detected.")

    return shell


def get_centroid(sparse_counts_mat: csr_matrix) -> np.ndarray:
    """Get the centroid for a raw counts matrix in scipy.sparse.csr_matrix format.

    Parameters
    ----------
    sparse_counts_mat: scipy.sparse.csr_matrix
        Sparse matrix of raw gene expression counts.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of the log normalized (1e4) centroid.

    Examples
    --------
    >>> centroid = get_centroid(data.get_matrix("counts"))
    >>> centroid = get_centroid(data.layers["counts"])
    """
    summed_counts = sparse_counts_mat.sum(axis=0).A
    normalization_factor = sparse_counts_mat.sum(axis=1).A.sum()
    centroid = np.log(1 + 1e4 * summed_counts / normalization_factor)
    return centroid


def get_cluster_centroids(
    data: Union[anndata.AnnData, pgio.UnimodalData, pgio.MultimodalData],
    target_gene_order: np.ndarray,
    cluster_key: str,
    cluster_label: Optional[str] = None,
    skip_null: bool = True,
) -> Tuple[np.ndarray, list]:
    """Get centroids of clusters based on raw read counts.

    Parameters
    ----------
    data: pegasusio.MultimodalData, pegasusio.UnimodalData, anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
    target_gene_order: numpy.ndarray
        An array containing the gene space.
    cluster_key: str
        The obs column key that contains cluster labels.
    cluster_label: optional, str
        The cluster label of interest. If None, then get the centroids of
        all clusters, otherwise get only the centroid for the cluster
        of interest
    skip_null: bool, default: True
        Whether to skip cells with null/nan cluster labels.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of the log normalized (1e4) cluster centroids.
    list
        A list of cluster labels corresponding to the order returned in centroids.

    Examples
    --------
    >>> centroids, cluster_idx = get_cluster_centroids(data, gene_order, "leiden_labels")
    """

    centroids = []
    cluster_idx = []

    aligned_data = align_dataset(data, target_gene_order)
    if skip_null:
        aligned_data = aligned_data[aligned_data.obs[cluster_key].notnull()].copy()
    aligned_data.obs[cluster_key] = aligned_data.obs[cluster_key].astype(str)

    if isinstance(aligned_data, pgio.UnimodalData):
        for i in set(aligned_data.obs[cluster_key]):
            if cluster_label is not None:
                i = cluster_label

            cluster_idx.append(i)
            if "counts" in aligned_data.list_keys():
                centroids.append(
                    get_centroid(
                        aligned_data[aligned_data.obs[cluster_key] == i]
                        .copy()
                        .get_matrix("counts")
                    )
                )
            elif "raw.X" in aligned_data.list_keys():
                centroids.append(
                    get_centroid(
                        aligned_data[aligned_data.obs[cluster_key] == i]
                        .copy()
                        .get_matrix("raw.X")
                    )
                )
            else:
                raise RuntimeError("Dataset incompatible: no counts matrix found")

            if cluster_label is not None:
                break

    if isinstance(aligned_data, anndata.AnnData):
        for i in set(aligned_data.obs[cluster_key]):
            if cluster_label is not None:
                i = cluster_label

            cluster_idx.append(i)
            if "counts" in aligned_data.layers:
                centroids.append(
                    get_centroid(
                        aligned_data[aligned_data.obs[cluster_key] == i]
                        .copy()
                        .layers["counts"]
                    )
                )
            else:
                raise RuntimeError("Dataset incompatible: no counts matrix found")

            if cluster_label is not None:
                break

    centroids = np.vstack(centroids)

    if np.isnan(centroids).any():
        raise RuntimeError(f"NaN detected in centroids.")

    return centroids, cluster_idx
