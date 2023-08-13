from typing import Dict, List, Optional, Tuple, Union

import numcodecs
import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

ARRAY_FORMATS = {
    "csr_matrix": csr_matrix,
    "csc_matrix": csc_matrix,
    "coo_matrix": coo_matrix,
}


class ZarrDataset:
    """A class that reads zarr datasets saved by AnnData from disk.
    Adapted from https://github.com/lilab-bcb/backedarray

    Example
    -------
        zarr_data = ZarrDataset("/data/dataset.zarr")
    """

    def __init__(self, store_path: str, mode: str = "r"):
        """Constructor.

        Parameters
        ----------
        store_path: str
            Path to folder containing all the dataset in zarr format.
        mode: str, default: "r"
            The persistence mode for the zarr dataset.

        Examples
        --------
        >>>  zarr_data = ZarrDataset("/data/dataset.zarr")
        """

        self.store_path = zarr.DirectoryStore(store_path)
        self.root = zarr.open_group(
            self.store_path, mode=mode, chunk_store=self.store_path
        )

    @property
    def dataset_info(self) -> Dict[str, list]:
        """Get a summary of the dataset info.

        Returns
        -------
        dict
            A dict containing information on the content of the dataset, such
            as keys in the various object attributes.

        Examples
        --------
        >>> zarr_data.dataset_info
        """

        d = {}
        if "var" in self.root.keys():
            d["var"] = list(self.root["var"].keys())
        if "obs" in self.root.keys():
            d["obs"] = list(self.root["obs"].keys())
        if "X" in self.root.keys():
            d["shape"] = self.root["X"].attrs["shape"]
        if "layers" in self.root:
            d["layers"] = list(self.root["layers"].keys())
        if "uns" in self.root:
            d["uns"] = list(self.root["uns"].keys())
        return d

    @property
    def X(self):
        if "X" in self.root.keys():
            X = self.root["X"]
            return self.get_matrix(X)
        return None

    @property
    def X_copy(self):
        if "X" in self.root.keys():
            X = self.root["X"]
            return self.get_matrix(X, in_mem=True)
        return None

    @property
    def counts(self):
        if "layers" in self.root.keys():
            layers = self.root["layers"]
            if "counts" in layers.keys():
                counts = layers["counts"]
                return self.get_matrix(counts)
        return None

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the gene expression matrix.

        Returns
        -------
        Tuple[int, int]
            A tuple of the form [nrows x ncolumns].

        Examples
        --------
        >>> zarr_data.shape
        """

        if "X" in self.root.keys():
            return self.root["X"].attrs["shape"]
        return None

    @property
    def var_index(self) -> pd.Index:
        """Get the var index.

        Returns
        -------
        pandas.Index
            A pandas Index containing the var index.

        Examples
        --------
        >>> zarr_data.var_index
        """

        if "var" in self.root.keys():
            return self.get_annotation_index(self.root["var"])
        return None

    @property
    def var(self) -> pd.DataFrame:
        """Get the var dataframe.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the var data.

        Examples
        --------
        >>> zarr_data.var
        """

        if "var" in self.root.keys():
            var = pd.DataFrame(
                {x: self.get_var(x) for x in self.dataset_info["var"] if x != "_index"}
            )
            var.index = self.var_index
            return var
        return None

    @property
    def obs_index(self) -> pd.Index:
        """Get the obs index.

        Returns
        -------
        pandas.Index
            A pandas Index containing the obs index.

        Examples
        --------
        >>> zarr_data.obs_index
        """

        if "obs" in self.root.keys():
            return self.get_annotation_index(self.root["obs"])
        return None

    @property
    def obs(self) -> pd.DataFrame:
        """Get the obs dataframe.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the obs data.

        Examples
        --------
        >>> zarr_data.obs
        """

        if "obs" in self.root.keys():
            obs = pd.DataFrame(
                {x: self.get_obs(x) for x in self.dataset_info["obs"] if x != "_index"}
            )
            obs.index = self.obs_index
            return obs
        return None

    def set_X(self, matrix: Union[csr_matrix, csc_matrix, coo_matrix]) -> None:
        """Set the X sparse matrix.
           This will overwrite the current stored X.

        Parameters
        ----------
        matrix: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.coo_matrix]
            The sparse matrix.

        Examples
        --------
        >>> zarr_data.set_X(matrix)
        """

        X = self.root.create_group("X", overwrite=True)
        self.set_matrix(X, matrix)

    def append_X(
        self, matrix: Union[csr_matrix, csc_matrix], axis: Optional[int] = None
    ) -> None:
        """Append to the X sparse matrix.

        Parameters
        ----------
        matrix: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
            The sparse matrix.

        Notes
        -----
           Only row-wise concatentation for csr_matrix.
           Only col-wise concatentation for csc_matrix.

        Examples
        --------
        >>> zarr_data.append_X(matrix)
        """

        if "X" in self.root.keys():
            self.append_matrix(self.root["X"], matrix, axis)

    def get_var(self, column: str) -> Union[np.ndarray, pd.Categorical]:
        """Get data.var[column] data.

        Parameters
        ----------
        column: str,
            Column name in var.

        Returns
        -------
        pandas.Series
            A pandas series containing the var data.

        Examples
        --------
        >>> zarr_data.get_var("symbol")
        """

        if "var" in self.root.keys():
            return self.get_annotation_column(self.root["var"], column)
        return None

    def get_obs(self, column: str) -> Union[np.ndarray, pd.Categorical]:
        """Get data.obs[column] data.

        Parameters
        ----------
        column: str,
            Column in obs.

        Returns
        -------
        pandas.Series
            A pandas series containing the obs data.

        Examples
        --------
        >>> zarr_data.get_obs("celltype_name")
        """

        if "obs" in self.root.keys():
            return self.get_annotation_column(self.root["obs"], column)
        return None

    def get_uns(self, key: str):
        """Get data.uns[key] data.

        Parameters
        ----------
        key: str,
            Key for the field in uns.

        Returns
        -------
        undefined
            The data in data.uns[key] in the format it was stored as.

        Examples
        --------
        >>> zarr_data.get_uns("orig_genes")
        """

        if "uns" in self.root.keys():
            group = self.root["uns"]
            if key in group:
                return group[key][...]
        return None

    def get_cell(self, idx: int) -> Union[csr_matrix, csc_matrix]:
        """Get gene expression data for one cell as sparse matrix.

        Parameters
        ----------
        idx: int,
            Numerical index of the cell.

        Examples
        --------
        >>> zarr_data.get_cell(42)
        """

        if "X" in self.root.keys():
            X = self.root["X"]
            encoding_type = X.attrs["encoding-type"]

            if encoding_type == "csr_matrix":
                return self.row_slice_csr(X, idx)
            elif encoding_type == "coo_matrix":
                return self.slice_coo(X, idx, axis=0)
            raise RuntimeError(
                f"Unsupported encoding-type for row slicing: {encoding_type}."
            )
        return None

    def get_layer_cell(self, layer_key: str, idx: int) -> Union[csr_matrix, csc_matrix]:
        """Get data for one cell from a layer as sparse matrix.

        Parameters
        ----------
        idx: int,
            Numerical index of the cell.

        Examples
        --------
        >>> zarr_data.get_layer_cell(42)
        """

        if "layers" in self.root.keys():
            layers = self.root["layers"]
            if layer_key in layers.keys():
                X = layers[layer_key]
                encoding_type = X.attrs["encoding-type"]

                if encoding_type == "csr_matrix":
                    return self.row_slice_csr(X, idx)
                elif encoding_type == "coo_matrix":
                    return self.slice_coo(X, idx, axis=0)
                raise RuntimeError(
                    f"Unsupported encoding-type for row slicing: {encoding_type}."
                )
        return None

    def get_gene(self, idx: int) -> Union[csr_matrix, csc_matrix]:
        """Get gene expression data for one gene as sparse matrix.

        Parameters
        ----------
        idx: int,
            Numerical index of the gene.

        Examples
        --------
        >>> zarr_data.get_gene(42)
        """

        if "X" in self.root.keys():
            X = self.root["X"]
            encoding_type = X.attrs["encoding-type"]

            if encoding_type == "csc_matrix":
                return self.col_slice_csc(X, idx)
            elif encoding_type == "coo_matrix":
                return self.slice_coo(X, idx, axis=1)
            raise RuntimeError(
                f"Unsupported encoding-type for col slicing: {encoding_type}."
            )
        return None

    def get_layer_gene(self, layer_key: str, idx: int) -> Union[csr_matrix, csc_matrix]:
        """Get data for one gene from a layer as sparse matrix.

        Parameters
        ----------
        idx: int,
            Numerical index of the cell.

        Examples
        --------
        >>> zarr_data.get_layer_gene(42)
        """

        if "layers" in self.root.keys():
            layers = self.root["layers"]
            if layer_key in layers.keys():
                X = layers[layer_key]
                encoding_type = X.attrs["encoding-type"]

                if encoding_type == "csc_matrix":
                    return self.col_slice_csc(X, idx)
                elif encoding_type == "coo_matrix":
                    return self.slice_coo(X, idx, axis=1)
                raise RuntimeError(
                    f"Unsupported encoding-type for col slicing: {encoding_type}."
                )
        return None

    def slice_with(self, group, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = group["data"]
        indices = group["indices"]
        indptr = group["indptr"]

        s = slice(*(indptr[idx : idx + 2]))
        new_data = data[s]
        new_indices = indices[s]
        new_indptr = [0, indptr[idx + 1] - indptr[idx]]
        return (new_data, new_indices, new_indptr)

    def slice_across(
        self, group, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = group["data"]
        indices = group["indices"]
        indptr = group["indptr"]

        s = [i for i, x in enumerate(indices) if x == idx]
        new_data = data[s]
        new_indices = [0] * len(new_data)

        new_indptr = [0]
        pos = 0
        for i in range(len(indptr) - 1):
            r = slice(*(indptr[i : i + 2]))
            if any([j >= r.start and j < r.stop for j in s[pos:]]):
                pos += 1
            new_indptr.append(pos)
        return (new_data, new_indices, new_indptr)

    def row_slice_csr(self, group, idx: int) -> csr_matrix:
        new_data, new_indices, new_indptr = self.slice_with(group, idx)
        shape = group.attrs["shape"]
        return csr_matrix((new_data, new_indices, new_indptr), shape=(1, shape[1]))

    def col_slice_csc(self, group: csc_matrix, idx: int) -> csc_matrix:
        new_data, new_indices, new_indptr = self.slice_with(group, idx)
        shape = group.attrs["shape"]
        return csc_matrix((new_data, new_indices, new_indptr), shape=(shape[0], 1))

    def slice_coo(self, group, idx: int, axis: int) -> coo_matrix:
        data = group["data"]
        row = group["row"]
        col = group["col"]
        shape = group.attrs["shape"]

        assert axis in [0, 1], "axis must be 0 or 1 for coo_matrix."
        if axis == 0:  # row slice
            s = [i for i, x in enumerate(row) if x == idx]
            new_data = data[s]
            new_row = [0] * len(new_data)
            new_col = col[s]
            return coo_matrix((new_data, (new_row, new_col)), shape=(1, shape[1]))
        elif axis == 1:  # col slice
            s = [i for i, x in enumerate(col) if x == idx]
            new_data = data[s]
            new_row = row[s]
            new_col = [0] * len(new_data)
            return coo_matrix((new_data, (new_row, new_col)), shape=(shape[0], 1))
        return None

    def get_matrix(
        self, group, in_mem: bool = False
    ) -> Union[csr_matrix, csc_matrix, coo_matrix]:
        encoding_type = group.attrs["encoding-type"]
        mtx = ARRAY_FORMATS[encoding_type](
            tuple(group.attrs["shape"]), dtype=group["data"].dtype
        )
        if encoding_type in ["csr_matrix", "csc_matrix"]:
            mtx.data = group["data"]
            mtx.indices = group["indices"]
            mtx.indptr = group["indptr"]
            if in_mem:
                mtx.data = mtx.data[...]
                mtx.indices = mtx.indices[...]
                mtx.indptr = mtx.indptr[...]
        elif encoding_type in ["coo_matrix"]:
            mtx.data = group["data"]
            mtx.row = group["row"]
            mtx.col = group["col"]
            if in_mem:
                mtx.data = mtx.data[...]
                mtx.row = mtx.row[...]
                mtx.col = mtx.col[...]
        else:
            raise RuntimeError(f"Unsupported encoding-type: {encoding_type}.")
        return mtx

    def set_matrix(
        self, group, matrix: Union[csr_matrix, csc_matrix, coo_matrix]
    ) -> None:
        encoding_type = type(matrix).__name__
        group.attrs.setdefault("encoding-type", encoding_type)
        group.attrs.setdefault("encoding-version", "0.1.0")
        group.attrs.setdefault("shape", list(matrix.shape))

        if encoding_type in ["csr_matrix", "csc_matrix"]:
            group.create_dataset("data", data=matrix.data, dtype=matrix.data.dtype)
            group.create_dataset(
                "indptr", data=matrix.indptr, dtype=matrix.indptr.dtype
            )
            group.create_dataset(
                "indices", data=matrix.indices, dtype=matrix.indices.dtype
            )
        elif encoding_type in ["coo_matrix"]:
            group.create_dataset("data", data=matrix.data, dtype=matrix.data.dtype)
            group.create_dataset("row", data=matrix.row, dtype=matrix.row.dtype)
            group.create_dataset("col", data=matrix.col, dtype=matrix.col.dtype)

    def append_matrix(
        self, group, matrix: Union[csr_matrix, csc_matrix], axis: Optional[int] = None
    ) -> None:
        encoding_type = group.attrs["encoding-type"]
        shape = group.attrs["shape"]

        if encoding_type == "csr_matrix":
            assert (
                shape[1] == matrix.shape[1]
            ), "csr_matrix must have same size of dimension 1 to be appended."
            new_shape = (shape[0] + matrix.shape[0], shape[1])
        elif encoding_type == "csc_matrix":
            assert (
                shape[0] == matrix.shape[0]
            ), "csc_matrix must have same size of dimension 0 to be appended."
            new_shape = (shape[0], shape[1] + matrix.shape[1])
        elif encoding_type == "coo_matrix":
            assert axis is not None and axis in [
                0,
                1,
            ], "axis must be 0 or 1 for coo_matrix."
            if axis == 0:
                assert (
                    shape[1] == matrix.shape[1]
                ), "coo_matrix must have same size of dimension 1 to be appended."
                new_shape = (shape[0] + matrix.shape[0], shape[1])
            elif axis == 1:
                assert (
                    shape[0] == matrix.shape[0]
                ), "coo_matrix must have same size of dimension 0 to be appended."
                new_shape = (shape[0], shape[1] + matrix.shape[1])

        if encoding_type in ["csr_matrix", "csc_matrix"]:
            # data
            data = group["data"]
            orig_data_size = data.shape[0]
            data.resize((orig_data_size + matrix.data.shape[0],))
            data[orig_data_size:] = matrix.data

            # indptr
            indptr = group["indptr"]
            orig_data_size = indptr.shape[0]
            append_offset = indptr[-1]
            indptr.resize((orig_data_size + matrix.indptr.shape[0] - 1,))
            indptr[orig_data_size:] = matrix.indptr[1:].astype(np.int64) + append_offset

            # indices
            indices = group["indices"]
            orig_data_size = indices.shape[0]
            indices.resize((orig_data_size + matrix.indices.shape[0],))
            indices[orig_data_size:] = matrix.indices
            group.attrs["shape"] = new_shape
        elif encoding_type in ["coo_matrix"]:
            # data
            data = group["data"]
            orig_data_size = data.shape[0]
            data.resize((orig_data_size + matrix.data.shape[0],))
            data[orig_data_size:] = matrix.data

            if axis == 0:
                # row
                row = group["row"]
                orig_data_size = row.shape[0]
                append_offset = matrix.shape[0]
                row.resize((orig_data_size + matrix.row.shape[0],))
                row[orig_data_size:] = matrix.row + append_offset

                # col
                col = group["col"]
                orig_data_size = col.shape[0]
                col.resize((orig_data_size + matrix.col.shape[0],))
                col[orig_data_size:] = matrix.col
            elif axis == 1:
                # row
                row = group["row"]
                orig_data_size = row.shape[0]
                row.resize((orig_data_size + matrix.row.shape[0],))
                row[orig_data_size:] = matrix.row

                # col
                col = group["col"]
                orig_data_size = col.shape[0]
                append_offset = matrix.shape[1]
                col.resize((orig_data_size + matrix.col.shape[0],))
                col[orig_data_size:] = matrix.col + append_offset
            group.attrs["shape"] = new_shape

    def get_annotation_index(self, group) -> pd.Index:
        group_index_field = group.attrs["_index"]
        idx = group[group_index_field][...]
        if pd.api.types.is_object_dtype(idx):
            idx = idx.astype(str)
        return pd.Index(idx)

    def get_annotation_column(
        self, group, column: str
    ) -> Union[np.ndarray, pd.Categorical]:
        if column in group:
            series = group[column]
            if (
                isinstance(series, zarr.hierarchy.Group)
                and "categories" in series.keys()
            ):
                categories = series["categories"][...]
                if pd.api.types.is_object_dtype(categories):
                    categories = categories.astype(str)
                ordered = series.attrs.get("ordered", False)
                values = pd.Categorical.from_codes(
                    series["codes"][...], categories, ordered=ordered
                )
            elif (
                isinstance(series, zarr.hierarchy.Group)
                and "categories" in series.attrs
            ):  # for older (<0.8.0) anndata saved zarrs
                categories = series.attrs["categories"]
                categories_dset = group[categories]
                categories = categories_dset[...]
                if pd.api.types.is_object_dtype(categories):
                    categories = categories.astype(str)
                ordered = categories_dset.attrs.get("ordered", False)
                values = pd.Categorical.from_codes(
                    series[...], categories, ordered=ordered
                )
            else:
                values = series[...]
            return values
        return None

    def set_annotation(self, annotation: str, df: pd.DataFrame):
        """Store annotation (i.e. obs, var) from a dataframe.
           This will overwrite the current stored annotation.

        Parameters
        ----------
        annotation: str,
            Annotation name (i.e. obs, var).

        Examples
        --------
        >>> zarr_data.set_annotation("obs", df)
        """

        anno = self.root.create_group(annotation, overwrite=True)
        anno.attrs.setdefault("_index", "_index")
        anno.attrs.setdefault("column-order", list(df.columns))
        anno.attrs.setdefault("encoding-type", "dataframe")
        anno.attrs.setdefault("encoding-version", "0.2.0")

        anno.create_dataset(
            "_index",
            data=df.index._values,
            dtype=df.index._values.dtype,
            object_codec=numcodecs.JSON(),
        )
        anno["_index"].attrs.setdefault("encoding-type", "string-array")
        anno["_index"].attrs.setdefault("encoding-version", "0.2.0")
        for k in df.columns:
            if isinstance(df[k], pd.Categorical):
                v = df[k]
                anno.create_group(k, overwrite=True)
                anno[k].attrs.setdefault("encoding-type", "categorical")
                anno[k].attrs.setdefault("encoding-version", "0.2.0")
                anno[k].attrs.setdefault("ordered", False)

                anno[k].create_dataset(
                    "categories",
                    data=v.categories._values,
                    dtype=v.categories._values.dtype,
                    object_codec=numcodecs.JSON(),
                )
                anno[k]["categories"].attrs.setdefault("encoding-type", "string-array")
                anno[k]["categories"].attrs.setdefault("encoding-version", "0.2.0")

                anno[k].create_dataset("codes", data=v.codes)
                anno[k]["codes"].attrs.setdefault("encoding-type", "array")
                anno[k]["codes"].attrs.setdefault("encoding-version", "0.2.0")
            elif isinstance(df[k], pd.Series):
                if df[k].dtype == "O":
                    anno.create_dataset(
                        k,
                        data=df[k]._values,
                        dtype=df[k]._values.dtype,
                        object_codec=numcodecs.JSON(),
                    )
                else:
                    anno.create_dataset(
                        k, data=df[k]._values, dtype=df[k]._values.dtype
                    )
                anno[k].attrs.setdefault("encoding-type", "array")
                anno[k].attrs.setdefault("encoding-version", "0.2.0")

    def append_annotation(self, annotation: str, df: pd.DataFrame):
        """Append annotation (i.e. obs, var) from a dataframe.

        Parameters
        ----------
        annotation: str,
            Annotation name (i.e. obs, var).

        Examples
        --------
        >>> zarr_data.append_annotation("obs", df)
        """

        group = self.root[annotation]
        columns = list(self.root[annotation].keys())
        current_df = pd.DataFrame(
            {x: self.get_annotation_column(group, x) for x in columns if x != "_index"}
        )
        current_df.index = self.get_annotation_index(group)
        self.set_annotation(annotation, pd.concat([current_df, df]))
