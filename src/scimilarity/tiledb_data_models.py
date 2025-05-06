import numpy as np
import os, re
import pandas as pd
import pytorch_lightning as pl
import random
from scipy.sparse import coo_matrix, diags
import tiledb
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional

from .utils import query_tiledb_df
from .ontologies import (
    import_cell_ontology,
    get_id_mapper,
    find_most_viable_parent,
    get_all_ancestors,
)

import logging

log = logging.getLogger(__name__)

cfg = tiledb.Config(
    {
        "sm.mem.total_budget": 50000000000,  # 50G
        # turn off tiledb multithreading
        "sm.compute_concurrency_level": 1,
        "sm.io_concurrency_level": 1,
        "sm.num_async_threads": 1,
        "sm.num_reader_threads": 1,
        "sm.num_tbb_threads": 1,
        "sm.num_writer_threads": 1,
        "vfs.num_threads": 1,
    }
)


class scDataset(Dataset):
    """A class that represents cells in TileDB.

    Parameters
    ----------
    data_df: pandas.DataFrame
        Pandas dataframe of valid cells.
    """

    def __init__(
        self,
        data_df: "pandas.DataFrame",
    ):
        self.data_df = data_df

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        return self.data_df.loc[[idx]].copy()


class CellSampler(Sampler[int]):
    """Sampler class for composition of cells in minibatch.

    Parameters
    ----------
    data_df: pandas.DataFrame
        DataFrame with column "sampling_weight"
    batch_size: int
        Batch size
    n_batches: int
        Number of batches to create. Should correspond to number of
        batches per epoch, as we are sampling with replacement.
    dynamic_weights: bool, default: False
        Dynamically lower the sampling weight of seen cells.
    weight_decay: float, default: 0.5
        Weight decay factor.
    """

    def __init__(
        self,
        data_df: "pandas.DataFrame",
        batch_size: int,
        n_batches: int,
        dynamic_weights: bool = False,
        weight_decay: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.data_df = data_df.copy()
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.dynamic_weights = dynamic_weights
        self.weight_decay = weight_decay

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        if self.dynamic_weights:
            col = "dynamic_weights"
            self.data_df[col] = self.data_df["sampling_weight"]
        else:
            col = "sampling_weight"

        a = self.data_df.index.values
        weights = self.data_df[col].values / self.data_df[col].values.sum()
        for _ in range(self.n_batches):
            batch = np.random.choice(
                a, size=self.batch_size, replace=False, p=weights
            ).tolist()
            if self.dynamic_weights:
                # lower the weight of previously sampled cells
                affected_cells = batch.index.values
                self.data_df.loc[affected_cells, col] = max(
                    self.data_df.loc[affected_cells, col] * weight_decay, 0.01
                )
                weights = self.data_df[col].values / self.data_df[col].values.sum()
            yield batch


class CellMultisetDataModule(pl.LightningDataModule):
    """A class to encapsulate cells in TileDB to train the model.

    Parameters
    ----------
    dataset_path: str
        Path to the directory containing the TileDB stores.
    cell_metadata_uri: str, default: "cell_metadata"
        Relative path to the cell metadata store.
    gene_annotation_uri: str, default: "gene_annotation"
        Relative path to the gene annotation store.
    counts_uri: str, default: "counts"
        Relative path to the counts matrix store.
    gene_order: str
        Use a given gene order as described in the specified file.
        One gene symbol per line.
    val_studies: List[str], optional, default: None
        List of studies to use as validation and test.
    exclude_studies: List[str], optional, default: None
        List of studies to exclude.
    exclude_samples: Dict[str, List[str]], optional, default: None
        Dict of samples to exclude in the form {study: [list of samples]}.
    label_id_column: str, default: "cellTypeOntologyID"
        Cell ontology ID column name.
    study_column: str, default: "datasetID"
        Study column name.
    sample_column: str, default: "sampleID"
        Sample column name.
    batch_size: int, default: 1000
        Batch size.
    num_workers: int, default: 1
        The number of worker threads for dataloaders
    lognorm: bool, default: True
        Whether to return log normalized expression instead of raw counts.
    target_sum: float, default: 1e4
        Target sum for log normalization.
    sparse: bool, default: False
        Use sparse matrices.
    remove_singleton_classes: bool, default: True
        Exclude cells with classes that exist in only one study.
    nan_string: str, default: "nan"
        A string representing NaN.
    sampler_cls: Sampler, default: CellSampler
        Sampler class to use for batching.
    dataset_cls: Dataset, default: scDataset
        Base Dataset class to use.
    n_batches: int, default: 100
        Number of batches to create in batch sampler. Should correspond to number of
        batches per epoch, as we are sampling with replacement.
    pin_memory: bool, default: False
        If True, uses pin memory in the DataLoaders.
    persistent_workers: bool, default: False
        If True, uses persistent workers in the DataLoaders.

    Examples
    --------
    >>> datamodule = MetricLearningZarrDataModule(
            dataset_path="/opt/cellarr_dataset"
            label_id_column="id",
            study_column="study",
            batch_size=1000,
            num_workers=1,
        )
    """

    def __init__(
        self,
        dataset_path: str,
        cell_metadata_uri: str = "cell_metadata",
        gene_annotation_uri: str = "gene_annotation",
        counts_uri: str = "counts",
        gene_order: Optional[str] = None,
        val_studies: Optional[List[str]] = None,
        exclude_studies: Optional[List[str]] = None,
        exclude_samples: Optional[Dict[str, List[str]]] = None,
        label_id_column: str = "cellTypeOntologyID",
        study_column: str = "datasetID",
        sample_column: str = "sampleID",
        filter_condition: Optional[str] = None,
        batch_size: int = 1000,
        num_workers: int = 0,
        lognorm: bool = True,
        target_sum: float = 1e4,
        sparse: bool = False,
        remove_singleton_classes: bool = True,
        nan_string: str = "nan",
        sampler_cls: Sampler = CellSampler,
        dataset_cls: Dataset = scDataset,
        n_batches: int = 100,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.cell_metadata_uri = cell_metadata_uri
        self.gene_annotation_uri = gene_annotation_uri
        self.counts_uri = counts_uri
        self.val_studies = val_studies
        self.exclude_studies = exclude_studies
        self.exclude_samples = exclude_samples
        self.label_id_column = label_id_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.filter_condition = filter_condition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.sparse = sparse
        self.remove_singleton_classes = remove_singleton_classes
        self.nan_string = nan_string
        self.sampler_cls = sampler_cls
        self.dataset_cls = dataset_cls
        self.n_batches = n_batches
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        if self.sparse:
            self.pin_memory = False

        self.cell_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.cell_metadata_uri), "r"
        )
        self.gene_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.gene_annotation_uri), "r"
        )

        self.matrix_shape = (
            self.cell_tdb.nonempty_domain()[0][1] + 1,
            self.gene_tdb.nonempty_domain()[0][1] + 1,
        )

        # get data with a limit to cells with counts and labels, and QC checks
        if self.filter_condition is None:
            self.filter_condition = f"{self.label_id_column}!='{self.nan_string}' and total_counts>1000 and n_genes_by_counts>500 and pct_counts_mt<20 and predicted_doublets==0"
        self.data_df = self.get_data(self.filter_condition)

        if self.exclude_studies is not None:
            self.data_df = self.data_df[
                ~self.data_df[self.study_column].isin(self.exclude_studies)
            ].copy()
        if self.exclude_samples is not None:
            self.data_df[":dummy:"] = (
                self.data_df[self.study_column]
                + "::::"
                + self.data_df[self.sample_column]
            )
            sample_list = [
                f"{k}::::{x}" for k, v in self.exclude_samples.items() for x in v
            ]
            self.data_df = self.data_df[
                ~self.data_df[":dummy:"].isin(sample_list)
            ].copy()
            self.data_df = self.data_df.drop(columns=":dummy:")

        # manual cell type harmonization
        self.data_df = self.harmonize_cell_types(self.data_df)

        self.val_df = None
        if self.val_studies is not None:
            # split out validation studies
            self.val_df = self.data_df[
                self.data_df[self.study_column].isin(self.val_studies)
            ].copy()
            self.train_df = self.data_df[
                ~self.data_df[self.study_column].isin(self.val_studies)
            ].copy()
            # limit validation cell types to those in the training data
            self.val_df = self.val_df[
                self.val_df[self.label_id_column].isin(
                    self.train_df[self.label_id_column].unique()
                )
            ].copy()
        else:
            self.train_df = self.data_df.copy()
        del self.data_df

        # limit to labels that exist in at least 2 studies
        if self.remove_singleton_classes:
            self.train_df = self.remove_singleton_label_ids(self.train_df)
            if self.val_df is not None:
                self.val_df = self.remove_singleton_label_ids(self.val_df)
                # limit validation cell types to those in the training data
                self.val_df = self.val_df[
                    self.val_df[self.label_id_column].isin(
                        self.train_df[self.label_id_column].unique()
                    )
                ].copy()

        self.label_name_column = "cellTypeName"
        self.train_df = self.get_sampler_weights(self.train_df)
        self.train_df = self.map_cell_type_id2name(self.train_df)
        if self.val_df is not None:
            self.val_df = self.get_sampler_weights(self.val_df)
            self.val_df = self.map_cell_type_id2name(self.val_df)

        log.info(f"Training data size: {self.train_df.shape}")
        if self.val_df is not None:
            log.info(f"Validation data size: {self.val_df.shape}")

        self.class_names = set(self.train_df[self.label_name_column].values)
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        genes = (
            self.gene_tdb.query(attrs=["cellarr_gene_index"])
            .df[:]["cellarr_gene_index"]
            .tolist()
        )
        if gene_order is not None:
            # gene space needs be aligned to the given gene order
            with open(gene_order, "r") as fh:
                self.gene_order = [line.strip() for line in fh]
            self.gene_indices = []
            for x in self.gene_order:
                try:
                    self.gene_indices.append(genes.index(x))
                except:
                    log.info(f"Gene not found: {x}")
                    pass
        else:
            self.gene_order = genes
            self.gene_indices = list(range(len(genes)))
        self.n_genes = len(self.gene_indices)  # used when creating training model

        self.train_df["label_int"] = self.train_df[self.label_name_column].map(
            self.label2int
        )
        self.train_dataset = scDataset(data_df=self.train_df)

        self.val_dataset = None
        if self.val_df is not None:
            self.val_df["label_int"] = self.val_df[self.label_name_column].map(
                self.label2int
            )
            self.val_dataset = scDataset(data_df=self.val_df)

        self.counts_tdb = tiledb.open(
            os.path.join(self.dataset_path, self.counts_uri), "r", config=cfg
        )
        self.counts_attr = self.counts_tdb.schema.attr(0).name

        self.cell_tdb.close()
        self.gene_tdb.close()

    def __del__(self):
        self.counts_tdb.close()

    def get_data(self, filter_condition: str):
        """Filter the tiledb cell metadata according to some filter condition and
           return the valid cells.

        Parameters
        ----------
        filter_condition: str
            A string that describes the filter condition according to tiledb search syntax.
        """

        return query_tiledb_df(
            self.cell_tdb,
            filter_condition,
            attrs=[self.study_column, self.sample_column, self.label_id_column],
        )

    def map_cell_type_id2name(self, data_df: "pandas.DataFrame"):
        """Map cell type ontology ID to name.

        Parameters
        ----------
        data_df: pandas.DataFrame
            DataFrame with a label id column and optionally a study column.
        """

        onto = import_cell_ontology()
        id2name = get_id_mapper(onto)
        data_df[self.label_name_column] = data_df[self.label_id_column].map(id2name)
        data_df = data_df.dropna()
        return data_df

    def remove_singleton_label_ids(
        self, data_df: "pandas.DataFrame", n_studies: int = 2
    ):
        """Ensure labels exist in at least a minimum number of studies.

        Parameters
        ----------
        data_df: pandas.DataFrame
            DataFrame with a label id column and optionally a study column.
        n_studies: int, default: 2
            The number of studies a label must exist in to be valid.
        """

        cell_type_counts = (
            data_df[[self.study_column, self.label_id_column]]
            .drop_duplicates()
            .groupby(self.label_id_column)
            .size()
            .sort_values(ascending=False)
        )
        singleton_labels = cell_type_counts[cell_type_counts <= 1].index
        well_represented_labels = cell_type_counts[cell_type_counts > 1].index

        # try to coarse grain singleton labels
        onto = import_cell_ontology()
        coarse_grain_dict = {label: label for label in well_represented_labels}
        for cell_type_id in singleton_labels:
            coarse_id = find_most_viable_parent(
                onto, cell_type_id, node_list=cell_type_counts.index
            )
            if coarse_id:
                coarse_grain_dict[cell_type_id] = coarse_id

        data_df[self.label_id_column] = data_df[self.label_id_column].map(
            coarse_grain_dict
        )
        cell_type_counts = (
            data_df[[self.study_column, self.label_id_column]]
            .drop_duplicates()
            .groupby(self.label_id_column)
            .size()
            .sort_values(ascending=False)
        )
        well_represented_labels = cell_type_counts[cell_type_counts >= n_studies].index

        data_df = data_df[
            data_df[self.label_id_column].isin(well_represented_labels)
        ].copy()
        return data_df

    def get_sampler_weights(
        self,
        data_df: "pandas.DataFrame",
        use_study: bool = True,
        class_target_sum: float = 1e4,
        study_target_sum: float = 1e6,
    ):
        """Get sampling weights and add to dataframe.

        Parameters
        ----------
        data_df: pandas.DataFrame
            DataFrame with a label id column and optionally a study column.
        use_study: bool, default: False
            Incorporate studies in sampler weights
        class_target_sum: float, default: 1e4
            Target sum for normalization of class counts.
        study_target_sum: float, default: 1e6
            Target sum for normalization of study counts.
        """

        class_count = data_df[self.label_id_column].value_counts()
        class_count = {
            x: np.log1p(class_count[x] / class_target_sum) for x in class_count.index
        }
        data_df["sampling_weight"] = data_df[self.label_id_column].apply(
            lambda x: 1.0 / class_count[x]
        )
        if use_study:
            study_count = data_df[self.study_column].value_counts()
            study_count = {
                x: np.log1p(study_count[x] / study_target_sum)
                for x in study_count.index
            }
            data_df["study_weight"] = data_df[self.study_column].apply(
                lambda x: 1.0 / study_count[x]
            )
            data_df["sampling_weight"] = (
                data_df["sampling_weight"] * data_df["study_weight"]
            )

        max_weight = np.quantile(data_df["sampling_weight"].values, 0.999) * 2
        data_df["sampling_weight"] = data_df["sampling_weight"].apply(
            lambda x: min(max_weight, x)
        )
        return data_df

    def harmonize_cell_types(self, data_df: "pandas.DataFrame"):
        """Manual harmonization of some cell types.

        Parameters
        ----------
        data_df: pandas.DataFrame
            DataFrame with a label id column.
        """

        data_df[self.label_id_column] = data_df[self.label_id_column].str.replace(
            "CL:0000792",  # CD4-positive, CD25-positive, alpha-beta regulatory T cell
            "CL:0000815",  # regulatory T cell
        )
        data_df[self.label_id_column] = data_df[self.label_id_column].str.replace(
            "CL:0001043",  # activated CD4-positive, alpha-beta T cell, human
            "CL:0000896",  # activated CD4-positive, alpha-beta T cell
        )
        data_df[self.label_id_column] = data_df[self.label_id_column].str.replace(
            "CL:0001056",  # dendritic cell, human
            "CL:0000451",  # dendritic cell
        )
        data_df[self.label_id_column] = data_df[self.label_id_column].str.replace(
            "CL:0001057",  # myeloid dendritic cell, human
            "CL:0000782",  # myeloid dendritic cell
        )
        data_df[self.label_id_column] = data_df[self.label_id_column].str.replace(
            "CL:0001058",  # plasmacytoid dendritic cell, human
            "CL:0000784",  # plasmacytoid dendritic cell
        )
        return data_df

    def collate(self, batch):
        """Collate tensors.

        Parameters
        ----------
        batch:
            Batch to collate.

        Returns
        -------
        tuple
            Gene expression, labels, and studies
        """

        df = pd.concat(batch)
        cell_idx = df.index.tolist()

        results = self.counts_tdb.multi_index[cell_idx, :]

        counts = coo_matrix(
            (results[self.counts_attr], (results["cell_index"], results["gene_index"])),
            shape=self.matrix_shape,
        ).tocsr()
        counts = counts[cell_idx, :]
        counts = counts[:, self.gene_indices]

        X = counts.astype(np.float32)
        if self.lognorm:
            # normalize to target sum
            row_sums = np.ravel(X.sum(axis=1))  # row sums as a 1D array
            # avoid division by zero by setting zero sums to one (they will remain zero after normalization)
            row_sums[row_sums == 0] = 1
            # create a sparse diagonal matrix with the inverse of the row sums
            inv_row_sums = diags(1 / row_sums).tocsr()
            # normalize the rows to sum to 1
            normalized_matrix = inv_row_sums.dot(X)
            # scale the rows sum to target_sum
            X = normalized_matrix.multiply(self.target_sum)
            X = X.log1p()

        X = torch.Tensor(X.toarray())
        if self.sparse:
            X = X.to_sparse()
        return (X, torch.Tensor(df["label_int"].values), df[self.study_column].values)

    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the training dataset.
        """

        self.train_sampler = self.sampler_cls(
            data_df=self.train_df,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
        )
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=self.pin_memory,
            batch_sampler=self.train_sampler,
            persistent_workers=self.persistent_workers,
            multiprocessing_context="fork",
        )

    def val_dataloader(self) -> DataLoader:
        """Load the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the validation dataset.
        """

        if self.val_dataset is None:
            return None
        self.val_sampler = self.sampler_cls(
            data_df=self.val_df,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
        )
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=self.pin_memory,
            batch_sampler=self.val_sampler,
            persistent_workers=self.persistent_workers,
            multiprocessing_context="fork",
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """

        return self.val_dataloader()
