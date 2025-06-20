import anndata
from collections import Counter
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional

from .utils import align_dataset
from .ontologies import (
    import_cell_ontology,
    get_id_mapper,
    find_most_viable_parent,
)


class scDataset(Dataset):
    """A class that represents a single cell dataset.

    Parameters
    ----------
    X: numpy.ndarray
        Gene expression vectors for every cell.
    Y: numpy.ndarray
        Text labels for every cell.
    study: numpy.ndarray
        The study identifier for every cell.
    """

    def __init__(self, X, Y, study=None):
        self.X = X
        self.Y = Y
        self.study = study

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # data, label, study
        return self.X[idx].toarray().flatten(), self.Y[idx], self.study[idx]


class scCollator:
    """A class to collate batch data."""

    def __init__(
        self,
        label2int: dict,
        sparse: bool = False,
    ):
        self.label2int = label2int
        self.sparse = sparse

    def __call__(self, batch):
        # tuple([list(t) for t in zip(*batch)])
        profiles, labels, studies = tuple(map(list, zip(*batch)))
        X = torch.squeeze(torch.Tensor(np.vstack(profiles)))
        if self.sparse:
            X = X.to_sparse()
        return (
            X,
            torch.Tensor([self.label2int[l] for l in labels]),
            np.array(studies),
        )


class MetricLearningDataModule(pl.LightningDataModule):
    """A class to encapsulate the anndata needed to train the model.

    Parameters
    ----------
    train_path: str
        Path to the training h5ad file.
    val_path: str, optional, default: None
        Path to the validataion h5ad file.
    label_column: str, default: "celltype_name"
        The column name containing ontology compliant cell type names.
    study_column: str, default: "study"
        The column name containing study identifiers.
    gene_order_file: str, optional
        Use a given gene order as described in the specified file rather than using the
        training dataset's gene order. One gene symbol per line.
    batch_size: int, default: 1000
        Batch size.
    num_workers: int, default: 1
        The number of worker threads for dataloaders.
    sparse: bool, default: False
        Use sparse matrices.
    remove_singleton_classes: bool, default: True
        Exclude cells with classes that exist in only one study.

    Examples
    --------
    >>> datamodule = MetricLearningDataModule(
            batch_size=1000,
            num_workers=1,
            label_column="celltype_name",
            train_path="train.h5ad",
        )
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        label_column: str = "celltype_name",
        study_column: str = "study",
        gene_order_file: Optional[str] = None,
        batch_size: int = 500,
        num_workers: int = 1,
        sparse: bool = False,
        remove_singleton_classes: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        multiprocessing_context: str = "fork",
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.label_column = label_column
        self.study_column = study_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sparse = sparse
        self.remove_singleton_classes = remove_singleton_classes
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        if self.sparse:
            self.pin_memory = False
        if self.num_workers == 0:
            self.persistent_workers = False

        # read in training dataset
        train_data = anndata.read_h5ad(self.train_path)

        # keep cells whose celltype labels have valid ontology id
        train_data = self.subset_valid_terms(train_data)

        if self.remove_singleton_classes:
            train_data = self.remove_singleton_label_ids(train_data)

        if (
            gene_order_file is not None
        ):  # gene space needs be aligned to the given gene order
            with open(gene_order_file, "r") as fh:
                self.gene_order = [line.strip() for line in fh]
            train_data = align_dataset(train_data, self.gene_order)
        else:  # training dataset gene space is the gene order
            self.gene_order = train_data.var.index.tolist()

        self.n_genes = train_data.shape[1]  # used when creating training model

        # map training labels to ints
        self.class_names = set(train_data.obs[self.label_column])
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {
            value: key for key, value in self.label2int.items()
        }  # used during training

        self.train_study = train_data.obs[self.study_column]  # studies
        self.train_Y = train_data.obs[self.label_column].values  # text labels
        self.train_dataset = scDataset(
            train_data.X, self.train_Y, study=self.train_study
        )

        self.val_dataset = None
        if val_path is not None:
            val_data = anndata.read_h5ad(self.val_path)
            val_data = align_dataset(
                self.subset_valid_terms(val_data), self.gene_order
            )  # gene space needs to match training set
            if self.remove_singleton_classes:
                val_data = self.remove_singleton_label_ids(val_data)
            val_data = val_data[
                val_data.obs[self.label_column].isin(self.class_names)
            ]  # labels need to be subsetted to training labels

            if val_data.shape[0] == 0:
                raise RuntimeError("No celltype labels have a valid ontology id.")
            self.val_study = val_data.obs[self.study_column]  # studies
            self.val_Y = val_data.obs[self.label_column].values  # text labels
            self.val_dataset = scDataset(val_data.X, self.val_Y, study=self.val_study)

        self.collator = scCollator(sparse=self.sparse, label2int=self.label2int)

    def subset_valid_terms(self, data: anndata.AnnData) -> anndata.AnnData:
        """Keep cells whose celltype labels have valid ontology id.

        Parameters
        ----------
        data: anndata.AnnData
            Annotated data to subset by valid ontology id.

        Returns
        -------
        anndata.AnnData
            An object containing the data whose celltype labels have
            valid ontology id.
        """

        # read in ontology terms
        name2id = {
            value: key for key, value in get_id_mapper(import_cell_ontology()).items()
        }
        valid_terms_idx = data.obs[self.label_column].isin(name2id.keys())
        if valid_terms_idx.any():
            return data[valid_terms_idx]
        raise RuntimeError("No celltype labels have a valid ontology id.")

    def remove_singleton_label_ids(
        self, data: anndata.AnnData, n_studies: int = 2
    ) -> anndata.AnnData:
        """Ensure labels exist in at least a minimum number of studies.

        Parameters
        ----------
        data: anndata.AnnData
            Annotated data to subset by valid ontology id.
        n_studies: int, default: 2
            The number of studies a label must exist in to be valid.
        """

        obs = data.obs.copy()
        cell_type_counts = (
            obs[[self.study_column, self.label_column]]
            .drop_duplicates()
            .groupby(self.label_column, observed=True)
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

        obs[self.label_column] = obs[self.label_column].map(coarse_grain_dict)
        cell_type_counts = (
            obs[[self.study_column, self.label_column]]
            .drop_duplicates()
            .groupby(self.label_column, observed=True)
            .size()
            .sort_values(ascending=False)
        )
        well_represented_labels = cell_type_counts[cell_type_counts >= n_studies].index

        data = data[obs[self.label_column].isin(well_represented_labels)].copy()
        return data

    def get_sampler_weights(
        self,
        labels: list,
        studies: Optional[list] = None,
        class_target_sum: float = 1e4,
        study_target_sum: float = 1e6,
    ) -> WeightedRandomSampler:
        """Get weighted random sampler.

        Parameters
        ----------
        dataset: scDataset
            Single cell dataset.

        Returns
        -------
        WeightedRandomSampler
            A WeightedRandomSampler object.
        """

        if studies is None:
            class_sample_count = Counter(labels)
            sample_weights = torch.Tensor([1.0 / class_sample_count[t] for t in labels])
        else:
            class_sample_count = Counter(labels)
            study_sample_count = Counter(studies)
            class_sample_count = {
                x: 1.0 / np.log1p(class_sample_count[x] / class_target_sum)
                for x in class_sample_count
            }
            study_sample_count = {
                x: 1.0 / np.log1p(study_sample_count[x] / study_target_sum)
                for x in study_sample_count
            }
            sample_weights = [
                class_sample_count[labels[i]] * study_sample_count[studies[i]]
                for i in range(len(labels))
            ]
            max_weight = np.quantile(sample_weights, 0.999) * 2
            sample_weights = torch.Tensor([min(max_weight, x) for x in sample_weights])
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.get_sampler_weights(self.train_Y, self.train_study),
            collate_fn=self.collator,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.get_sampler_weights(self.val_Y, self.val_study),
            collate_fn=self.collator,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context,
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """

        return self.val_dataloader()
