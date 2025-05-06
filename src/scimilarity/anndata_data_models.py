import anndata
from collections import Counter
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional

from .utils import align_dataset
from .ontologies import import_cell_ontology, get_id_mapper


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
        return self.X[idx].A, self.Y[idx], self.study[idx]


class MetricLearningDataModule(pl.LightningDataModule):
    """A class to encapsulate the anndata needed to train the model.

    Parameters
    ----------
    train_path: str
        Path to the training h5ad file.
    val_path: str, optional, default: None
        Path to the validataion h5ad file.
    obs_field: str, default: "celltype_name"
        The obs key name containing celltype labels.
    batch_size: int, default: 1000
        Batch size.
    num_workers: int, default: 1
        The number of worker threads for dataloaders.
    gene_order_file: str, optional
        Use a given gene order as described in the specified file rather than using the
        training dataset's gene order. One gene symbol per line.

    Examples
    --------
    >>> datamodule = MetricLearningDataModule(
            batch_size=1000,
            num_workers=1,
            obs_field="celltype_name",
            train_path="train.h5ad",
        )
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        obs_field: str = "celltype_name",
        batch_size: int = 500,
        num_workers: int = 1,
        gene_order_file: Optional[str] = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.obs_field = obs_field
        self.batch_size = batch_size
        self.num_workers = num_workers

        # read in ontology terms
        self.name2id = {
            value: key for key, value in get_id_mapper(import_cell_ontology()).items()
        }

        # read in training dataset
        train_data = anndata.read_h5ad(self.train_path)

        # keep cells whose celltype labels have valid ontology id
        train_data = self.subset_valid_terms(train_data)

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
        self.class_names = set(train_data.obs[obs_field])
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {
            value: key for key, value in self.label2int.items()
        }  # used during training

        train_study = train_data.obs["study"]  # studies
        self.train_Y = train_data.obs[obs_field].values  # text labels
        self.train_dataset = scDataset(train_data.X, self.train_Y, study=train_study)

        self.val_dataset = None
        if val_path is not None:
            val_data = anndata.read_h5ad(self.val_path)
            val_data = align_dataset(
                self.subset_valid_terms(val_data), self.gene_order
            )  # gene space needs to match training set
            val_data = val_data[
                val_data.obs[self.obs_field].isin(self.class_names)
            ]  # labels need to be subsetted to training labels

            if val_data.shape[0] == 0:
                raise RuntimeError("No celltype labels have a valid ontology id.")
            val_study = val_data.obs["study"]  # studies
            val_Y = val_data.obs[obs_field].values  # text labels
            self.val_dataset = scDataset(val_data.X, val_Y, study=val_study)

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

        valid_terms_idx = data.obs[self.obs_field].isin(self.name2id.keys())
        if valid_terms_idx.any():
            return data[valid_terms_idx]
        raise RuntimeError("No celltype labels have a valid ontology id.")

    def get_sampler_weights(
        self, labels: list, studies: Optional[list] = None
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
                x: np.log1p(class_sample_count[x] / 1e4) for x in class_sample_count
            }
            study_sample_count = {
                x: np.log1p(study_sample_count[x] / 1e5) for x in study_sample_count
            }
            sample_weights = torch.Tensor(
                [
                    1.0 / class_sample_count[labels[i]] / study_sample_count[studies[i]]
                    for i in range(len(labels))
                ]
            )
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    def collate(self, batch):
        """Collate tensors.

        Parameters
        ----------
        batch:
            Batch to collate.

        Returns
        -------
        tuple
            A Tuple[torch.Tensor, torch.Tensor, list] containing information
            on the collated tensors.
        """

        profiles, labels, studies = tuple(
            map(list, zip(*batch))
        )  # tuple([list(t) for t in zip(*batch)])
        return (
            torch.squeeze(torch.Tensor(np.vstack(profiles))),
            torch.Tensor([self.label2int[l] for l in labels]),
            studies,
        )

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
            sampler=self.get_sampler_weights(self.train_dataset),
            collate_fn=self.collate,
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
            sampler=self.get_sampler_weights(self.val_dataset),
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """

        return self.val_dataloader()
