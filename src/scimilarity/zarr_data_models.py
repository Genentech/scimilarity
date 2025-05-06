from collections import Counter
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from typing import Optional

from .zarr_dataset import ZarrDataset


class scDataset(Dataset):
    """A class that represent a collection of single cell datasets in zarr format.

    Parameters
    ----------
    data_list: list
        List of single-cell datasets.
    obs_celltype: str, default: "celltype_name"
        Cell type name.
    obs_study: str, default: "study"
        Study name.
    """

    def __init__(self, data_list, obs_celltype="celltype_name", obs_study="study"):
        self.data_list = data_list
        self.ncells_list = [data.shape[0] for data in data_list]
        self.ncells = sum(self.ncells_list)
        self.obs_celltype = obs_celltype
        self.obs_study = obs_study

        self.data_idx = [
            n for n in range(len(self.ncells_list)) for i in range(self.ncells_list[n])
        ]
        self.cell_idx = [
            i for n in range(len(self.ncells_list)) for i in range(self.ncells_list[n])
        ]

    def __len__(self):
        return self.ncells

    def __getitem__(self, idx):
        # data, label, study
        data_idx = self.data_idx[idx]
        cell_idx = self.cell_idx[idx]
        return (
            self.data_list[data_idx].get_cell(cell_idx).A,
            self.data_list[data_idx].get_obs(self.obs_celltype)[cell_idx],
            self.data_list[data_idx].get_obs(self.obs_study)[cell_idx],
        )


class MetricLearningDataModule(pl.LightningDataModule):
    """A class to encapsulate a collection of zarr datasets to train the model.

    Parameters
    ----------
    train_path: str
        Path to folder containing all training datasets.
        All datasets should be in zarr format, aligned to a known gene space, and
        cleaned to only contain valid cell ontology terms.
    gene_order: str
        Use a given gene order as described in the specified file. One gene
        symbol per line.
        IMPORTANT: the zarr datasets should already be in this gene order
        after preprocessing.
    val_path: str, optional, default: None
        Path to folder containing all validation datasets.
    obs_field: str, default: "celltype_name"
        The obs key name containing celltype labels.
    batch_size: int, default: 1000
        Batch size.
    num_workers: int, default: 1
        The number of worker threads for dataloaders

    Examples
    --------
    >>> datamodule = MetricLearningZarrDataModule(
            batch_size=1000,
            num_workers=1,
            obs_field="celltype_name",
            train_path="train",
            gene_order="gene_order.tsv",
        )
    """

    def __init__(
        self,
        train_path: str,
        gene_order: str,
        val_path: Optional[str] = None,
        obs_field: str = "celltype_name",
        batch_size: int = 1000,
        num_workers: int = 1,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # gene space needs be aligned to the given gene order
        with open(gene_order, "r") as fh:
            self.gene_order = [line.strip() for line in fh]

        self.n_genes = len(self.gene_order)  # used when creating training model

        train_data_list = []
        self.train_Y = []  # text labels
        self.train_study = []  # text studies

        if self.train_path[-1] != os.sep:
            self.train_path += os.sep

        self.train_file_list = [
            (
                root.replace(self.train_path, "").split(os.sep)[0],
                dirs[0].replace(".aligned.zarr", ""),
            )
            for root, dirs, files in os.walk(self.train_path)
            if dirs and dirs[0].endswith(".aligned.zarr")
        ]

        for study, sample in tqdm(self.train_file_list):
            data_path = os.path.join(
                self.train_path, study, sample, sample + ".aligned.zarr"
            )
            if os.path.isdir(data_path):
                zarr_data = ZarrDataset(data_path)
                train_data_list.append(zarr_data)
                self.train_Y.extend(zarr_data.get_obs(obs_field).astype(str).tolist())
                self.train_study.extend(zarr_data.get_obs("study").astype(str).tolist())

        # Lazy load training data from list of zarr datasets
        self.train_dataset = scDataset(train_data_list)

        self.class_names = set(self.train_Y)
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        self.val_dataset = None
        if self.val_path is not None:
            val_data_list = []
            self.val_Y = []
            self.val_study = []

            if self.val_path[-1] != os.sep:
                self.val_path += os.sep

            self.val_file_list = [
                (
                    root.replace(self.val_path, "").split(os.sep)[0],
                    dirs[0].replace(".aligned.zarr", ""),
                )
                for root, dirs, files in os.walk(self.val_path)
                if dirs and dirs[0].endswith(".aligned.zarr")
            ]

            for study, sample in tqdm(self.val_file_list):
                data_path = os.path.join(
                    self.val_path, study, sample, sample + ".aligned.zarr"
                )
                if os.path.isdir(data_path):
                    zarr_data = ZarrDataset(data_path)
                    val_data_list.append(zarr_data)
                    self.val_Y.extend(zarr_data.get_obs(obs_field).astype(str).tolist())
                    self.val_study.extend(
                        zarr_data.get_obs("study").astype(str).tolist()
                    )

            # Lazy load val data from list of zarr datasets
            self.val_dataset = scDataset(val_data_list)

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
            torch.Tensor([self.label2int[l] for l in labels]),  # text to int labels
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
            sampler=self.get_sampler_weights(self.train_Y, self.train_study),
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
            sampler=self.get_sampler_weights(self.val_Y, self.val_study),
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
