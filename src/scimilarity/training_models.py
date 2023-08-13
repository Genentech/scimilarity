import json
import os
from datetime import datetime
from typing import List, Optional

import hnswlib
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from scimilarity.triplet_selector import TripletSelector
from scimilarity.nn_models import Decoder, Encoder


class TripletLoss(torch.nn.TripletMarginLoss):
    """
    Wrapper for pytorch TripletMarginLoss.
    Triplets are generated using TripletSelector object which take embeddings and labels
    then return triplets.
    """

    def __init__(
        self,
        margin: float,
        sample_across_studies: bool = False,
        negative_selection: str = "semihard",
        perturb_labels: bool = False,
        perturb_labels_fraction: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.sample_across_studies = sample_across_studies
        self.triplet_selector = TripletSelector(
            margin=margin,
            negative_selection=negative_selection,
            perturb_labels=perturb_labels,
            perturb_labels_fraction=perturb_labels_fraction,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        int2label: dict,
        studies: torch.Tensor,
    ):
        if self.sample_across_studies is False:
            studies = None

        (
            triplets,
            num_violating_triplets,
            num_viable_triplets,
        ) = self.triplet_selector.get_triplets(embeddings, labels, int2label, studies)

        anchor, positive, negative = triplets
        return (
            F.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=self.margin,
                p=self.p,
                eps=self.eps,
                swap=self.swap,
                reduction="none",
            ),
            torch.tensor(num_violating_triplets, dtype=torch.float),
            torch.tensor(num_viable_triplets, dtype=torch.float),
        )


class MetricLearning(pl.LightningModule):
    """A class encapsulating the metric learning."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        alpha: float = 0.003,
        margin: float = 0.05,
        negative_selection: str = "semihard",
        sample_across_studies: bool = True,
        perturb_labels: bool = True,
        perturb_labels_fraction: float = 0.5,
        lr: float = 5e-3,
        l1: float = 1e-4,
        l2: float = 0.01,
        max_epochs: int = 500,
        residual: bool = False,
    ):
        """Constructor.

        Parameters (network)
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer

        Parameters (training)
        ----------
        alpha: float, default 0.003
            The weighting for triplet loss vs reconstruction loss.  This weighting sums to 1
            such that triplet loss weight is alpha and reconstruction loss weight is (1 - alpha).
        margin: float, default: 0.05
            The margin parameter in triplet loss.
        negative_selection: {"semihard", "hardest", "random"}, default: "semihard"
            The negative selection function.
        sample_across_studies: bool, default: True
            Whether to enforce anchor-positive pairs being from different studies.
        perturb_labels: bool, default: True
            Whether to perturb celltype labels by coarse graining the label based on cell ontology.
        perturb_labels_fraction: float, default: 0.5
            The fraction of cells per batch to perform label perturbation.
        lr: float, default: 5e-3
            The initial learning rate
        l1: float, default: 1e-4
            The l1 penalty lambda. A value of 0 will disable l1 penalty.
        l2: float, default: 1e-2
            The l2 penalty lambda (weight decay). A value of 0 will disable l2 penalty.
        max_epochs: int, default: 500
            The max epochs, used by the scheduler to determine lr annealing rate.

        Examples
        --------
        >>> datamodule = MetricLearningZarrDataModule(
                batch_size=1000,
                num_workers=1,
                obs_field="celltype_name",
                train_path="train",
                gene_order="gene_order.tsv",
            )
        >>> model = MetricLearning(datamodule.n_genes)
        """

        super().__init__()
        self.save_hyperparameters()
        valid_negative_selection = {"semihard", "hardest", "random"}
        if negative_selection not in valid_negative_selection:
            raise ValueError(
                f"Unknown return_type {negative_selection}."
                f"Options are: {valid_negative_selection}."
            )

        # network architecture
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.residual = residual

        # networks
        self.encoder = Encoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        # save layer sizes
        model_state_dict = self.encoder.state_dict()
        self.layer_sizes = {
            entry: list(model_state_dict[entry].size()) for entry in model_state_dict
        }

        # mixed loss weight
        self.alpha = alpha

        # constraints
        self.margin = margin
        self.negative_selection = negative_selection
        self.sample_across_studies = sample_across_studies
        self.perturb_labels = perturb_labels
        self.perturb_labels_fraction = perturb_labels_fraction

        # lr and regularization
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.max_epochs = max_epochs

        # losses
        self.triplet_loss_fn = TripletLoss(
            margin=self.margin,
            negative_selection=self.negative_selection,
            sample_across_studies=self.sample_across_studies,
            perturb_labels=self.perturb_labels,
            perturb_labels_fraction=self.perturb_labels_fraction,
        )
        self.mse_loss_fn = nn.MSELoss()

        self.scheduler = None

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=self.l2)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler,
        }  # pytorch-lightning required format

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def get_losses(self, batch, use_studies: bool = True):
        cells, labels, studies = batch
        if not use_studies:
            studies = None
        embedding, reconstruction = self(cells)
        triplet_losses, num_hard_triplets, num_viable_triplets = self.triplet_loss_fn(
            embedding, labels, self.trainer.datamodule.int2label, studies
        )
        reconstruction_loss = self.mse_loss_fn(cells, reconstruction)
        return (
            triplet_losses,
            reconstruction_loss,
            num_hard_triplets,
            num_viable_triplets,
        )

    def mixed_loss(self, triplet_loss, reconstruction_loss):
        if self.alpha == 0:
            return reconstruction_loss
        if self.alpha == 1:
            return triplet_loss
        return (self.alpha * triplet_loss) + ((1.0 - self.alpha) * reconstruction_loss)

    def training_step(self, batch, batch_idx):  # pytorch-lightning required parameters
        (
            triplet_losses,
            reconstruction_loss,
            num_hard_triplets,
            num_viable_triplets,
        ) = self.get_losses(batch)

        triplet_loss = triplet_losses.mean()
        num_nonzero_loss = (triplet_losses > 0).sum(dtype=torch.float).detach()
        hard_triplets = num_hard_triplets / num_viable_triplets

        loss = self.mixed_loss(triplet_loss, reconstruction_loss)

        current_lr = self.scheduler["scheduler"].get_last_lr()[0]

        if self.l1 > 0:  # use l1 penalty for first layer
            for layer in self.encoder.network:
                if isinstance(layer, nn.Linear):
                    l1_norm = sum(p.abs().sum() for p in layer.parameters())
                    l1_penalty = self.l1 * l1_norm * current_lr
                    loss += l1_penalty
                    self.log(
                        "train l1 penalty", l1_penalty, prog_bar=False, logger=True
                    )
                    break

        # if self.l2 > 0:  # use l2 penalty
        #    l2_regularization = []
        #    for layer in self.encoder.network:
        #        if isinstance(layer, nn.Linear):
        #            l2_norm = sum(p.pow(2).sum() for p in layer.parameters())
        #            l2_regularization.append(l2_norm)
        #    for layer in self.decoder.network:
        #        if isinstance(layer, nn.Linear):
        #            l2_norm = sum(p.pow(2).sum() for p in layer.parameters())
        #            l2_regularization.append(l2_norm)
        #    l2_penalty = (
        #        self.l2 * sum(l2_regularization[0:-1]) * current_lr
        #    )  # all but reconstruction layer
        #    loss += l2_penalty
        #    self.log("train l2 penalty", l2_penalty, prog_bar=False, logger=True)

        self.log("train loss", loss, prog_bar=False, logger=True)
        self.log("train triplet loss", triplet_loss, prog_bar=True, logger=True)
        self.log(
            "train reconstruction loss", reconstruction_loss, prog_bar=True, logger=True
        )
        self.log("train hard triplets", hard_triplets, prog_bar=True, logger=True)
        self.log("train num nonzero loss", num_nonzero_loss, prog_bar=True, logger=True)
        self.log(
            "train num hard triplets", num_hard_triplets, prog_bar=True, logger=True
        )
        self.log(
            "train num viable triplets", num_viable_triplets, prog_bar=True, logger=True
        )
        return {
            "loss": loss,
            "triplet_loss": triplet_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "train_hard_triplets": hard_triplets,
            "num_nonzeros_loss": num_nonzero_loss,
            "train_num_hard_triplets": num_hard_triplets,
            "train_num_viable_triplets": num_viable_triplets,
        }

    def validation_step(
        self, batch, batch_idx, dataloader_idx: Optional[int] = None
    ):  # pytorch-lightning required parameters
        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_step(batch, prefix="val")

    def validation_epoch_end(self, step_outputs: list):
        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_epoch(step_outputs, prefix="val")

    def test_step(self, batch, batch_idx):  # pytorch-lightning required parameters
        if self.trainer.datamodule.test_dataset is None:
            return {}
        return self._eval_step(batch, prefix="test")

    def test_epoch_end(self, step_outputs: list):
        if self.trainer.datamodule.test_dataset is None:
            return {}
        return self._eval_epoch(step_outputs, prefix="test")

    def _eval_step(self, batch, prefix: str):
        (
            triplet_losses,
            reconstruction_loss,
            num_hard_triplets,
            num_viable_triplets,
        ) = self.get_losses(batch, use_studies=False)

        triplet_loss = triplet_losses.mean()
        num_nonzero_loss = (triplet_losses > 0).sum()
        hard_triplets = num_hard_triplets / num_viable_triplets

        loss = self.mixed_loss(triplet_loss, reconstruction_loss)

        losses = {
            f"{prefix}_loss": loss,
            f"{prefix}_triplet_loss": triplet_loss,
            f"{prefix}_reconstruction_loss": reconstruction_loss,
            f"{prefix}_hard_triplets": hard_triplets,
            f"{prefix}_num_nonzero_loss": num_nonzero_loss,
            f"{prefix}_num_hard_triplets": num_hard_triplets,
            f"{prefix}_num_viable_triplets": num_viable_triplets,
        }
        return losses

    def _eval_epoch(self, step_outputs: list, prefix: str):
        loss = torch.Tensor([step[f"{prefix}_loss"] for step in step_outputs]).mean()
        triplet_loss = torch.Tensor(
            [step[f"{prefix}_triplet_loss"] for step in step_outputs]
        ).mean()
        reconstruction_loss = torch.Tensor(
            [step[f"{prefix}_reconstruction_loss"] for step in step_outputs]
        ).mean()
        hard_triplets = torch.Tensor(
            [step[f"{prefix}_hard_triplets"] for step in step_outputs]
        ).mean()
        num_nonzero_loss = torch.Tensor(
            [step[f"{prefix}_num_nonzero_loss"] for step in step_outputs]
        ).mean()
        num_hard_triplets = torch.Tensor(
            [step[f"{prefix}_num_hard_triplets"] for step in step_outputs]
        ).mean()
        num_viable_triplets = torch.Tensor(
            [step[f"{prefix}_num_viable_triplets"] for step in step_outputs]
        ).mean()

        self.log(f"{prefix} loss", loss, logger=True)
        self.log(f"{prefix} triplet loss", triplet_loss, logger=True)
        self.log(f"{prefix} reconstruction loss", reconstruction_loss, logger=True)
        self.log(f"{prefix} hard triplets", hard_triplets, logger=True)
        self.log(f"{prefix} num nonzero loss", num_nonzero_loss, logger=True)
        self.log(f"{prefix} num hard triplets", num_hard_triplets, logger=True)
        self.log(f"{prefix} num viable triplets", num_viable_triplets, logger=True)

        losses = {
            f"{prefix}_loss": loss,
            f"{prefix}_triplet_loss": triplet_loss,
            f"{prefix}_reconstruction_loss": reconstruction_loss,
            f"{prefix}_hard_triplets": hard_triplets,
            f"{prefix}_num_nonzero_loss": num_nonzero_loss,
            f"{prefix}_num_hard_triplets": num_hard_triplets,
            f"{prefix}_num_viable_triplets": num_viable_triplets,
        }
        return losses

    def get_dataset_embedding(self, dataset):
        embedding_parts = []
        labels_parts = []
        study_parts = []

        self.encoder.eval()
        with torch.inference_mode():
            buffer_size = 10000
            for i in range(0, len(dataset), buffer_size):
                profiles, labels_part, studies = dataset[i : i + buffer_size]
                profiles = torch.Tensor(profiles).cuda()

                embedding_parts.append(self.encoder(profiles).detach().cpu())
                labels_parts.extend(labels_part)
                study_parts.extend(studies.values)

        return torch.vstack(embedding_parts), pd.Categorical(labels_parts), study_parts

    def build_knn_classifier(
        self, embeddings: torch.Tensor, ef_construction=1000, M=80
    ):
        n_cells, latent_dim = embeddings.shape
        nn_reference = hnswlib.Index(
            space="cosine", dim=latent_dim
        )  # possible options are l2, cosine, or ip
        nn_reference.init_index(
            max_elements=n_cells, ef_construction=ef_construction, M=M
        )
        nn_reference.set_ef(ef_construction)
        nn_reference.add_items(embeddings, range(len(embeddings)))
        return nn_reference

    def save_all(
        self,
        model_path: str,
        save_knn: bool = False,
        ef_construction: int = 1000,
        M: int = 80,
    ):
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        # save NN model
        self.encoder.save_state(os.path.join(model_path, "encoder.ckpt"))
        self.decoder.save_state(os.path.join(model_path, "decoder.ckpt"))

        # save layer sizes as json, useful to infer model architecture
        with open(os.path.join(model_path, "layer_sizes.json"), "w") as f:
            f.write(json.dumps(self.layer_sizes))

        # save hyperparameters as json
        hyperparameters = {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "input_dropout": self.input_dropout,
            "margin": self.margin,
            "alpha": self.alpha,
            "negative_selection": self.negative_selection,
            "sample_across_studies": self.sample_across_studies,
            "perturb_labels": self.perturb_labels,
            "perturb_labels_fraction": self.perturb_labels_fraction,
            "lr": self.lr,
            "l1_lambda": self.l1,
            "l2_lambda": self.l2,
            "batch_size": self.trainer.datamodule.batch_size,
            "max_epochs": self.max_epochs,
            "residual": self.residual,
        }
        with open(os.path.join(model_path, "hyperparameters.json"), "w") as f:
            f.write(json.dumps(hyperparameters))

        # write gene order
        with open(os.path.join(model_path, "gene_order.tsv"), "w") as f:
            f.write("\n".join(self.trainer.datamodule.gene_order))

        # write reference labels
        with open(os.path.join(model_path, "reference_labels.tsv"), "w") as f:
            f.write("\n".join(self.trainer.datamodule.train_Y))

        # write dictionary to map label_ints to labels
        pd.Series(self.trainer.datamodule.int2label).to_csv(
            os.path.join(model_path, "label_ints.csv")
        )

        # write metadata: data paths, timestamp
        meta_data = {
            "train_path": self.trainer.datamodule.train_path,
            "val_path": self.trainer.datamodule.val_path,
            "test_path": self.trainer.datamodule.test_path,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            f.write(json.dumps(meta_data))

        if save_knn:  # build and save KNN model
            embeddings, labels, _ = self.get_dataset_embedding(
                self.trainer.datamodule.train_dataset
            )
            knn = self.build_knn_classifier(
                embeddings, ef_construction=ef_construction, M=M
            )
            knn.save_index(os.path.join(model_path, "kNN"))

    def load_state(
        self,
        encoder_filename: str,
        decoder_filename: str,
        use_gpu: bool = False,
        freeze: bool = False,
    ):
        self.encoder.load_state(encoder_filename, use_gpu)
        self.decoder.load_state(decoder_filename, use_gpu)

        if freeze:
            # encoder batchnorm freeze
            for i in range(len(self.encoder.network)):
                if isinstance(self.encoder.network[i], nn.BatchNorm1d):
                    for param in self.encoder.network[i].parameters():
                        param.requires_grad = False  # freeze

            # encoder linear freeze
            encoder_linear_idx = []
            for i in range(len(self.encoder.network)):
                if isinstance(self.encoder.network[i], nn.Linear):
                    encoder_linear_idx.append(i)
            for i in range(len(encoder_linear_idx)):
                if i < len(encoder_linear_idx) - 1:  # freeze all but bottleneck
                    for param in self.encoder.network[
                        encoder_linear_idx[i]
                    ].parameters():
                        param.requires_grad = False  # freeze
                else:
                    for param in self.encoder.network[
                        encoder_linear_idx[i]
                    ].parameters():
                        param.requires_grad = True  # unfreeze
