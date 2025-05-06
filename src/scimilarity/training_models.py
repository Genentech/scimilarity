import csv
from datetime import datetime
import json
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Optional, List

from .triplet_selector import TripletLoss
from .nn_models import Encoder, Decoder


class MetricLearning(pl.LightningModule):
    """A class encapsulating the metric learning.

    Parameters
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
    triplet_loss_weight: float, default 0.001
        The weighting for triplet loss vs reconstruction loss.  This weighting sums to 1
        such that triplet loss weight is triplet_loss_weight and reconstruction loss weight is (1 - triplet_loss_weight).
    margin: float, default: 0.05
        The margin parameter in triplet loss.
    negative_selection: {"semihard", "hardest", "random"}, default: "semihard"
        The negative selection function.
    sample_across_studies: bool, default: True
        Whether to enforce anchor-positive pairs being from different studies.
    perturb_labels: bool, default: False
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
    cosine_annealing_tmax: int, optional, default: None
        The number of epochs for T_max in cosine LR annealing.
        If None, use the max_epochs.
    track_triplets: str, optional, default: None
        Track the triplet composition used in triplet loss and store the files in this directory.
    track_triplets_above_step: int, default: -1,
        When tracking triplet composition, only track for global step above the given value.

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

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        triplet_loss_weight: float = 0.001,
        margin: float = 0.05,
        negative_selection: str = "semihard",
        sample_across_studies: bool = True,
        perturb_labels: bool = False,
        perturb_labels_fraction: float = 0.5,
        lr: float = 5e-3,
        l1: float = 1e-4,
        l2: float = 0.01,
        max_epochs: int = 500,
        cosine_annealing_tmax: Optional[int] = None,
        track_triplets: Optional[str] = None,
        track_triplets_above_step: int = -1,
    ):
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

        self.max_epochs = max_epochs
        self.cosine_annealing_tmax = cosine_annealing_tmax
        if (
            self.cosine_annealing_tmax is None
            or self.cosine_annealing_tmax > max_epochs
        ):
            self.cosine_annealing_tmax = max_epochs

        # networks
        self.encoder = Encoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
        )
        self.decoder = Decoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
        )

        # save layer sizes
        model_state_dict = self.encoder.state_dict()
        self.layer_sizes = {
            entry: list(model_state_dict[entry].size()) for entry in model_state_dict
        }

        # mixed loss weight
        self.triplet_loss_weight = triplet_loss_weight

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
        self.track_triplets = track_triplets
        self.track_triplets_above_step = track_triplets_above_step

        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        """Configure optimizers."""

        optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=self.l2)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cosine_annealing_tmax
            ),
            "interval": "epoch",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler,
        }  # pytorch-lightning required format

    def forward(self, x):
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        z: torch.Tensor
            Output tensor corresponding to the last encoder layer.
        x_hat: torch.Tensor
            Output tensor corresponding to the last decoder layer.
        """

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def get_losses(self, batch, use_studies: bool = True, val_metrics: bool = False):
        """Calculate the triplet and reconstruction loss.

        Parameters
        ----------
        batch:
            A batch as defined by a pytorch DataLoader.
        use_studies: bool, default: True
            Whether to use studies metadata in mining triplets and calculating triplet loss
        val_metrics: bool, default: False
            Whether to include extra validation metrics

        Returns
        -------
        triplet_loss: torch.Tensor
            Triplet loss.
        mse: torch.Tensor
            MSE reconstruction loss
        num_hard_triplets: torch.Tensor
            Number of hard triplets.
        num_viable_triplets: torch.Tensor
            Number of viable triplets.
        """

        cells, labels, studies = batch
        if not use_studies:
            studies = None

        if "sparse" in dir(self.trainer.datamodule) and self.trainer.datamodule.sparse:
            cells = cells.to_dense()
        embedding, reconstruction = self(cells)
        triplet_loss, num_hard_triplets, num_viable_triplets, triplets_idx = (
            self.triplet_loss_fn(
                embedding, labels, self.trainer.datamodule.int2label, studies
            )
        )
        mse = self.mse_loss_fn(cells, reconstruction)
        if val_metrics:
            asw = self.triplet_loss_fn.triplet_selector.get_asw(
                embedding, labels, self.trainer.datamodule.int2label
            )
            nmse = mse / cells.pow(2).mean()
            return (
                triplet_loss,
                mse,
                num_hard_triplets,
                num_viable_triplets,
                asw,
                nmse,
            )
        else:
            if (
                self.track_triplets is not None
                and self.trainer.global_step > self.track_triplets_above_step
            ):
                triplet_df = []
                for i, t in enumerate(["anchor", "positive", "negative"]):
                    df = pd.DataFrame(
                        {
                            f"{t}_label": [
                                self.trainer.datamodule.int2label[int(x)]
                                for x in labels[triplets_idx[i]]
                            ]
                        }
                    )
                    if studies is not None:
                        df[f"{t}_studies"] = studies[triplets_idx[i]]
                    triplet_df.append(df)
                pd.concat(triplet_df, axis=1).to_csv(
                    os.path.join(
                        self.track_triplets,
                        f"train_triplets.{self.trainer.global_step}.csv.gz",
                    ),
                    compression="gzip",
                    quoting=csv.QUOTE_NONNUMERIC,
                )

            return (
                triplet_loss,
                mse,
                num_hard_triplets,
                num_viable_triplets,
            )

    def get_mixed_loss(self, triplet_loss, mse):
        """Calculate the mixed loss.

        Parameters
        ----------
        triplet_loss: torch.Tensor
            Triplet loss.
        mse: torch.Tensor
            MSE reconstruction loss

        Returns
        -------
        torch.Tensor
            Mixed loss.
        """

        if self.triplet_loss_weight == 0:
            return mse
        if self.triplet_loss_weight == 1:
            return triplet_loss
        return (self.triplet_loss_weight * triplet_loss) + (
            (1.0 - self.triplet_loss_weight) * mse
        )

    def training_step(self, batch, batch_idx):
        """Pytorch-lightning training step.

        Parameters
        ----------
        batch:
            A batch as defined by a pytorch DataLoader.
        batch_idx:
            A batch index as defined by a pytorch-lightning.
        """

        (
            triplet_losses,
            mse,
            num_hard_triplets,
            num_viable_triplets,
        ) = self.get_losses(batch, val_metrics=False)

        triplet_loss = triplet_losses.mean()
        num_nonzero_loss = (triplet_losses > 0).sum(dtype=torch.float).detach()
        hard_triplets = num_hard_triplets / num_viable_triplets

        loss = self.get_mixed_loss(triplet_loss, mse)

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
        self.log("train mse", mse, prog_bar=True, logger=True)
        self.log("train hard triplets", hard_triplets, prog_bar=True, logger=True)
        self.log(
            "train num nonzero loss", num_nonzero_loss, prog_bar=False, logger=True
        )
        self.log(
            "train num hard triplets", num_hard_triplets, prog_bar=False, logger=True
        )
        self.log(
            "train num viable triplets",
            num_viable_triplets,
            prog_bar=False,
            logger=True,
        )
        return {
            "loss": loss,
            "triplet_loss": triplet_loss.detach(),
            "mse": mse.detach(),
            "hard_triplets": hard_triplets,
            "num_nonzeros_loss": num_nonzero_loss,
            "num_hard_triplets": num_hard_triplets,
            "num_viable_triplets": num_viable_triplets,
        }

    def on_validation_epoch_start(self):
        """Pytorch-lightning validation epoch start."""
        super().on_validation_epoch_start()
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """Pytorch-lightning validation step.

        Parameters
        ----------
        batch:
            A batch as defined by a pytorch DataLoader.
        batch_idx:
            A batch index as defined by a pytorch-lightning.
        """

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_step(batch, prefix="val")

    def on_validation_epoch_end(self):
        """Pytorch-lightning validation epoch end evaluation."""

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_epoch(prefix="val")

    def on_test_epoch_start(self):
        """Pytorch-lightning test epoch start."""
        super().on_test_epoch_start()
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        """Pytorch-lightning test step.

        Parameters
        ----------
        batch:
            A batch as defined by a pytorch DataLoader.
        batch_idx:
            A batch index as defined by a pytorch-lightning.
        """

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_step(batch, prefix="test")

    def on_test_epoch_end(self):
        """Pytorch-lightning test epoch end evaluation."""

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_epoch(prefix="test")

    def _eval_step(self, batch, prefix: str):
        """Evaluation of validation or test step.

        Parameters
        ----------
        batch:
            A batch as defined by a pytorch DataLoader.
        prefix: str
            A string prefix to label validation versus test evaluation.

        Returns
        -------
        dict
            A dictionary containing step evaluation metrics.
        """

        (
            triplet_losses,
            mse,
            num_hard_triplets,
            num_viable_triplets,
            asw,
            nmse,
        ) = self.get_losses(batch, use_studies=False, val_metrics=True)

        triplet_loss = triplet_losses.mean()
        num_nonzero_loss = (triplet_losses > 0).sum()
        hard_triplets = num_hard_triplets / num_viable_triplets
        evaluation_metric = (1 - asw) / 2 + nmse

        loss = self.get_mixed_loss(triplet_loss, mse)

        losses = {
            f"{prefix}_loss": loss,
            f"{prefix}_triplet_loss": triplet_loss,
            f"{prefix}_mse": mse,
            f"{prefix}_hard_triplets": hard_triplets,
            f"{prefix}_num_nonzero_loss": num_nonzero_loss,
            f"{prefix}_num_hard_triplets": num_hard_triplets,
            f"{prefix}_num_viable_triplets": num_viable_triplets,
            f"{prefix}_asw": asw,
            f"{prefix}_nmse": nmse,
            f"{prefix}_evaluation_metric": evaluation_metric,
        }

        if prefix == "val":
            self.val_step_outputs.append(losses)
        elif prefix == "test":
            self.test_step_outputs.append(losses)
        return losses

    def _eval_epoch(self, prefix: str):
        """Evaluation of validation or test epoch.

        Parameters
        ----------
        prefix: str
            A string prefix to label validation versus test evaluation.

        Returns
        -------
        dict
            A dictionary containing epoch evaluation metrics.
        """

        if prefix == "val":
            step_outputs = self.val_step_outputs
        elif prefix == "test":
            step_outputs = self.test_step_outputs

        loss = torch.Tensor([step[f"{prefix}_loss"] for step in step_outputs]).mean()
        triplet_loss = torch.Tensor(
            [step[f"{prefix}_triplet_loss"] for step in step_outputs]
        ).mean()
        mse = torch.Tensor([step[f"{prefix}_mse"] for step in step_outputs]).mean()
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
        asw = torch.Tensor([step[f"{prefix}_asw"] for step in step_outputs]).mean()
        nmse = torch.Tensor([step[f"{prefix}_nmse"] for step in step_outputs]).mean()
        evaluation_metric = torch.Tensor(
            [step[f"{prefix}_evaluation_metric"] for step in step_outputs]
        ).mean()

        self.log(f"{prefix} loss", loss, logger=True)
        self.log(f"{prefix} triplet loss", triplet_loss, logger=True)
        self.log(f"{prefix} mse", mse, logger=True)
        self.log(f"{prefix} hard triplets", hard_triplets, logger=True)
        self.log(f"{prefix} num nonzero loss", num_nonzero_loss, logger=True)
        self.log(f"{prefix} num hard triplets", num_hard_triplets, logger=True)
        self.log(f"{prefix} num viable triplets", num_viable_triplets, logger=True)
        self.log(f"{prefix} asw", asw, logger=True)
        self.log(f"{prefix} nmse", nmse, logger=True)
        self.log(f"{prefix} evaluation_metric", evaluation_metric, logger=True)

        losses = {
            f"{prefix}_loss": loss,
            f"{prefix}_triplet_loss": triplet_loss,
            f"{prefix}_mse": mse,
            f"{prefix}_hard_triplets": hard_triplets,
            f"{prefix}_num_nonzero_loss": num_nonzero_loss,
            f"{prefix}_num_hard_triplets": num_hard_triplets,
            f"{prefix}_num_viable_triplets": num_viable_triplets,
            f"{prefix}_asw": asw,
            f"{prefix}_nmse": nmse,
            f"{prefix}_evaluation_metric": evaluation_metric,
        }
        return losses

    def save_all(
        self,
        model_path: str,
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
            "triplet_loss_weight": self.triplet_loss_weight,
            "negative_selection": self.negative_selection,
            "sample_across_studies": self.sample_across_studies,
            "perturb_labels": self.perturb_labels,
            "perturb_labels_fraction": self.perturb_labels_fraction,
            "lr": self.lr,
            "l1_lambda": self.l1,
            "l2_lambda": self.l2,
            "batch_size": self.trainer.datamodule.batch_size,
            "max_epochs": self.max_epochs,
        }
        with open(os.path.join(model_path, "hyperparameters.json"), "w") as f:
            f.write(json.dumps(hyperparameters))

        # write gene order
        with open(os.path.join(model_path, "gene_order.tsv"), "w") as f:
            f.write("\n".join(self.trainer.datamodule.gene_order))

        # write dictionary to map label_ints to labels
        pd.Series(self.trainer.datamodule.int2label).to_csv(
            os.path.join(model_path, "label_ints.csv"),
            quoting=csv.QUOTE_NONNUMERIC,
        )

        # write metadata: data paths, timestamp
        meta_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if "train_path" in dir(self.trainer.datamodule):
            meta_data["train_path"] = self.trainer.datamodule.train_path
            meta_data["val_path"] = self.trainer.datamodule.val_path
        elif "cell_metadata_uri" in dir(self.trainer.datamodule):
            meta_data["cell_metadata_uri"] = self.trainer.datamodule.cell_metadata_uri
            meta_data["gene_annotation_uri"] = (
                self.trainer.datamodule.gene_annotation_uri
            )
            meta_data["counts_uri"] = self.trainer.datamodule.counts_uri

            self.trainer.datamodule.train_df.to_csv(
                os.path.join(model_path, "train_cells.csv.gz"),
                compression="gzip",
                quoting=csv.QUOTE_NONNUMERIC,
            )
            if self.trainer.datamodule.val_df is not None:
                self.trainer.datamodule.val_df.to_csv(
                    os.path.join(model_path, "val_cells.csv.gz"),
                    compression="gzip",
                    quoting=csv.QUOTE_NONNUMERIC,
                )
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            f.write(json.dumps(meta_data))

    def load_state(
        self,
        encoder_filename: str,
        decoder_filename: str,
        use_gpu: bool = False,
        freeze: bool = False,
    ):
        """Load model state.

        Parameters
        ----------
        encoder_filename: str
            Filename containing the encoder model state.
        decoder_filename: str
            Filename containing the decoder model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        freeze: bool, default: False
            Freeze all but bottleneck layer, used if pretraining the encoder.
        """

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
