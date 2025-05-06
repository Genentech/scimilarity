"""
This file contains the neural network architectures.
These are all you need for inference.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class Encoder(nn.Module):
    """A class that encapsulates the encoder.

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
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        torch.Tensor
            Output tensor corresponding to output layer.
        """

        for i, layer in enumerate(self.network):
            x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save model state.

        Parameters
        ----------
        filename: str
            Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])


class Decoder(nn.Module):
    """A class that encapsulates the decoder.

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
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        torch.Tensor
            Output tensor corresponding to output layer.
        """
        for i, layer in enumerate(self.network):
            x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save model state.

        Parameters
        ----------
        filename: str
            Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
