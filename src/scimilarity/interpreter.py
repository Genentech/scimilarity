from torch import nn
from typing import Optional, Union


class SimpleDist(nn.Module):
    """Calculates the distance between representations

    Parameters
    ----------
    encoder: torch.nn.Module
        The encoder pytorch object.
    """

    def __init__(self, encoder: "torch.nn.Module"):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        anchors: "torch.Tensor",
        negatives: "torch.Tensor",
    ):
        """Forward.

        Parameters
        ----------
        anchors: torch.Tensor
            Tensor for anchor or positive cells.
        negatives: torch.Tensor
            Tensor for negative cells.

        Returns
        -------
        float
            Sum of squares distance for the encoded tensors.
        """

        f_anc = self.encoder(anchors)
        f_neg = self.encoder(negatives)
        return ((f_neg - f_anc) ** 2).sum(dim=1)


class Interpreter:
    """A class that interprets significant genes.

    Parameters
    ----------
    encoder: torch.nn.Module
        The encoder pytorch object.
    gene_order: list
        The list of genes.

    Examples
    --------
    >>> interpreter = Interpreter(CellEmbedding("/opt/data/model").model)
    """

    def __init__(
        self,
        encoder: "torch.nn.Module",
        gene_order: list,
    ):
        from captum.attr import IntegratedGradients

        self.encoder = encoder
        self.dist_ig = IntegratedGradients(SimpleDist(self.encoder))
        self.gene_order = gene_order

    def get_attributions(
        self,
        anchors: Union["torch.Tensor", "numpy.ndarray", "scipy.sparse.csr_matrix"],
        negatives: Union["torch.Tensor", "numpy.ndarray", "scipy.sparse.csr_matrix"],
    ) -> "numpy.ndarray":
        """Returns attributions, which can later be aggregated.
        High attributions for genes that are expressed more highly in the anchor
        and that affect the distance between anchors and negatives strongly.

        Parameters
        ----------
        anchors:  numpy.ndarray, scipy.sparse.csr_matrix, torch.Tensor
            Tensor for anchor or positive cells.
        negatives: numpy.ndarray, scipy.sparse.csr_matrix, torch.Tensor
            Tensor for negative cells.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of attributions [num_cells x num_genes].

        Examples
        --------
        >>> attr = interpreter.get_attributions(anchors, negatives)
        """

        import numpy as np
        from scipy.sparse import csr_matrix
        import torch

        assert anchors.shape == negatives.shape

        if isinstance(anchors, np.ndarray):
            anc = torch.Tensor(anchors)
        elif isinstance(anchors, csr_matrix):
            anc = torch.Tensor(anchors.todense())
        else:
            anc = anchors

        if isinstance(negatives, np.ndarray):
            neg = torch.Tensor(negatives)
        elif isinstance(negatives, csr_matrix):
            neg = torch.Tensor(negatives.todense())
        else:
            neg = negatives

        # Check if model is on gpu device
        if next(self.encoder.parameters()).is_cuda:
            anc = anc.cuda()
            neg = neg.cuda()

        # attribute l2_dist(anchors, negatives)
        attr = self.dist_ig.attribute(
            anc,
            baselines=neg,  # integrate from negatives to anchors
            additional_forward_args=neg,
        )
        attr *= anc > neg
        attr = +attr.abs()  # signs unreliable, so use absolute value of attributions

        if next(self.encoder.parameters()).is_cuda:
            return attr.detach().cpu().numpy()
        return attr.detach().numpy()

    def get_ranked_genes(self, attrs: "numpy.ndarray") -> "pandas.DataFrame":
        """Get the ranked gene list based on highest attributions.

        Parameters
        ----------
        attr: numpy.ndarray
            Attributions matrix.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the ranked attributions for each gene

        Examples
        --------
        >>> attrs_df = interpreter.get_ranked_genes(attrs)
        """

        import numpy as np
        import pandas as pd

        mean_attrs = attrs.mean(axis=0)
        idx = mean_attrs.argsort()[::-1]
        df = {
            "gene": np.array(self.gene_order)[idx],
            "gene_idx": idx,
            "attribution": mean_attrs[idx],
            "attribution_std": attrs.std(axis=0)[idx],
            "cells": attrs.shape[0],
        }
        return pd.DataFrame(df)

    def plot_ranked_genes(
        self,
        attrs_df: "pandas.DataFrame",
        n_plot: int = 15,
        filename: Optional[str] = None,
    ):
        """Plot the ranked gene attributions.

        Parameters
        ----------
        attrs_df: pandas.DataFrame
            Dataframe of ranked attributions.
        n_plot: int
            The number of top genes to plot.
        filename: str, optional
            The filename to save to plot as.

        Examples
        --------
        >>> interpreter.plot_ranked_genes(attrs_df)
        """

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
        import seaborn as sns

        mpl.rcParams["pdf.fonttype"] = 42

        df = attrs_df.head(n_plot)
        ci = 1.96 * df["attribution_std"] / np.sqrt(df["cells"])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2), dpi=200)
        sns.barplot(ax=ax, data=df, x="gene", y="attribution", hue="gene", dodge=False)
        ax.set_yticks([])
        plt.tick_params(axis="x", which="major", labelsize=8, labelrotation=90)

        ax.errorbar(
            df["gene"].values,
            df["attribution"].values,
            yerr=ci,
            ecolor="black",
            fmt="none",
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        if filename:  # save the figure
            fig.savefig(filename, bbox_inches="tight")
