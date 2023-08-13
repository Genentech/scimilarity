from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from scipy.sparse import csr_matrix

mpl.rcParams["pdf.fonttype"] = 42


class SimpleDist(torch.nn.Module):
    """Calculates the distance between representations"""

    def __init__(self, encoder: torch.nn.Module):
        """Constructor.

         Parameters
         ----------
         encoder: torch.nn.Module
             The encoder pytorch object.
         """
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        anchors: torch.Tensor,
        negatives: torch.Tensor,
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
    """A class that interprets significant genes."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        gene_order: list,
    ):
        """Constructor.

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
        self.encoder = encoder
        self.dist_ig = IntegratedGradients(SimpleDist(self.encoder))
        self.gene_order = gene_order

    def get_attributions(
        self,
        anchors: Union[torch.Tensor, np.ndarray, csr_matrix],
        negatives: Union[torch.Tensor, np.ndarray, csr_matrix],
    ) -> np.ndarray:
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

    def get_ranked_genes(self, attrs: np.ndarray) -> pd.DataFrame:
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
        attrs_df: pd.DataFrame,
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

        df = attrs_df.head(n_plot)
        ci = 1.96 * df["attribution_std"] / np.sqrt(df["cells"])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=200)
        ax = sns.barplot(data=df, x="gene", y="attribution", yerr=ci, ax=ax)
        ax.set_yticks([])
        plt.tick_params(axis="x", which="major", labelsize=8, labelrotation=90)

        if filename:  # save the figure
            fig.savefig(filename, bbox_inches="tight")
