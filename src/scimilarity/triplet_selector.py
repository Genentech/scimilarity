from itertools import combinations
import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import List, Union, Optional

from .ontologies import (
    import_cell_ontology,
    get_id_mapper,
    get_all_ancestors,
    get_all_descendants,
    get_parents,
)


class TripletSelector:
    """For each anchor-positive pair, mine negative samples to create a triplet.

    Parameters
    ----------
    margin: float
        Triplet loss margin.
    negative_selection: str, default: "semihard"
        Method for negative selection: {"semihard", "hardest", "random"}.
    perturb_labels: bool, default: False
        Whether to perturb the ontology labels by coarse graining one level up.
    perturb_labels_fraction: float, default: 0.5
        The fraction of labels to perturb.

    Examples
    --------
    >>> triplet_selector = TripletSelector(margin=0.05, negative_selection="semihard")
    """

    def __init__(
        self,
        margin: float,
        negative_selection: str = "semihard",
        perturb_labels: bool = False,
        perturb_labels_fraction: float = 0.5,
    ):
        self.margin = margin
        self.negative_selection = negative_selection

        self.onto = import_cell_ontology()
        self.id2name = get_id_mapper(self.onto)
        self.name2id = {value: key for key, value in self.id2name.items()}

        self.perturb_labels = perturb_labels
        self.perturb_labels_fraction = perturb_labels_fraction

    def get_triplets_idx(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        int2label: dict,
        studies: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    ):
        """Get triplets as anchor, positive, and negative cell indices.

        Parameters
        ----------
        embeddings: numpy.ndarray, torch.Tensor
            Cell embeddings.
        labels: numpy.ndarray, torch.Tensor
            Cell labels in integer form.
        int2label: dict
            Dictionary to map labels in integer form to string
        studies: numpy.ndarray, torch.Tensor, optional, default: None
            Studies metadata for each cell.

        Returns
        -------
        triplets: Tuple[List, List, List]
            A tuple of lists containing anchor, positive, and negative cell indices.
        num_hard_triplets: int
            Number of hard triplets.
        num_viable_triplets: int
            Number of viable triplets.
        )
        """

        if isinstance(embeddings, torch.Tensor):
            distance_matrix = self.pdist(embeddings.detach().cpu().numpy())
        else:
            distance_matrix = self.pdist(embeddings)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        if studies is not None and isinstance(studies, torch.Tensor):
            studies = studies.detach().cpu().numpy()

        labels_ids = np.array([self.name2id[int2label[label]] for label in labels])
        labels_set = set(labels)

        if self.perturb_labels:
            labels_ids_set = set(labels_ids.tolist())
            label2int = {value: key for key, value in int2label.items()}
            perturb_list = random.choices(
                np.arange(len(labels)),
                k=int(len(labels) * self.perturb_labels_fraction),
            )
            for i in perturb_list:  # cells chosen for perturbation of labels
                term_id = self.name2id[int2label[labels[i]]]
                parents = set()
                # Max ancestor levels: 1=parents, 2=grandparents, ...
                max_ancestors = 1
                ancestor_level = 0
                while ancestor_level < max_ancestors:
                    ancestor_level += 1
                    if not parents:
                        parents = get_parents(self.onto, term_id)
                    else:
                        current = set()
                        for p in parents:
                            current = current | get_parents(self.onto, p)
                        parents = current
                    found = any((parent in labels_ids_set for parent in parents))
                    if found is True:
                        parents = list(parents)
                        np.random.shuffle(parents)
                        p = next(
                            parent for parent in parents if parent in labels_ids_set
                        )
                        labels[i] = label2int[self.id2name[p]]
                        break  # label perturbed, skip the rest of the ancestors

        triplets = []
        num_hard_triplets = 0
        num_viable_triplets = 0
        for label in labels_set:
            term_id = self.name2id[int2label[label]]
            ancestors = get_all_ancestors(self.onto, term_id)
            descendants = get_all_descendants(self.onto, term_id)
            violating_terms = ancestors.union(descendants)

            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue

            # negatives are labels that are not the same as current and not in violating terms
            negative_indices = np.where(
                np.logical_not(label_mask | np.isin(labels_ids, list(violating_terms)))
            )[0]

            # compute all pairs of anchor-positives
            anchor_positives = list(combinations(label_indices, 2))

            # enforce anchor and positive coming from different studies
            if studies is not None:
                anchor_positives = [
                    (anchor, positive)
                    for anchor, positive in anchor_positives
                    if studies[anchor] != studies[positive]
                ]

            for anchor_positive in anchor_positives:
                loss_values = (
                    distance_matrix[anchor_positive[0], anchor_positive[1]]
                    - distance_matrix[[anchor_positive[0]], negative_indices]
                    + self.margin
                )
                num_hard_triplets += (loss_values > 0).sum()
                num_viable_triplets += loss_values.size

                # select one negative for anchor positive pair based on selection function
                if self.negative_selection == "semihard":
                    hard_negative = self.semihard_negative(loss_values)
                elif self.negative_selection == "hardest":
                    hard_negative = self.hardest_negative(loss_values)
                elif self.negative_selection == "random":
                    hard_negative = self.random_negative(loss_values)
                else:
                    hard_negative = None

                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], hard_negative]
                    )

        if len(triplets) == 0:
            triplets.append([0, 0, 0])

        anchor_idx, positive_idx, negative_idx = tuple(
            map(list, zip(*triplets))
        )  # tuple([list(t) for t in zip(*triplets)])
        return (
            (
                anchor_idx,
                positive_idx,
                negative_idx,
            ),
            num_hard_triplets,
            num_viable_triplets,
        )

    def pdist(self, vectors: np.ndarray):
        """Get pair-wise distance between all cell embeddings.

        Parameters
        ----------
        vectors: numpy.ndarray
            Cell embeddings.

        Returns
        -------
        numpy.ndarray
            Distance matrix of cell embeddings.
        """

        vectors_squared_sum = (vectors**2).sum(axis=1)
        distance_matrix = (
            -2 * np.matmul(vectors, np.matrix.transpose(vectors))
            + vectors_squared_sum.reshape(1, -1)
            + vectors_squared_sum.reshape(-1, 1)
        )
        return distance_matrix

    def hardest_negative(self, loss_values):
        """Get hardest negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.

        Returns
        -------
        int
            Index of selection.
        """

        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def random_negative(self, loss_values):
        """Get random negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.

        Returns
        -------
        int
            Index of selection.
        """

        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def semihard_negative(self, loss_values):
        """Get a random semihard negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.

        Returns
        -------
        int
            Index of selection.
        """

        semihard_negatives = np.where(
            np.logical_and(loss_values < self.margin, loss_values > 0)
        )[0]
        return (
            np.random.choice(semihard_negatives)
            if len(semihard_negatives) > 0
            else None
        )

    def get_asw(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: List[str],
        int2label: dict,
        metric: str = "cosine",
    ) -> float:
        """Get the average silhouette width of celltypes, being aware of cell ontology such that
           ancestors are not considered inter-cluster and descendants are considered intra-cluster.

        Parameters
        ----------
        embeddings: numpy.ndarray, torch.Tensor
            Cell embeddings.
        labels: List[str]
            Celltype names.
        int2label: dict
            Dictionary to map labels in integer form to string
        metric: str, default: "cosine"
            The distance metric to use for scipy.spatial.distance.cdist().

        Returns
        -------
        asw: float
            The average silhouette width.

        Examples
        --------
        >>> asw = ontology_silhouette_width(embeddings, labels, metric="cosine")
        """

        if isinstance(embeddings, torch.Tensor):
            distance_matrix = self.pdist(embeddings.detach().cpu().numpy())
        else:
            distance_matrix = self.pdist(embeddings)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        sw = []
        for i, label1 in enumerate(labels):
            term_id1 = self.name2id[int2label[label1]]
            ancestors = get_all_ancestors(self.onto, term_id1)
            descendants = get_all_descendants(self.onto, term_id1)

            a_i = []
            b_i = {}
            for j, label2 in enumerate(labels):
                if i == j:
                    continue

                term_id2 = self.name2id[int2label[label2]]
                if term_id2 == term_id1 or term_id2 in descendants:  # intra-cluster
                    a_i.append(distance_matrix[i, j])
                elif (
                    term_id2 != term_id1 and term_id2 not in ancestors
                ):  # inter-cluster
                    if term_id2 not in b_i:
                        b_i[term_id2] = []
                    b_i[term_id2].append(distance_matrix[i, j])

            if len(a_i) <= 1 or not b_i:
                continue
            a_i = np.sum(a_i) / (len(a_i) - 1)
            b_i = np.min(
                [
                    np.sum(values) / len(values)
                    for values in b_i.values()
                    if len(values) > 1
                ]
            )

            s_i = (b_i - a_i) / np.max([a_i, b_i])
            sw.append(s_i)
        return np.mean(sw)


class TripletLoss(torch.nn.TripletMarginLoss):
    """
    Wrapper for pytorch TripletMarginLoss.
    Triplets are generated using TripletSelector object which take embeddings and labels
    then return triplets.

    Parameters
    ----------
    margin: float
        Triplet loss margin.
    sample_across_studies: bool, default: True
        Whether to enforce anchor-positive pairs being from different studies.
    negative_selection: str
        Method for negative selection: {"semihard", "hardest", "random"}
    perturb_labels: bool, default: False
        Whether to perturb the ontology labels by coarse graining one level up.
    perturb_labels_fraction: float, default: 0.5
        The fraction of labels to perturb

    Examples
    --------
    >>> triplet_loss = TripletLoss(margin=0.05)
    """

    def __init__(
        self,
        margin: float,
        sample_across_studies: bool = True,
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
            triplets_idx,
            num_violating_triplets,
            num_viable_triplets,
        ) = self.triplet_selector.get_triplets_idx(
            embeddings, labels, int2label, studies
        )

        anchor_idx, positive_idx, negative_idx = triplets_idx
        anchor = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]

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
            triplets_idx,
        )
