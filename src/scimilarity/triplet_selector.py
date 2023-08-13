from itertools import combinations
from typing import Optional, Union

import numpy as np
import random
import torch

from scimilarity.ontologies import (
    import_cell_ontology,
    get_all_ancestors,
    get_all_descendants,
    get_id_mapper,
    get_parents,
)


class TripletSelector:
    """
    For each anchor-positive pair, mine negative samples to create a triplet.
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
                max_ancestors = (
                    1  # range of ancestor levels. 1: parents, 2: grandparents, ...
                )
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
        total_hard_triplets = 0
        total_viable_triplets = 0
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
                total_hard_triplets += (loss_values > 0).sum()
                total_viable_triplets += loss_values.size

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
            total_hard_triplets,
            total_viable_triplets,
        )

    def get_triplets(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        int2label: dict,
        studies: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    ):
        (
            triplets_idx,
            total_hard_triplets,
            total_viable_triplets,
        ) = self.get_triplets_idx(embeddings, labels, int2label, studies)
        anchor_idx, positive_idx, negative_idx = triplets_idx
        return (
            (
                embeddings[anchor_idx],
                embeddings[positive_idx],
                embeddings[negative_idx],
            ),
            total_hard_triplets,
            total_viable_triplets,
        )

    def pdist(self, vectors):
        vectors_squared_sum = (vectors**2).sum(axis=1)
        distance_matrix = (
            -2 * np.matmul(vectors, np.matrix.transpose(vectors))
            + vectors_squared_sum.reshape(1, -1)
            + vectors_squared_sum.reshape(-1, 1)
        )
        return distance_matrix

    def hardest_negative(self, loss_values):
        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def random_negative(self, loss_values):
        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def semihard_negative(self, loss_values):
        semihard_negatives = np.where(
            np.logical_and(loss_values < self.margin, loss_values > 0)
        )[0]
        return (
            np.random.choice(semihard_negatives)
            if len(semihard_negatives) > 0
            else None
        )
