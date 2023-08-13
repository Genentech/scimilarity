import itertools
from typing import Tuple

import networkx as nx
import numpy as np
import obonet
import pandas as pd
from scipy.spatial.distance import cdist


def subset_nodes_to_set(nodes, restricted_set):
    return {node for node in nodes if node in restricted_set}


def import_cell_ontology(
    url="http://purl.obolibrary.org/obo/cl/cl-basic.obo",
) -> nx.DiGraph:
    """Read the taxrank ontology.

    Parameters
    ----------
    url: str
        URL for the cell ontology.

    Returns
    -------
    networkx.DiGraph
        DiGraph containing the cell ontology.
    """
    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_uberon_ontology(
    url="http://purl.obolibrary.org/obo/uberon/basic.obo",
) -> nx.DiGraph:
    """Read the uberon ontology.

    Parameters
    ----------
    url: str
        URL for the uberon ontology.

    Returns
    -------
    networkx.DiGraph
        DiGraph containing the uberon ontology.
    """
    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_doid_ontology(
    url="http://purl.obolibrary.org/obo/doid.obo",
) -> nx.DiGraph:
    """Read the doid ontology.

    Parameters
    ----------
    url: str
        URL for the doid ontology.

    Returns
    -------
    networkx.DiGraph
        DiGraph containing the doid ontology.
    """
    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_mondo_ontology(
    url="http://purl.obolibrary.org/obo/mondo.obo",
) -> nx.DiGraph:
    """Read the mondo ontology.

    Parameters
    ----------
    url: str
        URL for the mondo ontology.

    Returns
    -------
    networkx.DiGraph
        DiGraph containing the mondo ontology.
    """
    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def get_id_mapper(graph) -> dict:
    """Mapping from term ID to name.

    Parameters
    ----------
    graph: networkx.DiGraph
        onotology graph.

    Returns
    -------
    dict
        Dictionary containing the term ID to name mapper.
    """
    return {id_: data.get("name") for id_, data in graph.nodes(data=True)}


def get_children(graph, node, node_list=None):
    children = {item[1] for item in graph.out_edges(node)}
    if node_list is None:
        return children
    return subset_nodes_to_set(children, node_list)


def get_parents(graph, node, node_list=None):
    parents = {item[0] for item in graph.in_edges(node)}
    if node_list is None:
        return parents
    return subset_nodes_to_set(parents, node_list)


def get_siblings(graph, node):
    parents = get_parents(graph, node)
    siblings = set.union(
        *[set(get_children(graph, parent)) for parent in parents]
    ) - set([node])
    return siblings


def get_all_ancestors(graph, node, node_list=None, inclusive=False):
    ancestors = nx.ancestors(graph, node)
    if inclusive:
        ancestors = ancestors | {node}

    if node_list is None:
        return ancestors
    return subset_nodes_to_set(ancestors, node_list)


def get_all_descendants(graph, nodes, node_list=None, inclusive=False):
    if isinstance(nodes, str):  # one term id
        descendants = nx.descendants(graph, nodes)
    else:  # list of term ids
        descendants = set.union(*[nx.descendants(graph, node) for node in nodes])

    if inclusive:
        descendants = descendants | {nodes}

    if node_list is None:
        return descendants
    return subset_nodes_to_set(descendants, node_list)


def get_lowest_common_ancestor(graph, node1, node2):
    return nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
        graph, node1, node2
    )


def ontology_similarity(graph, term1, term2, blacklisted_terms=None):
    common_ancestors = get_all_ancestors(graph, term1).intersection(
        get_all_ancestors(graph, term2)
    )
    if blacklisted_terms is not None:
        common_ancestors -= blacklisted_terms
    return len(common_ancestors)


def all_pair_similarities(graph, used_terms, blacklisted_terms=None):
    node_pairs = itertools.combinations(used_terms, 2)
    similarity_df = pd.DataFrame(0, index=used_terms, columns=used_terms)
    for (term1, term2) in node_pairs:
        s = ontology_similarity(
            graph, term1, term2, blacklisted_terms=blacklisted_terms
        )  # too slow, cause recomputes each ancestor
        similarity_df.at[term1, term2] = s
    return similarity_df + similarity_df.T


def ontology_silhouette_width(
    embeddings: np.ndarray,
    labels: list,
    onto: nx.DiGraph,
    name2id: dict,
    metric: str = "cosine",
) -> Tuple[float, pd.DataFrame]:
    data = {"label": [], "intra": [], "inter": [], "sw": []}
    for i, name1 in enumerate(labels):
        term_id1 = name2id[name1]
        ancestors = get_all_ancestors(onto, term_id1)
        descendants = get_all_descendants(onto, term_id1)
        distances = cdist(embeddings[i].reshape(1, -1), embeddings, metric=metric).flatten()

        a_i = []
        b_i = {}
        for j, name2 in enumerate(labels):
            if i == j:
                continue

            term_id2 = name2id[name2]
            if term_id2 == term_id1 or term_id2 in descendants:  # intra-cluster
                a_i.append(distances[j])
            elif term_id2 != term_id1 and term_id2 not in ancestors:  # inter-cluster
                if term_id2 not in b_i:
                    b_i[term_id2] = []
                b_i[term_id2].append(distances[j])

        if len(a_i) <= 1 or not b_i:
            continue
        a_i = np.sum(a_i) / (len(a_i) - 1)
        b_i = np.min(
            [np.sum(values) / len(values) for values in b_i.values() if len(values) > 1]
        )

        s_i = (b_i - a_i) / np.max([a_i, b_i])

        data["label"].append(name1)
        data["intra"].append(a_i)
        data["inter"].append(b_i)
        data["sw"].append(s_i)
    return np.mean(data["sw"]), pd.DataFrame(data)
