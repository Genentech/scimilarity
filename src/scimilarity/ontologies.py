import networkx as nx
import obonet
import pandas as pd
from typing import Union, Tuple, List


def subset_nodes_to_set(nodes, restricted_set: Union[list, set]) -> nx.DiGraph:
    """Restrict nodes to a given set.

    Parameters
    ----------
    nodes: networkx.DiGraph
        Node graph.
    restricted_set: list, set
        Restricted node list.

    Returns
    -------
    networkx.DiGraph
        Node graph of restricted set.

    Examples
    --------
    >>> subset_nodes_to_set(nodes, node_list)
    """

    return {node for node in nodes if node in restricted_set}


def import_cell_ontology(
    url="http://purl.obolibrary.org/obo/cl/cl-basic.obo",
) -> nx.DiGraph:
    """Import taxrank cell ontology.

    Parameters
    ----------
    url: str, default: "http://purl.obolibrary.org/obo/cl/cl-basic.obo"
        The url of the ontology obo file.

    Returns
    -------
    networkx.DiGraph
        Node graph of ontology.

    Examples
    --------
    >>> onto = import_cell_ontology()
    """

    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_uberon_ontology(
    url="http://purl.obolibrary.org/obo/uberon/basic.obo",
) -> nx.DiGraph:
    """Import uberon tissue ontology.

    Parameters
    ----------
    url: str, default: "http://purl.obolibrary.org/obo/uberon/basic.obo"
        The url of the ontology obo file.

    Returns
    -------
    networkx.DiGraph
        Node graph of ontology.

    Examples
    --------
    >>> onto = import_uberon_ontology()
    """

    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_doid_ontology(
    url="http://purl.obolibrary.org/obo/doid.obo",
) -> nx.DiGraph:
    """Import doid disease ontology.

    Parameters
    ----------
    url: str, default: "http://purl.obolibrary.org/obo/doid.obo"
        The url of the ontology obo file.

    Returns
    -------
    networkx.DiGraph
        Node graph of ontology.

    Examples
    --------
    >>> onto = import_doid_ontology()
    """

    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def import_mondo_ontology(
    url="http://purl.obolibrary.org/obo/mondo.obo",
) -> nx.DiGraph:
    """Import mondo disease ontology.

    Parameters
    ----------
    url: str, default: "http://purl.obolibrary.org/obo/mondo.obo"
        The url of the ontology obo file.

    Returns
    -------
    networkx.DiGraph
        Node graph of ontology.

    Examples
    --------
    >>> onto = import_mondo_ontology()
    """

    graph = obonet.read_obo(url).reverse()  # flip for intuitiveness
    return nx.DiGraph(graph)  # return as graph


def get_id_mapper(graph) -> dict:
    """Mapping from term ID to name.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.

    Returns
    -------
    dict
        The id to name mapping dictionary.

    Examples
    --------
    >>> id2name = get_id_mapper(onto)
    """

    return {id_: data.get("name") for id_, data in graph.nodes(data=True)}


def get_children(graph, node, node_list=None) -> nx.DiGraph:
    """Get children nodes of a given node.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.

    Returns
    -------
    networkx.DiGraph
        Node graph of children.

    Examples
    --------
    >>> children = get_children(onto, id)
    """

    children = {item[1] for item in graph.out_edges(node)}
    if node_list is None:
        return children
    return subset_nodes_to_set(children, node_list)


def get_parents(graph, node, node_list=None) -> nx.DiGraph:
    """Get parent nodes of a given node.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.

    Returns
    -------
    networkx.DiGraph
        Node graph of parents.

    Examples
    --------
    >>> parents = get_parents(onto, id)
    """

    parents = {item[0] for item in graph.in_edges(node)}
    if node_list is None:
        return parents
    return subset_nodes_to_set(parents, node_list)


def get_siblings(graph, node, node_list=None) -> nx.DiGraph:
    """Get sibling nodes of a given node.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.

    Returns
    -------
    networkx.DiGraph
        Node graph of siblings.

    Examples
    --------
    >>> siblings = get_siblings(onto, id)
    """

    parents = get_parents(graph, node)
    siblings = set.union(
        *[set(get_children(graph, parent)) for parent in parents]
    ) - set([node])
    if node_list is None:
        return siblings
    return subset_nodes_to_set(siblings, node_list)


def get_all_ancestors(graph, node, node_list=None, inclusive=False) -> nx.DiGraph:
    """Get all ancestor nodes of a given node.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.
    inclusive: bool, default: False
        Whether to include the given node in the results.

    Returns
    -------
    networkx.DiGraph
        Node graph of ancestors.

    Examples
    --------
    >>> ancestors = get_all_ancestors(onto, id)
    """

    ancestors = nx.ancestors(graph, node)
    if inclusive:
        ancestors = ancestors | {node}

    if node_list is None:
        return ancestors
    return subset_nodes_to_set(ancestors, node_list)


def get_all_descendants(graph, nodes, node_list=None, inclusive=False) -> nx.DiGraph:
    """Get all descendant nodes of given node(s).

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    nodes: str, list
        ID of given node or a list of node IDs.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.
    inclusive: bool, default: False
        Whether to include the given node in the results.

    Returns
    -------
    networkx.DiGraph
        Node graph of descendants.

    Examples
    --------
    >>> descendants = get_all_descendants(onto, id)
    """

    if isinstance(nodes, str):  # one term id
        descendants = nx.descendants(graph, nodes)
    else:  # list of term ids
        descendants = set.union(*[nx.descendants(graph, node) for node in nodes])

    if inclusive:
        descendants = descendants | {nodes}

    if node_list is None:
        return descendants
    return subset_nodes_to_set(descendants, node_list)


def get_lowest_common_ancestor(graph, node1, node2) -> nx.DiGraph:
    """Get the lowest common ancestor of two nodes.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node1: str
        ID of node1.
    node2: str
        ID of node2.

    Returns
    -------
    networkx.DiGraph
        Node graph of descendants.

    Examples
    --------
    >>> common_ancestor = get_lowest_common_ancestor(onto, id1, id2)
    """

    return nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
        graph, node1, node2
    )


def find_most_viable_parent(graph, node, node_list):
    """Get most viable parent of a given node among the node_list.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node: str
        ID of given node.
    node_list: list, set, optional, default: None
        A restricted node list for filtering.

    Returns
    -------
    networkx.DiGraph
        Node graph of parents.

    Examples
    --------
    >>> coarse_grained = find_most_viable_parent(onto, id, celltype_list)
    """

    parents = get_parents(graph, node, node_list=node_list)
    if len(parents) == 0:
        coarse_grained = None
        all_parents = list(get_parents(graph, node))
        if len(all_parents) == 1:
            grandparents = get_parents(graph, all_parents[0], node_list=node_list)
            if len(grandparents) == 1:
                (coarse_grained,) = grandparents
    elif len(parents) == 1:
        (coarse_grained,) = parents
    else:
        for parent in list(parents):
            coarse_grained = None
            if get_all_ancestors(graph, parent, node_list=pd.Index(parents)):
                coarse_grained = parent
                break
    return coarse_grained


def ontology_similarity(graph, node1, node2, restricted_set=None) -> int:
    """Get the ontology similarity of two terms based on the number of common ancestors.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    node1: str
        ID of node1.
    node2: str
        ID of node2.
    restricted_set: set
        Set of restricted nodes to remove from their common ancestors.

    Returns
    -------
    int
        Number of common ancestors.

    Examples
    --------
    >>> onto_sim = ontology_similarity(onto, id1, id2)
    """

    common_ancestors = get_all_ancestors(graph, node1).intersection(
        get_all_ancestors(graph, node2)
    )
    if restricted_set is not None:
        common_ancestors -= restricted_set
    return len(common_ancestors)


def all_pair_similarities(graph, nodes, restricted_set=None) -> "pandas.DataFrame":
    """Get the ontology similarity of all pairs in a node list.

    Parameters
    ----------
    graph: networkx.DiGraph
        Node graph.
    nodes: list, set
        List of nodes.
    restricted_set: set
        Set of restricted nodes to remove from their common ancestors.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe showing similarity for all node pairs.

    Examples
    --------
    >>> onto_sim = all_pair_similarities(onto, id1, id2)
    """

    import itertools
    import pandas as pd

    node_pairs = itertools.combinations(nodes, 2)
    similarity_df = pd.DataFrame(0, index=nodes, columns=nodes)
    for node1, node2 in node_pairs:
        s = ontology_similarity(
            graph, node1, node2, restricted_set=restricted_set
        )  # too slow, cause recomputes each ancestor
        similarity_df.at[node1, node2] = s
    return similarity_df + similarity_df.T


def ontology_silhouette_width(
    embeddings: "numpy.ndarray",
    labels: List[str],
    onto: nx.DiGraph,
    name2id: dict,
    metric: str = "cosine",
) -> Tuple[float, "pandas.DataFrame"]:
    """Get the average silhouette width of celltypes, being aware of cell ontology such that
       ancestors are not considered inter-cluster and descendants are considered intra-cluster.

    Parameters
    ----------
    embeddings: numpy.ndarray
        Cell embeddings.
    labels: List[str]
        Celltype names.
    onto:
        Cell ontology graph object.
    name2id: dict
        A mapping dictionary of celltype name to id
    metric: str, default: "cosine"
        The distance metric to use for scipy.spatial.distance.cdist().

    Returns
    -------
    asw: float
        The average silhouette width.
    asw_df: pandas.DataFrame
        A dataframe containing silhouette width as well as
        inter and intra cluster distances for all cell types.

    Examples
    --------
    >>> asw, asw_df = ontology_silhouette_width(
            embeddings, labels, onto, name2id, metric="cosine"
        )
    """

    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist

    data = {"label": [], "intra": [], "inter": [], "sw": []}
    for i, name1 in enumerate(labels):
        term_id1 = name2id[name1]
        ancestors = get_all_ancestors(onto, term_id1)
        descendants = get_all_descendants(onto, term_id1)
        distances = cdist(
            embeddings[i].reshape(1, -1), embeddings, metric=metric
        ).flatten()

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
