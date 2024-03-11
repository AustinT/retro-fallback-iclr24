from __future__ import annotations

from syntheseus.search.graph.and_or import ANDOR_NODE, AndOrGraph


def leaf_distance_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Update a node's `leaf_distance`: the length of the shortest path to a leaf node.

    Return True if the node's `leaf_distance` changed.
    """
    successors = list(graph.successors(node))
    if successors:
        new_leaf_distance = min(n.data["leaf_distance"] for n in successors) + 1
    else:
        new_leaf_distance = 0

    old_leaf_distance = node.data.get("leaf_distance", None)
    node.data["leaf_distance"] = new_leaf_distance
    return new_leaf_distance != old_leaf_distance
