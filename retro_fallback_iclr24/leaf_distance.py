from __future__ import annotations

import math

from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode


def leaf_distance_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Update a node's `leaf_distance`: the length of the shortest synthesis plan
    downwards from this node to a leaf node.

    For OrNodes with no successors, the `leaf_distance` is 0. Otherwise,
    it is the smallest leaf distance of any child node plus 1.

    For AndNodes (which always have successors), the `leaf_distance` is the
    largest leaf distance of any child node plus 1 (because ALL children of an AND
    node must be included in a synthesis plan).

    Return True if the node's `leaf_distance` changed.
    """
    successor_leaf_distances = [n.data["leaf_distance"] for n in graph.successors(node)]
    if successor_leaf_distances:  # i.e. non-empty
        if isinstance(node, OrNode):
            new_leaf_distance = min(successor_leaf_distances) + 1
        elif isinstance(node, AndNode):
            new_leaf_distance = max(successor_leaf_distances) + 1
        else:
            raise RuntimeError(f"Unknown node type: {node}")
    else:
        new_leaf_distance = 0

    old_leaf_distance = node.data.get("leaf_distance", None)
    node.data["leaf_distance"] = new_leaf_distance
    return new_leaf_distance != old_leaf_distance


def reset_leaf_distance(node: ANDOR_NODE, graph: AndOrGraph) -> None:
    """
    Reset a node's `leaf_distance` to inf.
    """
    node.data["leaf_distance"] = math.inf
