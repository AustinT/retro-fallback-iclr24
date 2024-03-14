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


def root_distance_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Update the root distance: the longest synthesis path upwards towards
    the root node.

    For AndNodes (which only have one parent), it is the parent's root distance +1.

    For OrNodes, it is the max of any parent root distance, +1.
    """
    parent_root_distances = [n.data["root_distance"] for n in graph.predecessors(node)]
    if parent_root_distances:  # i.e. non-empty
        if isinstance(node, OrNode):
            new_root_distance = max(parent_root_distances) + 1
        elif isinstance(node, AndNode):
            new_root_distance = min(parent_root_distances) + 1
        else:
            raise RuntimeError(f"Unknown node type: {node}")
    else:
        assert node is graph.root_node
        new_root_distance = 0

    old_root_distance = node.data.get("root_distance", None)
    node.data["root_distance"] = new_root_distance
    return new_root_distance != old_root_distance


def reset_root_distance(node: ANDOR_NODE, graph: AndOrGraph) -> None:
    """
    Resets a node's "root distance" to infinity.
    """
    node.data["root_distance"] = math.inf
