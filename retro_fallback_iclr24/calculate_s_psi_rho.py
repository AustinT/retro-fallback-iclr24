"""
Contains functions to calculate the s, psi, and rho values used to choose which node to expand.

Calculation is generally done with _message passing_, i.e. updating nodes based on values
from their neighbours, then updating the neighbours based on the updated nodes, and so on.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

import numpy as np
from syntheseus.search.algorithms.best_first.base import PriorityQueue
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode

logger = logging.getLogger(__name__)


def s_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates "retro_fallback_s": an array holding a value of `s` for each sample.

    `s` is a binary variable, indicating whether a molecule is currently
    synthesizable (1) or not (0).

    For OR nodes, this is just max(success of children, purchase success).

    For AND nodes, this is just the product of the success of children and reaction feasibility.

    NOTE: updates depend only on children, so can avoid updating children in update prop.
    """

    # Step 1: calculate new success prob
    if isinstance(node, OrNode):
        success_outcomes = node.data["retro_fallback_b"]
        for or_child in graph.successors(node):
            success_outcomes = np.maximum(success_outcomes, or_child.data["retro_fallback_s"])

    elif isinstance(node, AndNode):
        # Success prob is product of reaction feasibility and children success probs
        success_outcomes = node.data["retro_fallback_f"]
        for and_child in graph.successors(node):
            success_outcomes = success_outcomes * and_child.data["retro_fallback_s"]
    else:
        raise TypeError("Only AND/OR nodes supported.")

    # Step 2: set the new value and return
    assert set(np.unique(success_outcomes)) <= {0, 1}
    old_value: Optional[np.ndarray] = node.data.get("retro_fallback_s", None)
    node.data["retro_fallback_s"] = success_outcomes
    value_changed = old_value is None or not np.allclose(old_value, success_outcomes)
    return value_changed


def psi_update(node: ANDOR_NODE, graph: AndOrGraph, max_feasibility_eps: float = 1e-7) -> bool:
    """
    Updates "retro_fallback_psi": an array holding a value of `psi` for each sample.

    `psi` indicates a potential success probablility achievable by each node
    if one synthesis plan under it is expanded.

    TODO: mention equation defining it in the paper

    psi values for AND nodes are capped at `1-max_feasibility_eps` to aid solving.

    NOTE: updates depend only on children, so can avoid updating children in update prop.
    """

    # Step 1: calculate new success prob
    if isinstance(node, OrNode):
        # Option 1: success from purchasing
        new_value = node.data["retro_fallback_b"]

        # Option 2: success comes from children
        for or_child in graph.successors(node):
            new_value = np.maximum(new_value, or_child.data["retro_fallback_psi"])

        # Option 3: success comes from estimate (only if not expanded)
        # NOTE: not expanded is slightly different from "cannot be expanded",
        # so this update is implicitly relying on nodes which cannot be expanded
        # not being given success estimates
        if not node.is_expanded and "retro_fallback_h" in node.data:
            new_value = np.maximum(new_value, node.data["retro_fallback_h"])

    elif isinstance(node, AndNode):
        # Success prob is product of reaction feasibility and children success probs
        new_value = np.minimum(node.data["retro_fallback_f"], 1 - max_feasibility_eps)  # clip to 1 - eps,
        for and_child in graph.successors(node):
            new_value = new_value * and_child.data["retro_fallback_psi"]
    else:
        raise TypeError("Only AND/OR nodes supported.")

    # Step 2: set the new value and return
    assert np.all(new_value <= 1) and np.all(new_value >= 0)  # should be float in [0, 1]
    old_value: Optional[np.ndarray] = node.data.get("retro_fallback_psi", None)
    node.data["retro_fallback_psi"] = new_value

    # Decide on tolerance for allclose
    atol = min(0.1 * max_feasibility_eps, 1e-6)
    rtol = min(0.9 * max_feasibility_eps, 1e-6)
    value_changed = old_value is None or not np.allclose(
        old_value,
        new_value,
        atol=atol,
        rtol=rtol,
    )
    return value_changed


def rho_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates "retro_fallback_rho": an array holding a value of `rho` for each sample.

    `rho` is essentially the estimated success probability of the single best synthesis
    plan which any node lies on.

    TODO: mention equation defining it in the paper

    NOTE: updates depend only on parents.
    """
    parents = list(graph.predecessors(node))

    # Step 1: calculate new success prob
    if isinstance(node, OrNode):
        if node is graph.root_node:

            # Special formula for root node
            new_value = node.data["retro_fallback_psi"]
        else:
            new_value = parents[0].data["retro_fallback_rho"]
            for parent in parents[1:]:
                new_value = np.maximum(new_value, parent.data["retro_fallback_rho"])

    elif isinstance(node, AndNode):
        assert len(parents) == 1
        parent = parents[0]

        # Do the calculation ignoring 0/0 = NaN, then correct for it later
        # Note that 1/0 can never occur
        with np.errstate(divide="ignore", invalid="ignore"):
            new_value = (
                parent.data["retro_fallback_rho"] / parent.data["retro_fallback_psi"] * node.data["retro_fallback_psi"]
            )
        new_value = np.nan_to_num(new_value, nan=0.0, posinf=np.nan, neginf=np.nan)

    else:
        raise TypeError("Only AND/OR nodes supported.")

    # Step 2: set the new value and return
    assert np.all(new_value <= 1) and np.all(new_value >= 0)  # should be float in [0, 1]
    old_value: Optional[np.ndarray] = node.data.get("retro_fallback_rho", None)
    node.data["retro_fallback_rho"] = new_value
    value_changed = old_value is None or not np.allclose(old_value, new_value)  # NOTE: tight tolerance
    return value_changed


@dataclass
class MessagePassingResult:
    nodes_updated: set[ANDOR_NODE]
    n_iter: int
    n_reset: int
    node_starting_values: dict[ANDOR_NODE, Any]


def add_to_priority_queue_if_priority_changed(queue: PriorityQueue, item: Any, priority: Any):
    if item in queue and queue.get_priority(item) == priority:
        return  # no need to re-add it (which would also remove it)
    queue.push_item(item, priority)


def message_passing_with_resets(
    *,
    nodes: Collection[ANDOR_NODE],
    graph: AndOrGraph,
    update_fn: Callable[[ANDOR_NODE, AndOrGraph], bool],
    update_predecessors: bool = True,
    update_successors: bool = True,
    reset_function: Optional[Callable[[ANDOR_NODE, AndOrGraph], None]] = None,
    node_value_tracker: Optional[Callable[[ANDOR_NODE], Any]] = None,
    update_priority_fn: Optional[Callable[[ANDOR_NODE], float]] = None,
    queue_entry_priority_fn: Optional[Callable[[ANDOR_NODE], float]] = None,  # TODO: rename this?
    num_visits_to_trigger_reset: int = 10_000,
    reset_visit_threshold: int = 1000,
    num_iter_to_reset_everything: Optional[int] = None,
) -> MessagePassingResult:
    """
    Function to run message passing on a graph while tracking how many times each node has
    been updated. If nodes are updated too many times, they are `reset` with an optionally
    provided `reset_function`. If too many total iterations are performed, all nodes
    in the graph are reset (this will only happen once).

    Specifically, once a node has been updated at least `num_visits_to_trigger_reset` times,
    all nodes visited more than `reset_visit_threshold` times are reset using the `reset_function`.

    Additional features are:
    - Tracking starting values of nodes. Useful to know how many nodes actually changed values.
    - Updating nodes in priority order (default is all nodes have the same priority)
    """

    def _default_priority_fn(n):
        return 0  # arbitrary constant

    assert num_visits_to_trigger_reset >= reset_visit_threshold, "Inconsistent reset thresholds"
    update_priority_fn = update_priority_fn or _default_priority_fn
    queue_entry_priority_fn = queue_entry_priority_fn or _default_priority_fn
    num_iter_to_reset_everything = num_iter_to_reset_everything or cast(int, math.inf)

    # Initialize queue of nodes to be updated
    update_queue = PriorityQueue()
    node_to_num_update_without_reset: defaultdict[ANDOR_NODE, int] = defaultdict(int)
    node_starting_values: dict[ANDOR_NODE, Any] = {}
    for n in sorted(nodes, key=queue_entry_priority_fn):
        add_to_priority_queue_if_priority_changed(update_queue, n, update_priority_fn(n))
        del n

    def get_neighbours_to_add(node):
        nodes_to_add = set()
        if update_predecessors:
            nodes_to_add.update(graph.predecessors(node))
        if update_successors:
            nodes_to_add.update(graph.successors(node))
        return nodes_to_add

    # Visit nodes
    n_iter = 0
    n_reset = 0
    reset_every_node = False  # will only do this once
    while len(update_queue) > 0:
        n_iter += 1
        node = update_queue.pop_item().item

        # Optionally track starting values
        if node_value_tracker is not None:
            node_starting_values.setdefault(node, node_value_tracker(node))

        # Perform the update and add neighbours to the queue if value changes
        if update_fn(node, graph):
            node_to_num_update_without_reset[node] += 1
            for n in sorted(get_neighbours_to_add(node), key=queue_entry_priority_fn):
                add_to_priority_queue_if_priority_changed(update_queue, n, update_priority_fn(n))
                del n

        # If a reset function is provided, potentially reset nodes
        if reset_function is not None and not reset_every_node:

            # First, check which nodes (if any) should be reset
            if n_iter >= num_iter_to_reset_everything:

                # Reset type 1: reset all nodes in the graph
                logger.warning(
                    "Too many iterations passed. "
                    f"n_iter={n_iter}, n_reset={n_reset}. "
                    "Will now reset every node in the graph."
                )
                reset_every_node = True
                nodes_to_reset = set(graph.nodes())
                update_queue = PriorityQueue()  # clear the queue, since everything will be re-added
            elif node_to_num_update_without_reset[node] >= num_visits_to_trigger_reset:

                # Reset type 2: only reset nodes updated more than `reset_visit_threshold` times
                nodes_to_reset = {
                    n
                    for n, num_updates in node_to_num_update_without_reset.items()
                    if num_updates >= reset_visit_threshold
                }

            else:

                # No resetting
                nodes_to_reset = set()

            # Second, do the resetting
            if nodes_to_reset:
                n_reset += 1

                nodes_to_add_to_queue = set()
                for reset_node in nodes_to_reset:

                    # First, record starting value
                    if node_value_tracker is not None and reset_node not in node_starting_values:
                        node_starting_values[reset_node] = node_value_tracker(reset_node)

                    # Second, do the reset
                    reset_function(reset_node, graph)
                    node_to_num_update_without_reset[reset_node] = 0

                    # Third, mark node (and neighbours) for addition to queue
                    nodes_to_add_to_queue.add(reset_node)
                    nodes_to_add_to_queue.update(get_neighbours_to_add(reset_node))

                    del reset_node

                # Add nodes to the queue (done at the end to avoid adding the same node multiple times)
                for n in sorted(nodes_to_add_to_queue, key=queue_entry_priority_fn):
                    add_to_priority_queue_if_priority_changed(update_queue, n, update_priority_fn(n))
                    del n

    return MessagePassingResult(set(node_to_num_update_without_reset.keys()), n_iter, n_reset, node_starting_values)


def reset_psi(node: ANDOR_NODE, graph: AndOrGraph) -> None:
    node.data["retro_fallback_psi"] = np.zeros_like(node.data["retro_fallback_psi"])


def reset_rho(node: ANDOR_NODE, graph: AndOrGraph) -> None:
    node.data["retro_fallback_rho"] = np.zeros_like(node.data["retro_fallback_rho"])
