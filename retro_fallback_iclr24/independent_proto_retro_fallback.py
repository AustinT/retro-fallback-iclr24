"""Main retro-fallback algorithm."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Collection
from typing import Optional, Sequence

import numpy as np
from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import run_message_passing

from .stochastic_processes.buyability import BinaryBuyability, IndependentBuyabilityModel
from .stochastic_processes.feasibility import IndependentFeasibilityModel

logger = logging.getLogger(__name__)


class IndependentProtoRetroFallback(
    AndOrSearchAlgorithm[int],
    ValueFunctionMixin[OrNode],
):
    def __init__(
        self,
        *args,
        feasibility_model: IndependentFeasibilityModel,
        buyability_model: Optional[IndependentBuyabilityModel] = None,
        **kwargs,
    ):
        kwargs.setdefault("prevent_repeat_mol_in_trees", True)  # sensible default setting
        super().__init__(*args, **kwargs)
        self.feasibility_model = feasibility_model  # \xi_f
        self.buyability_model = buyability_model or BinaryBuyability(  # \xi_b
            num_samples=self.feasibility_model.num_samples
        )

        # Check certain things
        if self.expand_purchasable_mols:
            raise ValueError("Currently not supported.")
        if not self.set_depth:
            raise ValueError("Currently not supported.")

    def reset(self) -> None:
        super().reset()
        self.feasibility_model.reset()
        self.buyability_model.reset()

    @property
    def requires_tree(self) -> bool:
        return True  # must use a tree!

    def setup(self, graph: AndOrGraph) -> None:
        # If there is only one node, set its success estimate arbitrarily
        # This saves a call to the value function
        if len(graph) == 1:
            # NOTE: could fail if input node is purchasble.
            # Deal with this in a nice way later
            graph.root_node.data.setdefault("rfb_success_estimate", 1.0)

        return super().setup(graph)

    def _run_from_graph_after_setup(self, graph: AndOrGraph) -> int:
        # Logging setup
        log_level = logging.DEBUG - 1
        logger_active = logger.isEnabledFor(log_level)

        # Run search until time limit or no nodes to expand
        step = 0
        terminate = False
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph) or terminate:
                break

            chosen_leaf = self._descend_tree_and_choose_node(graph)
            if chosen_leaf.is_expanded:
                # Should only happen if there are no other options,
                # so the algorithm will terminate
                action = "terminating because node is already expanded"
                terminate = True
            elif chosen_leaf.data["rfb_ind_succ_prob"] >= chosen_leaf.data["rfb_max_success_1_expand"]:
                action = "terminating because no improvement is estimated."
                terminate = True
            else:
                new_nodes = self.expand_node(chosen_leaf, graph)  # expand the node

                # Clear success estimate. This is important to avoid some edge cases
                # where the node is visited repeatedly. For example, if the depth
                # is >= max depth then the node will never actually be expanded but will
                # keep being visited due to a high success estimate
                chosen_leaf.data["rfb_success_estimate"] = 0.0

                # Run updates so that next node can be chosen
                updated_nodes = self.set_node_values(new_nodes + [chosen_leaf], graph)
                action = f"expanded: {len(new_nodes)} new nodes created, {len(updated_nodes)} nodes updated)"

            if logger_active:
                logger.log(
                    level=log_level,
                    msg=f"Step {step}:\tnode={chosen_leaf}, action={action}",
                )

        return step

    def _descend_tree_and_choose_node(self, graph) -> OrNode:
        """Returns a leaf node on the optimal expansion route."""

        # Descend the tree along the optimal route to find candidate nodes
        candidate_nodes: list[OrNode] = []
        nodes_to_descend: deque[OrNode] = deque([graph.root_node])
        while len(nodes_to_descend) > 0:
            node = nodes_to_descend.popleft()
            children = list(graph.successors(node))
            if len(children) == 0:
                # No children, so this is a candidate node
                candidate_nodes.append(node)
            else:
                # Find AndChildren with matching argmax
                v1, v2 = node.data["rfb_argmax_success_1_expand"]
                matching_children = [
                    child
                    for child in children
                    if math.isclose(child.data["rfb_max_success_1_expand"], v2, rel_tol=1e-4)
                    and math.isclose(child.data["rfb_ind_succ_prob"], v1, rel_tol=1e-4)
                ]
                assert len(matching_children) > 0
                chosen_and = matching_children[0]
                for grandchild in graph.successors(chosen_and):
                    nodes_to_descend.append(grandchild)

        # Now there should be at least one candidate node
        assert len(candidate_nodes) > 0

        # Choose candidates with lowest success probability,
        # breaking ties with creation time and depth
        candidate_nodes.sort(
            key=lambda _node: (
                _node.data["rfb_ind_succ_prob"],
                _node.creation_time,
                _node.depth,
            )
        )
        return candidate_nodes[0]

    def set_node_values(  # type: ignore[override]
        self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph
    ) -> Collection[ANDOR_NODE]:
        # Call superclass (to update depth)
        output_nodes = super().set_node_values(nodes, graph)
        del nodes  # unused

        # Fill in feasibilities, purchasabilities, and success estimates
        self._set_purchasabilities(
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode) and "rfb_marginal_purchase_prob" not in node.data
            ],
            graph=graph,
        )
        self._set_feasibilities(
            and_nodes=[
                node
                for node in output_nodes
                if isinstance(node, AndNode) and "rfb_marginal_rxn_feasible_prob" not in node.data
            ],
            graph=graph,
        )
        self._set_success_estimate(  # only for unexpanded leaf nodes
            or_nodes=[
                node
                for node in output_nodes
                if isinstance(node, OrNode) and "rfb_success_estimate" not in node.data and not node.is_expanded
            ],
            graph=graph,
        )

        # Update success probabilities efficiently
        return self._run_retro_fallback_updates(output_nodes, graph)

    def _set_purchasabilities(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        purchasabilities = self.buyability_model.marginal_probability({node.mol for node in or_nodes})
        for node in or_nodes:
            node.data["rfb_marginal_purchase_prob"] = purchasabilities[node.mol]

    def _set_feasibilities(self, and_nodes: Sequence[AndNode], graph: AndOrGraph) -> None:
        feasibilities = self.feasibility_model.marginal_probability({node.reaction for node in and_nodes})
        for node in and_nodes:
            node.data["rfb_marginal_rxn_feasible_prob"] = feasibilities[node.reaction]

    def _set_success_estimate(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        values = self.value_function(or_nodes, graph=graph)
        assert len(values) == len(or_nodes)
        for node, v in zip(or_nodes, values):
            node.data["rfb_success_estimate"] = v

    def _run_retro_fallback_updates(self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph) -> Collection[ANDOR_NODE]:
        # NOTE: the following updates assume that depth is set correctly.
        # Update success probability and max potential improvement from bottom-up,
        # sorting in reverse order of depth for efficiency
        return run_message_passing(
            graph=graph,
            nodes=sorted(nodes, key=lambda n: n.depth, reverse=True),
            update_fns=[
                independent_success_prob_update,
                max_success_1_expand_update,
            ],
            update_predecessors=True,
            update_successors=False,
        )


def independent_success_prob_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates "rfb_ind_succ_prob", the probability that a node can be synthesized/
    or purchased in the current graph assuming all probabilities are independent.
    Returns whether value changed.

    For OrNodes, this is 1 - (1 - purchase prob) * product(1 - child succ prob for all childre)

    For AndNodes, this is prod(child succ prob for all children)

    NOTE: updates depend only on children, so can avoid updating children in update prop.
    """

    # Step 1: calculate new success prob
    if isinstance(node, OrNode):
        # Compute the probability of failing to purchase the mol,
        # and of all children failing
        fail_prob = 1.0 - node.data["rfb_marginal_purchase_prob"]
        for or_child in graph.successors(node):
            fail_prob *= 1.0 - or_child.data["rfb_ind_succ_prob"]

        # Success prob is 1 - fail prob
        new_success_prob = 1.0 - fail_prob
    elif isinstance(node, AndNode):
        # Success prob is product of reaction feasibility and children success probs
        new_success_prob = node.data["rfb_marginal_rxn_feasible_prob"]
        for and_child in graph.successors(node):
            new_success_prob = new_success_prob * and_child.data["rfb_ind_succ_prob"]
    else:
        raise TypeError("Only AND/OR nodes supported.")

    # Step 2: set the new value and return
    assert 0 <= new_success_prob <= 1
    old_value: Optional[float] = node.data.get("rfb_ind_succ_prob")
    node.data["rfb_ind_succ_prob"] = new_success_prob
    value_changed = old_value is None or not math.isclose(  # counts as changed if it is not set before
        old_value, new_success_prob
    )
    return value_changed


def max_success_1_expand_update(node: ANDOR_NODE, graph: AndOrGraph) -> bool:
    """
    Updates the maximum predicted success if a single optimal route is chosen
    below this node, assuming that the "rfb_ind_succ_prob" of the expanded
    nodes is updated to "rfb_success_estimate". This value is called:
    "rfb_max_success_1_expand"

    For an AndNode, the max success probability occurs if each child node is
    expanded optimally. Therefore the value is the product of
    "rfb_max_success_1_expand" of its children nodes.

    For an OrNode, it is more complicated:
    - If it is unexpanded, then by definition the value is "rfb_success_estimate"
    - If it is expanded but has no children, then it is "rfb_ind_succ_prob"
      (its proven success value)
    - If it is expanded with children, it is the highest value achievable by taking
      1 AndNode child and setting its value to "rfb_max_success_1_expand".
      This turns out to be the child with the smallest value of (1 - estimate) / (1 - success prob)

      In this case it also keeps a record of which AndNode this was.

    NOTE: depends upon these values being correctly set beforehand:
    - "rfb_ind_succ_prob"
    - "rfb_marginal_rxn_feasible_prob"
    - "rfb_marginal_purchase_prob"
    - "rfb_success_estimate"

    NOTE: updates depend only on children, so can avoid updating children in update prop.
    """

    # Step 1: calculate new value
    children = list(graph.successors(node))
    improvement_params = None
    if isinstance(node, AndNode):
        # Product of marginal success prob and child values
        new_value = node.data["rfb_marginal_rxn_feasible_prob"]
        for child in children:
            new_value *= child.data["rfb_max_success_1_expand"]
    elif isinstance(node, OrNode):
        if len(children) == 0:
            if node.is_expanded:
                new_value = node.data["rfb_ind_succ_prob"]
            else:
                # Handle edge case where success estimate is lower than proven probability
                # (e.g. success estimator is not aware of which molecules are purchasable)
                new_value = max(node.data["rfb_success_estimate"], node.data["rfb_ind_succ_prob"])
        else:
            # Best node is determined by improvement factor = (1 - estimate) / (1 - success prob)
            # (always between 0 and 1; lower is better here)
            improvement_factors: list[float] = list()
            for child in children:
                if math.isclose(child.data["rfb_ind_succ_prob"], 1.0):
                    # if prob is 1, then estimate should also be 1, so factor is just 1
                    assert math.isclose(child.data["rfb_max_success_1_expand"], 1.0)
                    improvement_factors.append(1.0)
                else:
                    improvement_factors.append(
                        (1.0 - child.data["rfb_max_success_1_expand"]) / (1.0 - child.data["rfb_ind_succ_prob"])
                    )

            # New fail prob = previous fail prob * min improvement factor
            # Note: relies on his value being correctly set for this node (from success_prob_update)
            old_fail_prob = 1.0 - node.data["rfb_ind_succ_prob"]
            improvement_argmin = int(np.argmin(improvement_factors))
            new_fail_prob = old_fail_prob * improvement_factors[improvement_argmin]
            new_value = 1.0 - new_fail_prob

            # Record succ prob and max success 1 expand to search tree later
            c_best = children[improvement_argmin]
            improvement_params = (
                c_best.data["rfb_ind_succ_prob"],
                c_best.data["rfb_max_success_1_expand"],
            )
    else:
        raise TypeError("Only AND/OR nodes supported.")

    # Sanity check: new value should never be lower than proven success probability
    assert new_value >= node.data["rfb_ind_succ_prob"]  # estimate should never be lower than prob

    # Step 2: set the new value
    assert 0 <= new_value <= 1
    old_value: Optional[float] = node.data.get("rfb_max_success_1_expand")
    node.data["rfb_max_success_1_expand"] = new_value
    value_changed = old_value is None or not math.isclose(
        old_value, new_value
    )  # counts as changed if it is not set before

    # Step 3 (for OrNodes only): set new argmax.
    # (NOTE: it is more like a certificate now...)
    if isinstance(node, OrNode) and improvement_params is not None:
        old_argmax = node.data.get("rfb_argmax_success_1_expand")
        node.data["rfb_argmax_success_1_expand"] = improvement_params
        argmax_changed = (
            old_argmax is None
            or not math.isclose(old_argmax[0], improvement_params[0])
            or not math.isclose(old_argmax[1], improvement_params[1])
        )
    else:
        argmax_changed = False

    # Step 4: return
    return value_changed or argmax_changed
