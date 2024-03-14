"""Main retro-fallback algorithm."""

from __future__ import annotations

import logging
import math
from collections.abc import Collection
from typing import Optional, Sequence

import numpy as np
from syntheseus import Molecule, SingleProductReaction
from syntheseus.search.algorithms.base import AndOrSearchAlgorithm
from syntheseus.search.algorithms.best_first.base import GeneralBestFirstSearch, PriorityQueue
from syntheseus.search.algorithms.mixins import ValueFunctionMixin
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode

from .calculate_s_psi_rho import (
    MessagePassingResult,
    message_passing_with_resets,
    psi_update,
    reset_psi,
    reset_rho,
    rho_update,
    s_update,
)
from .graph_distances import leaf_distance_update, reset_leaf_distance, reset_root_distance, root_distance_update
from .stochastic_processes import BinaryBuyability, BuyabilityModel, FeasibilityModel

logger = logging.getLogger(__name__)


class RetroFallbackSearch(
    AndOrSearchAlgorithm[int],
    GeneralBestFirstSearch[AndOrGraph],
    ValueFunctionMixin[OrNode],
):

    def __init__(
        self,
        *args,
        feasibility_model: FeasibilityModel,
        buyability_model: Optional[BuyabilityModel] = None,
        early_stopping_SSP: float = 1.0,
        stepwise_log_level: int = logging.DEBUG - 1,
        **kwargs,
    ) -> None:
        # Before init, change default setting to run on a graph
        kwargs.setdefault("prevent_repeat_mol_in_trees", False)
        kwargs.setdefault("unique_nodes", True)

        # Normal init
        super().__init__(*args, **kwargs)

        # Set all attributes
        self.feasibility_model = feasibility_model  # \xi_f
        self.buyability_model = buyability_model or BinaryBuyability(  # \xi_b
            num_samples=self.feasibility_model.num_samples
        )
        assert (
            self.feasibility_model.num_samples == self.buyability_model.num_samples
        ), "Feasibility and buyability models must have the same number of samples."
        self.early_stopping_SSP = early_stopping_SSP
        self.stepwise_log_level = stepwise_log_level

        # Some default values which can optionally be overwritten
        self.track_update_changes = True
        self.num_visits_to_trigger_reset = 20
        self.reset_visit_threshold = 2

        # Check certain things
        if not self.set_depth:
            raise ValueError("Currently not supported: set_depth=False (some updates use depth information)")

    def reset(self) -> None:
        super().reset()
        self.feasibility_model.reset()
        self.buyability_model.reset()

    def should_stop_search(self, graph) -> bool:
        """In addition to checks from superclass, also check for SSP reaching a pre-defined threshold."""
        current_SSP = graph.root_node.data["retro_fallback_s"].mean()
        early_stopping_success = current_SSP > self.early_stopping_SSP
        if early_stopping_success:
            logger.debug(f"Early stopping: SSP={current_SSP} > {self.early_stopping_SSP}")
        return super().should_stop_search(graph) or early_stopping_success

    @property
    def requires_tree(self) -> bool:

        # Can also run on a graph (in fact this is the default)
        return False

    def setup(self, graph: AndOrGraph) -> None:

        # Clear and samples from stochastic processes
        for node in graph.nodes():
            node.data.pop("retro_fallback_b", None)
            node.data.pop("retro_fallback_f", None)

        # If there is only one node, set its success estimate arbitrarily
        # This saves a call to the value function
        if len(graph) == 1:
            graph.root_node.data.setdefault("retro_fallback_h", 0.5)  # arbitrary value between 0 and 1

        # Call superclass setup
        return super().setup(graph)

    def priority_function(self, node: ANDOR_NODE, graph: AndOrGraph):
        """
        Return the priority of a node (lower is better).

        Priority is based on the expected improvement in success probability,
        called "alpha" in the paper (higher values are better, so we use -alpha).

        For tie-breaking we use lowest success probability over all unsolved samples,
        and then node creation time. After that it is arbitrary.
        """
        # Find which samples the root node is solved in
        root_solved_mask = np.isclose(graph.root_node.data["retro_fallback_s"], 1.0)

        # Compute alpha by averaging over UNSOLVED samples
        alpha_value = float(np.ma.array(node.data["retro_fallback_rho"], mask=root_solved_mask).mean())

        # Compute tie-breaking value by averaging success outcomes
        success_tiebreaker = float(np.ma.array(node.data["retro_fallback_s"], mask=root_solved_mask).mean())

        # Edge case: if all values are solved, the mean value will also be masked.
        # Assign an arbitrary value of 0.0 in this case.
        if alpha_value is np.ma.masked:
            alpha_value = 0.0
            success_tiebreaker = 0.0

        return (-alpha_value, success_tiebreaker, node.creation_time)

    def _run_from_graph_after_setup(self, graph: AndOrGraph) -> int:

        # NOTE: this is essentially the same as the superclass method, but
        # with some modifications for logging, and handling  tuple-valued priorities.

        # Initialize queue.
        # Modifications from superclass:
        # - Priority function is tuple-valued
        # - inf priority cannot occur, so we don't check for it
        queue = PriorityQueue()
        for node in graph.nodes():
            if self.node_eligible_for_queue(node, graph):
                queue.push_item(node, self.priority_function(node, graph))
        logger.log(self.stepwise_log_level, f"Initial queue has {len(queue)} nodes")

        # Run search until time limit or queue is empty
        step = 0
        for step in range(self.limit_iterations):
            if self.should_stop_search(graph):
                break

            if len(queue) == 0:
                logger.debug(f"Queue is empty on step {step}. Stopping search.")
                break

            # Pop a node (and perform some validity checks)
            pq_item = queue.pop_item()
            node = pq_item.item
            assert self.node_eligible_for_queue(node, graph), "Nodes in the priority queue should always be eligible"

            # Visit node (generally means expanding it)
            new_nodes = list(self.visit_node(node, graph))

            # Update all node values (the most complex step of the algorithm)
            nodes_updated = self.set_node_values(new_nodes + [node], graph)

            # Add nodes to the queue whose priority may have changed (or are new)
            # dict.fromkeys is to preserve order and uniqueness
            # Modification from superclass: no handling inf priority
            for updated_node in dict.fromkeys(new_nodes + list(nodes_updated)):
                if self.node_eligible_for_queue(updated_node, graph):
                    updated_node_priority = self.priority_function(updated_node, graph)
                    already_in_queue_at_correct_priority = (
                        updated_node in queue and queue.get_priority(updated_node) == updated_node_priority
                    )
                    if not already_in_queue_at_correct_priority:
                        queue.push_item(updated_node, updated_node_priority)

            # Log results for this step (although don't bother making the string if logging is not active)
            # Modification: change numpy print options to get a reasonable preview
            if logger.isEnabledFor(self.stepwise_log_level):
                with np.printoptions(threshold=10, precision=3):
                    logging_str = (
                        f"Step: {step}, "
                        f"Nodes affected during visit: {len(new_nodes)}, "
                        f"Nodes updated: {len(nodes_updated)}, "
                        f"Graph size: {len(graph)}, "
                        f"Queue size: {len(queue)} (raw length: {queue.raw_len()}), "
                        f"Reaction model calls: {self.reaction_model.num_calls()}, "
                        f"Current SSP: {graph.root_node.data['retro_fallback_s'].mean():.5f}, "
                        f"Node visited: {node}"
                    )
                logger.log(
                    self.stepwise_log_level,
                    logging_str,
                )

        return step

    def _set_feasibilities_and_buyabilities(self, graph: AndOrGraph) -> None:
        r"""
        Sets \xi_f and \xi_b samples for retro-fallback.

        Because they are samples from the joint distribution,
        they are in general dependent, and therefore it must
        be done simultaneously for the whole graph.
        """

        # Get a list of molecules and their samples
        or_nodes_without_samples: set[OrNode] = set()
        and_nodes_without_samples: set[AndNode] = set()
        mol_to_samples: dict[Molecule, np.ndarray] = dict()
        rxn_to_samples: dict[SingleProductReaction, np.ndarray] = dict()
        for node in graph.nodes():
            # NOTE: this update does not account for the possibility
            # that nodes could occur multiple times in the graph but have different samples.
            # In theory this shouldn't happen because samples are reset at initialization,
            # but this could potentially be overridden by a user.
            if isinstance(node, OrNode):
                if "retro_fallback_b" not in node.data:
                    or_nodes_without_samples.add(node)
                else:
                    mol_to_samples[node.mol] = node.data["retro_fallback_b"]
            elif isinstance(node, AndNode):
                if "retro_fallback_f" not in node.data:
                    and_nodes_without_samples.add(node)
                else:
                    rxn_to_samples[node.reaction] = node.data["retro_fallback_f"]

        # Update 1: samples for new molecules
        # ==================================================

        # Now, calculate which molecules don't have samples.
        # This is not the same as nodes without samples,
        # since the same molecule could [potentially]
        # occur multiple times in the graph.
        mols_without_samples = set([node.mol for node in or_nodes_without_samples]) - set(mol_to_samples.keys())

        # If there are any molecules without samples, sample them
        if len(mols_without_samples) > 0:
            mol_to_samples.update(
                self.buyability_model.posterior_sample(inputs=mols_without_samples, observed_samples=mol_to_samples)
            )

        # Update nodes without samples
        for node in or_nodes_without_samples:
            node.data["retro_fallback_b"] = mol_to_samples[node.mol]

        # Update 2: samples for new reactions
        # ==================================================
        # (largely follows above)

        rxns_without_samples = set([node.reaction for node in and_nodes_without_samples]) - set(rxn_to_samples.keys())
        if len(rxns_without_samples) > 0:
            rxn_to_samples.update(
                self.feasibility_model.posterior_sample(inputs=rxns_without_samples, observed_samples=rxn_to_samples)
            )
        for node in and_nodes_without_samples:
            node.data["retro_fallback_f"] = rxn_to_samples[node.reaction]

    def _initialize_heuristic(self, or_nodes: Sequence[OrNode], graph: AndOrGraph) -> None:
        values: list[float] = self.value_function(or_nodes, graph=graph)
        assert len(values) == len(or_nodes)
        for node, v in zip(or_nodes, values):
            node.data["retro_fallback_h"] = v

    def set_node_values(self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph) -> Collection[ANDOR_NODE]:

        # Call superclass (to update depth)
        updated_nodes = super().set_node_values(nodes, graph)
        del nodes  # unused

        # Fill in feasibilities, buyabilities, and success estimates
        self._set_feasibilities_and_buyabilities(graph)
        self._initialize_heuristic(  # only for unexpanded leaf nodes
            or_nodes=[
                node
                for node in updated_nodes
                if isinstance(node, OrNode)
                and "retro_fallback_h" not in node.data
                and self.can_expand_node(node, graph)
            ],
            graph=graph,
        )

        # Update s, psi, rho
        return self._update_s_psi_rho(updated_nodes, graph)

    def _update_s_psi_rho(self, nodes: Collection[ANDOR_NODE], graph: AndOrGraph) -> Collection[ANDOR_NODE]:
        nodes_to_update = set(nodes)
        del nodes

        def _log_update_result(update_name: str, result: MessagePassingResult, tracking_fn) -> None:
            """Helper function for logging."""
            if tracking_fn is None:
                tracking_str = "No tracking of which nodes values actually changed."
            else:
                n_change = sum(
                    not np.allclose(result.node_starting_values[node], tracking_fn(node))
                    for node in result.nodes_updated
                )
                tracking_str = f"{n_change} / {len(result.nodes_updated)} nodes actually changed value."
            logger.log(
                self.stepwise_log_level,
                f"{update_name} updates: {len(result.nodes_updated)} / {len(graph)} nodes updated "
                f"in {result.n_iter} iterations "
                f"with {result.n_update} total updates. "
                f"and {result.n_reset} resets. "
                f"{tracking_str}",
            )

        # Define "tracking" functions for logging
        if self.track_update_changes:

            def s_tracker(node):
                return node.data["retro_fallback_s"]

            def psi_tracker(node):
                return node.data["retro_fallback_psi"]

            def rho_tracker(node):
                return node.data["retro_fallback_rho"]

        else:
            # If we don't track changes, set these to None
            # (type ignore is because "callable or None" type is not clear to mypy)
            s_tracker = psi_tracker = rho_tracker = None  # type: ignore[assignment]

        # Step 0) initialize all values to 0 for uninitialized nodes.
        # Initialize leaf distances to infinity
        for node in nodes_to_update:
            for key in [
                "retro_fallback_s",
                "retro_fallback_psi",
                "retro_fallback_rho",
            ]:
                if key not in node.data:
                    node.data[key] = np.zeros(self.feasibility_model.num_samples)
            node.data.setdefault("leaf_distance", math.inf)
            node.data.setdefault("root_distance", math.inf)

        # Step 1a) update "leaf_distance" (will be used to update s, psi in a good order)
        # Add optional reset to infinity (in case a cycle is formed)
        num_iter_to_reset_everything = 10 * len(graph)  # a large number which depends on the graph size
        leaf_distance_update_results = message_passing_with_resets(
            nodes=nodes_to_update,
            graph=graph,
            update_fn=leaf_distance_update,
            update_predecessors=True,
            update_successors=False,
            reset_function=reset_leaf_distance,
            num_visits_to_trigger_reset=self.num_visits_to_trigger_reset,
            reset_visit_threshold=self.reset_visit_threshold,
            num_iter_to_reset_everything=num_iter_to_reset_everything,
        )
        _log_update_result("leaf distance", leaf_distance_update_results, None)
        nodes_to_update.update(leaf_distance_update_results.nodes_updated)

        # Step 1b) update "root_distance" (will be used to update rho in a good order)
        root_distance_update_results = message_passing_with_resets(
            nodes=nodes_to_update,
            graph=graph,
            update_fn=root_distance_update,
            update_predecessors=False,
            update_successors=True,
            reset_function=reset_root_distance,
            num_visits_to_trigger_reset=self.num_visits_to_trigger_reset,
            reset_visit_threshold=self.reset_visit_threshold,
            num_iter_to_reset_everything=num_iter_to_reset_everything,
        )
        _log_update_result("root distance", root_distance_update_results, None)
        nodes_to_update.update(root_distance_update_results.nodes_updated)

        # Step 2) update "retro_fallback_s". Since it is non-decreasing we don't bother provided a reset function.
        s_update_result = message_passing_with_resets(
            nodes=nodes_to_update,
            graph=graph,
            update_fn=s_update,
            update_predecessors=True,
            update_successors=False,
            reset_function=None,  # shouldn't be required
            node_value_tracker=s_tracker,
            update_priority_fn=None,
            queue_entry_priority_fn=lambda node: node.data["leaf_distance"],
        )
        _log_update_result("s", s_update_result, s_tracker)
        nodes_to_update.update(s_update_result.nodes_updated)

        # Step 3) update "retro_fallback_psi". This may require resetting so a reset function is provided.
        # Updates are done by leaf-node distance since it is "bottom-up".
        psi_update_result = message_passing_with_resets(
            nodes=nodes_to_update,
            graph=graph,
            update_fn=psi_update,
            update_predecessors=True,
            update_successors=False,
            reset_function=reset_psi,
            node_value_tracker=psi_tracker,
            num_visits_to_trigger_reset=self.num_visits_to_trigger_reset,
            reset_visit_threshold=self.reset_visit_threshold,
            update_priority_fn=None,
            queue_entry_priority_fn=lambda node: node.data["leaf_distance"],
            num_iter_to_reset_everything=num_iter_to_reset_everything,
        )
        _log_update_result("Psi", psi_update_result, psi_tracker)
        nodes_to_update.update(psi_update_result.nodes_updated)

        # Step 4) update "retro_fallback_rho". This may require resetting so a reset function is provided.
        # Updates are done by depth since it is "top-down".
        rho_update_result = message_passing_with_resets(
            nodes=nodes_to_update,
            graph=graph,
            update_fn=rho_update,
            update_predecessors=False,
            update_successors=True,
            reset_function=reset_rho,
            node_value_tracker=rho_tracker,
            num_visits_to_trigger_reset=self.num_visits_to_trigger_reset,
            reset_visit_threshold=self.reset_visit_threshold,
            update_priority_fn=None,
            queue_entry_priority_fn=lambda node: node.data["root_distance"],
            num_iter_to_reset_everything=num_iter_to_reset_everything,
        )
        _log_update_result("Rho", rho_update_result, rho_tracker)
        nodes_to_update.update(rho_update_result.nodes_updated)

        return nodes_to_update
