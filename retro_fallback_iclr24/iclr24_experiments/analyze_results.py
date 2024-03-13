from __future__ import annotations

import collections
import logging
import math
import time
from typing import Any, Optional

import numpy as np
from syntheseus.search.algorithms.best_first.retro_star import (
    reaction_number_update,
)
from syntheseus.search.algorithms.mcts.molset import MolSetGraph
from syntheseus.search.analysis.route_extraction import iter_routes_time_order
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import run_message_passing

from retro_fallback_iclr24.metrics import (
    estimate_successful_synthesis_probability_over_time,
    iter_routes_feasibility_order,
)
from retro_fallback_iclr24.stochastic_processes import BuyabilityModel, FeasibilityModel

logger = logging.getLogger(__name__)


def get_nodes_in_route(graph) -> set:
    """
    Extract all nodes which are part of a solved synthesis route.
    In all cases this can be done by descending from the root node.
    """

    queue = collections.deque([graph.root_node])  # only nodes whose parents are in a route
    output = set()
    while len(queue) > 0:
        node = queue.popleft()
        if node in output:
            continue
        if node.has_solution:
            output.add(node)
            for child in graph.successors(node):
                queue.append(child)
    return output


def analyze_output_graph(
    output_graph: AndOrGraph,
    feasibility_model: FeasibilityModel,
    buyability_model: BuyabilityModel,
    analysis_times: list[float],
    calculate_most_feasible_route: bool = True,
    smiles_to_start_mols: Optional[dict[str, list[set[str]]]] = None,
) -> dict[str, Any]:
    """Get various analysis results for the output graph."""
    logger.debug("Starting analysis.")
    t0 = time.monotonic()

    # Insert analysis times (here always equal to num calls to rxn model)
    for node in output_graph.nodes():
        node.data["analysis_time"] = node.data["num_calls_rxn_model"]

    # Metric 1: solution time
    soln_time = get_first_solution_time(output_graph)
    t1 = time.monotonic()
    logger.debug(f"Solution time calculated in {t1 - t0:.2f} s.")

    # Metric 2: success probability over time
    success_probabilities_over_time = estimate_successful_synthesis_probability_over_time(
        output_graph,
        feasibility_model,
        buyability_model,
        max_times=analysis_times,  # type: ignore[arg-type]  # complains list[float] != list[Optional[float]]
    )
    t2 = time.monotonic()
    logger.debug(f"Success probabilities calculated in {t2 - t1:.2f} s.")

    # Metric 3 (currently deleted): number of routes over time (capped at M).
    # Finding the first M routes is enough to calculate this.
    # routes_time_order = list(
    #     iter_routes_time_order(output_graph, max_routes=max_num_routes)
    # )
    # routes_time_order = []
    # route_times = [max(n.data["analysis_time"] for n in r) for r in routes_time_order]
    # num_routes_over_time = [sum(t <= at for t in route_times) for at in analysis_times]
    # del routes_time_order, route_times
    # t3 = time.monotonic()
    # main_logger.debug(f"Number of routes calculated in {t3 - t2:.2f} s.")

    # A bunch of time-dependent metrics which cannot be computed simultaneously for different times
    num_nodes_over_time: list[int] = []
    search_duration_over_time: list[float] = []
    shortest_route_over_time: list[float] = []
    most_feasible_route_over_time: list[float] = []
    _most_feasible_route_mols = None  # Will use later
    for at in analysis_times:
        t0 = time.monotonic()

        # Two collections of nodes important for analysis
        nodes_before_at: list[ANDOR_NODE] = [n for n in output_graph.nodes() if n.data["analysis_time"] <= at]
        nodes_at_or_after_at: list[ANDOR_NODE] = [n for n in output_graph.nodes() if n.data["analysis_time"] >= at]

        # Metric 4: graph size over time
        num_nodes_over_time.append(len(nodes_before_at))

        # Metric 5: wallclock time taken to reach this number of nodes.
        # Because search can terminate early, it is not clear if this number of nodes would even be reached.
        # Therefore, we take the minimum time of any node at or after this time.
        if len(nodes_at_or_after_at) > 0:
            search_duration_over_time.append(
                (
                    min(n.creation_time for n in nodes_at_or_after_at) - output_graph.root_node.creation_time
                ).total_seconds()
            )
        else:
            search_duration_over_time.append(math.nan)
        t5 = time.monotonic()
        logger.debug(f"\tt={at}: Graph size/time calculated in {t5 - t0:.2f} s.")

        # Metric 6: shortest route over time.
        # Re-use retro* updates to calculate this.
        if soln_time <= at:
            # Set reaction/mol costs appropriately
            for node in output_graph.nodes():
                node.data["retro_star_reaction_number"] = math.inf  # initialize to overestimate
                node.data["reaction_number_estimate"] = math.inf  # Effectively no estimate
                if node.data["analysis_time"] > at:
                    node.data["retro_star_rxn_cost"] = math.inf
                    node.data["retro_star_mol_cost"] = math.inf
                else:
                    if isinstance(node, OrNode):
                        node.data["retro_star_mol_cost"] = 0.0 if node.mol.metadata["is_purchasable"] else math.inf
                    elif isinstance(node, AndNode):
                        node.data["retro_star_rxn_cost"] = 1.0
                    else:
                        raise ValueError(f"Invalid node type: {type(node)}")

            # Run updates
            run_message_passing(
                output_graph,
                nodes=output_graph.nodes(),
                update_fns=[reaction_number_update],
                update_predecessors=True,
                update_successors=False,
            )
            shortest_route_over_time.append(output_graph.root_node.data["retro_star_reaction_number"])

        else:
            shortest_route_over_time.append(math.inf)
        t6 = time.monotonic()
        logger.debug(f"\tt={at}: Shortest route calculated in {t6 - t5:.2f} s.")

        # Metric 7: most feasible route over time
        if not calculate_most_feasible_route:
            most_feasible_route_over_time.append(math.nan)
        elif soln_time <= at:
            # Set feasibilities for this time
            for node in output_graph.nodes():
                # NOTE: this is re-using node attributes from earlier success probability calculation
                if isinstance(node, OrNode):
                    node.data["retro_fallback_f_or_b"] = node.data["retro_fallback_b"]
                else:
                    node.data["retro_fallback_f_or_b"] = node.data["retro_fallback_f"]

                # Zero samples if analysis time exceeded
                if node.data["analysis_time"] > at:
                    node.data["retro_fallback_f_or_b"] = np.zeros_like(node.data["retro_fallback_f_or_b"])

            # Extract best solution
            most_feasible_solutions = list(iter_routes_feasibility_order(output_graph, max_routes=1))
            assert len(most_feasible_solutions) == 1
            most_feasible_route_over_time.append(math.exp(-most_feasible_solutions[0][0]))

            # Store the starting molecules (might be used later)
            if len(most_feasible_solutions[0][1]) == 1:
                _most_feasible_route_mols = {list(most_feasible_solutions[0][1])[0].mol}  # just the root node
            else:
                _most_feasible_route_mols = set(
                    output_graph.to_synthesis_graph(nodes=most_feasible_solutions[0][1]).get_starting_molecules()
                )
            del most_feasible_solutions
        else:
            # There are no solutions, just directly output 0
            most_feasible_route_over_time.append(0.0)
        t7 = time.monotonic()
        logger.debug(f"\tt={at}: Most feasible route calculated in {t7 - t6:.2f} s.")

        del at

    # (if provided): check if most feasible route matches ground truth starting molecules.
    # This metric was used in FusionRetro
    if smiles_to_start_mols is not None and output_graph.root_node.mol.smiles in smiles_to_start_mols:
        logger.debug("Performing starting molecule match check.")

        # Optionally get most feasible route
        if calculate_most_feasible_route:
            start_mol_match_most_feasible_route = False
        else:
            start_mol_match_most_feasible_route = None

        # Go through all start mol sets and check for matching (also for the most feasible route)
        for start_mol_smiles in smiles_to_start_mols[output_graph.root_node.mol.smiles]:
            start_mol_set = set(Molecule(s) for s in start_mol_smiles)

            if calculate_most_feasible_route:
                start_mol_match_most_feasible_route = start_mol_match_most_feasible_route or (
                    _most_feasible_route_mols == start_mol_set
                )

        logger.debug(f"Most feasible route matches start mols: result={start_mol_match_most_feasible_route}")
    else:
        logger.debug("Molecule match check skipped: no ground truth start mols.")
        start_mol_match_most_feasible_route = None

    return dict(
        solution_time=soln_time,
        success_probabilities=success_probabilities_over_time,
        # num_routes_over_time=num_routes_over_time,
        # cap_num_routes=max_num_routes,  # so it is clear what the max number of routes is
        num_nodes_over_time=num_nodes_over_time,
        search_duration_over_time=search_duration_over_time,
        shortest_route_over_time=shortest_route_over_time,
        most_feasible_route_over_time=most_feasible_route_over_time,
        num_nodes_part_of_a_route=len(get_nodes_in_route(output_graph)),
        start_mol_match_most_feasible_route=start_mol_match_most_feasible_route,
    )


def analyze_output_graph_molset(
    output_graph: MolSetGraph,
    feasibility_model: FeasibilityModel,
    buyability_model: BuyabilityModel,
    analysis_times: list[float],
    max_num_routes: int = 1_000_000,
) -> dict[str, Any]:
    """Alternative analysis procedure for MolSetGraphs. Works by extracting all routes."""

    logger.debug("Starting [MolSetGraph] analysis.")
    t0 = time.monotonic()

    # Insert analysis times (here always equal to num calls to rxn model)
    for node in output_graph.nodes():
        node.data["analysis_time"] = node.data["num_calls_rxn_model"]

    # Metric 1: solution time
    soln_time = get_first_solution_time(output_graph)
    t1 = time.monotonic()
    logger.debug(f"Solution time calculated in {t1 - t0:.2f} s.")

    # Remaining metrics require routes: extract them all
    routes = list(iter_routes_time_order(output_graph, max_routes=max_num_routes))
    logger.debug(f"Extracted {len(routes)} routes.")
    assert len(routes) < max_num_routes, "Some routes likely not extracted."

    # Figure out which reactions and starting molecules are in each route: this will be used to draw samples
    reactions_in_routes = [set(output_graph.to_synthesis_graph(r).nodes()) for r in routes]
    starting_mols_in_routes = [set(max(r, key=lambda n: n.depth).mols) for r in routes]

    # Draw the samples
    all_mols = set(m for li in starting_mols_in_routes for m in li)
    all_rxns = set(r for li in reactions_in_routes for r in li)
    logger.debug(f"Drawing samples for {len(all_mols)} mols and {len(all_rxns)} rxns.")
    rxn_samples = feasibility_model.prior_sample(all_rxns)
    mol_samples = buyability_model.prior_sample(all_mols)
    logger.debug("Done drawing samples.")

    # Assign a time to each route
    route_times = [max(n.data["analysis_time"] for n in r) for r in routes]

    # Compute success outcomes for each route
    success_outcomes = []
    for i, _ in enumerate(routes):
        arr = np.stack(
            [rxn_samples[r] for r in reactions_in_routes[i]] + [mol_samples[m] for m in starting_mols_in_routes[i]]
        )
        success_outcomes.append(np.min(arr, axis=0))
    success_outcomes = np.asarray(success_outcomes)

    # Compute various metrics over time
    success_probabilities_over_time: list[float] = []
    num_nodes_over_time: list[int] = []
    search_duration_over_time: list[float] = []
    shortest_route_over_time: list[float] = []
    most_feasible_route_over_time: list[float] = []
    for t in analysis_times:
        # Two collections of nodes important for analysis
        nodes_before_at = [n for n in output_graph.nodes() if n.data["analysis_time"] <= t]
        nodes_at_or_after_at = [n for n in output_graph.nodes() if n.data["analysis_time"] >= t]

        # Nodes over time
        num_nodes_over_time.append(len(nodes_before_at))
        if len(nodes_at_or_after_at) > 0:
            search_duration_over_time.append(
                (
                    min(n.creation_time for n in nodes_at_or_after_at) - output_graph.root_node.creation_time
                ).total_seconds()
            )
        else:
            search_duration_over_time.append(math.nan)

        # Various metrics
        if any(rt <= t for rt in route_times):
            success_probabilities_over_time.append(
                float(success_outcomes[np.asarray(route_times) <= t].max(axis=0).mean())
            )
            shortest_route_over_time.append(min([len(r) - 1 for r, rt in zip(routes, route_times) if rt <= t]))
            most_feasible_route_over_time.append(
                float(success_outcomes[np.asarray(route_times) <= t].mean(axis=1).max())
            )
        else:
            success_probabilities_over_time.append(0.0)
            shortest_route_over_time.append(math.inf)
            most_feasible_route_over_time.append(0.0)

    return dict(
        solution_time=soln_time,
        success_probabilities=success_probabilities_over_time,
        final_num_routes=len(routes),
        num_nodes_over_time=num_nodes_over_time,
        search_duration_over_time=search_duration_over_time,
        shortest_route_over_time=shortest_route_over_time,
        most_feasible_route_over_time=most_feasible_route_over_time,
        num_nodes_part_of_a_route=len(get_nodes_in_route(output_graph)),
    )
