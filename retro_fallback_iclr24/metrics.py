from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from syntheseus.search.graph.and_or import AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.message_passing import run_message_passing

from retro_fallback_iclr24.calculate_s_psi_rho import s_update
from retro_fallback_iclr24.stochastic_processes import BuyabilityModel, FeasibilityModel

logger = logging.getLogger(__name__)


def estimate_successful_synthesis_probability_over_time(
    graph: AndOrGraph,
    feasibility_model: FeasibilityModel,
    buyability_model: BuyabilityModel,
    max_times: list[Optional[float]] = [],
) -> list[float]:
    """
    Estimates SSP over time.
    Faster than calling single version repeatedly because it only draws samples once.

    NOTE: will overwrite existing samples of f/b on the graph.

    For efficiency, an initial computation of the success outcomes of each node is done
    using an outcome of 1 for any reaction or molecule with non-zero marginal
    feasibility/purchasability. Any unsuccessful reactions in this initial computation
    are not sampled, because they will not affect the success outcomes of the root node.
    This procedure saves time in large graphs where many nodes are unsolved.
    """

    output = []

    # Get molecules in graph
    mols = {node.mol for node in graph.nodes() if isinstance(node, OrNode)}
    rxns = {node.reaction for node in graph.nodes() if isinstance(node, AndNode)}
    logger.debug(f"Found {len(mols)} molecules and {len(rxns)} reactions in graph.")

    # Decide which nodes to draw samples for by solving for a "fake" s value
    # where all reactions and molecules whose marginal probabilities are non-zero
    # are marked as always feasible or always buyable.
    # Because this is deterministic, only a single sample is needed
    # Any molecule / reaction with s=0 in this scenario will definitely have s=0
    # later on, so we don't need to sample it (this saves a lot of time in practice).
    t = time.monotonic()
    mol_to_marginal = buyability_model.marginal_probability(mols)
    rxn_to_marginal = feasibility_model.marginal_probability(rxns)
    for node in graph.nodes():
        node.data["retro_fallback_s"] = np.zeros(feasibility_model.num_samples)
        if isinstance(node, OrNode):
            node.data["retro_fallback_b"] = np.ones(1) * float(mol_to_marginal[node.mol] > 0)
        elif isinstance(node, AndNode):
            node.data["retro_fallback_f"] = np.ones(1) * float(rxn_to_marginal[node.reaction] > 0)
        else:
            raise TypeError(f"Unexpected node type {type(node)}")
    run_message_passing(
        graph=graph,
        nodes=sorted(graph.nodes(), key=lambda n: n.depth, reverse=True),
        update_fns=[s_update],
        update_predecessors=True,
        update_successors=False,
    )
    mols_to_sample = {
        node.mol for node in graph.nodes() if isinstance(node, OrNode) and node.data["retro_fallback_s"].mean() > 0
    }
    rxns_to_sample = {
        node.reaction
        for node in graph.nodes()
        if isinstance(node, AndNode) and node.data["retro_fallback_s"].mean() > 0
    }
    logger.debug(
        f"Found {len(mols_to_sample)} mols and {len(rxns_to_sample)} reactions to sample "
        f"in {time.monotonic()-t:.2f} s."
    )

    # Get samples from models
    t = time.monotonic()
    rxn_samples = feasibility_model.prior_sample(rxns_to_sample)
    t2 = time.monotonic()
    logger.debug(f"Sampled {len(rxn_samples)} reactions in {t2-t:.2f} s.")
    mol_samples = buyability_model.prior_sample(mols_to_sample)
    t3 = time.monotonic()
    logger.debug(f"Sampled {len(mols_to_sample)} molecules in {t3-t2:.2f} s.")

    # Assign samples to nodes
    for i_t, max_time in enumerate(max_times):
        logger.debug(f"Start {i_t}th estimate of synthesis success at time {max_time}.")
        t_i = time.monotonic()
        for node in graph.nodes():
            # Initialize success outcomes
            node.data["retro_fallback_s"] = np.zeros(feasibility_model.num_samples)

            # Set samples
            in_time = (max_time is None) or (node.data["analysis_time"] <= max_time)
            if isinstance(node, OrNode):
                if node.mol in mols_to_sample and in_time:
                    samples = mol_samples[node.mol]
                else:
                    samples = np.zeros(feasibility_model.num_samples)
                node.data["retro_fallback_b"] = samples
            elif isinstance(node, AndNode):
                if node.reaction in rxns_to_sample and in_time:
                    samples = rxn_samples[node.reaction]
                else:
                    samples = np.zeros(feasibility_model.num_samples)
                node.data["retro_fallback_f"] = samples
            else:
                raise TypeError(f"Unexpected node type {type(node)}")

        t_i2 = time.monotonic()
        logger.debug(f"Assigned samples in {t_i2-t_i:.2f} s.")

        # Run message passing from retro-fallback
        run_message_passing(
            graph=graph,
            nodes=sorted(graph.nodes(), key=lambda n: n.depth, reverse=True),
            update_fns=[s_update],
            update_predecessors=True,
            update_successors=False,
        )
        t_i3 = time.monotonic()
        logger.debug(f"Calculated success outcomes in {t_i3-t_i2:.2f} s.")

        # Store mean success rate of root node as output
        output.append(float(graph.root_node.data["retro_fallback_s"].mean()))
        logger.debug(f"Estimated success rate: {output[-1]:.2f} in {time.monotonic() - t_i:.2f} s.")

    return output


def estimate_successful_synthesis_probability(
    graph: AndOrGraph,
    feasibility_model: FeasibilityModel,
    buyability_model: BuyabilityModel,
    max_time: Optional[float] = None,
) -> float:
    """Estimate SSP at a single time point."""
    return estimate_successful_synthesis_probability_over_time(
        graph=graph,
        feasibility_model=feasibility_model,
        buyability_model=buyability_model,
        max_times=[max_time],
    )[0]
