from __future__ import annotations

import logging

import numpy as np
from syntheseus.search.graph.molset import MolSetNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator

from retro_fallback_iclr24.iclr24_experiments.sascore_heuristic import mol_to_sa_succ_prob
from retro_fallback_iclr24.stochastic_processes import BuyabilityModel, FeasibilityModel

logger = logging.getLogger(__name__)


class _MCTS_SuccessEvaluator(NoCacheNodeEvaluator[MolSetNode]):
    """Base class with common methods to evaluate success probability."""

    def __init__(
        self,
        feasibility_model: FeasibilityModel,
        buyability_model: BuyabilityModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feasibility_model = feasibility_model
        self.buyability_model = buyability_model

    def get_reactions_and_mols(self, node, graph):
        """Get all reactions and molecules in the route to this node."""
        mols_need_to_purchase = set(node.mols)
        reactions = set()
        curr_node = node
        while curr_node != graph.root_node:
            parents = list(graph.predecessors(curr_node))
            assert len(parents) == 1, "Not a tree?"
            parent = parents[0]
            reactions.add(graph._graph.edges[parent, curr_node]["reaction"])
            curr_node = parent
        return mols_need_to_purchase, reactions

    def get_feas_and_pur_samples(self, node, graph):
        mols, rxns = self.get_reactions_and_mols(node, graph)
        purchase_samples = self.buyability_model.prior_sample(mols)
        feas_samples = self.feasibility_model.prior_sample(rxns)
        return purchase_samples, feas_samples


class MCTS_SuccessReward(_MCTS_SuccessEvaluator):
    """
    Reward is the success probability of a given route, estimated with samples.

    This reward will be only given `max_visit` number of times.
    After that, a reward of 0 is given to encourage exploration.
    """

    def __init__(self, *args, max_visit: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_visit = max_visit

    def _evaluate_nodes(self, nodes: list[MolSetNode], graph=None) -> list[float]:
        assert graph is not None
        output = []
        for node in nodes:
            if node.num_visit < self.max_visit:
                purchase_samples, feas_samples = self.get_feas_and_pur_samples(node, graph)
                all_outcomes = np.stack(list(purchase_samples.values()) + list(feas_samples.values()))
                output.append(float(np.average(np.min(all_outcomes, axis=0))))
            else:
                output.append(0.0)
                if node.num_visit == self.max_visit:
                    logger.debug(f"Node with id {id(node)} hit max visit count, not evaluating further.")

        return output


class MCTS_Optimistic_ValueFunction(_MCTS_SuccessEvaluator):
    """Value function is just feasibility of reactions in route. Mols assumed to be purchasable."""

    def _evaluate_nodes(self, nodes: list[MolSetNode], graph=None) -> list[float]:
        assert graph is not None
        output = []
        for node in nodes:
            _, feas_samples = self.get_feas_and_pur_samples(node, graph)
            if len(feas_samples) == 0:
                output.append(1.0)
            else:
                all_outcomes = np.stack(list(feas_samples.values()))
                output.append(float(np.average(np.min(all_outcomes, axis=0))))

        return output


class MCTS_SAscore_ValueFunction(_MCTS_SuccessEvaluator):
    """Like optimistic value function, but failing molecule samples are assigned SA score."""

    def _evaluate_nodes(self, nodes: list[MolSetNode], graph=None) -> list[float]:
        assert graph is not None
        output = []
        for node in nodes:
            purchase_samples, feas_samples = self.get_feas_and_pur_samples(node, graph)
            heuristic_samples = {
                mol: np.maximum(samp, mol_to_sa_succ_prob(mol)) for mol, samp in purchase_samples.items()
            }
            all_outcomes = np.stack(list(feas_samples.values()) + list(heuristic_samples.values()))
            output.append(float(np.average(np.prod(all_outcomes, axis=0))))

        return output


class MCTS_Marginal_Feasibility_Policy(NoCacheNodeEvaluator[MolSetNode]):
    """Policy outputs are just marginal feasibilities."""

    def __init__(
        self,
        feasibility_model: FeasibilityModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feasibility_model = feasibility_model

    def _evaluate_nodes(self, nodes: list[MolSetNode], graph=None) -> list[float]:
        rxns = []
        for node in nodes:
            parent = list(graph.predecessors(node))[0]
            rxns.append(graph._graph.edges[parent, node]["reaction"])
        marg_feas = self.feasibility_model.marginal_probability(set(rxns))
        return [marg_feas[rxn] for rxn in rxns]
