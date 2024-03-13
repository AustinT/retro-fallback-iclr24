from __future__ import annotations

import math

from syntheseus.search.graph.and_or import AndNode, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator

from retro_fallback_iclr24.stochastic_processes import BuyabilityModel, FeasibilityModel


def _neg_log_or_inf(p: float) -> float:
    if p > 0:
        return -math.log(p)
    else:
        return math.inf


class RetroStarMarginalFeasibilityCost(NoCacheNodeEvaluator[AndNode]):
    """Sets the cost of a reaction to be -log(marginal feasibility)."""

    def __init__(
        self,
        feasibility_model: FeasibilityModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feasibility_model = feasibility_model

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        feas_dict = self.feasibility_model.marginal_probability({n.reaction for n in nodes})
        return [_neg_log_or_inf(feas_dict[node.reaction]) for node in nodes]


class RetroStarMarginalBuyabilityCost(NoCacheNodeEvaluator[OrNode]):
    """Sets the cost of a molecule to be -log(marginal buyability)."""

    def __init__(
        self,
        buyability_model: BuyabilityModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.buyability_model = buyability_model

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        buy_dict = self.buyability_model.marginal_probability({n.mol for n in nodes})
        return [_neg_log_or_inf(buy_dict[node.mol]) for node in nodes]
