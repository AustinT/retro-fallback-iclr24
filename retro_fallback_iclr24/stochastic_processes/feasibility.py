from syntheseus import SingleProductReaction

from .base import IndependentStochasticProcess, StochasticProcess


class FeasibilityModel(StochasticProcess[SingleProductReaction]):
    """Base class for feasibility models."""

    pass


class IndependentFeasibilityModel(IndependentStochasticProcess[SingleProductReaction], FeasibilityModel):
    pass


class _MarginalConstantFeasibilityModel(FeasibilityModel):
    def __init__(
        self,
        constant: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0 <= constant <= 1
        self.constant = constant

    def marginal_probability(self, inputs: set[SingleProductReaction]) -> dict[SingleProductReaction, float]:
        return {rxn: self.constant for rxn in inputs}


class _MarginalRankFeasibilityModel(FeasibilityModel):
    """
    Feasibility model whose marginal probability is:

    constant / (1 + rank / rank_div)

    """

    def __init__(
        self,
        constant: float,
        rank_div: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0 <= constant <= 1
        self.constant = constant
        assert rank_div > 0
        self.rank_div = rank_div

    def marginal_probability(self, inputs: set[SingleProductReaction]) -> dict[SingleProductReaction, float]:
        return {rxn: self.constant / (1 + rxn.metadata["rank"] / self.rank_div) for rxn in inputs}


class ConstantIndependentFeasibility(_MarginalConstantFeasibilityModel, IndependentFeasibilityModel):
    pass


class RankIndependentFeasibility(_MarginalRankFeasibilityModel, IndependentFeasibilityModel):
    pass
