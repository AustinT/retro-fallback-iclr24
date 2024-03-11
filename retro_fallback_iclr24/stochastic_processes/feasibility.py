from syntheseus import SingleProductReaction

from .base import IndependentStochasticProcess, StochasticProcess


class FeasibilityModel(StochasticProcess[SingleProductReaction]):
    """Base class for feasibility models."""

    pass


class IndependentFeasibilityModel(IndependentStochasticProcess[SingleProductReaction], FeasibilityModel):
    pass
