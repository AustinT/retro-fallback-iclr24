from syntheseus import Molecule

from .base import IndependentStochasticProcess, StochasticProcess


class BuyabilityModel(StochasticProcess[Molecule]):
    r"""Base class for buyability models (\xi_b)."""

    pass


class IndependentBuyabilityModel(IndependentStochasticProcess[Molecule], BuyabilityModel):
    pass


class BinaryBuyability(IndependentBuyabilityModel):
    """Assigns buyability of 1.0 if `is_purchasable` else 0.0, deterministically and independently."""

    def marginal_probability(self, inputs: set[Molecule]) -> dict[Molecule, float]:
        return {mol: float(mol.metadata.get("is_purchasable", 0)) for mol in inputs}
