import numpy as np
import pytest
from syntheseus import Bag, Molecule, SingleProductReaction

from retro_fallback_iclr24.stochastic_processes.feasibility import (
    ConstantIndependentFeasibility,
    RankIndependentFeasibility,
)


@pytest.fixture
def reaction1():
    return SingleProductReaction(
        metadata={"rank": 1},
        product=Molecule("CCC"),
        reactants=Bag({Molecule("CC"), Molecule("C")}),
    )


@pytest.fixture
def reaction2():
    return SingleProductReaction(
        metadata={"rank": 2},
        product=Molecule("CC"),
        reactants=Bag({Molecule("C")}),
    )


def test_constant_independent_feasibility(reaction1, reaction2) -> None:
    model = ConstantIndependentFeasibility(num_samples=100_000, constant=0.5)

    # Check marginal probabilities
    rxns = {reaction1, reaction2}
    assert model.marginal_probability(rxns) == {reaction1: 0.5, reaction2: 0.5}

    # Check samples are empirically correctly distributed
    sample_dict = model.prior_sample(rxns)
    samples = np.stack(list(sample_dict.values()))
    assert np.allclose(samples.mean(axis=1), 0.5, atol=0.05)
    assert np.allclose(np.cov(samples), 0.25 * np.eye(2), atol=0.02)


def test_rank_independent_feasibility(reaction1, reaction2) -> None:
    model = RankIndependentFeasibility(num_samples=100_000, constant=0.5, rank_div=1)

    # Check marginal probabilities
    rxns = [reaction1, reaction2]
    expected_marginals = {reaction1: 0.5 / 2, reaction2: 0.5 / 3}
    assert model.marginal_probability(set(rxns)) == expected_marginals

    # Check samples are empirically correctly distributed
    sample_dict = model.prior_sample(set(rxns))
    samples = np.stack([sample_dict[r] for r in rxns])
    expected_means = np.asarray([expected_marginals[r] for r in rxns])
    assert np.allclose(samples.mean(axis=1), expected_means, atol=0.03)
    assert np.allclose(np.cov(samples), np.diag(expected_means) * np.diag(1 - expected_means), atol=0.02)
