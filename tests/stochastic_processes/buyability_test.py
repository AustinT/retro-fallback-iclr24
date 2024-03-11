import numpy as np
from syntheseus import Molecule

from retro_fallback_iclr24.stochastic_processes import BinaryBuyability


def test_binary_buyability():
    """Run a bunch of tests for binary buyability model."""

    # Create a binary buyability model
    NUM_SAMPLES = 100
    model = BinaryBuyability(num_samples=NUM_SAMPLES, seed=0)

    # Create some molecules
    C = Molecule("C", metadata={"is_purchasable": False})
    CC = Molecule("CC")  # NOTE: no explicit purchasability label
    CCC = Molecule("CCC", metadata={"is_purchasable": True})
    mols = {C, CC, CCC}

    # Sample from prior and check that the results are as expected
    prior_samples = model.prior_sample(mols)
    for mol in [C, CC]:
        assert np.allclose(prior_samples[mol], np.zeros(NUM_SAMPLES))
    assert np.allclose(prior_samples[CCC], np.ones(NUM_SAMPLES))

    # Check that the marginal probabilities are correct
    marginal_probs = model.marginal_probability(mols)
    assert marginal_probs[Molecule("C")] == 0.0
    assert marginal_probs[Molecule("CC")] == 0.0
    assert marginal_probs[Molecule("CCC")] == 1.0
    assert len(marginal_probs) == 3
