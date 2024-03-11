from __future__ import annotations

import numpy as np
import pytest
from syntheseus import Bag, Molecule, SingleProductReaction

from retro_fallback_iclr24.iclr24_experiments import latent_gp_feasibility

# 4 reactions with known covariance matrix (from jupyter notebook experimentation)
RXN_SMILES = [
    "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.CCCN(CCc1cccc(I)c1)C(=O)C1OC(C(=O)OC(C)(C)C)=CC(N=[N+]=[N-])C1NC(C)=O>>CCCN(CCc1cccc(I)c1)C(=O)C1OC(C(=O)OC(C)(C)C)=CC(NC(=O)OC(C)(C)C)C1NC(C)=O",
    "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.CCCC[Sn](CCCC)(CCCC)c1cccc(CCN(CCC)C(=O)C2OC(C(=O)OC(C)(C)C)=CC(N=[N+]=[N-])C2NC(C)=O)c1>>CCCC[Sn](CCCC)(CCCC)c1cccc(CCN(CCC)C(=O)C2OC(C(=O)OC(C)(C)C)=CC(NC(=O)OC(C)(C)C)C2NC(C)=O)c1",
    "CCCN(CCc1cccc(B(O)O)c1)C(=O)C1OC(C(=O)O)=CC(NC(=O)OCC[Si](C)(C)C)C1NC(C)=O.C[Si](C)(C)C=[N+]=[N-]>>CCCN(CCc1cccc(B(O)O)c1)C(=O)C1OC(C(=O)OC)=CC(NC(=O)OCC[Si](C)(C)C)C1NC(C)=O",
    "CCCN(CCc1cccc(B(O)O)c1)C(=O)C1OC(C(=O)O)=CC(N)C1N.C[Si](C)(C)C=[N+]=[N-]>>CCCN(CCc1cccc(B(O)O)c1)C(=O)C1OC(C(=O)OC)=CC(N)C1N",
]

RXN_KERNEL_VALUES = np.array(
    [
        [1.0, 0.74358974, 0.10956522, 0.08920188],
        [0.74358974, 1.0, 0.09489051, 0.07279693],
        [0.10956522, 0.09489051, 1.0, 0.61702128],
        [0.08920188, 0.07279693, 0.61702128, 1.0],
    ]
)

# NOTE: this might not be exact...
CONST_MODEL_EXPECTED_COVAR = np.array(
    [
        [0.24012376, 0.13343074, 0.02057991, 0.01634168],
        [0.13343074, 0.24103706, 0.01737938, 0.01336186],
        [0.02057991, 0.01737938, 0.2408656, 0.10223065],
        [0.01634168, 0.01336186, 0.10223065, 0.24000399],
    ]
)

RANK_MODEL_MEANS = np.array([0.6, 0.3, 0.2, 0.15])
# NOTE: value below also might not be exact...
RANK_MODEL_EXPECTED_COVAR = np.array(
    [
        [0.24034147, 0.10247617, 0.01396316, 0.00892761],
        [0.10247617, 0.21148746, 0.01062699, 0.00577004],
        [0.01396316, 0.01062699, 0.16067486, 0.05409579],
        [0.00892761, 0.00577004, 0.05409579, 0.1304354],
    ]
)


@pytest.fixture
def reactions():
    return [
        SingleProductReaction(
            reactants=Bag([Molecule(smi) for smi in rxn_smiles.split(">>")[0].split(".")]),
            product=Molecule(rxn_smiles.split(">>")[1]),
            metadata={"rank": i},
        )
        for i, rxn_smiles in enumerate(RXN_SMILES)
    ]


@pytest.fixture
def constant_latent_gp_model():
    return latent_gp_feasibility.Constant_Mean_Latent_GP_Feasibility(num_samples=10_000, constant=0.6)


@pytest.fixture
def rank_latent_gp_model():
    return latent_gp_feasibility.Rank_Mean_Latent_GP_Feasibility(num_samples=10_000, constant=0.6, rank_div=1)


class TestConstantMeanLatentGPModel:
    def test_marginal_probability(self, reactions, constant_latent_gp_model):
        marginal_probabilities = constant_latent_gp_model.marginal_probability(set(reactions))
        assert len(marginal_probabilities) == 4
        assert np.allclose([marginal_probabilities[rxn] for rxn in reactions], 0.6)

    def test_unconditional_latent_distribution(self, reactions, constant_latent_gp_model):
        constant_latent_gp_model._add_fingerprint_metadata_to_reactions(reactions)
        mu, cov = constant_latent_gp_model._get_latent_distribution(reactions, {})
        assert np.allclose(mu, 0.253, atol=0.001)  # this is the inverse CDF of 0.6
        assert np.allclose(cov, RXN_KERNEL_VALUES)

    def test_conditional_latent_distribution1(self, reactions, constant_latent_gp_model):
        constant_latent_gp_model._add_fingerprint_metadata_to_reactions(reactions)

        # We observe z(R1) = 0.5. Fill this in.
        reactions[0].metadata["LATENT_GP_SAMPLES"] = np.full(10_000, 0.5)

        # Get conditional distribution
        mu, cov = constant_latent_gp_model._get_latent_distribution(reactions[1:], {reactions[0]: np.ones(10_000)})

        # Mean matches expectation (note: not hand-verified, but should be fine)
        assert np.allclose(
            mu,
            np.stack([np.array([0.4367553, 0.28037163, 0.27534896]) for _ in range(10_000)]),
        )

        # Cov matches expectation (only partially verified by hand)
        assert np.allclose(
            cov,
            np.array(
                [
                    [0.4470754, 0.0134191, 0.00646747],
                    [0.0134191, 0.98799549, 0.60724787],
                    [0.00646747, 0.60724787, 0.99204304],
                ]
            ),
        )

    def _test_sample_mean_var(self, means, cov):
        # Assert means are all close to 0.6
        assert np.all(means >= 0.58) and np.all(means <= 0.62)

        # Assert covariance between samples is what we expect
        assert np.allclose(cov, CONST_MODEL_EXPECTED_COVAR, atol=0.01)

    def test_prior_sample(self, reactions, constant_latent_gp_model):
        samples = constant_latent_gp_model.prior_sample(reactions)
        sample_arr = np.stack([samples[rxn] for rxn in reactions])
        assert sample_arr.shape == (4, 10_000)
        means = np.mean(sample_arr, axis=1)
        cov = np.cov(sample_arr)

        self._test_sample_mean_var(means, cov)

    def test_posterior_sample(self, reactions, constant_latent_gp_model):
        """
        Test that sampling some reactions, then sampling from posterior gives same stats as directly sampling from prior.
        """

        # Do the sampling in 2 stages
        prior_samples = constant_latent_gp_model.prior_sample(reactions[:2])
        posterior_samples = constant_latent_gp_model.posterior_sample(reactions[2:], prior_samples)

        # Combine samples together
        all_samples = dict(prior_samples)
        all_samples.update(posterior_samples)
        sample_arr = np.stack([all_samples[rxn] for rxn in reactions])
        means = np.mean(sample_arr, axis=1)
        cov = np.cov(sample_arr)
        self._test_sample_mean_var(means, cov)

    def test_check_fails(self, reactions, constant_latent_gp_model):
        """Check that the sampling fails if observed samples don't have correct metadata."""
        with pytest.raises(AssertionError):
            constant_latent_gp_model.posterior_sample(reactions[2:], {rxn: np.ones(10_000) for rxn in reactions[:2]})

    def test_caching(self, reactions, constant_latent_gp_model):
        """
        Test that caching works as expected.
        We run it without the cache, which by default computes the cache and checks it.
        """

        # Prior sample on first reaction
        samples = constant_latent_gp_model.prior_sample(reactions[:1])

        # Posterior sample on second reaction. This should form a cache.
        samples.update(constant_latent_gp_model.posterior_sample(reactions[1:2], samples))

        # Posterior sample on third reaction. This should use the cache.
        constant_latent_gp_model.posterior_sample(reactions[2:], samples)


class TestRankMeanLatentGPModel:
    def test_marginals(self, reactions, rank_latent_gp_model):
        marginal_probabilities = rank_latent_gp_model.marginal_probability(reactions)
        assert np.allclose([marginal_probabilities[rxn] for rxn in reactions], RANK_MODEL_MEANS)

    def test_prior_sample(self, reactions, rank_latent_gp_model):
        samples = rank_latent_gp_model.prior_sample(reactions)
        sample_arr = np.stack([samples[rxn] for rxn in reactions])
        assert sample_arr.shape == (4, 10_000)

        # Means should match up
        means = np.mean(sample_arr, axis=1)
        assert np.all(np.abs(means - RANK_MODEL_MEANS) < 0.02)

        # Covariance should match reasonable value
        cov = np.cov(sample_arr)
        assert np.allclose(cov, RANK_MODEL_EXPECTED_COVAR, atol=0.02)
