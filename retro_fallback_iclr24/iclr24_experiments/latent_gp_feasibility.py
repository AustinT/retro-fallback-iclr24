"""Code for latent GP feasibility model."""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy import stats
from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.metrics.pairwise import manhattan_distances
from syntheseus import SingleProductReaction

from retro_fallback_iclr24.stochastic_processes.feasibility import (
    FeasibilityModel,
    _MarginalConstantFeasibilityModel,
    _MarginalRankFeasibilityModel,
)

from .fingerprints import smiles_to_fingerprint_arr

gp_feas_logger = logging.getLogger(__name__)


# Some keys which will be used to store metadata in reactions for latent GP
LATENT_GP_SAMPLES = "LATENT_GP_SAMPLES"
REACTANT_FP = "REACTANT_FP"
PRODUCT_FP = "PRODUCT_FP"


def _minmax_sim(x: np.ndarray, y: np.ndarray):
    """
    Min-max similarity between two vectors,
    defined as:

    sum_i min(x_i, y_i) / sum_i max(x_i, y_i)
    """
    # Compute required l1 norms
    x_norm = np.abs(x).sum(axis=1).reshape(-1, 1)
    y_norm = np.abs(y).sum(axis=1).reshape(1, -1)
    norm_sum = x_norm + y_norm
    pairwise_dist = manhattan_distances(x, y)

    # Calculate and return kernel
    eps = 1e-6  # to avoid division by zero
    return (norm_sum - pairwise_dist + eps) / (norm_sum + pairwise_dist + eps)


def _reaction_similarity_kernel(*, reactants1, product1, reactants2, product2):
    """
    Reaction similarity kernel, defined as:
    minmax_sim(r1|p1,r2|p2) * minmax_sim(r1-p1, r2-p2)
    where minmax_sim is the min-max similarity and | denotes concatenation.
    """

    # Concatenate both fingerprints to form "reaction fingerprints"
    fp1 = np.concatenate([reactants1, product1], axis=-1)
    fp2 = np.concatenate([reactants2, product2], axis=-1)

    # Calculate "mol similarity"
    mol_sim = _minmax_sim(fp1, fp2)

    # Calculate the differences between reactants and products
    fp_diff1 = np.abs(reactants1 - product1)
    fp_diff2 = np.abs(reactants2 - product2)

    # Calculate "reaction similarity"
    rxn_sim = _minmax_sim(fp_diff1, fp_diff2)

    # Return product of the two
    return mol_sim * rxn_sim


class _Latent_GP_FeasibilityModel(FeasibilityModel):
    """
    Feasibility model whose correlations are given by a latent GP.
    Posterior sampling is done by conditioning on the latent variables,
    rather than the observed data directly.

    Contains code to cache the covariance matrix for observed reactions
    to speed up computation (to overall O(N^3) complexity instead of O(N^4)).
    """

    def __init__(
        self,
        check_observed_samples: bool = True,
        use_cached_covar_matrix: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.check_observed_samples = check_observed_samples
        self.use_cached_covar_matrix = use_cached_covar_matrix
        self.noise_var = 1e-6

    def _check_reactions(self, samples: dict[SingleProductReaction, np.ndarray]):
        """
        Check that reactions have latent variables which match observed samples,
        and also fingerprint information.
        """
        for rxn, output in samples.items():
            assert LATENT_GP_SAMPLES in rxn.metadata
            assert np.allclose((rxn.metadata[LATENT_GP_SAMPLES] > 0).astype(float), output)

            # Check that fingerprints are there
            assert REACTANT_FP in rxn.metadata
            assert PRODUCT_FP in rxn.metadata

    def _get_latent_distribution(
        self,
        rxn_list: list[SingleProductReaction],
        observed_rxn_set: set[SingleProductReaction],
    ) -> tuple[np.ndarray, np.ndarray]:
        def _rxn_sort_key(rxn: SingleProductReaction):
            """Key used to order reactions to correctly recover cached covariance matrix."""
            return rxn.metadata.get("_latent_gp_cache_idx", -1)

        # Add labels to reactions to allow caching of covariance matrix in a stable order
        for i, rxn in enumerate(rxn_list):
            rxn.metadata["_latent_gp_cache_idx"] = i + len(observed_rxn_set)
            del i, rxn

        # Step 1: get marginal distribution of latent variable (i.e. not accounting for observed samples)
        marginal_probs = self.marginal_probability(set(rxn_list))
        mu_query_output = stats.norm.ppf(np.asarray([marginal_probs[rxn] for rxn in rxn_list]))
        assert mu_query_output.shape == (len(rxn_list),)
        fp_r_query = np.stack([rxn.metadata[REACTANT_FP] for rxn in rxn_list])
        fp_p_query = np.stack([rxn.metadata[PRODUCT_FP] for rxn in rxn_list])
        cov_query_query = _reaction_similarity_kernel(
            reactants1=fp_r_query,
            product1=fp_p_query,
            reactants2=fp_r_query,
            product2=fp_p_query,
        )
        cov_query_query_plus_noise = cov_query_query + self.noise_var * np.eye(len(cov_query_query))
        cov_query_output = cov_query_query_plus_noise.copy()
        assert cov_query_output.shape == (len(rxn_list), len(rxn_list))

        # Step 2: get conditional distribution of latent variable (i.e. accounting for observed samples)
        if len(observed_rxn_set) > 0:
            # Sort observed samples to allow cached covariance matrix to be used
            observed_rxn_list = sorted(observed_rxn_set, key=_rxn_sort_key)

            # Get stats of observed samples
            fp_r_cond = np.stack([rxn.metadata[REACTANT_FP] for rxn in observed_rxn_list])
            fp_p_cond = np.stack([rxn.metadata[PRODUCT_FP] for rxn in observed_rxn_list])
            observed_rxn_marginal_probs = self.marginal_probability(set(observed_rxn_list))
            mu_cond = stats.norm.ppf(np.asarray([observed_rxn_marginal_probs[rxn] for rxn in observed_rxn_list]))
            mu_cond = mu_cond[:, None]  # expand dims
            z_cond = np.stack([rxn.metadata[LATENT_GP_SAMPLES] for rxn in observed_rxn_list])

            # Get covariance matrix between condition and query samples
            t = time.monotonic()
            cov_cond_query = _reaction_similarity_kernel(
                reactants1=fp_r_cond,
                product1=fp_p_cond,
                reactants2=fp_r_query,
                product2=fp_p_query,
            )
            gp_feas_logger.debug(f"Cond-query covar calculated in {time.monotonic() - t} s")

            # Get cholesky factor of covariance matrix between condition samples.
            # This is either via explicit calculation or via cache.
            CHO_LOWER = True  # we compute lower triangular cholesky factor throughout
            cache_applicable = hasattr(self, "_rxns_cached") and observed_rxn_list == self._rxns_cached  # type: ignore[has-type]
            if self.use_cached_covar_matrix and cache_applicable:
                cov_cond_cond = self._covar_rxns_cached  # type: ignore[has-type]
                L_cond_cond_cho = self._cho_covar_rxns_cached  # type: ignore[has-type]
                gp_feas_logger.debug("Using cached covariance matrix & cho factor.")
            else:
                # Calculate covariance matrix between condition samples
                t = time.monotonic()
                cov_cond_cond = _reaction_similarity_kernel(
                    reactants1=fp_r_cond,
                    product1=fp_p_cond,
                    reactants2=fp_r_cond,
                    product2=fp_p_cond,
                )
                gp_feas_logger.debug(f"Cond-cond covariance calculated in {time.monotonic() - t} s")

                # Calculate cholesky factor of covariance matrix between condition samples
                t = time.monotonic()
                cov_cond_cond_plus_noise = cov_cond_cond + self.noise_var * np.eye(len(cov_cond_cond))
                L_cond_cond_cho = cholesky(cov_cond_cond_plus_noise, lower=CHO_LOWER)
                gp_feas_logger.debug(f"Cholesky factor calculated in {time.monotonic() - t} s")

                if cache_applicable:
                    # do a check to make sure the cache is correct
                    assert np.allclose(cov_cond_cond, self._covar_rxns_cached)  # type: ignore[has-type]
                    assert np.allclose(L_cond_cond_cho, self._cho_covar_rxns_cached)  # type: ignore[has-type]
                    gp_feas_logger.debug("Cached covariance matrix & cholesky factor matched calculated one.")
                else:
                    gp_feas_logger.debug("Cached covariance matrix not applicable.")

            # Conditional mean
            t = time.monotonic()
            mu_query_output = (  # NOTE: important cho_solve is done in this order to avoid N^2 solve op on many samples
                mu_query_output + (cho_solve((L_cond_cond_cho, CHO_LOWER), cov_cond_query).T @ (z_cond - mu_cond)).T
            )  # shape (num samples, num reactions)

            # Conditional covariance
            cov_query_output = cov_query_output - cov_cond_query.T @ cho_solve(
                (L_cond_cond_cho, CHO_LOWER), cov_cond_query
            )
            gp_feas_logger.debug(f"Conditional mean and covar calculated in {time.monotonic() - t} s")

            # For next iteration, prepare a cached version of full covariance matrix (i.e. query and condition points)
            # and its cholesky factor.
            # Useful if the next call conditions on the observed samples + new samples, which is the most likely scenario in retro-fallback.
            self._rxns_cached = observed_rxn_list + rxn_list
            self._covar_rxns_cached = np.concatenate(
                [
                    np.concatenate([cov_cond_cond, cov_cond_query], axis=1),
                    np.concatenate([cov_cond_query.T, cov_query_query], axis=1),
                ]
            )

            # New Cholesky factor calculation
            t = time.monotonic()
            new_cho_off_diag_block = solve_triangular(
                L_cond_cond_cho,
                cov_cond_query,
                lower=CHO_LOWER,
            )  # shape (num cond samples, num query samples)
            new_cho_diag_block = cholesky(  # NOTE: *includes* noise term
                cov_query_query_plus_noise - new_cho_off_diag_block.T @ new_cho_off_diag_block,
                lower=CHO_LOWER,
            )
            self._cho_covar_rxns_cached = np.zeros(self._covar_rxns_cached.shape)
            self._cho_covar_rxns_cached[: len(observed_rxn_list), : len(observed_rxn_list)] = L_cond_cond_cho
            self._cho_covar_rxns_cached[len(observed_rxn_list) :, : len(observed_rxn_list)] = new_cho_off_diag_block.T
            self._cho_covar_rxns_cached[len(observed_rxn_list) :, len(observed_rxn_list) :] = new_cho_diag_block
            gp_feas_logger.debug(f"Cholesky cache calculated in {time.monotonic() - t} s")

        return mu_query_output, cov_query_output

    def _add_fingerprint_metadata_to_reactions(self, reactions: list[SingleProductReaction]) -> None:
        for rxn in reactions:
            if REACTANT_FP not in rxn.metadata:
                rxn.metadata[REACTANT_FP] = smiles_to_fingerprint_arr([rxn.reaction_smiles.split(">>")[0]])[0]
            if PRODUCT_FP not in rxn.metadata:
                rxn.metadata[PRODUCT_FP] = smiles_to_fingerprint_arr([rxn.product.smiles])[0]

    def posterior_sample(
        self,
        reactions: set[SingleProductReaction],
        observed_samples: dict[SingleProductReaction, np.ndarray],
    ) -> dict[SingleProductReaction, np.ndarray]:
        gp_feas_logger.debug(
            f"Drawing posterior sample for {len(reactions)} reactions conditioned on {len(observed_samples)} observed samples."
        )
        t_start = time.monotonic()
        # Transform reactions into list (we need to assign an arbitrary order to them)
        rxn_list = list(reactions)
        del reactions

        # Potentially exit early if no reactions to sample
        if len(rxn_list) == 0:
            return dict()

        # Step 1: potentially check samples
        if self.check_observed_samples:
            self._check_reactions(observed_samples)

        # Step 2: get fingerprints for new reactions
        t = time.monotonic()
        self._add_fingerprint_metadata_to_reactions(rxn_list)
        gp_feas_logger.debug(f"Added fingerprint metadata in {time.monotonic() - t} s")

        # Step 3: get latent variable distribution
        t = time.monotonic()
        mu_query, cov_query = self._get_latent_distribution(rxn_list, set(observed_samples.keys()))
        gp_feas_logger.debug(f"Posterior latent variable distribution calculated in {time.monotonic() - t} s")

        # Step 4: sample from conditional distribution
        # (sample just the noise, whose distribution does not depend on observed samples, then add to conditional mean)
        t = time.monotonic()
        eps_query = self.generator.multivariate_normal(
            np.zeros(cov_query.shape[0]), cov_query, self.num_samples, method="cholesky"
        )
        z_query = mu_query + eps_query
        assert z_query.shape == (self.num_samples, len(rxn_list))
        for i, rxn in enumerate(rxn_list):
            rxn.metadata[LATENT_GP_SAMPLES] = z_query[:, i]
        gp_feas_logger.debug(f"Latent variable samples drawn in {time.monotonic() - t} s")

        # Step 5: transform samples into binary outcomes and return
        output = {rxn: (rxn.metadata[LATENT_GP_SAMPLES] > 0).astype(float) for rxn in rxn_list}
        gp_feas_logger.debug(f"Overall posterior sampling took {time.monotonic() - t_start} s")
        return output

    def reset(self):
        super().reset()

        # Clear caches
        for attr in "_rxns_cached", "_covar_rxns_cached", "_cho_covar_rxns_cached":
            if hasattr(self, attr):
                delattr(self, attr)


class Constant_Mean_Latent_GP_Feasibility(_MarginalConstantFeasibilityModel, _Latent_GP_FeasibilityModel):
    pass


class Rank_Mean_Latent_GP_Feasibility(_MarginalRankFeasibilityModel, _Latent_GP_FeasibilityModel):
    pass
