from __future__ import annotations

import abc
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class StochasticProcess(Generic[T], abc.ABC):
    """Base class for stochastic processes."""

    def __init__(self, num_samples: int, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def posterior_sample(
        self,
        inputs: set[T],
        observed_samples: dict[T, np.ndarray],
    ) -> dict[T, np.ndarray]:
        """Sample outcomes for a set of inputs, given a set of observed outcomes for other inputs."""

    def prior_sample(self, inputs: set[T]) -> dict[T, np.ndarray]:
        """Sample from the prior at at set of inputs."""
        return self.posterior_sample(inputs, {})

    @abc.abstractmethod
    def marginal_probability(self, inputs: set[T]) -> dict[T, float]:
        """Marginal probability for a set of inputs."""

    def reset(self):
        """Resets this model (deleting any caches/stored information)."""
        pass


class IndependentStochasticProcess(StochasticProcess[T]):
    """Subclass whose samples are independent, and therefore only depend on the marginal probabilities."""

    def posterior_sample(
        self,
        inputs: set[T],
        observed_samples: dict[T, np.ndarray],
    ) -> dict[T, np.ndarray]:

        # Partition inputs into observed and unobserved
        inputs_observed = set(observed_samples.keys()) & inputs
        inputs_unobserved = inputs - inputs_observed

        # Find marginal probabilities for unobserved inputs and sample outcomes
        marginal_probabilities = self.marginal_probability(inputs_unobserved)
        samples_for_unobserved = {
            inp: self.generator.binomial(1, p, self.num_samples) for inp, p in marginal_probabilities.items()
        }

        # Construct output dictionary
        output: dict[T, np.ndarray] = {}
        for inp in inputs:
            if inp in inputs_observed:
                output[inp] = observed_samples[inp]
            else:
                output[inp] = samples_for_unobserved[inp]
        return output
