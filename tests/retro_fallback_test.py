from __future__ import annotations

import pytest
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.conftest import *  # noqa: F403
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import *  # noqa: F403
from syntheseus.tests.search.conftest import RetrosynthesisTask

from retro_fallback_iclr24.retro_fallback import RetroFallbackSearch
from retro_fallback_iclr24.stochastic_processes.feasibility import ConstantIndependentFeasibility


class TestRetroFallback(BaseAlgorithmTest):
    time_limit_multiplier = 10.0  # generally slower than most algorithms
    time_limit_upper_bound_s = 0.1  # steps are much slower

    def setup_algorithm(self, **kwargs):
        kwargs.setdefault(
            "feasibility_model",
            ConstantIndependentFeasibility(constant=0.5, num_samples=10_000),
        )
        kwargs.setdefault("value_function", ConstantNodeEvaluator(1.0))
        return RetroFallbackSearch(**kwargs)

    @pytest.mark.skip(reason="RetroFallbackSearch operates on a graph by default")
    @pytest.mark.parametrize("prevent", [False, True])
    def test_prevent_repeat_mol_in_trees(
        self, prevent: bool, retrosynthesis_task5: RetrosynthesisTask  # noqa: F405
    ) -> None:
        pass

    @pytest.mark.parametrize("set_depth", [True, False])
    def test_set_depth(self, set_depth: bool, retrosynthesis_task4: RetrosynthesisTask) -> None:  # noqa: F405
        """Override the "False" case of this test because `set_depth` is required for RetroFallbackSearch."""
        if set_depth:
            super().test_set_depth(set_depth, retrosynthesis_task4)
        else:
            with pytest.raises(ValueError):
                super().test_set_depth(set_depth, retrosynthesis_task4)
