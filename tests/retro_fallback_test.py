from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from syntheseus import Molecule
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.conftest import *  # noqa: F403
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.algorithms.test_best_first import DictMolCost
from syntheseus.tests.search.conftest import *  # noqa: F403
from syntheseus.tests.search.conftest import RetrosynthesisTask

from retro_fallback_iclr24.metrics import estimate_successful_synthesis_probability, iter_routes_feasibility_order
from retro_fallback_iclr24.retro_fallback import RetroFallbackSearch
from retro_fallback_iclr24.stochastic_processes.buyability import BinaryBuyability
from retro_fallback_iclr24.stochastic_processes.feasibility import ConstantIndependentFeasibility


@pytest.fixture
def mol_value_fn() -> DictMolCost:
    """Return heuristic for hand-worked example."""
    return DictMolCost(
        smiles_to_cost={
            "CS": 0.95,
            "COCO": 0.8,
            "SCOS": 0.7,
        },
        default=0.1,
    )


@pytest.fixture
def rxn_feas_fn() -> ConstantIndependentFeasibility:
    """Return feasibility function for hand-worked example."""
    return ConstantIndependentFeasibility(constant=0.5, num_samples=100_000)


BY_HAND_STEP1_RXNS = [
    "C.OCS>>COCS",
    "CCOC>>COCS",
    "CO.CS>>COCS",
    "COC.S>>COCS",
    "COCO>>COCS",
    "OOCS>>COCS",
    "SCOS>>COCS",
]  # reactions we expect from one expansion


class TestRetroFallback(BaseAlgorithmTest):
    """
    Test that retro-fallback behaves as expected.

    Tests are also re-purposed to also test some analysis functions.
    """

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

    def test_by_hand_step1(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        rxn_feas_fn: ConstantIndependentFeasibility,
        mol_value_fn,  # noqa: F811
    ) -> None:
        r"""
        Test retro-fallback on a hand-worked example.

        In the first step, the algorithm should expand the root node,
        and the tree should have the following structure:

               ------------ COCS -----------------
              /      /      |     \    \     \     \
            C+OCS  CO+CS  COC+S  OOCS  SOCS  COCO  COCC
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            1,
            feasibility_model=rxn_feas_fn,
            value_function=mol_value_fn,
        )
        assert output_graph.reaction_smiles_counter() == Counter(BY_HAND_STEP1_RXNS)  # type: ignore  # unsure about rxn_counter
        assert len(output_graph) == 18
        assert not output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == math.inf
        assert np.allclose(output_graph.root_node.data["retro_fallback_s"], 0.0)

        # Check some rho values
        for smiles_str, expected_rho in [
            (
                "CO",
                0.95 * 0.5,  # this magic number is 0.95 estimated success from CS and 0.5 that CO.CS>>COCS succeeds
            ),
            ("CS", 0.95 * 0.5),
            ("C", 0.1 * 0.1 * 0.5),
            (
                "COCO",
                0.8 * 0.5,
            ),
            (
                "COCC",
                0.1 * 0.5,
            ),
        ]:
            assert math.isclose(
                output_graph._mol_to_node[Molecule(smiles_str)].data["retro_fallback_rho"].mean(),
                expected_rho,
                abs_tol=0.01,
            )

        # Check final success probability
        assert math.isclose(
            estimate_successful_synthesis_probability(
                output_graph, rxn_feas_fn, BinaryBuyability(rxn_feas_fn.num_samples)
            ),
            0.0,
        )

        # Check some leaf distances
        for smiles_str, expected_d in [
            ("COCS", 2),
            ("C", 0),
            ("COCO", 0),
        ]:
            assert output_graph._mol_to_node[Molecule(smiles_str)].data["leaf_distance"] == expected_d

        # Should not be any feasible routes
        with pytest.raises(StopIteration):
            next(iter_routes_feasibility_order(output_graph, max_routes=1))

    def test_by_hand_step2(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        rxn_feas_fn: ConstantIndependentFeasibility,
        mol_value_fn,  # noqa: F811
    ) -> None:
        r"""
        Continuation of test above.

        Should expand "CS", yielding 2 routes (from purchasable CC and CO)

               ------------ COCS -------------------------
              /           /      |         \    \     \     \
            C[1]+OCS  CO[2]+CS  COC+S[3]  OOCS  SOCS  COCO  COCC
                            |
                   -----------------
                  /         / \  \  \
                C[1]+S[3]  CC OS SS CO[2]

        Note: [X] indicate pairs of nodes which are the same. This is a graph, not a tree!
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            2,
            feasibility_model=rxn_feas_fn,
            value_function=mol_value_fn,
        )
        assert len(output_graph) == 26  # 3 less than in test_rfb_independent.py because it runs on a graph
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 2

        # Average success outcome should be the same as original algorithm
        expected_synthesis_probability = 0.375
        assert math.isclose(
            output_graph.root_node.data["retro_fallback_s"].mean(),
            expected_synthesis_probability,
            abs_tol=0.01,
        )

        # Check some rho values
        success_mask = np.isclose(output_graph.root_node.data["retro_fallback_s"], 1.0)
        for smiles_str, expected_rho in [
            (
                "COCO",
                0.5 * 0.8,  # success of COCO is independent of existing routes
            ),
            (
                "C",
                0.5 * 0.1 * 0.1,
            ),
        ]:
            success_array = np.ma.masked_array(
                output_graph._mol_to_node[Molecule(smiles_str)].data["retro_fallback_rho"],
                mask=success_mask,
            )
            assert math.isclose(
                success_array.mean(),
                expected_rho,
                abs_tol=0.01,
            )

        # Check final success probability
        assert math.isclose(
            estimate_successful_synthesis_probability(
                output_graph, rxn_feas_fn, BinaryBuyability(rxn_feas_fn.num_samples)
            ),
            expected_synthesis_probability,
            abs_tol=0.01,
        )

        # Check some leaf distances
        for smiles_str, expected_d in [
            ("COCS", 2),
            ("C", 0),
            ("S", 0),
            ("COCO", 0),
            ("CS", 2),
        ]:
            assert output_graph._mol_to_node[Molecule(smiles_str)].data["leaf_distance"] == expected_d

        # Most feasible route
        best_route_feasibility, _ = next(iter_routes_feasibility_order(output_graph, max_routes=1))
        best_route_feasibility = math.exp(-best_route_feasibility)
        assert math.isclose(best_route_feasibility, 0.25, abs_tol=0.01)

    def test_by_hand_step3(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        rxn_feas_fn: ConstantIndependentFeasibility,
        mol_value_fn,  # noqa: F811
    ) -> None:
        r"""
        Continuation of test above.

        Should expand "COCO", yielding 1 additional route.

               ------------ COCS ------------------------------
              /           /      |            \    \     \     \
            C[1]+OCS  CO[2]+CS  COC[4]+S[3]  OOCS  SOCS  COCO  COCC[5]
                            |                             |
                   -----------------                      |
                  /         / \  \  \                     |
                C[1]+S[3]  CC OS SS CO[2]                 |
                                                          |
                            -----------------------------------
                            /       /     /          \     \   \
                        C[1]+OCO  CO[2]  COC[4]+O  OOCO  SOCO  COCC[5]

        Note: [X] indicate pairs of nodes which are the same. This is a graph, not a tree!
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            3,
            feasibility_model=rxn_feas_fn,
            value_function=mol_value_fn,
        )
        assert len(output_graph) == 36
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 2

        # Average success outcome should be the same as original algorithm
        expected_synthesis_probability = 1 - (1 - 0.375) * (1 - 0.25)
        assert math.isclose(
            output_graph.root_node.data["retro_fallback_s"].mean(),
            expected_synthesis_probability,
            abs_tol=0.01,
        )

        # Check some rho values
        success_mask = np.isclose(output_graph.root_node.data["retro_fallback_s"], 1.0)
        for smiles_str, expected_rho in [
            (
                "COCC",
                0.5 * 0.1,  # NOTE: this number ignores contribution from below COCO
            ),
            (
                "SCOS",
                0.5 * 0.7,
            ),
        ]:
            success_array = np.ma.masked_array(
                output_graph._mol_to_node[Molecule(smiles_str)].data["retro_fallback_rho"],
                mask=success_mask,
            )
            assert math.isclose(
                success_array.mean(),
                expected_rho,
                abs_tol=0.01,
            )

        # Check final success probability
        assert math.isclose(
            estimate_successful_synthesis_probability(
                output_graph, rxn_feas_fn, BinaryBuyability(rxn_feas_fn.num_samples)
            ),
            expected_synthesis_probability,
            abs_tol=0.01,
        )

        # Check some leaf distances
        for smiles_str, expected_d in [
            ("COCS", 2),
            ("C", 0),
            ("S", 0),
            ("COCO", 2),
            ("CS", 2),
        ]:
            assert output_graph._mol_to_node[Molecule(smiles_str)].data["leaf_distance"] == expected_d

        # Most feasible route
        best_route_feasibility, _ = next(iter_routes_feasibility_order(output_graph, max_routes=1))
        best_route_feasibility = math.exp(-best_route_feasibility)
        assert math.isclose(best_route_feasibility, 0.25, abs_tol=0.01)
