import math
from collections import Counter

import pytest
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.tests.conftest import *  # noqa: F403
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.algorithms.test_best_first import DictMolCost
from syntheseus.tests.search.conftest import *  # noqa: F403
from syntheseus.tests.search.conftest import RetrosynthesisTask

from retro_fallback_iclr24.independent_proto_retro_fallback import IndependentProtoRetroFallback
from retro_fallback_iclr24.stochastic_processes.feasibility import (
    ConstantIndependentFeasibility,
)


@pytest.fixture
def mol_value_fn() -> DictMolCost:
    """Return success estimator for hand-worked example."""
    return DictMolCost(
        smiles_to_cost={
            "CS": 0.95,
            "COCO": 0.8,
            "SCOS": 0.7,
        },
        default=0.1,
    )


BY_HAND_STEP1_RXNS = [
    "C.OCS>>COCS",
    "CCOC>>COCS",
    "CO.CS>>COCS",
    "COC.S>>COCS",
    "COCO>>COCS",
    "OOCS>>COCS",
    "SCOS>>COCS",
]
BY_HAND_STEP2_RXNS = BY_HAND_STEP1_RXNS + [
    "C.S>>CS",
    "CC>>CS",
    "OS>>CS",
    "SS>>CS",
    "CO>>CS",
]
BY_HAND_STEP3_RXNS = BY_HAND_STEP2_RXNS + [
    "C.OCO>>COCO",
    "CCOC>>COCO",
    "CO>>COCO",
    "COC.O>>COCO",
    "OCOO>>COCO",
    "OCOS>>COCO",
]


class TestIndepdendentProtoRetroFallback(BaseAlgorithmTest):
    def setup_algorithm(self, **kwargs):
        kwargs.setdefault("feasibility_model", ConstantIndependentFeasibility(0.5, num_samples=1))
        kwargs.setdefault("value_function", ConstantNodeEvaluator(1.0))
        return IndependentProtoRetroFallback(**kwargs)

    @pytest.mark.parametrize("expand_purchasable_mols", [False, True])
    def test_expand_purchasable_mols(
        self,
        retrosynthesis_task1: RetrosynthesisTask,  # noqa: F405
        expand_purchasable_mols: bool,
    ) -> None:
        if expand_purchasable_mols:
            with pytest.raises(ValueError):
                super().test_expand_purchasable_mols(retrosynthesis_task1, expand_purchasable_mols)
        else:
            super().test_expand_purchasable_mols(retrosynthesis_task1, expand_purchasable_mols)

    @pytest.mark.parametrize("set_depth", [True, False])
    def test_set_depth(self, set_depth: bool, retrosynthesis_task4: RetrosynthesisTask) -> None:  # noqa: F405
        """
        Test the 'set_depth' argument, which toggles whether the 'depth'
        attribute is set during node updates.

        The test is run on a small finite tree for simplicity.
        """
        if set_depth:
            super().test_set_depth(set_depth, retrosynthesis_task4)
        else:
            with pytest.raises(ValueError):
                super().test_set_depth(set_depth, retrosynthesis_task4)

    def test_by_hand_step1(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Test retro-fallback on a hand-designed example with a custom reaction cost/value function
        (the default cost/value functions will just act the same as breadth-first search).
        Example is on retrosynthesis task2: it finds the 2 step solution then finds other solutions.

        In the first step, the algorithm should expand the root node,
        and the tree should have the following structure:

               ------------ COCS -----------------
              /      /      |     \    \     \     \
            C+OCS  CO+CS  COC+S  OOCS  SOCS  COCO  COCC
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            1,
            value_function=mol_value_fn,
        )
        assert output_graph.reaction_smiles_counter() == Counter(BY_HAND_STEP1_RXNS)  # type: ignore  # unsure about rxn_counter
        assert len(output_graph) == 18
        assert not output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == math.inf
        assert math.isclose(output_graph.root_node.data["rfb_ind_succ_prob"], 0.0)  # no solution
        assert math.isclose(
            output_graph.root_node.data["rfb_max_success_1_expand"],
            0.5 * 0.95,  # this magic number is 0.5 that CO.CS>>COCS succeeds and 0.95 estimated success from CS
        )

    def test_by_hand_step2(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        Should expand "CS", yielding 2 routes (from purchasable CC and CO)

               ------------ COCS -----------------
              /      /      |     \    \     \     \
            C+OCS  CO+CS  COC+S  OOCS  SOCS  COCO  COCC
                       |
                   -----------
                  /   / \  \  \
                C+S  CC OS SS CO
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            2,
            value_function=mol_value_fn,
        )
        assert output_graph.reaction_smiles_counter() == Counter(BY_HAND_STEP2_RXNS)  # type: ignore  # unsure about rxn_counter
        assert len(output_graph) == 29
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 2
        assert math.isclose(output_graph.root_node.data["rfb_ind_succ_prob"], 0.375)
        assert math.isclose(
            output_graph.root_node.data["rfb_max_success_1_expand"],
            1 - (1 - 0.375) * (1 - 0.5 * 0.8),  # success prob if COCO node attained estimated value of 0.8
        )

    def test_by_hand_step3(
        self,
        retrosynthesis_task2: RetrosynthesisTask,  # noqa: F405
        mol_value_fn: DictMolCost,
    ) -> None:
        r"""
        Continuation of test above.

        Should expand "COCO", yielding 1 additional route.

               ------------ COCS -----------------
              /      /      |     \    \     \     \
            C+OCS  CO+CS  COC+S  OOCS  SCOS*  COCO  COCC
                       |                      |
                   -----------                |
                  /   / \  \  \               |
                C+S  CC OS SS CO              |
                                 ------------------------
                                /  /     /     \     \   \
                            C+OCO  CO  COC+O  OOCO  SOCO  COCC
        """
        output_graph = self.run_alg_for_n_iterations(
            retrosynthesis_task2,
            3,
            value_function=mol_value_fn,
        )
        assert output_graph.reaction_smiles_counter() == Counter(BY_HAND_STEP3_RXNS)  # type: ignore  # unsure about rxn_counter
        assert output_graph.root_node.has_solution
        assert get_first_solution_time(output_graph) == 2
        assert math.isclose(
            output_graph.root_node.data["rfb_ind_succ_prob"],
            1 - (1 - 0.375) * (1 - 0.25),
        )
        assert math.isclose(
            output_graph.root_node.data["rfb_max_success_1_expand"],
            1
            - (1 - output_graph.root_node.data["rfb_ind_succ_prob"])
            * (1 - 0.5 * 0.7),  # if SCOS(*) attained a value of 0.7
        )
