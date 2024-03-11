from __future__ import annotations

import math
import os
import sys

import pytest
from rdkit.Chem import RDConfig
from syntheseus import Molecule

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

from syntheseus.search.graph.and_or import OrNode

from retro_fallback_iclr24.iclr24_experiments.sascore_heuristic import SAScoreCostHeuristic, SAScoreProbabilityHeuristic

SMILES_TO_SA_SCORE = {  # calculated offline
    "C": 7.3284153846153846,
    "Oc1ccccc1": 1.1765612788317714,
}


@pytest.fixture
def ornodes() -> list[OrNode]:
    return [OrNode(Molecule(smiles)) for smiles in SMILES_TO_SA_SCORE]


def test_probability_heuristic(ornodes: list[OrNode]) -> None:
    heuristic = SAScoreProbabilityHeuristic()
    scores = heuristic(ornodes)
    assert scores == pytest.approx([1.0 - (SMILES_TO_SA_SCORE[node.mol.smiles] - 1) / 10.0 for node in ornodes])


def test_cost_heuristic(ornodes: list[OrNode]) -> None:
    heuristic = SAScoreCostHeuristic()
    scores = heuristic(ornodes)
    assert scores == pytest.approx(
        [-math.log(1.0 - (SMILES_TO_SA_SCORE[node.mol.smiles] - 1) / 10.0) for node in ornodes]
    )
