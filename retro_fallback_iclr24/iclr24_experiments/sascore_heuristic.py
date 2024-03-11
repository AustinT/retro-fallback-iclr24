from __future__ import annotations

import math
import os
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator


def mol_to_sa_succ_prob(mol):
    sa_score = sascorer.calculateScore(Chem.MolFromSmiles(mol.smiles))
    return 1.0 - (sa_score - 1) / 10.0


class SAScoreProbabilityHeuristic(NoCacheNodeEvaluator[OrNode]):
    """
    Estimates probability as 1 - (SA_score-1)/10
    (linearly decreasing from 1 to 0 as SA_score increases)
    """

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        return [mol_to_sa_succ_prob(node.mol) for node in nodes]


class SAScoreCostHeuristic(SAScoreProbabilityHeuristic):
    """Cost = -log(probability)"""

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        probs = super()._evaluate_nodes(nodes, graph)
        return [-math.log(p) for p in probs]
