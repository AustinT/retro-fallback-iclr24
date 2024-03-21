"""
Holds specialized inventories used in this paper.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from syntheseus import Molecule
from syntheseus.search.mol_inventory import BaseMolInventory

from retro_fallback_iclr24.stochastic_processes.buyability import IndependentBuyabilityModel

EMOLECULES_INVENTORY_CSV = Path(__file__).parent / "eMolecules" / "emolecules_inventory.csv"
FUSION_RETRO_INVENTORY = Path(__file__).parent / "fusion_retro" / "zinc_stock_17_04_20.hdf5"
FUSION_RETRO_INCHI_STR = "fusion_retro_inchi_key"


class eMoleculesInventory(BaseMolInventory):
    def __init__(self, max_tier: int, eMolecules_file: str = str(EMOLECULES_INVENTORY_CSV), **kwargs):
        super().__init__(**kwargs)
        self.max_tier = max_tier

        # Read data frame
        df = pd.read_csv(eMolecules_file)
        smiles_list = df.smiles.to_list()
        tier_list = df.tier.to_list()

        # Make SMILES to tier dictionary
        self._smiles_to_tier = {s: int(tier) for s, tier in zip(smiles_list, tier_list)}

    def is_purchasable(self, mol: Molecule) -> bool:
        return self._smiles_to_tier.get(mol.smiles, math.inf) <= self.max_tier

    def fill_metadata(self, mol: Molecule) -> None:
        super().fill_metadata(mol)  # will fill is purchasable
        tier = self._smiles_to_tier.get(mol.smiles, None)
        if tier is not None:
            mol.metadata["emols_tier"] = tier


class eMoleculesTieredBuyabilityModel(IndependentBuyabilityModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tier_to_prob = {  # could manually overwrite if desired
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 0.5,
            4: 0.2,
            5: 0.05,
        }
        self.default_prob = 0.0  # for tiers not in the dictionary

    def marginal_probability(self, inputs: set[Molecule]) -> dict[Molecule, float]:
        mol_to_tier = {mol: mol.metadata.get("emols_tier", -1) for mol in inputs}
        return {mol: self.tier_to_prob.get(mol_to_tier[mol], self.default_prob) for mol in inputs}


class FusionRetroInventory(BaseMolInventory):
    def __init__(self, inventory_file: str = str(FUSION_RETRO_INVENTORY), **kwargs):
        super().__init__(**kwargs)

        # Read data frame
        df = pd.read_hdf(inventory_file, key="table")
        self._inchi_keys = set(df.inchi_key.to_list())

    def _fill_inchi_key(self, mol: Molecule) -> None:
        if FUSION_RETRO_INCHI_STR not in mol.metadata:
            mol.metadata[FUSION_RETRO_INCHI_STR] = Chem.MolToInchiKey(Chem.MolFromSmiles(mol.smiles))

    def is_purchasable(self, mol: Molecule) -> bool:
        self._fill_inchi_key(mol)
        return mol.metadata[FUSION_RETRO_INCHI_STR] in self._inchi_keys
