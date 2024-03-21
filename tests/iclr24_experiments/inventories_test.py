from __future__ import annotations

import pytest
from syntheseus import Molecule

from retro_fallback_iclr24.iclr24_experiments.inventories import (
    EMOLECULES_INVENTORY_CSV,
    FUSION_RETRO_INVENTORY,
    FusionRetroInventory,
    eMoleculesInventory,
    eMoleculesTieredBuyabilityModel,
)

SMILES_DEFINITELY_NOT_PURCHASABLE = "CC1OC(=O)C1N=C(O)OC1CCN(C(=O)c2ccccc2)CC1"  # from GuacaMol dataset
SMILES_DEFINITELY_PURCHASABLE = "CC(=O)C"  # acetone


@pytest.mark.skipif(not EMOLECULES_INVENTORY_CSV.exists(), reason="eMolecules inventory file not found")
def test_eMolecules_inventory() -> None:
    inventory = eMoleculesInventory(max_tier=2)
    assert inventory.is_purchasable(Molecule(SMILES_DEFINITELY_PURCHASABLE))
    assert not inventory.is_purchasable(Molecule(SMILES_DEFINITELY_NOT_PURCHASABLE))


def test_eMolecules_tiered_buyability_model() -> None:
    mol1 = Molecule(SMILES_DEFINITELY_PURCHASABLE, metadata={"emols_tier": 0})
    mol2 = Molecule(SMILES_DEFINITELY_NOT_PURCHASABLE, metadata={"emols_tier": 4})
    mol3 = Molecule("C")  # no emols tier
    marginal_probs = eMoleculesTieredBuyabilityModel(num_samples=10).marginal_probability({mol1, mol2, mol3})
    assert marginal_probs == {mol1: 1.0, mol2: 0.2, mol3: 0.0}


@pytest.mark.skipif(not FUSION_RETRO_INVENTORY.exists(), reason="Fusion Retro inventory file not found")
def test_fusion_retro_inventory() -> None:
    pytest.importorskip("tables")  # requires pytables
    inventory = FusionRetroInventory()
    assert inventory.is_purchasable(Molecule(SMILES_DEFINITELY_PURCHASABLE))
    assert not inventory.is_purchasable(Molecule(SMILES_DEFINITELY_NOT_PURCHASABLE))
