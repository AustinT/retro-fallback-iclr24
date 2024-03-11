from __future__ import annotations

import pytest
from syntheseus import Molecule

from retro_fallback_iclr24.iclr24_experiments.inventories import (
    EMOLECULES_INVENTORY_CSV,
    FUSION_RETRO_INVENTORY,
    FusionRetroInventory,
    eMoleculesInventory,
)

SMILES_DEFINITELY_NOT_PURCHASABLE = "CC1OC(=O)C1N=C(O)OC1CCN(C(=O)c2ccccc2)CC1"  # from GuacaMol dataset
SMILES_DEFINITELY_PURCHASABLE = "CC(=O)C"  # acetone


@pytest.mark.skipif(not EMOLECULES_INVENTORY_CSV.exists(), reason="eMolecules inventory file not found")
def test_eMolecules_inventory() -> None:
    inventory = eMoleculesInventory(max_tier=2)
    assert inventory.is_purchasable(Molecule(SMILES_DEFINITELY_PURCHASABLE))
    assert not inventory.is_purchasable(Molecule(SMILES_DEFINITELY_NOT_PURCHASABLE))


@pytest.mark.skipif(not FUSION_RETRO_INVENTORY.exists(), reason="Fusion Retro inventory file not found")
def test_fusion_retro_inventory() -> None:
    pytest.importorskip("tables")  # requires pytables
    inventory = FusionRetroInventory()
    assert inventory.is_purchasable(Molecule(SMILES_DEFINITELY_PURCHASABLE))
    assert not inventory.is_purchasable(Molecule(SMILES_DEFINITELY_NOT_PURCHASABLE))
