"""Python script to create and pre-process the eMolecules inventory."""

from __future__ import annotations

import collections
import json
import math
from typing import Any, Optional

import joblib
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

RAW_FILES = ["orders.tsv.gz", "quote_bb.tsv.gz"]


def safe_canon_smiles(smiles: str) -> Optional[str]:
    """Return Canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    # Create list of "smiles" to all listed purchase tiers
    smiles_to_purchase_tier = collections.defaultdict(list)
    num_nans = 0
    num_invalid = 0
    num_processed = 0
    for raw_file in RAW_FILES:
        df = pd.read_csv(raw_file, sep="\t", compression="gzip")

        # Canonicalize the SMILES in parallel with joblib
        smiles_iterator = tqdm(df["smiles"].to_list(), desc=f"SMILES from {raw_file}")
        canon_smiles = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(safe_canon_smiles)(smiles) for smiles in smiles_iterator
        )

        # Add the canonicalized SMILES to the dictionary
        for smiles, purchase_tier in zip(canon_smiles, df["highest tier"].to_list()):
            num_processed += 1
            if math.isnan(purchase_tier):
                num_nans += 1
            elif smiles is None:
                num_invalid += 1
            else:
                smiles_to_purchase_tier[smiles].append(purchase_tier)

    # Create a dataframe with the "smiles" and "tier".
    # Also do some analysis: how many molecules were found multiple times,
    # and how many molecules were found with multiple purchase tiers?
    num_repeat_mols = 0
    num_mols_with_diff_tiers = 0
    inventory: list[dict[str, Any]] = []
    for smiles, tiers in sorted(smiles_to_purchase_tier.items()):
        if len(tiers) > 1:
            num_repeat_mols += 1
        if len(set(tiers)) > 1:
            num_mols_with_diff_tiers += 1
        inventory.append(dict(smiles=smiles, tier=min(tiers)))

    # Output inventory
    inventory_df = pd.DataFrame(inventory)
    inventory_df = inventory_df.sort_values(by=["tier", "smiles"])  # sort by tier (smiles is tiebreaker)
    inventory_df.to_csv("emolecules_inventory.csv.gz", index=False, compression="gzip")

    # Output stats
    stats = dict(
        num_processed=num_processed,
        num_nans=num_nans,
        num_invalid=num_invalid,
        num_repeat_mols=num_repeat_mols,
        num_mols_with_diff_tiers=num_mols_with_diff_tiers,
        inventory_size=len(inventory),
        tier_counts=inventory_df["tier"].value_counts().to_dict(),
    )
    with open("emolecules_inventory_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
