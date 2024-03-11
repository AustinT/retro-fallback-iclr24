"""Script to process SMILES."""

from __future__ import annotations

import argparse
import random
from typing import Optional

import joblib
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def safe_canon_smiles(smiles: str) -> Optional[str]:
    """Return Canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--inventory_csv", type=str, required=True)
    args = parser.parse_args()

    # Read inventory
    inventory = pd.read_csv(args.inventory_csv)

    # Read input file
    with open(args.input_file, "r") as f:
        raw_smiles = [line.strip() for line in f.readlines()]

    # Canonicalize the SMILES in parallel with joblib
    canon_smiles = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(safe_canon_smiles)(smiles) for smiles in tqdm(raw_smiles, desc="Canonicalizing SMILES")
    )

    # Keep only unique SMILES not in the inventory
    smiles_to_keep = set(canon_smiles) - set(inventory["smiles"].to_list())
    smiles_to_keep.discard(None)

    # Shuffle and output
    smiles_list = list(smiles_to_keep)
    rng = random.Random(2024)
    rng.shuffle(smiles_list)
    with open(args.output_file, "w") as f:
        f.write("\n".join(smiles_list))


if __name__ == "__main__":
    main()
