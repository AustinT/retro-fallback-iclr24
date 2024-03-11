from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# Note: this method is tested implicitly with the GP feasibility model
def smiles_to_fingerprint_arr(smiles_list: list[str], radius: int = 1, n_bits: int = 2048) -> np.array:
    # Create mol objects
    mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    assert None not in mol_list

    # Get count fingerprint dictionaries
    fp_dicts = [
        rdMolDescriptors.GetMorganFingerprint(mol, radius=radius, useCounts=True).GetNonzeroElements()
        for mol in mol_list
    ]

    out = np.zeros((len(fp_dicts), n_bits))
    for i, fp in enumerate(fp_dicts):
        for k, v in fp.items():
            out[i, k % n_bits] += v
    return out
