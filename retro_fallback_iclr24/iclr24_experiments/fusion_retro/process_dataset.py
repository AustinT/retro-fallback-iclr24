"""Simple python script to process json dataset for FusionRetro."""

from __future__ import annotations

import argparse
import json

from rdkit import Chem

if __name__ == "__main__":
    # Read input file
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output_file_stem", type=str)
    args = parser.parse_args()
    with open(args.input, "r") as f:
        data = json.load(f)

    # Process data: create a list of canonical SMILES and starting molecule collection
    all_smiles = []
    smiles_to_molsets: dict[str, list] = dict()
    for sm, route_dict in data.items():
        sm = Chem.CanonSmiles(sm)
        all_smiles.append(sm)
        for rxn_idx in range(route_dict["num_reaction_trees"]):
            smiles_to_molsets.setdefault(sm, list()).append(
                [Chem.CanonSmiles(s) for s in route_dict[str(rxn_idx + 1)]["materials"]]
            )

    # Write output
    with open(args.output_file_stem + ".smiles", "w") as f:
        f.write("\n".join(all_smiles))

    with open(args.output_file_stem + ".json", "w") as f:
        json.dump(smiles_to_molsets, f, indent=2)
