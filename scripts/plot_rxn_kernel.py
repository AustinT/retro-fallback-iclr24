"""Script to plot some similar reactions."""

import argparse
import gzip

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from syntheseus import Molecule, SingleProductReaction

from retro_fallback_iclr24.iclr24_experiments.latent_gp_feasibility import (
    Constant_Mean_Latent_GP_Feasibility,
    _reaction_similarity_kernel,
)


def rxn_to_rdkit_rxn(rxn: SingleProductReaction) -> AllChem.ChemicalReaction:
    new_rxn = AllChem.ChemicalReaction()
    for mol in rxn.reactants:
        new_rxn.AddReactantTemplate(Chem.MolFromSmiles(mol.smiles))
    new_rxn.AddProductTemplate(Chem.MolFromSmiles(rxn.product.smiles))
    return new_rxn


def get_rxn_in_similarity_interval(all_rxns, idx1, s_low, s_high):
    query_reactions = [all_rxns[idx1]]
    compare_reactions = all_rxns[:idx1] + all_rxns[idx1 + 1 :]
    kern_values = _reaction_similarity_kernel(
        reactants1=np.stack([r.metadata["REACTANT_FP"] for r in query_reactions]),
        product1=np.stack([r.metadata["PRODUCT_FP"] for r in query_reactions]),
        reactants2=np.stack([r.metadata["REACTANT_FP"] for r in compare_reactions]),
        product2=np.stack([r.metadata["PRODUCT_FP"] for r in compare_reactions]),
    )
    kern_values = kern_values.flatten()
    eligible_idxs = np.where((kern_values >= s_low) & (kern_values <= s_high))[0]
    chosen_idx = eligible_idxs[0]
    return query_reactions[0], compare_reactions[chosen_idx], kern_values[chosen_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reactions_file", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Read reactions from file
    reactions: list[SingleProductReaction] = []
    with gzip.open(args.reactions_file, "rt") as f:
        for line in f.readlines():
            line = line.strip()
            r, p = line.split(">>")
            reactions.append(
                SingleProductReaction(
                    product=Molecule(p),
                    reactants=[Molecule(s) for s in r.split(".")],
                )
            )

    # Label with metadata
    feas_model = Constant_Mean_Latent_GP_Feasibility(num_samples=10, constant=0.5)
    feas_model._add_fingerprint_metadata_to_reactions(reactions)

    # Produce pictures of reactions in similarity intervals
    for pos, idx in enumerate([7, 102, 6002]):
        for interval in [
            (0.8, 0.9),
            (0.4, 0.6),
            (0.05, 0.1),
        ]:
            try:
                r1, r2, v = get_rxn_in_similarity_interval(reactions, idx, *interval)
                Draw.ReactionToImage(rxn_to_rdkit_rxn(r1)).save(f"{args.output_dir}/rxn_{pos}_{interval[0]}-a.png")
                Draw.ReactionToImage(rxn_to_rdkit_rxn(r2)).save(f"{args.output_dir}/rxn_{pos}_{interval[0]}-b.png")
                print(f"{idx} {interval} {v}")
            except IndexError:
                print(f"Could not find reaction in interval {interval} for idx {idx}")
