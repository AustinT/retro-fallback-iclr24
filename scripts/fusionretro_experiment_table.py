"""Script to make table for FusionRetro experiment."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plot_comparison import (
    ExperimentParams,
    alg_name_to_title,
    alg_sort,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Create results dict
    results_dict: dict[ExperimentParams, list[dict[str, Any]]] = dict()
    all_smiles: list[str] = []
    for result in sorted(Path(args.results_dir).rglob("*.json")):  # ensures SMILES are in order
        params = ExperimentParams(
            heuristic=result.parent.parent.name,
            alg=result.parent.parent.parent.name,
            feas_model=result.parent.parent.parent.parent.name,
        )
        results_dict.setdefault(params, list())
        with open(result, "r") as f:
            res = json.load(f)
            results_dict[params].append(res)
            all_smiles.append(res["smiles"])
            del res
        del result

    # Make list of SMILES encountered unique
    all_smiles = list(dict.fromkeys(all_smiles))

    # Print quick results summary
    for k, v in results_dict.items():
        print(f"{len(v)} results: {k}")
    print("Total number of unique molecules:", len(all_smiles))

    # Create lines in table
    METRICS = [r"Mean SSP (\%)", r"Solved (\%)", r"Precursor Match (\%)"]
    lines = [
        r"\begin{tabular}",
        "{ll" + r"r@{\hspace{0.02cm}$\pm$\hspace{0.02cm}}l@{\hspace{0.30cm}}" * len(METRICS) + "}",
        r"\toprule",
        "Algorithm & Heuristic & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in METRICS) + r"\\",
        r"\midrule",
    ]
    for expt_params, res_list in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):
        # Get heuristic name and algorithm name
        tokens = [
            alg_name_to_title(expt_params.alg),
            {"NONE": "N/A", "optimistic": "optimistic", "sascore": "SAScore"}[expt_params.heuristic],
        ]

        # First metric: mean SSP with standard error
        ssp_values = [100 * res["success_probabilities"][-1] for res in res_list]
        tokens.append(f"{np.mean(ssp_values):.2f}")
        tokens.append(f"{np.std(ssp_values) / np.sqrt(len(ssp_values)):.2f}")

        # Second metric: fraction solved (with standard error)
        solved = [float(v > 0) * 100 for v in ssp_values]
        tokens.append(f"{np.mean(solved):.2f}")
        tokens.append(f"{np.std(solved) / np.sqrt(len(solved)):.2f}")
        del solved

        # Third metric: matching (with standard error)
        start_mol_matches = [100 * float(res["start_mol_match_most_feasible_route"]) for res in res_list]
        tokens.append(f"{np.mean(start_mol_matches):.2f}")
        tokens.append(f"{np.std(start_mol_matches) / np.sqrt(len(start_mol_matches)):.2f}")
        lines.append(" & ".join(tokens) + r"\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    with open(Path(args.output_dir) / "fusionretro_regression.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
