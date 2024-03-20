"""Script to plot retro-fallback variability."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot_comparison import (
    feas_to_title,
    set_iclr_rc_params,
)


@dataclass(frozen=True)
class ExperimentParams:
    feas_model: str
    alg: str
    heuristic: str
    num_samples: int


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_fmt", type=str, default="pdf")
    args = parser.parse_args()

    set_iclr_rc_params()

    # Create results dict
    results_dict: dict[ExperimentParams, list[dict[str, Any]]] = dict()
    all_smiles: list[str] = []
    for result in sorted(Path(args.results_dir).rglob("*.json")):  # ensures SMILES are in order
        params = ExperimentParams(
            num_samples=int(result.parent.name.split("_")[-1]),
            heuristic=result.parent.parent.parent.name,
            alg=result.parent.parent.parent.parent.name,
            feas_model=result.parent.parent.parent.parent.parent.name,
        )
        results_dict.setdefault(params, list())
        with open(result, "r") as f:
            res = json.load(f)
            results_dict[params].append(res)
            all_smiles.append(res["smiles"])

    # Make list of SMILES encountered unique
    all_smiles = list(dict.fromkeys(all_smiles))

    # Print quick results summary
    for k, v in results_dict.items():
        print(f"{k}: {len(v)} results")
    print("Total number of unique molecules:", len(all_smiles))

    # In general, one plot for each type of heuristic
    all_feas_models = sorted({params.feas_model for params in results_dict})
    for heuristic_group in [
        ["optimistic", "NONE"],
        ["sascore", "NONE"],
    ]:

        # Plot variation in mean SSP over time
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            for res_type, res in sorted(results_dict.items(), key=lambda t: t[0].num_samples):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                # Only retro-fallback should be present
                assert res_type.alg == "retro-fallback"

                # Produce results for every SMILES
                smiles_to_ssp: dict[str, list[np.ndarray]] = dict()
                for r in res:
                    smiles_to_ssp.setdefault(r["smiles"], list()).append(r["success_probabilities"])
                    del r

                # Get mean SSP for each trial
                ssp_arr = np.stack(list(smiles_to_ssp.values()))
                mean_ssps = ssp_arr.mean(axis=0)

                plt.errorbar(
                    x=np.asarray(res[0]["analysis_times"]),
                    y=np.mean(mean_ssps, axis=0),
                    yerr=np.std(mean_ssps, axis=0, ddof=1),  # sample STD
                    fmt=".-",
                    capsize=3,
                    label=f"$k={res_type.num_samples}$",
                )

                del res_type, res, ssp_arr, mean_ssps

            # Settings for every axis
            plt.xscale("log")
            plt.xlabel("num. calls to $B$")
            if i == 0:
                plt.ylabel("mean SSP")

            plt.title(feas_to_title(feas_model))

        # Big legend below the plot
        fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=6, loc="upper center")

        # Save plot
        plt.savefig(Path(args.output_dir) / f"ssp_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)
