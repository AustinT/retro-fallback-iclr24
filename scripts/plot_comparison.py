"""Script to make algorithm comparison plots"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tueplots import figsizes, fonts, fontsizes


@dataclass(frozen=True)
class ExperimentParams:
    feas_model: str
    alg: str
    heuristic: str


def feas_to_title(feas_name: str) -> str:
    marg, corr = feas_name.split("-")
    marg = {"constant": "const.", "rank": "rank"}[marg]
    corr = {"independent": "ind.", "gp": "GP"}[corr]
    return f"$\\xi_f$ {marg}, {corr}"


def alg_sort(alg_name: str) -> int:
    name_to_priority = {
        "retro-fallback": 0,
        "breadth-first": 1,
        "retro-star": 2,
        "mcts": 3,
    }
    return name_to_priority.get(alg_name, 4)


def alg_name_to_title(alg_name: str) -> str:
    if alg_name == "mcts":
        return "MCTS"
    elif alg_name == "retro-star":
        return "retro*"
    else:
        return alg_name


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_fmt", type=str, default="pdf")
    parser.add_argument("--individual_smiles", action="store_true")
    args = parser.parse_args()

    # Update matplotlib rc settings to match ICLR 2024 styles
    plt.rcParams.update(fontsizes.iclr2024())
    plt.rcParams.update(fonts.iclr2024())
    plt.rcParams.update(figsizes.iclr2024())
    plt.rcParams.update(  # Use Times New Roman instead of Times
        {
            "font.serif": ["Times New Roman"],
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )

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

        # Plot 1) mean SSP across all molecules
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                # Concatenate results
                ssp_arr = np.asarray([r["success_probabilities"] for r in res])

                plt.errorbar(
                    x=np.asarray(res[0]["analysis_times"]),
                    y=np.mean(ssp_arr, axis=0),
                    yerr=np.std(ssp_arr, axis=0, ddof=1) / np.sqrt(ssp_arr.shape[0]),  # sample std
                    fmt=".-",
                    capsize=3,
                    label=alg_name_to_title(res_type.alg),
                )

                del res_type, res, ssp_arr

            # Settings for every axis
            plt.xscale("log")
            plt.xlabel("num. calls to $B$")
            if i == 0:
                plt.ylabel("mean SSP")

            plt.title(feas_to_title(feas_model))

        # Big legend below the plot
        fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=4, loc="upper center")

        # Save plot
        plt.savefig(Path(args.output_dir) / f"ssp_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)

        # Plot 2) fraction solved
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                # Fraction solved
                solved_arr = (np.asarray([r["success_probabilities"] for r in res]) > 0).astype(float)

                plt.plot(
                    np.asarray(res[0]["analysis_times"]),
                    np.mean(solved_arr, axis=0),
                    ".-",
                    label=alg_name_to_title(res_type.alg),
                )

                del res_type, res, solved_arr

            # Settings for every axis
            plt.xscale("log")
            plt.xlabel("num. calls to $B$")
            if i == 0:
                plt.ylabel("Fraction solved")

            plt.title(feas_to_title(feas_model))

        # Big legend below the plot
        fig.legend(*axes[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, -0.0), ncol=4, loc="upper center")

        # Save plot
        plt.savefig(Path(args.output_dir) / f"frac_solved_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)

        # Plot 3) shortest synthesis plan length
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            route_lengths = []
            alg_names = []
            for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                MAX_ROUTE_LEN = 20
                route_lengths.append([min(MAX_ROUTE_LEN, d["shortest_route_over_time"][-1]) for d in res])
                alg_names.append(res_type.alg)

                del res_type, res

            # make boxplot
            plt.boxplot(
                route_lengths,
                labels=alg_names,
            )
            ax.tick_params(axis="x", labelrotation=60)
            del route_lengths, alg_names

            if i == 0:
                plt.ylabel("Shortest plan")

            plt.title(feas_to_title(feas_model))

        # Save plot
        plt.savefig(Path(args.output_dir) / f"shortest_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)

        # Plot 4) synthesis plan with highest success probability
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            route_succ = []
            alg_names = []
            for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                MAX_ROUTE_LEN = 20
                route_succ.append([d["most_feasible_route_over_time"][-1] for d in res])
                alg_names.append(res_type.alg)

                del res_type, res

            plt.boxplot(route_succ, labels=alg_names)
            ax.tick_params(axis="x", labelrotation=60)
            del route_succ, alg_names

            if i == 0:
                plt.ylabel("Most successful plan")

            plt.title(feas_to_title(feas_model))

        # Save plot
        plt.savefig(Path(args.output_dir) / f"most_feasible_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)

        # Plot 5) (optionally) individual SMILES
        # ==================================================
        if args.individual_smiles:
            for smiles_idx, curr_smiles in enumerate(all_smiles):

                fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
                for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
                    plt.sca(ax)
                    for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                        # Only plot for this particular feasibility model
                        # and this particular heuristic
                        if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                            continue

                        ssp_arr = np.asarray([r["success_probabilities"] for r in res if r["smiles"] == curr_smiles])
                        median_ssp = np.median(ssp_arr, axis=0)
                        plt.errorbar(
                            x=np.asarray(res[0]["analysis_times"]),
                            y=median_ssp,
                            yerr=np.stack(
                                [
                                    median_ssp - np.min(ssp_arr, axis=0),
                                    np.max(ssp_arr, axis=0) - median_ssp,
                                ]
                            ),
                            fmt=".-",
                            capsize=3,
                            label=alg_name_to_title(res_type.alg),
                        )

                        del res_type, res, ssp_arr, median_ssp

                    if i == 0:
                        plt.ylabel("min/median/max SSP")

                    plt.title(feas_to_title(feas_model))

                # Big legend below the plot
                fig.legend(
                    *axes[0].get_legend_handles_labels(),
                    bbox_to_anchor=(0.5, -0.0),
                    ncol=4,
                    loc="upper center",
                    title=f"SMILES: {curr_smiles}",
                    title_fontsize="x-small",
                )

                # Save plot
                plt.savefig(Path(args.output_dir) / f"mol{smiles_idx:04d}_SSP_{heuristic_group[0]}.{args.save_fmt}")
                plt.close(fig=fig)

        # Plot 6) scatter plots with runtime
        # ==================================================
        fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
        for i, (ax, feas_model) in enumerate(zip(axes.flatten(), all_feas_models)):
            plt.sca(ax)
            for res_type, res in sorted(results_dict.items(), key=lambda t: alg_sort(t[0].alg)):

                # Only plot for this particular feasibility model
                # and this particular heuristic
                if res_type.feas_model != feas_model or res_type.heuristic not in heuristic_group:
                    continue

                # Only plot retro-fallback
                if res_type.alg != "retro-fallback":
                    continue

                # Get all node / runtime pairs from the dataset
                n_nodes = []
                times = []
                for d in res:
                    for n, t in zip(d["num_nodes_over_time"], d["search_duration_over_time"]):
                        if not np.isnan(t):
                            times.append(t)
                            n_nodes.append(n)
                n_nodes = np.asarray(n_nodes)
                times = np.asarray(times)

                # Figure out line of best fit (but only for long runtimes)
                LOG10_LINEAR_FIT_CUTOFF = 2.5
                mask = n_nodes > (10**LOG10_LINEAR_FIT_CUTOFF)  # type: ignore
                poly_best_fit = np.polynomial.polynomial.Polynomial.fit(
                    np.log10(n_nodes[mask]), np.log10(times[mask]), deg=1
                )
                plot_logt = np.linspace(LOG10_LINEAR_FIT_CUTOFF, 4, 25)

                # Plot the results
                plt_idxs = np.random.choice(
                    len(times), size=500, replace=False
                )  # only plot a subset of the data (otherwise it's too crowded)
                scatter = plt.loglog(n_nodes[plt_idxs], times[plt_idxs], ",")[0]
                plt.loglog(
                    10**plot_logt,
                    10 ** poly_best_fit(plot_logt),
                    "-",
                    color=scatter.get_color(),
                    label=f"p={poly_best_fit.convert().coef[-1]:.2f}",
                )

                del res_type, res, n_nodes, times

            plt.legend()
            plt.xlabel("Nodes in $\\mathcal{G}'$")
            if i == 0:
                plt.ylabel("Runtime (s)")

            plt.title(feas_to_title(feas_model))

        # Save plot
        plt.savefig(Path(args.output_dir) / f"runtimes_{heuristic_group[0]}.{args.save_fmt}")
        plt.close(fig=fig)
