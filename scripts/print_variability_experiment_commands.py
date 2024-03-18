from __future__ import annotations

import argparse
from pathlib import Path

from print_comparison_experiment_commands import (
    PREFIX_ENV_VARS,
    algorithms,
    datasets,
)

NUM_SAMPLE_ARRAY = [4, 16, 64, 256, 1024, 4096]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Root output directory")
    args = parser.parse_args()

    # Just do the one dataset, and change the number of replications / number of molecules
    dataset = datasets[1]
    assert dataset.name == "guacamol"
    dataset.num_replications = 10
    dataset.max_idx = dataset.batch_size  # just run 1 batch

    # Just do retro-fallback
    algorithm = algorithms[0]
    assert algorithm.name == "retro-fallback"

    # Loop over trials
    for trial in range(dataset.num_replications):

        # Loop over batches within this dataset
        for batch_start in range(0, dataset.max_idx, dataset.batch_size):
            batch_end = batch_start + dataset.batch_size

            # Loop over feasibility models
            for feas_model in dataset.feasibility_models:
                marg, corr = feas_model.split("-")

                # Loop over its heuristics
                for heur in algorithm.heuristics_to_test:

                    # Loop over number of samples
                    for num_samples in NUM_SAMPLE_ARRAY:

                        # Produce current output dir
                        curr_output_dir = (
                            Path(args.output_dir)
                            / dataset.name
                            / feas_model
                            / algorithm.name
                            / heur
                            / f"trial_{trial+1}"
                            / f"num_samples_{num_samples}"
                        )
                        curr_output_dir.mkdir(parents=True, exist_ok=True)

                        # Create command (except options for other algorithms are omitted)
                        command = (
                            PREFIX_ENV_VARS + "python scripts/main.py "
                            f"--smiles_file={dataset.smiles_file} "
                            f"--smiles_start_idx={batch_start} "
                            f"--smiles_end_idx={batch_end} "
                            f"--limit_rxn_model_calls={dataset.search_length} "
                            f"--marginal_feasibility={marg} "
                            f"--corr_feasibility={corr} "
                            f"--algorithm={algorithm.name} "
                            f"--num_samples={10_000} "
                            f"--heuristic={heur} "
                            f"--andor_graph_analysis "
                            f"--output_dir={str(curr_output_dir)} "
                            f"--num_samples_alg={num_samples} "
                            "--rfb_early_termination_prob=1.1 "  # i.e. never terminate early
                        )

                        # Append a re-direct of stdout and std err to log file
                        log_file_path = curr_output_dir / f"results_{batch_start:05d}-{batch_end:05d}.log"
                        command += f"> {str(log_file_path)} 2>&1"

                        # print the command
                        print(command)
