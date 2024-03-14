"""Print commands to run all experiments for this paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetExperimentParams:
    """Parameters for a single experiment."""

    name: str
    smiles_file: str
    num_replications: int
    search_length: int
    feasibility_models: list[str]
    max_idx: int
    batch_size: int
    inventory: str
    precursor_matching_file: Optional[str] = None


ALL_FEASIBILITY_MODELS = [
    "constant-independent",
    "constant-gp",
    "rank-independent",
    "rank-gp",
]

SMILES_FILE_PREFIX = "./retro_fallback_iclr24/iclr24_experiments"


datasets = [
    DatasetExperimentParams(
        name="retrostar190",
        smiles_file=f"{SMILES_FILE_PREFIX}/retrostar190/retrostar190.smiles",
        num_replications=5,
        search_length=501,
        feasibility_models=ALL_FEASIBILITY_MODELS,
        max_idx=190,
        batch_size=10,
        inventory="eMolecules",
    ),
    DatasetExperimentParams(
        name="guacamol",
        smiles_file=f"{SMILES_FILE_PREFIX}/guacamol/test-all.smiles",
        num_replications=2,
        search_length=201,
        feasibility_models=ALL_FEASIBILITY_MODELS,
        max_idx=500,
        batch_size=25,
        inventory="eMolecules",
    ),
    DatasetExperimentParams(
        name="fusionretro",
        smiles_file=f"{SMILES_FILE_PREFIX}/fusion_retro/test_dataset.smiles",
        num_replications=1,
        search_length=51,
        feasibility_models=["rank-independent"],
        max_idx=5838,
        batch_size=100,
        inventory="fusion-retro",
        precursor_matching_file=f"{SMILES_FILE_PREFIX}/fusion_retro/test_dataset.json",
    ),
]


@dataclass
class AlgorithmParams:
    """Parameters for each algorithm."""

    name: str
    heuristics_to_test: list[str]
    num_samples: Optional[int] = None
    tree: bool = False


STANDARD_HEURISTICS = ["optimistic", "sascore"]

algorithms = [
    AlgorithmParams(name="retro-fallback", heuristics_to_test=STANDARD_HEURISTICS, num_samples=1_000),
    AlgorithmParams(name="retro-star", heuristics_to_test=STANDARD_HEURISTICS, tree=True),
    AlgorithmParams(name="mcts", heuristics_to_test=STANDARD_HEURISTICS, tree=True),
    AlgorithmParams(name="breadth-first", heuristics_to_test=["NONE"]),
]


PREFIX_ENV_VARS = (
    'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
    "MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 "
    "PYTHONPATH=.:$PYTHONPATH "
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Root output directory")
    args = parser.parse_args()

    # Loop over datasets
    for dataset in datasets:

        # Loop over trials
        for trial in range(dataset.num_replications):

            # Loop over batches within this dataset
            for batch_start in range(0, dataset.max_idx, dataset.batch_size):
                batch_end = batch_start + dataset.batch_size

                # Loop over feasibility models
                for feas_model in dataset.feasibility_models:
                    marg, corr = feas_model.split("-")

                    # Loop over algorithms
                    for alg in algorithms:

                        # Loop over its heuristics
                        for heur in alg.heuristics_to_test:

                            # Produce current output dir
                            curr_output_dir = (
                                Path(args.output_dir) / dataset.name / feas_model / alg.name / heur / f"trial_{trial+1}"
                            )
                            curr_output_dir.mkdir(parents=True, exist_ok=True)

                            # Standard command for all algortthms
                            command = (
                                PREFIX_ENV_VARS + "python scripts/main.py "
                                f"--smiles_file={dataset.smiles_file} "
                                f"--smiles_start_idx={batch_start} "
                                f"--smiles_end_idx={batch_end} "
                                f"--limit_rxn_model_calls={dataset.search_length} "
                                f"--marginal_feasibility={marg} "
                                f"--corr_feasibility={corr} "
                                f"--algorithm={alg.name} "
                                f"--num_samples={10_000} "
                                f"--heuristic={heur} "
                                f"--andor_graph_analysis "
                                f"--output_dir={str(curr_output_dir)} "
                            )

                            # Optionally overwrite num_samples
                            if alg.num_samples:
                                command += f"--num_samples_alg={alg.num_samples} "

                            # Optionally add arg to run on a tree
                            if alg.tree:
                                command += "--tree "

                            # Optionally add start mol matching file
                            if dataset.precursor_matching_file:
                                command += f"--precursor_matching_file={dataset.precursor_matching_file} "

                            # Append a re-direct of stdout and std err to log file
                            log_file_path = curr_output_dir / f"results_{batch_start:05d}-{batch_end:05d}.log"
                            command += f"> {str(log_file_path)} 2>&1"

                            # print the command
                            print(command)
