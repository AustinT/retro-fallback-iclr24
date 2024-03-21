from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from pprint import pformat

from syntheseus import Molecule
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch,
    MolSet_BreadthFirstSearch,
)
from syntheseus.search.algorithms.mcts.base import pucb_bound
from syntheseus.search.algorithms.mcts.molset import MolSetGraph, MolSetMCTS
from syntheseus.search.graph.standardization import get_unique_node_andor_graph
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus_retro_star_benchmark import RetroStarReactionModel
from tqdm import tqdm

from retro_fallback_iclr24.iclr24_experiments.analyze_results import analyze_output_graph, analyze_output_graph_molset
from retro_fallback_iclr24.iclr24_experiments.inventories import (
    FusionRetroInventory,
    eMoleculesInventory,
    eMoleculesTieredBuyabilityModel,
)
from retro_fallback_iclr24.iclr24_experiments.latent_gp_feasibility import (
    Constant_Mean_Latent_GP_Feasibility,
    Rank_Mean_Latent_GP_Feasibility,
    gp_feas_logger,
)
from retro_fallback_iclr24.iclr24_experiments.mcts_stuff import (
    MCTS_Marginal_Feasibility_Policy,
    MCTS_Optimistic_ValueFunction,
    MCTS_SAscore_ValueFunction,
    MCTS_SuccessReward,
)
from retro_fallback_iclr24.iclr24_experiments.retro_star_stuff import (
    RetroStarMarginalBuyabilityCost,
    RetroStarMarginalFeasibilityCost,
)
from retro_fallback_iclr24.iclr24_experiments.sascore_heuristic import SAScoreCostHeuristic, SAScoreProbabilityHeuristic
from retro_fallback_iclr24.retro_fallback import RetroFallbackSearch
from retro_fallback_iclr24.stochastic_processes import BinaryBuyability
from retro_fallback_iclr24.stochastic_processes.feasibility import (
    ConstantIndependentFeasibility,
    RankIndependentFeasibility,
)

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--smiles_start_idx",
        type=int,
        default=None,
        help="Index of first SMILES to use.",
    )
    parser.add_argument(
        "--smiles_end_idx",
        type=int,
        default=None,
        help="Index of last SMILES to use (exclusive).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to output jsons to. Outputs will be named result_X.json.",
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=500,
        help="Allowed number of calls to reaction model.",
    )
    parser.add_argument(
        "--marginal_feasibility",
        type=str,
        default="constant",
        help="Type of feasibility model to use.",
    )
    parser.add_argument(
        "--corr_feasibility",
        type=str,
        default="independent",
        help="Type of feasibility model to use.",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="eMolecules",
        help="Which inventory/buyability model to use.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10_000,
        help="Number of samples to use for feasibility/buyability model (and success calculation).",
    )
    parser.add_argument(
        "--num_samples_alg",
        type=int,
        default=None,
        help="Number of samples to use during search (if different).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm to use.",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        default="optimistic",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Run algorithm on a tree instead of a graph.",
    )
    parser.add_argument(
        "--mcts_time_limit_s",
        type=float,
        default=1e4,
        help="Time limit for MCTS.",
    )
    parser.add_argument(
        "--precursor_matching_file",
        type=str,
        default=None,
        help="Optional JSON file containing ground truth start molecules. Used for start mol matching metric.",
    )
    parser.add_argument(
        "--rfb_early_termination_prob",
        type=float,
        default=0.99,
        help="Success probability value after which retro-fallback will terminate early.",
    )
    parser.add_argument(
        "--andor_graph_analysis",
        action="store_true",
        help="For OR graphs or algorithms using trees, convert to an AND/OR graph when performing analysis.",
    )
    parser.add_argument(
        "--skip_most_feasible_route",
        action="store_true",
        help="Flag to skip calculating the most feasible route (sometimes expensive).",
    )
    return parser


def run_search_and_analyze_results():
    logger.info("START OF SCRIPT")

    # Get arguments
    args = get_parser().parse_args()
    logger.info(args)

    # Ensure output dir exists
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load SMILES to test
    with open(args.smiles_file, "r") as f:
        test_smiles = [line.strip() for line in f.readlines()]
    if args.smiles_start_idx is None:
        smiles_idx_offset = 0
    else:
        smiles_idx_offset = args.smiles_start_idx
        test_smiles = test_smiles[args.smiles_start_idx : args.smiles_end_idx]
    logger.info(f"Loaded {len(test_smiles)} SMILES to test.")

    # Load ground truth start molecules
    smiles_to_start_mols = dict()
    if args.precursor_matching_file is not None:
        with open(args.precursor_matching_file, "r") as f:
            smiles_to_start_mols.update(json.load(f))
        logger.info(f"Loaded {len(smiles_to_start_mols)} ground truth start molecules.")

    # Make reaction model, and feasibility model
    rxn_model = RetroStarReactionModel(use_cache=True, default_num_results=50)
    _use_cache_covar = True  # common arg to all latent GP models
    if args.marginal_feasibility == "constant":
        _kwargs = dict(constant=0.5, num_samples=args.num_samples)
        if args.corr_feasibility == "independent":
            feasibility_model = ConstantIndependentFeasibility(**_kwargs)
        elif args.corr_feasibility == "gp":
            feasibility_model = Constant_Mean_Latent_GP_Feasibility(use_cached_covar_matrix=_use_cache_covar, **_kwargs)
        else:
            raise ValueError(f"Invalid corr_feasibility: {args.corr_feasibility}")
        del _kwargs
    elif args.marginal_feasibility == "rank":
        _kwargs = dict(constant=0.75, rank_div=10, num_samples=args.num_samples)
        if args.corr_feasibility == "independent":
            feasibility_model = RankIndependentFeasibility(**_kwargs)
        elif args.corr_feasibility == "gp":
            feasibility_model = Rank_Mean_Latent_GP_Feasibility(use_cached_covar_matrix=_use_cache_covar, **_kwargs)
        else:
            raise ValueError(f"Invalid corr_feasibility: {args.corr_feasibility}")
        del _kwargs
    else:
        raise NotImplementedError(f"Invalid marginal_feasibility: {args.marginal_feasibility}")

    # Make inventory and buyability model
    inventory_tokens = args.inventory.split("-")  # e.g. "eMolecules-tiered"
    if inventory_tokens[0] == "eMolecules":
        inventory = eMoleculesInventory(max_tier=2)
    elif inventory_tokens[0] == "fusion-retro":
        inventory = FusionRetroInventory()
    else:
        raise ValueError(f"Invalid inventory: {args.inventory}")
    if len(inventory_tokens) == 1:  # default is binary buyability
        buyability_model = BinaryBuyability(num_samples=args.num_samples)
    elif inventory_tokens[1] == "stochastic":
        buyability_model = eMoleculesTieredBuyabilityModel(num_samples=args.num_samples)
        assert isinstance(inventory, eMoleculesInventory), "Need to use this with eMolecules inventory!"
    else:
        raise ValueError(f"Invalid inventory: {args.inventory}")
    del inventory_tokens

    # Make the algorithm object
    high_integer = int(1e10)
    common_alg_kwargs = dict(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=args.limit_rxn_model_calls,
        stop_on_first_solution=False,
    )
    retro_star_like_common_alg_kwargs = dict(
        limit_iterations=high_integer,
        max_expansion_depth=30,
        prevent_repeat_mol_in_trees=args.tree,
        unique_nodes=not args.tree,
    )
    if args.algorithm == "retro-fallback":
        if args.heuristic == "optimistic":
            value_function = ConstantNodeEvaluator(1.0)
        elif args.heuristic == "sascore":
            value_function = SAScoreProbabilityHeuristic()
        else:
            raise ValueError(f"Invalid heuristic: {args.heuristic}")
        alg = RetroFallbackSearch(
            feasibility_model=feasibility_model,
            buyability_model=buyability_model,
            value_function=value_function,
            early_stopping_SSP=args.rfb_early_termination_prob,
            **common_alg_kwargs,
            **retro_star_like_common_alg_kwargs,
        )
    elif args.algorithm == "retro-star":
        if args.heuristic == "optimistic":
            value_function = ConstantNodeEvaluator(0.0)
        elif args.heuristic == "sascore":
            value_function = SAScoreCostHeuristic()
        else:
            raise ValueError(f"Invalid heuristic: {args.heuristic}")
        alg = RetroStarSearch(
            and_node_cost_fn=RetroStarMarginalFeasibilityCost(feasibility_model=feasibility_model),
            or_node_cost_fn=RetroStarMarginalBuyabilityCost(buyability_model=buyability_model),
            value_function=value_function,
            **common_alg_kwargs,
            **retro_star_like_common_alg_kwargs,
        )
    elif args.algorithm == "mcts":
        if args.heuristic == "optimistic":
            value_function = MCTS_Optimistic_ValueFunction(feasibility_model, buyability_model)
        elif args.heuristic == "sascore":
            value_function = MCTS_SAscore_ValueFunction(feasibility_model, buyability_model)
        else:
            raise ValueError(f"Invalid heuristic: {args.heuristic}")
        alg = MolSetMCTS(
            value_function=value_function,
            reward_function=MCTS_SuccessReward(feasibility_model, buyability_model),
            policy=MCTS_Marginal_Feasibility_Policy(feasibility_model),
            limit_iterations=high_integer,
            time_limit_s=args.mcts_time_limit_s,
            max_expansion_depth=30,
            prevent_repeat_mol_in_trees=True,
            unique_nodes=False,
            bound_constant=0.01,
            bound_function=pucb_bound,
            min_num_visit_to_expand=10,  # gather a lot of information before expanding
            **common_alg_kwargs,
        )
        alg.init_mcts_value = 1.0  # optimistic start so that all nodes are visited

        # Turn this down (otherwise it creates a lot of useless messages)
        gp_feas_logger.setLevel(logging.INFO)
    elif args.algorithm == "breadth-first":
        alg = AndOr_BreadthFirstSearch(
            unique_nodes=True,
            limit_iterations=high_integer,
            **common_alg_kwargs,
        )
    elif args.algorithm == "ORbreadth-first":
        alg = MolSet_BreadthFirstSearch(
            unique_nodes=False,  # OR graphs not fully supported
            limit_iterations=high_integer,
            **common_alg_kwargs,
        )
    else:
        raise ValueError(f"Invalid algorithm: {args.algorithm}")

    # Run the search
    analysis_times = [t for t in [1, 5, 10, 25, 50, 100, 150, 200, 250, 500, 1000] if t <= args.limit_rxn_model_calls]
    for i, smiles in enumerate(tqdm(test_smiles, dynamic_ncols=True, smoothing=0.0)):
        logger.info(f"Start search {i}/{len(test_smiles)}. SMILES: {smiles}")

        # Reset sample count in cast it changed
        num_samples_for_seach = args.num_samples_alg or args.num_samples
        feasibility_model.num_samples = num_samples_for_seach
        buyability_model.num_samples = num_samples_for_seach

        # Reset and run search
        t0 = time.monotonic()
        alg.reset()  # Clear cache to give fair comparison
        output_graph, _ = alg.run_from_mol(Molecule(smiles))
        t1 = time.monotonic()
        logger.debug(f"Search done in {t1 - t0:.2f} s. Starting analysis.")

        # Reset feasibility models for eval
        feasibility_model.num_samples = args.num_samples
        buyability_model.num_samples = args.num_samples

        # Potential convert to AND/OR graph for analysis
        if args.andor_graph_analysis and (isinstance(output_graph, MolSetGraph) or args.tree):
            graph_to_analyze = get_unique_node_andor_graph(output_graph)
            logger.debug(
                f"Converted OR graph to AND/OR graph for analysis. Resulting graph has {len(graph_to_analyze)} nodes."
            )
        else:
            graph_to_analyze = output_graph

        # Analyze output graph
        if isinstance(graph_to_analyze, MolSetGraph):
            analysis_results = analyze_output_graph_molset(
                graph_to_analyze, feasibility_model, buyability_model, analysis_times
            )
        else:
            analysis_results = analyze_output_graph(
                graph_to_analyze,
                feasibility_model,
                buyability_model,
                analysis_times,
                calculate_most_feasible_route=not args.skip_most_feasible_route,
                smiles_to_start_mols=smiles_to_start_mols,
            )
        t2 = time.monotonic()

        # Store search results
        search_results = dict(
            smiles=smiles,
            args=args.__dict__,
            final_num_nodes=len(output_graph),
            total_search_time=t1 - t0,
            total_analysis_time=t2 - t1,
            analysis_times=analysis_times,
            **analysis_results,
        )
        with open(output_dir_path / f"result_{i + 1 + smiles_idx_offset:06d}.json", "wt") as f:
            json.dump(search_results, f, indent=2)

        logger.info(f"Done search + analysis in {t2 - t0:.2f} s. Results:\n{pformat(search_results)}")

        # Free up memory for the next search
        del output_graph

    logger.info("END OF SCRIPT.")
