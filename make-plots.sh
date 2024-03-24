# Make all plots for the paper

# Make sure the top-level plots directory exists
plot_dir="./plots"
mkdir -p "${plot_dir}"


# Make comparison plots
for dataset in "retrostar190" "guacamol" "retrostar190-stochastic-inv" ; do

    # Make the output dir
    output_dir="${plot_dir}/comparison/${dataset}"
    mkdir -p "${output_dir}"

    python scripts/plot_comparison.py \
        --results_dir "results-comparison/${dataset}" \
        --output_dir "${output_dir}" \
        --save_fmt="pdf" \
       --individual_smiles

done

# Make variability plots
for dataset in "guacamol" ; do

    # Make the output dir
    output_dir="${plot_dir}/variation/${dataset}"
    mkdir -p "${output_dir}"

    python scripts/plot_variability.py \
        --results_dir "results-variation/${dataset}" \
        --output_dir "${output_dir}" \
        --save_fmt="pdf" \

done

# FusionRetro table
python scripts/fusionretro_experiment_table.py \
    --results_dir results-comparison/fusionretro \
    --output_dir plots/comparison/


# Plots for reaction kernel
# NOTE: this requires syntheseus / etc to be installed in this environment
output_dir="${plot_dir}/reaction-kernel"
mkdir -p "${output_dir}"
PYTHONPATH=.:$PYTHONPATH python scripts/plot_rxn_kernel.py \
    misc/reactions-from-backward-model.txt.gz \
    --output_dir "${output_dir}"
