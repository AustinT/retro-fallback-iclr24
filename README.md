# Retro-fallback (ICLR 2024)

[![Paper](http://img.shields.io/badge/paper-arxiv.2310.09270-B31B1B.svg)](https://arxiv.org/abs/2310.09270)
[![Conference](http://img.shields.io/badge/ICLR-2024-4b44ce.svg)](https://openreview.net/forum?id=dl0u4ODCuW)
![GitHub](https://img.shields.io/github/license/AustinT/retro-fallback-iclr24)

This is the official code to accompany the paper introducing retro-fallback:
an algorithm for retrosynthetic planning with uncertainty.
It contains a minimal python implementation of the methods
described in the paper, code to reproduce the experimental results,
all numerical results presented in the paper,
and code to re-create the plots in the paper.

## Code overview

The layout of this repository is as follows:

- `retro_fallback_iclr24/`: a minimal python package implementing retro-fallback and all associated metrics.
  This package is further segmented into two parts: methods specific to our ICLR experiments (see `retro_fallback_iclr24.iclr24_experiments`),
  and general code for retro-fallback and SSP (everything else).
- `tests/`: tests for our code (can be run with `pytest`)
- `misc/`: contains some data
- `official-results/`: contains all data for our experiment (in `.tar.gz` format)
- `scripts/`: all scripts associated with our code

## Installation

The code can be installed with pip by running

```bash
pip install .
```

This should install all dependencies.
If this does not work for you, you may need to install pytorch or other packages manually
using another method (e.g. conda),
then trying the pip installation again.

### Preparing datasets

The experiments run for this paper require downloading/preprocessing a small number of datasets.
Instructions for each dataset are in `README.md` files in the following directories:

- `retro_fallback_iclr24/iclr24_experiments/eMolecules`
- `retro_fallback_iclr24/iclr24_experiments/fusion_retro`
- `retro_fallback_iclr24/iclr24_experiments/guacamol` (NOTE: recreating dataset from scratch requires `eMolecules` inventory)
- `retro_fallback_iclr24/iclr24_experiments/retrostar190` (NOTE: this does not actually require you to run anything)

## Instructions to reproduce results

Our code is intended to be reproducible at multiple levels,
from re-making the plots in the paper to re-running our entire set of experiments.
Here we give instructions for each of these tasks.

### Reproducing the plots (most basic)

This does not require any special installation (with the exception of the reaction kernel plots).
First, extract our official data by running the following commands.

```bash
# Results for "comparison" experiments
for tarfile in official-results/comparison/*.tar.gz ; do
    tar -xzvf $tarfile -C results-comparison/
done

# Results for "variation" experiments
for tarfile in official-results/variation/*.tar.gz ; do
    tar -xzvf $tarfile -C results-variation/
done
```

NOTE: if you have run your own experiments this will overwrite the outputs from those experiments,
so be careful.

After that, all plots can be reproduced by running the script:

```bash
bash make-plots.sh
```

This requires the `tueplots` package to be installed, but otherwise does not require any ML libraries
(except for the final script to make reaction kernel plots, which are not central results).

### Re-running experiments

The results in this paper are based on a large number of experiments.
The commands to run these experiments can be printed by running:

```python
# comparison experiments
# (first argument is the directory for the experiments to output to; change if desired)
python scripts/print_comparison_experiment_commands.py results-comparison

# variation experiments
python scripts/print_variability_experiment_commands.py results-variation
```

Note that running these scripts will also create the directories for the results to be written to.
The commands can be run in any way you see fit.
The `xargs` program might be useful for this, e.g.:

```bash
python scripts/print_comparison_experiment_commands.py results-comparison | xargs -I {} -P 2 bash -c {}
```

This will run all commands, 2 in parallel (specified by `-P 2`).

After running the experiments,
it should produce similar results to those contained in the
`official-results` directory.
Plots can then be generated using the commands above.

## Development

This repository will not be under active development,
since its purpose is to reproduce the results presented in the paper.
However, if you want to make a PR,
please install `pre-commit` and run tests with `pytest`
to check that your PR does not break anything.
