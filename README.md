# Retro-fallback (ICLR 2024)

This is the official code to accompany the paper introducing retro-fallback.

## TODO

- installation instructions (including syntheseus retro star package)
- describe code structure (e.g. separation into "general" and "experiment-specific" code.)
- running / reproducting results
- development
- paper link
- add badges to README.
- in code, reference key equations in paper (e.g. for s, psi, rho, alpha)
- check spelling in comments
- describe instructions for setting up datasets (eMolecules, guacamol, etc)

## Datasets

The experiments run for this paper require downloading/preprocessing a small number of datasets.
Instructions for each dataset are in `README.md` files in the following directories:

- `retro_fallback_iclr24/iclr24_experiments/eMolecules`
- `retro_fallback_iclr24/iclr24_experiments/fusion_retro`
- `retro_fallback_iclr24/iclr24_experiments/guacamol` (NOTE: recreating dataset from scratch requires `eMolecules` inventory)
- `retro_fallback_iclr24/iclr24_experiments/retrostar190` (NOTE: this does not actually require you to run anything)

## Development

Please install `pre-commit` and run tests with `pytest`.
