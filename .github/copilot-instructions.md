# GitHub Copilot Instructions

## Project overview

This repository is a tutorial/blogpost on **active learning for pEC50 prediction** in drug discovery, focused on PXR (pregnane X receptor) inhibition. The main artifact is `active_learning_blog.ipynb`, a self-contained Jupyter notebook that walks through a full active learning benchmark using real experimental data.

## Domain context

- **Problem domain**: QSAR / cheminformatics / computational drug discovery
- **Target**: PXR — a promiscuous nuclear receptor; predicting pEC50 (–log₁₀ EC50)
- **Core concept**: Active learning with query-by-committee to reduce the number of expensive wet-lab assays needed to train a useful model
- **Acquisition strategies compared**: EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity

## Repository structure

```
active_learning_blog.ipynb   # Main notebook (the blogpost)
helpers.py                 # Core AL utilities: featurize, train_committee, run_active_learning, evaluate_on_test, smiles_to_ecfp4, smiles_to_gtm
conf.py                    # Global constants: STRATEGIES, STRATEGY_COLORS, STRATEGY_QUERY_KEYS
plots.py                   # All matplotlib/plotly plotting functions
```

## Key libraries and frameworks

- **`openadmet-models`** — primary ML framework; provides `CommitteeRegressor`, `ChemPropModel`, `ScaffoldSplitter`, `LightningTrainer`, `ChemPropFeaturizer`, calibration utilities. Install from [openadmet-models on GitHub](https://github.com/OpenADMET/openadmet-models).
- **`rdkit`** — molecular parsing and ECFP4 (Morgan) fingerprint generation
- **`ugtm`** — Generative Topographic Maps for 2-D chemical space embedding (Diversity strategy)
- **`lightning` (PyTorch Lightning)** — model training backend
- **`chemprop` / CheMeleon** — message-passing neural network backbone for property prediction
- **`pandas`**, **`numpy`**, **`scipy`** — data manipulation and statistics
- **`matplotlib`**, **`seaborn`**, **`plotly`** — visualization

## Molecular representations

- Molecules are stored as canonical SMILES strings (column `OPENADMET_CANONICAL_SMILES` or `smiles`)
- Models use `ChemPropFeaturizer` internally; fingerprints (`smiles_to_ecfp4`) are used for GTM embedding
- Data is loaded from Parquet files; scaffold splitting (`ScaffoldSplitter`) is preferred over random splits

## Active learning architecture

- **Committee**: `N_MODELS=5` independent `ChemPropModel` instances trained on bootstrapped labeled pools
- **Ensemble diversity**: deep ensembling (different random seeds) + bagging
- **Acquisition**: `query_batch(strategy, committee, X_unlabeled)` returns indices of next `M_QUERY` molecules to label
- **Loop**: `run_active_learning(strategy, seed, df_pool, df_test)` runs `K_ITER` iterations starting from `N_START` labeled molecules
- **Calibration**: isotonic regression post-hoc calibration on a held-out calibration set

## Coding conventions

- Python 3.10+, type hints encouraged for function signatures
- Helper functions live in `scripts/helpers.py`; plotting functions in `scripts/plots.py`; constants in `scripts/conf.py`
- Keep notebook cells focused: configuration → data loading → model training → results → visualization
- Intermediate results are checkpointed to `all_runs.pkl` to avoid re-running expensive training
- Use `STRATEGY_COLORS` from `conf.py` for consistent strategy colors across all plots
- Prefer `scaffold`-based splits over random splits when evaluating generalization

## Data conventions

- Activity column: `PXR_pEC50` (raw DataFrame), renamed to `pEC50` after splitting
- SMILES column: `OPENADMET_CANONICAL_SMILES` (raw), renamed to `smiles` after splitting
- Datasets are stored as Parquet files; use `pd.read_parquet()`
- Hit threshold: `pEC50 >= 7.0` is considered a "hit"
