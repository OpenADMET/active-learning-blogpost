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
active_learning_blog.ipynb        # Main notebook (the blogpost)
helpers.py                        # Core AL utilities and module-level constants
plots.py                          # All Plotly/Faerun plotting functions
results/                          # HTML outputs (tmap_selection.html, GTM animation HTMLs)
```

There is no separate `conf.py`. The constants `STRATEGIES`, `STRATEGY_COLORS`, and `STRATEGY_QUERY_KEYS` are defined at the top of `helpers.py` and imported directly from there.

## Key libraries and frameworks

- **`openadmet-models`** — primary ML framework; provides `CommitteeRegressor`, `ChemPropModel`, `ScaffoldSplitter`, `LightningTrainer`, `ChemPropFeaturizer`, `RegressionMetrics`, `UncertaintyMetrics`. Install from [openadmet-models on GitHub](https://github.com/OpenADMET/openadmet-models).
- **`chemographykit`** — provides `GTM` (Generative Topographic Map) and `calculate_latent_coords` for 2D chemical space embedding
- **`tmap`** + **`mhfp`** — TMAP layout using MHFP fingerprints and an LSH forest; produces minimum-spanning-tree visualizations
- **`faerun`** — interactive HTML scatter plots layered on TMAP layouts
- **`useful_rdkit_utils`** (`uru`) — RDKit descriptor calculation and scaling (`RDKitDescriptors`, `clean_and_scale_descriptors`)
- **`uncertainty_toolbox`** (`uct`) — uncertainty calibration (isotonic regression) and calibration curve computation
- **`lightning` (PyTorch Lightning)** — model training backend
- **`chemprop` / CheMeleon** — MPNN backbone for property prediction
- **`rdkit`** — molecular parsing
- **`pandas`**, **`numpy`**, **`scipy`** — data manipulation and statistics
- **`matplotlib`**, **`plotly`** — visualization

## Molecular representations

- Molecules are stored as canonical SMILES strings
- Raw dataset column: `OPENADMET_CANONICAL_SMILES`; renamed to `smiles` after splitting
- Models use `ChemPropFeaturizer` internally (graph-based, no explicit fingerprints)
- GTM embedding uses RDKit physicochemical descriptors (scaled via `useful_rdkit_utils`)
- TMAP layout uses MHFP (MinHash fingerprints) encoded with `MHFPEncoder` and indexed with an LSH forest
- Data is loaded from Parquet files; scaffold splitting (`ScaffoldSplitter`) is preferred over random splits

## Active learning architecture

- **Committee**: `N_MODELS=5` independent `ChemPropModel` instances trained on bootstrapped labeled pools
- **Ensemble diversity**: deep ensembling (different random seeds) + bootstrap bagging
- **Acquisition**: `query_batch(committee, smiles_unlabeled, strategy, best_y, size, ...)` returns indices of the next `M_QUERY=20` molecules to label
- **Loop**: `run_active_learning(df_pool, df_test, n_start, k_iter, seed, strategy)` runs `K_ITER=15` iterations starting from `N_START=100` labeled molecules
- **Calibration**: isotonic regression post-hoc calibration on a held-out calibration set (`df_cal`); visualized with `plot_calibration_curve_before_after`

## Helper functions (`helpers.py`)

| Function | Purpose |
|---|---|
| `smiles_to_gtm(smiles_list, ...)` | Fit a GTM and return `(gtm, crds_2d, resps, llhs)` |
| `smiles_to_tmap(smiles_list, ...)` | Compute TMAP layout; return `(x, y, s, t)` |
| `split_data(X, y, ...)` | Scaffold split into pool / cal / test |
| `featurize(smiles_list, y_list, shuffle)` | Return `(loader, scaler)` for ChemProp |
| `build_committee_member(seed, max_epochs, log_dir)` | Return `(model, trainer)` |
| `train_committee(smiles_labeled, y_labeled, ...)` | Train bootstrapped ensemble; return `CommitteeRegressor` |
| `query_batch(committee, smiles_unlabeled, strategy, best_y, size, ...)` | Select next batch; return `(indices, scores)` |
| `evaluate_on_test(committee, smiles_test, y_test)` | Return dict of MAE, R², Kendall τ, Spearman ρ, miscalibration area |
| `run_active_learning(df_pool, df_test, ...)` | Full AL loop; return `{"history": [...], "committee": ...}` |

## Plotting functions (`plots.py`)

| Function | Output |
|---|---|
| `plot_learning_curve_with_bands(...)` | Plotly line plot with ±1σ shading per strategy |
| `plot_hit_discovery_curve(...)` | Cumulative hits found vs. labeled pool size |
| `plot_calibration_curve_before_after(...)` | Before/after isotonic calibration curves |
| `plot_gtm_selection_animation(...)` | Animated Plotly scatter on GTM embedding |
| `plot_tmap_faerun(...)` | Interactive Faerun/TMAP HTML scatter; writes file and returns `Faerun` instance |

## Coding conventions

- Python 3.10+, type hints encouraged for function signatures
- All helper functions live in `helpers.py` at the repo root; all plot functions in `plots.py`
- Constants (`STRATEGIES`, `STRATEGY_COLORS`, `STRATEGY_QUERY_KEYS`) are defined in `helpers.py` — import them from there, not from a separate config file
- Keep notebook cells focused: configuration → data loading → GTM/TMAP embedding → AL loop → results unpacking → visualization → calibration
- Use `STRATEGY_COLORS` for consistent strategy colors across all plots
- Prefer scaffold-based splits over random splits when evaluating generalization
- Docstrings follow NumPy style

## Data conventions

- Activity column: `PXR_pEC50` (raw DataFrame), renamed to `pEC50` after splitting
- SMILES column: `OPENADMET_CANONICAL_SMILES` (raw), renamed to `smiles` after splitting
- Main dataset loaded from a Parquet file via `pd.read_parquet()`
- Hit threshold: `pEC50 >= 7.0` is considered a hit
