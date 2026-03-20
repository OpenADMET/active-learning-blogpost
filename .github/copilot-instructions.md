# GitHub Copilot Instructions

## Project overview

This repository is a tutorial/blogpost on **active learning for pEC50 prediction** in drug discovery, focused on PXR (pregnane X receptor) inhibition. The primary artifact is `blogpost.md`, a prose document that narrates the full active learning benchmark. All executable code lives in `src/`, with two entry points at the repository root.

## Domain context

- **Problem domain**: QSAR / cheminformatics / computational drug discovery
- **Target**: PXR — a promiscuous nuclear receptor; predicting pEC50 (–log₁₀ EC50)
- **Core concept**: Active learning with query-by-committee to reduce the number of expensive wet-lab assays needed to train a useful model
- **Acquisition strategies compared**: EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity

## Repository structure

```
blogpost.md                       # Primary document (the blogpost, prose + figure links)
run.py                            # Entry point 1: run the AL pipeline → results/setup.pkl + results/run_*.pkl
analysis.py                       # Entry point 2: consume results/*.pkl → all HTML/PNG figures
config.yaml                       # Experiment configuration (data paths, AL params, training params)
src/
    __init__.py
    config.py                     # ALConfig dataclass + load_config() — all config parsing/validation
    helpers.py                    # Core AL utilities and module-level constants
    plots.py                      # All Plotly/Faerun plotting functions
_data/
    octant_screening_compounds.csv  # Background compound library for visualisations
results/                          # Generated artifacts (HTML/PNG figures, pkl checkpoints)
```

### Entry points

- **`run.py`** — loads the dataset, scaffold-splits, fits the GTM embedding, and saves `results/setup.pkl`. Then runs `run_active_learning` for each requested `(strategy, seed)` pair, saving one `results/run_<STRATEGY>_seed<N>.pkl` per job. Skips jobs that already have output files (safe to resume). Designed for HPC: run `--setup-only` once, then dispatch one `--strategy S --seed N` job per node.
- **`analysis.py`** — loads `results/setup.pkl` and all `results/run_*.pkl` files, performs sanity checks on coverage and consistency, unpacks results into tidy DataFrames, and writes all interactive HTML + PNG figures to `results/`. Does not require access to the original dataset.

### Configuration

All experiment parameters live in `config.yaml` and are loaded into an `ALConfig` dataclass via `load_config()` in `src/config.py`. The config is frozen into `setup.pkl` at setup time so every downstream job uses an identical configuration.

`config.yaml` has three sections:

```yaml
data:
  dataset_path:          # Path to main labelled dataset (Parquet); tilde-expanded
  background_path:       # Path to background compound library (CSV/Parquet) for visualisations only
  background_smiles_col: # Column name for SMILES in the background file
  seed_data_path:        # Optional external pretraining dataset path (null = skip)
  seed_smiles_col:       # SMILES column in seed data file
  seed_activity_col:     # Activity column in seed data file

active_learning:
  strategies:   # List of strategies to run (subset of valid universe)
  seeds:        # List of outer random seeds (one HPC job per strategy×seed)
  k_iter:       # AL iterations per run
  query_size:   # Compounds queried per iteration
  n_start:      # Initial labeled pool size (0 when using seed_data_path)

training:
  n_models:     # Committee size (bootstrapped ensemble members)
  max_epochs:   # Max training epochs per committee member
```

The only CLI flags on `run.py` are `--setup-only`, `--config PATH`, `--strategy`, and `--seed`. All data paths and AL hyperparameters are in `config.yaml`.

### Checkpoint formats

`results/setup.pkl` keys: `df_pool`, `df_test`, `df_seed`, `gtm_coords_pool`, `gtm_coords_background`, `background_smiles`, `config` (an `ALConfig` instance).

`results/run_<STRATEGY>_seed<N>.pkl` keys: `strategy`, `seed`, `n_start`, `result` (contains `history` list and `committee`).

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
- **`PyYAML`** — config parsing in `src/config.py`
- **`kaleido`** — PNG export for Plotly figures in `analysis.py`

## Molecular representations

- Molecules are stored as canonical SMILES strings
- Raw dataset column: `OPENADMET_CANONICAL_SMILES`; renamed to `smiles` after splitting
- Models use `ChemPropFeaturizer` internally (graph-based, no explicit fingerprints)
- GTM embedding uses RDKit physicochemical descriptors (scaled via `useful_rdkit_utils`)
- TMAP layout uses MHFP (MinHash fingerprints) encoded with `MHFPEncoder` and indexed with an LSH forest
- Data is loaded from Parquet files; scaffold splitting (`ScaffoldSplitter`) is preferred over random splits

## Active learning architecture

- **Committee**: `n_models` (default 5) independent `ChemPropModel` instances trained on bootstrapped labeled pools; size and epochs set via `config.yaml`
- **Ensemble diversity**: deep ensembling (different random seeds per member) + bootstrap bagging; member seed = `seed+i`, iteration seed = `seed+k`
- **Acquisition**: `query_batch(committee, smiles_unlabeled, strategy, best_y, size=query_size, seed=seed+k)` returns `(indices, scores)`
- **Loop**: `run_active_learning(df_pool, df_test, n_start, k_iter, query_size, n_models, max_epochs, seed, strategy, df_seed, gtm_coords)` runs `k_iter` AL iterations from an initial labeled pool of size `n_start`
- **Calibration**: at each iteration, 10% of the current labeled pool (pool-acquired compounds only) is held out for isotonic regression calibration; pre- and post-calibration metrics are both recorded in the history
- **GTM visualization**: computed once in `run_setup` on `pool + background` molecules; coordinates sliced and passed as `gtm_coords` to avoid redundant refitting per job
- **Test loader**: pre-computed once before the loop and reused across iterations for efficiency

## `src/config.py`

- `ALConfig` — frozen dataclass with all 13 fields; picklable; stored in `setup.pkl`
- `load_config(path="config.yaml") -> ALConfig` — reads YAML, validates all fields with batched error reporting, returns `ALConfig`
- `_VALID_STRATEGIES` — strategy universe defined here (not imported from `helpers.py`) to avoid circular imports

## Helper functions (`src/helpers.py`)

| Function | Purpose |
|---|---|
| `smiles_to_gtm(smiles_list, ...)` | Fit a GTM and return `(gtm, crds_2d, resps, llhs)` |
| `smiles_to_tmap(smiles_list, ...)` | Compute TMAP layout; return `(x, y, s, t)` |
| `split_data(X, y, ...)` | Scaffold split into pool / test |
| `featurize(smiles_list, y_list, shuffle)` | Return `(loader, scaler)` for ChemProp |
| `build_committee_member(seed, max_epochs, log_dir)` | Return `(model, trainer)` |
| `train_committee(smiles_labeled, y_labeled, n_models, seed, max_epochs)` | Train bootstrapped ensemble; return `CommitteeRegressor` |
| `query_batch(committee, smiles_unlabeled, strategy, best_y, size, seed)` | Select next batch; return `(indices, scores)` |
| `evaluate_on_test(committee, y_test, X_test_loader, smiles_test)` | Return dict of MAE, R², Kendall τ, Spearman ρ, miscalibration area (pre- and post-calibration) |
| `run_active_learning(df_pool, df_test, ...)` | Full AL loop; return `{"history": [...], "committee": ...}` |

Constants exported from `src/helpers.py`: `STRATEGIES` (validation universe), `STRATEGY_COLORS`, `STRATEGY_QUERY_KEYS`.

## Plotting functions (`src/plots.py`)

| Function | Output |
|---|---|
| `plot_learning_curve_with_bands(...)` | Plotly line plot with ±1σ shading per strategy |
| `plot_hit_discovery_curve(...)` | Cumulative hits found vs. labeled pool size |
| `plot_calibration_curve_before_after(...)` | Before/after isotonic calibration curves |
| `plot_gtm_selection_animation(...)` | Animated Plotly scatter on GTM embedding |
| `plot_tmap_faerun(...)` | Interactive Faerun/TMAP HTML scatter; writes file and returns `Faerun` instance |

## Coding conventions

- Python 3.10+, type hints encouraged for function signatures
- `src/config.py` owns all config parsing — nothing else reads YAML or accesses raw config dicts
- `analysis.py` reads `cfg = setup["config"]` and uses `cfg.strategies` (never a hardcoded list)
- All helper functions live in `src/helpers.py`; all plot functions in `src/plots.py`
- Entry points (`run.py`, `analysis.py`) are at the repository root and import from `src`
- Use `STRATEGY_COLORS` (from `src/helpers.py`) for consistent strategy colors across all plots
- Prefer scaffold-based splits over random splits when evaluating generalization
- Docstrings follow NumPy style

## Data conventions

- Activity column: `PXR_pEC50` (raw DataFrame), renamed to `pEC50` after splitting
- SMILES column: `OPENADMET_CANONICAL_SMILES` (raw), renamed to `smiles` after splitting
- Main dataset loaded from a Parquet file via `pd.read_parquet()`; path set in `config.yaml`
- Hit threshold: `pEC50 >= 7.0` is considered a hit
- Background molecules (visualisation context only) are deduplicated against the main dataset at setup time
- Seed data (optional pretraining) is deduplicated against pool + test splits to prevent leakage
