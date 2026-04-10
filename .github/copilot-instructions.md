# GitHub Copilot Instructions

## Project overview

This repository is a tutorial/blogpost benchmarking **active learning with query-by-committee** for molecular potency prediction in drug discovery. Two targets are studied: PXR (pregnane X receptor, pEC50) and ASAP SARS-CoV-2 Mpro (pIC50). The primary artifact is `blogpost.md`. All executable code lives in `src/`, with two entry points at the repository root.

## Domain context

- **Problem domain**: QSAR / cheminformatics / computational drug discovery
- **Targets**: PXR (promiscuous nuclear receptor, pEC50) and ASAP SARS-CoV-2 Mpro (pIC50)
- **Core concept**: Active learning with query-by-committee to reduce the number of expensive wet-lab assays needed to train a useful model
- **Acquisition strategies**: EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity
- **2×2 design per target**: CheMeleon (pretrained MPNN) vs ChemProp (random init) × ChEMBL warm-start vs no ChEMBL

## Repository structure

```
blogpost.md                       # Primary document (the blogpost, prose + figure links)
run.py                            # Entry point 1: run the AL pipeline
analysis.py                       # Entry point 2: consume results/*.pkl → all HTML/PNG figures
makeitso.sh                       # SLURM submission script for all 8 configs
config/
    pxr_chemeleon_chembl_config.yaml
    pxr_chemeleon_config.yaml
    pxr_chemprop_chembl_config.yaml
    pxr_chemprop_config.yaml
    asap_chemeleon_chembl_config.yaml
    asap_chemeleon_config.yaml
    asap_chemprop_chembl_config.yaml
    asap_chemprop_config.yaml
src/
    __init__.py
    config.py                     # ALConfig dataclass + load_config() — all config parsing/validation
    helpers.py                    # Core AL utilities and module-level constants
    plots.py                      # All Plotly/Faerun plotting functions
data/
    pxr_challenge_train.csv       # PXR dataset (4140 rows; random 80/20 split → ~3312 pool)
    asap_potency.csv              # ASAP Mpro dataset (1031 Train / 297 Test; predefined split)
    chembl.csv                    # Optional ChEMBL seed training data
results/                          # Generated artifacts (one subdirectory per config)
```

### Entry points

- **`run.py`** — loads the dataset, applies the configured split type, fits the GTM embedding, and saves `<results_path>/setup_<split_type>.pkl`. Then runs `run_active_learning` for each requested `(strategy, seed)` pair, saving one `<results_path>/run_<split_type>_<STRATEGY>_seed<N>.pkl` per job. Skips jobs that already have output files (safe to resume). CLI flags: `--config PATH`, `--setup-only`, `--strategy S`, `--seed N`.
- **`analysis.py`** — loads `setup_<split>.pkl` and all `run_<split>_*.pkl` files from `results_path`, performs sanity checks, unpacks results into tidy DataFrames, and writes all interactive HTML + SVG figures. CLI flags: `--config PATH`, `--hit-threshold FLOAT`.

### Configuration

All experiment parameters live in per-experiment YAML files under `config/` and are loaded into an `ALConfig` dataclass via `load_config()` in `src/config.py`. The config is frozen into `setup_<split>.pkl` at setup time so every downstream job uses an identical configuration.

Each config YAML has four sections: `data`, `active_learning`, `training`, `clustering`.

Key `data` fields:
```yaml
data:
  dataset_path:           # Path to main labelled dataset (CSV or Parquet); tilde-expanded
  dataset_smiles_col:     # SMILES column name in the main dataset
  dataset_activity_col:   # Activity column name in the main dataset
  results_path:           # Output directory for pickles and figures
  dataset_split_seed:     # Random seed for pool/test splitter (ignored for "predefined")
  gtm_seed:               # Random seed for GTM embedding
  seed_data_path:         # Optional external pretraining dataset (null = skip)
  seed_smiles_col:        # SMILES column in seed data file
  seed_activity_col:      # Activity column in seed data file
  predefined_split_col:   # Column encoding predetermined split (default: "split")
```

Key `active_learning` fields:
```yaml
active_learning:
  strategies:   # List of strategies to run
  split_types:  # List of split types: scaffold, random, cluster, predefined
  seeds:        # List of outer random seeds (one HPC job per strategy×seed)
  k_iter:       # AL iterations per run
  query_size:   # Compounds queried per iteration
  n_start:      # Initial labeled pool size (must be > 0 when seed_data_path is null)
```

### Checkpoint formats

`<results_path>/setup_<split_type>.pkl` keys: `df_pool`, `df_test`, `df_seed`, `gtm_coords_pool`, `gtm_coords_background`, `background_smiles`, `config` (an `ALConfig` instance).

`<results_path>/run_<split_type>_<STRATEGY>_seed<N>.pkl` keys: `strategy`, `seed`, `n_start`, `result` (contains `history` list and `committee`).

## Key libraries and frameworks

- **`openadmet-models`** — primary ML framework; provides `CommitteeRegressor`, `ChemPropModel`, `ScaffoldSplitter`, `LightningTrainer`, `ChemPropFeaturizer`, `RegressionMetrics`, `UncertaintyMetrics`. Install from [openadmet-models on GitHub](https://github.com/OpenADMET/openadmet-models).
- **`chemographykit`** — provides `GTM` (Generative Topographic Map) and `calculate_latent_coords` for 2D chemical space embedding
- **`tmap`** + **`mhfp`** — TMAP layout using MHFP fingerprints and an LSH forest; produces minimum-spanning-tree visualizations
- **`faerun`** — interactive HTML scatter plots layered on TMAP layouts
- **`useful_rdkit_utils`** (`uru`) — RDKit descriptor calculation and scaling (`RDKitDescriptors`, `clean_and_scale_descriptors`)
- **`uncertainty_toolbox`** (`uct`) — uncertainty calibration and calibration curve computation
- **`lightning` (PyTorch Lightning)** — model training backend
- **`chemprop` / CheMeleon** — MPNN backbone for property prediction
- **`rdkit`** — molecular parsing
- **`pandas`**, **`numpy`**, **`scipy`** — data manipulation and statistics
- **`matplotlib`**, **`plotly`** — visualization
- **`PyYAML`** — config parsing in `src/config.py`
- **`kaleido`** — SVG/PNG export for Plotly figures in `analysis.py`

## Molecular representations

- Molecules are stored as SMILES strings (column name varies by dataset; renamed to `smiles` after loading)
- Models use `ChemPropFeaturizer` internally (graph-based, no explicit fingerprints)
- GTM embedding uses RDKit physicochemical descriptors (scaled via `useful_rdkit_utils`)
- TMAP layout uses MHFP (MinHash fingerprints) encoded with `MHFPEncoder` and indexed with an LSH forest

## Active learning architecture

- **Committee**: `n_models` (default 5) independent `ChemPropModel` instances trained on bootstrapped labeled pools; size and epochs set via config
- **Ensemble diversity**: deep ensembling + bootstrap bagging; member seed = `seed * n_models + i`, iteration seed = `seed + k`
- **Acquisition**: `query_batch(committee, smiles_unlabeled, strategy, best_y, size=query_size, seed=seed+k)` returns `(indices, scores)`
- **Loop**: `run_active_learning(...)` runs `k_iter` AL iterations from an initial labeled pool of size `n_start` (0 for ChEMBL-bootstrapped runs)
- **Calibration**: at each iteration, 10% of the current labeled pool is sampled *before* concatenating seed data, then used as a held-out calibration set; pre- and post-calibration metrics are both recorded
- **GTM visualization**: computed once in `run_setup` on `pool + background` molecules; coordinates sliced and passed as `gtm_coords` to avoid redundant refitting per job
- **Test loader**: pre-computed once before the loop and reused across iterations
- **x-axis** (`n_labeled`): counts only pool-acquired labels — ChEMBL runs start at x=0, no-ChEMBL runs start at x=n_start

## `src/config.py`

- `ALConfig` — frozen dataclass with all config fields; picklable; stored in `setup_<split>.pkl`
- `load_config(path) -> ALConfig` — reads YAML, validates all fields (including cross-field check: `n_start=0` requires `seed_data_path`), returns `ALConfig`
- `_VALID_STRATEGIES` and `_VALID_SPLIT_TYPES` — universe of allowed values

## Helper functions (`src/helpers.py`)

| Function | Purpose |
|---|---|
| `smiles_to_gtm(smiles_list, ...)` | Fit a GTM and return `(gtm, crds_2d, resps, llhs)` |
| `smiles_to_tmap(smiles_list, ...)` | Compute TMAP layout; return `(x, y, s, t)` |
| `split_data(X, y, ...)` | Scaffold/random/cluster split into pool / test |
| `featurize(smiles_list, y_list, shuffle)` | Return `(loader, scaler)` for ChemProp |
| `build_committee_member(seed, max_epochs, log_dir)` | Return `(model, trainer)` |
| `train_committee(smiles_labeled, y_labeled, n_models, seed, max_epochs)` | Train bootstrapped ensemble; return `CommitteeRegressor` |
| `query_batch(committee, smiles_unlabeled, strategy, best_y, size, seed)` | Select next batch; return `(indices, scores)` |
| `evaluate_on_test(committee, y_test, X_test_loader)` | Return dict of MAE, R², Kendall τ, Spearman ρ, miscalibration area |
| `run_active_learning(df_pool, df_test, ...)` | Full AL loop; return `{"history": [...], "committee": ...}` |

Constants exported: `STRATEGIES`, `STRATEGY_COLORS`, `STRATEGY_QUERY_KEYS`.

## Plotting functions (`src/plots.py`)

| Function | Output |
|---|---|
| `plot_learning_curve_with_bands(...)` | Plotly line plot with ±1σ shading per strategy |
| `plot_hit_discovery_curve(...)` | Cumulative hits found vs. labeled pool size |
| `plot_calibration_curve_before_after(...)` | Before/after calibration curves |
| `plot_gtm_selection_animation(...)` | Animated Plotly scatter on GTM embedding |
| `plot_tmap_faerun(...)` | Interactive Faerun/TMAP HTML scatter |

## Coding conventions

- Python 3.10+, type hints encouraged for function signatures
- `src/config.py` owns all config parsing — nothing else reads YAML or accesses raw config dicts
- `analysis.py` loads `cfg` from the YAML via `load_config(args.config)` (not from the setup pickle)
- All helper functions live in `src/helpers.py`; all plot functions in `src/plots.py`
- Use `STRATEGY_COLORS` for consistent strategy colors across all plots
- Docstrings follow NumPy style

## Data conventions

| Field | PXR | ASAP Mpro |
|---|---|---|
| SMILES column | `SMILES` | `CXSMILES` |
| Activity column | `pEC50` | `pIC50 (SARS-CoV-2 Mpro)` |
| Split type | `random` (80/20) | `predefined` (`Set` column: `Train`/`Test`) |
| Pool size | ~3312 | 1031 |
| Test size | ~828 | 297 |

After loading, both SMILES and activity columns are renamed to `smiles` and `pEC50` internally. The seed data (`chembl.csv`) uses `SMILES` and `pEC50` columns.

Hit threshold: activity ≥ 7.0 is considered a hit (override with `--hit-threshold`).
