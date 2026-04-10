# Active Learning for Potency Prediction

A reproducible benchmark of **active learning with query-by-committee** for predicting molecular potency in drug discovery. The repository benchmarks six acquisition strategies across two targets, two model architectures, and two data-bootstrapping conditions. All code is in `run.py`, `analysis.py`, and `src/`.

---

## Background

Wet-lab dose–response assays are expensive. Active learning lets a model guide its own data collection — querying the compounds most likely to improve the model or reveal new hits. This benchmark compares six acquisition strategies across multiple random seeds, producing learning curves with error bands and interactive chemical-space visualizations.

**Strategies compared:** EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity

**Targets:**
- **PXR** (pregnane X receptor) — ~3300 pool compounds after 80/20 random split
- **ASAP SARS-CoV-2 Mpro** — 1031 predefined train / 297 predefined test compounds

**2×2 experimental design per target:**

| | ChEMBL warm-start | No ChEMBL |
|---|---|---|
| **CheMeleon** (pretrained MPNN weights) | ✓ | ✓ |
| **ChemProp** (random init) | ✓ | ✓ |

---

## Repository structure

```
run.py                            # Pipeline entry point: setup + per-job AL runs
analysis.py                       # Visualization entry point: figures from saved results
config/
    pxr_chemeleon_chembl_config.yaml
    pxr_chemeleon_config.yaml
    pxr_chemprop_chembl_config.yaml
    pxr_chemprop_config.yaml
    asap_chemeleon_chembl_config.yaml
    asap_chemeleon_config.yaml
    asap_chemprop_chembl_config.yaml
    asap_chemprop_config.yaml
makeitso.sh                       # SLURM submission script for all 8 configs
blogpost.md                       # Prose narrative with embedded figure links
src/
    config.py                     # ALConfig dataclass + load_config() validation
    helpers.py                    # Core AL loop, committee training, acquisition functions
    plots.py                      # All Plotly/Faerun plotting functions
data/
    pxr_challenge_train.csv       # PXR labelled dataset
    asap_potency.csv              # ASAP Mpro labelled dataset (predefined split)
    chembl.csv                    # Optional ChEMBL seed training data
results/                          # Generated outputs (created at runtime, one folder per config)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install openadmet-models chemographykit tmap mhfp faerun useful_rdkit_utils \
            uncertainty_toolbox lightning chemprop rdkit pandas numpy scipy \
            matplotlib plotly pyyaml kaleido
```

`openadmet-models` must be installed from source: https://github.com/OpenADMET/openadmet-models

### 2. Run

```bash
# Step 1: setup (split + GTM embedding) — one per config
python run.py --config config/pxr_chemprop_config.yaml --setup-only

# Step 2: one job per (strategy, seed)
python run.py --config config/pxr_chemprop_config.yaml --strategy EI --seed 42

# Step 3: generate all figures for a config
python analysis.py --config config/pxr_chemprop_config.yaml
```

---

## HPC workflow (SLURM)

The `makeitso.sh` script runs setup for all 8 configs serially, then dispatches one SLURM job per `(config × strategy × seed)` — 240 jobs total.

```bash
bash makeitso.sh
```

Jobs are idempotent: if `run_<split>_<STRATEGY>_seed<N>.pkl` already exists it is skipped, so partial runs can be safely resumed.

---

## Configuration reference

Each YAML config has four sections: `data`, `active_learning`, `training`, and `clustering`.

### `data`

| Key | Type | Description |
|---|---|---|
| `dataset_path` | str | Path to the main labelled dataset (CSV or Parquet). Tilde-expanded. |
| `dataset_smiles_col` | str | SMILES column name in the main dataset. |
| `dataset_activity_col` | str | Activity column name in the main dataset. |
| `results_path` | str | Directory where pickles and figures are written. Created at runtime. |
| `dataset_split_seed` | int | Random seed for the pool/test splitter. Ignored for `"predefined"` split. |
| `gtm_seed` | int | Random seed for the GTM embedding. Does not affect model training. |
| `seed_data_path` | str or null | Optional external pretraining dataset. `null` = skip. |
| `seed_smiles_col` | str | SMILES column in the seed data file. |
| `seed_activity_col` | str | Activity column in the seed data file. |
| `predefined_split_col` | str | Column encoding a predetermined split (values: `"train"`/`"test"`, case-insensitive). Only used when `split_types` includes `"predefined"`. Default: `"split"`. |

### `active_learning`

| Key | Type | Description |
|---|---|---|
| `strategies` | list[str] | Acquisition strategies to run. Subset of `[EI, UCB, Random, Exploitation, Exploration, Diversity]`. |
| `split_types` | list[str] | Split types to run. Subset of `[scaffold, random, cluster, predefined]`. |
| `seeds` | list[int] | Outer random seeds. One HPC job per `(strategy, seed)` pair. |
| `k_iter` | int | AL iterations per run. |
| `query_size` | int | Compounds queried from the pool per iteration. |
| `n_start` | int | Initial labeled pool size. Must be `> 0` when `seed_data_path` is null. |

### `training`

| Key | Type | Description |
|---|---|---|
| `n_models` | int | Committee size (bootstrapped ensemble members). |
| `max_epochs` | int | Max training epochs per committee member. |
| `use_chemeleon` | bool | If `true`, initialise each committee member from CheMeleon pretrained weights. |

### `clustering` (optional)

| Key | Type | Description |
|---|---|---|
| `method` | str | `butina`, `kmeans`, or `bemis-murcko`. Default: `butina`. |
| `k_clusters` | int | Number of clusters for `kmeans`. |
| `butina_cutoff` | float | Tanimoto distance threshold for Butina clustering (0–1). Default: `0.65`. |

---

## Output files

All outputs are written to `results_path` specified in the config:

| File | Description |
|---|---|
| `setup_<split>.pkl` | Split DataFrames, GTM coordinates, background SMILES, frozen `ALConfig` |
| `run_<split>_<STRATEGY>_seed<N>.pkl` | Per-job AL history and final committee |
| `learning_curve_mae.html/.svg` | MAE learning curves with ±1σ bands per strategy |
| `learning_curve_ktau.html/.svg` | Kendall's τ learning curves with ±1σ bands |
| `hit_discovery_curve.html/.svg` | Cumulative hits (activity ≥ threshold) vs. labeled pool size |
| `gtm_selection_animation_exploitation.html/.svg` | Animated GTM showing Exploitation query selections |
| `tmap_selection.html` | Interactive TMAP (Faerun) colored by AL iteration |
| `calibration_curve.html/.svg` | Before/after uncertainty calibration curves |

---

## Active learning details

- **Model**: `CommitteeRegressor` of `n_models` independent `ChemPropModel` instances (MPNN backbone)
- **Diversity**: bootstrap bagging + different random seeds per member (`seed * n_models + i`) and per iteration (`seed + k`)
- **Calibration**: at each iteration, 10% of the current labeled pool is held out for scaling-factor calibration; pre- and post-calibration metrics are both recorded
- **Splits**: scaffold / random (80% pool / 20% test) or predefined (uses a `split` column in the dataset)
- **Hit threshold**: activity ≥ 7.0 by default; override with `--hit-threshold`
- **Seed data** (optional): external pretraining compounds included in every training step, never queried or evaluated; deduplicated against pool and test sets at setup time
- **x-axis**: `n_labeled` counts only pool-acquired labels (seed data excluded), so ChEMBL runs start at x=0 and no-ChEMBL runs start at x=n_start

---

## Data conventions

| Field | PXR | ASAP Mpro |
|---|---|---|
| SMILES column | `SMILES` | `CXSMILES` |
| Activity column | `pEC50` | `pIC50 (SARS-CoV-2 Mpro)` |
| Split type | random (80/20) | predefined (`Set` column: `Train`/`Test`) |
| Pool size | ~3312 | 1031 |
| Test size | ~828 | 297 |

After loading, both SMILES and activity columns are renamed to `smiles` and `pEC50` internally.
