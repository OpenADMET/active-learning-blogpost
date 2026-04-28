# Active Learning for Potency Prediction

A reproducible benchmark of **active learning with query-by-committee** for predicting molecular potency in drug discovery. The repository benchmarks six acquisition strategies across two targets, two model architectures, and two data-bootstrapping conditions. All code is in `run.py`, `analysis.py`, `synthetic_run.py`, `synthetic_analysis.py`, `analysis_combined.py`, and `src/`.

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
run.py                              # Pipeline entry point: setup + per-job AL runs
analysis.py                         # Visualization entry point: figures from saved results
analysis_combined.py                # Cross-config grid figures (ASAP 1×2, PXR 2×2)
synthetic_run.py                    # Synthetic oracle entry point: fast oracle-based runs
synthetic_analysis.py               # Synthetic oracle analysis: figures + oracle-tier tables
verify_stats.py                     # Statistical verification of blogpost claims
config/
    pxr_chemeleon_chembl_config.yaml
    pxr_chemeleon_config.yaml
    pxr_chemprop_chembl_config.yaml
    pxr_chemprop_config.yaml
    asap_chemeleon_chembl_config.yaml
    asap_chemeleon_config.yaml
    asap_chemprop_chembl_config.yaml
    asap_chemprop_config.yaml
    oracle_pxr_rho00_config.yaml    # Synthetic oracle configs (uncalibrated, ρ=0)
    oracle_pxr_rho03_config.yaml    # … ρ=0.3
    oracle_pxr_rho06_config.yaml    # … ρ=0.6
    oracle_pxr_rho09_config.yaml    # … ρ=0.9
    oracle_asap_rho00_config.yaml   # ASAP synthetic oracle configs
    oracle_asap_rho03_config.yaml
    oracle_asap_rho06_config.yaml
    oracle_asap_rho09_config.yaml
makeitso.sh                         # SLURM submission script for all real configs
blogpost.md                         # Prose narrative with embedded figure links
src/
    config.py                       # ALConfig dataclass + load_config() validation
    helpers.py                      # Core AL loop, committee training, acquisition functions
    plots.py                        # All Plotly/Faerun plotting functions
    synthetic.py                    # SyntheticOracle, CachedPredOracle, oracle AL loop
data/
    pxr_challenge_train.csv         # PXR labelled dataset
    asap_potency.csv                # ASAP Mpro labelled dataset (predefined split)
    chembl.csv                      # Optional ChEMBL seed training data
results/                            # Generated outputs (created at runtime, one folder per config)
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

### 2. Run real experiments

```bash
# Step 1: setup (split + GTM embedding) — one per config
python run.py --config config/pxr_chemprop_config.yaml --setup-only

# Step 2: one job per (strategy, seed)
python run.py --config config/pxr_chemprop_config.yaml --strategy EI --seed 42

# Step 3: generate all figures for a config
python analysis.py --config config/pxr_chemprop_config.yaml

# Step 4 (optional): cross-config combined grid figures
python analysis_combined.py --output-dir results/combined
```

### 3. Run synthetic oracle experiments (fast, no GPU)

```bash
# Setup (can reuse real setup pickles via data.setup_results_path in config)
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml --setup-only

# Single job
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml \
    --strategy EI --seed 42

# Analysis figures
python synthetic_analysis.py --config config/oracle_pxr_rho06_config.yaml

# Fast sanity check (forces k_iter=2, seeds=[42])
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml --smoke-test
```

---

## HPC workflow (SLURM)

The `makeitso.sh` script runs setup for all 8 real configs serially, then dispatches one SLURM job per `(config × strategy × seed)` — 240 jobs total.

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
| `setup_results_path` | str or null | (Synthetic oracle only) Directory to load pre-existing setup pickles from. `null` = build fresh. |

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

### `oracle` (synthetic oracle configs only)

| Key | Type | Description |
|---|---|---|
| `initial_mae` | float | Oracle MAE at iteration 0. |
| `final_mae` | float | Oracle MAE at the last iteration (ramps linearly). |
| `initial_rho` | float | Spearman ρ(σ, \|error\|) at iteration 0. |
| `final_rho` | float | Spearman ρ at the last iteration (ramps linearly). |
| `cache_path` | str or null | Path to a cached oracle predictions file. `null` = build fresh each run. |

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
| `calibration_area_per_iteration.html/.svg` | Miscalibration area (pre- and post-calibration) vs. iteration |
| `sigma_error_correlation.html/.svg` | Spearman ρ(σ, \|error\|) vs. labeled pool size |
| `gtm_selection_animation_exploitation.html/.svg` | Animated GTM showing Exploitation query selections |
| `tmap_selection.html` | Interactive TMAP (Faerun) colored by AL iteration |
| `tmap_partition.html/.svg` | TMAP colored by train/test partition |
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

## Synthetic oracle pipeline

`synthetic_run.py` + `synthetic_analysis.py` mirror the real pipeline exactly but replace committee training with a parametric `SyntheticOracle` that simulates predictions with configurable MAE and uncertainty quality (Spearman ρ between σ and |error|). Both parameters ramp linearly from initial to final values over the campaign.

This enables rapid ablation studies (no GPU required) over oracle tiers defined by their ρ:

| Config suffix | Initial ρ | Final ρ | Interpretation |
|---|---|---|---|
| `rho00` | 0.0 | 0.0 | Uncalibrated, random uncertainty |
| `rho03` | 0.0 | 0.3 | Weakly informative uncertainty |
| `rho06` | 0.0 | 0.6 | Moderately calibrated |
| `rho09` | 0.0 | 0.9 | Near-ideal uncertainty |

Results are saved in the identical pickle format as `run.py`, so `analysis.py` and `analysis_combined.py` work unchanged on synthetic oracle outputs.

---

## Statistical verification

`verify_stats.py` performs one-to-one verification of all statistical claims in `blogpost.md`. Run it after generating results:

```bash
python verify_stats.py
```

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
