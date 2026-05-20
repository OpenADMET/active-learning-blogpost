# Active Learning for Potency Prediction

A reproducible benchmark of **active learning with query-by-committee** for predicting molecular potency in drug discovery. Six acquisition strategies are compared across two targets, two model architectures, and two data-bootstrapping conditions.

---

## Background

Wet-lab dose–response assays are expensive. Active learning lets a model guide its own data collection — querying the compounds most likely to improve the model or reveal new hits. Each run uses a `CommitteeRegressor` of bootstrapped `ChemPropModel` ensembles (MPNN backbone); at each iteration 10% of the labeled pool is held out for uncertainty calibration, and `n_labeled` counts only pool-acquired labels (seed data excluded).

**Strategies:** EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity

**Targets:**
- **PXR** (pregnane X receptor) — ~3300 pool compounds after 80/20 random split
- **ASAP SARS-CoV-2 Mpro** — 1031 predefined train / 297 predefined test compounds

**2×2 experimental design per target:**

| | ChEMBL warm-start | No ChEMBL |
|---|---|---|
| **CheMeleon** (pretrained MPNN) | ✓ | ✓ |
| **ChemProp** (random init) | ✓ | ✓ |

Hit threshold: activity ≥ 7.0 by default; override with `--hit-threshold`.

---

## Repository structure

```
run.py                   # Pipeline entry point: setup + per-job AL runs
analysis.py              # Visualization: figures from saved results
analysis_combined.py     # Cross-config grid figures (ASAP 1×2, PXR 2×2)
synthetic_run.py         # Synthetic oracle entry point (fast, no GPU)
synthetic_analysis.py    # Synthetic oracle figures + oracle-tier tables
verify_stats.py          # Statistical verification of blogpost claims
makeitso.sh              # SLURM submission script for all real configs
blogpost.md              # Prose narrative with embedded figure links
config/                  # 8 real configs + 8 synthetic oracle configs (rho00/03/06/09)
src/
    config.py            # ALConfig dataclass + load_config() validation
    helpers.py           # Core AL loop, committee training, acquisition functions
    plots.py             # All Plotly/Faerun plotting functions
    synthetic.py         # SyntheticOracle, CachedPredOracle, oracle AL loop
data/
    pxr_challenge_train.csv   # PXR labelled dataset
    asap_potency.csv          # ASAP Mpro labelled dataset (predefined split)
    chembl.csv                # Optional ChEMBL seed training data
results/                 # Generated outputs (created at runtime, one folder per config)
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
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml --setup-only
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml --strategy EI --seed 42
python synthetic_analysis.py --config config/oracle_pxr_rho06_config.yaml

# Fast sanity check (forces k_iter=2, seeds=[42])
python synthetic_run.py --config config/oracle_pxr_rho06_config.yaml --smoke-test
```

---

## HPC workflow (SLURM)

`makeitso.sh` runs setup for all 8 real configs serially, then dispatches one SLURM job per `(config × strategy × seed)` — 240 jobs total. Jobs are idempotent; existing output files are skipped so partial runs can be safely resumed.

```bash
bash makeitso.sh
```

---

## Configuration reference

Each YAML config has four sections. The example configs in `config/` are the best starting point.

**`data`** — dataset path, SMILES/activity column names, results output directory, split seed, GTM seed, and optional seed data (ChEMBL). Use `predefined_split_col` when the dataset has a pre-assigned train/test column. Synthetic oracle configs may set `setup_results_path` to reuse an existing setup pickle.

**`active_learning`** — `strategies` (list), `split_types` (list), `seeds` (list), `k_iter` (iterations), `query_size` (compounds per iteration), `n_start` (initial labeled pool; must be > 0 when no seed data is provided).

**`training`** — `n_models` (committee size), `max_epochs`, `use_chemeleon` (bool; load pretrained weights).

**`clustering`** (optional) — `method` (`butina`/`kmeans`/`bemis-murcko`), `butina_cutoff` (default 0.65), `k_clusters`.

**`oracle`** (synthetic configs only) — `initial_mae`, `final_mae`, `initial_rho`, `final_rho` (all ramp linearly over iterations), `cache_path`.

---

## Output files

Each config's `results_path` receives two pickle files per run (`setup_<split>.pkl`, `run_<split>_<STRATEGY>_seed<N>.pkl`) plus a full set of HTML/SVG figures: learning curves (MAE, Kendall's τ), hit discovery curves, calibration curves, miscalibration area, σ–|error| correlation, animated GTM selections, and interactive TMAP/Faerun visualizations.

`analysis_combined.py` writes cross-config panels (hit/non-hit gap curves, mirror KDE animations) to the `--output-dir` directory.

---

## Synthetic oracle pipeline

`synthetic_run.py` + `synthetic_analysis.py` mirror the real pipeline exactly, replacing committee training with a parametric `SyntheticOracle`. MAE and uncertainty quality (Spearman ρ between σ and |error|) ramp linearly from initial to final values over the campaign, enabling rapid GPU-free ablations across oracle tiers (`rho00` → `rho09`). Output pickles are identical in format to `run.py`, so `analysis.py` and `analysis_combined.py` work unchanged.

---

## Statistical verification

```bash
python verify_stats.py
```

Performs one-to-one verification of all statistical claims in `blogpost.md`. Run after generating results.

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
