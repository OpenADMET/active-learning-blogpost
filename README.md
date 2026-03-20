# Active Learning for pEC50 Prediction

A reproducible benchmark of **active learning with query-by-committee** for predicting PXR (pregnane X receptor) pEC50 values in drug discovery. The repository accompanies a blogpost that narrates the full experiment; all code is in `run.py`, `analysis.py`, and `src/`.

---

## Background

Wet-lab dose–response assays are expensive. Active learning lets a model guide its own data collection — querying the compounds most likely to improve the model or reveal new hits. This benchmark compares six acquisition strategies across multiple random seeds, producing learning curves with error bands and interactive chemical-space visualizations.

**Strategies compared:** EI (Expected Improvement), UCB (Upper Confidence Bound), Random, Exploitation, Exploration, Diversity

---

## Repository structure

```
run.py                            # Pipeline entry point: setup + per-job AL runs
analysis.py                       # Visualization entry point: figures from saved results
config.yaml                       # Experiment configuration (edit this to configure runs)
blogpost.md                       # Prose narrative with embedded figure links
src/
    config.py                     # ALConfig dataclass + load_config() validation
    helpers.py                    # Core AL loop, committee training, acquisition functions
    plots.py                      # All Plotly/Faerun plotting functions
_data/
    octant_screening_compounds.csv  # Background compound library (visualisation context)
results/                          # Generated outputs (created at runtime)
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

### 2. Configure

Edit `config.yaml` to point at your dataset and set experiment parameters:

```yaml
data:
  dataset_path: path/to/your/dataset.parquet   # main labelled dataset
  background_path: _data/octant_screening_compounds.csv
  background_smiles_col: Smiles
  seed_data_path: null          # optional pretraining data; null = skip
  seed_smiles_col: smiles
  seed_activity_col: pEC50

active_learning:
  strategies: [EI, UCB, Random, Exploitation, Exploration, Diversity]
  seeds: [42, 43, 44, 45, 46]
  k_iter: 15
  query_size: 20
  n_start: 100

training:
  n_models: 5
  max_epochs: 20
```

### 3. Run

```bash
# Run everything locally (all strategies × all seeds, serially)
python run.py

# Or run a single (strategy, seed) pair
python run.py --strategy EI --seed 42

# Generate all figures
python analysis.py
```

---

## HPC workflow (SLURM)

Compute setup once, then dispatch one job per `(strategy, seed)` pair:

```bash
# Step 1: scaffold split + GTM embedding (fast, single node)
python run.py --setup-only

# Step 2: one job per pair
for strat in EI UCB Random Exploitation Exploration Diversity; do
    for seed in 42 43 44 45 46; do
        sbatch --job-name=al_${strat}_${seed} \
               --wrap="python run.py --strategy ${strat} --seed ${seed}"
    done
done

# Step 3: generate figures (after all jobs finish)
python analysis.py
```

Jobs are idempotent — if `results/run_<STRATEGY>_seed<N>.pkl` already exists it is skipped, so partial runs can be safely resumed.

---

## Configuration reference (`config.yaml`)

### `data`

| Key | Type | Description |
|---|---|---|
| `dataset_path` | str | Path to the main labelled dataset (Parquet). Tilde-expanded. |
| `background_path` | str | Background compound library for GTM/TMAP visualisations (CSV or Parquet). Never used for training. |
| `background_smiles_col` | str | SMILES column name in the background file. |
| `seed_data_path` | str or null | Optional external pretraining dataset. `null` = skip. |
| `seed_smiles_col` | str | SMILES column in the seed data file. |
| `seed_activity_col` | str | Activity column in the seed data file. |

### `active_learning`

| Key | Type | Description |
|---|---|---|
| `strategies` | list[str] | Acquisition strategies to run. Subset of `[EI, UCB, Random, Exploitation, Exploration, Diversity]`. |
| `seeds` | list[int] | Outer random seeds. One HPC job per `(strategy, seed)` pair. |
| `k_iter` | int | AL iterations per run. |
| `query_size` | int | Compounds queried from the pool per iteration. |
| `n_start` | int | Initial labeled pool size. Set to `0` when using `seed_data_path`. |

### `training`

| Key | Type | Description |
|---|---|---|
| `n_models` | int | Committee size (bootstrapped ensemble members). |
| `max_epochs` | int | Max training epochs per committee member. |

---

## Output files

All outputs are written to `results/`:

| File | Description |
|---|---|
| `setup.pkl` | Scaffold split, GTM coordinates, background SMILES, frozen `ALConfig` |
| `run_<STRATEGY>_seed<N>.pkl` | Per-job AL history and final committee |
| `learning_curve_mae.html/.png` | MAE learning curves with ±1σ bands per strategy |
| `learning_curve_ktau.html/.png` | Kendall's τ learning curves with ±1σ bands |
| `hit_discovery_curve.html/.png` | Cumulative hits (pEC50 ≥ 7.0) vs. labeled pool size |
| `gtm_selection_animation_exploitation.html/.png` | Animated GTM showing Exploitation query selections |
| `tmap_selection.html/.png` | Interactive TMAP (Faerun) colored by AL iteration (EI strategy) |
| `calibration_curve.html/.png` | Before/after isotonic uncertainty calibration |
| `pec50_distribution.png` | Activity distribution of the full dataset |

---

## Active learning details

- **Model**: `CommitteeRegressor` of `n_models` independent `ChemPropModel` instances (MPNN backbone)
- **Diversity**: bootstrap bagging + different random seeds per member (`seed+i`) and per iteration (`seed+k`)
- **Calibration**: at each iteration, 10% of the current labeled pool is held out for isotonic regression calibration; pre- and post-calibration metrics are both recorded
- **Split**: scaffold-based 80% pool / 20% test; `ScaffoldSplitter` from `openadmet-models`
- **Hit threshold**: pEC50 ≥ 7.0
- **Seed data** (optional): external pretraining compounds included in every training step, never queried or evaluated; deduplicated against pool and test sets

---

## Data conventions

| Column | Raw name | Post-split name |
|---|---|---|
| Activity | `PXR_pEC50` | `pEC50` |
| SMILES | `OPENADMET_CANONICAL_SMILES` | `smiles` |
