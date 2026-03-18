#!/usr/bin/env python
"""Active learning pipeline entry point.

Loads the PXR dataset, performs a scaffold split, fits the GTM embedding for
visualization, runs the active learning loop for every strategy, and checkpoints
everything to ``results/all_runs.pkl``.

Usage
-----
    python run.py

The script resumes gracefully from an existing checkpoint: strategies that are
already present in the pickle are skipped.  The checkpoint format stores the
enriched payload expected by ``analysis.py``::

    {
        "all_runs":              dict[str, dict],   # strategy → {history, committee}
        "df_pool":               pd.DataFrame,
        "df_cal":                pd.DataFrame,
        "df_test":               pd.DataFrame,
        "gtm_coords_pool":       np.ndarray,        # shape (n_pool, 2)
        "gtm_coords_background": np.ndarray,        # shape (n_bg, 2)
        "background_smiles":     list[str],
        "config":                dict,
    }
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openadmet.models.split.scaffold import ScaffoldSplitter

from src.helpers import (
    STRATEGIES,
    run_active_learning,
    smiles_to_gtm,
)

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
N_START = 100  # Initial labeled pool size
M_QUERY = 20  # Molecules queried per iteration
K_ITER = 15  # Active learning iterations
N_MODELS = 5  # Committee size
SEED = 42
MAX_EPOCHS = 20

# ── Data loading ───────────────────────────────────────────────────────────────
df = pd.read_parquet(
    "~/repos/skunkworks/chemprop_mve/datasets/pxr/pxr_combined_canonical.parquet"
)

background_smiles = list(
    pd.read_csv("_data/octant_screening_compounds.csv")["Smiles"].values
)
background_smiles = [
    x for x in background_smiles if x not in df["OPENADMET_CANONICAL_SMILES"].values
]

print(f"Dataset: {len(df)} compounds")
print(df["PXR_pEC50"].describe())

# Plot activity distribution
fig, ax = plt.subplots(1, dpi=150)
ax.hist(df["PXR_pEC50"], bins=30, color="teal", alpha=0.7)
ax.set_title("PXR pEC50 Distribution")
ax.set_xlabel("pEC50", fontweight="bold")
ax.set_ylabel("Count", fontweight="bold")
Path("results").mkdir(exist_ok=True)
fig.savefig("results/pec50_distribution.png", bbox_inches="tight")
plt.close(fig)

# ── Scaffold split ─────────────────────────────────────────────────────────────
splitter = ScaffoldSplitter(
    train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
)
X_pool, X_cal, X_test, y_pool, y_cal, y_test, _ = splitter.split(
    df["OPENADMET_CANONICAL_SMILES"], df["PXR_pEC50"]
)

df_pool = pd.DataFrame({"smiles": X_pool, "pEC50": y_pool}).reset_index(drop=True)
df_cal = pd.DataFrame({"smiles": X_cal, "pEC50": y_cal}).reset_index(drop=True)
df_test = pd.DataFrame({"smiles": X_test, "pEC50": y_test}).reset_index(drop=True)

min_pool_needed = N_START + K_ITER * M_QUERY
assert len(df_pool) >= min_pool_needed, (
    f"Pool too small: {len(df_pool)} < {min_pool_needed}. "
    "Increase dataset size or reduce N_START/K_ITER/M_QUERY."
)

print(f"Pool size: \t\t{len(df_pool)}")
print(f"Calibration size: \t{len(df_cal)}")
print(f"Test size: \t\t{len(df_test)}")

# ── GTM embedding (pool + background, for visualization) ──────────────────────
print("Fitting GTM on pool + background molecules...")
gtm_model, gtm_coords, resps, llhs = smiles_to_gtm(
    list(df_pool["smiles"].values) + background_smiles,
    device="cpu",
)
gtm_coords_pool = gtm_coords[: len(df_pool)]
gtm_coords_background = gtm_coords[len(df_pool) :]

# ── Load checkpoint or start fresh ────────────────────────────────────────────
pkl_path = Path("results/all_runs.pkl")
if pkl_path.exists():
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    # Support both the old format (bare all_runs dict) and the new enriched format
    if isinstance(data, dict) and "all_runs" in data:
        all_runs = data["all_runs"]
    else:
        all_runs = data
else:
    all_runs = {}

print(f"Strategies already completed: {list(all_runs.keys())}")

# ── Active learning loop ───────────────────────────────────────────────────────
for strategy in STRATEGIES:
    if strategy in all_runs:
        print(f"Skipping {strategy!r} (already in checkpoint).")
        continue

    print(f"\n{'─' * 60}")
    print(f"Running strategy: {strategy}")
    print(f"{'─' * 60}")

    all_runs[strategy] = run_active_learning(
        df_pool,
        df_test,
        n_start=N_START,
        k_iter=K_ITER,
        seed=SEED,
        strategy=strategy,
        verbose=True,
    )

    # Checkpoint after each strategy
    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "all_runs": all_runs,
                "df_pool": df_pool,
                "df_cal": df_cal,
                "df_test": df_test,
                "gtm_coords_pool": gtm_coords_pool,
                "gtm_coords_background": gtm_coords_background,
                "background_smiles": background_smiles,
                "config": {
                    "N_START": N_START,
                    "M_QUERY": M_QUERY,
                    "K_ITER": K_ITER,
                    "N_MODELS": N_MODELS,
                    "SEED": SEED,
                    "MAX_EPOCHS": MAX_EPOCHS,
                },
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Checkpoint saved after strategy '{strategy}'.")

print("\nAll strategies complete. Results saved to results/all_runs.pkl")
