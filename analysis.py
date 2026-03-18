#!/usr/bin/env python
"""Analysis and visualization entry point.

Loads the active learning results from ``results/all_runs.pkl`` (produced by
``run.py``) and generates all figures referenced in ``blogpost.md``.

Usage
-----
    python analysis.py

Generated outputs (written to ``results/``):
    learning_curve_mae.html                  — MAE learning curves per strategy
    learning_curve_ktau.html                 — Kendall's τ learning curves per strategy
    hit_discovery_curve.html                 — cumulative hits vs. labeled-pool size
    gtm_selection_animation_exploitation.html — animated GTM selection (Exploitation)
    tmap_selection.html                       — interactive TMAP (EI, via Faerun)
    calibration_curve.html                   — before/after isotonic calibration
"""

import copy
import pickle
import warnings
from pathlib import Path

import pandas as pd
import uncertainty_toolbox as uct

import src.plots as alp
from src.helpers import (
    STRATEGIES,
    STRATEGY_COLORS,
    evaluate_on_test,
    featurize,
    smiles_to_tmap,
)

warnings.filterwarnings("ignore")

# ── Load results ───────────────────────────────────────────────────────────────
pkl_path = Path("results/all_runs.pkl")
if not pkl_path.exists():
    raise FileNotFoundError(
        "results/all_runs.pkl not found. Run `python run.py` first."
    )

with open(pkl_path, "rb") as fh:
    data = pickle.load(fh)

all_runs = data["all_runs"]
df_pool = data["df_pool"]
df_cal = data["df_cal"]
df_test = data["df_test"]
gtm_coords_pool = data["gtm_coords_pool"]
gtm_coords_background = data["gtm_coords_background"]
background_smiles = data["background_smiles"]

print(f"Loaded results for strategies: {list(all_runs.keys())}")

# ── Unpack into tidy DataFrames ───────────────────────────────────────────────
records = []
pool_history_records = []

for strategy in STRATEGIES:
    run_data = all_runs[strategy]["history"]
    for step in run_data:
        records.append(
            {
                "strategy": strategy,
                "iteration": step["iteration"],
                "n_labeled": step["n_labeled"],
                "mae": step["mae"],
                "r2": step["r2"],
                "ktau": step["ktau"],
                "spearmanr": step["spearmanr"],
            }
        )
        for pval in step["pool_y_values"]:
            pool_history_records.append(
                {
                    "strategy": strategy,
                    "iteration": step["iteration"],
                    "pEC50": float(pval),
                }
            )

learning_curve_long = pd.DataFrame(records)
pool_history_long = pd.DataFrame(pool_history_records)
print(
    f"Curve rows: {len(learning_curve_long)}, "
    f"Pool history rows: {len(pool_history_long)}"
)

# Single-run: keep mean/lower/upper schema for downstream plotting compatibility
learning_curve_summary = (
    learning_curve_long[["strategy", "n_labeled", "mae", "ktau", "r2"]]
    .rename(columns={"mae": "mae_mean", "ktau": "ktau_mean", "r2": "r2_mean"})
    .copy()
)
for metric in ["mae", "ktau", "r2"]:
    learning_curve_summary[f"{metric}_lower"] = learning_curve_summary[f"{metric}_mean"]
    learning_curve_summary[f"{metric}_upper"] = learning_curve_summary[f"{metric}_mean"]

Path("results").mkdir(exist_ok=True)

# ── Learning curves ────────────────────────────────────────────────────────────
print("Generating learning curve (MAE)...")
fig = alp.plot_learning_curve_with_bands(
    learning_curve_summary,
    metric_col="mae",
    ylabel="MAE (pEC50 units)",
    strategy_order=STRATEGIES,
    color_map=STRATEGY_COLORS,
)
fig.write_html("results/learning_curve_mae.html")

print("Generating learning curve (Kendall's τ)...")
fig = alp.plot_learning_curve_with_bands(
    learning_curve_summary,
    metric_col="ktau",
    ylabel="Kendall's τ",
    strategy_order=STRATEGIES,
    color_map=STRATEGY_COLORS,
)
fig.write_html("results/learning_curve_ktau.html")

# ── Hit discovery curve ────────────────────────────────────────────────────────
print("Generating hit discovery curve...")
fig = alp.plot_hit_discovery_curve(
    pool_history_long,
    learning_curve_long,
    hit_threshold=7.0,
    strategy_order=STRATEGIES,
    color_map=STRATEGY_COLORS,
)
fig.write_html("results/hit_discovery_curve.html")

# ── GTM selection animation ────────────────────────────────────────────────────
_method = "Exploitation"
print(f"Generating GTM animation ({_method})...")
fig = alp.plot_gtm_selection_animation(
    gtm_coords=gtm_coords_pool,
    background_gtm_coords=gtm_coords_background,
    selection_history=all_runs[_method]["history"],
    title=f"{_method} Compound Selection in GTM Chemical Space",
)
fig.update_layout(autosize=False, width=800, height=800)
fig.write_html(f"results/gtm_selection_animation_{_method.lower()}.html")

# ── TMAP (Faerun) ──────────────────────────────────────────────────────────────
print("Computing TMAP layout...")
tmap_layout = smiles_to_tmap(list(df_pool["smiles"].values) + background_smiles)

print("Generating TMAP visualization...")
alp.plot_tmap_faerun(
    tmap_layout,
    n_background=len(background_smiles),
    smiles_list=df_pool["smiles"].values,
    selection_history=all_runs["EI"]["history"],
    point_scale=3,
    background_point_scale=1,
    title="Active Learning Selection (TMAP)",
    output_name="tmap_selection",
    output_path="./results/",
)

# ── Calibration ────────────────────────────────────────────────────────────────
committee_final = all_runs["EI"]["committee"]

print("Evaluating *before* calibration...")
res_pre = evaluate_on_test(committee_final, df_test["smiles"], df_test["pEC50"])

y_cal_arr = df_cal["pEC50"].values.reshape(-1, 1)
X_cal_loader, _ = featurize(df_cal["smiles"].values)

print("Calibrating...")
committee_calibrated = copy.deepcopy(committee_final)
committee_calibrated.calibrate_uncertainty(
    X_cal_loader, y_cal_arr, method="scaling-factor"
)

print("Evaluating *after* calibration...")
res_post = evaluate_on_test(committee_calibrated, df_test["smiles"], df_test["pEC50"])

exp_pre, obs_pre = uct.metrics_calibration.get_proportion_lists_vectorized(
    res_pre["y_test_pred"], res_pre["y_test_std"], df_test["pEC50"].values
)
exp_post, obs_post = uct.metrics_calibration.get_proportion_lists_vectorized(
    res_post["y_test_pred"], res_post["y_test_std"], df_test["pEC50"].values
)

fig = alp.plot_calibration_curve_before_after(exp_pre, obs_pre, exp_post, obs_post)
fig.write_html("results/calibration_curve.html")

print(f"\nMiscalibration Area Before: {res_pre['miscal_area']:.4f}")
print(f"Miscalibration Area After:  {res_post['miscal_area']:.4f}")

print("\nAll visualizations saved to results/")
