#!/usr/bin/env python
"""Analysis and visualization entry point.

Loads the active learning results from ``results/all_runs.pkl`` (produced by
``run.py``) and generates all figures referenced in ``blogpost.md``.

Usage
-----
    python analysis.py

Generated outputs (written to ``results/``):
    learning_curve_mae.html / .png            — MAE learning curves per strategy
    learning_curve_ktau.html / .png           — Kendall's τ learning curves per strategy
    hit_discovery_curve.html / .png           — cumulative hits vs. labeled-pool size
    gtm_selection_animation_exploitation.html / .png — animated GTM selection (Exploitation)
    tmap_selection.html / .png                — interactive TMAP (EI, via Faerun)
    calibration_curve.html / .png             — before/after isotonic calibration

Requires ``kaleido`` for PNG export (``pip install kaleido``).
"""

import asyncio
import copy
import logging
import os
import pickle
import warnings
from pathlib import Path

# kaleido v1 / choreographer emit noisy INFO logs; silence them
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from kaleido import Kaleido

import src.plots as alp
from src.helpers import (
    STRATEGIES,
    STRATEGY_COLORS,
    evaluate_on_test,
    featurize,
    smiles_to_tmap,
)

warnings.filterwarnings("ignore")


def main() -> None:
    # ── Load results ───────────────────────────────────────────────────────────
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
        learning_curve_summary[f"{metric}_lower"] = learning_curve_summary[
            f"{metric}_mean"
        ]
        learning_curve_summary[f"{metric}_upper"] = learning_curve_summary[
            f"{metric}_mean"
        ]

    Path("results").mkdir(exist_ok=True)

    # Accumulate (fig, png_path) pairs; all PNG writes are batched at the end in a
    # single Kaleido() session to avoid spawning a new subprocess per figure.
    _plotly_pngs: list[tuple] = []

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
    _plotly_pngs.append((fig, "results/learning_curve_mae.png"))

    print("Generating learning curve (Kendall's τ)...")
    fig = alp.plot_learning_curve_with_bands(
        learning_curve_summary,
        metric_col="ktau",
        ylabel="Kendall's τ",
        strategy_order=STRATEGIES,
        color_map=STRATEGY_COLORS,
    )
    fig.write_html("results/learning_curve_ktau.html")
    _plotly_pngs.append((fig, "results/learning_curve_ktau.png"))

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
    _plotly_pngs.append((fig, "results/hit_discovery_curve.png"))

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
    # GTM animation is animated; PNG would only capture an empty first frame.
    # A dedicated matplotlib static snapshot is generated below instead.

    # Static snapshot: final-state scatter colored by iteration
    print(f"Generating static GTM snapshot ({_method})...")
    _gtm_history = all_runs[_method]["history"]
    _gtm_iters = sorted({s["iteration"] for s in _gtm_history})
    _iter_to_cat = {it: idx + 1 for idx, it in enumerate(_gtm_iters)}
    _n_iter = len(_gtm_iters)
    _c = np.zeros(len(gtm_coords_pool), dtype=int)
    for _state in _gtm_history:
        for _idx in _state["selected_pool_indices"]:
            if _c[_idx] == 0:
                _c[_idx] = _iter_to_cat[_state["iteration"]]
    _base_cmap = mcm.get_cmap("viridis")
    _iter_colors = [_base_cmap(i / max(_n_iter - 1, 1)) for i in range(_n_iter)]
    _colors_list = [(0.75, 0.75, 0.75, 0.35)] + _iter_colors
    _point_colors = [_colors_list[ci] for ci in _c]
    fig_gtm_static, ax_gtm = plt.subplots(figsize=(8, 8))
    if len(gtm_coords_background) > 0:
        _bg_all = np.concatenate([gtm_coords_pool, gtm_coords_background], axis=0)
        ax_gtm.scatter(
            _bg_all[:, 0],
            _bg_all[:, 1],
            color=(0.75, 0.75, 0.75),
            s=4,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )
    else:
        ax_gtm.scatter(
            gtm_coords_pool[:, 0],
            gtm_coords_pool[:, 1],
            color=(0.75, 0.75, 0.75),
            s=4,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )
    ax_gtm.scatter(
        gtm_coords_pool[:, 0],
        gtm_coords_pool[:, 1],
        c=_point_colors,
        s=8,
        linewidths=0,
        zorder=2,
    )
    ax_gtm.set_title(f"{_method} Compound Selection in GTM Chemical Space")
    ax_gtm.set_xlabel("GTM dimension 1")
    ax_gtm.set_ylabel("GTM dimension 2")
    ax_gtm.set_facecolor("white")
    fig_gtm_static.patch.set_facecolor("white")
    _sm = plt.cm.ScalarMappable(
        cmap=_base_cmap, norm=mcolors.Normalize(vmin=0, vmax=_n_iter - 1)
    )
    _sm.set_array([])
    fig_gtm_static.colorbar(_sm, ax=ax_gtm, label="AL Iteration")
    fig_gtm_static.savefig(
        f"results/gtm_selection_animation_{_method.lower()}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_gtm_static)

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

    # Static TMAP snapshot colored by AL iteration
    print("Generating static TMAP snapshot...")
    _tx, _ty, _ts, _tt = tmap_layout
    _ei_history = all_runs["EI"]["history"]
    _ei_iters = sorted({s["iteration"] for s in _ei_history})
    _ei_iter_to_cat = {it: idx + 1 for idx, it in enumerate(_ei_iters)}
    _n_ei_iter = len(_ei_iters)
    _ct = np.zeros(len(_tx), dtype=int)
    for _state in _ei_history:
        for _idx in _state["selected_pool_indices"]:
            if _ct[_idx] == 0:
                _ct[_idx] = _ei_iter_to_cat[_state["iteration"]]
    _tmap_base_cmap = mcm.get_cmap("viridis")
    _tmap_colors = [(0.75, 0.75, 0.75, 0.15)] + [
        _tmap_base_cmap(i / max(_n_ei_iter - 1, 1)) for i in range(_n_ei_iter)
    ]
    _tmap_point_colors = [_tmap_colors[ci] for ci in _ct]
    fig_tmap_static, ax_tmap = plt.subplots(figsize=(10, 10))
    for _si, _ti in zip(_ts, _tt):
        ax_tmap.plot(
            [_tx[_si], _tx[_ti]],
            [_ty[_si], _ty[_ti]],
            color="gray",
            lw=0.15,
            alpha=0.3,
            zorder=1,
        )
    ax_tmap.scatter(_tx, _ty, c=_tmap_point_colors, s=2, linewidths=0, zorder=2)
    ax_tmap.set_title("Active Learning Selection (TMAP)")
    ax_tmap.axis("off")
    fig_tmap_static.patch.set_facecolor("white")
    _tsm = plt.cm.ScalarMappable(
        cmap=_tmap_base_cmap, norm=mcolors.Normalize(vmin=0, vmax=_n_ei_iter - 1)
    )
    _tsm.set_array([])
    fig_tmap_static.colorbar(_tsm, ax=ax_tmap, label="AL Iteration (EI)")
    fig_tmap_static.savefig("results/tmap_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig_tmap_static)

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
    res_post = evaluate_on_test(
        committee_calibrated, df_test["smiles"], df_test["pEC50"]
    )

    exp_pre, obs_pre = uct.metrics_calibration.get_proportion_lists_vectorized(
        res_pre["y_test_pred"], res_pre["y_test_std"], df_test["pEC50"].values
    )
    exp_post, obs_post = uct.metrics_calibration.get_proportion_lists_vectorized(
        res_post["y_test_pred"], res_post["y_test_std"], df_test["pEC50"].values
    )

    fig = alp.plot_calibration_curve_before_after(exp_pre, obs_pre, exp_post, obs_post)
    fig.write_html("results/calibration_curve.html")
    _plotly_pngs.append((fig, "results/calibration_curve.png"))

    print(f"\nMiscalibration Area Before: {res_pre['miscal_area']:.4f}")
    print(f"Miscalibration Area After:  {res_post['miscal_area']:.4f}")

    # ── Batch PNG export (single kaleido process) ──────────────────────────────────
    print(f"\nExporting {len(_plotly_pngs)} Plotly figures to PNG...")

    async def _export_pngs():
        async with Kaleido() as k:
            for _fig, _png_path in _plotly_pngs:
                await k.write_fig(_fig, _png_path)
                print(f"  saved {_png_path}")

    asyncio.run(_export_pngs())

    print("\nAll visualizations saved to results/")
    # kaleido v1 leaves a non-daemon background thread that prevents a clean exit;
    # os._exit(0) terminates immediately after all writes complete.
    os._exit(0)


if __name__ == "__main__":
    main()
