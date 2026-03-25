#!/usr/bin/env python
"""Analysis and visualization entry point.

Loads ``results/setup.pkl`` and all ``results/run_*.pkl`` files produced by
``run.py``, performs sanity checks on coverage and consistency, then generates
all figures referenced in ``blogpost.md``.

Usage
-----
    python analysis.py

Generated outputs (written to ``results/``)::
    learning_curve_mae.html / .png            — MAE learning curves per strategy
    learning_curve_ktau.html / .png           — Kendall’s τ learning curves per strategy
    hit_discovery_curve.html / .png           — cumulative hits vs. labeled-pool size
    gtm_selection_animation_exploitation.html / .png — animated GTM (Exploitation)
    tmap_selection.html / .png                — interactive TMAP (EI, via Faerun)
    calibration_curve.html / .png             — before/after isotonic calibration

Requires ``kaleido`` for PNG export (``pip install kaleido``).
"""

import asyncio
import logging
import os
import pickle
import re
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
from src.config import ALConfig
from src.helpers import (
    STRATEGY_COLORS,
    smiles_to_tmap,
)

warnings.filterwarnings("ignore")


def main() -> None:
    # ── Load setup ─────────────────────────────────────────────────────────────
    setup_path = Path("results/setup.pkl")
    if not setup_path.exists():
        raise FileNotFoundError(
            "results/setup.pkl not found. Run `python run.py --setup-only` first."
        )
    with open(setup_path, "rb") as fh:
        setup = pickle.load(fh)

    df_pool = setup["df_pool"]
    df_test = setup["df_test"]
    df_seed = setup.get("df_seed")
    gtm_coords_pool = setup["gtm_coords_pool"]
    cfg: ALConfig = setup["config"]

    if df_seed is not None:
        print(
            f"External seed data: {len(df_seed)} compounds "
            "(used in training, not counted in n_labeled on learning curves)."
        )

    # ── Load per-job results ───────────────────────────────────────────────────
    pattern = re.compile(r"^run_(.+)_seed(\d+)\.pkl$")
    job_results: dict[tuple[str, int], dict] = {}

    for pkl_file in sorted(Path("results").glob("run_*.pkl")):
        m = pattern.match(pkl_file.name)
        if not m:
            continue
        strat, seed = m.group(1), int(m.group(2))
        with open(pkl_file, "rb") as fh:
            job_data = pickle.load(fh)
        job_results[(strat, seed)] = job_data["result"]

    if not job_results:
        raise FileNotFoundError(
            "No results/run_*.pkl files found. "
            "Run `python run.py` (or individual jobs) first."
        )

    # ── Sanity checks ──────────────────────────────────────────────────────────
    warned = False

    # 1. Strategy coverage
    missing_strategies = [
        s for s in cfg.strategies if not any(k[0] == s for k in job_results)
    ]
    if missing_strategies:
        print(f"WARNING: Missing strategies entirely: {missing_strategies}")
        warned = True

    # 2. Seeds per strategy and cross-strategy seed-count consistency
    seeds_per_strategy: dict[str, list[int]] = {
        s: sorted(seed for (strat, seed) in job_results if strat == s)
        for s in cfg.strategies
    }
    present_counts = {s: len(v) for s, v in seeds_per_strategy.items() if v}
    unique_counts = set(present_counts.values())

    print("\nSeed coverage:")
    for s in cfg.strategies:
        n = len(seeds_per_strategy[s])
        seeds_str = str(seeds_per_strategy[s]) if n else "[]"
        print(f"  {s:<14} {n} seed(s): {seeds_str}")

    if len(unique_counts) > 1:
        print(
            f"WARNING: Seed count mismatch across strategies — "
            f"counts are {present_counts}. Learning curve bands may be inconsistent."
        )
        warned = True

    # 3. Iteration-count consistency (each run should have k_iter+1 steps)
    steps_per_run = {
        (strat, seed): len(result["history"])
        for (strat, seed), result in job_results.items()
    }
    unique_step_counts = set(steps_per_run.values())
    if len(unique_step_counts) > 1:
        print("WARNING: Iteration count differs across runs:")
        for (strat, seed), n in sorted(steps_per_run.items()):
            print(f"  {strat} seed={seed}: {n} iterations")
        warned = True
    else:
        print(f"\nIterations per run: {next(iter(unique_step_counts))}")

    if not warned:
        print("Sanity checks passed.")

    # ── Assemble all_runs ──────────────────────────────────────────────────────
    all_runs: dict[str, list] = {}
    for strategy in cfg.strategies:
        runs = [
            {"seed": seed, **job_results[(strategy, seed)]}
            for seed in seeds_per_strategy[strategy]
        ]
        if runs:
            all_runs[strategy] = runs

    print(f"\nStrategies loaded: {list(all_runs.keys())}")

    # ── Unpack into tidy DataFrames ───────────────────────────────────────────────
    records = []
    pool_history_records = []

    for strategy in cfg.strategies:
        if strategy not in all_runs:
            continue
        for run in all_runs[strategy]:
            seed = run["seed"]
            for step in run["history"]:
                records.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
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
                            "seed": seed,
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

    # Aggregate across seeds — mean ± std per (strategy, n_labeled).
    # When only one seed is present std=NaN → filled to 0 (no band).
    learning_curve_summary = (
        learning_curve_long.groupby(["strategy", "n_labeled"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ktau_mean=("ktau", "mean"),
            ktau_std=("ktau", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
        .fillna(0)
    )
    for metric in ["mae", "ktau", "r2"]:
        learning_curve_summary[f"{metric}_lower"] = (
            learning_curve_summary[f"{metric}_mean"]
            - learning_curve_summary[f"{metric}_std"]
        )
        learning_curve_summary[f"{metric}_upper"] = (
            learning_curve_summary[f"{metric}_mean"]
            + learning_curve_summary[f"{metric}_std"]
        )

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
        strategy_order=cfg.strategies,
        color_map=STRATEGY_COLORS,
    )
    fig.write_html("results/learning_curve_mae.html")
    _plotly_pngs.append((fig, "results/learning_curve_mae.png"))

    print("Generating learning curve (Kendall's τ)...")
    fig = alp.plot_learning_curve_with_bands(
        learning_curve_summary,
        metric_col="ktau",
        ylabel="Kendall's τ",
        strategy_order=cfg.strategies,
        color_map=STRATEGY_COLORS,
    )
    fig.write_html("results/learning_curve_ktau.html")
    _plotly_pngs.append((fig, "results/learning_curve_ktau.png"))

    # ── Hit discovery curve ────────────────────────────────────────────────────────
    print("Generating hit discovery curve...")
    # Pin to the first seed: plot_hit_discovery_curve groups by (strategy, iteration)
    # and counts hits, so mixing seeds would inflate counts by N_seeds.
    _vis_seed = all_runs[cfg.strategies[0]][0]["seed"]
    pool_history_vis = pool_history_long[pool_history_long["seed"] == _vis_seed]
    fig = alp.plot_hit_discovery_curve(
        pool_history_vis,
        learning_curve_long,
        hit_threshold=6.3,
        strategy_order=cfg.strategies,
        color_map=STRATEGY_COLORS,
    )
    fig.write_html("results/hit_discovery_curve.html")
    _plotly_pngs.append((fig, "results/hit_discovery_curve.png"))

    # ── GTM selection animation ────────────────────────────────────────────────────
    _method = "Exploitation"
    print(f"Generating GTM animation ({_method})...")
    fig = alp.plot_gtm_selection_animation(
        gtm_coords=gtm_coords_pool,
        selection_history=all_runs[_method][0]["history"],
        title=f"{_method} Compound Selection in GTM Chemical Space",
    )
    fig.update_layout(autosize=False, width=800, height=800)
    fig.write_html(f"results/gtm_selection_animation_{_method.lower()}.html")
    # GTM animation is animated; PNG would only capture an empty first frame.
    # A dedicated matplotlib static snapshot is generated below instead.

    # Static snapshot: final-state scatter colored by iteration
    print(f"Generating static GTM snapshot ({_method})...")
    _gtm_history = all_runs[_method][0]["history"]
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
    tmap_layout = smiles_to_tmap(list(df_pool["smiles"].values))

    print("Generating TMAP visualization...")
    alp.plot_tmap_faerun(
        tmap_layout,
        n_background=0,
        smiles_list=df_pool["smiles"].values,
        selection_history=all_runs["EI"][0]["history"],
        point_scale=3,
        background_point_scale=1,
        title="Active Learning Selection (TMAP)",
        output_name="tmap_selection",
        output_path="./results/",
    )

    # Static TMAP snapshot colored by AL iteration
    print("Generating static TMAP snapshot...")
    _tx, _ty, _ts, _tt = tmap_layout
    _ei_history = all_runs["EI"][0]["history"]
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
    # Calibration is performed per-iteration inside run_active_learning.
    # Visualize before/after using stored predictions from the final EI iteration.
    final_state = all_runs["EI"][0]["history"][-1]

    exp_pre, obs_pre = uct.metrics_calibration.get_proportion_lists_vectorized(
        final_state["y_test_pred_pre_cal"],
        final_state["y_test_std_pre_cal"],
        df_test["pEC50"].values,
    )
    exp_post, obs_post = uct.metrics_calibration.get_proportion_lists_vectorized(
        final_state["y_test_pred"],
        final_state["y_test_std"],
        df_test["pEC50"].values,
    )

    fig = alp.plot_calibration_curve_before_after(exp_pre, obs_pre, exp_post, obs_post)
    fig.write_html("results/calibration_curve.html")
    _plotly_pngs.append((fig, "results/calibration_curve.png"))

    print(f"\nMiscalibration Area Before: {final_state['miscal_area_pre_cal']:.4f}")
    print(f"Miscalibration Area After:  {final_state['miscal_area']:.4f}")

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
