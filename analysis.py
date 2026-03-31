#!/usr/bin/env python
"""Analysis and visualization entry point.

Loads ``results/setup_<split_type>.pkl`` files and all
``results/run_<split_type>_*.pkl`` files produced by ``run.py``, performs
sanity checks on coverage and consistency, then generates all figures
referenced in ``blogpost.md``.

Each performance-metric figure is produced as a two-panel side-by-side plot
when both scaffold and random split results are present (scaffold = left,
random = right). Single-strategy figures (GTM, TMAP, calibration curve) use
the scaffold split only (fallback to the first available split type).

Usage
-----
    python analysis.py

Generated outputs (written to ``results/``)::
    learning_curve_mae.html / .svg            — MAE learning curves per strategy
    learning_curve_ktau.html / .svg           — Kendall tau learning curves per strategy
    hit_discovery_curve.html / .svg           — cumulative hits vs. labeled-pool size
    calibration_area_per_iteration.html / .svg — miscalibration area per iteration
    sigma_error_correlation.html / .svg       — σ–|error| Spearman ρ per strategy
    gtm_selection_animation_exploitation.html / .svg — animated GTM (Exploitation)
    tmap_selection.html / .svg                — interactive TMAP (Exploitation, via Faerun)
    calibration_curve.html / .svg             — before/after scaling factor calibration

Requires ``kaleido`` for SVG export (``pip install kaleido``).
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

import matplotlib.cm as mcm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import uncertainty_toolbox as uct  # noqa: E402
from kaleido import Kaleido  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import src.plots as alp  # noqa: E402
from src.config import ALConfig  # noqa: E402
from src.helpers import (  # noqa: E402
    STRATEGY_COLORS,
    smiles_to_tmap,
)

warnings.filterwarnings("ignore")


def _combine_figures_side_by_side(
    fig_left: go.Figure,
    fig_right: go.Figure,
    title_left: str = "Scaffold Split",
    title_right: str = "Random Split",
) -> go.Figure:
    """Merge two Plotly figures into a two-column subplot.

    Traces from ``fig_left`` occupy the left panel; traces from ``fig_right``
    occupy the right panel. Legend entries are shown only once (left panel),
    so the right panel entries are suppressed. ``fill="tonexty"`` bands are
    handled correctly because Plotly restricts fills to traces sharing the
    same axis reference.

    Parameters
    ----------
    fig_left : go.Figure
        Figure for the left panel (typically scaffold split).
    fig_right : go.Figure
        Figure for the right panel (typically random split).
    title_left : str
        Subplot title for the left panel.
    title_right : str
        Subplot title for the right panel.

    Returns
    -------
    go.Figure
        Combined two-panel figure with doubled width.

    """
    combined = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[title_left, title_right],
        horizontal_spacing=0.10,
    )
    for trace in fig_left.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig_right.data:
        combined.add_trace(trace, row=1, col=2)

    # Suppress all right-panel legend entries — already shown in the left panel
    n_left = len(fig_left.data)
    for i in range(n_left, len(combined.data)):
        combined.data[i].showlegend = False

    # Mirror axis labels from the left figure to both panels
    x_title = fig_left.layout.xaxis.title.text or ""
    y_title = fig_left.layout.yaxis.title.text or ""
    combined.update_xaxes(title_text=x_title)
    combined.update_yaxes(title_text=y_title)

    w = fig_left.layout.width or 700
    h = fig_left.layout.height or 450
    combined.update_layout(width=w * 2, height=h)
    return combined


def main() -> None:
    """Load AL results, run sanity checks, assemble DataFrames, and write all figures.

    Reads all ``results/setup_<split_type>.pkl`` files and all matching
    ``results/run_<split_type>_*.pkl`` files produced by ``run.py``. For each
    split type, performs sanity checks, assembles tidy DataFrames, and
    aggregates per-strategy statistics. Multi-strategy figures are combined
    into two-panel side-by-side plots (scaffold left, random right) when both
    splits are available; single-strategy figures use the scaffold split.
    """
    # ── Load all setup pickles ─────────────────────────────────────────────────
    setups: dict[str, dict] = {}
    for setup_pkl in sorted(Path("results").glob("setup_*.pkl")):
        split_type = setup_pkl.stem[len("setup_"):]
        with open(setup_pkl, "rb") as fh:
            setups[split_type] = pickle.load(fh)

    if not setups:
        raise FileNotFoundError(
            "No results/setup_*.pkl files found. "
            "Run `python run.py --setup-only` first."
        )

    cfg: ALConfig = next(iter(setups.values()))["config"]
    available_split_types = list(setups.keys())
    print(f"Split types loaded: {available_split_types}")

    for split_type, setup in setups.items():
        df_seed = setup.get("df_seed")
        if df_seed is not None:
            print(
                f"[{split_type}] External seed data: {len(df_seed)} compounds "
                "(used in training, not counted in n_labeled on learning curves)."
            )

    # ── Load per-job results ───────────────────────────────────────────────────
    # Filename format: run_<split_type>_<strategy>_seed<N>.pkl
    # split_type is all-lowercase; strategy may be mixed-case with no underscores
    pattern = re.compile(r"^run_([a-z]+)_([A-Za-z]+)_seed(\d+)\.pkl$")
    job_results: dict[tuple[str, str, int], dict] = {}

    for pkl_file in sorted(Path("results").glob("run_*.pkl")):
        m = pattern.match(pkl_file.name)
        if not m:
            continue
        sp, strat, seed = m.group(1), m.group(2), int(m.group(3))
        with open(pkl_file, "rb") as fh:
            job_data = pickle.load(fh)
        job_results[(sp, strat, seed)] = job_data["result"]

    if not job_results:
        raise FileNotFoundError(
            "No results/run_*.pkl files found. "
            "Run `python run.py` (or individual jobs) first."
        )

    # ── Sanity checks (per split type) ────────────────────────────────────────
    warned = False
    for split_type in available_split_types:
        print(f"\n[{split_type}] Sanity checks:")
        split_jobs = {
            (st, sd): r
            for (sp, st, sd), r in job_results.items()
            if sp == split_type
        }

        missing_strategies = [
            s for s in cfg.strategies if not any(k[0] == s for k in split_jobs)
        ]
        if missing_strategies:
            print(f"  WARNING: Missing strategies entirely: {missing_strategies}")
            warned = True

        seeds_per_strategy: dict[str, list[int]] = {
            s: sorted(sd for (st, sd) in split_jobs if st == s)
            for s in cfg.strategies
        }
        present_counts = {s: len(v) for s, v in seeds_per_strategy.items() if v}
        unique_counts = set(present_counts.values())

        print("  Seed coverage:")
        for s in cfg.strategies:
            n = len(seeds_per_strategy[s])
            seeds_str = str(seeds_per_strategy[s]) if n else "[]"
            print(f"    {s:<14} {n} seed(s): {seeds_str}")

        if len(unique_counts) > 1:
            print(
                f"  WARNING: Seed count mismatch across strategies — "
                f"counts are {present_counts}."
            )
            warned = True

        steps_per_run = {
            (st, sd): len(r["history"]) for (st, sd), r in split_jobs.items()
        }
        unique_step_counts = set(steps_per_run.values())
        if len(unique_step_counts) > 1:
            print("  WARNING: Iteration count differs across runs:")
            for (st, sd), n in sorted(steps_per_run.items()):
                print(f"    {st} seed={sd}: {n} iterations")
            warned = True
        elif unique_step_counts:
            print(f"  Iterations per run: {next(iter(unique_step_counts))}")

    if not warned:
        print("Sanity checks passed.")

    # ── Build per-split DataFrames ─────────────────────────────────────────────
    per_split_data: dict[str, dict] = {}

    for split_type, setup in setups.items():
        _df_pool = setup["df_pool"]
        _df_test = setup["df_test"]
        _gtm_coords = setup["gtm_coords_pool"]

        split_jobs = {
            (st, sd): r
            for (sp, st, sd), r in job_results.items()
            if sp == split_type
        }
        seeds_per_strategy = {
            s: sorted(sd for (st, sd) in split_jobs if st == s)
            for s in cfg.strategies
        }

        _all_runs: dict[str, list] = {}
        for strategy in cfg.strategies:
            runs = [
                {"seed": seed, **split_jobs[(strategy, seed)]}
                for seed in seeds_per_strategy[strategy]
                if (strategy, seed) in split_jobs
            ]
            if runs:
                _all_runs[strategy] = runs

        print(f"\n[{split_type}] Strategies loaded: {list(_all_runs.keys())}")

        _records = []
        _pool_history_records = []
        for strategy in cfg.strategies:
            if strategy not in _all_runs:
                continue
            for run in _all_runs[strategy]:
                seed = run["seed"]
                for step in run["history"]:
                    _abs_err = np.abs(step["y_test_pred"] - _df_test["pEC50"].values)
                    _rho = spearmanr(step["y_test_std"], _abs_err).statistic
                    _records.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "iteration": step["iteration"],
                            "n_labeled": step["n_labeled"],
                            "mae": step["mae"],
                            "r2": step["r2"],
                            "ktau": step["ktau"],
                            "spearmanr": step["spearmanr"],
                            "miscal_area": step["miscal_area"],
                            "miscal_area_pre_cal": step["miscal_area_pre_cal"],
                            "sigma_error_rho": float(_rho),
                        }
                    )
                    for pval in step["pool_y_values"]:
                        _pool_history_records.append(
                            {
                                "strategy": strategy,
                                "seed": seed,
                                "iteration": step["iteration"],
                                "pEC50": float(pval),
                            }
                        )

        _lc_long = pd.DataFrame(_records)
        _ph_long = pd.DataFrame(_pool_history_records)
        print(
            f"[{split_type}] Curve rows: {len(_lc_long)}, "
            f"Pool history rows: {len(_ph_long)}"
        )

        _lc_summary = (
            _lc_long.groupby(["strategy", "n_labeled"])
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                ktau_mean=("ktau", "mean"),
                ktau_std=("ktau", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                miscal_area_mean=("miscal_area", "mean"),
                miscal_area_std=("miscal_area", "std"),
                miscal_area_pre_cal_mean=("miscal_area_pre_cal", "mean"),
                miscal_area_pre_cal_std=("miscal_area_pre_cal", "std"),
                sigma_error_rho_mean=("sigma_error_rho", "mean"),
                sigma_error_rho_std=("sigma_error_rho", "std"),
            )
            .reset_index()
            .fillna(0)
        )
        for metric in [
            "mae", "ktau", "r2", "miscal_area", "miscal_area_pre_cal", "sigma_error_rho"
        ]:
            _lc_summary[f"{metric}_lower"] = (
                _lc_summary[f"{metric}_mean"] - _lc_summary[f"{metric}_std"]
            )
            _lc_summary[f"{metric}_upper"] = (
                _lc_summary[f"{metric}_mean"] + _lc_summary[f"{metric}_std"]
            )

        per_split_data[split_type] = {
            "df_pool": _df_pool,
            "df_test": _df_test,
            "gtm_coords_pool": _gtm_coords,
            "all_runs": _all_runs,
            "learning_curve_long": _lc_long,
            "pool_history_long": _ph_long,
            "learning_curve_summary": _lc_summary,
        }

    # Primary split for single-strategy figures (scaffold preferred)
    _primary = "scaffold" if "scaffold" in per_split_data else next(iter(per_split_data))
    df_pool = per_split_data[_primary]["df_pool"]
    df_test = per_split_data[_primary]["df_test"]
    gtm_coords_pool = per_split_data[_primary]["gtm_coords_pool"]
    all_runs = per_split_data[_primary]["all_runs"]

    Path("results").mkdir(exist_ok=True)

    # Accumulate (fig, svg_path) pairs; all SVG writes are batched at the end in a
    # single Kaleido() session to avoid spawning a new subprocess per figure
    _plotly_svgs: list[tuple] = []

    def _save_multisplit(figs: dict[str, go.Figure], fname: str) -> None:
        """Combine split-type figures and write HTML + queue SVG export."""
        if "scaffold" in figs and "random" in figs:
            out = _combine_figures_side_by_side(figs["scaffold"], figs["random"])
        elif figs:
            out = next(iter(figs.values()))
        else:
            return
        out.write_html(f"results/{fname}.html")
        _plotly_svgs.append((out, f"results/{fname}.svg"))

    # ── Learning curves ────────────────────────────────────────────────────────────
    print("Generating learning curve (MAE)...")
    _figs: dict[str, go.Figure] = {}
    for _sp in ["scaffold", "random"]:
        if _sp not in per_split_data:
            continue
        _figs[_sp] = alp.plot_learning_curve_with_bands(
            per_split_data[_sp]["learning_curve_summary"],
            metric_col="mae",
            ylabel="MAE (pEC50 units)",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(_figs, "learning_curve_mae")

    print("Generating learning curve (Kendall's τ)...")
    _figs = {}
    for _sp in ["scaffold", "random"]:
        if _sp not in per_split_data:
            continue
        _figs[_sp] = alp.plot_learning_curve_with_bands(
            per_split_data[_sp]["learning_curve_summary"],
            metric_col="ktau",
            ylabel="Kendall's τ",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(_figs, "learning_curve_ktau")

    # ── Hit discovery curve ────────────────────────────────────────────────────────
    print("Generating hit discovery curve...")
    _hit_threshold = 6.3
    _figs = {}
    for _sp in ["scaffold", "random"]:
        if _sp not in per_split_data:
            continue
        _max_hits_sp = int(
            (per_split_data[_sp]["df_pool"]["pEC50"] >= _hit_threshold).sum()
        )
        _figs[_sp] = alp.plot_hit_discovery_curve(
            per_split_data[_sp]["pool_history_long"],
            per_split_data[_sp]["learning_curve_long"],
            hit_threshold=_hit_threshold,
            max_hits=_max_hits_sp,
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(_figs, "hit_discovery_curve")

    # ── GTM selection animation (primary split only) ───────────────────────────────
    _method = "Exploitation"
    print(f"Generating GTM animation ({_method}, {_primary} split)...")
    fig = alp.plot_gtm_selection_animation(
        gtm_coords=gtm_coords_pool,
        selection_history=all_runs[_method][0]["history"],
        title=f"{_method} Compound Selection in GTM Chemical Space",
    )
    fig.update_layout(autosize=False, width=800, height=800)
    fig.write_html(f"results/gtm_selection_animation_{_method.lower()}.html")
    # GTM animation is animated; PNG would only capture an empty first frame
    # A dedicated matplotlib static snapshot is generated below instead

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
    ax_gtm.set_xlabel("GTM dimension 1")
    ax_gtm.set_ylabel("GTM dimension 2")
    ax_gtm.set_facecolor("white")
    fig_gtm_static.patch.set_facecolor("white")
    ax_gtm.set_box_aspect(1)
    _sm = plt.cm.ScalarMappable(
        cmap=_base_cmap, norm=mcolors.Normalize(vmin=0, vmax=_n_iter - 1)
    )
    _sm.set_array([])
    _gtm_divider = make_axes_locatable(ax_gtm)
    _gtm_cax = _gtm_divider.append_axes("right", size="5%", pad=0.1)
    fig_gtm_static.colorbar(
        _sm, cax=_gtm_cax, label="AL Iteration"
    ).ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_gtm_static.savefig(
        f"results/gtm_selection_animation_{_method.lower()}.svg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_gtm_static)

    # ── TMAP (Faerun, primary split only) ──────────────────────────────────────────
    print("Computing TMAP layout...")
    tmap_layout = smiles_to_tmap(list(df_pool["smiles"].values))

    print("Generating TMAP visualization...")
    alp.plot_tmap_faerun(
        tmap_layout,
        n_background=0,
        smiles_list=df_pool["smiles"].values,
        selection_history=all_runs["Exploitation"][0]["history"],
        point_scale=3,
        background_point_scale=1,
        title="Active Learning Selection (TMAP)",
        output_name="tmap_selection",
        output_path="./results/",
    )

    # Static TMAP snapshot colored by AL iteration
    print("Generating static TMAP snapshot...")
    _tx, _ty, _ts, _tt = tmap_layout
    _exploitation_history = all_runs["Exploitation"][0]["history"]
    _exploitation_iters = sorted({s["iteration"] for s in _exploitation_history})
    _exploitation_iter_to_cat = {
        it: idx + 1 for idx, it in enumerate(_exploitation_iters)
    }
    _n_exploitation_iter = len(_exploitation_iters)
    _ct = np.zeros(len(_tx), dtype=int)
    for _state in _exploitation_history:
        for _idx in _state["selected_pool_indices"]:
            if _ct[_idx] == 0:
                _ct[_idx] = _exploitation_iter_to_cat[_state["iteration"]]
    _tmap_base_cmap = mcm.get_cmap("viridis")
    _tmap_colors = [(0.75, 0.75, 0.75, 0.15)] + [
        _tmap_base_cmap(i / max(_n_exploitation_iter - 1, 1))
        for i in range(_n_exploitation_iter)
    ]
    _tmap_point_colors = [_tmap_colors[ci] for ci in _ct]
    fig_tmap_static, ax_tmap = plt.subplots(figsize=(8.5, 8.5))
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
    ax_tmap.axis("off")
    fig_tmap_static.patch.set_facecolor("white")
    ax_tmap.set_box_aspect(1)
    _tsm = plt.cm.ScalarMappable(
        cmap=_tmap_base_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=_n_exploitation_iter - 1),
    )
    _tsm.set_array([])
    _tmap_divider = make_axes_locatable(ax_tmap)
    _tmap_cax = _tmap_divider.append_axes("right", size="5%", pad=0.1)
    fig_tmap_static.colorbar(
        _tsm, cax=_tmap_cax, label="AL Iteration (Exploitation)"
    ).ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_tmap_static.savefig("results/tmap_selection.svg", bbox_inches="tight")
    plt.close(fig_tmap_static)

    # ── Calibration (primary split only) ──────────────────────────────────────────
    # Calibration is performed per-iteration inside run_active_learning
    # Visualize before/after using stored predictions from the final Exploitation iteration
    final_state = all_runs["Exploitation"][0]["history"][-1]

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
    _plotly_svgs.append((fig, "results/calibration_curve.svg"))

    print(f"\nMiscalibration Area Before: {final_state['miscal_area_pre_cal']:.4f}")
    print(f"Miscalibration Area After:  {final_state['miscal_area']:.4f}")

    # ── Calibration area per iteration ─────────────────────────────────────────────
    print("Generating calibration area per iteration plot...")
    _figs = {}
    for _sp in ["scaffold", "random"]:
        if _sp not in per_split_data:
            continue
        _figs[_sp] = alp.plot_calibration_area_per_iteration(
            per_split_data[_sp]["learning_curve_summary"],
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(_figs, "calibration_area_per_iteration")

    # ── Sigma–error correlation ────────────────────────────────────────────────────
    print("Generating sigma–error correlation plot...")
    _figs = {}
    for _sp in ["scaffold", "random"]:
        if _sp not in per_split_data:
            continue
        _figs[_sp] = alp.plot_sigma_error_correlation(
            per_split_data[_sp]["learning_curve_summary"],
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(_figs, "sigma_error_correlation")

    # ── Batch PNG export (single kaleido process) ──────────────────────────────────
    print(f"\nExporting {len(_plotly_svgs)} Plotly figures to PNG...")

    # Export all Plotly figures in a single Kaleido subprocess session for efficiency
    async def _export_svgs():
        async with Kaleido() as k:
            for _fig, _svg_path in _plotly_svgs:
                await k.write_fig(_fig, _svg_path)
                print(f"  saved {_svg_path}")

    asyncio.run(_export_svgs())

    print("\nAll visualizations saved to results/")
    # kaleido v1 leaves a non-daemon background thread that prevents a clean exit;
    # os._exit(0) terminates immediately after all writes complete
    os._exit(0)


if __name__ == "__main__":
    main()
