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
    tmap_partition.html / .svg                — TMAP colored by train / test partition
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


def _remap_for_col2(d: dict) -> dict:
    """Remap single-panel axis refs to second-column subplot refs.

    ``add_hline`` / ``add_vline`` write shapes and annotations that reference
    ``xref="x domain"`` / ``yref="y"``.  When those objects are transplanted
    into the right panel of a two-column subplot the refs must point to the
    second set of axes (``x2 domain`` / ``y2``).

    Parameters
    ----------
    d : dict
        Shape or annotation dict (from ``to_plotly_json()``).

    Returns
    -------
    dict
        Copy of *d* with ``xref`` and ``yref`` remapped for column 2.
    """
    _MAP = {"x": "x2", "x domain": "x2 domain", "y": "y2"}
    out = dict(d)
    if "xref" in out:
        out["xref"] = _MAP.get(out["xref"], out["xref"])
    if "yref" in out and out["yref"] != "paper":
        out["yref"] = _MAP.get(out["yref"], out["yref"])
    return out


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

    Shapes and annotations (e.g. ``add_hline`` reference lines) are copied
    from both source figures with axis references adjusted for the correct
    subplot column.  Background colour and axis styling (grid colour, border
    lines, etc.) are inherited from ``fig_left`` so the combined figure
    matches the appearance of its source panels.

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

    # Left-justify subplot titles: move each from panel-centre to panel-left edge.
    # make_subplots always places the two titles as the first two annotations.
    _left_edge = combined.layout.xaxis.domain[0]
    _right_edge = combined.layout.xaxis2.domain[0]
    _title_annots = []
    for i, (annot, x_pos) in enumerate(
        zip(combined.layout.annotations, [_left_edge, _right_edge])
    ):
        d = annot.to_plotly_json()
        d["x"] = x_pos
        d["xanchor"] = "left"
        _title_annots.append(d)
    combined.update_layout(annotations=_title_annots)

    for trace in fig_left.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig_right.data:
        combined.add_trace(trace, row=1, col=2)

    # Suppress all right-panel legend entries — already shown in the left panel
    n_left = len(fig_left.data)
    for i in range(n_left, len(combined.data)):
        combined.data[i].showlegend = False

    # Copy shapes (e.g. add_hline reference lines) from both source figures.
    # Col-1 refs are already correct; col-2 refs need remapping.
    left_shapes = [s.to_plotly_json() for s in (fig_left.layout.shapes or [])]
    right_shapes = [
        _remap_for_col2(s.to_plotly_json()) for s in (fig_right.layout.shapes or [])
    ]
    if left_shapes or right_shapes:
        combined.update_layout(shapes=left_shapes + right_shapes)

    # Copy annotations (e.g. add_hline labels) from both source figures.
    # Append to the existing subplot-title annotations added by make_subplots.
    left_annots = [a.to_plotly_json() for a in (fig_left.layout.annotations or [])]
    right_annots = [
        _remap_for_col2(a.to_plotly_json())
        for a in (fig_right.layout.annotations or [])
    ]
    if left_annots or right_annots:
        existing = list(combined.layout.annotations or [])
        combined.update_layout(annotations=existing + left_annots + right_annots)

    # Propagate background colours from the left source figure
    combined.update_layout(
        plot_bgcolor=fig_left.layout.plot_bgcolor or "white",
        paper_bgcolor=fig_left.layout.paper_bgcolor or "white",
    )

    # Propagate axis style (grid, border lines) from the left source figure to
    # all panels.  Structural properties (title, domain, anchor) are excluded
    # because make_subplots assigns those and they differ per panel.
    _SKIP = {"title", "domain", "anchor", "matches", "scaleanchor", "scaleratio"}
    _xstyle = {
        k: v
        for k, v in fig_left.layout.xaxis.to_plotly_json().items()
        if k not in _SKIP
    }
    _ystyle = {
        k: v
        for k, v in fig_left.layout.yaxis.to_plotly_json().items()
        if k not in _SKIP
    }
    if _xstyle:
        combined.update_xaxes(**_xstyle)
    if _ystyle:
        combined.update_yaxes(**_ystyle)

    # Mirror axis labels from the left figure to both panels
    x_title = fig_left.layout.xaxis.title.text or ""
    y_title = fig_left.layout.yaxis.title.text or ""
    combined.update_xaxes(title_text=x_title)
    combined.update_yaxes(title_text=y_title)

    w = fig_left.layout.width or 700
    h = fig_left.layout.height or 450
    _bottom_margin = (
        fig_left.layout.margin.b if fig_left.layout.margin.b is not None else 80
    )
    combined.update_layout(width=w * 2, height=h, margin=dict(t=_bottom_margin))
    return combined


def load_setups(results_dir: str | Path = "results") -> dict[str, dict]:
    """Load all setup pickles from *results_dir*.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``setup_<split_type>.pkl`` files.

    Returns
    -------
    dict[str, dict]
        Mapping from split type (e.g. ``"scaffold"``, ``"random"``) to the
        unpickled setup dictionary containing ``df_pool``, ``df_test``,
        ``gtm_coords_pool``, and ``config`` keys.

    Raises
    ------
    FileNotFoundError
        If no ``setup_*.pkl`` files are found.
    """
    setups: dict[str, dict] = {}
    for setup_pkl in sorted(Path(results_dir).glob("setup_*.pkl")):
        split_type = setup_pkl.stem[len("setup_") :]
        with open(setup_pkl, "rb") as fh:
            setups[split_type] = pickle.load(fh)
    if not setups:
        raise FileNotFoundError(
            f"No {results_dir}/setup_*.pkl files found. "
            "Run `python run.py --setup-only` first."
        )
    return setups


def load_job_results(
    results_dir: str | Path = "results",
) -> dict[tuple[str, str, int], dict]:
    """Load all per-job result pickles from *results_dir*.

    Filename format: ``run_<split_type>_<strategy>_seed<N>.pkl``

    Parameters
    ----------
    results_dir : str or Path
        Directory containing ``run_*.pkl`` files.

    Returns
    -------
    dict[tuple[str, str, int], dict]
        Mapping from ``(split_type, strategy, seed)`` to the unpickled result
        dictionary (the ``"result"`` key from each file).

    Raises
    ------
    FileNotFoundError
        If no matching ``run_*.pkl`` files are found.
    """
    pattern = re.compile(r"^run_([a-z]+)_([A-Za-z]+)_seed(\d+)\.pkl$")
    job_results: dict[tuple[str, str, int], dict] = {}
    for pkl_file in sorted(Path(results_dir).glob("run_*.pkl")):
        m = pattern.match(pkl_file.name)
        if not m:
            continue
        sp, strat, seed = m.group(1), m.group(2), int(m.group(3))
        with open(pkl_file, "rb") as fh:
            job_data = pickle.load(fh)
        job_results[(sp, strat, seed)] = job_data["result"]
    if not job_results:
        raise FileNotFoundError(
            f"No {results_dir}/run_*.pkl files found. "
            "Run `python run.py` (or individual jobs) first."
        )
    return job_results


def run_sanity_checks(
    setups: dict[str, dict],
    job_results: dict[tuple[str, str, int], dict],
    cfg: ALConfig,
) -> None:
    """Print coverage and consistency checks across split types.

    Warns if any strategy is missing results, if seed counts differ across
    strategies, or if iteration counts differ across runs.

    Parameters
    ----------
    setups : dict[str, dict]
        Loaded setup dictionaries keyed by split type.
    job_results : dict[tuple[str, str, int], dict]
        Loaded per-job results keyed by ``(split_type, strategy, seed)``.
    cfg : ALConfig
        Active learning configuration (used for ``cfg.strategies``).
    """
    available_split_types = list(setups.keys())
    warned = False

    for split_type, setup in setups.items():
        df_seed = setup.get("df_seed")
        if df_seed is not None:
            print(
                f"[{split_type}] External seed data: {len(df_seed)} compounds "
                "(used in training, not counted in n_labeled on learning curves)."
            )

    for split_type in available_split_types:
        print(f"\n[{split_type}] Sanity checks:")
        split_jobs = {
            (st, sd): r for (sp, st, sd), r in job_results.items() if sp == split_type
        }

        missing_strategies = [
            s for s in cfg.strategies if not any(k[0] == s for k in split_jobs)
        ]
        if missing_strategies:
            print(f"  WARNING: Missing strategies entirely: {missing_strategies}")
            warned = True

        seeds_per_strategy: dict[str, list[int]] = {
            s: sorted(sd for (st, sd) in split_jobs if st == s) for s in cfg.strategies
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


def build_split_data(
    setups: dict[str, dict],
    job_results: dict[tuple[str, str, int], dict],
    cfg: ALConfig,
) -> dict[str, dict]:
    """Assemble tidy DataFrames and aggregated statistics for each split type.

    For each split type, extracts per-iteration metrics from the run histories
    and computes per-strategy mean and standard deviation over seeds.

    Parameters
    ----------
    setups : dict[str, dict]
        Loaded setup dictionaries keyed by split type.
    job_results : dict[tuple[str, str, int], dict]
        Loaded per-job results keyed by ``(split_type, strategy, seed)``.
    cfg : ALConfig
        Active learning configuration.

    Returns
    -------
    dict[str, dict]
        Mapping from split type to a dict with keys:

        ``df_pool``
            Pool DataFrame for this split.
        ``df_test``
            Test DataFrame for this split.
        ``gtm_coords_pool``
            2-D GTM coordinates for pool compounds.
        ``all_runs``
            ``dict[strategy, list[run_dict]]`` of raw run histories.
        ``learning_curve_long``
            Long-form DataFrame with one row per (strategy, seed, iteration).
        ``pool_history_long``
            Long-form DataFrame of pool pEC50 values over iterations.
        ``learning_curve_summary``
            Summary DataFrame with per-strategy mean/std/lower/upper columns.
    """
    per_split_data: dict[str, dict] = {}

    for split_type, setup in setups.items():
        _df_pool = setup["df_pool"]
        _df_test = setup["df_test"]
        _gtm_coords = setup["gtm_coords_pool"]

        split_jobs = {
            (st, sd): r for (sp, st, sd), r in job_results.items() if sp == split_type
        }
        seeds_per_strategy = {
            s: sorted(sd for (st, sd) in split_jobs if st == s) for s in cfg.strategies
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
            "mae",
            "ktau",
            "r2",
            "miscal_area",
            "miscal_area_pre_cal",
            "sigma_error_rho",
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

    return per_split_data


def _save_multisplit(
    figs: dict[str, go.Figure],
    fname: str,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Combine split-type figures and write HTML + queue SVG export.

    When both ``"scaffold"`` and ``"random"`` keys are present in *figs*, the
    two figures are merged into a two-panel side-by-side layout. Otherwise the
    single available figure is used as-is.

    Parameters
    ----------
    figs : dict[str, go.Figure]
        Mapping from split type to Plotly figure.
    fname : str
        Output filename stem (written to ``<results_dir>/<fname>.html``).
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list; ``(figure, svg_path)`` tuples are appended for later
        batch export via :func:`export_plotly_svgs`.
    xlim : tuple[float, float] or None, optional
        If provided, sets the x-axis range on the output figure.
    ylim : tuple[float, float] or None, optional
        If provided, sets the y-axis range on the output figure.
    """
    if "scaffold" in figs and "random" in figs:
        out = _combine_figures_side_by_side(figs["scaffold"], figs["random"])
    elif figs:
        out = next(iter(figs.values()))
    else:
        return
    if xlim is not None:
        out.update_xaxes(range=list(xlim))
    if ylim is not None:
        out.update_yaxes(range=list(ylim))
    out.write_html(f"{results_dir}/{fname}.html")
    svg_queue.append((out, f"{results_dir}/{fname}.svg"))


def generate_learning_curve_mae(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the MAE learning curve figure.

    Produces ``learning_curve_mae.html`` and queues the SVG export.

    Parameters
    ----------
    per_split_data : dict[str, dict]
        Per-split assembled data from :func:`build_split_data`.
    cfg : ALConfig
        Active learning configuration.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating learning curve (MAE)...")
    figs: dict[str, go.Figure] = {}
    for sp in ["scaffold", "random"]:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_learning_curve_with_bands(
            per_split_data[sp]["learning_curve_summary"],
            metric_col="mae",
            ylabel="MAE (pEC50 units)",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(
        figs, "learning_curve_mae", results_dir, svg_queue, xlim=xlim, ylim=ylim
    )


def generate_learning_curve_ktau(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the Kendall's τ learning curve figure.

    Produces ``learning_curve_ktau.html`` and queues the SVG export.

    Parameters
    ----------
    per_split_data : dict[str, dict]
        Per-split assembled data from :func:`build_split_data`.
    cfg : ALConfig
        Active learning configuration.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating learning curve (Kendall's τ)...")
    figs: dict[str, go.Figure] = {}
    for sp in ["scaffold", "random"]:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_learning_curve_with_bands(
            per_split_data[sp]["learning_curve_summary"],
            metric_col="ktau",
            ylabel="Kendall's τ",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(
        figs, "learning_curve_ktau", results_dir, svg_queue, xlim=xlim, ylim=ylim
    )


def generate_hit_discovery_curve(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    hit_threshold: float = 6.3,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the hit discovery curve figure.

    Produces ``hit_discovery_curve.html`` and queues the SVG export.

    Parameters
    ----------
    per_split_data : dict[str, dict]
        Per-split assembled data from :func:`build_split_data`.
    cfg : ALConfig
        Active learning configuration.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    hit_threshold : float, optional
        pEC50 value above which a compound is counted as a hit. Default 6.3.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating hit discovery curve...")
    figs: dict[str, go.Figure] = {}
    for sp in ["scaffold", "random"]:
        if sp not in per_split_data:
            continue
        max_hits_sp = int(
            (per_split_data[sp]["df_pool"]["pEC50"] >= hit_threshold).sum()
        )
        figs[sp] = alp.plot_hit_discovery_curve(
            per_split_data[sp]["pool_history_long"],
            per_split_data[sp]["learning_curve_long"],
            hit_threshold=hit_threshold,
            max_hits=max_hits_sp,
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(
        figs, "hit_discovery_curve", results_dir, svg_queue, xlim=xlim, ylim=ylim
    )


def generate_calibration_area_per_iteration(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the miscalibration area per iteration figure.

    Produces ``calibration_area_per_iteration.html`` and queues SVG export.

    Parameters
    ----------
    per_split_data : dict[str, dict]
        Per-split assembled data from :func:`build_split_data`.
    cfg : ALConfig
        Active learning configuration.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating calibration area per iteration plot...")
    figs: dict[str, go.Figure] = {}
    for sp in ["scaffold", "random"]:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_calibration_area_per_iteration(
            per_split_data[sp]["learning_curve_summary"],
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(
        figs,
        "calibration_area_per_iteration",
        results_dir,
        svg_queue,
        xlim=xlim,
        ylim=ylim,
    )


def generate_sigma_error_correlation(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the sigma–error Spearman correlation figure.

    Produces ``sigma_error_correlation.html`` and queues SVG export.

    Parameters
    ----------
    per_split_data : dict[str, dict]
        Per-split assembled data from :func:`build_split_data`.
    cfg : ALConfig
        Active learning configuration.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating sigma–error correlation plot...")
    figs: dict[str, go.Figure] = {}
    for sp in ["scaffold", "random"]:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_sigma_error_correlation(
            per_split_data[sp]["learning_curve_summary"],
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(
        figs, "sigma_error_correlation", results_dir, svg_queue, xlim=xlim, ylim=ylim
    )


def generate_calibration_curve(
    all_runs: dict[str, list],
    df_test: pd.DataFrame,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    strategy: str = "Exploitation",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate and save the before/after calibration curve figure.

    Uses predictions from the final iteration of *strategy*'s first seed run.
    Prints miscalibration area values before and after isotonic regression.
    Produces ``calibration_curve.html`` and queues SVG export.

    Parameters
    ----------
    all_runs : dict[str, list]
        Per-strategy run histories from the primary split.
    df_test : pd.DataFrame
        Test-set DataFrame with a ``pEC50`` column.
    results_dir : str or Path
        Directory to write output files.
    svg_queue : list of tuple
        Accumulator list for batch SVG export.
    strategy : str, optional
        Strategy whose final-iteration predictions are used. Default
        ``"Exploitation"``.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the output figure.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the output figure.
    """
    print("Generating calibration curve...")
    final_state = all_runs[strategy][0]["history"][-1]

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
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    fig.write_html(f"{results_dir}/calibration_curve.html")
    svg_queue.append((fig, f"{results_dir}/calibration_curve.svg"))

    print(f"\nMiscalibration Area Before: {final_state['miscal_area_pre_cal']:.4f}")
    print(f"Miscalibration Area After:  {final_state['miscal_area']:.4f}")


def generate_gtm_figures(
    all_runs: dict[str, list],
    gtm_coords_pool: np.ndarray,
    results_dir: str | Path,
    *,
    method: str = "Exploitation",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate GTM selection figures: animated Plotly HTML and static SVG.

    Writes an interactive animated Plotly figure to
    ``gtm_selection_animation_<method_lower>.html`` and a static matplotlib
    snapshot (compounds coloured by AL iteration) to the corresponding ``.svg``
    file.

    The animated HTML is not queued for Kaleido SVG export because animated
    frames collapse to an empty first frame; the static matplotlib SVG serves
    as the static representation instead.

    Parameters
    ----------
    all_runs : dict[str, list]
        Per-strategy run histories from the primary split.
    gtm_coords_pool : np.ndarray
        2-D GTM coordinates for pool compounds, shape ``(n_compounds, 2)``.
    results_dir : str or Path
        Directory to write output files.
    method : str, optional
        Strategy name whose run history is visualised. Default
        ``"Exploitation"``.
    xlim : tuple[float, float] or None, optional
        X-axis range override. Applied to both the Plotly figure and the
        matplotlib axes.
    ylim : tuple[float, float] or None, optional
        Y-axis range override. Applied to both the Plotly figure and the
        matplotlib axes.
    """
    print(f"Generating GTM animation ({method})...")
    fig = alp.plot_gtm_selection_animation(
        gtm_coords=gtm_coords_pool,
        selection_history=all_runs[method][0]["history"],
        title=f"{method} Compound Selection in GTM Chemical Space",
    )
    fig.update_layout(autosize=False, width=800, height=800)
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    fig.write_html(f"{results_dir}/gtm_selection_animation_{method.lower()}.html")

    # Static matplotlib snapshot: compounds coloured by first-acquired iteration
    print(f"Generating static GTM snapshot ({method})...")
    gtm_history = all_runs[method][0]["history"]
    gtm_iters = sorted({s["iteration"] for s in gtm_history})
    iter_to_cat = {it: idx + 1 for idx, it in enumerate(gtm_iters)}
    n_iter = len(gtm_iters)
    c = np.zeros(len(gtm_coords_pool), dtype=int)
    for state in gtm_history:
        for idx in state["selected_pool_indices"]:
            if c[idx] == 0:
                c[idx] = iter_to_cat[state["iteration"]]
    base_cmap = mcm.get_cmap("viridis")
    iter_colors = [base_cmap(i / max(n_iter - 1, 1)) for i in range(n_iter)]
    colors_list = [(0.75, 0.75, 0.75, 0.35)] + iter_colors
    point_colors = [colors_list[ci] for ci in c]

    fig_static, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        gtm_coords_pool[:, 0],
        gtm_coords_pool[:, 1],
        color=(0.75, 0.75, 0.75),
        s=4,
        alpha=0.25,
        linewidths=0,
        zorder=1,
    )
    ax.scatter(
        gtm_coords_pool[:, 0],
        gtm_coords_pool[:, 1],
        c=point_colors,
        s=8,
        linewidths=0,
        zorder=2,
    )
    ax.set_xlabel("GTM dimension 1")
    ax.set_ylabel("GTM dimension 2")
    ax.set_facecolor("white")
    fig_static.patch.set_facecolor("white")
    ax.set_box_aspect(1)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    sm = plt.cm.ScalarMappable(
        cmap=base_cmap, norm=mcolors.Normalize(vmin=0, vmax=n_iter - 1)
    )
    sm.set_array([])
    gtm_divider = make_axes_locatable(ax)
    gtm_cax = gtm_divider.append_axes("right", size="5%", pad=0.1)
    fig_static.colorbar(
        sm, cax=gtm_cax, label="AL Iteration"
    ).ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_static.savefig(
        f"{results_dir}/gtm_selection_animation_{method.lower()}.svg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_static)


def compute_tmap_layout(df_pool: pd.DataFrame) -> tuple:
    """Compute the TMAP layout for pool compounds.

    Wraps :func:`src.helpers.smiles_to_tmap`. This step is separated from
    :func:`generate_tmap_figures` because the layout computation is expensive
    and its result is shared between the interactive Faerun figure and the
    static matplotlib snapshot.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column.

    Returns
    -------
    tuple
        ``(x, y, s, t)`` TMAP layout arrays as returned by
        :func:`src.helpers.smiles_to_tmap`.
    """
    print("Computing TMAP layout...")
    return smiles_to_tmap(list(df_pool["smiles"].values))


def generate_tmap_figures(
    df_pool: pd.DataFrame,
    all_runs: dict[str, list],
    tmap_layout: tuple,
    results_dir: str | Path,
    *,
    strategy: str = "Exploitation",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate TMAP visualizations: interactive Faerun HTML and static SVG.

    Writes an interactive Faerun scatter to ``tmap_selection.html`` and a
    static matplotlib snapshot (compounds coloured by AL iteration) to
    ``tmap_selection.svg``.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column.
    all_runs : dict[str, list]
        Per-strategy run histories from the primary split.
    tmap_layout : tuple
        Pre-computed TMAP layout ``(x, y, s, t)`` from
        :func:`compute_tmap_layout`.
    results_dir : str or Path
        Directory to write output files.
    strategy : str, optional
        Strategy name whose run history is visualised. Default
        ``"Exploitation"``.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the static matplotlib snapshot.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the static matplotlib snapshot.
    """
    print("Generating TMAP visualization...")
    alp.plot_tmap_faerun_strategies(
        tmap_layout,
        smiles_list=list(df_pool["smiles"].values),
        strategies_histories={s: all_runs[s][0]["history"] for s in all_runs},
        point_scale=3,
        background_point_scale=1,
        title="Active Learning Selection (TMAP)",
        output_name="tmap_selection",
        output_path=f"{results_dir}/",
    )

    # Static matplotlib snapshot: compounds coloured by first-acquired iteration
    print("Generating static TMAP snapshot...")
    tx, ty, ts, tt = tmap_layout
    exploitation_history = all_runs[strategy][0]["history"]
    exploitation_iters = sorted({s["iteration"] for s in exploitation_history})
    exploitation_iter_to_cat = {
        it: idx + 1 for idx, it in enumerate(exploitation_iters)
    }
    n_exploitation_iter = len(exploitation_iters)
    ct = np.zeros(len(tx), dtype=int)
    for state in exploitation_history:
        for idx in state["selected_pool_indices"]:
            if ct[idx] == 0:
                ct[idx] = exploitation_iter_to_cat[state["iteration"]]
    tmap_base_cmap = mcm.get_cmap("viridis")
    tmap_colors = [(0.75, 0.75, 0.75, 0.15)] + [
        tmap_base_cmap(i / max(n_exploitation_iter - 1, 1))
        for i in range(n_exploitation_iter)
    ]
    tmap_point_colors = [tmap_colors[ci] for ci in ct]

    fig_static, ax = plt.subplots(figsize=(8.5, 8.5))
    for si, ti in zip(ts, tt):
        ax.plot(
            [tx[si], tx[ti]],
            [ty[si], ty[ti]],
            color="gray",
            lw=0.15,
            alpha=0.3,
            zorder=1,
        )
    ax.scatter(tx, ty, c=tmap_point_colors, s=2, linewidths=0, zorder=2)
    ax.axis("off")
    fig_static.patch.set_facecolor("white")
    ax.set_box_aspect(1)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    tsm = plt.cm.ScalarMappable(
        cmap=tmap_base_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=n_exploitation_iter - 1),
    )
    tsm.set_array([])
    tmap_divider = make_axes_locatable(ax)
    tmap_cax = tmap_divider.append_axes("right", size="5%", pad=0.1)
    fig_static.colorbar(
        tsm, cax=tmap_cax, label=f"AL Iteration ({strategy})"
    ).ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_static.savefig(f"{results_dir}/tmap_selection.svg", bbox_inches="tight")
    plt.close(fig_static)


def compute_tmap_partition_layout(
    df_pool: pd.DataFrame, df_test: pd.DataFrame
) -> tuple:
    """Compute a joint TMAP layout covering both pool and test compounds.

    Wraps :func:`src.helpers.smiles_to_tmap` on the concatenation of pool and
    test SMILES so that both partitions appear in the same minimum-spanning-tree
    embedding.  This layout is separate from the one produced by
    :func:`compute_tmap_layout` (pool-only), which is used for the AL selection
    figure.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column.
    df_test : pd.DataFrame
        Test DataFrame with a ``smiles`` column.

    Returns
    -------
    tuple
        ``(x, y, s, t)`` TMAP layout arrays as returned by
        :func:`src.helpers.smiles_to_tmap`, with pool compounds first
        (rows ``0 .. len(df_pool) - 1``) and test compounds last
        (rows ``len(df_pool) .. len(df_pool) + len(df_test) - 1``).
    """
    print("Computing TMAP partition layout (pool + test)...")
    all_smiles = list(df_pool["smiles"].values) + list(df_test["smiles"].values)
    return smiles_to_tmap(all_smiles)


def generate_tmap_partition_figures(
    df_pool: pd.DataFrame,
    df_test: pd.DataFrame,
    tmap_layout: tuple,
    results_dir: str | Path,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate train/test partition TMAP visualizations: Faerun HTML and static SVG.

    Writes an interactive Faerun scatter to ``tmap_partition.html`` (pool
    compounds colored as Train, test compounds as Test) and a static matplotlib
    snapshot to ``tmap_partition.svg``.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column (Train partition).
    df_test : pd.DataFrame
        Test DataFrame with a ``smiles`` column (Test partition).
    tmap_layout : tuple
        Pre-computed TMAP layout ``(x, y, s, t)`` from
        :func:`compute_tmap_partition_layout`, covering pool + test compounds in
        that order.
    results_dir : str or Path
        Directory to write output files.
    xlim : tuple[float, float] or None, optional
        X-axis range override for the static matplotlib snapshot.
    ylim : tuple[float, float] or None, optional
        Y-axis range override for the static matplotlib snapshot.
    """
    n_pool = len(df_pool)
    n_test = len(df_test)
    all_smiles = list(df_pool["smiles"].values) + list(df_test["smiles"].values)
    partition_labels = np.concatenate(
        [np.zeros(n_pool, dtype=int), np.ones(n_test, dtype=int)]
    )

    print("Generating TMAP partition visualization...")
    alp.plot_tmap_faerun_partition(
        tmap_layout,
        smiles_list=all_smiles,
        partition_labels=partition_labels,
        title="Train / Test Partition (TMAP)",
        output_name="tmap_partition",
        output_path=f"{results_dir}/",
    )

    # Static matplotlib snapshot: pool=blue, test=orange
    print("Generating static TMAP partition snapshot...")
    tx, ty, ts, tt = tmap_layout
    train_color = (0.122, 0.467, 0.706, 0.6)   # tab10 blue
    test_color = (1.0, 0.498, 0.055, 0.9)       # tab10 orange

    point_colors = [
        train_color if lbl == 0 else test_color for lbl in partition_labels
    ]

    fig_static, ax = plt.subplots(figsize=(8.5, 8.5))
    for si, ti in zip(ts, tt):
        ax.plot(
            [tx[si], tx[ti]],
            [ty[si], ty[ti]],
            color="gray",
            lw=0.15,
            alpha=0.3,
            zorder=1,
        )
    ax.scatter(tx, ty, c=point_colors, s=2, linewidths=0, zorder=2)
    ax.axis("off")
    fig_static.patch.set_facecolor("white")
    ax.set_box_aspect(1)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor=train_color[:3], markersize=6,
            label="Train",
        ),
        plt.Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor=test_color[:3], markersize=6,
            label="Test",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)
    fig_static.savefig(f"{results_dir}/tmap_partition.svg", bbox_inches="tight")
    plt.close(fig_static)


def export_plotly_svgs(svg_tasks: list[tuple]) -> None:
    """Export a batch of Plotly figures to SVG using a single Kaleido session.

    Spawning one Kaleido subprocess per figure is expensive; this function
    opens a single session and writes all figures sequentially.

    Parameters
    ----------
    svg_tasks : list of tuple
        Each element is ``(figure, svg_path)`` where *figure* is a
        :class:`plotly.graph_objects.Figure` and *svg_path* is the destination
        file path string.
    """
    print(f"\nExporting {len(svg_tasks)} Plotly figures to SVG...")

    async def _export():
        async with Kaleido() as k:
            for fig, svg_path in svg_tasks:
                await k.write_fig(fig, svg_path)
                print(f"  saved {svg_path}")

    asyncio.run(_export())


def main() -> None:
    """Load AL results, run sanity checks, assemble DataFrames, and write all figures.

    Orchestrates the full analysis pipeline:

    1. Load setup and job-result pickles from ``results/``.
    2. Run sanity checks on coverage and consistency.
    3. Assemble tidy DataFrames and per-strategy statistics for each split type.
    4. Generate all figures, writing HTML files and queuing SVG exports.
    5. Batch-export all Plotly figures to SVG via a single Kaleido session.

    Axis limit overrides for individual figures can be set by passing ``xlim``
    and ``ylim`` keyword arguments to the corresponding ``generate_*`` functions
    below.
    """
    setups = load_setups("results")
    cfg: ALConfig = next(iter(setups.values()))["config"]
    available_split_types = list(setups.keys())
    print(f"Split types loaded: {available_split_types}")

    job_results = load_job_results("results")
    run_sanity_checks(setups, job_results, cfg)
    per_split_data = build_split_data(setups, job_results, cfg)

    _primary = (
        "scaffold" if "scaffold" in per_split_data else next(iter(per_split_data))
    )
    all_runs = per_split_data[_primary]["all_runs"]
    df_pool = per_split_data[_primary]["df_pool"]
    df_test = per_split_data[_primary]["df_test"]
    gtm_coords_pool = per_split_data[_primary]["gtm_coords_pool"]

    Path("results").mkdir(exist_ok=True)
    svg_queue: list[tuple] = []

    generate_learning_curve_mae(
        per_split_data, cfg, "results", svg_queue, ylim=(0, 0.85)
    )
    generate_learning_curve_ktau(
        per_split_data, cfg, "results", svg_queue, ylim=(0, 0.65)
    )
    generate_hit_discovery_curve(
        per_split_data, cfg, "results", svg_queue, ylim=(0, 23)
    )
    generate_calibration_area_per_iteration(
        per_split_data, cfg, "results", svg_queue, ylim=(0, 0.4)
    )
    generate_sigma_error_correlation(
        per_split_data, cfg, "results", svg_queue, ylim=(0, 0.25)
    )
    generate_calibration_curve(all_runs, df_test, "results", svg_queue)
    generate_gtm_figures(all_runs, gtm_coords_pool, "results")
    tmap_layout = compute_tmap_layout(df_pool)
    generate_tmap_figures(df_pool, all_runs, tmap_layout, "results")
    tmap_partition_layout = compute_tmap_partition_layout(df_pool, df_test)
    generate_tmap_partition_figures(df_pool, df_test, tmap_partition_layout, "results")

    export_plotly_svgs(svg_queue)

    print("\nAll visualizations saved to results/")
    # kaleido v1 leaves a non-daemon background thread that prevents a clean exit;
    # os._exit(0) terminates immediately after all writes complete
    os._exit(0)


if __name__ == "__main__":
    main()
