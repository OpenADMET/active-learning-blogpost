#!/usr/bin/env python
"""Analysis and visualization entry point.

Loads ``<results_path>/setup_<split_type>.pkl`` files and all
``<results_path>/run_<split_type>_*.pkl`` files produced by ``run.py``, performs
sanity checks on coverage and consistency, then generates all figures
referenced in ``blogpost.md``.

Each performance-metric figure is produced as a two-panel side-by-side plot
when both scaffold and random split results are present (scaffold = left,
random = right). Single-strategy figures (GTM, TMAP, calibration curve) use
the scaffold split only (fallback to the first available split type).

Usage
-----
    python analysis.py --config asap_config.yaml
    python analysis.py --config pxr_config.yaml

The output directory is read from ``data.results_path`` in the config file.

Generated outputs (written to ``<results_path>/``)::
    learning_curve_mae.html / .svg            — MAE learning curves per strategy
    learning_curve_ktau.html / .svg           — Kendall tau learning curves per strategy
    hit_discovery_curve.html / .svg           — cumulative hits vs. labeled-pool size
    calibration_area_per_iteration.html / .svg — miscalibration area per iteration
    sigma_error_correlation.html / .svg       — σ–|error| Spearman ρ per strategy
    gtm_selection_animation_exploitation.html / .svg — animated GTM (Exploitation)
    tmap_selection.html / .svg                — interactive TMAP (Exploitation, via Faerun)
    tmap_partition.html / .svg                — TMAP colored by train / test partition
    calibration_curve.html / .svg             — before/after scaling factor calibration
    predicted_distribution_{strategy}_seed{seed}.html — animated hit/non-hit KDE (test set)
                                              per AL iteration; HTML-only

Requires ``kaleido`` for SVG export (``pip install kaleido``).
"""

import argparse
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
from scipy.stats import spearmanr  # noqa: E402

import src.plots as alp  # noqa: E402
from src.config import ALConfig, load_config  # noqa: E402
from src.helpers import (  # noqa: E402
    STRATEGY_COLORS,
    smiles_to_tmap,
)

warnings.filterwarnings("ignore")


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
    """Write a figure to HTML and queue it for SVG export.

    Takes the first (and expected only) figure from *figs* and writes it
    to ``<results_dir>/<fname>.html``, then appends it to *svg_queue* for
    batch SVG export via :func:`export_plotly_svgs`.

    Parameters
    ----------
    figs : dict[str, go.Figure]
        Mapping from split type to Plotly figure. Only the first entry is used.
    fname : str
        Output filename stem.
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
    if not figs:
        return
    out = next(iter(figs.values()))
    if xlim is not None:
        out.update_xaxes(range=list(xlim))
    if ylim is not None:
        out.update_yaxes(range=list(ylim))
    alp.write_html_both(fig=out, path=f"{results_dir}/{fname}.html")
    svg_queue.append((out, f"{results_dir}/{fname}.svg"))


def generate_learning_curve_mae(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
) -> None:
    """Generate and save the MAE learning curve figure.

    Produces ``learning_curve_mae.html`` and queues the SVG export.
    Y-axis bounds are autodetected from the data.

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
    """
    print("Generating learning curve (MAE)...")
    figs: dict[str, go.Figure] = {}
    for sp in per_split_data:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_learning_curve_with_bands(
            per_split_data[sp]["learning_curve_summary"],
            metric_col="mae",
            ylabel="MAE (pEC50 units)",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(figs, "learning_curve_mae", results_dir, svg_queue)


def generate_learning_curve_ktau(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
) -> None:
    """Generate and save the Kendall's τ learning curve figure.

    Produces ``learning_curve_ktau.html`` and queues the SVG export.
    Y-axis bounds are autodetected from the data; the lower bound is 0 when all
    values are non-negative, otherwise the minimum is autodetected.

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
    """
    print("Generating learning curve (Kendall's τ)...")
    figs: dict[str, go.Figure] = {}
    for sp in per_split_data:
        if sp not in per_split_data:
            continue
        figs[sp] = alp.plot_learning_curve_with_bands(
            per_split_data[sp]["learning_curve_summary"],
            metric_col="ktau",
            ylabel="Kendall's τ",
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(figs, "learning_curve_ktau", results_dir, svg_queue)


def generate_hit_discovery_curve(
    per_split_data: dict[str, dict],
    cfg: ALConfig,
    results_dir: str | Path,
    svg_queue: list[tuple],
    *,
    hit_threshold: float = 7.0,
) -> None:
    """Generate and save the hit discovery curve figure.

    Produces ``hit_discovery_curve.html`` and queues the SVG export.
    Y-axis bounds are autodetected from the data (lower bound fixed at 0).

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
        pEC50 value above which a compound is counted as a hit. Default 7.0.
    """
    print("Generating hit discovery curve...")
    figs: dict[str, go.Figure] = {}
    for sp in per_split_data:
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
            pool_size=len(per_split_data[sp]["df_pool"]),
            strategy_order=cfg.strategies,
            color_map=STRATEGY_COLORS,
        )
    _save_multisplit(figs, "hit_discovery_curve", results_dir, svg_queue)



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
    for sp in per_split_data:
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
    for sp in per_split_data:
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
    if strategy not in all_runs or not all_runs[strategy]:
        strategy = next(iter(all_runs))
        print(f"  Requested strategy missing; falling back to '{strategy}'")
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
    alp.write_html_both(fig=fig, path=f"{results_dir}/calibration_curve.html")
    svg_queue.append((fig, f"{results_dir}/calibration_curve.svg"))

    print(f"\nMiscalibration Area Before: {final_state['miscal_area_pre_cal']:.4f}")
    print(f"Miscalibration Area After:  {final_state['miscal_area']:.4f}")


def generate_gtm_figures(
    all_runs: dict[str, list],
    gtm_coords_pool: np.ndarray,
    results_dir: str | Path,
    *,
    method: str = "Exploitation",
    split_suffix: str = "",
    pool_activity: np.ndarray | None = None,
    hit_threshold: float = 7.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate GTM selection figures: animated Plotly HTML and static SVG.

    Writes an interactive animated Plotly figure to
    ``gtm_selection_animation_<method_lower><split_suffix>.html`` and a static
    matplotlib snapshot (compounds coloured by AL iteration) to the
    corresponding ``.svg`` file.

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
    split_suffix : str, optional
        String appended to output filenames before the extension, e.g.
        ``"_scaffold"``. Default is ``""`` (no suffix).
    pool_activity : np.ndarray or None, optional
        Activity values for all pool compounds (indexed by pool position).
        When provided, hit compounds (activity >= ``hit_threshold``) are
        highlighted in crimson; others in dark gray.
    hit_threshold : float, optional
        Hit threshold passed to :func:`src.plots.plot_gtm_selection_animation`.
        Default 7.0.
    xlim : tuple[float, float] or None, optional
        X-axis range override. Applied to both the Plotly figure and the
        matplotlib axes.
    ylim : tuple[float, float] or None, optional
        Y-axis range override. Applied to both the Plotly figure and the
        matplotlib axes.
    """
    if method not in all_runs or not all_runs[method]:
        method = next(iter(all_runs))
        print(f"  Requested strategy missing for GTM; falling back to '{method}'")
    print(f"Generating GTM animation ({method})...")
    fig = alp.plot_gtm_selection_animation(
        gtm_coords=gtm_coords_pool,
        selection_history=all_runs[method][0]["history"],
        title=f"{method} Compound Selection in GTM Chemical Space",
        pool_activity=pool_activity,
        hit_threshold=hit_threshold,
    )
    fig.update_layout(autosize=False, width=800, height=800)
    if xlim is not None:
        fig.update_xaxes(range=list(xlim))
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))
    alp.write_html_both(
        fig=fig,
        path=f"{results_dir}/gtm_selection_animation_{method.lower()}{split_suffix}.html",
        animation_opts=dict(
            frame=dict(duration=1200, redraw=True),
            transition=dict(duration=400),
        ),
    )

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
        f"{results_dir}/gtm_selection_animation_{method.lower()}{split_suffix}.svg",
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
        ``(x, y, s, t, edge_sims)`` TMAP layout arrays as returned by
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
    split_suffix: str = "",
    edge_similarity_threshold: float = 0.0,
    pool_activity: np.ndarray | None = None,
    hit_threshold: float = 7.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate TMAP visualizations: interactive Faerun HTML and static SVG.

    Writes an interactive Faerun scatter to
    ``tmap_selection<split_suffix>.html`` and a static matplotlib snapshot
    (compounds coloured by AL iteration) to
    ``tmap_selection<split_suffix>.svg``.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column.
    all_runs : dict[str, list]
        Per-strategy run histories from the primary split.
    tmap_layout : tuple
        Pre-computed TMAP layout ``(x, y, s, t, edge_sims)`` from
        :func:`compute_tmap_layout`.
    results_dir : str or Path
        Directory to write output files.
    strategy : str, optional
        Strategy name whose run history is visualised. Default
        ``"Exploitation"``.
    split_suffix : str, optional
        String appended to output filenames before the extension, e.g.
        ``"_scaffold"``. Default is ``""`` (no suffix).
    edge_similarity_threshold : float, optional
        Minimum Tanimoto/Jaccard similarity for an MST edge to be drawn in
        both the Faerun HTML and the static matplotlib snapshot. Default 0.0
        (all edges drawn).
    pool_activity : np.ndarray or None, optional
        Activity values for pool compounds (length = len(df_pool)). When
        provided, actives (activity >= hit_threshold) are rendered larger in
        the Faerun interactive plot.
    hit_threshold : float, optional
        Activity threshold for the active-size bump. Default 7.0.
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
        output_name=f"tmap_selection{split_suffix}",
        output_path=f"{results_dir}/",
        edge_similarity_threshold=edge_similarity_threshold,
        pool_activity=pool_activity,
        hit_threshold=hit_threshold,
    )

    # Static matplotlib snapshot: compounds coloured by first-acquired iteration
    print("Generating static TMAP snapshot...")
    tx, ty, ts, tt, t_edge_sims = tmap_layout
    if edge_similarity_threshold > 0.0:
        emask = t_edge_sims >= edge_similarity_threshold
        ts, tt = ts[emask], tt[emask]
    if strategy not in all_runs or not all_runs[strategy]:
        strategy = next(iter(all_runs))
        print(f"  Requested strategy missing for TMAP snapshot; falling back to '{strategy}'")
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
    fig_static.savefig(
        f"{results_dir}/tmap_selection{split_suffix}.svg", bbox_inches="tight"
    )
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
        ``(x, y, s, t, edge_sims)`` TMAP layout arrays as returned by
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
    split_suffix: str = "",
    edge_similarity_threshold: float = 0.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Generate train/test partition TMAP visualizations: Faerun HTML and static SVG.

    Writes an interactive Faerun scatter to
    ``tmap_partition<split_suffix>.html`` (pool compounds colored as Train,
    test compounds as Test) and a static matplotlib snapshot to
    ``tmap_partition<split_suffix>.svg``.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Pool DataFrame with a ``smiles`` column (Train partition).
    df_test : pd.DataFrame
        Test DataFrame with a ``smiles`` column (Test partition).
    tmap_layout : tuple
        Pre-computed TMAP layout ``(x, y, s, t, edge_sims)`` from
        :func:`compute_tmap_partition_layout`, covering pool + test compounds in
        that order.
    results_dir : str or Path
        Directory to write output files.
    split_suffix : str, optional
        String appended to output filenames before the extension, e.g.
        ``"_scaffold"``. Default is ``""`` (no suffix).
    edge_similarity_threshold : float, optional
        Minimum Tanimoto/Jaccard similarity for an MST edge to be drawn.
        Default 0.0 (all edges drawn).
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
        output_name=f"tmap_partition{split_suffix}",
        output_path=f"{results_dir}/",
        edge_similarity_threshold=edge_similarity_threshold,
    )

    # Static matplotlib snapshot: pool=blue, test=orange
    print("Generating static TMAP partition snapshot...")
    tx, ty, ts, tt, t_edge_sims = tmap_layout
    if edge_similarity_threshold > 0.0:
        emask = t_edge_sims >= edge_similarity_threshold
        ts, tt = ts[emask], tt[emask]
    train_color = (0.122, 0.467, 0.706, 0.6)  # tab10 blue
    test_color = (1.0, 0.498, 0.055, 0.9)  # tab10 orange

    point_colors = [train_color if lbl == 0 else test_color for lbl in partition_labels]

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
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=train_color[:3],
            markersize=6,
            label="Train",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=test_color[:3],
            markersize=6,
            label="Test",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)
    fig_static.savefig(
        f"{results_dir}/tmap_partition{split_suffix}.svg", bbox_inches="tight"
    )
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


def _build_distribution_frames(
    runs: list,
    df_pool: "pd.DataFrame | None",
    activity_col: str,
    strategy: str,
    hit_threshold: float,
) -> "tuple[list[dict], list[np.ndarray] | None, int]":
    """Assemble per-iteration merged frame data for a predicted-distribution animation.

    Concatenates predictions across all seeds at each iteration.  For
    **Exploitation** when ``df_pool`` is provided, unlabeled-pool acquisition
    scores are used (pool mode); all other strategy/config combinations use
    ``y_test_pred`` from each run's history (test mode).

    Parameters
    ----------
    runs : list of dict
        Per-seed run records, each with ``"history"`` and ``"seed"`` keys.
        Runs with empty histories are silently dropped.
    df_pool : pd.DataFrame or None
        Full pool DataFrame with an ``activity_col`` column.  Required for
        Exploitation pool mode; ignored otherwise.
    activity_col : str
        Column name for ground-truth activity values in ``df_pool``.
    strategy : str
        Strategy name (e.g. ``"Exploitation"``).  Only ``"Exploitation"`` with
        a non-None ``df_pool`` triggers pool mode.
    hit_threshold : float
        Activity value at or above which a compound is a hit (used only in pool
        mode to build per-frame true-activity arrays).

    Returns
    -------
    merged_history : list of dict
        One entry per valid iteration with keys ``"iteration"``,
        ``"n_labeled"``, and ``"y_test_pred"`` (concatenated across seeds).
    per_frame_y_true : list of np.ndarray or None
        Per-frame ground-truth activity arrays (pool mode only); ``None`` in
        test mode.
    n_seeds : int
        Number of valid (non-empty) seeds contributing to the merged data.
    """
    valid_runs = [r for r in runs if r["history"]]
    if not valid_runs:
        return [], None, 0
    n_seeds = len(valid_runs)
    n_iter = min(len(r["history"]) for r in valid_runs)
    y_pool = df_pool[activity_col].to_numpy() if df_pool is not None else None
    use_pool_mode = (strategy == "Exploitation" and y_pool is not None)

    merged_history: list[dict] = []
    per_frame_y_true: list[np.ndarray] | None = [] if use_pool_mode else None

    if use_pool_mode:
        assert y_pool is not None
        n_pool = len(y_pool)
        seed_labeled_masks: list[np.ndarray] = []
        for run in valid_runs:
            h = run["history"]
            lm = np.zeros(n_pool, dtype=bool)
            lm[h[0]["selected_pool_indices"]] = True
            seed_labeled_masks.append(lm)

        for k in range(n_iter):
            pool_preds_all: list[np.ndarray] = []
            y_true_all: list[np.ndarray] = []
            for si, run in enumerate(valid_runs):
                h = run["history"]
                if k > 0:
                    seed_labeled_masks[si][h[k]["selected_pool_indices"]] = True
                unlabeled_idx = np.where(~seed_labeled_masks[si])[0]
                raw = h[k].get("acquisition_scores")
                scores = np.asarray(raw) if raw is not None else np.array([])
                if len(scores) == len(unlabeled_idx) and len(scores) > 0:
                    pool_preds_all.append(scores)
                    y_true_all.append(y_pool[unlabeled_idx])
                # In pool mode: skip this seed/iter if scores unavailable.
                # Do NOT fall back to test set — that would mix populations.

            if not pool_preds_all:
                break

            ref_state = valid_runs[0]["history"][k]
            merged_history.append({
                "iteration": ref_state["iteration"],
                "n_labeled": ref_state["n_labeled"],
                "y_test_pred": np.concatenate(pool_preds_all),
            })
            assert per_frame_y_true is not None
            per_frame_y_true.append(np.concatenate(y_true_all))
    else:
        for k in range(n_iter):
            ref_state = valid_runs[0]["history"][k]
            merged_history.append({
                "iteration": ref_state["iteration"],
                "n_labeled": ref_state["n_labeled"],
                "y_test_pred": np.concatenate(
                    [r["history"][k]["y_test_pred"] for r in valid_runs]
                ),
            })

    return merged_history, per_frame_y_true, n_seeds


def _compute_gap_trajectory(
    runs: list,
    df_pool: "pd.DataFrame",
    activity_col: str,
    hit_threshold: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Compute the hit/non-hit predicted-activity gap per iteration for Exploitation.

    For each seed, reconstructs the unlabeled mask at each AL iteration and reads
    ``acquisition_scores`` (Exploitation pool predictions) to compute
    ``mean(scores[hits]) − mean(scores[non-hits])``.  Returns the per-iteration
    mean and standard deviation across seeds.

    Parameters
    ----------
    runs : list of dict
        Per-seed run records, each with ``"history"`` and ``"seed"`` keys.
        Runs with empty histories are silently dropped.
    df_pool : pd.DataFrame
        Full pool DataFrame with an ``activity_col`` column.
    activity_col : str
        Column name for ground-truth activity values in ``df_pool``.
    hit_threshold : float
        Activity value at or above which a compound is classified as a hit.

    Returns
    -------
    n_labeled : np.ndarray, shape (n_iter,)
        Number of labeled pool compounds at each iteration.
    gap_mean : np.ndarray, shape (n_iter,)
        Mean gap across seeds; ``nan`` where no valid seeds contributed.
    gap_std : np.ndarray, shape (n_iter,)
        Standard deviation of gap across seeds; ``nan`` similarly.

    """
    valid_runs = [r for r in runs if r["history"]]
    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    y_pool = df_pool[activity_col].to_numpy()
    n_pool = len(y_pool)
    n_iter = min(len(r["history"]) for r in valid_runs)

    per_seed_gaps: list[list[float]] = []
    n_labeled_list: list[int] = [valid_runs[0]["history"][k]["n_labeled"] for k in range(n_iter)]

    for run in valid_runs:
        h = run["history"]
        labeled_mask = np.zeros(n_pool, dtype=bool)
        labeled_mask[h[0]["selected_pool_indices"]] = True
        seed_gaps: list[float] = []
        for k in range(n_iter):
            if k > 0:
                labeled_mask[h[k]["selected_pool_indices"]] = True
            unlabeled_idx = np.where(~labeled_mask)[0]
            raw = h[k].get("acquisition_scores")
            if raw is None or len(raw) != len(unlabeled_idx):
                seed_gaps.append(float("nan"))
                continue
            scores = np.asarray(raw)
            y_true_unlab = y_pool[unlabeled_idx]
            hit_mask = y_true_unlab >= hit_threshold
            if hit_mask.sum() == 0 or (~hit_mask).sum() == 0:
                seed_gaps.append(float("nan"))
                continue
            seed_gaps.append(float(np.mean(scores[hit_mask]) - np.mean(scores[~hit_mask])))
        per_seed_gaps.append(seed_gaps)

    arr = np.array(per_seed_gaps, dtype=float)  # (n_seeds, n_iter)
    return (
        np.array(n_labeled_list),
        np.nanmean(arr, axis=0),
        np.nanstd(arr, axis=0),
    )


def generate_predicted_distribution_animation(
    all_runs: dict[str, list],
    df_test: pd.DataFrame,
    results_dir: str | Path,
    *,
    hit_threshold: float = 6.0,
    activity_col: str = "pEC50",
    model: str = "",
    df_pool: pd.DataFrame | None = None,
) -> None:
    """Generate and save animated hit/non-hit predicted-distribution figures.

    For every strategy, predictions across all seeds are pooled at each
    iteration by concatenating prediction arrays.  For **Exploitation**, when
    ``df_pool`` is provided, the unlabeled-pool acquisition scores (= committee
    mean predictions on the pool compounds the model is actually choosing from)
    are used instead of held-out test-set predictions.  The hit/non-hit mask is
    then recomputed each frame from the shrinking unlabeled pool's true
    activities.  All other strategies continue to use ``y_test_pred`` on the
    fixed test set.

    Saves one HTML file per strategy:
    ``predicted_distribution_{strategy}.html``.

    Figures are HTML-only; Plotly animations cannot be exported to static SVG.

    Parameters
    ----------
    all_runs : dict[str, list]
        Per-strategy run histories as returned by ``build_split_data``.
        Each value is a list of ``{"seed": int, "history": list[dict]}`` dicts.
    df_test : pd.DataFrame
        Held-out test DataFrame with an ``activity_col`` column for ground-truth
        activity values.  Row order must match ``y_test_pred`` in history states.
    results_dir : str or Path
        Directory to write output files.
    hit_threshold : float, optional
        Activity value at or above which a compound is counted as a hit.
        Default is 6.0.
    activity_col : str, optional
        Column in ``df_test`` with ground-truth activity values.
        Default is ``"pEC50"``.
    model : str, optional
        Model name shown as the first element of each figure title, e.g.
        ``"CheMeleon"`` or ``"ChemProp"``.  When empty, omitted from title.
        Default is ``""``.
    df_pool : pd.DataFrame, optional
        Full labeled-pool DataFrame with an ``activity_col`` column.  When
        provided, Exploitation animations use unlabeled-pool acquisition scores
        (mean predictions) and per-frame true labels from the pool rather than
        the held-out test set.  Default is ``None``.
    """
    results_dir = Path(results_dir)

    for strategy, runs in all_runs.items():
        merged_history, per_frame_y_true, n_seeds = _build_distribution_frames(
            runs, df_pool, activity_col, strategy, hit_threshold,
        )
        if not merged_history:
            continue

        df_test_tiled = pd.concat([df_test] * n_seeds, ignore_index=True)
        seed_label = f"{n_seeds} seeds"
        print(f"Generating predicted distribution animation: {strategy} ({seed_label}) ...")
        fig = alp.plot_predicted_distribution_animation(
            history=merged_history,
            df_test=df_test_tiled,
            hit_threshold=hit_threshold,
            activity_col=activity_col,
            strategy=strategy,
            seed_label=seed_label,
            model=model,
            per_frame_y_true=per_frame_y_true,
            n_seeds=n_seeds,
        )
        out_path = results_dir / f"predicted_distribution_{strategy}.html"
        alp.write_html_both(fig=fig, path=out_path, animation_opts=alp.DIST_ANIMATION_OPTS)
        print(f"  saved {out_path}")


def main() -> None:
    """Load AL results, run sanity checks, assemble DataFrames, and write all figures.

    Orchestrates the full analysis pipeline:

    1. Load setup and job-result pickles from the results directory.
    2. Run sanity checks on coverage and consistency.
    3. Assemble tidy DataFrames and per-strategy statistics for each split type.
    4. Generate all figures, writing HTML files and queuing SVG exports.
    5. Batch-export all Plotly figures to SVG via a single Kaleido session.

    Learning curve and hit discovery y-axis bounds are autodetected from the data.
    Axis limit overrides for calibration figures can be set by passing ``xlim``
    and ``ylim`` keyword arguments to ``generate_calibration_area_per_iteration``
    and ``generate_sigma_error_correlation`` below.
    """
    parser = argparse.ArgumentParser(
        description="Analysis and visualisation for the active learning pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML experiment config file. The results directory is "
        "read from data.results_path in this file. Default: config.yaml",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="pEC50 threshold above which a compound is counted as a hit. "
        "Overrides the value set in the config YAML (data.hit_threshold). "
        "Default: read from config (6.0 for PXR, 7.0 for ASAP).",
    )
    parser.add_argument(
        "--tmap-edge-threshold",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Minimum Tanimoto/Jaccard similarity for a TMAP MST edge to be "
        "drawn. Edges below this value are omitted from both the interactive "
        "Faerun HTML and the static SVG snapshot. Default: 0.0 (all edges drawn).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = Path(cfg.results_path).expanduser()

    # hit_threshold: CLI arg takes precedence; fall back to config-specified value
    hit_threshold: float = args.hit_threshold if args.hit_threshold is not None else cfg.hit_threshold

    setups = load_setups(results_dir)
    available_split_types = list(setups.keys())
    print(f"Split types loaded: {available_split_types}")

    job_results = load_job_results(results_dir)
    run_sanity_checks(setups, job_results, cfg)
    per_split_data = build_split_data(setups, job_results, cfg)

    _primary = "cluster" if "cluster" in per_split_data else next(iter(per_split_data))
    all_runs_primary = per_split_data[_primary]["all_runs"]
    df_test_primary = per_split_data[_primary]["df_test"]

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    svg_queue: list[tuple] = []

    generate_learning_curve_mae(per_split_data, cfg, results_dir, svg_queue)
    generate_learning_curve_ktau(per_split_data, cfg, results_dir, svg_queue)
    generate_hit_discovery_curve(
        per_split_data, cfg, results_dir, svg_queue,
        hit_threshold=hit_threshold,
    )
    generate_calibration_area_per_iteration(
        per_split_data, cfg, results_dir, svg_queue, ylim=(0, 0.4)
    )
    generate_sigma_error_correlation(
        per_split_data, cfg, results_dir, svg_queue, ylim=(0, 0.25)
    )
    generate_calibration_curve(
        all_runs_primary, df_test_primary, results_dir, svg_queue
    )

    # Distribution animation: Exploitation uses unlabeled-pool acquisition scores
    # (= committee mean predictions) to show the gap the model actually acts on.
    # Other strategies fall back to y_test_pred on the held-out test set.
    _df_pool_primary = per_split_data[_primary]["df_pool"]
    generate_predicted_distribution_animation(
        all_runs_primary,
        df_test_primary,
        results_dir,
        hit_threshold=hit_threshold,
        activity_col="pEC50",
        model="CheMeleon" if cfg.use_chemeleon else "ChemProp",
        df_pool=_df_pool_primary,
    )

    for split_type, split_data in per_split_data.items():
        suffix = f"_{split_type}"
        all_runs = split_data["all_runs"]
        df_pool = split_data["df_pool"]
        df_test = split_data["df_test"]
        gtm_coords_pool = split_data["gtm_coords_pool"]

        generate_gtm_figures(
            all_runs, gtm_coords_pool, results_dir,
            split_suffix=suffix,
            pool_activity=df_pool["pEC50"].values,
            hit_threshold=hit_threshold,
        )
        tmap_layout = compute_tmap_layout(df_pool)
        generate_tmap_figures(
            df_pool, all_runs, tmap_layout, results_dir,
            split_suffix=suffix,
            edge_similarity_threshold=args.tmap_edge_threshold,
            pool_activity=df_pool["pEC50"].values,
            hit_threshold=hit_threshold,
        )
        tmap_partition_layout = compute_tmap_partition_layout(df_pool, df_test)
        generate_tmap_partition_figures(
            df_pool, df_test, tmap_partition_layout, results_dir,
            split_suffix=suffix,
            edge_similarity_threshold=args.tmap_edge_threshold,
        )

    export_plotly_svgs(svg_queue)

    print(f"\nAll visualizations saved to {results_dir}/")
    # kaleido v1 leaves a non-daemon background thread that prevents a clean exit;
    # os._exit(0) terminates immediately after all writes complete
    os._exit(0)


if __name__ == "__main__":
    main()
