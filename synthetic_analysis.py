#!/usr/bin/env python
r"""Synthetic oracle analysis and visualisation entry point.

Mirrors ``analysis.py`` but targets the lightweight set of figures relevant to
the synthetic oracle ablation:

- MAE learning curves
- Kendall's τ learning curves
- Hit discovery curves
- σ–|error| Spearman ρ correlation (oracle validation plot)

Additionally supports a **tier comparison mode** (``--configs`` flag) that
loads results from multiple oracle tiers and generates side-by-side faceted
figures showing how acquisition strategies respond to varying uncertainty
quality levels.

Usage
-----
Per-tier analysis (writes to the tier's own results_path):

    python synthetic_analysis.py --config config/oracle_pxr_rho00_config.yaml
    python synthetic_analysis.py --config config/oracle_asap_rho06_config.yaml --hit-threshold 7.0

Tier comparison (writes to a separate output directory):

    python synthetic_analysis.py \
        --configs config/oracle_pxr_rho00_config.yaml \
                  config/oracle_pxr_rho03_config.yaml \
                  config/oracle_pxr_rho06_config.yaml \
                  config/oracle_pxr_ramping_config.yaml \
        --compare-output results/oracle_pxr_comparison \
        --tier-names "rho=0" "rho=0.3" "rho=0.6" "ramping"

Generated outputs (per-tier mode)
----------------------------------
    learning_curve_mae.html / .svg
    learning_curve_ktau.html / .svg
    hit_discovery_curve.html / .svg
    sigma_error_correlation.html / .svg

Generated outputs (tier comparison mode)
-----------------------------------------
    compare_mae.html / .svg          — MAE: 1×5 panel (per strategy), 4 tier curves
    compare_hit_discovery.html / .svg — Hits: same layout

Requires ``kaleido`` for SVG export.
"""

import argparse
import asyncio
import logging
import warnings
from pathlib import Path

logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from kaleido import Kaleido  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import src.plots as alp  # noqa: E402
from analysis import (  # noqa: E402
    build_split_data,
    generate_hit_discovery_curve,
    generate_learning_curve_ktau,
    generate_learning_curve_mae,
    generate_sigma_error_correlation,
    load_job_results,
    load_setups,
    run_sanity_checks,
)
from src.config import load_config  # noqa: E402
from src.helpers import STRATEGY_COLORS  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# SVG export (mirrors analysis.py)
# ---------------------------------------------------------------------------


def export_plotly_svgs(svg_tasks: list[tuple]) -> None:
    """Export a batch of Plotly figures to SVG using a single Kaleido session.

    Parameters
    ----------
    svg_tasks : list of tuple
        Each element is ``(figure, svg_path)``.
    """
    if not svg_tasks:
        return
    print(f"\nExporting {len(svg_tasks)} Plotly figures to SVG...")

    async def _export():
        async with Kaleido() as k:
            for fig, svg_path in svg_tasks:
                await k.write_fig(fig, svg_path)
                print(f"  saved {svg_path}")

    asyncio.run(_export())


# ---------------------------------------------------------------------------
# Per-tier analysis
# ---------------------------------------------------------------------------


def run_per_tier_analysis(cfg_path: str, hit_threshold: float = 7.0) -> None:
    """Run analysis for a single oracle tier and write figures to results_path.

    Generates MAE, Kendall's τ, hit discovery, and sigma–error correlation
    figures.

    Parameters
    ----------
    cfg_path : str
        Path to the oracle YAML config file.
    hit_threshold : float, optional
        pEC50 / pIC50 threshold for counting hits. Default 7.0.
    """
    cfg = load_config(cfg_path)
    results_dir = Path(cfg.results_path).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load setup pickles — may live in a shared (real-run) directory
    import yaml  # noqa: PLC0415

    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    setup_results_path = raw.get("data", {}).get("setup_results_path")

    # Try shared path first, fall back to own results_path
    for candidate in filter(None, [setup_results_path, str(results_dir)]):
        try:
            setups = load_setups(candidate)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(
            f"No setup_*.pkl found in {setup_results_path!r} or {results_dir!r}. "
            "Run synthetic_run.py --setup-only first."
        )

    job_results = load_job_results(results_dir)
    run_sanity_checks(setups, job_results, cfg)
    per_split_data = build_split_data(setups, job_results, cfg)

    svg_queue: list[tuple] = []
    generate_learning_curve_mae(per_split_data, cfg, results_dir, svg_queue)
    generate_learning_curve_ktau(per_split_data, cfg, results_dir, svg_queue)
    generate_hit_discovery_curve(
        per_split_data, cfg, results_dir, svg_queue, hit_threshold=hit_threshold
    )
    generate_sigma_error_correlation(per_split_data, cfg, results_dir, svg_queue)
    export_plotly_svgs(svg_queue)


# ---------------------------------------------------------------------------
# Tier comparison helpers
# ---------------------------------------------------------------------------


def _load_tier_data(
    cfg_paths: list[str],
) -> tuple[list, list[dict], list[str]]:
    """Load all tier configs, setups, and results.

    Parameters
    ----------
    cfg_paths : list[str]
        Paths to oracle YAML config files, one per tier.

    Returns
    -------
    cfgs : list[ALConfig]
        Loaded AL configs.
    per_split_list : list[dict]
        One ``per_split_data`` dict per tier.
    strategies : list[str]
        Strategies shared across all tiers (taken from the first config).
    """
    import yaml  # noqa: PLC0415

    cfgs = []
    per_split_list = []

    for path in cfg_paths:
        cfg = load_config(path)
        cfgs.append(cfg)
        results_dir = Path(cfg.results_path).expanduser()

        with open(path) as fh:
            raw = yaml.safe_load(fh)
        setup_results_path = raw.get("data", {}).get("setup_results_path")

        for candidate in filter(None, [setup_results_path, str(results_dir)]):
            try:
                setups = load_setups(candidate)
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(
                f"No setup_*.pkl found for tier {path!r}."
            )

        job_results = load_job_results(results_dir)
        per_split_data = build_split_data(setups, job_results, cfg)
        per_split_list.append(per_split_data)

    # Use strategies from the first tier (all tiers should match)
    strategies = cfgs[0].strategies
    return cfgs, per_split_list, strategies


# Muted palette for tier curves — distinguishable even in grayscale
_TIER_COLORS = ["#888888", "#4DBBAA", "#F05E2A", "#5B6FD4"]
_TIER_DASH = ["dot", "dash", "dashdot", "solid"]


def _build_tier_comparison_figure(
    per_split_list: list[dict],
    tier_names: list[str],
    strategies: list[str],
    metric_col: str,
    ylabel: str,
) -> go.Figure:
    """Build a 2×2 faceted Plotly figure for tier comparison.

    Each panel corresponds to one oracle tier (uncertainty configuration);
    each trace within a panel corresponds to one acquisition strategy.
    Shading (±1 σ) uses the same ``learning_curve_summary`` data as the
    per-tier plots.

    Parameters
    ----------
    per_split_list : list[dict]
        One per-split data dict per tier (from :func:`_load_tier_data`).
    tier_names : list[str]
        Human-readable tier labels, one per tier.
    strategies : list[str]
        Acquisition strategy names, shown as separate traces.
    metric_col : str
        Column name (without ``_mean``/``_std`` suffix) to plot on the y-axis.
    ylabel : str
        Y-axis label text.

    Returns
    -------
    go.Figure
        Faceted Plotly figure.
    """
    n_tiers = len(tier_names)
    n_cols = 3
    n_rows = (n_tiers + n_cols - 1) // n_cols

    from plotly.subplots import make_subplots  # noqa: PLC0415

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=tier_names,
        horizontal_spacing=0.04,
        vertical_spacing=0.14,
    )

    for tier_idx, (per_split_data, tier_name) in enumerate(
        zip(per_split_list, tier_names)
    ):
        row = tier_idx // n_cols + 1
        col = tier_idx % n_cols + 1

        split_key = next(iter(per_split_data))
        summary = per_split_data[split_key]["learning_curve_summary"]

        for strat_idx, strategy in enumerate(strategies):
            color = STRATEGY_COLORS.get(strategy, "#888888")
            s_df = summary[summary["strategy"] == strategy].sort_values("n_labeled")
            if s_df.empty:
                continue

            x = s_df["n_labeled"].values
            y_mean = s_df[f"{metric_col}_mean"].values
            y_lower = s_df[f"{metric_col}_lower"].values
            y_upper = s_df[f"{metric_col}_upper"].values

            # Shaded band
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_mean,
                    mode="lines",
                    name=strategy,
                    line=dict(color=color, width=2),
                    legendgroup=strategy,
                    showlegend=(tier_idx == 0),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=340 * n_rows + 80,
        width=420 * n_cols,
        template="plotly_white",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
        ),
        margin=dict(t=50, b=100, l=60, r=20),
    )
    # Y-axis labels only on left column
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text=ylabel, row=r, col=1)
    fig.update_xaxes(title_text="Labeled pool size")
    return fig


def _build_hit_comparison_figure(
    per_split_list: list[dict],
    tier_names: list[str],
    strategies: list[str],
    hit_threshold: float = 7.0,
) -> go.Figure:
    """Build a 1×N_strategies faceted hit discovery figure for tier comparison.

    Parameters
    ----------
    per_split_list : list[dict]
        One per-split data dict per tier.
    tier_names : list[str]
        Human-readable tier labels.
    strategies : list[str]
        Acquisition strategy names.
    hit_threshold : float, optional
        Activity threshold for counting hits. Default 7.0.

    Returns
    -------
    go.Figure
        Faceted Plotly figure.
    """
    from plotly.subplots import make_subplots  # noqa: PLC0415

    n_tiers = len(tier_names)
    n_cols = 3
    n_rows = (n_tiers + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=tier_names,
        horizontal_spacing=0.04,
        vertical_spacing=0.14,
    )

    for tier_idx, (per_split_data, tier_name) in enumerate(
        zip(per_split_list, tier_names)
    ):
        row = tier_idx // n_cols + 1
        col = tier_idx % n_cols + 1

        split_key = next(iter(per_split_data))
        pool_hist = per_split_data[split_key]["pool_history_long"]
        lc_long = per_split_data[split_key]["learning_curve_long"]
        df_pool = per_split_data[split_key]["df_pool"]

        for strat_idx, strategy in enumerate(strategies):
            color = STRATEGY_COLORS.get(strategy, "#888888")
            s_pool = pool_hist[pool_hist["strategy"] == strategy]
            s_lc = lc_long[lc_long["strategy"] == strategy]
            if s_pool.empty:
                continue

            seeds = s_pool["seed"].unique()
            all_x, all_y = [], []
            for seed in seeds:
                sp = s_pool[s_pool["seed"] == seed].sort_values("iteration")
                hits_found = []
                for it, grp in sp.groupby("iteration", sort=True):
                    cumulative = grp["pEC50"].values
                    n_hits = int((cumulative >= hit_threshold).sum())
                    n_lab = s_lc[
                        (s_lc["seed"] == seed) & (s_lc["iteration"] == it)
                    ]["n_labeled"]
                    if n_lab.empty:
                        continue
                    all_x.append(float(n_lab.values[0]))
                    all_y.append(n_hits)
                    hits_found.append((float(n_lab.values[0]), n_hits))

            if not all_x:
                continue

            df_tmp = (
                pd.DataFrame({"n_labeled": all_x, "hits": all_y})
                .groupby("n_labeled")
                .agg(hits_mean=("hits", "mean"), hits_std=("hits", "std"))
                .reset_index()
                .fillna(0)
            )
            x = df_tmp["n_labeled"].values
            y_mean = df_tmp["hits_mean"].values
            y_lower = np.maximum(df_tmp["hits_mean"].values - df_tmp["hits_std"].values, 0)
            y_upper = df_tmp["hits_mean"].values + df_tmp["hits_std"].values

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_mean,
                    mode="lines",
                    name=strategy,
                    line=dict(color=color, width=2),
                    legendgroup=strategy,
                    showlegend=(tier_idx == 0),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=340 * n_rows + 80,
        width=420 * n_cols,
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
        margin=dict(t=50, b=100, l=60, r=20),
    )
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Cumulative hits found", row=r, col=1)
    fig.update_xaxes(title_text="Labeled pool size")
    return fig


def run_tier_comparison(
    cfg_paths: list[str],
    tier_names: list[str],
    compare_output: str,
    hit_threshold: float = 7.0,
) -> None:
    """Generate tier comparison figures across multiple oracle configs.

    Produces two faceted figures (MAE and hit discovery), each with one panel
    per acquisition strategy and one trace per oracle tier.  Figures are written
    to *compare_output*.

    Parameters
    ----------
    cfg_paths : list[str]
        Paths to oracle YAML configs, one per tier.
    tier_names : list[str]
        Human-readable tier labels (same length as *cfg_paths*).
    compare_output : str
        Output directory path.
    hit_threshold : float, optional
        Activity threshold for hit discovery curve. Default 7.0.
    """
    if len(tier_names) != len(cfg_paths):
        raise ValueError("--tier-names must have the same length as --configs.")

    out_dir = Path(compare_output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(cfg_paths)} tier(s)...")
    cfgs, per_split_list, strategies = _load_tier_data(cfg_paths)
    print(f"Tiers loaded: {tier_names}")
    print(f"Strategies:   {strategies}")

    svg_queue: list[tuple] = []

    # MAE comparison
    print("Building tier comparison — MAE...")
    fig_mae = _build_tier_comparison_figure(
        per_split_list,
        tier_names,
        strategies,
        metric_col="mae",
        ylabel="MAE (pEC50 units)",
    )
    fig_mae.write_html(str(out_dir / "compare_mae.html"))
    svg_queue.append((fig_mae, str(out_dir / "compare_mae.svg")))

    # Kendall's τ comparison
    print("Building tier comparison — Kendall's τ...")
    fig_ktau = _build_tier_comparison_figure(
        per_split_list,
        tier_names,
        strategies,
        metric_col="ktau",
        ylabel="Kendall's τ",
    )
    fig_ktau.write_html(str(out_dir / "compare_ktau.html"))
    svg_queue.append((fig_ktau, str(out_dir / "compare_ktau.svg")))

    # Hit discovery comparison
    print("Building tier comparison — hit discovery...")
    fig_hits = _build_hit_comparison_figure(
        per_split_list,
        tier_names,
        strategies,
        hit_threshold=hit_threshold,
    )
    fig_hits.write_html(str(out_dir / "compare_hit_discovery.html"))
    svg_queue.append((fig_hits, str(out_dir / "compare_hit_discovery.svg")))

    export_plotly_svgs(svg_queue)
    print(f"\nTier comparison figures written to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch per-tier or tier-comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Synthetic oracle analysis and visualisation."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--config",
        metavar="PATH",
        help="Per-tier mode: path to a single oracle YAML config.",
    )
    mode_group.add_argument(
        "--configs",
        nargs="+",
        metavar="PATH",
        help="Tier comparison mode: paths to two or more oracle YAML configs.",
    )

    parser.add_argument(
        "--compare-output",
        metavar="DIR",
        default=None,
        help="Output directory for tier comparison figures (required with --configs).",
    )
    parser.add_argument(
        "--tier-names",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Human-readable tier labels for the comparison figure legend. "
            "Must match the number of --configs entries. "
            "Defaults to 'Tier 1', 'Tier 2', ... if omitted."
        ),
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=7.0,
        metavar="FLOAT",
        help="Activity threshold for counting hits. Default: 7.0",
    )

    args = parser.parse_args()

    if args.config:
        # Per-tier mode
        run_per_tier_analysis(args.config, hit_threshold=args.hit_threshold)

    else:
        # Tier comparison mode
        if not args.compare_output:
            parser.error("--compare-output is required when using --configs.")
        tier_names = args.tier_names or [
            f"Tier {i + 1}" for i in range(len(args.configs))
        ]
        run_tier_comparison(
            args.configs,
            tier_names=tier_names,
            compare_output=args.compare_output,
            hit_threshold=args.hit_threshold,
        )


if __name__ == "__main__":
    main()
