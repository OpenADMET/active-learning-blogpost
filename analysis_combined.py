#!/usr/bin/env python
"""Cross-configuration comparison analysis.

Loads results from all six experiment configurations and produces:

* **Individual panel figures** (one per config per metric) written to the output
  directory.
* **Combined multi-panel grid figures** (one per target per metric), enabling
  direct comparison across model initialization and ChEMBL seed-data conditions.
* **Mirror distribution animations** (one per target): animated ChemProp vs
  CheMeleon hit/non-hit predicted-activity distributions for the Exploitation
  strategy.

Config matrix
-------------
+-------------------------------+--------+----------+--------+-----------+
| Config dir                    | Target | Model    | ChEMBL | Split     |
+===============================+========+==========+========+===========+
| results/asap_chemprop         | ASAP   | ChemProp | no     | predefined|
| results/asap_chemeleon        | ASAP   | CheMeleon| no     | predefined|
| results/pxr_chemprop          | PXR    | ChemProp | no     | random    |
| results/pxr_chemeleon         | PXR    | CheMeleon| no     | random    |
| results/pxr_chemprop_chembl   | PXR    | ChemProp | yes    | random    |
| results/pxr_chemeleon_chembl  | PXR    | CheMeleon| yes    | random    |
+-------------------------------+--------+----------+--------+-----------+

Grid layouts
------------
* **ASAP** — 1 × 2: ``ChemProp | CheMeleon``
* **PXR**  — 2 × 2: rows = no-ChEMBL / +ChEMBL, cols = ChemProp / CheMeleon

Metrics (4 grid figures per target, plus 1 mirror animation per target, plus 1 cross-target gap figure)
-----------------------------------------------------------------------------------------------------
* MAE learning curve
* Kendall τ learning curve
* Hit discovery curve
* σ–|error| Spearman ρ
* ChemProp vs CheMeleon mirror distribution (Exploitation, animated)
* Hit/non-hit gap vs. labeled pool size (Exploitation, all configs, single panel)

Usage
-----
    python analysis_combined.py [--asap-hit-threshold FLOAT] [--pxr-hit-threshold FLOAT] [--output-dir DIR]

Output files are written to ``results/combined/`` by default (gitignored).
Pass ``--output-dir plots/`` to promote figures to the tracked plots directory
once they are ready.
"""

import argparse
import asyncio
import logging
import os
import warnings
from pathlib import Path

logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)

import plotly.graph_objects as go  # noqa: E402
from kaleido import Kaleido  # noqa: E402

import src.plots as alp  # noqa: E402
from analysis import build_split_data, load_job_results, load_setups, _build_distribution_frames, _compute_gap_trajectory  # noqa: E402
from src.config import load_config  # noqa: E402
from src.helpers import STRATEGY_COLORS  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------
# Each ASAP entry: (panel_label, config_yaml, results_dir, file_slug)
ASAP_CONFIGS = [
    (
        "ChemProp",
        "config/asap_chemprop_config.yaml",
        "results/asap_chemprop",
        "asap_chemprop",
    ),
    (
        "CheMeleon",
        "config/asap_chemeleon_config.yaml",
        "results/asap_chemeleon",
        "asap_chemeleon",
    ),
]

# Each PXR entry: (panel_label, config_yaml, results_dir, file_slug, grid_row, grid_col)
PXR_CONFIGS = [
    (
        "ChemProp",
        "config/pxr_chemprop_config.yaml",
        "results/pxr_chemprop",
        "pxr_chemprop",
        0,
        0,
    ),
    (
        "CheMeleon",
        "config/pxr_chemeleon_config.yaml",
        "results/pxr_chemeleon",
        "pxr_chemeleon",
        0,
        1,
    ),
    (
        "ChemProp + ChEMBL",
        "config/pxr_chemprop_chembl_config.yaml",
        "results/pxr_chemprop_chembl",
        "pxr_chemprop_chembl",
        1,
        0,
    ),
    (
        "CheMeleon + ChEMBL",
        "config/pxr_chemeleon_chembl_config.yaml",
        "results/pxr_chemeleon_chembl",
        "pxr_chemeleon_chembl",
        1,
        1,
    ),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_config_data(config_yaml: str, results_dir: str) -> dict:
    """Load and assemble all run data for a single config directory.

    Parameters
    ----------
    config_yaml : str
        Path to the YAML config file for this experiment.
    results_dir : str
        Path to the results directory containing setup and run pickles.

    Returns
    -------
    dict
        Assembled split data dict (keys: ``df_pool``, ``df_test``,
        ``gtm_coords_pool``, ``all_runs``, ``learning_curve_long``,
        ``pool_history_long``, ``learning_curve_summary``).
    """
    cfg = load_config(config_yaml)
    setups = load_setups(results_dir)
    job_results = load_job_results(results_dir)
    per_split = build_split_data(setups, job_results, cfg)
    # Each results dir has exactly one split type.
    split_key = next(iter(per_split))
    return per_split[split_key]


# ---------------------------------------------------------------------------
# Figure I/O helpers
# ---------------------------------------------------------------------------


def _save_fig(
    fig: go.Figure,
    stem: str,
    output_dir: Path,
    svg_queue: list,
) -> None:
    """Write HTML immediately and queue SVG for batch Kaleido export."""
    html_path = str(output_dir / f"{stem}.html")
    svg_path = str(output_dir / f"{stem}.svg")
    alp.write_html_both(fig=fig, path=html_path)
    print(f"  saved {html_path}")
    svg_queue.append((fig, svg_path))


async def _run_kaleido(svg_tasks: list) -> None:
    async with Kaleido() as k:
        for fig, path in svg_tasks:
            await k.write_fig(fig, path)
            print(f"  saved {path}")


def export_svgs(svg_tasks: list) -> None:
    """Batch-export all queued figures to SVG via a single Kaleido session."""
    print(f"\nExporting {len(svg_tasks)} SVG figures...")
    asyncio.run(_run_kaleido(svg_tasks))


# ---------------------------------------------------------------------------
# Hit-discovery helpers
# ---------------------------------------------------------------------------


def _hit_panel_data(data: dict, hit_threshold: float) -> dict:
    """Package a loaded data dict into the format expected by the grid plotter."""
    return {
        "pool_history_df": data["pool_history_long"],
        "learning_curve_df": data["learning_curve_long"],
        "max_hits": int((data["df_pool"]["pEC50"] >= hit_threshold).sum()),
        "pool_size": len(data["df_pool"]),
    }


# Color palette: encodes (model, ChEMBL) combinations consistently across plots.
_GAP_COLORS: dict[str, str] = {
    "ChemProp": "#DC143C",          # crimson
    "ChemProp + ChEMBL": "#8B0000", # dark red
    "CheMeleon": "#228B22",         # forest green
    "CheMeleon + ChEMBL": "#006400",# dark green
}
# Dash style: encodes target dataset.
_GAP_DASH: dict[str, str] = {
    "PXR": "solid",
    "ASAP": "dash",
}


def generate_hit_nonhit_gap_curve(
    asap_data: "list[dict]",
    pxr_data: "list[dict]",
    output_dir: Path,
    asap_hit_threshold: float = 7.0,
    pxr_hit_threshold: float = 6.0,
    svg_queue: "list | None" = None,
) -> None:
    """Generate and save the hit/non-hit gap vs. labeled pool size figure.

    Plots the Exploitation hit/non-hit predicted-activity gap for all model
    initializations (ChemProp, CheMeleon, and PXR-only ChEMBL variants) on a
    single panel.  Color encodes model initialization; line dash encodes
    target dataset (solid = PXR, dashed = ASAP).

    Parameters
    ----------
    asap_data : list of dict
        Loaded data dicts for ASAP configs (order matches ``ASAP_CONFIGS``).
    pxr_data : list of dict
        Loaded data dicts for PXR configs (order matches ``PXR_CONFIGS``).
    output_dir : Path
        Directory to write the output HTML (and queue SVG if provided).
    asap_hit_threshold : float
        Hit activity threshold for ASAP Mpro.
    pxr_hit_threshold : float
        Hit activity threshold for PXR.
    svg_queue : list or None
        If provided, ``(fig, svg_path)`` is appended for later batch export.

    """
    print("\n=== Hit/non-hit gap curve (Exploitation) ===")
    traces = []

    asap_configs_meta = [(label, "ASAP") for label, *_ in ASAP_CONFIGS]
    pxr_configs_meta  = [(label, "PXR")  for label, *_ in PXR_CONFIGS]

    for (label, target), data, hit_thr in [
        *zip(asap_configs_meta, asap_data, [asap_hit_threshold] * len(asap_data)),
        *zip(pxr_configs_meta,  pxr_data,  [pxr_hit_threshold]  * len(pxr_data)),
    ]:
        runs = data["all_runs"].get("Exploitation", [])
        if not runs:
            print(f"  [skip] {label} ({target}): no Exploitation runs")
            continue

        n_labeled, gap_mean, gap_std = _compute_gap_trajectory(
            runs,
            data["df_pool"],
            activity_col="pEC50",
            hit_threshold=hit_thr,
        )
        if len(n_labeled) == 0:
            print(f"  [skip] {label} ({target}): no gap data")
            continue

        trace_label = f"{label} ({target})"
        traces.append({
            "label": trace_label,
            "n_labeled": n_labeled,
            "gap_mean": gap_mean,
            "gap_std": gap_std,
            "color": _GAP_COLORS.get(label, "#888888"),
            "dash": _GAP_DASH.get(target, "solid"),
        })
        print(f"  {trace_label}: {len(n_labeled)} iterations")

    if not traces:
        print("  [skip] no valid traces — figure not generated.")
        return

    fig = alp.plot_hit_nonhit_gap_curve(traces)
    stem = "hit_nonhit_gap_Exploitation"
    html_path = str(output_dir / f"{stem}.html")
    svg_path = str(output_dir / f"{stem}.svg")
    alp.write_html_both(fig=fig, path=html_path)
    print(f"  saved {html_path}")
    if svg_queue is not None:
        svg_queue.append((fig, svg_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_mirror_distribution_animation(
    chemprop_data: dict,
    chemeleon_data: dict,
    output_dir: Path,
    file_slug: str,
    target_label: str,
    hit_threshold: float = 6.0,
    activity_col: str = "pEC50",
) -> None:
    """Generate and save a ChemProp vs CheMeleon mirror distribution animation.

    Builds pool-mode Exploitation frame data for both models and produces a
    single mirror figure (ChemProp top / CheMeleon bottom) saved to
    ``output_dir/<file_slug>_mirror_distribution_Exploitation.html``.

    Parameters
    ----------
    chemprop_data, chemeleon_data : dict
        Assembled split-data dicts as returned by ``load_config_data``.
    output_dir : Path
        Directory to write the output HTML.
    file_slug : str
        Short identifier prepended to the output filename (e.g. ``"pxr"``).
    target_label : str
        Human-readable target name used in log messages (e.g. ``"PXR"``).
    hit_threshold : float
        Hit activity threshold.
    activity_col : str
        Activity column name (default ``"pEC50"``).
    """
    print(f"\n=== {target_label} mirror distribution animation (Exploitation) ===")

    strategy = "Exploitation"
    cp_runs = chemprop_data["all_runs"].get(strategy, [])
    cm_runs = chemeleon_data["all_runs"].get(strategy, [])

    history_cp, y_true_cp, n_seeds_cp = _build_distribution_frames(
        cp_runs, chemprop_data["df_pool"], activity_col, strategy, hit_threshold,
    )
    history_cm, y_true_cm, n_seeds_cm = _build_distribution_frames(
        cm_runs, chemeleon_data["df_pool"], activity_col, strategy, hit_threshold,
    )

    if not history_cp or not history_cm:
        print("  [skip] insufficient frame data for one or both models.")
        return
    if y_true_cp is None or y_true_cm is None:
        print("  [skip] pool-mode y_true not available — check that df_pool was provided.")
        return

    n_seeds = max(n_seeds_cp, n_seeds_cm)
    seed_label = f"{n_seeds} seeds"

    print(f"  ChemProp frames: {len(history_cp)}, CheMeleon frames: {len(history_cm)}")
    fig = alp.plot_mirror_distribution_animation(
        history_a=history_cp,
        history_b=history_cm,
        label_a="ChemProp",
        label_b="CheMeleon",
        hit_threshold=hit_threshold,
        activity_col=activity_col,
        strategy=strategy,
        seed_label=seed_label,
        per_frame_y_true_a=y_true_cp,
        per_frame_y_true_b=y_true_cm,
        n_seeds=n_seeds,
    )
    out_path = output_dir / f"{file_slug}_mirror_distribution_Exploitation.html"
    alp.write_html_both(fig=fig, path=out_path, animation_opts=alp.DIST_ANIMATION_OPTS)
    print(f"  saved {out_path}")


def main() -> None:
    """Entry point: parse CLI arguments, load all configs, and write figures."""
    parser = argparse.ArgumentParser(
        description="Generate cross-config comparison figures for ASAP and PXR."
    )
    parser.add_argument(
        "--asap-hit-threshold",
        type=float,
        default=7.0,
        metavar="FLOAT",
        help="Hit threshold for ASAP Mpro figures (default: 7.0)",
    )
    parser.add_argument(
        "--pxr-hit-threshold",
        type=float,
        default=6.0,
        metavar="FLOAT",
        help="Hit threshold for PXR figures (default: 6.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/combined",
        metavar="DIR",
        help="Output directory for figures (default: results/combined/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asap_hit_thr: float = args.asap_hit_threshold
    pxr_hit_thr: float = args.pxr_hit_threshold

    svg_queue: list = []

    # -----------------------------------------------------------------------
    # Load all configs
    # -----------------------------------------------------------------------
    print("=== Loading ASAP configs ===")
    asap_data: list[dict] = []
    for label, cfg_yaml, results_dir, slug in ASAP_CONFIGS:
        print(f"  [{slug}] {results_dir}")
        asap_data.append(load_config_data(cfg_yaml, results_dir))

    print("\n=== Loading PXR configs ===")
    pxr_data: list[dict] = []
    for label, cfg_yaml, results_dir, slug, _r, _c in PXR_CONFIGS:
        print(f"  [{slug}] {results_dir}")
        pxr_data.append(load_config_data(cfg_yaml, results_dir))

    # Strategy list inferred from first loaded config.
    strategies = list(asap_data[0]["all_runs"].keys())

    # Learning-curve metric specs: (metric_col, y-axis label, filename slug)
    lc_metrics = [
        ("mae", "MAE (pEC50 units)", "mae"),
        ("ktau", "Kendall's τ", "ktau"),
        ("sigma_error_rho", "Spearman ρ(σ, |error|)", "sigma_error"),
    ]

    # -----------------------------------------------------------------------
    # Individual-panel figures
    # -----------------------------------------------------------------------
    print("\n=== Individual panel figures ===")

    # ASAP individual panels
    for (label, _, _, slug), data in zip(ASAP_CONFIGS, asap_data):
        for metric_col, ylabel, metric_slug in lc_metrics:
            fig = alp.plot_learning_curve_with_bands(
                data["learning_curve_summary"],
                metric_col=metric_col,
                ylabel=ylabel,
                strategy_order=strategies,
                color_map=STRATEGY_COLORS,
            )
            _save_fig(fig, f"{slug}_{metric_slug}", output_dir, svg_queue)

        fig = alp.plot_hit_discovery_curve(
            data["pool_history_long"],
            data["learning_curve_long"],
            hit_threshold=asap_hit_thr,
            max_hits=int((data["df_pool"]["pEC50"] >= asap_hit_thr).sum()),
            pool_size=len(data["df_pool"]),
            strategy_order=strategies,
            color_map=STRATEGY_COLORS,
        )
        _save_fig(fig, f"{slug}_hit_discovery", output_dir, svg_queue)

    # PXR individual panels
    for (label, _, _, slug, _r, _c), data in zip(PXR_CONFIGS, pxr_data):
        for metric_col, ylabel, metric_slug in lc_metrics:
            fig = alp.plot_learning_curve_with_bands(
                data["learning_curve_summary"],
                metric_col=metric_col,
                ylabel=ylabel,
                strategy_order=strategies,
                color_map=STRATEGY_COLORS,
            )
            _save_fig(fig, f"{slug}_{metric_slug}", output_dir, svg_queue)

        fig = alp.plot_hit_discovery_curve(
            data["pool_history_long"],
            data["learning_curve_long"],
            hit_threshold=pxr_hit_thr,
            max_hits=int((data["df_pool"]["pEC50"] >= pxr_hit_thr).sum()),
            pool_size=len(data["df_pool"]),
            strategy_order=strategies,
            color_map=STRATEGY_COLORS,
        )
        _save_fig(fig, f"{slug}_hit_discovery", output_dir, svg_queue)

    # -----------------------------------------------------------------------
    # Combined ASAP grid (1 × 2: ChemProp | CheMeleon)
    # -----------------------------------------------------------------------
    print("\n=== Combined ASAP grid figures ===")

    asap_lc_panels = [[
        (label, data["learning_curve_summary"])
        for (label, _, _, _slug), data in zip(ASAP_CONFIGS, asap_data)
    ]]
    asap_hit_panels: list[list[tuple[str, dict | None]]] = [[
        (label, _hit_panel_data(data, asap_hit_thr))
        for (label, _, _, _slug), data in zip(ASAP_CONFIGS, asap_data)
    ]]

    for metric_col, ylabel, metric_slug in lc_metrics:
        fig = alp.plot_learning_curve_grid(
            asap_lc_panels,
            metric_col=metric_col,
            ylabel=ylabel,
            strategy_order=strategies,
            color_map=STRATEGY_COLORS,
        )
        _save_fig(fig, f"asap_combined_{metric_slug}", output_dir, svg_queue)

    fig = alp.plot_hit_discovery_curve_grid(
        asap_hit_panels,
        strategy_order=strategies,
        color_map=STRATEGY_COLORS,
        hit_threshold=asap_hit_thr,
    )
    _save_fig(fig, "asap_combined_hit_discovery", output_dir, svg_queue)

    # -----------------------------------------------------------------------
    # Combined PXR grid (2 × 2)
    # Rows: no-ChEMBL (idx 0,1) / +ChEMBL (idx 2,3)
    # Cols: ChemProp / CheMeleon
    # -----------------------------------------------------------------------
    print("\n=== Combined PXR grid figures ===")

    pxr_lc_panels = [
        [
            (PXR_CONFIGS[0][0], pxr_data[0]["learning_curve_summary"]),
            (PXR_CONFIGS[1][0], pxr_data[1]["learning_curve_summary"]),
        ],
        [
            (PXR_CONFIGS[2][0], pxr_data[2]["learning_curve_summary"]),
            (PXR_CONFIGS[3][0], pxr_data[3]["learning_curve_summary"]),
        ],
    ]
    pxr_hit_panels: list[list[tuple[str, dict | None]]] = [
        [
            (PXR_CONFIGS[0][0], _hit_panel_data(pxr_data[0], pxr_hit_thr)),
            (PXR_CONFIGS[1][0], _hit_panel_data(pxr_data[1], pxr_hit_thr)),
        ],
        [
            (PXR_CONFIGS[2][0], _hit_panel_data(pxr_data[2], pxr_hit_thr)),
            (PXR_CONFIGS[3][0], _hit_panel_data(pxr_data[3], pxr_hit_thr)),
        ],
    ]

    for metric_col, ylabel, metric_slug in lc_metrics:
        fig = alp.plot_learning_curve_grid(
            pxr_lc_panels,
            metric_col=metric_col,
            ylabel=ylabel,
            strategy_order=strategies,
            color_map=STRATEGY_COLORS,
        )
        _save_fig(fig, f"pxr_combined_{metric_slug}", output_dir, svg_queue)

    fig = alp.plot_hit_discovery_curve_grid(
        pxr_hit_panels,
        strategy_order=strategies,
        color_map=STRATEGY_COLORS,
        hit_threshold=pxr_hit_thr,
    )
    _save_fig(fig, "pxr_combined_hit_discovery", output_dir, svg_queue)

    # -----------------------------------------------------------------------
    # Mirror distribution animations (ChemProp vs CheMeleon, Exploitation)
    # -----------------------------------------------------------------------
    generate_mirror_distribution_animation(
        asap_data[0],  # ChemProp (ASAP_CONFIGS index 0)
        asap_data[1],  # CheMeleon (ASAP_CONFIGS index 1)
        output_dir,
        file_slug="asap",
        target_label="ASAP",
        hit_threshold=asap_hit_thr,
    )
    generate_mirror_distribution_animation(
        pxr_data[0],  # ChemProp no-ChEMBL (PXR_CONFIGS index 0)
        pxr_data[1],  # CheMeleon no-ChEMBL (PXR_CONFIGS index 1)
        output_dir,
        file_slug="pxr",
        target_label="PXR",
        hit_threshold=pxr_hit_thr,
    )

    # -----------------------------------------------------------------------
    # Hit/non-hit gap curve (Exploitation, all configs, single panel)
    # -----------------------------------------------------------------------
    generate_hit_nonhit_gap_curve(
        asap_data=asap_data,
        pxr_data=pxr_data,
        output_dir=output_dir,
        asap_hit_threshold=asap_hit_thr,
        pxr_hit_threshold=pxr_hit_thr,
        svg_queue=svg_queue,
    )

    # -----------------------------------------------------------------------
    # Batch SVG export
    # -----------------------------------------------------------------------
    export_svgs(svg_queue)
    print(f"\nDone. All figures written to {output_dir}/")
    os._exit(0)


if __name__ == "__main__":
    main()
