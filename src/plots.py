"""Plotly and Faerun visualization functions for the active learning pipeline.

Each function returns a configured figure object ready to display or export.
Figures share a consistent visual style: white background, black axes, and
per-strategy colors from ``src.helpers.STRATEGY_COLORS``.
"""

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from faerun import Faerun


def plot_learning_curve_with_bands(
    learning_curve_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    strategy_order: list[str],
    color_map: dict | None = None,
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """Interactive line plot with shaded ±1σ band per strategy.

    Parameters
    ----------
    learning_curve_df : pd.DataFrame
        Must have columns: ``strategy``, ``n_labeled``, ``{metric_col}_mean``,
        ``{metric_col}_lower``, and ``{metric_col}_upper``.
    metric_col : str
        Base name of the metric; expects columns ``{metric_col}_mean``,
        ``{metric_col}_lower``, and ``{metric_col}_upper``.
    ylabel : str
        Y-axis label.
    strategy_order : list[str]
        Strategies to plot, controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with one mean line and one shaded ±1σ band per strategy.

    """
    fig = go.Figure()

    for strategy in strategy_order:
        df_sub = learning_curve_df[
            learning_curve_df["strategy"] == strategy
        ].sort_values("n_labeled")
        if df_sub.empty:
            continue

        color = color_map.get(strategy, None) if color_map else None

        # Upper bound — invisible line, anchors the fill
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub[f"{metric_col}_upper"],
                mode="lines",
                line=dict(width=0),
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
                **({"line_color": color} if color else {}),
            )
        )
        # Lower bound — fills back to the upper bound
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub[f"{metric_col}_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=(
                    _hex_to_rgba(color, 0.15) if color else "rgba(128,128,128,0.15)"
                ),
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub[f"{metric_col}_mean"],
                mode="lines",
                name=strategy,
                legendgroup=strategy,
                line=dict(width=2, color=color),
            )
        )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=ylabel,
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        plot_bgcolor="white",
        margin=dict(t=20),
        width=width,
        height=height,
    )
    return fig


def plot_hit_discovery_curve(
    pool_history_df: pd.DataFrame,
    learning_curve_df: pd.DataFrame,
    hit_threshold: float,
    max_hits: int,
    strategy_order: list[str],
    color_map: dict | None = None,
    value_col: str = "pEC50",
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """Interactive line plot of cumulative hits found vs. number of labeled molecules.

    Shows mean ± 1σ bands per strategy and a dashed reference line at the absolute
    hit ceiling.

    Parameters
    ----------
    pool_history_df : pd.DataFrame
        Must have columns: strategy, seed, iteration, {value_col}. One row per
        compound in the labeled pool at each (seed, iteration).
    learning_curve_df : pd.DataFrame
        Must have columns: strategy, seed, iteration, n_labeled. Used to map
        (strategy, seed, iteration) to labeled-pool size for the x-axis.
    hit_threshold : float
        Minimum value_col value for a compound to be counted as a hit.
    max_hits : int
        Total number of hits in the full compound pool — an absolute upper bound
        independent of strategy or seed. Used as the dashed reference line and as
        the ceiling for the upper error band.
    strategy_order : list[str]
        Strategies to plot, controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    value_col : str, optional
        Column name for the activity values used to define hits.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure showing cumulative hits vs. labeled pool size, with
        a dashed reference line at the total hit ceiling.

    """
    # Count hits per (strategy, seed, iteration), then map to n_labeled
    hits_per_seed_iter = (
        pool_history_df.groupby(["strategy", "seed", "iteration"])[value_col]
        .apply(lambda s: (s >= hit_threshold).sum())
        .reset_index(name="n_hits")
    )
    n_labeled = learning_curve_df[
        ["strategy", "seed", "iteration", "n_labeled"]
    ].drop_duplicates()
    hits_df = hits_per_seed_iter.merge(
        n_labeled, on=["strategy", "seed", "iteration"], how="left"
    )

    # Aggregate across seeds: mean ± std per (strategy, n_labeled)
    # std is NaN when only one seed is present; fill to 0 so bands collapse gracefully
    hits_summary = (
        hits_df.groupby(["strategy", "n_labeled"])["n_hits"]
        .agg(n_hits_mean="mean", n_hits_std="std")
        .reset_index()
        .fillna({"n_hits_std": 0})
    )
    hits_summary["n_hits_lower"] = (
        hits_summary["n_hits_mean"] - hits_summary["n_hits_std"]
    )
    hits_summary["n_hits_upper"] = np.minimum(
        hits_summary["n_hits_mean"] + hits_summary["n_hits_std"], max_hits
    )

    fig = go.Figure()

    for strategy in strategy_order:
        df_sub = hits_summary[hits_summary["strategy"] == strategy].sort_values(
            "n_labeled"
        )
        if df_sub.empty:
            continue
        color = color_map.get(strategy, None) if color_map else None

        # Upper bound — invisible line, anchors the fill
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["n_hits_upper"],
                mode="lines",
                line=dict(width=0),
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
                **({"line_color": color} if color else {}),
            )
        )
        # Lower bound — fills back to the upper bound
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["n_hits_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=(
                    _hex_to_rgba(color, 0.15) if color else "rgba(128,128,128,0.15)"
                ),
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["n_hits_mean"],
                mode="lines",
                name=strategy,
                legendgroup=strategy,
                line=dict(width=2, color=color),
            )
        )

    fig.add_hline(
        y=max_hits,
        line=dict(color="black", width=1.2, dash="dash"),
        annotation_text=f"Total hits in pool ({max_hits})",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=f"Hits Found ({value_col} ≥ {hit_threshold})",
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        plot_bgcolor="white",
        margin=dict(t=20),
        width=width,
        height=height,
    )
    return fig


def plot_calibration_curve_before_after(
    expected_before: np.ndarray,
    observed_before: np.ndarray,
    expected_after: np.ndarray,
    observed_after: np.ndarray,
    label_before="Before calibration",
    label_after="After calibration",
    width: int = 520,
    height: int = 520,
) -> go.Figure:
    """Interactive plot of diagonal (perfect calibration) and two calibration curves.

    Fills the area between each curve and the diagonal to show miscalibration.

    Parameters
    ----------
    expected_before, observed_before : np.ndarray
        Expected confidence levels and observed proportions before calibration.
    expected_after, observed_after : np.ndarray
        Expected confidence levels and observed proportions after calibration.
    label_before, label_after : str
        Legend labels for each calibration curve.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with the perfect-calibration diagonal and two
        calibration curves (before and after isotonic regression), with filled
        miscalibration areas.

    """
    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="black", dash="dash", width=1.5),
        )
    )

    # Before calibration
    # Diagonal reference (invisible) used as fill anchor
    fig.add_trace(
        go.Scatter(
            x=expected_before,
            y=expected_before,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=expected_before,
            y=observed_before,
            mode="lines",
            name=label_before,
            line=dict(color="#D62728", width=2),
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.1)",
        )
    )

    # After calibration
    # Diagonal reference (invisible) used as fill anchor
    fig.add_trace(
        go.Scatter(
            x=expected_after,
            y=expected_after,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=expected_after,
            y=observed_after,
            mode="lines",
            name=label_after,
            line=dict(color="#1f77b4", width=2),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.1)",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Expected Confidence Level",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            dtick=0.2,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            title="Observed Proportion",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            dtick=0.2,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="white",
        margin=dict(t=20, r=20),
        width=width,
        height=height,
    )
    return fig


def plot_gtm_selection_animation(
    gtm_coords: np.ndarray,
    selection_history: list[dict],
    title: str = "Active Learning Selection (GTM)",
    background_gtm_coords: np.ndarray | None = None,
) -> go.Figure:
    """Animated Plotly scatter of compound selections on a pre-computed GTM embedding.

    Renders one frame per active learning iteration with three layers:
    gray background (full pool), blue accumulation (all prior selections),
    and red highlight (current iteration's selections). Includes a slider
    and play/pause controls for use directly in a Jupyter Notebook.

    Parameters
    ----------
    gtm_coords : np.ndarray
        Array of shape (n_pool, 2) containing 2D GTM coordinates for every
        compound in the pool, in the same row order as ``df_pool``.
    selection_history : list[dict]
        List of per-iteration state dicts produced by ``run_active_learning``.
        Each dict must contain the key ``"selected_pool_indices"``, a list of
        integer indices into the pool corresponding to compounds selected that
        iteration.
    title : str, optional
        Title displayed above the plot.
    background_gtm_coords : np.ndarray or None, optional
        Array of shape (n_bg, 2) with GTM coordinates for additional background
        molecules not part of the active learning pool. When provided, these
        points are merged into the static gray background trace and displayed
        identically to unselected pool compounds (light gray, small).

    Returns
    -------
    go.Figure
        Animated Plotly figure with per-iteration frames, a play/pause button, and
        an iteration slider. Call ``fig.show()`` to display inline in a Jupyter Notebook.

    """
    gtm_coords = np.asarray(gtm_coords)
    if background_gtm_coords is not None:
        background_gtm_coords = np.asarray(background_gtm_coords)

    n_pool = len(gtm_coords)
    all_idx = np.arange(n_pool)

    # Merge background molecules into the gray layer (static, not animated)
    if background_gtm_coords is not None:
        bg_x = np.concatenate([gtm_coords[all_idx, 0], background_gtm_coords[:, 0]])
        bg_y = np.concatenate([gtm_coords[all_idx, 1], background_gtm_coords[:, 1]])
    else:
        bg_x = gtm_coords[all_idx, 0]
        bg_y = gtm_coords[all_idx, 1]

    frames = []
    slider_steps = []

    for i, state in enumerate(selection_history):
        current_idx = np.array(state["selected_pool_indices"], dtype=int)
        prior_idx = np.array(
            [idx for s in selection_history[:i] for idx in s["selected_pool_indices"]],
            dtype=int,
        )

        history_x = gtm_coords[prior_idx, 0] if len(prior_idx) > 0 else [None]
        history_y = gtm_coords[prior_idx, 1] if len(prior_idx) > 0 else [None]

        frame = go.Frame(
            data=[
                # Trace index 1: accumulated prior selections (blue)
                go.Scatter(
                    x=history_x,
                    y=history_y,
                    mode="markers",
                    marker=dict(
                        color="royalblue",
                        size=7,
                        opacity=0.75,
                        line=dict(color="white", width=0.5),
                    ),
                    name="Prior selections",
                    hoverinfo="skip",
                ),
                # Trace index 2: current iteration selections (red)
                go.Scatter(
                    x=gtm_coords[current_idx, 0],
                    y=gtm_coords[current_idx, 1],
                    mode="markers",
                    marker=dict(
                        color="crimson",
                        size=9,
                        opacity=0.9,
                        line=dict(color="white", width=0.5),
                    ),
                    name="Current iteration",
                    hoverinfo="skip",
                ),
            ],
            traces=[1, 2],
            name=str(i),
        )
        frames.append(frame)

        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        frame=dict(duration=600, redraw=True),
                        mode="immediate",
                        transition=dict(duration=200),
                    ),
                ],
                label=str(state["iteration"]),
            )
        )

    fig = go.Figure(
        data=[
            # Trace 0: full compound pool background (gray, constant)
            go.Scatter(
                x=bg_x,
                y=bg_y,
                mode="markers",
                marker=dict(color="lightgray", size=5, opacity=0.5),
                name="All compounds",
                hoverinfo="skip",
            ),
            # Trace 1: prior selections placeholder (overwritten by each frame)
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color="royalblue",
                    size=7,
                    opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                name="Prior selections",
                hoverinfo="skip",
            ),
            # Trace 2: current selection placeholder (overwritten by each frame)
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color="crimson",
                    size=9,
                    opacity=0.9,
                    line=dict(color="white", width=0.5),
                ),
                name="Current selection",
                hoverinfo="skip",
            ),
        ],
        frames=frames,
        layout=go.Layout(
            title=title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="GTM dimension 1", showgrid=False, zeroline=False),
            yaxis=dict(title="GTM dimension 2", showgrid=False, zeroline=False),
            legend=dict(itemsizing="constant"),
            hovermode=False,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0.85,
                    x=1.2,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=600, redraw=True),
                                    fromcurrent=True,
                                    loop=True,
                                    transition=dict(duration=200),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(
                        prefix="Iteration ",
                        visible=True,
                        xanchor="center",
                        font=dict(color="black"),
                    ),
                    font=dict(color="rgba(0,0,0,0)"),
                    pad=dict(t=50),
                    steps=slider_steps,
                )
            ],
        ),
    )

    return fig


def plot_tmap_faerun(
    tmap_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smiles_list: list[str],
    selection_history: list[dict],
    output_name: str = "tmap_selection",
    output_path: str = "./",
    title: str = "Active Learning Selection (TMAP)",
    colormap: str = "viridis",
    point_scale: float = 3.0,
    n_background: int = 0,
    background_point_scale: float = 1.0,
) -> Faerun:
    """Faerun scatter plot of compound selections on a pre-computed TMAP layout.

    Each compound node is colored by the active learning iteration in which it
    was first selected, using faerun's native categorical colormapping. Compounds
    never selected are labeled ``"Unselected"``. SMILES strings are embedded as
    labels for tooltip display. The TMAP minimum-spanning-tree is drawn as a tree
    layer. Calling ``.plot()`` on the returned object renders the interactive HTML.

    Parameters
    ----------
    tmap_layout : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t)`` as returned by ``helpers.smiles_to_tmap``, computed on
        **all** molecules in the order ``pool_smiles + background_smiles``. When
        ``n_background > 0`` the last ``n_background`` entries of x/y correspond
        to background molecules; the MST edges (s, t) span all molecules so the
        tree faithfully reflects the full chemical space.
    smiles_list : list[str]
        SMILES strings in the same row order as ``tmap_layout`` (pool first, then
        background). Shown as faerun labels / tooltips.
    selection_history : list[dict]
        List of per-iteration state dicts produced by ``run_active_learning``.
        Each dict must contain ``"selected_pool_indices"`` (list of int into the
        pool, i.e. indices ``0..len(smiles_list) - n_background - 1``) and
        ``"iteration"`` (int).
    output_name : str, optional
        Base filename (without extension) for faerun's HTML output.
    output_path : str, optional
        Directory in which to write the output file.
    title : str, optional
        Plot title.
    colormap : str, optional
        Matplotlib colormap name for categorical coloring. ``"viridis"`` is
        the default; ``"tab20"`` works well when many iterations need distinct
        hues.
    point_scale : float, optional
        Relative size of pool (foreground) scatter points. Default 3.0.
    n_background : int, optional
        Number of background molecules appended at the **end** of
        ``tmap_layout`` / ``smiles_list``. These entries are displayed as gray
        (``"Unselected"``) regardless of ``selection_history``. Default is 0
        (no background molecules).

        .. note::
            Because TMAP computes a single global MST layout, background and
            pool molecules must be embedded jointly. Compute coordinates with
            ``helpers.smiles_to_tmap(pool_smiles + background_smiles)`` and
            pass the full result as ``tmap_layout``.
    background_point_scale : float, optional
        Relative size of background scatter points. Default 1.0, smaller than
        the foreground default so background molecules recede visually.

    Returns
    -------
    Faerun
        Configured faerun instance. Call ``f.plot(output_name, output_path)``
        to regenerate the HTML, or ``f.plot(output_name, output_path,
        notebook_height=500)`` to display inline in a Jupyter Notebook.

    """
    x, y, s, t = tmap_layout

    # Build iteration label per compound: index 0 = "Unselected", index k+1 = str(k)
    iterations = sorted({state["iteration"] for state in selection_history})
    # Category 0 is reserved for "Unselected"
    iter_to_cat = {it: idx + 1 for idx, it in enumerate(iterations)}
    legend_labels = [(0, "Unselected")] + [
        (idx + 1, str(it)) for idx, it in enumerate(iterations)
    ]

    # Pool molecules: colored by first-selected iteration; background molecules: gray (0)
    c = np.zeros(len(x), dtype=int)
    for state in selection_history:
        cat = iter_to_cat[state["iteration"]]
        for idx in state["selected_pool_indices"]:
            if c[idx] == 0:  # keep the first iteration that selected this compound
                c[idx] = cat

    # Per-point sizes: foreground uses point_scale, background uses background_point_scale
    n_pool = len(x) - n_background
    s_vals = np.full(len(x), point_scale, dtype=float)
    if n_background > 0:
        s_vals[n_pool:] = background_point_scale

    # Build a ListedColormap: gray for "Unselected" (index 0), then N viridis
    # colors sampled across the full colormap range for each iteration category.
    # Using a ListedColormap avoids the matplotlib integer-indexing issue where
    # small integers (0, 1, 2…) all land at the dark end of continuous colormaps
    n_iter = len(iterations)
    base_cmap = mcm.get_cmap(colormap) if isinstance(colormap, str) else colormap
    iter_colors = [base_cmap(i / max(n_iter - 1, 1)) for i in range(n_iter)]
    listed_cmap = mcolors.ListedColormap(
        [(0.55, 0.55, 0.55, 1.0)] + iter_colors,
        N=1 + n_iter,
    )

    f = Faerun(
        title=title,
        clear_color="#111111",
        coords=False,
        view="front",
        thumbnail_width=500,
    )

    f.add_scatter(
        "tmap",
        {
            "x": x,
            "y": y,
            "c": c,
            "s": s_vals,
            "labels": smiles_list,
        },
        colormap=listed_cmap,
        categorical=True,
        has_legend=True,
        legend_title="AL Iteration",
        legend_labels=legend_labels,
        point_scale=1.0,
        shader="smoothCircle",
    )

    f.add_tree(
        "tmap_tree",
        {"from": s, "to": t, "x": x, "y": y},
        point_helper="tmap",
    )

    f.plot(output_name, output_path, template="smiles")
    return f


def plot_calibration_area_per_iteration(
    cal_area_df: pd.DataFrame,
    strategy_order: list[str],
    color_map: dict | None = None,
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """Line plot of miscalibration area per iteration for each strategy.

    Shows both before- and after-calibration values on the same panel. Solid lines
    represent post-calibration area; dashed lines represent pre-calibration (raw
    ensemble) area. Both are shown with ±1σ bands aggregated across seeds.
    Lower values indicate better calibration.

    Parameters
    ----------
    cal_area_df : pd.DataFrame
        Must have columns: ``strategy``, ``n_labeled``,
        ``miscal_area_mean``, ``miscal_area_lower``, ``miscal_area_upper``,
        ``miscal_area_pre_cal_mean``, ``miscal_area_pre_cal_lower``,
        ``miscal_area_pre_cal_upper``.
    strategy_order : list[str]
        Strategies to plot, controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with solid post-calibration lines, dashed
        pre-calibration lines, and ±1σ bands per strategy.

    """
    fig = go.Figure()

    for strategy in strategy_order:
        df_sub = cal_area_df[cal_area_df["strategy"] == strategy].sort_values(
            "n_labeled"
        )
        if df_sub.empty:
            continue

        color = color_map.get(strategy, None) if color_map else None
        fill_color = _hex_to_rgba(color, 0.15) if color else "rgba(128,128,128,0.15)"

        for prefix, dash, show_legend, label_suffix in [
            ("miscal_area", "solid", True, ""),
            ("miscal_area_pre_cal", "dot", True, " (pre-cal)"),
        ]:
            # Upper bound — invisible, anchors fill
            fig.add_trace(
                go.Scatter(
                    x=df_sub["n_labeled"],
                    y=df_sub[f"{prefix}_upper"],
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=f"{strategy}{label_suffix}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound — fills back to upper
            fig.add_trace(
                go.Scatter(
                    x=df_sub["n_labeled"],
                    y=df_sub[f"{prefix}_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    legendgroup=f"{strategy}{label_suffix}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=df_sub["n_labeled"],
                    y=df_sub[f"{prefix}_mean"],
                    mode="lines",
                    name=f"{strategy}{label_suffix}",
                    legendgroup=f"{strategy}{label_suffix}",
                    showlegend=show_legend,
                    line=dict(width=2, color=color, dash=dash),
                )
            )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title="Miscalibration Area (lower = better)",
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        plot_bgcolor="white",
        margin=dict(t=20),
        width=width,
        height=height,
    )
    return fig


def plot_sigma_error_correlation(
    summary_df: pd.DataFrame,
    strategy_order: list[str],
    color_map: dict | None = None,
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """Spearman ρ(σ, |error|) vs. labeled pool size for each strategy.

    A positive ρ indicates that the ensemble's predicted standard deviation σ
    correctly ranks test compounds by how wrong the model is — i.e. high-σ
    compounds tend to have larger absolute prediction errors. This is a more
    informative diagnostic than calibration-curve area when calibration and
    evaluation distributions differ (e.g. scaffold-split holdouts).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must have columns: ``strategy``, ``n_labeled``,
        ``sigma_error_rho_mean``, ``sigma_error_rho_lower``,
        ``sigma_error_rho_upper``.
    strategy_order : list[str]
        Strategies to plot, controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with Spearman ρ(σ, |error|) vs. labeled pool
        size, with ±1σ bands and a dashed y = 0 reference line.

    """
    fig = go.Figure()

    # y = 0 reference line
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    for strategy in strategy_order:
        df_sub = summary_df[summary_df["strategy"] == strategy].sort_values("n_labeled")
        if df_sub.empty:
            continue

        color = color_map.get(strategy, None) if color_map else None
        fill_color = _hex_to_rgba(color, 0.15) if color else "rgba(128,128,128,0.15)"

        # Upper bound — invisible, anchors fill
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["sigma_error_rho_upper"],
                mode="lines",
                line=dict(width=0),
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Lower bound — fills back to upper
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["sigma_error_rho_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                legendgroup=strategy,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["sigma_error_rho_mean"],
                mode="lines",
                name=strategy,
                legendgroup=strategy,
                showlegend=True,
                line=dict(width=2, color=color),
            )
        )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title="Spearman ρ(σ, |error|)",
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            # range=[-0.1, 0.3],
        ),
        plot_bgcolor="white",
        margin=dict(t=20),
        width=width,
        height=height,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a CSS hex color string to an ``rgba()`` string with the given alpha.

    Parameters
    ----------
    hex_color : str
        CSS hex color, with or without a leading ``#``. Both 3- and 6-digit
        forms are accepted (e.g., ``"#abc"`` or ``"#aabbcc"``).
    alpha : float
        Opacity value in the range [0.0, 1.0].

    Returns
    -------
    str
        CSS ``rgba()`` string, e.g. ``"rgba(170,187,204,0.5)"``.

    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
