import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def plot_learning_curve_with_bands(
    learning_curve_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    strategy_order: list[str],
    color_map: dict | None = None,
    figsize=(8, 5),
) -> plt.Figure:
    """
    Line plot with shaded ±1σ band per strategy.
    DataFrame must have columns: strategy, n_labeled, {metric_col}_mean, {metric_col}_lower, {metric_col}_upper.
    Uses matplotlib directly (no seaborn line functions for the band). Adds legend, axis labels, grid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for strategy in strategy_order:
        df_sub = learning_curve_df[
            learning_curve_df["strategy"] == strategy
        ].sort_values("n_labeled")
        if df_sub.empty:
            continue

        color = color_map.get(strategy, None) if color_map else None

        ax.plot(
            df_sub["n_labeled"],
            df_sub[f"{metric_col}_mean"],
            label=strategy,
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            df_sub["n_labeled"],
            df_sub[f"{metric_col}_lower"],
            df_sub[f"{metric_col}_upper"],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Number of Labeled Molecules")
    ax.set_ylabel(ylabel)
    ax.legend(title="Strategy")
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    return fig


def plot_hit_discovery_curve(
    pool_history_df: pd.DataFrame,
    learning_curve_df: pd.DataFrame,
    hit_threshold: float,
    strategy_order: list[str],
    color_map: dict | None = None,
    value_col: str = "pEC50",
    figsize=(8, 5),
) -> plt.Figure:
    """
    Line plot of cumulative hits found in the labeled pool vs. number of labeled molecules,
    with one line per strategy and a dashed reference line at the total hits available in the pool.

    Parameters
    ----------
    pool_history_df : pd.DataFrame
        Must have columns: strategy, iteration, {value_col}. One row per compound
        in the labeled pool at each iteration.
    learning_curve_df : pd.DataFrame
        Must have columns: strategy, iteration, n_labeled. Used to map iteration to
        labeled-pool size for the x-axis.
    hit_threshold : float
        Minimum value_col value for a compound to be counted as a hit.
    strategy_order : list[str]
        Strategies to plot, controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to color. Falls back to matplotlib default if None.
    value_col : str, optional
        Column name for the activity values used to define hits.
    figsize : tuple, optional
        Figure size passed to matplotlib.

    Returns
    -------
    plt.Figure

    """
    # Count hits per (strategy, iteration) and merge with n_labeled for the x-axis
    hits_per_iter = (
        pool_history_df.groupby(["strategy", "iteration"])[value_col]
        .apply(lambda s: (s >= hit_threshold).sum())
        .reset_index(name="n_hits")
    )
    n_labeled = learning_curve_df[
        ["strategy", "iteration", "n_labeled"]
    ].drop_duplicates()
    hits_df = hits_per_iter.merge(n_labeled, on=["strategy", "iteration"], how="left")

    # Best-case reference: max hits found by any strategy at the final iteration
    last_iter = pool_history_df["iteration"].max()
    max_hits = int(
        pool_history_df[pool_history_df["iteration"] == last_iter]
        .groupby("strategy")[value_col]
        .apply(lambda s: (s >= hit_threshold).sum())
        .max()
    )

    fig, ax = plt.subplots(figsize=figsize)

    for strategy in strategy_order:
        df_sub = hits_df[hits_df["strategy"] == strategy].sort_values("n_labeled")
        if df_sub.empty:
            continue
        color = color_map.get(strategy, None) if color_map else None
        ax.plot(
            df_sub["n_labeled"],
            df_sub["n_hits"],
            label=strategy,
            color=color,
            linewidth=2,
        )

    # Reference line showing the best any strategy achieved by the final iteration
    ax.axhline(
        max_hits,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Max hits found ({max_hits})",
    )

    ax.set_xlabel("Number of Labeled Molecules")
    ax.set_ylabel(f"Hits Found ({value_col} ≥ {hit_threshold})")
    ax.legend(title="Strategy")
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    return fig


def plot_calibration_curve_before_after(
    expected_before: np.ndarray,
    observed_before: np.ndarray,
    expected_after: np.ndarray,
    observed_after: np.ndarray,
    label_before="Before calibration",
    label_after="After calibration",
    figsize=(6, 6),
) -> plt.Figure:
    """
    Plot diagonal (perfect calibration) + two calibration curves.
    Fill between curve and diagonal to show miscalibration area. Legend with label strings.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    ax.plot(
        expected_before, observed_before, label=label_before, color="#d62728", marker=""
    )
    ax.plot(
        expected_after, observed_after, label=label_after, color="#2ca02c", marker=""
    )

    ax.fill_between(
        expected_before, expected_before, observed_before, color="#d62728", alpha=0.1
    )
    ax.fill_between(
        expected_after, observed_after, expected_after, alpha=0.1, color="#2ca02c"
    )

    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Proportion")
    ax.set_title("Uncertainty Calibration")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    return fig


def plot_gtm_selection_animation(
    gtm_coords: np.ndarray,
    selection_history: list[dict],
    title: str = "Active Learning Selection (GTM)",
) -> go.Figure:
    """
    Animated Plotly scatter of compound selections on a pre-computed GTM embedding.

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

    Returns
    -------
    go.Figure
        Plotly figure with animation frames, a play/pause button, and an
        iteration slider. Call ``fig.show()`` to render in a Jupyter Notebook.

    """
    n_pool = len(gtm_coords)
    all_idx = np.arange(n_pool)

    # Build per-frame data: accumulated history (blue) and current selections (red)
    frames = []
    slider_steps = []

    for i, state in enumerate(selection_history):
        current_idx = np.array(state["selected_pool_indices"], dtype=int)

        # Accumulate all selections from prior iterations as historical context
        prior_idx = np.array(
            [idx for s in selection_history[:i] for idx in s["selected_pool_indices"]],
            dtype=int,
        )

        history_x = gtm_coords[prior_idx, 0] if len(prior_idx) > 0 else np.array([])
        history_y = gtm_coords[prior_idx, 1] if len(prior_idx) > 0 else np.array([])

        frame = go.Frame(
            data=[
                # Trace index 1: accumulated prior selections (blue)
                go.Scatter(
                    x=history_x,
                    y=history_y,
                    mode="markers",
                    marker=dict(color="royalblue", size=7, opacity=0.75),
                    name="Prior selections",
                ),
                # Trace index 2: current iteration selections (red)
                go.Scatter(
                    x=gtm_coords[current_idx, 0],
                    y=gtm_coords[current_idx, 1],
                    mode="markers",
                    marker=dict(color="crimson", size=9, opacity=0.9),
                    name=f"Iteration {state['iteration']}",
                ),
            ],
            # Only update the history and current traces; pool stays fixed
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
                label=f"Iter {state['iteration']}",
            )
        )

    # Base figure: pool trace is always visible and never changes
    fig = go.Figure(
        data=[
            # Trace 0: full compound pool background (gray, constant)
            go.Scatter(
                x=gtm_coords[all_idx, 0],
                y=gtm_coords[all_idx, 1],
                mode="markers",
                marker=dict(color="lightgray", size=5, opacity=0.5),
                name="All compounds",
            ),
            # Trace 1: prior selections placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="royalblue", size=7, opacity=0.75),
                name="Prior selections",
            ),
            # Trace 2: current selection placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="crimson", size=9, opacity=0.9),
                name="Current selection",
            ),
        ],
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis=dict(title="GTM dimension 1", showgrid=False, zeroline=False),
            yaxis=dict(title="GTM dimension 2", showgrid=False, zeroline=False),
            legend=dict(itemsizing="constant"),
            hovermode="closest",
            # Play/pause controls
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.05,
                    x=0.0,
                    xanchor="left",
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
                        prefix="Active Learning — ", visible=True, xanchor="center"
                    ),
                    pad=dict(t=50),
                    steps=slider_steps,
                )
            ],
        ),
    )

    return fig


def plot_strategy_label_efficiency(
    learning_curve_df: pd.DataFrame,
    target_mae: float,
    strategy_order: list[str],
    color_map: dict | None = None,
    figsize=(8, 4),
) -> plt.Figure:
    """
    Bar chart: number of labeled molecules required to first reach target_mae.
    DataFrame must have columns: strategy, n_labeled, mae_mean.
    If strategy never reaches target, show a 'Did not reach' annotation.
    Annotate bars with exact values.
    """
    fig, ax = plt.subplots(figsize=figsize)

    efficiency = {}

    max_labeled = learning_curve_df["n_labeled"].max()
    sentinel_height = max_labeled * 1.05

    for strategy in strategy_order:
        df_sub = learning_curve_df[
            learning_curve_df["strategy"] == strategy
        ].sort_values("n_labeled")
        # Find first row where mae_mean <= target_mae
        reached = df_sub[df_sub["mae_mean"] <= target_mae]
        if not reached.empty:
            efficiency[strategy] = reached.iloc[0]["n_labeled"]
        else:
            efficiency[strategy] = np.nan

    strategies = list(efficiency.keys())
    values = list(efficiency.values())

    bars = []
    for i, strategy in enumerate(strategies):
        val = values[i]
        color = color_map.get(strategy, "gray") if color_map else "steelblue"

        if np.isnan(val):
            # Did not reach
            bar = ax.bar(strategy, sentinel_height, color=color, alpha=0.4, hatch="///")
            bars.append(bar[0])

            # Annotate
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                sentinel_height / 2,
                "Did not reach",
                ha="center",
                va="center",
                color="black",
                rotation=90,
                fontweight="bold",
            )
        else:
            # Reached
            bar = ax.bar(strategy, val, color=color)
            bars.append(bar[0])

            # Annotate
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                val,
                f"{int(val)}",
                ha="center",
                va="bottom",
            )

    ax.set_ylabel("Labeled Samples to Reach Target")
    sns.despine()
    return fig
