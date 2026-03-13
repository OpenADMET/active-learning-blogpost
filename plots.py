import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw


def plot_learning_curve_with_bands(
    learning_curve_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    strategy_order: list[str],
    color_map: dict | None = None,
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """
    Interactive line plot with shaded ±1σ band per strategy.

    DataFrame must have columns: strategy, n_labeled, {metric_col}_mean,
    {metric_col}_lower, {metric_col}_upper.

    Parameters
    ----------
    learning_curve_df : pd.DataFrame
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
                line=dict(width=2, color=color),
            )
        )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=ylabel,
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        plot_bgcolor="white",
        width=width,
        height=height,
    )
    return fig


def plot_hit_discovery_curve(
    pool_history_df: pd.DataFrame,
    learning_curve_df: pd.DataFrame,
    hit_threshold: float,
    strategy_order: list[str],
    color_map: dict | None = None,
    value_col: str = "pEC50",
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """
    Interactive line plot of cumulative hits found vs. number of labeled molecules,
    with one line per strategy and a dashed reference line at the max hits found.

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
        Mapping of strategy name to CSS color string.
    value_col : str, optional
        Column name for the activity values used to define hits.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
    """
    hits_per_iter = (
        pool_history_df.groupby(["strategy", "iteration"])[value_col]
        .apply(lambda s: (s >= hit_threshold).sum())
        .reset_index(name="n_hits")
    )
    n_labeled = learning_curve_df[
        ["strategy", "iteration", "n_labeled"]
    ].drop_duplicates()
    hits_df = hits_per_iter.merge(n_labeled, on=["strategy", "iteration"], how="left")

    last_iter = pool_history_df["iteration"].max()
    max_hits = int(
        pool_history_df[pool_history_df["iteration"] == last_iter]
        .groupby("strategy")[value_col]
        .apply(lambda s: (s >= hit_threshold).sum())
        .max()
    )

    fig = go.Figure()

    for strategy in strategy_order:
        df_sub = hits_df[hits_df["strategy"] == strategy].sort_values("n_labeled")
        if df_sub.empty:
            continue
        color = color_map.get(strategy, None) if color_map else None
        fig.add_trace(
            go.Scatter(
                x=df_sub["n_labeled"],
                y=df_sub["n_hits"],
                mode="lines",
                name=strategy,
                line=dict(width=2, color=color),
            )
        )

    fig.add_hline(
        y=max_hits,
        line=dict(color="black", width=1.2, dash="dash"),
        annotation_text=f"Max hits found ({max_hits})",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=f"Hits Found ({value_col} ≥ {hit_threshold})",
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1),
        plot_bgcolor="white",
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
    width: int = 500,
    height: int = 500,
) -> go.Figure:
    """
    Interactive plot of diagonal (perfect calibration) and two calibration curves.
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

    # --- Before calibration ---
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
            line=dict(color="#d62728", width=2),
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.1)",
        )
    )

    # --- After calibration ---
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
            line=dict(color="#2ca02c", width=2),
            fill="tonexty",
            fillcolor="rgba(44,160,44,0.1)",
        )
    )

    fig.update_layout(
        title="Uncertainty Calibration",
        xaxis=dict(title="Expected Confidence Level", range=[0, 1], showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="Observed Proportion", range=[0, 1], showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        plot_bgcolor="white",
        width=width,
        height=height,
    )
    return fig


def plot_gtm_selection_animation(
    gtm_coords: np.ndarray,
    selection_history: list[dict],
    smiles_list: list[str],
    title: str = "Active Learning Selection (GTM)",
    img_size: tuple[int, int] = (200, 150),
) -> go.Figure:
    """
    Animated Plotly scatter of compound selections on a pre-computed GTM embedding.

    Renders one frame per active learning iteration with three layers:
    gray background (full pool), blue accumulation (all prior selections),
    and red highlight (current iteration's selections). Molecule structures
    appear in hover tooltips as embedded PNG images. Includes a slider
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
    smiles_list : list[str]
        SMILES strings in the same row order as ``gtm_coords``. Used to
        generate molecule-structure hover images.
    title : str, optional
        Title displayed above the plot.
    img_size : tuple[int, int], optional
        Width and height in pixels for each molecule thumbnail in the tooltip.

    Returns
    -------
    go.Figure
        Plotly figure with animation frames, a play/pause button, and an
        iteration slider. Call ``fig.show()`` to render in a Jupyter Notebook.

    """
    n_pool = len(gtm_coords)
    all_idx = np.arange(n_pool)
    hover_imgs = _smiles_to_hover_images(smiles_list, img_size=img_size)

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

        prior_hover = (
            [f'<b>SMILES:</b> {smiles_list[j]}<br><img src="{hover_imgs[j]}">'
             for j in prior_idx]
            if len(prior_idx) > 0 else []
        )
        current_hover = [
            f'<b>SMILES:</b> {smiles_list[j]}<br><img src="{hover_imgs[j]}">'
            for j in current_idx
        ]

        frame = go.Frame(
            data=[
                # Trace index 1: accumulated prior selections (blue)
                go.Scatter(
                    x=history_x,
                    y=history_y,
                    mode="markers",
                    marker=dict(color="royalblue", size=7, opacity=0.75),
                    name="Prior selections",
                    text=prior_hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
                # Trace index 2: current iteration selections (red)
                go.Scatter(
                    x=gtm_coords[current_idx, 0],
                    y=gtm_coords[current_idx, 1],
                    mode="markers",
                    marker=dict(color="crimson", size=9, opacity=0.9),
                    name=f"Iteration {state['iteration']}",
                    text=current_hover,
                    hovertemplate="%{text}<extra></extra>",
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

    pool_hover = [
        f'<b>SMILES:</b> {smi}<br><img src="{img}">'
        for smi, img in zip(smiles_list, hover_imgs)
    ]

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
                text=pool_hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            # Trace 1: prior selections placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="royalblue", size=7, opacity=0.75),
                name="Prior selections",
                text=[],
                hovertemplate="%{text}<extra></extra>",
            ),
            # Trace 2: current selection placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="crimson", size=9, opacity=0.9),
                name="Current selection",
                text=[],
                hovertemplate="%{text}<extra></extra>",
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
    width: int = 700,
    height: int = 400,
) -> go.Figure:
    """
    Interactive bar chart: number of labeled molecules required to first reach target_mae.

    DataFrame must have columns: strategy, n_labeled, mae_mean.
    Strategies that never reach the target are shown as hatched bars with a
    "Did not reach" annotation.

    Parameters
    ----------
    learning_curve_df : pd.DataFrame
    target_mae : float
    strategy_order : list[str]
    color_map : dict or None, optional
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
    """
    max_labeled = learning_curve_df["n_labeled"].max()
    sentinel_height = max_labeled * 1.05

    bar_x, bar_y, bar_colors, bar_patterns, hover_texts = [], [], [], [], []
    did_not_reach = []

    for strategy in strategy_order:
        df_sub = learning_curve_df[
            learning_curve_df["strategy"] == strategy
        ].sort_values("n_labeled")
        reached = df_sub[df_sub["mae_mean"] <= target_mae]
        color = color_map.get(strategy, "steelblue") if color_map else "steelblue"

        bar_x.append(strategy)
        bar_colors.append(color)

        if not reached.empty:
            val = int(reached.iloc[0]["n_labeled"])
            bar_y.append(val)
            bar_patterns.append("")
            hover_texts.append(f"{val} labeled samples")
        else:
            bar_y.append(sentinel_height)
            bar_patterns.append("/")
            hover_texts.append("Did not reach target")
            did_not_reach.append(strategy)

    fig = go.Figure(
        go.Bar(
            x=bar_x,
            y=bar_y,
            marker=dict(
                color=bar_colors,
                opacity=[0.4 if s in did_not_reach else 1.0 for s in bar_x],
                pattern=dict(shape=bar_patterns, fgcolor="rgba(0,0,0,0.4)"),
            ),
            text=[
                str(v) if s not in did_not_reach else ""
                for s, v in zip(bar_x, bar_y)
            ],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="x+text",
        )
    )

    for strategy in did_not_reach:
        idx = bar_x.index(strategy)
        fig.add_annotation(
            x=strategy,
            y=bar_y[idx] / 2,
            text="Did not reach",
            showarrow=False,
            font=dict(color="black", size=11),
            textangle=-90,
        )

    fig.update_layout(
        yaxis_title="Labeled Samples to Reach Target",
        yaxis=dict(range=[0, sentinel_height * 1.15], showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        showlegend=False,
        width=width,
        height=height,
    )
    return fig


def plot_tmap_selection_animation(
    tmap_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smiles_list: list[str],
    selection_history: list[dict],
    title: str = "Active Learning Selection (TMAP)",
    img_size: tuple[int, int] = (200, 150),
) -> go.Figure:
    """
    Animated Plotly scatter of compound selections on a pre-computed TMAP layout.

    Renders one frame per active learning iteration with four layers:
    the TMAP minimum-spanning-tree edges (gray, permanent background), all
    pool nodes (light gray, permanent), accumulated prior selections (blue),
    and the current-iteration selections (red). Molecule structures appear in
    hover tooltips as embedded PNG images. Includes a slider and play/pause
    controls for use directly in a Jupyter Notebook.

    Parameters
    ----------
    tmap_layout : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t)`` as returned by ``helpers.smiles_to_tmap``:
        node x-coordinates, node y-coordinates, MST edge source indices,
        and MST edge target indices.
    smiles_list : list[str]
        SMILES strings in the same row order as the ``tmap_layout`` arrays.
        Used to generate molecule-structure hover images.
    selection_history : list[dict]
        List of per-iteration state dicts produced by ``run_active_learning``.
        Each dict must contain ``"selected_pool_indices"`` (list of integer
        indices into the pool) and ``"iteration"`` (int).
    title : str, optional
        Title displayed above the plot.
    img_size : tuple[int, int], optional
        Width and height in pixels for each molecule thumbnail in the tooltip.

    Returns
    -------
    go.Figure
        Plotly figure with animation frames, a play/pause button, and an
        iteration slider. Call ``fig.show()`` to render in a Jupyter Notebook.
    """
    x, y, s, t = tmap_layout
    hover_imgs = _smiles_to_hover_images(smiles_list, img_size=img_size)

    # Build a single edge trace with None separators (draws the MST as lines)
    edge_x, edge_y = [], []
    for src, dst in zip(s, t):
        edge_x.extend([x[src], x[dst], None])
        edge_y.extend([y[src], y[dst], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.4)", width=0.5),
        hoverinfo="skip",
        showlegend=False,
    )

    pool_hover = [
        f'<b>SMILES:</b> {smi}<br><img src="{img}">'
        for smi, img in zip(smiles_list, hover_imgs)
    ]

    pool_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color="lightgray", size=5, opacity=0.5),
        name="All compounds",
        text=pool_hover,
        hovertemplate="%{text}<extra></extra>",
    )

    frames = []
    slider_steps = []

    for i, state in enumerate(selection_history):
        current_idx = np.array(state["selected_pool_indices"], dtype=int)
        prior_idx = np.array(
            [idx for s_ in selection_history[:i] for idx in s_["selected_pool_indices"]],
            dtype=int,
        )

        prior_hover = (
            [f'<b>SMILES:</b> {smiles_list[j]}<br><img src="{hover_imgs[j]}">'
             for j in prior_idx]
            if len(prior_idx) > 0 else []
        )
        current_hover = [
            f'<b>SMILES:</b> {smiles_list[j]}<br><img src="{hover_imgs[j]}">'
            for j in current_idx
        ]

        frame = go.Frame(
            data=[
                # Trace index 2: accumulated prior selections (blue)
                go.Scatter(
                    x=x[prior_idx] if len(prior_idx) > 0 else np.array([]),
                    y=y[prior_idx] if len(prior_idx) > 0 else np.array([]),
                    mode="markers",
                    marker=dict(color="royalblue", size=7, opacity=0.75),
                    name="Prior selections",
                    text=prior_hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
                # Trace index 3: current iteration selections (red)
                go.Scatter(
                    x=x[current_idx],
                    y=y[current_idx],
                    mode="markers",
                    marker=dict(color="crimson", size=9, opacity=0.9),
                    name=f"Iteration {state['iteration']}",
                    text=current_hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
            ],
            traces=[2, 3],
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

    fig = go.Figure(
        data=[
            # Trace 0: MST edges (permanent)
            edge_trace,
            # Trace 1: full compound pool background (gray, constant)
            pool_trace,
            # Trace 2: prior selections placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="royalblue", size=7, opacity=0.75),
                name="Prior selections",
                hovertemplate="%{text}<extra></extra>",
                text=[],
            ),
            # Trace 3: current selection placeholder (overwritten by each frame)
            go.Scatter(
                x=np.array([]),
                y=np.array([]),
                mode="markers",
                marker=dict(color="crimson", size=9, opacity=0.9),
                name="Current selection",
                hovertemplate="%{text}<extra></extra>",
                text=[],
            ),
        ],
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis=dict(title="TMAP dimension 1", showgrid=False, zeroline=False),
            yaxis=dict(title="TMAP dimension 2", showgrid=False, zeroline=False),
            legend=dict(itemsizing="constant"),
            hovermode="closest",
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


def _smiles_to_hover_images(
    smiles_list: list[str],
    img_size: tuple[int, int] = (200, 150),
) -> list[str]:
    """Render each SMILES as a base64-encoded PNG data URI for hover tooltips.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to render.
    img_size : tuple[int, int]
        ``(width, height)`` in pixels for each rendered molecule image.

    Returns
    -------
    list[str]
        List of ``"data:image/png;base64,..."`` URI strings, one per input
        SMILES. Invalid SMILES produce a 1×1 transparent PNG placeholder.
    """
    w, h = img_size
    uris = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        buf = io.BytesIO()
        if mol is not None:
            img = Draw.MolToImage(mol, size=(w, h))
        else:
            from PIL import Image
            img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a CSS hex color string to an rgba() string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
