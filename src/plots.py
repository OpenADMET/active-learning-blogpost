"""Plotly and Faerun visualization functions for the active learning pipeline.

Each function returns a configured figure object ready to display or export.
Figures share a consistent visual style: white background, black axes, and
per-strategy colors from ``src.helpers.STRATEGY_COLORS``.
"""

import uuid
from pathlib import Path
from typing import Any

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from faerun import Faerun
from plotly.subplots import make_subplots

# Animation timing for plot_predicted_distribution_animation.
# Used by write_html(animation_opts=...) so scrub and playback share the same speed.
# redraw=True is required so that trace names (legend "n=X") and layout title
# ("X labeled") are re-applied on every frame during playback, not just on scrub.
# transition_ms=0: KDE curves don't interpolate meaningfully — any non-zero
# transition produces nonsensical intermediate density shapes.
_DIST_FRAME_MS: int = 1000
_DIST_TRANSITION_MS: int = 0
DIST_ANIMATION_OPTS: dict = dict(
    frame=dict(duration=_DIST_FRAME_MS, redraw=True),
    transition=dict(duration=_DIST_TRANSITION_MS),
)

# GTM animation timing (matches the inline values used in analysis.py).
_GTM_FRAME_MS: int = 1200
_GTM_TRANSITION_MS: int = 400

# JS injected into animated HTML exports via post_script to provide a single
# play/pause toggle button above the chart.  Placeholders are substituted by
# make_play_pause_script().  No auto-play: the animation only starts on click.
_PLAY_PAUSE_SCRIPT_TEMPLATE = r"""
(function() {
  var gd = document.getElementById('__DIV_ID__');
  if (!gd) return;
  var playing = false;
  var playToken = 0;
  var toolbar = document.createElement('div');
  toolbar.style.cssText = 'margin:4px 0;padding-left:4px;';
  var btn = document.createElement('button');
  btn.innerHTML = '\u25B6 Play';
  btn.style.cssText = 'padding:5px 14px;font-size:13px;cursor:pointer;'
    + 'background:#555;color:#fff;border:1px solid #888;border-radius:4px;';
  toolbar.appendChild(btn);
  gd.insertAdjacentElement('beforebegin', toolbar);

  function setPlaying(isPlaying) {
    playing = isPlaying;
    btn.innerHTML = isPlaying ? '\u23F8 Pause' : '\u25B6 Play';
  }

  function getFrameNames() {
    var frames = (gd._transitionData && gd._transitionData._frames) || [];
    return frames.map(function(frame) { return frame.name; }).filter(Boolean);
  }

  function getStartIndex(frameNames) {
    var sliders = gd._fullLayout && gd._fullLayout.sliders;
    var active = sliders && sliders.length ? sliders[0].active : 0;
    if (typeof active !== 'number' || !isFinite(active) || active < 0) {
      return 0;
    }
    if (active >= frameNames.length - 1) {
      return 0;
    }
    return active;
  }

  function stopAnimation() {
    playToken += 1;
    setPlaying(false);
    return Plotly.animate(gd, [null], {
      mode: 'immediate',
      frame: {duration: 0, redraw: false},
      transition: {duration: 0, ordering: 'layout first'}
    });
  }

  function playAnimation() {
    var frameNames = getFrameNames();
    if (!frameNames.length) return;

    var startIndex = getStartIndex(frameNames);
    var token = playToken + 1;
    playToken = token;
    setPlaying(true);

    function playFrame(index) {
      if (token !== playToken) return;
      if (index >= frameNames.length) {
        setPlaying(false);
        return;
      }

      Plotly.animate(gd, [frameNames[index]], {
        mode: 'immediate',
        frame: {duration: 0, redraw: __REDRAW__},
        transition: {
          duration: __TRANSITION_MS__,
          easing: '__EASING__',
          ordering: 'layout first'
        }
      }).then(function() {
        if (token !== playToken) return;
        Plotly.relayout(gd, {'sliders[0].active': index});
        setTimeout(function() { playFrame(index + 1); }, __FRAME_MS__);
      }).catch(function() {});
    }

    playFrame(startIndex);
  }

  btn.addEventListener('click', function() {
    if (!playing) {
      playAnimation();
    } else {
      stopAnimation();
    }
  });
})();
"""


def make_play_pause_script(frame_ms: int, transition_ms: int, easing: str, redraw: bool = False) -> str:
    """Return a JS play/pause toggle script with the given animation parameters.

    The returned string is suitable for passing as ``post_script`` to
    ``write_html_both`` (or directly to ``fig.write_html``).  It inserts a
    single ▶/⏸ toggle button above the chart via DOM injection; no Plotly
    ``updatemenus`` are needed.  The animation does **not** auto-play —
    playback starts only when the user clicks the button.

    Parameters
    ----------
    frame_ms : int
        Duration of each frame in milliseconds during playback.
    transition_ms : int
        Transition duration between frames in milliseconds.
    easing : str
        CSS/Plotly easing name (e.g. ``"linear"``, ``"circle"``).
    redraw : bool
        Whether to force a full Plotly redraw on each frame.  Must be
        ``True`` when frames include layout changes (e.g. title ``text``) or
        trace ``name`` updates (legend labels), otherwise those properties
        only update on manual scrub.  Default is ``False``.

    Returns
    -------
    str
        JavaScript snippet ready for ``post_script``.
    """
    return (
        _PLAY_PAUSE_SCRIPT_TEMPLATE
        .replace("__FRAME_MS__", str(frame_ms))
        .replace("__TRANSITION_MS__", str(transition_ms))
        .replace("__EASING__", easing)
        .replace("__REDRAW__", "true" if redraw else "false")
    )


# Pre-built convenience scripts for the two animation types in this repo.
DIST_PLAY_PAUSE_SCRIPT: str = make_play_pause_script(_DIST_FRAME_MS, _DIST_TRANSITION_MS, "linear", redraw=True)
GTM_PLAY_PAUSE_SCRIPT: str = make_play_pause_script(_GTM_FRAME_MS, _GTM_TRANSITION_MS, "circle", redraw=False)


def write_html_both(
    fig: go.Figure,
    path: str | Path,
    *,
    animation_opts: dict | None = None,
    auto_play: bool = False,
    post_script: str | None = None,
) -> None:
    """Write a figure to both an embedded HTML file and a CDN-linked HTML file.

    Two files are always written:

    1. **Embedded** — ``path`` with ``include_plotlyjs=True`` (~3 MB larger, works
       offline).
    2. **CDN** — ``path.parent / "cdn" / path.name`` with
       ``include_plotlyjs="cdn"`` (tiny, requires internet).

    The ``cdn/`` subdirectory is created automatically if it does not exist.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to export.
    path : str or Path
        Destination path for the self-contained (embedded) HTML file.
    animation_opts : dict, optional
        Passed directly to ``fig.write_html`` for both versions.
    auto_play : bool
        Whether to auto-play animations on page load.  Default is ``False``.
    post_script : str, optional
        JavaScript snippet injected after Plotly initialises the figure.
        Pass ``DIST_PLAY_PAUSE_SCRIPT`` or ``GTM_PLAY_PAUSE_SCRIPT`` to inject
        the DOM play/pause toggle button for animated figures.
    """
    path = Path(path)
    div_id = str(uuid.uuid4())

    # CSS injected via post_script: center the figure div and clamp it to the
    # container width so it never overflows a narrow CMS column (e.g. Ghost).
    centering_css = (
        "(function(){"
        "var s=document.createElement('style');"
        "s.textContent='#" + div_id + "{max-width:100%!important;"
        "margin-left:auto!important;margin-right:auto!important;"
        "display:block!important;}';"
        "document.head.appendChild(s);"
        "})();"
    )

    user_script = post_script.replace("__DIV_ID__", div_id) if post_script is not None else None
    combined_script = "\n".join(filter(None, [centering_css, user_script]))

    kwargs: dict = dict(
        auto_play=auto_play,
        div_id=div_id,
        post_script=combined_script,
        config={"responsive": True},
    )
    if animation_opts is not None:
        kwargs["animation_opts"] = animation_opts

    fig.write_html(str(path), include_plotlyjs=True, **kwargs)

    cdn_path = path.parent / "cdn" / path.name
    cdn_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(cdn_path), include_plotlyjs="cdn", **kwargs)


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

    # Autodetect y-axis bounds from the plotted strategies' data.
    # Lower bound: 0 when all values are non-negative; 5% below the minimum otherwise.
    # Upper bound: 5% above the maximum of the upper band.
    relevant = learning_curve_df[learning_curve_df["strategy"].isin(strategy_order)]
    y_min = relevant[f"{metric_col}_lower"].min() if not relevant.empty else 0.0
    y_max = relevant[f"{metric_col}_upper"].max() if not relevant.empty else 1.0
    y_lower = 0.0 if y_min >= 0 else y_min * 1.05
    y_upper = y_max * 1.05

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=ylabel,
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(range=[y_lower, y_upper], showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        plot_bgcolor="white",
        margin=dict(t=40),
        width=width,
        height=height,
    )
    return fig


def plot_hit_discovery_curve(
    pool_history_df: pd.DataFrame,
    learning_curve_df: pd.DataFrame,
    hit_threshold: float,
    max_hits: int,
    pool_size: int,
    strategy_order: list[str],
    color_map: dict | None = None,
    value_col: str = "pEC50",
    width: int = 700,
    height: int = 450,
) -> go.Figure:
    """Interactive line plot of cumulative hits found vs. number of labeled molecules.

    Shows mean ± 1σ bands per strategy, a diagonal dashed gray reference line for
    the expected hits under random (uniform) acquisition, and a dashed reference line
    at the absolute hit ceiling.

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
    pool_size : int
        Total number of compounds in the pool. Defines the x-extent of the diagonal
        random-acquisition baseline (a linear ramp from (0, 0) to (pool_size, max_hits)).
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
        a diagonal random-acquisition baseline and a dashed reference line at the
        total hit ceiling.

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

    # Random baseline: linear ramp from (0, 0) to (pool_size, max_hits)
    fig.add_trace(
        go.Scatter(
            x=[0, pool_size],
            y=[0, max_hits],
            mode="lines",
            name="Random (expected)",
            line=dict(color="#888888", width=1.2, dash="dash"),
            hoverinfo="skip",
        )
    )

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

    # Autodetect upper y bound: 5% above the max of the upper bands and the hit ceiling.
    y_upper = max(
        hits_summary["n_hits_upper"].max() if not hits_summary.empty else 0,
        max_hits,
    ) * 1.10

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=f"Hits Found ({value_col} ≥ {hit_threshold})",
        legend_title="Strategy",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        yaxis=dict(range=[0, y_upper], showgrid=True, gridcolor="rgba(0,0,0,0.1)", gridwidth=1, showline=True, linecolor="black", linewidth=1, mirror=True),
        plot_bgcolor="white",
        margin=dict(t=40),
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
        margin=dict(t=40, r=20),
        width=width,
        height=height,
    )
    return fig


def plot_gtm_selection_animation(
    gtm_coords: np.ndarray,
    selection_history: list[dict],
    title: str = "Active Learning Selection (GTM)",
    background_gtm_coords: np.ndarray | None = None,
    pool_activity: np.ndarray | None = None,
    hit_threshold: float = 7.0,
) -> go.Figure:
    """Animated Plotly scatter of compound selections on a pre-computed GTM embedding.

    All pool compounds are present in two permanent data traces from the first frame
    (inactive pool and active pool). Frames animate only ``marker.size`` and
    ``marker.color`` per-point arrays — no points are added or removed across frames.
    This produces a smooth grow-in-place effect as compounds are selected (Plotly
    interpolates numeric ``marker.size`` arrays over the transition duration).

    Actives are always rendered on top (higher trace index). The legend shows three
    static entries: "Unqueried" (light gray), "Inactive" (dark gray), "Active"
    (crimson). A final "End" frame shows the campaign end-state with all selections
    at resting size and no current-iteration highlight.

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
        molecules not part of the active learning pool. When provided, these are
        shown as a static light-gray layer beneath the animated pool traces.
    pool_activity : np.ndarray or None, optional
        Activity values for all pool compounds, indexed by pool position.
        When provided, compounds with activity >= ``hit_threshold`` are coloured
        crimson; others are dark gray. When None, all selected compounds are
        shown in dark gray.
    hit_threshold : float, optional
        Activity threshold above which a compound is considered a hit and
        coloured crimson. Default 7.0.

    Returns
    -------
    go.Figure
        Animated Plotly figure with per-iteration frames, a play/pause button, and
        an iteration slider. Call ``fig.show()`` to display inline in a Jupyter Notebook.

    """
    _COLOR_POOL_BG  = "rgb(200, 200, 200)"   # static background: all pool compounds
    _COLOR_INACTIVE = "rgb(75, 75, 75)"       # queried inactive: dark gray, fully opaque
    _COLOR_ACTIVE   = "rgb(220, 20, 60)"      # queried active: crimson, fully opaque
    _SIZE_INVISIBLE = 0    # unqueried on data traces (shown by bg trace instead)
    _SIZE_SMALL     = 6    # prior selections
    _SIZE_LARGE     = 12   # current-iteration selections

    gtm_coords = np.asarray(gtm_coords)
    if background_gtm_coords is not None:
        background_gtm_coords = np.asarray(background_gtm_coords)

    n_pool = len(gtm_coords)

    # Split pool into inactive and active sub-groups using ground-truth activity.
    # Active compounds always live in the higher-index trace so they render on top.
    if pool_activity is not None:
        active_pool_mask = np.asarray(pool_activity) >= hit_threshold
    else:
        active_pool_mask = np.zeros(n_pool, dtype=bool)

    inactive_pool_idx = np.where(~active_pool_mask)[0]
    active_pool_idx   = np.where(active_pool_mask)[0]

    # Trace layout (fixed): [bg] + [3 legend-only dummies] + [inactive data] + [active data]
    _animated = [4, 5]

    def _make_marker_arrays(
        trace_pool_idx: np.ndarray,
        queried_set: set,
        current_set: set,
        color: str,
    ) -> dict:
        """Return marker dict with per-point size arrays for one data trace.

        Unqueried points use size=0 (invisible); the static background trace
        provides their light-gray appearance.  All colors are fully opaque.
        """
        sizes: list[int] = []
        for pi in trace_pool_idx:
            if pi in current_set:
                sizes.append(_SIZE_LARGE)
            elif pi in queried_set:
                sizes.append(_SIZE_SMALL)
            else:
                sizes.append(_SIZE_INVISIBLE)
        return dict(color=color, size=sizes, line=dict(color="white", width=0.5))

    def _make_frame(prior_set: set, current_set: set, name: str) -> go.Frame:
        queried_set = prior_set | current_set
        return go.Frame(
            data=[
                go.Scatter(
                    mode="markers",
                    marker=_make_marker_arrays(
                        inactive_pool_idx, queried_set, current_set, _COLOR_INACTIVE
                    ),
                    hoverinfo="skip",
                ),
                go.Scatter(
                    mode="markers",
                    marker=_make_marker_arrays(
                        active_pool_idx, queried_set, current_set, _COLOR_ACTIVE
                    ),
                    hoverinfo="skip",
                ),
            ],
            traces=_animated,
            name=name,
        )

    frames = []
    slider_steps = []
    prior_set: set = set()

    for i, state in enumerate(selection_history):
        current_set = set(state["selected_pool_indices"])
        frames.append(_make_frame(prior_set, current_set, str(i)))
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        frame=dict(duration=1200, redraw=True),
                        mode="immediate",
                        transition=dict(duration=400),
                    ),
                ],
                label=str(state["iteration"]),
            )
        )
        prior_set = prior_set | current_set

    # End-state frame: all selections as visited, no current-iteration highlight
    frames.append(_make_frame(prior_set, set(), "end"))
    slider_steps.append(
        dict(
            method="animate",
            args=[
                ["end"],
                dict(
                    frame=dict(duration=1200, redraw=True),
                    mode="immediate",
                    transition=dict(duration=400),
                ),
            ],
            label="End",
        )
    )

    def _initial_markers(color: str, n: int) -> dict:
        return dict(
            color=color,
            size=[_SIZE_INVISIBLE] * n,
            line=dict(color="white", width=0.5),
        )

    # Static background: all pool compounds + optional background molecules shown as
    # light gray.  These never animate; the data traces render on top when queried.
    bg_x = np.array(gtm_coords[:, 0])
    bg_y = np.array(gtm_coords[:, 1])
    if background_gtm_coords is not None:
        bg_x = np.concatenate([bg_x, background_gtm_coords[:, 0]])
        bg_y = np.concatenate([bg_y, background_gtm_coords[:, 1]])

    figure_data = [
        go.Scatter(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(color=_COLOR_POOL_BG, size=5),
            hoverinfo="skip",
            showlegend=False,
        )
    ]

    # Three legend-only dummy traces
    figure_data.extend([
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_COLOR_POOL_BG, size=8),
            name="Unqueried", hoverinfo="skip", showlegend=True,
        ),
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_COLOR_INACTIVE, size=8, line=dict(color="white", width=0.5)),
            name="Inactive", hoverinfo="skip", showlegend=True,
        ),
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_COLOR_ACTIVE, size=8, line=dict(color="white", width=0.5)),
            name="Active", hoverinfo="skip", showlegend=True,
        ),
    ])

    # Permanent data traces: all pool compounds, initially size=0 (invisible).
    # The static bg trace provides their light-gray unqueried appearance.
    figure_data.append(
        go.Scatter(
            x=gtm_coords[inactive_pool_idx, 0],
            y=gtm_coords[inactive_pool_idx, 1],
            mode="markers",
            marker=_initial_markers(_COLOR_INACTIVE, len(inactive_pool_idx)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure_data.append(
        go.Scatter(
            x=gtm_coords[active_pool_idx, 0],
            y=gtm_coords[active_pool_idx, 1],
            mode="markers",
            marker=_initial_markers(_COLOR_ACTIVE, len(active_pool_idx)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig = go.Figure(
        data=figure_data,
        frames=frames,
        layout=go.Layout(
            title=title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="GTM dimension 1", showgrid=False, zeroline=False),
            yaxis=dict(title="GTM dimension 2", showgrid=False, zeroline=False),
            legend=dict(itemsizing="constant"),
            hovermode=False,
            updatemenus=[],
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


def _build_iteration_categories(
    selection_history: list[dict],
    n_points: int,
    colormap: str | mcolors.Colormap = "viridis",
) -> tuple[np.ndarray, mcolors.ListedColormap, list[tuple]]:
    """Build per-compound iteration category data for a single selection history.

    Encodes which AL iteration first selected each compound.  Index 0 is
    reserved for "Unselected" (gray); indices 1..N map to the sorted iteration
    numbers.

    Parameters
    ----------
    selection_history : list[dict]
        Per-iteration state dicts from ``run_active_learning``, each containing
        ``"iteration"`` and ``"selected_pool_indices"``.
    n_points : int
        Total number of points (pool + any background).  Must be ≥ the maximum
        pool index referenced in ``selection_history``.
    colormap : str or Colormap, optional
        Matplotlib colormap for the iteration colors.  Default ``"viridis"``.

    Returns
    -------
    c : np.ndarray
        Integer array of length ``n_points``.  0 = Unselected, k = iteration
        category k (1-based).
    listed_cmap : mcolors.ListedColormap
        ListedColormap with gray at index 0 and one viridis color per iteration.
    legend_labels : list of tuple
        ``[(category_int, label_str), ...]`` suitable for Faerun's
        ``legend_labels`` argument.
    """
    iterations = sorted({state["iteration"] for state in selection_history})
    iter_to_cat = {it: idx + 1 for idx, it in enumerate(iterations)}
    legend_labels = [(0, "Unselected")] + [
        (idx + 1, str(it)) for idx, it in enumerate(iterations)
    ]

    c = np.zeros(n_points, dtype=int)
    for state in selection_history:
        cat = iter_to_cat[state["iteration"]]
        for idx in state["selected_pool_indices"]:
            if c[idx] == 0:
                c[idx] = cat

    n_iter = len(iterations)
    base_cmap = mcm.get_cmap(colormap) if isinstance(colormap, str) else colormap
    iter_colors = [base_cmap(i / max(n_iter - 1, 1)) for i in range(n_iter)]
    listed_cmap = mcolors.ListedColormap(
        [(0.55, 0.55, 0.55, 1.0)] + iter_colors,
        N=1 + n_iter,
    )

    return c, listed_cmap, legend_labels


def plot_tmap_faerun(
    tmap_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smiles_list: list[str],
    selection_history: list[dict],
    output_name: str = "tmap_selection",
    output_path: str = "./",
    title: str = "Active Learning Selection (TMAP)",
    colormap: str = "viridis",
    point_scale: float = 3.0,
    n_background: int = 0,
    background_point_scale: float = 1.0,
    edge_similarity_threshold: float = 0.0,
    pool_activity: np.ndarray | None = None,
    hit_threshold: float = 7.0,
    active_scale_multiplier: float = 1.5,
) -> Faerun:
    """Faerun scatter plot of compound selections on a pre-computed TMAP layout.

    Each compound node is colored by the active learning iteration in which it
    was first selected, using faerun's native categorical colormapping. Compounds
    never selected are labeled ``"Unselected"``. SMILES strings are embedded as
    labels for tooltip display. The TMAP minimum-spanning-tree is drawn as a tree
    layer. Calling ``.plot()`` on the returned object renders the interactive HTML.

    Parameters
    ----------
    tmap_layout : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t, edge_sims)`` as returned by ``helpers.smiles_to_tmap``, computed on
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
    edge_similarity_threshold : float, optional
        Minimum Tanimoto/Jaccard similarity for an MST edge to be drawn.
        Edges below this threshold are omitted from the tree layer. Default 0.0
        (all edges drawn).
    pool_activity : np.ndarray or None, optional
        Activity values (length = number of pool compounds, i.e.
        ``len(smiles_list) - n_background``). When provided, compounds with
        activity >= ``hit_threshold`` are rendered at
        ``point_scale * active_scale_multiplier`` rather than ``point_scale``.
    hit_threshold : float, optional
        Activity threshold above which a compound is considered active.
        Default 7.0.
    active_scale_multiplier : float, optional
        Size multiplier applied to active compounds relative to ``point_scale``.
        Default 1.5 (actives appear 50% larger than inactives).

    Returns
    -------
    Faerun
        Configured faerun instance. Call ``f.plot(output_name, output_path)``
        to regenerate the HTML, or ``f.plot(output_name, output_path,
        notebook_height=500)`` to display inline in a Jupyter Notebook.

    """
    x, y, s, t, edge_sims = tmap_layout

    c, listed_cmap, legend_labels = _build_iteration_categories(
        selection_history, len(x), colormap
    )

    # Per-point sizes: foreground uses point_scale, background uses background_point_scale
    n_pool = len(x) - n_background
    s_vals = np.full(len(x), point_scale, dtype=float)
    if n_background > 0:
        s_vals[n_pool:] = background_point_scale
    if pool_activity is not None:
        active_mask = pool_activity[:n_pool] >= hit_threshold
        s_vals[:n_pool][active_mask] *= active_scale_multiplier

    if edge_similarity_threshold > 0.0:
        mask = edge_sims >= edge_similarity_threshold
        s, t = s[mask], t[mask]

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


def plot_tmap_faerun_strategies(
    tmap_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smiles_list: list[str],
    strategies_histories: dict[str, list[dict]],
    output_name: str = "tmap_selection",
    output_path: str = "./",
    title: str = "Active Learning Selection (TMAP)",
    colormap: str = "viridis",
    point_scale: float = 3.0,
    n_background: int = 0,
    background_point_scale: float = 1.0,
    edge_similarity_threshold: float = 0.0,
    pool_activity: np.ndarray | None = None,
    hit_threshold: float = 7.0,
    active_scale_multiplier: float = 1.5,
) -> Faerun:
    """Faerun scatter plot of compound selections for multiple strategies with dropdown.

    Each strategy is a separate Faerun series selectable via a dropdown in the
    viewer.  Within each series, nodes are colored by the AL iteration in which
    the compound was first selected; unselected compounds are gray.  The MST is
    drawn as a tree layer shared across all series.

    Parameters
    ----------
    tmap_layout : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t, edge_sims)`` as returned by ``helpers.smiles_to_tmap``.
    smiles_list : list[str]
        SMILES strings in the same row order as ``tmap_layout``.
    strategies_histories : dict[str, list[dict]]
        Mapping from strategy name to its per-iteration state dicts
        (e.g. ``all_runs[strategy][0]["history"]`` for seed 0).  Keys become
        the dropdown labels in the Faerun viewer.
    output_name : str, optional
        Base filename (without extension) for faerun's HTML output.
    output_path : str, optional
        Directory in which to write the output file.
    title : str, optional
        Plot title.
    colormap : str, optional
        Matplotlib colormap name for iteration colors. Default ``"viridis"``.
    point_scale : float, optional
        Relative size of pool (foreground) scatter points. Default 3.0.
    n_background : int, optional
        Number of background molecules appended at the end of ``tmap_layout``
        / ``smiles_list``. Default is 0.
    background_point_scale : float, optional
        Relative size of background scatter points. Default 1.0.
    edge_similarity_threshold : float, optional
        Minimum Tanimoto/Jaccard similarity for an MST edge to be drawn.
        Edges below this threshold are omitted. Default 0.0 (all edges drawn).
    pool_activity : np.ndarray or None, optional
        Activity values (length = number of pool compounds). When provided,
        compounds with activity >= ``hit_threshold`` are rendered at
        ``point_scale * active_scale_multiplier``. Activity is strategy-
        independent, so one size array is shared across all strategy series.
    hit_threshold : float, optional
        Activity threshold above which a compound is considered active.
        Default 7.0.
    active_scale_multiplier : float, optional
        Size multiplier for active compounds relative to ``point_scale``.
        Default 1.5.

    Returns
    -------
    Faerun
        Configured faerun instance.
    """
    x, y, s, t, edge_sims = tmap_layout
    n_pool = len(x) - n_background
    s_vals = np.full(len(x), point_scale, dtype=float)
    if n_background > 0:
        s_vals[n_pool:] = background_point_scale
    if pool_activity is not None:
        active_mask = pool_activity[:n_pool] >= hit_threshold
        s_vals[:n_pool][active_mask] *= active_scale_multiplier

    if edge_similarity_threshold > 0.0:
        mask = edge_sims >= edge_similarity_threshold
        s, t = s[mask], t[mask]

    all_c: list[np.ndarray] = []
    all_cmaps: list[mcolors.ListedColormap] = []
    all_legend_labels: list[list[tuple]] = []

    for history in strategies_histories.values():
        c, listed_cmap, legend_labels = _build_iteration_categories(
            history, len(x), colormap
        )
        all_c.append(c)
        all_cmaps.append(listed_cmap)
        all_legend_labels.append(legend_labels)

    n_strategies = len(strategies_histories)
    strategy_names = list(strategies_histories.keys())

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
            "c": all_c,
            "s": [s_vals] * n_strategies,
            "labels": smiles_list,
        },
        colormap=all_cmaps,
        categorical=[True] * n_strategies,
        has_legend=True,
        legend_title=["AL Iteration"] * n_strategies,
        legend_labels=all_legend_labels,
        series_title=strategy_names,
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


def plot_tmap_faerun_partition(
    tmap_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smiles_list: list[str],
    partition_labels: np.ndarray,
    output_name: str = "tmap_partition",
    output_path: str = "./",
    title: str = "Train / Test Partition (TMAP)",
    train_color: tuple = (0.122, 0.467, 0.706, 1.0),
    test_color: tuple = (0.839, 0.153, 0.157, 1.0),
    train_singleton_color: tuple = (0.682, 0.780, 0.910, 1.0),
    test_singleton_color: tuple = (0.984, 0.604, 0.600, 1.0),
    edge_similarity_threshold: float = 0.0,
    point_scale: float = 3.0,
) -> Faerun:
    """Faerun scatter plot of pool and test compounds on a pre-computed TMAP layout.

    Each node is colored by its data partition.  Two or four categories are
    supported:

    * ``0`` — Train
    * ``1`` — Test
    * ``2`` — Train, singleton Bemis-Murcko scaffold (light blue)
    * ``3`` — Test, singleton Bemis-Murcko scaffold (light red)

    SMILES strings are embedded as labels for tooltip display. The TMAP
    minimum-spanning-tree is drawn as a tree layer.

    Parameters
    ----------
    tmap_layout : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t, edge_sims)`` as returned by ``helpers.smiles_to_tmap``,
        computed on **all** molecules in the order ``pool_smiles + test_smiles``.
    smiles_list : list[str]
        SMILES strings in the same row order as ``tmap_layout``.
    partition_labels : np.ndarray
        Integer array of length ``len(smiles_list)``.  Use values 0/1 for the
        basic two-category plot, or 0/1/2/3 to additionally distinguish
        singleton-scaffold compounds.
    output_name : str, optional
        Base filename (without extension) for faerun's HTML output.
    output_path : str, optional
        Directory in which to write the output file.
    title : str, optional
        Plot title.
    train_color : tuple, optional
        RGBA color for Train nodes. Default is matplotlib tab10 blue.
    test_color : tuple, optional
        RGBA color for Test nodes. Default is matplotlib tab10 red.
    train_singleton_color : tuple, optional
        RGBA color for Train nodes with singleton scaffolds. Default is a light
        blue (``#aec7e8``).
    test_singleton_color : tuple, optional
        RGBA color for Test nodes with singleton scaffolds. Default is a light
        red (``#fb9a99``).
    edge_similarity_threshold : float, optional
        Minimum Tanimoto/Jaccard similarity for an MST edge to be drawn.
        Edges below this threshold are omitted. Default 0.0 (all edges drawn).
    point_scale : float, optional
        Relative size of scatter points. Default 3.0.

    Returns
    -------
    Faerun
        Configured faerun instance.
    """
    x, y, s, t, edge_sims = tmap_layout

    if edge_similarity_threshold > 0.0:
        mask = edge_sims >= edge_similarity_threshold
        s, t = s[mask], t[mask]

    n_cats = int(partition_labels.max()) + 1
    all_colors = [train_color, test_color, train_singleton_color, test_singleton_color]
    all_labels = [
        (0, "Train"),
        (1, "Test"),
        (2, "Train (singleton scaffold)"),
        (3, "Test (singleton scaffold)"),
    ]
    legend_labels = all_labels[:n_cats]
    listed_cmap = mcolors.ListedColormap(all_colors[:n_cats], N=n_cats)

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
            "c": partition_labels.tolist(),
            "s": np.full(len(x), point_scale, dtype=float),
            "labels": smiles_list,
        },
        colormap=listed_cmap,
        categorical=True,
        has_legend=True,
        legend_title="Partition",
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
        margin=dict(t=40),
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
        margin=dict(t=40),
        width=width,
        height=height,
    )
    return fig


def plot_hit_nonhit_gap_curve(
    traces: "list[dict]",
    hit_threshold: float = 6.0,
    ylabel: str = "Hit / Non-Hit Gap (activity units)",
    width: int = 750,
    height: int = 475,
) -> go.Figure:
    """Hit/non-hit predicted-activity gap vs. labeled pool size.

    Plots one mean line with ±1 SD shaded band per trace.  Each trace
    represents a (target, model-initialization) combination under the
    **Exploitation** strategy.

    Parameters
    ----------
    traces : list of dict
        Each dict must contain:

        * ``"label"`` (str) — legend entry.
        * ``"n_labeled"`` (array-like) — x-axis values.
        * ``"gap_mean"`` (array-like) — mean gap at each iteration.
        * ``"gap_std"`` (array-like) — standard deviation of gap across seeds.
        * ``"color"`` (str) — CSS hex or RGB color string for the line.
        * ``"dash"`` (str) — Plotly dash style, e.g. ``"solid"`` or ``"dash"``.
    hit_threshold : float
        Hit threshold shown in the y-axis label.
    ylabel : str
        Y-axis label.
    width, height : int, optional
        Figure dimensions in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with one mean line and one ±1 SD shaded band
        per trace, plus a y = 0 reference line.

    """
    fig = go.Figure()

    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.25)", width=1, dash="dash"))

    for tr in traces:
        label: str = tr["label"]
        x = np.asarray(tr["n_labeled"])
        mean = np.asarray(tr["gap_mean"])
        std = np.asarray(tr["gap_std"])
        color: str = tr["color"]
        dash: str = tr.get("dash", "solid")

        upper = mean + std
        lower = mean - std
        fill_color = _hex_to_rgba(color, 0.15) if color.startswith("#") else "rgba(128,128,128,0.15)"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line=dict(width=0),
                legendgroup=label,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                legendgroup=label,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=True,
                line=dict(width=2, color=color, dash=dash),
            )
        )

    fig.update_layout(
        xaxis_title="Number of Labeled Molecules",
        yaxis_title=ylabel,
        legend_title="Config",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            gridwidth=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        plot_bgcolor="white",
        margin=dict(t=40),
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


def plot_learning_curve_grid(
    panels: list[list[tuple[str, "pd.DataFrame | None"]]],
    metric_col: str,
    ylabel: str,
    strategy_order: list[str],
    color_map: dict | None = None,
    cell_width: int = 550,
    cell_height: int = 400,
) -> go.Figure:
    """Multi-panel grid of learning curves with ±1σ bands.

    Parameters
    ----------
    panels : list of list of (title, summary_df or None)
        Row-major 2-D grid of panels. ``None`` entries render as blank axes.
        Each non-``None`` ``summary_df`` must have the same columns expected
        by :func:`plot_learning_curve_with_bands`.
    metric_col : str
        Base name of the metric; expects ``{metric_col}_mean``,
        ``{metric_col}_lower``, and ``{metric_col}_upper`` columns.
    ylabel : str
        Y-axis label applied to all panels.
    strategy_order : list[str]
        Strategies to plot; controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    cell_width, cell_height : int, optional
        Dimensions of each individual panel in pixels.

    Returns
    -------
    go.Figure
        Combined Plotly figure with one subplot per non-``None`` panel.
        Legend entries appear once (first non-``None`` panel only).
    """
    n_rows = len(panels)
    n_cols = max(len(row) for row in panels)

    subplot_titles = []
    for row in panels:
        for title, _ in row:
            subplot_titles.append(title)
        for _ in range(n_cols - len(row)):
            subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    # Compute shared y-axis bounds across all non-None panels.
    all_uppers, all_lowers, all_x_maxes = [], [], []
    for row in panels:
        for _, df in row:
            if df is None:
                continue
            relevant = df[df["strategy"].isin(strategy_order)]
            if relevant.empty:
                continue
            all_uppers.append(relevant[f"{metric_col}_upper"].max())
            all_lowers.append(relevant[f"{metric_col}_lower"].min())
            all_x_maxes.append(relevant["n_labeled"].max())
    y_max = max(all_uppers) if all_uppers else 1.0
    y_min = min(all_lowers) if all_lowers else 0.0
    y_lower = 0.0 if y_min >= 0 else y_min * 1.05
    y_upper = y_max * 1.05
    x_upper = max(all_x_maxes) * 1.02 if all_x_maxes else 1.0

    legend_shown: set[str] = set()

    for r_idx, row in enumerate(panels):
        for c_idx, (_, df) in enumerate(row):
            row_num, col_num = r_idx + 1, c_idx + 1
            if df is None:
                continue

            for strategy in strategy_order:
                df_sub = df[df["strategy"] == strategy].sort_values("n_labeled")
                if df_sub.empty:
                    continue
                color = color_map.get(strategy) if color_map else None
                show_legend = strategy not in legend_shown
                if show_legend:
                    legend_shown.add(strategy)

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
                    ),
                    row=row_num,
                    col=col_num,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sub["n_labeled"],
                        y=df_sub[f"{metric_col}_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=(
                            _hex_to_rgba(color, 0.15)
                            if color
                            else "rgba(128,128,128,0.15)"
                        ),
                        legendgroup=strategy,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_num,
                    col=col_num,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sub["n_labeled"],
                        y=df_sub[f"{metric_col}_mean"],
                        mode="lines",
                        name=strategy,
                        legendgroup=strategy,
                        showlegend=show_legend,
                        line=dict(width=2, color=color),
                    ),
                    row=row_num,
                    col=col_num,
                )

    _axis_style = dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        gridwidth=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )
    fig.update_xaxes(range=[0, x_upper], title_text="Number of Labeled Molecules", **_axis_style)
    fig.update_yaxes(range=[y_lower, y_upper], title_text=ylabel, **_axis_style)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Strategy",
        width=cell_width * n_cols,
        height=cell_height * n_rows,
        margin=dict(t=60, b=60),
    )
    return fig


def plot_hit_discovery_curve_grid(
    panels: list[list[tuple[str, "dict | None"]]],
    strategy_order: list[str],
    color_map: dict | None = None,
    hit_threshold: float = 7.0,
    value_col: str = "pEC50",
    cell_width: int = 550,
    cell_height: int = 400,
) -> go.Figure:
    """Multi-panel grid of hit discovery curves.

    Parameters
    ----------
    panels : list of list of (title, panel_data or None)
        Row-major 2-D grid. ``None`` entries render as blank axes.
        Each non-``None`` ``panel_data`` dict must have keys:

        ``pool_history_df``
            Long-form pool history DataFrame (columns: strategy, seed,
            iteration, {value_col}).
        ``learning_curve_df``
            Long-form learning curve DataFrame (columns: strategy, seed,
            iteration, n_labeled).
        ``max_hits``
            Total number of hits in the pool (integer ceiling).
        ``pool_size``
            Total number of compounds in the pool.

    strategy_order : list[str]
        Strategies to plot; controls legend order.
    color_map : dict or None, optional
        Mapping of strategy name to CSS color string.
    hit_threshold : float, optional
        Activity threshold for counting hits. Default 7.0.
    value_col : str, optional
        Column name in pool_history_df for activity values.
    cell_width, cell_height : int, optional
        Dimensions of each individual panel in pixels.

    Returns
    -------
    go.Figure
        Combined Plotly figure. Legend entries appear once.
    """
    n_rows = len(panels)
    n_cols = max(len(row) for row in panels)

    subplot_titles = []
    for row in panels:
        for title, _ in row:
            subplot_titles.append(title)
        for _ in range(n_cols - len(row)):
            subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    # Compute per-panel summaries first (needed for global y bounds).
    summaries: list[list[tuple[Any, ...] | None]] = []
    for row in panels:
        row_cells: list[tuple[Any, ...] | None] = []
        for _, data in row:
            if data is None:
                row_cells.append(None)
                continue
            pool_history_df = data["pool_history_df"]
            learning_curve_df = data["learning_curve_df"]
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
                hits_summary["n_hits_mean"] + hits_summary["n_hits_std"],
                data["max_hits"],
            )
            row_cells.append((hits_summary, data["max_hits"], data["pool_size"]))
        summaries.append(row_cells)

    # Shared y and x upper bounds across all panels.
    global_y_upper = 0.0
    global_x_upper = 0.0
    for r_idx, row_cells in enumerate(summaries):
        for c_idx, cell in enumerate(row_cells):
            if cell is None:
                continue
            hits_summary, max_hits, pool_size = cell
            candidate = max(
                hits_summary["n_hits_upper"].max() if not hits_summary.empty else 0,
                max_hits,
            ) * 1.10
            global_y_upper = max(global_y_upper, candidate)
            if not hits_summary.empty:
                global_x_upper = max(global_x_upper, hits_summary["n_labeled"].max())

    legend_shown: set[str] = set()

    for r_idx, (row, row_cells) in enumerate(zip(panels, summaries)):
        for c_idx, ((_, data), cell) in enumerate(zip(row, row_cells)):
            row_num, col_num = r_idx + 1, c_idx + 1
            if data is None or cell is None:
                continue
            hits_summary, max_hits, pool_size = cell
            color = color_map.get("Random") if color_map else None

            # Random acquisition baseline (diagonal ramp).
            fig.add_trace(
                go.Scatter(
                    x=[0, pool_size],
                    y=[0, max_hits],
                    mode="lines",
                    name="Random (expected)",
                    legendgroup="__random_baseline__",
                    showlegend="__random_baseline__" not in legend_shown,
                    line=dict(color="#888888", width=1.2, dash="dash"),
                    hoverinfo="skip",
                ),
                row=row_num,
                col=col_num,
            )
            legend_shown.add("__random_baseline__")

            for strategy in strategy_order:
                df_sub = hits_summary[
                    hits_summary["strategy"] == strategy
                ].sort_values("n_labeled")
                if df_sub.empty:
                    continue
                color = color_map.get(strategy) if color_map else None
                show_legend = strategy not in legend_shown
                if show_legend:
                    legend_shown.add(strategy)

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
                    ),
                    row=row_num,
                    col=col_num,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sub["n_labeled"],
                        y=df_sub["n_hits_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=(
                            _hex_to_rgba(color, 0.15)
                            if color
                            else "rgba(128,128,128,0.15)"
                        ),
                        legendgroup=strategy,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row_num,
                    col=col_num,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sub["n_labeled"],
                        y=df_sub["n_hits_mean"],
                        mode="lines",
                        name=strategy,
                        legendgroup=strategy,
                        showlegend=show_legend,
                        line=dict(width=2, color=color),
                    ),
                    row=row_num,
                    col=col_num,
                )

            # Hit ceiling reference line (per subplot).
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=max_hits,
                y1=max_hits,
                xref=f"x{'' if (r_idx * n_cols + c_idx) == 0 else r_idx * n_cols + c_idx + 1} domain",
                yref=f"y{'' if (r_idx * n_cols + c_idx) == 0 else r_idx * n_cols + c_idx + 1}",
                line=dict(color="black", width=1.2, dash="dash"),
            )
            # Annotation for the hit ceiling.
            axis_idx = r_idx * n_cols + c_idx
            fig.add_annotation(
                x=1,
                y=max_hits,
                xref=f"x{'' if axis_idx == 0 else axis_idx + 1} domain",
                yref=f"y{'' if axis_idx == 0 else axis_idx + 1}",
                text=f"Total hits in pool ({max_hits})",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11),
            )

    _axis_style = dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        gridwidth=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )
    fig.update_xaxes(range=[0, global_x_upper * 1.02], title_text="Number of Labeled Molecules", **_axis_style)
    fig.update_yaxes(
        range=[0, global_y_upper],
        title_text=f"Hits Found ({value_col} \u2265 {hit_threshold})",
        **_axis_style,
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Strategy",
        width=cell_width * n_cols,
        height=cell_height * n_rows,
        margin=dict(t=60, b=60),
    )
    return fig


def plot_predicted_distribution_animation(
    history: list[dict],
    df_test: pd.DataFrame,
    hit_threshold: float = 6.0,
    activity_col: str = "pEC50",
    strategy: str = "Exploitation",
    seed_label: str = "Seed 42",
    model: str = "",
    n_kde_points: int = 400,
    width: int = 800,
    height: int = 500,
    per_frame_y_true: list[np.ndarray] | None = None,
    n_seeds: int = 1,
) -> go.Figure:
    """Animated figure showing committee predicted-activity distributions for hits vs non-hits.

    At each AL iteration two overlapping, semi-transparent KDE curves are drawn
    using the committee's predictions — one for compounds that are ground-truth
    hits (activity ≥ threshold) and one for non-hits.  Vertical lines mark each
    group's mean prediction and a horizontal annotation shows the gap value.  A
    Plotly slider lets viewers scrub through iterations; Play/Pause buttons run
    the animation.

    By default, predictions come from ``"y_test_pred"`` in each history state
    and ground-truth labels from ``df_test``.  When ``per_frame_y_true`` is
    provided, ``"y_test_pred"`` is treated as unlabeled-pool predictions and the
    corresponding per-frame true-activity arrays are used instead of ``df_test``.
    In that case the hit/non-hit mask and compound counts are recomputed each
    frame, so the legend accurately reflects the shrinking unlabeled pool.

    Parameters
    ----------
    history : list[dict]
        Per-iteration state records from ``run_active_learning``, each
        containing at minimum ``"iteration"``, ``"n_labeled"``, and
        ``"y_test_pred"`` (``np.ndarray``).  When combining across seeds,
        ``y_test_pred`` should be the concatenation of all seeds' predictions
        and ``df_test`` (or ``per_frame_y_true``) should be sized to match.
    df_test : pd.DataFrame
        Held-out test DataFrame with an ``activity_col`` column giving
        ground-truth activity values.  Ignored when ``per_frame_y_true`` is
        provided.
    hit_threshold : float, optional
        Activity value at or above which a compound is counted as a hit.
        Default is 6.0.
    activity_col : str, optional
        Column in ``df_test`` containing the ground-truth activity values.
        Default is ``"pEC50"``.
    strategy : str, optional
        Strategy name shown in figure titles.  Default is ``"Exploitation"``.
    seed_label : str, optional
        Pre-formatted seed description shown in the figure title, e.g.
        ``"Seed 42"`` for a single run or ``"5 seeds"`` for a combined run.
        Default is ``"Seed 42"``.
    model : str, optional
        Model name shown as the first element of the figure title (e.g.
        ``"CheMeleon"`` or ``"ChemProp"``).  When empty, omitted from title.
        Default is ``""``.
    n_kde_points : int, optional
        Number of evenly spaced points used to evaluate each KDE curve.
        Default is 400.
    width : int, optional
        Figure width in pixels.  Default is 800.
    height : int, optional
        Figure height in pixels.  Default is 500.
    per_frame_y_true : list of np.ndarray, optional
        When provided, a list of length ``len(history)`` where each element is
        a 1-D array of ground-truth activity values for the compounds whose
        predictions appear in that frame's ``"y_test_pred"``.  Use this when
        predictions come from the unlabeled pool (which shrinks each iteration)
        rather than from a fixed test set.  Default is ``None``.
    n_seeds : int, optional
        Number of independent seeds whose predictions have been concatenated
        into each frame's ``"y_test_pred"``.  Legend counts (n_hits, n_nonhits)
        are divided by this value and floored so they represent a single-seed
        equivalent.  Default is ``1``.

    Returns
    -------
    go.Figure
        Animated Plotly figure.  The figure is intended for HTML export only
        (Plotly animations are not supported in static SVG/PNG output).

    Raises
    ------
    ValueError
        If ``history`` is empty.
    """
    from scipy.stats import gaussian_kde  # local import — not needed elsewhere

    if not history:
        raise ValueError("history is empty; nothing to animate.")

    _pool_mode = per_frame_y_true is not None

    if not _pool_mode:
        y_true_fixed = df_test[activity_col].to_numpy()
        hit_mask_fixed = y_true_fixed >= hit_threshold
        nonhit_mask_fixed = ~hit_mask_fixed
        n_hits_fixed = int(hit_mask_fixed.sum())
        n_nonhits_fixed = int(nonhit_mask_fixed.sum())

    # Colour palette: crimson for hits (matches GTM active color), gray for non-hits
    _HIT_COLOR = "rgba(220, 20, 60, 0.55)"
    _HIT_LINE = "rgba(220, 20, 60, 1.0)"
    _NONHIT_COLOR = "rgba(120, 120, 120, 0.40)"
    _NONHIT_LINE = "rgba(80, 80, 80, 1.0)"
    _GAP_COLOR = "#444444"

    # KDE grid spans the full fixed axis range [0, 10] so curves taper
    # smoothly to zero at the edges rather than cutting off abruptly.
    x_grid = np.linspace(0.0, 10.0, n_kde_points)

    def _kde_trace(
        y_pred_subset: np.ndarray,
        x_grid: np.ndarray,
        fill_color: str,
        line_color: str,
        name: str,
        showlegend: bool,
    ) -> go.Scatter:
        """Return a filled KDE scatter trace for one activity class."""
        if len(y_pred_subset) < 2:
            density = np.zeros_like(x_grid)
        else:
            kde = gaussian_kde(y_pred_subset)
            density = kde(x_grid)
        return go.Scatter(
            x=x_grid.tolist(),
            y=density.tolist(),
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=1.5),
            name=name,
            showlegend=showlegend,
        )

    def _vline_trace(
        x_val: float,
        y_top: float,
        color: str,
        name: str,
        showlegend: bool,
    ) -> go.Scatter:
        """Return a vertical dashed line trace at x_val reaching y_top."""
        return go.Scatter(
            x=[x_val, x_val],
            y=[0, y_top],
            mode="lines",
            line=dict(color=color, dash="dash", width=1.5),
            name=name,
            showlegend=showlegend,
        )

    # Pre-pass: compute per-frame peak densities to fix vline tops before building frames.
    frame_ymaxes_pre: list[float] = []
    for ki, state in enumerate(history):
        y_pred_pre = np.asarray(state["y_test_pred"])
        if _pool_mode:
            _yt = per_frame_y_true[ki]  # type: ignore[index]
            _hm = _yt >= hit_threshold
            y_hits_pre = y_pred_pre[_hm]
            y_nonhits_pre = y_pred_pre[~_hm]
        else:
            y_hits_pre = y_pred_pre[hit_mask_fixed]
            y_nonhits_pre = y_pred_pre[nonhit_mask_fixed]
        densities: list[float] = []
        for subset in (y_hits_pre, y_nonhits_pre):
            if len(subset) >= 2:
                densities.extend(gaussian_kde(subset)(x_grid).tolist())
        frame_ymaxes_pre.append(max(densities) if densities else 1e-6)
    global_y_max_pre = max(frame_ymaxes_pre) * 1.15
    vline_offset = 0.1 * global_y_max_pre

    # Fixed global heights computed once so all frames are consistent.
    # Vlines and gap bracket all reference the same anchor: the highest KDE
    # peak across any frame plus a 10%-of-axis fixed offset.
    _peak_y = max(frame_ymaxes_pre)
    _vline_top = _peak_y + vline_offset       # top of all vertical dashed lines
    _gap_bracket_y = _vline_top               # gap bracket sits right at vline tops
    _gap_text_y = _gap_bracket_y + 0.15 * vline_offset  # small clearance above bracket

    frames: list[go.Frame] = []
    frame_ymaxes: list[float] = []

    for ki, state in enumerate(history):
        y_pred = np.asarray(state["y_test_pred"])
        n_labeled = state["n_labeled"]
        k = state["iteration"]

        if _pool_mode:
            _yt = per_frame_y_true[ki]  # type: ignore[index]
            hit_mask = _yt >= hit_threshold
            nonhit_mask = ~hit_mask
            n_hits = int(hit_mask.sum())
            n_nonhits = int(nonhit_mask.sum())
        else:
            hit_mask = hit_mask_fixed
            nonhit_mask = nonhit_mask_fixed
            n_hits = n_hits_fixed
            n_nonhits = n_nonhits_fixed

        y_hits = y_pred[hit_mask]
        y_nonhits = y_pred[nonhit_mask]

        mean_hit = float(np.mean(y_hits))
        mean_nonhit = float(np.mean(y_nonhits))
        gap = mean_hit - mean_nonhit

        trace_hits = _kde_trace(
            y_hits, x_grid, _HIT_COLOR, _HIT_LINE,
            f"Hits (≥{hit_threshold:.1f}, n={int(np.floor(n_hits / n_seeds))})", showlegend=True,
        )
        trace_nonhits = _kde_trace(
            y_nonhits, x_grid, _NONHIT_COLOR, _NONHIT_LINE,
            f"Non-hits (n={int(np.floor(n_nonhits / n_seeds))})", showlegend=True,
        )

        all_density = list(np.array(trace_hits.y)) + list(np.array(trace_nonhits.y))
        y_max = max(float(np.max(all_density)), 1e-6)
        frame_ymaxes.append(y_max)

        trace_vhit = _vline_trace(
            mean_hit, _vline_top, _HIT_LINE,
            f"Mean hit ŷ = {mean_hit:.2f}", showlegend=False,
        )
        trace_vnonhit = _vline_trace(
            mean_nonhit, _vline_top, _NONHIT_LINE,
            f"Mean non-hit ŷ = {mean_nonhit:.2f}", showlegend=False,
        )

        # Horizontal gap bracket: line trace + separate text trace slightly above.
        # Both use the globally fixed heights computed in the pre-pass so the
        # bracket sits at a consistent position above the KDE peaks every frame.
        trace_gap_line = go.Scatter(
            x=[mean_nonhit, mean_hit],
            y=[_gap_bracket_y, _gap_bracket_y],
            mode="lines",
            line=dict(color=_GAP_COLOR, width=2),
            showlegend=False,
            name="gap",
        )
        trace_gap_text = go.Scatter(
            x=[mean_nonhit - 0.2],
            y=[_gap_text_y],
            mode="text",
            text=[f"gap = {gap:+.2f}"],
            textposition="middle left",
            textfont=dict(size=13, color=_GAP_COLOR),
            showlegend=False,
            name="gap_label",
        )

        model_prefix = f"Model: {model} | " if model else ""
        frame_title = (
            f"{model_prefix}Strategy: {strategy} | {seed_label} | "
            f"{n_labeled} labeled"
        )
        frames.append(
            go.Frame(
                data=[trace_hits, trace_nonhits, trace_vhit, trace_vnonhit, trace_gap_line, trace_gap_text],
                name=str(k),
                layout=go.Layout(title_text=frame_title),
            )
        )

    global_y_max = max(max(frame_ymaxes) * 1.15, _gap_text_y * 1.12)
    initial_frame = frames[0]
    fig = go.Figure(
        data=initial_frame.data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=initial_frame.layout.title.text,
                font=dict(size=14),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=f"Committee mean predicted {activity_col} ({'unlabeled pool' if _pool_mode else 'test set'})",
                range=[0, 10],
                showgrid=True,
                gridcolor="#eeeeee",
                zeroline=False,
                linecolor="black",
                linewidth=1,
                mirror=True,
            ),
            yaxis=dict(
                title="Density",
                range=[0, global_y_max],
                showgrid=True,
                gridcolor="#eeeeee",
                zeroline=False,
                linecolor="black",
                linewidth=1,
                mirror=True,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
            width=width,
            height=height,
            margin=dict(r=90, b=140),
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
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(s["iteration"])],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(s["iteration"]),
                        )
                        for s in history
                    ],
                )
            ],
            updatemenus=[],
        ),
    )
    return fig



def plot_mirror_distribution_animation(
    history_a: list[dict],
    history_b: list[dict],
    label_a: str = "ChemProp",
    label_b: str = "CheMeleon",
    hit_threshold: float = 6.0,
    activity_col: str = "pEC50",
    strategy: str = "Exploitation",
    seed_label: str = "5 seeds",
    n_kde_points: int = 400,
    width: int = 900,
    height: int = 700,
    per_frame_y_true_a: "list | None" = None,
    per_frame_y_true_b: "list | None" = None,
    n_seeds: int = 1,
) -> go.Figure:
    """Animated mirror figure comparing two models' hit/non-hit predicted distributions.

    Displays model A (e.g. ChemProp) in the **positive y half** and model B
    (e.g. CheMeleon) in the **negative y half** of a single shared axes.  Both
    halves use the same global y-scale derived from the maximum KDE peak across
    both models and all frames.  A thick zero line acts as the mirror axis.

    Both models must be in pool mode (``per_frame_y_true_a`` and
    ``per_frame_y_true_b`` are required).  Frames are aligned to
    ``min(len(history_a), len(history_b))``.

    Parameters
    ----------
    history_a, history_b : list of dict
        Per-iteration state records for each model.  Each entry must contain
        ``"iteration"``, ``"n_labeled"``, and ``"y_test_pred"`` (pool scores
        concatenated across seeds).
    label_a, label_b : str
        Display names for the two models (used in legend group titles and
        layout annotations).
    hit_threshold : float
        Activity cutoff defining hits.  Default is 6.0.
    activity_col : str
        Activity column name used in axis labels.  Default is ``"pEC50"``.
    strategy : str
        Strategy name shown in the figure title.  Default is ``"Exploitation"``.
    seed_label : str
        Pre-formatted seed count string for the title, e.g. ``"5 seeds"``.
    n_kde_points : int
        KDE evaluation grid resolution.  Default is 400.
    width, height : int
        Figure dimensions in pixels.
    per_frame_y_true_a, per_frame_y_true_b : list of np.ndarray
        Per-frame ground-truth activity arrays (one per iteration) for pool mode.
    n_seeds : int
        Number of seeds concatenated into each frame; legend counts are divided
        by this value and floored to give single-seed equivalents.

    Returns
    -------
    go.Figure
        Animated Plotly figure (HTML export only).
    """
    import numpy as _np
    from scipy.stats import gaussian_kde

    if not history_a or not history_b:
        raise ValueError("Both history_a and history_b must be non-empty.")
    if per_frame_y_true_a is None or per_frame_y_true_b is None:
        raise ValueError(
            "plot_mirror_distribution_animation requires pool mode: "
            "per_frame_y_true_a and per_frame_y_true_b must be provided."
        )

    n_frames = min(len(history_a), len(history_b))
    history_a = history_a[:n_frames]
    history_b = history_b[:n_frames]
    per_frame_y_true_a = per_frame_y_true_a[:n_frames]
    per_frame_y_true_b = per_frame_y_true_b[:n_frames]

    _HIT_COLOR_A = "rgba(220, 20, 60, 0.55)"
    _HIT_LINE_A = "rgba(220, 20, 60, 1.0)"
    _HIT_COLOR_B = "rgba(34, 139, 34, 0.55)"
    _HIT_LINE_B = "rgba(34, 139, 34, 1.0)"
    _NONHIT_COLOR = "rgba(120, 120, 120, 0.40)"
    _NONHIT_LINE = "rgba(80, 80, 80, 1.0)"
    _GAP_COLOR = "#444444"

    x_grid = _np.linspace(0.0, 10.0, n_kde_points)

    def _dens(y_subset, sign=1.0):
        if len(y_subset) < 2:
            return _np.zeros_like(x_grid)
        return sign * gaussian_kde(y_subset)(x_grid)

    # Pre-pass: shared global y-scale across both models and all frames
    frame_ymaxes_pre: list[float] = []
    for ki in range(n_frames):
        y_pred_a = _np.asarray(history_a[ki]["y_test_pred"])
        y_pred_b = _np.asarray(history_b[ki]["y_test_pred"])
        yt_a = per_frame_y_true_a[ki]
        yt_b = per_frame_y_true_b[ki]
        hm_a = yt_a >= hit_threshold
        hm_b = yt_b >= hit_threshold
        densities: list[float] = []
        for pred, hm in [(y_pred_a, hm_a), (y_pred_b, hm_b)]:
            for subset in (pred[hm], pred[~hm]):
                if len(subset) >= 2:
                    densities.extend(gaussian_kde(subset)(x_grid).tolist())
        frame_ymaxes_pre.append(max(densities) if densities else 1e-6)

    global_y_max_pre = max(frame_ymaxes_pre) * 1.15
    vline_offset = 0.1 * global_y_max_pre
    _peak_y = max(frame_ymaxes_pre)
    _vline_top = _peak_y + vline_offset
    _gap_bracket_y = _vline_top
    _gap_text_y = _gap_bracket_y + 0.15 * vline_offset

    frames: list[go.Frame] = []
    frame_ymaxes: list[float] = []

    for ki in range(n_frames):
        state_a = history_a[ki]
        state_b = history_b[ki]
        y_pred_a = _np.asarray(state_a["y_test_pred"])
        y_pred_b = _np.asarray(state_b["y_test_pred"])
        n_labeled = state_a["n_labeled"]
        k = state_a["iteration"]

        yt_a = per_frame_y_true_a[ki]
        yt_b = per_frame_y_true_b[ki]
        hm_a = yt_a >= hit_threshold
        hm_b = yt_b >= hit_threshold

        y_hits_a = y_pred_a[hm_a]
        y_nonhits_a = y_pred_a[~hm_a]
        y_hits_b = y_pred_b[hm_b]
        y_nonhits_b = y_pred_b[~hm_b]

        mean_hit_a = float(_np.mean(y_hits_a)) if len(y_hits_a) else float("nan")
        mean_nonhit_a = float(_np.mean(y_nonhits_a)) if len(y_nonhits_a) else float("nan")
        mean_hit_b = float(_np.mean(y_hits_b)) if len(y_hits_b) else float("nan")
        mean_nonhit_b = float(_np.mean(y_nonhits_b)) if len(y_nonhits_b) else float("nan")
        gap_a = mean_hit_a - mean_nonhit_a
        gap_b = mean_hit_b - mean_nonhit_b

        n_hits_a = int(hm_a.sum())
        n_nonhits_a = int((~hm_a).sum())
        n_hits_b = int(hm_b.sum())
        n_nonhits_b = int((~hm_b).sum())

        kde_hits_a = _dens(y_hits_a, sign=1.0)
        kde_nonhits_a = _dens(y_nonhits_a, sign=1.0)
        kde_hits_b = _dens(y_hits_b, sign=-1.0)
        kde_nonhits_b = _dens(y_nonhits_b, sign=-1.0)

        y_max = max(float(kde_hits_a.max()), float(kde_nonhits_a.max()), 1e-6)
        frame_ymaxes.append(y_max)

        n_hits_a_d = int(_np.floor(n_hits_a / n_seeds))
        n_nonhits_a_d = int(_np.floor(n_nonhits_a / n_seeds))
        n_hits_b_d = int(_np.floor(n_hits_b / n_seeds))
        n_nonhits_b_d = int(_np.floor(n_nonhits_b / n_seeds))

        # Model A — top half (positive y)
        t_hits_a = go.Scatter(
            x=x_grid.tolist(), y=kde_hits_a.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=_HIT_COLOR_A, line=dict(color=_HIT_LINE_A, width=1.5),
            name=f"Hits (≥{hit_threshold:.1f}, n={n_hits_a_d})",
            legendgroup="a", legendgrouptitle=dict(text=label_a),
            showlegend=True,
        )
        t_nonhits_a = go.Scatter(
            x=x_grid.tolist(), y=kde_nonhits_a.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=_NONHIT_COLOR, line=dict(color=_NONHIT_LINE, width=1.5),
            name=f"Non-hits (n={n_nonhits_a_d})",
            legendgroup="a", showlegend=True,
        )
        t_vhit_a = go.Scatter(
            x=[mean_hit_a, mean_hit_a], y=[0, _vline_top],
            mode="lines", line=dict(color=_HIT_LINE_A, dash="dash", width=1.5),
            showlegend=False, name="mean_hit_a",
        )
        t_vnonhit_a = go.Scatter(
            x=[mean_nonhit_a, mean_nonhit_a], y=[0, _vline_top],
            mode="lines", line=dict(color=_NONHIT_LINE, dash="dash", width=1.5),
            showlegend=False, name="mean_nonhit_a",
        )
        t_gap_line_a = go.Scatter(
            x=[mean_nonhit_a, mean_hit_a], y=[_gap_bracket_y, _gap_bracket_y],
            mode="lines", line=dict(color=_GAP_COLOR, width=2),
            showlegend=False, name="gap_a",
        )
        t_gap_text_a = go.Scatter(
            x=[mean_nonhit_a - 0.2], y=[_gap_text_y],
            mode="text", text=[f"gap = {gap_a:+.2f}"],
            textposition="middle left",
            textfont=dict(size=13, color=_GAP_COLOR),
            showlegend=False, name="gap_label_a",
        )

        # Model B — bottom half (negative y)
        t_hits_b = go.Scatter(
            x=x_grid.tolist(), y=kde_hits_b.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=_HIT_COLOR_B, line=dict(color=_HIT_LINE_B, width=1.5),
            name=f"Hits (≥{hit_threshold:.1f}, n={n_hits_b_d})",
            legendgroup="b", legendgrouptitle=dict(text=label_b),
            showlegend=True,
        )
        t_nonhits_b = go.Scatter(
            x=x_grid.tolist(), y=kde_nonhits_b.tolist(),
            mode="lines", fill="tozeroy",
            fillcolor=_NONHIT_COLOR, line=dict(color=_NONHIT_LINE, width=1.5),
            name=f"Non-hits (n={n_nonhits_b_d})",
            legendgroup="b", showlegend=True,
        )
        t_vhit_b = go.Scatter(
            x=[mean_hit_b, mean_hit_b], y=[0, -_vline_top],
            mode="lines", line=dict(color=_HIT_LINE_B, dash="dash", width=1.5),
            showlegend=False, name="mean_hit_b",
        )
        t_vnonhit_b = go.Scatter(
            x=[mean_nonhit_b, mean_nonhit_b], y=[0, -_vline_top],
            mode="lines", line=dict(color=_NONHIT_LINE, dash="dash", width=1.5),
            showlegend=False, name="mean_nonhit_b",
        )
        t_gap_line_b = go.Scatter(
            x=[mean_nonhit_b, mean_hit_b], y=[-_gap_bracket_y, -_gap_bracket_y],
            mode="lines", line=dict(color=_GAP_COLOR, width=2),
            showlegend=False, name="gap_b",
        )
        t_gap_text_b = go.Scatter(
            x=[mean_nonhit_b - 0.2], y=[-_gap_text_y],
            mode="text", text=[f"gap = {gap_b:+.2f}"],
            textposition="middle left",
            textfont=dict(size=13, color=_GAP_COLOR),
            showlegend=False, name="gap_label_b",
        )

        frame_title = f"Strategy: {strategy} | {seed_label} | {n_labeled} labeled"
        frames.append(go.Frame(
            data=[
                t_hits_a, t_nonhits_a, t_vhit_a, t_vnonhit_a, t_gap_line_a, t_gap_text_a,
                t_hits_b, t_nonhits_b, t_vhit_b, t_vnonhit_b, t_gap_line_b, t_gap_text_b,
            ],
            name=str(k),
            layout=go.Layout(title_text=frame_title),
        ))

    global_y_max = max(max(frame_ymaxes) * 1.15, _gap_text_y * 1.12)
    initial_frame = frames[0]
    fig = go.Figure(
        data=initial_frame.data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=initial_frame.layout.title.text,
                font=dict(size=14),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=f"Committee mean predicted {activity_col} (unlabeled pool)",
                range=[0, 10],
                showgrid=True, gridcolor="#eeeeee",
                zeroline=False, linecolor="black", linewidth=1, mirror=True,
            ),
            yaxis=dict(
                title="Density",
                range=[-global_y_max, global_y_max],
                showgrid=True, gridcolor="#eeeeee",
                zeroline=False, linecolor="black", linewidth=1, mirror=True,
            ),
            shapes=[dict(
                type="line",
                x0=0, x1=1, y0=0, y1=0,
                xref="paper", yref="y",
                line=dict(color="black", width=2),
                layer="above",
            )],
            annotations=[
                dict(
                    text=f"▲ {label_a}",
                    x=0.01, y=_vline_top * 0.92,
                    xref="paper", yref="y",
                    showarrow=False,
                    font=dict(size=13, color="#333333"),
                    xanchor="left",
                ),
                dict(
                    text=f"▼ {label_b}",
                    x=0.01, y=-_vline_top * 0.92,
                    xref="paper", yref="y",
                    showarrow=False,
                    font=dict(size=13, color="#333333"),
                    xanchor="left",
                ),
            ],
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                x=0.98, y=0.98, xanchor="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc", borderwidth=1,
                tracegroupgap=10,
            ),
            width=width,
            height=height,
            margin=dict(r=90, b=140),
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    prefix="Iteration ",
                    visible=True,
                    xanchor="center",
                    font=dict(color="black"),
                ),
                font=dict(color="rgba(0,0,0,0)"),
                pad=dict(t=50),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(s["iteration"])],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(s["iteration"]),
                    )
                    for s in history_a[:n_frames]
                ],
            )],
            updatemenus=[],
        ),
    )
    return fig
