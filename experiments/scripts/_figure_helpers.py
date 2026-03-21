# ruff: noqa: N802, N803, N806
"""Shared drawing utilities for arXiv results figures.

Wraps reusable functions from generate_algorithm_overview.py and plotting_styles.py
to avoid duplication across the neighbourhood, shortest-path, and round-trip figure
scripts.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

# Re-exports from generate_algorithm_overview
# Re-exports from plotting_styles
from experiments.plotting_styles import (  # noqa: E402, F401
    PAUL_TOL_BRIGHT,
    TOKEN_COLORS,
    apply_ieee_style,
    save_figure,
    tokenize_for_display,
)

# Re-export from exp1_shortest_path
from experiments.scripts.exp1_shortest_path import (  # noqa: E402, F401
    levenshtein_with_backtrace,
)
from experiments.scripts.generate_algorithm_overview import (  # noqa: E402, F401
    _NODE_TYPE_COLORS,
    _compute_dag_layout,
    compute_d2s_ghost_state,
    draw_dag,
    node_display_label,
    pick_snapshots,
    render_token_heatmap_horizontal,
)
from isalsr.adapters.sympy_adapter import SympyAdapter  # noqa: E402
from isalsr.core.canonical import canonical_string, levenshtein  # noqa: E402, F401
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/figures/arXiv_figures"

# Colors for edit operation types
EDIT_COLORS = {
    "substitute": PAUL_TOL_BRIGHT["red"],  # #EE6677
    "insertion": PAUL_TOL_BRIGHT["blue"],  # #4477AA
    "deletion": PAUL_TOL_BRIGHT["yellow"],  # #CCBB44
}

# Panel colors for edit groups
PANEL_COLORS = {
    "substitution": PAUL_TOL_BRIGHT["blue"],  # #4477AA
    "insertion": PAUL_TOL_BRIGHT["green"],  # #228833
    "deletion": PAUL_TOL_BRIGHT["red"],  # #EE6677
    "base": "#444444",
}


def dag_from_string(string: str, num_vars: int) -> LabeledDAG:
    """Build a LabeledDAG from an IsalSR instruction string.

    Args:
        string: IsalSR instruction string.
        num_vars: Number of input variables.

    Returns:
        The resulting LabeledDAG.
    """
    s2d = StringToDAG(string, num_variables=num_vars)
    return s2d.run()


def dag_to_sympy_latex(dag: LabeledDAG) -> str:
    """Convert a LabeledDAG to a LaTeX math expression string via SymPy.

    Handles multi-output DAGs by finding the single non-VAR/non-CONST sink.
    Falls back to "?" if conversion fails.

    Args:
        dag: The labeled DAG.

    Returns:
        LaTeX string suitable for matplotlib math rendering.
    """
    import sympy

    from isalsr.core.node_types import NodeType

    adapter = SympyAdapter()
    try:
        expr = adapter.to_sympy(dag)
        return sympy.latex(expr)
    except Exception:
        # Try to find a single meaningful output node
        sinks = []
        for i in range(dag.node_count):
            if len(dag.out_neighbors(i)) == 0:
                label = dag.node_label(i)
                if label not in (NodeType.VAR, NodeType.CONST):
                    sinks.append(i)
        if len(sinks) == 1:
            # Build expression for just that subtree
            try:
                expr = adapter._node_to_sympy(dag, sinks[0], {})
                return sympy.latex(expr)
            except Exception:
                pass
        return "?"


def draw_math_label(ax: plt.Axes, latex_str: str, fontsize: int = 8) -> None:
    """Render a LaTeX math expression centered in the given axes.

    Args:
        ax: Matplotlib axes (typically a narrow row below the DAG).
        latex_str: LaTeX expression string (without $ delimiters).
        fontsize: Font size.
    """
    ax.text(
        0.5,
        0.5,
        f"${latex_str}$",
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    ax.axis("off")


def draw_expression_cell(
    ax_dag: plt.Axes,
    ax_heatmap: plt.Axes,
    dag: LabeledDAG,
    canon_string: str,
    *,
    ax_math: plt.Axes | None = None,
    node_size: float = 0.35,
    cell_width: float = 8.0,
    cell_height: float = 7.0,
    math_fontsize: int = 7,
    title: str = "",
    title_fontsize: int = 8,
    ghost_nodes: set[int] | None = None,
    ghost_edges: set[tuple[int, int]] | None = None,
    pos: dict[int, tuple[float, float]] | None = None,
    current_token_idx: int = -2,
) -> dict[int, tuple[float, float]]:
    """Draw a complete expression cell: DAG + token heatmap + optional math label.

    Args:
        ax_dag: Axes for the DAG visualization.
        ax_heatmap: Axes for the token heatmap strip.
        dag: The labeled DAG to draw.
        canon_string: The instruction string to display.
        ax_math: Optional axes for the math expression label.
        node_size: DAG node circle radius.
        cell_width: Token heatmap cell width.
        cell_height: Token heatmap cell height.
        math_fontsize: Font size for the math expression.
        title: Optional title above the cell.
        title_fontsize: Font size for the title.
        ghost_nodes: Ghost node IDs for D2S visualization.
        ghost_edges: Ghost edge tuples for D2S visualization.
        pos: Pre-computed DAG layout positions.
        current_token_idx: Last completed token index (-2 = all completed).

    Returns:
        The DAG layout positions used.
    """
    tokens = tokenize_for_display(canon_string)
    n_tokens = len(tokens)

    # Determine current_token_idx: -2 means all completed
    if current_token_idx == -2:
        current_token_idx = n_tokens - 1

    # Draw DAG
    pos = draw_dag(
        ax_dag,
        dag,
        ghost_nodes=ghost_nodes,
        ghost_edges=ghost_edges,
        node_size=node_size,
        pos=pos,
    )

    if title:
        ax_dag.set_title(title, fontsize=title_fontsize, pad=3)

    # Draw token heatmap
    render_token_heatmap_horizontal(
        ax_heatmap,
        tokens,
        current_token_idx,
        cell_width=cell_width,
        cell_height=cell_height,
    )

    # Draw math label if axes provided
    if ax_math is not None:
        try:
            latex_str = dag_to_sympy_latex(dag)
            draw_math_label(ax_math, latex_str, fontsize=math_fontsize)
        except Exception:
            logger.warning("Could not render math expression for DAG")
            ax_math.axis("off")

    return pos


def add_background_panel(
    fig: plt.Figure,
    axes_list: list[plt.Axes],
    color: str,
    label: str = "",
    alpha: float = 0.04,
    label_fontsize: int = 9,
    pad: float = 0.008,
    pad_x: float | None = None,
    pad_y: float | None = None,
    extend_top: float | None = None,
    extend_bottom: float | None = None,
) -> None:
    """Add a semi-transparent rounded background panel around a group of axes.

    Args:
        fig: The figure.
        axes_list: List of axes to enclose.
        color: Panel face/edge color.
        label: Optional label text (rendered above the panel).
        alpha: Panel transparency.
        label_fontsize: Label font size.
        pad: Padding around the bounding box (used if pad_x/pad_y not set).
        pad_x: Horizontal padding (overrides pad).
        pad_y: Vertical padding (overrides pad).
        extend_top: If set, extend the panel top to this y coordinate (figure fraction).
        extend_bottom: If set, extend the panel bottom to this y coordinate (figure fraction).
    """
    from matplotlib.transforms import Bbox

    px = pad_x if pad_x is not None else pad
    py = pad_y if pad_y is not None else pad

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    bboxes = []
    for ax in axes_list:
        bb = ax.get_tightbbox(renderer)
        if bb is not None:
            bboxes.append(bb.transformed(fig.transFigure.inverted()))
    if not bboxes:
        return

    bb = Bbox.union(bboxes)

    x0 = bb.x0 - px
    y0 = bb.y0 - py
    x1 = bb.x1 + px
    y1 = bb.y1 + py

    if extend_top is not None:
        y1 = max(y1, extend_top)
    if extend_bottom is not None:
        y0 = min(y0, extend_bottom)

    fig.patches.append(
        mpatches.FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.005",
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=0.8,
            transform=fig.transFigure,
            zorder=0,
        )
    )
    if label:
        fig.text(
            bb.x0 + bb.width / 2,
            y1 + 0.008,
            label,
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            fontweight="bold",
            color=color,
            transform=fig.transFigure,
        )
