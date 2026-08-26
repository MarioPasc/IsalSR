# ruff: noqa: N802, N803, N806
"""Shortest path figure for the IsalSR arXiv paper.

Demonstrates that Levenshtein distance on canonical strings induces a meaningful
metric on expression DAGs. Shows a complete shortest edit path between two
expressions, with DAG visualizations and mathematical expressions at each step.

Pair: x_0^2 -> sin(x_0) + x_0 (distance=2, all intermediates valid)
  Source:       V^VkPnc  (x_0 ^ 1.0, effectively x_0^2 with constant)
  Intermediate: V+VkPnc  (x_0 + 1.0)
  Target:       V+VsPnc  (x_0 + sin(x_0))

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_shortest_path.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.scripts._figure_helpers import (
    EDIT_COLORS,
    OUTPUT_DIR,
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    dag_to_sympy_latex,
    draw_dag,
    draw_math_label,
    levenshtein_with_backtrace,
    render_token_heatmap_horizontal,
    save_figure,
    tokenize_for_display,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SOURCE_CANON = "V+VcPnc"  # cos(x_0) + x_0
TARGET_CANON = "VcVkpv+Ppc"  # cos(x_0) + 1.0
NUM_VARS = 1

# --- Global visual parameters ---
NODE_SIZE = 0.7
TOKEN_CELL_WIDTH = 11.0
TOKEN_CELL_HEIGHT = 11.0
MATH_FONTSIZE = 12


def _reconstruct_intermediates(
    source: str,
    target: str,
    operations: list,
) -> list[str]:
    """Reconstruct intermediate strings by applying edit operations one at a time.

    Args:
        source: Source string.
        target: Target string.
        operations: List of edit operations from levenshtein_with_backtrace.

    Returns:
        List of intermediate strings (including source and target).
    """
    intermediates = [source]
    current = list(source)
    offset = 0  # Track position shifts from insertions/deletions

    for op in operations:
        if op[0] == "match":
            continue
        elif op[0] == "substitute":
            _, i, j, old_char, new_char = op
            pos = i + offset
            if 0 <= pos < len(current):
                current[pos] = new_char
            intermediates.append("".join(current))
        elif op[0] == "insert":
            _, i, j, char = op
            pos = i + offset
            current.insert(pos, char)
            offset += 1
            intermediates.append("".join(current))
        elif op[0] == "delete":
            _, i, j, char = op
            pos = i + offset
            if 0 <= pos < len(current):
                current.pop(pos)
                offset -= 1
            intermediates.append("".join(current))

    return intermediates


def _try_parse_string(s: str, num_vars: int) -> tuple[bool, object | None, str]:
    """Try to parse a string as an IsalSR instruction.

    Returns:
        (is_valid, dag_or_None, latex_or_empty)
    """
    from isalsr.core.string_to_dag import StringToDAG

    try:
        s2d = StringToDAG(s, num_variables=num_vars)
        dag = s2d.run()
        if dag.node_count <= num_vars:
            return False, None, ""
        try:
            latex = dag_to_sympy_latex(dag)
        except Exception:
            latex = "?"
        return True, dag, latex
    except Exception:
        return False, None, ""


def _draw_edit_arrow(
    fig: plt.Figure,
    ax_from: plt.Axes,
    ax_to: plt.Axes,
    op_type: str,
    op_label: str,
) -> None:
    """Draw an annotated arrow between two axes showing the edit operation.

    Args:
        fig: The figure.
        ax_from: Source axes.
        ax_to: Target axes.
        op_type: Edit operation type ('substitute', 'insert', 'delete').
        op_label: Text label for the operation.
    """
    color = EDIT_COLORS.get(op_type, "#888888")

    # Create connection arrow between axes
    con = ConnectionPatch(
        xyA=(1.0, 0.5),
        xyB=(0.0, 0.5),
        coordsA="axes fraction",
        coordsB="axes fraction",
        axesA=ax_from,
        axesB=ax_to,
        arrowstyle="-|>",
        color=color,
        linewidth=1.5,
        mutation_scale=12,
    )
    fig.add_artist(con)

    # Add label in the middle
    bb_from = ax_from.get_position()
    bb_to = ax_to.get_position()
    mid_x = (bb_from.x1 + bb_to.x0) / 2
    mid_y = (bb_from.y0 + bb_from.y1) / 2

    fig.text(
        mid_x,
        mid_y + 0.02,
        op_label,
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color=color,
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "pad": 1.5,
            "edgecolor": color,
            "linewidth": 0.5,
            "boxstyle": "round,pad=0.15",
        },
    )


def generate_shortest_path_figure() -> str:
    """Generate the shortest path figure."""
    apply_ieee_style()

    # Compute Levenshtein with backtrace
    dist, ops = levenshtein_with_backtrace(SOURCE_CANON, TARGET_CANON)
    logger.info("Edit distance: %d, operations: %d", dist, len(ops))

    # Reconstruct intermediate strings
    intermediates = _reconstruct_intermediates(SOURCE_CANON, TARGET_CANON, ops)
    logger.info("Intermediate strings: %d (including endpoints)", len(intermediates))

    # Parse each intermediate
    steps = []
    for i, s in enumerate(intermediates):
        is_valid, dag, latex = _try_parse_string(s, NUM_VARS)
        steps.append(
            {
                "string": s,
                "valid": is_valid,
                "dag": dag,
                "latex": latex,
                "is_endpoint": i == 0 or i == len(intermediates) - 1,
            }
        )
        logger.info(
            "  Step %d: %r (valid=%s, latex=%s)",
            i,
            s,
            is_valid,
            latex if latex else "N/A",
        )

    # Collect edit operations (non-match)
    edit_ops = [op for op in ops if op[0] != "match"]

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    n_steps = len(steps)
    fig_width = max(10.0, 1.9 * n_steps)
    fig_height = 4.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Grid: n_steps columns for cells
    gs = GridSpec(
        3,
        n_steps,
        figure=fig,
        height_ratios=[3.0, 0.4, 0.35],
        hspace=0.08,
        wspace=0.25,
        left=0.04,
        right=0.96,
        top=0.90,
        bottom=0.05,
    )

    # Column labels at consistent y-level using fig.text
    col_width_frac = (0.96 - 0.04) / n_steps
    col_title_y = 0.96
    for col in range(n_steps):
        col_x = 0.04 + (col + 0.5) * col_width_frac
        if col == 0:
            label = "Source"
            color = PAUL_TOL_BRIGHT["green"]
        elif col == n_steps - 1:
            label = "Target"
            color = PAUL_TOL_BRIGHT["blue"]
        else:
            label = f"Step {col}"
            color = "0.3"
        fig.text(
            col_x,
            col_title_y,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    dag_axes = []
    hm_axes = []
    math_axes = []

    for col in range(n_steps):
        step = steps[col]
        ax_dag = fig.add_subplot(gs[0, col])
        ax_hm = fig.add_subplot(gs[1, col])
        ax_math = fig.add_subplot(gs[2, col])
        dag_axes.append(ax_dag)
        hm_axes.append(ax_hm)
        math_axes.append(ax_math)

        if step["valid"] and step["dag"] is not None:
            # Draw valid DAG
            tokens = tokenize_for_display(step["string"])
            draw_dag(ax_dag, step["dag"], node_size=NODE_SIZE)

            # Heatmap
            render_token_heatmap_horizontal(
                ax_hm,
                tokens,
                len(tokens) - 1,
                cell_width=TOKEN_CELL_WIDTH,
                cell_height=TOKEN_CELL_HEIGHT,
            )

            # Math expression
            if step["latex"] and step["latex"] != "?":
                draw_math_label(ax_math, step["latex"], fontsize=MATH_FONTSIZE)
            else:
                ax_math.axis("off")
        else:
            # Invalid intermediate: show placeholder
            ax_dag.text(
                0.5,
                0.5,
                "?",
                ha="center",
                va="center",
                fontsize=20,
                color="0.7",
                fontweight="bold",
                transform=ax_dag.transAxes,
            )
            ax_dag.set_facecolor("#F8F8F8")
            ax_dag.set_xticks([])
            ax_dag.set_yticks([])
            for spine in ax_dag.spines.values():
                spine.set_edgecolor("0.8")
                spine.set_linewidth(0.5)
                spine.set_linestyle("dashed")
            # Show the raw string even if invalid
            ax_hm.text(
                0.5,
                0.5,
                step["string"],
                ha="center",
                va="center",
                fontsize=5,
                fontfamily="monospace",
                color="0.5",
                transform=ax_hm.transAxes,
            )
            ax_hm.axis("off")
            ax_math.axis("off")

    # Draw edit arrows between consecutive cells
    edit_idx = 0
    for col in range(n_steps - 1):
        if edit_idx < len(edit_ops):
            op = edit_ops[edit_idx]
            op_type = op[0]
            if op_type == "substitute":
                label = f"{op[3]}->{op[4]}"
            elif op_type == "insert":
                label = f"ins '{op[3]}'"
            elif op_type == "delete":
                label = f"del '{op[3]}'"
            else:
                label = "?"

            _draw_edit_arrow(fig, dag_axes[col], dag_axes[col + 1], op_type, label)
            edit_idx += 1

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fig_shortest_path")
    saved = save_figure(fig, out_path)
    plt.close(fig)
    logger.info("Saved shortest path figure: %s", saved)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_shortest_path_figure()
