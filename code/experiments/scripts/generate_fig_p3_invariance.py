# ruff: noqa: N802, N803, N806
"""P3 (Canonical Invariance + Idempotence) figure for the IsalSR arXiv paper.

Demonstrates Property P3:
  1. Invariance:  D ≅ S2D(canonical(D))
  2. Idempotence: canonical(D) = canonical(S2D(canonical(D)))

Expression: sin(x_0) + cos(x_0)  (CONST-free for exact round-trip)

Layout (horizontal flow):
  [DAG D] --canonical--> [w** heatmap] --S2D--> [DAG D''] --canonical--> [w**' heatmap]
  Token heatmaps sit at arrow level between the DAG columns.

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_p3_invariance.py
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
    OUTPUT_DIR,
    _compute_dag_layout,
    apply_ieee_style,
    draw_dag,
    draw_math_label,
    render_token_heatmap_horizontal,
    save_figure,
    tokenize_for_display,
)
from isalsr.core.canonical import pruned_canonical_string
from isalsr.core.string_to_dag import StringToDAG

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

NUM_VARS = 1
EXPRESSION_LATEX = r"\sin(x_0) + \cos(x_0)"

# Visual parameters
FONTSIZE = 18
NODE_SIZE = 0.65
TOKEN_CELL_WIDTH = 14.0
TOKEN_CELL_HEIGHT = 0

FIG_WIDTH = 20.0
FIG_HEIGHT = 5.0

CLR_ARROW = "#555555"


def _build_expression_dag():
    """Build sin(x_0) + cos(x_0) from SymPy."""
    import sympy

    from isalsr.adapters.sympy_adapter import SympyAdapter

    x = sympy.Symbol("x_0")
    expr = sympy.sin(x) + sympy.cos(x)
    adapter = SympyAdapter()
    return adapter.from_sympy(expr, [x])


def generate_p3_figure() -> str:
    """Generate the P3 invariance + idempotence figure."""
    apply_ieee_style()

    # =========================================================================
    # Compute P3 data
    # =========================================================================
    logger.info("Building expression DAG (sin(x_0) + cos(x_0))...")
    D = _build_expression_dag()
    pos_D = _compute_dag_layout(D)

    w_star = pruned_canonical_string(D)
    logger.info("w** = canonical(D) = %r (len=%d)", w_star, len(w_star))

    s2d = StringToDAG(w_star, num_variables=NUM_VARS)
    D_prime = s2d.run()
    pos_D_prime = _compute_dag_layout(D_prime)
    logger.info("D'': %d nodes, %d edges", D_prime.node_count, D_prime.edge_count)

    w_star_prime = pruned_canonical_string(D_prime)
    logger.info("w**' = canonical(D'') = %r", w_star_prime)
    logger.info(
        "Invariance: %s, Idempotence: %s",
        w_star == pruned_canonical_string(D),
        w_star == w_star_prime,
    )

    tokens_w = tokenize_for_display(w_star)
    tokens_wp = tokenize_for_display(w_star_prime)

    # =========================================================================
    # Figure layout: 2 DAG columns with heatmaps at arrow level between them
    #
    #   [DAG D]  -->  [heatmap w**]  -->  [DAG D'']  -->  [heatmap w**']
    #
    # GridSpec: 2 rows (main + math) x 4 cols (DAG, hm, DAG, hm)
    # =========================================================================
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[4.0, 0.5],
        width_ratios=[1.3, 0.7, 1.3, 0.7],
        hspace=0.05,
        wspace=0.12,
        left=0.04,
        right=0.96,
        top=0.88,
        bottom=0.06,
    )

    # --- Column labels via fig.text at consistent y ---
    label_y = 0.87
    col_positions = []  # store (center_x) for each column
    for col_idx in range(4):
        # Compute center x from gridspec
        pos = gs[0, col_idx].get_position(fig)
        cx = (pos.x0 + pos.x1) / 2
        col_positions.append(cx)

    labels = [r"$D$", r"$w^{**}$", r"$D''$", r"$w^{**'}$"]
    for col_idx, label in enumerate(labels):
        fig.text(
            col_positions[col_idx],
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE,
            fontweight="bold",
            color="#333333",
        )

    # --- Column 0: DAG D ---
    ax_dag_D = fig.add_subplot(gs[0, 0])
    ax_math_D = fig.add_subplot(gs[1, 0])
    draw_dag(ax_dag_D, D, pos=pos_D, node_size=NODE_SIZE)
    draw_math_label(ax_math_D, EXPRESSION_LATEX, fontsize=FONTSIZE)

    # --- Column 1: Token heatmap w** (vertically centered) ---
    ax_hm_w = fig.add_subplot(gs[0, 1])
    render_token_heatmap_horizontal(
        ax_hm_w,
        tokens_w,
        len(tokens_w) - 1,
        cell_width=TOKEN_CELL_WIDTH,
        cell_height=TOKEN_CELL_HEIGHT,
    )
    ax_math_w = fig.add_subplot(gs[1, 1])
    ax_math_w.axis("off")

    # --- Column 2: DAG D'' ---
    ax_dag_Dp = fig.add_subplot(gs[0, 2])
    ax_math_Dp = fig.add_subplot(gs[1, 2])
    draw_dag(ax_dag_Dp, D_prime, pos=pos_D_prime, node_size=NODE_SIZE)
    draw_math_label(ax_math_Dp, EXPRESSION_LATEX, fontsize=FONTSIZE)

    # --- Column 3: Token heatmap w**' (vertically centered) ---
    ax_hm_wp = fig.add_subplot(gs[0, 3])
    render_token_heatmap_horizontal(
        ax_hm_wp,
        tokens_wp,
        len(tokens_wp) - 1,
        cell_width=TOKEN_CELL_WIDTH,
        cell_height=TOKEN_CELL_HEIGHT,
    )
    ax_math_wp = fig.add_subplot(gs[1, 3])
    ax_math_wp.axis("off")

    # --- Flow arrows ---
    arrow_pairs = [
        (ax_dag_D, ax_hm_w, r"canonical$(D)$"),
        (ax_hm_w, ax_dag_Dp, r"S2D$(w^{**}, m)$"),
        (ax_dag_Dp, ax_hm_wp, r"canonical$(D'')$"),
    ]
    for ax_from, ax_to, label in arrow_pairs:
        con = ConnectionPatch(
            xyA=(1.0, 0.5),
            xyB=(0.0, 0.5),
            coordsA="axes fraction",
            coordsB="axes fraction",
            axesA=ax_from,
            axesB=ax_to,
            arrowstyle="-|>",
            color=CLR_ARROW,
            linewidth=2.0,
            mutation_scale=16,
        )
        fig.add_artist(con)

        bb_from = ax_from.get_position()
        bb_to = ax_to.get_position()
        mid_x = (bb_from.x1 + bb_to.x0) / 2
        mid_y = (bb_from.y0 + bb_from.y1) / 2

        fig.text(
            mid_x,
            mid_y + 0.06,
            label,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 4,
            fontstyle="italic",
            color=CLR_ARROW,
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "pad": 2.5,
                "edgecolor": CLR_ARROW,
                "linewidth": 0.5,
                "boxstyle": "round,pad=0.2",
            },
        )

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fig_p3_invariance")
    saved = save_figure(fig, out_path)
    plt.close(fig)
    logger.info("Saved P3 invariance figure: %s", saved)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_p3_figure()
