# ruff: noqa: N803, N806
"""Algorithm overview figure for the IsalSR paper.

Generates a horizontal figure showing S2D (String-to-DAG) and D2S (DAG-to-String)
execution steps for sin(x_0)*x_1 + cos(x_0). Adapted from IsalGraph's algorithm_figures.py
for directed labeled DAGs.

Layout:
    - Top row: S2D execution (5 snapshots, left to right)
    - Bottom row: D2S execution (5 snapshots, left to right)
    Each snapshot column shows: CDLL ring, instruction heatmap, DAG

All visual parameters are defined in the PARAMS dict at the top of this file.
To tweak the figure, edit any value in PARAMS and re-run:
    cd /home/mpascual/research/code/IsalSR && \
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_algorithm_overview.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.plotting_styles import (
    TOKEN_COLORS,
    apply_ieee_style,
    save_figure,
    tokenize_for_display,
)
from isalsr.core.canonical import canonical_string
from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

logger = logging.getLogger(__name__)

# =============================================================================
# FIGURE PARAMETERS — edit this dict to tweak the figure, then re-run.
# =============================================================================

PARAMS: dict = {
    # --- Output ---
    "output_dir": "/media/mpascual/Sandisk2TB/research/isalsr/results/figures",
    # --- Snapshots ---
    "n_snapshots": 5,  # number of evenly-spaced snapshots per algorithm
    # --- Combined figure (S2D + D2S) ---
    "combined_col_width": 2.8,  # inches per snapshot column
    "combined_extra_width": 0.5,  # extra horizontal inches
    "combined_height": 10.0,  # total height (inches)
    "combined_hspace": 0.30,  # vertical gap between S2D and D2S groups
    "combined_top": 0.92,
    "combined_bottom": 0.05,
    "combined_left": 0.08,  # left margin (room for rotated label)
    "combined_right": 0.97,
    # --- Individual figures (S2D-only / D2S-only) ---
    "single_col_width": 2.8,  # inches per snapshot column
    "single_extra_width": 0.5,
    "single_height": 5.5,  # total height (inches)
    "single_left": 0.03,  # narrower left margin (no rotated label)
    # --- Row height ratios [CDLL ring, token heatmap, DAG] ---
    "row_ratios": [1.0, 0.20, 2.0],  # DAG gets ~63% of the vertical space
    "row_hspace": 0.10,  # vertical gap between rows
    "row_wspace": 0.20,  # horizontal gap between columns
    # --- CDLL ring ---
    "cdll_ring_radius": 2.3,  # radius of the ring circle
    "cdll_node_radius": 0.85,  # radius of each node on the ring
    "cdll_node_fontsize": 12,  # font inside CDLL nodes
    "cdll_edge_width": 0.6,  # line width of ring connections
    "cdll_edge_color": "0.6",
    # --- Pointers (pi / sigma) ---
    "pointer_arrow_offset": 0.9,  # how far outside the ring the arrow starts
    "pointer_label_offset": 0.7,  # how far outside the ring the label sits
    "pointer_arrow_width": 1.4,  # arrow line width
    "pointer_arrow_scale": 10,  # arrowhead mutation scale
    "pointer_fontsize": 12,  # font for pi/sigma labels
    # --- DAG nodes ---
    "dag_node_radius": 0.5,  # radius of DAG node circles
    "dag_node_fontsize": 12,  # font inside DAG nodes
    "dag_node_linewidth": 0.6,  # circle border width
    # --- DAG layout (Sugiyama) ---
    "dag_layer_x_spacing": 4,  # horizontal distance between nodes in same layer
    "dag_layer_y_spacing": 2,  # vertical distance between layers
    "dag_barycentric_sweeps": 4,  # number of barycentric ordering sweeps
    "dag_axis_padding": 0.3,  # extra padding around DAG content
    # --- DAG edges ---
    "dag_edge_width_solid": 1.0,
    "dag_edge_width_ghost": 0.4,
    "dag_edge_color_solid": "0.25",
    "dag_edge_color_ghost": "0.82",
    "dag_edge_arrow_scale": 12,
    # --- Token heatmap ---
    "token_cell_width": 12.4,
    "token_cell_height": 11.3,
    "token_fontsize": 9,  # font inside token cells
    # --- Step titles ---
    "step_title_fontsize": 12,
    "step_title_pad": 10,
    # --- Legend ---
    "legend_fontsize": 12,
    # --- Group box labels (rotated, combined figure only) ---
    "group_label_fontsize": 10,
    "group_box_alpha": 0.06,
    "group_box_linewidth": 1.0,
    "group_box_pad": 0.008,
    # --- Colors (Paul Tol colorblind-safe) ---
    "color_primary": "#EE6677",  # primary pointer (red)
    "color_secondary": "#4477AA",  # secondary pointer (blue)
    "color_new_node": "#CCBB44",  # newly inserted node (yellow)
    "color_default_node": "#DDDDDD",  # fallback node color
    "color_s2d_group": "#EE6677",  # S2D group box tint
    "color_d2s_group": "#4477AA",  # D2S group box tint
    "color_s2d_label": "#CC3355",  # S2D rotated label
    "color_d2s_label": "#335588",  # D2S rotated label
}

# Node type colors for DAG visualization
_NODE_TYPE_COLORS: dict[NodeType, str] = {
    NodeType.VAR: "#4477AA",  # blue
    NodeType.ADD: "#228833",  # green
    NodeType.MUL: "#44AA66",  # light green
    NodeType.SUB: "#EE7733",  # orange
    NodeType.DIV: "#DDAA33",  # amber
    NodeType.SIN: "#AA3377",  # magenta
    NodeType.COS: "#CC6699",  # pink
    NodeType.EXP: "#6644AA",  # purple
    NodeType.LOG: "#8866BB",  # lavender
    NodeType.SQRT: "#9977CC",  # light purple
    NodeType.POW: "#AA7744",  # brown
    NodeType.ABS: "#BB88DD",  # light magenta
    NodeType.NEG: "#CC3355",  # dark rose
    NodeType.INV: "#66BBAA",  # teal
    NodeType.CONST: "#CCBB44",  # gold  # noqa: E261
}

# Convenience aliases (read from PARAMS for consistency)
PRIMARY_COLOR = PARAMS["color_primary"]
SECONDARY_COLOR = PARAMS["color_secondary"]
NEW_NODE_COLOR = PARAMS["color_new_node"]
_DEFAULT_NODE_COLOR = PARAMS["color_default_node"]


# =============================================================================
# Node label rendering
# =============================================================================


def node_display_label(dag: LabeledDAG, node_id: int) -> str:
    """Return a display label string for a DAG node.

    Uses LaTeX math notation for operators and variable subscripts.

    Args:
        dag: The DAG containing the node.
        node_id: The node index.

    Returns:
        Display-ready string.
    """
    label = dag.node_label(node_id)
    data = dag.node_data(node_id)
    if label == NodeType.VAR:
        return f"$x_{{{data.get('var_index', '?')}}}$"
    if label == NodeType.ADD:
        return "$+$"
    if label == NodeType.MUL:
        return r"$\times$"
    if label == NodeType.SUB:
        return "$-$"
    if label == NodeType.DIV:
        return r"$\div$"
    if label == NodeType.SIN:
        return "sin"
    if label == NodeType.COS:
        return "cos"
    if label == NodeType.EXP:
        return "exp"
    if label == NodeType.LOG:
        return "log"
    if label == NodeType.SQRT:
        return r"$\sqrt{}$"
    if label == NodeType.POW:
        return r"$\wedge$"
    if label == NodeType.ABS:
        return r"$|\cdot|$"
    if label == NodeType.NEG:
        return "neg"
    if label == NodeType.INV:
        return "inv"
    if label == NodeType.CONST:
        return "$c$"
    return "?"


# =============================================================================
# CDLL extraction
# =============================================================================


def extract_cdll_order(
    cdll: CircularDoublyLinkedList,
    dag_snapshot: LabeledDAG,
    start_ptr: int,
) -> list[tuple[int, int, NodeType | None]]:
    """Extract ordered list of (cdll_idx, graph_node, label) from CDLL traversal.

    Args:
        cdll: The CDLL instance.
        dag_snapshot: The DAG snapshot for label lookups.
        start_ptr: CDLL node index to start traversal from.

    Returns:
        List of (cdll_index, graph_node_id, node_label_or_None).
    """
    if cdll.size() == 0:
        return []
    order: list[tuple[int, int, NodeType | None]] = []
    current = start_ptr
    for _ in range(cdll.size()):
        graph_node = cdll.get_value(current)
        label: NodeType | None = None
        if graph_node < dag_snapshot.node_count:
            label = dag_snapshot.node_label(graph_node)
        order.append((current, graph_node, label))
        current = cdll._next[current]
    return order


# =============================================================================
# Snapshot selection
# =============================================================================


def pick_snapshots(trace: list, n: int = 5) -> list[int]:
    """Select n evenly-spaced snapshot indices (always includes first and last).

    Args:
        trace: The trace list.
        n: Number of snapshots to pick.

    Returns:
        List of indices into the trace.
    """
    total = len(trace)
    if total <= n:
        return list(range(total))
    indices = [0]
    step = (total - 1) / (n - 1)
    for i in range(1, n - 1):
        indices.append(round(i * step))
    indices.append(total - 1)
    return indices


# =============================================================================
# CDLL ring drawing (adapted for labeled DAGs)
# =============================================================================


def draw_cdll_ring(
    ax: plt.Axes,
    cdll_order: list[tuple[int, int, NodeType | None]],
    primary_cdll_idx: int,
    secondary_cdll_idx: int,
    dag: LabeledDAG,
    *,
    new_graph_node: int | None = None,
    radius: float | None = None,
    node_radius: float | None = None,
) -> None:
    """Draw CDLL as a circular ring with labeled nodes and pointer arrows.

    Unlike IsalGraph's version, nodes display operation labels instead of
    plain integer IDs. Node fill colors correspond to operation type.

    Args:
        ax: Matplotlib axes.
        cdll_order: CDLL traversal from extract_cdll_order.
        primary_cdll_idx: CDLL node index of primary pointer.
        secondary_cdll_idx: CDLL node index of secondary pointer.
        dag: DAG snapshot for label lookups.
        new_graph_node: If set, highlight this graph node in yellow.
        radius: Ring radius. If None, reads from PARAMS.
        node_radius: Individual node circle radius. If None, reads from PARAMS.
    """
    if radius is None:
        radius = PARAMS["cdll_ring_radius"]
    if node_radius is None:
        node_radius = PARAMS["cdll_node_radius"]

    n = len(cdll_order)
    if n == 0:
        ax.axis("off")
        return

    # Map cdll indices to positions in the order list
    cdll_idx_to_pos = {entry[0]: i for i, entry in enumerate(cdll_order)}
    pri_pos = cdll_idx_to_pos.get(primary_cdll_idx, 0)
    sec_pos = cdll_idx_to_pos.get(secondary_cdll_idx, 0)

    # Compute node positions (evenly spaced on circle, starting from top)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    # Draw edges (lines between consecutive nodes)
    for i in range(n):
        j = (i + 1) % n
        ax.annotate(
            "",
            xy=positions[j],
            xytext=positions[i],
            arrowprops={
                "arrowstyle": "-",
                "color": PARAMS["cdll_edge_color"],
                "linewidth": PARAMS["cdll_edge_width"],
            },
        )

    # Draw nodes with operation labels
    for i, (_cdll_idx, graph_node, label) in enumerate(cdll_order):
        x, y = positions[i]

        # Determine fill color
        if new_graph_node is not None and graph_node == new_graph_node:
            color = NEW_NODE_COLOR
        elif i == pri_pos and i == sec_pos:
            color = "#AA77BB"  # Blend for overlap
        elif i == pri_pos:
            color = PRIMARY_COLOR
        elif i == sec_pos:
            color = SECONDARY_COLOR
        elif label is not None:
            color = _NODE_TYPE_COLORS.get(label, _DEFAULT_NODE_COLOR)
        else:
            color = _DEFAULT_NODE_COLOR

        circle = plt.Circle(
            (x, y),
            node_radius,
            facecolor=color,
            edgecolor="0.3",
            linewidth=0.6,
            zorder=3,
        )
        ax.add_patch(circle)

        # Show operation label inside node
        display_label = node_display_label(dag, graph_node) if graph_node < dag.node_count else "?"
        ax.text(
            x,
            y,
            display_label,
            ha="center",
            va="center",
            fontsize=PARAMS["cdll_node_fontsize"],
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # Draw pointer arrows from outside the ring
    arrow_radius = radius + PARAMS["pointer_arrow_offset"]
    _draw_pointer_arrow(
        ax,
        positions[pri_pos],
        angles[pri_pos],
        arrow_radius,
        node_radius,
        PRIMARY_COLOR,
        r"$\pi$",
    )

    if sec_pos != pri_pos:
        _draw_pointer_arrow(
            ax,
            positions[sec_pos],
            angles[sec_pos],
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            r"$\sigma$",
        )
    else:
        # Overlap: offset secondary arrow slightly
        offset_angle = angles[pri_pos] + 0.35
        _draw_pointer_arrow(
            ax,
            positions[sec_pos],
            offset_angle,
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            r"$\sigma$",
        )

    margin = arrow_radius + 0.45
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_pointer_arrow(
    ax: plt.Axes,
    target_pos: tuple[float, float],
    angle: float,
    arrow_radius: float,
    node_radius: float,
    color: str,
    label: str,
) -> None:
    """Draw a labeled pointer arrow from outside the ring toward a node.

    The arrow tip stops at the circle edge (not the center) by computing
    the intersection point in data coordinates.
    """
    start_x = arrow_radius * np.cos(angle)
    start_y = arrow_radius * np.sin(angle)

    # Compute arrow endpoint at the circle edge (not the center)
    dx = target_pos[0] - start_x
    dy = target_pos[1] - start_y
    dist = np.sqrt(dx**2 + dy**2)
    if dist > 1e-6:
        edge_x = target_pos[0] - (dx / dist) * node_radius
        edge_y = target_pos[1] - (dy / dist) * node_radius
    else:
        edge_x, edge_y = target_pos

    ax.annotate(
        "",
        xy=(edge_x, edge_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": PARAMS["pointer_arrow_width"],
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": PARAMS["pointer_arrow_scale"],
        },
    )
    label_x = (arrow_radius + PARAMS["pointer_label_offset"]) * np.cos(angle)
    label_y = (arrow_radius + PARAMS["pointer_label_offset"]) * np.sin(angle)
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=PARAMS["pointer_fontsize"],
        fontweight="bold",
        color=color,
    )


# =============================================================================
# DAG drawing (topological layout for directed labeled graph)
# =============================================================================


def _compute_dag_layout(dag: LabeledDAG) -> dict[int, tuple[float, float]]:
    """Compute a layered layout for a DAG with barycentric edge-crossing reduction.

    Uses a Sugiyama-style layered approach:
    1. Assign layers by longest path from sources (VAR nodes at bottom).
    2. Order nodes within each layer by the **barycenter** (mean x-position
       of parent nodes), which minimizes edge crossings.
    3. Apply 2 top-down + 2 bottom-up ordering sweeps for refinement.

    Args:
        dag: The DAG to lay out.

    Returns:
        Dictionary mapping node ID to (x, y) position.
    """
    if dag.node_count == 0:
        return {}

    n = dag.node_count
    layer: list[int] = [0] * n
    order = dag.topological_sort()
    for node in order:
        for neighbor in dag.out_neighbors(node):
            if neighbor < n:
                layer[neighbor] = max(layer[neighbor], layer[node] + 1)

    max_layer = max(layer) if layer else 0
    layers: dict[int, list[int]] = {lv: [] for lv in range(max_layer + 1)}
    for node in range(n):
        layers[layer[node]].append(node)

    # Initial ordering: sort by node ID within each layer (stable baseline).
    for lv in layers:
        layers[lv] = sorted(layers[lv])

    # Assign initial x-positions (evenly spaced).
    pos_x: dict[int, float] = {}
    for lv, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos_x[node] = float(i)

    # Barycentric ordering sweeps to minimize edge crossings.
    for _sweep in range(PARAMS["dag_barycentric_sweeps"]):
        # Top-down: order layer lv by mean x of parents (in layer lv-1).
        for lv in range(1, max_layer + 1):
            bary: dict[int, float] = {}
            for node in layers[lv]:
                parents = [p for p in dag.in_neighbors(node) if layer[p] == lv - 1]
                if parents:
                    bary[node] = sum(pos_x[p] for p in parents) / len(parents)
                else:
                    bary[node] = pos_x[node]
            layers[lv] = sorted(layers[lv], key=lambda nd: bary[nd])
            for i, node in enumerate(layers[lv]):
                pos_x[node] = float(i)

        # Bottom-up: order layer lv by mean x of children (in layer lv+1).
        for lv in range(max_layer - 1, -1, -1):
            bary = {}
            for node in layers[lv]:
                children = [c for c in dag.out_neighbors(node) if layer[c] == lv + 1]
                if children:
                    bary[node] = sum(pos_x[c] for c in children) / len(children)
                else:
                    bary[node] = pos_x[node]
            layers[lv] = sorted(layers[lv], key=lambda nd: bary[nd])
            for i, node in enumerate(layers[lv]):
                pos_x[node] = float(i)

    # Final positions: center each layer horizontally.
    # Spacing tuned so nodes (radius 0.30) don't overlap.
    pos: dict[int, tuple[float, float]] = {}
    for lv, nodes in layers.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * PARAMS["dag_layer_x_spacing"]
            y = lv * PARAMS["dag_layer_y_spacing"]
            pos[node] = (x, y)

    return pos


def draw_dag(
    ax: plt.Axes,
    dag: LabeledDAG,
    *,
    pos: dict[int, tuple[float, float]] | None = None,
    ghost_nodes: set[int] | None = None,
    ghost_edges: set[tuple[int, int]] | None = None,
    node_size: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Draw a labeled DAG with optional ghost rendering for D2S.

    Solid nodes/edges are already-encoded parts. Ghost (dashed) nodes/edges
    show parts not yet encoded.

    Args:
        ax: Matplotlib axes.
        dag: The full DAG to draw.
        pos: Pre-computed layout positions. If None, computed automatically.
        ghost_nodes: Set of node IDs to render as ghosts (not yet encoded).
        ghost_edges: Set of (src, tgt) edges to render as ghosts.
        node_size: Radius of node circles.

    Returns:
        Layout position dict.
    """
    if node_size is None:
        node_size = PARAMS["dag_node_radius"]

    n = dag.node_count
    if n == 0:
        ax.axis("off")
        return {}

    if pos is None:
        pos = _compute_dag_layout(dag)

    if ghost_nodes is None:
        ghost_nodes = set()
    if ghost_edges is None:
        ghost_edges = set()

    # Collect all edges
    all_edges: list[tuple[int, int]] = []
    for src in range(n):
        for tgt in dag.out_neighbors(src):
            all_edges.append((src, tgt))

    # Draw edges first (behind nodes)
    for src, tgt in all_edges:
        if src not in pos or tgt not in pos:
            continue
        is_ghost = (src, tgt) in ghost_edges
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        ux, uy = dx / dist, dy / dist
        sx, sy = x0 + ux * node_size, y0 + uy * node_size
        ex, ey = x1 - ux * node_size, y1 - uy * node_size

        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": PARAMS["dag_edge_color_ghost"]
                if is_ghost
                else PARAMS["dag_edge_color_solid"],
                "linewidth": PARAMS["dag_edge_width_ghost"]
                if is_ghost
                else PARAMS["dag_edge_width_solid"],
                "linestyle": (0, (3, 3)) if is_ghost else "solid",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": PARAMS["dag_edge_arrow_scale"],
            },
            zorder=1,
        )

    # Draw nodes
    for node_id in range(n):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        is_ghost = node_id in ghost_nodes
        label = dag.node_label(node_id)

        if is_ghost:
            facecolor = "white"
            edgecolor = "0.7"
            linestyle = "dashed"
            text_color = "0.5"
        else:
            facecolor = _NODE_TYPE_COLORS.get(label, _DEFAULT_NODE_COLOR)
            edgecolor = "0.3"
            linestyle = "solid"
            text_color = "white"

        circle = plt.Circle(
            (x, y),
            node_size,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=PARAMS["dag_node_linewidth"],
            linestyle=linestyle,
            zorder=3,
        )
        ax.add_patch(circle)

        display_label = node_display_label(dag, node_id)
        ax.text(
            x,
            y,
            display_label,
            ha="center",
            va="center",
            fontsize=PARAMS["dag_node_fontsize"],
            fontweight="bold",
            color=text_color,
            zorder=4,
        )

    # Set axis limits with tight padding around actual content.
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad = node_size + PARAMS["dag_axis_padding"]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 + pad, node_size + 0.5)
        half_h = max((max(ys) - min(ys)) / 2 + pad, node_size + 0.5)
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)

    ax.set_aspect("equal")
    ax.axis("off")
    return pos


# =============================================================================
# Instruction heatmap (horizontal colored token strip)
# =============================================================================


def render_token_heatmap_horizontal(
    ax: plt.Axes,
    tokens: list[str],
    current_token_idx: int,
    *,
    cell_width: float | None = None,
    cell_height: float | None = None,
) -> None:
    """Render instruction tokens as a horizontal row of colored cells.

    Processed tokens have full opacity; remaining tokens are dimmed.

    Args:
        ax: Matplotlib axes.
        tokens: List of tokens (single-char or two-char).
        current_token_idx: Index of last completed token (-1 for none).
        cell_width: Width of each cell.
        cell_height: Height of each cell.
    """
    if cell_width is None:
        cell_width = PARAMS["token_cell_width"]
    if cell_height is None:
        cell_height = PARAMS["token_cell_height"]

    n = len(tokens)
    if n == 0:
        ax.axis("off")
        return

    for i, token in enumerate(tokens):
        color = TOKEN_COLORS.get(token, "#000000")
        completed = i <= current_token_idx
        alpha = 1.0 if completed else 0.15
        x = i * cell_width

        rect = mpatches.FancyBboxPatch(
            (x, -cell_height / 2),
            cell_width * 0.85,
            cell_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.4" if completed else "0.8",
            linewidth=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_width * 0.425,
            0,
            token,
            ha="center",
            va="center",
            fontsize=PARAMS["token_fontsize"],
            fontfamily="monospace",
            fontweight="bold" if i == current_token_idx else "normal",
            color="white" if completed else "0.6",
        )

    ax.set_xlim(-cell_width * 0.2, n * cell_width + cell_width * 0.1)
    ax.set_ylim(-cell_height, cell_height)
    ax.set_aspect("auto")
    ax.axis("off")


# =============================================================================
# D2S ghost state computation
# =============================================================================


def compute_d2s_ghost_state(
    full_dag: LabeledDAG,
    output_dag: LabeledDAG,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Determine which nodes and edges in the full DAG are ghosts (not yet encoded).

    The D2S output_dag contains the nodes and edges encoded so far. Nodes in
    the full_dag that have no corresponding node in the output_dag are ghosts.

    Note: This comparison uses raw node IDs, which is correct when the D2S
    operates on a canonical-derived DAG (where the i2o mapping is identity
    because S2D and D2S traverse nodes in the same order). For DAGs with
    non-identity i2o mappings, a mapping-aware version would be needed.

    Args:
        full_dag: The complete target DAG.
        output_dag: The partial DAG built by D2S so far.

    Returns:
        (ghost_nodes, ghost_edges) -- sets of IDs/edge tuples that are not yet encoded.
    """
    ghost_nodes: set[int] = set()
    for i in range(full_dag.node_count):
        if i >= output_dag.node_count:
            ghost_nodes.add(i)

    ghost_edges: set[tuple[int, int]] = set()
    for src in range(full_dag.node_count):
        for tgt in full_dag.out_neighbors(src):
            if (
                src >= output_dag.node_count
                or tgt >= output_dag.node_count
                or not output_dag.has_edge(src, tgt)
            ):
                ghost_edges.add((src, tgt))

    return ghost_nodes, ghost_edges


# =============================================================================
# Main figure generation
# =============================================================================


def _generate_traces(
    canonical_str: str,
    num_variables: int,
):
    """Generate S2D and D2S traces for a canonical string.

    Returns:
        (s2d_dag, s2d_trace, s2d_tokens, d2s_str, d2s_trace, d2s_tokens,
         s2d_dag_pos, d2s_target_dag, d2s_target_pos)
    """
    s2d = StringToDAG(canonical_str, num_variables=num_variables)
    s2d_dag = s2d.run(trace=True)
    s2d_trace = s2d._trace_log

    d2s = DAGToString(s2d_dag, initial_node=0)
    d2s_str = d2s.run(trace=True)
    d2s_trace = d2s._trace_log

    logger.info("Canonical string: %r (len=%d)", canonical_str, len(canonical_str))
    logger.info("D2S string: %r (len=%d)", d2s_str, len(d2s_str))
    logger.info("S2D trace: %d snapshots, D2S trace: %d snapshots", len(s2d_trace), len(d2s_trace))

    s2d_tokens = tokenize_for_display(canonical_str)
    d2s_tokens = tokenize_for_display(d2s_str)

    d2s_target_dag = s2d_dag
    d2s_target_pos = _compute_dag_layout(d2s_target_dag)
    s2d_final_dag = s2d_trace[-1][0]
    s2d_dag_pos = _compute_dag_layout(s2d_final_dag)

    return (
        s2d_dag,
        s2d_trace,
        s2d_tokens,
        d2s_str,
        d2s_trace,
        d2s_tokens,
        s2d_dag_pos,
        d2s_target_dag,
        d2s_target_pos,
    )


def _render_s2d_snapshots(fig, gs, s2d_trace, s2d_tokens, s2d_dag_pos, n_cols, s2d_indices):
    """Render S2D snapshots into a gridspec. Returns axes lists."""
    cdll_axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    instr_axes = [fig.add_subplot(gs[1, c]) for c in range(n_cols)]
    graph_axes = [fig.add_subplot(gs[2, c]) for c in range(n_cols)]

    for col, snap_idx in enumerate(s2d_indices):
        dag_snap, cdll_snap, pri, sec, tokens_so_far = s2d_trace[snap_idx]

        cdll_order = extract_cdll_order(cdll_snap, dag_snap, pri)
        new_node = None
        if snap_idx > 0:
            prev_dag = s2d_trace[snap_idx - 1][0]
            if dag_snap.node_count > prev_dag.node_count:
                new_node = dag_snap.node_count - 1

        draw_cdll_ring(
            cdll_axes[col],
            cdll_order,
            pri,
            sec,
            dag_snap,
            new_graph_node=new_node,
        )
        step_label = "Init" if snap_idx == 0 else f"Step {snap_idx}"
        cdll_axes[col].set_title(
            step_label,
            fontsize=PARAMS["step_title_fontsize"],
            fontweight="bold",
            pad=PARAMS["step_title_pad"],
        )

        current_token_idx = len(tokens_so_far) - 1
        render_token_heatmap_horizontal(instr_axes[col], s2d_tokens, current_token_idx)

        draw_dag(
            graph_axes[col],
            dag_snap,
            pos={k: v for k, v in s2d_dag_pos.items() if k < dag_snap.node_count},
        )

    for col in range(len(s2d_indices), n_cols):
        cdll_axes[col].axis("off")
        instr_axes[col].axis("off")
        graph_axes[col].axis("off")

    return cdll_axes, instr_axes, graph_axes


def _render_d2s_snapshots(
    fig,
    gs,
    d2s_trace,
    d2s_tokens,
    d2s_target_dag,
    d2s_target_pos,
    n_cols,
    d2s_indices,
):
    """Render D2S snapshots into a gridspec. Returns axes lists."""
    cdll_axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    instr_axes = [fig.add_subplot(gs[1, c]) for c in range(n_cols)]
    graph_axes = [fig.add_subplot(gs[2, c]) for c in range(n_cols)]

    for col, snap_idx in enumerate(d2s_indices):
        output_dag_snap, cdll_snap, pri, sec, prefix = d2s_trace[snap_idx]

        cdll_order = extract_cdll_order(cdll_snap, output_dag_snap, pri)
        new_node = None
        if snap_idx > 0:
            prev_dag = d2s_trace[snap_idx - 1][0]
            if output_dag_snap.node_count > prev_dag.node_count:
                new_node = output_dag_snap.node_count - 1

        draw_cdll_ring(
            cdll_axes[col],
            cdll_order,
            pri,
            sec,
            output_dag_snap,
            new_graph_node=new_node,
        )
        step_label = "Init" if snap_idx == 0 else f"Step {snap_idx}"
        cdll_axes[col].set_title(
            step_label,
            fontsize=PARAMS["step_title_fontsize"],
            fontweight="bold",
            pad=PARAMS["step_title_pad"],
        )

        d2s_tokens_so_far = tokenize_for_display(prefix)
        current_token_idx = len(d2s_tokens_so_far) - 1
        render_token_heatmap_horizontal(instr_axes[col], d2s_tokens, current_token_idx)

        ghost_nodes, ghost_edges = compute_d2s_ghost_state(d2s_target_dag, output_dag_snap)
        draw_dag(
            graph_axes[col],
            d2s_target_dag,
            pos=d2s_target_pos,
            ghost_nodes=ghost_nodes,
            ghost_edges=ghost_edges,
        )

    for col in range(len(d2s_indices), n_cols):
        cdll_axes[col].axis("off")
        instr_axes[col].axis("off")
        graph_axes[col].axis("off")

    return cdll_axes, instr_axes, graph_axes


def _add_legend(fig):
    """Add the shared pointer legend to a figure."""
    legend_handles = [
        mpatches.Patch(facecolor=PRIMARY_COLOR, label=r"$\pi$ (primary)"),
        mpatches.Patch(facecolor=SECONDARY_COLOR, label=r"$\sigma$ (secondary)"),
        mpatches.Patch(facecolor=NEW_NODE_COLOR, label="New node"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=PARAMS["legend_fontsize"],
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )


def _add_single_group_box(fig, axes_rows, color, label, *, show_label=True):
    """Add a colored background box and optional rotated label around a single group."""
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    bboxes = []
    for row in axes_rows:
        for ax in row:
            bb = ax.get_tightbbox(renderer)
            if bb is not None:
                bboxes.append(bb.transformed(fig.transFigure.inverted()))
    if not bboxes:
        return
    bb = Bbox.union(bboxes)

    pad = PARAMS["group_box_pad"]
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (bb.x0 - pad, bb.y0 - pad),
            bb.width + 2 * pad,
            bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor=color,
            alpha=PARAMS["group_box_alpha"],
            edgecolor=color,
            linewidth=PARAMS["group_box_linewidth"],
            transform=fig.transFigure,
            zorder=0,
        )
    )
    if show_label:
        fig.text(
            bb.x0 - pad - 0.025,
            bb.y0 + bb.height / 2,
            label,
            ha="center",
            va="center",
            fontsize=PARAMS["group_label_fontsize"],
            fontweight="bold",
            color=color,
            transform=fig.transFigure,
            rotation=90,
        )


def generate_s2d_figure(
    canonical_str: str,
    num_variables: int,
    output_dir: str,
    s2d_trace=None,
    s2d_tokens=None,
    s2d_dag_pos=None,
) -> str:
    """Generate a standalone S2D (String-to-DAG) figure.

    Args:
        canonical_str: The canonical string for the expression.
        num_variables: Number of input variables.
        output_dir: Directory to save figures.
        s2d_trace: Pre-computed trace (optional, recomputed if None).
        s2d_tokens: Pre-computed tokens (optional).
        s2d_dag_pos: Pre-computed layout (optional).

    Returns:
        Base path of saved figure (without extension).
    """
    if s2d_trace is None:
        s2d = StringToDAG(canonical_str, num_variables=num_variables)
        s2d.run(trace=True)
        s2d_trace = s2d._trace_log
        s2d_tokens = tokenize_for_display(canonical_str)
        s2d_dag_pos = _compute_dag_layout(s2d_trace[-1][0])

    s2d_indices = pick_snapshots(s2d_trace, PARAMS["n_snapshots"])
    n_cols = len(s2d_indices)

    fig = plt.figure(
        figsize=(
            PARAMS["single_col_width"] * n_cols + PARAMS["single_extra_width"],
            PARAMS["single_height"],
        )
    )
    gs = GridSpec(
        3,
        n_cols,
        figure=fig,
        height_ratios=PARAMS["row_ratios"],
        hspace=PARAMS["row_hspace"],
        wspace=PARAMS["row_wspace"],
        top=0.90,
        bottom=0.08,
        left=0.03,
        right=0.97,
    )

    cdll_ax, instr_ax, graph_ax = _render_s2d_snapshots(
        fig,
        gs,
        s2d_trace,
        s2d_tokens,
        s2d_dag_pos,
        n_cols,
        s2d_indices,
    )

    _add_single_group_box(
        fig,
        [cdll_ax, instr_ax, graph_ax],
        "#EE6677",
        "String-to-DAG (S2D)",
        show_label=False,
    )
    _add_legend(fig)

    output_path = os.path.join(output_dir, "fig_s2d")
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("S2D figure saved: %s", output_path)
    return output_path


def generate_d2s_figure(
    canonical_str: str,
    num_variables: int,
    output_dir: str,
    d2s_trace=None,
    d2s_tokens=None,
    d2s_target_dag=None,
    d2s_target_pos=None,
) -> str:
    """Generate a standalone D2S (DAG-to-String) figure.

    Args:
        canonical_str: The canonical string for the expression.
        num_variables: Number of input variables.
        output_dir: Directory to save figures.
        d2s_trace: Pre-computed trace (optional, recomputed if None).
        d2s_tokens: Pre-computed tokens (optional).
        d2s_target_dag: Pre-computed target DAG (optional).
        d2s_target_pos: Pre-computed target layout (optional).

    Returns:
        Base path of saved figure (without extension).
    """
    if d2s_trace is None:
        s2d = StringToDAG(canonical_str, num_variables=num_variables)
        s2d_dag = s2d.run(trace=True)
        d2s = DAGToString(s2d_dag, initial_node=0)
        d2s_str = d2s.run(trace=True)
        d2s_trace = d2s._trace_log
        d2s_tokens = tokenize_for_display(d2s_str)
        d2s_target_dag = s2d_dag
        d2s_target_pos = _compute_dag_layout(d2s_target_dag)

    d2s_indices = pick_snapshots(d2s_trace, PARAMS["n_snapshots"])
    n_cols = len(d2s_indices)

    fig = plt.figure(
        figsize=(
            PARAMS["single_col_width"] * n_cols + PARAMS["single_extra_width"],
            PARAMS["single_height"],
        )
    )
    gs = GridSpec(
        3,
        n_cols,
        figure=fig,
        height_ratios=PARAMS["row_ratios"],
        hspace=PARAMS["row_hspace"],
        wspace=PARAMS["row_wspace"],
        top=0.90,
        bottom=0.08,
        left=0.03,
        right=0.97,
    )

    cdll_ax, instr_ax, graph_ax = _render_d2s_snapshots(
        fig,
        gs,
        d2s_trace,
        d2s_tokens,
        d2s_target_dag,
        d2s_target_pos,
        n_cols,
        d2s_indices,
    )

    _add_single_group_box(
        fig,
        [cdll_ax, instr_ax, graph_ax],
        "#4477AA",
        "DAG-to-String (D2S)",
        show_label=False,
    )
    _add_legend(fig)

    output_path = os.path.join(output_dir, "fig_d2s")
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("D2S figure saved: %s", output_path)
    return output_path


def generate_algorithm_overview(
    canonical_str: str,
    full_dag: LabeledDAG,
    num_variables: int,
    output_dir: str,
) -> str:
    """Generate the combined algorithm overview figure with S2D and D2S traces.

    Also generates individual S2D-only and D2S-only figures.

    Args:
        canonical_str: The canonical string for the expression.
        full_dag: The complete expression DAG.
        num_variables: Number of input variables.
        output_dir: Directory to save figures.

    Returns:
        Base path of the combined figure (without extension).
    """
    # --- Generate traces (shared across all three figures) ---
    (
        s2d_dag,
        s2d_trace,
        s2d_tokens,
        d2s_str,
        d2s_trace,
        d2s_tokens,
        s2d_dag_pos,
        d2s_target_dag,
        d2s_target_pos,
    ) = _generate_traces(canonical_str, num_variables)

    s2d_indices = pick_snapshots(s2d_trace, PARAMS["n_snapshots"])
    d2s_indices = pick_snapshots(d2s_trace, PARAMS["n_snapshots"])
    n_cols = max(len(s2d_indices), len(d2s_indices))

    logger.info("S2D snapshot indices: %s (of %d)", s2d_indices, len(s2d_trace))
    logger.info("D2S snapshot indices: %s (of %d)", d2s_indices, len(d2s_trace))

    # === Combined figure (S2D top, D2S bottom) ===
    fig = plt.figure(
        figsize=(
            PARAMS["combined_col_width"] * n_cols + PARAMS["combined_extra_width"],
            PARAMS["combined_height"],
        )
    )

    outer_gs = GridSpec(
        2,
        1,
        figure=fig,
        hspace=PARAMS["combined_hspace"],
        top=PARAMS["combined_top"],
        bottom=PARAMS["combined_bottom"],
        left=PARAMS["combined_left"],
        right=PARAMS["combined_right"],
    )

    gs_s2d = outer_gs[0].subgridspec(
        3,
        n_cols,
        height_ratios=PARAMS["row_ratios"],
        hspace=PARAMS["row_hspace"],
        wspace=PARAMS["row_wspace"],
    )
    gs_d2s = outer_gs[1].subgridspec(
        3,
        n_cols,
        height_ratios=PARAMS["row_ratios"],
        hspace=PARAMS["row_hspace"],
        wspace=PARAMS["row_wspace"],
    )

    s2d_cdll_ax, s2d_instr_ax, s2d_graph_ax = _render_s2d_snapshots(
        fig,
        gs_s2d,
        s2d_trace,
        s2d_tokens,
        s2d_dag_pos,
        n_cols,
        s2d_indices,
    )
    d2s_cdll_ax, d2s_instr_ax, d2s_graph_ax = _render_d2s_snapshots(
        fig,
        gs_d2s,
        d2s_trace,
        d2s_tokens,
        d2s_target_dag,
        d2s_target_pos,
        n_cols,
        d2s_indices,
    )

    _add_group_boxes_horizontal(
        fig,
        [s2d_cdll_ax, s2d_instr_ax, s2d_graph_ax],
        [d2s_cdll_ax, d2s_instr_ax, d2s_graph_ax],
    )
    _add_legend(fig)

    output_path = os.path.join(output_dir, "fig_algorithm_overview")
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Combined algorithm overview saved: %s", output_path)

    # === Individual S2D figure ===
    generate_s2d_figure(
        canonical_str,
        num_variables,
        output_dir,
        s2d_trace=s2d_trace,
        s2d_tokens=s2d_tokens,
        s2d_dag_pos=s2d_dag_pos,
    )

    # === Individual D2S figure ===
    generate_d2s_figure(
        canonical_str,
        num_variables,
        output_dir,
        d2s_trace=d2s_trace,
        d2s_tokens=d2s_tokens,
        d2s_target_dag=d2s_target_dag,
        d2s_target_pos=d2s_target_pos,
    )

    return output_path


def _add_group_boxes_horizontal(
    fig: plt.Figure,
    s2d_axes: list[list[plt.Axes]],
    d2s_axes: list[list[plt.Axes]],
) -> None:
    """Add grouped boxes and labels for the horizontal overview layout.

    Args:
        fig: The figure.
        s2d_axes: 3-element list of [cdll_axes_row, instr_axes_row, graph_axes_row].
        d2s_axes: 3-element list of [cdll_axes_row, instr_axes_row, graph_axes_row].
    """
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    pad = PARAMS["group_box_pad"]

    def _group_bbox(axes_rows: list[list[plt.Axes]]) -> Bbox:
        bboxes = []
        for row in axes_rows:
            for ax in row:
                bb = ax.get_tightbbox(renderer)
                if bb is not None:
                    bboxes.append(bb.transformed(fig.transFigure.inverted()))
        if not bboxes:
            return Bbox([[0, 0], [1, 1]])
        return Bbox.union(bboxes)

    s2d_bb = _group_bbox(s2d_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (s2d_bb.x0 - pad, s2d_bb.y0 - pad),
            s2d_bb.width + 2 * pad,
            s2d_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor=PARAMS["color_s2d_group"],
            alpha=PARAMS["group_box_alpha"],
            edgecolor=PARAMS["color_s2d_group"],
            linewidth=PARAMS["group_box_linewidth"],
            transform=fig.transFigure,
            zorder=0,
        )
    )

    d2s_bb = _group_bbox(d2s_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (d2s_bb.x0 - pad, d2s_bb.y0 - pad),
            d2s_bb.width + 2 * pad,
            d2s_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor=PARAMS["color_d2s_group"],
            alpha=PARAMS["group_box_alpha"],
            edgecolor=PARAMS["color_d2s_group"],
            linewidth=PARAMS["group_box_linewidth"],
            transform=fig.transFigure,
            zorder=0,
        )
    )

    # Group titles (rotated, to the left of boxes)
    fig.text(
        s2d_bb.x0 - pad - 0.025,
        s2d_bb.y0 + s2d_bb.height / 2,
        "String-to-DAG (S2D)",
        ha="center",
        va="center",
        fontsize=PARAMS["group_label_fontsize"],
        fontweight="bold",
        color=PARAMS["color_s2d_label"],
        transform=fig.transFigure,
        rotation=90,
    )
    fig.text(
        d2s_bb.x0 - pad - 0.025,
        d2s_bb.y0 + d2s_bb.height / 2,
        "DAG-to-String (D2S)",
        ha="center",
        va="center",
        fontsize=PARAMS["group_label_fontsize"],
        fontweight="bold",
        color=PARAMS["color_d2s_label"],
        transform=fig.transFigure,
        rotation=90,
    )

    # Horizontal divider between the two groups
    mid_y = (s2d_bb.y0 + d2s_bb.y1) / 2
    fig.add_artist(
        plt.Line2D(
            [max(s2d_bb.x0, d2s_bb.x0) - pad, min(s2d_bb.x1, d2s_bb.x1) + pad],
            [mid_y, mid_y],
            transform=fig.transFigure,
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Generate all algorithm overview figures using PARAMS.

    Edit the PARAMS dict at the top of this file to tweak any parameter,
    then re-run:  python experiments/scripts/generate_algorithm_overview.py
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()

    output_dir = PARAMS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Build expression: sin(x_0)*x_1 + cos(x_0)
    # k=4, 2 variables, 4 op types (SIN, COS, MUL, ADD),
    # depth=3, 16-token canonical string.
    dag = LabeledDAG(7)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x_0
    dag.add_node(NodeType.VAR, var_index=1)  # 1: x_1
    dag.add_node(NodeType.SIN)  # 2: sin(x_0)
    dag.add_edge(0, 2)
    dag.add_node(NodeType.COS)  # 3: cos(x_0)
    dag.add_edge(0, 3)
    dag.add_node(NodeType.MUL)  # 4: sin(x_0)*x_1
    dag.add_edge(2, 4)
    dag.add_edge(1, 4)
    dag.add_node(NodeType.ADD)  # 5: sin(x_0)*x_1 + cos(x_0)
    dag.add_edge(4, 5)
    dag.add_edge(3, 5)

    logger.info("Expression: sin(x_0)*x_1 + cos(x_0)")
    logger.info("DAG: k=4, 6 edges, depth=3")

    canon = canonical_string(dag)
    logger.info("Canonical string: %r (len=%d)", canon, len(canon))

    generate_algorithm_overview(canon, dag, num_variables=2, output_dir=output_dir)

    logger.info("Done. Figures saved in %s", output_dir)


if __name__ == "__main__":
    main()
