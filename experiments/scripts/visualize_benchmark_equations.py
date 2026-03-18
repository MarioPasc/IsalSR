#!/usr/bin/env python
"""Visualize representative equations from each benchmark cache.

For each of the 5 benchmarks, selects 4 diverse equations from the precomputed
cache and generates a publication-quality figure showing:
  (1) The DAG as a directed graph with labeled nodes (topological layout)
  (2) The colored canonical string (per-token coloring)
  (3) The SymPy expression rendered in LaTeX
  (4) Structural statistics (k, edges, depth, operation types)

Selection algorithm: greedy diversification in (n_internal, n_distinct_ops, depth)
feature space, seeded by the expression closest to the median n_internal.

Output: PDF + PNG per benchmark in the specified output directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import sympy  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.plotting_styles import (  # noqa: E402
    IEEE_TEXT_WIDTH_INCHES,
    PLOT_SETTINGS,
    TOKEN_COLORS,
    apply_ieee_style,
    save_figure,
    tokenize_for_display,
)
from isalsr.adapters.networkx_adapter import NetworkXAdapter  # noqa: E402
from isalsr.adapters.sympy_adapter import SympyAdapter  # noqa: E402
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import (  # noqa: E402
    BINARY_OPS,
    LEAF_TYPES,
    UNARY_OPS,
    VARIADIC_OPS,
    NodeType,
)
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.precomputed.cache_entry import dag_depth  # noqa: E402

log = logging.getLogger(__name__)

# =============================================================================
# Benchmark configuration
# =============================================================================

BENCHMARKS: list[dict[str, str]] = [
    {"name": "nguyen_1var", "dir": "generate_cache_nguyen_1var"},
    {"name": "nguyen_2var", "dir": "generate_cache_nguyen_2var"},
    {"name": "feynman_1var", "dir": "generate_cache_feynman_1var"},
    {"name": "feynman_2var", "dir": "generate_cache_feynman_2var"},
    {"name": "feynman_3var", "dir": "generate_cache_feynman_3var"},
]

BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "nguyen_1var": "Nguyen (1 variable)",
    "nguyen_2var": "Nguyen (2 variables)",
    "feynman_1var": "Feynman (1 variable)",
    "feynman_2var": "Feynman (2 variables)",
    "feynman_3var": "Feynman (3 variables)",
}

# Number of equations to select per benchmark
N_EQUATIONS = 4

# =============================================================================
# Node rendering configuration (Paul Tol colorblind-safe)
# =============================================================================

NODE_COLORS: dict[str, str] = {
    "variadic": "#228833",  # green
    "binary": "#EE6677",  # red
    "unary": "#AA3377",  # purple
    "leaf": "#BBBBBB",  # gray
}


def node_color(label: NodeType) -> str:
    """Return the display color for a node based on its operation category."""
    if label in VARIADIC_OPS:
        return NODE_COLORS["variadic"]
    if label in BINARY_OPS:
        return NODE_COLORS["binary"]
    if label in UNARY_OPS:
        return NODE_COLORS["unary"]
    return NODE_COLORS["leaf"]


def node_display_label(label: NodeType, data: dict[str, int | float]) -> str:
    """Return the display string for a node in the DAG figure."""
    display_map: dict[NodeType, str] = {
        NodeType.ADD: "$+$",
        NodeType.MUL: r"$\times$",
        NodeType.SUB: "$-$",
        NodeType.DIV: r"$\div$",
        NodeType.SIN: "sin",
        NodeType.COS: "cos",
        NodeType.EXP: "exp",
        NodeType.LOG: "log",
        NodeType.SQRT: r"$\sqrt{}$",
        NodeType.POW: r"$\wedge$",
        NodeType.ABS: r"$|\cdot|$",
        NodeType.NEG: "neg",
        NodeType.INV: "inv",
        NodeType.CONST: "$c$",
    }
    if label == NodeType.VAR:
        vi = data.get("var_index", 0)
        return f"$x_{{{int(vi)}}}$"
    return display_map.get(label, label.name)


# =============================================================================
# Graph layout
# =============================================================================


def topological_layout(graph: nx.DiGraph) -> dict[int, tuple[float, float]]:
    """Hierarchical layout: sources at bottom, sinks at top.

    Assigns levels by longest path from any source (topological depth).
    Within each level, nodes are centered horizontally.

    Args:
        graph: A directed acyclic graph.

    Returns:
        Position dict mapping node -> (x, y).
    """
    levels: dict[int, int] = {}
    for node in nx.topological_sort(graph):
        preds = list(graph.predecessors(node))
        levels[node] = max((levels[p] for p in preds), default=-1) + 1

    # Group by level.
    level_nodes: dict[int, list[int]] = {}
    for n, lev in levels.items():
        level_nodes.setdefault(lev, []).append(n)

    pos: dict[int, tuple[float, float]] = {}
    for lev, nodes in level_nodes.items():
        w = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            x = (i - (w - 1) / 2) * 1.5  # center horizontally
            y = lev * 1.5  # level spacing
            pos[n] = (x, y)
    return pos


# =============================================================================
# Equation selection
# =============================================================================


def select_diverse_equations(
    canonical_strings: list[str],
    num_variables: int,
    n_select: int = 4,
) -> list[tuple[str, LabeledDAG]]:
    """Select diverse equations from canonical strings via greedy diversification.

    Feature vector: (n_internal, n_distinct_op_types, depth).

    Args:
        canonical_strings: List of unique canonical strings.
        num_variables: Number of input variables for S2D.
        n_select: Number of equations to select.

    Returns:
        List of (canonical_string, LabeledDAG) tuples, sorted by n_internal.
    """

    # Build DAGs and compute features.
    candidates: list[tuple[str, LabeledDAG, np.ndarray]] = []
    for cs in canonical_strings:
        try:
            dag = StringToDAG(cs, num_variables=num_variables).run()
        except Exception:
            log.warning("Failed to build DAG from canonical string: %s", cs)
            continue

        n_internal = dag.node_count - num_variables

        # Filter: skip trivial expressions with fewer than 2 internal nodes.
        if n_internal < 2:
            continue

        # Filter: must have a valid single output node (well-formed expression).
        try:
            dag.output_node()
        except ValueError:
            continue

        # Filter: must have a valid SymPy expression (for the LaTeX rendering).
        try:
            SympyAdapter().to_sympy(dag)
        except Exception:
            continue

        # Distinct op types (excluding VAR and CONST -- actual operations only).
        op_types: set[NodeType] = set()
        for i in range(dag.node_count):
            label = dag.node_label(i)
            if label not in LEAF_TYPES:
                op_types.add(label)

        n_distinct_ops = len(op_types)

        # Filter: must have at least one actual operation (not just leaves).
        if n_distinct_ops < 1:
            continue

        depth = dag_depth(dag)

        features = np.array([n_internal, n_distinct_ops, depth], dtype=float)
        candidates.append((cs, dag, features))

    if len(candidates) <= n_select:
        log.warning(
            "Only %d candidates available (requested %d)",
            len(candidates),
            n_select,
        )
        result = [(cs, dag) for cs, dag, _ in candidates]
        result.sort(key=lambda x: x[1].node_count)
        return result

    # Extract feature matrix for distance computation.
    feature_matrix = np.array([f for _, _, f in candidates])

    # Normalize features to [0, 1] for fair distance computation.
    fmin = feature_matrix.min(axis=0)
    fmax = feature_matrix.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0  # avoid div by zero
    feature_norm = (feature_matrix - fmin) / frange

    # Seed: closest to median n_internal.
    median_n_internal = np.median(feature_matrix[:, 0])
    seed_idx = int(np.argmin(np.abs(feature_matrix[:, 0] - median_n_internal)))

    selected_indices: list[int] = [seed_idx]

    # Greedy diversification.
    for _ in range(n_select - 1):
        best_idx = -1
        best_min_dist = -1.0
        for i in range(len(candidates)):
            if i in selected_indices:
                continue
            # Min distance to all already-selected.
            min_dist = min(
                float(np.linalg.norm(feature_norm[i] - feature_norm[j])) for j in selected_indices
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            selected_indices.append(best_idx)

    # Sort by n_internal ascending for visual progression.
    selected_indices.sort(key=lambda i: feature_matrix[i, 0])

    return [(candidates[i][0], candidates[i][1]) for i in selected_indices]


# =============================================================================
# Colored string rendering (axes-based, using transAxes)
# =============================================================================


def render_colored_tokens_on_axes(
    ax: plt.Axes,
    string: str,
    fontsize: int = 9,
) -> None:
    """Render a colored canonical string centered on the axes.

    Uses transAxes coordinates so the string is positioned relative to
    the axes, not data coordinates. Each token is colored according to
    TOKEN_COLORS.

    Args:
        ax: Matplotlib Axes to draw on.
        string: IsalSR instruction string.
        fontsize: Font size for rendering.
    """
    tokens = tokenize_for_display(string)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Build a list of (token, color) pairs.
    parts: list[tuple[str, str]] = []
    for token in tokens:
        color = TOKEN_COLORS.get(token, "#333333")
        parts.append((token, color))

    # Render using a single annotation with multiple text segments.
    # We use fig.canvas.get_renderer() and manual x-offset advancement.
    fig = ax.figure
    # Force a draw to get renderer.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Start from center-left; we will measure total width then re-center.
    # First pass: compute total width in axes fraction.
    dummy_texts = []
    total_width = 0.0
    for token_str, _ in parts:
        txt = ax.text(
            0,
            0,
            token_str,
            fontfamily="monospace",
            fontsize=fontsize,
            transform=ax.transAxes,
        )
        txt.draw(renderer)
        bbox = txt.get_window_extent(renderer=renderer)
        inv = ax.transAxes.inverted()
        axes_bbox = inv.transform(bbox)
        w = axes_bbox[1][0] - axes_bbox[0][0]
        total_width += w
        dummy_texts.append(txt)

    # Remove dummy texts.
    for txt in dummy_texts:
        txt.remove()

    # Second pass: render centered.
    x_start = 0.5 - total_width / 2
    x = x_start
    y = 0.5
    for token_str, color in parts:
        txt = ax.text(
            x,
            y,
            token_str,
            fontfamily="monospace",
            fontsize=fontsize,
            color=color,
            transform=ax.transAxes,
            va="center",
            ha="left",
        )
        txt.draw(renderer)
        bbox = txt.get_window_extent(renderer=renderer)
        inv = ax.transAxes.inverted()
        axes_bbox = inv.transform(bbox)
        w = axes_bbox[1][0] - axes_bbox[0][0]
        x += w


# =============================================================================
# Single equation row rendering
# =============================================================================


def render_equation_row(
    fig: plt.Figure,
    gs_row: gridspec.SubplotSpec,
    canonical_str: str,
    dag: LabeledDAG,
    row_label: str,
    num_variables: int,
) -> None:
    """Render one equation row in the figure.

    Layout:
      Left (70%): DAG visualization
      Right (30%): Statistics
      Bottom-left: Colored canonical string
      Bottom-right: LaTeX expression

    Args:
        fig: The parent figure.
        gs_row: GridSpec row to subdivide.
        canonical_str: The canonical instruction string.
        dag: The LabeledDAG to visualize.
        row_label: Row panel label (e.g., "(a)").
        num_variables: Number of input variables.
    """
    # Subdivide the row: 3 sub-rows (DAG+stats, string, expression).
    inner = gs_row.subgridspec(3, 2, height_ratios=[4, 1, 1], width_ratios=[7, 3])

    # --- DAG axes (top-left) ---
    ax_dag = fig.add_subplot(inner[0, 0])
    _draw_dag(ax_dag, dag, num_variables)
    ax_dag.set_title(
        row_label,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        loc="left",
        fontweight="bold",
        pad=4,
    )

    # --- Stats axes (top-right) ---
    ax_stats = fig.add_subplot(inner[0, 1])
    _draw_stats(ax_stats, dag, num_variables)

    # --- Colored string axes (bottom-left, spans both columns) ---
    ax_string = fig.add_subplot(inner[1, :])
    render_colored_tokens_on_axes(ax_string, canonical_str, fontsize=9)

    # --- LaTeX expression axes (bottom row, spans both columns) ---
    ax_latex = fig.add_subplot(inner[2, :])
    _draw_latex_expression(ax_latex, dag)


def _draw_dag(
    ax: plt.Axes,
    dag: LabeledDAG,
    num_variables: int,
) -> None:
    """Draw the DAG on the given axes using NetworkX."""
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")

    nx_adapter = NetworkXAdapter()
    nxg = nx_adapter.to_external(dag)

    if nxg.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    pos = topological_layout(nxg)

    # Prepare node colors and labels.
    colors: list[str] = []
    labels: dict[int, str] = {}
    for n in nxg.nodes():
        label = dag.node_label(n)
        data = dag.node_data(n)
        colors.append(node_color(label))
        labels[n] = node_display_label(label, data)

    # Node sizes: larger for operations, smaller for leaves.
    sizes: list[int] = []
    for n in nxg.nodes():
        label = dag.node_label(n)
        if label in LEAF_TYPES:
            sizes.append(500)
        else:
            sizes.append(700)

    # Draw.
    nx.draw_networkx_nodes(
        nxg,
        pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors="#333333",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        nxg,
        pos,
        ax=ax,
        labels=labels,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        nxg,
        pos,
        ax=ax,
        edge_color="#555555",
        width=1.2,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=12,
        min_target_margin=12,
    )


def _draw_stats(
    ax: plt.Axes,
    dag: LabeledDAG,
    num_variables: int,
) -> None:
    """Draw structural statistics on the given axes."""
    ax.axis("off")

    n_internal = dag.node_count - num_variables
    n_edges = dag.edge_count
    depth = dag_depth(dag)

    # Collect distinct operation types (non-leaf).
    op_types: set[NodeType] = set()
    for i in range(dag.node_count):
        label = dag.node_label(i)
        if label not in LEAF_TYPES:
            op_types.add(label)

    # Format operation names.
    op_names = sorted(nt.name.lower() for nt in op_types)
    op_str = ", ".join(op_names) if op_names else "(none)"

    stats_lines = [
        f"$k = {n_internal}$",
        f"edges $= {n_edges}$",
        f"depth $= {depth}$",
        f"ops: {{{op_str}}}",
    ]

    y_start = 0.85
    y_step = 0.2
    for i, line in enumerate(stats_lines):
        ax.text(
            0.1,
            y_start - i * y_step,
            line,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            fontfamily="serif",
        )


def _draw_latex_expression(
    ax: plt.Axes,
    dag: LabeledDAG,
) -> None:
    """Render the SymPy expression as LaTeX on the given axes."""
    ax.axis("off")

    try:
        adapter = SympyAdapter()
        sympy_expr = adapter.to_sympy(dag)
        latex_str = sympy.latex(sympy_expr)
        display_str = f"$f(\\mathbf{{x}}) = {latex_str}$"
    except Exception as e:
        log.warning("Failed to convert DAG to SymPy: %s", e)
        display_str = "(expression unavailable)"

    ax.text(
        0.5,
        0.5,
        display_str,
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="center",
    )


# =============================================================================
# Main figure generation
# =============================================================================


def generate_benchmark_figure(
    cache_path: Path,
    benchmark_name: str,
    output_dir: Path,
) -> list[str]:
    """Generate the visualization figure for one benchmark.

    Args:
        cache_path: Path to the cache_merged.h5 file.
        benchmark_name: Benchmark identifier (e.g., "nguyen_1var").
        output_dir: Directory for output files.

    Returns:
        List of saved file paths.
    """
    log.info("Processing benchmark: %s", benchmark_name)

    with h5py.File(cache_path, "r") as f:
        num_variables = int(f.attrs["num_variables"])

        # Load unique canonical strings.
        ucp_raw = f["canonical_index/unique_canonical_pruned"][:]
        canonical_strings: list[str] = []
        for item in ucp_raw:
            if isinstance(item, bytes):
                canonical_strings.append(item.decode("utf-8"))
            else:
                canonical_strings.append(str(item))

    log.info(
        "  Loaded %d unique canonical strings (num_variables=%d)",
        len(canonical_strings),
        num_variables,
    )

    # Select diverse equations.
    selected = select_diverse_equations(canonical_strings, num_variables, n_select=N_EQUATIONS)

    if not selected:
        log.warning("  No equations selected for %s, skipping.", benchmark_name)
        return []

    log.info("  Selected %d equations", len(selected))

    # Create figure.
    n_rows = len(selected)
    row_height = 2.4  # inches per row
    fig_height = n_rows * row_height + 0.6  # extra for suptitle

    fig = plt.figure(figsize=(IEEE_TEXT_WIDTH_INCHES, fig_height))

    display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    fig.suptitle(
        display_name,
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
        fontweight="bold",
        y=0.98,
    )

    outer_gs = gridspec.GridSpec(
        n_rows,
        1,
        figure=fig,
        hspace=0.55,
        top=0.94,
        bottom=0.02,
        left=0.04,
        right=0.96,
    )

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for i, (cs, dag) in enumerate(selected):
        label = panel_labels[i] if i < len(panel_labels) else f"({i + 1})"
        render_equation_row(fig, outer_gs[i], cs, dag, label, num_variables)

    # Save.
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = str(output_dir / f"benchmark_{benchmark_name}_equations")
    saved = save_figure(fig, base_path, formats=("pdf", "png"))
    plt.close(fig)

    for path in saved:
        log.info("  Saved: %s", path)

    return saved


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize representative equations from benchmark caches.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing generate_cache_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output PDF/PNG files.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=None,
        help="Specific benchmarks to process (default: all 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    apply_ieee_style()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    # Filter benchmarks if specified.
    benchmarks = BENCHMARKS
    if args.benchmarks:
        allowed = set(args.benchmarks)
        benchmarks = [b for b in benchmarks if b["name"] in allowed]

    all_saved: list[str] = []
    for bm in benchmarks:
        cache_path = cache_dir / bm["dir"] / "cache_merged.h5"
        if not cache_path.exists():
            log.warning("Cache not found: %s, skipping.", cache_path)
            continue
        saved = generate_benchmark_figure(cache_path, bm["name"], output_dir)
        all_saved.extend(saved)

    log.info("Done. Total files saved: %d", len(all_saved))


if __name__ == "__main__":
    main()
