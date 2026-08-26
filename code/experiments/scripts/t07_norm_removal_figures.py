"""T07 figures — visualise norm-removal study results.

Produces two publication-ready figures:

``fig_const_why``
    Four-panel explanation for readers: (a) orphan CONST before repair,
    (b) same DAG after repair, (c) pre-T15 merge of non-isomorphic DAGs,
    (d) the equivariance counterexample showing keep-arm silent wrong answer.

``fig_norm_removal_results``
    Per-population comparison: ρ side-by-side, distinct-string counts, and
    equivariance failure rates for keep and drop arms.  The adversarial
    column is visually distinct.

Usage
-----
    python -m experiments.scripts.t07_norm_removal_figures \\
        --in /tmp/t07_smoke/results.json \\
        --out /tmp/t07_smoke/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes

log = logging.getLogger("t07.figures")

# ---------------------------------------------------------------------------
# Colours (Paul Tol bright — colorblind-safe)
# ---------------------------------------------------------------------------
_C_KEEP = "#4477AA"  # blue
_C_DROP = "#EE6677"  # red
_C_ADV = "#AA3377"  # purple
_C_CONST = "#CCBB44"  # yellow — highlights CONST nodes
_C_EDGE_REPAIR = "#228833"  # green — the added creation edge
_C_GHOST = "#BBBBBB"  # grey — suppressed annotations
_C_NODE = "#F0F0F0"  # light grey — default node fill

# ---------------------------------------------------------------------------
# Standalone node-label helper (mirrors generate_algorithm_overview.py)
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[NodeType, str] = {
    NodeType.ADD: "$+$",
    NodeType.MUL: r"$\times$",
    NodeType.SUB: "$-$",
    NodeType.DIV: r"$\div$",
    NodeType.SIN: "sin",
    NodeType.COS: "cos",
    NodeType.EXP: "exp",
    NodeType.LOG: "log",
    NodeType.SQRT: r"$\sqrt{\ }$",  # thin-space avoids empty-group mathtext crash
    NodeType.POW: r"$\wedge$",
    NodeType.ABS: r"$|\cdot|$",
    NodeType.NEG: "neg",
    NodeType.INV: "inv",
    NodeType.CONST: "$c$",
}


def _node_label(dag: LabeledDAG, node_id: int) -> str:
    """Return display label for one DAG node.

    Args:
        dag: The DAG.
        node_id: Node index.

    Returns:
        Display-ready string with LaTeX where appropriate.
    """
    label = dag.node_label(node_id)
    if label == NodeType.VAR:
        idx = dag.node_data(node_id).get("var_index", "?")
        return f"$x_{{{idx}}}$"
    return _LABEL_MAP.get(label, "?")


# ---------------------------------------------------------------------------
# Simple DAG layout and drawing
# ---------------------------------------------------------------------------


def _dag_pos(dag: LabeledDAG) -> dict[int, tuple[float, float]]:
    """Compute a simple layered layout via BFS from VAR nodes.

    Args:
        dag: The labeled DAG to lay out.

    Returns:
        Dict mapping node index to (x, y) coordinates.
    """
    n = dag.node_count
    layer: dict[int, int] = {}
    queue = [i for i in range(n) if dag.node_label(i) == NodeType.VAR]
    for v in queue:
        layer[v] = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in dag.out_neighbors(u):
            if v not in layer or layer[v] <= layer[u]:
                layer[v] = layer[u] + 1
                queue.append(v)

    max_layer = max(layer.values()) if layer else 0
    by_layer: dict[int, list[int]] = {}
    for v, lay in layer.items():
        by_layer.setdefault(lay, []).append(v)
    # Nodes not in any layer (unreachable from VAR) get layer max+1
    for v in range(n):
        if v not in layer:
            lv = max_layer + 1
            layer[v] = lv
            by_layer.setdefault(lv, []).append(v)
    max_layer = max(layer.values())

    pos: dict[int, tuple[float, float]] = {}
    for lay, nodes in by_layer.items():
        for j, v in enumerate(sorted(nodes)):
            x = j - (len(nodes) - 1) / 2.0
            y = max_layer - lay
            pos[v] = (x, y)
    return pos


def _draw_dag_ax(
    ax: Axes,
    dag: LabeledDAG,
    *,
    node_colors: dict[int, str] | None = None,
    highlight_edges: set[tuple[int, int]] | None = None,
    highlight_edge_color: str = _C_EDGE_REPAIR,
    title: str = "",
    annotation: str = "",
    node_r: float = 0.28,
    pos: dict[int, tuple[float, float]] | None = None,
) -> None:
    """Draw a labeled DAG on *ax*.

    Args:
        ax: Matplotlib axes.
        dag: The DAG to draw.
        node_colors: Optional per-node fill colours.
        highlight_edges: Edges to draw in a different colour (dashed).
        highlight_edge_color: Colour for highlighted edges.
        title: Axes title.
        annotation: Text shown below the DAG.
        node_r: Radius of node circles.
        pos: Pre-computed positions.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=4)

    if dag.node_count == 0:
        return

    if pos is None:
        pos = _dag_pos(dag)
    nc = node_colors or {}
    he = highlight_edges or set()

    # Compute axis limits
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = node_r * 2
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad - 0.4, max(ys) + pad)

    # Draw edges
    for src in range(dag.node_count):
        for tgt in dag.out_neighbors(src):
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            dx, dy = x1 - x0, y1 - y0
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                continue
            ux, uy = dx / dist, dy / dist
            sx, sy = x0 + ux * node_r, y0 + uy * node_r
            ex, ey = x1 - ux * node_r, y1 - uy * node_r
            is_hi = (src, tgt) in he
            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": highlight_edge_color if is_hi else "0.25",
                    "linewidth": 2.0 if is_hi else 0.9,
                    "linestyle": (0, (4, 2)) if is_hi else "solid",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 12,
                },
                annotation_clip=False,
            )

    # Draw nodes
    for v, (x, y) in pos.items():
        fc = nc.get(v, _C_NODE)
        lw = 1.8 if fc != _C_NODE else 0.8
        circ = mpatches.Circle(
            (x, y), node_r, facecolor=fc, edgecolor="0.2", linewidth=lw, zorder=3
        )
        ax.add_patch(circ)
        ax.text(x, y, _node_label(dag, v), ha="center", va="center", fontsize=8, zorder=4)

    if annotation:
        ax.text(
            0.5,
            -0.02,
            annotation,
            ha="center",
            va="top",
            fontsize=7,
            color="0.3",
            transform=ax.transAxes,
            wrap=True,
        )


# ---------------------------------------------------------------------------
# Figure 1: fig_const_why — four-panel explanation
# ---------------------------------------------------------------------------


def _make_panel_a() -> tuple[LabeledDAG, dict[int, str]]:
    """DAG: x_0 * c, orphan CONST (no in-edge)."""
    d = LabeledDAG(8)
    d.add_node(NodeType.VAR, var_index=0)  # 0 = x_0
    d.add_node(NodeType.CONST, const_value=1.0)  # 1 = c
    d.add_node(NodeType.MUL)  # 2 = *
    d.add_edge(0, 2)  # x_0 -> *
    d.add_edge(1, 2)  # c -> *   (c has this out-edge but no in-edge)
    return d, {1: _C_CONST}


def _make_panel_b() -> tuple[LabeledDAG, dict[int, str], set[tuple[int, int]]]:
    """Same DAG after 𝒩 repair: x_0 -> c added."""
    d = LabeledDAG(8)
    d.add_node(NodeType.VAR, var_index=0)
    d.add_node(NodeType.CONST, const_value=1.0)
    d.add_node(NodeType.MUL)
    d.add_edge(0, 2)
    d.add_edge(1, 2)
    d.add_edge(0, 1)  # repair edge: x_0 -> c
    return d, {1: _C_CONST}, {(0, 1)}


def _make_panel_c_left() -> LabeledDAG:
    """DAG A: x_0, x_1, CONST (child of x_0), SIN (child of CONST), ADD."""
    d = LabeledDAG(12)
    d.add_node(NodeType.VAR, var_index=0)  # 0
    d.add_node(NodeType.VAR, var_index=1)  # 1
    c = d.add_node(NodeType.CONST, const_value=1.0)  # 2
    s = d.add_node(NodeType.SIN)  # 3
    a = d.add_node(NodeType.ADD)  # 4
    d.add_edge(0, c)  # x_0 -> c  (creation edge from pre-T15 policy)
    d.add_edge(c, s)  # c -> sin
    d.add_edge(s, a)  # sin -> add
    d.add_edge(1, a)  # x_1 -> add
    return d


def _make_panel_c_right() -> LabeledDAG:
    """DAG B: same shape but CONST hangs off SIN, not x_0.  Non-isomorphic."""
    d = LabeledDAG(12)
    d.add_node(NodeType.VAR, var_index=0)
    d.add_node(NodeType.VAR, var_index=1)
    s = d.add_node(NodeType.SIN)  # 2 = sin
    c = d.add_node(NodeType.CONST, const_value=1.0)  # 3
    a = d.add_node(NodeType.ADD)  # 4
    d.add_edge(0, s)
    d.add_edge(s, c)  # sin -> c  (creation edge from pre-T15 policy)
    d.add_edge(c, a)
    d.add_edge(1, a)
    return d


def _make_panel_d() -> tuple[LabeledDAG, LabeledDAG]:
    """Verbatim fixture counterexample (d and d2)."""
    d = LabeledDAG(16)
    for i in range(3):
        d.add_node(NodeType.VAR, var_index=i)
    c1 = d.add_node(NodeType.CONST, const_value=1.0)  # 3
    c2 = d.add_node(NodeType.CONST, const_value=1.0)  # 4
    node_a = d.add_node(NodeType.SIN)  # 5
    node_b = d.add_node(NodeType.SIN)  # 6
    d.add_edge(c2, node_a)
    d.add_edge(node_a, 0)
    d.add_edge(c1, node_b)
    d.add_edge(node_b, 1)
    d2 = permute_internal_nodes(d, [1, 0, 2, 3])
    return d, d2


def make_fig_const_why(out_dir: Path) -> None:
    """Generate fig_const_why.{png,pdf} — four-panel CONST normalisation explainer.

    Args:
        out_dir: Directory to write figures into.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    # Panel (a): orphan CONST before repair
    dag_a, nc_a = _make_panel_a()
    _draw_dag_ax(
        ax_a,
        dag_a,
        node_colors=nc_a,
        title="(a) Host DAG as it arrives",
        annotation="$c$ has in-degree 0 ⟹ no V/v token creates it ⟹ unencodable in Σ_SR",
    )

    # Panel (b): after repair
    dag_b, nc_b, he_b = _make_panel_b()
    _draw_dag_ax(
        ax_b,
        dag_b,
        node_colors=nc_b,
        highlight_edges=he_b,
        highlight_edge_color=_C_EDGE_REPAIR,
        title="(b) After 𝒩 repair (production)",
        annotation="Value unchanged: CONST ignores in-edges during evaluation",
    )

    # Panel (c): pre-T15 policy merging non-isomorphic DAGs
    dag_c1 = _make_panel_c_left()
    dag_c2 = _make_panel_c_right()

    # Draw both DAGs side-by-side inside panel (c)
    ax_c.axis("off")
    ax_c.set_title("(c) Pre-T15 policy: two DAGs, one string", fontsize=9, pad=4)
    inner_left = ax_c.inset_axes([0.0, 0.05, 0.48, 0.82])
    inner_right = ax_c.inset_axes([0.52, 0.05, 0.48, 0.82])
    _draw_dag_ax(inner_left, dag_c1, node_colors={2: _C_CONST})
    _draw_dag_ax(inner_right, dag_c2, node_colors={3: _C_CONST})
    ax_c.text(
        0.5,
        0.0,
        "Both mapped to same canonical string ⟹ merged (wrong)",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.3",
        transform=ax_c.transAxes,
    )

    # Panel (d): equivariance counterexample
    dag_d, dag_d2 = _make_panel_d()
    ax_d.axis("off")
    ax_d.set_title(
        "(d) Keep arm: equivariance failure (isomorphic → different strings)", fontsize=9, pad=4
    )
    inner_d = ax_d.inset_axes([0.0, 0.05, 0.48, 0.82])
    inner_d2 = ax_d.inset_axes([0.52, 0.05, 0.48, 0.82])
    nc_d = {3: _C_CONST, 4: _C_CONST}  # c1, c2
    _draw_dag_ax(inner_d, dag_d, node_colors=nc_d)
    _draw_dag_ax(inner_d2, dag_d2, node_colors=nc_d)
    ax_d.text(
        0.5,
        0.0,
        "d and d2 are isomorphic (c₁↔c₂ swap) but keep arm gives different strings.\n"
        "Drop arm correctly raises RuntimeError on both.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.3",
        transform=ax_d.transAxes,
    )

    fig.suptitle(
        "CONST creation-edge normalisation (𝒩): role and limits",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()

    base = str(out_dir / "fig_const_why")
    for fmt in ("png", "pdf"):
        fig.savefig(f"{base}.{fmt}", dpi=300, bbox_inches="tight")
        log.info("Wrote %s.%s", base, fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: fig_norm_removal_results — per-population comparison bars
# ---------------------------------------------------------------------------


def _safe(v: Any, default: float = float("nan")) -> float:
    """Return *v* as float, or *default* if None/NaN/string.

    Args:
        v: Value to convert.
        default: Fallback for non-numeric values.

    Returns:
        Float value.
    """
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def make_fig_norm_removal_results(data: dict[str, Any], out_dir: Path) -> None:
    """Generate fig_norm_removal_results.{png,pdf} — per-population arm comparison.

    Args:
        data: Loaded results.json (from t07_norm_removal_study.py or aggregate).
        out_dir: Output directory.
    """
    # Support both a single-task result and an aggregated summary
    if "populations" in data:
        # aggregate summary
        by_pop = data["populations"]

        def _get_arm(pop: str, arm: str) -> dict[str, Any]:
            pop_data = by_pop.get(pop, {})
            return cast(dict[str, Any], pop_data.get("arms", {}).get(arm, {}))

        def _get_cmp(pop: str) -> dict[str, Any]:
            return cast(dict[str, Any], by_pop.get(pop, {}).get("comparisons", {}))

        populations = [p for p in ("synthetic", "adversarial", "bingo", "udfs") if p in by_pop]

        def _rho(pop: str, arm: str) -> float:
            a = _get_arm(pop, arm)
            return _safe(a.get("rho_lower_bound", a.get("rho")))

        def _eq_fail_rate(pop: str, arm: str) -> float:
            a = _get_arm(pop, arm)
            n = _safe(a.get("n_equivariance_samples", 0), 0)
            f = _safe(a.get("n_equivariance_failures", 0), 0)
            return f / n if n > 0 else float("nan")

        def _n_distinct(pop: str, arm: str) -> float:
            a = _get_arm(pop, arm)
            v = a.get("n_unique_lower_bound", a.get("n_unique", 0))
            return _safe(v, 0)
    else:
        # Single task: data is the result for one population
        populations = [data["population"]]

        def _get_arm(pop: str, arm: str) -> dict[str, Any]:  # noqa: F811
            return cast(dict[str, Any], data.get("arms", {}).get(arm, {}))

        def _get_cmp(pop: str) -> dict[str, Any]:  # noqa: F811
            return cast(dict[str, Any], data.get("comparisons", {}))

        def _rho(pop: str, arm: str) -> float:  # noqa: F811
            return _safe(_get_arm(pop, arm).get("rho"))

        def _eq_fail_rate(pop: str, arm: str) -> float:  # noqa: F811
            a = _get_arm(pop, arm)
            n = _safe(a.get("n_equivariance_samples", 0), 0)
            f = _safe(a.get("n_equivariance_failures", 0), 0)
            return f / n if n > 0 else float("nan")

        def _n_distinct(pop: str, arm: str) -> float:  # noqa: F811
            return _safe(_get_arm(pop, arm).get("n_unique", 0), 0)

    if not populations:
        log.warning("No populations found in results — skipping figure 2")
        return

    n_pop = len(populations)
    fig, axes = plt.subplots(1, 3, figsize=(3.5 * n_pop, 4.0))
    if n_pop == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(n_pop)

    # Panel 1: ρ (reduction factor)
    ax_rho = axes[0]
    rho_keep = [_rho(p, "keep") for p in populations]
    rho_drop = [_rho(p, "drop") for p in populations]
    ax_rho.bar(x - bar_width / 2, rho_keep, bar_width, label="keep", color=_C_KEEP, alpha=0.85)
    ax_rho.bar(x + bar_width / 2, rho_drop, bar_width, label="drop", color=_C_DROP, alpha=0.85)
    ax_rho.set_ylabel("Reduction factor ρ")
    ax_rho.set_title("ρ = n_ok / n_unique")
    ax_rho.set_xticks(x)
    pop_labels = [("adversarial*" if p == "adversarial" else p) for p in populations]
    ax_rho.set_xticklabels(pop_labels, rotation=15, ha="right")
    ax_rho.legend(fontsize=8)
    ax_rho.axhline(1.0, color="0.5", linewidth=0.7, linestyle="--")
    ax_rho.set_ylim(bottom=0)

    # Highlight adversarial column
    if "adversarial" in populations:
        adv_idx = populations.index("adversarial")
        ax_rho.axvspan(adv_idx - 0.5, adv_idx + 0.5, alpha=0.08, color=_C_ADV)

    # Panel 2: distinct strings
    ax_nd = axes[1]
    nd_keep = [_n_distinct(p, "keep") for p in populations]
    nd_drop = [_n_distinct(p, "drop") for p in populations]
    ax_nd.bar(x - bar_width / 2, nd_keep, bar_width, label="keep", color=_C_KEEP, alpha=0.85)
    ax_nd.bar(x + bar_width / 2, nd_drop, bar_width, label="drop", color=_C_DROP, alpha=0.85)
    ax_nd.set_ylabel("Distinct canonical strings")
    ax_nd.set_title("n_unique (lower bound for pooled tasks)")
    ax_nd.set_xticks(x)
    ax_nd.set_xticklabels(pop_labels, rotation=15, ha="right")
    ax_nd.legend(fontsize=8)
    if "adversarial" in populations:
        adv_idx = populations.index("adversarial")
        ax_nd.axvspan(adv_idx - 0.5, adv_idx + 0.5, alpha=0.08, color=_C_ADV)

    # Panel 3: equivariance failure rate
    ax_eq = axes[2]
    eq_keep = [_eq_fail_rate(p, "keep") for p in populations]
    eq_drop = [_eq_fail_rate(p, "drop") for p in populations]
    eq_keep_plot = [v if math.isfinite(v) else 0.0 for v in eq_keep]
    eq_drop_plot = [v if math.isfinite(v) else 0.0 for v in eq_drop]
    bars_k = ax_eq.bar(
        x - bar_width / 2, eq_keep_plot, bar_width, label="keep", color=_C_KEEP, alpha=0.85
    )
    bars_d = ax_eq.bar(
        x + bar_width / 2, eq_drop_plot, bar_width, label="drop", color=_C_DROP, alpha=0.85
    )
    # Mark NaN bars with a cross-hatch
    for bar, v in zip(list(bars_k) + list(bars_d), eq_keep + eq_drop, strict=False):
        if not math.isfinite(v):
            bar.set_hatch("///")
            bar.set_edgecolor("0.4")
    ax_eq.set_ylabel("Equivariance failure rate")
    ax_eq.set_title("Permutation equivariance failures\n(0 expected for correct arm)")
    ax_eq.set_xticks(x)
    ax_eq.set_xticklabels(pop_labels, rotation=15, ha="right")
    ax_eq.legend(fontsize=8)
    ax_eq.set_ylim(bottom=0)
    if "adversarial" in populations:
        adv_idx = populations.index("adversarial")
        ax_eq.axvspan(adv_idx - 0.5, adv_idx + 0.5, alpha=0.08, color=_C_ADV)
        ax_eq.text(
            adv_idx,
            ax_eq.get_ylim()[1] * 0.95,
            "keep: silent wrong\ndrop: loud refusal",
            ha="center",
            va="top",
            fontsize=7,
            color=_C_ADV,
        )

    fig.suptitle(
        "T07: keep vs drop of 𝒩  |  * adversarial column highlighted",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout()

    base = str(out_dir / "fig_norm_removal_results")
    for fmt in ("png", "pdf"):
        fig.savefig(f"{base}.{fmt}", dpi=300, bbox_inches="tight")
        log.info("Wrote %s.%s", base, fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for T07 figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="input",
        required=True,
        help="Path to results.json (single task) or summary.json (aggregate).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory to write figures into.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data = json.loads(Path(args.input).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating fig_const_why...")
    make_fig_const_why(out_dir)

    log.info("Generating fig_norm_removal_results...")
    make_fig_norm_removal_results(data, out_dir)

    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
