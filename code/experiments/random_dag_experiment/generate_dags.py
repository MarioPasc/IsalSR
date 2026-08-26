"""Random-DAG preview generator for the screening factor grid.

Generates one representative DAG per cell of a 16-cell screening design
over (k, m, op_set), renders each as a layered PNG, and emits a manifest
plus a metadata CSV. Used to inspect the *boundary* of the proposed
random-DAG benchmark before committing compute on Picasso.

References
----------
- Lample & Charton (2020). Deep Learning for Symbolic Mathematics. ICLR.
  Random unary-binary tree generator (Appendix C).
- Weisfeiler & Leman (1968). The reduction of a graph to a canonical form.
- Lopez-Rubio (2025). IsalGraph. arXiv:2512.10429v2.

Usage
-----
    python -m experiments.random_dag_experiment.generate_dags \\
        --output-dir experiments/random_dag_experiment/outputs \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

# Path setup for in-place runs.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.synthetic_scalability.run_synthetic_scalability import (
    generate_random_expr_tree,
    tree_to_labeled_dag,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("random_dag_experiment")

# ---------------------------------------------------------------------------
# Operator subsets
# ---------------------------------------------------------------------------
OP_SETS: dict[str, tuple[list[NodeType], list[NodeType]]] = {
    # poly: pure polynomial — ADD/MUL build any polynomial via repeated MUL,
    # NEG subsumes subtraction in the commutative encoding.
    "poly": (
        [NodeType.ADD, NodeType.MUL],
        [NodeType.NEG],
    ),
    # poly_trig: Taylor-friendly extension.
    "poly_trig": (
        [NodeType.ADD, NodeType.MUL],
        [NodeType.NEG, NodeType.SIN, NodeType.COS],
    ),
    # full: mirrors the production alphabet (no CONST, no POW, no SUB/DIV).
    "full": (
        [NodeType.ADD, NodeType.MUL],
        [NodeType.NEG, NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.LOG, NodeType.INV],
    ),
}

# Display label per operator (LaTeX-ish ASCII).
LABEL_GLYPH: dict[NodeType, str] = {
    NodeType.ADD: "+",
    NodeType.MUL: "x",  # multiplication glyph (avoid Unicode in PNG fonts)
    NodeType.SUB: "-",
    NodeType.DIV: "/",
    NodeType.POW: "^",
    NodeType.NEG: "neg",
    NodeType.INV: "1/.",
    NodeType.SIN: "sin",
    NodeType.COS: "cos",
    NodeType.EXP: "exp",
    NodeType.LOG: "log",
    NodeType.SQRT: "sqrt",
    NodeType.ABS: "abs",
    NodeType.CONST: "c",
}

# Color palette by category.
COLOR_BY_CATEGORY: dict[str, str] = {
    "var": "#9ec5e8",
    "poly": "#a7d7a7",
    "trig": "#f5b366",
    "transcendental": "#e8908d",
    "const": "#d8d8d8",
}


def _category(label: NodeType) -> str:
    """Map a NodeType to its visual category (for coloring)."""
    if label == NodeType.VAR:
        return "var"
    if label == NodeType.CONST:
        return "const"
    if label in {NodeType.ADD, NodeType.MUL, NodeType.SUB, NodeType.NEG, NodeType.POW}:
        return "poly"
    if label in {NodeType.SIN, NodeType.COS}:
        return "trig"
    return "transcendental"


# ---------------------------------------------------------------------------
# Factor grid (16 cells)
# ---------------------------------------------------------------------------


@dataclass
class GridCell:
    """One design point in the screening grid."""

    cell_id: int
    k: int
    m: int
    op_set: str
    role: str = ""

    @property
    def name(self) -> str:
        return f"cell{self.cell_id:02d}_k{self.k}_m{self.m}_{self.op_set}"


GRID: list[GridCell] = [
    GridCell(1, 6, 1, "poly", "min-k, single-var, pure poly (max VAR sharing)"),
    GridCell(2, 6, 4, "poly", "min-k, multi-var, pure poly (low sharing)"),
    GridCell(3, 6, 2, "poly_trig", "min-k, mid-var, mixed ops"),
    GridCell(4, 6, 2, "full", "min-k, mid-var, full alphabet"),
    GridCell(5, 9, 1, "poly", "mid-k, single-var, pure poly"),
    GridCell(6, 9, 2, "poly", "mid-k, mid-var, pure poly"),
    GridCell(7, 9, 4, "poly_trig", "mid-k, multi-var, mixed"),
    GridCell(8, 9, 2, "full", "mid-k, full alphabet"),
    GridCell(9, 12, 1, "poly", "large-k, single-var (max sharing)"),
    GridCell(10, 12, 2, "poly", "large-k, mid-var, pure poly"),
    GridCell(11, 12, 4, "full", "large-k, multi-var, full alphabet"),
    GridCell(12, 12, 2, "poly_trig", "large-k, mid-var, mixed"),
    GridCell(13, 15, 1, "poly", "XL single-var (extreme sharing)"),
    GridCell(14, 15, 2, "poly", "XL pure poly"),
    GridCell(15, 15, 4, "full", "XL multi-var, full alphabet"),
    GridCell(16, 15, 2, "poly_trig", "XL Taylor-extension"),
]


# ---------------------------------------------------------------------------
# HARD_GRID — designed for the "sweet-spot search". Larger k_target and richer
# operator sets so that the random tree, after collapse, still has ≥10 internal
# nodes and a non-polynomial substructure. Avoids the trivial-target failure
# mode of single-variable poly cells (Add(x, -x) → 0).
# ---------------------------------------------------------------------------
HARD_GRID: list[GridCell] = [
    GridCell(101, 18, 2, "poly_trig", "k=18, m=2, poly+trig"),
    GridCell(102, 18, 3, "poly", "k=18, m=3, pure poly"),
    GridCell(103, 22, 2, "poly", "k=22, m=2, pure poly"),
    GridCell(104, 22, 3, "poly_trig", "k=22, m=3, poly+trig"),
    GridCell(105, 26, 2, "poly_trig", "k=26, m=2, poly+trig"),
    GridCell(106, 26, 4, "full", "k=26, m=4, full alphabet"),
    GridCell(107, 22, 4, "full", "k=22, m=4, full alphabet"),
    GridCell(108, 30, 3, "poly_trig", "k=30, m=3, poly+trig (XL)"),
]


_GRIDS: dict[str, list[GridCell]] = {"default": GRID, "hard": HARD_GRID}


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------


@dataclass
class DAGMetrics:
    """Structural summary of one generated DAG."""

    cell_id: int
    k: int  # k_requested
    k_actual: int  # internal nodes after associative collapse
    m: int
    op_set: str
    role: str
    n_total_nodes: int
    n_edges: int
    n_vars_used: int
    max_in_degree: int
    max_out_degree: int
    depth: int
    n_perm_samples: int
    n_unique_canon: int
    n_distinct_fingerprints: int
    estimated_RF: float
    canon_invariant: bool
    canonical_len: int
    label_histogram: dict[str, int] = field(default_factory=dict)


def _depth(dag: LabeledDAG) -> int:
    """Longest path length in a DAG (in edges).

    Uses a topological DP. Returns 0 for a DAG with no edges.
    """
    n = dag.node_count
    indeg = [0] * n
    for v in range(n):
        indeg[v] = len(list(dag.in_neighbors(v)))

    queue = [v for v in range(n) if indeg[v] == 0]
    longest = [0] * n
    order: list[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for w in dag.out_neighbors(u):
            indeg[w] -= 1
            if longest[w] < longest[u] + 1:
                longest[w] = longest[u] + 1
            if indeg[w] == 0:
                queue.append(w)
    return max(longest) if longest else 0


def _label_histogram(dag: LabeledDAG) -> dict[str, int]:
    """Count nodes by NodeType name."""
    hist: dict[str, int] = {}
    for i in range(dag.node_count):
        name = dag.node_label(i).name
        hist[name] = hist.get(name, 0) + 1
    return hist


def _structural_fingerprint(dag: LabeledDAG) -> tuple:
    """ID-sensitive structural fingerprint of a labelled DAG.

    Two permutations of internal node IDs that yield the **same** fingerprint
    correspond to the same labelled DAG up to identity (i.e. the permutation
    is in Aut(D)). Number of distinct fingerprints across all k! permutations
    therefore equals k!/|Aut(D)| (the size of the orbit). This is the
    *non-canonical* counterpart to the canonical string — by construction,
    fast_canonical_string collapses all of these to a single canonical, so
    counting canonicals would always yield 1.
    """
    n = dag.node_count
    return tuple(
        (
            dag.node_label(i).name,
            tuple(sorted(dag.in_neighbors(i))),
            tuple(sorted(dag.out_neighbors(i))),
        )
        for i in range(n)
    )


def _estimate_RF(dag: LabeledDAG, k: int, max_perms: int) -> tuple[int, int, int, bool]:
    """Sample permutations and measure both canonical invariance and orbit size.

    Returns (n_perm_samples, n_unique_canonicals, n_distinct_fingerprints,
    canon_invariant). RF ≈ samples / n_distinct_fingerprints is a
    Monte-Carlo proxy for k!/|Aut(D)|; canon_invariant is True iff
    n_unique_canonicals == 1 (the IsalSR core guarantee).
    """
    if k == 0:
        return (1, 1, 1, True)
    k_fact = math.factorial(k)
    if k_fact <= max_perms:
        from itertools import permutations as _perms

        perm_iter = (list(p) for p in _perms(range(k)))
        n_samples = k_fact
    else:
        rng = random.Random(0xC0FFEE + k)
        base = list(range(k))

        def _gen():
            for _ in range(max_perms):
                p = base.copy()
                for i in range(k - 1, 0, -1):
                    j = rng.randint(0, i)
                    p[i], p[j] = p[j], p[i]
                yield p

        perm_iter = _gen()
        n_samples = max_perms

    canonicals: set[str] = set()
    fingerprints: set[tuple] = set()
    for perm in perm_iter:
        dag_p = permute_internal_nodes(dag, perm)
        canonicals.add(fast_canonical_string(dag_p, mode="wl_only"))
        fingerprints.add(_structural_fingerprint(dag_p))
    return (n_samples, len(canonicals), len(fingerprints), len(canonicals) == 1)


def compute_metrics(
    cell: GridCell,
    dag: LabeledDAG,
    perm_samples: int,
) -> tuple[DAGMetrics, str]:
    """Compute the metric record + canonical string for one DAG."""
    n = dag.node_count
    n_vars_used = sum(
        1
        for i in range(cell.m)
        if dag.node_label(i) == NodeType.VAR and len(list(dag.out_neighbors(i))) > 0
    )
    max_in = max((len(list(dag.in_neighbors(i))) for i in range(n)), default=0)
    max_out = max((len(list(dag.out_neighbors(i))) for i in range(n)), default=0)
    depth = _depth(dag)

    canon = fast_canonical_string(dag, mode="wl_only")
    k_actual = max(0, dag.node_count - cell.m)
    n_samples, n_unique_canon, n_distinct_fp, canon_inv = _estimate_RF(dag, k_actual, perm_samples)
    rf = float(n_samples) / max(1, n_distinct_fp)

    metrics = DAGMetrics(
        cell_id=cell.cell_id,
        k=cell.k,
        k_actual=k_actual,
        m=cell.m,
        op_set=cell.op_set,
        role=cell.role,
        n_total_nodes=n,
        n_edges=dag.edge_count,
        n_vars_used=n_vars_used,
        max_in_degree=max_in,
        max_out_degree=max_out,
        depth=depth,
        n_perm_samples=n_samples,
        n_unique_canon=n_unique_canon,
        n_distinct_fingerprints=n_distinct_fp,
        estimated_RF=rf,
        canon_invariant=canon_inv,
        canonical_len=len(canon),
        label_histogram=_label_histogram(dag),
    )
    return metrics, canon


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _node_text(label: NodeType, node_id: int, var_index: int | None) -> str:
    if label == NodeType.VAR:
        return f"x{var_index}" if var_index is not None else "x?"
    glyph = LABEL_GLYPH.get(label, label.name.lower())
    return glyph


def dag_to_formula(dag: LabeledDAG) -> tuple[str, str]:
    """Convert DAG to (sympy_str, latex_str) for figure rendering.

    Returns ("", "") on failure (e.g., adapter exception or empty DAG).
    """
    try:
        import sympy as sp

        from isalsr.adapters.sympy_adapter import SympyAdapter as _SympyAdapter

        # Use the structural expression as built from the DAG; do NOT
        # simplify via sp.expand because random NEG/ADD/MUL chains can
        # collapse to trivial constants (e.g. x + (-x) -> 0), destroying
        # the target function we want to fit.
        expr = _SympyAdapter().to_sympy(dag)
        return str(expr), sp.latex(expr)
    except Exception as e:
        log.debug("Formula extraction failed: %s", e)
        return "", ""


def _wrap_text(s: str, width: int = 90, max_lines: int = 4) -> str:
    """Soft-wrap a long string for display on the figure caption."""
    if not s:
        return ""
    out: list[str] = []
    while s and len(out) < max_lines:
        if len(s) <= width:
            out.append(s)
            break
        cut = s.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(s[:cut])
        s = s[cut:].lstrip()
    if s and len(out) == max_lines:
        out[-1] = out[-1].rstrip() + " ..."
    return "\n".join(out)


def _evaluate_target(formula: str, m: int, X: np.ndarray) -> np.ndarray:
    """Evaluate the SymPy formula at the rows of X (shape (n, m))."""
    import sympy as sp

    expr = sp.sympify(formula)
    syms = [sp.Symbol(f"x_{i}") for i in range(m)]
    fn = sp.lambdify(syms, expr, modules="numpy")
    if m == 1:
        y = fn(X[:, 0])
    else:
        y = fn(*X.T)
    y = np.asarray(y, dtype=np.float64)
    if y.shape == ():
        y = np.full(X.shape[0], float(y))
    return y.reshape(-1)


def _plot_target_data(
    ax: object,
    formula: str,
    m: int,
    *,
    n_train: int = 80,
    n_test: int = 40,
    bound: float = 1.0,
    seed: int = 0,
) -> None:
    """Draw the target evaluation panel.

    - m=1: dense true curve overlaid with train/test scatter.
    - m=2: contour of the true function over [-bound, bound]^2 with
      train/test points overlaid.
    - m>=3: scatter of (x_0 projection, y) for visual reference only.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (ax is matplotlib already)

    rng = np.random.default_rng(seed)
    X_tr = rng.uniform(-bound, bound, size=(n_train, m))
    X_te = rng.uniform(-bound, bound, size=(n_test, m))

    try:
        y_tr = _evaluate_target(formula, m, X_tr)
        y_te = _evaluate_target(formula, m, X_te)
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"target evaluation failed:\n{e}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_axis_off()
        return

    finite = np.isfinite(y_tr)
    y_tr_f = y_tr[finite]
    X_tr_f = X_tr[finite]
    finite_te = np.isfinite(y_te)
    y_te_f = y_te[finite_te]
    X_te_f = X_te[finite_te]

    if y_tr_f.size == 0:
        ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    if m == 1:
        x_dense = np.linspace(-bound, bound, 400).reshape(-1, 1)
        try:
            y_dense = _evaluate_target(formula, m, x_dense)
            mask = np.isfinite(y_dense)
            ax.plot(x_dense[mask, 0], y_dense[mask], color="#2ca02c", lw=2.0, label="target f(x)")
        except Exception:
            pass
        ax.scatter(X_tr_f[:, 0], y_tr_f, s=20, color="#1f77b4", alpha=0.75, label="train")
        ax.scatter(
            X_te_f[:, 0], y_te_f, s=24, color="#ff7f0e", marker="^", alpha=0.85, label="test"
        )
        ax.set_xlabel("$x_0$", fontsize=9)
        ax.set_ylabel("$f(x)$", fontsize=9)
    elif m == 2:
        gx = np.linspace(-bound, bound, 60)
        gy = np.linspace(-bound, bound, 60)
        XX, YY = np.meshgrid(gx, gy)
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1)
        try:
            ZZ = _evaluate_target(formula, m, XY).reshape(XX.shape)
            ZZ = np.where(np.isfinite(ZZ), ZZ, np.nan)
            cs = ax.contourf(XX, YY, ZZ, levels=20, cmap="viridis")
            from matplotlib import pyplot as _plt  # local import for colorbar

            _plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            pass
        ax.scatter(
            X_tr_f[:, 0],
            X_tr_f[:, 1],
            s=18,
            color="white",
            edgecolor="black",
            linewidths=0.5,
            label="train",
        )
        ax.scatter(
            X_te_f[:, 0],
            X_te_f[:, 1],
            s=22,
            color="red",
            edgecolor="black",
            linewidths=0.5,
            marker="^",
            label="test",
        )
        ax.set_xlabel("$x_0$", fontsize=9)
        ax.set_ylabel("$x_1$", fontsize=9)
    else:
        ax.scatter(X_tr_f[:, 0], y_tr_f, s=20, color="#1f77b4", alpha=0.75, label="train")
        ax.scatter(
            X_te_f[:, 0], y_te_f, s=24, color="#ff7f0e", marker="^", alpha=0.85, label="test"
        )
        ax.set_xlabel(f"$x_0$ (projection from $m={m}$)", fontsize=9)
        ax.set_ylabel("$f(x)$", fontsize=9)

    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    ax.set_title("Target evaluation", fontsize=9)


def render_dag(
    dag: LabeledDAG,
    out_path: str,
    title: str,
    formula: str = "",
    latex_formula: str = "",
    m: int = 1,
    data_seed: int = 0,
) -> None:
    """Render a 2-panel PNG: DAG (left) + target-evaluation plot (right).

    The target panel shows the function the DAG implements, along with a
    scatter of randomly-sampled train/test points used downstream by the
    synthetic Bingo runner.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    from isalsr.adapters.networkx_adapter import NetworkXAdapter

    g = NetworkXAdapter().to_external(dag)

    gens: dict[int, int] = {}
    for v in nx.topological_sort(g):
        preds = list(g.predecessors(v))
        gens[v] = 0 if not preds else 1 + max(gens[u] for u in preds)
    by_layer: dict[int, list[int]] = {}
    for node, layer in gens.items():
        by_layer.setdefault(layer, []).append(node)
    pos: dict[int, tuple[float, float]] = {}
    max_layer_width = max(len(v) for v in by_layer.values())
    for layer, nodes in by_layer.items():
        nodes_sorted = sorted(nodes)
        n_in_layer = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            x = (i + 1) / (n_in_layer + 1) * max_layer_width
            pos[node] = (x, -float(layer))

    node_colors: list[str] = []
    node_labels: dict[int, str] = {}
    for node in g.nodes():
        attrs = g.nodes[node]
        label = NodeType[attrs["label"]]
        node_colors.append(COLOR_BY_CATEGORY[_category(label)])
        node_labels[node] = _node_text(label, node, attrs.get("var_index"))

    width = max(12, max_layer_width * 1.2 + 6)
    height = max(5, len(by_layer) * 1.0)
    fig, (ax_dag, ax_data) = plt.subplots(
        1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    nx.draw_networkx_edges(
        g, pos, ax=ax_dag, arrowsize=12, edge_color="#444444", width=1.0, alpha=0.85
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax_dag,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(g, pos, labels=node_labels, ax=ax_dag, font_size=9)

    handles = [
        mpatches.Patch(color=COLOR_BY_CATEGORY["var"], label="VAR"),
        mpatches.Patch(color=COLOR_BY_CATEGORY["poly"], label="poly (ADD/MUL/NEG)"),
        mpatches.Patch(color=COLOR_BY_CATEGORY["trig"], label="trig (SIN/COS)"),
        mpatches.Patch(color=COLOR_BY_CATEGORY["transcendental"], label="transcendental"),
    ]
    ax_dag.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    ax_dag.set_title("DAG", fontsize=10)
    ax_dag.set_axis_off()

    if formula:
        _plot_target_data(ax_data, formula, m, seed=data_seed)
    else:
        ax_data.text(
            0.5,
            0.5,
            "no formula available",
            ha="center",
            va="center",
            transform=ax_data.transAxes,
        )
        ax_data.set_axis_off()

    fig.suptitle(title, fontsize=10)

    if latex_formula or formula:
        rendered = False
        if latex_formula:
            try:
                fig.text(
                    0.5,
                    0.015,
                    f"$f(x) = {latex_formula}$",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                rendered = True
            except Exception:
                rendered = False
        if not rendered and formula:
            wrapped = _wrap_text(f"f(x) = {formula}", width=110, max_lines=3)
            fig.text(0.5, 0.015, wrapped, ha="center", va="bottom", fontsize=8, family="monospace")

    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tree post-processing
# ---------------------------------------------------------------------------


def collapse_associative_tree(node: ExprTreeNode) -> None:
    """Inline same-type ADD/MUL children to produce variable-arity nodes.

    After collapse, a node with label ADD has no ADD children (their children
    are absorbed); same for MUL. The resulting tree therefore exercises
    in-degree > 2 on associative ops, which is the structural lever the
    Lample–Charton generator alone does not produce.
    """
    for child in node.children:
        collapse_associative_tree(child)
    if node.node_type in (NodeType.ADD, NodeType.MUL):
        merged: list[ExprTreeNode] = []
        for child in node.children:
            if child.node_type == node.node_type and child.children:
                merged.extend(child.children)
            else:
                merged.append(child)
        node.children = merged


def ensure_all_vars_used(root: ExprTreeNode, m: int, rng: np.random.Generator) -> bool:
    """Ensure every variable index in [0, m) appears in at least one leaf.

    Overwrites randomly chosen leaves' var_index to cover missing variables.
    Returns False if there are fewer leaves than m (cannot cover); the
    caller should regenerate with a fresh tree.
    """
    leaves: list[ExprTreeNode] = []
    used: set[int] = set()

    def _collect(n: ExprTreeNode) -> None:
        if not n.children:
            leaves.append(n)
            if n.var_index is not None:
                used.add(n.var_index)
        for c in n.children:
            _collect(c)

    _collect(root)
    missing = [v for v in range(m) if v not in used]
    if not missing:
        return True
    if len(leaves) < m:
        return False
    pick = rng.choice(len(leaves), size=len(missing), replace=False)
    for i, var in zip(pick.tolist(), missing, strict=True):
        leaves[int(i)].var_index = var
    return True


def generate_one(
    cell: GridCell, seed: int, max_tries: int = 32
) -> tuple[LabeledDAG, np.random.Generator]:
    """Generate one DAG for the cell, with var-coverage and arity-collapse.

    Steps per attempt:
      1. Sample a random unary–binary tree (Lample–Charton).
      2. Overwrite leaves to ensure every variable index appears.
      3. Collapse same-type ADD/MUL chains into variable-arity nodes.
      4. Convert tree → LabeledDAG (shares VAR nodes).
      5. Verify all m VAR nodes have ≥1 outgoing edge in the final DAG.
    """
    binary_ops, unary_ops = OP_SETS[cell.op_set]
    base_seed = seed + cell.cell_id * 1009
    for attempt in range(max_tries):
        rng = np.random.default_rng(base_seed + attempt * 7919)
        tree = generate_random_expr_tree(cell.k, cell.m, binary_ops, unary_ops, rng)
        if not ensure_all_vars_used(tree, cell.m, rng):
            continue
        collapse_associative_tree(tree)
        dag = tree_to_labeled_dag(tree, cell.m)
        if all(len(list(dag.out_neighbors(i))) > 0 for i in range(cell.m)):
            return dag, rng
    raise RuntimeError(
        f"Could not produce a DAG with all {cell.m} variables connected "
        f"for {cell.name} after {max_tries} attempts."
    )


def run(output_dir: str, seed: int, perm_samples: int, grid_name: str = "default") -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "dag_metadata.csv")
    manifest_path = os.path.join(output_dir, "manifest.json")
    gallery_path = os.path.join(output_dir, "gallery.md")

    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "experiment": "random_dag_experiment_preview",
        "seed": seed,
        "perm_samples_per_cell": perm_samples,
        "operator_sets": {
            name: {
                "binary": [op.name for op in bins],
                "unary": [op.name for op in uns],
            }
            for name, (bins, uns) in OP_SETS.items()
        },
        "cells": [],
    }

    gallery_lines: list[str] = [
        "# Random DAG screening — preview gallery",
        "",
        f"_Generated with seed={seed}, perm_samples={perm_samples}._",
        "",
        "Each row below is one design cell. RF estimate = (# permutations sampled)",
        "/ (# unique canonical strings); RF ≈ k! / |Aut(D)|.",
        "",
    ]

    grid_cells = _GRIDS.get(grid_name)
    if grid_cells is None:
        raise ValueError(f"Unknown grid '{grid_name}', expected one of {list(_GRIDS)}")
    log.info("Using grid '%s' (%d cells)", grid_name, len(grid_cells))

    for cell in grid_cells:
        log.info(
            "Cell %02d: k=%d m=%d op=%s — %s",
            cell.cell_id,
            cell.k,
            cell.m,
            cell.op_set,
            cell.role,
        )
        dag, _ = generate_one(cell, seed)
        metrics, canon = compute_metrics(cell, dag, perm_samples)
        formula, latex_formula = dag_to_formula(dag)

        png_name = f"{cell.name}.png"
        png_path = os.path.join(fig_dir, png_name)
        k_actual = metrics.n_total_nodes - cell.m
        title = (
            f"Cell {cell.cell_id:02d}  k_req={cell.k} k_actual={k_actual}  "
            f"m={cell.m}  ops={cell.op_set}\n"
            f"nodes={metrics.n_total_nodes} edges={metrics.n_edges} "
            f"max_in={metrics.max_in_degree} depth={metrics.depth}  "
            f"RF≈{metrics.estimated_RF:.2f}  vars_used={metrics.n_vars_used}/{cell.m}"
        )
        render_dag(
            dag,
            png_path,
            title,
            formula=formula,
            latex_formula=latex_formula,
            m=cell.m,
            data_seed=seed + cell.cell_id,
        )

        rows.append(
            {
                "cell_id": metrics.cell_id,
                "k": metrics.k,
                "k_actual": metrics.k_actual,
                "m": metrics.m,
                "op_set": metrics.op_set,
                "role": metrics.role,
                "n_total_nodes": metrics.n_total_nodes,
                "n_edges": metrics.n_edges,
                "n_vars_used": metrics.n_vars_used,
                "max_in_degree": metrics.max_in_degree,
                "max_out_degree": metrics.max_out_degree,
                "depth": metrics.depth,
                "n_perm_samples": metrics.n_perm_samples,
                "n_unique_canon": metrics.n_unique_canon,
                "n_distinct_fingerprints": metrics.n_distinct_fingerprints,
                "estimated_RF": round(metrics.estimated_RF, 4),
                "canon_invariant": int(metrics.canon_invariant),
                "k_factorial": math.factorial(metrics.k_actual),
                "canonical_len": metrics.canonical_len,
            }
        )
        manifest["cells"].append(
            {
                "cell_id": cell.cell_id,
                "k": cell.k,
                "k_actual": metrics.k_actual,
                "m": cell.m,
                "op_set": cell.op_set,
                "role": cell.role,
                "metrics": {
                    "n_total_nodes": metrics.n_total_nodes,
                    "n_edges": metrics.n_edges,
                    "n_vars_used": metrics.n_vars_used,
                    "max_in_degree": metrics.max_in_degree,
                    "max_out_degree": metrics.max_out_degree,
                    "depth": metrics.depth,
                    "n_perm_samples": metrics.n_perm_samples,
                    "n_unique_canon": metrics.n_unique_canon,
                    "n_distinct_fingerprints": metrics.n_distinct_fingerprints,
                    "estimated_RF": round(metrics.estimated_RF, 4),
                    "canon_invariant": metrics.canon_invariant,
                    "k_factorial": math.factorial(metrics.k_actual),
                    "canonical_len": metrics.canonical_len,
                    "label_histogram": metrics.label_histogram,
                },
                "canonical_string": canon,
                "sympy_expression": formula,
                "latex_expression": latex_formula,
                "figure": f"figures/{png_name}",
            }
        )

        gallery_lines.extend(
            [
                f"## Cell {cell.cell_id:02d} — k={cell.k}, m={cell.m}, ops=`{cell.op_set}`",
                "",
                f"_{cell.role}_",
                "",
                f"![cell {cell.cell_id:02d}](figures/{png_name})",
                "",
                f"- nodes={metrics.n_total_nodes}, edges={metrics.n_edges}, "
                f"vars_used={metrics.n_vars_used}/{cell.m}",
                f"- max_in_degree={metrics.max_in_degree}, "
                f"max_out_degree={metrics.max_out_degree}, depth={metrics.depth}",
                f"- estimated RF ≈ **{metrics.estimated_RF:.2f}** "
                f"({metrics.n_distinct_fingerprints} distinct fingerprints / "
                f"{metrics.n_perm_samples} perms; k_actual={metrics.k_actual}, "
                f"k! = {math.factorial(metrics.k_actual)})",
                f"- canonical invariance: "
                f"{'OK (1 unique canonical)' if metrics.canon_invariant else 'FAIL'}",
                f"- canonical len = {metrics.canonical_len}",
                f"- target: $f(x) = {latex_formula}$"
                if latex_formula
                else f"- target: `{formula}`",
                "",
            ]
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell_id",
                "k",
                "k_actual",
                "m",
                "op_set",
                "role",
                "n_total_nodes",
                "n_edges",
                "n_vars_used",
                "max_in_degree",
                "max_out_degree",
                "depth",
                "n_perm_samples",
                "n_unique_canon",
                "n_distinct_fingerprints",
                "estimated_RF",
                "canon_invariant",
                "k_factorial",
                "canonical_len",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(gallery_path, "w") as f:
        f.write("\n".join(gallery_lines))

    log.info("Wrote: %s", csv_path)
    log.info("Wrote: %s", manifest_path)
    log.info("Wrote: %s", gallery_path)
    log.info("Figures: %s/*.png", fig_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--perm-samples",
        type=int,
        default=200,
        help="Permutations sampled per DAG to estimate RF (default 200).",
    )
    p.add_argument(
        "--grid",
        type=str,
        default="default",
        choices=list(_GRIDS.keys()),
        help="Which grid preset to render: 'default' (16 cells, k=6..15) or "
        "'hard' (8 cells, k=18..30, designed for sweet-spot search).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir, args.seed, args.perm_samples, grid_name=args.grid)


if __name__ == "__main__":
    main()
