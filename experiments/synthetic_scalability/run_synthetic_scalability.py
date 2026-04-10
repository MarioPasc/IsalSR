"""Synthetic Scalability Experiment -- rho and canonicalization time vs. k.

Generates random expression DAGs via the Lample-Charton (2020) random
unary-binary tree method, permutes internal node IDs exhaustively (k<=11
with default max_perms=50M) or by sampling, canonicalizes each permutation with
``fast_canonical_string(mode="wl_only")``, and measures the empirical
reduction factor rho = N_perm / N_unique and wall-clock canonicalization time.

Scientific rationale: isolates the intrinsic combinatorial reduction from
canonicalization, free from the bias of any particular SR method's search.

References:
    - Lample & Charton (2020). Deep Learning for Symbolic Mathematics. ICLR.
    - Flajolet & Sedgewick (2009). Analytic Combinatorics. Cambridge UP.
    - Gutjahr (1991). Uniform random generation of expressions. Computing 47.

Reproducibility note:
    Set ``PYTHONHASHSEED=0`` for fully deterministic canonical_len across
    runs.  Python's hash randomisation changes WL hash ordering in
    fast_canonical_string, which may select a different (but equivalent)
    canonical string.  rho and all other structural columns are always
    reproducible regardless of PYTHONHASHSEED.

Usage (local):
    PYTHONHASHSEED=0 python experiments/synthetic_scalability/run_synthetic_scalability.py \\
        --output-dir /tmp/synth_test \\
        --n-expr 5 --max-perms 50 \\
        --k-values "1,2,3,4" --m-values "1,2"

Author: Mario Pascual Gonzalez (mpascual@uma.es)
Date: 2026-04-10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Path setup for non-installed runs (project convention).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from isalsr.core.canonical import CanonicalTimeoutError, fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator set (commutative decomposition, matching main experiments).
# ---------------------------------------------------------------------------
BINARY_OPS: list[NodeType] = [NodeType.ADD, NodeType.MUL, NodeType.POW]
UNARY_OPS: list[NodeType] = [
    NodeType.SIN,
    NodeType.COS,
    NodeType.EXP,
    NodeType.LOG,
    NodeType.NEG,
    NodeType.INV,
]
ALL_OPS: list[NodeType] = BINARY_OPS + UNARY_OPS

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
FIELDNAMES: list[str] = [
    "k",
    "m",
    "expr_id",
    "seed",
    "n_nodes",
    "n_edges",
    "n_perms",
    "n_unique_canonicals",
    "rho",
    "rho_over_kfact",
    "canonical_len",
    "mean_canon_time_s",
    "std_canon_time_s",
    "total_canon_time_s",
    "timeout_count",
]


# ===================================================================
# Random expression tree generation (Lample-Charton 2020)
# ===================================================================


@dataclass
class ExprTreeNode:
    """Node in a random expression tree (before conversion to LabeledDAG)."""

    node_type: NodeType
    children: list[ExprTreeNode] = field(default_factory=list)
    var_index: int | None = None


def generate_random_expr_tree(
    k: int,
    m: int,
    binary_ops: list[NodeType],
    unary_ops: list[NodeType],
    rng: np.random.Generator,
) -> ExprTreeNode:
    """Generate a random expression tree with exactly *k* internal nodes.

    Uses the "grow" method from Lample & Charton (2020): maintain a list
    of open leaf slots, iteratively replace one with a randomly chosen
    operator (unary or binary), and create the appropriate number of new
    open slots. After all *k* operators are placed, label each remaining
    open leaf with a variable drawn uniformly from {x_0, ..., x_{m-1}}.

    The open-slot count starts at 1 and is monotonically non-decreasing
    (unary: +0, binary: +1), so any operator is valid at every step —
    no constraint-aware sampling is required.

    Args:
        k: Number of internal (operator) nodes.
        m: Number of input variables.
        binary_ops: Allowed binary operators.
        unary_ops: Allowed unary operators.
        rng: NumPy random generator for reproducibility.

    Returns:
        Root of the expression tree.
    """
    all_ops = binary_ops + unary_ops
    n_all = len(all_ops)
    n_binary = len(binary_ops)

    # Start with a single open-leaf placeholder (the root).
    root = ExprTreeNode(node_type=NodeType.VAR)  # temporary
    open_leaves: list[ExprTreeNode] = [root]

    for _ in range(k):
        # Pick a random open leaf to expand.
        idx = int(rng.integers(0, len(open_leaves)))
        node = open_leaves[idx]
        # Remove it from open list (swap-and-pop for O(1)).
        open_leaves[idx] = open_leaves[-1]
        open_leaves.pop()

        # Choose a random operator.
        op_idx = int(rng.integers(0, n_all))
        op = all_ops[op_idx]
        node.node_type = op
        node.var_index = None

        # Create children.
        n_children = 2 if op_idx < n_binary else 1
        for _ in range(n_children):
            child = ExprTreeNode(node_type=NodeType.VAR)  # placeholder
            node.children.append(child)
            open_leaves.append(child)

    # Assign variables to remaining open leaves.
    for leaf in open_leaves:
        leaf.var_index = int(rng.integers(0, m))

    return root


def _count_internal(node: ExprTreeNode) -> int:
    """Count internal (non-leaf) nodes in the tree."""
    if node.var_index is not None and not node.children:
        return 0
    return 1 + sum(_count_internal(c) for c in node.children)


# ===================================================================
# Tree -> LabeledDAG conversion
# ===================================================================


def tree_to_labeled_dag(root: ExprTreeNode, m: int) -> LabeledDAG:
    """Convert an expression tree to a LabeledDAG, reusing VAR nodes.

    Multiple leaves referencing the same variable share a single VAR node,
    making the result a proper DAG (not necessarily a tree).

    Edge direction follows IsalSR convention: child -> parent (data flow).
    For binary operators, children are added in tree order (children[0]
    first), preserving operand order via ``_input_order`` (critical for
    non-commutative ops like POW).

    Args:
        root: Root of the expression tree.
        m: Number of input variables (VAR nodes 0..m-1 are pre-created).

    Returns:
        A LabeledDAG with m VAR nodes + k internal nodes.
    """
    k = _count_internal(root)
    dag = LabeledDAG(max_nodes=m + k)

    # Pre-create VAR nodes at indices 0..m-1.
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)

    def _build(node: ExprTreeNode) -> int:
        """Recursive DFS: returns the dag node ID for *node*."""
        # Leaf: reuse existing VAR node.
        if node.var_index is not None and not node.children:
            return node.var_index

        # Internal node.
        dag_id = dag.add_node(node.node_type)
        for child in node.children:
            child_id = _build(child)
            # Edge: child provides input to this node (data flow).
            # Use add_edge (checked) because binary ops whose children
            # map to the same VAR node would create a duplicate
            # (source, target) pair.  add_edge_unchecked would corrupt
            # _edge_count vs. the set-based adjacency.
            dag.add_edge(child_id, dag_id)
        return dag_id

    _build(root)
    return dag


# ===================================================================
# Permutation generators (O(1) memory)
# ===================================================================


def _sampled_perms_gen(k: int, n: int, seed: int) -> Iterator[list[int]]:
    """Yield *n* random permutations of range(k) via Fisher-Yates.

    Streaming version: O(k) memory per permutation, never materialises
    the full list.  Reproduces the same sequence as
    ``isalsr.core.permutations.random_permutations`` for the same seed.
    """
    rng = random.Random(seed)
    base = list(range(k))
    for _ in range(n):
        p = base.copy()
        for i in range(k - 1, 0, -1):
            j = rng.randint(0, i)
            p[i], p[j] = p[j], p[i]
        yield p


# ===================================================================
# Per-expression analysis
# ===================================================================


def run_single_expression(
    dag: LabeledDAG,
    k: int,
    m: int,
    expr_id: int,
    seed: int,
    max_perms: int,
    timeout: float,
) -> dict[str, object]:
    """Permute, canonicalize, and measure rho for one expression DAG.

    Memory-efficient: permutations are streamed (never materialised) and
    timing statistics are computed via Welford's online algorithm, so peak
    memory is O(k) regardless of the number of permutations.

    Args:
        dag: The expression DAG (m VAR + k internal nodes).
        k: Number of internal nodes.
        m: Number of input variables.
        expr_id: Expression index within the (k, m) cell.
        seed: Per-expression seed (for permutation sampling).
        max_perms: Maximum permutations to test.
        timeout: Canonicalization timeout in seconds.

    Returns:
        Row dict matching FIELDNAMES.
    """
    k_factorial = math.factorial(k)

    # Streaming permutation iterator (O(1) memory).
    exhaustive = k_factorial <= max_perms
    if exhaustive:
        perm_iter: Iterator[tuple[int, ...]] | Iterator[list[int]]
        perm_iter = itertools.permutations(range(k))
        n_perms = k_factorial
    else:
        perm_iter = _sampled_perms_gen(k, max_perms, seed)
        n_perms = max_perms

    canonical_set: set[str] = set()
    first_canonical: str | None = None
    timeout_count = 0

    # Welford's online algorithm for mean/variance of timing.
    w_count = 0
    w_mean = 0.0
    w_m2 = 0.0
    w_total = 0.0

    for perm in perm_iter:
        dag_p = permute_internal_nodes(dag, list(perm))

        t0 = time.perf_counter()
        try:
            canon: str | None = fast_canonical_string(dag_p, timeout=timeout, mode="wl_only")
        except CanonicalTimeoutError:
            timeout_count += 1
            canon = None
        dt = time.perf_counter() - t0

        # Welford update (includes both successful and timed-out calls).
        w_count += 1
        delta = dt - w_mean
        w_mean += delta / w_count
        w_m2 += delta * (dt - w_mean)
        w_total += dt

        if canon is not None:
            canonical_set.add(canon)
            if first_canonical is None:
                first_canonical = canon

    n_unique = len(canonical_set)
    successful = n_perms - timeout_count

    if n_unique > 0:
        rho = successful / n_unique
        rho_over_kfact = rho / k_factorial
    else:
        rho = float("nan")
        rho_over_kfact = float("nan")

    mean_t = w_mean if w_count > 0 else float("nan")
    std_t = math.sqrt(w_m2 / (w_count - 1)) if w_count >= 2 else 0.0

    return {
        "k": k,
        "m": m,
        "expr_id": expr_id,
        "seed": seed,
        "n_nodes": dag.node_count,
        "n_edges": dag.edge_count,
        "n_perms": n_perms,
        "n_unique_canonicals": n_unique,
        "rho": rho,
        "rho_over_kfact": rho_over_kfact,
        "canonical_len": len(first_canonical) if first_canonical else -1,
        "mean_canon_time_s": mean_t,
        "std_canon_time_s": std_t,
        "total_canon_time_s": w_total,
        "timeout_count": timeout_count,
    }


# ===================================================================
# CSV I/O
# ===================================================================


def write_csv_rows(
    path: str,
    rows: list[dict[str, object]],
    *,
    append: bool = False,
) -> None:
    """Write rows to CSV, optionally appending."""
    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


# ===================================================================
# Cell runner (one (k, m) combination)
# ===================================================================


def run_cell(
    k: int,
    m: int,
    n_expr: int,
    max_perms: int,
    timeout: float,
    global_seed: int,
) -> list[dict[str, object]]:
    """Run all expressions for a single (k, m) cell.

    Args:
        k: Number of internal nodes.
        m: Number of input variables.
        n_expr: Number of expressions to generate.
        max_perms: Max permutations per expression.
        timeout: Canonicalization timeout in seconds.
        global_seed: Global seed for reproducibility.

    Returns:
        List of result rows (one per expression).
    """
    k_factorial = math.factorial(k)
    n_perms_actual = min(k_factorial, max_perms)
    log.info(
        "Cell k=%d, m=%d: %d expressions x %d perms (%s)",
        k,
        m,
        n_expr,
        n_perms_actual,
        "exhaustive" if k_factorial <= max_perms else "sampled",
    )

    rows: list[dict[str, object]] = []
    for expr_id in range(n_expr):
        # Deterministic per-expression seed (collision-free for k<=99,
        # m<=9, expr_id<=999).
        expr_seed = global_seed + k * 10_000 + m * 1_000 + expr_id

        rng = np.random.default_rng(expr_seed)
        tree = generate_random_expr_tree(k, m, BINARY_OPS, UNARY_OPS, rng)
        dag = tree_to_labeled_dag(tree, m)

        row = run_single_expression(dag, k, m, expr_id, expr_seed, max_perms, timeout)
        rows.append(row)

        if (expr_id + 1) % 50 == 0:
            log.info("  k=%d, m=%d: %d/%d expressions done", k, m, expr_id + 1, n_expr)

    return rows


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic scalability: rho and canon time vs. k",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output CSV and metadata JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--n-expr",
        type=int,
        default=200,
        help="Expressions per (k, m) cell (default: 200).",
    )
    parser.add_argument(
        "--max-perms",
        type=int,
        default=50_000_000,
        help="Max permutations per expression (default: 50000000). Exhaustive up to k=11.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Canonicalization timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,25",
        help="Comma-separated k values (default: 1-12,14,16,18,20,25).",
    )
    parser.add_argument(
        "--m-values",
        type=str,
        default="1,2,3",
        help="Comma-separated m values (default: 1,2,3).",
    )
    return parser.parse_args()


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Entry point."""
    args = parse_args()
    k_values = [int(x) for x in args.k_values.split(",")]
    m_values = [int(x) for x in args.m_values.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    # Fragment mode (single cell, typical for SLURM array) vs. full mode.
    if len(k_values) == 1 and len(m_values) == 1:
        csv_path = os.path.join(args.output_dir, f"synth_k{k_values[0]}_m{m_values[0]}.csv")
    else:
        csv_path = os.path.join(args.output_dir, "synthetic_scalability_results.csv")

    log.info("Output CSV: %s", csv_path)
    log.info("k values: %s, m values: %s", k_values, m_values)
    log.info(
        "n_expr=%d, max_perms=%d, timeout=%.1fs, seed=%d",
        args.n_expr,
        args.max_perms,
        args.timeout,
        args.seed,
    )

    all_rows: list[dict[str, object]] = []
    first_cell = True

    for k in k_values:
        for m in m_values:
            t_cell = time.perf_counter()
            rows = run_cell(k, m, args.n_expr, args.max_perms, args.timeout, args.seed)
            dt_cell = time.perf_counter() - t_cell

            all_rows.extend(rows)
            write_csv_rows(csv_path, rows, append=not first_cell)
            first_cell = False

            log.info(
                "Cell k=%d, m=%d done in %.1fs (%d rows written)",
                k,
                m,
                dt_cell,
                len(rows),
            )

    # --- Metadata JSON ---
    metadata: dict[str, object] = {
        "experiment": "synthetic_scalability",
        "description": (
            "Synthetic rho and canonicalization time vs. k. "
            "Random expression DAGs via Lample-Charton (2020) method."
        ),
        "operator_set": {
            "binary": [op.name for op in BINARY_OPS],
            "unary": [op.name for op in UNARY_OPS],
            "total": len(ALL_OPS),
        },
        "k_values": k_values,
        "m_values": m_values,
        "n_expr": args.n_expr,
        "max_perms": args.max_perms,
        "timeout_s": args.timeout,
        "global_seed": args.seed,
        "canonicalization_algorithm": "fast_canonical_string(mode='wl_only')",
        "tree_generation_method": "Lample-Charton random unary-binary tree",
        "total_rows": len(all_rows),
    }
    try:
        from experiments.models.hardware_info import collect_hardware_info

        metadata["hardware"] = collect_hardware_info()
    except ImportError:
        log.warning("Could not import hardware_info; skipping hardware metadata.")

    meta_path = os.path.join(args.output_dir, "synthetic_scalability_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata written to %s", meta_path)
    log.info("Done. %d total rows across %d cells.", len(all_rows), len(k_values) * len(m_values))


if __name__ == "__main__":
    main()
