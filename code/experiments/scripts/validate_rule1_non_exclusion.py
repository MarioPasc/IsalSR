"""Empirically validate the Rule 1 non-exclusion property across all BINARY_OPS.

Rule 1 (first-operand eligibility): a binary-op candidate c with at least one
recorded in-neighbour is eligible from acting pointer u only if
ordered_inputs(c)[0] == u (u would become the base/minuend/dividend operand).

Implementation location: canonical.py, lines 607-614 (_step / V branch) and
684-691 (_step / v branch); mirrored in _fast_step at lines 991-998 and
1060-1067.

Prose-vs-implementation divergence (finding):
    The manuscript prose names only "Pow node" as the Rule 1 target.
    The implementation applies the predicate to ALL BINARY_OPS = {SUB, DIV, POW}
    (the ``BINARY_OPS`` frozenset in node_types.py).  Table 3 caption says
    "binary non-commutative node", which matches the implementation.
    The prose should be corrected to say "binary non-commutative node (SUB, DIV, POW)".

Population design (three phases, per-op reporting):

Phase 2 -- Direct construction (default 1 500 DAGs per op x 3 ops = 4 500 total):
    For each op in {POW, SUB, DIV}, a node of that type is created with its first
    in-edge from x0 and its second in-edge from a different variable x_j.
    By construction, count_rule1_exclusions_per_op[op] >= 1 for every DAG.
    This guarantees non-vacuous testing: Rule 1 fires at least once per DAG,
    yet fast_canonical_string must still succeed.

Phase 3 -- String-based random population (default 15 000 strings):
    Random strings over a biased alphabet where V^/v^, V-/v-, and V//v/ each
    appear 6x as often as other V/v tokens.  A DAG qualifies if it contains any
    of {POW, SUB, DIV} and satisfies the reachability precondition.  Per-op
    stats are accumulated independently (a DAG containing both POW and SUB
    contributes to both op counters).

Backend: ``backend='cpp'`` is used explicitly when the C++ extension is
available, so Python backtracking cost does not create spurious timeouts.

Usage:
    python -m experiments.scripts.validate_rule1_non_exclusion --out result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from isalsr.core.canonical import CanonicalTimeoutError, fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NodeType
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger(__name__)

# The three non-commutative binary ops that Rule 1 governs.
BINARY_OP_TYPES: tuple[NodeType, ...] = (NodeType.POW, NodeType.SUB, NodeType.DIV)

# ---------------------------------------------------------------------------
# Biased random alphabet for Phase 3.
# V^/v^, V-/v-, V//v/ are each weighted 6x over other V/v tokens so that each
# of the three binary op types appears frequently in the generated population.
# ---------------------------------------------------------------------------
_SINGLE_TOKENS: list[str] = list("NPnpCc")
_BINARY_WEIGHT: list[str] = ["V^", "v^", "V-", "v-", "V/", "v/"] * 6  # 36 tokens
# Other V/v tokens — exclude ^, -, / since those are in _BINARY_WEIGHT.
_OTHER_V: list[str] = [f"{p}{lbl}" for p in "Vv" for lbl in "+*scelragik"]  # 22 tokens
BIASED_ALPHABET: list[str] = _SINGLE_TOKENS + _BINARY_WEIGHT + _OTHER_V  # 64 total

# Extra ops for direct-construction DAGs (non-binary so they don't add Rule 1 pairs).
_EXTRA_OPS: list[NodeType] = [
    NodeType.ADD,
    NodeType.MUL,
    NodeType.SIN,
    NodeType.COS,
    NodeType.NEG,
    NodeType.ABS,
]


# ---------------------------------------------------------------------------
# Backend detection — prefer C++ to avoid Python backtracking cost confound.
# ---------------------------------------------------------------------------
def _detect_backend() -> str:
    """Return ``'cpp'`` if the C++ extension is importable, else ``'python'``."""
    try:
        from isalsr.core import _native  # type: ignore[attr-defined]

        del _native
        return "cpp"
    except ImportError:
        return "python"


_EFFECTIVE_BACKEND: str = _detect_backend()


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------


def count_rule1_exclusions_per_op(dag: LabeledDAG) -> dict[NodeType, int]:
    """Count Rule 1 excluded (parent, candidate) pairs, broken out per op type.

    For each binary-op node c with at least one recorded in-neighbour, count
    the parents u of c for which ordered_inputs(c)[0] != u, grouped by c's
    label (POW, SUB, or DIV).

    Args:
        dag: The labeled DAG to analyse.

    Returns:
        Dict mapping each of {POW, SUB, DIV} to its exclusion count.
        A count of 0 for an op means Rule 1 did not restrict that op in this DAG.
    """
    counts: dict[NodeType, int] = {NodeType.POW: 0, NodeType.SUB: 0, NodeType.DIV: 0}
    for c in range(dag.node_count):
        label = dag.node_label_unchecked(c)
        if label not in BINARY_OPS:
            continue
        inputs = dag.ordered_inputs(c)
        if not inputs:
            continue
        first_operand = inputs[0]
        for u in dag.in_neighbors_raw(c):
            if u != first_operand:
                counts[label] += 1
    return counts


def count_rule1_exclusions(dag: LabeledDAG) -> int:
    """Count total Rule 1 excluded (parent, candidate) pairs across all binary ops.

    Convenience wrapper over ``count_rule1_exclusions_per_op`` that returns the
    sum across {POW, SUB, DIV}.

    Args:
        dag: The labeled DAG to analyse.

    Returns:
        Total count of excluded (parent, candidate) pairs.
    """
    return sum(count_rule1_exclusions_per_op(dag).values())


def has_pow_node(dag: LabeledDAG) -> bool:
    """Return True iff the DAG contains at least one POW node.

    Args:
        dag: The labeled DAG to inspect.

    Returns:
        True if any node has label NodeType.POW.
    """
    return any(dag.node_label_unchecked(i) == NodeType.POW for i in range(dag.node_count))


def has_op_node(dag: LabeledDAG, op: NodeType) -> bool:
    """Return True iff the DAG contains at least one node with the given label.

    Args:
        dag: The labeled DAG to inspect.
        op: The node type to search for.

    Returns:
        True if any node has the given label.
    """
    return any(dag.node_label_unchecked(i) == op for i in range(dag.node_count))


def satisfies_reachability(dag: LabeledDAG) -> bool:
    """Return True iff every non-VAR node is reachable from some VAR node.

    Implements the Round-Trip Fidelity reachability precondition via BFS
    from all VAR source nodes following out-edges.

    Args:
        dag: The labeled DAG to inspect.

    Returns:
        True iff the reachability precondition holds for all non-VAR nodes.
    """
    n = dag.node_count
    visited: bytearray = bytearray(n)
    queue: list[int] = []
    for i in range(n):
        if dag.node_label_unchecked(i) == NodeType.VAR:
            visited[i] = 1
            queue.append(i)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nb in dag.out_neighbors_raw(node):
            if not visited[nb]:
                visited[nb] = 1
                queue.append(nb)
    return all(bool(visited[i]) or dag.node_label_unchecked(i) == NodeType.VAR for i in range(n))


# ---------------------------------------------------------------------------
# DAG construction helpers
# ---------------------------------------------------------------------------


def build_exclusion_dag(
    m: int,
    extra_nodes: int,
    op_type: NodeType,
    rng: random.Random,
) -> LabeledDAG:
    """Build a DAG guaranteed to have count_rule1_exclusions_per_op[op_type] >= 1.

    Structure: m VAR nodes + 1 node of type ``op_type`` (first in-edge = x0 as
    first operand, second in-edge = x_j as second operand, j != 0) +
    ``extra_nodes`` unary or variadic nodes connected to random VAR nodes (for
    structural diversity).

    All non-VAR nodes are connected to VAR nodes, so the reachability precondition
    is satisfied by construction.  count_rule1_exclusions_per_op[op_type] >= 1
    because (x_j, op_node) is always an excluded pair: ordered_inputs[0] = x0
    but x_j != x0 is also a parent.

    Args:
        m: Number of input variables (must be >= 2).
        extra_nodes: Number of additional non-binary nodes (0-4 recommended).
        op_type: Binary op type to construct (must be in BINARY_OPS).
        rng: Random number generator for selecting x_j and extra ops.

    Returns:
        A labeled DAG with op_type node having two distinct variable parents.
    """
    max_nodes = m + 1 + extra_nodes + 1
    dag = LabeledDAG(max_nodes)

    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)

    # op_type node: first operand = x0, second operand = x_j (j in [1, m-1]).
    second_var = rng.randint(1, m - 1)
    op_idx = dag.node_count
    dag.add_node(op_type)
    dag.add_edge(0, op_idx)  # x0 -> op  (first: ordered_inputs[0] = 0)
    dag.add_edge(second_var, op_idx)  # x_j -> op (second: excluded by Rule 1)

    for _ in range(extra_nodes):
        extra_op = rng.choice(_EXTRA_OPS)
        parent = rng.randint(0, m - 1)
        node_idx = dag.node_count
        dag.add_node(extra_op)
        dag.add_edge(parent, node_idx)  # connected to VAR -> reachable

    return dag


def build_guaranteed_dags() -> list[tuple[LabeledDAG, int]]:
    """Construct small DAGs with count_rule1_exclusions > 0 for all three op types.

    Returns:
        List of (dag, num_variables) pairs, two per op type.
    """
    results: list[tuple[LabeledDAG, int]] = []

    # POW minimal: x0 --(base)--> p, x1 --(exp)--> p
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.POW)
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    results.append((dag, 2))

    # POW larger: ADD(x0,x1) base, x2 exp
    dag = LabeledDAG(5)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.VAR, var_index=2)
    dag.add_node(NodeType.ADD)
    dag.add_node(NodeType.POW)
    dag.add_edge(0, 3)
    dag.add_edge(1, 3)
    dag.add_edge(3, 4)  # a -> p (base, first)
    dag.add_edge(2, 4)  # x2 -> p (exp, second)
    results.append((dag, 3))

    # SUB minimal: x0 --(minuend)--> s, x1 --(subtrahend)--> s
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.SUB)
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    results.append((dag, 2))

    # DIV minimal: x0 --(numerator)--> d, x1 --(denominator)--> d
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.DIV)
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    results.append((dag, 2))

    return results


# ---------------------------------------------------------------------------
# Per-op stat accumulation helpers
# ---------------------------------------------------------------------------

# Type alias for the per-op stats dict (one entry per NodeType in BINARY_OP_TYPES).
_OpStats = dict[str, int]


def _make_per_op_stats() -> dict[NodeType, _OpStats]:
    """Create a zeroed per-op stats dict for all three binary op types."""
    return {
        op: {
            "n_tested": 0,
            "n_with_exclusions": 0,
            "n_canonicalized": 0,
            "n_roundtrip_ok": 0,
            "n_runtime_failures": 0,
            "n_timeout_failures": 0,
        }
        for op in BINARY_OP_TYPES
    }


def _accumulate(
    per_op: dict[NodeType, _OpStats],
    dag: LabeledDAG,
    per_op_excl: dict[NodeType, int],
    c_ok: bool,
    rt_ok: bool,
    is_to: bool,
) -> None:
    """Update per-op stats for one DAG.

    A DAG contributes to the stats for op X if it contains at least one node
    of type X.  A single DAG may contribute to multiple op categories.

    Args:
        per_op: Mutable stats dict updated in-place.
        dag: The DAG that was processed.
        per_op_excl: Exclusion counts per op for this DAG.
        c_ok: True iff fast_canonical_string succeeded.
        rt_ok: True iff the round-trip check passed (only if c_ok).
        is_to: True iff the failure was a CanonicalTimeoutError.
    """
    for op in BINARY_OP_TYPES:
        if not has_op_node(dag, op):
            continue
        per_op[op]["n_tested"] += 1
        if per_op_excl[op] > 0:
            per_op[op]["n_with_exclusions"] += 1
        if c_ok:
            per_op[op]["n_canonicalized"] += 1
            if rt_ok:
                per_op[op]["n_roundtrip_ok"] += 1
        elif is_to:
            per_op[op]["n_timeout_failures"] += 1
        else:
            per_op[op]["n_runtime_failures"] += 1


# ---------------------------------------------------------------------------
# Single-DAG check
# ---------------------------------------------------------------------------


def check_dag(
    dag: LabeledDAG,
    m: int,
    timeout: float,
    backend: str,
) -> tuple[dict[NodeType, int], bool, bool, bool]:
    """Compute per-op Rule 1 exclusion counts and run round-trip check.

    Args:
        dag: Labeled DAG satisfying the reachability precondition.
        m: Number of input variables.
        timeout: Per-DAG time budget for fast_canonical_string.
        backend: Canonicaliser backend (``'cpp'`` or ``'python'``).

    Returns:
        (per_op_excl, canon_ok, roundtrip_ok, is_timeout) where:
            per_op_excl: count_rule1_exclusions_per_op(dag).
            canon_ok: fast_canonical_string completed without error.
            roundtrip_ok: dag.is_isomorphic(S2D(fcs, m)) -- only meaningful
                when canon_ok is True.
            is_timeout: True iff the failure was CanonicalTimeoutError.
                A RuntimeError ("no valid operation found") would indicate
                a potential Rule 1 non-exclusion violation.
    """
    per_op_excl = count_rule1_exclusions_per_op(dag)

    try:
        fcs = fast_canonical_string(dag, timeout=timeout, backend=backend)
    except CanonicalTimeoutError:
        return per_op_excl, False, False, True
    except RuntimeError:
        return per_op_excl, False, False, False

    try:
        decoded = StringToDAG(fcs, num_variables=m).run()
        rt_ok: bool = dag.is_isomorphic(decoded)
    except Exception:  # noqa: BLE001 - any decode failure is a round-trip failure
        rt_ok = False

    return per_op_excl, True, rt_ok, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Rule 1 non-exclusion empirical validation across all BINARY_OPS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=15_000,
        help="Number of random strings for Phase 3 (default: 15000).",
    )
    parser.add_argument(
        "--n-direct-per-op",
        type=int,
        default=1_500,
        help="Directly-constructed exclusion DAGs per op in Phase 2 (default: 1500).",
    )
    parser.add_argument(
        "--num-variables",
        type=int,
        default=2,
        help="Input variable count for Phase 3 random strings (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Per-DAG canonicalisation timeout in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write JSON summary to this file path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backend = _EFFECTIVE_BACKEND
    if backend != "cpp":
        log.warning("C++ backend unavailable; using Python backend (may produce timeouts).")
    log.info(
        "Rule 1 non-exclusion: seed=%d n_direct_per_op=%d n_strings=%d m=%d "
        "timeout=%.1fs backend=%s",
        args.seed,
        args.n_direct_per_op,
        args.n,
        args.num_variables,
        args.timeout,
        backend,
    )

    rng = random.Random(args.seed)
    m: int = args.num_variables

    per_op = _make_per_op_stats()
    n_total_tested: int = 0
    n_canonicalized: int = 0
    n_roundtrip_ok: int = 0
    failure_examples: list[dict[str, Any]] = []

    # ---- Phase 1: hand-crafted guaranteed DAGs (all three op types) -----------
    log.info("Phase 1: 4 hand-crafted guaranteed DAGs (POW x2, SUB x1, DIV x1)")
    for g_dag, g_m in build_guaranteed_dags():
        per_op_excl, c_ok, rt_ok, is_to = check_dag(g_dag, g_m, args.timeout, backend)
        n_total_tested += 1
        if c_ok:
            n_canonicalized += 1
            if rt_ok:
                n_roundtrip_ok += 1
        _accumulate(per_op, g_dag, per_op_excl, c_ok, rt_ok, is_to)
        if not c_ok and len(failure_examples) < 10:
            failure_examples.append({"source": "guaranteed", "is_timeout": is_to})

    # ---- Phase 2: directly-constructed exclusion DAGs (all three op types) ----
    n_direct = args.n_direct_per_op
    log.info(
        "Phase 2: %d DAGs per op x 3 ops = %d total (100%% exclusion rate by construction)",
        n_direct,
        n_direct * 3,
    )
    for op_type in BINARY_OP_TYPES:
        for _idx in range(n_direct):
            m_local = rng.choice([2, 3, 4])
            extra = rng.randint(0, 4)
            dag = build_exclusion_dag(m_local, extra, op_type, rng)

            per_op_excl, c_ok, rt_ok, is_to = check_dag(dag, m_local, args.timeout, backend)
            n_total_tested += 1
            if c_ok:
                n_canonicalized += 1
                if rt_ok:
                    n_roundtrip_ok += 1
            _accumulate(per_op, dag, per_op_excl, c_ok, rt_ok, is_to)

            if not c_ok and len(failure_examples) < 10:
                failure_examples.append(
                    {"source": "direct", "op": op_type.name, "m": m_local, "is_timeout": is_to}
                )

        log.info(
            "Phase 2 %s done: n_with_excl=%d / %d, runtime_fail=%d",
            op_type.name,
            per_op[op_type]["n_with_exclusions"],
            per_op[op_type]["n_tested"],
            per_op[op_type]["n_runtime_failures"],
        )

    # ---- Phase 3: string-based random population (all three op types) --------
    log.info("Phase 3: up to %d random strings (V^/V-/V/ equally biased 6x)", args.n)
    for i in range(args.n):
        word = "".join(rng.choice(BIASED_ALPHABET) for _ in range(rng.randint(4, 12)))
        try:
            dag = StringToDAG(word, num_variables=m).run()
        except Exception:  # noqa: BLE001 - malformed random strings expected
            continue

        # Filter: at least one binary op present AND reachability satisfied.
        if not any(has_op_node(dag, op) for op in BINARY_OP_TYPES):
            continue
        if not satisfies_reachability(dag):
            continue

        per_op_excl, c_ok, rt_ok, is_to = check_dag(dag, m, args.timeout, backend)
        n_total_tested += 1
        if c_ok:
            n_canonicalized += 1
            if rt_ok:
                n_roundtrip_ok += 1
        _accumulate(per_op, dag, per_op_excl, c_ok, rt_ok, is_to)

        if not c_ok and len(failure_examples) < 10:
            failure_examples.append({"source": "string", "word": word, "is_timeout": is_to})

        if (i + 1) % 5_000 == 0:
            log.info(
                "Phase 3: %d/%d strings | total_tested=%d | "
                "POW excl>0=%d | SUB excl>0=%d | DIV excl>0=%d | "
                "runtime_fail POW/SUB/DIV=%d/%d/%d",
                i + 1,
                args.n,
                n_total_tested,
                per_op[NodeType.POW]["n_with_exclusions"],
                per_op[NodeType.SUB]["n_with_exclusions"],
                per_op[NodeType.DIV]["n_with_exclusions"],
                per_op[NodeType.POW]["n_runtime_failures"],
                per_op[NodeType.SUB]["n_runtime_failures"],
                per_op[NodeType.DIV]["n_runtime_failures"],
            )

    # ---- Summary ---------------------------------------------------------------
    vacuous_ops = [op.name for op in BINARY_OP_TYPES if per_op[op]["n_with_exclusions"] == 0]
    vacuous_warning = (
        f"WARNING: Rule 1 never excluded any candidate for: {vacuous_ops}" if vacuous_ops else ""
    )
    if vacuous_warning:
        log.warning(vacuous_warning)

    log.info("=== RESULTS ===")
    log.info("Backend: %s", backend)
    log.info("Total DAGs tested : %d", n_total_tested)
    log.info("Total canonicalized: %d", n_canonicalized)
    log.info("Total round-trip OK: %d", n_roundtrip_ok)
    log.info("Per-op breakdown:")
    for op in BINARY_OP_TYPES:
        s = per_op[op]
        log.info(
            "  %-3s  tested=%d  excl>0=%d  canonicalized=%d  rt_ok=%d  timeout=%d  runtime_fail=%d",
            op.name,
            s["n_tested"],
            s["n_with_exclusions"],
            s["n_canonicalized"],
            s["n_roundtrip_ok"],
            s["n_timeout_failures"],
            s["n_runtime_failures"],
        )

    summary: dict[str, Any] = {
        "seed": args.seed,
        "backend": backend,
        "n_direct_per_op": args.n_direct_per_op,
        "n_strings_phase3": args.n,
        "num_variables_phase3": m,
        "n_total_tested": n_total_tested,
        "n_canonicalized": n_canonicalized,
        "n_roundtrip_ok": n_roundtrip_ok,
        "per_op": {
            op.name: {
                "n_tested": per_op[op]["n_tested"],
                "n_with_exclusions": per_op[op]["n_with_exclusions"],
                "n_canonicalized": per_op[op]["n_canonicalized"],
                "n_roundtrip_ok": per_op[op]["n_roundtrip_ok"],
                "n_runtime_failures_rule1_attributable": per_op[op]["n_runtime_failures"],
                "n_timeout_failures": per_op[op]["n_timeout_failures"],
            }
            for op in BINARY_OP_TYPES
        },
        "vacuous_test_warning": vacuous_warning,
        "prose_vs_implementation_divergence": (
            "Rule 1 prose says 'Pow node'; the implementation (canonical.py "
            "BINARY_OPS filter) covers {SUB, DIV, POW}.  "
            "Table 3 caption ('binary non-commutative node') matches the "
            "implementation.  The prose should read 'binary non-commutative "
            "node (SUB, DIV, POW)' to be consistent."
        ),
        "failure_examples": failure_examples[:10],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        log.info("JSON summary written to %s", out_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
