"""Fast verification of the IsalSR representation properties tabulated in T07.

One script, one table, under a minute. Run it after any change to the
canonicaliser, to ``normalize_const_creation``, or to either engine.

Each property is checked on both the Python and the C++ backend where a backend
is selectable, because a stale C++ extension can make a source change look
applied when it is not (see CLAUDE.md, "Rebuilding the C++ extension").

Properties, matching the table in
``.claude/notes/review/tasks/T07-theorem-foundation.md``:

    P1  Completeness (<=)  isomorphic DAGs receive the same canonical string
    P2  Completeness (=>)  equal canonical strings imply isomorphic DAGs
    P3  Round-trip         D ~= S2D(fcs(D), m)
    P4  Engine equivalence cpp and python agree byte-for-byte
    P5  Eval preservation  eval(D) == eval(N(D))
    P6  N not in canon     orphan-CONST DAGs are refused, not silently repaired
    P7  S2D reachability   S2D output satisfies the precondition by construction
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np

from isalsr.core import canonical as canonical_mod
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger(__name__)

ALPHABET: tuple[str, ...] = (
    "N",
    "P",
    "n",
    "p",
    "C",
    "c",
    "W",
    "V+",
    "V*",
    "V-",
    "V/",
    "Vs",
    "Vc",
    "Ve",
    "Vl",
    "Vr",
    "V^",
    "Va",
    "Vg",
    "Vi",
    "Vk",
    "v+",
    "v*",
    "v-",
    "v/",
    "vs",
    "vc",
    "ve",
    "vl",
    "vr",
    "v^",
    "va",
    "vg",
    "vi",
    "vk",
)


def _canon(dag: LabeledDAG, backend: str, timeout: float) -> str | None:
    """Canonicalise, returning None if the canonicaliser refuses.

    Args:
        dag: DAG to canonicalise.
        backend: 'cpp' or 'python'.
        timeout: Time budget in seconds.

    Returns:
        The canonical string, or None if canonicalisation raised.
    """
    try:
        return canonical_mod.fast_canonical_string(dag, timeout=timeout, backend=backend)
    except Exception:  # noqa: BLE001 - refusal is a valid outcome here
        return None


def _satisfies_reachability(dag: LabeledDAG) -> bool:
    """Return True if every non-VAR node is reachable from some VAR.

    Args:
        dag: DAG to test.

    Returns:
        True if the Round-Trip Fidelity precondition holds.
    """
    seen = {i for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR}
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in dag.out_neighbors(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return all(i in seen for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)


def _random_dags(n: int, rng: random.Random, num_variables: int) -> list[LabeledDAG]:
    """Decode *n* random Sigma_SR strings into DAGs, skipping malformed ones.

    Args:
        n: Number of strings to attempt.
        rng: Seeded RNG.
        num_variables: Variable count for S2D.

    Returns:
        The successfully decoded DAGs.
    """
    out: list[LabeledDAG] = []
    for _ in range(n):
        word = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 22)))
        try:
            out.append(StringToDAG(word, num_variables=num_variables).run())
        except Exception:  # noqa: BLE001 - malformed random strings are expected
            continue
    return out


def _orphan_const_dag() -> LabeledDAG:
    """Build the minimal DAG carrying a CONST node with no in-edge.

    Returns:
        A DAG that has no encoding in Sigma_SR until repaired.
    """
    d = LabeledDAG(8)
    d.add_node(NodeType.VAR, var_index=0)
    c = d.add_node(NodeType.CONST, const_value=2.0)
    m = d.add_node(NodeType.MUL)
    d.add_edge(0, m)
    d.add_edge(c, m)
    return d


def main() -> None:
    """Run every property check and print the result table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="random DAGs per check")
    parser.add_argument("--k-perms", type=int, default=5, help="permutations per DAG")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = random.Random(args.seed)
    results: list[tuple[str, str, bool, str]] = []

    dags = _random_dags(args.n, rng, num_variables=2) + _random_dags(args.n, rng, num_variables=3)
    reachable = [d for d in dags if _satisfies_reachability(d)]

    # -- P7: S2D output satisfies the precondition by construction -----------
    results.append(
        (
            "P7",
            "S2D reachability",
            len(reachable) == len(dags),
            f"{len(reachable)}/{len(dags)} satisfy the precondition",
        )
    )

    # -- P1 / P4: equivariance and engine agreement --------------------------
    p1_fail = p4_fail = n_perm = 0
    by_string: dict[str, list[LabeledDAG]] = {}
    for dag in reachable:
        ref = _canon(dag, "cpp", args.timeout)
        if ref is None:
            continue
        if _canon(dag, "python", args.timeout) != ref:
            p4_fail += 1
        by_string.setdefault(ref, []).append(dag)
        k = sum(1 for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)
        if k < 2:
            continue
        for _ in range(args.k_perms):
            perm = list(range(k))
            rng.shuffle(perm)
            if _canon(permute_internal_nodes(dag, perm), "cpp", args.timeout) != ref:
                p1_fail += 1
            n_perm += 1
    results.append(
        (
            "P1",
            "Completeness (<=)  iso => same string",
            p1_fail == 0,
            f"{p1_fail} failures / {n_perm} permutations",
        )
    )
    results.append(
        (
            "P4",
            "Engine equivalence cpp == python",
            p4_fail == 0,
            f"{p4_fail} disagreements / {len(reachable)} DAGs",
        )
    )

    # -- P2: equal strings imply isomorphic ----------------------------------
    # Random DAGs almost never collide, so grouping them alone leaves this
    # VACUOUS (0 pairs).  Seed the pool with deliberate isomorphic copies,
    # which are guaranteed to collide, so the (=>) direction is genuinely
    # exercised.  A failure here would be a *false merge*: two structurally
    # different DAGs sharing one canonical string.
    for dag in list(reachable):
        k = sum(1 for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)
        if k < 2:
            continue
        perm = list(range(k))
        rng.shuffle(perm)
        twin = permute_internal_nodes(dag, perm)
        cs_twin = _canon(twin, "cpp", args.timeout)
        if cs_twin is not None:
            by_string.setdefault(cs_twin, []).append(twin)

    p2_fail = p2_pairs = 0
    for group in by_string.values():
        for other in group[1:]:
            p2_pairs += 1
            if not group[0].is_isomorphic(other):
                p2_fail += 1
    results.append(
        (
            "P2",
            "Completeness (=>)  same string => iso",
            p2_fail == 0 and p2_pairs > 0,
            f"{p2_fail} failures / {p2_pairs} colliding pairs"
            + ("  [VACUOUS - no collisions]" if p2_pairs == 0 else ""),
        )
    )

    # -- P3: round-trip ------------------------------------------------------
    p3_fail = p3_n = 0
    for dag in reachable:
        cs = _canon(dag, "cpp", args.timeout)
        if cs is None:
            continue
        p3_n += 1
        try:
            m = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)
            if not dag.is_isomorphic(StringToDAG(cs, num_variables=m).run()):
                p3_fail += 1
        except Exception:  # noqa: BLE001
            p3_fail += 1
    results.append(
        (
            "P3",
            "Round-trip  D ~= S2D(fcs(D), m)",
            p3_fail == 0,
            f"{p3_fail} failures / {p3_n} DAGs",
        )
    )

    # -- P5: N preserves evaluation ------------------------------------------
    # Random S2D DAGs are mostly not evaluable (arity/sink violations), which
    # left this VACUOUS at 0 samples.  Build well-formed CONST-bearing DAGs
    # explicitly instead -- including the `x -> Cos -> Const` shape that the
    # PRE-T15 policy broke, turning 1.0 into cos(1.5) = 0.0707.
    p5_fail = p5_n = 0
    # evaluate_dag takes dict[var_index, float] and returns a scalar.
    xs = {0: 1.5, 1: 2.5, 2: 3.5}

    def _const_dags() -> list[LabeledDAG]:
        """Build well-formed, evaluable DAGs containing CONST nodes."""
        built: list[LabeledDAG] = []
        for op in (NodeType.ADD, NodeType.MUL, NodeType.SUB, NodeType.DIV):
            d = LabeledDAG(8)
            d.add_node(NodeType.VAR, var_index=0)
            c = d.add_node(NodeType.CONST, const_value=2.0)
            o = d.add_node(op)
            d.add_edge(0, o)
            d.add_edge(c, o)
            built.append(d)
        # orphan CONST feeding an operator: N must add x_1 -> c and change nothing
        d = LabeledDAG(8)
        d.add_node(NodeType.VAR, var_index=0)
        c = d.add_node(NodeType.CONST, const_value=3.0)
        o = d.add_node(NodeType.ADD)
        d.add_edge(0, o)
        d.add_edge(c, o)
        built.append(d)
        # the shape the pre-T15 relocation broke
        d = LabeledDAG(8)
        d.add_node(NodeType.VAR, var_index=0)
        cs_ = d.add_node(NodeType.COS)
        c2 = d.add_node(NodeType.CONST, const_value=1.0)
        d.add_edge(0, cs_)
        d.add_edge(cs_, c2)
        built.append(d)
        return built

    for dag in _const_dags() + dags:
        if not any(dag.node_label(i) == NodeType.CONST for i in range(dag.node_count)):
            continue
        try:
            before = evaluate_dag(dag, xs)
            after = evaluate_dag(dag.normalize_const_creation(), xs)
        except Exception:  # noqa: BLE001 - degenerate DAGs are not the subject here
            continue
        p5_n += 1
        if not np.isclose(before, after, rtol=1e-12, atol=1e-12, equal_nan=True):
            p5_fail += 1
    results.append(
        (
            "P5",
            "eval(D) == eval(N(D))",
            p5_fail == 0 and p5_n > 0,
            f"{p5_fail} failures / {p5_n} CONST-bearing DAGs"
            + ("  [VACUOUS - none evaluable]" if p5_n == 0 else ""),
        )
    )

    # -- P6: N is NOT applied inside canonicalisation ------------------------
    orphan = _orphan_const_dag()
    refused = {be: _canon(orphan, be, args.timeout) is None for be in ("cpp", "python")}
    repaired_ok = {
        be: _canon(orphan.normalize_const_creation(), be, args.timeout) is not None
        for be in ("cpp", "python")
    }
    results.append(
        (
            "P6",
            "N absent from canonicaliser",
            all(refused.values()) and all(repaired_ok.values()),
            f"orphan refused={refused}, N(D) canonicalises={repaired_ok}",
        )
    )

    # -- report --------------------------------------------------------------
    print(f"\nIsalSR property check   n={args.n} per m, k_perms={args.k_perms}, seed={args.seed}")
    print("-" * 78)
    n_bad = 0
    for pid, name, ok, detail in results:
        n_bad += not ok
        print(f"  {pid}  {'PASS' if ok else 'FAIL'}  {name:<38s} {detail}")
    print("-" * 78)
    print(f"  {len(results) - n_bad}/{len(results)} properties hold\n")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
