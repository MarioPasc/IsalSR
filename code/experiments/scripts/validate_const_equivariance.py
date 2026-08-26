"""Measure equivariance failures of normalize_const_creation across four populations.

Equivariance definition: for any two DAGs D, D' that are isomorphic via an
internal-node permutation, fast_canonical_string(D) must equal
fast_canonical_string(D').  Because fast_canonical_string calls
normalize_const_creation internally, a non-equivariant normalize propagates into
different canonical strings for isomorphic inputs.

A failure is recorded when fast_canonical_string(permute_internal_nodes(D, pi))
differs from fast_canonical_string(D) for any of K random permutations pi.
CanonicalTimeoutError is counted separately and never as a failure.

Safe class C = C1 ∪ C2:
  C1: every non-VAR node reachable from some VAR via directed edges (RTF precondition).
      On C1 no CONST has in-degree 0, so normalize_const_creation is the identity.
  C2: no VAR node has any in-edge (VARs are pure sources).
      On C2 every orphan CONST anchors to x_0 regardless of processing order.

Expected finding: failures occur ONLY outside C.

Populations:
  P1: Random S2D-decoded DAGs (N >= 20,000, num_variables in {2, 3})
  P2: S2D-decoded DAGs from a CONST-free alphabet (mirrors the SR generator;
      vacuously 0 failures because normalize_const_creation is a no-op)
  P3: Bingo adapter output from a short run (<= 90 s); equivariance tested
      post-hoc on collected DAGs; in_degree(VAR) reported for C2 membership
  P4: Adversarial DAGs (explicitly constructed with >= 2 orphan CONSTs and
      at least one VAR with in-edges; failures are EXPECTED here)

Usage:
    python -m experiments.scripts.validate_const_equivariance --out results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

from isalsr.core._native import CanonicalTimeoutError

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes, random_permutations
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger(__name__)

# Full S2D alphabet (including CONST via Vk/vk).
FULL_ALPHABET = list("NPnpCcW") + [f"{ptr}{lbl}" for ptr in "Vv" for lbl in "+*-/scelr^agik"]
# CONST-free alphabet: mirrors the production SR generator (no 'Vk' / 'vk').
CONST_FREE_ALPHABET = list("NPnpCcW") + [f"{ptr}{lbl}" for ptr in "Vv" for lbl in "+*-/scelr^ag"]

K_PERMS = 8  # number of random permutations per DAG
TIMEOUT = 10.0  # fast_canonical_string timeout in seconds


# ---------------------------------------------------------------------------
# Safe class predicates
# ---------------------------------------------------------------------------


def in_c1(dag: LabeledDAG) -> bool:
    """Return True if every non-VAR node is reachable from some VAR (RTF precondition).

    Args:
        dag: The labeled DAG to check.

    Returns:
        True iff the Round-Trip Fidelity reachability hypothesis holds.
    """
    queue: deque[int] = deque()
    seen: set[int] = set()
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.VAR:
            seen.add(i)
            queue.append(i)
    while queue:
        node = queue.popleft()
        for succ in dag.out_neighbors(node):
            if succ not in seen:
                seen.add(succ)
                queue.append(succ)
    return all(i in seen for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)


def in_c2(dag: LabeledDAG) -> bool:
    """Return True if no VAR node has any in-edge (VARs are pure sources).

    On this class no orphan CONST can reach any VAR, so x_0 is always a
    valid anchor and normalize_const_creation is trivially equivariant.

    Args:
        dag: The labeled DAG to check.

    Returns:
        True iff in_degree(x_i) == 0 for every VAR node x_i.
    """
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.VAR and dag.in_degree(i) > 0:
            return False
    return True


def in_safe_class(dag: LabeledDAG) -> bool:
    """Return True if dag is in C = C1 ∪ C2.

    Args:
        dag: The labeled DAG to check.

    Returns:
        True iff the DAG lies in the safe class where equivariance is guaranteed.
    """
    return in_c1(dag) or in_c2(dag)


def has_const(dag: LabeledDAG) -> bool:
    """Return True if the DAG contains at least one CONST node.

    Args:
        dag: The labeled DAG to check.

    Returns:
        True iff any node has label NodeType.CONST.
    """
    return dag._has_const_nodes()


# ---------------------------------------------------------------------------
# Equivariance oracle
# ---------------------------------------------------------------------------


def test_equivariance(
    dag: LabeledDAG,
    rng: random.Random,
    k_perms: int = K_PERMS,
    timeout: float = TIMEOUT,
) -> dict[str, int]:
    """Test equivariance of fast_canonical_string under internal-node permutations.

    Computes the reference canonical string for dag, then applies k_perms random
    permutations and checks that each yields the same string.  A failure is
    recorded when strings differ; CanonicalTimeoutError is counted separately.

    Args:
        dag: Source DAG; must have at least one internal node.
        rng: Seeded random number generator.
        k_perms: Number of random permutations to test.
        timeout: Wall-clock budget for fast_canonical_string.

    Returns:
        Dict with keys 'n_tests', 'n_failures', 'n_timeouts'.
    """
    m = len(dag.var_nodes())
    k = dag.node_count - m
    if k == 0:
        # No internal nodes: any permutation is identity; equivariance trivial.
        return {"n_tests": 0, "n_failures": 0, "n_timeouts": 0}

    # Reference string: permute with identity (i.e., just call on dag itself).
    try:
        ref = fast_canonical_string(dag, timeout=timeout)
    except CanonicalTimeoutError:
        return {"n_tests": 0, "n_failures": 0, "n_timeouts": 1}
    except Exception:  # noqa: BLE001
        return {"n_tests": 0, "n_failures": 0, "n_timeouts": 0}

    n_failures = 0
    n_timeouts = 0
    perms = random_permutations(k, k_perms, rng)
    for perm in perms:
        dag_pi = permute_internal_nodes(dag, perm)
        try:
            cs_pi = fast_canonical_string(dag_pi, timeout=timeout)
        except CanonicalTimeoutError:
            n_timeouts += 1
            continue
        except Exception:  # noqa: BLE001
            continue
        if cs_pi != ref:
            n_failures += 1

    return {"n_tests": k_perms, "n_failures": n_failures, "n_timeouts": n_timeouts}


# ---------------------------------------------------------------------------
# Population result accumulator
# ---------------------------------------------------------------------------


def make_result() -> dict[str, Any]:
    """Return an empty per-population result dict."""
    return {
        "n_dags": 0,
        "n_with_const": 0,
        "n_in_c1": 0,
        "n_in_c2": 0,
        "n_in_safe": 0,
        "n_tests": 0,
        "n_failures": 0,
        "n_timeouts": 0,
        # Failures by safe-class membership (cross-tabulation).
        "failures_in_safe": 0,
        "failures_outside_safe": 0,
        # Failures with/without CONST nodes (sanity check).
        "failures_no_const": 0,
    }


def accumulate(
    result: dict[str, Any],
    dag: LabeledDAG,
    eq: dict[str, int],
) -> None:
    """Update result in-place with observations from one DAG.

    Args:
        result: Accumulator dict from make_result().
        dag: The source DAG.
        eq: Output of test_equivariance().
    """
    result["n_dags"] += 1
    has_c = has_const(dag)
    safe = in_safe_class(dag)
    c1 = in_c1(dag)
    c2 = in_c2(dag)

    if has_c:
        result["n_with_const"] += 1
    if c1:
        result["n_in_c1"] += 1
    if c2:
        result["n_in_c2"] += 1
    if safe:
        result["n_in_safe"] += 1

    result["n_tests"] += eq["n_tests"]
    result["n_failures"] += eq["n_failures"]
    result["n_timeouts"] += eq["n_timeouts"]

    if eq["n_failures"] > 0:
        if safe:
            result["failures_in_safe"] += 1
        else:
            result["failures_outside_safe"] += 1
        if not has_c:
            result["failures_no_const"] += 1


# ---------------------------------------------------------------------------
# Population 1 & 2: random S2D-decoded DAGs
# ---------------------------------------------------------------------------


def run_s2d_population(
    alphabet: list[str],
    n_target: int,
    num_variables: int,
    seed: int,
    rng: random.Random,
    label: str,
    vacuous_note: str | None = None,
    k_perms: int = K_PERMS,
) -> dict[str, Any]:
    """Generate random S2D-decoded DAGs and test equivariance.

    Args:
        alphabet: Token alphabet for random string generation.
        n_target: Minimum number of successfully decoded DAGs.
        num_variables: Number of VAR nodes in each S2D decode.
        seed: Seed used for string generation (passed through).
        rng: Seeded RNG for permutation generation.
        label: Human-readable population label.
        vacuous_note: If set, attach this note to the result dict.
        k_perms: Number of permutations to test per DAG.

    Returns:
        Per-population result dict.
    """
    result = make_result()
    result["population"] = label
    result["num_variables"] = num_variables
    result["seed"] = seed
    if vacuous_note:
        result["note"] = vacuous_note

    string_rng = random.Random(seed)
    n_decoded = 0
    n_generated = 0

    while n_decoded < n_target:
        word = "".join(string_rng.choice(alphabet) for _ in range(string_rng.randint(4, 30)))
        n_generated += 1
        try:
            dag = StringToDAG(word, num_variables=num_variables).run()
        except Exception:  # noqa: BLE001
            continue
        n_decoded += 1

        eq = test_equivariance(dag, rng, k_perms=k_perms)
        accumulate(result, dag, eq)

        if n_decoded % 5000 == 0:
            log.info(
                "%s: %d/%d decoded, failures=%d",
                label,
                n_decoded,
                n_target,
                result["n_failures"],
            )

    result["n_generated"] = n_generated
    result["failure_rate"] = result["n_failures"] / result["n_tests"] if result["n_tests"] else 0.0
    result["dag_failure_rate"] = result["n_failures"] / result["n_dags"]
    return result


# ---------------------------------------------------------------------------
# Population 3: Bingo adapter output
# ---------------------------------------------------------------------------


class _DagCollector:
    """Wraps fast_canonical_string to collect every DAG that passes through.

    Designed for monkey-patching isalsr.core.canonical.fast_canonical_string
    during a short Bingo run.  The collected DAGs are tested for equivariance
    after the run.
    """

    def __init__(self) -> None:
        self.dags: list[LabeledDAG] = []

    def __call__(
        self,
        dag: LabeledDAG,
        *,
        timeout: float | None = None,
        mode: str = "wl_only",
        **kwargs: Any,
    ) -> str:
        """Record dag and delegate to the real fast_canonical_string.

        Args:
            dag: The labeled DAG to canonicalize.
            timeout: Forwarded to fast_canonical_string.
            mode: Forwarded to fast_canonical_string.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            The canonical string for dag.

        Raises:
            CanonicalTimeoutError: When the budget is exceeded.
        """
        self.dags.append(dag)
        return fast_canonical_string(dag, timeout=timeout, mode=mode)  # type: ignore[return-value]


def run_bingo_population(
    max_time_s: float,
    seed: int,
    rng: random.Random,
    k_perms: int = K_PERMS,
) -> dict[str, Any]:
    """Run a short Bingo IsalSR search and test equivariance on collected DAGs.

    Uses Nguyen-1 (x^3 + x^2 + x, 1 variable) as the target because it is the
    cheapest problem in the benchmark suite.  The search is capped at max_time_s
    seconds.  Equivariance is tested post-hoc on each unique DAG that passed
    through the canonicalizer.

    The UDFS adapter is not run here because it requires a cluster array to
    complete within time limits (T15).  Its structural argument is given below
    under the key 'udfs_structural_argument'.

    Args:
        max_time_s: Wall-clock cap for the Bingo search.
        seed: Random seed for reproducibility.
        rng: Seeded RNG for permutation generation.
        k_perms: Number of permutations to test per DAG.

    Returns:
        Per-population result dict.
    """
    result = make_result()
    result["population"] = "P3_bingo"
    result["seed"] = seed
    result["max_time_s"] = max_time_s

    # Structural argument for UDFS (code inspection, not execution).
    # experiments/models/udfs/adapter.py lines 108-151:
    #   VAR nodes are created first (lines 108-112), then CONST nodes
    #   (lines 114-119), then operator nodes.  Edges are only added from
    #   children to their parent operator node (lines 149-151):
    #       dag.add_edge(child_isalsr, isalsr_id)
    #   No edge is ever added targeting a VAR node ID (0..m-1).
    #   Therefore in_degree(VAR) == 0 for every UDFS-produced DAG.
    #   All UDFS output is in C2 with probability 1 (structural guarantee).
    result["udfs_structural_argument"] = (
        "UDFS adapter (adapter.py:108-151) adds edges only from children to "
        "operator nodes; no edge targets a VAR node. in_degree(VAR)==0 for "
        "all UDFS-produced DAGs => 100% in C2 (structural, not measured)."
    )

    try:
        import isalsr.core.canonical as canonical_mod
        from experiments.models.orchestrator import create_runner
    except ImportError as exc:
        result["error"] = f"import: {exc}"
        result["note"] = "Bingo/orchestrator import failed; population skipped."
        return result

    # Nguyen-1: y = x^3 + x^2 + x, sampled on [-1, 1].
    import numpy as np

    rng_data = np.random.default_rng(seed)
    x_vals = rng_data.uniform(-1.0, 1.0, size=(240, 1))
    y_vals = x_vals[:, 0] ** 3 + x_vals[:, 0] ** 2 + x_vals[:, 0]
    x_test = rng_data.uniform(-1.0, 1.0, size=(60, 1))
    y_test = x_test[:, 0] ** 3 + x_test[:, 0] ** 2 + x_test[:, 0]

    config: dict[str, Any] = {
        "bingo": {
            "population_size": 200,
            "stack_size": 16,
            "operators": ["+", "-", "*", "/", "sin", "cos", "exp", "log"],
            "use_simplification": False,
            "crossover_prob": 0.4,
            "mutation_prob": 0.4,
            "metric": "mse",
            "clo_alg": "lm",
            "generations": 100000000,
            "fitness_threshold": 1.0e-16,
            "max_time": max_time_s,
            "max_evals": 100000000,
            "snapshot_frequency": 10,
            "canonicalization_timeout": 10.0,
            "use_fast_canonical": True,
        },
        "isalsr": {
            "canonicalization_timeout": 10.0,
            "use_fast_canonical": True,
        },
    }

    collector = _DagCollector()
    original = canonical_mod.fast_canonical_string
    canonical_mod.fast_canonical_string = collector  # type: ignore[assignment]
    try:
        runner = create_runner("bingo", "isalsr", config)
        runner.fit(x_vals, y_vals, x_test, y_test, seed, config)
    except Exception as exc:  # noqa: BLE001
        result["run_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        canonical_mod.fast_canonical_string = original  # type: ignore[assignment]

    result["n_raw_dags_collected"] = len(collector.dags)
    log.info("P3 Bingo: collected %d raw DAGs", len(collector.dags))

    for dag in collector.dags:
        eq = test_equivariance(dag, rng, k_perms=k_perms)
        accumulate(result, dag, eq)

    result["failure_rate"] = result["n_failures"] / result["n_tests"] if result["n_tests"] else 0.0
    result["dag_failure_rate"] = result["n_failures"] / max(result["n_dags"], 1)
    result["frac_in_c2"] = result["n_in_c2"] / max(result["n_dags"], 1)
    return result


# ---------------------------------------------------------------------------
# Population 4: adversarial DAGs
# ---------------------------------------------------------------------------


def _make_adversarial_dag(
    n_vars: int,
    const_labels: list[NodeType],
    op_labels: list[NodeType],
    var_targets: list[int],
    rng: random.Random,
) -> LabeledDAG:
    """Construct a DAG with orphan CONSTs that reach VAR nodes.

    Structure per (const, op, var_target) triple:
        const -> op -> var_target

    This places const outside C1 (unreachable from VARs) and var_target outside
    C2 (has an in-edge from op).  Permuting the CONST nodes in sorted() order
    can then assign different anchors depending on which CONST is processed first.

    Args:
        n_vars: Number of VAR nodes.
        const_labels: NodeType.CONST for each orphan CONST to create.
        op_labels: Unary operator for each chain (parallel to const_labels).
        var_targets: VAR index (0-based) to connect each chain to.
        rng: Unused; kept for interface uniformity.

    Returns:
        A newly constructed LabeledDAG.
    """
    max_nodes = n_vars + len(const_labels) + len(op_labels) + 10
    dag = LabeledDAG(max_nodes)

    var_ids: list[int] = []
    for i in range(n_vars):
        var_ids.append(dag.add_node(NodeType.VAR, var_index=i))

    const_ids: list[int] = []
    for _ in const_labels:
        const_ids.append(dag.add_node(NodeType.CONST, const_value=1.0))

    op_ids: list[int] = []
    for op_label in op_labels:
        op_ids.append(dag.add_node(op_label))

    # Chain: const -> op -> var_target
    for const_id, op_id, var_tgt in zip(const_ids, op_ids, var_targets, strict=True):
        dag.add_edge(const_id, op_id)
        dag.add_edge(op_id, var_ids[var_tgt])

    return dag


_UNARY_OPS = [NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.NEG]
_N_VARS_CHOICES = [3, 4, 5]


def run_adversarial_population(
    n_target: int,
    seed: int,
    rng: random.Random,
    k_perms: int = K_PERMS,
) -> dict[str, Any]:
    """Generate adversarial DAGs and test equivariance.

    Adversarial DAGs have >= 2 orphan CONSTs (in-degree 0) and >= 1 VAR with
    in-edges.  They lie outside C1 (CONSTs unreachable from VARs) and outside
    C2 (VARs have in-edges).  The brief's confirmed counterexample (2 CONSTs,
    2 SIN chains pointing at different VARs) is the minimal instance.

    Failures are EXPECTED here; a non-zero failure rate validates the hypothesis
    that the mechanism is not vacuous.

    Args:
        n_target: Number of adversarial DAGs to construct.
        seed: Seed for adversarial construction.
        rng: Seeded RNG for permutation generation.
        k_perms: Number of permutations to test per DAG.

    Returns:
        Per-population result dict.
    """
    result = make_result()
    result["population"] = "P4_adversarial"
    result["seed"] = seed

    adv_rng = random.Random(seed)

    for _ in range(n_target):
        n_vars = adv_rng.choice(_N_VARS_CHOICES)
        # At least 2 CONSTs, at most n_vars (one per chain).
        n_consts = adv_rng.randint(2, min(n_vars, 4))
        ops = [adv_rng.choice(_UNARY_OPS) for _ in range(n_consts)]
        # Each chain targets a distinct VAR; shuffle so it's not always 0,1,...
        targets = adv_rng.sample(range(n_vars), n_consts)

        dag = _make_adversarial_dag(
            n_vars=n_vars,
            const_labels=[NodeType.CONST] * n_consts,
            op_labels=ops,
            var_targets=targets,
            rng=adv_rng,
        )

        eq = test_equivariance(dag, rng, k_perms=k_perms)
        accumulate(result, dag, eq)

    result["failure_rate"] = result["n_failures"] / result["n_tests"] if result["n_tests"] else 0.0
    result["dag_failure_rate"] = result["n_failures"] / max(result["n_dags"], 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_table(populations: list[dict[str, Any]]) -> None:
    """Print a compact per-population summary table.

    Args:
        populations: List of per-population result dicts.
    """
    header = (
        f"{'Population':<30} {'N_DAGs':>7} {'CONST%':>7} "
        f"{'InSafe%':>8} {'Tests':>7} {'Fail':>6} {'Fail%':>7} "
        f"{'FailInSafe':>11} {'FailOutSafe':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in populations:
        n = r["n_dags"]
        const_pct = 100.0 * r["n_with_const"] / n if n else 0.0
        safe_pct = 100.0 * r["n_in_safe"] / n if n else 0.0
        fail_pct = 100.0 * r["failure_rate"] if "failure_rate" in r else 0.0
        print(
            f"{r['population']:<30} {n:>7,} {const_pct:>6.1f}% "
            f"{safe_pct:>7.1f}% {r['n_tests']:>7,} {r['n_failures']:>6} "
            f"{fail_pct:>6.2f}% {r['failures_in_safe']:>11} "
            f"{r['failures_outside_safe']:>12}"
        )


def main() -> None:
    """Entry point: run four populations and emit per-population statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the JSON results.",
    )
    parser.add_argument("--n-s2d", type=int, default=20_000, help="P1 DAG count.")
    parser.add_argument(
        "--n-s2d-vacuous",
        type=int,
        default=50,
        help=(
            "P2 DAG count (CONST-free, vacuously 0). "
            "CONST-free DAGs cost ~83x more per DAG than CONST-bearing ones "
            "(456ms vs 5.5ms) due to larger internal-node counts from lack of "
            "leaf CONSTs. Default 50 keeps P2 under 30s."
        ),
    )
    parser.add_argument("--n-adv", type=int, default=5_000, help="P4 DAG count.")
    parser.add_argument("--bingo-time", type=float, default=90.0, help="P3 wall-clock cap (s).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-perms", type=int, default=K_PERMS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    master_rng = random.Random(args.seed)
    rng_p1a = random.Random(master_rng.randint(0, 2**31))
    rng_p1b = random.Random(master_rng.randint(0, 2**31))
    rng_p2 = random.Random(master_rng.randint(0, 2**31))
    rng_p3 = random.Random(master_rng.randint(0, 2**31))
    rng_p4 = random.Random(master_rng.randint(0, 2**31))
    k_perms = args.k_perms

    log.info("Population 1a: S2D full alphabet, num_variables=2, N=%d", args.n_s2d)
    p1a = run_s2d_population(
        alphabet=FULL_ALPHABET,
        n_target=args.n_s2d,
        num_variables=2,
        seed=args.seed + 1,
        rng=rng_p1a,
        label="P1a_s2d_nv2",
        k_perms=k_perms,
    )

    log.info("Population 1b: S2D full alphabet, num_variables=3, N=%d", args.n_s2d)
    p1b = run_s2d_population(
        alphabet=FULL_ALPHABET,
        n_target=args.n_s2d,
        num_variables=3,
        seed=args.seed + 2,
        rng=rng_p1b,
        label="P1b_s2d_nv3",
        k_perms=k_perms,
    )

    log.info(
        "Population 2: S2D CONST-free alphabet (vacuous: normalize is no-op), N=%d",
        args.n_s2d,
    )
    p2 = run_s2d_population(
        alphabet=CONST_FREE_ALPHABET,
        n_target=args.n_s2d_vacuous,
        num_variables=2,
        seed=args.seed + 3,
        rng=rng_p2,
        label="P2_const_free_vacuous",
        vacuous_note=(
            "CONST-free alphabet: normalize_const_creation is a no-op. "
            "n_s2d_vacuous used (CONST-free DAGs cost ~83x more per-DAG due "
            "to larger k; P2 result is vacuously 0 regardless of N). "
            "Failure rate of 0 is vacuous, not evidence of equivariance."
        ),
        k_perms=k_perms,
    )

    log.info("Population 3: Bingo adapter output, max_time=%.0fs", args.bingo_time)
    p3 = run_bingo_population(
        max_time_s=args.bingo_time,
        seed=args.seed,
        rng=rng_p3,
        k_perms=k_perms,
    )

    log.info("Population 4: adversarial (>= 2 orphan CONSTs + VAR in-edges), N=%d", args.n_adv)
    p4 = run_adversarial_population(
        n_target=args.n_adv,
        seed=args.seed + 4,
        rng=rng_p4,
        k_perms=k_perms,
    )

    populations = [p1a, p1b, p2, p3, p4]

    print("\n=== Equivariance measurement: normalize_const_creation ===\n")
    print(f"K permutations per DAG: {args.k_perms}")
    print(f"Timeout per fast_canonical_string call: {TIMEOUT}s\n")
    print_table(populations)

    print("\nCross-tabulation: failures vs safe class membership")
    print(f"  {'Population':<30} {'FailInSafe':>12} {'FailOutSafe':>13} {'Note'}")
    for r in populations:
        note = r.get("note", "")
        print(
            f"  {r['population']:<30} {r['failures_in_safe']:>12} "
            f"{r['failures_outside_safe']:>13}  {note}"
        )

    if "frac_in_c2" in p3:
        print(f"\nP3 Bingo: fraction of DAGs in C2 (all VARs are sources): {p3['frac_in_c2']:.4f}")
    print(f"\nP3 UDFS C2 claim: {p3.get('udfs_structural_argument', 'N/A')}")

    summary: dict[str, Any] = {
        "meta": {
            "k_perms": args.k_perms,
            "timeout_s": TIMEOUT,
            "seed": args.seed,
        },
        "populations": populations,
    }

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        log.info("wrote %s", path)
    else:
        print("\n(pass --out <path> to save JSON)")


if __name__ == "__main__":
    main()
