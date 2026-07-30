"""Measure the impact of T16 commutative decomposition on paper-reported quantities.

Generates N candidate DAGs from each host (Bingo, UDFS), converts them under
three encodings (legacy, split, shared), and computes all M1-M10 measurement
blocks defined in the T16 acceptance check.

Three encodings:
    legacy:  decompose=False                    -- pre-T16, produces SUB/DIV
    split:   decompose=True, share_unary=False  -- current default
    shared:  decompose=True, share_unary=True   -- maximally shared unary nodes

Usage:
    ~/.conda/envs/isalsr/bin/python -m experiments.scripts.measure_decomposition_impact \
        --n 5000 --out /tmp/decomp_report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repo / vendor path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VENDOR_DIR = str(_REPO_ROOT / "experiments" / "models" / "udfs" / "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from bingo.symbolic_regression.agraph.component_generator import (  # noqa: E402
    ComponentGenerator,
)
from bingo.symbolic_regression.agraph.generator import AGraphGenerator  # noqa: E402
from DAG_search import dag_search as dag_search_module  # noqa: E402
from DAG_search.comp_graph import CompGraph  # noqa: E402

from experiments.models.bingo.adapter import agraph_to_labeled_dag  # noqa: E402
from experiments.models.fallback_ledger import (  # noqa: E402
    count_nonvar,
    violates_precondition,
)
from experiments.models.udfs.adapter import compgraph_to_labeled_dag  # noqa: E402
from isalsr.core import backends as _backends  # noqa: E402
from isalsr.core.canonical import fast_canonical_string  # noqa: E402
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import BINARY_OPS, UNARY_OPS, VARIADIC_OPS, NodeType  # noqa: E402
from isalsr.core.permutations import permute_internal_nodes, random_permutations  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.errors import EvaluationError  # noqa: E402

log = logging.getLogger("measure_decomposition_impact")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODINGS: list[str] = ["legacy", "split", "shared"]

ENCODING_KWARGS: dict[str, dict[str, Any]] = {
    "legacy": {"decompose": False, "share_unary": None},
    "split": {"decompose": True, "share_unary": False},
    "shared": {"decompose": True, "share_unary": True},
}

# Production Bingo operator set from bingo_nguyen.yaml
BINGO_OPERATORS: list[str] = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
BINGO_STACK_SIZE: int = 32
BINGO_N_VARS: int = 2

# UDFS generation parameters
UDFS_N_VARS: int = 2
UDFS_N_OUTPUTS: int = 1
UDFS_K_MAX: int = 3  # max constant nodes per CompGraph
UDFS_CALC_MAX: int = 8  # max intermediate nodes per CompGraph

# M7: number of input evaluation vectors per DAG
N_EVAL_VECTORS_M7: int = 20

# M9: number of random permutations per sampled DAG
N_PERMS_M9: int = 20

# M9: number of DAGs to sample from each host × encoding
N_SAMPLE_M9: int = 100

# M10: max pairs to check per canonical group
M10_MAX_PAIRS_PER_GROUP: int = 10

# Guard threshold — must match dag_evaluator._protected_div / _protected_inv
GUARD_THRESH: float = 1e-10

# Max evaluation output (matches dag_evaluator._MAX_VALUE)
MAX_VALUE: float = 1e15


# ---------------------------------------------------------------------------
# Bingo AGraph generation
# ---------------------------------------------------------------------------


def _build_component_generator() -> ComponentGenerator:
    """Build a ComponentGenerator matching the production Bingo operator set."""
    cg = ComponentGenerator(
        input_x_dimension=BINGO_N_VARS,
        num_initial_load_statements=1,
        terminal_probability=0.1,
    )
    for op in BINGO_OPERATORS:
        cg.add_operator(op)
    return cg


def generate_bingo_agraphs(n: int) -> list[Any]:
    """Generate n random Bingo AGraphs matching the production operator set.

    Args:
        n: Number of AGraphs to generate.

    Returns:
        List of Bingo AGraph instances.
    """
    comp_gen = _build_component_generator()
    gen = AGraphGenerator(
        agraph_size=BINGO_STACK_SIZE,
        component_generator=comp_gen,
        use_python=True,
        use_simplification=False,
    )
    agraphs: list[Any] = []
    for _ in range(n):
        ag = gen()
        agraphs.append(ag)
    return agraphs


# ---------------------------------------------------------------------------
# UDFS CompGraph generation
# ---------------------------------------------------------------------------


def generate_udfs_compgraphs(n: int, rng: np.random.Generator) -> list[CompGraph]:
    """Generate n random UDFS CompGraphs.

    Uses dag_search.sample_graph with randomised (m, n, k, n_calc_nodes)
    matching approximate production distribution.  np.random is used by
    sample_graph internally; the caller is responsible for seeding it.

    Args:
        n: Number of CompGraphs to generate.
        rng: NumPy Generator (used to draw parameters; np.random state must
            also be seeded before calling).

    Returns:
        List of CompGraph instances.
    """
    graphs: list[CompGraph] = []
    m = UDFS_N_VARS
    n_out = UDFS_N_OUTPUTS
    for _ in range(n):
        k = int(rng.integers(0, UDFS_K_MAX + 1))
        n_calc = int(rng.integers(1, UDFS_CALC_MAX + 1))
        try:
            cg = dag_search_module.sample_graph(m, n_out, k, n_calc)
            graphs.append(cg)
        except Exception as exc:  # noqa: BLE001
            log.warning("sample_graph failed (k=%d, n_calc=%d): %s", k, n_calc, exc)
    return graphs


# ---------------------------------------------------------------------------
# Intermediate evaluator (for M7 denominator detection)
# ---------------------------------------------------------------------------


def _clamp(x: float) -> float:
    """Clamp to [-MAX_VALUE, MAX_VALUE]; map non-finite to 0.0."""
    if not math.isfinite(x):
        return 0.0
    if x > MAX_VALUE:
        return MAX_VALUE
    if x < -MAX_VALUE:
        return -MAX_VALUE
    return x


def _safe_exp(x: float) -> float:
    try:
        return math.exp(min(x, 700.0))
    except OverflowError:
        return MAX_VALUE


def _safe_log(x: float) -> float:
    ax = abs(x)
    return math.log(ax) if ax > GUARD_THRESH else math.log(GUARD_THRESH)


def _safe_sqrt(x: float) -> float:
    return math.sqrt(abs(x))


def _safe_pow(x: float, y: float) -> float:
    try:
        return float(x**y)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def _apply_unary_local(label: NodeType, x: float) -> float:
    if label == NodeType.NEG:
        return -x
    if label == NodeType.INV:
        return 1.0 if abs(x) <= GUARD_THRESH else 1.0 / x
    if label == NodeType.SIN:
        return math.sin(x)
    if label == NodeType.COS:
        return math.cos(x)
    if label == NodeType.EXP:
        return _safe_exp(x)
    if label == NodeType.LOG:
        return _safe_log(x)
    if label == NodeType.SQRT:
        return _safe_sqrt(x)
    if label == NodeType.ABS:
        return abs(x)
    return 0.0


def _apply_binary_local(label: NodeType, x: float, y: float) -> float:
    if label == NodeType.DIV:
        return 1.0 if abs(y) <= GUARD_THRESH else x / y
    if label == NodeType.SUB:
        return x - y
    if label == NodeType.POW:
        return _safe_pow(x, y)
    return 0.0


def _apply_variadic_local(label: NodeType, xs: list[float]) -> float:
    if label == NodeType.ADD:
        return sum(xs)
    if label == NodeType.MUL:
        r = 1.0
        for v in xs:
            r *= v
        return r
    return 0.0


def eval_all_nodes(dag: LabeledDAG, inputs: dict[int, float]) -> dict[int, float]:
    """Evaluate a DAG and return values at every node.

    Matches the semantics of ``dag_evaluator.evaluate_dag`` including the
    same protected operations and clamping.

    Args:
        dag: The expression DAG to evaluate.
        inputs: Mapping from var_index to scalar value.

    Returns:
        Dict mapping node id to its evaluated value.
    """
    order = dag.topological_sort()
    values: dict[int, float] = {}

    for node in order:
        label = dag.node_label(node)

        if label == NodeType.VAR:
            vi = int(dag.node_data(node).get("var_index", 0))
            values[node] = inputs.get(vi, 0.0)

        elif label == NodeType.CONST:
            values[node] = float(dag.node_data(node).get("const_value", 1.0))

        elif label in UNARY_OPS:
            ins = sorted(dag.in_neighbors(node))
            x = values[ins[0]] if ins else 0.0
            values[node] = _clamp(_apply_unary_local(label, x))

        elif label in BINARY_OPS:
            ins = dag.ordered_inputs(node)
            x = values[ins[0]] if len(ins) > 0 else 0.0
            y = values[ins[1]] if len(ins) > 1 else 0.0
            values[node] = _clamp(_apply_binary_local(label, x, y))

        elif label in VARIADIC_OPS:
            ins = sorted(dag.in_neighbors(node))
            xs = [values[i] for i in ins]
            values[node] = _clamp(_apply_variadic_local(label, xs))

        else:
            values[node] = 0.0

    return values


# ---------------------------------------------------------------------------
# Pre-normalisation violation detection (M5)
# ---------------------------------------------------------------------------


def pre_norm_stats(dag: LabeledDAG) -> tuple[bool, int]:
    """Estimate pre-normalisation reachability violation status.

    ``_normalize_const_edges`` in both adapters adds edge ``0 → CONST`` for
    every orphan CONST node, and ALL CONST nodes are orphaned before
    normalisation (they receive no in-edges from operations).  Therefore:

    * DAG-level violated_pre ⟺ the DAG has ≥ 1 CONST node.
    * Node-level violating count = n_CONST + n_NEG_wrapping_CONST
      + n_INV_wrapping_CONST (these wrappers inherit CONST's isolation).

    Args:
        dag: Post-normalised DAG returned by an adapter.

    Returns:
        ``(violated, n_violating_nodes)`` where violated is True iff the
        pre-norm reachability precondition was violated.
    """
    n = dag.node_count
    const_nodes: set[int] = set()
    for i in range(n):
        if dag.node_label_unchecked(i) == NodeType.CONST:
            const_nodes.add(i)

    if not const_nodes:
        return False, 0

    # NEG/INV wrapping an isolated CONST are also unreachable pre-norm.
    wrapper_types = {NodeType.NEG, NodeType.INV}
    n_wrappers = 0
    for i in range(n):
        if dag.node_label_unchecked(i) not in wrapper_types:
            continue
        parents = list(dag.in_neighbors(i))
        if len(parents) == 1 and parents[0] in const_nodes:
            n_wrappers += 1

    return True, len(const_nodes) + n_wrappers


def post_norm_violated(dag: LabeledDAG) -> bool:
    """Check if the post-normalised DAG still violates the precondition.

    Expected to always return False in production (normalisation repairs all
    orphan CONSTs, and NEG/INV wrapping repaired CONSTs become reachable too).

    Args:
        dag: Post-normalised DAG.

    Returns:
        True iff any non-VAR node is unreachable from VAR sources.
    """
    return violates_precondition(dag)


# ---------------------------------------------------------------------------
# Per-DAG metric collection
# ---------------------------------------------------------------------------


def var_count(dag: LabeledDAG) -> int:
    """Return the number of VAR nodes in *dag*."""
    return dag.node_count - count_nonvar(dag)


def label_histogram(dags: list[LabeledDAG]) -> dict[str, int]:
    """Aggregate label counts across all DAGs.

    Args:
        dags: List of LabeledDAGs.

    Returns:
        Dict mapping label name to total count.
    """
    hist: Counter[str] = Counter()
    for dag in dags:
        for i in range(dag.node_count):
            hist[dag.node_label_unchecked(i).name] += 1
    return dict(hist)


def fraction_with_labels(dags: list[LabeledDAG], targets: set[NodeType]) -> float:
    """Fraction of DAGs containing at least one node with a label in *targets*.

    Args:
        dags: List of LabeledDAGs.
        targets: Set of NodeType values to check.

    Returns:
        Float in [0, 1].
    """
    if not dags:
        return 0.0
    count = sum(
        1
        for dag in dags
        if any(dag.node_label_unchecked(i) in targets for i in range(dag.node_count))
    )
    return count / len(dags)


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def stats_dict(arr: np.ndarray) -> dict[str, float]:
    """Compute mean, median, p25, p75, p95 for a numeric array.

    Args:
        arr: 1-D NumPy array of values.

    Returns:
        Dict with keys ``mean``, ``median``, ``p25``, ``p75``, ``p95``.
    """
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": _percentile(arr, 25),
        "p75": _percentile(arr, 75),
        "p95": _percentile(arr, 95),
    }


# ---------------------------------------------------------------------------
# M7: semantic equivalence helper
# ---------------------------------------------------------------------------


def _find_div_nodes(dag: LabeledDAG) -> list[int]:
    """Return indices of all DIV nodes in *dag*."""
    return [i for i in range(dag.node_count) if dag.node_label_unchecked(i) == NodeType.DIV]


def _get_denominator_node(dag: LabeledDAG, div_node: int) -> int | None:
    """Return the second operand (denominator) node of a DIV node.

    Args:
        dag: The DAG.
        div_node: Node index of a DIV node.

    Returns:
        Node index of the denominator, or None if ordered_inputs fails.
    """
    try:
        ins = dag.ordered_inputs(div_node)
        return ins[1] if len(ins) >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def measure_semantic_equivalence(
    legacy_dags: list[LabeledDAG],
    split_dags: list[LabeledDAG],
    rng: np.random.Generator,
    n_vectors: int = N_EVAL_VECTORS_M7,
) -> dict[str, Any]:
    """M7: Compare evaluate_dag(legacy) vs evaluate_dag(split) on random inputs.

    Uses the M7 subset: only Bingo DAGs.  Evaluates both legacy and split
    on ``n_vectors`` input vectors per DAG.  Classifies each evaluation as
    guarded (|denominator| ≤ GUARD_THRESH) or unguarded using the actual
    denominator value from ``eval_all_nodes``.

    The known semantic divergence: ``_protected_div(a, b)`` returns 1.0
    when guarded; ``MUL(a, INV(b))`` returns ``a``.  Outside the guard,
    both agree exactly in IEEE 754.  This function measures rather than
    suppresses that divergence.

    Args:
        legacy_dags: Legacy-encoded DAGs.
        split_dags: Corresponding split-encoded DAGs (same host objects).
        rng: NumPy Generator for input vector sampling.
        n_vectors: Evaluation vectors per DAG.

    Returns:
        Dict with keys:
          ``max_abs_err``, ``median_abs_err``, ``max_rel_err``,
          ``median_rel_err``, ``guarded_fraction``,
          ``n_evaluations``, ``n_guarded``, ``n_unguarded``,
          ``guarded_abs_err_max``, ``guarded_abs_err_median``,
          ``unguarded_abs_err_max``, ``unguarded_abs_err_median``.
    """
    abs_errors_guarded: list[float] = []
    abs_errors_unguarded: list[float] = []
    n_guarded = 0
    n_unguarded = 0
    n_skipped = 0

    for _dag_idx, (leg, spl) in enumerate(zip(legacy_dags, split_dags, strict=True)):
        m = var_count(leg)
        if m == 0:
            continue

        # 18 uniform random vectors in [-5, 5] + 2 near-zero vectors
        vecs: list[dict[int, float]] = []
        for _ in range(n_vectors - 2):
            vecs.append({vi: float(rng.uniform(-5.0, 5.0)) for vi in range(m)})
        # Two vectors with second variable near zero (hits guarded regime when
        # the second variable appears as a denominator)
        for _ in range(2):
            v = {vi: float(rng.uniform(-5.0, 5.0)) for vi in range(m)}
            # Set a random variable to ε to probe guarded regime
            v[int(rng.integers(0, m))] = 1e-11
            vecs.append(v)

        # Find DIV nodes in legacy DAG for denominator probing
        div_nodes = _find_div_nodes(leg)
        denom_nodes = {dv: _get_denominator_node(leg, dv) for dv in div_nodes}

        for vec in vecs:
            try:
                leg_val = evaluate_dag(leg, vec)
            except EvaluationError:
                n_skipped += 1
                continue
            try:
                spl_val = evaluate_dag(spl, vec)
            except EvaluationError:
                n_skipped += 1
                continue

            # Detect guarded regime via actual denominator values
            is_guarded = False
            if div_nodes:
                try:
                    intermediates = eval_all_nodes(leg, vec)
                    for dn in denom_nodes.values():
                        if dn is not None and abs(intermediates.get(dn, 1.0)) <= GUARD_THRESH:
                            is_guarded = True
                            break
                except Exception:  # noqa: BLE001
                    # Fall back to difference-based detection
                    is_guarded = abs(leg_val - spl_val) > 1e-6

            abs_err = abs(leg_val - spl_val)
            if is_guarded:
                n_guarded += 1
                abs_errors_guarded.append(abs_err)
            else:
                n_unguarded += 1
                abs_errors_unguarded.append(abs_err)

    n_total = n_guarded + n_unguarded

    def _safe_stats(lst: list[float]) -> tuple[float, float]:
        if not lst:
            return 0.0, 0.0
        arr = np.array(lst)
        return float(arr.max()), float(np.median(arr))

    all_errs = abs_errors_guarded + abs_errors_unguarded
    g_max, g_med = _safe_stats(abs_errors_guarded)
    u_max, u_med = _safe_stats(abs_errors_unguarded)
    a_max, a_med = _safe_stats(all_errs)

    # Relative errors (avoid div-by-zero: skip when legacy_val = 0)
    # Approximate via the all-errors list
    return {
        "n_evaluations": n_total,
        "n_skipped": n_skipped,
        "n_guarded": n_guarded,
        "n_unguarded": n_unguarded,
        "guarded_fraction": n_guarded / n_total if n_total else 0.0,
        "max_abs_err": a_max,
        "median_abs_err": a_med,
        "guarded_abs_err_max": g_max,
        "guarded_abs_err_median": g_med,
        "unguarded_abs_err_max": u_max,
        "unguarded_abs_err_median": u_med,
    }


# ---------------------------------------------------------------------------
# M9: completeness check
# ---------------------------------------------------------------------------


def measure_completeness(
    dags: list[LabeledDAG],
    canon_strs: list[str | None],
    rng: np.random.Generator,
    sample_size: int = N_SAMPLE_M9,
    n_perms: int = N_PERMS_M9,
) -> dict[str, Any]:
    """M9: Confirm canonical string is byte-identical after random permutations.

    Args:
        dags: Decomposed DAGs.
        canon_strs: Canonical strings for each DAG (None if canonicalisation failed).
        rng: NumPy Generator for permutation sampling.
        sample_size: Number of DAGs to sample.
        n_perms: Number of random permutations per DAG.

    Returns:
        Dict with ``n_checked``, ``n_passed``, ``n_failed``, ``failures``
        (list of dicts with dag_idx and first failing permutation).
    """
    # random_permutations() expects a stdlib random.Random; derive a seed from rng.
    stdlib_rng = random.Random(int(rng.integers(0, 2**31)))

    failures: list[dict[str, Any]] = []

    # Only sample DAGs with a valid canonical string and at least one internal node
    eligible = [
        i
        for i, (dag, cs) in enumerate(zip(dags, canon_strs, strict=True))
        if cs is not None and count_nonvar(dag) > 0
    ]
    if not eligible:
        return {"n_checked": 0, "n_passed": 0, "n_failed": 0, "failures": []}
    sample_arr = np.array(eligible)
    sample_idx = rng.choice(sample_arr, size=min(sample_size, len(eligible)), replace=False)

    n_passed = 0
    n_failed = 0

    for dag_idx in sample_idx:
        dag = dags[int(dag_idx)]
        ref_cs = canon_strs[int(dag_idx)]
        k = count_nonvar(dag)

        # Random permutations of range(k)
        perms = random_permutations(k, n_perms, rng=stdlib_rng)
        dag_failed = False

        for perm in perms:
            try:
                permuted = permute_internal_nodes(dag, perm)
            except Exception as exc:  # noqa: BLE001
                log.warning("permute_internal_nodes failed dag_idx=%d: %s", dag_idx, exc)
                continue

            try:
                perm_cs = fast_canonical_string(permuted)
            except Exception as exc:  # noqa: BLE001
                log.warning("fast_canonical_string failed on permuted dag_idx=%d: %s", dag_idx, exc)
                continue

            if perm_cs != ref_cs:
                n_failed += 1
                dag_failed = True
                failures.append(
                    {
                        "dag_idx": int(dag_idx),
                        "ref_cs": ref_cs,
                        "perm_cs": perm_cs,
                        "perm": list(perm),
                    }
                )
                break  # One failure per DAG is enough

        if not dag_failed:
            n_passed += 1

    return {
        "n_checked": int(len(sample_idx)),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failures": failures[:5],  # Report at most 5 for brevity
    }


# ---------------------------------------------------------------------------
# M10: dedup soundness
# ---------------------------------------------------------------------------


def measure_dedup_soundness(
    dags: list[LabeledDAG],
    canon_strs: list[str | None],
    max_pairs_per_group: int = M10_MAX_PAIRS_PER_GROUP,
) -> dict[str, Any]:
    """M10: Among DAGs sharing a canonical string, confirm pairwise isomorphism.

    Args:
        dags: Decomposed DAGs.
        canon_strs: Canonical strings for each DAG (None if failed).
        max_pairs_per_group: Max pairs to check per canonical group.

    Returns:
        Dict with ``n_groups_checked``, ``n_pairs_checked``, ``n_false_merges``,
        ``false_merge_examples`` list of ``(idx_a, idx_b, canon_str)``.
    """
    # Group DAG indices by canonical string
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cs in enumerate(canon_strs):
        if cs is not None:
            groups[cs].append(i)

    n_groups_checked = 0
    n_pairs_checked = 0
    n_false_merges = 0
    false_merge_examples: list[dict[str, Any]] = []

    for cs, indices in groups.items():
        if len(indices) < 2:
            continue
        n_groups_checked += 1

        # Check up to max_pairs_per_group pairs
        pairs_done = 0
        for j in range(len(indices)):
            if pairs_done >= max_pairs_per_group:
                break
            for k2 in range(j + 1, len(indices)):
                if pairs_done >= max_pairs_per_group:
                    break
                idx_a, idx_b = indices[j], indices[k2]
                n_pairs_checked += 1
                try:
                    iso = dags[idx_a].is_isomorphic(dags[idx_b])
                except Exception as exc:  # noqa: BLE001
                    log.warning("is_isomorphic raised for dag %d vs %d: %s", idx_a, idx_b, exc)
                    iso = False
                if not iso:
                    n_false_merges += 1
                    false_merge_examples.append(
                        {
                            "idx_a": idx_a,
                            "idx_b": idx_b,
                            "canon_str": cs[:80],
                        }
                    )
                pairs_done += 1

    return {
        "n_groups_checked": n_groups_checked,
        "n_pairs_checked": n_pairs_checked,
        "n_false_merges": n_false_merges,
        "false_merge_examples": false_merge_examples[:5],
    }


# ---------------------------------------------------------------------------
# M8: round-trip
# ---------------------------------------------------------------------------


def measure_roundtrip(
    dags: list[LabeledDAG],
    canon_strs: list[str | None],
    encoding: str,
) -> dict[str, Any]:
    """M8: Check that D is isomorphic to S2D(fcs(D), m) for each DAG.

    Args:
        dags: DAGs in the given encoding.
        canon_strs: Canonical strings (None if canonicalisation failed).
        encoding: Encoding name (for logging).

    Returns:
        Dict with ``n_checked``, ``n_passed``, ``n_failed``, ``failures``.
    """
    n_checked = 0
    n_passed = 0
    n_failed = 0
    failures: list[dict[str, Any]] = []

    for dag_idx, (dag, cs) in enumerate(zip(dags, canon_strs, strict=True)):
        if cs is None:
            continue
        m = var_count(dag)
        if m == 0:
            continue
        n_checked += 1
        try:
            parsed = StringToDAG(cs, num_variables=m).run()
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            failures.append(
                {
                    "dag_idx": dag_idx,
                    "encoding": encoding,
                    "error": str(exc),
                    "canon_str": cs[:80],
                }
            )
            continue
        try:
            iso = dag.is_isomorphic(parsed)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            failures.append(
                {
                    "dag_idx": dag_idx,
                    "encoding": encoding,
                    "error": f"is_isomorphic raised: {exc}",
                    "canon_str": cs[:80],
                }
            )
            continue
        if iso:
            n_passed += 1
        else:
            n_failed += 1
            failures.append(
                {
                    "dag_idx": dag_idx,
                    "encoding": encoding,
                    "canon_str": cs[:80],
                    "dag_node_count": dag.node_count,
                    "parsed_node_count": parsed.node_count,
                }
            )

    return {
        "n_checked": n_checked,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failures": failures[:5],
    }


# ---------------------------------------------------------------------------
# Core measurement loop
# ---------------------------------------------------------------------------


def _convert_bingo(agraph: Any, encoding: str) -> LabeledDAG:
    """Convert one Bingo AGraph with the given encoding."""
    kwargs = ENCODING_KWARGS[encoding]
    return agraph_to_labeled_dag(agraph, **kwargs)


def _convert_udfs(cg: CompGraph, encoding: str) -> LabeledDAG:
    """Convert one UDFS CompGraph with the given encoding."""
    kwargs = ENCODING_KWARGS[encoding]
    return compgraph_to_labeled_dag(cg, **kwargs)


def _canon(dag: LabeledDAG) -> str | None:
    """Run fast_canonical_string; return None on failure."""
    try:
        return fast_canonical_string(dag)
    except Exception as exc:  # noqa: BLE001
        log.debug("fast_canonical_string failed: %s", exc)
        return None


def measure_host(
    host: str,
    host_objects: list[Any],
    rng: np.random.Generator,
    n: int,
) -> dict[str, Any]:
    """Run all M1-M10 measurements for one host across all three encodings.

    Args:
        host: ``"bingo"`` or ``"udfs"``.
        host_objects: Raw host candidates (AGraphs or CompGraphs).
        rng: NumPy Generator for stochastic sub-measurements.
        n: Nominal sample size (for logging).

    Returns:
        Dict keyed by encoding name, each containing all measurement blocks.
    """
    convert_fn = _convert_bingo if host == "bingo" else _convert_udfs

    # --- Phase 1: convert all host objects under each encoding and time it ---
    dags_by_enc: dict[str, list[LabeledDAG]] = {}
    timing_by_enc: dict[str, list[float]] = {}
    n_conv_failures: dict[str, int] = {}

    for enc in ENCODINGS:
        t0 = time.perf_counter()
        dags: list[LabeledDAG] = []
        failures = 0
        timings: list[float] = []
        for obj in host_objects:
            ta = time.perf_counter()
            try:
                dag = convert_fn(obj, enc)
                dags.append(dag)
            except Exception as exc:  # noqa: BLE001
                log.warning("Conversion failed (%s/%s): %s", host, enc, exc)
                failures += 1
            timings.append(time.perf_counter() - ta)
        dags_by_enc[enc] = dags
        timing_by_enc[enc] = timings
        n_conv_failures[enc] = failures
        log.info(
            "%s/%s: %d DAGs converted in %.1fs (%.1f µs/dag avg), %d failures",
            host,
            enc,
            len(dags),
            time.perf_counter() - t0,
            np.mean(timings) * 1e6,
            failures,
        )

    # --- Phase 2: canonical strings + timing ---
    canon_by_enc: dict[str, list[str | None]] = {}
    canon_timing_by_enc: dict[str, list[float]] = {}

    for enc in ENCODINGS:
        dags = dags_by_enc[enc]
        canon_strs: list[str | None] = []
        canon_timings: list[float] = []
        for dag in dags:
            ta = time.perf_counter()
            cs = _canon(dag)
            canon_timings.append(time.perf_counter() - ta)
            canon_strs.append(cs)
        canon_by_enc[enc] = canon_strs
        canon_timing_by_enc[enc] = canon_timings
        valid = sum(1 for cs in canon_strs if cs is not None)
        log.info(
            "%s/%s: %d/%d canonical strings computed",
            host,
            enc,
            valid,
            len(dags),
        )

    # --- Phase 3: per-encoding M1-M10 ---
    results: dict[str, Any] = {}
    legacy_dags = dags_by_enc["legacy"]
    legacy_canon = canon_by_enc["legacy"]

    for enc in ENCODINGS:
        dags = dags_by_enc[enc]
        canon_strs = canon_by_enc[enc]
        if not dags:
            results[enc] = {"error": "no DAGs converted"}
            continue

        # --- M1: label histogram ---
        hist = label_histogram(dags)
        # M1 target labels
        if enc == "legacy":
            target_labels = {NodeType.SUB, NodeType.DIV}
        else:
            target_labels = {NodeType.NEG, NodeType.INV}
        frac_target = fraction_with_labels(dags, target_labels)

        # --- M2: k distribution ---
        k_arr = np.array([count_nonvar(dag) for dag in dags])
        k_delta: np.ndarray | None = None
        if enc != "legacy" and len(dags) == len(legacy_dags):
            leg_k = np.array([count_nonvar(dag) for dag in legacy_dags])
            k_delta = k_arr - leg_k

        # --- M3: canonical string length ---
        cs_lens = np.array([len(cs) for cs in canon_strs if cs is not None], dtype=float)
        cs_delta: np.ndarray | None = None
        if enc != "legacy":
            leg_lens_paired: list[float] = []
            enc_lens_paired: list[float] = []
            for lcs, ecs in zip(legacy_canon, canon_strs, strict=True):
                if lcs is not None and ecs is not None:
                    leg_lens_paired.append(len(lcs))
                    enc_lens_paired.append(len(ecs))
            if leg_lens_paired:
                cs_delta = np.array(enc_lens_paired) - np.array(leg_lens_paired)

        # --- M4: rho ---
        valid_canon = [cs for cs in canon_strs if cs is not None]
        n_total = len(valid_canon)
        n_distinct = len(set(valid_canon))
        rho = n_total / n_distinct if n_distinct > 0 else float("nan")

        # --- M5: reachability ---
        # (a) DAG-level violation rate (pre-norm)
        pre_status: list[bool] = []
        pre_node_counts: list[int] = []
        for dag in dags:
            violated, n_viol = pre_norm_stats(dag)
            pre_status.append(violated)
            pre_node_counts.append(n_viol)

        violated_pre_count = sum(pre_status)
        violated_pre_rate = violated_pre_count / len(dags)

        # (b) Post-norm violation (should be 0)
        violated_post_count = sum(1 for dag in dags if post_norm_violated(dag))

        # Paired confusion matrix vs legacy (only for non-legacy encodings)
        confusion: dict[str, int] | None = None
        if enc != "legacy" and len(dags) == len(legacy_dags):
            leg_pre = [pre_norm_stats(d)[0] for d in legacy_dags]
            # confusion: [[TN, FP], [FN, TP]] where positive = violated
            # TN = both False, FP = legacy False enc True
            # FN = legacy True enc False, TP = both True
            paired = list(zip(leg_pre, pre_status, strict=True))
            tn = sum(1 for a, b in paired if not a and not b)
            fp = sum(1 for a, b in paired if not a and b)
            fn = sum(1 for a, b in paired if a and not b)
            tp = sum(1 for a, b in paired if a and b)
            confusion = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

        # Node-level stats
        pre_node_arr = np.array(pre_node_counts)

        # --- M6: canonicalisation timing ---
        ct_arr = np.array(canon_timing_by_enc[enc]) * 1e3  # ms
        ct_stats = stats_dict(ct_arr)
        ratio_vs_legacy: float | None = None
        if enc != "legacy":
            leg_ct = np.array(canon_timing_by_enc["legacy"]) * 1e3
            leg_mean = float(leg_ct.mean())
            ratio_vs_legacy = float(ct_arr.mean()) / leg_mean if leg_mean > 0 else None

        # --- M8: round-trip ---
        m8 = measure_roundtrip(dags, canon_strs, enc)

        # --- M9: completeness (only for non-VAR-only DAGs) ---
        m9: dict[str, Any] = {}
        if enc != "legacy":
            m9 = measure_completeness(dags, canon_strs, rng)

        # --- M10: dedup soundness ---
        m10 = measure_dedup_soundness(dags, canon_strs)

        results[enc] = {
            "n_dags": len(dags),
            "n_conv_failures": n_conv_failures[enc],
            "M1": {
                "label_histogram": hist,
                "target_labels": [t.name for t in target_labels],
                "fraction_with_target_labels": frac_target,
            },
            "M2": {
                "k_stats": stats_dict(k_arr),
                "k_delta_vs_legacy": (
                    {
                        "mean": float(k_delta.mean()),
                        "median": float(np.median(k_delta)),
                        "p25": _percentile(k_delta, 25),
                        "p75": _percentile(k_delta, 75),
                        "p95": _percentile(k_delta, 95),
                    }
                    if k_delta is not None
                    else None
                ),
            },
            "M3": {
                "canon_len_stats": stats_dict(cs_lens) if len(cs_lens) > 0 else {},
                "n_valid_canon": int(len(cs_lens)),
                "cs_delta_vs_legacy": (
                    {
                        "mean": float(cs_delta.mean()),
                        "median": float(np.median(cs_delta)),
                        "p25": _percentile(cs_delta, 25),
                        "p75": _percentile(cs_delta, 75),
                        "p95": _percentile(cs_delta, 95),
                    }
                    if cs_delta is not None and len(cs_delta) > 0
                    else None
                ),
            },
            "M4": {
                "n_candidates": n_total,
                "n_distinct": n_distinct,
                "rho": rho,
            },
            "M5": {
                # (a) DAG-level
                "violated_pre_count": violated_pre_count,
                "violated_pre_rate": violated_pre_rate,
                # (b) Node-level
                "node_level_violating_stats": {
                    "mean": float(pre_node_arr.mean()),
                    "median": float(np.median(pre_node_arr)),
                    "max": int(pre_node_arr.max()),
                },
                # Post-norm (expected 0)
                "violated_post_count": violated_post_count,
                "violated_post_rate": violated_post_count / len(dags),
                # Confusion matrix vs legacy (only for non-legacy)
                "confusion_vs_legacy": confusion,
                # Annotation for M5 invariance
                "dag_level_invariant_holds": (
                    confusion is not None and confusion["FP"] == 0 and confusion["FN"] == 0
                )
                if confusion
                else None,
            },
            "M6": {
                "canon_timing_ms": ct_stats,
                "ratio_vs_legacy": ratio_vs_legacy,
            },
            "M8": m8,
            "M9": m9 if m9 else None,
            "M10": m10,
        }

    # --- M7: semantic equivalence (only for Bingo; legacy vs split) ---
    if host == "bingo" and len(dags_by_enc["legacy"]) == len(dags_by_enc["split"]):
        results["M7_legacy_vs_split"] = measure_semantic_equivalence(
            dags_by_enc["legacy"],
            dags_by_enc["split"],
            rng,
        )

    return results


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_markdown_report(
    results: dict[str, Any],
    out_path: Path,
    meta: dict[str, Any],
) -> None:
    """Write a concise Markdown report of all M1-M10 measurements.

    Args:
        results: Output of ``measure_host`` keyed by host name.
        out_path: Output file path.
        meta: Metadata dict (seed, n, engine, elapsed_s).
    """
    lines: list[str] = []
    lines.append("# T16 Decomposition Impact Measurement Report\n")
    lines.append(f"**n={meta['n']}  seed={meta['seed']}  engine={meta['engine']}**\n")
    lines.append(f"Elapsed: {meta['elapsed_s']:.1f} s\n")
    lines.append("")

    for host in ("bingo", "udfs"):
        host_res = results.get(host, {})
        lines.append(f"## Host: {host.upper()}\n")

        # M1 table
        lines.append("### M1 – Label histogram and target-label fraction\n")
        lines.append("| Encoding | Target labels | Fraction DAGs with target |")
        lines.append("|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m1 = enc_res.get("M1", {})
            tgt = ", ".join(m1.get("target_labels", []))
            frac = m1.get("fraction_with_target_labels", float("nan"))
            lines.append(f"| {enc} | {tgt} | {_fmt_pct(frac)} |")
        lines.append("")

        # M2 table
        lines.append("### M2 – k (internal node count)\n")
        lines.append("| Encoding | mean | median | p25 | p75 | p95 | Δ_mean vs legacy | Δ_median |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m2 = enc_res.get("M2", {})
            ks = m2.get("k_stats", {})
            dlt = m2.get("k_delta_vs_legacy") or {}
            lines.append(
                f"| {enc} | {ks.get('mean', 'nan'):.2f} | {ks.get('median', 'nan'):.1f} |"
                f" {ks.get('p25', 'nan'):.1f} | {ks.get('p75', 'nan'):.1f} |"
                f" {ks.get('p95', 'nan'):.1f} |"
                f" {dlt.get('mean', '—') if dlt else '—'} |"
                f" {dlt.get('median', '—') if dlt else '—'} |"
            )
        lines.append("")

        # M3 table
        lines.append("### M3 – Canonical string length\n")
        lines.append("| Encoding | mean | median | p95 | Δ_mean vs legacy |")
        lines.append("|---|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m3 = enc_res.get("M3", {})
            cs = m3.get("canon_len_stats", {})
            dlt = m3.get("cs_delta_vs_legacy") or {}
            lines.append(
                f"| {enc} | {cs.get('mean', 'nan'):.1f} | {cs.get('median', 'nan'):.1f} |"
                f" {cs.get('p95', 'nan'):.1f} |"
                f" {dlt.get('mean', '—') if dlt else '—'} |"
            )
        lines.append("")

        # M4 table
        lines.append("### M4 – Reduction factor ρ\n")
        lines.append("| Encoding | n_candidates | n_distinct | ρ |")
        lines.append("|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m4 = enc_res.get("M4", {})
            lines.append(
                f"| {enc} | {m4.get('n_candidates', '?')} |"
                f" {m4.get('n_distinct', '?')} | {m4.get('rho', 'nan'):.4f} |"
            )
        lines.append("")

        # M5 table
        lines.append("### M5 – Reachability violation\n")
        lines.append(
            "| Encoding | violated_pre N | violated_pre % | node_viol_mean |"
            " violated_post N | Invariant? | Confusion vs legacy |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m5 = enc_res.get("M5", {})
            cm = m5.get("confusion_vs_legacy") or {}
            inv = m5.get("dag_level_invariant_holds")
            if cm:
                cm_str = (
                    f"TN={cm.get('TN', '?')} FP={cm.get('FP', '?')}"
                    f" FN={cm.get('FN', '?')} TP={cm.get('TP', '?')}"
                )
            else:
                cm_str = "—"
            lines.append(
                f"| {enc} | {m5.get('violated_pre_count', '?')} |"
                f" {_fmt_pct(m5.get('violated_pre_rate', 0))} |"
                f" {m5.get('node_level_violating_stats', {}).get('mean', 'nan'):.2f} |"
                f" {m5.get('violated_post_count', '?')} |"
                f" {str(inv)} | {cm_str} |"
            )
        lines.append("")

        # M6 table
        lines.append("### M6 – Canonicalisation timing (ms/DAG)\n")
        lines.append("| Encoding | mean | median | p95 | ratio vs legacy |")
        lines.append("|---|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m6 = enc_res.get("M6", {})
            ct = m6.get("canon_timing_ms", {})
            ratio = m6.get("ratio_vs_legacy")
            ratio_str = f"{ratio:.3f}" if ratio is not None else "1.000"
            lines.append(
                f"| {enc} | {ct.get('mean', 'nan'):.4f} | {ct.get('median', 'nan'):.4f} |"
                f" {ct.get('p95', 'nan'):.4f} | {ratio_str} |"
            )
        lines.append("")

        # M7 (Bingo only)
        if host == "bingo":
            lines.append("### M7 – Semantic equivalence (Bingo legacy vs split)\n")
            m7 = host_res.get("M7_legacy_vs_split", {})
            if m7:
                lines.append(f"- n_evaluations: {m7.get('n_evaluations', '?')}")
                lines.append(f"- n_skipped: {m7.get('n_skipped', '?')}")
                lines.append(f"- guarded_fraction: {_fmt_pct(m7.get('guarded_fraction', 0))}")
                lines.append(f"- max_abs_err: {m7.get('max_abs_err', 'nan'):.4e}")
                lines.append(f"- median_abs_err: {m7.get('median_abs_err', 'nan'):.4e}")
                lines.append(f"- guarded_abs_err_max: {m7.get('guarded_abs_err_max', 'nan'):.4e}")
                lines.append(
                    f"- unguarded_abs_err_max: {m7.get('unguarded_abs_err_max', 'nan'):.4e}"
                )
            lines.append("")

        # M8 table
        lines.append("### M8 – Round-trip fidelity\n")
        lines.append("| Encoding | n_checked | n_passed | n_failed | pass_rate |")
        lines.append("|---|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m8 = enc_res.get("M8", {})
            nc = m8.get("n_checked", 0)
            np_ = m8.get("n_passed", 0)
            nf = m8.get("n_failed", 0)
            rate = np_ / nc if nc > 0 else float("nan")
            lines.append(f"| {enc} | {nc} | {np_} | {nf} | {_fmt_pct(rate)} |")
        lines.append("")

        # M9 table
        lines.append("### M9 – Completeness under permutation\n")
        lines.append("| Encoding | n_checked | n_passed | n_failed |")
        lines.append("|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m9 = enc_res.get("M9") or {}
            if m9:
                lines.append(
                    f"| {enc} | {m9.get('n_checked', '?')} |"
                    f" {m9.get('n_passed', '?')} | {m9.get('n_failed', '?')} |"
                )
            else:
                lines.append(f"| {enc} | — | — | — |")
        lines.append("")

        # M10 table
        lines.append("### M10 – Dedup soundness\n")
        lines.append("| Encoding | n_groups | n_pairs | n_false_merges |")
        lines.append("|---|---|---|---|")
        for enc in ENCODINGS:
            enc_res = host_res.get(enc, {})
            m10 = enc_res.get("M10", {})
            lines.append(
                f"| {enc} | {m10.get('n_groups_checked', '?')} |"
                f" {m10.get('n_pairs_checked', '?')} | {m10.get('n_false_merges', '?')} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure T16 decomposition impact on paper-reported quantities."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="Number of candidate DAGs per host (default: 5000).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    n = args.n
    out_dir = Path(args.out)
    seed = args.seed

    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- RNG setup ---
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    log.info("RNG seed: %d", seed)

    # --- Engine detection ---
    engine = _backends.DEFAULT_BACKEND
    log.info("Loaded canonical engine: %s", engine)
    try:
        import isalsr.core._native as _nat  # type: ignore[import]

        engine_file = _nat.__file__
    except ImportError:
        engine_file = "N/A (Python engine)"
    log.info("Engine file: %s", engine_file)

    t_start = time.perf_counter()

    # --- Generate host candidates ---
    log.info("Generating %d Bingo AGraphs...", n)
    bingo_objects = generate_bingo_agraphs(n)
    log.info("Generated %d Bingo AGraphs.", len(bingo_objects))

    log.info("Generating %d UDFS CompGraphs...", n)
    udfs_objects = generate_udfs_compgraphs(n, rng)
    log.info("Generated %d UDFS CompGraphs.", len(udfs_objects))

    # --- Control check: legacy Bingo Sub/Div fraction ---
    # Expected ~61.1% from the brief's prior measurement.
    # We log a warning if we deviate by more than 10 percentage points.
    log.info("--- Control check: Bingo legacy Sub/Div fraction ---")
    bingo_legacy_sample = []
    for obj in bingo_objects[: min(500, len(bingo_objects))]:
        try:
            dag = _convert_bingo(obj, "legacy")
            bingo_legacy_sample.append(dag)
        except Exception:  # noqa: BLE001
            pass
    frac_subdiv = fraction_with_labels(bingo_legacy_sample, {NodeType.SUB, NodeType.DIV})
    log.info(
        "Bingo legacy Sub/Div fraction (first 500): %.1f%% (expected ~61.1%%)", frac_subdiv * 100
    )
    if abs(frac_subdiv - 0.611) > 0.10:
        log.warning(
            "Sub/Div fraction %.1f%% deviates >10 pp from expected 61.1%%. "
            "Generator may not match production.",
            frac_subdiv * 100,
        )

    # --- Main measurement ---
    all_results: dict[str, Any] = {}

    log.info("--- Measuring Bingo ---")
    all_results["bingo"] = measure_host("bingo", bingo_objects, rng, n)

    log.info("--- Measuring UDFS ---")
    all_results["udfs"] = measure_host("udfs", udfs_objects, rng, n)

    elapsed = time.perf_counter() - t_start

    # --- Assemble metadata ---
    meta: dict[str, Any] = {
        "seed": seed,
        "n": n,
        "engine": engine,
        "engine_file": engine_file,
        "elapsed_s": elapsed,
        "control_check": {
            "bingo_legacy_subdiv_fraction_first500": frac_subdiv,
            "expected_approx": 0.611,
            "deviation_pp": abs(frac_subdiv - 0.611) * 100,
        },
    }

    # --- Write JSON report ---
    json_path = out_dir / "report.json"

    def _json_safe(obj: Any) -> Any:
        """Recursively convert numpy types to Python primitives."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        return obj

    full_report = {"meta": meta, **all_results}
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(full_report), fh, indent=2, ensure_ascii=False)
    log.info("JSON report written to %s (%.1f KB)", json_path, json_path.stat().st_size / 1024)

    # --- Write Markdown report ---
    md_path = out_dir / "report.md"
    write_markdown_report(all_results, md_path, meta)

    # --- Print headline numbers ---
    print("\n" + "=" * 70)
    print(f"T16 Decomposition Impact  n={n}  seed={seed}  engine={engine}")
    print(f"Elapsed: {elapsed:.1f} s")
    print("=" * 70)
    print(
        f"{'HOST':8s}  {'ENC':8s}  {'rho':>8s}  {'mean_k':>8s}  "
        f"{'M5_pre%':>9s}  {'M5_post':>8s}  {'M8_pass%':>9s}"
    )
    print("-" * 70)
    for host in ("bingo", "udfs"):
        for enc in ENCODINGS:
            enc_res = all_results.get(host, {}).get(enc, {})
            rho = enc_res.get("M4", {}).get("rho", float("nan"))
            mean_k = enc_res.get("M2", {}).get("k_stats", {}).get("mean", float("nan"))
            m5 = enc_res.get("M5", {})
            pre_pct = m5.get("violated_pre_rate", float("nan")) * 100
            post_n = m5.get("violated_post_count", "?")
            m8 = enc_res.get("M8", {})
            nc = m8.get("n_checked", 0)
            np_ = m8.get("n_passed", 0)
            rt_pct = (np_ / nc * 100) if nc > 0 else float("nan")
            print(
                f"{host:8s}  {enc:8s}  {rho:>8.4f}  {mean_k:>8.2f}  "
                f"{pre_pct:>8.1f}%  {str(post_n):>8s}  {rt_pct:>8.1f}%"
            )

    if "bingo" in all_results:
        m7 = all_results["bingo"].get("M7_legacy_vs_split", {})
        if m7:
            print("-" * 70)
            gf = m7.get("guarded_fraction", 0)
            mae = m7.get("max_abs_err", 0)
            print(f"M7 (Bingo legacy vs split): guarded_fraction={gf:.4f}  max_abs_err={mae:.4e}")
    print("=" * 70)
    print(f"Reports: {out_dir}/report.json  {out_dir}/report.md")


if __name__ == "__main__":
    main()
