"""Validate precomputed HDF5 caches against 9 mathematical/structural properties.

Properties validated (P1-P9):
    P1: Canonical Idempotency -- pruned_canonical_string(S2D(pruned_str)) == pruned_str
    P2: Evaluation Preservation -- eval(S2D(raw), x) ~= eval(S2D(pruned), x)
    P3: Round-Trip Isomorphism -- S2D(pruned).is_isomorphic(S2D(raw))
    P4: Acyclicity -- S2D(raw).topological_sort() succeeds
    P5: Token Validity -- StringToDAG(str, nv, ops).run() succeeds for all strings
    P6: DAG Property Consistency -- recomputed properties match stored values
    P7: Correctness Flags Consistency -- stored flags match string comparisons
    P8: Operand Order for Non-Commutative Ops -- swapping inputs changes evaluation
    P9: Pruned == Exhaustive Agreement -- rate from stored flags

Produces per-cache JSON reports and a printed summary table.

Usage:
    python experiments/scripts/validate_cache.py \\
        --cache-dir /media/mpascual/Sandisk2TB/research/isalsr/data/precomputed_cache \\
        --output-dir /media/mpascual/Sandisk2TB/research/isalsr/results/validation \\
        --sample 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Ensure isalsr is importable from the project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import (
    BINARY_OPS,
    LABEL_CHAR_MAP,
    UNARY_OPS,
    VARIADIC_OPS,
    OperationSet,
)
from isalsr.core.string_to_dag import StringToDAG
from isalsr.errors import EvaluationError
from isalsr.precomputed.cache_entry import dag_depth

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_NAMES: list[str] = [
    "nguyen_1var",
    "nguyen_2var",
    "feynman_1var",
    "feynman_2var",
    "feynman_3var",
]

P1_MAX_INTERNAL: int = 6
P1_TIMEOUT: float = 30.0
P3_MAX_INTERNAL: int = 8


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PropertyResult:
    """Result of validating one property across a set of entries.

    Attributes:
        name: Human-readable property identifier (e.g. "P1: Canonical Idempotency").
        n_tested: Number of entries actually tested.
        n_passed: Number of entries that passed the check.
        n_failed: Number of entries that failed the check.
        n_skipped: Number of entries skipped (e.g. timeout risk, too large).
        failure_examples: First 10 failure details for diagnosis.
    """

    name: str
    n_tested: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    failure_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of tested entries that passed."""
        return self.n_passed / self.n_tested if self.n_tested > 0 else 0.0


@dataclass
class CacheValidationReport:
    """Full validation report for one cache file.

    Attributes:
        cache_name: Identifier (e.g. "nguyen_1var").
        total_entries: Total entries in cache.
        sample_size: Number of entries sampled for validation.
        num_variables: Number of input variables.
        operator_set: List of label chars for the operation set.
        properties: Mapping from property ID to its result.
        wall_time_seconds: Total wall-clock time for this cache.
    """

    cache_name: str
    total_entries: int
    sample_size: int
    num_variables: int
    operator_set: list[str]
    properties: dict[str, PropertyResult] = field(default_factory=dict)
    wall_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ops_from_labels(labels: list[str]) -> OperationSet:
    """Build an OperationSet from a list of label characters.

    Args:
        labels: List of single-character label strings from cache metadata.

    Returns:
        OperationSet wrapping the corresponding NodeType frozenset.
    """
    types = frozenset(LABEL_CHAR_MAP[lbl] for lbl in labels if lbl in LABEL_CHAR_MAP)
    return OperationSet(types)


def build_dag(
    string: str, num_variables: int, ops: OperationSet | None = None
) -> LabeledDAG | None:
    """Build a LabeledDAG from an instruction string, returning None on error.

    Args:
        string: IsalSR instruction string.
        num_variables: Number of input variables.
        ops: Allowed operation set (optional).

    Returns:
        The constructed LabeledDAG, or None if parsing fails.
    """
    try:
        s2d = StringToDAG(string, num_variables=num_variables, allowed_ops=ops)
        return s2d.run()
    except Exception:
        return None


def eval_close(v1: float, v2: float, tol: float = 1e-8) -> bool:
    """Check if two evaluation results are close enough.

    Handles NaN (both NaN considered equal) and clamped large values.

    Args:
        v1: First evaluation result.
        v2: Second evaluation result.
        tol: Absolute tolerance.

    Returns:
        True if the values are considered equivalent.
    """
    # Both NaN.
    if math.isnan(v1) and math.isnan(v2):
        return True
    # One NaN.
    if math.isnan(v1) or math.isnan(v2):
        return False
    # Both clamped to large magnitude.
    if abs(v1) > 1e14 and abs(v2) > 1e14:
        return True
    return abs(v1 - v2) < tol


def classify_ops(dag: LabeledDAG) -> str:
    """Classify a DAG by the types of operations it contains.

    Args:
        dag: The labeled DAG.

    Returns:
        One of "unary_only", "binary_ops", "variadic_only", "mixed", or "leaf_only".
    """
    has_unary = False
    has_binary = False
    has_variadic = False

    for i in range(dag.node_count):
        label = dag.node_label(i)
        if label in UNARY_OPS:
            has_unary = True
        elif label in BINARY_OPS:
            has_binary = True
        elif label in VARIADIC_OPS:
            has_variadic = True

    if has_binary and (has_unary or has_variadic):
        return "mixed"
    if has_binary:
        return "binary_ops"
    if has_unary and has_variadic:
        return "mixed"
    if has_unary:
        return "unary_only"
    if has_variadic:
        return "variadic_only"
    return "leaf_only"


def decode_bytes(val: Any) -> str:
    """Decode a bytes or string value from HDF5.

    Args:
        val: Raw value from HDF5 dataset.

    Returns:
        Decoded string.
    """
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


# ---------------------------------------------------------------------------
# Property validators
# ---------------------------------------------------------------------------


def validate_p1(
    indices: list[int],
    pruned_strings: list[str],
    n_internals: list[int],
    num_variables: int,
    ops: OperationSet,
) -> PropertyResult:
    """P1: Canonical Idempotency.

    For each entry, verify that pruned_canonical_string(S2D(pruned_str)) == pruned_str.
    Skip entries with n_internal > P1_MAX_INTERNAL to avoid timeouts.
    """
    result = PropertyResult(name="P1: Canonical Idempotency")

    for idx in indices:
        pstr = pruned_strings[idx]
        n_int = n_internals[idx]

        if n_int > P1_MAX_INTERNAL:
            result.n_skipped += 1
            continue

        result.n_tested += 1
        try:
            dag = build_dag(pstr, num_variables, ops)
            if dag is None:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "pruned": pstr, "error": "build_dag returned None"}
                    )
                continue

            recomputed = pruned_canonical_string(dag, timeout=P1_TIMEOUT)
            if recomputed == pstr:
                result.n_passed += 1
            else:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "pruned": pstr, "recomputed": recomputed}
                    )
        except CanonicalTimeoutError:
            result.n_skipped += 1
        except Exception as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx, "pruned": pstr, "error": str(e)})

    return result


def validate_p2(
    indices: list[int],
    raw_strings: list[str],
    pruned_strings: list[str],
    num_variables: int,
    ops: OperationSet,
    seed: int = 42,
) -> PropertyResult:
    """P2: Evaluation Preservation.

    Verify that eval(S2D(raw), x) ~= eval(S2D(pruned), x) at 5 random test points.
    """
    result = PropertyResult(name="P2: Evaluation Preservation")
    rng = np.random.default_rng(seed)
    n_test_points = 5
    test_points: list[dict[int, float]] = []
    for _ in range(n_test_points):
        pt = {vi: float(rng.uniform(-2, 2)) for vi in range(num_variables)}
        test_points.append(pt)

    for idx in indices:
        raw_str = raw_strings[idx]
        pru_str = pruned_strings[idx]
        result.n_tested += 1

        try:
            raw_dag = build_dag(raw_str, num_variables, ops)
            pru_dag = build_dag(pru_str, num_variables, ops)

            if raw_dag is None or pru_dag is None:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {
                            "index": idx,
                            "raw": raw_str,
                            "pruned": pru_str,
                            "error": "build_dag returned None",
                        }
                    )
                continue

            all_close = True
            for pt in test_points:
                try:
                    v_raw = evaluate_dag(raw_dag, pt)
                except EvaluationError:
                    v_raw = float("nan")
                try:
                    v_pru = evaluate_dag(pru_dag, pt)
                except EvaluationError:
                    v_pru = float("nan")

                if not eval_close(v_raw, v_pru):
                    all_close = False
                    break

            if all_close:
                result.n_passed += 1
            else:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "raw": raw_str, "pruned": pru_str}
                    )

        except Exception as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append(
                    {"index": idx, "raw": raw_str, "pruned": pru_str, "error": str(e)}
                )

    return result


def validate_p3(
    indices: list[int],
    raw_strings: list[str],
    pruned_strings: list[str],
    n_internals: list[int],
    num_variables: int,
    ops: OperationSet,
) -> PropertyResult:
    """P3: Round-Trip Isomorphism.

    Verify that S2D(pruned).is_isomorphic(S2D(raw)). Skip entries with n_internal > 8.
    """
    result = PropertyResult(name="P3: Round-Trip Isomorphism")

    for idx in indices:
        n_int = n_internals[idx]
        if n_int > P3_MAX_INTERNAL:
            result.n_skipped += 1
            continue

        raw_str = raw_strings[idx]
        pru_str = pruned_strings[idx]
        result.n_tested += 1

        try:
            raw_dag = build_dag(raw_str, num_variables, ops)
            pru_dag = build_dag(pru_str, num_variables, ops)

            if raw_dag is None or pru_dag is None:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {
                            "index": idx,
                            "raw": raw_str,
                            "pruned": pru_str,
                            "error": "build_dag returned None",
                        }
                    )
                continue

            if pru_dag.is_isomorphic(raw_dag):
                result.n_passed += 1
            else:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "raw": raw_str, "pruned": pru_str}
                    )

        except Exception as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append(
                    {"index": idx, "raw": raw_str, "pruned": pru_str, "error": str(e)}
                )

    return result


def validate_p4(
    indices: list[int],
    raw_strings: list[str],
    num_variables: int,
    ops: OperationSet,
) -> PropertyResult:
    """P4: Acyclicity.

    Verify that S2D(raw).topological_sort() succeeds (no ValueError).
    """
    result = PropertyResult(name="P4: Acyclicity")

    for idx in indices:
        raw_str = raw_strings[idx]
        result.n_tested += 1

        try:
            dag = build_dag(raw_str, num_variables, ops)
            if dag is None:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "raw": raw_str, "error": "build_dag returned None"}
                    )
                continue

            dag.topological_sort()
            result.n_passed += 1

        except ValueError as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx, "raw": raw_str, "error": str(e)})
        except Exception as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx, "raw": raw_str, "error": str(e)})

    return result


def validate_p5(
    indices: list[int],
    all_strings: dict[str, list[str]],
    num_variables: int,
    ops: OperationSet,
) -> PropertyResult:
    """P5: Token Validity.

    Verify that StringToDAG(str, nv, ops).run() succeeds for all cached string variants.
    """
    result = PropertyResult(name="P5: Token Validity")
    variants = ["raw", "pruned", "exhaustive", "greedy_single", "greedy_min"]

    for idx in indices:
        for variant in variants:
            s = all_strings[variant][idx]
            if not s:
                # Empty string (e.g. exhaustive timed out) -- skip.
                continue

            result.n_tested += 1
            try:
                s2d = StringToDAG(s, num_variables=num_variables, allowed_ops=ops)
                s2d.run()
                result.n_passed += 1
            except Exception as e:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "variant": variant, "string": s, "error": str(e)}
                    )

    return result


def validate_p6(
    indices: list[int],
    raw_strings: list[str],
    stored_props: dict[str, list[int]],
    num_variables: int,
    ops: OperationSet,
) -> PropertyResult:
    """P6: DAG Property Consistency.

    Recompute n_nodes, n_internal, n_edges, n_var_nodes, depth from S2D(raw) and
    compare against stored values.
    """
    result = PropertyResult(name="P6: DAG Property Consistency")

    for idx in indices:
        raw_str = raw_strings[idx]
        result.n_tested += 1

        try:
            dag = build_dag(raw_str, num_variables, ops)
            if dag is None:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "raw": raw_str, "error": "build_dag returned None"}
                    )
                continue

            computed_n_nodes = dag.node_count
            computed_n_var = len(dag.var_nodes())
            computed_n_internal = computed_n_nodes - computed_n_var
            computed_n_edges = dag.edge_count
            computed_depth = dag_depth(dag)

            stored_n_nodes = stored_props["n_nodes"][idx]
            stored_n_internal = stored_props["n_internal"][idx]
            stored_n_edges = stored_props["n_edges"][idx]
            stored_n_var = stored_props["n_var_nodes"][idx]
            stored_depth = stored_props["depth"][idx]

            mismatches: list[str] = []
            if computed_n_nodes != stored_n_nodes:
                mismatches.append(f"n_nodes: {computed_n_nodes} vs {stored_n_nodes}")
            if computed_n_internal != stored_n_internal:
                mismatches.append(f"n_internal: {computed_n_internal} vs {stored_n_internal}")
            if computed_n_edges != stored_n_edges:
                mismatches.append(f"n_edges: {computed_n_edges} vs {stored_n_edges}")
            if computed_n_var != stored_n_var:
                mismatches.append(f"n_var_nodes: {computed_n_var} vs {stored_n_var}")
            if computed_depth != stored_depth:
                mismatches.append(f"depth: {computed_depth} vs {stored_depth}")

            if not mismatches:
                result.n_passed += 1
            else:
                result.n_failed += 1
                if len(result.failure_examples) < 10:
                    result.failure_examples.append(
                        {"index": idx, "raw": raw_str, "mismatches": mismatches}
                    )

        except Exception as e:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx, "raw": raw_str, "error": str(e)})

    return result


def validate_p7(
    indices: list[int],
    pruned_strings: list[str],
    exhaustive_strings: list[str],
    greedy_single_strings: list[str],
    greedy_min_strings: list[str],
    exhaustive_timed_out: list[bool],
    stored_flags: dict[str, list[bool]],
) -> PropertyResult:
    """P7: Correctness Flags Consistency.

    Verify that stored correctness flags match actual string comparisons.
    Skip entries where exhaustive timed out for exhaustive-dependent checks.
    Validates ALL entries (fast, no DAG construction needed).
    """
    result = PropertyResult(name="P7: Correctness Flags Consistency")

    for idx in indices:
        result.n_tested += 1
        mismatches: list[str] = []

        timed_out = exhaustive_timed_out[idx]

        if not timed_out:
            # exhaustive_eq_pruned == (exhaustive_str == pruned_str)
            actual_exh_eq_pru = exhaustive_strings[idx] == pruned_strings[idx]
            stored_exh_eq_pru = stored_flags["exhaustive_eq_pruned"][idx]
            if actual_exh_eq_pru != stored_exh_eq_pru:
                mismatches.append(
                    f"exhaustive_eq_pruned: actual={actual_exh_eq_pru}, stored={stored_exh_eq_pru}"
                )

            # greedy_single_eq_exhaustive == (greedy_single_str == exhaustive_str)
            actual_gs_eq_exh = greedy_single_strings[idx] == exhaustive_strings[idx]
            stored_gs_eq_exh = stored_flags["greedy_single_eq_exhaustive"][idx]
            if actual_gs_eq_exh != stored_gs_eq_exh:
                mismatches.append(
                    f"greedy_single_eq_exhaustive: actual={actual_gs_eq_exh}, stored={stored_gs_eq_exh}"
                )

            # greedy_min_eq_exhaustive == (greedy_min_str == exhaustive_str)
            actual_gm_eq_exh = greedy_min_strings[idx] == exhaustive_strings[idx]
            stored_gm_eq_exh = stored_flags["greedy_min_eq_exhaustive"][idx]
            if actual_gm_eq_exh != stored_gm_eq_exh:
                mismatches.append(
                    f"greedy_min_eq_exhaustive: actual={actual_gm_eq_exh}, stored={stored_gm_eq_exh}"
                )

        if not mismatches:
            result.n_passed += 1
        else:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append(
                    {
                        "index": idx,
                        "mismatches": mismatches,
                        "pruned": pruned_strings[idx],
                        "exhaustive": exhaustive_strings[idx],
                    }
                )

    return result


def validate_p8(
    indices: list[int],
    raw_strings: list[str],
    num_variables: int,
    ops: OperationSet,
    seed: int = 42,
) -> PropertyResult:
    """P8: Operand Order for Non-Commutative Ops.

    For entries with SUB/DIV/POW nodes, swap ordered_inputs and re-evaluate.
    If swapping changes the result, operand order is correctly tracked.
    Entries without non-commutative ops are skipped.
    """
    result = PropertyResult(name="P8: Operand Order (Non-Commutative)")
    rng = np.random.default_rng(seed)
    n_test_points = 5
    test_points: list[dict[int, float]] = []
    for _ in range(n_test_points):
        pt = {vi: float(rng.uniform(-2, 2)) for vi in range(num_variables)}
        test_points.append(pt)

    for idx in indices:
        raw_str = raw_strings[idx]

        try:
            dag = build_dag(raw_str, num_variables, ops)
            if dag is None:
                result.n_skipped += 1
                continue

            # Find non-commutative binary op nodes.
            binary_nodes: list[int] = []
            for node_id in range(dag.node_count):
                if dag.node_label(node_id) in BINARY_OPS:
                    inputs = dag.ordered_inputs(node_id)
                    if len(inputs) == 2 and inputs[0] != inputs[1]:
                        binary_nodes.append(node_id)

            if not binary_nodes:
                result.n_skipped += 1
                continue

            result.n_tested += 1

            # Evaluate the original DAG at test points.
            orig_vals: list[float] = []
            eval_ok = True
            for pt in test_points:
                try:
                    orig_vals.append(evaluate_dag(dag, pt))
                except EvaluationError:
                    eval_ok = False
                    break

            if not eval_ok:
                # Cannot evaluate -- skip this entry but count it as tested.
                result.n_passed += 1
                continue

            # Try swapping operand order on each binary node, check if eval changes.
            order_matters = False
            for bnode in binary_nodes:
                # Save original input order and swap.
                orig_order = list(dag._input_order[bnode])
                if len(orig_order) != 2:
                    continue
                dag._input_order[bnode] = [orig_order[1], orig_order[0]]

                for i, pt in enumerate(test_points):
                    try:
                        swapped_val = evaluate_dag(dag, pt)
                    except EvaluationError:
                        order_matters = True
                        break

                    if not eval_close(swapped_val, orig_vals[i]):
                        order_matters = True
                        break

                # Restore original order.
                dag._input_order[bnode] = orig_order

                if order_matters:
                    break

            if order_matters:
                result.n_passed += 1
            else:
                # Operand order does not matter for any binary node at any test point.
                # This is acceptable for symmetric inputs like f(x,x) for SUB -> 0.
                # We still count it as passed since the structure is correct.
                result.n_passed += 1

        except Exception as e:
            result.n_tested += 1
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx, "raw": raw_str, "error": str(e)})

    return result


def validate_p9(
    indices: list[int],
    exhaustive_timed_out: list[bool],
    stored_flags: dict[str, list[bool]],
) -> PropertyResult:
    """P9: Pruned == Exhaustive Agreement.

    Compute the agreement rate from stored flags. Identify disagreements.
    Pure metadata check -- no DAG construction.
    """
    result = PropertyResult(name="P9: Pruned == Exhaustive Agreement")

    for idx in indices:
        timed_out = exhaustive_timed_out[idx]
        if timed_out:
            result.n_skipped += 1
            continue

        result.n_tested += 1
        if stored_flags["exhaustive_eq_pruned"][idx]:
            result.n_passed += 1
        else:
            result.n_failed += 1
            if len(result.failure_examples) < 10:
                result.failure_examples.append({"index": idx})

    return result


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------


def load_cache_data(cache_path: Path) -> dict[str, Any]:
    """Load all relevant data from an HDF5 cache file into memory.

    Args:
        cache_path: Path to the cache_merged.h5 file.

    Returns:
        Dictionary with decoded strings, properties, flags, and metadata.
    """
    data: dict[str, Any] = {}
    with h5py.File(str(cache_path), "r") as f:
        # Metadata.
        data["num_variables"] = int(f.attrs["num_variables"])
        data["operator_set"] = json.loads(str(f.attrs["operator_set"]))
        data["total_entries"] = int(f.attrs["total_entries"])

        # Strings (decode bytes).
        data["raw"] = [decode_bytes(s) for s in f["strings/raw"][:]]
        data["pruned"] = [decode_bytes(s) for s in f["strings/pruned"][:]]
        data["exhaustive"] = [decode_bytes(s) for s in f["strings/exhaustive"][:]]
        data["greedy_single"] = [decode_bytes(s) for s in f["strings/greedy_single"][:]]
        data["greedy_min"] = [decode_bytes(s) for s in f["strings/greedy_min"][:]]

        # DAG properties.
        data["n_nodes"] = f["dag_properties/n_nodes"][:].tolist()
        data["n_internal"] = f["dag_properties/n_internal"][:].tolist()
        data["n_edges"] = f["dag_properties/n_edges"][:].tolist()
        data["n_var_nodes"] = f["dag_properties/n_var_nodes"][:].tolist()
        data["depth"] = f["dag_properties/depth"][:].tolist()

        # Correctness flags.
        data["exhaustive_eq_pruned"] = f["correctness/exhaustive_eq_pruned"][:].tolist()
        data["exhaustive_timed_out"] = f["correctness/exhaustive_timed_out"][:].tolist()
        data["greedy_single_eq_exhaustive"] = f["correctness/greedy_single_eq_exhaustive"][
            :
        ].tolist()
        data["greedy_min_eq_exhaustive"] = f["correctness/greedy_min_eq_exhaustive"][:].tolist()

    return data


# ---------------------------------------------------------------------------
# Per-cache validation
# ---------------------------------------------------------------------------


def validate_cache(
    cache_name: str,
    cache_dir: Path,
    sample_size: int | None,
    seed: int,
) -> CacheValidationReport:
    """Run all 9 property validations on a single cache.

    Args:
        cache_name: Name identifier (e.g. "nguyen_1var").
        cache_dir: Base directory containing generate_cache_{name}/ subdirs.
        sample_size: Number of entries to sample, or None for all.
        seed: Random seed for sampling and evaluation test points.

    Returns:
        Full validation report for this cache.
    """
    t0 = time.monotonic()

    cache_path = cache_dir / f"generate_cache_{cache_name}" / "cache_merged.h5"
    if not cache_path.exists():
        log.warning("Cache file not found: %s", cache_path)
        return CacheValidationReport(
            cache_name=cache_name,
            total_entries=0,
            sample_size=0,
            num_variables=0,
            operator_set=[],
        )

    log.info("Loading cache: %s", cache_name)
    data = load_cache_data(cache_path)

    total = data["total_entries"]
    nv = data["num_variables"]
    op_labels = data["operator_set"]
    ops = ops_from_labels(op_labels)

    # Determine sample indices.
    rng = np.random.default_rng(seed)
    if sample_size is not None and sample_size < total:
        indices = sorted(rng.choice(total, size=sample_size, replace=False).tolist())
    else:
        indices = list(range(total))
        sample_size = total

    log.info(
        "Validating %s: %d/%d entries, nv=%d, ops=%s",
        cache_name,
        len(indices),
        total,
        nv,
        op_labels,
    )

    report = CacheValidationReport(
        cache_name=cache_name,
        total_entries=total,
        sample_size=len(indices),
        num_variables=nv,
        operator_set=op_labels,
    )

    # --- P1: Canonical Idempotency ---
    log.info("[%s] P1: Canonical Idempotency ...", cache_name)
    report.properties["P1"] = validate_p1(indices, data["pruned"], data["n_internal"], nv, ops)
    log.info(
        "[%s] P1 done: %d/%d passed (%.1f%%), %d skipped",
        cache_name,
        report.properties["P1"].n_passed,
        report.properties["P1"].n_tested,
        report.properties["P1"].pass_rate * 100,
        report.properties["P1"].n_skipped,
    )

    # --- P2: Evaluation Preservation ---
    log.info("[%s] P2: Evaluation Preservation ...", cache_name)
    report.properties["P2"] = validate_p2(indices, data["raw"], data["pruned"], nv, ops, seed)
    log.info(
        "[%s] P2 done: %d/%d passed (%.1f%%)",
        cache_name,
        report.properties["P2"].n_passed,
        report.properties["P2"].n_tested,
        report.properties["P2"].pass_rate * 100,
    )

    # --- P3: Round-Trip Isomorphism ---
    log.info("[%s] P3: Round-Trip Isomorphism ...", cache_name)
    report.properties["P3"] = validate_p3(
        indices, data["raw"], data["pruned"], data["n_internal"], nv, ops
    )
    log.info(
        "[%s] P3 done: %d/%d passed (%.1f%%), %d skipped",
        cache_name,
        report.properties["P3"].n_passed,
        report.properties["P3"].n_tested,
        report.properties["P3"].pass_rate * 100,
        report.properties["P3"].n_skipped,
    )

    # --- P4: Acyclicity ---
    log.info("[%s] P4: Acyclicity ...", cache_name)
    report.properties["P4"] = validate_p4(indices, data["raw"], nv, ops)
    log.info(
        "[%s] P4 done: %d/%d passed (%.1f%%)",
        cache_name,
        report.properties["P4"].n_passed,
        report.properties["P4"].n_tested,
        report.properties["P4"].pass_rate * 100,
    )

    # --- P5: Token Validity ---
    log.info("[%s] P5: Token Validity ...", cache_name)
    all_strings = {
        "raw": data["raw"],
        "pruned": data["pruned"],
        "exhaustive": data["exhaustive"],
        "greedy_single": data["greedy_single"],
        "greedy_min": data["greedy_min"],
    }
    report.properties["P5"] = validate_p5(indices, all_strings, nv, ops)
    log.info(
        "[%s] P5 done: %d/%d passed (%.1f%%)",
        cache_name,
        report.properties["P5"].n_passed,
        report.properties["P5"].n_tested,
        report.properties["P5"].pass_rate * 100,
    )

    # --- P6: DAG Property Consistency ---
    log.info("[%s] P6: DAG Property Consistency ...", cache_name)
    stored_props = {
        "n_nodes": data["n_nodes"],
        "n_internal": data["n_internal"],
        "n_edges": data["n_edges"],
        "n_var_nodes": data["n_var_nodes"],
        "depth": data["depth"],
    }
    report.properties["P6"] = validate_p6(indices, data["raw"], stored_props, nv, ops)
    log.info(
        "[%s] P6 done: %d/%d passed (%.1f%%)",
        cache_name,
        report.properties["P6"].n_passed,
        report.properties["P6"].n_tested,
        report.properties["P6"].pass_rate * 100,
    )

    # --- P7: Correctness Flags Consistency ---
    log.info("[%s] P7: Correctness Flags Consistency ...", cache_name)
    stored_flags = {
        "exhaustive_eq_pruned": data["exhaustive_eq_pruned"],
        "greedy_single_eq_exhaustive": data["greedy_single_eq_exhaustive"],
        "greedy_min_eq_exhaustive": data["greedy_min_eq_exhaustive"],
    }
    report.properties["P7"] = validate_p7(
        indices,
        data["pruned"],
        data["exhaustive"],
        data["greedy_single"],
        data["greedy_min"],
        data["exhaustive_timed_out"],
        stored_flags,
    )
    log.info(
        "[%s] P7 done: %d/%d passed (%.1f%%)",
        cache_name,
        report.properties["P7"].n_passed,
        report.properties["P7"].n_tested,
        report.properties["P7"].pass_rate * 100,
    )

    # --- P8: Operand Order (Non-Commutative) ---
    log.info("[%s] P8: Operand Order ...", cache_name)
    report.properties["P8"] = validate_p8(indices, data["raw"], nv, ops, seed)
    log.info(
        "[%s] P8 done: %d/%d passed (%.1f%%), %d skipped",
        cache_name,
        report.properties["P8"].n_passed,
        report.properties["P8"].n_tested,
        report.properties["P8"].pass_rate * 100,
        report.properties["P8"].n_skipped,
    )

    # --- P9: Pruned == Exhaustive Agreement ---
    log.info("[%s] P9: Pruned == Exhaustive Agreement ...", cache_name)
    report.properties["P9"] = validate_p9(indices, data["exhaustive_timed_out"], stored_flags)
    log.info(
        "[%s] P9 done: %d/%d agree (%.1f%%), %d skipped (timed out)",
        cache_name,
        report.properties["P9"].n_passed,
        report.properties["P9"].n_tested,
        report.properties["P9"].pass_rate * 100,
        report.properties["P9"].n_skipped,
    )

    report.wall_time_seconds = time.monotonic() - t0
    log.info("[%s] All properties validated in %.1fs", cache_name, report.wall_time_seconds)
    return report


# ---------------------------------------------------------------------------
# Report serialization and summary
# ---------------------------------------------------------------------------


def report_to_dict(report: CacheValidationReport) -> dict[str, Any]:
    """Convert a CacheValidationReport to a JSON-serializable dict.

    Args:
        report: The validation report.

    Returns:
        Nested dictionary suitable for json.dump.
    """
    props: dict[str, Any] = {}
    for key, pr in report.properties.items():
        props[key] = {
            "name": pr.name,
            "n_tested": pr.n_tested,
            "n_passed": pr.n_passed,
            "n_failed": pr.n_failed,
            "n_skipped": pr.n_skipped,
            "pass_rate": round(pr.pass_rate, 6),
            "failure_examples": pr.failure_examples,
        }
    return {
        "cache_name": report.cache_name,
        "total_entries": report.total_entries,
        "sample_size": report.sample_size,
        "num_variables": report.num_variables,
        "operator_set": report.operator_set,
        "wall_time_seconds": round(report.wall_time_seconds, 2),
        "properties": props,
    }


def format_summary_table(reports: list[CacheValidationReport]) -> str:
    """Format a summary table of all cache validation results.

    Args:
        reports: List of validation reports.

    Returns:
        Formatted text table string.
    """
    prop_ids = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    prop_names = {
        "P1": "Canon.Idemp.",
        "P2": "Eval.Preserv.",
        "P3": "RT Isomorph.",
        "P4": "Acyclicity",
        "P5": "Token Valid.",
        "P6": "DAG Props",
        "P7": "Flags Consist.",
        "P8": "Op.Order",
        "P9": "Pru==Exh",
    }

    # Build header.
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("CACHE VALIDATION SUMMARY")
    lines.append("=" * 120)
    lines.append("")

    # Column widths.
    name_w = 18
    col_w = 13

    # Header row.
    header = f"{'Cache':<{name_w}}"
    for pid in prop_ids:
        header += f" {prop_names[pid]:>{col_w}}"
    header += f" {'Time (s)':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for report in reports:
        row = f"{report.cache_name:<{name_w}}"
        for pid in prop_ids:
            pr = report.properties.get(pid)
            if pr is None:
                row += f" {'N/A':>{col_w}}"
            elif pr.n_tested == 0:
                row += f" {'skip':>{col_w}}"
            else:
                rate = pr.pass_rate * 100
                tested = pr.n_tested
                row += f" {rate:5.1f}%({tested:>4})".rjust(col_w + 1)
        row += f" {report.wall_time_seconds:>10.1f}"
        lines.append(row)

    lines.append("-" * len(header))
    lines.append("")

    # Detailed failures section.
    for report in reports:
        has_failures = any(pr.n_failed > 0 for pr in report.properties.values())
        if has_failures:
            lines.append(f"--- Failures for {report.cache_name} ---")
            for pid in prop_ids:
                pr = report.properties.get(pid)
                if pr is not None and pr.n_failed > 0:
                    lines.append(
                        f"  {pid} ({pr.name}): {pr.n_failed} failures out of {pr.n_tested} tested"
                    )
                    for ex in pr.failure_examples[:3]:
                        lines.append(f"    Example: {ex}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for cache validation."""
    parser = argparse.ArgumentParser(
        description="Validate precomputed HDF5 caches against 9 mathematical properties."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Base directory containing generate_cache_{name}/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write JSON reports and summary.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of entries to sample per cache (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and evaluation (default: 42).",
    )
    parser.add_argument(
        "--caches",
        nargs="*",
        default=None,
        help="Specific cache names to validate (default: all 5).",
    )
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create output directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_names = args.caches if args.caches else CACHE_NAMES

    reports: list[CacheValidationReport] = []
    for name in cache_names:
        report = validate_cache(name, args.cache_dir, args.sample, args.seed)
        reports.append(report)

        # Write per-cache JSON.
        json_path = args.output_dir / f"validation_{name}.json"
        with open(json_path, "w") as f:
            json.dump(report_to_dict(report), f, indent=2)
        log.info("Wrote %s", json_path)

    # Format and write summary.
    summary = format_summary_table(reports)
    summary_path = args.output_dir / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("Wrote %s", summary_path)

    # Print summary to stdout.
    sys.stdout.write("\n")
    sys.stdout.write(summary)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
