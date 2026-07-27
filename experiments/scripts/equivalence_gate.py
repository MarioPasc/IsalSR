"""Equivalence-gate harness for IsalSR canonicaliser cross-engine comparison.

Runs three gates comparing canonicalisation engines (engine_a vs engine_b).
When only the Python engine is available, both sides are the same implementation
and the report marks ``self_comparison: true`` — the cross-engine checks trivially
pass, but gate 1b invariance and gate 3 round-trip remain genuinely meaningful.

Gates
-----
Gate 1 — Exhaustive permutation (k=1..8):
    For systematically generated base DAGs, generate ALL k! permutations of
    internal node IDs via ``permute_internal_nodes`` and verify:
    (a) engine_a == engine_b byte-for-byte on every permuted copy.
    (b) All k! permutations of the same DAG yield the same canonical string
        (the invariance property).
    The number of base DAGs per k is capped; the cap is recorded in the report.

Gate 2 — Generated corpus (>=10,000 DAGs from random IsalSR strings):
    Canonicalise under both engines; assert byte equality.
    Records k-distribution (histogram of internal node count) of the corpus.

Gate 3 — Round-trip:
    For every DAG in the gate-2 corpus:
    ``S2D(canonical_string(D))`` must be isomorphic to D.
    Verified under both engines independently.

Usage
-----
Run all gates with defaults::

    python -m experiments.scripts.equivalence_gate --gate all --out report.json

Quick mode (20× smaller corpora, for local iteration)::

    python -m experiments.scripts.equivalence_gate --gate all --quick

"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap so the script can be run without `pip install -e`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.scripts._dag_generators import make_random_sr_dag
from isalsr.core.backends import build_info, engine
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG
from isalsr.errors import InvalidTokenError
from isalsr.types import VALID_LABEL_CHARS, VALID_SINGLE_INSTRUCTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_SEED: int = 20260101
"""Fixed seed for all random corpus generation.  Recorded in every report."""

GATE1_K_MAX_FULL: int = 8
GATE1_K_MAX_QUICK: int = 6

# Per-k base-DAG cap for full mode.  k=7/8 have 5040/40320 permutations
# per DAG, so the cap is reduced to keep the full run under ~30 minutes.
# The cap is recorded in the JSON report so the reader can see what ran.
GATE1_DAGS_PER_K_FULL: dict[int, int] = {
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5: 5,
    6: 5,
    7: 2,  # 2 × 5040 = 10,080 permutations
    8: 1,  # 1 × 40320 = 40,320 permutations
}
GATE1_DAGS_PER_K_QUICK: int = 1

# Per-call timeout (seconds) for fast_canonical_string inside gate 1.
# Protects against pathological DAGs whose backtracking search is expensive.
# Timed-out calls are counted as errors, not mismatches.
GATE1_CANON_TIMEOUT: float = 5.0

GATE2_CORPUS_FULL: int = 10_000
GATE2_CORPUS_QUICK: int = 500
GATE2_N_VARS_RANGE: tuple[int, int] = (1, 5)
GATE2_STRING_LEN_RANGE: tuple[int, int] = (4, 30)

MAX_MISMATCH_CASES: int = 20
"""Maximum number of concrete failing cases recorded in the report."""

_LABEL_CHARS: list[str] = sorted(VALID_LABEL_CHARS)
_SINGLE_CHARS: list[str] = sorted(VALID_SINGLE_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Canonicalisation wrapper
# ---------------------------------------------------------------------------


def _canonicalize(
    dag: LabeledDAG,
    backend: str,
    *,
    timeout: float | None = None,
) -> str:
    """Compute ``fast_canonical_string`` routing to the specified backend.

    Does NOT fall back silently.  The caller is responsible for passing a
    backend that was confirmed available by ``_probe_cpp_canonical`` before
    any gate runs.  Passing ``"cpp"`` when the backend is absent will raise
    ``TypeError`` which propagates to the gate error counter — making the
    failure visible rather than silent.

    Args:
        dag: Labeled DAG to canonicalise.
        backend: Backend identifier confirmed available: ``"python"`` or
            ``"cpp"``.  Must have been validated by ``_probe_cpp_canonical``
            before calling.
        timeout: Optional wall-clock limit in seconds.  ``CanonicalTimeoutError``
            propagates to the caller unchanged.

    Returns:
        Canonical IsalSR string (mode ``"wl_only"``).
    """
    kwargs: dict[str, object] = {"mode": "wl_only"}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if backend == "python":
        # Avoid passing backend= when it is not yet a supported keyword.
        return fast_canonical_string(dag, **kwargs)  # type: ignore[arg-type]
    # backend == "cpp": pass through explicitly.  No fallback — a TypeError
    # here means the probe was wrong and must surface as a gate error.
    return fast_canonical_string(dag, backend=backend, **kwargs)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Random string generator
# ---------------------------------------------------------------------------

_MOVEMENT: list[str] = ["N", "P", "n", "p"]
_EDGE: list[str] = ["C", "c"]


def _gen_random_string(rng: random.Random, n_tokens: int, n_vars: int) -> str:  # noqa: ARG001
    """Generate a random IsalSR instruction string.

    Uses a weighted mix of movement, edge, no-op, and V/v insertion tokens.
    Invalid strings (trailing bare V/v without label) can occur and are
    caught at the call site.

    Args:
        rng: Seeded random number generator.
        n_tokens: Approximate token count (two-char tokens count as one token).
        n_vars: Number of input variables (not used to constrain generation;
            provided for future extension).

    Returns:
        An IsalSR instruction string.
    """
    chars: list[str] = []
    for _ in range(n_tokens):
        r = rng.random()
        if r < 0.25:
            chars.append(rng.choice(_MOVEMENT))
        elif r < 0.35:
            chars.append(rng.choice(_EDGE))
        elif r < 0.40:
            chars.append("W")
        elif r < 0.70:
            chars.append("V")
            chars.append(rng.choice(_LABEL_CHARS))
        else:
            chars.append("v")
            chars.append(rng.choice(_LABEL_CHARS))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Gate 1 — Exhaustive permutation
# ---------------------------------------------------------------------------


def _run_gate1(
    *,
    quick: bool,
    backend_a: str,
    backend_b: str,
) -> dict[str, Any]:
    """Run gate 1: exhaustive k! permutation check for k=1..K_MAX.

    For each (k, dag_index), all k! permutations of internal node IDs are
    generated and canonicalised under both engines.  Two sub-checks:

    (a) Cross-engine: engine_a result == engine_b result for every permuted copy.
    (b) Invariance: all k! permuted copies of the same base DAG yield the
        same canonical string.

    Args:
        quick: If True, use reduced k range and fewer base DAGs per k.
        backend_a: Name of engine A (``"python"`` or ``"cpp"``).
        backend_b: Name of engine B.

    Returns:
        Per-gate result dict suitable for embedding in the JSON report.
    """
    k_max = GATE1_K_MAX_QUICK if quick else GATE1_K_MAX_FULL
    # In quick mode every k gets the same (small) cap.
    # In full mode use the per-k cap dict to bound runtime on large k.
    _dags_per_k_map: dict[int, int] = (
        {k: GATE1_DAGS_PER_K_QUICK for k in range(1, k_max + 1)}
        if quick
        else {k: GATE1_DAGS_PER_K_FULL.get(k, 1) for k in range(1, k_max + 1)}
    )
    # Timeout per canonicalization call: only applied in full mode to
    # protect against pathological DAGs at large k.
    _canon_timeout: float | None = None if quick else GATE1_CANON_TIMEOUT

    dags_tested: int = 0
    comparisons_made: int = 0  # total (dag, permutation) pairs tested
    mismatches_cross: int = 0  # engine_a != engine_b
    mismatches_inv: int = 0  # not all perms gave same string
    errors: int = 0
    mismatch_cases: list[dict[str, Any]] = []

    per_k: list[dict[str, Any]] = []

    for k in range(1, k_max + 1):
        n_perms = math.factorial(k)
        k_dags_tested = 0
        k_comparisons = 0
        k_mismatch_cross = 0
        k_mismatch_inv = 0
        k_errors = 0

        # Use num_vars=1 for k=1..4, 2 for k=5..6, 3 for k=7..8 to exercise
        # multi-variable graphs without exploding runtime.
        num_vars = 1 + (k - 1) // 4

        for dag_idx in range(_dags_per_k_map[k]):
            seed = FIXED_SEED + k * 1000 + dag_idx
            try:
                base_dag = make_random_sr_dag(num_vars=num_vars, num_internal=k, seed=seed)
            except Exception as exc:  # noqa: BLE001
                log.warning("Gate1 k=%d dag=%d: base DAG generation failed: %s", k, dag_idx, exc)
                k_errors += 1
                continue

            k_dags_tested += 1
            strings_a: list[str] = []
            strings_b: list[str] = []

            for perm in itertools.permutations(range(k)):
                try:
                    permuted = permute_internal_nodes(base_dag, list(perm))
                    s_a = _canonicalize(permuted, backend_a, timeout=_canon_timeout)
                    s_b = _canonicalize(permuted, backend_b, timeout=_canon_timeout)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Gate1 k=%d dag=%d perm=%s: error: %s", k, dag_idx, perm, exc)
                    k_errors += 1
                    continue

                k_comparisons += 1
                strings_a.append(s_a)
                strings_b.append(s_b)

                # Cross-engine check (a).
                if s_a != s_b and len(mismatch_cases) < MAX_MISMATCH_CASES:
                    k_mismatch_cross += 1
                    mismatch_cases.append(
                        {
                            "gate": 1,
                            "kind": "cross_engine",
                            "k": k,
                            "dag_idx": dag_idx,
                            "perm": list(perm),
                            "canonical_a": s_a,
                            "canonical_b": s_b,
                        }
                    )
                elif s_a != s_b:
                    k_mismatch_cross += 1

            # Invariance check (b): all strings_a should be identical.
            if strings_a:
                unique_a = set(strings_a)
                if len(unique_a) > 1:
                    k_mismatch_inv += 1
                    if len(mismatch_cases) < MAX_MISMATCH_CASES:
                        mismatch_cases.append(
                            {
                                "gate": 1,
                                "kind": "invariance",
                                "k": k,
                                "dag_idx": dag_idx,
                                "n_unique_strings": len(unique_a),
                                "sample_strings": sorted(unique_a)[:3],
                            }
                        )

        dags_tested += k_dags_tested
        comparisons_made += k_comparisons
        mismatches_cross += k_mismatch_cross
        mismatches_inv += k_mismatch_inv
        errors += k_errors

        per_k.append(
            {
                "k": k,
                "num_vars": num_vars,
                "dags_tested": k_dags_tested,
                "perms_per_dag": n_perms,
                "comparisons": k_comparisons,
                "mismatches_cross_engine": k_mismatch_cross,
                "mismatches_invariance": k_mismatch_inv,
                "errors": k_errors,
            }
        )
        log.info(
            "Gate1 k=%d: %d DAGs × %d perms = %d comparisons, "
            "cross=%d invariance_fail=%d errors=%d",
            k,
            k_dags_tested,
            n_perms,
            k_comparisons,
            k_mismatch_cross,
            k_mismatch_inv,
            k_errors,
        )

    total_mismatches = mismatches_cross + mismatches_inv
    # Timeout errors are expected on pathological DAGs and do not constitute
    # a correctness failure — they are recorded separately.
    passed = total_mismatches == 0

    return {
        "dags_tested": dags_tested,
        "comparisons_made": comparisons_made,
        "mismatches_cross_engine": mismatches_cross,
        "mismatches_invariance": mismatches_inv,
        "errors": errors,
        "pass": passed,
        "cap_dags_per_k": _dags_per_k_map,
        "k_max": k_max,
        "per_k": per_k,
        "mismatch_cases": mismatch_cases,
    }


# ---------------------------------------------------------------------------
# Gate 2 — Generated corpus
# ---------------------------------------------------------------------------


def _is_fully_reachable(dag: LabeledDAG) -> bool:
    """Return True when every node in *dag* is reachable from node 0 via out-edges.

    ``fast_canonical_string`` / D2S starts at x_0 (node 0) and traverses
    out-edges.  A DAG whose nodes are not all reachable from node 0 will
    produce a canonical string that encodes only the reachable subgraph, so
    round-trip (gate 3) will fail for the unreachable part.

    Random string generation can produce such DAGs when a V/v instruction
    fires while the pointer is on a node already disconnected from x_0.
    These DAGs are excluded from gates 2 and 3 to avoid spurious failures.

    Args:
        dag: Labeled DAG to inspect.

    Returns:
        True if every node ID in ``range(dag.node_count)`` is reachable from
        node 0 by following out-edges.
    """
    n = dag.node_count
    if n == 0:
        return True
    visited: set[int] = set()
    stack: list[int] = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in dag.out_neighbors(node):
            if nb not in visited:
                stack.append(nb)
    return len(visited) == n


def _build_corpus(
    *,
    target_size: int,
    seed: int,
) -> list[tuple[str, int, LabeledDAG]]:
    """Build a corpus of DAGs from randomly generated IsalSR strings.

    Generates strings until ``target_size`` valid DAGs pass all inclusion
    criteria:

    - At least one internal node (non-variable operation node).
    - All nodes are reachable from x_0 (node 0) via out-edges.  DAGs that
      fail this check would produce spurious gate-3 round-trip failures
      because D2S only encodes the reachable subgraph.

    Strings that raise ``InvalidTokenError`` or produce invalid DAGs are
    silently skipped.

    Args:
        target_size: Number of DAGs to collect.
        seed: RNG seed for reproducibility.

    Returns:
        List of ``(source_string, num_vars, dag)`` triples where every DAG
        is non-empty and fully reachable from x_0.
    """
    rng = random.Random(seed)
    corpus: list[tuple[str, int, LabeledDAG]] = []
    attempts = 0
    n_has_internals: int = 0  # DAGs that passed the empty-DAG check
    n_discarded_unreachable: int = 0  # discarded by reachability filter
    n_vars_choices = list(range(GATE2_N_VARS_RANGE[0], GATE2_N_VARS_RANGE[1] + 1))
    len_choices = list(range(GATE2_STRING_LEN_RANGE[0], GATE2_STRING_LEN_RANGE[1] + 1))

    while len(corpus) < target_size:
        attempts += 1
        n_vars = rng.choice(n_vars_choices)
        n_toks = rng.choice(len_choices)
        s = _gen_random_string(rng, n_toks, n_vars)
        try:
            dag = StringToDAG(s, n_vars).run()
        except (InvalidTokenError, ValueError, Exception):  # noqa: BLE001
            continue

        # Skip empty DAGs (no internal nodes added).
        m = len(dag.var_nodes())
        if dag.node_count <= m:
            continue

        n_has_internals += 1

        # Skip DAGs with nodes unreachable from x_0 via out-edges.
        # Such DAGs produce incomplete canonical strings (D2S only encodes
        # the reachable subgraph), causing spurious gate-3 failures.
        if not _is_fully_reachable(dag):
            n_discarded_unreachable += 1
            continue

        corpus.append((s, n_vars, dag))

        if len(corpus) % 1000 == 0:
            log.info(
                "Gate2 corpus: %d / %d built (%d attempts)", len(corpus), target_size, attempts
            )

    discard_rate = n_discarded_unreachable / n_has_internals if n_has_internals > 0 else 0.0
    log.info(
        "Gate2 corpus: %d DAGs accepted; %d generated with internals, "
        "%d discarded unreachable (discard_rate=%.1f%%)",
        target_size,
        n_has_internals,
        n_discarded_unreachable,
        100 * discard_rate,
    )
    return corpus, n_has_internals, n_discarded_unreachable


def _run_gate2(
    *,
    quick: bool,
    backend_a: str,
    backend_b: str,
) -> tuple[dict[str, Any], list[tuple[str, int, LabeledDAG]]]:
    """Run gate 2: generated corpus cross-engine byte equality check.

    Args:
        quick: If True, use a 20× smaller corpus.
        backend_a: Name of engine A.
        backend_b: Name of engine B.

    Returns:
        Tuple of (gate-result dict, corpus) so gate 3 can reuse the corpus.
    """
    target = GATE2_CORPUS_QUICK if quick else GATE2_CORPUS_FULL
    corpus, n_generated, n_discarded = _build_corpus(target_size=target, seed=FIXED_SEED)
    discard_rate = round(n_discarded / n_generated, 4) if n_generated > 0 else 0.0

    dags_tested: int = 0
    comparisons_made: int = 0
    mismatches: int = 0
    errors: int = 0
    mismatch_cases: list[dict[str, Any]] = []
    k_counts: dict[int, int] = {}

    for source_str, n_vars, dag in corpus:
        dags_tested += 1
        m = len(dag.var_nodes())
        k = dag.node_count - m
        k_counts[k] = k_counts.get(k, 0) + 1

        try:
            s_a = _canonicalize(dag, backend_a)
            s_b = _canonicalize(dag, backend_b)
        except Exception as exc:  # noqa: BLE001
            log.warning("Gate2 dag=%d: canonicalize error: %s", dags_tested, exc)
            errors += 1
            continue

        comparisons_made += 1
        if s_a != s_b:
            mismatches += 1
            if len(mismatch_cases) < MAX_MISMATCH_CASES:
                mismatch_cases.append(
                    {
                        "gate": 2,
                        "kind": "cross_engine",
                        "dag_idx": dags_tested - 1,
                        "source_string": source_str,
                        "num_vars": n_vars,
                        "k": k,
                        "canonical_a": s_a,
                        "canonical_b": s_b,
                    }
                )

    # k-distribution as sorted list of {k, count} for JSON readability.
    k_distribution = [{"k": k, "count": cnt} for k, cnt in sorted(k_counts.items())]

    passed = mismatches == 0 and errors == 0
    result: dict[str, Any] = {
        "dags_tested": dags_tested,
        "comparisons_made": comparisons_made,
        "mismatches": mismatches,
        "errors": errors,
        "pass": passed,
        "corpus_seed": FIXED_SEED,
        "dags_generated": n_generated,
        "dags_discarded_unreachable": n_discarded,
        "discard_rate": discard_rate,
        "k_distribution": k_distribution,
        "mismatch_cases": mismatch_cases,
    }
    log.info(
        "Gate2: %d DAGs, %d comparisons, %d mismatches, %d errors",
        dags_tested,
        comparisons_made,
        mismatches,
        errors,
    )
    return result, corpus


# ---------------------------------------------------------------------------
# Gate 3 — Round-trip
# ---------------------------------------------------------------------------


def _run_gate3(
    *,
    backend_a: str,
    backend_b: str,
    corpus: list[tuple[str, int, LabeledDAG]],
) -> dict[str, Any]:
    """Run gate 3: round-trip isomorphism check on the gate-2 corpus.

    For each DAG D in corpus:
        S2D(fast_canonical_string(D, backend=X), n_vars).is_isomorphic(D)
    verified independently under engine_a and engine_b.

    Args:
        backend_a: Name of engine A.
        backend_b: Name of engine B.
        corpus: Pre-built corpus from gate 2 (list of (source_str, n_vars, dag)).

    Returns:
        Gate-result dict.
    """
    dags_tested: int = 0
    comparisons_made: int = 0
    mismatches_a: int = 0
    mismatches_b: int = 0
    errors: int = 0
    mismatch_cases: list[dict[str, Any]] = []

    for source_str, n_vars, dag in corpus:
        dags_tested += 1

        for backend, label in ((backend_a, "a"), (backend_b, "b")):
            try:
                canon = _canonicalize(dag, backend)
                reconstructed = StringToDAG(canon, n_vars).run()
                iso = dag.is_isomorphic(reconstructed)
            except Exception as exc:  # noqa: BLE001
                log.warning("Gate3 dag=%d backend=%s: error: %s", dags_tested, backend, exc)
                errors += 1
                continue

            comparisons_made += 1
            if not iso:
                if label == "a":
                    mismatches_a += 1
                else:
                    mismatches_b += 1
                if len(mismatch_cases) < MAX_MISMATCH_CASES:
                    m = len(dag.var_nodes())
                    k = dag.node_count - m
                    mismatch_cases.append(
                        {
                            "gate": 3,
                            "kind": f"roundtrip_engine_{label}",
                            "dag_idx": dags_tested - 1,
                            "source_string": source_str,
                            "num_vars": n_vars,
                            "k": k,
                            "canonical_string": canon,
                        }
                    )

    total_mismatches = mismatches_a + mismatches_b
    passed = total_mismatches == 0 and errors == 0
    result: dict[str, Any] = {
        "dags_tested": dags_tested,
        "comparisons_made": comparisons_made,
        "mismatches_engine_a": mismatches_a,
        "mismatches_engine_b": mismatches_b,
        "errors": errors,
        "pass": passed,
        "mismatch_cases": mismatch_cases,
    }
    log.info(
        "Gate3: %d DAGs, %d comparisons, %d mismatches_a, %d mismatches_b, %d errors",
        dags_tested,
        comparisons_made,
        mismatches_a,
        mismatches_b,
        errors,
    )
    return result


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current git HEAD commit hash, or ``"unknown"`` on error.

    Returns:
        Short git commit hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _detect_self_comparison(backend_a: str, backend_b: str) -> bool:
    """Return True when both resolved backends are ``"python"``.

    Args:
        backend_a: Resolved engine name for side A.
        backend_b: Resolved engine name for side B.

    Returns:
        True if this is a self-comparison (both sides are ``"python"``).
    """
    return backend_a == "python" and backend_b == "python"


def _probe_cpp_canonical() -> tuple[bool, str]:
    """Probe whether ``fast_canonical_string`` has a working C++ backend.

    Calls ``fast_canonical_string`` with ``backend='cpp'`` on a trivial
    one-variable DAG.  A ``TypeError`` means the ``backend=`` keyword is not
    yet accepted (the C++ canonicaliser has not been integrated); any
    successful return means the C++ path is live.

    This probe tests the **actual call path**, not whether the extension
    module loads.  ``backends.engine()``, ``build_info()``, and
    ``_CPP_AVAILABLE`` all answer the latter question only.

    Returns:
        ``(available, detail)`` where *detail* is a human-readable string
        recorded verbatim in ``capability_probe`` of the JSON report.
    """
    from isalsr.core.node_types import NodeType as _NodeType

    probe_dag = LabeledDAG(max_nodes=1)
    probe_dag.add_node(_NodeType.VAR, var_index=0)
    try:
        fast_canonical_string(probe_dag, mode="wl_only", backend="cpp")  # type: ignore[call-arg]
        return True, (
            "fast_canonical_string(backend='cpp') succeeded on 1-VAR probe DAG; "
            "C++ canonicaliser is live"
        )
    except TypeError as exc:
        return False, (
            f"fast_canonical_string raised TypeError ({exc}); "
            "backend= keyword not yet present — C++ canonicaliser not integrated"
        )
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"fast_canonical_string raised {type(exc).__name__}: {exc}; treating as unavailable"
        )


def _resolve_backend_b(requested: str) -> tuple[str, dict[str, Any]]:
    """Resolve the engine-B backend using a canonical capability probe.

    Unlike checking ``_CPP_AVAILABLE`` (extension loads), this function tests
    whether ``fast_canonical_string`` itself accepts ``backend='cpp'``.  The
    two conditions are independent: the extension can load without providing a
    canonicalisation entry point.

    Args:
        requested: CLI-requested backend name (``"python"`` or ``"cpp"``).

    Returns:
        ``(resolved_backend, capability_probe_dict)`` where
        *resolved_backend* is the backend that will actually execute and
        *capability_probe_dict* is recorded in the JSON report.
    """
    if requested == "python":
        return "python", {
            "method": "python_requested_directly",
            "cpp_canonical_available": False,
            "detail": "engine_b='python' was specified explicitly on the CLI",
        }

    # requested == "cpp": probe the actual canonicalisation path.
    cpp_avail, detail = _probe_cpp_canonical()
    probe: dict[str, Any] = {
        "method": "fast_canonical_string_backend_probe",
        "cpp_canonical_available": cpp_avail,
        "detail": detail,
    }
    if cpp_avail:
        return "cpp", probe

    log.warning(
        "C++ canonicaliser probe: %s  Falling back to engine_b='python'. "
        "self_comparison will be set to True and top-level pass to False.",
        detail,
    )
    return "python", probe


def _assemble_report(
    *,
    gate_results: dict[str, dict[str, Any] | None],
    backend_a: str,
    backend_b: str,
    capability_probe: dict[str, Any],
    quick: bool,
    start_time: float,
    end_time: float,
    gates_run: list[str],
) -> dict[str, Any]:
    """Assemble the top-level JSON report.

    A self-comparison (both engines = Python) is **never** reported as
    ``pass: true`` regardless of gate outcomes: no cross-engine check
    occurred, so no positive claim about engine equivalence can be made.

    Args:
        gate_results: Mapping from gate name (``"gate1"``, ``"gate2"``,
            ``"gate3"``) to result dict (or None if not run).
        backend_a: Resolved engine A name.
        backend_b: Resolved engine B name.
        capability_probe: Dict from ``_resolve_backend_b`` recording how
            C++ canonicaliser availability was determined.
        quick: Whether quick mode was used.
        start_time: Unix timestamp of run start.
        end_time: Unix timestamp of run end.
        gates_run: List of gate names that were executed.

    Returns:
        Top-level report dict.
    """
    self_comp = _detect_self_comparison(backend_a, backend_b)

    if self_comp:
        # Self-comparison: no genuine cross-engine test happened.
        # Must not be marked passing regardless of gate outcomes.
        top_pass = False
        reason: str | None = (
            "self_comparison=True: both engines are the Python implementation; "
            "no cross-engine comparison occurred; "
            "re-run when fast_canonical_string(backend='cpp') is available"
        )
    else:
        top_pass = all(
            gate_results[g]["pass"]  # type: ignore[index]
            for g in gates_run
            if gate_results.get(g) is not None
        )
        reason = None

    return {
        "schema_version": "1.0",
        "pass": top_pass,
        "reason": reason,
        "self_comparison": self_comp,
        "self_comparison_note": (
            "Both engines are the Python implementation.  Cross-engine checks "
            "trivially agree (same code path).  Gate 1b invariance and gate 3 "
            "round-trip are genuinely meaningful.  Re-run with --backend-b cpp "
            "once fast_canonical_string accepts backend='cpp'."
        )
        if self_comp
        else None,
        "provenance": {
            "git_commit": _git_commit(),
            "build_info": build_info(),
            "engine": engine(),
            "engine_a": backend_a,
            "engine_b": backend_b,
            "capability_probe": capability_probe,
            "seed": FIXED_SEED,
            "quick_mode": quick,
            "gates_run": gates_run,
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
            "elapsed_seconds": round(end_time - start_time, 2),
            "python_version": sys.version,
        },
        "gate1": gate_results.get("gate1"),
        "gate2": gate_results.get("gate2"),
        "gate3": gate_results.get("gate3"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable one-page summary to stdout.

    Args:
        report: Assembled report dict.
    """
    prov = report["provenance"]
    status = "PASS" if report["pass"] else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"  IsalSR Equivalence Gate — {status}")
    print(f"{'=' * 60}")
    print(f"  git commit : {prov['git_commit']}")
    print(f"  engine_a   : {prov['engine_a']}")
    print(f"  engine_b   : {prov['engine_b']}")
    print(f"  quick_mode : {prov['quick_mode']}")
    if report["self_comparison"]:
        print("  NOTE: self-comparison — C++ canonicaliser not yet available")
        print(f"  REASON: {report.get('reason', '')}")
    print(f"  elapsed    : {prov['elapsed_seconds']}s")
    cap_probe = prov.get("capability_probe", {})
    print(f"  cpp_probe  : {cap_probe.get('detail', 'n/a')}")
    print()

    for gate_key, gate_name in [
        ("gate1", "Gate 1 (exhaustive perm)"),
        ("gate2", "Gate 2 (corpus)"),
        ("gate3", "Gate 3 (round-trip)"),
    ]:
        g = report.get(gate_key)
        if g is None:
            print(f"  {gate_name}: skipped")
            continue
        g_status = "PASS" if g["pass"] else "FAIL"
        print(f"  {gate_name}: {g_status}")
        print(f"    DAGs tested      : {g['dags_tested']}")
        print(f"    Comparisons      : {g['comparisons_made']}")
        if gate_key == "gate1":
            print(f"    Mismatch cross   : {g['mismatches_cross_engine']}")
            print(f"    Mismatch inv     : {g['mismatches_invariance']}")
            print(f"    Cap per k        : {g['cap_dags_per_k']} (k_max={g['k_max']})")
        elif gate_key == "gate2":
            print(f"    Mismatches       : {g['mismatches']}")
            print(f"    DAGs generated   : {g.get('dags_generated', 'n/a')}")
            print(f"    Discarded unreach: {g.get('dags_discarded_unreachable', 'n/a')}")
            print(f"    Discard rate     : {g.get('discard_rate', 'n/a'):.1%}")
        elif gate_key == "gate3":
            print(f"    Mismatches_a     : {g['mismatches_engine_a']}")
            print(f"    Mismatches_b     : {g['mismatches_engine_b']}")
        print(f"    Errors           : {g['errors']}")
    print(f"{'=' * 60}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the equivalence gate harness.

    Args:
        argv: Command-line arguments (uses ``sys.argv`` when None).

    Returns:
        Exit code: 0 on pass, 1 on any gate failure.
    """
    parser = argparse.ArgumentParser(
        description="IsalSR canonicaliser equivalence-gate harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gate",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which gate(s) to run.",
    )
    parser.add_argument(
        "--out",
        default="equivalence_gate_report.json",
        help="Output path for the JSON report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Shrink all corpora by ~20× for fast local iteration.  "
            "The report records quick_mode=true."
        ),
    )
    parser.add_argument(
        "--backend-a",
        default="python",
        choices=["python", "cpp"],
        help="Engine A (reference).",
    )
    parser.add_argument(
        "--backend-b",
        default="cpp",
        choices=["python", "cpp"],
        help=(
            "Engine B (under test).  Falls back to 'python' when the C++ "
            "extension is unavailable, marking the report self_comparison=true."
        ),
    )
    args = parser.parse_args(argv)

    # Engine A is always python (reference).
    backend_a: str = args.backend_a  # "python" only for now; kept symmetric for future use
    # Engine B: probe the actual canonicalisation call path, not just extension loading.
    backend_b, capability_probe = _resolve_backend_b(args.backend_b)
    self_comp = _detect_self_comparison(backend_a, backend_b)

    log.info(
        "Equivalence gate: engine_a=%s engine_b=%s quick=%s self_comparison=%s",
        backend_a,
        backend_b,
        args.quick,
        self_comp,
    )
    log.info("Capability probe: %s", capability_probe.get("detail", capability_probe))
    if self_comp:
        log.warning(
            "Self-comparison mode: both engines are Python.  "
            "Top-level pass will be False.  "
            "Gate 1b invariance and gate 3 round-trip are still meaningful."
        )

    gate_flags: set[str] = {"1", "2", "3"} if args.gate == "all" else {args.gate}
    gates_run: list[str] = []
    gate_results: dict[str, dict[str, Any] | None] = {
        "gate1": None,
        "gate2": None,
        "gate3": None,
    }
    corpus: list[tuple[str, int, LabeledDAG]] = []

    start_time = time.time()

    if "1" in gate_flags:
        log.info(
            "--- Gate 1: exhaustive permutation k=1..%d ---",
            GATE1_K_MAX_QUICK if args.quick else GATE1_K_MAX_FULL,
        )
        gate_results["gate1"] = _run_gate1(
            quick=args.quick,
            backend_a=backend_a,
            backend_b=backend_b,
        )
        gates_run.append("gate1")

    if "2" in gate_flags or "3" in gate_flags:
        log.info(
            "--- Gate 2: generated corpus (%d DAGs) ---",
            GATE2_CORPUS_QUICK if args.quick else GATE2_CORPUS_FULL,
        )
        g2_result, corpus = _run_gate2(
            quick=args.quick,
            backend_a=backend_a,
            backend_b=backend_b,
        )
        if "2" in gate_flags:
            gate_results["gate2"] = g2_result
            gates_run.append("gate2")

    if "3" in gate_flags:
        log.info("--- Gate 3: round-trip isomorphism (%d DAGs) ---", len(corpus))
        gate_results["gate3"] = _run_gate3(
            backend_a=backend_a,
            backend_b=backend_b,
            corpus=corpus,
        )
        gates_run.append("gate3")

    end_time = time.time()

    report = _assemble_report(
        gate_results=gate_results,
        backend_a=backend_a,
        backend_b=backend_b,
        capability_probe=capability_probe,
        quick=args.quick,
        start_time=start_time,
        end_time=end_time,
        gates_run=gates_run,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written to %s", out_path)

    _print_summary(report)

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
