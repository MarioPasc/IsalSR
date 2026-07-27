"""Differential tests: C++ fast_canonical_string vs Python (wl_only mode).

Acceptance checks implemented here:
  2. Byte-exact differential test on ≥3,000 DAGs (1–5 vars, varied length).
     fast_canonical_string(d, mode="wl_only", backend="cpp")
     == fast_canonical_string(d, mode="wl_only", backend="python")
     Zero mismatches required.
  3. WL hash vector conformance: C++ wl_node_hash reproduces every entry in
     tests/data/wl_hash_vectors.json (the shared oracle).

Tested invariants:
  5  — pair sort by |a|+|b| (via conformant canonical output).
  8  — binary op operand order via ordered_inputs (via conformant output).
  9  — normalize_const_creation applied inside C++ entry point.
  10 — label-char is the first sort key (via conformant output).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from isalsr.core import _native as _cpp_ext  # type: ignore[attr-defined]
from isalsr.core.canonical import CanonicalTimeoutError, fast_canonical_string
from isalsr.core.string_to_dag import StringToDAG

_nt: Any = _cpp_ext.testing

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_WL_VECTORS_PATH = _DATA_DIR / "wl_hash_vectors.json"

# ---------------------------------------------------------------------------
# String generator (same as test_native_s2d.py)
# ---------------------------------------------------------------------------

_LABEL_CHARS: list[str] = list("+*-/scelr^akgi")
_SINGLE_CHARS: list[str] = list("NnPpCcW")


def _gen_valid_string(rng: random.Random, n_tokens: int) -> str:
    """Generate a valid IsalSR instruction string with *n_tokens* logical tokens."""
    parts: list[str] = []
    for _ in range(n_tokens):
        if rng.random() < 0.5:
            parts.append(rng.choice(_SINGLE_CHARS))
        else:
            parts.append(rng.choice("Vv"))
            parts.append(rng.choice(_LABEL_CHARS))
    return "".join(parts)


# ---------------------------------------------------------------------------
# WL hash vector conformance (Acceptance check 3)
# ---------------------------------------------------------------------------


def _load_wl_vectors() -> list[dict[str, Any]]:
    with open(_WL_VECTORS_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "vec",
    _load_wl_vectors(),
    ids=[v["case"] for v in _load_wl_vectors()],
)
def test_wl_hash_vector_conformance(vec: dict[str, Any]) -> None:
    """C++ wl_node_hash must match the shared oracle for every test vector."""
    label_value: str = vec["label_value"]
    children: list[int] = vec["children_hashes"]
    expected: int = vec["expected_uint64"]

    # children_hashes are already sorted in the oracle file
    result: int = _cpp_ext.wl_node_hash(label_value, children)
    assert result == expected, (
        f"WL hash mismatch for case={vec['case']!r}: C++ returned {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Differential test — 3,000 DAGs (Acceptance check 2)
# ---------------------------------------------------------------------------


def test_differential_3000_dags() -> None:
    """Byte-exact differential: backend='cpp' == backend='python' for 3,000 DAGs.

    Generates 3,000 (seed, n_vars, string) triples, builds a Python LabeledDAG
    from each, and compares fast_canonical_string with both backends.
    Zero mismatches are required.

    Note: a small number of randomly generated strings produce degenerate DAGs
    for which the Python oracle also raises RuntimeError ("no valid operation
    found") — a pre-existing issue in the Python algorithm that is outside the
    scope of this implementation.  For these cases the C++ backend raises the
    same RuntimeError, confirming consistent behaviour.  They are counted in
    ``n_both_fail`` and are not treated as mismatches.  All other 3,000 seeds
    must match exactly.
    """
    mismatches: list[tuple[int, int, str, str, str]] = []
    n_both_fail = 0

    for seed in range(3000):
        rng = random.Random(seed)
        n_vars = rng.randint(1, 5)
        n_tokens = rng.randint(0, 20)
        s = _gen_valid_string(rng, n_tokens)

        dag = StringToDAG(s, n_vars).run()

        # Attempt Python backend
        try:
            py_result = fast_canonical_string(dag, mode="wl_only", backend="python")
            py_ok = True
        except RuntimeError:
            py_ok = False
            py_result = ""

        # Attempt C++ backend — must agree with Python on success/failure
        try:
            cpp_result = fast_canonical_string(dag, mode="wl_only", backend="cpp")
            cpp_ok = True
        except RuntimeError:
            cpp_ok = False
            cpp_result = ""

        if not py_ok and not cpp_ok:
            # Both fail with RuntimeError: consistent pre-existing limitation
            n_both_fail += 1
            continue

        # One failed and the other did not, OR both succeeded but differ
        if py_ok != cpp_ok or cpp_result != py_result:
            mismatches.append((seed, n_vars, s, py_result, cpp_result))

    n_total = 3000
    n_miss = len(mismatches)
    assert n_miss == 0, (
        f"{n_miss}/{n_total} mismatches (plus {n_both_fail} consistent pre-existing failures).\n"
        + "\n".join(
            f"  seed={s}, n_vars={nv}, str={st!r}\n    py ={py!r}\n    cpp={cp!r}"
            for s, nv, st, py, cp in mismatches[:5]
        )
    )
    # Report consistent failure count informally (not an assertion)
    if n_both_fail > 0:
        import warnings

        warnings.warn(
            f"test_differential_3000_dags: {n_both_fail} DAGs where both backends "
            "raise RuntimeError (pre-existing Python oracle limitation).",
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_vars", [1, 2, 3, 5])
def test_empty_string_backend_parity(n_vars: int) -> None:
    """Empty string → canonical is '' on both backends."""
    dag = StringToDAG("", n_vars).run()
    assert fast_canonical_string(dag, mode="wl_only", backend="python") == ""
    assert fast_canonical_string(dag, mode="wl_only", backend="cpp") == ""


def test_single_node_expressions() -> None:
    """Single internal node (V+, V*, Vs, etc.) — backends agree."""
    mismatches = []
    for lchar in _LABEL_CHARS:
        for n_vars in (1, 2, 3):
            s = f"V{lchar}"
            try:
                dag = StringToDAG(s, n_vars).run()
            except Exception:
                continue
            py_r = fast_canonical_string(dag, mode="wl_only", backend="python")
            cpp_r = fast_canonical_string(dag, mode="wl_only", backend="cpp")
            if py_r != cpp_r:
                mismatches.append((s, n_vars, py_r, cpp_r))
    assert mismatches == [], mismatches


def test_timeout_cpp_raises_canonical_timeout_error() -> None:
    """Timeout triggers CanonicalTimeoutError from the C++ backend."""
    # Use a string with multiple nodes to ensure search takes time
    rng = random.Random(7)
    s = _gen_valid_string(rng, 30)
    dag = StringToDAG(s, 3).run()
    with pytest.raises(CanonicalTimeoutError):
        fast_canonical_string(dag, mode="wl_only", backend="cpp", timeout=1e-9)


def test_backend_cpp_default_when_native_available() -> None:
    """With _native present, backend=None should use C++ (DEFAULT_BACKEND='cpp')."""
    from isalsr.core.backends import DEFAULT_BACKEND

    assert DEFAULT_BACKEND == "cpp", (
        "Expected DEFAULT_BACKEND='cpp' when _native is loaded, got " + repr(DEFAULT_BACKEND)
    )
    dag = StringToDAG("V+V*", 2).run()
    # backend=None should match backend='cpp'
    r_none = fast_canonical_string(dag, mode="wl_only", backend=None)
    r_cpp = fast_canonical_string(dag, mode="wl_only", backend="cpp")
    assert r_none == r_cpp


def test_backend_python_explicit() -> None:
    """Explicitly requesting backend='python' always uses the Python path."""
    dag = StringToDAG("V+V*", 2).run()
    r_py = fast_canonical_string(dag, mode="wl_only", backend="python")
    r_cpp = fast_canonical_string(dag, mode="wl_only", backend="cpp")
    assert r_py == r_cpp  # results are the same, not just no error


def test_invalid_backend_raises() -> None:
    """Unknown backend name raises ValueError."""
    dag = StringToDAG("V+", 1).run()
    with pytest.raises(ValueError, match="Unknown backend"):
        fast_canonical_string(dag, mode="wl_only", backend="invalid")


@pytest.mark.parametrize("mode", ["wl_tiebreak", "tuple_only"])
def test_non_wl_only_modes_fall_back_to_python(mode: str) -> None:
    """wl_tiebreak and tuple_only modes fall back to Python path (no C++ port)."""
    dag = StringToDAG("V+V*", 2).run()
    # Should not raise; C++ path only supports wl_only
    r = fast_canonical_string(dag, mode=mode, backend="cpp")  # type: ignore[arg-type]
    r_py = fast_canonical_string(dag, mode=mode, backend="python")  # type: ignore[arg-type]
    assert r == r_py


def test_const_nodes_normalized_in_cpp() -> None:
    """DAGs with CONST nodes: C++ applies normalize_const_creation internally."""
    # Vk creates a CONST node; its creation edge must be normalized
    dag = StringToDAG("VkV+", 1).run()
    py_r = fast_canonical_string(dag, mode="wl_only", backend="python")
    cpp_r = fast_canonical_string(dag, mode="wl_only", backend="cpp")
    assert cpp_r == py_r, f"CONST node mismatch: py={py_r!r}, cpp={cpp_r!r}"


def test_larger_dags_differential() -> None:
    """Longer strings (≥15 tokens) also match between backends."""
    mismatches = []
    for seed in range(200):
        rng = random.Random(seed + 10000)
        n_vars = rng.randint(1, 5)
        n_tokens = rng.randint(15, 40)
        s = _gen_valid_string(rng, n_tokens)
        dag = StringToDAG(s, n_vars).run()
        py_r = fast_canonical_string(dag, mode="wl_only", backend="python")
        cpp_r = fast_canonical_string(dag, mode="wl_only", backend="cpp")
        if py_r != cpp_r:
            mismatches.append((seed, n_vars, s, py_r, cpp_r))
    assert mismatches == [], f"{len(mismatches)} mismatches: {mismatches[:2]}"


def test_wl_node_hash_unsorted_children_not_equal_sorted() -> None:
    """Sanity: hash of [h1, h2] != hash of [h2, h1] when h1 != h2 (not sorted == error)."""
    h1, h2 = 12345678901234567890 % (2**64), 98765432109876543210 % (2**64)
    r_sorted = _cpp_ext.wl_node_hash("+", [min(h1, h2), max(h1, h2)])
    r_unsorted = _cpp_ext.wl_node_hash("+", [max(h1, h2), min(h1, h2)])
    # Not necessarily different (hash collisions), but we check function is callable
    # The test primarily ensures the API accepts list[int] correctly.
    assert isinstance(r_sorted, int)
    assert isinstance(r_unsorted, int)
    # For this specific input, they should differ (different byte sequences)
    assert r_sorted != r_unsorted, "Expected different hashes for different input orders"
