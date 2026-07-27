"""Tests for cross-session stability of the FNV-1a WL subtree hash.

Verifies that:
1. ``_wl_node_hash`` reproduces every entry in ``tests/data/wl_hash_vectors.json``.
2. ``_native.fnv1a64`` agrees with ``_wl_node_hash`` for plain byte-string inputs
   (i.e., empty ``children_hashes``), confirming the C extension uses identical
   FNV-1a parameters (offset basis 0xcbf29ce484222325, prime 0x100000001b3).
3. ``fast_canonical_string(mode="wl_only")`` returns the same result when called
   under three different PYTHONHASHSEED environments (end-to-end stability).

These vectors constitute the shared pin point that the forthcoming C++ canonical
engine will be validated against.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from isalsr.core import _native
from isalsr.core.canonical import _wl_node_hash

# ---------------------------------------------------------------------------
# Load shared test vectors
# ---------------------------------------------------------------------------

_VECTORS_PATH = Path(__file__).parent.parent / "data" / "wl_hash_vectors.json"
_VECTORS: list[dict] = json.loads(_VECTORS_PATH.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cases() -> list[tuple[str, str, tuple[int, ...], int]]:
    """Return (case_id, label_value, children_hashes, expected_uint64) tuples."""
    return [
        (v["case"], v["label_value"], tuple(v["children_hashes"]), v["expected_uint64"])
        for v in _VECTORS
    ]


# ---------------------------------------------------------------------------
# Test 1: Python _wl_node_hash reproduces every vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,label_value,children_hashes,expected", _load_cases())
def test_wl_node_hash_matches_vector(
    case_id: str,
    label_value: str,
    children_hashes: tuple[int, ...],
    expected: int,
) -> None:
    """``_wl_node_hash`` reproduces the pinned test vector exactly.

    Args:
        case_id: Human-readable case identifier (used in parametrize id).
        label_value: The ``NodeType.value`` string fed to the hash function.
        children_hashes: Sorted child hash tuple (already sorted in JSON).
        expected: Expected 64-bit unsigned integer result.
    """
    result = _wl_node_hash(label_value, children_hashes)
    assert result == expected, (
        f"Case {case_id!r}: _wl_node_hash({label_value!r}, {children_hashes}) "
        f"= {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Test 2: _native.fnv1a64 agrees with _wl_node_hash for empty-children cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,label_value,children_hashes,expected",
    [row for row in _load_cases() if not row[2]],  # only empty-children cases
)
def test_native_fnv1a64_agrees_with_python(
    case_id: str,
    label_value: str,
    children_hashes: tuple[int, ...],
    expected: int,
) -> None:
    """``_native.fnv1a64`` agrees with ``_wl_node_hash`` for plain byte inputs.

    When ``children_hashes`` is empty, ``_wl_node_hash(label, ())`` reduces to
    FNV-1a applied to ``label.encode('utf-8')`` with no further mixing — exactly
    what ``_native.fnv1a64(label.encode('utf-8'))`` computes. Any divergence
    indicates a parameter mismatch in the C extension.

    Args:
        case_id: Human-readable case identifier.
        label_value: NodeType value string.
        children_hashes: Empty tuple (enforced by parametrize filter).
        expected: Expected 64-bit hash.
    """
    assert not children_hashes, "This test only applies to empty-children cases"
    native_result = _native.fnv1a64(label_value.encode("utf-8"))
    python_result = _wl_node_hash(label_value, ())
    assert native_result == python_result == expected, (
        f"Case {case_id!r}: native={native_result}, python={python_result}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Test 3: End-to-end cross-PYTHONHASHSEED stability
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = (
    Path(__file__).parent.parent.parent
    / (".." if False else "")
    / (
        "tmp/claude-1000/-home-mpascual-research-code-IsalSR/"
        "34912311-a981-4a5f-845d-0ad3421737b4/scratchpad/hashseed_probe.py"
    )
)

# Embedded minimal probe — avoids dependency on a temp-dir path.
_MINI_PROBE = """\
import json, sys, random
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.string_to_dag import StringToDAG

MOVES = ["N", "P", "n", "p", "C", "c", "W"]
LABELS = ["+", "*", "-", "/", "s", "c", "e", "l", "r", "a", "g", "i", "k"]

rng = random.Random(20260727)
dags = []
while len(dags) < 30:
    tokens = []
    for _ in range(rng.randint(6, 22)):
        if rng.random() < 0.55:
            tokens.append(rng.choice(["V", "v"]) + rng.choice(LABELS))
        else:
            tokens.append(rng.choice(MOVES))
    expr = "".join(tokens)
    try:
        d = StringToDAG(expr, num_variables=2).run()
    except Exception:
        continue
    if d.node_count < 4:
        continue
    dags.append(d)

canon = [fast_canonical_string(d, mode="wl_only") for d in dags]
json.dump(canon, sys.stdout)
"""


def _run_probe(seed: int) -> list[str]:
    """Run the mini probe under a fixed PYTHONHASHSEED, return canonical list."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    result = subprocess.run(
        [sys.executable, "-c", _MINI_PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("seed_pair", [(0, 42), (0, 1337), (42, 1337)])
def test_canonical_stable_across_hashseeds(seed_pair: tuple[int, int]) -> None:
    """``fast_canonical_string(mode='wl_only')`` is identical across PYTHONHASHSEED values.

    Runs the canonical computation in two fresh subprocesses with different
    PYTHONHASHSEED values and asserts zero differences across 30 DAGs.

    Args:
        seed_pair: Two PYTHONHASHSEED values to compare.
    """
    seed_a, seed_b = seed_pair
    canon_a = _run_probe(seed_a)
    canon_b = _run_probe(seed_b)
    assert len(canon_a) == len(canon_b) == 30
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(canon_a, canon_b, strict=True)) if a != b]
    assert not diffs, (
        f"PYTHONHASHSEED {seed_a} vs {seed_b}: "
        f"{len(diffs)}/{len(canon_a)} canonical strings differ. "
        f"First diff at index {diffs[0][0]}: {diffs[0][1]!r} vs {diffs[0][2]!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: _wl_node_hash output is always a valid uint64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,label_value,children_hashes,expected", _load_cases())
def test_wl_node_hash_is_uint64(
    case_id: str,
    label_value: str,
    children_hashes: tuple[int, ...],
    expected: int,
) -> None:
    """``_wl_node_hash`` always returns a value in [0, 2^64).

    Args:
        case_id: Human-readable case identifier.
        label_value: NodeType value string.
        children_hashes: Sorted child hash tuple.
        expected: Expected value (unused; presence confirms the vector loaded).
    """
    result = _wl_node_hash(label_value, children_hashes)
    assert 0 <= result < (1 << 64), f"Case {case_id!r}: result {result} is out of uint64 range"
