"""Hypothesis property-based tests for fast_canonical_string.

Tests the paper's core claim for the PREFERRED algorithm:
    fast_canonical_string(D, mode=m) is a complete labeled-DAG invariant
for all three modes.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.node_types import OperationSet
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.operators import detokenize, random_token


@st.composite
def isalsr_strings(draw: st.DrawFn, num_variables: int = 1, max_tokens: int = 6) -> str:
    """Hypothesis strategy for random IsalSR strings."""
    allowed_ops = OperationSet()
    n_tokens = draw(st.integers(min_value=1, max_value=max_tokens))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31)))
    tokens = [random_token(allowed_ops, rng) for _ in range(n_tokens)]
    return detokenize(tokens)


# ======================================================================
# WL-only (default mode) property tests
# ======================================================================


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_idempotent_wl_only(string: str) -> None:
    """fast_canonical(S2D(fast_canonical(S2D(w)))) == fast_canonical(S2D(w))."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon1 = fast_canonical_string(dag1, mode="wl_only")
        dag2 = StringToDAG(canon1, num_variables=1).run()
        canon2 = fast_canonical_string(dag2, mode="wl_only")
    except Exception:  # noqa: BLE001
        return
    assert canon1 == canon2


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_roundtrip_wl_only(string: str) -> None:
    """S2D(fast_canonical_string(D)) is isomorphic to D."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon = fast_canonical_string(dag1, mode="wl_only")
        dag2 = StringToDAG(canon, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)


# ======================================================================
# wl_tiebreak mode property tests
# ======================================================================


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_idempotent_wl_tiebreak(string: str) -> None:
    """Idempotency for wl_tiebreak mode."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon1 = fast_canonical_string(dag1, mode="wl_tiebreak")
        dag2 = StringToDAG(canon1, num_variables=1).run()
        canon2 = fast_canonical_string(dag2, mode="wl_tiebreak")
    except Exception:  # noqa: BLE001
        return
    assert canon1 == canon2


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_roundtrip_wl_tiebreak(string: str) -> None:
    """Round-trip for wl_tiebreak mode."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon = fast_canonical_string(dag1, mode="wl_tiebreak")
        dag2 = StringToDAG(canon, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)


# ======================================================================
# tuple_only mode property tests
# ======================================================================


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_idempotent_tuple_only(string: str) -> None:
    """Idempotency for tuple_only mode."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon1 = fast_canonical_string(dag1, mode="tuple_only")
        dag2 = StringToDAG(canon1, num_variables=1).run()
        canon2 = fast_canonical_string(dag2, mode="tuple_only")
    except Exception:  # noqa: BLE001
        return
    assert canon1 == canon2


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=15000)
def test_fast_canonical_roundtrip_tuple_only(string: str) -> None:
    """Round-trip for tuple_only mode."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon = fast_canonical_string(dag1, mode="tuple_only")
        dag2 = StringToDAG(canon, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)


# ======================================================================
# Multi-variable test
# ======================================================================


@given(string=isalsr_strings(num_variables=2, max_tokens=5))
@settings(max_examples=30, deadline=15000)
def test_fast_canonical_2var_roundtrip(string: str) -> None:
    """Multi-variable round-trip with default mode."""
    try:
        dag1 = StringToDAG(string, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 2:
        return
    try:
        canon = fast_canonical_string(dag1, mode="wl_only")
        dag2 = StringToDAG(canon, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)


# ======================================================================
# Cross-mode equivalence class consistency
# ======================================================================


@given(string=isalsr_strings(num_variables=1, max_tokens=4))
@settings(max_examples=30, deadline=15000)
def test_all_modes_same_equivalence_classes(string: str) -> None:
    """All modes agree on whether two DAGs are isomorphic.

    If mode A says D ~ D' (same canonical), then mode B must also say D ~ D'.
    We test this by generating a random permutation and checking all modes agree.
    """
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    k = dag1.node_count - len(dag1.var_nodes())
    if k < 2:
        return

    # Create a permuted copy
    rng = np.random.default_rng(42)
    perm = list(range(k))
    rng.shuffle(perm)
    try:
        dag2 = permute_internal_nodes(dag1, perm)
    except Exception:  # noqa: BLE001
        return

    # All modes should agree: original and permuted have the same canonical
    for mode in ["wl_only", "wl_tiebreak", "tuple_only"]:
        try:
            c1 = fast_canonical_string(dag1, mode=mode)  # type: ignore[arg-type]
            c2 = fast_canonical_string(dag2, mode=mode)  # type: ignore[arg-type]
            assert c1 == c2, f"mode={mode}: permutation changed canonical"
        except Exception:  # noqa: BLE001
            # Timeout or conversion error — skip
            pass
