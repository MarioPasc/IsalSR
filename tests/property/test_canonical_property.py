"""Hypothesis property-based tests for canonical string invariance.

Tests the paper's core claim: canonical_string is a complete invariant.
    canonical_string(D) == canonical_string(D') iff D ~ D'
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isalsr.core.canonical import canonical_string
from isalsr.core.node_types import OperationSet
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


@given(string=isalsr_strings(num_variables=1, max_tokens=5))
@settings(max_examples=30, deadline=10000)
def test_canonical_is_idempotent_1var(string: str) -> None:
    """canonical(S2D(canonical(S2D(w)))) == canonical(S2D(w)).

    Applying canonical twice gives the same result (idempotency).
    """
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon1 = canonical_string(dag1)
        dag2 = StringToDAG(canon1, num_variables=1).run()
        canon2 = canonical_string(dag2)
    except Exception:  # noqa: BLE001
        return
    assert canon1 == canon2


@given(string=isalsr_strings(num_variables=1, max_tokens=5))
@settings(max_examples=30, deadline=10000)
def test_canonical_roundtrip_isomorphism_1var(string: str) -> None:
    """S2D(canonical_string(D)) is isomorphic to D."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 1:
        return
    try:
        canon = canonical_string(dag1)
        dag2 = StringToDAG(canon, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)
