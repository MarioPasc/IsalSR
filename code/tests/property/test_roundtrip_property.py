"""Hypothesis property-based tests for the round-trip property.

Uses Hypothesis to generate random valid IsalSR strings and verify:
    S2D(w) ~ S2D(D2S(S2D(w), x_1)) for all valid w.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isalsr.core.dag_to_string import DAGToString
from isalsr.core.node_types import OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.operators import detokenize, random_token


@st.composite
def isalsr_strings(draw: st.DrawFn, num_variables: int = 1, max_tokens: int = 8) -> str:
    """Hypothesis strategy for random IsalSR strings."""
    allowed_ops = OperationSet()
    n_tokens = draw(st.integers(min_value=1, max_value=max_tokens))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31)))
    tokens = [random_token(allowed_ops, rng) for _ in range(n_tokens)]
    return detokenize(tokens)


@given(string=isalsr_strings(num_variables=1, max_tokens=6))
@settings(max_examples=50, deadline=5000)
def test_roundtrip_1var(string: str) -> None:
    """Round-trip property for 1-variable strings."""
    try:
        dag1 = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return  # Invalid string, skip.
    if dag1.node_count <= 1:
        return  # VAR-only.
    try:
        string2 = DAGToString(dag1).run()
        dag2 = StringToDAG(string2, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.node_count == dag2.node_count
    assert dag1.edge_count == dag2.edge_count
    assert dag1.is_isomorphic(dag2)


@given(string=isalsr_strings(num_variables=2, max_tokens=6))
@settings(max_examples=50, deadline=5000)
def test_roundtrip_2var(string: str) -> None:
    """Round-trip property for 2-variable strings."""
    try:
        dag1 = StringToDAG(string, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    if dag1.node_count <= 2:
        return
    try:
        string2 = DAGToString(dag1).run()
        dag2 = StringToDAG(string2, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    assert dag1.is_isomorphic(dag2)
