"""Hypothesis property-based tests for DAG acyclicity.

Uses Hypothesis to generate random IsalSR strings and verify:
    The resulting DAG from S2D is always acyclic (topological sort succeeds).

This is the fundamental safety property of the C/c cycle check.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isalsr.core.node_types import OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.operators import detokenize, random_token


@st.composite
def isalsr_strings(draw: st.DrawFn, num_variables: int = 1, max_tokens: int = 10) -> str:
    """Hypothesis strategy for random IsalSR strings."""
    allowed_ops = OperationSet()
    n_tokens = draw(st.integers(min_value=0, max_value=max_tokens))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31)))
    tokens = [random_token(allowed_ops, rng) for _ in range(n_tokens)]
    return detokenize(tokens)


@given(string=isalsr_strings(num_variables=1, max_tokens=10))
@settings(max_examples=100, deadline=5000)
def test_dag_always_acyclic_1var(string: str) -> None:
    """Every DAG produced by S2D is acyclic (topological sort succeeds)."""
    try:
        dag = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return  # Invalid string, skip.
    # If topological sort raises, the DAG has a cycle (should never happen).
    order = dag.topological_sort()
    assert len(order) == dag.node_count


@given(string=isalsr_strings(num_variables=2, max_tokens=10))
@settings(max_examples=100, deadline=5000)
def test_dag_always_acyclic_2var(string: str) -> None:
    """Every 2-variable DAG produced by S2D is acyclic."""
    try:
        dag = StringToDAG(string, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    order = dag.topological_sort()
    assert len(order) == dag.node_count
