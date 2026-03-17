"""Hypothesis property-based tests for evaluation consistency.

Verifies that DAG evaluation produces finite results for all valid strings.
(SymPy comparison deferred to integration tests due to import cost.)

Requires: numpy.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from isalsr.core.dag_evaluator import evaluate_dag
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
def test_evaluation_always_finite_1var(string: str) -> None:
    """DAG evaluation always returns a finite value (protected ops work)."""
    try:
        dag = StringToDAG(string, num_variables=1).run()
    except Exception:  # noqa: BLE001
        return
    if dag.node_count <= 1:
        return
    try:
        result = evaluate_dag(dag, {0: 1.5})
    except Exception:  # noqa: BLE001
        return  # Evaluation error (e.g., wrong arity) is acceptable.
    assert math.isfinite(result), f"Non-finite result: {result}"


@given(string=isalsr_strings(num_variables=2, max_tokens=6))
@settings(max_examples=50, deadline=5000)
def test_evaluation_always_finite_2var(string: str) -> None:
    """DAG evaluation always returns finite for 2-variable expressions."""
    try:
        dag = StringToDAG(string, num_variables=2).run()
    except Exception:  # noqa: BLE001
        return
    if dag.node_count <= 2:
        return
    try:
        result = evaluate_dag(dag, {0: 1.5, 1: -0.5})
    except Exception:  # noqa: BLE001
        return
    assert math.isfinite(result), f"Non-finite result: {result}"
