"""Numerical evaluation of expression DAGs.

Evaluates a LabeledDAG on numerical input data using topological sort.
No external dependencies -- operates on Python lists and the math module.
The evaluation/ layer wraps this for vectorized numpy evaluation.

Protected operations follow standard symbolic regression practice:
    - Koza (1992). Genetic Programming.
    - Petersen et al. (2021). DSR. NeurIPS.

Restriction: ZERO external dependencies. Only Python stdlib + math.
"""

from __future__ import annotations

import logging
import math
from functools import reduce

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, UNARY_OPS, VARIADIC_OPS, NodeType
from isalsr.errors import EvaluationError

log = logging.getLogger(__name__)

# Maximum absolute value for clamping outputs to avoid overflow propagation.
_MAX_VALUE = 1e15


def evaluate_dag(dag: LabeledDAG, inputs: dict[int, float]) -> float:
    """Evaluate an expression DAG on scalar inputs.

    Args:
        dag: The expression DAG to evaluate.
        inputs: Mapping from **var_index** (0, 1, 2, ...) to scalar values.
            For example, ``{0: 3.14, 1: 2.72}`` sets x_1=3.14, x_2=2.72.

    Returns:
        The scalar output value of the expression.

    Raises:
        EvaluationError: If a required var_index is missing from inputs,
            or if the DAG has no operation nodes (only variables).
    """
    order = dag.topological_sort()
    values: dict[int, float] = {}

    for node in order:
        label = dag.node_label(node)

        if label == NodeType.VAR:
            var_idx_raw = dag.node_data(node).get("var_index")
            if var_idx_raw is None:
                raise EvaluationError(f"VAR node {node} has no var_index")
            var_idx = int(var_idx_raw)
            if var_idx not in inputs:
                raise EvaluationError(
                    f"Missing input for var_index={var_idx} (node {node}). "
                    f"Provided: {sorted(inputs.keys())}"
                )
            values[node] = inputs[var_idx]

        elif label == NodeType.CONST:
            values[node] = float(dag.node_data(node).get("const_value", 1.0))

        elif label in UNARY_OPS:
            in_nodes = sorted(dag.in_neighbors(node))
            if len(in_nodes) != 1:
                raise EvaluationError(
                    f"Unary op {label.name} (node {node}) expects 1 input, got {len(in_nodes)}"
                )
            values[node] = _apply_unary(label, values[in_nodes[0]])

        elif label in BINARY_OPS:
            in_nodes = sorted(dag.in_neighbors(node))
            if len(in_nodes) != 2:
                raise EvaluationError(
                    f"Binary op {label.name} (node {node}) expects 2 inputs, got {len(in_nodes)}"
                )
            values[node] = _apply_binary(label, values[in_nodes[0]], values[in_nodes[1]])

        elif label in VARIADIC_OPS:
            in_nodes = sorted(dag.in_neighbors(node))
            if len(in_nodes) < 2:
                raise EvaluationError(
                    f"Variadic op {label.name} (node {node}) expects >=2 inputs, "
                    f"got {len(in_nodes)}"
                )
            values[node] = _apply_variadic(label, [values[s] for s in in_nodes])

        else:
            raise EvaluationError(f"Unknown node type {label} at node {node}")

        # Clamp to finite range to prevent overflow propagation.
        values[node] = _clamp(values[node])

    # Return the output node's value.
    try:
        out = dag.output_node()
    except ValueError as e:
        raise EvaluationError(str(e)) from e
    return values[out]


# ======================================================================
# Operation dispatch
# ======================================================================


def _apply_unary(label: NodeType, x: float) -> float:
    """Apply a unary operation with numerical protection."""
    if label == NodeType.SIN:
        return math.sin(x)
    if label == NodeType.COS:
        return math.cos(x)
    if label == NodeType.EXP:
        return _protected_exp(x)
    if label == NodeType.LOG:
        return _protected_log(x)
    if label == NodeType.SQRT:
        return _protected_sqrt(x)
    if label == NodeType.ABS:
        return abs(x)
    raise EvaluationError(f"Unknown unary op: {label}")


def _apply_binary(label: NodeType, x: float, y: float) -> float:
    """Apply a binary operation with numerical protection.

    Input order: x has lower node ID, y has higher node ID (sorted).
    For SUB: result = x - y. For DIV: result = x / y. For POW: result = x ^ y.
    """
    if label == NodeType.SUB:
        return x - y
    if label == NodeType.DIV:
        return _protected_div(x, y)
    if label == NodeType.POW:
        return _protected_pow(x, y)
    raise EvaluationError(f"Unknown binary op: {label}")


def _apply_variadic(label: NodeType, xs: list[float]) -> float:
    """Apply a variadic operation (commutative, so input order doesn't matter)."""
    if label == NodeType.ADD:
        return reduce(lambda a, b: a + b, xs)
    if label == NodeType.MUL:
        return reduce(lambda a, b: a * b, xs)
    raise EvaluationError(f"Unknown variadic op: {label}")


# ======================================================================
# Protected operations (numerical stability)
# ======================================================================


def _protected_log(x: float) -> float:
    """Protected logarithm: log(|x| + epsilon)."""
    return math.log(abs(x) + 1e-10)


def _protected_div(x: float, y: float) -> float:
    """Protected division: x / y if |y| > epsilon, else 1.0."""
    if abs(y) > 1e-10:
        return x / y
    return 1.0


def _protected_sqrt(x: float) -> float:
    """Protected square root: sqrt(|x|)."""
    return math.sqrt(abs(x))


def _protected_exp(x: float) -> float:
    """Protected exponential: exp(clip(x, -500, 500))."""
    return math.exp(max(-500.0, min(500.0, x)))


def _protected_pow(x: float, y: float) -> float:
    """Protected power: |x|^y with overflow protection."""
    base = abs(x) + 1e-10
    exp = max(-100.0, min(100.0, y))
    try:
        result = float(base**exp)
    except OverflowError:
        return _MAX_VALUE
    if not math.isfinite(result):
        return _MAX_VALUE
    return result


def _clamp(value: float) -> float:
    """Clamp value to finite range [-MAX_VALUE, MAX_VALUE]."""
    if math.isnan(value):
        return 0.0
    if value > _MAX_VALUE:
        return _MAX_VALUE
    if value < -_MAX_VALUE:
        return -_MAX_VALUE
    return value
