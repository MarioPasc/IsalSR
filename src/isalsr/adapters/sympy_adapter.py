"""SymPy Expr <-> LabeledDAG adapter.

Critical for verifying expression correctness, pretty-printing, simplification,
and benchmark definition (SymPy expressions converted to DAGs).

Optional dependency: sympy >= 1.12
"""

from __future__ import annotations

from typing import Any

import sympy
from sympy import (
    Abs,
    Add,
    Mul,
    Pow,
    Symbol,
    cos,
    exp,
    log,
    sin,
    sqrt,
)

from isalsr.adapters.base import DAGAdapter
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


class SympyAdapter(DAGAdapter[sympy.Expr]):
    """Convert between SymPy expressions and LabeledDAGs."""

    def to_sympy(self, dag: LabeledDAG) -> sympy.Expr:
        """Convert a LabeledDAG to a SymPy expression.

        Traverses the DAG in topological order, building the SymPy
        expression bottom-up.

        Args:
            dag: The expression DAG.

        Returns:
            A SymPy expression equivalent to the DAG.
        """
        order = dag.topological_sort()
        node_exprs: dict[int, sympy.Expr] = {}

        for node in order:
            label = dag.node_label(node)
            data = dag.node_data(node)
            # BUG FIX B9: Use ordered_inputs for binary ops to preserve
            # operand order (SUB, DIV, POW are non-commutative).
            from isalsr.core.node_types import BINARY_OPS as _BIN_OPS

            if label in _BIN_OPS:
                in_nodes = dag.ordered_inputs(node)
            else:
                in_nodes = sorted(dag.in_neighbors(node))

            if label == NodeType.VAR:
                var_idx = int(data.get("var_index", 0))
                node_exprs[node] = Symbol(f"x_{var_idx}")

            elif label == NodeType.CONST:
                val = float(data.get("const_value", 1.0))
                node_exprs[node] = sympy.Float(val)

            elif label == NodeType.ADD:
                args = [node_exprs[n] for n in in_nodes]
                node_exprs[node] = Add(*args)

            elif label == NodeType.MUL:
                args = [node_exprs[n] for n in in_nodes]
                node_exprs[node] = Mul(*args)

            elif label == NodeType.SUB:
                node_exprs[node] = node_exprs[in_nodes[0]] - node_exprs[in_nodes[1]]

            elif label == NodeType.DIV:
                node_exprs[node] = node_exprs[in_nodes[0]] / node_exprs[in_nodes[1]]

            elif label == NodeType.SIN:
                node_exprs[node] = sin(node_exprs[in_nodes[0]])

            elif label == NodeType.COS:
                node_exprs[node] = cos(node_exprs[in_nodes[0]])

            elif label == NodeType.EXP:
                node_exprs[node] = exp(node_exprs[in_nodes[0]])

            elif label == NodeType.LOG:
                node_exprs[node] = log(node_exprs[in_nodes[0]])

            elif label == NodeType.SQRT:
                node_exprs[node] = sqrt(node_exprs[in_nodes[0]])

            elif label == NodeType.POW:
                node_exprs[node] = Pow(node_exprs[in_nodes[0]], node_exprs[in_nodes[1]])

            elif label == NodeType.ABS:
                node_exprs[node] = Abs(node_exprs[in_nodes[0]])

            elif label == NodeType.NEG:
                node_exprs[node] = -node_exprs[in_nodes[0]]

            elif label == NodeType.INV:
                node_exprs[node] = sympy.Integer(1) / node_exprs[in_nodes[0]]

        out = dag.output_node()
        return node_exprs[out]

    def to_external(self, dag: LabeledDAG) -> sympy.Expr:
        """Alias for to_sympy (implements DAGAdapter interface)."""
        return self.to_sympy(dag)

    def from_sympy(self, expr: sympy.Expr, variables: list[Symbol]) -> LabeledDAG:
        """Convert a SymPy expression to a LabeledDAG.

        Args:
            expr: SymPy expression.
            variables: Ordered list of SymPy Symbols (x_0, x_1, ...).

        Returns:
            A LabeledDAG representing the expression.
        """
        var_map: dict[Symbol, int] = {s: i for i, s in enumerate(variables)}
        memo: dict[int, int] = {}  # id(expr) -> dag node ID (shared subexpr detection)

        # Count nodes for capacity estimate.
        node_count = _count_nodes(expr)
        dag = LabeledDAG(max_nodes=len(variables) + node_count + 10)

        # Pre-insert VAR nodes.
        for i, _sym in enumerate(variables):
            dag.add_node(NodeType.VAR, var_index=i)

        _build_dag(expr, dag, var_map, memo)

        # Add creation edges for CONST nodes (required for D2S reachability).
        # CONST nodes are evaluation-neutral leaves that ignore in-edges,
        # but D2S needs them reachable from VAR nodes via outgoing edges.
        # Normalize: always create from x_1 (node 0).
        for i in range(dag.node_count):
            if dag.node_label(i) == NodeType.CONST and dag.in_degree(i) == 0:
                dag.add_edge(0, i)

        return dag

    def from_external(self, obj: sympy.Expr) -> LabeledDAG:
        """Convert SymPy expression to DAG (requires variables to be inferred).

        Variables are extracted as sorted free symbols.
        """
        free_syms = sorted(obj.free_symbols, key=lambda s: str(s))
        symbols = [s for s in free_syms if isinstance(s, Symbol)]
        return self.from_sympy(obj, symbols)


def _count_nodes(expr: sympy.Expr) -> int:
    """Estimate node count for capacity allocation."""
    if isinstance(expr, Symbol):
        return 0  # Already pre-inserted as VAR.
    if isinstance(expr, sympy.Number):
        return 1
    return 1 + sum(_count_nodes(a) for a in expr.args)


def _build_dag(
    expr: sympy.Expr,
    dag: LabeledDAG,
    var_map: dict[Symbol, int],
    memo: dict[int, int],
) -> int:
    """Recursively build DAG nodes for a SymPy expression.

    Returns the DAG node ID for this expression.
    """
    expr_id = id(expr)
    if expr_id in memo:
        return memo[expr_id]

    # Symbol → VAR (already pre-inserted).
    if isinstance(expr, Symbol):
        if expr in var_map:
            memo[expr_id] = var_map[expr]
            return var_map[expr]
        raise ValueError(f"Unknown symbol: {expr}")

    # Number → CONST.
    if isinstance(expr, sympy.Number):
        node = dag.add_node(NodeType.CONST, const_value=float(expr))
        memo[expr_id] = node
        return node

    # Determine operation type and build children first.
    child_ids: list[int] = []
    for arg in expr.args:
        child_ids.append(_build_dag(arg, dag, var_map, memo))

    # Map SymPy function to NodeType.
    node_type = _sympy_to_node_type(expr)
    node = dag.add_node(node_type)

    # Add edges: children provide input to this node.
    for child in child_ids:
        dag.add_edge(child, node)

    memo[expr_id] = node
    return node


def _sympy_to_node_type(expr: Any) -> NodeType:  # noqa: ANN401
    """Map a SymPy expression to its NodeType."""
    if isinstance(expr, Add):
        return NodeType.ADD
    if isinstance(expr, Mul):
        return NodeType.MUL
    if isinstance(expr, Pow):
        return NodeType.POW
    if isinstance(expr, sin):
        return NodeType.SIN
    if isinstance(expr, cos):
        return NodeType.COS
    if isinstance(expr, exp):
        return NodeType.EXP
    if isinstance(expr, log):
        return NodeType.LOG
    if isinstance(expr, sqrt):
        return NodeType.SQRT
    if isinstance(expr, Abs):
        return NodeType.ABS
    raise ValueError(f"Unsupported SymPy expression type: {type(expr).__name__}")
