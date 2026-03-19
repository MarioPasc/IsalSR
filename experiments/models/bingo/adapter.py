"""AGraph <-> LabeledDAG bidirectional adapter.

Converts between Bingo's AGraph command array (Nx3 integer stack) and
IsalSR's LabeledDAG. Handles operator mapping, edge direction, operand
ordering, and unused-row filtering.

Bingo command array format:
    Row i: [op_code, param1, param2]
    - op_code: integer operator (0=VARIABLE, 1=CONSTANT, 2=ADD, etc.)
    - param1, param2: indices to earlier rows (for binary ops)
    - Topological order guaranteed (params < current row index)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NodeType

# ======================================================================
# Operator mapping: Bingo op code -> IsalSR NodeType
# ======================================================================

BINGO_OP_TO_ISALSR: dict[int, NodeType] = {
    0: NodeType.VAR,  # VARIABLE
    1: NodeType.CONST,  # CONSTANT
    2: NodeType.ADD,  # ADDITION
    3: NodeType.SUB,  # SUBTRACTION
    4: NodeType.MUL,  # MULTIPLICATION
    5: NodeType.DIV,  # DIVISION
    6: NodeType.SIN,  # SIN
    7: NodeType.COS,  # COS
    8: NodeType.EXP,  # EXPONENTIAL
    9: NodeType.LOG,  # LOGARITHM
    10: NodeType.POW,  # POWER
    11: NodeType.ABS,  # ABS
    12: NodeType.SQRT,  # SQRT
}

# Bingo unary ops (arity 1): use only param1
BINGO_UNARY_OPS: frozenset[int] = frozenset({6, 7, 8, 9, 11, 12})
# Bingo binary ops (arity 2): use param1 and param2
BINGO_BINARY_OPS: frozenset[int] = frozenset({2, 3, 4, 5, 10})
# Bingo terminals (arity 0)
BINGO_TERMINALS: frozenset[int] = frozenset({0, 1, -1})


# ======================================================================
# AGraph -> LabeledDAG
# ======================================================================


def agraph_to_labeled_dag(
    agraph: Any,
    const_values: tuple[float, ...] | None = None,
) -> LabeledDAG:
    """Convert a Bingo AGraph to an IsalSR LabeledDAG.

    Handles:
    - VARIABLE rows: deduplicated by param1 (variable index)
    - CONSTANT rows: each creates a new CONST node
    - Unary ops: edge from param1 row
    - Binary ops: edges from param1 (first operand) then param2 (second)
    - Unused rows: filtered via get_utilized_commands()
    - CONST normalization: invariant #9

    Args:
        agraph: Bingo AGraph instance.
        const_values: Optional constant values. If None, uses agraph.constants.

    Returns:
        IsalSR LabeledDAG.
    """
    cmd = agraph.command_array  # Nx3 numpy array
    utilized = agraph.get_utilized_commands()  # list of bool, length N

    if const_values is None:
        try:
            cv = agraph.constants
            # Bingo stores constants as (n,1) or (n,) ndarray; flatten to 1D
            const_values = tuple(float(np.asarray(v).flat[0]) for v in cv)
        except (AttributeError, TypeError):
            const_values = ()

    n_rows = len(cmd)

    # Identify distinct variable indices from utilized VARIABLE rows
    var_indices: set[int] = set()
    for i in range(n_rows):
        if utilized[i] and cmd[i, 0] == 0:  # VARIABLE
            var_indices.add(int(cmd[i, 1]))

    m = max(var_indices) + 1 if var_indices else 1

    # Count utilized non-terminal rows for capacity estimate
    n_utilized = sum(1 for i in range(n_rows) if utilized[i])
    dag = LabeledDAG(max_nodes=m + n_utilized + 10)

    # Create VAR nodes (one per distinct variable index)
    var_node_map: dict[int, int] = {}  # var_index -> dag node id
    for vi in range(m):
        dag_id = dag.add_node(NodeType.VAR, var_index=vi)
        var_node_map[vi] = dag_id

    # Map command array row -> dag node id
    row_to_node: dict[int, int] = {}

    # Map VARIABLE rows to their VAR nodes
    for i in range(n_rows):
        if utilized[i] and cmd[i, 0] == 0:
            row_to_node[i] = var_node_map[int(cmd[i, 1])]

    # Process utilized rows top-to-bottom (topological order)
    for i in range(n_rows):
        if not utilized[i]:
            continue

        op_code = int(cmd[i, 0])
        param1 = int(cmd[i, 1])
        param2 = int(cmd[i, 2])

        if op_code == 0:
            # VARIABLE: already mapped
            continue

        if op_code == 1 or op_code == -1:
            # CONSTANT or INTEGER
            val = 1.0
            if const_values and op_code == 1 and param1 < len(const_values):
                val = float(const_values[param1])
            dag_id = dag.add_node(NodeType.CONST, const_value=val)
            row_to_node[i] = dag_id
            continue

        # Operation node
        if op_code not in BINGO_OP_TO_ISALSR:
            raise ValueError(f"Unsupported Bingo op code: {op_code}")

        node_type = BINGO_OP_TO_ISALSR[op_code]
        dag_id = dag.add_node(node_type)
        row_to_node[i] = dag_id

        if op_code in BINGO_UNARY_OPS:
            # Unary: edge from param1
            src = row_to_node.get(param1)
            if src is not None:
                dag.add_edge(src, dag_id)
        elif op_code in BINGO_BINARY_OPS:
            # Binary: param1 = first operand, param2 = second operand
            # Sequential add_edge preserves order via _input_order (invariant B8)
            src1 = row_to_node.get(param1)
            src2 = row_to_node.get(param2)
            if src1 is not None:
                dag.add_edge(src1, dag_id)
            if src2 is not None and src2 != src1:
                dag.add_edge(src2, dag_id)
            elif src2 is not None and src2 == src1:
                # Self-referencing (e.g., x+x): edge already added.
                # For ADD/MUL this is fine (single edge means one input).
                # For SUB/DIV/POW: x-x=0, x/x=1 are constant expressions.
                pass

    # Normalize CONST creation edges (invariant #9)
    _normalize_const_edges(dag)

    return dag


def _normalize_const_edges(dag: LabeledDAG) -> None:
    """Ensure all CONST nodes have at least one in-edge from node 0 (x_1)."""
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.CONST and dag.in_degree(i) == 0:
            dag.add_edge(0, i)


# ======================================================================
# LabeledDAG -> AGraph (for testing roundtrips)
# ======================================================================


def labeled_dag_to_agraph(dag: LabeledDAG) -> Any:
    """Convert an IsalSR LabeledDAG to a Bingo AGraph.

    Builds a topological-ordered command array from the DAG.

    Args:
        dag: IsalSR LabeledDAG.

    Returns:
        Bingo AGraph.
    """
    from bingo.symbolic_regression.agraph.agraph import AGraph

    isalsr_to_bingo: dict[NodeType, int] = {v: k for k, v in BINGO_OP_TO_ISALSR.items()}

    order = dag.topological_sort()
    isalsr_to_row: dict[int, int] = {}
    commands: list[list[int]] = []
    const_values: list[float] = []
    const_count = 0

    for node_id in order:
        label = dag.node_label(node_id)
        row_idx = len(commands)
        isalsr_to_row[node_id] = row_idx

        if label == NodeType.VAR:
            var_idx = dag.node_data(node_id).get("var_index", 0)
            commands.append([0, int(var_idx), int(var_idx)])

        elif label == NodeType.CONST:
            val = dag.node_data(node_id).get("const_value", 1.0)
            commands.append([1, const_count, const_count])
            const_values.append(float(val))
            const_count += 1

        else:
            op_code = isalsr_to_bingo.get(label)
            if op_code is None:
                raise ValueError(f"Cannot map IsalSR {label} to Bingo op")

            if label in BINARY_OPS:
                in_nodes = dag.ordered_inputs(node_id)
            else:
                in_nodes = sorted(dag.in_neighbors(node_id))

            if op_code in BINGO_UNARY_OPS:
                p1 = isalsr_to_row[in_nodes[0]] if in_nodes else 0
                commands.append([op_code, p1, p1])
            elif op_code in BINGO_BINARY_OPS:
                p1 = isalsr_to_row[in_nodes[0]] if len(in_nodes) > 0 else 0
                p2 = isalsr_to_row[in_nodes[1]] if len(in_nodes) > 1 else p1
                commands.append([op_code, p1, p2])

    ag = AGraph(use_simplification=False)
    ag._command_array = np.array(commands, dtype=int)
    ag._notify_modification()

    if const_values:
        ag.set_local_optimization_params(np.array(const_values))

    return ag
