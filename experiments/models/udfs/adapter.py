"""CompGraph <-> LabeledDAG bidirectional adapter.

Converts between UDFS's CompGraph representation and IsalSR's LabeledDAG.
Handles operation mapping, edge direction, operand ordering, and
identity node collapsing.

Reference: UDFS uses sub_l/sub_r/div_l/div_r for ordered binary ops,
while IsalSR uses edge insertion order via _input_order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure vendored DAG_search is importable
_vendor_dir = str(Path(__file__).parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

from DAG_search.comp_graph import CompGraph  # noqa: E402

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ======================================================================
# Operation mapping: UDFS -> IsalSR
# ======================================================================

# UDFS operations that encode operand order in the op name.
# sub_l(a, b) = a - b → SUB with children in original order
# sub_r(a, b) = b - a → SUB with children reversed
# div_l(a, b) = a / b → DIV with children in original order
# div_r(a, b) = b / a → DIV with children reversed
UDFS_OP_TO_ISALSR: dict[str, NodeType] = {
    "+": NodeType.ADD,
    "*": NodeType.MUL,
    "sub_l": NodeType.SUB,
    "sub_r": NodeType.SUB,
    "div_l": NodeType.DIV,
    "div_r": NodeType.DIV,
    "sin": NodeType.SIN,
    "cos": NodeType.COS,
    "exp": NodeType.EXP,
    "log": NodeType.LOG,
    "sqrt": NodeType.SQRT,
    "inv": NodeType.INV,
    "neg": NodeType.NEG,
}

# Ops where children order must be reversed to match IsalSR's
# "first add_edge = first operand" convention.
REVERSED_OPS = frozenset({"sub_r", "div_r"})

ISALSR_TO_UDFS_OP: dict[NodeType, str] = {
    NodeType.ADD: "+",
    NodeType.MUL: "*",
    NodeType.SUB: "sub_l",
    NodeType.DIV: "div_l",
    NodeType.SIN: "sin",
    NodeType.COS: "cos",
    NodeType.EXP: "exp",
    NodeType.LOG: "log",
    NodeType.SQRT: "sqrt",
    NodeType.INV: "inv",
    NodeType.NEG: "neg",
}


# ======================================================================
# CompGraph -> LabeledDAG
# ======================================================================


def compgraph_to_labeled_dag(
    cg: CompGraph,
    const_values: Any = None,
) -> LabeledDAG:
    """Convert a UDFS CompGraph to an IsalSR LabeledDAG.

    Handles:
    - Input nodes [0, m) → VAR nodes with var_index
    - Constant nodes [m, m+k) → CONST nodes
    - Identity nodes ('=') → collapsed (mapped to their child)
    - sub_r/div_r → reversed operand order
    - Edge direction: UDFS children → node matches IsalSR source → target

    Args:
        cg: UDFS CompGraph.
        const_values: Optional array of constant values.

    Returns:
        IsalSR LabeledDAG.
    """
    m = cg.inp_dim
    k = cg.n_consts
    n = cg.outp_dim

    dag = LabeledDAG(max_nodes=len(cg.node_dict) + 10)

    # udfs_id -> isalsr_id (or None if collapsed)
    node_map: dict[int, int] = {}

    # 1. Create VAR nodes [0, m)
    for i in range(m):
        isalsr_id = dag.add_node(NodeType.VAR, var_index=i)
        node_map[i] = isalsr_id

    # 2. Create CONST nodes [m, m+k)
    for i in range(m, m + k):
        val = 1.0
        if const_values is not None and (i - m) < len(const_values):
            val = float(const_values[i - m])
        isalsr_id = dag.add_node(NodeType.CONST, const_value=val)
        node_map[i] = isalsr_id

    # 3. Process remaining nodes in topological (evaluation) order
    eval_order = cg.eval_order if cg.eval_order is not None else sorted(cg.node_dict.keys())

    for udfs_id in eval_order:
        if udfs_id in node_map:
            continue

        children, op = cg.node_dict[udfs_id]

        if op == "=":
            # Identity node — collapse by mapping to its single child
            child_isalsr = _resolve(node_map, children[0])
            node_map[udfs_id] = child_isalsr
            continue

        if op not in UDFS_OP_TO_ISALSR:
            raise ValueError(f"Unsupported UDFS operation: {op!r}")

        node_type = UDFS_OP_TO_ISALSR[op]
        isalsr_id = dag.add_node(node_type)
        node_map[udfs_id] = isalsr_id

        # Determine child order
        ordered_children = list(children)
        if op in REVERSED_OPS:
            ordered_children = list(reversed(ordered_children))

        # Add edges: child → new node (preserving operand order)
        for child_udfs in ordered_children:
            child_isalsr = _resolve(node_map, child_udfs)
            dag.add_edge(child_isalsr, isalsr_id)

    # Normalize CONST creation edges (invariant #9)
    _normalize_const_edges(dag)

    return dag


def _resolve(node_map: dict[int, int], udfs_id: int) -> int:
    """Resolve a UDFS node ID to its IsalSR node ID, following identity collapses."""
    return node_map[udfs_id]


def _normalize_const_edges(dag: LabeledDAG) -> None:
    """Ensure all CONST nodes have at least one in-edge from node 0 (x_1).

    CONST nodes are evaluation-neutral leaves but need a creation edge
    for D2S reachability. Normalize: create edge from x_1 (node 0).
    """
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.CONST and dag.in_degree(i) == 0:
            dag.add_edge(0, i)


# ======================================================================
# LabeledDAG -> CompGraph
# ======================================================================


def labeled_dag_to_compgraph(
    dag: LabeledDAG,
    const_values: list[float] | None = None,
) -> CompGraph:
    """Convert an IsalSR LabeledDAG to a UDFS CompGraph.

    Args:
        dag: IsalSR LabeledDAG.
        const_values: Values for CONST nodes. If None, uses 1.0.

    Returns:
        UDFS CompGraph.
    """
    # Count variables and constants
    var_nodes = []
    const_nodes = []
    op_nodes = []

    for i in range(dag.node_count):
        label = dag.node_label(i)
        if label == NodeType.VAR:
            var_nodes.append(i)
        elif label == NodeType.CONST:
            const_nodes.append(i)
        else:
            op_nodes.append(i)

    m = len(var_nodes)
    k = len(const_nodes)
    n = 1  # single output

    # Build node_dict
    # UDFS layout: [0..m) inputs, [m..m+k) consts, [m+k..m+k+n) outputs, [m+k+n..) intermediates
    isalsr_to_udfs: dict[int, int] = {}

    # Map VAR nodes
    for isalsr_id in var_nodes:
        var_idx = dag.node_data(isalsr_id).get("var_index", 0)
        isalsr_to_udfs[isalsr_id] = var_idx

    # Map CONST nodes
    for j, isalsr_id in enumerate(const_nodes):
        isalsr_to_udfs[isalsr_id] = m + j

    # Map operation nodes
    next_udfs_id = m + k + n  # after output node
    for isalsr_id in op_nodes:
        isalsr_to_udfs[isalsr_id] = next_udfs_id
        next_udfs_id += 1

    # Build node_dict
    node_dict: dict[int, tuple[tuple[int, ...], str]] = {}

    # Input nodes
    for i in range(m):
        node_dict[i] = ((), "=")

    # Const nodes
    for i in range(m, m + k):
        node_dict[i] = ((), "=")

    # Operation nodes
    for isalsr_id in op_nodes:
        label = dag.node_label(isalsr_id)
        udfs_id = isalsr_to_udfs[isalsr_id]

        # Get ordered inputs
        from isalsr.core.node_types import BINARY_OPS

        if label in BINARY_OPS:
            in_nodes = dag.ordered_inputs(isalsr_id)
        else:
            in_nodes = sorted(dag.in_neighbors(isalsr_id))

        children = tuple(isalsr_to_udfs[c] for c in in_nodes)
        op = ISALSR_TO_UDFS_OP.get(label)
        if op is None:
            raise ValueError(f"Cannot map IsalSR {label} to UDFS operation")

        node_dict[udfs_id] = (children, op)

    # Output node (identity '=')
    out_node = dag.output_node()
    out_udfs = m + k  # first output position
    node_dict[out_udfs] = ((isalsr_to_udfs[out_node],), "=")

    cg = CompGraph(m, n, k, node_dict=node_dict)
    return cg
