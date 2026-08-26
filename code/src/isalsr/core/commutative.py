"""DAG-level conversion between non-commutative and commutative representations.

Implements the unary decomposition from GraphSR (Xiang et al., NeurIPS 2025):
    SUB(a, b) = ADD(a, NEG(b))
    DIV(a, b) = MUL(a, INV(b))

This eliminates non-commutative binary ops (except POW), enabling
simpler isomorphism checking where operand order is irrelevant for
all variadic ops.

Semantic invariant: for any valid expression DAG D and inputs x,
    evaluate_dag(to_commutative(D), x) == evaluate_dag(D, x)
    evaluate_dag(from_commutative(to_commutative(D)), x) == evaluate_dag(D, x)

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import logging

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import (
    NodeType,
)

log = logging.getLogger(__name__)


def to_commutative(dag: LabeledDAG) -> LabeledDAG:
    """Convert a DAG with SUB/DIV nodes to use ADD+NEG / MUL+INV.

    Each SUB node with ordered_inputs [a, b] becomes:
        NEG node with edge mapped_b -> NEG
        ADD node with edges mapped_a -> ADD, NEG -> ADD

    Each DIV node with ordered_inputs [a, b] becomes:
        INV node with edge mapped_b -> INV
        MUL node with edges mapped_a -> MUL, INV -> MUL

    All other nodes (VAR, CONST, unary, variadic, POW) are copied as-is
    with their input edges remapped through the old->new node mapping.

    Args:
        dag: The input expression DAG (may contain SUB/DIV nodes).

    Returns:
        A new LabeledDAG with SUB replaced by ADD+NEG and DIV by MUL+INV.
        If the input contains no SUB or DIV nodes, the result is a
        structural copy of the input.
    """
    # Count SUB and DIV nodes to allocate extra capacity.
    n = dag.node_count
    extra = 0
    for i in range(n):
        label = dag.node_label(i)
        if label in (NodeType.SUB, NodeType.DIV):
            extra += 1

    new_dag = LabeledDAG(n + extra)

    # Mapping from old node ID -> new node ID.
    node_map: dict[int, int] = {}

    order = dag.topological_sort()

    for old_node in order:
        label = dag.node_label(old_node)
        data = dag.node_data(old_node)

        if label == NodeType.SUB:
            # SUB(a, b) -> ADD(a, NEG(b))
            old_inputs = dag.ordered_inputs(old_node)
            if len(old_inputs) != 2:
                raise ValueError(f"SUB node {old_node} has {len(old_inputs)} inputs, expected 2")
            mapped_a = node_map[old_inputs[0]]
            mapped_b = node_map[old_inputs[1]]

            # Create NEG node: mapped_b -> NEG
            neg_node = new_dag.add_node(NodeType.NEG)
            new_dag.add_edge(mapped_b, neg_node)

            # Create ADD node: mapped_a -> ADD, NEG -> ADD
            add_node = new_dag.add_node(NodeType.ADD)
            new_dag.add_edge(mapped_a, add_node)
            new_dag.add_edge(neg_node, add_node)

            # Map old SUB node -> new ADD node (for outgoing edges).
            node_map[old_node] = add_node

        elif label == NodeType.DIV:
            # DIV(a, b) -> MUL(a, INV(b))
            old_inputs = dag.ordered_inputs(old_node)
            if len(old_inputs) != 2:
                raise ValueError(f"DIV node {old_node} has {len(old_inputs)} inputs, expected 2")
            mapped_a = node_map[old_inputs[0]]
            mapped_b = node_map[old_inputs[1]]

            # Create INV node: mapped_b -> INV
            inv_node = new_dag.add_node(NodeType.INV)
            new_dag.add_edge(mapped_b, inv_node)

            # Create MUL node: mapped_a -> MUL, INV -> MUL
            mul_node = new_dag.add_node(NodeType.MUL)
            new_dag.add_edge(mapped_a, mul_node)
            new_dag.add_edge(inv_node, mul_node)

            # Map old DIV node -> new MUL node.
            node_map[old_node] = mul_node

        else:
            # Copy node as-is (VAR, CONST, unary, variadic, POW).
            new_node = new_dag.add_node(
                label,
                var_index=int(data["var_index"]) if "var_index" in data else None,
                const_value=float(data["const_value"]) if "const_value" in data else None,
            )
            node_map[old_node] = new_node

            # Reconnect input edges using the mapping.
            # ordered_inputs preserves insertion order, which is critical for
            # non-commutative binary ops (only POW reaches here since SUB/DIV
            # are handled above). For other node types, order is irrelevant
            # but preserved for consistency.
            for src in dag.ordered_inputs(old_node):
                new_dag.add_edge(node_map[src], new_node)

    return new_dag


def from_commutative(dag: LabeledDAG) -> LabeledDAG:
    """Convert a commutative DAG back to non-commutative form where possible.

    Pattern-matches and collapses:
        ADD(a, NEG(b)) -> SUB(a, b)  when ADD has exactly 2 inputs
                                      and NEG has out_degree == 1
        MUL(a, INV(b)) -> DIV(a, b)  when MUL has exactly 2 inputs
                                      and INV has out_degree == 1

    The NEG/INV node is absorbed into the binary op and removed from
    the output DAG. If a NEG/INV feeds multiple consumers, it is NOT
    absorbed (it would change the DAG structure).

    Args:
        dag: The input expression DAG (commutative form with NEG/INV).

    Returns:
        A new LabeledDAG with matched patterns collapsed to SUB/DIV.
        Unmatched NEG/INV nodes are preserved as-is.
    """
    n = dag.node_count

    # Phase 1: Identify collapsible (variadic_node, unary_node) pairs.
    # A pair is collapsible iff:
    #   - variadic_node is ADD/MUL with exactly 2 inputs
    #   - one input is NEG/INV (respectively) with out_degree == 1
    #   - the NEG/INV is used ONLY by this variadic node
    absorbed: set[int] = set()  # NEG/INV nodes that will be absorbed.
    collapse_info: dict[int, tuple[NodeType, int, int]] = {}
    # Maps variadic_node -> (new_label, first_operand, second_operand)
    # where second_operand is the input to the NEG/INV (not the NEG/INV itself).

    for node in range(n):
        label = dag.node_label(node)

        if label == NodeType.ADD:
            _try_collapse(dag, node, NodeType.NEG, NodeType.SUB, absorbed, collapse_info)
        elif label == NodeType.MUL:
            _try_collapse(dag, node, NodeType.INV, NodeType.DIV, absorbed, collapse_info)

    # Phase 2: Build the new DAG.
    # Nodes that are absorbed (NEG/INV) are skipped entirely.
    # Collapsed variadic nodes become binary ops.
    new_dag = LabeledDAG(n - len(absorbed))

    node_map: dict[int, int] = {}
    order = dag.topological_sort()

    for old_node in order:
        if old_node in absorbed:
            continue

        label = dag.node_label(old_node)
        data = dag.node_data(old_node)

        if old_node in collapse_info:
            # This ADD/MUL becomes SUB/DIV.
            new_label, first_op, second_op = collapse_info[old_node]
            new_node = new_dag.add_node(new_label)
            node_map[old_node] = new_node

            # Wire operands in correct order: first_op, then second_op.
            # first_op and second_op are old node IDs already resolved
            # through the NEG/INV's input.
            new_dag.add_edge(node_map[first_op], new_node)
            new_dag.add_edge(node_map[second_op], new_node)
        else:
            # Copy node as-is.
            new_node = new_dag.add_node(
                label,
                var_index=int(data["var_index"]) if "var_index" in data else None,
                const_value=float(data["const_value"]) if "const_value" in data else None,
            )
            node_map[old_node] = new_node

            # Reconnect inputs using the mapping, skipping absorbed nodes.
            # ordered_inputs preserves insertion order (critical for POW).
            for src in dag.ordered_inputs(old_node):
                if src not in absorbed:
                    new_dag.add_edge(node_map[src], new_node)

    return new_dag


def _try_collapse(
    dag: LabeledDAG,
    variadic_node: int,
    unary_type: NodeType,
    binary_type: NodeType,
    absorbed: set[int],
    collapse_info: dict[int, tuple[NodeType, int, int]],
) -> None:
    """Try to collapse a variadic+unary pair into a binary op.

    Checks if ``variadic_node`` (ADD or MUL) has exactly 2 inputs,
    one of which is a ``unary_type`` (NEG or INV) with out_degree 1.
    If so, records the collapse and marks the unary node as absorbed.

    For ADD(a, NEG(b)): the non-NEG input ``a`` is the first operand
    of SUB, and the input to NEG (which is ``b``) is the second operand.

    For MUL(a, INV(b)): the non-INV input ``a`` is the first operand
    of DIV, and the input to INV (which is ``b``) is the second operand.

    Args:
        dag: The DAG being analyzed.
        variadic_node: The ADD or MUL node to check.
        unary_type: The unary node type to look for (NEG or INV).
        binary_type: The binary node type to create (SUB or DIV).
        absorbed: Set of nodes to be absorbed (mutated in place).
        collapse_info: Dict mapping collapsed nodes to their info (mutated).
    """
    inputs = dag.ordered_inputs(variadic_node)
    if len(inputs) != 2:
        return

    # Check each input to see if it is the target unary type.
    # We need exactly one to be the unary type with out_degree 1.
    unary_candidates: list[tuple[int, int]] = []
    # Each entry: (position_in_inputs, unary_node_id)

    for pos, inp in enumerate(inputs):
        if inp in absorbed:
            # Already claimed by another collapse.
            continue
        if dag.node_label(inp) == unary_type and dag.out_degree(inp) == 1:
            unary_candidates.append((pos, inp))

    if len(unary_candidates) != 1:
        # Either zero candidates (no match) or two candidates (ambiguous:
        # ADD(NEG(a), NEG(b)) -- we don't collapse this case since
        # it would require choosing which NEG to absorb, and the result
        # NEG(a) - NEG(b) is wrong. Leave as-is.)
        return

    unary_pos, unary_node = unary_candidates[0]
    other_pos = 1 - unary_pos

    # The unary node's single input is the "second operand" of the binary op.
    unary_inputs = dag.ordered_inputs(unary_node)
    if len(unary_inputs) != 1:
        return

    second_operand = unary_inputs[0]
    first_operand = inputs[other_pos]

    absorbed.add(unary_node)
    collapse_info[variadic_node] = (binary_type, first_operand, second_operand)


__all__: list[str] = [
    "to_commutative",
    "from_commutative",
]
