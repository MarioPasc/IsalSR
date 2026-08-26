"""Unit tests for _input_order tracking in LabeledDAG.

The _input_order field (list[list[int]]) records the order in which in-edges
are added to each node. This is critical for non-commutative binary ops
(SUB, DIV, POW) where operand order determines evaluation semantics:
    - For x - y: _input_order of the SUB node is [x_id, y_id]
      meaning first_operand=x, second_operand=y.
    - V/v creates the first edge, C/c creates subsequent edges.

This module tests 8 categories:
1. Basic add_edge tracking for all op types
2. remove_edge maintenance
3. undo_node cleanup
4. Backtracking cycles (add -> remove -> re-add with different source)
5. Duplicate edge rejection (no spurious _input_order modification)
6. Cycle detection rejection (no spurious _input_order modification)
7. ordered_inputs accessor (returns copy, matches _input_order)
8. Complex DAG patterns (diamond, chain, fan-out, fan-in)
"""

from __future__ import annotations

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ======================================================================
# Category 1: Basic add_edge tracking
# ======================================================================


class TestBasicAddEdgeTracking:
    """Verify _input_order is populated correctly when edges are added."""

    def test_single_edge_unary(self) -> None:
        """sin(x): edge x->sin populates _input_order[sin] = [x]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        assert dag._input_order[sin] == [x]

    def test_two_edges_binary_sub(self) -> None:
        """x - y: _input_order[sub] = [x, y] -- order matters for SUB."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [x, y]

    def test_two_edges_binary_div(self) -> None:
        """x / y: _input_order[div] = [x, y] -- order matters for DIV."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        div = dag.add_node(NodeType.DIV)
        dag.add_edge(x, div)
        dag.add_edge(y, div)
        assert dag._input_order[div] == [x, y]

    def test_two_edges_binary_pow(self) -> None:
        """x ^ y: _input_order[pow] = [x, y] -- order matters for POW."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        pw = dag.add_node(NodeType.POW)
        dag.add_edge(x, pw)
        dag.add_edge(y, pw)
        assert dag._input_order[pw] == [x, y]

    def test_reversed_order_binary_sub(self) -> None:
        """y - x: _input_order[sub] = [y, x] -- reversed from x - y."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(y, sub)  # y first
        dag.add_edge(x, sub)  # x second
        assert dag._input_order[sub] == [y, x]

    def test_variadic_add_three_inputs(self) -> None:
        """x + y + z: _input_order[add] = [x, y, z] in insertion order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        add = dag.add_node(NodeType.ADD)
        dag.add_edge(x, add)
        dag.add_edge(y, add)
        dag.add_edge(z, add)
        assert dag._input_order[add] == [x, y, z]

    def test_variadic_mul_three_inputs(self) -> None:
        """x * y * z: _input_order[mul] = [x, y, z] in insertion order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        mul = dag.add_node(NodeType.MUL)
        dag.add_edge(x, mul)
        dag.add_edge(y, mul)
        dag.add_edge(z, mul)
        assert dag._input_order[mul] == [x, y, z]

    def test_leaf_nodes_have_empty_input_order(self) -> None:
        """VAR and CONST nodes have no incoming edges, so empty _input_order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        k = dag.add_node(NodeType.CONST, const_value=1.0)
        assert dag._input_order[x] == []
        assert dag._input_order[k] == []

    def test_edge_source_not_target_tracked(self) -> None:
        """_input_order tracks source nodes of incoming edges, not outgoing."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        # x has no incoming edges, so its _input_order should be empty
        assert dag._input_order[x] == []
        # sin has one incoming edge from x
        assert dag._input_order[sin] == [x]


# ======================================================================
# Category 2: remove_edge maintenance
# ======================================================================


class TestRemoveEdgeMaintenance:
    """Verify _input_order is updated correctly when edges are removed."""

    def test_remove_only_edge(self) -> None:
        """Remove x->sin: _input_order[sin] becomes []."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        assert dag._input_order[sin] == [x]
        dag.remove_edge(x, sin)
        assert dag._input_order[sin] == []

    def test_remove_first_of_two_edges(self) -> None:
        """Remove x from x-y (SUB): _input_order[sub] becomes [y]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        dag.remove_edge(x, sub)
        assert dag._input_order[sub] == [y]

    def test_remove_second_of_two_edges(self) -> None:
        """Remove y from x-y (SUB): _input_order[sub] becomes [x]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        dag.remove_edge(y, sub)
        assert dag._input_order[sub] == [x]

    def test_remove_nonexistent_edge_no_change(self) -> None:
        """remove_edge for non-existent edge returns False, no _input_order change."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        original = list(dag._input_order[sub])
        result = dag.remove_edge(y, sub)
        assert result is False
        assert dag._input_order[sub] == original

    def test_remove_middle_of_three_edges(self) -> None:
        """Remove y from x+y+z (ADD): _input_order[add] becomes [x, z]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        add = dag.add_node(NodeType.ADD)
        dag.add_edge(x, add)
        dag.add_edge(y, add)
        dag.add_edge(z, add)
        dag.remove_edge(y, add)
        assert dag._input_order[add] == [x, z]


# ======================================================================
# Category 3: undo_node cleanup
# ======================================================================


class TestUndoNodeCleanup:
    """Verify _input_order is cleaned up for the removed node AND that the
    node is removed from other nodes' _input_order lists."""

    def test_undo_clears_removed_node_input_order(self) -> None:
        """undo_node on SUB node clears its own _input_order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [x, y]
        dag.undo_node()
        # After undo, _input_order[sub_index] should be empty
        assert dag._input_order[2] == []

    def test_undo_removes_node_from_others_input_order(self) -> None:
        """undo_node removes the undone node from other nodes' _input_order.

        Scenario: x(0) -> add(2), add(2) -> sin(3), y(1) -> sin(3).
        undo_node removes sin(3). sin's entry in add's outgoing edges is
        handled, but add(2) is NOT in sin(3)'s input order -- sin(3) IS in
        add(2)'s _input_order only if there's an edge from sin(3) to add(2).

        Better scenario: x(0) -> mul(2), mul(2) -> sub(3), y(1) -> sub(3).
        undo_node removes sub(3). sub(3) had outgoing edges to nobody (it's
        the last node), and incoming edges from mul(2) and y(1).
        Check: mul(2) _input_order should NOT contain sub(3) (it shouldn't,
        because sub(3) was a TARGET, not a source, of mul(2)'s edges).

        The real test: if undone node had outgoing edges, targets' _input_order
        must be cleaned.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        add = dag.add_node(NodeType.ADD)  # 2
        sin = dag.add_node(NodeType.SIN)  # 3
        dag.add_edge(x, add)  # x -> add
        dag.add_edge(y, add)  # y -> add
        dag.add_edge(add, sin)  # add -> sin
        assert dag._input_order[sin] == [add]
        # Now undo sin (last node). This removes edge add->sin.
        # add should remain in _input_order[sin] is irrelevant since sin
        # is removed. But verify add's _input_order is untouched.
        dag.undo_node()
        assert dag.node_count == 3
        assert dag._input_order[add] == [x, y]
        # sin's slot should be cleared
        assert dag._input_order[3] == []

    def test_undo_node_with_outgoing_edges_cleans_targets(self) -> None:
        """If the undone node has outgoing edges, those targets' _input_order
        must have the undone node removed.

        Build: x(0), add(1), sin(2). Edges: x->add, add->sin.
        undo_node removes sin(2): sin has no outgoing edges, so no cleanup
        of other targets needed from sin's outgoing. sin's incoming (from add)
        is cleaned.

        Better: x(0), sin(1), add(2). Edges: x->sin, x->add, sin->add.
        undo_node removes add(2): add had incoming from x and sin.
        Check x._input_order and sin._input_order are unchanged (add was
        a target, not a source, of their out-edges).

        Actually, the key case: node being undone has OUTGOING edges.
        Build: x(0), add(1), sub(2). Edges: x->add, x->sub, sub->add.
        Wait, sub->add would mean sub provides input to add.
        _input_order[add] = [x, sub].
        undo_node removes sub(2). sub's outgoing edge is sub->add.
        So add's _input_order should have sub removed: [x].
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        add = dag.add_node(NodeType.ADD)  # 1
        k = dag.add_node(NodeType.CONST, const_value=2.0)  # 2

        dag.add_edge(x, add)  # x -> add
        dag.add_edge(k, add)  # k -> add
        assert dag._input_order[add] == [x, k]

        # undo_node removes k (node 2, the last).
        # k's outgoing edge is k->add.
        # So add's _input_order should lose k.
        dag.undo_node()
        assert dag.node_count == 2
        assert dag._input_order[add] == [x]

    def test_undo_node_incoming_edges_cleaned(self) -> None:
        """When undone node has incoming edges, those sources' _out_adj is cleaned.

        Build: x(0), y(1), sub(2). Edges: x->sub, y->sub.
        undo_node removes sub(2). sub had incoming from x and y.
        x's out_adj should no longer contain sub. y same.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        sub = dag.add_node(NodeType.SUB)  # 2
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        dag.undo_node()
        assert dag.out_degree(x) == 0
        assert dag.out_degree(y) == 0
        assert dag._input_order[2] == []

    def test_undo_preserves_unrelated_input_order(self) -> None:
        """Undoing the last node should not affect _input_order of unrelated nodes."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        add = dag.add_node(NodeType.ADD)  # 2
        sin = dag.add_node(NodeType.SIN)  # 3
        dag.add_edge(x, add)  # x -> add
        dag.add_edge(y, add)  # y -> add
        dag.add_edge(x, sin)  # x -> sin (unrelated to add)

        dag.undo_node()  # removes sin(3)
        # add's _input_order should be untouched
        assert dag._input_order[add] == [x, y]


# ======================================================================
# Category 4: Backtracking cycles (add -> remove -> re-add)
# ======================================================================


class TestBacktrackingCycles:
    """Simulate what canonical.py does: add edge, backtrack (remove), try
    different source. Verify _input_order reflects the final state."""

    def test_add_remove_add_different_source(self) -> None:
        """Add x->sub, remove x->sub, add y->sub.
        Final _input_order[sub] = [y]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)

        dag.add_edge(x, sub)
        assert dag._input_order[sub] == [x]
        dag.remove_edge(x, sub)
        assert dag._input_order[sub] == []
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [y]

    def test_backtrack_second_operand(self) -> None:
        """Build x-y, backtrack second edge, replace with x-z.
        _input_order[sub] goes: [x] -> [x,y] -> [x] -> [x,z]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        sub = dag.add_node(NodeType.SUB)

        dag.add_edge(x, sub)
        assert dag._input_order[sub] == [x]
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [x, y]

        # Backtrack: remove y->sub
        dag.remove_edge(y, sub)
        assert dag._input_order[sub] == [x]

        # Try z instead
        dag.add_edge(z, sub)
        assert dag._input_order[sub] == [x, z]

    def test_full_backtrack_and_rebuild(self) -> None:
        """Build x-y, fully backtrack both edges, rebuild as y-x.
        This tests that operand reversal is correctly tracked."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)

        # Build x - y
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [x, y]

        # Full backtrack
        dag.remove_edge(y, sub)
        dag.remove_edge(x, sub)
        assert dag._input_order[sub] == []

        # Rebuild as y - x (reversed operands)
        dag.add_edge(y, sub)
        dag.add_edge(x, sub)
        assert dag._input_order[sub] == [y, x]

    def test_multiple_backtrack_iterations(self) -> None:
        """Simulate canonical search trying multiple candidates for an edge.
        Each iteration adds then removes, then the final one sticks."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        add = dag.add_node(NodeType.ADD)

        # First attempt: x -> add
        dag.add_edge(x, add)
        dag.remove_edge(x, add)

        # Second attempt: y -> add
        dag.add_edge(y, add)
        dag.remove_edge(y, add)

        # Third attempt: z -> add (this one sticks)
        dag.add_edge(z, add)
        assert dag._input_order[add] == [z]


# ======================================================================
# Category 5: Duplicate edge rejection
# ======================================================================


class TestDuplicateEdgeRejection:
    """add_edge returns False for duplicates -- verify _input_order is NOT modified."""

    def test_duplicate_does_not_append(self) -> None:
        """Adding the same edge twice should not duplicate in _input_order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        assert dag._input_order[sin] == [x]

        result = dag.add_edge(x, sin)
        assert result is False
        assert dag._input_order[sin] == [x]  # still just one entry

    def test_duplicate_binary_does_not_append(self) -> None:
        """For x-y, re-adding x->sub should not create [x, y, x]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        assert dag._input_order[sub] == [x, y]

        result = dag.add_edge(x, sub)
        assert result is False
        assert dag._input_order[sub] == [x, y]  # unchanged

    def test_duplicate_preserves_edge_count(self) -> None:
        """Duplicate edge should not change edge_count."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        count_before = dag.edge_count
        dag.add_edge(x, sin)
        assert dag.edge_count == count_before


# ======================================================================
# Category 6: Cycle detection rejection
# ======================================================================


class TestCycleDetectionRejection:
    """add_edge returns False for cycles -- verify _input_order is NOT modified."""

    def test_self_loop_no_input_order_change(self) -> None:
        """Self-loop rejection should not touch _input_order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        assert dag._input_order[x] == []
        result = dag.add_edge(x, x)
        assert result is False
        assert dag._input_order[x] == []

    def test_direct_cycle_no_input_order_change(self) -> None:
        """A->B, then B->A rejected. _input_order[A] should stay empty."""
        dag = LabeledDAG(max_nodes=10)
        a = dag.add_node(NodeType.VAR, var_index=0)
        b = dag.add_node(NodeType.ADD)
        dag.add_edge(a, b)
        assert dag._input_order[a] == []
        assert dag._input_order[b] == [a]

        result = dag.add_edge(b, a)
        assert result is False
        # Both _input_orders should be unchanged
        assert dag._input_order[a] == []
        assert dag._input_order[b] == [a]

    def test_indirect_cycle_no_input_order_change(self) -> None:
        """A->B->C, then C->A rejected. No _input_order modification."""
        dag = LabeledDAG(max_nodes=10)
        a = dag.add_node(NodeType.VAR, var_index=0)
        b = dag.add_node(NodeType.SIN)
        c = dag.add_node(NodeType.EXP)
        dag.add_edge(a, b)
        dag.add_edge(b, c)

        original_a = list(dag._input_order[a])
        original_b = list(dag._input_order[b])
        original_c = list(dag._input_order[c])

        result = dag.add_edge(c, a)
        assert result is False

        assert dag._input_order[a] == original_a
        assert dag._input_order[b] == original_b
        assert dag._input_order[c] == original_c


# ======================================================================
# Category 7: ordered_inputs accessor
# ======================================================================


class TestOrderedInputsAccessor:
    """Verify ordered_inputs returns a copy and matches _input_order."""

    def test_returns_correct_order(self) -> None:
        """ordered_inputs should return elements in insertion order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        sub = dag.add_node(NodeType.SUB)
        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        assert dag.ordered_inputs(sub) == [x, y]

    def test_returns_copy_not_reference(self) -> None:
        """Modifying the returned list should not affect internal state."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)

        inputs = dag.ordered_inputs(sin)
        inputs.append(999)  # mutate the returned copy

        # Internal state should be unchanged
        assert dag._input_order[sin] == [x]
        assert dag.ordered_inputs(sin) == [x]

    def test_matches_internal_input_order(self) -> None:
        """ordered_inputs(node) should equal _input_order[node] element-wise."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        y = dag.add_node(NodeType.VAR, var_index=1)
        z = dag.add_node(NodeType.VAR, var_index=2)
        mul = dag.add_node(NodeType.MUL)
        dag.add_edge(z, mul)
        dag.add_edge(x, mul)
        dag.add_edge(y, mul)
        assert dag.ordered_inputs(mul) == dag._input_order[mul]
        assert dag.ordered_inputs(mul) == [z, x, y]

    def test_ordered_inputs_empty_for_leaf(self) -> None:
        """Leaf nodes (VAR, CONST) have no inputs."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        k = dag.add_node(NodeType.CONST, const_value=3.14)
        assert dag.ordered_inputs(x) == []
        assert dag.ordered_inputs(k) == []

    def test_ordered_inputs_invalid_node_raises(self) -> None:
        """ordered_inputs on an invalid node should raise IndexError."""
        dag = LabeledDAG(max_nodes=10)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(IndexError):
            dag.ordered_inputs(5)

    def test_ordered_inputs_is_new_list_each_call(self) -> None:
        """Two calls to ordered_inputs should return distinct list objects."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)
        sin = dag.add_node(NodeType.SIN)
        dag.add_edge(x, sin)
        list1 = dag.ordered_inputs(sin)
        list2 = dag.ordered_inputs(sin)
        assert list1 == list2
        assert list1 is not list2


# ======================================================================
# Category 8: Complex DAG patterns
# ======================================================================


class TestComplexDAGPatterns:
    """Diamond, chain, fan-out, fan-in patterns with _input_order verification."""

    def test_diamond_shared_subexpression(self) -> None:
        """Diamond: x(0) -> add(2), y(1) -> add(2), add(2) -> sub(3), add(2) -> mul(4).

        This represents expressions like (x+y) - f((x+y)), where (x+y) is shared.
        Check _input_order for all non-leaf nodes.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        add = dag.add_node(NodeType.ADD)  # 2
        sub = dag.add_node(NodeType.SUB)  # 3
        mul = dag.add_node(NodeType.MUL)  # 4

        dag.add_edge(x, add)  # x -> add
        dag.add_edge(y, add)  # y -> add
        dag.add_edge(add, sub)  # add -> sub (first operand of sub)
        dag.add_edge(add, mul)  # add -> mul (first operand of mul)
        dag.add_edge(y, sub)  # y -> sub (second operand of sub)
        dag.add_edge(x, mul)  # x -> mul (second operand of mul)

        assert dag._input_order[add] == [x, y]
        assert dag._input_order[sub] == [add, y]
        assert dag._input_order[mul] == [add, x]

    def test_chain_unary(self) -> None:
        """Chain: x -> sin -> cos -> exp.
        Each node has exactly one input in its _input_order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        sin = dag.add_node(NodeType.SIN)  # 1
        cos = dag.add_node(NodeType.COS)  # 2
        exp = dag.add_node(NodeType.EXP)  # 3

        dag.add_edge(x, sin)
        dag.add_edge(sin, cos)
        dag.add_edge(cos, exp)

        assert dag._input_order[x] == []
        assert dag._input_order[sin] == [x]
        assert dag._input_order[cos] == [sin]
        assert dag._input_order[exp] == [cos]

    def test_fan_out(self) -> None:
        """Fan-out: x feeds into sin, cos, and exp.
        x._input_order is empty; each unary has [x]."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        sin = dag.add_node(NodeType.SIN)  # 1
        cos = dag.add_node(NodeType.COS)  # 2
        exp = dag.add_node(NodeType.EXP)  # 3

        dag.add_edge(x, sin)
        dag.add_edge(x, cos)
        dag.add_edge(x, exp)

        assert dag._input_order[x] == []
        assert dag._input_order[sin] == [x]
        assert dag._input_order[cos] == [x]
        assert dag._input_order[exp] == [x]

    def test_fan_in(self) -> None:
        """Fan-in: x, y, z all feed into add.
        _input_order[add] = [x, y, z] in insertion order."""
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        z = dag.add_node(NodeType.VAR, var_index=2)  # 2
        add = dag.add_node(NodeType.ADD)  # 3

        dag.add_edge(x, add)
        dag.add_edge(y, add)
        dag.add_edge(z, add)

        assert dag._input_order[add] == [x, y, z]

    def test_two_level_expression_tree(self) -> None:
        """(x - y) / (x + y): full two-level expression.

        Structure:
            x(0), y(1)
            sub(2): x->sub, y->sub     => _input_order[sub] = [x, y]
            add(3): x->add, y->add     => _input_order[add] = [x, y]
            div(4): sub->div, add->div  => _input_order[div] = [sub, add]
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        sub = dag.add_node(NodeType.SUB)  # 2
        add = dag.add_node(NodeType.ADD)  # 3
        div = dag.add_node(NodeType.DIV)  # 4

        dag.add_edge(x, sub)
        dag.add_edge(y, sub)
        dag.add_edge(x, add)
        dag.add_edge(y, add)
        dag.add_edge(sub, div)
        dag.add_edge(add, div)

        assert dag._input_order[sub] == [x, y]
        assert dag._input_order[add] == [x, y]
        assert dag._input_order[div] == [sub, add]

    def test_diamond_with_backtracking(self) -> None:
        """Diamond pattern built with backtracking.

        Build (x+y) - z, then backtrack the z edge and replace with (x+y) - x.
        Final _input_order[sub] = [add, x].
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        y = dag.add_node(NodeType.VAR, var_index=1)  # 1
        z = dag.add_node(NodeType.VAR, var_index=2)  # 2
        add = dag.add_node(NodeType.ADD)  # 3
        sub = dag.add_node(NodeType.SUB)  # 4

        dag.add_edge(x, add)
        dag.add_edge(y, add)
        dag.add_edge(add, sub)
        dag.add_edge(z, sub)
        assert dag._input_order[sub] == [add, z]

        # Backtrack: remove z -> sub
        dag.remove_edge(z, sub)
        assert dag._input_order[sub] == [add]

        # Replace with x -> sub (creating shared subexpression diamond)
        dag.add_edge(x, sub)
        assert dag._input_order[sub] == [add, x]

    def test_sin_x_minus_cos_x(self) -> None:
        """sin(x) - cos(x): non-commutative with shared variable.

        Structure:
            x(0)
            sin(1): x->sin   => _input_order[sin] = [x]
            cos(2): x->cos   => _input_order[cos] = [x]
            sub(3): sin->sub, cos->sub  => _input_order[sub] = [sin, cos]

        This means sin(x) - cos(x), NOT cos(x) - sin(x).
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        sin = dag.add_node(NodeType.SIN)  # 1
        cos = dag.add_node(NodeType.COS)  # 2
        sub = dag.add_node(NodeType.SUB)  # 3

        dag.add_edge(x, sin)
        dag.add_edge(x, cos)
        dag.add_edge(sin, sub)
        dag.add_edge(cos, sub)

        assert dag._input_order[sin] == [x]
        assert dag._input_order[cos] == [x]
        assert dag._input_order[sub] == [sin, cos]

    def test_cos_x_minus_sin_x_different_order(self) -> None:
        """cos(x) - sin(x): reversed operand order from sin(x) - cos(x).

        Same graph structure but _input_order[sub] = [cos, sin].
        This distinguishes the two expressions semantically.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        sin = dag.add_node(NodeType.SIN)  # 1
        cos = dag.add_node(NodeType.COS)  # 2
        sub = dag.add_node(NodeType.SUB)  # 3

        dag.add_edge(x, sin)
        dag.add_edge(x, cos)
        dag.add_edge(cos, sub)  # cos is first operand
        dag.add_edge(sin, sub)  # sin is second operand

        assert dag._input_order[sub] == [cos, sin]

    def test_x_pow_const(self) -> None:
        """x ^ k: _input_order determines base vs exponent.

        _input_order[pow] = [x, k] means x is the base, k is the exponent.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        k = dag.add_node(NodeType.CONST, const_value=2.0)  # 1
        pw = dag.add_node(NodeType.POW)  # 2

        dag.add_edge(x, pw)
        dag.add_edge(k, pw)

        assert dag._input_order[pw] == [x, k]

    def test_const_pow_x_reversed(self) -> None:
        """k ^ x: reversed from x ^ k.

        _input_order[pow] = [k, x] means k is the base, x is the exponent.
        """
        dag = LabeledDAG(max_nodes=10)
        x = dag.add_node(NodeType.VAR, var_index=0)  # 0
        k = dag.add_node(NodeType.CONST, const_value=2.0)  # 1
        pw = dag.add_node(NodeType.POW)  # 2

        dag.add_edge(k, pw)  # k is first (base)
        dag.add_edge(x, pw)  # x is second (exponent)

        assert dag._input_order[pw] == [k, x]
