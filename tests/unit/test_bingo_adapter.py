"""Unit tests for the Bingo AGraph <-> LabeledDAG adapter."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

bingo = pytest.importorskip("bingo")

from bingo.symbolic_regression.agraph.agraph import AGraph  # noqa: E402

from experiments.models.bingo.adapter import (  # noqa: E402
    agraph_to_labeled_dag,
    labeled_dag_to_agraph,
)
from isalsr.core.canonical import pruned_canonical_string  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402


def _make_agraph(cmd):
    """Helper to create an AGraph from a command array."""
    ag = AGraph(use_simplification=False)
    ag._command_array = np.array(cmd, dtype=int)
    ag._notify_modification()
    return ag


class TestAGraphToLabeledDAG:
    def test_addition(self):
        """x_0 + x_1"""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.VAR
        assert dag.node_label(2) == NodeType.ADD

    def test_sin(self):
        """sin(x_0)"""
        ag = _make_agraph([[0, 0, 0], [6, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 2
        assert dag.node_label(1) == NodeType.SIN

    def test_subtraction_operand_order(self):
        """x_0 - x_1: first operand = param1 = x_0"""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(2) == NodeType.SUB
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_subtraction_reversed(self):
        """x_1 - x_0: first operand = param1 = x_1"""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 1, 0]])
        dag = agraph_to_labeled_dag(ag)
        inputs = dag.ordered_inputs(2)
        assert inputs == [1, 0]

    def test_different_subtraction_canonical_strings(self):
        """x-y and y-x must have different canonical strings."""
        ag1 = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        ag2 = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 1, 0]])
        cs1 = pruned_canonical_string(agraph_to_labeled_dag(ag1))
        cs2 = pruned_canonical_string(agraph_to_labeled_dag(ag2))
        assert cs1 != cs2

    def test_division(self):
        """x_0 / x_1"""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [5, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(2) == NodeType.DIV
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_const_node(self):
        """c + x_0"""
        ag = _make_agraph([[0, 0, 0], [1, 0, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 3
        assert dag.node_label(1) == NodeType.CONST
        assert dag.in_degree(1) >= 1  # CONST has creation edge

    def test_unused_rows_filtered(self):
        """Only utilized rows should be converted."""
        # Row 0: x_0, Row 1: x_1 (unused), Row 2: x_0*x_0 (unused),
        # Row 3: x_0+x_0 (unused), Row 4: sin(x_0) (output)
        ag = _make_agraph(
            [
                [0, 0, 0],
                [0, 1, 0],
                [4, 0, 1],
                [2, 0, 2],
                [6, 0, 0],
            ]
        )
        utilized = ag.get_utilized_commands()
        dag = agraph_to_labeled_dag(ag)
        cs = pruned_canonical_string(dag)
        assert cs == "Vs"  # sin(x_0)

    def test_variable_deduplication(self):
        """Multiple rows referencing same variable produce one VAR node."""
        ag = _make_agraph([[0, 0, 0], [0, 0, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        # Only 1 VAR node (both rows reference x_0)
        var_count = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)
        assert var_count == 1

    def test_canonical_isomorphism(self):
        """Isomorphic AGraphs produce the same canonical string."""
        # x_0 + x_1
        ag1 = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        # x_1 + x_0 (addition is commutative)
        ag2 = _make_agraph([[0, 1, 0], [0, 0, 0], [2, 0, 1]])
        cs1 = pruned_canonical_string(agraph_to_labeled_dag(ag1))
        cs2 = pruned_canonical_string(agraph_to_labeled_dag(ag2))
        assert cs1 == cs2

    def test_exp(self):
        """exp(x_0)"""
        ag = _make_agraph([[0, 0, 0], [8, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.EXP

    def test_log(self):
        """log(x_0)"""
        ag = _make_agraph([[0, 0, 0], [9, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.LOG

    def test_cos(self):
        """cos(x_0)"""
        ag = _make_agraph([[0, 0, 0], [7, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.COS


class TestLabeledDAGToAGraph:
    def test_roundtrip_addition(self):
        """AGraph → LabeledDAG → AGraph roundtrip."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2)

    def test_roundtrip_sin(self):
        """sin(x_0) roundtrip."""
        ag = _make_agraph([[0, 0, 0], [6, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[0.5], [1.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-10)
