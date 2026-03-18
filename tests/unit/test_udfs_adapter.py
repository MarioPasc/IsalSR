"""Unit tests for the UDFS CompGraph <-> LabeledDAG adapter."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Add vendored DAG_search to path
_vendor_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "experiments",
    "models",
    "udfs",
    "vendor",
)
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

# Skip if torch not available
torch = pytest.importorskip("torch")

from DAG_search.comp_graph import CompGraph

from experiments.models.udfs.adapter import (
    compgraph_to_labeled_dag,
    labeled_dag_to_compgraph,
)
from isalsr.core.canonical import pruned_canonical_string
from isalsr.core.node_types import NodeType


def _make_compgraph(m, n, k, node_dict):
    """Helper to create a CompGraph."""
    return CompGraph(m, n, k, node_dict=node_dict)


class TestCompGraphToLabeledDAG:
    def test_addition(self):
        """x_0 + x_1"""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 3  # 2 VAR + 1 ADD
        assert dag.edge_count == 2
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.VAR
        assert dag.node_label(2) == NodeType.ADD

    def test_sin(self):
        """sin(x_0)"""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "sin"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 2
        assert dag.node_label(1) == NodeType.SIN

    def test_sub_l(self):
        """x_0 - x_1 (sub_l)"""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "sub_l"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(2) == NodeType.SUB
        # First operand should be x_0 (node 0)
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_sub_r_reverses_children(self):
        """sub_r(x_0, x_1) = x_1 - x_0"""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "sub_r"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(2) == NodeType.SUB
        inputs = dag.ordered_inputs(2)
        assert inputs == [1, 0]  # reversed

    def test_div_l(self):
        """x_0 / x_1 (div_l)"""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "div_l"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(2) == NodeType.DIV
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_div_r_reverses_children(self):
        """div_r(x_0, x_1) = x_1 / x_0"""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "div_r"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        inputs = dag.ordered_inputs(2)
        assert inputs == [1, 0]

    def test_identity_collapse(self):
        """Identity nodes ('=') should be collapsed."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "sin"),
                2: ((1,), "="),  # output identity
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        # Only 2 nodes: VAR and SIN (identity collapsed)
        assert dag.node_count == 2

    def test_const_node(self):
        """x_0 + c_0"""
        cg = _make_compgraph(
            1,
            1,
            1,
            {
                0: ((), "="),  # x_0
                1: ((), "="),  # c_0
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 3
        assert dag.node_label(1) == NodeType.CONST
        # CONST should have creation edge from node 0 (x_0)
        assert dag.in_degree(1) >= 1

    def test_neg(self):
        """neg(x_0) = -x_0"""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "neg"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(1) == NodeType.NEG

    def test_inv(self):
        """inv(x_0) = 1/x_0"""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "inv"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(1) == NodeType.INV

    def test_canonical_string_consistency(self):
        """Two isomorphic CompGraphs should produce the same canonical string."""
        # x_0 + x_1 (order 1)
        cg1 = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        # x_0 + x_1 (same graph, just different build)
        cg2 = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((1, 0), "+"),
                3: ((2,), "="),
            },
        )
        dag1 = compgraph_to_labeled_dag(cg1)
        dag2 = compgraph_to_labeled_dag(cg2)
        cs1 = pruned_canonical_string(dag1)
        cs2 = pruned_canonical_string(dag2)
        # x_0+x_1 and x_1+x_0 should be isomorphic for commutative ADD
        assert cs1 == cs2


class TestLabeledDAGToCompGraph:
    def test_roundtrip_simple(self):
        """CompGraph → LabeledDAG → CompGraph roundtrip."""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        cg2 = labeled_dag_to_compgraph(dag)
        assert cg2.inp_dim == 2
        assert cg2.outp_dim == 1
        assert cg2.n_consts == 0
        assert cg2.n_nodes() >= 3  # 2 inputs + 1 op + 1 output

    def test_roundtrip_with_const(self):
        """Round-trip with a constant node."""
        cg = _make_compgraph(
            1,
            1,
            1,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        cg2 = labeled_dag_to_compgraph(dag)
        assert cg2.n_consts == 1
