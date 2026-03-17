"""Integration tests for NetworkX adapter.

Requires: networkx >= 3.0
"""

from __future__ import annotations

nx = __import__("pytest").importorskip("networkx")

from isalsr.adapters.networkx_adapter import NetworkXAdapter
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


class TestNetworkXRoundTrip:
    """DAG → nx.DiGraph → DAG round-trip."""

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        adapter = NetworkXAdapter()
        g = adapter.to_external(sin_x_dag)
        dag2 = adapter.from_external(g)
        assert sin_x_dag.is_isomorphic(dag2)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        adapter = NetworkXAdapter()
        g = adapter.to_external(x_plus_y_dag)
        dag2 = adapter.from_external(g)
        assert x_plus_y_dag.is_isomorphic(dag2)


class TestNetworkXAttributes:
    """Node attributes preserved in conversion."""

    def test_labels_preserved(self, sin_x_dag: LabeledDAG) -> None:
        adapter = NetworkXAdapter()
        g = adapter.to_external(sin_x_dag)
        assert g.nodes[0]["label"] == "VAR"
        assert g.nodes[1]["label"] == "SIN"

    def test_var_index_preserved(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        adapter = NetworkXAdapter()
        g = adapter.to_external(dag)
        assert g.nodes[0]["var_index"] == 0
        assert g.nodes[1]["var_index"] == 1

    def test_const_value_preserved(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=3.14)
        dag.add_edge(0, 1)
        adapter = NetworkXAdapter()
        g = adapter.to_external(dag)
        assert g.nodes[1]["const_value"] == 3.14

    def test_edge_direction(self, sin_x_dag: LabeledDAG) -> None:
        adapter = NetworkXAdapter()
        g = adapter.to_external(sin_x_dag)
        assert g.has_edge(0, 1)
        assert not g.has_edge(1, 0)
