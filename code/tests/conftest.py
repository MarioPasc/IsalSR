"""Shared test fixtures for IsalSR.

Provides reusable DAGs, expressions, and test data for all test modules.
"""

from __future__ import annotations

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


@pytest.fixture()
def empty_dag() -> LabeledDAG:
    """Empty DAG with capacity 10."""
    return LabeledDAG(max_nodes=10)


@pytest.fixture()
def single_var_dag() -> LabeledDAG:
    """DAG with one VAR node (x_1)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)
    return dag


@pytest.fixture()
def two_var_dag() -> LabeledDAG:
    """DAG with two VAR nodes (x_1, x_2)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    return dag


@pytest.fixture()
def x_plus_y_dag() -> LabeledDAG:
    """Expression DAG for x + y: 3 nodes (x, y, +), 2 edges (x->+, y->+)."""
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # node 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # node 1
    add = dag.add_node(NodeType.ADD)  # node 2
    dag.add_edge(x, add)
    dag.add_edge(y, add)
    return dag


@pytest.fixture()
def sin_x_dag() -> LabeledDAG:
    """Expression DAG for sin(x): 2 nodes (x, sin), 1 edge (x->sin)."""
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # node 0
    sin = dag.add_node(NodeType.SIN)  # node 1
    dag.add_edge(x, sin)
    return dag


@pytest.fixture()
def sin_x_mul_y_dag() -> LabeledDAG:
    """Expression DAG for sin(x) * y: 4 nodes, 3 edges."""
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    sin = dag.add_node(NodeType.SIN)  # 2
    mul = dag.add_node(NodeType.MUL)  # 3
    dag.add_edge(x, sin)
    dag.add_edge(sin, mul)
    dag.add_edge(y, mul)
    return dag
