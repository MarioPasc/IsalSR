"""Greedy D2S from x_1 only (single starting node).

The simplest and fastest D2S variant. Not a graph invariant.

Restriction: ZERO external dependencies.
"""

from __future__ import annotations

from isalsr.core.algorithms.base import D2SAlgorithm
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG


class GreedySingleD2S(D2SAlgorithm):
    """Greedy DAG-to-string from x_1 (node 0) only."""

    def encode(self, dag: LabeledDAG) -> str:
        """Encode via greedy D2S from x_1."""
        if dag.node_count == 0:
            return ""
        num_vars = len(dag.var_nodes())
        if dag.node_count == num_vars and dag.edge_count == 0:
            return ""
        return DAGToString(dag, initial_node=0).run()

    @property
    def name(self) -> str:
        return "greedy-single"
