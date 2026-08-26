"""Greedy D2S from all variable nodes, pick shortest then lexmin.

More robust than single-start but still not a true invariant.

Restriction: ZERO external dependencies.
"""

from __future__ import annotations

from isalsr.core.algorithms.base import D2SAlgorithm
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG


class GreedyMinD2S(D2SAlgorithm):
    """Greedy DAG-to-string from all VAR nodes, pick shortest (lexmin ties)."""

    def encode(self, dag: LabeledDAG) -> str:
        """Encode via greedy D2S from each VAR node, return best."""
        if dag.node_count == 0:
            return ""
        var_nodes = dag.var_nodes()
        if dag.node_count == len(var_nodes) and dag.edge_count == 0:
            return ""

        best: str | None = None
        for v in var_nodes:
            try:
                w = DAGToString(dag, initial_node=v).run()
            except (ValueError, RuntimeError):
                continue
            if best is None or (len(w), w) < (len(best), best):
                best = w

        if best is None:
            raise ValueError("No valid encoding found from any VAR node.")
        return best

    @property
    def name(self) -> str:
        return "greedy-min"
