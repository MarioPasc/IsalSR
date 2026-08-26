"""Shared DAG generators for arXiv benchmark experiments.

Provides controlled random DAG generation with exact internal node counts.
Extracted from pruning_experiment.py for reuse across experiments.
"""

from __future__ import annotations

import random

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


def make_random_sr_dag(
    num_vars: int,
    num_internal: int,
    seed: int,
    *,
    include_pow: bool = False,
) -> LabeledDAG:
    """Generate a random SR expression DAG with exactly num_internal internal nodes.

    Args:
        num_vars: Number of input variables (VAR nodes).
        num_internal: Number of internal (operation) nodes.
        seed: Random seed for reproducibility.
        include_pow: If True, include POW in binary ops.

    Returns:
        A LabeledDAG with num_vars + num_internal nodes.
    """
    rng = random.Random(seed)
    total = num_vars + num_internal
    dag = LabeledDAG(max_nodes=total + 2)

    for i in range(num_vars):
        dag.add_node(NodeType.VAR, var_index=i)

    unary_ops = [NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.LOG, NodeType.ABS]
    binary_ops = [NodeType.ADD, NodeType.MUL, NodeType.SUB, NodeType.DIV]
    if include_pow:
        binary_ops.append(NodeType.POW)

    for _ in range(num_internal):
        existing = list(range(dag.node_count))
        if rng.random() < 0.6 or len(existing) < 2:
            op = rng.choice(unary_ops)
            parent = rng.choice(existing)
            nid = dag.add_node(op)
            dag.add_edge(parent, nid)
        else:
            op = rng.choice(binary_ops)
            parents = rng.sample(existing, 2)
            nid = dag.add_node(op)
            for p in parents:
                dag.add_edge(p, nid)

    return dag
