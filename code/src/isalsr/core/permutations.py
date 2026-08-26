"""Permutation of internal nodes in labeled DAGs.

Given a LabeledDAG with m variable nodes and k internal nodes,
creates an isomorphic copy with internal nodes renumbered according
to a permutation. Variables remain fixed (indices 0..m-1).

Invariant: permute_internal_nodes(D, pi) is isomorphic to D for all
valid permutations pi.

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


def permute_internal_nodes(
    dag: LabeledDAG,
    perm: Sequence[int],
) -> LabeledDAG:
    """Create an isomorphic copy of *dag* with permuted internal node IDs.

    Variables (nodes with label VAR) retain their original positions
    (0..m-1). Internal nodes are remapped: the node at original
    position m+i is placed at new position m+perm[i].

    The mapping phi is:
        phi(v) = v              if v < m  (variable, fixed)
        phi(m+i) = m + perm[i]  for i in 0..k-1  (internal, permuted)

    Args:
        dag: Source labeled DAG with m VAR nodes at positions 0..m-1.
        perm: Permutation of range(k), where k = dag.node_count - m.
            perm[i] = j means original internal node m+i maps to m+j.

    Returns:
        A new LabeledDAG isomorphic to *dag* with renumbered internals.

    Raises:
        ValueError: If perm is not a valid permutation of range(k).
    """
    var_nodes = dag.var_nodes()
    m = len(var_nodes)
    n = dag.node_count
    k = n - m

    # Validate permutation.
    perm_list = list(perm)
    if sorted(perm_list) != list(range(k)):
        raise ValueError(f"perm must be a permutation of range({k}), got {perm_list}")

    # Build forward and inverse mappings.
    # phi: old_node -> new_node
    phi: list[int] = list(range(n))  # identity for variables
    for i in range(k):
        phi[m + i] = m + perm_list[i]

    # inv_phi: new_node -> old_node
    inv_phi: list[int] = list(range(n))  # identity for variables
    for i in range(k):
        inv_phi[m + perm_list[i]] = m + i

    # Create new DAG.
    new_dag = LabeledDAG(max_nodes=n)

    # Add nodes in new-ID order (0, 1, ..., n-1).
    # LabeledDAG.add_node() assigns IDs sequentially, so the j-th call
    # to add_node() creates node with ID j.
    for new_id in range(n):
        old_id = inv_phi[new_id]
        label = dag.node_label(old_id)
        data = dag.node_data(old_id)

        if label == NodeType.VAR:
            var_idx: int | None = data.get("var_index")  # type: ignore
            new_dag.add_node(label, var_index=var_idx)
        elif label == NodeType.CONST:
            new_dag.add_node(label, const_value=data.get("const_value"))
        else:
            new_dag.add_node(label)

    # Add edges, preserving operand order (critical for SUB/DIV/POW).
    # For each target node, iterate ordered_inputs in the original DAG
    # and add edges in the same order via the permutation mapping.
    for new_target in range(n):
        old_target = inv_phi[new_target]
        for old_source in dag.ordered_inputs(old_target):
            new_source = phi[old_source]
            new_dag.add_edge_unchecked(new_source, new_target)

    return new_dag


def random_permutations(
    k: int,
    n_samples: int,
    rng: random.Random,
) -> list[list[int]]:
    """Generate n_samples random permutations of range(k).

    Uses Fisher-Yates shuffle for uniform sampling.

    Args:
        k: Permutation size.
        n_samples: Number of permutations to generate.
        rng: Random number generator for reproducibility.

    Returns:
        List of permutations (each a list of ints).
    """
    result: list[list[int]] = []
    base = list(range(k))
    for _ in range(n_samples):
        p = base.copy()
        # Fisher-Yates shuffle
        for i in range(k - 1, 0, -1):
            j = rng.randint(0, i)
            p[i], p[j] = p[j], p[i]
        result.append(p)
    return result
