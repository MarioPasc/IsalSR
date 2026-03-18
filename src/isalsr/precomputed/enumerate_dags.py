"""Systematic DAG enumeration for cache generation (Phase 1).

Enumerates all valid labeled DAGs with k internal nodes for a given
operator set and variable count. This is the most thorough generation
mode but is only feasible for small k (typically k <= 5-6).

Status: STUB — implement after sampled mode is validated.

Reference: docs/design/precomputed_cache_design.md, Section 2.1.
"""

from __future__ import annotations

from collections.abc import Iterator

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import OperationSet


def enumerate_dags(
    num_variables: int,
    max_internal_nodes: int,
    allowed_ops: OperationSet,
) -> Iterator[LabeledDAG]:
    """Enumerate all valid labeled DAGs up to a given size.

    Args:
        num_variables: Number of input variables (m).
        max_internal_nodes: Maximum number of non-VAR nodes (k_max).
        allowed_ops: Allowed operation set.

    Yields:
        LabeledDAG instances, one per unique DAG skeleton + labeling.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError(
        "Systematic DAG enumeration is not yet implemented. "
        "Use sampled mode (generate_cache.py --mode sampled) instead."
    )
