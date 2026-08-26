"""Cache entry dataclasses and DAG property helpers.

Defines the frozen dataclasses for individual cache records and aggregate
statistics. Zero external dependencies (stdlib + isalsr.core only).

Reference: docs/design/precomputed_cache_design.md, Section 6.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from isalsr.core.labeled_dag import LabeledDAG

# ======================================================================
# DAG property helpers
# ======================================================================


def dag_depth(dag: LabeledDAG) -> int:
    """Compute the longest path length (in edges) in a DAG.

    Uses topological sort + dynamic programming. O(V + E).

    For a DAG with only VAR nodes (no edges), returns 0.
    For a chain x -> sin -> exp, returns 2.

    Args:
        dag: The labeled DAG.

    Returns:
        The length (in edges) of the longest directed path.
    """
    if dag.node_count == 0:
        return 0

    order = dag.topological_sort()
    depth: list[int] = [0] * dag.node_count

    for node in order:
        for parent in dag.in_neighbors(node):
            candidate = depth[parent] + 1
            if candidate > depth[node]:
                depth[node] = candidate

    return max(depth) if depth else 0


# ======================================================================
# CacheEntry
# ======================================================================


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A single precomputed cache entry for one DAG.

    Stores the output of all four D2S algorithms, DAG structural
    properties, per-algorithm timing, and correctness flags.

    Attributes:
        raw_string: The original IsalSR string (Level 1 lookup key).
        num_variables: Number of input variables (m).
        n_nodes: Total node count in the DAG.
        n_internal: Number of non-VAR nodes (k).
        n_edges: Edge count.
        n_var_nodes: Number of VAR nodes (should equal num_variables).
        depth: Longest path length in the DAG (edges).
        greedy_single: GreedySingleD2S output string.
        greedy_min: GreedyMinD2S output string.
        pruned: PrunedExhaustiveD2S output string (pruned canonical).
        exhaustive: ExhaustiveD2S output string (true canonical), or ""
            if the computation timed out.
        exhaustive_timed_out: True if exhaustive hit the timeout.
        timing_greedy_single: Wall-clock seconds for GreedySingleD2S.
        timing_greedy_min: Wall-clock seconds for GreedyMinD2S.
        timing_pruned: Wall-clock seconds for PrunedExhaustiveD2S.
        timing_exhaustive: Wall-clock seconds for ExhaustiveD2S
            (-1.0 if timed out).
        is_canonical: True if raw_string equals the pruned canonical.
        exhaustive_eq_pruned: True if exhaustive == pruned.
        greedy_single_eq_exhaustive: True if greedy_single == exhaustive.
        greedy_min_eq_exhaustive: True if greedy_min == exhaustive.
    """

    raw_string: str
    num_variables: int
    n_nodes: int
    n_internal: int
    n_edges: int
    n_var_nodes: int
    depth: int
    greedy_single: str
    greedy_min: str
    pruned: str
    exhaustive: str
    exhaustive_timed_out: bool
    timing_greedy_single: float
    timing_greedy_min: float
    timing_pruned: float
    timing_exhaustive: float
    is_canonical: bool
    exhaustive_eq_pruned: bool
    greedy_single_eq_exhaustive: bool
    greedy_min_eq_exhaustive: bool


# ======================================================================
# CacheStats
# ======================================================================


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Aggregate statistics for a cache file.

    Attributes:
        total_entries: Total number of cache entries.
        unique_canonical_pruned: Number of unique pruned canonical strings.
        unique_canonical_exhaustive: Number of unique exhaustive canonical
            strings (excluding timeouts).
        exhaustive_timeout_count: Number of entries where exhaustive timed out.
        exhaustive_eq_pruned_count: Number of entries where exhaustive == pruned.
        greedy_single_eq_exhaustive_count: Entries where greedy_single == exhaustive.
        greedy_min_eq_exhaustive_count: Entries where greedy_min == exhaustive.
        avg_depth: Mean DAG depth across all entries.
        max_depth: Maximum DAG depth across all entries.
        avg_internal_nodes: Mean number of internal nodes.
    """

    total_entries: int
    unique_canonical_pruned: int
    unique_canonical_exhaustive: int
    exhaustive_timeout_count: int
    exhaustive_eq_pruned_count: int
    greedy_single_eq_exhaustive_count: int
    greedy_min_eq_exhaustive_count: int
    avg_depth: float
    max_depth: int
    avg_internal_nodes: float
