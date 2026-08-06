"""Regression tests for the k=0 structural-scope guard (D3, 2026-08-06).

The defect these lock down: every zero-internal-node DAG canonicalises to ``""``,
so ``f(x) = x_0`` and ``f(x) = x_1`` shared a deduplication / fitness-cache key.
Bingo transferred the cached fitness between them; UDFS would have skipped
evaluating the second outright.

Found by the T04 Mode 1 replay on the Stage D trace stream before the campaign
committed any core-hours (EXECUTION-PLAN §4.4 D3).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.models.structural_scope import (
    STRUCTURAL_SCOPE_REASON,
    count_internal_nodes,
    is_structural,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


def _vars_only(n_vars: int) -> LabeledDAG:
    """A DAG of ``n_vars`` isolated VAR nodes: the initial state, k = 0."""
    dag = LabeledDAG(max_nodes=max(n_vars, 4))
    for _ in range(n_vars):
        dag.add_node(NodeType.VAR)
    return dag


def _var_plus_op(op: NodeType = NodeType.SIN) -> LabeledDAG:
    """A DAG with one internal node: k = 1."""
    dag = LabeledDAG(max_nodes=4)
    v = dag.add_node(NodeType.VAR)
    o = dag.add_node(op)
    dag.add_edge(v, o)
    return dag


class TestTheDefectItself:
    """Characterise the collision, so the guard's reason stays visible."""

    @pytest.mark.parametrize("n_vars", [1, 2, 3, 5])
    def test_every_zero_internal_node_dag_canonicalises_to_the_empty_string(
        self, n_vars: int
    ) -> None:
        assert fast_canonical_string(_vars_only(n_vars)) == ""

    def test_dags_of_different_size_are_indistinguishable_at_k_zero(self) -> None:
        """This is why '' must never be a dedup key: it does not encode m."""
        assert fast_canonical_string(_vars_only(1)) == fast_canonical_string(_vars_only(2))

    def test_a_bare_constant_is_structural_and_stays_distinguishable(self) -> None:
        """CONST is created by ``Vk``, so it IS encoded -- k = 1, not k = 0.

        The creation edge is mandatory: invariant 9 makes CONST in-degree-0
        repair a PRODUCER-side step, and the canonicaliser raises rather than
        assuming it (that is what keeps ``fcs`` a pure function of the DAG).
        The host adapters apply ``_normalize_const_edges`` for exactly this.
        """
        dag = LabeledDAG(max_nodes=4)
        v = dag.add_node(NodeType.VAR)
        c = dag.add_node(NodeType.CONST)
        dag.add_edge(v, c)
        assert fast_canonical_string(dag) != ""
        assert is_structural(dag)


class TestPredicate:
    @pytest.mark.parametrize("n_vars", [1, 2, 3, 5])
    def test_bare_variables_are_not_structural(self, n_vars: int) -> None:
        assert not is_structural(_vars_only(n_vars))
        assert count_internal_nodes(_vars_only(n_vars)) == 0

    def test_one_operator_is_structural(self) -> None:
        assert is_structural(_var_plus_op())
        assert count_internal_nodes(_var_plus_op()) == 1

    def test_the_empty_dag_is_not_structural(self) -> None:
        assert not is_structural(LabeledDAG(max_nodes=4))

    def test_reason_string_is_stable(self) -> None:
        """The trace records it; changing it silently breaks stream analysis."""
        assert STRUCTURAL_SCOPE_REASON == "k0_nonstructural"


class TestRhoAccounting:
    """rho must be a ratio over ONE population.

    Regression for the bias found in Stage C v5b: ``n_total`` is incremented
    when a candidate is first seen, several steps before the DAG exists to be
    classified.  The first version of the guard left k=0 candidates in
    ``n_total`` while excluding them from ``n_unique``, so
    ``rho = n_total / n_unique`` mixed two populations and came out **12.15 %**
    too high on bingo/hash and **13.86 %** on bingo/isalsr -- in the direction
    that flatters IsalSR.
    """

    def test_recording_a_nonstructural_candidate_undoes_n_total(self) -> None:
        pytest.importorskip("bingo")
        from experiments.models.bingo.isalsr_runner import _CanonicalDeduplicator

        d = _CanonicalDeduplicator()
        d.n_total = 10
        d.n_unique = 4

        d.record_nonstructural()

        assert d.n_nonstructural == 1
        assert d.n_total == 9, (
            "n_total must exclude the k=0 candidate, or rho is a ratio over "
            "two different populations"
        )

    def test_candidates_seen_stays_recoverable(self) -> None:
        pytest.importorskip("bingo")
        from experiments.models.bingo.isalsr_runner import _CanonicalDeduplicator

        d = _CanonicalDeduplicator()
        seen = 0
        for i in range(20):
            d.n_total += 1
            seen += 1
            if i % 4 == 0:
                d.record_nonstructural()

        assert d.n_total + d.n_nonstructural == seen, (
            "total_dags_explored + n_nonstructural must reconstruct the number "
            "of candidates the host actually produced"
        )
        assert d.n_nonstructural == 5
        assert d.n_total == 15


class TestAdapterLevelCollision:
    """The end-to-end statement, on the real Bingo adapter."""

    def test_x0_and_x1_share_a_canonical_string_but_not_semantics(self) -> None:
        pytest.importorskip("bingo")
        from bingo.symbolic_regression.agraph.agraph import AGraph

        from experiments.models.bingo.adapter import agraph_to_labeled_dag

        def build(cmd: list[tuple[int, int, int]]) -> tuple[str, str, LabeledDAG]:
            g = AGraph()
            g.command_array = np.array(cmd, dtype=int)
            dag = agraph_to_labeled_dag(g)
            return str(g), fast_canonical_string(dag), dag

        expr0, canon0, dag0 = build([(0, 0, 0)])
        expr1, canon1, dag1 = build([(0, 1, 1)])

        assert expr0 != expr1, "the two candidates must be different functions"
        assert canon0 == canon1 == "", "and they do share the empty canonical key"

        # Which is exactly why both must be excluded from dedup.
        assert not is_structural(dag0)
        assert not is_structural(dag1)
