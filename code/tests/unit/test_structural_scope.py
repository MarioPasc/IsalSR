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
    NONSTRUCTURAL_KEY_PREFIX,
    STRUCTURAL_SCOPE_REASON,
    count_internal_nodes,
    is_structural,
    nonstructural_key,
    recorded_key,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


def _vars_only(n_vars: int) -> LabeledDAG:
    """A DAG of ``n_vars`` isolated VAR nodes: the initial state, k = 0.

    ``var_index`` is set because the host adapters set it and
    :func:`isalsr.baselines.serialise` requires it -- a VAR node without one is
    not a shape any adapter emits.
    """
    dag = LabeledDAG(max_nodes=max(n_vars, 4))
    for i in range(n_vars):
        node = dag.add_node(NodeType.VAR)
        dag.node_data(node)["var_index"] = i
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


class TestNonstructuralKey:
    """k=0 candidates get a SOUND key, and stay in the rho accounting.

    Two errors are locked out here, and they pull in opposite directions:

    * Keying k=0 on the canonical string equates ``x_0`` with ``x_1`` -- a
      search-correctness defect (the fitness cache hands the second the first's
      fitness; UDFS skips it entirely).
    * *Excluding* k=0 from the accounting to avoid that is an over-correction.
      Bare-variable candidates are ordinary redundancy -- on a one-variable
      problem every one is literally the same DAG -- and dropping them
      understated rho by 12.2 % on Stage C v5b.
    """

    def test_key_separates_dags_the_canonical_string_equates(self) -> None:
        one, two = _vars_only(1), _vars_only(2)
        assert fast_canonical_string(one) == fast_canonical_string(two) == ""
        assert nonstructural_key(one) != nonstructural_key(two), (
            "the k=0 key must distinguish what the canonical string cannot, "
            "or the fitness cache stays corruptible"
        )

    def test_identical_k0_dags_still_share_a_key(self) -> None:
        """The m=1 case: all bare-variable candidates ARE the same DAG."""
        assert nonstructural_key(_vars_only(1)) == nonstructural_key(_vars_only(1))

    def test_key_cannot_collide_with_a_canonical_string(self) -> None:
        """No Sigma_SR word begins with '#', so the namespaces are disjoint."""
        assert nonstructural_key(_vars_only(2)).startswith(NONSTRUCTURAL_KEY_PREFIX)
        assert NONSTRUCTURAL_KEY_PREFIX.startswith("#")
        for dag in (_var_plus_op(), _var_plus_op(NodeType.COS)):
            assert not fast_canonical_string(dag).startswith("#")

    def test_n_total_is_not_decremented_for_k0(self) -> None:
        """The over-correction regression.

        k=0 candidates must remain in ``n_total``: they are real candidates that
        the deduplication really does collapse. Removing them made rho 12.2 %
        too low -- a self-inflicted penalty. ``_CanonicalDeduplicator`` must
        therefore expose no method that walks ``n_total`` backwards.
        """
        pytest.importorskip("bingo")
        from experiments.models.bingo.isalsr_runner import _CanonicalDeduplicator

        assert not hasattr(_CanonicalDeduplicator(), "record_nonstructural"), (
            "record_nonstructural decremented n_total; k=0 candidates belong "
            "in the rho accounting under a sound key instead"
        )


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


class TestRecordedKey:
    """``recorded_key`` reproduces what the production runners persist.

    The D3 verifiers (``stage_d_trace._spot_check_one`` and
    ``stage_d_mode1_replay._cross_check``) re-derive the key of a traced
    candidate and compare it byte-exact against the recorded value.  Both
    runners substitute :func:`nonstructural_key` at k=0, so a verifier that
    re-canonicalises unconditionally reports a mismatch on every bare-variable
    record -- and reports it as an *engine* disagreement, which it is not:
    both engines return ``""`` there.  Regression for that false alarm.
    """

    def test_structural_dag_keeps_its_canonical_string(self) -> None:
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)
        node = dag.add_node(NodeType.SIN)
        dag.add_edge(0, node)
        canonical = fast_canonical_string(dag)

        assert canonical != ""
        assert recorded_key(dag, canonical) == canonical

    def test_bare_variable_gets_the_substitution_not_the_empty_string(self) -> None:
        dag = _vars_only(1)
        canonical = fast_canonical_string(dag)

        assert canonical == ""
        assert recorded_key(dag, canonical) == nonstructural_key(dag)
        assert recorded_key(dag, canonical).startswith(NONSTRUCTURAL_KEY_PREFIX)

    def test_distinct_bare_variables_get_distinct_keys(self) -> None:
        """The whole point of the substitution: x_0 and x_0,x_1 must not merge."""
        one, two = _vars_only(1), _vars_only(2)

        assert fast_canonical_string(one) == fast_canonical_string(two) == ""
        assert recorded_key(one, "") != recorded_key(two, "")

    def test_both_engines_agree_at_k_zero(self) -> None:
        """The k=0 key is engine-independent, so no D3 verifier may flag it."""
        pytest.importorskip("isalsr.core._native", reason="C++ extension not built")
        dag = _vars_only(1)

        cpp = fast_canonical_string(dag, backend="cpp")
        python = fast_canonical_string(dag, backend="python")

        assert cpp == python == ""
        assert recorded_key(dag, cpp) == recorded_key(dag, python)
