"""Tests for the naive fixed-order-hash search variant (``hash`` arm).

The ``hash`` arm answers reviewer comment R1.4: it is identical to the
``isalsr`` arm in every respect except the equivalence relation used to
decide whether a candidate DAG has been seen before.

- ``isalsr``: ``hash(fast_canonical_string(dag))`` -- complete invariant.
- ``hash``:   ``fixed_order_hash(dag, FixedOrder.TOPOLOGICAL)`` -- sound but
  incomplete.

These tests pin the wiring (factory, output tree, schema fields) and the key
parity between the arm's deduplicator and the reference implementation in
:mod:`isalsr.baselines.fixed_order_hash`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.models.io_utils import ensure_output_structure
from experiments.models.orchestrator import create_runner
from experiments.models.schemas import SearchSpaceResults
from isalsr.baselines import FixedOrder, fixed_order_hash, serialise
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


def _two_var_dag() -> LabeledDAG:
    """Return ``sin(x_0) + x_1`` as a labeled DAG."""
    dag = LabeledDAG(16)
    x0 = dag.add_node(NodeType.VAR, var_index=0)
    x1 = dag.add_node(NodeType.VAR, var_index=1)
    s = dag.add_node(NodeType.SIN)
    a = dag.add_node(NodeType.ADD)
    dag.add_edge(x0, s)
    dag.add_edge(s, a)
    dag.add_edge(x1, a)
    return dag


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_create_runner_accepts_hash_variant(method: str) -> None:
    """The runner factory must build a ``hash`` arm for both hosts."""
    runner = create_runner(method, "hash", {})
    assert runner.variant == "hash"
    assert runner.name == method


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_hash_runner_ignores_atlas(method: str) -> None:
    """The atlas stores canonical hashes; the hash arm must not consult it."""
    runner = create_runner(method, "hash", {}, atlas=object())
    assert runner._atlas is None


def test_create_runner_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        create_runner("bingo", "not_a_variant", {})


def test_output_structure_is_variant_driven(tmp_path: Path) -> None:
    """``ensure_output_structure`` must create a subdir per requested arm."""
    paths = ensure_output_structure(
        tmp_path,
        "bingo",
        "nguyen",
        "Nguyen-1",
        variants=["baseline", "isalsr", "hash"],
    )
    assert set(paths) >= {"problem", "baseline", "isalsr", "hash"}
    for key in ("baseline", "isalsr", "hash"):
        assert paths[key].is_dir()


def test_output_structure_keeps_legacy_arms(tmp_path: Path) -> None:
    """Existing result trees must still resolve when only two arms are asked for."""
    paths = ensure_output_structure(tmp_path, "udfs", "nguyen", "Nguyen-1")
    assert set(paths) >= {"problem", "baseline", "isalsr"}


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------


def test_search_space_results_carries_shadow_fields() -> None:
    """The three shadow cardinality estimates must round-trip through the schema."""
    ss = SearchSpaceResults(
        total_dags_explored=10,
        unique_canonical_dags=4,
        empirical_reduction_factor=2.5,
        max_internal_nodes_seen=3,
        theoretical_reduction_bound=6.0,
        redundancy_rate=0.6,
        shadow_distinct_insertion=8.0,
        shadow_distinct_topological=7.0,
        shadow_distinct_topological_commutative=6.0,
    )
    d = ss.__dict__ if not hasattr(ss, "_asdict") else dict(ss._asdict())
    assert d["shadow_distinct_insertion"] == pytest.approx(8.0)
    assert d["shadow_distinct_topological"] == pytest.approx(7.0)
    assert d["shadow_distinct_topological_commutative"] == pytest.approx(6.0)


def test_search_space_shadow_fields_default_to_none() -> None:
    """Arms that do not run shadow counters must still construct."""
    ss = SearchSpaceResults(
        total_dags_explored=1,
        unique_canonical_dags=1,
        empirical_reduction_factor=1.0,
        max_internal_nodes_seen=0,
        theoretical_reduction_bound=1.0,
        redundancy_rate=0.0,
    )
    assert ss.shadow_distinct_insertion is None


# ----------------------------------------------------------------------
# Key parity: the deduplicator must produce exactly the reference key
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_dedup_key_matches_reference_fixed_order_hash(module_name: str) -> None:
    """``key_mode='hash'`` must key on ``fixed_order_hash(dag, TOPOLOGICAL)``."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(key_mode="hash")
    dag = _two_var_dag()
    rep = dedup.representation_string(dag)
    assert rep == serialise(dag, FixedOrder.TOPOLOGICAL)
    assert hash(rep) == fixed_order_hash(dag, FixedOrder.TOPOLOGICAL)


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_dedup_key_mode_canonical_is_default(module_name: str) -> None:
    """The default arm must remain the canonical string, unchanged."""
    from isalsr.core.canonical import fast_canonical_string

    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator()
    assert dedup.key_mode == "canonical"
    dag = _two_var_dag()
    assert dedup.representation_string(dag) == fast_canonical_string(dag, timeout=dedup.timeout)


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_dedup_rejects_unknown_key_mode(module_name: str) -> None:
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    with pytest.raises(ValueError, match="key_mode"):
        mod._CanonicalDeduplicator(key_mode="bogus")


# ----------------------------------------------------------------------
# Shadow counters
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_shadow_counters_are_off_by_default(module_name: str) -> None:
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator()
    assert dedup.shadow_counts() == {}


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_shadow_counters_track_all_three_orders(module_name: str) -> None:
    """Feeding N pairwise-distinct DAGs must give an estimate near N for each order."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)

    dags = []
    for n_extra in range(5):
        dag = LabeledDAG(16)
        prev = dag.add_node(NodeType.VAR, var_index=0)
        for _ in range(n_extra + 1):
            nxt = dag.add_node(NodeType.SIN)
            dag.add_edge(prev, nxt)
            prev = nxt
        dags.append(dag)

    for dag in dags:
        dedup.record_shadow(dag)
        dedup.record_shadow(dag)  # duplicates must not inflate the estimate

    counts = dedup.shadow_counts()
    assert set(counts) == {
        "shadow_distinct_insertion",
        "shadow_distinct_topological",
        "shadow_distinct_topological_commutative",
    }
    for key, value in counts.items():
        assert value == pytest.approx(len(dags), rel=0.05), key


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_shadow_memory_is_constant_in_stream_length(module_name: str) -> None:
    """The sketch must not grow with the number of candidates fed to it."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    reg = dedup._shadow[FixedOrder.TOPOLOGICAL]._registers  # type: ignore[index]
    before = len(reg)
    dag = _two_var_dag()
    for _ in range(2000):
        dedup.record_shadow(dag)
    assert len(dedup._shadow[FixedOrder.TOPOLOGICAL]._registers) == before  # type: ignore[index]


# ----------------------------------------------------------------------
# Soundness / incompleteness of the hash arm relative to the canonical arm
# ----------------------------------------------------------------------


def test_fixed_order_key_is_incomplete_where_canonical_is_complete() -> None:
    """Two isomorphic DAGs with different numbering: canonical merges, hash may not.

    This is the scientific content of the arm -- it must be measurable, not
    assumed. The assertion is one-sided: the canonical key MUST merge them.
    """
    from isalsr.core.canonical import fast_canonical_string

    # sin(x_0) * cos(x_1), built in two different node orders.
    a = LabeledDAG(16)
    ax0 = a.add_node(NodeType.VAR, var_index=0)
    ax1 = a.add_node(NodeType.VAR, var_index=1)
    a_s = a.add_node(NodeType.SIN)
    a_c = a.add_node(NodeType.COS)
    a_m = a.add_node(NodeType.MUL)
    a.add_edge(ax0, a_s)
    a.add_edge(ax1, a_c)
    a.add_edge(a_s, a_m)
    a.add_edge(a_c, a_m)

    b = LabeledDAG(16)
    bx0 = b.add_node(NodeType.VAR, var_index=0)
    bx1 = b.add_node(NodeType.VAR, var_index=1)
    b_c = b.add_node(NodeType.COS)
    b_s = b.add_node(NodeType.SIN)
    b_m = b.add_node(NodeType.MUL)
    b.add_edge(bx0, b_s)
    b.add_edge(bx1, b_c)
    b.add_edge(b_c, b_m)
    b.add_edge(b_s, b_m)

    assert fast_canonical_string(a) == fast_canonical_string(b)
    # The fixed-order keys are permitted to differ; soundness only requires
    # that a shared key implies isomorphism, never the converse.
    assert isinstance(fixed_order_hash(a, FixedOrder.TOPOLOGICAL), int)
    assert isinstance(fixed_order_hash(b, FixedOrder.TOPOLOGICAL), int)
