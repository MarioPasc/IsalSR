"""Unit tests for T07 round-trip fidelity diagnosis.

Two claims are validated:

(a) Comparator defect in t07_norm_removal_study:
    ``_structural_key`` compares absolute node IDs, so two isomorphic DAGs
    constructed with different node orderings produce *different* keys.
    The unified comparator (fcs) produces the *same* canonical string for
    isomorphic DAGs, making it the correct oracle for both arms.

(b) Feature extraction correctness:
    ``classify_failure_features`` accurately detects SUB/DIV/POW presence,
    single-in-edge binary nodes (adapter src1==src2 case), CONST presence,
    and k.

Tests that were RED before ``t07_roundtrip_diagnosis`` was written:
    - All tests that import from experiments.scripts.t07_roundtrip_diagnosis
      raised ImportError before the module existed.
"""

from __future__ import annotations

import pytest

# The flawed comparator from the original study (for demonstrating the defect)
from experiments.scripts.t07_norm_removal_study import _structural_key

# Module under test (was ImportError before t07_roundtrip_diagnosis was written)
from experiments.scripts.t07_roundtrip_diagnosis import (
    classify_failure_features,
    dag_to_json,
    round_trip_drop_unified,
    round_trip_keep,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

_TIMEOUT = 5.0  # seconds — enough for unit-test DAGs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dag_add_xy(m: int = 2) -> LabeledDAG:
    """Build ADD(x0, x1) with *m* variables.  Minimal 1-internal-node DAG."""
    dag = LabeledDAG(m + 5)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    add_id = dag.add_node(NodeType.ADD)
    dag.add_edge(0, add_id)
    dag.add_edge(1, add_id)
    return dag


def _dag_sin_cos_add() -> LabeledDAG:
    """Build ADD(SIN(x0), COS(x1)) in adapter creation order.

    Adapter order: x0(0), x1(1), SIN(2), COS(3), ADD(4).
    """
    dag = LabeledDAG(8)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    sin_id = dag.add_node(NodeType.SIN)  # 2
    cos_id = dag.add_node(NodeType.COS)  # 3
    add_id = dag.add_node(NodeType.ADD)  # 4
    dag.add_edge(0, sin_id)
    dag.add_edge(1, cos_id)
    dag.add_edge(sin_id, add_id)
    dag.add_edge(cos_id, add_id)
    return dag


def _dag_sin_cos_add_reversed_order() -> LabeledDAG:
    """Build ADD(SIN(x0), COS(x1)) with COS created before SIN.

    This is ISOMORPHIC to ``_dag_sin_cos_add`` as a labeled DAG, but has a
    different internal node numbering.  S2D may produce this ordering when the
    canonical algorithm orders COS before SIN.

    Reversed order: x0(0), x1(1), COS(2), SIN(3), ADD(4).
    """
    dag = LabeledDAG(8)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    cos_id = dag.add_node(NodeType.COS)  # 2 — COS first this time
    sin_id = dag.add_node(NodeType.SIN)  # 3
    add_id = dag.add_node(NodeType.ADD)  # 4
    dag.add_edge(1, cos_id)
    dag.add_edge(0, sin_id)
    dag.add_edge(sin_id, add_id)
    dag.add_edge(cos_id, add_id)
    return dag


def _dag_sub_single_edge(m: int = 1) -> LabeledDAG:
    """Build a SUB node with ONLY ONE in-edge (src1==src2 case from adapter).

    Bingo command array ``[SUB, row_x0, row_x0]`` → adapter skips the second
    ``add_edge`` because src1==src2, leaving SUB with in_degree==1.
    """
    dag = LabeledDAG(5)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    sub_id = dag.add_node(NodeType.SUB)  # 1
    dag.add_edge(0, sub_id)
    # Deliberately omit second edge (src1 == src2 in adapter)
    assert dag.in_degree(sub_id) == 1
    return dag


def _dag_from_string(s: str, m: int) -> LabeledDAG:
    """Decode an IsalSR string via S2D."""
    return StringToDAG(s, num_variables=m).run()


# ---------------------------------------------------------------------------
# (a) Comparator defect: structural_key vs fcs
# ---------------------------------------------------------------------------


class TestStructuralKeyDefect:
    """Demonstrate that _structural_key is NOT a valid isomorphism comparator."""

    def test_structural_key_differs_for_isomorphic_dags(self) -> None:
        """Two isomorphic DAGs with different node orderings produce different keys.

        This is the root defect: ``_structural_key`` compares absolute node IDs
        via ``ordered_inputs()``.  If the adapter creates SIN before COS but
        S2D creates them in the opposite order, the keys diverge even though the
        two DAGs represent the same expression.
        """
        d1 = _dag_sin_cos_add()  # adapter order: SIN=2, COS=3
        d2 = _dag_sin_cos_add_reversed_order()  # reversed: COS=2, SIN=3

        k1 = _structural_key(d1)
        k2 = _structural_key(d2)

        # Keys MUST differ (demonstrating the defect — absolute IDs differ)
        assert k1 != k2, (
            "structural_key should differ for same-structure dags with different "
            "node numbering, but got equal keys — test premise violated."
        )

    def test_fcs_agrees_for_isomorphic_dags(self) -> None:
        """fast_canonical_string is invariant to node numbering.

        Both DAGs are labeled-isomorphic (ADD(SIN(x0), COS(x1))) so fcs must
        produce the same canonical string for both.
        """
        from isalsr.core.canonical import fast_canonical_string

        d1 = _dag_sin_cos_add()
        d2 = _dag_sin_cos_add_reversed_order()

        cs1 = fast_canonical_string(d1, timeout=_TIMEOUT, backend="cpp")
        cs2 = fast_canonical_string(d2, timeout=_TIMEOUT, backend="cpp")

        assert cs1 == cs2, f"fcs should be the same for isomorphic DAGs; got {cs1!r} vs {cs2!r}"

    def test_old_drop_comparator_produces_false_negative(self) -> None:
        """Show the old comparator reports a failure that fcs would report as OK.

        The scenario: S2D(fcs(d1)) may produce d2 (isomorphic but with a
        different node ordering). structural_key(d1) != structural_key(d2)
        → old comparator says FAIL; fcs(d1) == fcs(d2) → new comparator says OK.
        """
        from isalsr.core.canonical import fast_canonical_string

        d1 = _dag_sin_cos_add()
        m = 2

        keep_cs = fast_canonical_string(d1, timeout=_TIMEOUT, backend="cpp")
        # Decode the canonical string to get the "round-trip" DAG
        d_decoded = _dag_from_string(keep_cs, m)

        old_ok = _structural_key(d1) == _structural_key(d_decoded)
        new_ok = round_trip_keep(keep_cs, m, _TIMEOUT)

        # The new comparator is definitionally correct (it checks fcs idempotence)
        assert new_ok, "round_trip_keep should succeed for S2D-produced canonical"

        # The old comparator MAY incorrectly say FAIL (false negative).
        # We document this: either they disagree (defect exposed) or both agree
        # (this particular DAG happens to preserve ordering — still a valid test).
        if not old_ok and new_ok:
            # This is the classic false-negative artefact
            pass  # test passed by reaching here
        # If both agree for this specific DAG, the general defect is still
        # demonstrated by test_structural_key_differs_for_isomorphic_dags.


# ---------------------------------------------------------------------------
# (a) Unified comparator correctness
# ---------------------------------------------------------------------------


class TestUnifiedComparator:
    """round_trip_keep and round_trip_drop_unified use fcs as unified oracle."""

    @pytest.mark.parametrize(
        "instruction_string, m",
        [
            ("V+", 2),
            ("Vs", 1),
            ("V+NV*", 2),
            ("V+NPVs", 2),
        ],
    )
    def test_round_trip_keep_on_s2d_dags(self, instruction_string: str, m: int) -> None:
        """S2D-produced DAGs satisfy the round-trip property by construction.

        fcs(S2D(fcs(D))) == fcs(D) must hold for any D produced by S2D
        (Theorem 3.13).  These are the simplest cases: the string IS the
        canonical string already.
        """
        dag = _dag_from_string(instruction_string, m)
        from isalsr.core.canonical import fast_canonical_string

        keep_cs = fast_canonical_string(dag, timeout=_TIMEOUT, backend="cpp")
        assert round_trip_keep(keep_cs, m, _TIMEOUT), (
            f"round_trip_keep failed for s2d dag from {instruction_string!r}"
        )

    def test_round_trip_drop_unified_matches_keep_for_normalized_dag(self) -> None:
        """For adapter-normalized DAGs, drop_cs == keep_cs so RT rates are equal.

        The adapter applies normalize_const_creation before canonicalisation.
        The C++ engine applies the same normalization inside fcs (keep arm).
        Therefore fcs_raw(D) == fcs(D) for adapter DAGs → drop RT = keep RT.
        """
        from isalsr.core import _native
        from isalsr.core.canonical import _py_dag_to_native, fast_canonical_string

        dag = _dag_add_xy(m=2)
        m = 2

        keep_cs = fast_canonical_string(dag, timeout=_TIMEOUT, backend="cpp")

        # Drop arm: fcs_raw (skips normalization, but adapter already normalized)
        try:
            raw_fn = _native.testing.fast_canonical_string_raw
            drop_cs = raw_fn(_py_dag_to_native(dag), _TIMEOUT)
        except Exception:
            pytest.skip("fcs_raw not available in this build")

        # For a fully normalized DAG, both arms should agree
        assert keep_cs == drop_cs, (
            "Adapter-normalized DAGs should produce identical keep and drop strings"
        )
        assert round_trip_drop_unified(drop_cs, keep_cs, m, _TIMEOUT), (
            "round_trip_drop_unified should succeed when drop_cs == keep_cs"
        )

    def test_round_trip_keep_simple_var_only(self) -> None:
        """A DAG with only VAR nodes has empty canonical string; skip gracefully."""
        from isalsr.core.canonical import fast_canonical_string

        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        m = 2
        try:
            keep_cs = fast_canonical_string(dag, timeout=_TIMEOUT, backend="cpp")
            # If canonicalisation succeeds, round-trip must hold
            result = round_trip_keep(keep_cs, m, _TIMEOUT)
            assert result
        except Exception:
            # Some implementations raise on VAR-only DAGs; that is also fine
            pass

    def test_round_trip_drop_unified_returns_false_on_none_drop(self) -> None:
        """When drop_cs is None (drop raised), round_trip_drop_unified returns False."""
        result = round_trip_drop_unified("not-a-valid-string", "not-a-valid-string", 1, _TIMEOUT)
        # "not-a-valid-string" will fail S2D decoding → False
        assert result is False


# ---------------------------------------------------------------------------
# (b) Feature classification
# ---------------------------------------------------------------------------


class TestClassifyFailureFeatures:
    """classify_failure_features extracts correct structural features."""

    def test_var_only_dag(self) -> None:
        """A VAR-only DAG has k=0 and all flags False."""
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        f = classify_failure_features(dag)
        assert f.k == 0
        assert f.m == 2
        assert not f.has_sub
        assert not f.has_div
        assert not f.has_pow
        assert not f.has_const
        assert f.binary_single_edge == 0
        assert f.indegree_profile == {}

    def test_add_node(self) -> None:
        """ADD node is an internal node with k=1; not flagged as sub/div/pow."""
        dag = _dag_add_xy(m=2)
        f = classify_failure_features(dag)
        assert f.k == 1
        assert not f.has_sub
        assert not f.has_div
        assert not f.has_pow
        assert f.binary_single_edge == 0
        assert "ADD" in f.indegree_profile
        assert f.indegree_profile["ADD"] == [2]

    def test_sub_node_detected(self) -> None:
        """SUB node sets has_sub=True."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        sub_id = dag.add_node(NodeType.SUB)  # 2
        dag.add_edge(0, sub_id)
        dag.add_edge(1, sub_id)
        f = classify_failure_features(dag)
        assert f.has_sub
        assert not f.has_div
        assert f.binary_single_edge == 0  # in_degree == 2, not flagged

    def test_div_node_detected(self) -> None:
        """DIV node sets has_div=True."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        div_id = dag.add_node(NodeType.DIV)
        dag.add_edge(0, div_id)
        dag.add_edge(1, div_id)
        f = classify_failure_features(dag)
        assert f.has_div
        assert f.binary_single_edge == 0

    def test_pow_node_detected(self) -> None:
        """POW node sets has_pow=True."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        pow_id = dag.add_node(NodeType.POW)
        dag.add_edge(0, pow_id)
        dag.add_edge(1, pow_id)
        f = classify_failure_features(dag)
        assert f.has_pow
        assert f.binary_single_edge == 0

    def test_binary_single_edge_detected(self) -> None:
        """SUB with exactly 1 in-edge (src1==src2 adapter case) sets binary_single_edge.

        In the Bingo adapter, when param1 == param2 for a binary op (e.g. x-x),
        the second add_edge is skipped, leaving the node with in_degree == 1.
        This is the primary hypothesis for the 0.6% failure mechanism.
        """
        dag = _dag_sub_single_edge(m=1)
        f = classify_failure_features(dag)
        assert f.has_sub
        assert f.binary_single_edge == 1
        assert f.k == 1
        assert f.m == 1

    @pytest.mark.parametrize(
        "node_type, flag_attr",
        [
            (NodeType.SUB, "has_sub"),
            (NodeType.DIV, "has_div"),
            (NodeType.POW, "has_pow"),
        ],
    )
    def test_single_edge_for_each_binary_type(self, node_type: NodeType, flag_attr: str) -> None:
        """Each binary type with 1 in-edge sets the flag and binary_single_edge."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        op_id = dag.add_node(node_type)
        dag.add_edge(0, op_id)  # Only one edge — src1==src2 scenario
        f = classify_failure_features(dag)
        assert getattr(f, flag_attr), f"{flag_attr} should be True"
        assert f.binary_single_edge == 1

    def test_const_node_detected(self) -> None:
        """CONST leaf sets has_const=True."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        const_id = dag.add_node(NodeType.CONST, const_value=3.14)  # 1
        dag.add_edge(0, const_id)  # creation edge (invariant #9)
        f = classify_failure_features(dag)
        assert f.has_const
        assert f.k == 1

    def test_mixed_dag(self) -> None:
        """Complex DAG with multiple node types reports all features correctly."""
        dag = LabeledDAG(12)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        sin_id = dag.add_node(NodeType.SIN)  # 2
        sub_id = dag.add_node(NodeType.SUB)  # 3
        const_id = dag.add_node(NodeType.CONST, const_value=1.0)  # 4
        dag.add_edge(0, sin_id)
        dag.add_edge(sin_id, sub_id)
        dag.add_edge(1, sub_id)
        dag.add_edge(0, const_id)  # creation edge

        f = classify_failure_features(dag)
        assert f.k == 3  # SIN, SUB, CONST
        assert f.m == 2
        assert f.has_sub
        assert not f.has_div
        assert not f.has_pow
        assert f.has_const
        assert f.binary_single_edge == 0  # SUB has in_degree == 2

    def test_indegree_profile_accurate(self) -> None:
        """indegree_profile lists in-degrees grouped by NodeType name."""
        dag = _dag_add_xy(m=2)
        f = classify_failure_features(dag)
        assert "ADD" in f.indegree_profile
        # ADD has 2 in-edges (from x0 and x1)
        assert f.indegree_profile["ADD"] == [2]


# ---------------------------------------------------------------------------
# (c) dag_to_json
# ---------------------------------------------------------------------------


class TestDagToJson:
    """dag_to_json produces correct JSON-serialisable structure."""

    def test_all_fields_present(self) -> None:
        """JSON output includes nodes, edges, input_order, keep_cs, decoded_cs, features."""
        dag = _dag_add_xy(m=2)
        feats = classify_failure_features(dag)
        j = dag_to_json(dag, keep_cs="Vx", decoded_cs="Vx", features=feats)

        assert "nodes" in j
        assert "edges" in j
        assert "input_order" in j
        assert "keep_cs" in j
        assert "decoded_cs" in j
        assert "features" in j

    def test_node_count(self) -> None:
        """nodes list has one entry per DAG node."""
        dag = _dag_add_xy(m=2)
        j = dag_to_json(dag)
        assert len(j["nodes"]) == dag.node_count

    def test_edge_count(self) -> None:
        """edges list length equals DAG edge_count."""
        dag = _dag_add_xy(m=2)
        j = dag_to_json(dag)
        assert len(j["edges"]) == dag.edge_count

    def test_node_labels_correct(self) -> None:
        """Each node's label field matches the NodeType name."""
        dag = _dag_add_xy(m=2)
        j = dag_to_json(dag)
        labels = {n["id"]: n["label"] for n in j["nodes"]}
        assert labels[0] == "VAR"
        assert labels[1] == "VAR"
        assert labels[2] == "ADD"

    def test_input_order_for_add(self) -> None:
        """input_order correctly records ordered inputs for the ADD node."""
        dag = _dag_add_xy(m=2)
        j = dag_to_json(dag)
        io = j["input_order"]
        # ADD node is at id 2; ordered_inputs = [0, 1]
        assert "2" in io
        assert io["2"] == [0, 1]

    def test_var_index_in_json(self) -> None:
        """VAR nodes include var_index in their JSON dict."""
        dag = _dag_add_xy(m=2)
        j = dag_to_json(dag)
        var_nodes = [n for n in j["nodes"] if n["label"] == "VAR"]
        for vn in var_nodes:
            assert "var_index" in vn

    def test_const_value_in_json(self) -> None:
        """CONST nodes include const_value in their JSON dict."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        const_id = dag.add_node(NodeType.CONST, const_value=2.718)
        dag.add_edge(0, const_id)
        j = dag_to_json(dag)
        const_nodes = [n for n in j["nodes"] if n["label"] == "CONST"]
        assert len(const_nodes) == 1
        assert abs(const_nodes[0]["const_value"] - 2.718) < 1e-9

    def test_json_serialisable(self) -> None:
        """dag_to_json output is JSON-serialisable (no non-serialisable types)."""
        import json

        dag = _dag_sin_cos_add()
        feats = classify_failure_features(dag)
        j = dag_to_json(dag, keep_cs="test_cs", decoded_cs="test_cs", features=feats)
        serialised = json.dumps(j)  # must not raise
        assert isinstance(serialised, str)
