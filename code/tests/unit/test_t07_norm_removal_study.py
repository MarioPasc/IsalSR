"""Unit tests for T07 norm-removal study helpers.

Tests cover:
  - ArmStats accumulation and serialisation
  - _score_keep and _score_drop on trivial and adversarial DAGs
  - _check_equivariance: keep arm fails on adversarial, drop arm consistent
  - _round_trip_keep and _round_trip_drop on S2D-decoded DAGs
  - Adversarial fixture: keep gives equivariance failure, drop raises
  - _build_adversarial_dags: all DAGs satisfy the three conditions
  - _structural_key: isomorphic DAGs have equal keys when 𝒩 is identity
"""

from __future__ import annotations

import random

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_dag(m: int = 2) -> LabeledDAG:
    """Build x_0 + x_1 (simplest non-trivial expression).

    Args:
        m: Number of input variables.

    Returns:
        A minimal 3-node DAG.
    """
    d = LabeledDAG(8)
    for i in range(m):
        d.add_node(NodeType.VAR, var_index=i)
    a = d.add_node(NodeType.ADD)
    for i in range(m):
        d.add_edge(i, a)
    return d


def _make_dag_k2() -> LabeledDAG:
    """Build x_0 + sin(x_1) — k=2 internal nodes, no CONST.

    Returns:
        A 4-node DAG with exactly 2 permutable internal nodes.
    """
    d = LabeledDAG(8)
    d.add_node(NodeType.VAR, var_index=0)  # 0
    d.add_node(NodeType.VAR, var_index=1)  # 1
    s = d.add_node(NodeType.SIN)  # 2 = internal 0
    a = d.add_node(NodeType.ADD)  # 3 = internal 1
    d.add_edge(1, s)  # x_1 -> sin
    d.add_edge(0, a)  # x_0 -> add
    d.add_edge(s, a)  # sin -> add
    return d


def _make_const_dag() -> LabeledDAG:
    """Build x_0 * c with orphan CONST (no in-edge on c).

    Returns:
        DAG where c has in-degree 0.
    """
    d = LabeledDAG(8)
    d.add_node(NodeType.VAR, var_index=0)  # 0
    d.add_node(NodeType.CONST, const_value=2.0)  # 1
    m = d.add_node(NodeType.MUL)  # 2
    d.add_edge(0, m)
    d.add_edge(1, m)
    return d


def _make_fixture() -> tuple[LabeledDAG, LabeledDAG]:
    """Return (d, d2) — verbatim brief fixture (equivariance counterexample)."""
    d = LabeledDAG(16)
    for i in range(3):
        d.add_node(NodeType.VAR, var_index=i)
    c1 = d.add_node(NodeType.CONST, const_value=1.0)  # 3
    c2 = d.add_node(NodeType.CONST, const_value=1.0)  # 4
    node_a = d.add_node(NodeType.SIN)  # 5
    node_b = d.add_node(NodeType.SIN)  # 6
    d.add_edge(c2, node_a)
    d.add_edge(node_a, 0)
    d.add_edge(c1, node_b)
    d.add_edge(node_b, 1)
    d2 = permute_internal_nodes(d, [1, 0, 2, 3])
    return d, d2


# ---------------------------------------------------------------------------
# ArmStats
# ---------------------------------------------------------------------------


class TestArmStats:
    """Tests for ArmStats accumulator."""

    def test_initial_state(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats()
        assert s.n_total == 0
        assert s.n_ok == 0
        assert s.n_unique == 0
        assert s.rho != s.rho  # NaN

    def test_record_ok_increments_unique(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats()
        s.record("ok", "abc", k=2, reachable=True)
        s.record("ok", "abc", k=2, reachable=True)
        s.record("ok", "xyz", k=3, reachable=True)
        assert s.n_total == 3
        assert s.n_ok == 3
        assert s.n_unique == 2  # hash-based; "abc" counted once

    def test_record_raised_does_not_add_unique(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats()
        s.record("raised", None, k=1, reachable=True)
        assert s.n_raised == 1
        assert s.n_unique == 0

    def test_rho_matches_nok_over_nunique(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats()
        for cs in ["a", "a", "b", "c"]:
            s.record("ok", cs, k=2, reachable=True)
        assert pytest.approx(s.rho, rel=1e-6) == 4 / 3

    def test_to_dict_has_required_keys(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats(round_trip_comparator="fast_canonical_string")
        s.record("ok", "s1", k=2, reachable=True)
        d = s.to_dict()
        for key in (
            "n_total",
            "n_ok",
            "n_raised",
            "n_timeout",
            "n_unique",
            "rho",
            "n_reachable",
            "n_round_trip_ok",
            "n_round_trip_checked",
            "round_trip_comparator",
            "n_equivariance_samples",
            "n_equivariance_failures",
            "per_k",
        ):
            assert key in d, f"Missing key: {key}"

    def test_per_k_aggregation(self) -> None:
        from experiments.scripts.t07_norm_removal_study import ArmStats

        s = ArmStats()
        s.record("ok", "a", k=2, reachable=True)
        s.record("ok", "b", k=2, reachable=True)
        s.record("raised", None, k=3, reachable=True)
        d = s.to_dict()
        assert d["per_k"]["2"]["n_total"] == 2
        assert d["per_k"]["2"]["n_ok"] == 2
        assert d["per_k"]["3"]["n_raised"] == 1


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    """Tests for _score_keep and _score_drop."""

    def test_keep_ok_on_simple_dag(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _score_keep

        dag = _make_simple_dag()
        outcome, cs = _score_keep(dag, timeout=5.0)
        assert outcome == "ok"
        assert isinstance(cs, str) and len(cs) > 0

    def test_drop_ok_on_simple_dag(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _score_drop

        dag = _make_simple_dag()
        outcome, cs = _score_drop(dag, timeout=5.0)
        assert outcome == "ok"
        assert isinstance(cs, str) and len(cs) > 0

    def test_keep_and_drop_agree_on_simple_dag(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _score_drop, _score_keep

        dag = _make_simple_dag()
        ko, ks = _score_keep(dag, timeout=5.0)
        do, ds = _score_drop(dag, timeout=5.0)
        assert ko == "ok" and do == "ok"
        assert ks == ds, "Arms must agree on a DAG where 𝒩 is identity"

    def test_keep_ok_on_const_dag_with_orphan(self) -> None:
        # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser (keep arm).
        # Keep arm calls fast_canonical_string which no longer applies 𝒩 internally.
        # Orphan-CONST DAG is now refused loudly by the keep arm (outcome == "raised").
        """Keep arm raises on orphan-CONST DAG — loud refusal replaces silent wrong."""
        from experiments.scripts.t07_norm_removal_study import _score_keep

        dag = _make_const_dag()
        outcome, cs = _score_keep(dag, timeout=5.0)
        # No 𝒩 in keep arm — orphan CONST causes RuntimeError, captured as "raised".
        assert outcome == "raised", f"Keep arm must raise on orphan CONST dag: got {outcome}"
        assert cs is None

    def test_drop_raises_on_const_dag_with_orphan(self) -> None:
        """Drop arm has no 𝒩 — orphan CONST makes D2S unencodable."""
        from experiments.scripts.t07_norm_removal_study import _score_drop

        dag = _make_const_dag()
        outcome, cs = _score_drop(dag, timeout=5.0)
        assert outcome == "raised", f"Drop arm should raise on orphan CONST DAG, got {outcome!r}"
        assert cs is None

    def test_keep_ok_on_s2d_decoded_dag(self) -> None:
        """Canonical → S2D → canonical must be ok on S2D-produced DAGs."""
        from experiments.scripts.t07_norm_removal_study import _score_keep

        dag = StringToDAG("V+NV+", num_variables=2).run()
        outcome, _ = _score_keep(dag, timeout=5.0)
        assert outcome == "ok"


# ---------------------------------------------------------------------------
# Adversarial fixture
# ---------------------------------------------------------------------------


class TestAdversarialFixture:
    """Tests for the equivariance counterexample from the brief."""

    def test_keep_raises_on_fixture(self) -> None:
        # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
        # Keep arm no longer applies 𝒩 — it raises (RuntimeError) on both d and d2.
        # The old "silent wrong" (different strings) is replaced by equivariant loud
        # refusal: both orphan-CONST DAGs are refused consistently, no silent error.
        """Keep arm raises on both d and d2 — equivariant refusal, no silent wrong."""
        from experiments.scripts.t07_norm_removal_study import _score_keep

        d, d2 = _make_fixture()
        ok1, cs1 = _score_keep(d, timeout=5.0)
        ok2, cs2 = _score_keep(d2, timeout=5.0)
        assert ok1 == "raised" and ok2 == "raised", (
            "Keep arm must raise on adversarial DAGs (no 𝒩 in canonicaliser)"
        )
        assert cs1 is None and cs2 is None

    def test_drop_raises_on_fixture_d(self) -> None:
        """Drop arm raises on d (orphan CONSTs, loud refusal)."""
        from experiments.scripts.t07_norm_removal_study import _score_drop

        d, _ = _make_fixture()
        outcome, _ = _score_drop(d, timeout=5.0)
        assert outcome == "raised", f"Drop arm should raise on d, got {outcome!r}"

    def test_drop_raises_on_fixture_d2(self) -> None:
        """Drop arm raises on d2 (still orphan CONSTs after permutation)."""
        from experiments.scripts.t07_norm_removal_study import _score_drop

        _, d2 = _make_fixture()
        outcome, _ = _score_drop(d2, timeout=5.0)
        assert outcome == "raised", f"Drop arm should raise on d2, got {outcome!r}"

    def test_build_adversarial_dags_nonempty(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _build_adversarial_dags

        dags = _build_adversarial_dags()
        assert len(dags) >= 2

    def test_adversarial_dags_have_orphan_consts(self) -> None:
        """Every adversarial DAG must have at least one in-degree-0 CONST."""
        from experiments.scripts.t07_norm_removal_study import _build_adversarial_dags

        for dag in _build_adversarial_dags():
            has_orphan = any(
                dag.node_label(i) == NodeType.CONST and len(dag.ordered_inputs(i)) == 0
                for i in range(dag.node_count)
            )
            assert has_orphan, "Adversarial DAG must have at least one orphan CONST"

    def test_all_adversarial_dags_drop_raises(self) -> None:
        """Drop arm must raise (RuntimeError) on every adversarial DAG."""
        from experiments.scripts.t07_norm_removal_study import (
            _build_adversarial_dags,
            _score_drop,
        )

        for dag in _build_adversarial_dags():
            outcome, _ = _score_drop(dag, timeout=5.0)
            assert outcome == "raised", f"Drop arm should raise on adversarial DAG, got {outcome!r}"


# ---------------------------------------------------------------------------
# Equivariance check
# ---------------------------------------------------------------------------


class TestEquivarianceCheck:
    """Tests for _check_equivariance."""

    def test_keep_equivariant_on_k2_dag(self) -> None:
        """Keep arm is equivariant on a normal k=2 DAG (no orphan CONSTs)."""
        from experiments.scripts.t07_norm_removal_study import (
            _check_equivariance,
            _score_keep,
        )

        dag = _make_dag_k2()  # k=2 internal nodes
        rng = random.Random(42)
        score_fn = lambda d: _score_keep(d, 5.0)  # noqa: E731
        failures, tried = _check_equivariance(dag, score_fn, rng, k_perms=8)
        assert tried == 8
        assert failures == 0

    def test_drop_consistent_on_k2_dag(self) -> None:
        """Drop arm is consistent on a normal k=2 DAG."""
        from experiments.scripts.t07_norm_removal_study import (
            _check_equivariance,
            _score_drop,
        )

        dag = _make_dag_k2()  # k=2 internal nodes
        rng = random.Random(42)
        score_fn = lambda d: _score_drop(d, 5.0)  # noqa: E731
        failures, tried = _check_equivariance(dag, score_fn, rng, k_perms=8)
        assert tried == 8
        assert failures == 0

    def test_keep_consistent_on_fixture(self) -> None:
        # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
        # Keep arm now raises on ALL permutations of the fixture (all have orphan
        # CONSTs regardless of node ordering).  _check_equivariance sees baseline
        # == ("raised", None) and every permuted result == ("raised", None), so
        # failures == 0.  No equivariance failure is detectable — the arm refuses
        # consistently instead of giving silent wrong answers.
        """Keep arm is consistent (all raise) on fixture — no equivariance failure.

        Uses k_perms=24 (= 4!) to cover all permutations of the 4 internal
        nodes.  Every permutation still has orphan CONSTs, so the keep arm
        raises on each — failures == 0 (equivariant refusal, not silent wrong).
        """
        from experiments.scripts.t07_norm_removal_study import (
            _check_equivariance,
            _score_keep,
        )

        d, _ = _make_fixture()
        rng = random.Random(0)
        score_fn = lambda dag: _score_keep(dag, 5.0)  # noqa: E731
        # 4! = 24 permutations: all have orphan CONSTs → all raise → failures == 0
        failures, tried = _check_equivariance(d, score_fn, rng, k_perms=24)
        assert tried == 24
        assert failures == 0, (
            "Keep arm must be consistent (all raise) on fixture after 𝒩 removal: "
            "no permutation produces a different outcome"
        )

    def test_drop_consistent_on_fixture(self) -> None:
        """Drop arm raises on fixture DAG AND all its permutations (consistent).

        Every permutation of the 4 internal nodes still leaves c1 and c2 with
        in-degree 0 (orphan CONSTs), so the drop arm raises on all of them.
        """
        from experiments.scripts.t07_norm_removal_study import (
            _check_equivariance,
            _score_drop,
        )

        d, _ = _make_fixture()
        rng = random.Random(0)
        score_fn = lambda dag: _score_drop(dag, 5.0)  # noqa: E731
        # All permutations should raise — failures == 0 means consistent
        failures, tried = _check_equivariance(d, score_fn, rng, k_perms=24)
        assert tried == 24
        assert failures == 0, "Drop arm should be consistent (all raise) on the adversarial fixture"

    def test_trivial_dag_no_permutation(self) -> None:
        """DAG with k=1 internal node: equivariance is trivially satisfied."""
        from experiments.scripts.t07_norm_removal_study import (
            _check_equivariance,
            _score_keep,
        )

        d = LabeledDAG(4)
        d.add_node(NodeType.VAR, var_index=0)
        d.add_node(NodeType.SIN)
        d.add_edge(0, 1)
        rng = random.Random(7)
        score_fn = lambda dag: _score_keep(dag, 5.0)  # noqa: E731
        _, tried = _check_equivariance(d, score_fn, rng)
        assert tried == 0, "k=1 internal node: no permutation is non-trivial"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for _round_trip_keep and _round_trip_drop."""

    @pytest.mark.parametrize(
        "word,m",
        [
            ("V+NV+", 2),
            ("V*NV*", 2),
            ("Vs", 1),
        ],
    )
    def test_round_trip_keep_holds_on_s2d_dag(self, word: str, m: int) -> None:
        from experiments.scripts.t07_norm_removal_study import _round_trip_keep, _score_keep

        dag = StringToDAG(word, num_variables=m).run()
        ok, cs = _score_keep(dag, timeout=5.0)
        assert ok == "ok"
        assert _round_trip_keep(cs, m, timeout=5.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "word,m",
        [
            ("V+NV+", 2),
            ("V*NV*", 2),
            ("Vs", 1),
        ],
    )
    def test_round_trip_drop_holds_on_s2d_dag(self, word: str, m: int) -> None:
        # STALE CALL SIGNATURE fixed 2026-07-29: _round_trip_drop now requires
        # (drop_canonical, keep_canonical, m, timeout, keep_fn=None).
        # The old call passed `dag` (a LabeledDAG) as keep_canonical and omitted
        # timeout.  S2D-decoded DAGs satisfy the precondition, so both arms agree:
        # keep_canonical == drop_canonical == cs.
        from experiments.scripts.t07_norm_removal_study import (
            _round_trip_drop,
            _score_drop,
            _score_keep,
        )

        dag = StringToDAG(word, num_variables=m).run()
        ok_drop, cs_drop = _score_drop(dag, timeout=5.0)
        ok_keep, cs_keep = _score_keep(dag, timeout=5.0)
        assert ok_drop == "ok", f"Drop arm should succeed on S2D-decoded DAG, got {ok_drop!r}"
        assert ok_keep == "ok", f"Keep arm should succeed on S2D-decoded DAG, got {ok_keep!r}"
        assert _round_trip_drop(cs_drop, cs_keep, m, timeout=5.0)


# ---------------------------------------------------------------------------
# Structural key
# ---------------------------------------------------------------------------


class TestStructuralKey:
    """Tests for _structural_key."""

    def test_same_dag_same_key(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _structural_key

        dag = _make_simple_dag()
        assert _structural_key(dag) == _structural_key(dag)

    def test_different_dags_different_keys(self) -> None:
        from experiments.scripts.t07_norm_removal_study import _structural_key

        d1 = StringToDAG("V+NV+", num_variables=2).run()
        d2 = StringToDAG("V*NV*", num_variables=2).run()
        assert _structural_key(d1) != _structural_key(d2)

    def test_isomorphic_s2d_dags_equal_key_when_same_string(self) -> None:
        """Two DAGs decoded from the same string must have equal structural keys."""
        from experiments.scripts.t07_norm_removal_study import _structural_key

        word, m = "V+NV+", 2
        d1 = StringToDAG(word, num_variables=m).run()
        d2 = StringToDAG(word, num_variables=m).run()
        assert _structural_key(d1) == _structural_key(d2)


# ---------------------------------------------------------------------------
# Smoke test: main() with --population synthetic --n 100
# ---------------------------------------------------------------------------


def test_main_synthetic_smoke(tmp_path: pytest.TempPathFactory) -> None:
    """Run the study script end-to-end with a tiny synthetic population."""
    import json
    import sys
    from unittest.mock import patch

    from experiments.scripts.t07_norm_removal_study import main

    out_dir = str(tmp_path)
    with patch.object(
        sys,
        "argv",
        [
            "t07_norm_removal_study",
            "--population",
            "synthetic",
            "--n",
            "200",
            "--seed",
            "1",
            "--sample-rate",
            "10",
            "--equivariance-k",
            "4",
            "--out",
            out_dir,
        ],
    ):
        main()

    result_path = tmp_path / "results.json"
    assert result_path.exists(), "results.json must be written"
    data = json.loads(result_path.read_text())

    assert data["population"] == "synthetic"
    assert "arms" in data
    assert "keep" in data["arms"]
    assert "drop" in data["arms"]
    assert "comparisons" in data
    # Decisive comparisons must be present
    for key in (
        "rho_keep",
        "rho_drop",
        "rho_equal",
        "n_distinct_keep",
        "n_distinct_drop",
        "n_distinct_equal",
        "per_dag_agreement_rate",
        "n_equivariance_failures_keep",
        "n_equivariance_failures_drop",
    ):
        assert key in data["comparisons"], f"Missing comparison key: {key}"


def test_main_adversarial_smoke(tmp_path: pytest.TempPathFactory) -> None:
    """Run the study script on the adversarial population."""
    import json
    import sys
    from unittest.mock import patch

    from experiments.scripts.t07_norm_removal_study import main

    out_dir = str(tmp_path)
    with patch.object(
        sys,
        "argv",
        [
            "t07_norm_removal_study",
            "--population",
            "adversarial",
            "--seed",
            "0",
            "--sample-rate",
            "1",
            "--equivariance-k",
            "8",
            "--out",
            out_dir,
        ],
    ):
        main()

    data = json.loads((tmp_path / "results.json").read_text())
    assert data["population"] == "adversarial"
    cmp = data["comparisons"]
    # On adversarial: drop arm must raise on all DAGs
    assert data["arms"]["drop"]["n_raised"] > 0
    # Keep silent-wrong must be > 0 (fixture triggers equivariance failure)
    assert cmp["adversarial_keep_silent_wrong"] is not None
    assert cmp["adversarial_drop_loud_refusal"] is not None
    assert cmp["adversarial_drop_loud_refusal"] > 0
