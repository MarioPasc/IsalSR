"""Unit tests for the T04 Mode-1 replay of the Stage-D certification streams.

The fixture stream is hand-built so every ratio is known in closed form, and the
two hard correctness checks are exercised against deliberately broken inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.models.stage_d_trace import (
    CANDIDATES_FILE,
    StageDTracer,
)
from experiments.models.structural_scope import recorded_key
from experiments.scripts import stage_d_mode1_replay as m1
from isalsr.baselines.fixed_order_hash import FixedOrder
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

INS = FixedOrder.INSERTION.value
TOPO = FixedOrder.TOPOLOGICAL.value
TOPO_C = FixedOrder.TOPOLOGICAL_COMMUTATIVE.value


# --------------------------------------------------------------------------- #
# The hand-built stream
# --------------------------------------------------------------------------- #
#
# Six candidates in three isomorphism classes, each class holding two members
# that differ only in internal numbering or operand-insertion order:
#
#   A , B    x0 + x1, edges added (0,1) resp. (1,0)          k = 1
#   S1, S2   sin(x0) + cos(x0), SIN/COS numbered either way   k = 3
#   G1, G2   sin(x0) + sin(x1), SIN nodes numbered either way k = 3
#
# Verified serialisations (see the module docstring of stage_d_mode1_replay for
# the ladder's refinement order):
#
#   insertion               all six distinct   -> rho_total = 6/6 = 1.0
#   topological             all six distinct   -> rho_exact[topo] = 1.0
#   topological_commutative three distinct     -> rho_exact[topo_comm] = 2.0
#   canonical               three distinct     -> rho_iso = 2.0


def _add(edge_order: tuple[int, int]) -> LabeledDAG:
    dag = LabeledDAG(max_nodes=8)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    node = dag.add_node(NodeType.ADD)
    for src in edge_order:
        dag.add_edge(src, node)
    return dag


def _sin_cos(sin_first: bool) -> LabeledDAG:
    dag = LabeledDAG(max_nodes=8)
    dag.add_node(NodeType.VAR, var_index=0)
    first = NodeType.SIN if sin_first else NodeType.COS
    second = NodeType.COS if sin_first else NodeType.SIN
    u1 = dag.add_node(first)
    u2 = dag.add_node(second)
    dag.add_edge(0, u1)
    dag.add_edge(0, u2)
    node = dag.add_node(NodeType.ADD)
    dag.add_edge(u1, node)
    dag.add_edge(u2, node)
    return dag


def _two_sin(swap: bool) -> LabeledDAG:
    dag = LabeledDAG(max_nodes=8)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    u1 = dag.add_node(NodeType.SIN)
    u2 = dag.add_node(NodeType.SIN)
    dag.add_edge(1 if swap else 0, u1)
    dag.add_edge(0 if swap else 1, u2)
    node = dag.add_node(NodeType.ADD)
    dag.add_edge(u1, node)
    dag.add_edge(u2, node)
    return dag


def stream_dags() -> list[LabeledDAG]:
    """Return the six-candidate fixture stream, in stream order."""
    return [
        _add((0, 1)),
        _add((1, 0)),
        _sin_cos(sin_first=True),
        _sin_cos(sin_first=False),
        _two_sin(swap=False),
        _two_sin(swap=True),
    ]


def write_trace(trace_dir: Path, dags: list[LabeledDAG]) -> Path:
    """Persist *dags* through the real tracer and return the trace directory."""
    tracer = StageDTracer.from_env(
        {
            "ISALSR_STAGE_D_TRACE": "1",
            "ISALSR_STAGE_D_TRACE_DIR": str(trace_dir),
            "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": "1",
        }
    )
    for dag in dags:
        tracer.begin()
        tracer.note_eval_time(1e-4)
        tracer.record(dag=dag, representation=fast_canonical_string(dag), t_canon=2e-4)
    tracer.close(run={"method": "bingo", "seed": 1, "problem": "Pagie-1"})
    return trace_dir


def write_raw_trace(trace_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write *rows* verbatim as ``candidates.jsonl`` and return the directory."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / CANDIDATES_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return trace_dir


@pytest.fixture
def traced(tmp_path: Path) -> Path:
    return write_trace(tmp_path / "c2_trace", stream_dags())


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #


class TestRatios:
    """rho_total / rho_exact / rho_iso on a stream with known duplicate structure."""

    def test_stream_loads_completely(self, traced: Path) -> None:
        records, report = m1.load_stream(traced)
        assert len(records) == 6
        assert report.n_lines == 6
        assert report.n_malformed == 0
        assert report.n_deserialise_failures == 0
        assert report.n_canon_failures == 0

    def test_replay_is_faithful_to_the_run(self, traced: Path) -> None:
        _records, report = m1.load_stream(traced)
        assert report.digest_mismatches == []
        assert report.canonical_mismatches == []

    def test_run_identity_recovered(self, traced: Path) -> None:
        _records, report = m1.load_stream(traced)
        assert report.run["method"] == "bingo"
        assert report.run["seed"] == 1

    def test_overall_ratios(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        ratios = m1.compute_ratios(records)["overall"]
        assert ratios["n"] == 6
        assert ratios["distinct_total"] == 6
        assert ratios["distinct_exact"] == {INS: 6, TOPO: 6, TOPO_C: 3}
        assert ratios["distinct_iso"] == 3
        assert ratios["rho_total"] == pytest.approx(1.0)
        assert ratios["rho_exact"][INS] == pytest.approx(1.0)
        assert ratios["rho_exact"][TOPO] == pytest.approx(1.0)
        assert ratios["rho_exact"][TOPO_C] == pytest.approx(2.0)
        assert ratios["rho_iso"] == pytest.approx(2.0)

    def test_rho_total_equals_rho_exact_insertion(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        overall = m1.compute_ratios(records)["overall"]
        assert overall["rho_total"] == overall["rho_exact"][INS]

    def test_monotonicity_holds(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        assert m1.compute_ratios(records)["monotonicity_ok"] is True

    def test_stratified_by_k(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        by_k = m1.compute_ratios(records)["by_k"]
        assert set(by_k) == {"1", "3"}
        assert by_k["1"]["n"] == 2
        assert by_k["1"]["rho_total"] == pytest.approx(1.0)
        assert by_k["1"]["rho_iso"] == pytest.approx(2.0)
        assert by_k["3"]["n"] == 4
        assert by_k["3"]["distinct_iso"] == 2
        assert by_k["3"]["rho_iso"] == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("repeats", "expected_iso"),
        [(1, 2.0), (2, 4.0), (5, 10.0)],
    )
    def test_ratio_scales_with_duplication(
        self, tmp_path: Path, repeats: int, expected_iso: float
    ) -> None:
        dags = stream_dags() * repeats
        trace = write_trace(tmp_path / f"r{repeats}", dags)
        records, _ = m1.load_stream(trace)
        overall = m1.compute_ratios(records)["overall"]
        assert overall["n"] == 6 * repeats
        assert overall["rho_iso"] == pytest.approx(expected_iso)

    def test_empty_stream_yields_zero_ratios(self, tmp_path: Path) -> None:
        trace = write_raw_trace(tmp_path / "empty", [])
        records, _ = m1.load_stream(trace)
        overall = m1.compute_ratios(records)["overall"]
        assert overall["n"] == 0
        assert overall["rho_total"] == 0.0
        assert overall["rho_iso"] == 0.0

    def test_malformed_and_unserialisable_lines_are_counted_not_fatal(self, tmp_path: Path) -> None:
        good, _ = m1.load_stream(write_trace(tmp_path / "g", stream_dags()[:1]))
        rows = [
            {"i": 0, "serialisation": good[0].serialisations[INS], "canonical": None},
            {"i": 1, "serialisation": "3|not-a-record"},
            {"i": 2},
        ]
        trace = write_raw_trace(tmp_path / "mixed", rows)
        (trace / CANDIDATES_FILE).open("a").write("{not json\n")
        records, report = m1.load_stream(trace)
        assert len(records) == 1
        assert report.n_malformed == 1
        assert report.n_deserialise_failures == 1


# --------------------------------------------------------------------------- #
# Hard check 1 — hash soundness
# --------------------------------------------------------------------------- #


class TestHashSoundness:
    """A shared fixed-order digest must imply a shared canonical string."""

    def test_clean_stream_is_sound(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        result = m1.check_hash_soundness(records)
        assert result["sound"] is True
        for block in result["by_order"].values():
            assert block["n_unsound_merges"] == 0
            assert block["n_digest_collisions"] == 0

    def test_unsound_merge_is_detected_and_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stub the digest AND the non-insertion serialisations so that two DAGs
        with different canonical strings look identical to the topological rung.

        That is exactly the failure mode T04 AC-1 forbids: the baseline merges
        two non-isomorphic expressions and inflates its own reduction factor.
        """
        trace = write_trace(tmp_path / "unsound", [_add((0, 1)), _sin_cos(sin_first=True)])
        real_serialise = m1.serialise
        monkeypatch.setattr(m1, "fixed_order_digest", lambda dag, order: 42)
        monkeypatch.setattr(
            m1,
            "serialise",
            lambda dag, order: (
                real_serialise(dag, order) if order is FixedOrder.INSERTION else "COLLIDING"
            ),
        )
        records, _ = m1.load_stream(trace)
        result = m1.check_hash_soundness(records)

        assert result["sound"] is False
        topo = result["by_order"][TOPO]
        assert topo["n_unsound_merges"] == 1
        pair = topo["unsound_merges"][0]
        assert pair["left"]["i"] == 0
        assert pair["right"]["i"] == 1
        assert pair["left"]["canonical"] != pair["right"]["canonical"]
        assert pair["left"]["serialisation_insertion"] != pair["right"]["serialisation_insertion"]

    def test_digest_collision_is_reported_separately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trace = write_trace(tmp_path / "collide", [_add((0, 1)), _sin_cos(sin_first=True)])
        monkeypatch.setattr(m1, "fixed_order_digest", lambda dag, order: 7)
        records, _ = m1.load_stream(trace)
        result = m1.check_hash_soundness(records)
        assert result["sound"] is False
        block = result["by_order"][INS]
        assert block["n_unsound_merges"] == 0
        assert block["n_digest_collisions"] == 1

    def test_unsound_merge_causes_a_loud_non_zero_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        trace = write_trace(tmp_path / "loud", [_add((0, 1)), _sin_cos(sin_first=True)])
        real_serialise = m1.serialise
        monkeypatch.setattr(m1, "fixed_order_digest", lambda dag, order: 42)
        monkeypatch.setattr(
            m1,
            "serialise",
            lambda dag, order: (
                real_serialise(dag, order) if order is FixedOrder.INSERTION else "COLLIDING"
            ),
        )
        out_json = tmp_path / "out" / "replay.json"
        out_md = tmp_path / "out" / "replay.md"
        with caplog.at_level("ERROR"):
            code = m1.main(
                [
                    "--trace-dir",
                    str(trace),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )
        assert code == 2
        assert "UNSOUND MERGE" in caplog.text
        report = json.loads(out_json.read_text())
        assert report["hash_soundness_ok"] is False
        assert report["ok"] is False
        assert "UNSOUND MERGE" in out_md.read_text()


# --------------------------------------------------------------------------- #
# Hard check 2 — IsalSR soundness
# --------------------------------------------------------------------------- #


class TestIsalSRSoundness:
    """A shared canonical string must imply ``is_isomorphic``."""

    def test_clean_stream_is_sound(self, traced: Path) -> None:
        records, _ = m1.load_stream(traced)
        result = m1.check_isalsr_soundness(records, max_classes=10)
        assert result["sound"] is True
        assert result["n_classes_total"] == 3
        assert result["n_classes_checked"] == 3
        assert result["largest_class_sizes"] == [2, 2, 2]
        assert result["n_pairs_checked"] == 3

    def test_runs_on_the_largest_classes_only(self, tmp_path: Path) -> None:
        dags = [_add((0, 1)), _add((1, 0)), _add((0, 1)), _sin_cos(sin_first=True)]
        trace = write_trace(tmp_path / "sizes", dags)
        records, _ = m1.load_stream(trace)
        result = m1.check_isalsr_soundness(records, max_classes=1)
        assert result["n_classes_total"] == 2
        assert result["n_classes_checked"] == 1
        assert result["largest_class_sizes"] == [3]
        assert result["n_pairs_checked"] == 2

    @pytest.mark.parametrize("max_classes", [0, -1])
    def test_zero_classes_checks_nothing(self, traced: Path, max_classes: int) -> None:
        records, _ = m1.load_stream(traced)
        result = m1.check_isalsr_soundness(records, max_classes=max_classes)
        assert result["n_classes_checked"] == 0
        assert result["n_pairs_checked"] == 0
        assert result["sound"] is True

    def test_violation_is_detected_and_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collapse every DAG into one canonical class; the members are not
        isomorphic, so the invariant's soundness must fail."""
        trace = write_trace(tmp_path / "isoviol", stream_dags())
        # Drop the recorded canonicals so the fidelity check does not fire first
        # and the isomorphism check is isolated.
        rows = [
            {**json.loads(line), "canonical": None}
            for line in (trace / CANDIDATES_FILE).read_text().splitlines()
        ]
        stripped = write_raw_trace(tmp_path / "isoviol_raw", rows)
        monkeypatch.setattr(m1, "fast_canonical_string", lambda dag, backend=None: "SAME")
        records, _ = m1.load_stream(stripped)
        result = m1.check_isalsr_soundness(records, max_classes=10)
        assert result["sound"] is False
        assert result["n_classes_total"] == 1
        assert result["n_failures"] > 0
        failure = result["failures"][0]
        assert failure["canonical"] == "SAME"
        assert failure["left"]["i"] != failure["right"]["i"]

    def test_violation_causes_a_loud_non_zero_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        trace = write_trace(tmp_path / "isoloud", stream_dags())
        rows = [
            {**json.loads(line), "canonical": None}
            for line in (trace / CANDIDATES_FILE).read_text().splitlines()
        ]
        stripped = write_raw_trace(tmp_path / "isoloud_raw", rows)
        monkeypatch.setattr(m1, "fast_canonical_string", lambda dag, backend=None: "SAME")
        out_json = tmp_path / "iso.json"
        with caplog.at_level("ERROR"):
            code = m1.main(["--trace-dir", str(stripped), "--out-json", str(out_json)])
        assert code == 2
        assert "ISALSR SOUNDNESS VIOLATION" in caplog.text
        assert json.loads(out_json.read_text())["isalsr_soundness_ok"] is False


# --------------------------------------------------------------------------- #
# Replay fidelity
# --------------------------------------------------------------------------- #


class TestReplayFidelity:
    """Recomputed keys must agree with the ones recorded during the run."""

    def test_tampered_recorded_canonical_is_caught(self, tmp_path: Path) -> None:
        trace = write_trace(tmp_path / "tamper", stream_dags())
        rows = [json.loads(line) for line in (trace / CANDIDATES_FILE).read_text().splitlines()]
        rows[2]["canonical"] = rows[2]["canonical"] + "W"
        tampered = write_raw_trace(tmp_path / "tampered", rows)
        _records, report = m1.load_stream(tampered)
        assert len(report.canonical_mismatches) == 1
        assert report.canonical_mismatches[0]["i"] == 2

    def test_tampered_recorded_digest_is_caught(self, tmp_path: Path) -> None:
        trace = write_trace(tmp_path / "dtamper", stream_dags())
        rows = [json.loads(line) for line in (trace / CANDIDATES_FILE).read_text().splitlines()]
        rows[1][f"digest_{TOPO}"] = 1
        tampered = write_raw_trace(tmp_path / "dtampered", rows)
        _records, report = m1.load_stream(tampered)
        assert len(report.digest_mismatches) == 1
        assert report.digest_mismatches[0]["order"] == TOPO

    def test_fidelity_failure_fails_the_run(self, tmp_path: Path) -> None:
        trace = write_trace(tmp_path / "fid", stream_dags())
        rows = [json.loads(line) for line in (trace / CANDIDATES_FILE).read_text().splitlines()]
        rows[0]["canonical"] = "WRONG"
        tampered = write_raw_trace(tmp_path / "fid_raw", rows)
        out_json = tmp_path / "fid.json"
        assert m1.main(["--trace-dir", str(tampered), "--out-json", str(out_json)]) == 2
        assert json.loads(out_json.read_text())["replay_fidelity_ok"] is False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCli:
    """Argument handling, discovery and report emission."""

    def test_clean_run_exits_zero_and_writes_both_reports(
        self, traced: Path, tmp_path: Path
    ) -> None:
        out_json = tmp_path / "r" / "replay.json"
        out_md = tmp_path / "r" / "replay.md"
        code = m1.main(
            [
                "--trace-dir",
                str(traced),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
                "--max-classes",
                "5",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        report = json.loads(out_json.read_text())
        assert report["ok"] is True
        assert report["n_streams"] == 1
        assert set(report["rho_definitions"]) == {"rho_total", "rho_exact", "rho_iso"}
        md = out_md.read_text()
        assert "rho_iso" in md
        assert "R1.4 decision" in md

    def test_by_method_pooling(self, traced: Path, tmp_path: Path) -> None:
        out_json = tmp_path / "pooled.json"
        m1.main(["--trace-dir", str(traced), "--out-json", str(out_json)])
        pooled = json.loads(out_json.read_text())["by_method"]
        assert set(pooled) == {"bingo"}
        assert pooled["bingo"]["n"] == 6
        assert pooled["bingo"]["rho_iso"] == pytest.approx(2.0)

    def test_results_root_discovery(self, tmp_path: Path) -> None:
        root = tmp_path / "campaign"
        for name in ("bingo/seed_1", "udfs/seed_1"):
            write_trace(root / name / "c2_trace", stream_dags())
        assert len(m1.discover_trace_dirs(root)) == 2
        out_json = tmp_path / "all.json"
        assert m1.main(["--results-root", str(root), "--out-json", str(out_json)]) == 0
        assert json.loads(out_json.read_text())["n_streams"] == 2

    def test_no_input_exits_three(self) -> None:
        assert m1.main([]) == 3

    def test_missing_stream_exits_three(self, tmp_path: Path) -> None:
        assert m1.main(["--trace-dir", str(tmp_path / "nope")]) == 3

    def test_forced_python_backend(self, traced: Path, tmp_path: Path) -> None:
        out_json = tmp_path / "py.json"
        assert (
            m1.main(
                [
                    "--trace-dir",
                    str(traced),
                    "--canonical-backend",
                    "python",
                    "--out-json",
                    str(out_json),
                ]
            )
            == 0
        )
        assert json.loads(out_json.read_text())["canonical_backend"] == "python"

    def test_r14_null_result_is_recorded(self, traced: Path, tmp_path: Path) -> None:
        """rho_exact[insertion] == 1.00 on the fixture, so the report must say so."""
        out_md = tmp_path / "null.md"
        m1.main(["--trace-dir", str(traced), "--out-md", str(out_md)])
        assert "null result" in out_md.read_text()

    def test_r14_non_null_result_is_recorded(self, tmp_path: Path) -> None:
        trace = write_trace(tmp_path / "dup", [_add((0, 1)), _add((0, 1))])
        out_md = tmp_path / "notnull.md"
        m1.main(["--trace-dir", str(trace), "--out-md", str(out_md)])
        assert "not** a null result" in out_md.read_text()


# --------------------------------------------------------------------------- #
# k=0 records must replay clean (regression, 2026-08-25)
# --------------------------------------------------------------------------- #


class TestNonStructuralRecordsReplayClean:
    """A bare-variable candidate must not read as an engine disagreement.

    The production runners record ``nonstructural_key(dag)`` in place of the
    canonical string whenever k=0.  ``_cross_check`` used to re-canonicalise
    unconditionally and compare the raw ``""`` against that recorded ``#k0:...``
    value, so **every** k=0 record landed in ``canonical_mismatches`` and the
    replay reported ``REPLAY FIDELITY FAILURE``.  Bare variables are common in a
    GP population, so this fired on essentially every traced run.
    """

    @staticmethod
    def _bare_variable() -> LabeledDAG:
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)
        return dag

    @staticmethod
    def _write(trace_dir: Path, dags: list[LabeledDAG]) -> Path:
        """Persist *dags* the way the production runners do (with the k=0 key)."""
        tracer = StageDTracer.from_env(
            {
                "ISALSR_STAGE_D_TRACE": "1",
                "ISALSR_STAGE_D_TRACE_DIR": str(trace_dir),
                "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": "1",
            }
        )
        for dag in dags:
            tracer.begin()
            tracer.note_eval_time(1e-4)
            tracer.record(
                dag=dag,
                representation=recorded_key(dag, fast_canonical_string(dag)),
                t_canon=2e-4,
            )
        tracer.close(run={"method": "bingo", "seed": 1, "problem": "Nguyen-1"})
        return trace_dir

    def test_stream_of_bare_variables_reports_no_mismatch(self, tmp_path: Path) -> None:
        trace_dir = self._write(tmp_path / "c2_trace", [self._bare_variable() for _ in range(3)])

        _records, load = m1.load_stream(trace_dir, None)

        assert load.n_replayable == 3
        assert load.canonical_mismatches == [], load.canonical_mismatches
        assert load.digest_mismatches == []

    def test_mixed_stream_reports_no_mismatch(self, tmp_path: Path) -> None:
        dags = [*stream_dags(), self._bare_variable()]
        trace_dir = self._write(tmp_path / "c2_trace", dags)

        _records, load = m1.load_stream(trace_dir, None)

        assert load.n_replayable == len(dags)
        assert load.canonical_mismatches == [], load.canonical_mismatches

    def test_a_genuine_canonical_mismatch_is_still_caught(self, tmp_path: Path) -> None:
        """The substitution must not blunt the check it lives inside."""
        trace_dir = self._write(tmp_path / "c2_trace", [_add((0, 1))])
        path = trace_dir / CANDIDATES_FILE
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["canonical"] = "V+"  # a real Sigma_SR word, but the wrong one
        write_raw_trace(trace_dir, rows)

        _records, load = m1.load_stream(trace_dir, None)

        assert len(load.canonical_mismatches) == 1
        assert load.canonical_mismatches[0]["recorded"] == "V+"
