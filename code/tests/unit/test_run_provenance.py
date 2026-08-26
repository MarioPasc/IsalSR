"""Tests for the C2 run-provenance layer.

Covers the four EXECUTION-PLAN items that must be in the code *before* the
campaign launches, because each measures something that exists only while a run
is happening and that no post-hoc pass can recover:

============  ==========================================================
Plan item     What it guarantees
============  ==========================================================
A7 / A7-BUG   ``engine`` and build identity reach ``run_log.json``
C1.9-BUG      the five T06 fallback paths reach ``run_log.json``
P3 / C4       all three arms of a cell provably saw the same data
P4 / §5.5     a killed run leaves a named row, not a silent gap
============  ==========================================================

The tests are written against the *serialised* artefact wherever the plan's
check reads the artefact, because that is what a pre-flight check will actually
parse. A field that exists on a dataclass but does not survive
``to_dict``/``from_dict`` fails the check just as completely as one that was
never added.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from experiments.models.fallback_ledger import FallbackLedger
from experiments.models.hardware_info import collect_hardware_info, peak_rss_gb
from experiments.models.provenance import config_sha256, data_fingerprint
from experiments.models.schemas import (
    BestExpression,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
)
from experiments.models.status_ledger import (
    LEDGER_COLUMNS,
    RunStatus,
    collect_status_ledger,
    load_status,
    reconcile,
    write_status,
)

# ====================================================================== #
# A7 / A7-BUG -- engine and build identity
# ====================================================================== #


class TestHardwareProvenance:
    """A7: the run log must identify the node, the code and the engine."""

    @pytest.fixture(scope="class")
    def hardware(self) -> dict:
        return collect_hardware_info()

    @pytest.mark.parametrize(
        "field",
        [
            # A7-BUG: the field that read "<none>" in a live T04 probe and
            # makes check C1.14 ("every task records engine == native")
            # possible at all.
            "engine",
            "build_hash",
            "isa_level",
            "avx512f",
            "compiler",
            "native_module_path",
            "native_module_mtime",
            # Node identity, for the B5/B6 node-heterogeneity covariate.
            "cpu_model",
            "hostname",
            # Code identity. SP-1 treats a dirty tree as invalidating, which is
            # only checkable if the -dirty suffix is recorded at run time.
            "git_hash",
            "git_describe",
            "git_dirty",
            # Allocation identity, for the C1.11 memory profile.
            "slurm_job_id",
            "slurm_array_task_id",
            "mem_requested_gb",
        ],
    )
    def test_field_present(self, hardware: dict, field: str) -> None:
        assert field in hardware, f"{field} missing: EXECUTION-PLAN A7 is not satisfied"

    def test_preexisting_fields_retained(self, hardware: dict) -> None:
        """C1-era readers must keep working; A7 extends, it does not rename."""
        for field in (
            "cpu",
            "cpu_count",
            "ram_gb",
            "python_version",
            "platform",
            "os",
            "conda_env",
            "git_hash",
            "timestamp",
        ):
            assert field in hardware

    def test_engine_is_a_known_value(self, hardware: dict) -> None:
        """C1.14 asserts equality against a fixed string, so the vocabulary is
        normalised here: the backend registry says ``cpp``, the plan says
        ``native``."""
        assert hardware["engine"] in {"native", "python", "unknown"}

    def test_engine_reports_actual_dispatch_not_the_compiled_default(self) -> None:
        """B2's defect, guarded.

        Until 2026-07-31 the engine was read from the compiled-in default and
        the ``ISALSR_ENGINE`` override was bypassed, so a probe reported
        ``native`` whichever engine ran -- passing while proving nothing. The
        capture must follow the override.
        """
        pytest.importorskip("isalsr.core.backends")
        previous = os.environ.get("ISALSR_ENGINE")
        os.environ["ISALSR_ENGINE"] = "python"
        try:
            assert collect_hardware_info()["engine"] == "python"
        finally:
            if previous is None:
                os.environ.pop("ISALSR_ENGINE", None)
            else:
                os.environ["ISALSR_ENGINE"] = previous

    def test_peak_rss_is_plausible(self) -> None:
        """Guards the unit bug this module shipped with once: ``ru_maxrss`` is
        kilobytes on Linux, so dividing by 1024 yields MB, and a fresh process
        reported "163 GB"."""
        peak = peak_rss_gb()
        assert 0.0 < peak < 16.0, f"{peak} GB for a test process is not a GB reading"


# ====================================================================== #
# P3 / C4 -- cross-arm data identity
# ====================================================================== #


class TestDataFingerprint:
    """P3: the paired design's premise, made checkable."""

    @staticmethod
    def _arrays(seed: int = 0):
        rng = np.random.default_rng(seed)
        return (
            rng.normal(size=(40, 3)),
            rng.normal(size=40),
            rng.normal(size=(10, 3)),
            rng.normal(size=10),
        )

    def test_deterministic(self) -> None:
        arrays = self._arrays()
        assert data_fingerprint(*arrays) == data_fingerprint(*arrays)

    def test_is_a_sha256(self) -> None:
        digest = data_fingerprint(*self._arrays())
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_different_seeds_differ(self) -> None:
        assert data_fingerprint(*self._arrays(0)) != data_fingerprint(*self._arrays(1))

    def test_detects_a_single_perturbed_element(self) -> None:
        """The failure mode is subtle drift, not a wholesale swap."""
        x_train, y_train, x_test, y_test = self._arrays()
        perturbed = y_train.copy()
        perturbed[7] += 1e-12
        assert data_fingerprint(x_train, y_train, x_test, y_test) != data_fingerprint(
            x_train, perturbed, x_test, y_test
        )

    def test_insensitive_to_container_dtype_and_contiguity(self) -> None:
        """The certified object is the sample, not the container carrying it."""
        x_train, y_train, x_test, y_test = self._arrays()
        reference = data_fingerprint(x_train, y_train, x_test, y_test)
        non_contiguous = np.asfortranarray(x_train)
        as_list = y_train.tolist()
        assert data_fingerprint(non_contiguous, as_list, x_test, y_test) == reference

    def test_train_test_swap_is_detected(self) -> None:
        """Position matters: the four arrays enter under fixed names."""
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
        u, v = rng.normal(size=10), rng.normal(size=10)
        assert data_fingerprint(a, u, b, v) != data_fingerprint(b, v, a, u)

    def test_shape_is_committed_separately(self) -> None:
        """``(1000, 1)`` and ``(1000,)`` share bytes and are different data."""
        rng = np.random.default_rng(4)
        flat = rng.normal(size=12)
        column = flat.reshape(12, 1)
        other = rng.normal(size=3)
        assert data_fingerprint(column, other, column, other) != data_fingerprint(
            flat, other, flat, other
        )

    def test_rejects_unfingerprintable_input(self) -> None:
        with pytest.raises(ValueError, match="cannot be cast"):
            data_fingerprint("not an array", [1.0], [[1.0]], [1.0])


class TestConfigDigest:
    def test_content_addressed(self, tmp_path) -> None:
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text("experiment:\n  method: udfs\n")
        second.write_text("experiment:\n  method: udfs\n")
        assert config_sha256(first) == config_sha256(second)

    def test_edit_changes_the_digest(self, tmp_path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("max_time: 43200\n")
        before = config_sha256(path)
        path.write_text("max_time: 900\n")
        assert config_sha256(path) != before

    def test_missing_file_never_raises(self, tmp_path) -> None:
        """Provenance capture must not be the reason a 12 h run fails to start."""
        assert config_sha256(tmp_path / "absent.yaml") == "unavailable"


# ====================================================================== #
# C1.9-BUG -- the five fallback paths reach the run log
# ====================================================================== #


def _make_run_log(**search_space_overrides) -> RunLog:
    """Minimal RunLog for serialisation round-trip tests."""
    return RunLog(
        metadata=RunMetadata(
            method="bingo",
            representation="isalsr",
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=1,
            data_fingerprint="a" * 64,
            config_sha256="b" * 64,
        ),
        regression=RegressionResults(
            r2_train=1.0,
            r2_test=1.0,
            nrmse_train=0.0,
            nrmse_test=0.0,
            mse_test=0.0,
            solution_recovered=True,
            jaccard_index=1.0,
            model_complexity=3,
        ),
        time=TimeResults(
            wall_clock_total_s=1.0,
            wall_clock_search_only_s=1.0,
            canonicalization_precomputed_s=0.0,
            canonicalization_runtime_s=0.0,
            cache_hit_rate=0.0,
            cache_hits=0,
            cache_misses=0,
            estimated_time_saved_s=0.0,
            time_to_r2_099_s=None,
            time_to_r2_0999_s=None,
            evaluation_time_s=1.0,
            overhead_time_s=0.0,
        ),
        search_space=SearchSpaceResults(
            total_dags_explored=10,
            unique_canonical_dags=5,
            empirical_reduction_factor=2.0,
            max_internal_nodes_seen=4,
            theoretical_reduction_bound=24.0,
            redundancy_rate=0.5,
            **search_space_overrides,
        ),
        best_expression=BestExpression(
            symbolic_form="x_0",
            isalsr_string="",
            canonical_string="",
            n_nodes=1,
            n_edges=0,
        ),
    )


#: The five fallback paths of T06, plus the denominators without which a count
#: is not a rate, plus the atlas partition that makes the rates interpretable.
LEDGER_FIELDS = (
    "ledger_enabled",
    "ledger_sample_rate",
    "n_ledger_seen",
    "n_ledger_sampled",
    "n_violations_pre",
    "n_violations_post",
    "n_canon_timeouts",
    "n_conversion_failures",
    "n_canon_raised",
    "n_atlas_hits",
)


class TestFallbackFieldsReachTheRunLog:
    """C1.9-BUG: a walk of a probe run_log.json found no reachability field."""

    def test_ledger_exports_every_field(self) -> None:
        ledger = FallbackLedger()
        exported = ledger.to_search_space_fields()
        assert set(exported) == set(LEDGER_FIELDS)

    def test_exported_names_are_accepted_by_the_schema(self) -> None:
        """The mapping and the dataclass must not drift apart."""
        run_log = _make_run_log(**FallbackLedger().to_search_space_fields())
        for field in LEDGER_FIELDS:
            assert getattr(run_log.search_space, field) is not None

    @pytest.mark.parametrize("field", LEDGER_FIELDS)
    def test_field_survives_serialisation(self, field: str) -> None:
        """The pre-flight check parses JSON, not a live dataclass."""
        run_log = _make_run_log(**FallbackLedger().to_search_space_fields())
        payload = json.loads(json.dumps(run_log.to_dict()))
        assert field in payload["results"]["search_space"]

    def test_counts_are_carried_faithfully(self) -> None:
        ledger = FallbackLedger()
        ledger.enabled = True
        ledger.n_seen = 476
        ledger.n_sampled = 476
        ledger.violated_pre = 476
        ledger.violated_post = 0
        ledger.timeout = 2
        ledger.conversion_failure = 1
        ledger.canon_raised = 3
        ledger.atlas_hit = 40
        exported = ledger.to_search_space_fields()
        assert exported["n_ledger_seen"] == 476
        assert exported["n_violations_pre"] == 476
        assert exported["n_violations_post"] == 0
        assert exported["n_canon_timeouts"] == 2
        assert exported["n_conversion_failures"] == 1
        assert exported["n_canon_raised"] == 3
        assert exported["n_atlas_hits"] == 40

    def test_dead_ledger_is_distinguishable_from_a_zero_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SP-6, stated as an assertion.

        A zero-everywhere ledger means the counters are dead, not that the rates
        are zero. ``ledger_enabled`` plus the sampled denominator is what tells
        the two apart, and a campaign that cannot tell them apart has no
        evidence base for R1.2.

        The environment is pinned explicitly: ``FallbackLedger`` reads
        ``ISALSR_LEDGER_ENABLED`` at construction, other suites set it, and a
        test of the disabled state that silently ran enabled would assert
        nothing.
        """
        monkeypatch.delenv("ISALSR_LEDGER_ENABLED", raising=False)
        dead = _make_run_log(**FallbackLedger().to_search_space_fields())
        assert dead.search_space.ledger_enabled is False

        live = FallbackLedger()
        live.enabled = True
        live.n_seen = 1000
        live.n_sampled = 1000
        measured = _make_run_log(**live.to_search_space_fields())
        assert measured.search_space.ledger_enabled is True
        assert measured.search_space.n_ledger_sampled == 1000

    def test_absent_ledger_is_none_not_zero(self) -> None:
        """The baseline arm does not canonicalise. ``None`` says "not asked";
        ``0`` would say "asked and none occurred"."""
        baseline = _make_run_log()
        for field in LEDGER_FIELDS:
            assert getattr(baseline.search_space, field) is None


class TestRunMetadataThreeArms:
    """A7: the schema must accept the hash arm and carry the fingerprints."""

    @pytest.mark.parametrize("arm", ["baseline", "hash", "isalsr"])
    def test_round_trip(self, arm: str) -> None:
        original = _make_run_log()
        metadata = RunMetadata(
            method="udfs",
            representation=arm,
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=7,
            hardware={"engine": "native"},
            data_fingerprint="c" * 64,
            config_sha256="d" * 64,
        )
        restored = RunMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
        assert restored == metadata
        assert restored.representation == arm
        assert original.metadata.data_fingerprint == "a" * 64

    def test_pre_c2_log_still_loads(self) -> None:
        """C1 artefacts have neither fingerprint; they must not fail to parse."""
        legacy = {
            "method": "bingo",
            "representation": "baseline",
            "benchmark": "nguyen",
            "problem": "Nguyen-1",
            "seed": 3,
            "hardware": {},
            "hyperparameters": {},
        }
        restored = RunMetadata.from_dict(legacy)
        assert restored.data_fingerprint == ""
        assert restored.config_sha256 == ""


# ====================================================================== #
# P4 / §5.5 -- the anti-1,465 rule
# ====================================================================== #


class TestStatusLedger:
    """P4: every run leaves a record, including the ones killed from outside."""

    @staticmethod
    def _status(arm: str = "isalsr", seed: int = 1, **kwargs) -> RunStatus:
        defaults = {
            "method": "bingo",
            "arm": arm,
            "benchmark": "hard",
            "problem": "Korns-12",
            "seed": seed,
        }
        return RunStatus(**{**defaults, **kwargs})

    def test_write_then_load_round_trip(self, tmp_path) -> None:
        status = self._status(engine="native", data_fingerprint="e" * 64)
        write_status(status, tmp_path / "seed_01")
        restored = load_status(tmp_path / "seed_01" / "status.json")
        assert restored is not None
        assert restored.engine == "native"
        assert restored.data_fingerprint == "e" * 64

    def test_written_before_the_search_so_a_sigkill_leaves_a_row(self, tmp_path) -> None:
        """The C1 shortfall was 36 OOM kills. ``SIGKILL`` reaches no handler, so
        a try/except ledger records nothing for the dominant failure mode. The
        write-ahead row is the whole mechanism."""
        seed_directory = tmp_path / "seed_01"
        write_status(self._status(), seed_directory)
        # No terminal write happens -- this models the process being killed.
        survivor = load_status(seed_directory / "status.json")
        assert survivor is not None
        assert survivor.terminal_status == "started"

    def test_terminal_write_overwrites_the_started_row(self, tmp_path) -> None:
        seed_directory = tmp_path / "seed_01"
        status = self._status()
        write_status(status, seed_directory)
        status.terminal_status = "completed"
        status.exit_code = 0
        status.wall_clock_s = 42.0
        write_status(status, seed_directory)
        restored = load_status(seed_directory / "status.json")
        assert restored is not None
        assert restored.terminal_status == "completed"
        assert restored.wall_clock_s == 42.0

    def test_write_is_atomic_leaving_no_stray_temp_file(self, tmp_path) -> None:
        seed_directory = tmp_path / "seed_01"
        write_status(self._status(), seed_directory)
        assert not list(seed_directory.glob(".*tmp"))

    def test_corrupt_record_returns_none_rather_than_raising(self, tmp_path) -> None:
        """A corrupt record is a finding; it must not stop the reconciliation
        that would report it."""
        path = tmp_path / "status.json"
        path.write_text("{ this is not json")
        assert load_status(path) is None

    def test_missing_record_returns_none(self, tmp_path) -> None:
        assert load_status(tmp_path / "nope.json") is None

    def test_collect_writes_every_column(self, tmp_path) -> None:
        write_status(self._status(arm="baseline", seed=1), tmp_path / "a" / "seed_01")
        write_status(self._status(arm="isalsr", seed=1), tmp_path / "b" / "seed_01")
        rows = collect_status_ledger(tmp_path, tmp_path / "status_ledger.csv")
        assert len(rows) == 2
        with (tmp_path / "status_ledger.csv").open() as handle:
            written = list(csv.DictReader(handle))
        assert len(written) == 2
        assert set(written[0]) == set(LEDGER_COLUMNS)

    def test_collect_is_deterministic(self, tmp_path) -> None:
        """Two collections over one root must be byte-identical, or a diff
        between two audits is uninterpretable."""
        for seed in (3, 1, 2):
            write_status(self._status(seed=seed), tmp_path / f"s{seed}" / f"seed_{seed:02d}")
        first = collect_status_ledger(tmp_path, tmp_path / "one.csv")
        second = collect_status_ledger(tmp_path, tmp_path / "two.csv")
        assert [s.seed for s in first] == [s.seed for s in second] == [1, 2, 3]
        assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()

    def test_ledger_columns_match_the_dataclass(self) -> None:
        assert "extra" not in LEDGER_COLUMNS
        assert "terminal_status" in LEDGER_COLUMNS
        assert "max_rss_gb" in LEDGER_COLUMNS
        assert "data_fingerprint" in LEDGER_COLUMNS


class TestReconciliation:
    """C1.15 / E6: a mismatch must NAME the cells, not just count them."""

    @staticmethod
    def _rows() -> list[RunStatus]:
        def make(arm: str, seed: int, terminal: str) -> RunStatus:
            return RunStatus(
                method="bingo",
                arm=arm,
                benchmark="hard",
                problem="Vlad-2",
                seed=seed,
                terminal_status=terminal,  # type: ignore[arg-type]
            )

        return [
            make("baseline", 1, "completed"),
            make("isalsr", 1, "started"),  # killed from outside
            make("hash", 1, "failed"),
        ]

    def test_names_killed_and_failed_and_missing(self) -> None:
        expected = {
            ("bingo", "baseline", "Vlad-2", 1),
            ("bingo", "isalsr", "Vlad-2", 1),
            ("bingo", "hash", "Vlad-2", 1),
            ("bingo", "isalsr", "Vlad-2", 2),  # never started at all
        }
        report = reconcile(self._rows(), expected)
        assert report["n_expected"] == 4
        assert report["n_observed"] == 3
        assert report["n_completed"] == 1
        assert report["missing"] == [("bingo", "isalsr", "Vlad-2", 2)]
        assert report["killed"] == [("bingo", "isalsr", "Vlad-2", 1)]
        assert report["failed"] == [("bingo", "hash", "Vlad-2", 1)]
        assert report["reconciled"] is False

    def test_complete_campaign_reconciles(self) -> None:
        rows = [
            RunStatus(
                method="udfs",
                arm=arm,
                benchmark="nguyen",
                problem="Nguyen-1",
                seed=1,
                terminal_status="completed",
            )
            for arm in ("baseline", "hash", "isalsr")
        ]
        expected = {("udfs", arm, "Nguyen-1", 1) for arm in ("baseline", "hash", "isalsr")}
        report = reconcile(rows, expected)
        assert report["reconciled"] is True
        assert report["n_completed"] == 3
