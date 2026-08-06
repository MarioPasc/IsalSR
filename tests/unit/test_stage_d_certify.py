"""Unit tests for the Campaign-C2 Stage D certifier.

The property under test is not "does it pass clean data" -- that is cheap and
almost worthless. It is "does it FAIL damaged data, name the offender, and never
raise". Each damage test mutates exactly one thing on a synthetic 12-cell Stage D
root and asserts the specific criterion flips to FAIL.

The synthetic root is built from the locked registry in
``experiments.scripts.stage_d_task_spec`` rather than from a second enumeration,
so a change to the registry cannot leave the tests certifying a shape the
launcher no longer submits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.scripts.stage_d_certify import (
    DEFAULT_WALL_S,
    MEM_ROUND_STEP_GB,
    _parse_elapsed_to_s,
    _round_up_gb,
    certify,
    main,
)
from experiments.scripts.stage_d_task_spec import STAGE_D_CERTIFICATION_CELLS, StageDCell

# ---------------------------------------------------------------------- #
# Fixture parameters. Chosen so a CLEAN root passes every threshold with
# room to spare, and each damage test has to move exactly one number.
# ---------------------------------------------------------------------- #

#: 40,000 s of a 57,600 s wall leaves 30.6 % headroom (D1.1 wants >= 10 %).
CLEAN_ELAPSED_S = 40_000.0

#: Peak RSS as a fraction of the request. 0.40 leaves 60 % headroom
#: (D1.2 wants >= 30 %).
CLEAN_PEAK_FRACTION = 0.40

#: C1's published per-problem values, measured 2026-08-04 from
#: ``wl_subtree_unified/analysis``. Used to build the synthetic reference so the
#: D1.6 band arithmetic is exercised against real magnitudes.
C1_RHO_DELTA: dict[tuple[str, str], float] = {
    ("bingo", "Pagie-1"): 0.8338,
    ("bingo", "Korns-12"): 0.8214,
    ("bingo", "Vladislavleva-2"): 0.8320,
    ("udfs", "Pagie-1"): 0.7412,
    ("udfs", "Korns-12"): 0.2823,
    ("udfs", "Vladislavleva-2"): 0.3921,
}
C1_R2_DELTA: dict[tuple[str, str], float] = {
    ("bingo", "Pagie-1"): -0.0758,
    ("bingo", "Korns-12"): 0.0,
    ("bingo", "Vladislavleva-2"): 0.0576,
    ("udfs", "Pagie-1"): -0.0182,
    ("udfs", "Korns-12"): -0.0,
    ("udfs", "Vladislavleva-2"): 0.0076,
}

#: Stage D rho per arm. isalsr sits above every reconstructed C1 rho
#: (max 1 + 0.8338 = 1.8338) and above the hash arm, satisfying D1.5 and D1.6.
CLEAN_RHO: dict[str, float] = {"baseline": 1.0, "hash": 1.5, "isalsr": 2.1}

#: Stage D r2_test per arm. The paired delta is +0.02 everywhere, inside the
#: +/-0.15 band against every C1 delta above.
CLEAN_R2: dict[str, float] = {"baseline": 0.90, "hash": 0.91, "isalsr": 0.92}

#: One SLURM array id per submission group, mirroring one sbatch per group.
#: Used only by the array-style negative test; the clean fixture joins on the
#: RAW per-task id, which is what ``status.json`` actually records.
GROUP_ARRAY_IDS: dict[str, str] = {
    "udfs": "7001",
    "bingo_std": "7002",
    "bingo_isalsr": "7003",
}

#: Base of the raw, per-task SLURM job id. ``JobIDRaw`` is unique per task, so
#: the fixture must be too: a base-plus-group-index scheme collides across
#: groups and silently joins a cell's memory onto its neighbour.
RAW_JOB_ID_BASE = 7100


# ====================================================================== #
# Synthetic root builder
# ====================================================================== #


def _run_log(spec: StageDCell) -> dict[str, Any]:
    """Return a schema-complete run_log payload for one Stage D cell."""
    dedup = spec.arm in ("hash", "isalsr")
    rho = CLEAN_RHO[spec.arm]
    ledger: dict[str, Any] = {
        "ledger_enabled": True if dedup else None,
        "ledger_sample_rate": 16 if dedup else None,
        "n_ledger_seen": 100_000 if dedup else None,
        "n_ledger_sampled": 6_250 if dedup else None,
        "n_violations_pre": 3 if dedup else None,
        "n_violations_post": 0 if dedup else None,
        "n_canon_timeouts": 0 if dedup else None,
        "n_canon_raised": 0 if dedup else None,
        "n_atlas_hits": 42 if dedup else None,
        "n_conversion_failures": 0 if dedup else None,
        # Structural-scope exclusion (D3, 2026-08-06).
        "n_nonstructural": 7 if dedup else None,
    }
    return {
        "metadata": {
            "method": spec.method,
            "representation": spec.arm,
            "benchmark": spec.suite,
            "problem": spec.problem,
            "seed": spec.seed,
            "hardware": {"engine": "native", "cpu_model": "test"},
            "hyperparameters": {"max_time": 43200},
            "data_fingerprint": f"fp-{spec.problem}-{spec.seed}",
            "config_sha256": "cfg" * 10,
        },
        "results": {
            "regression": {
                "r2_train": CLEAN_R2[spec.arm] + 0.01,
                "r2_test": CLEAN_R2[spec.arm],
                "nrmse_train": 0.01,
                "nrmse_test": 0.02,
                "mse_test": 0.001,
                "solution_recovered": False,
                "jaccard_index": 0.8,
                "model_complexity": 11,
                "n_nonfinite_test_predictions": 0,
            },
            "time": {
                "wall_clock_total_s": CLEAN_ELAPSED_S,
                "wall_clock_search_only_s": 39_000.0,
                "canonicalization_precomputed_s": 0.0,
                "canonicalization_runtime_s": 2_000.0 if dedup else 0.0,
                "cache_hit_rate": 0.3,
                "cache_hits": 30,
                "cache_misses": 70,
                "estimated_time_saved_s": 12.0,
                "time_to_r2_099_s": None,
                "time_to_r2_0999_s": None,
                "evaluation_time_s": 39_000.0,
                "overhead_time_s": 3_000.0 if dedup else 0.0,
                "conversion_time_s": 1_000.0 if dedup else 0.0,
                "shadow_time_s": 500.0 if dedup else 0.0,
            },
            "search_space": {
                "total_dags_explored": 210_000,
                "unique_canonical_dags": int(210_000 / rho),
                "empirical_reduction_factor": rho,
                "max_internal_nodes_seen": 14,
                "theoretical_reduction_bound": 5040.0,
                "redundancy_rate": 1.0 - 1.0 / rho,
                "shadow_distinct_insertion": 190_000.0 if dedup else None,
                "shadow_distinct_topological": 180_000.0 if dedup else None,
                "shadow_distinct_topological_commutative": 170_000.0 if dedup else None,
                "shadow_distinct_host_native": 195_000.0 if dedup else None,
                "n_shadow_failures": 0 if dedup else None,
                "penalised_in_population_mean": 3.0 if dedup else 0.0,
                "penalised_in_population_max": 9.0 if dedup else 0.0,
                **ledger,
            },
        },
        "best_expression": {
            "symbolic_form": "x_0**2 + x_1",
            "isalsr_string": "V+NV*" if dedup else "",
            "canonical_string": "V+NV*" if dedup else "",
            "n_nodes": 9,
            "n_edges": 8,
        },
    }


def _raw_job_id(spec: StageDCell) -> str:
    """Return the unique raw SLURM job id the fixture assigns to a cell."""
    return str(RAW_JOB_ID_BASE + spec.index)


def _status(spec: StageDCell) -> dict[str, Any]:
    """Return a completed status.json payload for one Stage D cell."""
    return {
        "method": spec.method,
        "arm": spec.arm,
        "benchmark": spec.suite,
        "problem": spec.problem,
        "seed": spec.seed,
        "terminal_status": "completed",
        "exit_code": 0,
        "wall_clock_s": CLEAN_ELAPSED_S,
        "max_rss_gb": spec.mem_gb * CLEAN_PEAK_FRACTION,
        "node_cpu_model": "test",
        "hostname": "sr01",
        "engine": "native",
        "git_commit": "abc123",
        "config_sha256": "cfg" * 10,
        "data_fingerprint": f"fp-{spec.problem}-{spec.seed}",
        "slurm_job_id": _raw_job_id(spec),
        "slurm_array_task_id": str(spec.group_index),
        "n_nan_metrics": 0,
        "nan_fields": "",
        "exception_class": "",
        "exception_message": "",
        "started_at": "2026-08-04T00:00:00+00:00",
        "finished_at": "2026-08-04T11:06:40+00:00",
        "extra": {},
    }


def _write_trajectory(path: Path) -> None:
    """Write a monotone three-row trajectory.csv."""
    from experiments.models.schemas import TRAJECTORY_COLUMNS

    rows = [
        (0.0, 0, 0.10, 0.9, 1000, 800, "x", 3, 0.0),
        (20000.0, 500, 0.80, 0.2, 100000, 60000, "x", 9, 900.0),
        (39000.0, 999, 0.92, 0.02, 210000, 100000, "x", 11, 2000.0),
    ]
    lines = [",".join(TRAJECTORY_COLUMNS)]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rss(path: Path, peak_gb: float, n: int = 5) -> None:
    """Write an rss_timeseries.csv whose vmhwm high-water mark is ``peak_gb``."""
    lines = ["timestamp_s,vmrss_kb,vmhwm_kb"]
    hwm = 0.0
    for i in range(n):
        rss = peak_gb * (0.4 + 0.6 * (i + 1) / n)
        hwm = max(hwm, rss)
        lines.append(f"{i * 600},{int(rss * 1024**2)},{int(hwm * 1024**2)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_c1_reference(directory: Path) -> Path:
    """Write a synthetic campaign-C1 analysis directory.

    Only the two artefact families the certifier reads are produced:
    ``cross_problem_dominance_*`` for the per-problem deltas and
    ``three_axis_summary_*`` for the cohort mean that validates the rho
    reconstruction.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for method in ("bingo", "udfs"):
        problems = [p for (m, p) in C1_RHO_DELTA if m == method]
        rho_deltas = [C1_RHO_DELTA[(method, p)] for p in problems]
        r2_deltas = [C1_R2_DELTA[(method, p)] for p in problems]
        (directory / f"cross_problem_dominance_{method}_benchmark.json").write_text(
            json.dumps(
                {
                    "r2_test": {"problem_names": problems, "problem_deltas": r2_deltas},
                    "empirical_reduction_factor": {
                        "problem_names": problems,
                        "problem_deltas": rho_deltas,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / f"three_axis_summary_{method}_benchmark.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "search_space": {
                        "mean_reduction_factor": 1.0 + sum(rho_deltas) / len(rho_deltas)
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return directory


def _write_sacct(path: Path, peaks_gb: dict[str, float] | None = None) -> Path:
    """Write a ``JobID,MaxRSS`` export in the producer's exact shape.

    ``slurm/c2_stage_d/aggregate_worker.sh`` runs
    ``sacct -o JobIDRaw,MaxRSS`` and keeps only ``.batch`` steps, so every row's
    JobID carries a ``.batch`` suffix and the numeric part is the RAW id, not the
    ``<array_id>_<task>`` form. Both details are reproduced here.
    """
    peaks = peaks_gb or {}
    lines = ["JobID,MaxRSS"]
    for spec in STAGE_D_CERTIFICATION_CELLS:
        default = spec.mem_gb * CLEAN_PEAK_FRACTION
        gb = peaks.get(spec.label, default)
        lines.append(f"{_raw_job_id(spec)}.batch,{int(gb * 1024**2)}K")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    """Build a fully valid 12-cell synthetic Stage D root."""
    from experiments.models.status_ledger import LEDGER_COLUMNS

    root = tmp_path / "stage_d"
    ledger_rows: list[dict[str, Any]] = []
    for spec in STAGE_D_CERTIFICATION_CELLS:
        seed_dir = spec.run_dir(root)
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run_log.json").write_text(
            json.dumps(_run_log(spec), indent=2), encoding="utf-8"
        )
        # status.json records the RAW job id, which is what the JobIDRaw join
        # depends on; a JobID join would look for "<array>_<task>" and miss.
        status = _status(spec)
        (seed_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        _write_trajectory(seed_dir / "trajectory.csv")
        # The sampler peak is deliberately BELOW sacct's on the clean root, so
        # the max() in _peak_of is exercised in both directions across tests.
        _write_rss(seed_dir / "rss_timeseries.csv", spec.mem_gb * CLEAN_PEAK_FRACTION * 0.9)
        ledger_rows.append(status)

    lines = [",".join(LEDGER_COLUMNS)]
    for row in ledger_rows:
        lines.append(",".join(str(row[c]) for c in LEDGER_COLUMNS))
    (root / "status_ledger.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def c1_reference(tmp_path: Path) -> Path:
    """Build a synthetic campaign-C1 analysis directory."""
    return _write_c1_reference(tmp_path / "c1_analysis")


@pytest.fixture
def sacct_csv(tmp_path: Path) -> Path:
    """Build a clean sacct export for the 12 cells."""
    return _write_sacct(tmp_path / "stage_d_maxrss.csv")


# ====================================================================== #
# Helpers
# ====================================================================== #


def _run(
    root: Path,
    sacct: Path | None = None,
    reference: Path | None = None,
    wall_s: float = DEFAULT_WALL_S,
) -> dict[str, str]:
    """Certify a root and return ``{criterion_id: status}``."""
    results = certify(root=root, sacct_csv=sacct, c1_reference=reference, wall_s=wall_s)
    return {r.id: r.status for r in results}


def _detail(
    root: Path,
    criterion: str,
    sacct: Path | None = None,
    reference: Path | None = None,
    wall_s: float = DEFAULT_WALL_S,
) -> dict[str, Any]:
    """Certify a root and return one criterion's detail mapping."""
    results = certify(root=root, sacct_csv=sacct, c1_reference=reference, wall_s=wall_s)
    return next(r.detail for r in results if r.id == criterion)


def _cell(problem: str, method: str, arm: str) -> StageDCell:
    """Return the registry entry for one cell."""
    return next(
        c
        for c in STAGE_D_CERTIFICATION_CELLS
        if c.problem == problem and c.method == method and c.arm == arm
    )


def _patch_run_log(root: Path, spec: StageDCell, mutate: Any) -> None:
    """Apply ``mutate`` to one cell's run_log payload and write it back."""
    path = spec.run_dir(root) / "run_log.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ====================================================================== #
# Clean root
# ====================================================================== #


def test_clean_root_passes_every_criterion(
    clean_root: Path, sacct_csv: Path, c1_reference: Path
) -> None:
    results = certify(
        root=clean_root, sacct_csv=sacct_csv, c1_reference=c1_reference, wall_s=DEFAULT_WALL_S
    )
    for r in results:
        assert r.status == "PASS", (
            f"{r.id} failed on a clean root: observed={r.observed!r} "
            f"detail={json.dumps(r.detail, default=str)[:2000]}"
        )


def test_every_criterion_emits_a_verdict(clean_root: Path, c1_reference: Path) -> None:
    statuses = _run(clean_root, reference=c1_reference)
    assert set(statuses) == {f"D1.{i}" for i in range(1, 9)}
    assert all(v in {"PASS", "FAIL", "SKIP"} for v in statuses.values())


def test_all_twelve_registry_cells_are_certified(clean_root: Path) -> None:
    detail = _detail(clean_root, "D1.1")
    assert len(detail["per_cell"]) == len(STAGE_D_CERTIFICATION_CELLS) == 12


# ====================================================================== #
# D1.1 -- wall-clock headroom
# ====================================================================== #


def test_short_wall_headroom_fails_d1_1(clean_root: Path) -> None:
    # 95 % of the wall leaves 5 % headroom, below the 10 % the criterion wants.
    spec = _cell("Korns-12", "bingo", "isalsr")
    path = spec.run_dir(clean_root) / "status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["wall_clock_s"] = DEFAULT_WALL_S * 0.95
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _run(clean_root)["D1.1"] == "FAIL"
    detail = _detail(clean_root, "D1.1")
    assert detail["headroom_below_threshold"]["count"] == 1
    assert spec.label in detail["headroom_below_threshold"]["examples"][0]
    assert detail["min_headroom_frac"] == pytest.approx(0.05, abs=1e-6)


def test_missing_cell_is_named_by_d1_1_and_d1_3(clean_root: Path) -> None:
    spec = _cell("Vladislavleva-2", "bingo", "hash")
    for name in ("run_log.json", "status.json", "trajectory.csv", "rss_timeseries.csv"):
        (spec.run_dir(clean_root) / name).unlink()
    spec.run_dir(clean_root).rmdir()

    statuses = _run(clean_root)
    assert statuses["D1.1"] == "FAIL"
    assert statuses["D1.3"] == "FAIL"
    assert spec.label in json.dumps(_detail(clean_root, "D1.1")["no_run_directory"])


def test_sacct_elapsed_column_is_preferred_over_status_json(
    clean_root: Path, tmp_path: Path
) -> None:
    path = tmp_path / "sacct_elapsed.csv"
    lines = ["JobID,MaxRSS,Elapsed"]
    for spec in STAGE_D_CERTIFICATION_CELLS:
        gb = spec.mem_gb * CLEAN_PEAK_FRACTION
        lines.append(f"{_raw_job_id(spec)}.batch,{int(gb * 1024**2)}K,04:00:00")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail = _detail(clean_root, "D1.1", sacct=path)
    assert {row["elapsed_source"] for row in detail["per_cell"]} == {"sacct:Elapsed"}
    assert {row["elapsed_s"] for row in detail["per_cell"]} == {14400.0}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("04:00:00", 14400.0),
        ("1-02:03:04", 93784.0),
        ("12:34", 754.0),
        ("00:00:30.500", 30.5),
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_elapsed_variants(raw: str, expected: float | None) -> None:
    assert _parse_elapsed_to_s(raw) == expected


# ====================================================================== #
# D1.2 -- memory and the production recommendation
# ====================================================================== #


def test_maxrss_at_95_percent_of_request_fails_d1_2(clean_root: Path, tmp_path: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    sacct = _write_sacct(tmp_path / "hot.csv", peaks_gb={spec.label: spec.mem_gb * 0.95})
    assert _run(clean_root, sacct=sacct)["D1.2"] == "FAIL"
    detail = _detail(clean_root, "D1.2", sacct=sacct)
    assert detail["headroom_below_threshold"]["count"] == 1
    assert spec.label in detail["headroom_below_threshold"]["examples"][0]


def test_rss_timeseries_peak_beats_a_lower_sacct_value(clean_root: Path, tmp_path: Path) -> None:
    spec = _cell("Pagie-1", "udfs", "isalsr")
    _write_rss(spec.run_dir(clean_root) / "rss_timeseries.csv", peak_gb=11.0)
    sacct = _write_sacct(tmp_path / "low.csv", peaks_gb={spec.label: 2.0})

    row = next(
        r for r in _detail(clean_root, "D1.2", sacct=sacct)["per_cell"] if r["cell"] == spec.label
    )
    assert row["peak_source"] == "rss_timeseries:vmhwm_kb"
    assert row["peak_gb"] == pytest.approx(11.0, rel=1e-3)


def test_sacct_maxrss_beats_a_lower_timeseries_peak(clean_root: Path, tmp_path: Path) -> None:
    spec = _cell("Pagie-1", "udfs", "isalsr")
    _write_rss(spec.run_dir(clean_root) / "rss_timeseries.csv", peak_gb=2.0)
    sacct = _write_sacct(tmp_path / "high.csv", peaks_gb={spec.label: 9.0})

    row = next(
        r for r in _detail(clean_root, "D1.2", sacct=sacct)["per_cell"] if r["cell"] == spec.label
    )
    assert row["peak_source"] == "sacct:MaxRSS"
    assert row["peak_gb"] == pytest.approx(9.0, rel=1e-3)


def test_missing_rss_timeseries_degrades_gracefully(
    clean_root: Path, sacct_csv: Path, c1_reference: Path
) -> None:
    spec = _cell("Korns-12", "bingo", "baseline")
    (spec.run_dir(clean_root) / "rss_timeseries.csv").unlink()

    results = certify(root=clean_root, sacct_csv=sacct_csv, c1_reference=c1_reference)
    statuses = {r.id: r.status for r in results}
    # sacct still supplies a peak for that cell, so the criterion still passes,
    # but the missing sampler file is reported rather than swallowed.
    assert statuses["D1.2"] == "PASS"
    detail = next(r.detail for r in results if r.id == "D1.2")
    assert detail["missing_or_unusable_rss_timeseries"]["count"] == 1
    assert spec.label in detail["missing_or_unusable_rss_timeseries"]["examples"][0]


def test_no_memory_evidence_at_all_fails_d1_2(clean_root: Path) -> None:
    for spec in STAGE_D_CERTIFICATION_CELLS:
        (spec.run_dir(clean_root) / "rss_timeseries.csv").unlink()
    # No --sacct-csv either: there is now no memory evidence anywhere.
    assert _run(clean_root)["D1.2"] == "FAIL"
    assert _detail(clean_root, "D1.2")["unmeasured_cells"]["count"] == 12


def test_sacct_csv_join_uses_jobidraw_batch_semantics(clean_root: Path, sacct_csv: Path) -> None:
    detail = _detail(clean_root, "D1.2", sacct=sacct_csv)
    join = detail["sacct_join"]
    assert join["n_rows"] == 12
    assert join["n_matched"] == 12
    assert join["n_unmatched"] == 0
    assert join["errors"] == []
    # The trap is documented in the emitted evidence, not only in the code.
    assert "JobIDRaw" in join["trap_note"]
    assert "sacct -X" in join["trap_note"]


def test_array_style_jobid_does_not_join(clean_root: Path, tmp_path: Path) -> None:
    # This is the 42-of-1,260 failure mode: joining on JobID, which reads
    # "<array_id>_<task>" for an array while status.json holds the raw id.
    path = tmp_path / "array_style.csv"
    lines = ["JobID,MaxRSS"]
    for spec in STAGE_D_CERTIFICATION_CELLS:
        gb = spec.mem_gb * CLEAN_PEAK_FRACTION
        array_id = GROUP_ARRAY_IDS[spec.group]
        lines.append(f"{array_id}_{spec.group_index}.batch,{int(gb * 1024**2)}K")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    join = _detail(clean_root, "D1.2", sacct=path)["sacct_join"]
    assert join["n_matched"] == 0
    assert join["n_unmatched"] == 12


def test_unreadable_sacct_csv_is_reported_not_raised(clean_root: Path, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    join = _detail(clean_root, "D1.2", sacct=missing)["sacct_join"]
    assert join["source"] == "absent"
    assert "missing" in join["errors"][0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.1, 8),
        (8.0, 8),
        (8.1, 16),
        (100.0, 104),
        (145.7, 152),
        (float("nan"), 8),
        (-5.0, 8),
    ],
)
def test_round_up_gb_never_rounds_down(raw: float, expected: int) -> None:
    assert _round_up_gb(raw) == expected
    assert _round_up_gb(raw) % MEM_ROUND_STEP_GB == 0
    if raw == raw and raw > 0:  # noqa: PLR0124 - NaN guard
        assert _round_up_gb(raw) >= raw


def test_production_recommendation_is_emitted_per_method_arm(
    clean_root: Path, sacct_csv: Path
) -> None:
    detail = _detail(clean_root, "D1.2", sacct=sacct_csv)
    rows = {r["group"]: r for r in detail["production_recommendation_by_method_arm"]}
    expected_groups = {(c.method, c.arm) for c in STAGE_D_CERTIFICATION_CELLS}
    assert set(rows) == {f"{m}/{a}" for m, a in expected_groups}
    row = rows["bingo/isalsr"]
    # peak = 256 x 0.40 = 102.4 GB -> 102.4 / 0.70 = 146.3 -> 152 GB.
    assert row["requested_gb"] == 256
    assert row["peak_gb"] == pytest.approx(102.4, rel=1e-3)
    assert row["recommended_gb"] == 152
    assert row["margin_frac"] == pytest.approx((152 - 102.4) / 152, rel=1e-3)
    assert row["vmrss_p50_gb"] is not None
    assert row["vmrss_p95_gb"] is not None
    assert row["peak_source"] in ("sacct:MaxRSS", "rss_timeseries:vmhwm_kb")


def test_recommendation_is_emitted_even_when_part_a_fails(clean_root: Path, tmp_path: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    sacct = _write_sacct(tmp_path / "hot.csv", peaks_gb={spec.label: spec.mem_gb * 0.95})
    detail = _detail(clean_root, "D1.2", sacct=sacct)
    rows = {r["group"]: r for r in detail["production_recommendation_by_method_arm"]}
    assert rows["bingo/isalsr"]["recommended_gb"] is not None


# ====================================================================== #
# D1.3 -- artefact completeness
# ====================================================================== #


def test_missing_spec_field_fails_d1_3(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    _patch_run_log(clean_root, spec, lambda p: p["results"]["time"].pop("conversion_time_s"))

    assert _run(clean_root)["D1.3"] == "FAIL"
    named = json.dumps(_detail(clean_root, "D1.3")["run_log_violations"])
    assert "results.time.conversion_time_s" in named
    assert spec.label in named


def test_missing_shadow_time_field_fails_d1_3(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "udfs", "hash")
    _patch_run_log(clean_root, spec, lambda p: p["results"]["time"].pop("shadow_time_s"))
    assert "results.time.shadow_time_s" in json.dumps(
        _detail(clean_root, "D1.3")["run_log_violations"]
    )


def test_empty_trajectory_fails_d1_3(clean_root: Path) -> None:
    spec = _cell("Korns-12", "bingo", "hash")
    from experiments.models.schemas import TRAJECTORY_COLUMNS

    (spec.run_dir(clean_root) / "trajectory.csv").write_text(
        ",".join(TRAJECTORY_COLUMNS) + "\n", encoding="utf-8"
    )
    assert _run(clean_root)["D1.3"] == "FAIL"
    assert "empty" in json.dumps(_detail(clean_root, "D1.3")["trajectory_violations"])


def test_missing_ledger_row_fails_d1_3(clean_root: Path) -> None:
    path = clean_root / "status_ledger.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    assert _run(clean_root)["D1.3"] == "FAIL"
    assert _detail(clean_root, "D1.3")["cells_absent_from_status_ledger"]["count"] == 1


def test_absent_status_ledger_is_reported_not_raised(clean_root: Path) -> None:
    (clean_root / "status_ledger.csv").unlink()
    detail = _detail(clean_root, "D1.3")
    assert "missing" in detail["status_ledger_error"]
    assert detail["cells_absent_from_status_ledger"]["count"] == 12


# ====================================================================== #
# D1.4 -- the T08 AC-7 evidence
# ====================================================================== #


@pytest.mark.parametrize("problem", ["Korns-12", "Vladislavleva-2"])
@pytest.mark.parametrize("metric", ["r2_test", "r2_train"])
def test_nan_r2_on_nan_problem_fails_d1_4(clean_root: Path, problem: str, metric: str) -> None:
    spec = _cell(problem, "bingo", "isalsr")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["regression"].__setitem__(metric, float("nan"))
    )
    assert _run(clean_root)["D1.4"] == "FAIL"
    named = json.dumps(_detail(clean_root, "D1.4")["non_finite"])
    assert metric in named
    assert spec.label in named


def test_inf_r2_also_fails_d1_4(clean_root: Path) -> None:
    spec = _cell("Korns-12", "bingo", "isalsr")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["regression"].__setitem__("r2_test", float("inf")),
    )
    assert _run(clean_root)["D1.4"] == "FAIL"


def test_d1_4_scopes_to_bingo_isalsr_on_the_nan_problems(clean_root: Path) -> None:
    # A NaN elsewhere is a real defect but it is NOT what D1.4 certifies; it is
    # caught by the schema criterion instead.
    spec = _cell("Pagie-1", "udfs", "baseline")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["regression"].__setitem__("r2_test", float("nan")),
    )
    assert _run(clean_root)["D1.4"] == "PASS"
    assert len(_detail(clean_root, "D1.4")["per_cell"]) == 2


# ====================================================================== #
# D1.5 -- rho ordering
# ====================================================================== #


def test_rho_hash_above_rho_isalsr_fails_d1_5(clean_root: Path) -> None:
    spec = _cell("Vladislavleva-2", "bingo", "hash")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["search_space"].__setitem__("empirical_reduction_factor", 9.9),
    )
    assert _run(clean_root)["D1.5"] == "FAIL"
    named = json.dumps(_detail(clean_root, "D1.5")["violations"])
    assert "Vladislavleva-2" in named
    assert "rho_hash=9.900000" in named


def test_equal_rho_is_ordered_and_passes_d1_5(clean_root: Path) -> None:
    spec = _cell("Korns-12", "bingo", "hash")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["search_space"].__setitem__(
            "empirical_reduction_factor", CLEAN_RHO["isalsr"]
        ),
    )
    assert _run(clean_root)["D1.5"] == "PASS"


def test_d1_5_matches_on_method_and_problem(clean_root: Path) -> None:
    rows = _detail(clean_root, "D1.5")["per_pair"]
    assert len(rows) == 4  # 3 Bingo problems + 1 UDFS problem
    assert all(r["ordered"] for r in rows)


# ====================================================================== #
# D1.6 -- the C1 neighbourhood
# ====================================================================== #


def test_missing_c1_reference_skips_without_blocking(clean_root: Path, tmp_path: Path) -> None:
    results = certify(
        root=clean_root, sacct_csv=None, c1_reference=tmp_path / "nope", wall_s=DEFAULT_WALL_S
    )
    d16 = next(r for r in results if r.id == "D1.6")
    assert d16.status == "SKIP"
    assert d16.blocking is False
    assert sum(1 for r in results if r.blocking and r.status != "PASS") == 0


def test_c1_reference_none_skips_without_blocking(clean_root: Path) -> None:
    results = certify(root=clean_root, sacct_csv=None, c1_reference=None)
    d16 = next(r for r in results if r.id == "D1.6")
    assert d16.status == "SKIP"
    assert "--c1-reference not supplied" in json.dumps(d16.detail["errors"])


def test_rho_reconstruction_crosscheck_is_reported_and_validates(
    clean_root: Path, c1_reference: Path
) -> None:
    detail = _detail(clean_root, "D1.6", reference=c1_reference)
    checks = detail["rho_reconstruction_crosscheck"]
    assert checks["bingo"]["validated"] is True
    assert checks["udfs"]["validated"] is True
    assert checks["bingo"]["published_mean_reduction_factor"] is not None


def test_rho_drop_versus_c1_fails_d1_6(clean_root: Path, c1_reference: Path) -> None:
    # C1 Bingo/Pagie-1 rho = 1.8338; 0.5 is far below the 10 % drop floor.
    spec = _cell("Pagie-1", "bingo", "isalsr")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["search_space"].__setitem__("empirical_reduction_factor", 0.5),
    )
    assert _run(clean_root, reference=c1_reference)["D1.6"] == "FAIL"
    detail = _detail(clean_root, "D1.6", reference=c1_reference)
    assert detail["rho_excursions"]["count"] == 1
    row = next(
        r for r in detail["rho_comparisons"] if r["method"] == "bingo" and r["problem"] == "Pagie-1"
    )
    assert row["verdict"].startswith("FAIL")
    assert "decomposition is not reaching the canonicaliser" in row["explanation"]


def test_small_rho_drop_inside_tolerance_passes_and_is_explained(
    clean_root: Path, c1_reference: Path
) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["search_space"].__setitem__(
            "empirical_reduction_factor", 1.8338 * 0.95
        ),
    )
    assert _run(clean_root, reference=c1_reference)["D1.6"] == "PASS"
    row = next(
        r
        for r in _detail(clean_root, "D1.6", reference=c1_reference)["rho_comparisons"]
        if r["method"] == "bingo" and r["problem"] == "Pagie-1"
    )
    assert row["verdict"].startswith("PASS")
    assert "single-seed tolerance" in row["explanation"]


def test_r2_exceeding_c1_fails_d1_6_and_flags_the_direction(
    clean_root: Path, c1_reference: Path
) -> None:
    # C1 Bingo/Korns-12 delta = 0.0; pushing the isalsr arm to 1.0 against a
    # 0.90 baseline gives delta = +0.10... still inside the band, so go further.
    spec = _cell("Korns-12", "bingo", "baseline")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["regression"].__setitem__("r2_test", 0.50)
    )
    assert _run(clean_root, reference=c1_reference)["D1.6"] == "FAIL"
    row = next(
        r
        for r in _detail(clean_root, "D1.6", reference=c1_reference)["r2_comparisons"]
        if r["method"] == "bingo" and r["problem"] == "Korns-12"
    )
    assert row["verdict"] == "FAIL (C2 materially exceeds C1)"
    assert "dataset, the split and the metric" in row["explanation"]
    assert row["excess"] == pytest.approx(0.42, abs=1e-6)


def test_r2_below_c1_fails_d1_6_with_the_opposite_direction(
    clean_root: Path, c1_reference: Path
) -> None:
    spec = _cell("Korns-12", "bingo", "isalsr")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["regression"].__setitem__("r2_test", 0.50)
    )
    row = next(
        r
        for r in _detail(clean_root, "D1.6", reference=c1_reference)["r2_comparisons"]
        if r["method"] == "bingo" and r["problem"] == "Korns-12"
    )
    assert row["verdict"] == "FAIL (C2 materially below C1)"


def test_d1_6_report_carries_the_engine_and_t16_explanations(
    clean_root: Path, c1_reference: Path
) -> None:
    definition = _detail(clean_root, "D1.6", reference=c1_reference)["neighbourhood_definition"]
    assert "NOT attributable to the engine" in definition["engine_note"]
    assert "22 %" in definition["rho_rationale"]
    assert "no budget asymmetry" in definition["r2_rationale"]
    assert "0.10" in definition["rho_rule"]
    assert "0.15" in definition["r2_rule"]


# ====================================================================== #
# D1.7 -- timing under the new accounting
# ====================================================================== #


def test_absent_t_canon_fails_d1_7(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["time"].pop("canonicalization_runtime_s")
    )
    assert _run(clean_root)["D1.7"] == "FAIL"
    assert "canonicalization_runtime_s" in json.dumps(
        _detail(clean_root, "D1.7")["missing_or_nonfinite_fields"]
    )


def test_zero_t_canon_on_a_dedup_arm_fails_d1_7(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "hash")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["time"].__setitem__("canonicalization_runtime_s", 0.0),
    )
    assert _run(clean_root)["D1.7"] == "FAIL"
    assert _detail(clean_root, "D1.7")["zero_T_canon_on_dedup_arm"]["count"] == 1


def test_zero_t_eval_fails_d1_7(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "udfs", "baseline")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["time"].__setitem__("evaluation_time_s", 0.0)
    )
    assert _run(clean_root)["D1.7"] == "FAIL"
    assert _detail(clean_root, "D1.7")["zero_T_eval"]["count"] == 1


def test_zero_t_canon_on_the_baseline_arm_is_correct_not_a_failure(clean_root: Path) -> None:
    # The baseline arm never canonicalises, so T_canon == 0 is the expected
    # value there and must not fail the criterion.
    assert _run(clean_root)["D1.7"] == "PASS"
    row = next(
        r
        for r in _detail(clean_root, "D1.7")["per_cell"]
        if r["cell"] == _cell("Pagie-1", "bingo", "baseline").label
    )
    assert row["T_canon_s"] == 0.0


def test_overhead_counts_conversion_and_reports_shadow_separately(clean_root: Path) -> None:
    detail = _detail(clean_root, "D1.7")
    row = next(
        r for r in detail["per_cell"] if r["cell"] == _cell("Pagie-1", "bingo", "isalsr").label
    )
    # canon 2000 + conversion 1000 = 3000 over 39000 s of evaluation.
    assert row["overhead_s_computed"] == pytest.approx(3000.0)
    assert row["overhead_pct_of_eval"] == pytest.approx(100 * 3000 / 39000, abs=1e-3)
    # The canon-only figure is reported alongside, which is what makes the
    # accounting change legible instead of looking like a regression: the
    # overhead figure is strictly the larger of the two by construction.
    assert row["canon_only_pct_of_eval"] == pytest.approx(100 * 2000 / 39000, abs=1e-3)
    assert row["overhead_pct_of_eval"] > row["canon_only_pct_of_eval"]
    assert row["shadow_pct_of_eval"] == pytest.approx(100 * 500 / 39000, abs=1e-3)
    assert row["reported_matches_computed"] is True
    assert "ACCOUNTING CHANGE, not a regression" in detail["accounting_note"]


def test_overhead_mismatch_between_reported_and_computed_is_disclosed(clean_root: Path) -> None:
    spec = _cell("Pagie-1", "bingo", "isalsr")
    _patch_run_log(
        clean_root, spec, lambda p: p["results"]["time"].__setitem__("overhead_time_s", 2000.0)
    )
    row = next(r for r in _detail(clean_root, "D1.7")["per_cell"] if r["cell"] == spec.label)
    assert row["reported_matches_computed"] is False


# ====================================================================== #
# D1.8 -- the pre-flight MANIFEST
# ====================================================================== #


def test_manifest_is_written_and_validates_non_strict(clean_root: Path) -> None:
    from experiments.models.manifest import load_manifest

    assert _run(clean_root)["D1.8"] == "PASS"
    path = clean_root / "c2_preflight" / "stage_d_manifest.json"
    assert path.exists()

    manifest = load_manifest(path)
    assert manifest.seeds == [101]
    assert sorted(manifest.arms) == ["baseline", "hash", "isalsr"]
    assert manifest.node_constraint == "sr"
    # One split per (method, arm): 2 methods x 3 arms.
    assert len(manifest.submission_splits) == 6
    assert sum(s.n_tasks for s in manifest.submission_splits) == len(STAGE_D_CERTIFICATION_CELLS)


def test_manifest_splits_are_internally_consistent(clean_root: Path) -> None:
    detail = _detail(clean_root, "D1.8")
    splits = detail["submission_splits"]
    assert [s["index"] for s in splits] == list(range(1, len(splits) + 1))
    for split in splits:
        assert split["n_tasks"] == split["n_problems"] * 1


def test_strict_validation_would_reject_the_preflight_manifest(clean_root: Path) -> None:
    # The scope note is not decoration: it is the reason strict_campaign=False
    # is correct here, and this test pins that the strict mode really does
    # reject a 12-cell, 1-seed manifest.
    from experiments.models.manifest import (
        ManifestValidationError,
        load_manifest,
        validate_manifest,
    )

    _run(clean_root)
    manifest = load_manifest(clean_root / "c2_preflight" / "stage_d_manifest.json")
    validate_manifest(manifest, strict_campaign=False)
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest, strict_campaign=True)


def test_truncated_manifest_fails_d1_8(clean_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.scripts.stage_d_certify as mod

    # Drop the config digests: a manifest that names no configuration carries no
    # provenance, and the validator says so in both modes.
    monkeypatch.setattr(mod, "_config_digests", lambda: ([], ["forced empty for test"]))
    assert _run(clean_root)["D1.8"] == "FAIL"
    problems = json.dumps(_detail(clean_root, "D1.8")["validation_problems"])
    assert "configs" in problems


def test_manifest_missing_operator_sets_fails_d1_8(
    clean_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.scripts.stage_d_certify as mod

    monkeypatch.setattr(mod, "_operator_sets", lambda: ({}, ["forced empty for test"]))
    assert _run(clean_root)["D1.8"] == "FAIL"
    problems = json.dumps(_detail(clean_root, "D1.8")["validation_problems"])
    assert "bingo_operators" in problems


# ====================================================================== #
# CLI
# ====================================================================== #


def test_main_writes_both_reports_and_exits_zero_on_a_clean_root(
    clean_root: Path, sacct_csv: Path, c1_reference: Path, tmp_path: Path
) -> None:
    out_json = tmp_path / "out" / "stage_d_certification.json"
    out_md = tmp_path / "out" / "stage_d_certification.md"
    code = main(
        [
            "--root",
            str(clean_root),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--sacct-csv",
            str(sacct_csv),
            "--c1-reference",
            str(c1_reference),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "GO"
    assert payload["n_blocking_failures"] == 0
    assert payload["n_cells"] == 12
    assert payload["wall_s"] == DEFAULT_WALL_S
    assert set(f"D1.{i}" for i in range(1, 9)) <= set(payload)

    report = out_md.read_text(encoding="utf-8")
    assert "Stage D certification" in report
    assert "NOT attributable to the engine" in report
    assert "ACCOUNTING CHANGE" in report


def test_main_exits_one_on_a_blocking_failure(
    clean_root: Path, sacct_csv: Path, tmp_path: Path
) -> None:
    spec = _cell("Korns-12", "bingo", "isalsr")
    _patch_run_log(
        clean_root,
        spec,
        lambda p: p["results"]["regression"].__setitem__("r2_test", float("nan")),
    )
    code = main(
        [
            "--root",
            str(clean_root),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-md",
            str(tmp_path / "o.md"),
            "--sacct-csv",
            str(sacct_csv),
        ]
    )
    assert code == 1
    payload = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "NO-GO"
    assert payload["D1.4"]["status"] == "FAIL"


def test_main_on_an_absent_root_fails_without_raising(tmp_path: Path) -> None:
    code = main(
        [
            "--root",
            str(tmp_path / "nothing_here"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-md",
            str(tmp_path / "o.md"),
        ]
    )
    assert code == 1
    payload = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert payload["D0"]["status"] == "FAIL"


def test_certify_never_raises_on_an_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    results = certify(root=root, sacct_csv=None, c1_reference=None)
    assert {r.id for r in results} == {f"D1.{i}" for i in range(1, 9)}
    blocking = [r for r in results if r.blocking]
    assert all(r.status == "FAIL" for r in blocking if r.id != "D1.8")
