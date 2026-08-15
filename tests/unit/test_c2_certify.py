"""Unit tests for the Campaign-C2 Stage C certifier.

The property under test is not "does it pass clean data" -- that is cheap and
almost worthless. It is "does it FAIL damaged data, name the offender, and never
raise". Each damage test mutates exactly one thing on a synthetic root and
asserts the specific criterion flips to FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.scripts.c2_certify import (
    ARMS,
    RUN_LOG_FIELD_SPEC,
    _parse_maxrss_to_gb,
    _percentile,
    certify,
    main,
)

METHODS_2 = ("udfs", "bingo")
SEEDS_3 = (0, 101, 102)
PROBLEM = "Nguyen-1"
BENCH = "nguyen"


# ====================================================================== #
# Synthetic root builder
# ====================================================================== #


def _run_log(method: str, arm: str, seed: int) -> dict[str, Any]:
    """Return a schema-complete run_log payload for one cell."""
    dedup = arm in ("hash", "isalsr")
    ledger: dict[str, Any] = {
        "ledger_enabled": True if dedup else None,
        "ledger_sample_rate": 16 if dedup else None,
        "n_ledger_seen": 1000 if dedup else None,
        "n_ledger_sampled": 64 if dedup else None,
        "n_violations_pre": 2 if dedup else None,
        "n_violations_post": 0 if dedup else None,
        "n_canon_timeouts": 0 if dedup else None,
        "n_canon_raised": 0 if dedup else None,
        "n_atlas_hits": 5 if dedup else None,
        "n_conversion_failures": 1 if dedup else None,
        # Structural-scope exclusion (D3, 2026-08-06): zero-internal-node
        # candidates, evaluated but never deduplicated.  None on the baseline
        # arm, which converts nothing.
        "n_nonstructural": 3 if dedup else None,
    }
    rho = {"baseline": 1.0, "hash": 1.4, "isalsr": 1.9}[arm]
    # T19 structural telemetry.  Present on all three arms -- that is the point
    # of it -- but the `unique` block is None on baseline, which holds no cache.
    # Values rise with the arm so a fixture that silently lost the block cannot
    # still look like a plausible three-arm comparison.
    mean_k = {"baseline": 5.5, "hash": 6.1, "isalsr": 6.8}[arm]
    complexity: dict[str, Any] = {
        "complexity_sampling_mode": "population" if method == "bingo" else "stream",
        "complexity_sample_rate": 25 if method == "bingo" else 31,
        "complexity_n_sampled": 500,
        "complexity_time_s": 0.02,
        "complexity_n_failures": 0,
        "complexity_mean_k": mean_k,
        "complexity_std_k": 1.2,
        "complexity_median_k": mean_k,
        "complexity_p90_k": mean_k + 2.0,
        "complexity_max_k": mean_k + 5.0,
        "complexity_mean_depth": 4.0,
        "complexity_median_depth": 4.0,
        "complexity_mean_edges": 6.0,
        "complexity_mean_n_op": mean_k - 1.0,
        "complexity_mean_n_const": 1.0,
        "complexity_mean_shared": 1.1,
        "complexity_mean_sharing_surplus": 1.3,
        "complexity_mean_nonlinear": 2.0,
        "complexity_mean_op_entropy": 1.5,
        "complexity_mean_max_in_degree": 2.0,
        "complexity_unique_n_sampled": 300 if dedup else None,
        "complexity_unique_mean_k": mean_k + 0.5 if dedup else None,
        "complexity_unique_mean_depth": 4.5 if dedup else None,
        "complexity_unique_mean_nonlinear": 2.2 if dedup else None,
        "complexity_unique_mean_op_entropy": 1.6 if dedup else None,
    }
    return {
        "metadata": {
            "method": method,
            "representation": arm,
            "benchmark": BENCH,
            "problem": PROBLEM,
            "seed": seed,
            "hardware": {"engine": "native", "cpu_model": "test"},
            "hyperparameters": {"max_time": 15},
            "data_fingerprint": f"fp-{PROBLEM}-{seed}",
            "config_sha256": "cfg" * 10,
        },
        "results": {
            "regression": {
                "r2_train": 0.99,
                "r2_test": 0.98,
                "nrmse_train": 0.01,
                "nrmse_test": 0.02,
                "mse_test": 0.001,
                "solution_recovered": True,
                "jaccard_index": 0.8,
                "model_complexity": 7,
                "n_nonfinite_test_predictions": 0,
            },
            "time": {
                "wall_clock_total_s": 20.0,
                "wall_clock_search_only_s": 15.0,
                "canonicalization_precomputed_s": 0.0,
                "canonicalization_runtime_s": 0.5 if dedup else 0.0,
                "cache_hit_rate": 0.3,
                "cache_hits": 3,
                "cache_misses": 7,
                "estimated_time_saved_s": 0.1,
                "time_to_r2_099_s": None,
                "time_to_r2_0999_s": None,
                "evaluation_time_s": 10.0,
                "overhead_time_s": 1.0,
                "conversion_time_s": 0.3 if dedup else 0.0,
                "shadow_time_s": 0.2 if dedup else 0.0,
            },
            "search_space": {
                "total_dags_explored": 190,
                "unique_canonical_dags": int(190 / rho),
                "empirical_reduction_factor": rho,
                "max_internal_nodes_seen": 6,
                "theoretical_reduction_bound": 720.0,
                "redundancy_rate": 1.0 - 1.0 / rho,
                "shadow_distinct_insertion": None,
                "shadow_distinct_topological": None,
                "shadow_distinct_topological_commutative": None,
                "shadow_distinct_host_native": None,
                "n_shadow_failures": None,
                "penalised_in_population_mean": 2.0 if dedup else 0.0,
                "penalised_in_population_max": 5.0 if dedup else 0.0,
                **ledger,
                **complexity,
            },
        },
        "best_expression": {
            "symbolic_form": "x_0**3 + x_0",
            "isalsr_string": "V+NV*" if dedup else "",
            "canonical_string": "V+NV*" if dedup else "",
            "n_nodes": 5,
            "n_edges": 4,
        },
    }


def _status(method: str, arm: str, seed: int) -> dict[str, Any]:
    """Return a completed status.json payload for one cell."""
    return {
        "method": method,
        "arm": arm,
        "benchmark": BENCH,
        "problem": PROBLEM,
        "seed": seed,
        "terminal_status": "completed",
        "exit_code": 0,
        "wall_clock_s": 20.0,
        "max_rss_gb": 1.5,
        "node_cpu_model": "test",
        "hostname": "testhost",
        "engine": "native",
        "git_commit": "abc123",
        "config_sha256": "cfg" * 10,
        "data_fingerprint": f"fp-{PROBLEM}-{seed}",
        "slurm_job_id": "9001",
        "slurm_array_task_id": str(seed),
        "n_nan_metrics": 0,
        "nan_fields": "",
        "exception_class": "",
        "exception_message": "",
        "started_at": "2026-08-03T00:00:00+00:00",
        "finished_at": "2026-08-03T00:00:20+00:00",
        "extra": {},
    }


def _trajectory(path: Path) -> None:
    """Write a monotone three-row trajectory.csv."""
    from experiments.models.schemas import TRAJECTORY_COLUMNS

    rows = [
        (0.0, 0, 0.10, 0.9, 10, 8, "x", 3, 0.0),
        (5.0, 1, 0.50, 0.5, 90, 60, "x", 4, 0.2),
        (15.0, 2, 0.98, 0.02, 190, 100, "x", 5, 0.3),
    ]
    lines = [",".join(TRAJECTORY_COLUMNS)]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate(path: Path, method: str, arm: str) -> None:
    """Write an aggregate.csv with one row per metric."""
    from experiments.models.analyzer.aggregation import METRIC_EXTRACTORS
    from experiments.models.schemas import AGGREGATE_COLUMNS

    lines = [",".join(AGGREGATE_COLUMNS)]
    for metric in METRIC_EXTRACTORS:
        lines.append(",".join([method, arm, BENCH, PROBLEM, metric] + ["0.0"] * 7))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_metric() -> dict[str, Any]:
    """Return one PairedStatsMetric payload."""
    return {
        "baseline_mean": 0.9,
        "baseline_std": 0.01,
        "isalsr_mean": 0.95,
        "isalsr_std": 0.01,
        "mean_diff": 0.05,
        "std_diff": 0.005,
        "shapiro_wilk_p": 0.5,
        "normality_assumed": True,
        "test_used": "paired_t",
        "statistic": 2.0,
        "p_value_raw": 0.25,
        "p_value_holm": 0.25,
        "cohens_d": 1.0,
        "cohens_d_ci_lower": 0.1,
        "cohens_d_ci_upper": 2.0,
        "mean_diff_ci_lower": 0.0,
        "mean_diff_ci_upper": 0.1,
    }


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    """Build a fully valid 18-cell synthetic Stage C root."""
    from experiments.models.status_ledger import LEDGER_COLUMNS

    root = tmp_path / "root"
    ledger_rows: list[dict[str, Any]] = []
    for method in METHODS_2:
        problem_dir = root / method / BENCH / "nguyen_1"
        for arm in ARMS:
            for seed in SEEDS_3:
                sd = problem_dir / arm / f"seed_{seed:02d}"
                sd.mkdir(parents=True, exist_ok=True)
                (sd / "run_log.json").write_text(
                    json.dumps(_run_log(method, arm, seed), indent=2), encoding="utf-8"
                )
                status = _status(method, arm, seed)
                (sd / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                _trajectory(sd / "trajectory.csv")
                ledger_rows.append(status)
            _aggregate(problem_dir / arm / "aggregate.csv", method, arm)
        for fname in (
            "paired_stats.json",
            "paired_stats_hash_vs_baseline.json",
            "paired_stats_isalsr_vs_hash.json",
        ):
            (problem_dir / fname).write_text(
                json.dumps(
                    {
                        "method": method,
                        "benchmark": BENCH,
                        "problem": PROBLEM,
                        "metrics": {"r2_test": _paired_metric()},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    lines = [",".join(LEDGER_COLUMNS)]
    for row in ledger_rows:
        lines.append(",".join(str(row[c]) for c in LEDGER_COLUMNS))
    (root / "status_ledger.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (root / "metadata.json").write_text(
        json.dumps(
            {"config": {"benchmarks": {BENCH: {"train_size": 20, "test_size": 100}}}},
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _run(root: Path) -> dict[str, str]:
    """Certify a root and return ``{criterion_id: status}``."""
    results = certify(
        root=root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=None,
    )
    return {r.id: r.status for r in results}


def _detail(root: Path, criterion: str) -> dict[str, Any]:
    """Certify a root and return one criterion's detail mapping."""
    results = certify(
        root=root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=None,
    )
    return next(r.detail for r in results if r.id == criterion)


# ====================================================================== #
# Clean root
# ====================================================================== #


def test_clean_root_passes_every_structural_criterion(clean_root: Path) -> None:
    results = certify(
        root=clean_root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=None,
    )
    statuses = {r.id: r.status for r in results}
    details = {r.id: (r.observed, r.detail) for r in results}
    for cid in (
        "C1.1",
        "C1.2",
        "C1.3",
        "C1.6",
        "C1.7",
        "C1.8",
        "C1.9",
        "C1.10",
        "C1.12",
        "C1.13",
        "C1.14",
        "C1.15",
        "C1.16",
        "C1.17",
        "C2",
        "C4",
    ):
        # The message carries observed + detail so an intermittent failure is
        # diagnosable from the pytest output alone rather than needing a repro.
        assert statuses[cid] == "PASS", (
            f"{cid} failed on a clean root: observed={details[cid][0]!r} "
            f"detail={json.dumps(details[cid][1], default=str)}"
        )


def test_every_criterion_emits_a_verdict(clean_root: Path) -> None:
    statuses = _run(clean_root)
    expected_ids = {f"C1.{i}" for i in range(1, 18)} | {"C2", "C4"}
    assert set(statuses) == expected_ids
    assert all(v in {"PASS", "FAIL", "SKIP"} for v in statuses.values())


# ====================================================================== #
# Damage detection -- the actual acceptance property
# ====================================================================== #


def test_deleted_run_log_fails_and_names_the_cell(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_101" / "run_log.json"
    target.unlink()
    statuses = _run(clean_root)
    assert statuses["C1.1"] == "FAIL"
    assert statuses["C1.15"] == "FAIL"
    named = json.dumps(_detail(clean_root, "C1.15"))
    assert "udfs/isalsr/Nguyen-1/101" in named


def test_nan_r2_test_fails_c1_3(clean_root: Path) -> None:
    target = clean_root / "bingo" / BENCH / "nguyen_1" / "baseline" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["results"]["regression"]["r2_test"] = float("nan")
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.3"] == "FAIL"
    assert "r2_test" in json.dumps(_detail(clean_root, "C1.3")["non_finite_metrics"])


def test_zero_n_ledger_sampled_fails_c1_9(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_102" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["results"]["search_space"]["n_ledger_sampled"] = 0
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.9"] == "FAIL"
    detail = _detail(clean_root, "C1.9")
    assert detail["n_ledger_sampled_zero_or_missing"]["count"] == 1


def test_altered_data_fingerprint_fails_c4(clean_root: Path) -> None:
    target = clean_root / "bingo" / BENCH / "nguyen_1" / "hash" / "seed_101" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["metadata"]["data_fingerprint"] = "tampered"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C4"] == "FAIL"
    detail = _detail(clean_root, "C4")
    assert detail["cross_arm_disagreement"]["count"] == 1


def test_rho_below_one_names_the_broken_counter_diagnosis(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["results"]["search_space"]["empirical_reduction_factor"] = 0.5
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.6"] == "FAIL"
    assert any("BROKEN COUNTER" in d for d in _detail(clean_root, "C1.6")["diagnoses"])


def test_rho_all_one_names_the_dead_hook_diagnosis(clean_root: Path) -> None:
    for seed in SEEDS_3:
        for method in METHODS_2:
            target = (
                clean_root
                / method
                / BENCH
                / "nguyen_1"
                / "isalsr"
                / f"seed_{seed:02d}"
                / "run_log.json"
            )
            payload = json.loads(target.read_text())
            ss = payload["results"]["search_space"]
            ss["empirical_reduction_factor"] = 1.0
            ss["unique_canonical_dags"] = ss["total_dags_explored"]
            target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.6"] == "FAIL"
    assert any("DEAD DEDUP HOOK" in d for d in _detail(clean_root, "C1.6")["diagnoses"])


def test_rho_hash_above_rho_isalsr_fails_c1_7(clean_root: Path) -> None:
    for seed in SEEDS_3:
        target = (
            clean_root / "udfs" / BENCH / "nguyen_1" / "hash" / f"seed_{seed:02d}" / "run_log.json"
        )
        payload = json.loads(target.read_text())
        payload["results"]["search_space"]["empirical_reduction_factor"] = 9.9
        target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.7"] == "FAIL"


def test_populated_ledger_on_baseline_fails_c1_8(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "baseline" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["results"]["search_space"]["n_ledger_seen"] = 12
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.8"] == "FAIL"


def test_forbidden_alphabet_label_fails_c1_13(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["best_expression"]["canonical_string"] = "V+NV-"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.13"] == "FAIL"


def test_pow_outside_operator_set_is_disclosed_not_blocking(clean_root: Path) -> None:
    """Pow on a method whose operator set lacks it is counted, never blocking.

    These strings describe the SymPy-round-tripped best expression, not the
    candidate stream: SymPy writes ``sqrt(x)`` as ``Pow(x, 1/2)`` and ``x/y`` as
    ``x*Pow(y, -1)``. UDFS's vendored ``NODE_ARITY`` has no ``pow`` and its
    adapter has no ``POW`` mapping, so the search cannot have produced one, and
    failing C1.13 on it would fail the criterion for SymPy's notation. The
    candidate-stream assertion is check B3.
    """
    for method in ("bingo", "udfs"):
        target = clean_root / method / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "run_log.json"
        payload = json.loads(target.read_text())
        payload["best_expression"]["canonical_string"] = "V+NV^"
        payload["best_expression"]["isalsr_string"] = "V+NV^"
        target.write_text(json.dumps(payload), encoding="utf-8")
        assert _run(clean_root)["C1.13"] == "PASS", f"Pow must not block C1.13 for {method}"

    # ...but it is still recorded, so the disclosure obligation is discharged.
    disclosed = _detail(clean_root, "C1.13")["pow_outside_operator_set"]
    assert disclosed, "Pow outside the operator set must still be counted and reported"


def test_sub_and_div_remain_blocking(clean_root: Path) -> None:
    """A '-' or '/' has no encoding in the decomposed Sigma_SR at all."""
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["best_expression"]["canonical_string"] = "V+NV-"
    payload["best_expression"]["isalsr_string"] = "V+NV-"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.13"] == "FAIL"


def test_non_native_engine_fails_c1_14(clean_root: Path) -> None:
    target = clean_root / "bingo" / BENCH / "nguyen_1" / "hash" / "seed_00" / "run_log.json"
    payload = json.loads(target.read_text())
    payload["metadata"]["hardware"]["engine"] = "python"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert _run(clean_root)["C1.14"] == "FAIL"


def test_started_status_is_a_time_kill(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "baseline" / "seed_00" / "status.json"
    payload = json.loads(target.read_text())
    payload["terminal_status"] = "started"
    target.write_text(json.dumps(payload), encoding="utf-8")
    statuses = _run(clean_root)
    assert statuses["C1.12"] == "FAIL"
    assert statuses["C1.15"] == "FAIL"


def test_non_monotone_trajectory_fails_c1_10(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "trajectory.csv"
    lines = target.read_text().splitlines()
    lines[-1] = lines[-1].replace("15.0,2,0.98", "1.0,2,0.10")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run(clean_root)["C1.10"] == "FAIL"


def test_missing_status_ledger_fails_c2(clean_root: Path) -> None:
    (clean_root / "status_ledger.csv").unlink()
    assert _run(clean_root)["C2"] == "FAIL"


def test_missing_contrast_file_fails_c1_16(clean_root: Path) -> None:
    (clean_root / "udfs" / BENCH / "nguyen_1" / "paired_stats_isalsr_vs_hash.json").unlink()
    assert _run(clean_root)["C1.16"] == "FAIL"


def test_c1_16_passes_when_the_declared_seed_count_matches(clean_root: Path) -> None:
    """Positive control for the regression below."""
    assert _run(clean_root)["C1.16"] == "PASS"


def test_c1_16_honours_the_declared_seed_count(clean_root: Path) -> None:
    """C1.16 must validate against the seeds it is GIVEN, not ``DEFAULT_SEEDS``.

    The fixture root holds three paired seeds, so declaring a four-seed campaign
    must fail. Before the 2026-08-14 fix the check compared every contrast
    against the hardcoded Stage C smoke default of three, which made C1.16
    unpassable for any campaign with a different seed count: the real C2 run
    reported ``0/420 valid`` and a spurious **NO-GO** on complete, correct data.
    A blocking check whose failure probability rises with the sample size is
    worse than no check at all.
    """
    results = certify(
        root=clean_root,
        expected_tasks=24,
        max_time=15.0,
        wall_slack=600.0,
        seeds=(*SEEDS_3, 103),
        sacct_csv=None,
    )
    status = {r.id: r.status for r in results}
    assert status["C1.16"] == "FAIL"


def test_wrong_aggregate_row_count_fails_c1_17(clean_root: Path) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "aggregate.csv"
    lines = target.read_text().splitlines()
    target.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert _run(clean_root)["C1.17"] == "FAIL"


# ====================================================================== #
# Robustness: never crash
# ====================================================================== #


@pytest.mark.parametrize(
    "damage",
    ["truncate", "empty", "not_json", "not_object", "wrong_types"],
)
def test_corrupt_run_log_never_raises(clean_root: Path, damage: str) -> None:
    target = clean_root / "udfs" / BENCH / "nguyen_1" / "isalsr" / "seed_00" / "run_log.json"
    if damage == "truncate":
        target.write_text(target.read_text()[:40], encoding="utf-8")
    elif damage == "empty":
        target.write_text("", encoding="utf-8")
    elif damage == "not_json":
        target.write_text("<html>500</html>", encoding="utf-8")
    elif damage == "not_object":
        target.write_text("[1, 2, 3]", encoding="utf-8")
    else:
        target.write_text(json.dumps({"metadata": {"seed": "zero"}}), encoding="utf-8")
    statuses = _run(clean_root)
    assert statuses["C1.2"] == "FAIL"
    assert len(statuses) == 19


def test_empty_root_never_raises(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    statuses = _run(root)
    assert len(statuses) == 19
    assert statuses["C1.1"] == "FAIL"


def test_absent_root_exits_one_without_raising(tmp_path: Path) -> None:
    code = main(
        [
            "--root",
            str(tmp_path / "does_not_exist"),
            "--out-json",
            str(tmp_path / "o.json"),
            "--out-md",
            str(tmp_path / "o.md"),
        ]
    )
    assert code == 1
    assert json.loads((tmp_path / "o.json").read_text())["verdict"] == "NO-GO"


def test_partial_root_reports_honestly(clean_root: Path) -> None:
    import shutil

    shutil.rmtree(clean_root / "bingo")
    results = certify(
        root=clean_root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=None,
    )
    c15 = next(r for r in results if r.id == "C1.15")
    # The universe collapses to what remains discoverable; the check still runs
    # and still reports observed against expected rather than raising.
    assert c15.observed == 9
    assert c15.detail["expected_set_source"] == "disk"


def test_certifier_writes_only_the_two_out_paths(clean_root: Path, tmp_path: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in clean_root.rglob("*") if p.is_file()}
    out_dir = tmp_path / "reports"
    code = main(
        [
            "--root",
            str(clean_root),
            "--out-json",
            str(out_dir / "cert.json"),
            "--out-md",
            str(out_dir / "cert.md"),
            "--expected-tasks",
            "18",
            "--max-time",
            "15",
        ]
    )
    assert code in (0, 1)
    after = {p: p.stat().st_mtime_ns for p in clean_root.rglob("*") if p.is_file()}
    assert before == after, "the certifier mutated the results root"
    assert (out_dir / "cert.json").exists()
    assert (out_dir / "cert.md").exists()


# ====================================================================== #
# Helpers
# ====================================================================== #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("garbage", None),
        ("0", 0.0),
        ("1G", 1.0),
        ("1024M", 1.0),
        ("2048K", 2048 / 1024**2),
        ("1T", 1024.0),
    ],
)
def test_parse_maxrss_to_gb(raw: str, expected: float | None) -> None:
    result = _parse_maxrss_to_gb(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([], 50, None),
        ([1.0], 99, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.0),
        ([1.0, 2.0, 3.0, 4.0], 100, 4.0),
    ],
)
def test_percentile(values: list[float], q: float, expected: float | None) -> None:
    assert _percentile(values, q) == expected


def test_field_spec_covers_every_schema_field() -> None:
    """The C1.2 spec must enumerate every field of every schema dataclass."""
    from dataclasses import fields as dc_fields

    from experiments.models.schemas import (
        BestExpression,
        RegressionResults,
        RunMetadata,
        SearchSpaceResults,
        TimeResults,
    )

    spec_leaves = {path[-1] for path, _t, _n in RUN_LOG_FIELD_SPEC}
    for cls in (RunMetadata, RegressionResults, TimeResults, SearchSpaceResults, BestExpression):
        for f in dc_fields(cls):
            assert f.name in spec_leaves, f"{cls.__name__}.{f.name} missing from RUN_LOG_FIELD_SPEC"


def test_sacct_csv_join_and_recommendation(clean_root: Path, tmp_path: Path) -> None:
    sacct = tmp_path / "sacct.csv"
    sacct.write_text(
        "JobID,MaxRSS\n" + "\n".join(f"9001_{s}.batch,{2 + i}G" for i, s in enumerate(SEEDS_3)),
        encoding="utf-8",
    )
    results = certify(
        root=clean_root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=sacct,
    )
    c11 = next(r for r in results if r.id == "C1.11")
    assert c11.status == "PASS"
    assert c11.detail["source"].startswith("sacct:")
    assert c11.detail["recommended_production_mem_gb"] == pytest.approx(6.0)
    assert c11.blocking is False


def test_missing_sacct_falls_back_to_status_json(clean_root: Path, tmp_path: Path) -> None:
    results = certify(
        root=clean_root,
        expected_tasks=18,
        max_time=15.0,
        wall_slack=600.0,
        seeds=SEEDS_3,
        sacct_csv=tmp_path / "absent.csv",
    )
    c11 = next(r for r in results if r.id == "C1.11")
    assert "status.json" in c11.detail["source"]
    assert c11.detail["pooled_gb"]["n"] == 18
