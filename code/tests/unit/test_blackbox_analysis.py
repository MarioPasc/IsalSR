"""Unit tests for the SRBench black-box analysis (``experiments.scripts.blackbox``).

The analysis reuses the main campaign's statistics unchanged, so these tests
pin only what is new: the pre-declared inclusion rule, the stop-reason
classifier, the exact-or-asymptotic branch of the signed-rank test and its
floor at N = 20, the flattening of the test records, and the formatting of the
numbers the appendix prints. One integration test audits the real corpus when
it is present on this machine and is skipped otherwise (the capsule is offline
and does not carry the corpus).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from experiments.models.analyzer.aggregation import compute_cross_problem_dominance
from experiments.models.schemas import PairedStats, PairedStatsMetric
from experiments.scripts.blackbox import audit, latex, measures, stopping
from experiments.scripts.blackbox.config import (
    BINGO_MAX_EVALS,
    BUDGET_S,
    DATASETS,
    DEFAULT_CORPUS,
    SEEDS,
)
from experiments.scripts.blackbox.cpdt import (
    exact_wilcoxon,
    flatten_cpdt,
    primary_p_value,
    wilcoxon_floor,
)

# ----------------------------------------------------------------------
# Inclusion rule (PLAN D7)
# ----------------------------------------------------------------------


def _write_cell(
    directory: Path,
    *,
    status: str = "completed",
    exit_code: int = 0,
    r2_test: float | None = 0.5,
    nrmse_test: float | None = 0.7,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    log = {"results": {"regression": {"r2_test": r2_test, "nrmse_test": nrmse_test}}}
    (directory / "run_log.json").write_text(json.dumps(log), encoding="utf-8")
    (directory / "status.json").write_text(
        json.dumps({"terminal_status": status, "exit_code": exit_code}), encoding="utf-8"
    )


def test_cell_reason_accepts_a_completed_finite_cell(tmp_path: Path) -> None:
    _write_cell(tmp_path / "ok")
    assert audit._cell_reason(tmp_path / "ok") is None


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"status": "failed"}, "terminal_status"),
        ({"exit_code": 137}, "exit_code"),
        ({"r2_test": None}, "r2_test not finite"),
        ({"nrmse_test": float("nan")}, "nrmse_test not finite"),
    ],
)
def test_cell_reason_names_the_failing_criterion(
    tmp_path: Path, kwargs: dict[str, Any], fragment: str
) -> None:
    _write_cell(tmp_path / "bad", **kwargs)
    reason = audit._cell_reason(tmp_path / "bad")
    assert reason is not None and fragment in reason


def test_cell_reason_flags_a_missing_run_log(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert audit._cell_reason(tmp_path / "empty") == "run_log.json missing"


def test_fallback_drops_only_the_failing_host_dataset() -> None:
    report = audit.AuditReport()
    report.issues.append(audit.CellIssue("bingo", DATASETS[3], "hash", 7, "exit_code=1"))
    audit._apply_fallback(report)
    assert DATASETS[3] not in report.included["bingo"]
    assert len(report.included["bingo"]) == len(DATASETS) - 1
    assert report.included["udfs"] == list(DATASETS)
    assert report.excluded["bingo"] == {DATASETS[3]: ["hash/seed_07: exit_code=1"]}
    assert report.excluded["udfs"] == {}


def test_declared_grid_is_1200_cells() -> None:
    cells = audit.expected_cells()
    assert len(cells) == 2 * len(DATASETS) * 3 * len(SEEDS) == 1200
    assert len(set(cells)) == len(cells)


# ----------------------------------------------------------------------
# Stop reasons
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "wall", "r2", "at_cap", "expected"),
    [
        ("bingo", 5_000.0, 0.8, True, "evaluations"),
        # Bingo checks the evaluation cap before the time limit.
        ("bingo", 42_335.0, 0.8, True, "evaluations"),
        ("bingo", 42_335.0, 0.8, False, "time"),
        ("bingo", 900.0, 1.0, False, "fitness"),
        ("bingo", 900.0, 0.99, False, "other"),
        # UDFS has no evaluation cap; the flag is ignored there.
        ("udfs", 43_200.5, 0.6, True, "time"),
        ("udfs", 120.0, 1.0, None, "fitness"),
    ],
)
def test_classify_follows_bingo_check_order(
    method: str, wall: float, r2: float, at_cap: bool | None, expected: str
) -> None:
    assert stopping.classify(method, wall, r2, at_cap) == expected


def test_counter_prefers_the_exact_reading() -> None:
    below = stopping.CounterReading(
        exact=BINGO_MAX_EVALS - 1, last_capture=BINGO_MAX_EVALS, max_step=10
    )
    assert below.at_cap() is False
    lagging = stopping.CounterReading(exact=None, last_capture=BINGO_MAX_EVALS - 500, max_step=600)
    assert lagging.at_cap() is True
    short = stopping.CounterReading(exact=None, last_capture=BINGO_MAX_EVALS - 500, max_step=400)
    assert short.at_cap() is False
    assert stopping.CounterReading(None, None, 0).at_cap() is None


def test_read_counter_uses_the_convergence_log(tmp_path: Path) -> None:
    n_evals = np.array([0, 40_000_000, 80_000_000, 99_999_000], dtype=np.int32)
    np.savez_compressed(tmp_path / "convergence_log.npz", n_evals=n_evals)
    log = {"results": {"search_space": {"total_dags_explored": 123}}}
    hashed = stopping.read_counter(tmp_path, "hash", log)
    assert hashed == stopping.CounterReading(
        exact=None, last_capture=99_999_000, max_step=40_000_000
    )
    native = stopping.read_counter(tmp_path, "baseline", log)
    assert native.exact == 123 and native.at_cap() is False


def test_time_rule_covers_bingo_checkpoint_stop() -> None:
    # Bingo stops when under a quarter of a checkpoint remains, near 42,335 s.
    assert stopping.classify("bingo", 42_329.0, 0.7, False) == "time"
    assert BUDGET_S > 42_329.0


# ----------------------------------------------------------------------
# Signed-rank branch and floor at N = 20
# ----------------------------------------------------------------------


def test_scipy_is_exact_without_zeros_or_ties() -> None:
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.1, 1.0, size=20)
    assert exact_wilcoxon(list(deltas))
    auto = stats.wilcoxon(deltas, alternative="greater", zero_method="zsplit").pvalue
    exact = stats.wilcoxon(
        deltas, alternative="greater", zero_method="zsplit", method="exact"
    ).pvalue
    assert auto == pytest.approx(exact, rel=1e-12)


def test_scipy_falls_back_to_the_normal_approximation_with_a_zero() -> None:
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.1, 1.0, size=20)
    deltas[4] = 0.0
    assert not exact_wilcoxon(list(deltas))
    auto = stats.wilcoxon(deltas, alternative="greater", zero_method="zsplit").pvalue
    approx = stats.wilcoxon(
        deltas, alternative="greater", zero_method="zsplit", method="approx"
    ).pvalue
    assert auto == pytest.approx(approx, rel=1e-12)


def test_tie_snap_counts_as_a_zero() -> None:
    deltas = [float(i + 1) for i in range(19)] + [5e-7]
    assert not exact_wilcoxon(deltas)


def _paired(problem: str, delta: float) -> PairedStats:
    ps = PairedStats(method="bingo", benchmark="srbench_blackbox", problem=problem, n_seeds=10)
    ps.metrics["r2_test"] = PairedStatsMetric(
        baseline_mean=0.5,
        baseline_std=0.0,
        isalsr_mean=0.5 + delta,
        isalsr_std=0.0,
        mean_diff=delta,
        std_diff=0.0,
        shapiro_wilk_p=1.0,
        normality_assumed=True,
        test_used="paired_t",
        statistic=0.0,
        p_value_raw=1.0,
        p_value_holm=None,
        cohens_d=0.0,
        cohens_d_ci_lower=0.0,
        cohens_d_ci_upper=0.0,
        mean_diff_ci_lower=0.0,
        mean_diff_ci_upper=0.0,
        n=10,
    )
    return ps


def test_all_positive_skewed_deltas_reach_the_exact_floor() -> None:
    # Geometric deltas fail Shapiro-Wilk, so the main campaign's test takes the
    # signed-rank branch; with every sign positive the exact p is 2^-20.
    stats_list = [_paired(f"p{i}", 1e-4 * 2.0**i) for i in range(20)]
    res = compute_cross_problem_dominance(stats_list, "r2_test", "bingo", "srbench_blackbox")
    assert res.test_used == "wilcoxon_signed_rank"
    assert res.p_value_one_sided == pytest.approx(wilcoxon_floor(20, "greater"), rel=1e-12)
    assert wilcoxon_floor(20, "greater") == 2.0**-20
    assert wilcoxon_floor(20, "two-sided") == 2.0**-19


# ----------------------------------------------------------------------
# Test records
# ----------------------------------------------------------------------


def _record(test: str, alternative: str, deltas: list[float]) -> dict[str, Any]:
    return {
        "test_used": test,
        "alternative": alternative,
        "n_problems": len(deltas),
        "n_wins": sum(d > 0 for d in deltas),
        "n_ties": 0,
        "n_losses": sum(d < 0 for d in deltas),
        "shapiro_wilk_p": 0.01,
        "cohens_d": 0.3,
        "cohens_d_ci_lower": -0.1,
        "cohens_d_ci_upper": 0.6,
        "mean_delta": 0.01,
        "p_value_one_sided": 0.2,
        "p_value_two_sided": 0.4,
        "p_value_holm": 0.4,
        "problem_deltas": deltas,
    }


def test_flatten_reports_the_primary_p_of_each_alternative() -> None:
    deltas = [0.01 * (i + 1) * (-1) ** i for i in range(20)]
    doc = {
        "contrasts": {
            "isalsr_vs_baseline": {"r2_test": _record("wilcoxon_signed_rank", "greater", deltas)},
            "isalsr_vs_hash": {"r2_test": _record("wilcoxon_signed_rank", "two-sided", deltas)},
        }
    }
    rows = {(r["contrast"], r["metric"]): r for r in flatten_cpdt(doc, "bingo")}
    assert rows[("isalsr_vs_baseline", "r2_test")]["p_primary"] == 0.2
    assert rows[("isalsr_vs_hash", "r2_test")]["p_primary"] == 0.4
    assert rows[("isalsr_vs_baseline", "r2_test")]["wilcoxon_exact"] == 1
    assert rows[("isalsr_vs_baseline", "r2_test")]["at_exact_floor"] == 0


def test_primary_p_of_a_descriptive_record_is_nan() -> None:
    rec = _record("descriptive_definitional_baseline", "descriptive", [1.0])
    rec["p_value_one_sided"] = float("nan")
    assert math.isnan(primary_p_value(rec))


# ----------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------


def test_generation_ratio_is_seed_matched() -> None:
    extra = []
    for seed in SEEDS:
        extra.append(
            {"method": "bingo", "dataset": "a", "arm": "baseline", "seed": seed, "generations": 100}
        )
        extra.append(
            {"method": "bingo", "dataset": "a", "arm": "isalsr", "seed": seed, "generations": 150}
        )
        extra.append(
            {"method": "bingo", "dataset": "a", "arm": "hash", "seed": seed, "generations": 120}
        )
    out = measures.generation_ratios(extra)
    assert out["isalsr"]["median_over_datasets"] == pytest.approx(1.5)
    assert out["hash"]["median_over_datasets"] == pytest.approx(1.2)


def test_termination_summary_lists_datasets_per_reason() -> None:
    extra = [
        {"method": "bingo", "arm": "isalsr", "dataset": "a", "termination": "evaluations"},
        {"method": "bingo", "arm": "isalsr", "dataset": "b", "termination": "time"},
    ]
    out = measures.termination_summary(extra)
    assert out["bingo"]["isalsr"]["tally"] == {"evaluations": 1, "time": 1}
    assert out["bingo"]["isalsr"]["datasets"]["time"] == ["b"]
    assert out["udfs"]["baseline"]["tally"] == {}


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def test_macro_names_hold_letters_only() -> None:
    assert latex.macro_name("udfs_rho_sd") == r"\bbxUdfsRhoSd"
    with pytest.raises(ValueError):
        latex.macro_name("udfs_r2_test")


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.62857, "0.629"),
        (8016074661.96, "8.0{\\times}10^{9}"),
        (-1234.5, "-1.2{\\times}10^{3}"),
        (float("nan"), "---"),
    ],
)
def test_fmt_num_switches_to_exponent_form(x: float, expected: str) -> None:
    assert latex.fmt_num(x, 3) == expected


def test_fmt_p_plain_carries_no_significance_marks() -> None:
    assert latex.fmt_p_plain(5.15e-13) == "5.2{\\times}10^{-13}"
    assert latex.fmt_p_plain(0.6386) == "0.639"
    assert latex.fmt_p_plain(float("nan")) == "---"


def test_best_worst_marks_survive_extreme_values() -> None:
    marks = latex.mark_best_worst(
        {"baseline": 2.2e6, "hash": 3.644, "isalsr": 8.0e8}, higher_is_better=False, digits=3
    )
    assert marks["hash"] == r"\mathbf{3.644}"
    assert marks["isalsr"] == r"\underline{8.0{\times}10^{8}}"


def test_macros_file_defines_each_name_once() -> None:
    text = latex.macros_file({r"\bbxA": "1", r"\bbxB": "2"})
    assert text.count(r"\newcommand{\bbxA}{1}") == 1
    assert text.count(r"\newcommand") == 2


# ----------------------------------------------------------------------
# The real corpus, when present
# ----------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not DEFAULT_CORPUS.is_dir(), reason="black-box corpus not on this machine")
def test_real_corpus_is_complete_and_single_provenance() -> None:
    report = audit.run_audit(DEFAULT_CORPUS)
    assert report.complete
    assert report.n_completed == 1200
    assert report.excluded == {"udfs": {}, "bingo": {}}
    assert not report.e6_e7["provenance"]["mixed"]
    records = stopping.scan(DEFAULT_CORPUS)
    assert len(records) == 1200
    check = stopping.validate_native(records)
    assert check.get("disagree", 0) == 0 and check["agree"] == 200
