"""Per-contrast CPDT for the three-arm C2 campaign (contrast policy, 2026-08-04).

Campaign C2 has three arms (baseline, hash, isalsr). The cross-problem dominance
test used to run only on the primary contrast (baseline -> isalsr). This module
pins the contrast policy Mario decided on 2026-08-04:

===========================  ==================  ==============  ==================
Contrast (arm_a -> arm_b)    r2_test / r2_train  nrmse_test      rho / redundancy
===========================  ==================  ==============  ==================
baseline -> isalsr           greater             less            descriptive
hash -> isalsr               two-sided           two-sided       greater
baseline -> hash             two-sided           two-sided       descriptive
===========================  ==================  ==============  ==================

"Descriptive" contrasts are definitional: the baseline arm has rho == 1 by
construction, so a test of "isalsr merges more than a representation that never
merges" is tautological. The result object is still produced (W/T/L, effect
size, CIs, per-problem deltas); only the p-values are withheld.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.models.analyze import run_cross_problem_dominance_test
from experiments.models.analyzer.aggregation import (
    CPDT_CONTRAST_POLICY,
    CPDT_METRIC_ALTERNATIVES,
    apply_holm_across_contrasts,
    compute_cross_problem_dominance,
)
from experiments.models.schemas import CrossProblemDominanceResult, PairedStats, PairedStatsMetric

DESCRIPTIVE_MARKER = "descriptive_definitional_baseline"

_CPDT_METRICS = (
    "r2_test",
    "r2_train",
    "nrmse_test",
    "empirical_reduction_factor",
    "redundancy_rate",
)


# ----------------------------------------------------------------------
# Helpers (mirroring tests/unit/test_stats_fairness_fixes.py)
# ----------------------------------------------------------------------


def _metric(delta: float) -> PairedStatsMetric:
    """Build a PairedStatsMetric whose arm_b - arm_a mean equals ``delta``."""
    return PairedStatsMetric(
        baseline_mean=0.0,
        baseline_std=0.0,
        isalsr_mean=delta,
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
        n=30,
    )


def _paired_stats(per_metric: dict[str, list[float]]) -> list[PairedStats]:
    """Wrap {metric: delta vector} as one PairedStats per problem."""
    lengths = {len(v) for v in per_metric.values()}
    assert len(lengths) == 1, "all metrics need the same number of problems"
    n = lengths.pop()
    return [
        PairedStats(
            method="bingo",
            benchmark="benchmark",
            problem=f"P{i:03d}",
            metrics={m: _metric(v[i]) for m, v in per_metric.items()},
            n_seeds=30,
        )
        for i in range(n)
    ]


def _rng_deltas(seed: int, n: int, loc: float, scale: float) -> list[float]:
    return (np.random.default_rng(seed).normal(loc, scale, n)).tolist()


def _three_contrast_inputs(n: int = 12) -> dict[str, list[PairedStats]]:
    """Synthetic three-arm input with distinct deltas per contrast."""
    primary = _paired_stats(
        {
            "r2_test": _rng_deltas(1, n, 0.020, 0.010),
            "r2_train": _rng_deltas(2, n, 0.030, 0.010),
            "nrmse_test": _rng_deltas(3, n, -0.020, 0.010),
            "empirical_reduction_factor": _rng_deltas(4, n, 0.500, 0.100),
            "redundancy_rate": _rng_deltas(5, n, 0.300, 0.050),
        }
    )
    isalsr_vs_hash = _paired_stats(
        {
            "r2_test": _rng_deltas(6, n, 0.004, 0.010),
            "r2_train": _rng_deltas(7, n, 0.006, 0.010),
            "nrmse_test": _rng_deltas(8, n, -0.003, 0.010),
            "empirical_reduction_factor": _rng_deltas(9, n, 0.120, 0.050),
            "redundancy_rate": _rng_deltas(10, n, 0.080, 0.030),
        }
    )
    hash_vs_baseline = _paired_stats(
        {
            "r2_test": _rng_deltas(11, n, 0.012, 0.010),
            "r2_train": _rng_deltas(12, n, 0.018, 0.010),
            "nrmse_test": _rng_deltas(13, n, -0.014, 0.010),
            "empirical_reduction_factor": _rng_deltas(14, n, 0.380, 0.100),
            "redundancy_rate": _rng_deltas(15, n, 0.220, 0.050),
        }
    )
    return {
        "isalsr_vs_baseline": primary,
        "isalsr_vs_hash": isalsr_vs_hash,
        "hash_vs_baseline": hash_vs_baseline,
    }


def _holm_by_hand(p_values: list[float]) -> list[float]:
    """Holm step-down, computed independently of statsmodels."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    running = 0.0
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


# ----------------------------------------------------------------------
# (a) The policy table matches the decision, cell by cell
# ----------------------------------------------------------------------

_POLICY_CELLS: list[tuple[str, str, str, str | None]] = [
    ("baseline", "isalsr", "r2_test", "greater"),
    ("baseline", "isalsr", "r2_train", "greater"),
    ("baseline", "isalsr", "nrmse_test", "less"),
    ("baseline", "isalsr", "empirical_reduction_factor", None),
    ("baseline", "isalsr", "redundancy_rate", None),
    ("hash", "isalsr", "r2_test", "two-sided"),
    ("hash", "isalsr", "r2_train", "two-sided"),
    ("hash", "isalsr", "nrmse_test", "two-sided"),
    ("hash", "isalsr", "empirical_reduction_factor", "greater"),
    ("hash", "isalsr", "redundancy_rate", "greater"),
    ("baseline", "hash", "r2_test", "two-sided"),
    ("baseline", "hash", "r2_train", "two-sided"),
    ("baseline", "hash", "nrmse_test", "two-sided"),
    ("baseline", "hash", "empirical_reduction_factor", None),
    ("baseline", "hash", "redundancy_rate", None),
]


@pytest.mark.parametrize(("arm_a", "arm_b", "metric", "expected"), _POLICY_CELLS)
def test_policy_cell(arm_a: str, arm_b: str, metric: str, expected: str | None) -> None:
    assert CPDT_CONTRAST_POLICY[(arm_a, arm_b)][metric] == expected


def test_policy_has_exactly_three_contrasts() -> None:
    assert set(CPDT_CONTRAST_POLICY) == {
        ("baseline", "isalsr"),
        ("hash", "isalsr"),
        ("baseline", "hash"),
    }


def test_policy_covers_every_cpdt_metric() -> None:
    for key, table in CPDT_CONTRAST_POLICY.items():
        assert set(table) == set(CPDT_METRIC_ALTERNATIVES), key
    assert set(CPDT_METRIC_ALTERNATIVES) == set(_CPDT_METRICS)


def test_unknown_contrast_falls_back_to_metric_alternatives() -> None:
    ps = _paired_stats({"r2_test": _rng_deltas(21, 10, 0.02, 0.01)})
    res = compute_cross_problem_dominance(
        ps, "r2_test", method="bingo", benchmark="benchmark", arm_a="foo", arm_b="bar"
    )
    assert res.alternative == CPDT_METRIC_ALTERNATIVES["r2_test"] == "greater"


# ----------------------------------------------------------------------
# (b) Two-arm regression: primary contrast is untouched
# ----------------------------------------------------------------------

_LEGACY_FIELDS = (
    "alternative",
    "n_problems",
    "n_wins",
    "n_ties",
    "n_losses",
    "problem_names",
    "problem_deltas",
    "shapiro_wilk_p",
    "normality_assumed",
    "test_used",
    "statistic",
    "p_value_one_sided",
    "p_value_two_sided",
    "cohens_d",
    "cohens_d_ci_lower",
    "cohens_d_ci_upper",
    "mean_delta",
    "mean_delta_ci_lower",
    "mean_delta_ci_upper",
)


@pytest.mark.parametrize("metric", ["r2_test", "r2_train", "nrmse_test"])
def test_primary_contrast_identical_to_legacy_call(metric: str) -> None:
    """Default-arg call reproduces the pre-change (explicit-alternative) result."""
    deltas = _rng_deltas(31, 14, 0.02 if metric != "nrmse_test" else -0.02, 0.015)
    ps = _paired_stats({metric: deltas})
    legacy = compute_cross_problem_dominance(
        ps,
        metric,
        method="bingo",
        benchmark="benchmark",
        alternative=CPDT_METRIC_ALTERNATIVES[metric],
    )
    new = compute_cross_problem_dominance(ps, metric, method="bingo", benchmark="benchmark")
    for fname in _LEGACY_FIELDS:
        assert getattr(new, fname) == getattr(legacy, fname), fname
    assert new.arm_a == "baseline"
    assert new.arm_b == "isalsr"


def test_two_arm_root_holm_equals_primary_p(tmp_path: Path) -> None:
    """No hash arm -> Holm family of size 1 -> adjusted p == raw primary p."""
    ps = _paired_stats(
        {
            "r2_test": _rng_deltas(41, 14, 0.02, 0.015),
            "r2_train": _rng_deltas(42, 14, 0.03, 0.015),
            "nrmse_test": _rng_deltas(43, 14, -0.02, 0.015),
            "empirical_reduction_factor": _rng_deltas(44, 14, 0.5, 0.1),
            "redundancy_rate": _rng_deltas(45, 14, 0.3, 0.05),
        }
    )
    out = run_cross_problem_dominance_test(ps, "bingo", "benchmark", tmp_path)

    assert set(out["contrasts"]) == {"isalsr_vs_baseline"}
    for metric in ("r2_test", "r2_train", "nrmse_test"):
        entry = out[metric]
        assert entry["p_value_holm"] == pytest.approx(entry["p_value_one_sided"])
    for metric in ("empirical_reduction_factor", "redundancy_rate"):
        assert out[metric]["p_value_holm"] is None
        assert out[metric]["test_used"] == DESCRIPTIVE_MARKER


def test_two_arm_json_keeps_top_level_shape(tmp_path: Path) -> None:
    ps = _paired_stats(
        {m: _rng_deltas(50 + i, 12, 0.02, 0.01) for i, m in enumerate(_CPDT_METRICS)}
    )
    run_cross_problem_dominance_test(ps, "bingo", "benchmark", tmp_path)
    path = tmp_path / "cross_problem_dominance_bingo_benchmark.json"
    with open(path) as f:
        payload = json.load(f)
    # Every metric remains a top-level key holding a full result dict, as the
    # LaTeX table generator reads it.
    for metric in _CPDT_METRICS:
        assert metric in payload
        assert payload[metric]["metric"] == metric
        assert "cohens_d" in payload[metric]
    assert "contrasts" in payload


# ----------------------------------------------------------------------
# (c) Three-contrast behaviour
# ----------------------------------------------------------------------


@pytest.fixture
def three_contrast_out(tmp_path: Path) -> dict[str, Any]:
    inputs = _three_contrast_inputs()
    return run_cross_problem_dominance_test(
        inputs["isalsr_vs_baseline"],
        "bingo",
        "benchmark",
        tmp_path,
        contrast_stats={
            "isalsr_vs_hash": inputs["isalsr_vs_hash"],
            "hash_vs_baseline": inputs["hash_vs_baseline"],
        },
    )


def test_three_contrasts_present(three_contrast_out: dict[str, Any]) -> None:
    assert set(three_contrast_out["contrasts"]) == {
        "isalsr_vs_baseline",
        "isalsr_vs_hash",
        "hash_vs_baseline",
    }
    for contrast in three_contrast_out["contrasts"].values():
        assert set(contrast) == set(_CPDT_METRICS)


def test_arm_labels(three_contrast_out: dict[str, Any]) -> None:
    contrasts = three_contrast_out["contrasts"]
    assert contrasts["isalsr_vs_baseline"]["r2_test"]["arm_a"] == "baseline"
    assert contrasts["isalsr_vs_baseline"]["r2_test"]["arm_b"] == "isalsr"
    assert contrasts["isalsr_vs_hash"]["r2_test"]["arm_a"] == "hash"
    assert contrasts["isalsr_vs_hash"]["r2_test"]["arm_b"] == "isalsr"
    assert contrasts["hash_vs_baseline"]["r2_test"]["arm_a"] == "baseline"
    assert contrasts["hash_vs_baseline"]["r2_test"]["arm_b"] == "hash"


def test_top_level_is_the_primary_contrast(three_contrast_out: dict[str, Any]) -> None:
    for metric in _CPDT_METRICS:
        assert (
            three_contrast_out[metric]
            == three_contrast_out["contrasts"]["isalsr_vs_baseline"][metric]
        )


def test_two_sided_contrasts_record_both_p(three_contrast_out: dict[str, Any]) -> None:
    for name in ("isalsr_vs_hash", "hash_vs_baseline"):
        entry = three_contrast_out["contrasts"][name]["r2_test"]
        assert entry["alternative"] == "two-sided"
        assert math.isfinite(entry["p_value_one_sided"])
        assert math.isfinite(entry["p_value_two_sided"])
        # One-sided p is taken in the direction of the observed mean delta,
        # so it never exceeds the two-sided p.
        assert entry["p_value_one_sided"] <= entry["p_value_two_sided"] + 1e-12


def test_holm_family_of_three_for_r2_test(three_contrast_out: dict[str, Any]) -> None:
    contrasts = three_contrast_out["contrasts"]
    names = ["isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline"]
    raw = []
    for name in names:
        e = contrasts[name]["r2_test"]
        raw.append(
            e["p_value_two_sided"] if e["alternative"] == "two-sided" else e["p_value_one_sided"]
        )
    expected = _holm_by_hand(raw)
    got = [contrasts[name]["r2_test"]["p_value_holm"] for name in names]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
    # Holm never decreases a p-value and preserves the raw ordering.
    for r, a in zip(raw, got, strict=True):
        assert a >= r - 1e-15
    assert [names[i] for i in np.argsort(raw)] == [names[i] for i in np.argsort(got, kind="stable")]


def test_rho_tested_only_on_hash_to_isalsr(three_contrast_out: dict[str, Any]) -> None:
    contrasts = three_contrast_out["contrasts"]
    for metric in ("empirical_reduction_factor", "redundancy_rate"):
        tested = contrasts["isalsr_vs_hash"][metric]
        assert tested["alternative"] == "greater"
        assert math.isfinite(tested["p_value_one_sided"])
        # Family of one -> Holm is the identity.
        assert tested["p_value_holm"] == pytest.approx(tested["p_value_one_sided"])
        for name in ("isalsr_vs_baseline", "hash_vs_baseline"):
            desc = contrasts[name][metric]
            assert desc["test_used"] == DESCRIPTIVE_MARKER
            assert math.isnan(desc["p_value_one_sided"])
            assert math.isnan(desc["p_value_two_sided"])
            assert desc["p_value_holm"] is None


def test_descriptive_rows_keep_effect_size_and_counts(three_contrast_out: dict[str, Any]) -> None:
    desc = three_contrast_out["contrasts"]["isalsr_vs_baseline"]["empirical_reduction_factor"]
    assert desc["n_problems"] == 12
    assert desc["n_wins"] + desc["n_ties"] + desc["n_losses"] == 12
    assert math.isfinite(desc["cohens_d"])
    assert math.isfinite(desc["cohens_d_ci_lower"])
    assert math.isfinite(desc["cohens_d_ci_upper"])
    assert math.isfinite(desc["mean_delta"])
    assert math.isfinite(desc["mean_delta_ci_lower"])
    assert math.isfinite(desc["mean_delta_ci_upper"])
    assert len(desc["problem_deltas"]) == 12
    assert len(desc["problem_names"]) == 12


def test_descriptive_effect_size_matches_a_tested_call() -> None:
    """Withholding the p-value must not perturb the estimation half."""
    deltas = _rng_deltas(61, 12, 0.5, 0.1)
    ps = _paired_stats({"empirical_reduction_factor": deltas})
    desc = compute_cross_problem_dominance(
        ps, "empirical_reduction_factor", method="bingo", benchmark="benchmark"
    )
    tested = compute_cross_problem_dominance(
        ps,
        "empirical_reduction_factor",
        method="bingo",
        benchmark="benchmark",
        alternative="greater",
    )
    assert desc.test_used == DESCRIPTIVE_MARKER
    assert tested.test_used != DESCRIPTIVE_MARKER
    np.testing.assert_allclose(desc.cohens_d, tested.cohens_d, rtol=1e-12)
    np.testing.assert_allclose(desc.mean_delta, tested.mean_delta, rtol=1e-12)
    assert (desc.n_wins, desc.n_ties, desc.n_losses) == (
        tested.n_wins,
        tested.n_ties,
        tested.n_losses,
    )


def test_apply_holm_across_contrasts_skips_descriptive() -> None:
    inputs = _three_contrast_inputs(n=10)
    arms = {
        "isalsr_vs_baseline": ("baseline", "isalsr"),
        "isalsr_vs_hash": ("hash", "isalsr"),
        "hash_vs_baseline": ("baseline", "hash"),
    }
    results: dict[str, dict[str, CrossProblemDominanceResult]] = {}
    for name, ps in inputs.items():
        a, b = arms[name]
        results[name] = {
            m: compute_cross_problem_dominance(
                ps, m, method="bingo", benchmark="benchmark", arm_a=a, arm_b=b
            )
            for m in _CPDT_METRICS
        }
    apply_holm_across_contrasts(results)
    assert results["isalsr_vs_baseline"]["redundancy_rate"].p_value_holm is None
    assert results["hash_vs_baseline"]["redundancy_rate"].p_value_holm is None
    solo = results["isalsr_vs_hash"]["redundancy_rate"]
    assert solo.p_value_holm == pytest.approx(solo.p_value_one_sided)


# ----------------------------------------------------------------------
# (d) Schema round-trip and legacy tolerance
# ----------------------------------------------------------------------


def _result_kwargs() -> dict[str, Any]:
    return {
        "method": "bingo",
        "benchmark": "benchmark",
        "metric": "r2_test",
        "alternative": "greater",
        "n_problems": 3,
        "n_wins": 2,
        "n_ties": 1,
        "n_losses": 0,
        "problem_names": ["P000", "P001", "P002"],
        "problem_deltas": [0.1, 0.0, 0.2],
        "shapiro_wilk_p": 0.5,
        "normality_assumed": True,
        "test_used": "t_one_sample",
        "statistic": 1.5,
        "p_value_one_sided": 0.04,
        "p_value_two_sided": 0.08,
        "cohens_d": 0.8,
        "cohens_d_ci_lower": 0.1,
        "cohens_d_ci_upper": 1.5,
        "mean_delta": 0.1,
        "mean_delta_ci_lower": 0.0,
        "mean_delta_ci_upper": 0.2,
    }


def test_schema_defaults_and_roundtrip() -> None:
    res = CrossProblemDominanceResult(**_result_kwargs())
    assert res.arm_a == "baseline"
    assert res.arm_b == "isalsr"
    assert res.p_value_holm is None
    res.arm_a, res.arm_b, res.p_value_holm = "hash", "isalsr", 0.12
    again = CrossProblemDominanceResult.from_dict(res.to_dict())
    assert again == res


def test_schema_from_dict_accepts_legacy_dict() -> None:
    legacy = _result_kwargs()
    assert "arm_a" not in legacy and "p_value_holm" not in legacy
    res = CrossProblemDominanceResult.from_dict(legacy)
    assert (res.arm_a, res.arm_b, res.p_value_holm) == ("baseline", "isalsr", None)


def test_schema_from_dict_ignores_unknown_keys() -> None:
    payload = _result_kwargs() | {"some_future_field": 1}
    res = CrossProblemDominanceResult.from_dict(payload)
    assert res.metric == "r2_test"
