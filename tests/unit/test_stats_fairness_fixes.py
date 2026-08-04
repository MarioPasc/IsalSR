"""Fairness fixes in the paired-statistics pipeline (audit, 2026-08-04).

Four defects are pinned here, each with the regression direction made explicit
so a future change that reverts one is caught by a failing inequality rather
than by a silently different p-value:

1. **CPDT tie policy** (``compute_cross_problem_dominance``). Ties were counted
   with a 1e-6 threshold for display but the *raw* delta vector was handed to
   the test, and SciPy's default ``zero_method="wilcox"`` then discarded exact
   zeros. Both make the test anti-conservative when the non-tied deltas lean
   one way. The tested vector is now the snapped vector and zeros are kept via
   ``zero_method="zsplit"`` (Pratt 1959, JASA 54(287):655-667; Demsar 2006,
   JMLR 7:1-30).
2. **Supplementary per-problem test** (``_paired_test``). An exception returned
   ``p = 1.0``, indistinguishable from a decided null; it now returns NaN.
3. **Cross-method Friedman**. Columns were built from bare arrays that could
   misalign across variants, and the critical-difference machinery ranks larger
   values first regardless of metric direction, inverting the CD diagram for
   ``nrmse_test`` and ``wall_clock_total_s``.
4. **SymPy failure accounting**. A generic exception scored ``False``/``0.0``;
   it is now ``None`` (excluded), matching the timeout policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats as sp_stats

from experiments.figures.models.generate_tables import _paired_test
from experiments.models.analyzer import cross_method as cm
from experiments.models.analyzer.aggregation import compute_cross_problem_dominance
from experiments.models.analyzer.statistical_tests import critical_difference_data
from experiments.models.schemas import PairedStats, PairedStatsMetric

TIE_THRESHOLD = 1e-6


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _metric(delta: float) -> PairedStatsMetric:
    """Build a PairedStatsMetric whose isalsr - baseline mean equals ``delta``."""
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


def _paired_stats(deltas: list[float], metric_name: str = "r2_test") -> list[PairedStats]:
    """Wrap a delta vector as one PairedStats per problem."""
    return [
        PairedStats(
            method="bingo",
            benchmark="benchmark",
            problem=f"P{i:03d}",
            metrics={metric_name: _metric(d)},
            n_seeds=30,
        )
        for i, d in enumerate(deltas)
    ]


def _cpdt(deltas: list[float]) -> Any:
    return compute_cross_problem_dominance(
        _paired_stats(deltas),
        metric_name="r2_test",
        method="bingo",
        benchmark="benchmark",
    )


# Sub-threshold noise that leans positive: 40 of the 60 ties are numerically
# above zero. This is the configuration in which the old code was
# anti-conservative -- the noise entered the signed-rank test as evidence.
_NOISE_LEANING_POSITIVE = [
    *np.linspace(1e-8, 5e-7, 40).tolist(),
    *np.linspace(-5e-7, -1e-8, 20).tolist(),
]
_BIG = [0.01] * 7 + [-0.01] * 3


# ----------------------------------------------------------------------
# 1. CPDT tie policy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "ties"),
    [
        ("subthreshold_noise", _NOISE_LEANING_POSITIVE),
        ("exact_zeros", [0.0] * 60),
    ],
)
def test_cpdt_tie_counts_come_from_the_tested_vector(label: str, ties: list[float]) -> None:
    """W/T/L are computed from the same snapped vector the test consumes."""
    res = _cpdt([*ties, *_BIG])

    assert res.n_problems == 70
    assert (res.n_wins, res.n_ties, res.n_losses) == (7, 60, 3)
    assert res.n_wins + res.n_ties + res.n_losses == res.n_problems


@pytest.mark.parametrize(
    ("label", "ties"),
    [
        ("subthreshold_noise", _NOISE_LEANING_POSITIVE),
        ("exact_zeros", [0.0] * 60),
    ],
)
def test_cpdt_p_matches_zsplit_on_snapped_vector(label: str, ties: list[float]) -> None:
    """The reported p is exactly Wilcoxon(zsplit) on the snapped deltas."""
    deltas = [*ties, *_BIG]
    res = _cpdt(deltas)

    d_arr = np.array(deltas)
    snapped = np.where(np.abs(d_arr) <= TIE_THRESHOLD, 0.0, d_arr)
    expected = sp_stats.wilcoxon(snapped, alternative="greater", zero_method="zsplit")

    assert res.test_used == "wilcoxon_signed_rank"
    np.testing.assert_allclose(res.p_value_one_sided, float(expected.pvalue), rtol=1e-12)
    np.testing.assert_allclose(res.statistic, float(expected.statistic), rtol=1e-12)


@pytest.mark.parametrize(
    ("label", "ties"),
    [
        ("subthreshold_noise", _NOISE_LEANING_POSITIVE),
        ("exact_zeros", [0.0] * 60),
    ],
)
def test_cpdt_is_more_conservative_than_the_old_behaviour(label: str, ties: list[float]) -> None:
    """Regression direction: the fix must raise p, never lower it.

    The old call was ``wilcoxon(raw_deltas, alternative=...)`` with SciPy's
    default ``zero_method="wilcox"``. Either defect alone -- unsnapped noise or
    dropped zeros -- lets 60 tied problems act as evidence for the alternative.
    """
    deltas = [*ties, *_BIG]
    res = _cpdt(deltas)
    old_p = float(sp_stats.wilcoxon(np.array(deltas), alternative="greater").pvalue)

    assert res.p_value_one_sided > old_p


def test_cpdt_all_ties_is_a_null_result() -> None:
    """Every delta below threshold: no test is defined, p = 1."""
    res = _cpdt(np.linspace(-1e-6, 1e-6, 41).tolist())

    assert res.test_used == "all_zeros"
    assert res.p_value_one_sided == 1.0
    assert res.p_value_two_sided == 1.0
    assert res.n_ties == res.n_problems == 41
    assert (res.n_wins, res.n_losses) == (0, 0)


def test_cpdt_normal_deltas_still_take_the_t_branch() -> None:
    """Snapping does not divert a well-behaved delta vector off the t-test."""
    rng = np.random.default_rng(7)
    deltas = (rng.normal(0.02, 0.01, 40)).tolist()
    res = _cpdt(deltas)

    d_test = np.array(deltas)  # no value is within the tie threshold
    expected = sp_stats.ttest_1samp(d_test, 0.0, alternative="greater")

    assert res.test_used == "t_one_sample"
    assert res.normality_assumed
    np.testing.assert_allclose(res.p_value_one_sided, float(expected.pvalue), rtol=1e-12)


def test_cpdt_effect_sizes_stay_on_raw_deltas() -> None:
    """Snapping is a test-decision rule; the reported mean delta is unsnapped."""
    deltas = [*_NOISE_LEANING_POSITIVE, *_BIG]
    res = _cpdt(deltas)

    np.testing.assert_allclose(res.mean_delta, float(np.mean(deltas)), rtol=1e-12)
    np.testing.assert_allclose(res.problem_deltas, deltas, rtol=0, atol=0)
    # The snap can move the mean by at most the threshold, so the two agree to
    # well inside any reported precision.
    snapped_mean = float(
        np.mean(np.where(np.abs(np.array(deltas)) <= TIE_THRESHOLD, 0.0, np.array(deltas)))
    )
    assert abs(res.mean_delta - snapped_mean) <= TIE_THRESHOLD


# ----------------------------------------------------------------------
# 2. Supplementary per-problem test
# ----------------------------------------------------------------------


def test_paired_test_all_zero_diffs_is_an_explicit_null() -> None:
    """Identical arms: d = 0 and p = 1, without invoking an undefined test."""
    values = {s: 0.5 for s in range(10)}
    assert _paired_test(values, dict(values)) == (0.0, 1.0)


def test_paired_test_too_few_seeds_is_nan() -> None:
    """Fewer than three pairs is undetermined, not a null."""
    d, p = _paired_test({0: 1.0, 1: 2.0}, {0: 1.5, 1: 2.5})
    assert np.isnan(d)
    assert np.isnan(p)


def test_paired_test_exception_is_nan_not_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SciPy failure must not be reported as a decided null (p = 1.0)."""
    import experiments.figures.models.generate_tables as gt

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("shapiro exploded")

    monkeypatch.setattr(gt.sp_stats, "shapiro", _boom)

    d, p = _paired_test({s: float(s) for s in range(10)}, {s: float(s) + 0.3 for s in range(10)})
    assert np.isnan(p), "an exception must propagate as NaN, not as p = 1.0"
    assert not np.isnan(d), "Cohen's d is computed before the test and stays finite"


def test_paired_test_wilcoxon_keeps_ties() -> None:
    """Non-normal diffs with ties use zsplit, matching the CPDT tie policy."""
    bl = {s: 0.0 for s in range(20)}
    is_ = {s: (0.0 if s < 14 else 1.0) for s in range(20)}
    _, p = _paired_test(bl, is_)

    b = np.array([bl[s] for s in range(20)])
    i = np.array([is_[s] for s in range(20)])
    expected = sp_stats.wilcoxon(b, i, zero_method="zsplit").pvalue
    np.testing.assert_allclose(p, float(expected), rtol=1e-12)


# ----------------------------------------------------------------------
# 3. Cross-method Friedman construction
# ----------------------------------------------------------------------


def _synthetic_groups(
    problems: list[str],
    values: dict[str, list[float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Build a {method: {variant: {problem: value}}} bundle for two methods."""
    keys = ["udfs_baseline", "udfs_isalsr", "bingo_baseline", "bingo_isalsr"]
    out: dict[str, dict[str, dict[str, float]]] = {"udfs": {}, "bingo": {}}
    for k in keys:
        method, variant = k.split("_")
        out[method][variant] = dict(zip(problems, values[k], strict=True))
    return out


_PROBLEMS = [f"P{i}" for i in range(8)]
_VALUES = {
    "udfs_baseline": [10.0, 9.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1],
    "udfs_isalsr": [11.0, 10.0, 12.0, 11.5, 10.5, 11.2, 10.8, 11.1],
    "bingo_baseline": [5.0, 4.0, 6.0, 5.5, 4.5, 5.2, 4.8, 5.1],
    "bingo_isalsr": [6.0, 5.0, 7.0, 6.5, 5.5, 6.2, 5.8, 6.1],
}


def _run_friedman(
    monkeypatch: pytest.MonkeyPatch,
    bundle: dict[str, dict[str, dict[str, float]]],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    monkeypatch.setattr(
        cm,
        "load_cross_method_results",
        lambda *_args, **_kwargs: bundle,
    )
    return cm.cross_method_friedman(
        Path("/nonexistent"),
        ["udfs", "bingo"],
        "benchmark",
        lambda rl: 0.0,
        higher_is_better=higher_is_better,
    )


def test_direction_flips_ranks_but_not_chi2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negating for a lower-is-better metric reorients ranks, not the test."""
    bundle = _synthetic_groups(_PROBLEMS, _VALUES)
    hi = _run_friedman(monkeypatch, bundle, higher_is_better=True)
    lo = _run_friedman(monkeypatch, bundle, higher_is_better=False)

    np.testing.assert_allclose(hi["chi2"], lo["chi2"], rtol=1e-12)
    np.testing.assert_allclose(hi["p_value"], lo["p_value"], rtol=1e-12)
    np.testing.assert_allclose(hi["cd_value"], lo["cd_value"], rtol=1e-12)

    hi_order = list(np.argsort(hi["avg_ranks"]))
    lo_order = list(np.argsort(lo["avg_ranks"]))
    assert hi_order == lo_order[::-1], "rank order must invert with the direction"
    assert hi["higher_is_better"] is True
    assert lo["higher_is_better"] is False


def test_int_matrix_yields_fractional_average_ranks() -> None:
    """An integer matrix must not truncate the .5 ranks that ties produce."""
    data = np.array([[1, 1, 2, 3], [1, 1, 2, 3]], dtype=int)
    result = critical_difference_data(data, ["a", "b", "c", "d"])

    assert result.avg_ranks.dtype == np.float64
    # Columns 0 and 1 tie for last (values 1, 1) -> average rank (3 + 4) / 2.
    np.testing.assert_allclose(result.avg_ranks, [3.5, 3.5, 2.0, 1.0], rtol=1e-12)


def test_non_finite_problem_is_dropped_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN group mean removes that problem and the name is reported."""
    values = {k: list(v) for k, v in _VALUES.items()}
    values["bingo_isalsr"][3] = float("nan")
    bundle = _synthetic_groups(_PROBLEMS, values)
    out = _run_friedman(monkeypatch, bundle, higher_is_better=True)

    assert out["n_problems"] == 7
    assert out["n_problems_dropped"] == 1
    assert out["dropped_problems"] == ["P3"]
    assert "P3" not in out["problem_names"]


def test_problem_missing_from_one_variant_is_excluded_and_columns_stay_aligned() -> None:
    """Intersection on names, so no column is shifted by a missing problem."""
    values = {k: list(v) for k, v in _VALUES.items()}
    bundle = _synthetic_groups(_PROBLEMS, values)
    del bundle["bingo"]["isalsr"]["P2"]

    matrix, names, problems, dropped = cm.build_cross_method_matrix(bundle, ["udfs", "bingo"])

    assert dropped == ["P2"]
    assert problems == [p for p in _PROBLEMS if p != "P2"]
    assert matrix.shape == (7, 4)
    assert names == ["udfs_baseline", "udfs_isalsr", "bingo_baseline", "bingo_isalsr"]
    for col, key in enumerate(names):
        expected = [v for p, v in zip(_PROBLEMS, values[key], strict=True) if p != "P2"]
        np.testing.assert_allclose(matrix[:, col], expected, rtol=1e-12)


def test_load_cross_method_results_means_over_finite_values_only(tmp_path: Path) -> None:
    """Non-finite per-seed values are skipped; an all-non-finite problem is NaN."""

    class _RL:
        def __init__(self, v: float) -> None:
            self.v = v

    written: dict[Path, list[_RL]] = {}
    for problem, vals in (("P0", [1.0, np.inf, 3.0]), ("P1", [np.nan, np.inf])):
        for variant in ("baseline", "isalsr"):
            d = tmp_path / "udfs" / "benchmark" / problem / variant
            d.mkdir(parents=True)
            written[d] = [_RL(v) for v in vals]

    import experiments.models.analyzer.cross_method as mod

    original = mod.load_all_run_logs
    try:
        mod.load_all_run_logs = lambda d: written.get(Path(d), [])  # type: ignore[assignment]
        out = mod.load_cross_method_results(tmp_path, ["udfs"], "benchmark", lambda rl: float(rl.v))
    finally:
        mod.load_all_run_logs = original  # type: ignore[assignment]

    np.testing.assert_allclose(out["udfs"]["baseline"]["P0"], 2.0, rtol=1e-12)
    assert np.isnan(out["udfs"]["isalsr"]["P1"])


# ----------------------------------------------------------------------
# 4. SymPy failure accounting
# ----------------------------------------------------------------------


def test_solution_recovered_returns_none_when_subtraction_raises() -> None:
    """A SymPy blowup during the difference is undetermined, not a failure."""
    sympy = pytest.importorskip("sympy")
    from experiments.models.analyzer.metrics import solution_recovered

    class _Exploding:
        free_symbols: set[Any] = set()

        def __sub__(self, _other: Any) -> Any:
            raise ValueError("cannot subtract")

        def __rsub__(self, _other: Any) -> Any:
            raise ValueError("cannot subtract")

    x = sympy.Symbol("x")
    assert solution_recovered(_Exploding(), x + 1) is None


def test_jaccard_returns_none_when_normalisation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SymPy blowup during normalisation is undetermined, not 0.0.

    The failure is injected in ``sympy.simplify`` because that is where the
    real blowups happen; a non-``sympy.Basic`` input does *not* reach the
    handler (``_get_subexpressions`` returns the empty set for it and 0.0 comes
    out of the normal path).
    """
    sympy = pytest.importorskip("sympy")
    from experiments.models.analyzer import metrics as met

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("simplify exploded")

    monkeypatch.setattr(sympy, "simplify", _boom)

    x = sympy.Symbol("x")
    assert met.jaccard_index(x**2 + x, x**2) is None
