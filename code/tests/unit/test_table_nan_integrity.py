"""Regression tests for NaN handling and seed pairing in the LaTeX table generator.

These cover three defects found by T08 (reviewer comment R2.7) in
``experiments/figures/models/generate_tables.py``. Each test fails against the
pre-fix code:

D1. ``_fmt_bold_underline`` marked a NaN as the better value. ``bl > is`` is
    ``False`` whenever ``is`` is NaN, so control fell through to the ``else``
    branch and the NaN was emitted bold while a finite value was underlined.
    In the submitted appendix this typeset a real ``R^2 = 0.9385`` as worse
    than a missing observation.

D2. Per-problem cells used ``np.mean``, so a single NaN seed out of 30 made the
    whole cell NaN. The analyzer pipeline (``analyzer/aggregation.py``) uses
    ``np.nanmean`` for the same quantity, so the appendix table and the headline
    test disagreed by construction.

D3. Paired statistics were formed by positional truncation ``bl[:n], is_[:n]``
    with ``n = min(len(bl), len(is_))``. Seed identity is not carried, so a
    single missing cell shifts every subsequent pair onto mismatched seeds.
    14 of 50 Bingo problems in the submitted campaign were affected.

The invariant the fix establishes: a NaN is a *missing observation*, never a
value that wins, and pairing is by seed number, never by list position.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.figures.models.generate_tables import (
    _cohens_d_with_ci,
    _fmt_bold_underline,
    _nanmean,
    _pair_by_seed,
    _paired_test,
)

# ----------------------------------------------------------------------
# D1. NaN must never be marked as the better value
# ----------------------------------------------------------------------


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_nan_is_never_bold(higher_is_better: bool) -> None:
    """A NaN on either side must never receive the bold (better) mark."""
    finite, missing = 0.9385, float("nan")

    bl_s, is_s = _fmt_bold_underline(finite, missing, ".4f", higher_is_better)
    assert "mathbf" not in is_s, "NaN was marked as the better value"
    assert "mathbf" in bl_s, "the finite value should win against a missing one"

    # ... and symmetrically with the NaN on the baseline side.
    bl_s, is_s = _fmt_bold_underline(missing, finite, ".4f", higher_is_better)
    assert "mathbf" not in bl_s, "NaN was marked as the better value"
    assert "mathbf" in is_s, "the finite value should win against a missing one"


def test_nan_renders_as_explicit_marker_not_bold() -> None:
    """The submitted table's exact Vlad-2 row: 0.9385 vs NaN, higher is better."""
    bl_s, is_s = _fmt_bold_underline(0.9385, float("nan"), ".4f", higher_is_better=True)
    assert "underline" not in bl_s, "a real R^2 was typeset as worse than a missing value"
    assert "nan" not in is_s.lower() or "mathbf" not in is_s


def test_both_nan_gets_no_marks() -> None:
    """Two missing observations are not comparable; neither may be bold."""
    nan = float("nan")
    bl_s, is_s = _fmt_bold_underline(nan, nan, ".4f", higher_is_better=True)
    assert "mathbf" not in bl_s and "mathbf" not in is_s
    assert "underline" not in bl_s and "underline" not in is_s


def test_finite_comparison_is_unchanged() -> None:
    """The fix must not alter behaviour on ordinary finite input."""
    bl_s, is_s = _fmt_bold_underline(0.90, 0.95, ".4f", higher_is_better=True)
    assert "underline" in bl_s and "mathbf" in is_s

    bl_s, is_s = _fmt_bold_underline(0.10, 0.05, ".4f", higher_is_better=False)
    assert "underline" in bl_s and "mathbf" in is_s

    # Equal-displaying values get neither mark.
    bl_s, is_s = _fmt_bold_underline(0.00001, 0.00002, ".4f", higher_is_better=True)
    assert "mathbf" not in bl_s and "underline" not in is_s


# ----------------------------------------------------------------------
# D2. One NaN seed must not destroy the whole cell
# ----------------------------------------------------------------------


def test_one_nan_seed_does_not_nan_the_cell() -> None:
    """29 finite seeds + 1 NaN must aggregate to the mean of the 29."""
    vals = [0.99] * 29 + [float("nan")]
    got = _nanmean(vals)
    assert math.isfinite(got)
    assert got == pytest.approx(0.99)
    assert math.isnan(float(np.mean(vals))), "pre-fix behaviour: np.mean propagates"


def test_nanmean_matches_analyzer_pipeline() -> None:
    """The table generator must agree with analyzer/aggregation.py on the same input."""
    vals = np.array([0.9] * 10 + [float("nan")] + [0.8] * 5)
    assert _nanmean(list(vals)) == pytest.approx(float(np.nanmean(vals)))


def test_nanmean_of_all_missing_is_nan_not_zero() -> None:
    """An entirely missing cell stays missing; it must not silently become 0.0."""
    assert math.isnan(_nanmean([float("nan")] * 5))
    assert math.isnan(_nanmean([]))


def test_nanmean_ignores_infinities() -> None:
    """inf is an evaluation failure, not a large value."""
    assert _nanmean([1.0, 3.0, float("inf")]) == pytest.approx(2.0)
    assert _nanmean([1.0, 3.0, float("-inf")]) == pytest.approx(2.0)


# ----------------------------------------------------------------------
# D3. Pairing must be by seed, not by list position
# ----------------------------------------------------------------------


def test_pair_by_seed_aligns_on_seed_number() -> None:
    """A gap in one arm must not shift the other arm's pairing."""
    bl = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    is_ = {1: 1.1, 3: 1.3, 4: 1.4}  # seed 2 missing
    b, i = _pair_by_seed(bl, is_)
    assert list(b) == [0.1, 0.3, 0.4]
    assert list(i) == [1.1, 1.3, 1.4]


def test_pair_by_seed_reproduces_the_korns12_shape() -> None:
    """The real failure: baseline has 28 seeds, IsalSR 21, overlapping partially.

    Positional truncation pairs baseline seed 2 with IsalSR seed 3 and stays
    wrong for every subsequent index.
    """
    bl_seeds = [s for s in range(1, 31) if s not in (26, 29)]
    is_seeds = [s for s in range(1, 31) if s not in (2, 4, 6, 13, 19, 20, 22, 23, 24)]
    bl = {s: float(s) for s in bl_seeds}
    is_ = {s: float(s) for s in is_seeds}

    b, i = _pair_by_seed(bl, is_)
    assert np.array_equal(b, i), "paired values must come from the same seed"
    # 21 IsalSR seeds, of which 26 and 29 are absent from the baseline arm.
    assert len(b) == len(set(bl_seeds) & set(is_seeds)) == 19

    # Demonstrate the pre-fix defect explicitly.
    n = min(len(bl_seeds), len(is_seeds))
    positional_bl = np.array([bl[s] for s in bl_seeds])[:n]
    positional_is = np.array([is_[s] for s in is_seeds])[:n]
    assert not np.array_equal(positional_bl, positional_is), (
        "positional truncation should mispair -- if this passes, the fixture is wrong"
    )


def test_pair_by_seed_drops_nan_pairs() -> None:
    """Pairwise deletion: a NaN on either side removes that pair, not the cell."""
    bl = {1: 0.1, 2: 0.2, 3: float("nan"), 4: 0.4}
    is_ = {1: 1.1, 2: float("nan"), 3: 1.3, 4: 1.4}
    b, i = _pair_by_seed(bl, is_)
    assert list(b) == [0.1, 0.4]
    assert list(i) == [1.1, 1.4]


def test_paired_test_uses_only_common_seeds() -> None:
    """_paired_test must be invariant to a missing seed in the other arm."""
    rng = np.random.default_rng(0)
    base = {s: float(v) for s, v in enumerate(rng.normal(0, 1, 30), start=1)}
    # A real effect plus noise, so the paired difference has non-zero variance.
    noise = rng.normal(0, 0.1, 30)
    treat = {s: v + 0.5 + float(noise[i]) for i, (s, v) in enumerate(base.items())}

    d_full, p_full = _paired_test(base, treat)

    # Remove one seed from the treatment arm only.
    treat_gap = {s: v for s, v in treat.items() if s != 7}
    d_gap, p_gap = _paired_test(base, treat_gap)

    assert d_full > 1.0 and d_gap > 1.0, "a large positive effect must survive the gap"
    assert math.isfinite(p_full) and math.isfinite(p_gap)
    assert p_full < 0.01 and p_gap < 0.01


def test_paired_test_survives_a_nan_seed() -> None:
    """A NaN seed must reduce the pair count, not make d and p undefined."""
    rng = np.random.default_rng(1)
    base = {s: float(v) for s, v in enumerate(rng.normal(0, 1, 30), start=1)}
    treat = {s: v + 0.5 for s, v in base.items()}
    treat[13] = float("nan")

    d, p = _paired_test(base, treat)
    assert math.isfinite(d) and math.isfinite(p)


def test_cohens_d_ci_survives_a_nan_seed() -> None:
    """The bootstrap CI must be finite when one of 30 seeds is missing."""
    rng = np.random.default_rng(2)
    base = {s: float(v) for s, v in enumerate(rng.normal(0, 1, 30), start=1)}
    treat = {s: v + 0.5 for s, v in base.items()}
    treat[5] = float("nan")

    d, lo, hi = _cohens_d_with_ci(base, treat, n_boot=200)
    assert math.isfinite(d) and math.isfinite(lo) and math.isfinite(hi)
    assert lo <= d <= hi


def test_too_few_common_seeds_is_not_a_silent_zero() -> None:
    """Fewer than three paired seeds must not be reported as d = 0, p = 1."""
    bl = {1: 0.1, 2: 0.2}
    is_ = {5: 1.1, 6: 1.2}  # no overlap at all
    d, p = _paired_test(bl, is_)
    assert math.isnan(d) and math.isnan(p)
