"""Regression tests for the vectorised Cohen's d bootstrap CI.

``cohens_d_ci_bootstrap`` was a Python loop over 10,000 resamples; it is now one
blocked draw. The CI bounds it returns are a **reported quantity** (Table 2, the
supplementary effect-size columns), so "faster and statistically equivalent" is
not good enough — the replacement must be **bit-identical**. That is what these
tests assert, against a reference implementation of the original loop kept here
verbatim.

If a NumPy upgrade ever changes how ``Generator.integers`` consumes the stream,
these tests fail rather than the numbers silently moving.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from experiments.models.analyzer.effect_sizes import (
    cohens_d_ci_bootstrap,
    cohens_d_paired,
)


def _reference_loop(
    differences: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """The pre-2026-08-05 implementation, verbatim. Do not optimise."""
    rng = np.random.default_rng(seed)
    n = len(differences)
    if n < 2:
        return 0.0, 0.0

    boot_ds = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(differences, size=n, replace=True)
        sd = np.std(sample, ddof=1)
        boot_ds[b] = np.mean(sample) / sd if sd > 1e-10 else 0.0

    alpha = 1 - ci
    lower = float(np.percentile(boot_ds, 100 * alpha / 2))
    upper = float(np.percentile(boot_ds, 100 * (1 - alpha / 2)))
    return lower, upper


# --------------------------------------------------------------------------
# Bit-identity — the property the change stands or falls on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 15, 20, 30, 50])
def test_bit_identical_to_the_reference_loop(n: int) -> None:
    rng = np.random.default_rng(7)
    d = rng.normal(0.01, 0.05, size=n)
    assert cohens_d_ci_bootstrap(d) == _reference_loop(d)


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_bit_identical_across_seeds(seed: int) -> None:
    d = np.random.default_rng(3).normal(0.0, 1.0, size=20)
    assert cohens_d_ci_bootstrap(d, seed=seed) == _reference_loop(d, seed=seed)


@pytest.mark.parametrize("n_boot", [10, 100, 1000, 10000])
def test_bit_identical_across_resample_counts(n_boot: int) -> None:
    d = np.random.default_rng(11).normal(0.02, 0.1, size=20)
    assert cohens_d_ci_bootstrap(d, n_boot=n_boot) == _reference_loop(d, n_boot=n_boot)


@pytest.mark.parametrize("ci", [0.80, 0.90, 0.95, 0.99])
def test_bit_identical_across_confidence_levels(ci: float) -> None:
    d = np.random.default_rng(13).normal(0.0, 0.3, size=20)
    assert cohens_d_ci_bootstrap(d, ci=ci) == _reference_loop(d, ci=ci)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("all-zero (sd == 0 on every resample)", [0.0] * 20),
        ("all-identical non-zero", [0.7] * 20),
        ("one outlier", [0.0] * 19 + [1e6]),
        ("mixed signs", [-1.0, 1.0] * 10),
        ("tiny magnitudes near the 1e-10 guard", [1e-12, -1e-12] * 10),
        ("R2-like saturation deltas", [0.0] * 15 + [1e-7] * 5),
        ("negatives only", [-0.5, -0.25, -0.75, -0.1, -0.9]),
    ],
)
def test_bit_identical_on_degenerate_inputs(name: str, values: list[float]) -> None:
    """The sd <= 1e-10 branch is where a vectorised rewrite most easily drifts."""
    d = np.asarray(values, dtype=float)
    assert cohens_d_ci_bootstrap(d) == _reference_loop(d), name


def test_bit_identical_on_a_wide_random_sweep() -> None:
    """200 random inputs at C2's seed count, so a rare branch cannot hide."""
    rng = np.random.default_rng(99)
    for _ in range(200):
        n = int(rng.integers(2, 31))
        scale = float(10 ** rng.uniform(-8, 3))
        d = rng.normal(0.0, scale, size=n)
        assert cohens_d_ci_bootstrap(d, n_boot=200) == _reference_loop(d, n_boot=200)


def test_integer_input_is_handled_like_the_loop() -> None:
    d = np.array([1, 2, 3, 4, 5])
    assert cohens_d_ci_bootstrap(d) == _reference_loop(d)


def test_list_input_is_handled_like_the_loop() -> None:
    """`np.asarray` was added by the rewrite; the loop accepted lists via choice."""
    d = [0.1, 0.2, 0.3, 0.4]
    assert cohens_d_ci_bootstrap(d) == _reference_loop(d)  # type: ignore[arg-type]


def test_input_is_not_mutated() -> None:
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    before = d.copy()
    cohens_d_ci_bootstrap(d)
    np.testing.assert_array_equal(d, before)


# --------------------------------------------------------------------------
# Contract preserved
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_lengths_short_circuit(n: int) -> None:
    assert cohens_d_ci_bootstrap(np.zeros(n)) == (0.0, 0.0)


def test_ci_brackets_the_point_estimate_on_a_clean_effect() -> None:
    d = np.random.default_rng(5).normal(1.0, 0.2, size=25)
    lo, hi = cohens_d_ci_bootstrap(d)
    assert lo <= cohens_d_paired(d) <= hi


def test_repeated_calls_are_deterministic() -> None:
    d = np.random.default_rng(17).normal(0.05, 0.1, size=20)
    assert cohens_d_ci_bootstrap(d) == cohens_d_ci_bootstrap(d)


# --------------------------------------------------------------------------
# The reason for the change
# --------------------------------------------------------------------------


def test_is_materially_faster_than_the_loop() -> None:
    """~95 % of C2's aggregation job was this loop; a regression would restore it.

    A deliberately loose threshold (10x, against 51x measured at n=20) so the
    test reports a genuine regression rather than CI jitter.
    """
    d = np.random.default_rng(23).normal(0.01, 0.05, size=20)

    t0 = time.perf_counter()
    _reference_loop(d)
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    cohens_d_ci_bootstrap(d)
    t_vec = time.perf_counter() - t0

    assert t_vec * 10 < t_loop, f"loop {t_loop:.4f}s vs vectorised {t_vec:.4f}s"
