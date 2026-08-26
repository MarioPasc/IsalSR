"""Unit tests for hard-tier benchmark definitions.

Validates each of the 10 hard problems:
    - target_fn evaluates correctly at canonical points
    - generate_data returns expected shapes for each sampling protocol
    - Outputs are finite (no NaN, no Inf) for the canonical seed
    - Per-protocol invariants:
        * Pagie-1 grid contains no x=0 or y=0
        * Korns-12 yields exactly 2000 train / 2000 test
        * Keijzer-6 test extrapolates beyond training range
        * Vladislavleva-4 yields 1024 / 5000
        * Vladislavleva-2 yields 100 / 221
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.datasets.hard import (
    HARD_BENCHMARKS,
    generate_data,
    get_benchmark,
    harmonic,
)

# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------


def test_registry_has_ten_problems() -> None:
    assert len(HARD_BENCHMARKS) == 10


def test_registry_names_are_unique() -> None:
    names = [b["name"] for b in HARD_BENCHMARKS]
    assert len(names) == len(set(names))


def test_get_benchmark_lookup() -> None:
    for b in HARD_BENCHMARKS:
        assert get_benchmark(b["name"]) is b


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown hard benchmark"):
        get_benchmark("DoesNotExist")


def test_each_dict_has_required_keys() -> None:
    required = {
        "name",
        "expression",
        "sympy_expression",
        "sympy_variables",
        "num_variables",
        "var_ranges",
        "target_fn",
        "sampling",
    }
    for b in HARD_BENCHMARKS:
        assert required <= set(b.keys()), f"{b['name']} missing keys"


# ----------------------------------------------------------------------
# Target function correctness at canonical points
# ----------------------------------------------------------------------


def test_i15_10_canonical_values() -> None:
    bench = get_benchmark("I.15.10")
    # m0=2, v=1, c=2 -> v/c = 0.5 -> sqrt(1-0.25)=sqrt(0.75) -> 2*1/sqrt(0.75)
    y = bench["target_fn"](np.array([2.0]), np.array([1.0]), np.array([2.0]))
    assert np.isclose(y[0], 2.0 / np.sqrt(0.75))


def test_i30_3_canonical_values() -> None:
    bench = get_benchmark("I.30.3")
    # n=2, theta=2 -> sin(2)^2 / sin(1)^2
    y = bench["target_fn"](np.array([2.0]), np.array([2.0]))
    assert np.isclose(y[0], (np.sin(2.0) ** 2) / (np.sin(1.0) ** 2))


def test_i37_4_canonical_values() -> None:
    bench = get_benchmark("I.37.4")
    # I1=4, I2=4, delta=0 -> 4 + 4 + 2*4*1 = 16
    y = bench["target_fn"](np.array([4.0]), np.array([4.0]), np.array([0.0]))
    assert np.isclose(y[0], 16.0)


def test_ii11_27_canonical_values() -> None:
    """Clausius-Mossotti polarisation, ``n*alpha/(1 - n*alpha/3)*epsilon*Ef``.

    Definition corrected 2026-08-02 to match the AI Feynman database; see
    ``docs/md_files/changes/feynman_definition_corrections.md``.
    """
    bench = get_benchmark("II.11.27")
    # n=alpha=0 -> the whole numerator vanishes.
    y = bench["target_fn"](np.array([0.0]), np.array([0.0]), np.array([1.5]), np.array([2.0]))
    assert np.isclose(y[0], 0.0)
    # n=alpha=1, eps=Ef=1 -> 1/(1 - 1/3) = 3/2
    y2 = bench["target_fn"](np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert np.isclose(y2[0], 1.5)
    # Linear in epsilon and in Ef.
    y3 = bench["target_fn"](np.array([1.0]), np.array([1.0]), np.array([2.0]), np.array([2.0]))
    assert np.isclose(y3[0], 1.5 * 4.0)


def test_iii17_37_canonical_values() -> None:
    """Angular distribution ``beta*(1 + alpha*cos(theta))``.

    Definition corrected 2026-08-02 to match the AI Feynman database; see
    ``docs/md_files/changes/feynman_definition_corrections.md``.
    """
    bench = get_benchmark("III.17.37")
    # theta = pi/2 -> cos = 0 -> f = beta
    y = bench["target_fn"](np.array([3.0]), np.array([4.0]), np.array([np.pi / 2]))
    assert np.isclose(y[0], 3.0)
    # theta = 0 -> cos = 1 -> f = beta*(1 + alpha)
    y2 = bench["target_fn"](np.array([2.0]), np.array([4.0]), np.array([0.0]))
    assert np.isclose(y2[0], 10.0)


def test_pagie1_canonical_values() -> None:
    bench = get_benchmark("Pagie-1")
    # x=1, y=1 -> 1/(1+1) + 1/(1+1) = 1.0
    y = bench["target_fn"](np.array([1.0]), np.array([1.0]))
    assert np.isclose(y[0], 1.0)
    # x=2, y=2 -> 1/(1+1/16) + 1/(1+1/16) = 2 * 16/17
    y2 = bench["target_fn"](np.array([2.0]), np.array([2.0]))
    assert np.isclose(y2[0], 2 * 16.0 / 17.0)


def test_korns12_canonical_values() -> None:
    bench = get_benchmark("Korns-12")
    # cos(0)*sin(0) = 0 -> y = 2.0
    z = np.array([0.0])
    y = bench["target_fn"](z, z, z, z, z)
    assert np.isclose(y[0], 2.0)


def test_vlad4_canonical_values() -> None:
    bench = get_benchmark("Vladislavleva-4")
    # All x_i = 3 -> denominator = 5 + 0 = 5 -> y = 10/5 = 2
    z = np.array([3.0])
    y = bench["target_fn"](z, z, z, z, z)
    assert np.isclose(y[0], 2.0)


def test_vlad2_canonical_values() -> None:
    bench = get_benchmark("Vladislavleva-2")
    x = np.array([1.0])
    expected = (
        np.exp(-1.0) * 1.0 * (np.cos(1.0) * np.sin(1.0)) * (np.cos(1.0) * np.sin(1.0) ** 2 - 1.0)
    )
    y = bench["target_fn"](x)
    assert np.isclose(y[0], expected)


def test_keijzer6_canonical_values() -> None:
    bench = get_benchmark("Keijzer-6")
    # H(1) = 1
    y = bench["target_fn"](np.array([1.0]))
    assert np.isclose(y[0], 1.0)
    # H(2) = 1.5
    y = bench["target_fn"](np.array([2.0]))
    assert np.isclose(y[0], 1.5)
    # H(4) = 1 + 0.5 + 1/3 + 0.25 = 25/12
    y = bench["target_fn"](np.array([4.0]))
    assert np.isclose(y[0], 25.0 / 12.0)
    # Vector input
    y = bench["target_fn"](np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(y, [1.0, 1.5, 11.0 / 6.0, 25.0 / 12.0])


def test_harmonic_helper_matches_eulermascheroni_for_large_n() -> None:
    # H(10000) ~ ln(10000) + gamma (Euler-Mascheroni 0.5772)
    n = np.array([10000.0])
    y = harmonic(n)
    expected = math.log(10000.0) + 0.5772156649015329
    assert abs(y[0] - expected) < 5e-5


# ----------------------------------------------------------------------
# Sampling protocol shapes & determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_train,expected_test",
    [
        ("I.15.10", 1000, 250),
        ("I.30.3", 1000, 250),
        ("I.37.4", 1000, 250),
        ("II.11.27", 1000, 250),
        ("III.17.37", 1000, 250),
        ("Korns-12", 2000, 2000),
        ("Vladislavleva-4", 1024, 5000),
        ("Vladislavleva-2", 100, 221),
        ("Keijzer-6", 50, 120),
    ],
)
def test_sampling_sizes(name: str, expected_train: int, expected_test: int) -> None:
    bench = get_benchmark(name)
    x_train, y_train, x_test, y_test = generate_data(
        bench, n_samples=1250, train_ratio=0.8, seed=42
    )
    assert x_train.shape[0] == expected_train, f"{name} train size"
    assert x_test.shape[0] == expected_test, f"{name} test size"
    assert y_train.shape[0] == expected_train
    assert y_test.shape[0] == expected_test
    assert x_train.shape[1] == bench["num_variables"]
    assert x_test.shape[1] == bench["num_variables"]


def test_pagie1_grid_size_676_2500() -> None:
    bench = get_benchmark("Pagie-1")
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    assert x_train.shape == (676, 2)
    assert x_test.shape == (2500, 2)
    assert y_train.shape == (676,)
    assert y_test.shape == (2500,)


def test_pagie1_grid_skips_zero() -> None:
    bench = get_benchmark("Pagie-1")
    x_train, _, x_test, _ = generate_data(bench, seed=42)
    # No grid point should have x=0 or y=0 (where 1/x^4 is undefined).
    assert not np.any(np.isclose(x_train[:, 0], 0.0, atol=1e-9))
    assert not np.any(np.isclose(x_train[:, 1], 0.0, atol=1e-9))
    assert not np.any(np.isclose(x_test[:, 0], 0.0, atol=1e-9))
    assert not np.any(np.isclose(x_test[:, 1], 0.0, atol=1e-9))


def test_keijzer6_test_extrapolates_train() -> None:
    bench = get_benchmark("Keijzer-6")
    x_train, _, x_test, _ = generate_data(bench, seed=42)
    train_max = x_train.max()
    test_max = x_test.max()
    assert test_max > train_max  # extrapolation
    assert train_max == 50.0
    assert test_max == 120.0
    # Train domain is fully integer.
    assert np.all(np.equal(x_train, np.rint(x_train)))


def test_keijzer6_target_is_harmonic_not_log() -> None:
    bench = get_benchmark("Keijzer-6")
    x_train, y_train, _, _ = generate_data(bench, seed=42)
    # y[0] = H(1) = 1.0 exactly; ln(1) = 0; differentiating signal.
    assert np.isclose(y_train[0], 1.0)
    # y[1] = H(2) = 1.5
    assert np.isclose(y_train[1], 1.5)


def test_vlad2_test_grid_step() -> None:
    bench = get_benchmark("Vladislavleva-2")
    _, _, x_test, _ = generate_data(bench, seed=42)
    diffs = np.diff(x_test[:, 0])
    assert np.allclose(diffs, 0.05, atol=1e-9)


def test_uniform_sampling_is_seed_deterministic() -> None:
    # Same seed -> identical sample for any uniform-protocol problem.
    bench = get_benchmark("II.11.27")
    a = generate_data(bench, seed=7)[0]
    b = generate_data(bench, seed=7)[0]
    assert np.array_equal(a, b)
    # Different seeds -> different samples.
    c = generate_data(bench, seed=8)[0]
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# Numerical sanity (no NaN / Inf in y for the canonical seed)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", HARD_BENCHMARKS, ids=lambda b: b["name"])
def test_outputs_finite_for_canonical_seed(bench: dict) -> None:
    _, y_train, _, y_test = generate_data(bench, seed=42)
    assert np.all(np.isfinite(y_train)), f"{bench['name']} train NaN/Inf"
    assert np.all(np.isfinite(y_test)), f"{bench['name']} test NaN/Inf"


# ----------------------------------------------------------------------
# SymPy ground-truth presence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", HARD_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_expression_evaluates(bench: dict) -> None:
    """Every problem ships a pre-built SymPy expression that evaluates
    numerically (so the orchestrator's solution_recovery comparison can run).
    """
    import sympy

    expr = bench["sympy_expression"]
    syms = bench["sympy_variables"]
    assert isinstance(expr, sympy.Expr)
    assert len(syms) == bench["num_variables"]
    # Substitute test point: lower bound of each var.
    subs = {s: lo for s, (lo, _) in zip(syms, bench["var_ranges"], strict=True)}
    val = float(expr.subs(subs).evalf())
    assert math.isfinite(val), f"{bench['name']} SymPy eval not finite: {val}"
