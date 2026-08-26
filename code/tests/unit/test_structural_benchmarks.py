"""Unit tests for structural benchmark definitions.

Validates each of the 10 structural problems:
    - target_fn evaluates correctly at canonical points
    - generate_data returns expected shapes
    - Outputs are finite (no NaN, no Inf) for the canonical seed
    - Vlad-7 uses published 300/1200 protocol
    - II.11.28 domain stays safe (denominator > 0)
    - SymPy ground-truth expressions evaluate finitely
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.datasets.structural import (
    STRUCTURAL_BENCHMARKS,
    generate_data,
    get_benchmark,
)

# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------


def test_registry_has_ten_problems() -> None:
    assert len(STRUCTURAL_BENCHMARKS) == 10


def test_registry_names_are_unique() -> None:
    names = [b["name"] for b in STRUCTURAL_BENCHMARKS]
    assert len(names) == len(set(names))


def test_get_benchmark_lookup() -> None:
    for b in STRUCTURAL_BENCHMARKS:
        assert get_benchmark(b["name"]) is b


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown structural benchmark"):
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
    for b in STRUCTURAL_BENCHMARKS:
        assert required <= set(b.keys()), f"{b['name']} missing keys"


# ----------------------------------------------------------------------
# Target function correctness at canonical points
# ----------------------------------------------------------------------


def test_i29_16_canonical_values() -> None:
    bench = get_benchmark("I.29.16")
    # x1=3, x2=4, theta1=theta2=1 -> sqrt(9+16-24*cos(0)) = sqrt(1) = 1
    y = bench["target_fn"](np.array([3.0]), np.array([4.0]), np.array([1.0]), np.array([1.0]))
    assert np.isclose(y[0], 1.0)
    # x1=x2=1, theta1=theta2=0 -> sqrt(1+1-2*cos(0)) = sqrt(0) = 0
    y2 = bench["target_fn"](np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert np.isclose(y2[0], 0.0, atol=1e-10)


def test_i50_26_canonical_values() -> None:
    bench = get_benchmark("I.50.26")
    # x1=2, omega=0, t=1, alpha=1 -> 2*(cos(0)+1*cos(0)^2) = 2*(1+1) = 4
    y = bench["target_fn"](np.array([2.0]), np.array([0.0]), np.array([1.0]), np.array([1.0]))
    assert np.isclose(y[0], 4.0)


def test_i16_6_canonical_values() -> None:
    bench = get_benchmark("I.16.6")
    # c=5, v=1, u=1 -> (1+1)/(1+1/25) = 2/(26/25) = 50/26
    y = bench["target_fn"](np.array([5.0]), np.array([1.0]), np.array([1.0]))
    assert np.isclose(y[0], 50.0 / 26.0)


def test_ii11_28_canonical_values() -> None:
    bench = get_benchmark("II.11.28")
    # n=0, alpha=0 -> 1 + 0/(1-0) = 1
    y = bench["target_fn"](np.array([0.0]), np.array([0.0]))
    assert np.isclose(y[0], 1.0)
    # n=1, alpha=1 -> 1 + 1/(1-1/3) = 1 + 1/(2/3) = 1 + 1.5 = 2.5
    y2 = bench["target_fn"](np.array([1.0]), np.array([1.0]))
    assert np.isclose(y2[0], 2.5)


def test_iii14_14_canonical_values() -> None:
    bench = get_benchmark("III.14.14")
    # I0=1, q=1, V=0, kb=1, T=1 -> 1*(exp(0)-1) = 0
    y = bench["target_fn"](
        np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([1.0])
    )
    assert np.isclose(y[0], 0.0, atol=1e-10)
    # I0=1, q=1, V=1, kb=1, T=1 -> exp(1)-1
    y2 = bench["target_fn"](
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
    )
    assert np.isclose(y2[0], np.exp(1.0) - 1.0)


def test_vlad7_canonical_values() -> None:
    bench = get_benchmark("Vlad-7")
    # x1=3, x2=3 -> 0 + 2*sin((-1)*(-1)) = 2*sin(1)
    y = bench["target_fn"](np.array([3.0]), np.array([3.0]))
    assert np.isclose(y[0], 2 * np.sin(1.0))
    # x1=4, x2=4 -> (1)(1) + 2*sin(0) = 1
    y2 = bench["target_fn"](np.array([4.0]), np.array([4.0]))
    assert np.isclose(y2[0], 1.0)


def test_r2_canonical_values() -> None:
    bench = get_benchmark("R2")
    # x=0 -> (0-0+1)/(0+1) = 1
    y = bench["target_fn"](np.array([0.0]))
    assert np.isclose(y[0], 1.0)
    # x=1 -> (1-3+1)/(1+1) = -1/2
    y2 = bench["target_fn"](np.array([1.0]))
    assert np.isclose(y2[0], -0.5)


def test_r3_canonical_values() -> None:
    bench = get_benchmark("R3")
    # x=0 -> 0/1 = 0
    y = bench["target_fn"](np.array([0.0]))
    assert np.isclose(y[0], 0.0)
    # x=1 -> 2/5 = 0.4
    y2 = bench["target_fn"](np.array([1.0]))
    assert np.isclose(y2[0], 0.4)


def test_keijzer11_canonical_values() -> None:
    bench = get_benchmark("Keijzer-11")
    # x=1, y=1 -> 1*1 + sin(0) = 1
    y = bench["target_fn"](np.array([1.0]), np.array([1.0]))
    assert np.isclose(y[0], 1.0)
    # x=0, y=0 -> 0 + sin((-1)*(-1)) = sin(1)
    y2 = bench["target_fn"](np.array([0.0]), np.array([0.0]))
    assert np.isclose(y2[0], np.sin(1.0))


def test_liv14_canonical_values() -> None:
    bench = get_benchmark("Liv-14")
    # x1=0, x2=0 -> 0+0+0+sin(0)+sin(0) = 0
    y = bench["target_fn"](np.array([0.0]), np.array([0.0]))
    assert np.isclose(y[0], 0.0, atol=1e-10)
    # x1=1, x2=0 -> 1+1+1+sin(1)+sin(0) = 3+sin(1)
    y2 = bench["target_fn"](np.array([1.0]), np.array([0.0]))
    assert np.isclose(y2[0], 3.0 + np.sin(1.0))


# ----------------------------------------------------------------------
# Sampling protocol shapes & determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_train,expected_test",
    [
        ("I.29.16", 1000, 250),
        ("I.50.26", 1000, 250),
        ("I.16.6", 1000, 250),
        ("II.11.28", 1000, 250),
        ("III.14.14", 1000, 250),
        ("R2", 1000, 250),
        ("R3", 1000, 250),
        ("Keijzer-11", 1000, 250),
        ("Liv-14", 1000, 250),
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


def test_vlad7_uses_published_protocol_300_1200() -> None:
    bench = get_benchmark("Vlad-7")
    x_train, y_train, x_test, y_test = generate_data(
        bench, n_samples=1250, train_ratio=0.8, seed=42
    )
    assert x_train.shape == (300, 2)
    assert x_test.shape == (1200, 2)
    assert y_train.shape == (300,)
    assert y_test.shape == (1200,)


def test_uniform_sampling_is_seed_deterministic() -> None:
    bench = get_benchmark("I.29.16")
    a = generate_data(bench, seed=7)[0]
    b = generate_data(bench, seed=7)[0]
    assert np.array_equal(a, b)
    c = generate_data(bench, seed=8)[0]
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# Domain safety
# ----------------------------------------------------------------------


def test_ii11_28_domain_safe() -> None:
    """II.11.28 denominator (1 - n*alpha/3) > 0 on [0,1]^2."""
    bench = get_benchmark("II.11.28")
    x_train, _, x_test, _ = generate_data(bench, seed=42)
    for x in [x_train, x_test]:
        n, alpha = x[:, 0], x[:, 1]
        denom = 1 - n * alpha / 3
        assert np.all(denom > 0), "II.11.28 denominator <= 0 in domain"


def test_iii14_14_exp_arg_bounded() -> None:
    """III.14.14 exp argument qV/(kbT) stays bounded on [1,5]x[1,2]^4."""
    bench = get_benchmark("III.14.14")
    x_train, _, x_test, _ = generate_data(bench, seed=42)
    for x in [x_train, x_test]:
        q, volt, kb, t = x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        exp_arg = q * volt / (kb * t)
        assert np.all(exp_arg <= 5.0), "III.14.14 exp arg too large"


# ----------------------------------------------------------------------
# Numerical sanity (no NaN / Inf in y for the canonical seed)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STRUCTURAL_BENCHMARKS, ids=lambda b: b["name"])
def test_outputs_finite_for_canonical_seed(bench: dict) -> None:
    _, y_train, _, y_test = generate_data(bench, seed=42)
    assert np.all(np.isfinite(y_train)), f"{bench['name']} train NaN/Inf"
    assert np.all(np.isfinite(y_test)), f"{bench['name']} test NaN/Inf"


# ----------------------------------------------------------------------
# SymPy ground-truth presence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STRUCTURAL_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_expression_evaluates(bench: dict) -> None:
    import sympy

    expr = bench["sympy_expression"]
    syms = bench["sympy_variables"]
    assert isinstance(expr, sympy.Expr)
    assert len(syms) == bench["num_variables"]
    subs = {s: lo + 0.1 for s, (lo, _) in zip(syms, bench["var_ranges"], strict=True)}
    val = float(expr.subs(subs).evalf())
    assert math.isfinite(val), f"{bench['name']} SymPy eval not finite: {val}"
