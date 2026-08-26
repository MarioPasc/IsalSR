"""Unit tests for roundoff benchmark definitions.

Validates each of the 8 roundoff problems:
    - target_fn evaluates correctly at canonical points
    - generate_data returns expected shapes (1000 train / 250 test)
    - Outputs are finite (no NaN, no Inf) for the canonical seed
    - SymPy ground-truth expressions evaluate finitely
    - Domain safety (no singularities in the sampled domain)
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from benchmarks.datasets.roundoff import (
    ROUNDOFF_BENCHMARKS,
    generate_data,
    get_benchmark,
)

# ----------------------------------------------------------------------
# Registry sanity
# ----------------------------------------------------------------------


def test_registry_has_eight_problems() -> None:
    assert len(ROUNDOFF_BENCHMARKS) == 8


def test_registry_names_are_unique() -> None:
    names = [b["name"] for b in ROUNDOFF_BENCHMARKS]
    assert len(names) == len(set(names))


def test_get_benchmark_lookup() -> None:
    for b in ROUNDOFF_BENCHMARKS:
        assert get_benchmark(b["name"]) is b


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown roundoff benchmark"):
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
    for b in ROUNDOFF_BENCHMARKS:
        assert required <= set(b.keys()), f"{b['name']} missing keys"


# ----------------------------------------------------------------------
# Target function correctness at canonical points
# ----------------------------------------------------------------------


def test_r1_target_fn() -> None:
    fn = get_benchmark("R1")["target_fn"]
    x = np.array([0.0, 1.0, -1.0, 0.5])
    expected = (x + 1) ** 3 / (x**2 - x + 1)
    np.testing.assert_allclose(fn(x), expected, rtol=1e-12)


def test_iii_10_19_target_fn() -> None:
    fn = get_benchmark("III.10.19")["target_fn"]
    result = fn(np.array([2.0]), np.array([1.0]), np.array([2.0]), np.array([3.0]))
    expected = 2.0 * np.sqrt(1.0 + 4.0 + 9.0)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_pagie2_target_fn() -> None:
    fn = get_benchmark("Pagie-2")["target_fn"]
    x = np.array([1.0])
    y = np.array([2.0])
    z = np.array([3.0])
    expected = 1.0**4 / (1.0**4 + 1) + 2.0**4 / (2.0**4 + 1) + 3.0**4 / (3.0**4 + 1)
    np.testing.assert_allclose(fn(x, y, z), expected, rtol=1e-12)


def test_pagie2_at_zero() -> None:
    fn = get_benchmark("Pagie-2")["target_fn"]
    result = fn(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_liv4_target_fn() -> None:
    fn = get_benchmark("Liv-4")["target_fn"]
    x = np.array([1.0, 2.0])
    expected = np.log(x + 1) + np.log(x**2 + 1) + np.log(x)
    np.testing.assert_allclose(fn(x), expected, rtol=1e-12)


def test_liv19_target_fn() -> None:
    fn = get_benchmark("Liv-19")["target_fn"]
    x = np.array([1.0, 2.0, 5.0])
    expected = np.log(x**2 + x) + np.log(x**3 + x)
    np.testing.assert_allclose(fn(x), expected, rtol=1e-12)


def test_ii_11_3_target_fn() -> None:
    fn = get_benchmark("II.11.3")["target_fn"]
    q, ef, m, w0, w = 2.0, 3.0, 1.0, 4.0, 1.0
    expected = q * ef / (m * (w0**2 - w**2))
    result = fn(
        np.array([q]),
        np.array([ef]),
        np.array([m]),
        np.array([w0]),
        np.array([w]),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_ii_11_3_no_singularity() -> None:
    bench = get_benchmark("II.11.3")
    ranges = bench["var_ranges"]
    w0_min, w_max = ranges[3][0], ranges[4][1]
    assert w0_min**2 - w_max**2 > 0, "Domain must avoid resonance singularity"


def test_i_13_12_target_fn() -> None:
    fn = get_benchmark("I.13.12")["target_fn"]
    m1, m2, r1, r2, g = 2.0, 3.0, 1.0, 4.0, 5.0
    expected = g * m1 * m2 * (1.0 / r2 - 1.0 / r1)
    result = fn(
        np.array([m1]),
        np.array([m2]),
        np.array([r1]),
        np.array([r2]),
        np.array([g]),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_i_44_4_target_fn() -> None:
    fn = get_benchmark("I.44.4")["target_fn"]
    n, kb, t, v1, v2 = 2.0, 3.0, 4.0, 1.0, 5.0
    expected = n * kb * t * np.log(v2 / v1)
    result = fn(
        np.array([n]),
        np.array([kb]),
        np.array([t]),
        np.array([v1]),
        np.array([v2]),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


# ----------------------------------------------------------------------
# Sampling protocol shapes & determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", ROUNDOFF_BENCHMARKS, ids=lambda b: b["name"])
def test_sampling_sizes(bench: dict) -> None:
    x_train, y_train, x_test, y_test = generate_data(
        bench, n_samples=1250, train_ratio=0.8, seed=42
    )
    assert x_train.shape[0] == 1000, f"{bench['name']} train size"
    assert x_test.shape[0] == 250, f"{bench['name']} test size"
    assert y_train.shape[0] == 1000
    assert y_test.shape[0] == 250
    assert x_train.shape[1] == bench["num_variables"]
    assert x_test.shape[1] == bench["num_variables"]


def test_uniform_sampling_is_seed_deterministic() -> None:
    bench = get_benchmark("R1")
    a = generate_data(bench, seed=7)[0]
    b = generate_data(bench, seed=7)[0]
    assert np.array_equal(a, b)
    c = generate_data(bench, seed=8)[0]
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# Numerical sanity (no NaN / Inf in y for the canonical seed)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", ROUNDOFF_BENCHMARKS, ids=lambda b: b["name"])
def test_outputs_finite_for_canonical_seed(bench: dict) -> None:
    _, y_train, _, y_test = generate_data(bench, seed=42)
    assert np.all(np.isfinite(y_train)), f"{bench['name']} train NaN/Inf"
    assert np.all(np.isfinite(y_test)), f"{bench['name']} test NaN/Inf"


# ----------------------------------------------------------------------
# SymPy ground-truth presence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", ROUNDOFF_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_expression_present(bench: dict) -> None:
    assert "sympy_expression" in bench
    assert "sympy_variables" in bench
    assert isinstance(bench["sympy_expression"], sympy.Basic)
    assert len(bench["sympy_variables"]) == bench["num_variables"]


@pytest.mark.parametrize("bench", ROUNDOFF_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_evaluates_finitely(bench: dict) -> None:
    expr = bench["sympy_expression"]
    variables = bench["sympy_variables"]
    ranges = bench["var_ranges"]
    midpoints = {v: float((lo + hi) / 2) for v, (lo, hi) in zip(variables, ranges, strict=True)}
    result = float(expr.subs(midpoints))
    assert np.isfinite(result), f"{bench['name']} sympy eval not finite"
