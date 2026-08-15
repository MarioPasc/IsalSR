"""Unit tests for the Feynman-remainder benchmark definitions (R3.1 draw).

Validates each of the 6 problems:
    - registry shape and key completeness (T1)
    - generate_data shapes and seed determinism (T2)
    - sympy ground truth agrees with target_fn to rtol=1e-10 (T3)
    - outputs finite across 20 seeds (T4)
    - domain guards asserted on sampled data (T5)
    - get_benchmark round-trip and error path (T6)
    - agreement with the committed AI Feynman catalogue (T7)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sympy

from benchmarks.datasets.feynman_catalogue import AIFEYNMAN_120
from benchmarks.datasets.feynman_remainder import (
    FEYNMAN_REMAINDER_BENCHMARKS,
    generate_data,
    get_benchmark,
)

EXPECTED_NAMES = [
    "I.12.2",
    "II.34.29a",
    "II.34.29b",
    "III.19.51",
    "III.4.32",
    "test_4",
]

REQUIRED_KEYS = {
    "name",
    "expression",
    "sympy_expression",
    "sympy_variables",
    "num_variables",
    "var_ranges",
    "target_fn",
    "sampling",
}


# ----------------------------------------------------------------------
# T1 -- registry
# ----------------------------------------------------------------------


def test_registry_has_six_problems() -> None:
    assert len(FEYNMAN_REMAINDER_BENCHMARKS) == 6


def test_registry_names_are_exactly_the_draw() -> None:
    assert [b["name"] for b in FEYNMAN_REMAINDER_BENCHMARKS] == EXPECTED_NAMES


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_required_keys_present(bench: dict[str, Any]) -> None:
    assert set(bench.keys()) >= REQUIRED_KEYS


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_arity_is_consistent(bench: dict[str, Any]) -> None:
    nv = bench["num_variables"]
    assert len(bench["var_ranges"]) == nv
    assert len(bench["sympy_variables"]) == nv
    assert isinstance(bench["sympy_expression"], sympy.Basic)


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_sampling_is_plain_uniform(bench: dict[str, Any]) -> None:
    assert bench["sampling"] == {"type": "uniform"}


# ----------------------------------------------------------------------
# T2 -- generate_data shapes and determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_generate_data_shapes(bench: dict[str, Any]) -> None:
    nv = bench["num_variables"]
    x_train, y_train, x_test, y_test = generate_data(
        bench, n_samples=1250, train_ratio=0.8, seed=42
    )
    assert x_train.shape == (1000, nv)
    assert y_train.shape == (1000,)
    assert x_test.shape == (250, nv)
    assert y_test.shape == (250,)


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_generate_data_is_seed_deterministic(bench: dict[str, Any]) -> None:
    a_x, a_y, _, _ = generate_data(bench, seed=7)
    b_x, b_y, _, _ = generate_data(bench, seed=7)
    assert np.array_equal(a_x, b_x)
    assert np.array_equal(a_y, b_y)


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_different_seeds_give_different_data(bench: dict[str, Any]) -> None:
    a_x = generate_data(bench, seed=7)[0]
    c_x = generate_data(bench, seed=8)[0]
    assert not np.array_equal(a_x, c_x)


# ----------------------------------------------------------------------
# T3 -- sympy ground truth agrees with target_fn (decisive test)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_matches_target_fn(bench: dict[str, Any]) -> None:
    x_train, y_train, _, _ = generate_data(bench, seed=42)
    lam = sympy.lambdify(bench["sympy_variables"], bench["sympy_expression"], "numpy")
    y_sympy = np.asarray(lam(*[x_train[:, i] for i in range(bench["num_variables"])]), dtype=float)
    np.testing.assert_allclose(y_sympy, y_train, rtol=1e-10)


# ----------------------------------------------------------------------
# T4 -- finiteness across 20 seeds
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_outputs_finite_across_seeds(bench: dict[str, Any], seed: int) -> None:
    _, y_train, _, y_test = generate_data(bench, seed=seed)
    assert np.all(np.isfinite(y_train)), f"{bench['name']} train NaN/Inf at seed {seed}"
    assert np.all(np.isfinite(y_test)), f"{bench['name']} test NaN/Inf at seed {seed}"


# ----------------------------------------------------------------------
# T5 -- domain guards on sampled data
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_iii_4_32_denominator_strictly_positive(seed: int) -> None:
    bench = get_benchmark("III.4.32")
    x_train, _, x_test, _ = generate_data(bench, seed=seed)
    for x in (x_train, x_test):
        h, omega, kb, t = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        arg = (h / (2.0 * np.pi)) * omega / (kb * t)
        assert np.all(arg > 0.0)
        assert np.all(np.expm1(arg) > 0.0)


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_test_4_radicand_strictly_positive(seed: int) -> None:
    bench = get_benchmark("test_4")
    x_train, _, x_test, _ = generate_data(bench, seed=seed)
    for x in (x_train, x_test):
        m, e_n, u, ell, r = (x[:, i] for i in range(5))
        radicand = 2.0 / m * (e_n - u - ell**2 / (2.0 * m * r**2))
        assert np.all(radicand > 0.0), f"min radicand {radicand.min()}"


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_i_12_2_denominator_bounded_away_from_zero(seed: int) -> None:
    bench = get_benchmark("I.12.2")
    x_train, _, x_test, _ = generate_data(bench, seed=seed)
    for x in (x_train, x_test):
        epsilon, r = x[:, 2], x[:, 3]
        denom = 4.0 * np.pi * epsilon * r**3
        assert np.all(np.abs(denom) > 1.0)


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_iii_19_51_denominator_bounded_away_from_zero(seed: int) -> None:
    bench = get_benchmark("III.19.51")
    x_train, _, x_test, _ = generate_data(bench, seed=seed)
    for x in (x_train, x_test):
        h, n, epsilon = x[:, 2], x[:, 3], x[:, 4]
        denom = 2.0 * (4.0 * np.pi * epsilon) ** 2 * (h / (2.0 * np.pi)) ** 2 * n**2
        assert np.all(np.abs(denom) > 1e-3)


# ----------------------------------------------------------------------
# T6 -- get_benchmark
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_get_benchmark_round_trip(name: str) -> None:
    assert get_benchmark(name)["name"] == name


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown feynman_remainder benchmark"):
        get_benchmark("DoesNotExist")


# ----------------------------------------------------------------------
# T7 -- cross-check against the committed AI Feynman catalogue
# ----------------------------------------------------------------------

_CATALOGUE = {row["id"]: row for row in AIFEYNMAN_120}


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_name_present_in_catalogue(bench: dict[str, Any]) -> None:
    assert bench["name"] in _CATALOGUE


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_num_variables_matches_catalogue(bench: dict[str, Any]) -> None:
    assert bench["num_variables"] == _CATALOGUE[bench["name"]]["num_variables"]


@pytest.mark.parametrize("bench", FEYNMAN_REMAINDER_BENCHMARKS, ids=lambda b: b["name"])
def test_var_ranges_match_catalogue(bench: dict[str, Any]) -> None:
    catalogue_ranges = [(v["low"], v["high"]) for v in _CATALOGUE[bench["name"]]["variables"]]
    assert list(bench["var_ranges"]) == catalogue_ranges
