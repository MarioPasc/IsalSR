"""Unit tests for the ODE-Strogatz benchmark definitions.

Validates each of the 14 problems:
    - registry shape and required keys
    - the vendored PMLB files load as (400, 3) with no NaN/Inf
    - **transcription**: target_fn on the published (x, y) columns reproduces
      the published target column (the decisive test)
    - the SymPy ground truth agrees with target_fn on the same columns, which
      is what makes ``solution_recovered`` trustworthy
    - the 300/100 split is exhaustive, disjoint and seed-deterministic
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sympy

from benchmarks.datasets.strogatz import (
    STROGATZ_BENCHMARKS,
    data_key,
    data_path,
    generate_data,
    get_benchmark,
    load_published,
)

_REQUIRED_KEYS = {
    "name",
    "expression",
    "sympy_expression",
    "sympy_variables",
    "num_variables",
    "var_ranges",
    "target_fn",
    "sampling",
}

_EXPECTED_NAMES = [
    "Strogatz-bacres1",
    "Strogatz-bacres2",
    "Strogatz-barmag1",
    "Strogatz-barmag2",
    "Strogatz-glider1",
    "Strogatz-glider2",
    "Strogatz-lv1",
    "Strogatz-lv2",
    "Strogatz-predprey1",
    "Strogatz-predprey2",
    "Strogatz-shearflow1",
    "Strogatz-shearflow2",
    "Strogatz-vdp1",
    "Strogatz-vdp2",
]

# Transcription tolerance. The published targets are stored as ~15-digit
# decimals, so agreement is expected at double-precision round-off level;
# these bounds are three orders of magnitude looser than that. They are NOT
# to be relaxed: a failure means the equation or the column mapping is wrong.
_RTOL = 1e-6
_ATOL = 1e-8


def _published_columns(
    bench: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, target)`` of the vendored file backing ``bench``."""
    features, target = load_published(data_key(bench["name"]))
    return features[:, 0], features[:, 1], target


# ----------------------------------------------------------------------
# T1 -- registry sanity
# ----------------------------------------------------------------------


def test_registry_has_fourteen_problems() -> None:
    assert len(STROGATZ_BENCHMARKS) == 14


def test_registry_names_match_expected() -> None:
    assert [b["name"] for b in STROGATZ_BENCHMARKS] == _EXPECTED_NAMES


def test_registry_names_are_unique() -> None:
    names = [b["name"] for b in STROGATZ_BENCHMARKS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_each_dict_has_required_keys(bench: dict) -> None:
    assert set(bench.keys()) >= _REQUIRED_KEYS, f"{bench['name']} missing keys"


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_num_variables_is_two(bench: dict) -> None:
    assert bench["num_variables"] == 2
    assert len(bench["sympy_variables"]) == 2
    assert [str(v) for v in bench["sympy_variables"]] == ["x_0", "x_1"]
    assert len(bench["var_ranges"]) == 2


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_sampling_protocol_is_published_fixed(bench: dict) -> None:
    sampling = bench["sampling"]
    assert sampling["type"] == "published_fixed"
    assert sampling["n_train_override"] == 300
    assert sampling["n_test_override"] == 100


# ----------------------------------------------------------------------
# T2 -- vendored data integrity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_vendored_file_exists(bench: dict) -> None:
    path = data_path(data_key(bench["name"]))
    assert path.is_file(), f"missing vendored file {path}"
    assert path.is_absolute() or not str(path).startswith("."), "path must not be CWD-relative"


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_vendored_file_shape_and_finiteness(bench: dict) -> None:
    features, target = load_published(data_key(bench["name"]))
    assert features.shape == (400, 2)
    assert target.shape == (400,)
    assert np.all(np.isfinite(features)), f"{bench['name']} non-finite features"
    assert np.all(np.isfinite(target)), f"{bench['name']} non-finite target"


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_var_ranges_match_published_columns(bench: dict) -> None:
    features, _ = load_published(data_key(bench["name"]))
    for i, (lo, hi) in enumerate(bench["var_ranges"]):
        assert lo == pytest.approx(float(features[:, i].min()))
        assert hi == pytest.approx(float(features[:, i].max()))
        assert lo < hi


# ----------------------------------------------------------------------
# T3 -- transcription (decisive)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_target_fn_reproduces_published_target(bench: dict) -> None:
    x, y, target = _published_columns(bench)
    predicted = np.asarray(bench["target_fn"](x, y), dtype=np.float64)
    predicted = np.broadcast_to(predicted, target.shape)
    np.testing.assert_allclose(
        predicted,
        target,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{bench['name']} transcription mismatch",
    )


# ----------------------------------------------------------------------
# T4 -- SymPy ground truth agrees with target_fn
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_expression_present(bench: dict) -> None:
    assert isinstance(bench["sympy_expression"], sympy.Basic)
    free = bench["sympy_expression"].free_symbols
    assert free <= set(bench["sympy_variables"])


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_matches_published_target(bench: dict) -> None:
    x, y, target = _published_columns(bench)
    fn = sympy.lambdify(bench["sympy_variables"], bench["sympy_expression"], "numpy")
    predicted = np.asarray(fn(x, y), dtype=np.float64)
    predicted = np.broadcast_to(predicted, target.shape)
    np.testing.assert_allclose(
        predicted,
        target,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{bench['name']} sympy/target mismatch",
    )


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_sympy_matches_target_fn(bench: dict) -> None:
    x, y, _ = _published_columns(bench)
    fn = sympy.lambdify(bench["sympy_variables"], bench["sympy_expression"], "numpy")
    expected = np.broadcast_to(np.asarray(bench["target_fn"](x, y), dtype=np.float64), x.shape)
    predicted = np.broadcast_to(np.asarray(fn(x, y), dtype=np.float64), x.shape)
    np.testing.assert_allclose(
        predicted, expected, rtol=_RTOL, atol=_ATOL, err_msg=f"{bench['name']}"
    )


# ----------------------------------------------------------------------
# T5 -- split protocol
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_split_shapes(bench: dict) -> None:
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    assert x_train.shape == (300, 2)
    assert y_train.shape == (300,)
    assert x_test.shape == (100, 2)
    assert y_test.shape == (100,)


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_split_ignores_n_samples_and_train_ratio(bench: dict) -> None:
    a = generate_data(bench, n_samples=400, train_ratio=0.75, seed=3)
    b = generate_data(bench, n_samples=12345, train_ratio=0.1, seed=3)
    for lhs, rhs in zip(a, b, strict=True):
        assert np.array_equal(lhs, rhs)


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_split_is_a_partition_of_the_published_rows(bench: dict) -> None:
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    features, target = load_published(data_key(bench["name"]))

    rows = np.column_stack
    train_rows = {tuple(r) for r in rows((x_train, y_train)).tolist()}
    test_rows = {tuple(r) for r in rows((x_test, y_test)).tolist()}
    all_rows = {tuple(r) for r in rows((features, target)).tolist()}

    assert len(train_rows) == 300
    assert len(test_rows) == 100
    assert train_rows.isdisjoint(test_rows)
    assert train_rows | test_rows == all_rows


def test_split_differs_across_seeds() -> None:
    bench = get_benchmark("Strogatz-vdp1")
    a = generate_data(bench, seed=7)[0]
    b = generate_data(bench, seed=8)[0]
    assert not np.array_equal(a, b)


def test_split_is_seed_deterministic() -> None:
    bench = get_benchmark("Strogatz-vdp1")
    a = generate_data(bench, seed=7)
    b = generate_data(bench, seed=7)
    for lhs, rhs in zip(a, b, strict=True):
        assert np.array_equal(lhs, rhs)


def test_unknown_sampling_type_raises() -> None:
    bench = dict(get_benchmark("Strogatz-vdp1"))
    bench["sampling"] = {"type": "uniform"}
    with pytest.raises(ValueError, match="Unknown sampling type"):
        generate_data(bench)


# ----------------------------------------------------------------------
# T6 -- returned arrays are finite
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
@pytest.mark.parametrize("seed", [0, 42])
def test_outputs_finite(bench: dict, seed: int) -> None:
    for arr in generate_data(bench, seed=seed):
        assert np.all(np.isfinite(arr)), f"{bench['name']} non-finite at seed {seed}"


# ----------------------------------------------------------------------
# T7 -- lookup
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bench", STROGATZ_BENCHMARKS, ids=lambda b: b["name"])
def test_get_benchmark_round_trips(bench: dict) -> None:
    assert get_benchmark(bench["name"]) is bench


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strogatz benchmark"):
        get_benchmark("DoesNotExist")


def test_data_key_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strogatz benchmark"):
        data_key("Strogatz-nope")
