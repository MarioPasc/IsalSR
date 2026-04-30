"""Roundoff benchmark definitions: 8 problems to bring the total from 42 to 50.

All problems satisfy the structural-bottleneck screening criterion
(n_nontrivial_constants = 0, k >= 5) from the bottleneck-type analysis
(``docs/md_files/changes/bottleneck_type_analysis.md``, 2026-04-19).

AI Feynman (Udrescu & Tegmark, 2020):
    - III.10.19   Magnetic moment × L2-norm (k=7, 4 vars)
    - II.11.3     Driven oscillator resonance (k=6, 5 vars)
    - I.13.12     Gravitational PE difference (k=6, 5 vars)
    - I.44.4      Isothermal work with log (k=5, 5 vars)

GP-classic / DSO benchmarks:
    - R1          (DSO/Koza) — rational cubic (k=7, 1 var)
    - Pagie-2     (Pagie & Hogeweg, 1997) — 3D Pagie (k=10, 3 vars)
    - Liv-4       (Mundhenk et al., 2021) — three-log sum (k=8, 1 var)
    - Liv-19      (Mundhenk et al., 2021) — log-of-polynomial (k=9, 1 var)

All problems use uniform sampling (1000 train / 250 test) with no overrides.
The operator set matches ``experiments/configs/bingo_hard.yaml`` (sqrt, pow
included) — no new operators are required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import sympy


def _make_roundoff(
    name: str,
    expression: str,
    sympy_expression: sympy.Expr,
    sympy_variables: list[sympy.Symbol],
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Create a roundoff-benchmark specification dict."""
    return {
        "name": name,
        "expression": expression,
        "sympy_expression": sympy_expression,
        "sympy_variables": sympy_variables,
        "num_variables": num_variables,
        "var_ranges": var_ranges,
        "target_fn": target_fn,
        "sampling": sampling,
    }


_x = [sympy.Symbol(f"x_{i}") for i in range(6)]


# ----------------------------------------------------------------------
# AI Feynman (uniform sampling, 1000 train + 250 test)
# Variable ranges from the canonical Feynman CSV (Udrescu & Tegmark, 2020).
# ----------------------------------------------------------------------

_FEYNMAN_ROUNDOFF: list[dict[str, Any]] = [
    _make_roundoff(
        "III.10.19",
        "mom * sqrt(Bx^2 + By^2 + Bz^2)",
        _x[0] * sympy.sqrt(_x[1] ** 2 + _x[2] ** 2 + _x[3] ** 2),
        _x[:4],
        4,
        [(1.0, 5.0)] * 4,  # mom, Bx, By, Bz
        lambda mom, bx, by, bz: mom * np.sqrt(bx**2 + by**2 + bz**2),
        {"type": "uniform"},
    ),
    _make_roundoff(
        "II.11.3",
        "q*Ef / (m*(omega_0^2 - omega^2))",
        _x[0] * _x[1] / (_x[2] * (_x[3] ** 2 - _x[4] ** 2)),
        _x[:5],
        5,
        [
            (1.0, 5.0),  # q
            (1.0, 5.0),  # Ef
            (1.0, 3.0),  # m
            (3.0, 5.0),  # omega_0
            (1.0, 2.0),  # omega
        ],
        lambda q, ef, m, w0, w: q * ef / (m * (w0**2 - w**2)),
        # omega_0^2 - omega^2 >= 9 - 4 = 5 > 0 always.
        {"type": "uniform"},
    ),
    _make_roundoff(
        "I.13.12",
        "G*m1*m2*(1/r2 - 1/r1)",
        _x[4] * _x[0] * _x[1] * (1 / _x[3] - 1 / _x[2]),
        _x[:5],
        5,
        [(1.0, 5.0)] * 5,  # m1, m2, r1, r2, G
        lambda m1, m2, r1, r2, g: g * m1 * m2 * (1.0 / r2 - 1.0 / r1),
        # r1, r2 >= 1.0 so no division-by-zero.
        {"type": "uniform"},
    ),
    _make_roundoff(
        "I.44.4",
        "n*kb*T*ln(V2/V1)",
        _x[0] * _x[1] * _x[2] * sympy.log(_x[4] / _x[3]),
        _x[:5],
        5,
        [(1.0, 5.0)] * 5,  # n, kb, T, V1, V2
        lambda n, kb, t, v1, v2: n * kb * t * np.log(v2 / v1),
        # V1, V2 >= 1.0 so V2/V1 > 0.
        {"type": "uniform"},
    ),
]


# ----------------------------------------------------------------------
# GP-classic / DSO benchmarks (uniform sampling, 1000 train + 250 test)
# ----------------------------------------------------------------------

_GP_ROUNDOFF: list[dict[str, Any]] = [
    _make_roundoff(
        "R1",
        "(x+1)^3 / (x^2 - x + 1)",
        (_x[0] + 1) ** 3 / (_x[0] ** 2 - _x[0] + 1),
        _x[:1],
        1,
        [(-1.0, 1.0)],  # DSO/Koza canonical domain
        lambda x: (x + 1) ** 3 / (x**2 - x + 1),
        # Denominator = (x - 1/2)^2 + 3/4 >= 3/4 > 0 always.
        {"type": "uniform"},
    ),
    _make_roundoff(
        "Pagie-2",
        "1/(1+x^(-4)) + 1/(1+y^(-4)) + 1/(1+z^(-4))",
        _x[0] ** 4 / (_x[0] ** 4 + 1)
        + _x[1] ** 4 / (_x[1] ** 4 + 1)
        + _x[2] ** 4 / (_x[2] ** 4 + 1),
        _x[:3],
        3,
        [(-5.0, 5.0)] * 3,  # Matches Pagie-1 domain
        lambda x, y, z: x**4 / (x**4 + 1) + y**4 / (y**4 + 1) + z**4 / (z**4 + 1),
        # Numerically stable form: x^4/(x^4+1) avoids x^(-4) overflow at x=0.
        {"type": "uniform"},
    ),
    _make_roundoff(
        "Liv-4",
        "ln(x+1) + ln(x^2+1) + ln(x)",
        sympy.log(_x[0] + 1) + sympy.log(_x[0] ** 2 + 1) + sympy.log(_x[0]),
        _x[:1],
        1,
        [(0.1, 10.0)],  # x > 0 required for ln(x); x > -1 for ln(x+1).
        lambda x: np.log(x + 1) + np.log(x**2 + 1) + np.log(x),
        {"type": "uniform"},
    ),
    _make_roundoff(
        "Liv-19",
        "ln(x^2+x) + ln(x^3+x)",
        sympy.log(_x[0] ** 2 + _x[0]) + sympy.log(_x[0] ** 3 + _x[0]),
        _x[:1],
        1,
        [(0.1, 10.0)],  # x > 0 ensures x^2+x > 0 and x^3+x > 0.
        lambda x: np.log(x**2 + x) + np.log(x**3 + x),
        {"type": "uniform"},
    ),
]


ROUNDOFF_BENCHMARKS: list[dict[str, Any]] = _FEYNMAN_ROUNDOFF + _GP_ROUNDOFF


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------


def _sample_uniform(
    benchmark: dict[str, Any],
    n_samples: int,
    train_ratio: float,
    seed: int,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Uniform random sampling with optional per-problem size overrides."""
    sampling = benchmark["sampling"]
    n_train_override = sampling.get("n_train_override")
    n_test_override = sampling.get("n_test_override")
    if n_train_override is not None and n_test_override is not None:
        n_train = int(n_train_override)
        n_test = int(n_test_override)
    else:
        n_train = int(n_samples * train_ratio)
        n_test = n_samples - n_train

    rng = np.random.default_rng(seed)
    nv = benchmark["num_variables"]
    var_ranges = benchmark["var_ranges"]
    fn = benchmark["target_fn"]

    x_train = np.column_stack([rng.uniform(lo, hi, n_train) for lo, hi in var_ranges])
    x_test = np.column_stack([rng.uniform(lo, hi, n_test) for lo, hi in var_ranges])

    args_train = [x_train[:, i] for i in range(nv)]
    args_test = [x_test[:, i] for i in range(nv)]
    y_train = fn(*args_train)
    y_test = fn(*args_test)

    return x_train, y_train, x_test, y_test


def generate_data(
    benchmark: dict[str, Any],
    n_samples: int = 1250,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Generate train/test data for a roundoff benchmark.

    Same signature as ``cherrypicked.generate_data`` for orchestrator
    compatibility. All 8 problems use uniform sampling.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    sampling_type = benchmark["sampling"]["type"]

    if sampling_type == "uniform":
        return _sample_uniform(benchmark, n_samples, train_ratio, seed)
    raise ValueError(f"Unknown sampling type: {sampling_type}")


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a roundoff benchmark by name."""
    for b in ROUNDOFF_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown roundoff benchmark: {name}. Available: {[b['name'] for b in ROUNDOFF_BENCHMARKS]}"
    )
