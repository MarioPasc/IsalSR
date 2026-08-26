"""Structural benchmark definitions: 10 structural-bottleneck problems.

Ten problems drawn from published SR benchmarks that satisfy the
structural-bottleneck screening criterion (n_nontrivial_constants = 0, k >= 5)
from the bottleneck-type analysis
(``docs/md_files/changes/bottleneck_type_analysis.md``, 2026-04-19). The
criterion is a property of the target expression alone and is applied without
reference to any solver's measured performance.

Screening procedure: ``docs/md_files/changes/candidate_problem_screening.md``

AI Feynman (Udrescu & Tegmark, 2020):
    - I.29.16     Law of cosines (k=11, 4 vars)
    - I.50.26     Nonlinear oscillation (k=8, 4 vars)
    - I.16.6      Relativistic velocity addition (k=6, 3 vars)
    - II.11.28    Clausius-Mossotti (k=6, 2 vars)
    - III.14.14   Shockley diode equation (k=6, 5 vars)

GP-classic / DSO benchmarks:
    - Vlad-7      (Vladislavleva et al., 2009) — product + sin (k=9, 2 vars)
    - R2          (DSO/Koza) — rational quintic (k=9, 1 var)
    - R3          (DSO/Koza) — rational sextic (k=11, 1 var)
    - Keijzer-11  (Keijzer, 2003) — bivariate trig (k=6, 2 vars)
    - Liv-14      (Mundhenk et al., 2021) — poly+trig hybrid (k=8, 2 vars)

All problems use uniform sampling. Vlad-7 uses the published Vladislavleva
protocol (300 train / 1200 test); all others use the standard 1000 / 250.
The operator set matches ``experiments/configs/bingo_hard.yaml`` — no new
operators are required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import sympy


def _make_structural(
    name: str,
    expression: str,
    sympy_expression: sympy.Expr,
    sympy_variables: list[sympy.Symbol],
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Create a structural-benchmark specification dict."""
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


_x = [sympy.Symbol(f"x_{i}") for i in range(5)]


# ----------------------------------------------------------------------
# AI Feynman (uniform sampling, 1000 train + 250 test)
# Variable ranges from the canonical Feynman CSV (Udrescu & Tegmark, 2020).
# ----------------------------------------------------------------------

_FEYNMAN_STRUCTURAL: list[dict[str, Any]] = [
    _make_structural(
        "I.29.16",
        "sqrt(x1^2 + x2^2 - 2*x1*x2*cos(theta1 - theta2))",
        sympy.sqrt(_x[0] ** 2 + _x[1] ** 2 - 2 * _x[0] * _x[1] * sympy.cos(_x[2] - _x[3])),
        _x[:4],
        4,
        [(1.0, 5.0)] * 4,  # x1, x2, theta1, theta2
        lambda x1, x2, t1, t2: np.sqrt(x1**2 + x2**2 - 2 * x1 * x2 * np.cos(t1 - t2)),
        # Argument of sqrt = (x1 - x2*cos(d))^2 + (x2*sin(d))^2 >= 0 always.
        {"type": "uniform"},
    ),
    _make_structural(
        "I.50.26",
        "x1*(cos(omega*t) + alpha*cos(omega*t)^2)",
        _x[0] * (sympy.cos(_x[1] * _x[2]) + _x[3] * sympy.cos(_x[1] * _x[2]) ** 2),
        _x[:4],
        4,
        [(1.0, 3.0)] * 4,  # x1, omega, t, alpha
        lambda x1, omega, t, alpha: x1 * (np.cos(omega * t) + alpha * np.cos(omega * t) ** 2),
        {"type": "uniform"},
    ),
    _make_structural(
        "I.16.6",
        "(u + v) / (1 + u*v/c^2)",
        (_x[2] + _x[1]) / (1 + _x[2] * _x[1] / _x[0] ** 2),
        _x[:3],
        3,
        [(1.0, 5.0)] * 3,  # Feynman CSV order: c, v, u
        lambda c, v, u: (u + v) / (1 + u * v / c**2),
        # Denominator = 1 + uv/c^2 > 1 for positive inputs.
        {"type": "uniform"},
    ),
    _make_structural(
        "II.11.28",
        "1 + n*alpha / (1 - n*alpha/3)",
        1 + _x[0] * _x[1] / (1 - _x[0] * _x[1] / 3),
        _x[:2],
        2,
        [(0.0, 1.0), (0.0, 1.0)],  # n, alpha
        lambda n, alpha: 1 + n * alpha / (1 - n * alpha / 3),
        # Singularity at n*alpha=3; domain [0,1]^2 keeps n*alpha <= 1 < 3.
        {"type": "uniform"},
    ),
    _make_structural(
        "III.14.14",
        "I_0 * (exp(q*Volt/(kb*T)) - 1)",
        _x[0] * (sympy.exp(_x[1] * _x[2] / (_x[3] * _x[4])) - 1),
        _x[:5],
        5,
        [
            (1.0, 5.0),  # I_0
            (1.0, 2.0),  # q
            (1.0, 2.0),  # Volt
            (1.0, 2.0),  # kb
            (1.0, 2.0),  # T
        ],
        lambda i0, q, volt, kb, t: i0 * (np.exp(q * volt / (kb * t)) - 1),
        # Max exp argument = 2*2/(1*1) = 4.0 -> exp(4) ~ 54.6.
        {"type": "uniform"},
    ),
]


# ----------------------------------------------------------------------
# GP-classic / DSO benchmarks (per-problem sampling protocols)
# ----------------------------------------------------------------------

_GP_STRUCTURAL: list[dict[str, Any]] = [
    _make_structural(
        "Vlad-7",
        "(x1-3)*(x2-3) + 2*sin((x1-4)*(x2-4))",
        (_x[0] - 3) * (_x[1] - 3) + 2 * sympy.sin((_x[0] - 4) * (_x[1] - 4)),
        _x[:2],
        2,
        [(0.05, 6.05), (0.05, 6.05)],  # Vladislavleva (2009) canonical domain
        lambda x1, x2: (x1 - 3) * (x2 - 3) + 2 * np.sin((x1 - 4) * (x2 - 4)),
        # Published Vladislavleva protocol: 300 random train, 1200 random test.
        {"type": "uniform", "n_train_override": 300, "n_test_override": 1200},
    ),
    _make_structural(
        "R2",
        "(x^5 - 3*x^3 + 1) / (x^2 + 1)",
        (_x[0] ** 5 - 3 * _x[0] ** 3 + 1) / (_x[0] ** 2 + 1),
        _x[:1],
        1,
        [(-1.0, 1.0)],
        lambda x: (x**5 - 3 * x**3 + 1) / (x**2 + 1),
        # Denominator x^2 + 1 >= 1 always.
        {"type": "uniform"},
    ),
    _make_structural(
        "R3",
        "(x^6 + x^5) / (x^4 + x^3 + x^2 + x + 1)",
        (_x[0] ** 6 + _x[0] ** 5) / (_x[0] ** 4 + _x[0] ** 3 + _x[0] ** 2 + _x[0] + 1),
        _x[:1],
        1,
        [(-1.0, 1.0)],
        lambda x: (x**6 + x**5) / (x**4 + x**3 + x**2 + x + 1),
        # Denominator = (x^5 - 1)/(x - 1) for x != 1; min ~ 0.674 on [-1,1].
        {"type": "uniform"},
    ),
    _make_structural(
        "Keijzer-11",
        "x*y + sin((x-1)*(y-1))",
        _x[0] * _x[1] + sympy.sin((_x[0] - 1) * (_x[1] - 1)),
        _x[:2],
        2,
        [(-3.0, 3.0), (-3.0, 3.0)],  # McDermott et al. (2012) domain
        lambda x, y: x * y + np.sin((x - 1) * (y - 1)),
        {"type": "uniform"},
    ),
    _make_structural(
        "Liv-14",
        "x1^3 + x1^2 + x1 + sin(x1) + sin(x2^2)",
        _x[0] ** 3 + _x[0] ** 2 + _x[0] + sympy.sin(_x[0]) + sympy.sin(_x[1] ** 2),
        _x[:2],
        2,
        [(-1.0, 1.0), (-1.0, 1.0)],
        lambda x1, x2: x1**3 + x1**2 + x1 + np.sin(x1) + np.sin(x2**2),
        {"type": "uniform"},
    ),
]


STRUCTURAL_BENCHMARKS: list[dict[str, Any]] = _FEYNMAN_STRUCTURAL + _GP_STRUCTURAL


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
    """Generate train/test data for a structural benchmark.

    Same signature as ``hard.generate_data`` for orchestrator compatibility.
    All 10 problems use uniform sampling; the dispatch is included for
    forward-compatibility.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    sampling_type = benchmark["sampling"]["type"]

    if sampling_type == "uniform":
        return _sample_uniform(benchmark, n_samples, train_ratio, seed)
    raise ValueError(f"Unknown sampling type: {sampling_type}")


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a structural benchmark by name."""
    for b in STRUCTURAL_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown structural benchmark: {name}. "
        f"Available: {[b['name'] for b in STRUCTURAL_BENCHMARKS]}"
    )
