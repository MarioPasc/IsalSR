"""Feynman-remainder benchmark definitions: the 6 equations drawn by rule R3.1.

The six problems below are the pre-registered R3.1 extension draw over the AI
Feynman database (Udrescu & Tegmark, *Sci. Adv.* 6(16):eaay2631, 2020), taken
from the Sigma_SR-eligible pool that no existing IsalSR tier already covers.
The draw is recorded in ``docs/md_files/changes/r31_extension_selection_draw.json``;
the problem ``name`` strings here are the canonical AI Feynman ids used there and
in the catalogue ``benchmarks/datasets/feynman_catalogue.py``.

Equations (published variable order, PMLB variable ranges):
    - I.12.2      Coulomb force (4 vars)
    - II.34.29a   Magnetic moment of a spin (3 vars)
    - II.34.29b   Zeeman splitting energy (5 vars)
    - III.19.51   Bohr-model energy levels (5 vars)
    - III.4.32    Bose-Einstein mean occupation number (4 vars)
    - test_4      Radial velocity in a central potential (5 vars, bonus equation)

All six use uniform sampling (1000 train / 250 test) with no per-problem
overrides, matching every other Feynman-derived tier in the suite.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import sympy


def _make_feynman_remainder(
    name: str,
    expression: str,
    sympy_expression: sympy.Expr,
    sympy_variables: list[sympy.Symbol],
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Create a Feynman-remainder benchmark specification dict.

    Args:
        name: Canonical AI Feynman equation id.
        expression: Human-readable formula, in the published form.
        sympy_expression: Ground-truth expression over ``sympy_variables``.
        sympy_variables: Symbols ``x_0 ... x_{n-1}`` in declared variable order.
        num_variables: Number of input variables.
        var_ranges: Per-variable ``(low, high)`` sampling bounds.
        target_fn: Vectorised numpy evaluator, positional args in declared order.
        sampling: Sampling protocol descriptor.

    Returns:
        The benchmark specification dict.
    """
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
# AI Feynman remainder (uniform sampling, 1000 train + 250 test)
# Variable ranges from the PMLB feynman_* metadata, verbatim.
# ----------------------------------------------------------------------

FEYNMAN_REMAINDER_BENCHMARKS: list[dict[str, Any]] = [
    _make_feynman_remainder(
        "I.12.2",
        "q1*q2*r/(4*pi*epsilon*r**3)",
        _x[0] * _x[1] * _x[3] / (4 * sympy.pi * _x[2] * _x[3] ** 3),
        _x[:4],
        4,
        [(1.0, 5.0)] * 4,  # q1, q2, epsilon, r
        lambda q1, q2, epsilon, r: q1 * q2 * r / (4.0 * np.pi * epsilon * r**3),
        # Published form keeps r in the numerator and r**3 in the denominator.
        # epsilon, r >= 1.0 so the denominator is bounded away from zero.
        {"type": "uniform"},
    ),
    _make_feynman_remainder(
        "II.34.29a",
        "q*h/(4*pi*m)",
        _x[0] * _x[1] / (4 * sympy.pi * _x[2]),
        _x[:3],
        3,
        [(1.0, 5.0)] * 3,  # q, h, m
        lambda q, h, m: q * h / (4.0 * np.pi * m),
        # m >= 1.0 so no division-by-zero.
        {"type": "uniform"},
    ),
    _make_feynman_remainder(
        "II.34.29b",
        "g_*mom*B*Jz/(h/(2*pi))",
        _x[0] * _x[3] * _x[4] * _x[2] / (_x[1] / (2 * sympy.pi)),
        _x[:5],
        5,
        [(1.0, 5.0)] * 5,  # g_, h, Jz, mom, B
        lambda g_, h, jz, mom, b: g_ * mom * b * jz / (h / (2.0 * np.pi)),
        # h >= 1.0 so h/(2*pi) > 0.
        {"type": "uniform"},
    ),
    _make_feynman_remainder(
        "III.19.51",
        "-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
        -_x[0]
        * _x[1] ** 4
        / (2 * (4 * sympy.pi * _x[4]) ** 2 * (_x[2] / (2 * sympy.pi)) ** 2)
        * (1 / _x[3] ** 2),
        _x[:5],
        5,
        [(1.0, 5.0)] * 5,  # m, q, h, n, epsilon
        lambda m, q, h, n, epsilon: (
            -m
            * q**4
            / (2.0 * (4.0 * np.pi * epsilon) ** 2 * (h / (2.0 * np.pi)) ** 2)
            * (1.0 / n**2)
        ),
        # epsilon, h, n >= 1.0 so every denominator factor is bounded away from zero.
        {"type": "uniform"},
    ),
    _make_feynman_remainder(
        "III.4.32",
        "1/(exp((h/(2*pi))*omega/(kb*T))-1)",
        1 / (sympy.exp((_x[0] / (2 * sympy.pi)) * _x[1] / (_x[2] * _x[3])) - 1),
        _x[:4],
        4,
        [(1.0, 5.0)] * 4,  # h, omega, kb, T
        lambda h, omega, kb, t: 1.0 / (np.exp((h / (2.0 * np.pi)) * omega / (kb * t)) - 1.0),
        # The exponent is bounded below by (1/(2*pi))*1/(5*5) = 1/(50*pi) > 0,
        # so exp(arg) - 1 > 0 and the reciprocal is finite everywhere.
        {"type": "uniform"},
    ),
    _make_feynman_remainder(
        # Bonus equation: PMLB distributes the 20 bonus equations as
        # feynman_test_1 ... feynman_test_20. Physics: radial velocity of a
        # particle in a central potential.
        "test_4",
        "sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))",
        sympy.sqrt(2 / _x[0] * (_x[1] - _x[2] - _x[3] ** 2 / (2 * _x[0] * _x[4] ** 2))),
        _x[:5],
        5,
        [
            (1.0, 3.0),  # m
            (8.0, 12.0),  # E_n
            (1.0, 3.0),  # U
            (1.0, 3.0),  # L
            (1.0, 3.0),  # r
        ],
        lambda m, e_n, u, ell, r: np.sqrt(2.0 / m * (e_n - u - ell**2 / (2.0 * m * r**2))),
        # Radicand >= 8 - 3 - 9/2 = 0.5 > 0 at the worst corner
        # (E_n=8, U=3, L=3, m=1, r=1), so the sqrt is real everywhere.
        {"type": "uniform"},
    ),
]


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
    """Uniform random sampling with optional per-problem size overrides.

    Args:
        benchmark: Benchmark specification dict.
        n_samples: Total sample budget when no override is present.
        train_ratio: Fraction of ``n_samples`` used for training.
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.
    """
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
    """Generate train/test data for a Feynman-remainder benchmark.

    Same signature as ``roundoff.generate_data`` for orchestrator compatibility.
    All 6 problems use uniform sampling.

    Args:
        benchmark: Benchmark specification dict.
        n_samples: Total sample budget (1250 -> 1000 train / 250 test).
        train_ratio: Fraction of ``n_samples`` used for training.
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If the benchmark declares an unknown sampling type.
    """
    sampling_type = benchmark["sampling"]["type"]

    if sampling_type == "uniform":
        return _sample_uniform(benchmark, n_samples, train_ratio, seed)
    raise ValueError(f"Unknown sampling type: {sampling_type}")


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a Feynman-remainder benchmark by name.

    Args:
        name: Canonical AI Feynman equation id.

    Returns:
        The matching benchmark specification dict.

    Raises:
        ValueError: If no benchmark carries that name.
    """
    for b in FEYNMAN_REMAINDER_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown feynman_remainder benchmark: {name}. "
        f"Available: {[b['name'] for b in FEYNMAN_REMAINDER_BENCHMARKS]}"
    )
