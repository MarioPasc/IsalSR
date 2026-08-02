"""Feynman physics equations benchmark definitions.

Selected equations from Liu2025 Table 2 (GraphDSR, Neural Networks 187:107405).
Original source: Udrescu & Tegmark (2020). AI Feynman. Science Advances 6(16).

Data configuration follows Liu2025 Section 4.1:
    - Train/test split: 80/20
    - Seed: 42 for reproducibility

Every problem carries an explicit ``sympy_expression``. Before 2026-08-02 the
tier relied on the orchestrator's string-parse fallback, which only handles one-
and two-variable targets, so ``solution_recovered`` was silently uncomputable for
five of these ten (I.14.3, I.12.4, II.3.24, I.10.7, I.48.20). Stage C's criterion
C1.5 requires it on 70/70 problems.

Three definitions were corrected on 2026-08-02 to match the AI Feynman database
(I.39.10, I.12.4, II.3.24); see
``docs/md_files/changes/feynman_definition_corrections.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy


def _make_feynman(
    feynman_id: str,
    expression: str,
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
    sympy_expression: sympy.Expr,
) -> dict[str, Any]:
    """Create a Feynman benchmark specification dict.

    Args:
        feynman_id: Canonical AI Feynman equation id.
        expression: Human-readable target expression.
        num_variables: Input dimensionality.
        var_ranges: Sampling range per variable, in column order.
        target_fn: Vectorised target, taking one array per variable in column order.
        sympy_expression: Ground truth over ``x_0 … x_{n-1}``, in the same column
            order, used by ``solution_recovered``.

    Returns:
        The benchmark specification dict.
    """
    return {
        "name": feynman_id,
        "expression": expression,
        "num_variables": num_variables,
        "var_ranges": var_ranges,
        "target_fn": target_fn,
        "sympy_expression": sympy_expression,
        "sympy_variables": [sympy.Symbol(f"x_{i}") for i in range(num_variables)],
    }


_x = [sympy.Symbol(f"x_{i}") for i in range(3)]


# ======================================================================
# Selected Feynman equations from Liu2025 Table 2 (verified against PDF page 7)
# ======================================================================

FEYNMAN_BENCHMARKS: list[dict[str, Any]] = [
    _make_feynman(
        "I.6.20a",
        "exp(-theta^2/2) / sqrt(2*pi)",
        1,
        [(1.0, 3.0)],
        lambda theta: np.exp(-(theta**2) / 2) / np.sqrt(2 * math.pi),
        sympy.exp(-(_x[0] ** 2) / 2) / sympy.sqrt(2 * sympy.pi),
    ),
    _make_feynman(
        "I.12.1",
        "mu * N_s",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda mu, n_s: mu * n_s,
        _x[0] * _x[1],
    ),
    _make_feynman(
        "I.14.3",
        "m * g * z",
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        lambda m, g, z: m * g * z,
        _x[0] * _x[1] * _x[2],
    ),
    _make_feynman(
        "I.25.13",
        "q / C",
        2,
        [(1.0, 3.0), (1.0, 3.0)],
        lambda q, c: q / c,
        _x[0] / _x[1],
    ),
    _make_feynman(
        "I.34.27",
        "hbar * omega",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda hbar, omega: hbar * omega,
        _x[0] * _x[1],
    ),
    _make_feynman(
        "I.39.10",
        "1.5 * p_r * V",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda p_r, v: 1.5 * p_r * v,
        sympy.Rational(3, 2) * _x[0] * _x[1],
    ),
    _make_feynman(
        "I.12.4",
        "Ef = q1 * r / (4 * pi * epsilon * r^3)",
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],  # q1, epsilon, r ∈ [1,5]
        lambda q1, epsilon, r: q1 * r / (4 * math.pi * epsilon * r**3),
        _x[0] * _x[2] / (4 * sympy.pi * _x[1] * _x[2] ** 3),
    ),
    _make_feynman(
        "II.3.24",
        "flux = Pwr / (4 * pi * r^2)",
        2,
        [(1.0, 5.0), (1.0, 5.0)],  # Pwr, r ∈ [1,5]
        lambda pwr, r: pwr / (4 * math.pi * r**2),
        _x[0] / (4 * sympy.pi * _x[1] ** 2),
    ),
    _make_feynman(
        "I.10.7",
        "m0 / sqrt(1 - v^2/c^2)",
        3,
        [(1.0, 5.0), (1.0, 2.0), (3.0, 10.0)],  # Table 2: m_0∈[1,5], v∈[1,2], c∈[3,10]
        lambda m0, v, c: m0 / np.sqrt(1 - (v / c) ** 2),
        _x[0] / sympy.sqrt(1 - _x[1] ** 2 / _x[2] ** 2),
    ),
    _make_feynman(
        "I.48.20",
        "m*c^2 / sqrt(1 - (v/c)^2)",
        3,
        [(1.0, 5.0), (3.0, 10.0), (1.0, 2.0)],  # m∈[1,5], c∈[3,10], v∈[1,2] (v<c required)
        lambda m, c, v: m * c**2 / np.sqrt(1 - (v / c) ** 2),
        _x[0] * _x[1] ** 2 / sympy.sqrt(1 - _x[2] ** 2 / _x[1] ** 2),
    ),
]


def generate_data(
    benchmark: dict[str, Any],
    n_samples: int = 200,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Generate train/test data for a Feynman benchmark.

    Following Liu2025 Section 4.1: 80/20 train/test split.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    nv = benchmark["num_variables"]
    var_ranges = benchmark["var_ranges"]
    fn = benchmark["target_fn"]

    # Generate all samples.
    x_all = np.column_stack([rng.uniform(lo, hi, n_samples) for lo, hi in var_ranges])

    # Compute target.
    args = [x_all[:, i] for i in range(nv)]
    y_all = fn(*args)

    # Train/test split.
    n_train = int(n_samples * train_ratio)
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    return x_train, y_train, x_test, y_test


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a Feynman benchmark by ID (e.g., 'I.6.20a')."""
    for b in FEYNMAN_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown benchmark: {name}. Available: {[b['name'] for b in FEYNMAN_BENCHMARKS]}"
    )
