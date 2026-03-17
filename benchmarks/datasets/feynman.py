"""Feynman physics equations benchmark definitions.

Selected equations from Liu2025 Table 2 (GraphDSR, Neural Networks 187:107405).
Original source: Udrescu & Tegmark (2020). AI Feynman. Science Advances 6(16).

Data configuration follows Liu2025 Section 4.1:
    - Train/test split: 80/20
    - Seed: 42 for reproducibility
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def _make_feynman(
    feynman_id: str,
    expression: str,
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
) -> dict[str, Any]:
    """Create a Feynman benchmark specification dict."""
    return {
        "name": feynman_id,
        "expression": expression,
        "num_variables": num_variables,
        "var_ranges": var_ranges,
        "target_fn": target_fn,
    }


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
    ),
    _make_feynman(
        "I.12.1",
        "mu * N_s",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda mu, n_s: mu * n_s,
    ),
    _make_feynman(
        "I.14.3",
        "m * g * z",
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        lambda m, g, z: m * g * z,
    ),
    _make_feynman(
        "I.25.13",
        "q / C",
        2,
        [(1.0, 3.0), (1.0, 3.0)],
        lambda q, c: q / c,
    ),
    _make_feynman(
        "I.34.27",
        "hbar * omega",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda hbar, omega: hbar * omega,
    ),
    _make_feynman(
        "I.39.10",
        "0.5 * p_r * V",
        2,
        [(1.0, 5.0), (1.0, 5.0)],
        lambda p_r, v: 0.5 * p_r * v,
    ),
    _make_feynman(
        "I.12.4",
        "q1 / (4 * pi * r * c)",
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],  # q1, r, c (Table 2: q_i, r, c ∈ [1,5])
        lambda q1, r, c: q1 / (4 * math.pi * r * c),
    ),
    _make_feynman(
        "II.3.24",
        "F_E = p * r / (4 * pi)",
        2,
        [(1.0, 5.0), (1.0, 5.0)],  # Table 2: p, r ∈ [1,5]
        lambda p, r: p * r / (4 * math.pi),
    ),
    _make_feynman(
        "I.10.7",
        "m0 / sqrt(1 - v^2/c^2)",
        3,
        [(1.0, 5.0), (1.0, 2.0), (3.0, 10.0)],  # Table 2: m_0∈[1,5], γ∈[1,2], c∈[3,10]
        lambda m0, v, c: m0 / np.sqrt(1 - (v / c) ** 2),
    ),
    _make_feynman(
        "I.48.20",
        "m*c^2 / sqrt(1 - (v/c)^2)",
        3,
        [(1.0, 5.0), (1.0, 2.0), (3.0, 10.0)],  # Table 2: m∈[1,5], c∈[1,2], v∈... (approx)
        lambda m, c, v: m * c**2 / np.sqrt(1 - (v / c) ** 2),
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
