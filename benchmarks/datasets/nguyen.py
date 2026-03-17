"""Nguyen symbolic regression benchmark definitions.

12 standard Nguyen benchmarks used across the SR literature.
EXACT expressions from Liu2025 (GraphDSR, Neural Networks 187:107405), Table 1.
Originally from: Uy et al. (2011). Semantically-based crossover in GP.

Data configuration follows Liu2025 Section 4.1:
    - Training: 20 points uniformly sampled from x_range
    - Testing: 100 points uniformly sampled from x_range
    - Seed: 42 for reproducibility
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _make_benchmark(
    name: str,
    expression: str,
    num_variables: int,
    x_range: tuple[float, float],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
) -> dict[str, Any]:
    """Create a benchmark specification dict."""
    return {
        "name": name,
        "expression": expression,
        "num_variables": num_variables,
        "x_range": x_range,
        "target_fn": target_fn,
    }


# ======================================================================
# Nguyen-1 through Nguyen-12 (Liu2025 Table 1, verified against PDF page 6)
# ======================================================================

NGUYEN_BENCHMARKS: list[dict[str, Any]] = [
    _make_benchmark(
        "Nguyen-1",
        "x^3 + x^2 + x",
        1,
        (-1.0, 1.0),
        lambda x: x**3 + x**2 + x,
    ),
    _make_benchmark(
        "Nguyen-2",
        "x^4 + x^3 + x^2 + x",
        1,
        (-1.0, 1.0),
        lambda x: x**4 + x**3 + x**2 + x,
    ),
    _make_benchmark(
        "Nguyen-3",
        "x^5 + x^4 + x^3 + x^2 + x",
        1,
        (-1.0, 1.0),
        lambda x: x**5 + x**4 + x**3 + x**2 + x,
    ),
    _make_benchmark(
        "Nguyen-4",
        "x^6 + x^5 + x^4 + x^3 + x^2 + x",
        1,
        (-1.0, 1.0),
        lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
    ),
    _make_benchmark(
        "Nguyen-5",
        "sin(x^2) * cos(x) - 1",
        1,
        (-1.0, 1.0),
        lambda x: np.sin(x**2) * np.cos(x) - 1,
    ),
    _make_benchmark(
        "Nguyen-6",
        "sin(x) + sin(x + x^2)",
        1,
        (-1.0, 1.0),
        lambda x: np.sin(x) + np.sin(x + x**2),
    ),
    _make_benchmark(
        "Nguyen-7",
        "log(x + 1) + log(x^2 + 1)",
        1,
        (0.0, 2.0),
        lambda x: np.log(x + 1) + np.log(x**2 + 1),
    ),
    _make_benchmark(
        "Nguyen-8",
        "sqrt(x)",
        1,
        (0.0, 4.0),
        lambda x: np.sqrt(x),
    ),
    _make_benchmark(
        "Nguyen-9",
        "sin(x) + sin(y^2)",
        2,
        (-1.0, 1.0),
        lambda x, y: np.sin(x) + np.sin(y**2),
    ),
    _make_benchmark(
        "Nguyen-10",
        "2 * sin(x) * cos(y)",
        2,
        (-1.0, 1.0),
        lambda x, y: 2 * np.sin(x) * np.cos(y),
    ),
    _make_benchmark(
        "Nguyen-11",
        "x^y",
        2,
        (0.0, 1.0),
        lambda x, y: np.power(np.abs(x) + 1e-10, y),  # protected for x~0
    ),
    _make_benchmark(
        "Nguyen-12",
        "x^4 - x^3 + 0.5*y^2 - y",
        2,
        (-1.0, 1.0),
        lambda x, y: x**4 - x**3 + 0.5 * y**2 - y,
    ),
]


def generate_data(
    benchmark: dict[str, Any],
    n_train: int = 20,
    n_test: int = 100,
    seed: int = 42,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Generate train/test data for a Nguyen benchmark.

    Following Liu2025 Section 4.1: uniform sampling within x_range.

    Args:
        benchmark: A benchmark dict from NGUYEN_BENCHMARKS.
        n_train: Number of training points (default 20, per Liu2025).
        n_test: Number of test points (default 100).
        seed: Random seed for reproducibility.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    nv = benchmark["num_variables"]
    lo, hi = benchmark["x_range"]
    fn = benchmark["target_fn"]

    x_train = rng.uniform(lo, hi, (n_train, nv))
    x_test = rng.uniform(lo, hi, (n_test, nv))

    if nv == 1:
        y_train = fn(x_train[:, 0])
        y_test = fn(x_test[:, 0])
    elif nv == 2:
        y_train = fn(x_train[:, 0], x_train[:, 1])
        y_test = fn(x_test[:, 0], x_test[:, 1])
    else:
        raise ValueError(f"Unsupported num_variables: {nv}")

    return x_train, y_train, x_test, y_test


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a benchmark by name (e.g., 'Nguyen-1')."""
    for b in NGUYEN_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown benchmark: {name}. Available: {[b['name'] for b in NGUYEN_BENCHMARKS]}"
    )
