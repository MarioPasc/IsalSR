"""NumPy-vectorized protected mathematical operations.

Provides numerically safe vectorized operations for expression evaluation.
These mirror the scalar protected ops in ``core.dag_evaluator`` but operate
on NumPy arrays for efficient batch evaluation.

Standard in symbolic regression: Koza (1992), Schmidt & Lipson (2009).

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_VALUE = 1e15


def protected_log(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected logarithm: log(|x| + epsilon)."""
    return np.log(np.abs(x) + 1e-10)  # type: ignore[no-any-return]


def protected_div(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected division: x / y where |y| > epsilon, else 1.0."""
    return np.where(np.abs(y) > 1e-10, x / y, 1.0)


def protected_sqrt(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected square root: sqrt(|x|)."""
    return np.sqrt(np.abs(x))  # type: ignore[no-any-return]


def protected_exp(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected exponential: exp(clip(x, -500, 500))."""
    return np.exp(np.clip(x, -500.0, 500.0))  # type: ignore[no-any-return]


def protected_pow(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected power: |x|^clip(y) with overflow protection."""
    base = np.abs(x) + 1e-10
    exponent = np.clip(y, -100.0, 100.0)
    result = np.power(base, exponent)
    return np.clip(result, -_MAX_VALUE, _MAX_VALUE)  # type: ignore[no-any-return]


def protected_inv(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Protected multiplicative inverse: 1/x where |x| > epsilon, else 1.0.

    Semantically equivalent to ``protected_div(np.ones_like(x), x)``.
    """
    return np.where(np.abs(x) > 1e-10, 1.0 / x, 1.0)


def clamp(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Clamp array values: NaN -> 0, clip to [-MAX_VALUE, MAX_VALUE]."""
    result = np.where(np.isnan(x), 0.0, x)
    return np.clip(result, -_MAX_VALUE, _MAX_VALUE)
