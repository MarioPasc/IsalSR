"""Hard-tier symbolic regression benchmark definitions.

Ten additional problems selected per the hard-benchmark proposal
(``docs/md_files/changes/hard_benchmark_proposal.md``, 2026-04-13):

Extended Feynman (Udrescu & Tegmark, 2020):
    - I.15.10    Relativistic momentum
    - I.30.3     Single-slit diffraction intensity
    - I.37.4     Two-beam interference
    - II.11.27   Clausius-Mossotti polarisation
    - III.17.37  Angular distribution beta*(1 + alpha*cos(theta))

GP-hard classics:
    - Pagie-1         (Pagie & Hogeweg, 1997)
    - Korns-12        (Korns, 2011) — subsampled to 2000/2000 for UDFS budget
    - Vladislavleva-4 (Vladislavleva et al., 2009)
    - Vladislavleva-2 (Salustowicz & Schmidhuber, 1997)
    - Keijzer-6       (Keijzer, 2003)

Each problem's sampling protocol is encoded in a ``sampling`` dict, allowing
grid-, uniform-, and integer-grid-based schemes to coexist. The
``generate_data`` function dispatches on ``sampling.type`` and IGNORES the
top-level ``n_samples`` / ``train_ratio`` for grid-based problems
(Pagie-1, Keijzer-6, Vladislavleva-2) whose data size is defined by the
canonical protocol.

Operator set expansion: Pagie-1 needs x^(-4), I.15.10 and I.37.4 need sqrt.
The matching YAML configs (``experiments/configs/{udfs,bingo}_hard.yaml``)
include ``sqrt`` and ``pow`` in the operator set.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def harmonic(x: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    """Harmonic-number target for Keijzer-6.

    H(n) = sum_{i=1}^{n} 1/i  for integer n >= 1.

    Evaluated exactly for integer inputs; any non-integer is rounded to the
    nearest positive integer (the canonical Keijzer-6 protocol uses integer
    x from 1..50 for training and 1..120 for testing).
    """
    n_int = np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)
    n_int = np.maximum(n_int, 1)  # Defensive; domain starts at 1.
    out = np.zeros_like(n_int, dtype=np.float64)
    # Compute once up to the largest n, then gather.
    n_max = int(n_int.max())
    inv = 1.0 / np.arange(1, n_max + 1, dtype=np.float64)
    cum = np.cumsum(inv)  # cum[i-1] = H(i)
    out = cum[n_int - 1]
    return out


def _make_hard(
    name: str,
    expression: str,
    sympy_expression: sympy.Expr,
    sympy_variables: list[sympy.Symbol],
    num_variables: int,
    var_ranges: list[tuple[float, float]],
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Create a hard-benchmark specification dict."""
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


# SymPy symbols reused across problems.
_x = [sympy.Symbol(f"x_{i}") for i in range(5)]


# ----------------------------------------------------------------------
# Extended Feynman (uniform sampling, 1000 train + 250 test)
# ----------------------------------------------------------------------

_FEYNMAN_HARD: list[dict[str, Any]] = [
    _make_hard(
        "I.15.10",
        "m0*v / sqrt(1 - v^2/c^2)",
        _x[0] * _x[1] / sympy.sqrt(1 - _x[1] ** 2 / _x[2] ** 2),
        _x[:3],
        3,
        [(1.0, 5.0), (1.0, 2.0), (3.0, 10.0)],  # m0, v, c (v<c by construction)
        lambda m0, v, c: m0 * v / np.sqrt(1 - (v / c) ** 2),
        {"type": "uniform"},
    ),
    _make_hard(
        "I.30.3",
        "sin(n*theta/2)^2 / sin(theta/2)^2",
        sympy.sin(_x[0] * _x[1] / 2) ** 2 / sympy.sin(_x[1] / 2) ** 2,
        _x[:2],
        2,
        [(1.0, 5.0), (1.0, 5.0)],  # n, theta (theta in [1,5] avoids 0, 2*pi)
        lambda n, theta: np.sin(n * theta / 2) ** 2 / np.sin(theta / 2) ** 2,
        {"type": "uniform"},
    ),
    _make_hard(
        "I.37.4",
        "I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
        _x[0] + _x[1] + 2 * sympy.sqrt(_x[0] * _x[1]) * sympy.cos(_x[2]),
        _x[:3],
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],  # I1, I2, delta
        lambda i1, i2, delta: i1 + i2 + 2 * np.sqrt(i1 * i2) * np.cos(delta),  # noqa: E501
        {"type": "uniform"},
    ),
    _make_hard(
        "II.11.27",
        "n*alpha/(1 - n*alpha/3) * epsilon * Ef",
        _x[0] * _x[1] / (1 - _x[0] * _x[1] / 3) * _x[2] * _x[3],
        _x[:4],
        4,
        # n, alpha, epsilon, Ef -- the database ranges.  n, alpha <= 1 keeps
        # 1 - n*alpha/3 in [2/3, 1], so the denominator never approaches zero.
        [(0.0, 1.0), (0.0, 1.0), (1.0, 2.0), (1.0, 2.0)],
        lambda n, alpha, eps, ef: n * alpha / (1 - n * alpha / 3) * eps * ef,
        {"type": "uniform"},
    ),
    _make_hard(
        "III.17.37",
        "beta*(1 + alpha*cos(theta))",
        _x[0] * (1 + _x[1] * sympy.cos(_x[2])),
        _x[:3],
        3,
        [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],  # beta, alpha, theta
        lambda beta, alpha, theta: beta * (1 + alpha * np.cos(theta)),
        {"type": "uniform"},
    ),
]

# ----------------------------------------------------------------------
# GP-hard classics (per-problem sampling protocols)
# ----------------------------------------------------------------------

_GP_HARD: list[dict[str, Any]] = [
    _make_hard(
        "Pagie-1",
        "1/(1 + x^(-4)) + 1/(1 + y^(-4))",
        1 / (1 + _x[0] ** (-4)) + 1 / (1 + _x[1] ** (-4)),
        _x[:2],
        2,
        [(-5.0, 5.0), (-5.0, 5.0)],
        lambda x, y: 1.0 / (1.0 + x ** (-4)) + 1.0 / (1.0 + y ** (-4)),
        # Canonical Pagie-1 protocol (GPBenchmarks.org):
        # Train: 26x26 grid with spacing 0.4 over [-5, 5] -> naturally skips 0.
        # Test: 50x50 grid with spacing ~0.2 over [-5, 5] -> also skips 0 if offset.
        {"type": "grid_2d_skip_zero", "train_step": 0.4, "test_step": 0.2},
    ),
    _make_hard(
        "Korns-12",
        "2.0 - 2.1*cos(9.8*x1)*sin(1.3*x5)",
        sympy.Float(2.0)
        - sympy.Float(2.1)
        * sympy.cos(sympy.Float(9.8) * _x[0])
        * sympy.sin(sympy.Float(1.3) * _x[4]),
        _x[:5],
        5,
        [(-50.0, 50.0)] * 5,  # x1..x5 ∈ [-50, 50]; only x1, x5 are relevant.
        lambda x1, x2, x3, x4, x5: 2.0 - 2.1 * np.cos(9.8 * x1) * np.sin(1.3 * x5),
        # Subsampled from Korns (2011) canonical 10000/10000 to 2000/2000 for
        # UDFS single-process budget (12h); preserves 5-variable feature-selection
        # difficulty. Documented deviation.
        {"type": "uniform", "n_train_override": 2000, "n_test_override": 2000},
    ),
    _make_hard(
        "Vladislavleva-4",
        "10 / (5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)",
        10
        / (
            5
            + (_x[0] - 3) ** 2
            + (_x[1] - 3) ** 2
            + (_x[2] - 3) ** 2
            + (_x[3] - 3) ** 2
            + (_x[4] - 3) ** 2
        ),
        _x[:5],
        5,
        [(0.05, 6.05)] * 5,
        lambda x1, x2, x3, x4, x5: (
            10.0
            / (5.0 + (x1 - 3) ** 2 + (x2 - 3) ** 2 + (x3 - 3) ** 2 + (x4 - 3) ** 2 + (x5 - 3) ** 2)
        ),
        {"type": "uniform", "n_train_override": 1024, "n_test_override": 5000},
    ),
    _make_hard(
        "Vladislavleva-2",
        "exp(-x)*x^3*(cos(x)*sin(x))*(cos(x)*sin(x)^2 - 1)",
        sympy.exp(-_x[0])
        * _x[0] ** 3
        * (sympy.cos(_x[0]) * sympy.sin(_x[0]))
        * (sympy.cos(_x[0]) * sympy.sin(_x[0]) ** 2 - 1),
        _x[:1],
        1,
        [(0.05, 10.0)],
        lambda x: np.exp(-x) * x**3 * (np.cos(x) * np.sin(x)) * (np.cos(x) * np.sin(x) ** 2 - 1),
        # Train: 100 uniform in [0.05, 10].
        # Test: 221 grid from 0.05 stepping 0.05 up to (inclusive) ~11.05.
        {
            "type": "grid_1d_train_uniform_test_grid",
            "n_train": 100,
            "test_step": 0.05,
            "test_lo": 0.05,
            "test_hi": 11.05,
        },
    ),
    _make_hard(
        "Keijzer-6",
        "log(x)",  # Approximation target; true y is harmonic number H(x).
        sympy.log(_x[0]) + sympy.Float(0.5772156649015329),  # Euler-Mascheroni approx.
        _x[:1],
        1,
        [(1.0, 120.0)],
        harmonic,
        # Train: x = 1, 2, ..., 50 (integer).
        # Test:  x = 1, 2, ..., 120 (integer) — extrapolation beyond training.
        {"type": "integer_grid", "train_lo": 1, "train_hi": 50, "test_lo": 1, "test_hi": 120},
    ),
]


HARD_BENCHMARKS: list[dict[str, Any]] = _FEYNMAN_HARD + _GP_HARD


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
    """Uniform random sampling (Feynman / Vlad-4 / Korns-12)."""
    sampling = benchmark["sampling"]
    # Allow per-problem override of train/test sizes.
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


def _sample_pagie1_grid(
    benchmark: dict[str, Any],
    seed: int,  # noqa: ARG001 — grid is deterministic; seed kept for API parity.
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """2D grid sampling skipping x=0 and y=0 (Pagie-1).

    Train: 26x26 grid over [-5, 5] with step 0.4 -> 676 points.
        The grid {-5, -4.6, ..., -0.2, 0.2, ..., 5.0} naturally skips 0.
    Test:  50x50 grid over [-5, 5] with step ~0.2 -> 2500 points.
        Offset to also skip exact zeros in both dims.
    """
    sampling = benchmark["sampling"]
    train_step = sampling["train_step"]
    test_step = sampling["test_step"]
    lo, hi = benchmark["var_ranges"][0]

    # Train grid: start at lo, 26 points with step 0.4.
    train_axis = np.arange(lo, hi + train_step / 2, train_step)
    # Defensive: ensure no point is exactly zero (shouldn't happen with spacing 0.4 from -5).
    train_axis = train_axis[~np.isclose(train_axis, 0.0, atol=1e-9)]

    # Test grid: start offset from lo to avoid integer points; skip zero.
    test_axis = np.arange(lo + test_step / 2.0, hi, test_step)
    test_axis = test_axis[~np.isclose(test_axis, 0.0, atol=1e-9)]

    tx, ty = np.meshgrid(train_axis, train_axis, indexing="ij")
    x_train = np.column_stack([tx.ravel(), ty.ravel()])

    ttx, tty = np.meshgrid(test_axis, test_axis, indexing="ij")
    x_test = np.column_stack([ttx.ravel(), tty.ravel()])

    fn = benchmark["target_fn"]
    y_train = fn(x_train[:, 0], x_train[:, 1])
    y_test = fn(x_test[:, 0], x_test[:, 1])

    return x_train, y_train, x_test, y_test


def _sample_vlad2(
    benchmark: dict[str, Any],
    seed: int,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Uniform train + grid test (Vladislavleva-2)."""
    sampling = benchmark["sampling"]
    n_train = int(sampling["n_train"])
    test_step = sampling["test_step"]
    test_lo = sampling["test_lo"]
    test_hi = sampling["test_hi"]
    train_lo, train_hi = benchmark["var_ranges"][0]

    rng = np.random.default_rng(seed)
    x_train_1d = rng.uniform(train_lo, train_hi, n_train)
    x_test_1d = np.arange(test_lo, test_hi + test_step / 2, test_step)

    x_train = x_train_1d.reshape(-1, 1)
    x_test = x_test_1d.reshape(-1, 1)

    fn = benchmark["target_fn"]
    y_train = fn(x_train_1d)
    y_test = fn(x_test_1d)

    return x_train, y_train, x_test, y_test


def _sample_integer_grid(
    benchmark: dict[str, Any],
    seed: int,  # noqa: ARG001 — integer grid is deterministic.
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Integer-grid sampling (Keijzer-6).

    Train: x = train_lo .. train_hi inclusive (integers).
    Test:  x = test_lo .. test_hi inclusive (integers), covers and extends
           beyond the train range.
    """
    s = benchmark["sampling"]
    train = np.arange(s["train_lo"], s["train_hi"] + 1, dtype=np.float64)
    test = np.arange(s["test_lo"], s["test_hi"] + 1, dtype=np.float64)

    x_train = train.reshape(-1, 1)
    x_test = test.reshape(-1, 1)

    fn = benchmark["target_fn"]
    y_train = fn(train)
    y_test = fn(test)

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
    """Generate train/test data for a hard benchmark.

    Uses the same signature as ``feynman.generate_data`` for orchestrator
    compatibility, but internally dispatches on the ``sampling.type`` key
    inside each benchmark dict. For grid-based protocols
    (Pagie-1, Keijzer-6, Vladislavleva-2), ``n_samples`` and ``train_ratio``
    are IGNORED — the canonical per-problem sizes are always used.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    sampling_type = benchmark["sampling"]["type"]

    if sampling_type == "uniform":
        return _sample_uniform(benchmark, n_samples, train_ratio, seed)
    if sampling_type == "grid_2d_skip_zero":
        return _sample_pagie1_grid(benchmark, seed)
    if sampling_type == "grid_1d_train_uniform_test_grid":
        return _sample_vlad2(benchmark, seed)
    if sampling_type == "integer_grid":
        return _sample_integer_grid(benchmark, seed)
    raise ValueError(f"Unknown sampling type: {sampling_type}")


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a hard benchmark by name."""
    for b in HARD_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown hard benchmark: {name}. Available: {[b['name'] for b in HARD_BENCHMARKS]}"
    )


# Silence unused-import warning for math (kept for future trig-constant insertion).
_ = math
