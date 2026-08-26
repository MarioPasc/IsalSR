"""ODE-Strogatz benchmark definitions: the 14 published two-state ODE problems.

These are SRBench's ground-truth track minus the AI-Feynman problems
(La Cava et al., NeurIPS 2021, Datasets & Benchmarks). Each problem is one
component of a two-dimensional dynamical system taken from Strogatz's
*Nonlinear Dynamics and Chaos*: the two state variables ``x`` and ``y`` are the
features and the target is the corresponding time derivative.

Unlike every other suite in ``benchmarks/datasets/``, these are **published
fixed datasets**, not generated ones. The 400 rows per problem are vendored
byte-verbatim from PMLB (Romano et al., 2021; MIT licence) under
``data/strogatz/strogatz_<key>.tsv.gz``; see ``data/strogatz/PROVENANCE.md``.
``generate_data`` therefore only *splits* the published rows (300 train /
100 test, SRBench's 75/25) and ignores ``n_samples`` / ``train_ratio``.

The 14 ground-truth equations below were transcribed from the ODE-Strogatz
generator (``simulate_ode.m``) and cross-checked against each PMLB
``metadata.yaml``. ``shearflow1`` is published as ``cot(y)*cos(x)``; it is
written here as ``cos(y)*cos(x)/sin(y)`` because Sigma_SR has no ``cot`` label
and supplies the reciprocal through ``Inv``. No equation is simplified.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import sympy

_DATA_DIR = Path(__file__).parent / "data" / "strogatz"

_N_ROWS = 400
_N_TRAIN = 300
_N_TEST = 100

_STROGATZ_KEYS: tuple[str, ...] = (
    "bacres1",
    "bacres2",
    "barmag1",
    "barmag2",
    "glider1",
    "glider2",
    "lv1",
    "lv2",
    "predprey1",
    "predprey2",
    "shearflow1",
    "shearflow2",
    "vdp1",
    "vdp2",
)


# ----------------------------------------------------------------------
# Vendored-data loading
# ----------------------------------------------------------------------


def data_key(name: str) -> str:
    """Map a benchmark name to its vendored-file key.

    Args:
        name: Benchmark name, e.g. ``"Strogatz-vdp1"``.

    Returns:
        The PMLB dataset key, e.g. ``"vdp1"``.

    Raises:
        ValueError: If ``name`` does not follow the ``Strogatz-<key>`` form
            or the key is not one of the 14 known problems.
    """
    prefix, sep, key = name.partition("-")
    if prefix != "Strogatz" or not sep or key not in _STROGATZ_KEYS:
        raise ValueError(f"Unknown strogatz benchmark: {name}")
    return key


def data_path(key: str) -> Path:
    """Return the absolute path of a vendored ODE-Strogatz file.

    Args:
        key: PMLB dataset key, e.g. ``"vdp1"``.

    Returns:
        Path to ``data/strogatz/strogatz_<key>.tsv.gz``, resolved relative to
        this module so it works from any working directory (SLURM included).
    """
    return _DATA_DIR / f"strogatz_{key}.tsv.gz"


@cache
def load_published(
    key: str,
) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """Load the 400 published rows of one ODE-Strogatz problem.

    The column order is read from the header line rather than assumed.

    Args:
        key: PMLB dataset key, e.g. ``"vdp1"``.

    Returns:
        A pair ``(features, target)`` where ``features`` has shape
        ``(400, 2)`` with columns ``(x, y)`` and ``target`` has shape
        ``(400,)``.

    Raises:
        ValueError: If the header is not ``target``/``x``/``y`` or the file
            does not contain exactly 400 rows.
    """
    path = data_path(key)
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        table = np.loadtxt(handle, dtype=np.float64)

    if set(header) != {"target", "x", "y"}:
        raise ValueError(f"Unexpected header in {path}: {header}")
    if table.shape != (_N_ROWS, 3):
        raise ValueError(f"Expected ({_N_ROWS}, 3) in {path}, got {table.shape}")

    features = np.column_stack((table[:, header.index("x")], table[:, header.index("y")]))
    target = table[:, header.index("target")]
    return features, target


def _empirical_ranges(key: str) -> list[tuple[float, float]]:
    """Compute per-column ``[min, max]`` of the published features.

    These ranges are **documentation** (Appendix D.1), not a sampling domain:
    nothing is ever drawn from them, since the rows are fixed and published.
    """
    features, _ = load_published(key)
    return [
        (float(features[:, i].min()), float(features[:, i].max())) for i in range(features.shape[1])
    ]


# ----------------------------------------------------------------------
# Benchmark registry
# ----------------------------------------------------------------------


def _make_strogatz(
    key: str,
    expression: str,
    sympy_expression: sympy.Expr,
    target_fn: Callable[..., np.ndarray[Any, np.dtype[Any]]],
) -> dict[str, Any]:
    """Create an ODE-Strogatz benchmark specification dict."""
    return {
        "name": f"Strogatz-{key}",
        "expression": expression,
        "sympy_expression": sympy_expression,
        "sympy_variables": list(_x),
        "num_variables": 2,
        "var_ranges": _empirical_ranges(key),
        "target_fn": target_fn,
        "sampling": {
            "type": "published_fixed",
            "n_train_override": _N_TRAIN,
            "n_test_override": _N_TEST,
        },
    }


# x_0 is the state variable x, x_1 is y, in the published column order.
_x = [sympy.Symbol(f"x_{i}") for i in range(2)]
_X, _Y = _x

_HALF = sympy.Rational(1, 2)
_THIRD = sympy.Rational(1, 3)
_TENTH = sympy.Rational(1, 10)


STROGATZ_BENCHMARKS: list[dict[str, Any]] = [
    # Bacterial respiration (Strogatz, Ex. 7.3.5).
    _make_strogatz(
        "bacres1",
        "20 - x - x*y/(1 + 0.5*x^2)",
        20 - _X - (_X * _Y) / (1 + _HALF * _X**2),
        lambda x, y: 20 - x - (x * y) / (1 + 0.5 * x**2),
    ),
    _make_strogatz(
        "bacres2",
        "10 - x*y/(1 + 0.5*x^2)",
        10 - (_X * _Y) / (1 + _HALF * _X**2),
        lambda x, y: 10 - (x * y) / (1 + 0.5 * x**2),
    ),
    # Bar magnets (Strogatz, Ex. 8.6.9).
    _make_strogatz(
        "barmag1",
        "0.5*sin(x - y) - sin(x)",
        _HALF * sympy.sin(_X - _Y) - sympy.sin(_X),
        lambda x, y: 0.5 * np.sin(x - y) - np.sin(x),
    ),
    _make_strogatz(
        "barmag2",
        "0.5*sin(y - x) - sin(y)",
        _HALF * sympy.sin(_Y - _X) - sympy.sin(_Y),
        lambda x, y: 0.5 * np.sin(y - x) - np.sin(y),
    ),
    # Glider (Strogatz, Ex. 6.4.6).
    _make_strogatz(
        "glider1",
        "-0.05*x^2 - sin(y)",
        -sympy.Rational(1, 20) * _X**2 - sympy.sin(_Y),
        lambda x, y: -0.05 * x**2 - np.sin(y),
    ),
    _make_strogatz(
        "glider2",
        "x - cos(y)/x",
        _X - sympy.cos(_Y) / _X,
        lambda x, y: x - np.cos(y) / x,
    ),
    # Lotka-Volterra competition (Strogatz, Ex. 6.4.4).
    _make_strogatz(
        "lv1",
        "3*x - 2*x*y - x^2",
        3 * _X - 2 * _X * _Y - _X**2,
        lambda x, y: 3 * x - 2 * x * y - x**2,
    ),
    _make_strogatz(
        "lv2",
        "2*y - x*y - y^2",
        2 * _Y - _X * _Y - _Y**2,
        lambda x, y: 2 * y - x * y - y**2,
    ),
    # Predator-prey (Strogatz, Ex. 7.2.19).
    _make_strogatz(
        "predprey1",
        "x*(4 - x - y/(1 + x))",
        _X * (4 - _X - _Y / (1 + _X)),
        lambda x, y: x * (4 - x - y / (1 + x)),
    ),
    _make_strogatz(
        "predprey2",
        "y*(x/(1 + x) - 0.075*y)",
        _Y * (_X / (1 + _X) - sympy.Rational(3, 40) * _Y),
        lambda x, y: y * (x / (1 + x) - 0.075 * y),
    ),
    # Shear flow (Strogatz, Ex. 6.6.1). cot(y) is written cos(y)/sin(y).
    _make_strogatz(
        "shearflow1",
        "cot(y)*cos(x)  [written cos(y)*cos(x)/sin(y)]",
        sympy.cos(_Y) * sympy.cos(_X) / sympy.sin(_Y),
        lambda x, y: np.cos(y) * np.cos(x) / np.sin(y),
    ),
    _make_strogatz(
        "shearflow2",
        "(cos(y)^2 + 0.1*sin(y)^2)*sin(x)",
        (sympy.cos(_Y) ** 2 + _TENTH * sympy.sin(_Y) ** 2) * sympy.sin(_X),
        lambda x, y: (np.cos(y) ** 2 + 0.1 * np.sin(y) ** 2) * np.sin(x),
    ),
    # Van der Pol oscillator (Strogatz, Ex. 7.5.1), Lienard form.
    _make_strogatz(
        "vdp1",
        "10*(y - (1/3)*(x^3 - x))",
        10 * (_Y - _THIRD * (_X**3 - _X)),
        lambda x, y: 10 * (y - (1.0 / 3.0) * (x**3 - x)),
    ),
    _make_strogatz(
        "vdp2",
        "-(1/10)*x",
        -_TENTH * _X,
        # y is deliberately unused: dx2/dt depends on x only.
        lambda x, y: -(1.0 / 10.0) * x,
    ),
]


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------


def _split_published(
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
    """Split the published rows into train/test with a seeded permutation.

    Per-problem size overrides take precedence over ``n_samples`` and
    ``train_ratio``, exactly as in ``roundoff._sample_uniform``. All 14
    ODE-Strogatz problems carry overrides (300/100), so the caller's request
    is ignored; the fallback path exists only for symmetry with the other
    suites.
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

    features, target = load_published(data_key(benchmark["name"]))
    if n_train + n_test > features.shape[0]:
        raise ValueError(
            f"Requested {n_train} + {n_test} rows but only "
            f"{features.shape[0]} are published for {benchmark['name']}"
        )

    perm = np.random.default_rng(seed).permutation(features.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]

    return (
        features[train_idx],
        target[train_idx],
        features[test_idx],
        target[test_idx],
    )


def generate_data(
    benchmark: dict[str, Any],
    n_samples: int = 400,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
]:
    """Generate train/test data for an ODE-Strogatz benchmark.

    Same signature as ``roundoff.generate_data`` for orchestrator
    compatibility. Because the rows are published rather than sampled,
    ``n_samples`` and ``train_ratio`` are ignored whenever the benchmark
    carries ``n_train_override``/``n_test_override`` (all 14 do).

    Args:
        benchmark: An entry of ``STROGATZ_BENCHMARKS``.
        n_samples: Ignored when size overrides are present.
        train_ratio: Ignored when size overrides are present.
        seed: Seed of the permutation that defines the split.

    Returns:
        ``(X_train, y_train, X_test, y_test)`` with shapes ``(300, 2)``,
        ``(300,)``, ``(100, 2)``, ``(100,)``.

    Raises:
        ValueError: If the benchmark declares an unknown sampling type.
    """
    sampling_type = benchmark["sampling"]["type"]

    if sampling_type == "published_fixed":
        return _split_published(benchmark, n_samples, train_ratio, seed)
    raise ValueError(f"Unknown sampling type: {sampling_type}")


def get_benchmark(name: str) -> dict[str, Any]:
    """Get an ODE-Strogatz benchmark by name.

    Args:
        name: Benchmark name, e.g. ``"Strogatz-vdp1"``.

    Returns:
        The benchmark specification dict.

    Raises:
        ValueError: If no benchmark carries that name.
    """
    for b in STROGATZ_BENCHMARKS:
        if b["name"] == name:
            return b
    raise ValueError(
        f"Unknown strogatz benchmark: {name}. Available: {[b['name'] for b in STROGATZ_BENCHMARKS]}"
    )
