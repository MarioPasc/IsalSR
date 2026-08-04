"""Tests for experiments.models.analyzer.metrics.

Two things are under test:

1. **Metric preservation.** ``jaccard_index`` and ``solution_recovered`` are
   reported in the paper, so their values must not drift. The pinned table in
   ``PINNED`` was measured against the pre-timeout implementation and any
   change to it is a change to a published number.

   This matters because the obvious optimisation -- dropping
   ``sympy.simplify`` from ``_get_subexpressions``, which dominates the cost --
   is **not** metric-preserving. It diverges on the Pythagorean identity and on
   cancelling rationals, where SymPy's automatic evaluation is not enough.
   ``test_simplify_is_load_bearing`` pins those three cases explicitly so a
   future optimiser sees the counter-example before making the change.

2. **Timeout semantics.** An unbounded ``sympy.simplify`` in the post-search
   path hung a Picasso probe cell past its wall limit and cost the whole run
   (2026-08-02, T04 probe task 7). The bound distinguishes three outcomes:

   =============== ====================== ==========================
   outcome         ``solution_recovered`` meaning
   =============== ====================== ==========================
   proved equal    ``True``               recovered
   proved unequal  ``False``              not recovered
   timed out       ``None``               undetermined -- excluded
   SymPy raised    ``None``               undetermined -- excluded
   =============== ====================== ==========================

   Recording a timeout as ``False`` would be a false negative landing
   preferentially on whichever arm finds larger expressions, biasing the
   paired comparison. A SymPy exception is treated identically (fairness
   audit, 2026-08-04): conversion blowups correlate with expression size for
   the same reason timeouts do.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")

from experiments.models.analyzer.metrics import (  # noqa: E402
    jaccard_index,
    solution_recovered,
)

x, y = sympy.symbols("x y")


# ----------------------------------------------------------------------
# 1. Metric preservation
# ----------------------------------------------------------------------

# (name, found, true, expected_jaccard, expected_recovered)
PINNED: list[tuple[str, Any, Any, float, bool]] = [
    ("identical", x**3 + x**2 + x, x**3 + x**2 + x, 1.0, True),
    ("subset", x**3 + x**2 + x, x**3 + x**2, 0.7142857142857143, False),
    ("disjoint", sympy.sin(x), sympy.exp(y), 0.0, False),
    (
        "pythagorean",
        sympy.sin(x) ** 2 + sympy.cos(x) ** 2,
        sympy.Integer(1),
        0.14285714285714285,
        True,
    ),
    ("pyth_embedded", y * (sympy.sin(x) ** 2 + sympy.cos(x) ** 2), y, 0.125, True),
    ("cancelling_rational", (x**2 - 1) / (x - 1), x + 1, 0.2222222222222222, True),
    (
        "pagie1_true",
        1 / (1 + x ** (-4)) + 1 / (1 + y ** (-4)),
        1 / (1 + x ** (-4)) + 1 / (1 + y ** (-4)),
        1.0,
        True,
    ),
    (
        "pagie1_perturbed",
        1 / (1 + x ** (-4)) + 1 / (1 + y ** (-4)),
        1 / (1 + x ** (-4)) + 1 / (1 + y ** (-3)),
        0.47058823529411764,
        False,
    ),
    (
        "log_sum",
        sympy.log(x) + sympy.log(x + 1) + sympy.log(x + 2),
        sympy.log(x) + sympy.log(x + 1),
        0.5,
        False,
    ),
    ("sqrt_nest", sympy.sqrt(x**2 + y**2), sympy.sqrt(x**2 + y**2) + x, 0.8888888888888888, False),
]


@pytest.mark.parametrize(
    ("name", "found", "true", "expected"),
    [(n, f, t, j) for n, f, t, j, _ in PINNED],
    ids=[n for n, *_ in PINNED],
)
def test_jaccard_index_pinned(name: str, found: Any, true: Any, expected: float) -> None:
    """Jaccard must not drift from the published implementation."""
    assert jaccard_index(found, true) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("name", "found", "true", "expected"),
    [(n, f, t, r) for n, f, t, _, r in PINNED],
    ids=[n for n, *_ in PINNED],
)
def test_solution_recovered_pinned(name: str, found: Any, true: Any, expected: bool) -> None:
    """Solution recovery must not drift from the published implementation."""
    assert solution_recovered(found, true) is expected


def test_simplify_is_load_bearing() -> None:
    """Counter-example set: dropping simplify() from _get_subexpressions changes Jaccard.

    Measured 2026-08-02. A cheaper canonicaliser (``str``, ``srepr``) gives
    0.0, 0.1111 and 0.1 on these three pairs respectively; ``cancel`` recovers
    the rational case but not the trigonometric ones. Do not "optimise"
    ``_get_subexpressions`` by removing the call -- bound its runtime instead.
    """
    assert jaccard_index(sympy.sin(x) ** 2 + sympy.cos(x) ** 2, sympy.Integer(1)) > 0.0
    assert jaccard_index(y * (sympy.sin(x) ** 2 + sympy.cos(x) ** 2), y) == pytest.approx(0.125)
    assert jaccard_index((x**2 - 1) / (x - 1), x + 1) == pytest.approx(0.2222222222222222)


# ----------------------------------------------------------------------
# 2. Timeout semantics
# ----------------------------------------------------------------------


@pytest.fixture()
def slow_simplify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sympy.simplify block, deterministically and machine-independently."""

    def _blocking(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(30)
        return sympy.Integer(0)

    monkeypatch.setattr(sympy, "simplify", _blocking)


@pytest.mark.usefixtures("slow_simplify")
def test_solution_recovered_timeout_is_none() -> None:
    """A timeout is undetermined (None), never a negative result (False)."""
    t0 = time.perf_counter()
    result = solution_recovered(x + 1, x + 1, timeout_s=0.25)
    elapsed = time.perf_counter() - t0

    assert result is None
    assert elapsed < 5.0, "timeout did not actually interrupt the call"


@pytest.mark.usefixtures("slow_simplify")
def test_jaccard_timeout_is_none() -> None:
    """Jaccard reports None on timeout so it is excluded, not counted as 0.0."""
    result = jaccard_index(x**3 + x**2 + x, x**3 + x**2, timeout_s=0.25)
    assert result is None


def test_sympy_exception_is_none_not_false() -> None:
    """A SymPy exception is undetermined (None), like a timeout.

    ``False``/``0.0`` used to be returned here, which is a *decided* negative.
    SymPy blowups correlate with expression size exactly as timeouts do, so
    scoring them as failures lands preferentially on whichever arm finds larger
    expressions -- the same asymmetric-failure argument that made a timeout
    ``None``. ``False`` stays reserved for a proved non-equivalence.
    """
    assert solution_recovered("not an expression", x + 1) is None
    # ``jaccard_index`` does *not* raise here: ``_get_subexpressions`` returns
    # the empty set for a non-``sympy.Basic`` input, so 0.0 comes out of the
    # normal path, not the exception handler. Its exception handler is covered
    # in tests/unit/test_stats_fairness_fixes.py.
    assert jaccard_index("not an expression", x + 1) == 0.0


def test_timeout_disabled_by_nonpositive() -> None:
    """timeout_s <= 0 disables the bound (used by offline re-analysis)."""
    assert solution_recovered(x + 1, x + 1, timeout_s=0) is True


# ----------------------------------------------------------------------
# 3. Propagation: None must reach the aggregator as NaN and be excluded
# ----------------------------------------------------------------------


def test_none_extracts_as_nan_and_is_excluded() -> None:
    """An undetermined seed shrinks N rather than dragging the mean down."""
    from experiments.models.analyzer.aggregation import METRIC_EXTRACTORS

    class _Reg:
        solution_recovered: bool | None = None
        jaccard_index: float | None = None

    class _RL:
        regression = _Reg()

    rl: Any = _RL()
    assert np.isnan(METRIC_EXTRACTORS["solution_recovered"](rl))
    assert np.isnan(METRIC_EXTRACTORS["jaccard_index"](rl))

    # nanmean over [1.0, 1.0, nan] is 1.0, not 0.667 -- the undetermined seed
    # is excluded from the denominator rather than counted as a failure.
    assert float(np.nanmean(np.array([1.0, 1.0, np.nan]))) == pytest.approx(1.0)
