"""Runtime scoring policy for expressions undefined on part of the evaluation set.

T08 / reviewer comment R2.7. In the submitted campaign two Bingo-IsalSR runs
recovered expressions that are well-defined on the training domain and undefined
on part of the *test* domain:

  * Vlad-2 seed 23 -- ``log(-0.443*x_0 - 3.140*(...)*exp(-x_0) + 1.448)`` on an
    extrapolation grid, where the argument goes negative. ``r2_train = 0.9973``,
    ``r2_test = NaN``.
  * Korns-12 seed 30 -- ``exp(cos(...)**2 / sin(x_4))``, which overflows to
    ``+inf`` as ``sin(x_4) -> 0``. ``r2_train = 0.0237``, ``r2_test = NaN``.

Host protected operators guard division by zero but neither ``log`` of a
negative argument nor ``exp`` overflow.

Policy: such a model is **unusable on that evaluation set** and is scored as
such (``R^2 = 0``, ``NRMSE = 1``, ``MSE = Var[y]``), with the number of
non-finite predictions recorded separately so the failure stays countable. It is
*not* scored on the subset where it happens to be defined.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.models.analyzer.metrics import (
    count_nonfinite_predictions,
    mse,
    nrmse,
    r_squared,
)

# ----------------------------------------------------------------------
# The counter
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preds", "expected"),
    [
        ([1.0, 2.0, 3.0], 0),
        ([1.0, float("nan"), 3.0], 1),
        ([float("inf"), 2.0, float("-inf")], 2),
        ([float("nan")] * 4, 4),
    ],
)
def test_count_nonfinite_predictions(preds: list[float], expected: int) -> None:
    assert count_nonfinite_predictions(np.array(preds)) == expected


# ----------------------------------------------------------------------
# No metric may return NaN
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_no_metric_returns_nan_when_a_prediction_is_bad(bad: float) -> None:
    """A single undefined test point must not make any metric NaN."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, bad, 4.0, 5.0])

    for name, val in [
        ("r2", r_squared(y_true, y_pred)),
        ("nrmse", nrmse(y_true, y_pred)),
        ("mse", mse(y_true, y_pred)),
    ]:
        assert math.isfinite(val), f"{name} returned a non-finite value"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_unusable_model_scores_as_no_better_than_the_mean(bad: float) -> None:
    """R^2 = 0, NRMSE = 1, MSE = Var[y] -- the 'predict the mean' baseline."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, bad, 4.0, 5.0])

    assert r_squared(y_true, y_pred) == 0.0
    assert nrmse(y_true, y_pred) == pytest.approx(1.0)
    assert mse(y_true, y_pred) == pytest.approx(float(np.var(y_true)))


def test_scoring_is_not_restricted_to_the_finite_subset() -> None:
    """A model must not be rewarded for being undefined where it fails.

    Here the prediction is perfect on every point except one, where it is NaN.
    Scoring the finite subset would give R^2 = 1.0 -- a better score than an
    honest model that is defined everywhere but slightly wrong.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    near_perfect_but_undefined_once = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
    honest_but_imperfect = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    r2_undefined = r_squared(y_true, near_perfect_but_undefined_once)
    r2_honest = r_squared(y_true, honest_but_imperfect)

    assert r2_undefined == 0.0
    assert r2_honest > r2_undefined, "an undefined model outscored a usable one"


def test_finite_predictions_are_scored_exactly_as_before() -> None:
    """The guard must be inert on well-behaved input."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 1, 200)
    y_pred = y_true + rng.normal(0, 0.1, 200)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    assert r_squared(y_true, y_pred) == pytest.approx(1.0 - ss_res / ss_tot)
    assert nrmse(y_true, y_pred) == pytest.approx(
        float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / float(np.std(y_true))
    )
    assert mse(y_true, y_pred) == pytest.approx(float(np.mean((y_true - y_pred) ** 2)))


def test_perfect_prediction_still_scores_one() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y.copy()) == pytest.approx(1.0)
    assert nrmse(y, y.copy()) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# The two real cases from the submitted campaign
# ----------------------------------------------------------------------


def test_vlad2_extrapolation_signature() -> None:
    """log() of a negative argument on the extrapolated part of the grid.

    Good fit on the training range, undefined beyond it -- Vlad-2's test grid
    extends past its training range by construction.
    """
    x_train = np.linspace(1.0, 5.0, 50)
    x_test = np.linspace(-1.0, 5.0, 60)  # extends below the training range

    def model(x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(x)

    y_train_true, y_test_true = np.log(np.abs(x_train) + 1), np.log(np.abs(x_test) + 1)

    assert count_nonfinite_predictions(model(x_train)) == 0
    assert count_nonfinite_predictions(model(x_test)) > 0

    # Training score is unaffected; test score degrades to the mean baseline.
    assert math.isfinite(r_squared(y_train_true, model(x_train)))
    assert r_squared(y_test_true, model(x_test)) == 0.0


def test_korns12_exp_overflow_signature() -> None:
    """exp(k / sin(x)) overflows to +inf as sin(x) -> 0."""
    x = np.linspace(0.01, np.pi - 0.01, 40)
    x = np.append(x, np.pi)  # sin(pi) == 0 numerically -> division blows up

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        preds = np.exp(1.0 / np.sin(x))

    y_true = np.linspace(0, 1, len(x))
    assert count_nonfinite_predictions(preds) > 0
    assert r_squared(y_true, preds) == 0.0
    assert math.isfinite(nrmse(y_true, preds))
