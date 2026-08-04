"""Regression tests for C1.10 -- ``trajectory.csv`` must be one quantity.

Stage C (2026-08-04) failed C1.10 on 452 of 1,260 cells. A scan of all 1,260
trajectories showed **100 % of the violations sat at the final row and nowhere
else**, on both hosts (UDFS 241, Bingo 218): the intermediate rows carried
training R-squared while the final row carried *test* R-squared, so ``best_r2``
was two different quantities in one series and decreased whenever
``r2_test < r2_train`` -- which is most of the time.

That is the same defect class as the ``n_dags_explored`` mix-up fixed on
2026-08-03: intermediate rows measuring one population and the final row
another. These tests pin the invariant for both hosts so it cannot come back.

Test metrics are not lost by the fix; they remain authoritative in
``run_log.json``'s ``results.regression``, which is what every analyzer reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.models.schemas import TrajectoryRow


def _monotone_non_decreasing(values: list[float]) -> bool:
    return all(b >= a - 1e-12 for a, b in zip(values, values[1:], strict=False))


class TestTrajectoryIsASingleQuantity:
    """``best_r2`` must be the same measurement in every row of the series."""

    def test_final_row_uses_train_not_test_bingo(self) -> None:
        """Bingo's final trajectory row reports the train R2 of the best model.

        Built directly from the translator so the test fails against the
        pre-fix code, where the last row carried ``r_squared(y_test, ...)``.
        """
        translator = _make_bingo_translator()
        rows = translator.to_trajectory(_bingo_raw_result())
        r2 = [row.best_r2 for row in rows]

        assert _monotone_non_decreasing(r2), (
            f"best_r2 must be monotone non-decreasing; got {r2}. "
            "A drop at the last row means the final row switched to test R2."
        )
        # The regression bites specifically when test R2 is worse than train.
        assert rows[-1].best_r2 == pytest.approx(_TRAIN_R2, abs=1e-9)
        assert rows[-1].best_r2 != pytest.approx(_TEST_R2, abs=1e-9)

    def test_final_row_uses_train_not_test_udfs(self) -> None:
        """UDFS's final trajectory row reports the train R2 of the best model."""
        translator = _make_udfs_translator()
        rows = translator.to_trajectory(_udfs_raw_result())
        r2 = [row.best_r2 for row in rows]

        assert _monotone_non_decreasing(r2), f"best_r2 must be monotone non-decreasing; got {r2}."
        assert rows[-1].best_r2 == pytest.approx(_TRAIN_R2, abs=1e-9)
        assert rows[-1].best_r2 != pytest.approx(_TEST_R2, abs=1e-9)

    @pytest.mark.parametrize("host", ["bingo", "udfs"])
    def test_series_is_monotone_under_a_worse_test_split(self, host: str) -> None:
        """The invariant holds precisely where the old code broke it.

        The failure mode needed ``r2_test < r2_train``; a fixture where the two
        coincide would pass against the buggy code too and certify nothing.
        """
        assert _TEST_R2 < _TRAIN_R2, "fixture must exercise the failing direction"
        translator = _make_bingo_translator() if host == "bingo" else _make_udfs_translator()
        raw = _bingo_raw_result() if host == "bingo" else _udfs_raw_result()
        rows = translator.to_trajectory(raw)
        assert _monotone_non_decreasing([row.best_r2 for row in rows])

    def test_trajectory_row_columns_unchanged(self) -> None:
        """The fix must not alter the CSV header -- C1 artefacts still parse."""
        row = TrajectoryRow(
            timestamp_s=0.0,
            iteration=0,
            best_r2=0.0,
            best_nrmse=0.0,
            n_dags_explored=0,
            n_unique_canonical=0,
            current_expr="",
            current_complexity=0,
            cache_hit_rate_cumulative=0.0,
        )
        assert row.COLUMNS[:4] == [
            "timestamp_s",
            "iteration",
            "best_r2",
            "best_nrmse",
        ]


# --------------------------------------------------------------------------
# Fixtures.  Deliberately constructed so that r2_test < r2_train, which is the
# only regime in which the pre-fix code produced a non-monotone series.
# --------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_Y_TRAIN = np.linspace(1.0, 10.0, 64)
_Y_TEST = np.linspace(1.0, 10.0, 32)
# Train predictions are near-perfect; test predictions are visibly worse.
_YP_TRAIN = _Y_TRAIN + 0.01 * _RNG.standard_normal(_Y_TRAIN.size)
_YP_TEST = _Y_TEST + 0.50 * _RNG.standard_normal(_Y_TEST.size)


def _r2(y: np.ndarray, yp: np.ndarray) -> float:
    return float(1.0 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2))


_TRAIN_R2 = _r2(_Y_TRAIN, _YP_TRAIN)
_TEST_R2 = _r2(_Y_TEST, _YP_TEST)


def _make_bingo_translator():  # type: ignore[no-untyped-def]
    from experiments.models.bingo.translator import BingoTranslator

    translator = BingoTranslator.__new__(BingoTranslator)
    translator._y_train = _Y_TRAIN  # type: ignore[attr-defined]
    translator._y_test = _Y_TEST  # type: ignore[attr-defined]
    return translator


def _make_udfs_translator():  # type: ignore[no-untyped-def]
    from experiments.models.udfs.translator import UDFSTranslator

    translator = UDFSTranslator.__new__(UDFSTranslator)
    translator._y_train = _Y_TRAIN  # type: ignore[attr-defined]
    translator._y_test = _Y_TEST  # type: ignore[attr-defined]
    return translator


def _bingo_raw_result():  # type: ignore[no-untyped-def]
    from experiments.models.bingo.runner import BingoRawResult, BingoTrajectorySnapshot

    var_y = float(np.var(_Y_TRAIN))
    # Two snapshots climbing toward, but not exceeding, the exact train R2.
    snaps = [
        BingoTrajectorySnapshot(
            timestamp_s=1.0,
            generation=10,
            best_fitness=(1.0 - 0.90) * var_y,
            n_evals=100,
            n_total_dags=100,
            n_unique_canonical=80,
            n_skipped=20,
        ),
        BingoTrajectorySnapshot(
            timestamp_s=2.0,
            generation=20,
            # Must climb ABOVE the test R2 (~0.96), or the pre-fix final row
            # would still look monotone and the test would certify nothing.
            best_fitness=(1.0 - 0.99) * var_y,
            n_evals=200,
            n_total_dags=200,
            n_unique_canonical=150,
            n_skipped=50,
        ),
    ]
    raw = BingoRawResult.__new__(BingoRawResult)
    raw.trajectory_snapshots = snaps  # type: ignore[attr-defined]
    raw.y_pred_train = _YP_TRAIN  # type: ignore[attr-defined]
    raw.y_pred_test = _YP_TEST  # type: ignore[attr-defined]
    raw.wall_clock_s = 3.0  # type: ignore[attr-defined]
    raw.n_generations = 30  # type: ignore[attr-defined]
    raw.n_total_dags = 300  # type: ignore[attr-defined]
    raw.n_unique_canonical = 200  # type: ignore[attr-defined]
    raw.n_skipped = 100  # type: ignore[attr-defined]
    raw.best_agraph = None  # type: ignore[attr-defined]
    raw.best_sympy = None  # type: ignore[attr-defined]
    return raw


def _udfs_raw_result():  # type: ignore[no-untyped-def]
    from experiments.models.udfs.runner import TrajectorySnapshot, UDFSRawResult

    var_y = float(np.var(_Y_TRAIN))
    snaps = [
        TrajectorySnapshot(
            timestamp_s=1.0,
            total_evals=100,
            best_loss=(1.0 - 0.90) * var_y,
        ),
        TrajectorySnapshot(
            timestamp_s=2.0,
            total_evals=200,
            # Must climb ABOVE the test R2 (~0.96) -- see the Bingo fixture.
            best_loss=(1.0 - 0.99) * var_y,
        ),
    ]
    raw = UDFSRawResult.__new__(UDFSRawResult)
    raw.trajectory_snapshots = snaps  # type: ignore[attr-defined]
    raw.y_pred_train = _YP_TRAIN  # type: ignore[attr-defined]
    raw.y_pred_test = _YP_TEST  # type: ignore[attr-defined]
    raw.wall_clock_s = 3.0  # type: ignore[attr-defined]
    raw.total_evals = 300  # type: ignore[attr-defined]
    raw.n_total_dags = 300  # type: ignore[attr-defined]
    raw.n_unique_canonical = 200  # type: ignore[attr-defined]
    raw.n_skipped = 100  # type: ignore[attr-defined]
    raw.best_sympy = None  # type: ignore[attr-defined]
    return raw
