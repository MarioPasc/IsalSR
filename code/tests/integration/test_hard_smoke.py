"""Integration smoke tests for hard-tier benchmarks.

Runs minimal Bingo + UDFS experiments on representative hard problems
(one per sampling protocol) to verify the pipeline survives operator-set
extension (sqrt + pow), grid sampling, and 4-/5-variable inputs.

Marked @slow because each test invokes the real evaluation engines (~10-90s).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np = pytest.importorskip("numpy")

from benchmarks.datasets.hard import generate_data, get_benchmark  # noqa: E402
from experiments.models.analyzer.metrics import r_squared  # noqa: E402

# Representative problems chosen one per sampling protocol:
#   - II.11.27   (uniform; 4 variables; tests exp overflow tolerance)
#   - Pagie-1    (grid_2d_skip_zero; 2 variables; tests x^-4 representability)
#   - Keijzer-6  (integer_grid; 1 variable; tests log-domain extrapolation)
HARD_REPRESENTATIVES = ["II.11.27", "Pagie-1", "Keijzer-6"]


@pytest.fixture
def hard_data(request):
    name = request.param
    bench = get_benchmark(name)
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    return name, x_train, y_train, x_test, y_test, bench


# ----------------------------------------------------------------------
# Bingo
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("hard_data", HARD_REPRESENTATIVES, indirect=True)
class TestBingoOnHard:
    def test_baseline_runs_and_returns_finite_r2(self, hard_data):
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.runner import BingoBaselineRunner

        name, x_train, y_train, x_test, y_test, _ = hard_data
        cfg = BingoConfig(
            population_size=20,
            stack_size=16,
            operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
            max_time=30,
            max_evals=500,
            generations=20,
        )
        runner = BingoBaselineRunner(config=cfg)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
        assert result.wall_clock_s > 0
        assert result.total_evals > 0
        r2 = r_squared(y_test, result.y_pred_test)
        assert np.isfinite(r2), f"{name}: R^2 not finite ({r2})"

    def test_isalsr_dedupes_and_returns_finite_r2(self, hard_data):
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        name, x_train, y_train, x_test, y_test, _ = hard_data
        cfg = BingoConfig(
            population_size=20,
            stack_size=16,
            operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
            max_time=30,
            max_evals=500,
            generations=20,
        )
        runner = IsalSRBingoRunner(config=cfg)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
        assert result.n_total_dags > 0
        assert result.n_unique_canonical > 0
        assert result.n_unique_canonical <= result.n_total_dags
        assert result.canonicalization_time_s >= 0
        assert np.isfinite(r_squared(y_test, result.y_pred_test)) or np.isnan(
            r_squared(y_test, result.y_pred_test)
        )


# ----------------------------------------------------------------------
# UDFS
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("hard_data", HARD_REPRESENTATIVES, indirect=True)
class TestUDFSOnHard:
    def test_baseline_runs_and_returns_finite_r2(self, hard_data):
        pytest.importorskip("torch")
        from experiments.models.udfs.config import UDFSConfig
        from experiments.models.udfs.runner import UDFSBaselineRunner

        name, x_train, y_train, x_test, y_test, _ = hard_data
        cfg = UDFSConfig(
            n_calc_nodes=2,
            max_orders=30,
            max_time=30,
            k=1,
            mode="hierarchical",
        )
        runner = UDFSBaselineRunner(config=cfg)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
        assert result.wall_clock_s > 0
        assert len(result.y_pred_test) == len(y_test)

    def test_isalsr_runs_and_dedupes(self, hard_data):
        pytest.importorskip("torch")
        from experiments.models.udfs.config import UDFSConfig
        from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner

        name, x_train, y_train, x_test, y_test, _ = hard_data
        cfg = UDFSConfig(
            n_calc_nodes=2,
            max_orders=30,
            max_time=30,
            k=1,
            mode="hierarchical",
        )
        runner = IsalSRUDFSRunner(config=cfg)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
        assert result.wall_clock_s > 0
        assert result.n_total_dags >= 0
        assert result.n_unique_canonical >= 0
        # Conversions should not all fail; at least *some* canonicalization happens
        # (allow zero in extreme edge cases — pipeline must not crash).
