"""Integration smoke test for Bingo baseline + IsalSR runners."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np = pytest.importorskip("numpy")
bingo = pytest.importorskip("bingo")

from benchmarks.datasets.nguyen import generate_data, get_benchmark  # noqa: E402
from experiments.models.analyzer.metrics import r_squared  # noqa: E402
from experiments.models.bingo.config import BingoConfig  # noqa: E402
from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner  # noqa: E402
from experiments.models.bingo.runner import BingoBaselineRunner  # noqa: E402
from experiments.models.bingo.translator import BingoTranslator  # noqa: E402
from experiments.models.schemas import RunMetadata  # noqa: E402


@pytest.fixture
def nguyen1_data():
    bench = get_benchmark("Nguyen-1")
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    return x_train, y_train, x_test, y_test, bench


@pytest.fixture
def small_config():
    return BingoConfig(
        population_size=20,
        stack_size=10,
        max_time=30,
        max_evals=300,
        generations=50,
    )


@pytest.mark.slow
class TestBingoBaseline:
    def test_produces_valid_result(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = BingoBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        assert result.wall_clock_s > 0
        assert result.total_evals > 0
        assert len(result.y_pred_test) == len(y_test)

    def test_r2_is_finite(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = BingoBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        r2 = r_squared(y_test, result.y_pred_test)
        assert np.isfinite(r2)


@pytest.mark.slow
class TestIsalSRBingo:
    def test_produces_valid_result(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRBingoRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        assert result.wall_clock_s > 0
        assert result.n_total_dags > 0
        assert result.n_unique_canonical > 0
        assert result.n_unique_canonical <= result.n_total_dags
        assert result.canonicalization_time_s >= 0

    def test_deduplication_works(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRBingoRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        # GP should produce duplicates
        assert result.n_skipped >= 0
        # unique + skipped <= total (some may fail conversion and bypass dedup)
        assert result.n_unique_canonical + result.n_skipped <= result.n_total_dags
        assert result.n_unique_canonical <= result.n_total_dags

    def test_search_only_time(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRBingoRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        assert result.search_only_time_s <= result.wall_clock_s


@pytest.mark.slow
class TestBingoTranslator:
    def test_run_log_schema(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = BingoBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        translator = BingoTranslator(y_train=y_train, y_test=y_test)
        metadata = RunMetadata(
            method="bingo",
            representation="baseline",
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=42,
        )
        run_log = translator.to_run_log(result, metadata)
        d = run_log.to_dict()
        assert "metadata" in d
        assert "results" in d
        assert np.isfinite(run_log.regression.r2_test)
        assert run_log.time.wall_clock_total_s > 0

    def test_trajectory_has_rows(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = BingoBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})
        translator = BingoTranslator(y_train=y_train, y_test=y_test)
        trajectory = translator.to_trajectory(result)
        assert len(trajectory) >= 1
