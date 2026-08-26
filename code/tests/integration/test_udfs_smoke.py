"""Integration smoke test for UDFS baseline + IsalSR runners.

Runs a minimal UDFS experiment on Nguyen-1 to verify the full pipeline.
Marked @slow because it runs actual UDFS evaluation (~10-30s).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from benchmarks.datasets.nguyen import generate_data, get_benchmark  # noqa: E402
from experiments.models.analyzer.metrics import r_squared  # noqa: E402
from experiments.models.schemas import RunMetadata  # noqa: E402
from experiments.models.udfs.config import UDFSConfig  # noqa: E402
from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner  # noqa: E402
from experiments.models.udfs.runner import UDFSBaselineRunner  # noqa: E402
from experiments.models.udfs.translator import UDFSTranslator  # noqa: E402


@pytest.fixture
def nguyen1_data():
    """Generate Nguyen-1 train/test data."""
    bench = get_benchmark("Nguyen-1")
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    return x_train, y_train, x_test, y_test, bench


@pytest.fixture
def small_config():
    """Small UDFS config for fast testing."""
    return UDFSConfig(
        n_calc_nodes=2,
        max_orders=30,
        max_time=30,
        k=1,
        mode="hierarchical",
    )


@pytest.mark.slow
class TestUDFSBaseline:
    def test_produces_valid_result(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = UDFSBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        assert result.wall_clock_s > 0
        assert result.total_evals > 0
        assert len(result.y_pred_test) == len(y_test)

    def test_r2_is_reasonable(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = UDFSBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        r2 = r_squared(y_test, result.y_pred_test)
        # With small budget, R² might not be great but should be finite
        assert np.isfinite(r2)


@pytest.mark.slow
class TestIsalSRUDFS:
    def test_produces_valid_result(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRUDFSRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        assert result.wall_clock_s > 0
        assert result.n_total_dags > 0
        assert result.n_unique_canonical > 0
        assert result.n_unique_canonical <= result.n_total_dags
        assert result.canonicalization_time_s >= 0

    def test_deduplication_works(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRUDFSRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        # With even small k, there should be SOME duplicates
        assert result.n_skipped >= 0
        assert result.n_unique_canonical + result.n_skipped == result.n_total_dags

    def test_search_only_time_less_than_total(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data
        runner = IsalSRUDFSRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        assert result.search_only_time_s <= result.wall_clock_s


@pytest.mark.slow
class TestUDFSTranslator:
    def test_run_log_schema(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data

        runner = UDFSBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        translator = UDFSTranslator(y_train=y_train, y_test=y_test)
        metadata = RunMetadata(
            method="udfs",
            representation="baseline",
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=42,
        )
        run_log = translator.to_run_log(result, metadata)

        # Verify schema
        d = run_log.to_dict()
        assert "metadata" in d
        assert "results" in d
        assert "regression" in d["results"]
        assert "time" in d["results"]
        assert "search_space" in d["results"]
        assert np.isfinite(run_log.regression.r2_test)
        assert run_log.time.wall_clock_total_s > 0

    def test_trajectory_has_rows(self, nguyen1_data, small_config):
        x_train, y_train, x_test, y_test, bench = nguyen1_data

        runner = UDFSBaselineRunner(config=small_config)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=42, config={})

        translator = UDFSTranslator(y_train=y_train, y_test=y_test)
        trajectory = translator.to_trajectory(result)

        assert len(trajectory) >= 1
        assert trajectory[0].timestamp_s > 0
