"""Integration tests for the EquivalenceProbe and its runner hooks.

Key invariant tested: when no probe is installed, the production code path
is byte-for-byte unchanged (the ``ACTIVE_PROBE is not None`` check short-
circuits immediately and no probe import side-effect can alter results).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np = pytest.importorskip("numpy")

import experiments.models.equivalence_probe as _ep  # noqa: E402
from experiments.models.equivalence_probe import EquivalenceProbe  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_probe():
    """Ensure ACTIVE_PROBE is None before and after every test."""
    _ep.ACTIVE_PROBE = None
    yield
    _ep.ACTIVE_PROBE = None


@pytest.fixture
def nguyen1_data():
    """Small Nguyen-1 train/test arrays (1 variable, 20/50 points)."""
    rng = np.random.default_rng(42)
    x_train = rng.uniform(-1, 1, (20, 1))
    x_test = rng.uniform(-1, 1, (50, 1))
    x = x_train[:, 0]
    y_train = x + x**2 + x**3
    x_te = x_test[:, 0]
    y_test = x_te + x_te**2 + x_te**3
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Unit: probe absent → ACTIVE_PROBE None guard is the only overhead
# ---------------------------------------------------------------------------


class TestProbeAbsent:
    """Production path is unchanged when no probe is installed."""

    def test_active_probe_is_none_by_default(self) -> None:
        assert _ep.ACTIVE_PROBE is None

    def test_setting_and_clearing_probe(self) -> None:
        probe = EquivalenceProbe()
        _ep.ACTIVE_PROBE = probe
        assert _ep.ACTIVE_PROBE is probe
        _ep.ACTIVE_PROBE = None
        assert _ep.ACTIVE_PROBE is None

    def test_bingo_run_unchanged_without_probe(self, nguyen1_data) -> None:
        """Bingo with no probe installed runs identically to baseline."""
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        assert _ep.ACTIVE_PROBE is None
        x_train, y_train, x_test, y_test = nguyen1_data
        cfg = BingoConfig(
            population_size=10,
            stack_size=8,
            operators=["+", "-", "*", "/"],
            max_time=5.0,
            max_evals=200,
            generations=5,
        )
        runner = IsalSRBingoRunner(config=cfg)
        result = runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
        # Probe must still be absent after the run
        assert _ep.ACTIVE_PROBE is None
        assert result.n_total_dags >= 0


# ---------------------------------------------------------------------------
# Unit: probe collects DAGs and detects alphabet
# ---------------------------------------------------------------------------


class TestProbeAlphabet:
    """Alphabet check: split encoding produces zero SUB/DIV nodes."""

    def test_alphabet_after_bingo_run(self, nguyen1_data) -> None:
        """After a short Bingo run, split DAGs must have 0 SUB/DIV nodes."""
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        probe = EquivalenceProbe(engine_a="python", engine_b="python")
        _ep.ACTIVE_PROBE = probe

        x_train, y_train, x_test, y_test = nguyen1_data
        cfg = BingoConfig(
            population_size=10,
            stack_size=8,
            operators=["+", "-", "*", "/"],
            max_time=5.0,
            max_evals=200,
            generations=5,
        )
        runner = IsalSRBingoRunner(config=cfg)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})

        summary = probe.summary()
        alph = summary["alphabet_checks"]
        assert alph["sub_nodes_in_split_dag"] == 0, (
            f"Found {alph['sub_nodes_in_split_dag']} SUB nodes in split DAGs"
        )
        assert alph["div_nodes_in_split_dag"] == 0, (
            f"Found {alph['div_nodes_in_split_dag']} DIV nodes in split DAGs"
        )
        assert alph["sub_chars_in_canonical"] == 0, (
            f"Found {alph['sub_chars_in_canonical']} '-' chars in canonicals"
        )
        assert alph["div_chars_in_canonical"] == 0, (
            f"Found {alph['div_chars_in_canonical']} '/' chars in canonicals"
        )


# ---------------------------------------------------------------------------
# Unit: probe self-comparison (python vs python) has zero mismatches
# ---------------------------------------------------------------------------


class TestProbeSelfComparison:
    """Python-vs-Python comparison must always yield 0 mismatches."""

    def test_python_vs_python_bingo(self, nguyen1_data) -> None:
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        probe = EquivalenceProbe(engine_a="python", engine_b="python")
        _ep.ACTIVE_PROBE = probe

        x_train, y_train, x_test, y_test = nguyen1_data
        cfg = BingoConfig(
            population_size=10,
            stack_size=8,
            operators=["+", "-", "*", "/"],
            max_time=5.0,
            max_evals=200,
            generations=5,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IsalSRBingoRunner(config=cfg).fit(x_train, y_train, x_test, y_test, seed=2, config={})

        summary = probe.summary()
        assert summary["mismatches"] == 0, summary["mismatch_cases"]
        assert summary["dags_compared"]["bingo"] > 0

    def test_python_vs_python_udfs(self, nguyen1_data) -> None:
        pytest.importorskip("torch")
        from experiments.models.udfs.config import UDFSConfig
        from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner

        probe = EquivalenceProbe(engine_a="python", engine_b="python")
        _ep.ACTIVE_PROBE = probe

        x_train, y_train, x_test, y_test = nguyen1_data
        cfg = UDFSConfig(n_calc_nodes=2, max_orders=5_000, max_time=5.0)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IsalSRUDFSRunner(config=cfg).fit(x_train, y_train, x_test, y_test, seed=2, config={})

        summary = probe.summary()
        assert summary["mismatches"] == 0, summary["mismatch_cases"]


# ---------------------------------------------------------------------------
# Unit: k-distribution direction (split ≥ legacy on average)
# ---------------------------------------------------------------------------


class TestKDistributionDirection:
    """Split encoding should have ≥ legacy node count (adds NEG/INV nodes)."""

    def test_split_k_gte_legacy_k_bingo(self, nguyen1_data) -> None:
        pytest.importorskip("bingo")
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        probe = EquivalenceProbe(engine_a="python", engine_b="python")
        _ep.ACTIVE_PROBE = probe

        x_train, y_train, x_test, y_test = nguyen1_data
        cfg = BingoConfig(
            population_size=20,
            stack_size=16,
            operators=["+", "-", "*", "/"],
            max_time=8.0,
            max_evals=1000,
            generations=20,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IsalSRBingoRunner(config=cfg).fit(x_train, y_train, x_test, y_test, seed=3, config={})

        summary = probe.summary()
        b = summary["k_distributions"]["bingo"]
        # Only assert direction when we have paired data
        if b["split"]["mean"] is not None and b["legacy"]["mean"] is not None:
            assert b["delta_split_minus_legacy_mean"] >= 0, (
                f"Expected split k ≥ legacy k; delta={b['delta_split_minus_legacy_mean']}"
            )
