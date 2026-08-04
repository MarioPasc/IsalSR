"""Cost attribution of the dedup wrapper's untimed per-candidate work.

The dedup arms (``isalsr``, ``hash``) run two blocks of per-candidate work that
happen inside the fixed wall-clock budget but were outside every timer, so the
run log booked them as *search* time:

1. the adapter conversion host object -> ``LabeledDAG``, which is genuine method
   cost (the representation cannot be keyed without it), and
2. the T04 shadow cardinality sketches, which are pure audit instrumentation.

These tests pin the two new totals, their propagation through the raw result and
the translator, and the two redefined derived quantities::

    wall_clock_search_only_s = max(0, wall - canon - conversion - shadow)
    overhead_time_s          = canon + conversion

Both hosts run live: a UDFS micro-run costs ~1.1 s and a Bingo micro-run ~0.5 s
on the development workstation, which is inside the unit-test budget.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from experiments.models.schemas import RunLog, RunMetadata

# Budgets for the micro-runs.  Small enough to keep the suite fast, large enough
# that the search evaluates a few hundred candidates (so the accumulators are
# fed) on every host.
_MAX_TIME_S = 5.0
_UDFS_MAX_ORDERS = 2_000
_BINGO_POP = 10
_BINGO_GENS = 5
_BINGO_MAX_EVALS = 200

_TOL = 1e-6


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a 1-variable quadratic sample: ``y = x^2 + x``."""
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-1.0, 1.0, size=(40, 1))
    x_test = rng.uniform(-1.0, 1.0, size=(20, 1))
    y_train = (x_train[:, 0] ** 2 + x_train[:, 0]).astype(np.float64)
    y_test = (x_test[:, 0] ** 2 + x_test[:, 0]).astype(np.float64)
    return x_train, y_train, x_test, y_test


def _metadata(method: str, representation: str) -> RunMetadata:
    return RunMetadata(
        method=method,
        representation=representation,
        benchmark="unit",
        problem="Quadratic-1",
        seed=1,
    )


def _udfs_config() -> Any:
    from experiments.models.udfs.config import UDFSConfig

    return UDFSConfig(n_calc_nodes=2, max_orders=_UDFS_MAX_ORDERS, max_time=_MAX_TIME_S)


def _bingo_config() -> Any:
    from experiments.models.bingo.config import BingoConfig

    return BingoConfig(
        population_size=_BINGO_POP,
        stack_size=8,
        operators=["+", "-", "*", "/"],
        max_time=_MAX_TIME_S,
        max_evals=_BINGO_MAX_EVALS,
        generations=_BINGO_GENS,
    )


def _run(runner: Any, data: Any, config: dict[str, Any]) -> Any:
    x_train, y_train, x_test, y_test = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.fit(x_train, y_train, x_test, y_test, seed=1, config=config)


def _udfs_runner(variant: str) -> Any:
    from experiments.models.udfs.isalsr_runner import HashUDFSRunner, IsalSRUDFSRunner
    from experiments.models.udfs.runner import UDFSBaselineRunner

    cfg = _udfs_config()
    if variant == "isalsr":
        return IsalSRUDFSRunner(config=cfg)
    if variant == "hash":
        return HashUDFSRunner(config=cfg)
    return UDFSBaselineRunner(config=cfg)


def _bingo_runner(variant: str) -> Any:
    from experiments.models.bingo.isalsr_runner import HashBingoRunner, IsalSRBingoRunner
    from experiments.models.bingo.runner import BingoBaselineRunner

    cfg = _bingo_config()
    if variant == "isalsr":
        return IsalSRBingoRunner(config=cfg)
    if variant == "hash":
        return HashBingoRunner(config=cfg)
    return BingoBaselineRunner(config=cfg)


def _translate(method: str, raw: Any, data: Any, representation: str) -> RunLog:
    _x_train, y_train, _x_test, y_test = data
    if method == "udfs":
        from experiments.models.udfs.translator import UDFSTranslator

        translator: Any = UDFSTranslator(y_train, y_test)
    else:
        from experiments.models.bingo.translator import BingoTranslator

        translator = BingoTranslator(y_train, y_test)
    return translator.to_run_log(raw, _metadata(method, representation))


def _runner(method: str, variant: str) -> Any:
    return _udfs_runner(variant) if method == "udfs" else _bingo_runner(variant)


def _skip_if_missing(method: str) -> None:
    if method == "udfs":
        pytest.importorskip("torch")
    else:
        pytest.importorskip("bingo")


# ----------------------------------------------------------------------
# The two new accumulators, measured on a live run
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_isalsr_arm_times_conversion_and_shadow(method: str, tiny_data: Any) -> None:
    """The isalsr arm must charge itself for conversion and for the sketches."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "isalsr"), tiny_data, {})

    assert raw.n_total_dags > 0, "micro-run produced no candidates; budgets too small"
    assert raw.conversion_time_s > 0.0
    assert raw.shadow_time_s > 0.0


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_shadow_disabled_leaves_shadow_time_exactly_zero(method: str, tiny_data: Any) -> None:
    """``shadow_hash: false`` must leave the shadow accumulator untouched."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "isalsr"), tiny_data, {"shadow_hash": False})

    assert raw.n_total_dags > 0
    assert raw.shadow_time_s == 0.0
    # The conversion still happens: it is not instrumentation.
    assert raw.conversion_time_s > 0.0


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_hash_arm_times_conversion(method: str, tiny_data: Any) -> None:
    """The hash arm converts too, so it must carry the same conversion cost."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "hash"), tiny_data, {})

    assert raw.n_total_dags > 0
    assert raw.conversion_time_s > 0.0


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_baseline_arm_reports_zero_for_both(method: str, tiny_data: Any) -> None:
    """No wrapper, no wrapper cost."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "baseline"), tiny_data, {})

    assert raw.conversion_time_s == 0.0
    assert raw.shadow_time_s == 0.0


# ----------------------------------------------------------------------
# Derived quantities in the run log
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_run_log_search_only_subtracts_all_three(method: str, tiny_data: Any) -> None:
    """search_only = wall - canon - conversion - shadow, exactly."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "isalsr"), tiny_data, {})
    log = _translate(method, raw, tiny_data, "isalsr")
    t = log.time

    assert t.conversion_time_s > 0.0
    assert t.shadow_time_s > 0.0
    expected = (
        t.wall_clock_total_s - t.canonicalization_runtime_s - t.conversion_time_s - t.shadow_time_s
    )
    np.testing.assert_allclose(t.wall_clock_search_only_s, expected, atol=_TOL, rtol=0.0)


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_run_log_overhead_is_canon_plus_conversion(method: str, tiny_data: Any) -> None:
    """Overhead is the representation layer's cost; the sketches stay out of it."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "isalsr"), tiny_data, {})
    log = _translate(method, raw, tiny_data, "isalsr")
    t = log.time

    expected = t.canonicalization_runtime_s + t.conversion_time_s
    np.testing.assert_allclose(t.overhead_time_s, expected, atol=_TOL, rtol=0.0)
    # Instrumentation is reported, never charged.
    assert t.overhead_time_s < t.overhead_time_s + t.shadow_time_s


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_baseline_run_log_keeps_legacy_time_identities(method: str, tiny_data: Any) -> None:
    """Baseline logs must be numerically unchanged by this schema addition."""
    _skip_if_missing(method)
    raw = _run(_runner(method, "baseline"), tiny_data, {})
    log = _translate(method, raw, tiny_data, "baseline")
    t = log.time

    assert t.conversion_time_s == 0.0
    assert t.shadow_time_s == 0.0
    assert t.overhead_time_s == 0.0
    np.testing.assert_allclose(
        t.wall_clock_search_only_s, t.wall_clock_total_s, atol=_TOL, rtol=0.0
    )


def test_search_only_is_clamped_at_zero() -> None:
    """A pathological accounting overshoot must not produce negative search time."""
    from experiments.models.udfs.runner import UDFSRawResult
    from experiments.models.udfs.translator import UDFSTranslator

    y = np.array([1.0, 2.0, 3.0])
    raw = UDFSRawResult(
        wall_clock_s=1.0,
        seed=1,
        y_pred_train=np.array([1.0, 2.0, 3.0]),
        y_pred_test=np.array([1.0, 2.0, 3.0]),
        canonicalization_time_s=0.8,
        conversion_time_s=0.5,
        shadow_time_s=0.4,
        search_only_time_s=0.2,
    )
    log = UDFSTranslator(y, y).to_run_log(raw, _metadata("udfs", "isalsr"))
    assert log.time.wall_clock_search_only_s == 0.0


# ----------------------------------------------------------------------
# Schema: round-trip and legacy tolerance
# ----------------------------------------------------------------------


def _minimal_run_log_dict() -> dict[str, Any]:
    return {
        "metadata": {
            "method": "udfs",
            "representation": "isalsr",
            "benchmark": "unit",
            "problem": "Quadratic-1",
            "seed": 1,
            "hardware": {},
            "hyperparameters": {},
            "data_fingerprint": "",
            "config_sha256": "",
        },
        "results": {
            "regression": {
                "r2_train": 1.0,
                "r2_test": 1.0,
                "nrmse_train": 0.0,
                "nrmse_test": 0.0,
                "mse_test": 0.0,
                "solution_recovered": True,
                "jaccard_index": 1.0,
                "model_complexity": 3,
                "n_nonfinite_test_predictions": 0,
            },
            "time": {
                "wall_clock_total_s": 10.0,
                "wall_clock_search_only_s": 8.0,
                "canonicalization_precomputed_s": 0.0,
                "canonicalization_runtime_s": 2.0,
                "cache_hit_rate": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "estimated_time_saved_s": 0.0,
                "time_to_r2_099_s": None,
                "time_to_r2_0999_s": None,
                "evaluation_time_s": 8.0,
                "overhead_time_s": 2.0,
            },
            "search_space": {
                "total_dags_explored": 10,
                "unique_canonical_dags": 5,
                "empirical_reduction_factor": 2.0,
                "max_internal_nodes_seen": 3,
                "theoretical_reduction_bound": 6.0,
                "redundancy_rate": 0.5,
            },
        },
        "best_expression": {
            "symbolic_form": "x_0",
            "isalsr_string": "",
            "canonical_string": "",
            "n_nodes": 1,
            "n_edges": 0,
        },
    }


def test_legacy_run_log_without_new_keys_loads_with_zeros() -> None:
    """Artifacts written before this change must still deserialise."""
    log = RunLog.from_dict(_minimal_run_log_dict())
    assert log.time.conversion_time_s == 0.0
    assert log.time.shadow_time_s == 0.0
    assert log.search_space.penalised_in_population_mean == 0.0
    assert log.search_space.penalised_in_population_max == 0.0


def test_run_log_with_new_keys_round_trips() -> None:
    """A fresh artifact must survive to_dict -> from_dict unchanged."""
    d = _minimal_run_log_dict()
    d["results"]["time"]["conversion_time_s"] = 1.25
    d["results"]["time"]["shadow_time_s"] = 0.5
    d["results"]["search_space"]["penalised_in_population_mean"] = 3.5
    d["results"]["search_space"]["penalised_in_population_max"] = 9.0

    log = RunLog.from_dict(d)
    assert log.time.conversion_time_s == pytest.approx(1.25)
    assert log.time.shadow_time_s == pytest.approx(0.5)
    assert log.search_space.penalised_in_population_mean == pytest.approx(3.5)
    assert log.search_space.penalised_in_population_max == pytest.approx(9.0)

    again = RunLog.from_dict(log.to_dict())
    assert again.time.conversion_time_s == pytest.approx(1.25)
    assert again.time.shadow_time_s == pytest.approx(0.5)
    assert again.search_space.penalised_in_population_mean == pytest.approx(3.5)
    assert again.search_space.penalised_in_population_max == pytest.approx(9.0)


# ----------------------------------------------------------------------
# Bingo effective-population disclosure
# ----------------------------------------------------------------------


def test_bingo_isalsr_reports_penalised_population(tiny_data: Any) -> None:
    """``n_penalised_per_gen`` must reach the run log instead of being discarded."""
    pytest.importorskip("bingo")
    raw = _run(_bingo_runner("isalsr"), tiny_data, {})
    log = _translate("bingo", raw, tiny_data, "isalsr")

    assert raw.penalised_in_population_mean >= 0.0
    assert raw.penalised_in_population_max >= raw.penalised_in_population_mean
    assert log.search_space.penalised_in_population_mean == pytest.approx(
        raw.penalised_in_population_mean
    )
    assert log.search_space.penalised_in_population_max == pytest.approx(
        raw.penalised_in_population_max
    )


def test_udfs_leaves_penalised_population_at_zero(tiny_data: Any) -> None:
    """The disclosure does not apply to UDFS, which has no population penalty."""
    pytest.importorskip("torch")
    raw = _run(_udfs_runner("isalsr"), tiny_data, {})
    log = _translate("udfs", raw, tiny_data, "isalsr")

    assert log.search_space.penalised_in_population_mean == 0.0
    assert log.search_space.penalised_in_population_max == 0.0
