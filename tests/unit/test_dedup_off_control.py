"""Tests for the ``nodedup`` control arm (C3).

The ``nodedup`` arm keeps the whole IsalSR wrapper installed -- the UDFS
module-level ``evaluate_cgraph`` patch, Bingo's ``IsalSREvaluation`` subclass,
the adapter conversion, the canonicalisation and every counter -- and disables
exactly one thing: the *suppression* of a candidate whose canonical string has
already been seen. It exists to separate "the wrapper perturbs the search" from
"deduplication changes the search": with suppression off, every candidate the
baseline would evaluate must still reach the host's fitness evaluation.

The two load-bearing properties tested here are:

1. **Completeness.** With dedup off, the number of host evaluations equals the
   number of candidates offered (``n_total``), not the number of distinct
   canonical strings.
2. **Instrumentation is preserved.** ``n_total`` and ``n_unique`` still count,
   so the run still reports a reduction factor rho = n_total / n_unique > 1 on a
   stream that contains duplicates, while nothing was suppressed
   (``n_skipped == 0``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# ----------------------------------------------------------------------
# Candidate streams: 3 distinct expressions, each offered twice.
# ----------------------------------------------------------------------


def _udfs_stream() -> list[Any]:
    """Return 6 UDFS ``CompGraph`` objects spanning 3 canonical classes.

    Each of ``sin(x)``, ``sin(sin(x))`` and ``sin(sin(sin(x)))`` is built twice
    as an independent object, so the stream has 6 candidates and 3 distinct
    canonical strings.
    """
    from DAG_search.comp_graph import CompGraph

    stream: list[Any] = []
    for n_sin in (1, 2, 3):
        for _ in range(2):
            node_dict: dict[int, tuple[list[int], str]] = {0: ([], "inp")}
            for i in range(1, n_sin + 1):
                node_dict[i] = ([i - 1], "sin")
            stream.append(CompGraph(1, 1, 0, node_dict=node_dict))
    return stream


def _bingo_stream() -> list[Any]:
    """Return 6 Bingo ``AGraph`` objects spanning 3 canonical classes.

    Same construction as :func:`_udfs_stream`: each expression appears twice as
    an independent, unevaluated individual.
    """
    from bingo.symbolic_regression.agraph.agraph import AGraph

    stream: list[Any] = []
    for n_sin in (1, 2, 3):
        for _ in range(2):
            rows = [[0, 0, 0]] + [[6, i, i] for i in range(n_sin)]
            agraph = AGraph(use_simplification=False)
            agraph._command_array = np.array(rows, dtype=int)
            agraph._notify_modification()
            stream.append(agraph)
    return stream


# ----------------------------------------------------------------------
# UDFS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dedup_enabled", "expected_host_calls"),
    [(True, 3), (False, 6)],
)
def test_udfs_suppression_flag_controls_host_calls(
    dedup_enabled: bool, expected_host_calls: int
) -> None:
    """Dedup off routes every candidate to the host; dedup on routes 3 of 6."""
    pytest.importorskip("DAG_search.comp_graph")
    from experiments.models.udfs.isalsr_runner import _CanonicalDeduplicator

    calls: list[Any] = []

    def _original(cgraph, x, loss_fkt, opt_mode="grid_zoom", loss_thresh=None):  # noqa: ANN001,ANN202,N803
        calls.append(cgraph)
        return np.array([]), 1.0

    dedup = _CanonicalDeduplicator(dedup_enabled=dedup_enabled)
    wrapped = dedup.wrap_evaluate_cgraph(_original)
    for cgraph in _udfs_stream():
        wrapped(cgraph, np.zeros((2, 1)), None)

    assert len(calls) == expected_host_calls
    assert dedup.n_total == 6
    assert dedup.n_unique == 3
    assert dedup.n_skipped == (3 if dedup_enabled else 0)


def test_udfs_dedup_off_still_reports_reduction_factor() -> None:
    """rho = n_total / n_unique stays > 1 with suppression disabled."""
    pytest.importorskip("DAG_search.comp_graph")
    from experiments.models.udfs.isalsr_runner import _CanonicalDeduplicator

    def _original(cgraph, x, loss_fkt, opt_mode="grid_zoom", loss_thresh=None):  # noqa: ANN001,ANN202,N803
        return np.array([]), 1.0

    dedup = _CanonicalDeduplicator(dedup_enabled=False)
    wrapped = dedup.wrap_evaluate_cgraph(_original)
    for cgraph in _udfs_stream():
        wrapped(cgraph, np.zeros((2, 1)), None)

    np.testing.assert_allclose(dedup.n_total / dedup.n_unique, 2.0, rtol=0.0)


# ----------------------------------------------------------------------
# Bingo
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dedup_enabled", "expected_fitness_calls"),
    [(True, 3), (False, 6)],
)
def test_bingo_suppression_flag_controls_fitness_calls(
    dedup_enabled: bool, expected_fitness_calls: int
) -> None:
    """Dedup off evaluates every individual; dedup on evaluates 3 of 6."""
    pytest.importorskip("bingo")
    from experiments.models.bingo.isalsr_runner import (
        IsalSREvaluation,
        _CanonicalDeduplicator,
    )

    n_calls = 0

    def _fitness(indv: Any) -> float:
        nonlocal n_calls
        n_calls += 1
        return 1.0

    dedup = _CanonicalDeduplicator(dedup_enabled=dedup_enabled)
    evaluation = IsalSREvaluation(_fitness, dedup=dedup)
    population = _bingo_stream()
    evaluation._serial_eval(population)

    assert n_calls == expected_fitness_calls
    assert dedup.n_total == 6
    assert dedup.n_unique == 3
    assert dedup.n_skipped == (3 if dedup_enabled else 0)
    if not dedup_enabled:
        assert all(np.isfinite(indv.fitness) for indv in population)
        assert all(indv.genetic_age == 0 for indv in population)


def test_bingo_dedup_off_overrides_population_enforcement() -> None:
    """``enforce_dedup=True`` is inert when the deduplicator is disabled.

    Population-level enforcement rejects duplicates *and* answers repeats from
    ``fitness_cache``, both of which are suppression. The control arm must do
    neither, so the flag has to be neutralised rather than merely bypassed.
    """
    pytest.importorskip("bingo")
    from experiments.models.bingo.isalsr_runner import (
        IsalSREvaluation,
        _CanonicalDeduplicator,
    )

    dedup = _CanonicalDeduplicator(dedup_enabled=False)
    evaluation = IsalSREvaluation(lambda indv: 1.0, dedup=dedup, enforce_dedup=True)

    assert evaluation._enforce_dedup is False


# ----------------------------------------------------------------------
# Orchestrator factory
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_create_runner_nodedup_arm(method: str) -> None:
    """``--variants nodedup`` yields the IsalSR runner with suppression off."""
    pytest.importorskip(f"experiments.models.{method}.isalsr_runner")
    from experiments.models.orchestrator import create_runner

    runner = create_runner(method, "nodedup", {})
    assert runner.variant == "nodedup"
    assert runner._dedup_enabled is False


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_create_runner_isalsr_arm_unchanged(method: str) -> None:
    """The ``isalsr`` arm keeps suppression on and keeps its arm name."""
    pytest.importorskip(f"experiments.models.{method}.isalsr_runner")
    from experiments.models.orchestrator import create_runner

    runner = create_runner(method, "isalsr", {})
    assert runner.variant == "isalsr"
    assert runner._dedup_enabled is True
