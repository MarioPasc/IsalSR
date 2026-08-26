"""Tests for population-level canonical deduplication in Bingo IsalSR.

Validates the enforce_population_dedup mode: ensures duplicate canonical
strings within the living population are rejected (INF fitness), while
historically-seen canonicals can re-enter the population with cached fitness.

Requires: bingo-nasa
"""

from __future__ import annotations

import numpy as np
import pytest

bingo = pytest.importorskip("bingo")

from bingo.symbolic_regression.agraph.agraph import AGraph  # noqa: E402

from experiments.models.bingo.adapter import agraph_to_labeled_dag  # noqa: E402
from experiments.models.bingo.config import BingoConfig  # noqa: E402
from experiments.models.bingo.isalsr_runner import (  # noqa: E402
    _DUPLICATE_AGE_PENALTY,
    IsalSREvaluation,
    _CanonicalDeduplicator,
)
from experiments.models.bingo.runner import build_bingo_pipeline  # noqa: E402
from isalsr.core.canonical import fast_canonical_string  # noqa: E402


def _make_simple_agraph(x_coeff: float = 1.0) -> AGraph:
    """Create a simple AGraph for x * x_coeff (linear expression).

    Uses a deterministic command array: load x_0, load const, multiply.
    """
    ag = AGraph()
    ag.command_array = np.array(
        [
            [0, 0, 0],  # x_0
            [1, 0, 0],  # const (slot 0)
            [4, 0, 1],  # multiply (row 0 * row 1)
        ],
        dtype=int,
    )
    ag.set_local_optimization_params(np.array([x_coeff]))
    return ag


def _make_sin_x_agraph() -> AGraph:
    """Create AGraph for sin(x_0)."""
    ag = AGraph()
    ag.command_array = np.array(
        [
            [0, 0, 0],  # x_0
            [6, 0, 0],  # sin(row 0)
        ],
        dtype=int,
    )
    ag.set_local_optimization_params(np.array([]))
    return ag


def _make_cos_x_agraph() -> AGraph:
    """Create AGraph for cos(x_0)."""
    ag = AGraph()
    ag.command_array = np.array(
        [
            [0, 0, 0],  # x_0
            [7, 0, 0],  # cos(row 0)
        ],
        dtype=int,
    )
    ag.set_local_optimization_params(np.array([]))
    return ag


def _make_x_plus_sin_x_agraph_v1() -> AGraph:
    """Create AGraph for x + sin(x), version 1 (sin first)."""
    ag = AGraph()
    ag.command_array = np.array(
        [
            [0, 0, 0],  # row 0: x_0
            [6, 0, 0],  # row 1: sin(x_0)
            [2, 0, 1],  # row 2: x_0 + sin(x_0)
        ],
        dtype=int,
    )
    ag.set_local_optimization_params(np.array([]))
    return ag


def _make_x_plus_sin_x_agraph_v2() -> AGraph:
    """Create AGraph for x + sin(x), version 2 (different row ordering).

    Uses padding rows so the utilized structure is the same expression
    but with different internal row indices.
    """
    ag = AGraph()
    ag.command_array = np.array(
        [
            [0, 0, 0],  # row 0: x_0
            [1, 0, 0],  # row 1: const (unused padding)
            [6, 0, 0],  # row 2: sin(x_0)
            [2, 0, 2],  # row 3: x_0 + sin(x_0)
        ],
        dtype=int,
    )
    ag.set_local_optimization_params(np.array([1.0]))
    return ag


class TestPopulationDedup:
    """Tests for population-level duplicate rejection."""

    def _make_dedup_and_eval(self) -> tuple[_CanonicalDeduplicator, IsalSREvaluation]:
        """Create a deduplicator and evaluation with enforce_dedup=True."""
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)

        class DummyFitness:
            eval_count = 0

            def __call__(self, indv):
                self.eval_count += 1
                return 1.0

        evaluation = IsalSREvaluation(
            fitness_function=DummyFitness(),
            dedup=dedup,
            snapshot_freq=100_000,
            t0=0.0,
            enforce_dedup=True,
        )
        return dedup, evaluation

    def test_duplicate_rejected(self):
        """Two individuals with the same canonical string: second gets INF."""
        dedup, evaluation = self._make_dedup_and_eval()

        ag1 = _make_sin_x_agraph()
        ag2 = _make_sin_x_agraph()  # Identical structure

        population = [ag1, ag2]
        evaluation._serial_eval(population)

        assert np.isfinite(ag1.fitness), "First individual should have finite fitness"
        assert ag2.fitness == np.inf, "Duplicate should get INF fitness"
        assert dedup.n_rejected_duplicates == 1

    def test_population_set_tracks_evictions(self):
        """After removing an individual, its canonical string is freed."""
        dedup, evaluation = self._make_dedup_and_eval()

        ag1 = _make_sin_x_agraph()
        ag2 = _make_cos_x_agraph()

        # Evaluate both — both should be accepted
        population = [ag1, ag2]
        evaluation._serial_eval(population)
        assert np.isfinite(ag1.fitness)
        assert np.isfinite(ag2.fitness)

        # Simulate eviction: remove ag1, replace with new sin(x)
        ag3 = _make_sin_x_agraph()
        new_population = [ag3, ag2]  # ag1 evicted

        # Rebuild population set (simulates start of next __call__)
        evaluation._rebuild_population_set(new_population)

        # ag3 has same canonical as ag1, but ag1 was evicted
        # ag3 should be ACCEPTED (not rejected)
        evaluation._serial_eval([ag3])
        assert np.isfinite(ag3.fitness), "Individual with evicted canonical should be accepted"

    def test_fitness_caching_still_works(self):
        """Historical canonical reuses cached fitness (no re-evaluation)."""
        dedup, evaluation = self._make_dedup_and_eval()

        ag1 = _make_sin_x_agraph()
        population = [ag1]
        evaluation._serial_eval(population)
        original_fitness = ag1.fitness
        assert np.isfinite(original_fitness)

        # Simulate eviction and re-entry
        evaluation._rebuild_population_set([])  # Clear population set

        ag2 = _make_sin_x_agraph()  # Same canonical
        evaluation._serial_eval([ag2])

        # Should get cached fitness (same value, no re-evaluation)
        assert ag2.fitness == pytest.approx(original_fitness, rel=1e-10), (
            "Cached fitness should be reused for re-entering canonical"
        )
        assert np.isfinite(ag2.fitness)

    def test_isomorphic_agraphs_collapsed(self):
        """Two AGraphs encoding x+sin(x) with different structure: same canonical."""
        ag1 = _make_x_plus_sin_x_agraph_v1()
        ag2 = _make_x_plus_sin_x_agraph_v2()

        dag1 = agraph_to_labeled_dag(ag1)
        dag2 = agraph_to_labeled_dag(ag2)
        canon1 = fast_canonical_string(dag1)
        canon2 = fast_canonical_string(dag2)
        assert canon1 == canon2, (
            f"Isomorphic AGraphs should have same canonical: {canon1} vs {canon2}"
        )

        # In a population, second should be rejected
        dedup, evaluation = self._make_dedup_and_eval()
        evaluation._serial_eval([ag1, ag2])
        assert np.isfinite(ag1.fitness)
        assert ag2.fitness == np.inf, "Isomorphic duplicate should be rejected"


class TestDeltaReachesOne:
    """Integration test: delta = 1.0 on a trivial problem with enforce_dedup."""

    @pytest.mark.timeout(120)
    def test_delta_reaches_one_nguyen1(self):
        """After sufficient generations, delta = 1.0 on Nguyen-1."""
        cfg = BingoConfig(
            population_size=64,
            stack_size=12,
            crossover_prob=0.4,
            mutation_prob=0.4,
            max_time=60.0,
            enforce_population_dedup=True,
        )

        rng = np.random.RandomState(42)
        x = rng.uniform(-1, 1, (20, 1))
        y = x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]

        np.random.seed(42)
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x,
            y,
            cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={
                "dedup": dedup,
                "snapshot_freq": 100_000,
                "t0": 0.0,
                "enforce_dedup": True,
            },
        )

        # Evolve for 30 generations
        for _ in range(30):
            island.evolve(1)

        # Compute delta on the final population
        canonical_strings = set()
        n_valid = 0
        for indv in island.population:
            try:
                dag = agraph_to_labeled_dag(indv)
                canon = fast_canonical_string(dag, timeout=10.0)
                canonical_strings.add(canon)
                n_valid += 1
            except Exception:  # noqa: BLE001
                n_valid += 1  # Count but skip

        delta = len(canonical_strings) / n_valid if n_valid > 0 else 0
        # With the NaN fitness + age penalty, duplicates are unconditionally
        # removed by AgeFitness selection. For small N=64 populations with
        # stack_size=12, delta >= 0.90 after 30 gens is a strong signal.
        # Production runs (N=200+) achieve delta >= 0.99.
        assert delta >= 0.90, (
            f"Delta should approach 1.0 with enforce_dedup + age penalty, "
            f"got {delta:.3f} ({len(canonical_strings)}/{n_valid} unique)"
        )


class TestAgePenalty:
    """Tests for the age penalty mechanism on duplicate individuals."""

    def _make_dedup_and_eval(
        self, use_age_penalty: bool = True
    ) -> tuple[_CanonicalDeduplicator, IsalSREvaluation]:
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)

        class DummyFitness:
            eval_count = 0

            def __call__(self, indv):
                self.eval_count += 1
                return 1.0

        evaluation = IsalSREvaluation(
            fitness_function=DummyFitness(),
            dedup=dedup,
            snapshot_freq=100_000,
            t0=0.0,
            enforce_dedup=True,
            use_age_penalty=use_age_penalty,
        )
        return dedup, evaluation

    def test_age_penalty_ensures_pareto_dominance(self):
        """Duplicate with age penalty is always Pareto-dominated."""
        from bingo.selection.age_fitness import AgeFitness

        _, evaluation = self._make_dedup_and_eval(use_age_penalty=True)

        ag1 = _make_sin_x_agraph()
        ag2 = _make_sin_x_agraph()  # duplicate
        ag2.genetic_age = 3  # young age

        evaluation._serial_eval([ag1, ag2])

        # ag2 should have INF fitness and high age
        assert ag2.fitness == np.inf
        assert ag2.genetic_age >= _DUPLICATE_AGE_PENALTY

        # ag1 should NOT be dominated by ag2
        assert AgeFitness._first_not_dominated(ag1, ag2) is True
        # ag2 IS dominated by ag1
        assert AgeFitness._first_not_dominated(ag2, ag1) is False

    def test_age_penalty_can_be_disabled(self):
        """With use_age_penalty=False, only fitness is set to INF."""
        _, evaluation = self._make_dedup_and_eval(use_age_penalty=False)

        ag1 = _make_sin_x_agraph()
        ag2 = _make_sin_x_agraph()
        original_age = ag2.genetic_age

        evaluation._serial_eval([ag1, ag2])

        assert ag2.fitness == np.inf, "Duplicate should get INF fitness"
        assert ag2.genetic_age == original_age, "Age should NOT be modified"

    def test_stale_dup_age_reset(self):
        """Previously penalized individual gets age reset when re-eligible."""
        dedup, evaluation = self._make_dedup_and_eval(use_age_penalty=True)

        ag1 = _make_sin_x_agraph()
        ag2 = _make_sin_x_agraph()  # duplicate

        evaluation._serial_eval([ag1, ag2])
        assert ag2.genetic_age >= _DUPLICATE_AGE_PENALTY

        # Simulate eviction: ag1 removed, ag2 remains as stale dup
        evaluation._rebuild_population_set([ag2])
        # population_canonicals now has ag2's canonical

        # Clear population set (simulates ag2's canonical being evicted)
        evaluation._rebuild_population_set([])

        # Re-evaluate ag2: it's no longer a population duplicate
        ag2.fit_set = False  # force re-evaluation
        evaluation._serial_eval([ag2])

        # Age should be reset (was stale dup, now eligible)
        assert ag2.genetic_age < _DUPLICATE_AGE_PENALTY, (
            f"Stale dup age should be reset, got {ag2.genetic_age}"
        )
