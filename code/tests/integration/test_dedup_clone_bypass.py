"""Tests for the VarAnd clone bypass fix (2026-04-01).

Verifies that IsalSREvaluation correctly deduplicates offspring created
by VarAnd's parent.copy() path, which retains fit_set=True.

The bug: ~36% of offspring (P(no crossover)×P(no mutation) = 0.6×0.6)
bypass dedup because AGraph.copy() preserves fit_set=True and
_serial_eval guarded on ``not indv.fit_set``.

The fix: track object IDs of established population members; force
dedup processing for any individual with an unrecognized ID.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

np = pytest.importorskip("numpy")
bingo = pytest.importorskip("bingo")

from bingo.symbolic_regression.agraph.agraph import AGraph  # noqa: E402

from experiments.models.bingo.adapter import agraph_to_labeled_dag  # noqa: E402
from experiments.models.bingo.config import BingoConfig  # noqa: E402
from experiments.models.bingo.isalsr_runner import (  # noqa: E402
    IsalSREvaluation,
    _CanonicalDeduplicator,
)
from experiments.models.bingo.runner import build_bingo_pipeline  # noqa: E402
from isalsr.core.canonical import fast_canonical_string  # noqa: E402

# ======================================================================
# Helpers
# ======================================================================


def _make_simple_agraph() -> AGraph:
    """Create a minimal evaluated AGraph: f(x) = x + x."""
    ag = AGraph(use_simplification=False)
    # stack: row0=X_0, row1=X_0, row2=X_0+X_0
    ag.command_array = np.array(
        [[0, 0, 0], [0, 0, 0], [2, 0, 1]],
        dtype=int,
    )
    return ag


def _make_different_agraph() -> AGraph:
    """Create a different AGraph: f(x) = x * x (distinct canonical string)."""
    ag = AGraph(use_simplification=False)
    # stack: row0=X_0, row1=X_0, row2=X_0*X_0
    ag.command_array = np.array(
        [[0, 0, 0], [0, 0, 0], [4, 0, 1]],
        dtype=int,
    )
    return ag


def _canonical(ag: AGraph) -> str:
    dag = agraph_to_labeled_dag(ag)
    return fast_canonical_string(dag, timeout=10.0)


def _population_delta_finite(population: list) -> tuple[float, int, int]:
    """Compute effective diversity ratio among finite-fitness individuals.

    Returns (delta, n_unique, n_finite).
    """
    cs_list = []
    for ag in population:
        if hasattr(ag, "fitness") and ag.fit_set and np.isfinite(ag.fitness):
            try:
                cs_list.append(_canonical(ag))
            except Exception:
                cs_list.append(f"__FAIL_{id(ag)}__")
    n_finite = len(cs_list)
    n_unique = len(set(cs_list))
    delta = n_unique / n_finite if n_finite > 0 else 0.0
    return delta, n_unique, n_finite


# ======================================================================
# Unit-level: VarAnd clone detection
# ======================================================================


class TestVarAndCloneDetection:
    """Verify that _serial_eval processes VarAnd clones (new ID + fit_set=True).

    Tests simulate the real MuPlusLambda call sequence:
      1. evaluation(initial_pop)   — initial evaluation (all fit_set=False)
      2. evaluation(parents)       — parent call (all fit_set=True → _parent_ids set)
      3. evaluation(offspring)     — offspring call (some fit_set=False → clones detected)
    """

    def test_agraph_copy_preserves_fit_set(self):
        """Precondition: AGraph.copy() retains fit_set=True (Bingo behavior)."""
        ag = _make_simple_agraph()
        ag.fitness = 5.0
        assert ag.fit_set is True

        clone = ag.copy()
        assert clone.fit_set is True, "AGraph.copy() should preserve fit_set"
        assert np.array_equal(clone.command_array, ag.command_array)
        assert id(clone) != id(ag)

    def test_clone_with_fit_set_is_deduped(self):
        """A VarAnd clone (new ID, fit_set=True) must be caught by dedup.

        Simulates: initial eval → parent call → offspring call with clone.
        """

        class DummyFitness:
            eval_count = 0

            def __call__(self, indv):
                self.eval_count += 1
                return 1.0

        fitness_fn = DummyFitness()
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=10.0)
        evaluation = IsalSREvaluation(fitness_fn, dedup=dedup)

        # Step 1: Initial evaluation (simulates evaluate_population)
        original = _make_simple_agraph()
        evaluation([original])

        assert original.fit_set is True
        assert np.isfinite(original.fitness)
        assert dedup.n_unique == 1
        assert dedup.n_skipped == 0

        # Step 2: Parent call (simulates evaluation(parents) in generational_step)
        evaluation([original])

        # Step 3: Offspring call with clone + 1 fresh individual
        # (AddRandomIndividuals always adds 1 fresh → offspring batch has fit_set=False)
        clone = original.copy()
        fresh = _make_different_agraph()  # fit_set=False, different structure
        assert clone.fit_set is True
        assert id(clone) != id(original)

        evaluation([clone, fresh])

        assert dedup.n_skipped == 1, "Clone must be caught by dedup"
        assert clone.fitness == np.inf, "Clone must get inf fitness"

    def test_established_parent_is_skipped(self):
        """An established parent (known ID, fit_set=True) should be skipped."""

        class DummyFitness:
            eval_count = 0

            def __call__(self, indv):
                self.eval_count += 1
                return 1.0

        fitness_fn = DummyFitness()
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=10.0)
        evaluation = IsalSREvaluation(fitness_fn, dedup=dedup)

        # Step 1: Initial evaluation
        original = _make_simple_agraph()
        evaluation([original])

        original_fitness = original.fitness
        n_total_before = dedup.n_total

        # Step 2: Parent call — original is a known parent, must be skipped
        evaluation([original])

        assert dedup.n_total == n_total_before, "Established parent must be skipped"
        assert original.fitness == original_fitness, "Parent fitness must not change"

    def test_multiple_clones_all_deduped(self):
        """Multiple clones of the same parent are all caught."""

        class DummyFitness:
            eval_count = 0

            def __call__(self, indv):
                self.eval_count += 1
                return 1.0

        fitness_fn = DummyFitness()
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=10.0)
        evaluation = IsalSREvaluation(fitness_fn, dedup=dedup)

        # Step 1: Initial evaluation
        original = _make_simple_agraph()
        evaluation([original])

        # Step 2: Parent call
        evaluation([original])

        # Step 3: Offspring call with 5 clones + 1 fresh
        clones = [original.copy() for _ in range(5)]
        fresh = _make_different_agraph()  # different structure, fit_set=False
        evaluation(clones + [fresh])

        assert dedup.n_skipped == 5
        assert all(c.fitness == np.inf for c in clones)


# ======================================================================
# Integration: population-level diversity invariant
# ======================================================================


@pytest.fixture
def nguyen1_data():
    from benchmarks.datasets.nguyen import generate_data, get_benchmark

    bench = get_benchmark("Nguyen-1")
    x_train, y_train, x_test, y_test = generate_data(bench, seed=42)
    return x_train, y_train


@pytest.fixture
def small_cfg():
    return BingoConfig(
        population_size=64,
        stack_size=16,
        operators=["+", "-", "*", "/"],
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        max_time=120.0,
        max_evals=10_000_000,
    )


@pytest.mark.slow
class TestPopulationDiversityInvariant:
    """Verify that δ_finite = 1.0 after initial evaluation (Proposition X.4)."""

    def test_delta_finite_1_at_gen0(self, nguyen1_data, small_cfg):
        """After initial evaluate_population, all finite-fitness individuals
        must have unique canonical strings."""
        x_train, y_train = nguyen1_data
        np.random.seed(42)

        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            small_cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={"dedup": dedup, "snapshot_freq": 100_000, "t0": 0.0},
        )
        evaluation._fitness_counter = fitness_fn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evaluate_population()

        delta, n_unique, n_finite = _population_delta_finite(island.population)
        assert delta == 1.0, (
            f"δ_finite must be 1.0 at gen 0, got {delta:.3f} (unique={n_unique}, finite={n_finite})"
        )

    def test_delta_finite_1_after_evolve(self, nguyen1_data, small_cfg):
        """After evolve(1), all finite-fitness individuals must still have
        unique canonical strings — the fix must catch VarAnd clones."""
        x_train, y_train = nguyen1_data
        np.random.seed(42)

        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            small_cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={"dedup": dedup, "snapshot_freq": 100_000, "t0": 0.0},
        )
        evaluation._fitness_counter = fitness_fn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evaluate_population()
            island.evolve(1)

        delta, n_unique, n_finite = _population_delta_finite(island.population)
        assert delta == 1.0, (
            f"δ_finite must be 1.0 after evolve(1), got {delta:.3f} "
            f"(unique={n_unique}, finite={n_finite})"
        )

    def test_delta_finite_1_after_10_generations(self, nguyen1_data, small_cfg):
        """After 10 generations of evolution, δ_finite must remain 1.0."""
        x_train, y_train = nguyen1_data
        np.random.seed(7)

        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            small_cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={"dedup": dedup, "snapshot_freq": 100_000, "t0": 0.0},
        )
        evaluation._fitness_counter = fitness_fn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evaluate_population()
            for _ in range(10):
                island.evolve(1)

        delta, n_unique, n_finite = _population_delta_finite(island.population)
        assert delta == 1.0, (
            f"δ_finite must be 1.0 after 10 gens, got {delta:.3f} "
            f"(unique={n_unique}, finite={n_finite})"
        )

    def test_dedup_catches_more_with_fix(self, nguyen1_data, small_cfg):
        """The fix must increase n_skipped compared to a hypothetical
        unpatched version (VarAnd clones are now caught)."""
        x_train, y_train = nguyen1_data
        np.random.seed(42)

        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=30.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            small_cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={"dedup": dedup, "snapshot_freq": 100_000, "t0": 0.0},
        )
        evaluation._fitness_counter = fitness_fn

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evaluate_population()
            for _ in range(5):
                island.evolve(1)

        # With pop_size=64 and 5 gens, the dedup should catch significantly
        # more duplicates than a trivial amount
        assert dedup.n_skipped > 0, "Dedup must catch some duplicates"
        assert dedup.n_total > dedup.n_unique, "Some individuals must be duplicates"
