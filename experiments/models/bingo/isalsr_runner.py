"""IsalSR-enhanced Bingo runner.

Subclasses Bingo's Evaluation to add canonical string deduplication.
For each individual evaluated, converts AGraph → LabeledDAG, computes
pruned canonical string, and skips isomorphic duplicates by assigning
them infinite fitness.

Architectural difference from UDFS: Instead of monkey-patching, we
subclass Evaluation._serial_eval (Bingo's component-swapping design).

Bug fix (2026-04-01): VarAnd clone bypass.
  Bingo's VarAnd creates offspring via parent.copy() when crossover
  doesn't fire (~60% of the time). AGraph.copy() preserves fit_set=True,
  so these unmodified clones were SKIPPED by _serial_eval's
  ``not indv.fit_set`` guard — bypassing dedup entirely.  ~36% of
  offspring (P(no crossover)×P(no mutation) = 0.6×0.6) entered the
  combined pool with the parent's fitness, allowing selection to keep
  both parent and clone.
  Fix: track object IDs of established population members.  Any
  individual with an unrecognized ID is forced through dedup regardless
  of fit_set.
"""

from __future__ import annotations

import gc
import logging
import time
import warnings
from typing import Any

import numpy as np
from bingo.evaluation.evaluation import Evaluation

from experiments.models.base_runner import ModelRunner
from experiments.models.bingo.adapter import agraph_to_labeled_dag
from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.runner import (
    BingoRawResult,
    BingoTrajectorySnapshot,
    build_bingo_pipeline,
    extract_sympy,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heap fragmentation mitigation for long evolutionary runs.
#
# CPython's pymalloc uses 256 KB arenas.  Over 10 K+ generations of
# create/GC cycles, surviving objects (canonical_seen entries, AGraphs
# kept by selection) pin arenas — preventing glibc from returning pages
# to the OS.  At gen ~19 K this exhausts 64 GB on I.10.7 / I.48.20.
#
# Calling glibc malloc_trim(0) after gc.collect() releases pages whose
# entire arena has been freed.  Combined with PYTHONMALLOC=malloc in the
# SLURM worker (bypasses pymalloc arenas entirely), this keeps RSS under
# control for 12-hour runs.
# ---------------------------------------------------------------------------
try:
    _LIBC = __import__("ctypes").CDLL("libc.so.6")

    def _release_heap() -> None:
        """Run full GC then return freed glibc pages to the OS."""
        gc.collect()
        _LIBC.malloc_trim(0)

except (OSError, AttributeError):
    # Non-Linux or missing libc — fall back to plain gc.collect
    def _release_heap() -> None:  # type: ignore[misc]
        gc.collect()


class _CanonicalDeduplicator:
    """Tracks canonical strings and deduplication statistics.

    Uses hash-based storage (``set[int]``) instead of storing full canonical
    strings.  This reduces per-entry memory from ~150 bytes (``set[str]``)
    to ~28 bytes (``set[int]``), preventing OOM on long evolutionary runs
    with millions of unique individuals.

    The 64-bit Python hash gives collision probability < 3×10⁻⁶ for 10 M
    entries (birthday bound n²/2⁶⁵), which is negligible for our use case.
    """

    def __init__(
        self,
        use_fast_canonical: bool = True,
        timeout: float = 60.0,
        atlas: Any = None,
    ):
        self.use_fast_canonical = use_fast_canonical
        self.timeout = timeout
        self.atlas = atlas  # AtlasLookup | None
        # Historical hash set — used for fitness caching & legacy dedup
        self.canonical_seen: set[int] = set()
        # Fitness cache: canon_hash → fitness (for re-entry after eviction)
        self.fitness_cache: dict[int, float] = {}
        # Population-level dedup: canonical strings of currently alive members
        self.population_canonicals: set[str] = set()
        # Map id(indv) → canonical string for population set rebuild
        self.id_to_canonical: dict[int, str] = {}
        self.n_total: int = 0
        self.n_unique: int = 0
        self.n_skipped: int = 0
        self.n_rejected_duplicates: int = 0  # Population-level rejections
        self.canon_time_total: float = 0.0
        # Atlas-specific stats
        self.atlas_hits: int = 0
        self.atlas_misses: int = 0
        self.atlas_lookup_time: float = 0.0
        self.canon_fallback_time: float = 0.0


class IsalSREvaluation(Evaluation):
    """Bingo Evaluation with IsalSR canonical deduplication.

    Overrides _serial_eval to intercept each individual BEFORE fitness
    evaluation. If the canonical string is already seen, assigns infinite
    fitness (worst possible) and skips the expensive fitness call.

    Also captures periodic trajectory snapshots. AgeFitnessEA extends
    MuPlusLambda, which calls ``evaluation()`` exactly 2× per generation
    (parents then offspring). Snapshots are taken every ``snapshot_freq``
    generations.
    """

    def __init__(
        self,
        fitness_function: Any,
        dedup: _CanonicalDeduplicator,
        snapshot_freq: int = 10,
        t0: float = 0.0,
        enforce_dedup: bool = False,
        **kwargs: Any,
    ):
        super().__init__(fitness_function, **kwargs)
        self.dedup = dedup
        self._snapshot_freq = snapshot_freq
        self._t0 = t0
        self._enforce_dedup = enforce_dedup
        self._call_count = 0
        self._best_fitness = float("inf")
        self.snapshots: list[BingoTrajectorySnapshot] = []
        # Set after build_bingo_pipeline returns
        self._fitness_counter: Any = None
        # Parent-ID registry for VarAnd clone detection (fix 2026-04-01).
        # MuPlusLambda calls evaluation(parents) then evaluation(offspring).
        # In the parent call, every individual has fit_set=True. We record
        # their object IDs.  In the offspring call, any individual whose ID
        # is NOT in _parent_ids is an offspring (even if fit_set=True from
        # AGraph.copy()) and is forced through dedup.
        self._parent_ids: set[int] = set()

    def _rebuild_population_set(self, population: Any) -> None:
        """Rebuild canonical set from current living population.

        Called at the START of each __call__ to synchronize with
        selection/replacement that happened since the last call.
        Uses id(indv) → canonical mapping from previous _serial_eval calls.
        Safe because all population members are alive (held by reference).
        """
        self.dedup.population_canonicals.clear()
        new_id_map: dict[int, str] = {}
        for indv in population:
            canon = self.dedup.id_to_canonical.get(id(indv))
            if canon is not None:
                self.dedup.population_canonicals.add(canon)
                new_id_map[id(indv)] = canon
        # Prune stale entries (evicted individuals)
        self.dedup.id_to_canonical = new_id_map

    def __call__(self, population: Any) -> None:
        # Detect the parent call: all individuals already evaluated AND
        # not the very first evaluation (initial pop has fit_set=False).
        all_evaluated = all(indv.fit_set for indv in population)
        if all_evaluated and self._call_count > 0:
            self._parent_ids = {id(indv) for indv in population}

            # Rebuild population canonical set from PARENTS only.
            # MuPlusLambda calls eval(parents) then eval(offspring).
            # Rebuilding here ensures parent canonicals are in the set
            # before offspring are evaluated.  Offspring canonicals are
            # added incrementally in _serial_eval.
            if self._enforce_dedup:
                self._rebuild_population_set(population)

        super().__call__(population)
        # MuPlusLambda calls __call__ 2x per generation (parents + offspring)
        self._call_count += 1
        if self._call_count % 2 == 0:
            gen = self._call_count // 2

            # Generation-boundary heap release: the most effective point
            # because both parent and offspring evaluation are done and
            # all transient LabeledDAG / canonical string objects from
            # this generation are unreachable.
            _release_heap()

            if gen % self._snapshot_freq == 0:
                n_evals = (
                    self._fitness_counter.eval_count if self._fitness_counter is not None else 0
                )
                self.snapshots.append(
                    BingoTrajectorySnapshot(
                        timestamp_s=time.perf_counter() - self._t0,
                        generation=gen,
                        best_fitness=self._best_fitness,
                        n_evals=n_evals,
                        n_total_dags=self.dedup.n_total,
                        n_unique_canonical=self.dedup.n_unique,
                        n_skipped=self.dedup.n_skipped,
                    )
                )

    # Intra-generation GC interval.  Reduced from 100 K to 5 K to
    # limit peak RSS between generation boundaries on hard problems
    # (I.10.7, I.48.20) where each generation processes >4 K individuals.
    _GC_INTERVAL = 5_000

    def _serial_eval(self, population):  # type: ignore[override]
        for indv in population:
            is_parent = id(indv) in self._parent_ids

            # Process if: (a) standard unevaluated individual, OR
            # (b) NOT a known parent (catches VarAnd clones that
            #     inherited fit_set=True from parent via AGraph.copy()), OR
            # (c) enforce_dedup parent with INF fitness (was a rejected
            #     duplicate in a previous gen — may be eligible now if
            #     the original was evicted by selection).
            is_stale_dup = self._enforce_dedup and is_parent and indv.fitness == np.inf
            should_process = self._redundant or not indv.fit_set or not is_parent or is_stale_dup
            if not should_process:
                continue

            self.dedup.n_total += 1

            # Periodic heap release (GC + malloc_trim)
            if self.dedup.n_total % self._GC_INTERVAL == 0:
                _release_heap()

            # Convert AGraph → LabeledDAG
            try:
                dag = agraph_to_labeled_dag(indv)
            except Exception:  # noqa: BLE001
                # Conversion failed: evaluate normally (only if unevaluated)
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)
                    if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                        self._best_fitness = indv.fitness
                continue

            # Resolve canonical hash: atlas fast-path or online fallback
            t0 = time.perf_counter()
            canon_hash: int | None = None
            canonical: str | None = None

            if self.dedup.atlas is not None:
                canon_hash, was_hit = self.dedup.atlas.lookup_dag(dag)
                dt = time.perf_counter() - t0
                self.dedup.atlas_lookup_time += dt
                if was_hit:
                    self.dedup.atlas_hits += 1
                else:
                    self.dedup.atlas_misses += 1

            # Population dedup needs the full canonical string (not just hash)
            need_canonical_str = canon_hash is None or self._enforce_dedup
            if need_canonical_str and canonical is None:
                # No atlas or atlas miss: compute canonical string
                t0_canon = time.perf_counter()
                try:
                    if self.dedup.use_fast_canonical:
                        from isalsr.core.canonical import fast_canonical_string

                        canonical = fast_canonical_string(
                            dag,
                            timeout=self.dedup.timeout,
                        )
                    else:
                        from isalsr.core.canonical import canonical_string

                        canonical = canonical_string(
                            dag,
                            timeout=self.dedup.timeout,
                        )
                except Exception:  # noqa: BLE001
                    self.dedup.canon_fallback_time += time.perf_counter() - t0_canon
                    self.dedup.canon_time_total += time.perf_counter() - t0
                    if not indv.fit_set:
                        indv.fitness = self.fitness_function(indv)
                        if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                            self._best_fitness = indv.fitness
                    continue
                self.dedup.canon_fallback_time += time.perf_counter() - t0_canon
                if canon_hash is None:
                    canon_hash = hash(canonical)

            self.dedup.canon_time_total += time.perf_counter() - t0

            if self._enforce_dedup:
                # --- Population-level dedup with fitness caching ---

                # Check if this canonical is already held by a living member
                if canonical in self.dedup.population_canonicals:
                    self.dedup.n_rejected_duplicates += 1
                    self.dedup.n_skipped += 1
                    indv.fitness = np.inf
                    continue

                # Fitness caching: reuse fitness if seen historically
                if canon_hash in self.dedup.fitness_cache:
                    indv.fitness = self.dedup.fitness_cache[canon_hash]
                elif not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)

                # Cache the fitness for future reuse
                if np.isfinite(indv.fitness):
                    self.dedup.fitness_cache[canon_hash] = indv.fitness

                # Register in historical set and population set
                if canon_hash not in self.dedup.canonical_seen:
                    self.dedup.n_unique += 1
                self.dedup.canonical_seen.add(canon_hash)
                self.dedup.population_canonicals.add(canonical)
                self.dedup.id_to_canonical[id(indv)] = canonical

                if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                    self._best_fitness = indv.fitness
            else:
                # --- Legacy dedup: historical hash rejection ---
                if canon_hash in self.dedup.canonical_seen:
                    self.dedup.n_skipped += 1
                    indv.fitness = np.inf
                    continue

                self.dedup.canonical_seen.add(canon_hash)
                self.dedup.n_unique += 1
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)
                if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                    self._best_fitness = indv.fitness


class IsalSRBingoRunner(ModelRunner):
    """Runs Bingo with IsalSR canonical deduplication."""

    def __init__(self, config: BingoConfig | None = None, atlas: Any = None):
        self._config = config or BingoConfig()
        self._atlas = atlas  # AtlasLookup | None

    @property
    def name(self) -> str:
        return "bingo"

    @property
    def variant(self) -> str:
        return "isalsr"

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        config: dict[str, Any],
    ) -> BingoRawResult:
        cfg = BingoConfig.from_dict(config) if config else self._config

        np.random.seed(seed)

        dedup = _CanonicalDeduplicator(
            use_fast_canonical=cfg.use_fast_canonical,
            timeout=cfg.canonicalization_timeout,
            atlas=self._atlas,
        )

        t0 = time.perf_counter()

        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={
                "dedup": dedup,
                "snapshot_freq": cfg.snapshot_frequency,
                "t0": t0,
                "enforce_dedup": cfg.enforce_population_dedup,
            },
        )
        # fitness_fn (ExplicitRegression) has eval_count; set after build
        evaluation._fitness_counter = fitness_fn  # type: ignore[union-attr]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evolve_until_convergence(
                max_generations=cfg.generations,
                fitness_threshold=cfg.fitness_threshold,
                max_fitness_evaluations=cfg.max_evals,
                convergence_check_frequency=10,
                max_time=cfg.max_time,
            )
        wall_clock = time.perf_counter() - t0

        # Extract results
        total_evals = fitness_fn.eval_count
        best_agraph = None
        best_sympy_expr = None
        best_fitness = float("inf")
        y_pred_train = np.full(len(y_train), np.nan)
        y_pred_test = np.full(len(y_test), np.nan)
        n_gens = island.generational_age

        try:
            from experiments.models.bingo.runner import _with_timeout

            best_agraph = island.get_best_individual()
            best_fitness = best_agraph.fitness
            best_sympy_expr = _with_timeout(lambda: extract_sympy(best_agraph))
            pred_train = _with_timeout(
                lambda: best_agraph.evaluate_equation_at(x_train).flatten(),
                60,
            )
            pred_test = _with_timeout(
                lambda: best_agraph.evaluate_equation_at(x_test).flatten(),
                60,
            )
            if pred_train is not None:
                y_pred_train = pred_train
            if pred_test is not None:
                y_pred_test = pred_test
        except Exception:  # noqa: BLE001
            log.debug("Failed to extract Bingo IsalSR results", exc_info=True)

        search_only = wall_clock - dedup.canon_time_total
        snapshots = evaluation.snapshots  # type: ignore[union-attr]

        log.info(
            "IsalSR Bingo: total=%d unique=%d skipped=%d pop_rejected=%d "
            "canon=%.2fs atlas_hits=%d misses=%d gens=%d",
            dedup.n_total,
            dedup.n_unique,
            dedup.n_skipped,
            dedup.n_rejected_duplicates,
            dedup.canon_time_total,
            dedup.atlas_hits,
            dedup.atlas_misses,
            n_gens,
        )

        return BingoRawResult(
            wall_clock_s=wall_clock,
            seed=seed,
            best_agraph=best_agraph,
            best_sympy=best_sympy_expr,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            total_evals=total_evals,
            best_fitness=best_fitness,
            n_generations=n_gens,
            trajectory_snapshots=snapshots,
            n_total_dags=dedup.n_total,
            n_unique_canonical=dedup.n_unique,
            n_skipped=dedup.n_skipped,
            canonicalization_time_s=dedup.canon_time_total,
            search_only_time_s=search_only,
            atlas_hits=dedup.atlas_hits,
            atlas_misses=dedup.atlas_misses,
            atlas_lookup_time_s=dedup.atlas_lookup_time,
            canon_fallback_time_s=dedup.canon_fallback_time,
        )
