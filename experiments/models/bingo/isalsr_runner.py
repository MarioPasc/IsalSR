"""IsalSR-enhanced Bingo runner.

Subclasses Bingo's Evaluation to add canonical string deduplication.
For each individual evaluated, converts AGraph → LabeledDAG, computes
pruned canonical string, and skips isomorphic duplicates by assigning
them infinite fitness.

Architectural difference from UDFS: Instead of monkey-patching, we
subclass Evaluation._serial_eval (Bingo's component-swapping design).
"""

from __future__ import annotations

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
        use_pruned: bool = True,
        timeout: float = 60.0,
        atlas: Any = None,
    ):
        self.use_pruned = use_pruned
        self.timeout = timeout
        self.atlas = atlas  # AtlasLookup | None
        self.canonical_seen: set[int] = set()
        self.n_total: int = 0
        self.n_unique: int = 0
        self.n_skipped: int = 0
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
        **kwargs: Any,
    ):
        super().__init__(fitness_function, **kwargs)
        self.dedup = dedup
        self._snapshot_freq = snapshot_freq
        self._t0 = t0
        self._call_count = 0
        self._best_fitness = float("inf")
        self.snapshots: list[BingoTrajectorySnapshot] = []
        # Set after build_bingo_pipeline returns
        self._fitness_counter: Any = None

    def __call__(self, population: Any) -> None:
        super().__call__(population)
        # MuPlusLambda calls __call__ 2x per generation (parents + offspring)
        self._call_count += 1
        if self._call_count % 2 == 0:
            gen = self._call_count // 2
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

    def _serial_eval(self, population):  # type: ignore[override]
        for indv in population:
            if self._redundant or not indv.fit_set:
                self.dedup.n_total += 1

                # Convert AGraph → LabeledDAG
                try:
                    dag = agraph_to_labeled_dag(indv)
                except Exception:  # noqa: BLE001
                    # Conversion failed: evaluate normally
                    indv.fitness = self.fitness_function(indv)
                    if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                        self._best_fitness = indv.fitness
                    continue

                # Resolve canonical hash: atlas fast-path or online fallback
                t0 = time.perf_counter()
                canon_hash: int | None = None

                if self.dedup.atlas is not None:
                    canon_hash, was_hit = self.dedup.atlas.lookup_dag(dag)
                    dt = time.perf_counter() - t0
                    self.dedup.atlas_lookup_time += dt
                    if was_hit:
                        self.dedup.atlas_hits += 1
                    else:
                        self.dedup.atlas_misses += 1

                if canon_hash is None:
                    # No atlas or atlas miss: compute canonical string
                    t0_canon = time.perf_counter()
                    try:
                        if self.dedup.use_pruned:
                            from isalsr.core.canonical import pruned_canonical_string

                            canonical = pruned_canonical_string(
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
                        indv.fitness = self.fitness_function(indv)
                        if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                            self._best_fitness = indv.fitness
                        continue
                    self.dedup.canon_fallback_time += time.perf_counter() - t0_canon
                    canon_hash = hash(canonical)

                self.dedup.canon_time_total += time.perf_counter() - t0

                # Deduplication check (hash-based for memory efficiency)
                if canon_hash in self.dedup.canonical_seen:
                    self.dedup.n_skipped += 1
                    indv.fitness = np.inf  # Sets fit_set=True, worst fitness
                    continue

                self.dedup.canonical_seen.add(canon_hash)
                self.dedup.n_unique += 1
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
            use_pruned=cfg.use_pruned,
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
            "IsalSR Bingo: total=%d unique=%d skipped=%d canon=%.2fs "
            "atlas_hits=%d misses=%d gens=%d",
            dedup.n_total,
            dedup.n_unique,
            dedup.n_skipped,
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
