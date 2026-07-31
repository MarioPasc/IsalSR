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
from experiments.models.bingo.adapter import (
    BINGO_BINARY_OPS,
    BINGO_UNARY_OPS,
    agraph_to_labeled_dag,
)
from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.runner import (
    BingoRawResult,
    BingoTrajectorySnapshot,
    build_bingo_pipeline,
    extract_sympy,
)
from experiments.models.fallback_ledger import FallbackLedger
from isalsr.baselines import FixedOrder, HyperLogLog, fixed_order_hash, serialise
from isalsr.baselines.host_native import HostNativeRecord, host_native_serialise

log = logging.getLogger(__name__)

# Deduplication key modes.
#   "canonical"   -- the IsalSR arm: complete labeled-DAG isomorphism invariant.
#   "host_native" -- the naive baseline of reviewer comment R1.4: the host's own
#                    command array in the host's own row order.
#   "hash"        -- the steel-manned second rung: a fixed order over the
#                    *adapter's* output, which concedes IsalSR's own renumbering.
# Everything else about the arms — when the hook fires, what a duplicate does to
# the search, the counters, the snapshots — is identical by construction.
KEY_MODES = ("canonical", "host_native", "hash")

# The fixed order used by the "hash" (adapter-order) key mode.
HASH_ARM_ORDER = FixedOrder.TOPOLOGICAL

# Key mode of the ``hash`` *arm*.  The arm keys on the host's own representation;
# ``key_mode="hash"`` remains available as a configurable second rung.
HASH_ARM_KEY_MODE = "host_native"

# HyperLogLog precision for the shadow sketches.  p=16 gives 65,536 registers
# (64 KB per sketch) and a relative standard error of 1.04/sqrt(2^16) = 0.41 %.
# p=14 (s.e. 0.81 %) could not resolve the measured +1.17 % Bingo shadow gap,
# which sat at ~1.4 sigma.  Memory stays ~10^4x below an exact set[int].
SHADOW_HLL_PRECISION = 16

# RunLog field name per fixed order, for the shadow cardinality sketches.
SHADOW_FIELDS: dict[FixedOrder, str] = {
    FixedOrder.INSERTION: "shadow_distinct_insertion",
    FixedOrder.TOPOLOGICAL: "shadow_distinct_topological",
    FixedOrder.TOPOLOGICAL_COMMUTATIVE: "shadow_distinct_topological_commutative",
}


def bingo_host_native_records(agraph: Any) -> list[HostNativeRecord]:
    """Extract host-native records from a Bingo ``AGraph``.

    Iterates ``agraph.command_array`` in **its own row order** and emits, per
    utilised row, the row index, the opcode, and the operand row indices exactly
    as Bingo stores them.  No renumbering and no topological pass: Bingo's row
    order is already the host's own order.

    Two restrictions are applied, both for the same reason -- the excluded data
    is genetic material that Bingo never reads, so keying on it would split
    candidates that are the same expression:

    - Non-utilised rows are dropped (``get_utilized_commands``).  Bingo stacks
      carry dead code.
    - For a unary or terminal row only ``param1`` is emitted.  ``param2`` is
      never read at those arities and is junk: on a 3,085-candidate Nguyen-style
      stream, 1,679 of 2,595 utilised unary rows (64.7 %) carried
      ``param1 != param2``.

    Args:
        agraph: A Bingo ``AGraph``.

    Returns:
        The host-native records in ``command_array`` row order.
    """
    cmd = agraph.command_array
    utilized = agraph.get_utilized_commands()
    records: list[HostNativeRecord] = []
    for row in range(len(cmd)):
        if not utilized[row]:
            continue
        op_code = int(cmd[row, 0])
        if op_code in BINGO_BINARY_OPS:
            operands: tuple[int, ...] = (int(cmd[row, 1]), int(cmd[row, 2]))
        elif op_code in BINGO_UNARY_OPS:
            operands = (int(cmd[row, 1]),)
        else:
            # Terminal: param1 is the variable index (VARIABLE) or the constant
            # slot (CONSTANT), not a row index, but it is still host-stored data
            # the label depends on.
            operands = (int(cmd[row, 1]),)
        records.append((row, f"op{op_code}", operands))
    return records


# Age penalty for duplicate individuals.  Must be large enough that the
# duplicate is Pareto-dominated by ANY other individual in AgeFitnessEA
# selection on both (genetic_age, fitness) dimensions.  With fitness=inf
# and age=_DUPLICATE_AGE_PENALTY, _first_not_dominated(other, dup) is
# always True for any finite-fitness, finite-age individual, guaranteeing
# immediate removal by selection.
_DUPLICATE_AGE_PENALTY = 10_000_000


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
        ledger: FallbackLedger | None = None,
        key_mode: str = "canonical",
        shadow_hash: bool = False,
    ):
        if key_mode not in KEY_MODES:
            raise ValueError(f"Unknown key_mode: {key_mode!r} (expected one of {KEY_MODES})")
        self.use_fast_canonical = use_fast_canonical
        self.timeout = timeout
        self.atlas = atlas  # AtlasLookup | None
        self.ledger: FallbackLedger | None = ledger
        self.key_mode = key_mode
        # Shadow distinct-cardinality sketches over the full candidate stream.
        # HyperLogLog(p=14) is ~16 KB each and constant in stream length; an
        # exact set[int] would cost 1-2 GB at 10^7 candidates.
        self._shadow: dict[FixedOrder, HyperLogLog] = (
            {order: HyperLogLog(p=SHADOW_HLL_PRECISION) for order in SHADOW_FIELDS}
            if shadow_hash
            else {}
        )
        self.n_shadow_failures: int = 0
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

    def representation_string(self, dag: Any, host: Any = None) -> str:
        """Return the string whose hash is this arm's deduplication key.

        Args:
            dag: The candidate ``LabeledDAG``, as produced by the host adapter
                (so the T16 SUB/DIV decomposition is already applied).
            host: The originating Bingo ``AGraph``.  Required by the
                ``"host_native"`` key mode, ignored by the other two.

        Returns:
            The canonical string for the ``"canonical"`` arm, the host-native
            serialisation for the ``"host_native"`` arm, or the adapter-order
            fixed serialisation for the ``"hash"`` arm.

        Raises:
            ValueError: If ``key_mode`` is ``"host_native"`` and *host* is None.
        """
        if self.key_mode == "host_native":
            if host is None:
                raise ValueError("key_mode='host_native' requires the host AGraph")
            return host_native_serialise(bingo_host_native_records(host))
        if self.key_mode == "hash":
            return serialise(dag, HASH_ARM_ORDER)
        if self.use_fast_canonical:
            from isalsr.core.canonical import fast_canonical_string  # noqa: PLC0415

            return fast_canonical_string(dag, timeout=self.timeout)
        from isalsr.core.canonical import canonical_string  # noqa: PLC0415

        return canonical_string(dag, timeout=self.timeout)

    def record_shadow(self, dag: Any) -> None:
        """Feed one candidate into every enabled fixed-order cardinality sketch.

        A serialisation failure is counted and ignored: the shadow counters are
        instrumentation and must never change the arm's search behaviour.

        Args:
            dag: The same ``LabeledDAG`` object the deduplication key sees.
        """
        if not self._shadow:
            return
        for order, sketch in self._shadow.items():
            try:
                sketch.add(fixed_order_hash(dag, order))
            except Exception:  # noqa: BLE001
                self.n_shadow_failures += 1

    def shadow_counts(self) -> dict[str, float]:
        """Return the shadow distinct-cardinality estimates by RunLog field name."""
        return {SHADOW_FIELDS[order]: sketch.count() for order, sketch in self._shadow.items()}


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
        use_age_penalty: bool = True,
        **kwargs: Any,
    ):
        super().__init__(fitness_function, **kwargs)
        self.dedup = dedup
        self._snapshot_freq = snapshot_freq
        self._t0 = t0
        self._enforce_dedup = enforce_dedup
        self._use_age_penalty = use_age_penalty
        self._call_count = 0
        self._best_fitness = float("inf")
        self.snapshots: list[BingoTrajectorySnapshot] = []
        # Dense per-generation convergence data (all individuals' fitness).
        self.convergence_data: list[tuple[int, float, int, np.ndarray]] = []
        # Per-generation count of penalised-in-pool duplicates
        # (age >= _DUPLICATE_AGE_PENALTY).  Measures effective-population
        # shortfall: N_effective = len(population) - n_penalised_per_gen[g].
        self.n_penalised_per_gen: list[tuple[int, int, int]] = []
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

    def _capture_convergence(self, generation: int, population: Any) -> None:
        """Record every individual's fitness for convergence analysis.

        Also counts penalised-in-pool duplicates (age >= _DUPLICATE_AGE_PENALTY)
        to measure the effective-population shortfall introduced by the
        "mark +inf, age=10^7" dedup strategy (see TECHNICAL_REPORT §2.8).
        """
        n_evals = self._fitness_counter.eval_count if self._fitness_counter is not None else 0
        fitness_arr = np.array(
            [
                indv.fitness if hasattr(indv, "fitness") and indv.fit_set else np.inf
                for indv in population
            ],
            dtype=np.float64,
        )
        n_penalised = sum(
            1 for indv in population if getattr(indv, "genetic_age", 0) >= _DUPLICATE_AGE_PENALTY
        )
        self.convergence_data.append(
            (generation, time.perf_counter() - self._t0, n_evals, fitness_arr)
        )
        self.n_penalised_per_gen.append((generation, n_penalised, len(population)))

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

        # Capture gen 0 after initial evaluation
        if self._call_count == 1:
            self._capture_convergence(0, population)

        if self._call_count % 2 == 0:
            gen = self._call_count // 2
            # Dense per-generation capture: all individuals' fitness
            self._capture_convergence(gen, population)

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
            # (c) enforce_dedup parent with non-finite fitness (was a
            #     rejected duplicate in a previous gen — may be eligible
            #     now if the original was evicted by selection).
            is_stale_dup = self._enforce_dedup and is_parent and not np.isfinite(indv.fitness)
            should_process = self._redundant or not indv.fit_set or not is_parent or is_stale_dup
            if not should_process:
                continue

            self.dedup.n_total += 1

            # Periodic heap release (GC + malloc_trim)
            if self.dedup.n_total % self._GC_INTERVAL == 0:
                _release_heap()

            # Convert AGraph → LabeledDAG
            # record_pre is called inside agraph_to_labeled_dag, before
            # _normalize_const_edges, to measure RTF precondition violations.
            try:
                dag = agraph_to_labeled_dag(indv, ledger=self.dedup.ledger)
            except Exception:  # noqa: BLE001
                # Conversion failed: count in ledger (full-rate, O(1))
                if self.dedup.ledger is not None:
                    self.dedup.ledger.record_conversion_failure()
                # Evaluate normally (only if unevaluated)
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)
                    if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                        self._best_fitness = indv.fitness
                continue

            # Record post-normalisation state (before canonicalisation).
            # violated_post should be 0 if _normalize_const_edges worked correctly.
            if self.dedup.ledger is not None:
                self.dedup.ledger.record_post(dag)

            # Probe hook — outside every timer and outside the canonicalisation
            # try/except, so it fires for every successfully converted DAG
            # (including atlas hits and subsequent failures).  Inert when no
            # probe is installed: one is-not-None check, nothing else.
            import experiments.models.equivalence_probe as _ep_bingo  # noqa: PLC0415

            if _ep_bingo.ACTIVE_PROBE is not None:
                _ep_bingo.ACTIVE_PROBE.record_bingo(dag, indv)

            # Shadow cardinality sketches — fed from the SAME LabeledDAG the
            # deduplication key sees, so the two streams are identical.  Sits
            # outside every timer: this is instrumentation, not arm cost.
            self.dedup.record_shadow(dag)

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
                    if self.dedup.ledger is not None:
                        self.dedup.ledger.record_atlas_hit(dag)
                else:
                    self.dedup.atlas_misses += 1

            # Population dedup needs the full canonical string (not just hash)
            need_canonical_str = canon_hash is None or self._enforce_dedup
            if need_canonical_str and canonical is None:
                # No atlas or atlas miss: compute this arm's representation
                # string (canonical string, or fixed-order serialisation for
                # the "hash" arm).  Its cost lands in canon_time_total either
                # way; metadata.representation disambiguates.
                t0_canon = time.perf_counter()
                try:
                    canonical = self.dedup.representation_string(dag, indv)
                except Exception as _exc:  # noqa: BLE001
                    if self.dedup.ledger is not None:
                        from isalsr.core.canonical import CanonicalTimeoutError

                        if isinstance(_exc, CanonicalTimeoutError):
                            self.dedup.ledger.record_timeout(dag)
                        else:
                            self.dedup.ledger.record_canon_raised(dag)
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
                    if self._use_age_penalty:
                        indv.genetic_age = _DUPLICATE_AGE_PENALTY
                    continue

                # Stale duplicate re-entry: individual was previously penalized
                # (age >= _DUPLICATE_AGE_PENALTY) but its canonical is no longer
                # held by a living member (original was evicted by selection).
                # Reset age so the individual competes fairly.
                if indv.genetic_age >= _DUPLICATE_AGE_PENALTY:
                    indv.genetic_age = 0

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
                    if self._use_age_penalty:
                        indv.genetic_age = _DUPLICATE_AGE_PENALTY
                    continue

                self.dedup.canonical_seen.add(canon_hash)
                self.dedup.n_unique += 1
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)
                if np.isfinite(indv.fitness) and indv.fitness < self._best_fitness:
                    self._best_fitness = indv.fitness


def purge_penalized(population: list) -> list:
    """Remove age-penalized duplicates from a population list.

    Bingo's AgeFitness tournament selection doesn't prioritize removing
    inf-fitness individuals — it removes targets randomly across the
    combined pool.  Penalized duplicates can survive when non-penalized
    individuals are also removed via Pareto dominance.  Call this after
    ``island.evolve()`` to guarantee all duplicates are purged.

    Args:
        population: List of Bingo AGraph individuals (modified in place).

    Returns:
        The filtered list (same object, for convenience).
    """
    i = 0
    while i < len(population):
        if population[i].genetic_age >= _DUPLICATE_AGE_PENALTY:
            population.pop(i)
        else:
            i += 1
    return population


class IsalSRBingoRunner(ModelRunner):
    """Runs Bingo with IsalSR canonical deduplication."""

    #: Deduplication key mode; overridden by the "hash" arm subclass.
    KEY_MODE: str = "canonical"

    def __init__(self, config: BingoConfig | None = None, atlas: Any = None):
        self._config = config or BingoConfig()
        # The atlas maps DAGs to CANONICAL hashes, so it is only sound for the
        # canonical arm.  The hash arm must compute its own key every time.
        self._atlas = atlas if self.KEY_MODE == "canonical" else None
        self.last_shadow: dict[str, float] = {}

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

        ledger = FallbackLedger()

        # Shadow counters default ON for the canonical arm (they are what the
        # R1.4 answer measures) and OFF for the hash arm, which already keys on
        # a fixed order.  ``shadow_hash: false`` in the YAML disables them.
        shadow_hash = bool(config.get("shadow_hash", self.KEY_MODE == "canonical"))

        dedup = _CanonicalDeduplicator(
            use_fast_canonical=cfg.use_fast_canonical,
            timeout=cfg.canonicalization_timeout,
            atlas=self._atlas,
            ledger=ledger,
            key_mode=self.KEY_MODE,
            shadow_hash=shadow_hash,
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
        convergence_data = evaluation.convergence_data  # type: ignore[union-attr]

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
        if ledger.enabled:
            log.info("FallbackLedger: %s", ledger.to_dict())
        self.last_ledger: FallbackLedger = ledger
        self.last_shadow = dedup.shadow_counts()
        if self.last_shadow:
            log.info(
                "Shadow cardinalities: %s (serialisation failures=%d)",
                self.last_shadow,
                dedup.n_shadow_failures,
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
            convergence_data=convergence_data,
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


class HashBingoRunner(IsalSRBingoRunner):
    """Runs Bingo with naive host-native-serialisation deduplication.

    The ``hash`` arm of reviewer comment R1.4. It differs from
    :class:`IsalSRBingoRunner` in exactly one thing: the deduplication key is the
    hash of the host's own utilised ``command_array`` rows, serialised in the
    host's own row order — sound but incomplete — instead of the hash of the
    complete canonical string. The evaluation hook, the VarAnd clone detection
    via ``_parent_ids``, the duplicate penalty, the counters and the trajectory
    snapshots are inherited unchanged, which is what makes the paired comparison
    valid.

    Set ``KEY_MODE = "hash"`` on a subclass to key on a fixed order over the
    *adapter's* output instead; that rung concedes IsalSR's own renumbering and
    is therefore a steel-manned, not a naive, baseline.
    """

    KEY_MODE = HASH_ARM_KEY_MODE

    @property
    def variant(self) -> str:
        return "hash"
