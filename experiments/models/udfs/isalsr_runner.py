"""IsalSR-enhanced UDFS runner.

Wraps UDFS DAGRegressor with canonical string deduplication.
For each CompGraph evaluated by UDFS, converts to LabeledDAG,
computes the pruned canonical string, and skips isomorphic duplicates.

Strategy: Monkey-patch the module-level `evaluate_cgraph` function
during the UDFS run to intercept each graph evaluation. This is the
cleanest subclass-only approach (no vendored code modifications).
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

# Ensure vendored DAG_search is importable
_vendor_dir = str(Path(__file__).parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

import DAG_search.dag_search as dag_search_module  # noqa: E402
from DAG_search.dag_search import DAGRegressor  # noqa: E402

from experiments.models.alphabet_guard import validate_udfs_operators
from experiments.models.base_runner import ModelRunner
from experiments.models.fallback_ledger import FallbackLedger
from experiments.models.stage_d_trace import StageDTracer
from experiments.models.structural_scope import is_structural, nonstructural_key
from experiments.models.udfs.adapter import compgraph_to_labeled_dag
from experiments.models.udfs.config import UDFSConfig
from experiments.models.udfs.runner import TrajectorySnapshot, UDFSRawResult
from isalsr.baselines import FixedOrder, HyperLogLog, fixed_order_hash, serialise  # noqa: E402
from isalsr.baselines.host_native import (  # noqa: E402
    HostNativeRecord,
    host_native_hash,
    host_native_serialise,
)

log = logging.getLogger(__name__)

# Deduplication key modes.
#   "canonical"   -- the IsalSR arm: complete labeled-DAG isomorphism invariant.
#   "host_native" -- the naive baseline of reviewer comment R1.4: the host's own
#                    CompGraph in the host's own ``node_dict`` key order.
#   "hash"        -- the steel-manned second rung: a fixed order over the
#                    *adapter's* output, which concedes IsalSR's own renumbering.
# Everything else about the arms is identical by construction.
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

# RunLog field name for the host-native shadow sketch.  Unlike the three above it
# is keyed on the host's own ``node_dict`` entries in the host's own key order,
# so it is the only shadow counter that is free of the adapter's renumbering.
SHADOW_HOST_NATIVE_FIELD = "shadow_distinct_host_native"
SHADOW_FAILURES_FIELD = "n_shadow_failures"


def udfs_host_native_records(cgraph: Any) -> list[HostNativeRecord]:
    """Extract host-native records from a UDFS ``CompGraph``.

    Iterates ``cg.node_dict`` in **its own key order** and emits, per node, the
    node's key, its UDFS operation string, and its child list exactly as UDFS
    stores it.  ``cg.eval_order`` is deliberately not consulted: it is a
    *computed* topological order, and keying on it would reintroduce the partial
    canonicalisation that this baseline exists to avoid.

    Every entry of ``node_dict`` is emitted, including unused ``'inp'`` and
    ``'const'`` terminals.  Those terminals are not dead code: the UDFS adapter
    creates a ``VAR``/``CONST`` node for each of them regardless of use, so
    dropping them here would let two CompGraphs with different constant budgets
    share a key while their adapter outputs, and hence their canonical strings,
    differ -- a soundness violation.

    Args:
        cgraph: A UDFS ``CompGraph``.

    Returns:
        The host-native records in ``node_dict`` order.
    """
    return [
        (int(key), str(op), tuple(int(child) for child in children))
        for key, (children, op) in cgraph.node_dict.items()
    ]


class _KeyResolution(NamedTuple):
    """Outcome of resolving one candidate's deduplication key.

    Attributes:
        canon_hash: The live, process-local key, or ``None`` on failure.
        representation: The string the key was taken of, or ``None`` on an atlas
            hit (where no string is computed) or on failure.
        t_canon_s: Seconds spent resolving the key.
        fallback: One of ``none``, ``atlas_hit``, ``timeout``, ``canon_raised``.
    """

    canon_hash: int | None
    representation: str | None
    t_canon_s: float
    fallback: str


class _CanonicalDeduplicator:
    """Tracks canonical strings and deduplication statistics.

    Uses hash-based storage (``set[int]``) instead of storing full canonical
    strings.  This reduces per-entry memory from ~150 bytes (``set[str]``)
    to ~28 bytes (``set[int]``), preventing OOM on long runs.

    The 64-bit Python hash gives collision probability < 3×10⁻⁶ for 10 M
    entries (birthday bound n²/2⁶⁵), which is negligible for our use case.
    """

    def __init__(
        self,
        use_fast_canonical: bool = True,
        timeout: float = 60.0,
        snapshot_freq: int = 1000,
        t0: float = 0.0,
        atlas: Any = None,
        ledger: FallbackLedger | None = None,
        key_mode: str = "canonical",
        shadow_hash: bool = False,
        dedup_enabled: bool = True,
    ):
        if key_mode not in KEY_MODES:
            raise ValueError(f"Unknown key_mode: {key_mode!r} (expected one of {KEY_MODES})")
        self.use_fast_canonical = use_fast_canonical
        # Suppression switch for the C3 control arm.  When False the wrapper is
        # still installed, the conversion and the canonicalisation still run and
        # every counter still counts, but no candidate is ever answered from
        # ``canonical_seen`` -- the host evaluates all of them.  This isolates
        # "the wrapper perturbs the search" from "dedup changes the search".
        self.dedup_enabled = dedup_enabled
        self.timeout = timeout
        self.atlas = atlas  # AtlasLookup | None
        self.ledger: FallbackLedger | None = ledger
        # Stage-D detailed trace (EXECUTION-PLAN §4.4 D2).  Constructed from the
        # environment; inert unless ISALSR_STAGE_D_TRACE is set, which the worker
        # does on exactly one cell.  Every hook below is one attribute read when
        # disabled, and no file is created.
        self.tracer: StageDTracer = StageDTracer.from_env()
        self.key_mode = key_mode
        # Shadow distinct-cardinality sketches over the full candidate stream.
        # HyperLogLog(p=14) is ~16 KB each and constant in stream length; an
        # exact set[int] would cost 1-2 GB at 10^7 candidates.
        self._shadow: dict[FixedOrder, HyperLogLog] = (
            {order: HyperLogLog(p=SHADOW_HLL_PRECISION) for order in SHADOW_FIELDS}
            if shadow_hash
            else {}
        )
        # Fourth sketch, keyed on the host's own CompGraph instead of the
        # adapter's LabeledDAG.  Kept out of ``_shadow`` because it consumes a
        # different object; same precision, same constant memory.
        self._shadow_host_native: HyperLogLog | None = (
            HyperLogLog(p=SHADOW_HLL_PRECISION) if shadow_hash else None
        )
        # Whether any host object was ever offered to ``record_shadow``.  Call
        # sites that only have the DAG (unit tests, offline replay) leave the
        # host-native counter undefined rather than reporting a spurious 0.
        self._host_native_offered: bool = False
        self.n_shadow_failures: int = 0
        self.canonical_seen: set[int] = set()
        self.n_total = 0
        self.n_unique = 0
        self.n_skipped = 0
        # Zero-internal-node candidates: outside the invariant's domain, so
        # scored but never deduplicated and excluded from rho.  Expected to
        # stay 0 for UDFS (n_calc_nodes = 5 campaign-wide); a non-zero value
        # means the search reached a configuration this campaign did not
        # anticipate.  See experiments/models/structural_scope.py.
        self.n_nonstructural = 0
        self.canon_time_total = 0.0
        # Adapter conversion cost (host CompGraph -> LabeledDAG).  Inside the
        # wall-clock budget and part of the method, so it must be subtracted
        # from search time and charged as overhead.
        self.conversion_time_total = 0.0
        # Shadow sketch cost.  Inside the wall-clock budget but pure audit
        # instrumentation: subtracted from search time, never charged as
        # overhead.  Stays exactly 0.0 when the sketches are disabled.
        self.shadow_time_total = 0.0
        self._snapshot_freq = snapshot_freq
        self._t0 = t0
        self._best_loss = float("inf")
        self.snapshots: list[TrajectorySnapshot] = []
        self._original_evaluate: Any = None
        # Atlas-specific stats
        self.atlas_hits: int = 0
        self.atlas_misses: int = 0
        self.atlas_lookup_time: float = 0.0
        self.canon_fallback_time: float = 0.0

    def _maybe_snapshot(self) -> None:
        """Append a trajectory snapshot if it's time."""
        if self.n_total % self._snapshot_freq == 0:
            self.snapshots.append(
                TrajectorySnapshot(
                    timestamp_s=time.perf_counter() - self._t0,
                    total_evals=self.n_total,
                    best_loss=self._best_loss,
                )
            )

    def representation_string(self, dag: Any, host: Any = None) -> str:
        """Return the string whose hash is this arm's deduplication key.

        Args:
            dag: The candidate ``LabeledDAG``, as produced by the host adapter
                (so the T16 SUB/DIV decomposition is already applied).
            host: The originating UDFS ``CompGraph``.  Required by the
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
                raise ValueError("key_mode='host_native' requires the host CompGraph")
            return host_native_serialise(udfs_host_native_records(host))
        if self.key_mode == "hash":
            return serialise(dag, HASH_ARM_ORDER)
        if self.use_fast_canonical:
            from isalsr.core.canonical import fast_canonical_string  # noqa: PLC0415

            return fast_canonical_string(dag, timeout=self.timeout)
        from isalsr.core.canonical import canonical_string  # noqa: PLC0415

        return canonical_string(dag, timeout=self.timeout)

    def record_shadow(self, dag: Any, host: Any = None) -> None:
        """Feed one candidate into every enabled cardinality sketch.

        A serialisation failure is counted and ignored: the shadow counters are
        instrumentation and must never change the arm's search behaviour.

        Args:
            dag: The same ``LabeledDAG`` object the deduplication key sees, used
                by the three adapter-order sketches.
            host: The originating UDFS ``CompGraph``, used by the host-native
                sketch.  The host object is required because avoiding the
                adapter is the whole point of that counter; it is never
                re-derived from *dag*.  When ``None``, the host-native sketch is
                left untouched and stays unreported.
        """
        if not self._shadow:
            return
        # Timed from here, not from the call site: with the sketches disabled the
        # method returns above having done no work, so ``shadow_time_total``
        # stays exactly 0.0 rather than accumulating call overhead.
        t0 = time.perf_counter()
        for order, sketch in self._shadow.items():
            try:
                sketch.add(fixed_order_hash(dag, order))
            except Exception:  # noqa: BLE001
                self.n_shadow_failures += 1
        if host is not None and self._shadow_host_native is not None:
            self._host_native_offered = True
            try:
                self._shadow_host_native.add(host_native_hash(udfs_host_native_records(host)))
            except Exception:  # noqa: BLE001
                self.n_shadow_failures += 1
        self.shadow_time_total += time.perf_counter() - t0

    def shadow_counts(self) -> dict[str, float]:
        """Return the shadow distinct-cardinality estimates by RunLog field name.

        The host-native entry is present only if at least one host object was
        offered to :meth:`record_shadow`; otherwise the counter has no defined
        value and the RunLog field stays ``None``.
        """
        if not self._shadow:
            return {}
        counts = {SHADOW_FIELDS[order]: sketch.count() for order, sketch in self._shadow.items()}
        if self._shadow_host_native is not None and self._host_native_offered:
            counts[SHADOW_HOST_NATIVE_FIELD] = self._shadow_host_native.count()
        # Ships with the cardinalities, not beside them: a count is only
        # interpretable together with the number of records that failed to
        # serialise.  Reported only when the sketches actually ran, so a
        # shadow-off run leaves the field None rather than claiming zero
        # failures for work it never did.
        counts[SHADOW_FAILURES_FIELD] = self.n_shadow_failures
        return counts

    def _resolve_canonical_hash(self, labeled_dag: Any, host: Any = None) -> int | None:
        """Resolve the canonical hash for a DAG: atlas fast-path or online fallback.

        Thin wrapper over :meth:`_resolve_key` kept for callers that need only
        the key (``experiments/scripts/verify_udfs_dedup.py``).

        Args:
            labeled_dag: The adapter's ``LabeledDAG`` for the candidate.
            host: The originating UDFS ``CompGraph``, needed by the
                ``"host_native"`` key mode.

        Returns:
            The deduplication key, or ``None`` if key resolution failed.
        """
        return self._resolve_key(labeled_dag, host).canon_hash

    def _resolve_key(self, labeled_dag: Any, host: Any = None) -> _KeyResolution:
        """Resolve this arm's deduplication key: atlas fast-path or online fallback.

        Updates atlas/fallback timing stats as a side effect.

        Args:
            labeled_dag: The adapter's ``LabeledDAG`` for the candidate.
            host: The originating UDFS ``CompGraph``, needed by the
                ``"host_native"`` key mode.

        Returns:
            The key, the representation string that produced it (``None`` on an
            atlas hit or a failure), the seconds spent, and the fallback path.
        """
        t0 = time.perf_counter()
        canon_hash: int | None = None

        # Atlas fast-path
        if self.atlas is not None:
            canon_hash, was_hit = self.atlas.lookup_dag(labeled_dag)
            dt = time.perf_counter() - t0
            self.atlas_lookup_time += dt
            if was_hit:
                self.atlas_hits += 1
                self.canon_time_total += dt
                if self.ledger is not None:
                    self.ledger.record_atlas_hit(labeled_dag)
                return _KeyResolution(canon_hash, None, dt, "atlas_hit")
            self.atlas_misses += 1

        # Online fallback: compute this arm's representation string (canonical
        # string, or fixed-order serialisation for the "hash" arm).  Its cost
        # lands in canon_time_total either way; metadata.representation
        # disambiguates.
        t0_canon = time.perf_counter()
        try:
            canonical = self.representation_string(labeled_dag, host)
        except Exception as _exc:  # noqa: BLE001
            from isalsr.core.canonical import CanonicalTimeoutError

            is_timeout = isinstance(_exc, CanonicalTimeoutError)
            if self.ledger is not None:
                if is_timeout:
                    self.ledger.record_timeout(labeled_dag)
                else:
                    self.ledger.record_canon_raised(labeled_dag)
            self.canon_fallback_time += time.perf_counter() - t0_canon
            dt_canon = time.perf_counter() - t0
            self.canon_time_total += dt_canon
            return _KeyResolution(None, None, dt_canon, "timeout" if is_timeout else "canon_raised")

        self.canon_fallback_time += time.perf_counter() - t0_canon
        dt_canon = time.perf_counter() - t0
        self.canon_time_total += dt_canon
        return _KeyResolution(hash(canonical), canonical, dt_canon, "none")

    def _traced_evaluate(
        self,
        cgraph: Any,
        x: Any,
        loss_fkt: Any,
        opt_mode: str,
        loss_thresh: Any,
    ) -> Any:
        """Call the host's evaluator, charging its wall time to the Stage-D tracer.

        The two ``perf_counter`` calls run only while the tracer has selected the
        current candidate for sampling, so with tracing off this costs one
        attribute read on top of the direct call it replaces.

        Args:
            cgraph: The UDFS ``CompGraph`` under evaluation.
            x: The design matrix passed through unchanged.
            loss_fkt: The loss functional passed through unchanged.
            opt_mode: The constant-optimisation mode passed through unchanged.
            loss_thresh: The early-exit threshold passed through unchanged.

        Returns:
            Whatever the original ``evaluate_cgraph`` returned.
        """
        if not self.tracer.sampling:
            return self._original_evaluate(cgraph, x, loss_fkt, opt_mode, loss_thresh)
        t0 = time.perf_counter()
        result = self._original_evaluate(cgraph, x, loss_fkt, opt_mode, loss_thresh)
        self.tracer.note_eval_time(time.perf_counter() - t0)
        return result

    def wrap_evaluate_cgraph(self, original_fn: Any) -> Any:
        """Create a wrapper around evaluate_cgraph with canonical dedup."""
        self._original_evaluate = original_fn

        def wrapped(cgraph, X, loss_fkt, opt_mode="grid_zoom", loss_thresh=None):  # noqa: N803
            self.n_total += 1

            # Stage-D trace: open a candidate.  Returns False (and does nothing
            # else) unless the tracer is enabled AND this candidate is on the
            # deterministic 1-in-N sampling grid.
            tracer = self.tracer
            tracer.begin()

            # record_pre is called inside compgraph_to_labeled_dag, before
            # _normalize_const_edges, to measure RTF precondition violations.
            t_conv = time.perf_counter()
            try:
                labeled_dag = compgraph_to_labeled_dag(cgraph, ledger=self.ledger)
            except Exception:  # noqa: BLE001
                # A refused candidate still consumed the budget, so the failed
                # conversion is charged like a successful one.
                self.conversion_time_total += time.perf_counter() - t_conv
                # Conversion failed: count in ledger (full-rate, O(1))
                if self.ledger is not None:
                    self.ledger.record_conversion_failure()
                # Evaluate normally
                result = self._traced_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
                consts, loss = result
                if np.isfinite(loss) and loss < self._best_loss:
                    self._best_loss = loss
                self._maybe_snapshot()
                tracer.record(fallback="conversion_failure")
                return result
            self.conversion_time_total += time.perf_counter() - t_conv

            # Record post-normalisation state (before canonicalisation).
            if self.ledger is not None:
                self.ledger.record_post(labeled_dag)

            # Probe hook — inert in production (one is-not-None check).
            # Only fires in the fast-canonical branch when a gate driver
            # installs a probe before the run.
            if self.use_fast_canonical:
                import experiments.models.equivalence_probe as _ep  # noqa: PLC0415

                if _ep.ACTIVE_PROBE is not None:
                    _ep.ACTIVE_PROBE.record_udfs(labeled_dag, cgraph)

            # Shadow cardinality sketches — fed from the SAME LabeledDAG the
            # deduplication key sees, so the two streams are identical.  The
            # CompGraph goes with it: the host-native sketch must see the host's
            # own node_dict, not the adapter's renumbering.  Sits outside every
            # timer: this is instrumentation, not arm cost.
            self.record_shadow(labeled_dag, cgraph)

            key = self._resolve_key(labeled_dag, cgraph)
            canon_hash = key.canon_hash

            if canon_hash is None:
                # Canonicalization failed — evaluate normally
                result = self._traced_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
                consts, loss = result
                if np.isfinite(loss) and loss < self._best_loss:
                    self._best_loss = loss
                self._maybe_snapshot()
                tracer.record(dag=labeled_dag, t_canon=key.t_canon_s, fallback=key.fallback)
                return result

            # --- Structural scope (D3, 2026-08-06) -------------------------
            # Zero internal nodes => a bare input variable, whose canonical
            # string is "" for every m and every output variable.  Keying the
            # dedup on "" would make UDFS SKIP evaluating x_i for i > 0 once
            # x_0 had been seen.  Outside the invariant's domain (0! = 1: no
            # redundancy to collapse), so score it and count nothing.
            #
            # In practice unreachable here -- all seven campaign configs pin
            # n_calc_nodes = 5, so UDFS cannot emit k = 0 -- but the guard is
            # symmetric with Bingo's on purpose: an arm-specific dedup rule is
            # exactly the kind of asymmetry the paired design cannot tolerate.
            if not is_structural(labeled_dag):
                # k=0: "" is not a usable key -- it equates x_0 with x_1, so
                # UDFS would mark the second a duplicate and never evaluate it.
                # Substitute a sound key and fall through: these are ordinary
                # redundancy and belong in rho.  See structural_scope.py.
                self.n_nonstructural += 1
                canon_hash = hash(nonstructural_key(labeled_dag))

            is_duplicate = canon_hash in self.canonical_seen
            if is_duplicate and self.dedup_enabled:
                self.n_skipped += 1
                n_consts = cgraph.n_consts
                dummy_consts = np.zeros(n_consts) if n_consts > 0 else np.array([])
                self._maybe_snapshot()
                tracer.record(
                    dag=labeled_dag,
                    representation=key.representation,
                    representation_kind=self.key_mode,
                    representation_hash=canon_hash,
                    t_canon=key.t_canon_s,
                    fallback=key.fallback,
                    dedup_hit=True,
                )
                return dummy_consts, np.inf

            # Counters are unconditional: with suppression off the arm still
            # reports rho = n_total / n_unique, which is then a measurement of
            # the redundancy the host produced without acting on it.
            if not is_duplicate:
                self.canonical_seen.add(canon_hash)
                self.n_unique += 1
            result = self._traced_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
            consts, loss = result
            if np.isfinite(loss) and loss < self._best_loss:
                self._best_loss = loss
            self._maybe_snapshot()
            tracer.record(
                dag=labeled_dag,
                representation=key.representation,
                representation_kind=self.key_mode,
                representation_hash=canon_hash,
                t_canon=key.t_canon_s,
                fallback=key.fallback,
                dedup_hit=is_duplicate,
            )
            return result

        return wrapped


@contextmanager
def _patched_evaluate(deduplicator: _CanonicalDeduplicator):
    """Context manager that patches evaluate_cgraph with dedup wrapper."""
    original = dag_search_module.evaluate_cgraph
    dag_search_module.evaluate_cgraph = deduplicator.wrap_evaluate_cgraph(original)
    try:
        yield deduplicator
    finally:
        dag_search_module.evaluate_cgraph = original


class IsalSRUDFSRunner(ModelRunner):
    """Runs UDFS with IsalSR canonical deduplication.

    Args:
        config: UDFS configuration; a default one is built when omitted.
        atlas: Optional ``AtlasLookup`` for O(1) canonical lookup. Ignored
            unless ``KEY_MODE`` is ``"canonical"``.
        dedup_enabled: When ``False``, the wrapper and the canonicalisation stay
            installed but no candidate is suppressed -- the ``nodedup`` control
            arm of check C3. The reported ``variant`` changes accordingly, so the
            run lands in its own output directory.
    """

    #: Deduplication key mode; overridden by the "hash" arm subclass.
    KEY_MODE: str = "canonical"

    def __init__(
        self,
        config: UDFSConfig | None = None,
        atlas: Any = None,
        dedup_enabled: bool = True,
    ):
        # UDFS takes no operator set from the configuration: its search
        # enumerates the vendored node table, so that table is what has to be
        # encodable.  Checked here rather than assumed, because an operator
        # without an image in the alphabet of Definition 3.2 is refused by the
        # adapter, counted as a conversion failure and then evaluated without
        # deduplication, which depresses the reduction factor silently.
        validate_udfs_operators()
        self._config = config or UDFSConfig()
        # The atlas maps DAGs to CANONICAL hashes, so it is only sound for the
        # canonical arm.  The hash arm must compute its own key every time.
        self._atlas = atlas if self.KEY_MODE == "canonical" else None
        self._dedup_enabled = dedup_enabled
        self.last_shadow: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "udfs"

    @property
    def variant(self) -> str:
        return "isalsr" if self._dedup_enabled else "nodedup"

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        config: dict[str, Any],
    ) -> UDFSRawResult:
        """Run UDFS with IsalSR canonical deduplication."""
        cfg = UDFSConfig.from_dict(config) if config else self._config
        kwargs = cfg.to_dag_regressor_kwargs()

        regressor = DAGRegressor(**kwargs)
        regressor.random_state = seed

        t0 = time.perf_counter()
        ledger = FallbackLedger()
        # Shadow counters default ON for the canonical arm (they are what the
        # R1.4 answer measures) and OFF for the hash arm, which already keys on
        # a fixed order.  ``shadow_hash: false`` in the YAML disables them.
        shadow_hash = bool(config.get("shadow_hash", self.KEY_MODE == "canonical"))

        dedup = _CanonicalDeduplicator(
            use_fast_canonical=cfg.use_fast_canonical,
            timeout=cfg.canonicalization_timeout,
            snapshot_freq=cfg.snapshot_frequency,
            t0=t0,
            atlas=self._atlas,
            ledger=ledger,
            key_mode=self.KEY_MODE,
            shadow_hash=shadow_hash,
            dedup_enabled=self._dedup_enabled,
        )

        with _patched_evaluate(dedup), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor.fit(x_train, y_train, verbose=0)
        wall_clock = time.perf_counter() - t0

        # Extract results
        total_evals = getattr(regressor, "total_evals", 0)
        best_sympy = None
        y_pred_train = np.full(len(y_train), np.nan)
        y_pred_test = np.full(len(y_test), np.nan)
        best_loss = float("inf")
        n_top = 0

        if hasattr(regressor, "cgraph") and regressor.cgraph is not None:
            try:
                best_sympy = regressor.model()
            except Exception:  # noqa: BLE001
                log.debug("Failed to extract SymPy model", exc_info=True)

            try:
                y_pred_train = regressor.predict(x_train)
                y_pred_test = regressor.predict(x_test)
            except Exception:  # noqa: BLE001
                log.debug("Prediction failed", exc_info=True)

        if hasattr(regressor, "results") and regressor.results:
            losses = regressor.results.get("losses", [])
            if losses:
                best_loss = float(min(losses))
            n_top = len(regressor.results.get("graphs", []))

        # Every block the wrapper runs inside the budget is removed, not just the
        # canonicalisation: conversion and the shadow sketches were previously
        # booked as search time.
        search_only = max(
            0.0,
            wall_clock
            - dedup.canon_time_total
            - dedup.conversion_time_total
            - dedup.shadow_time_total,
        )

        log.info(
            "IsalSR UDFS: total=%d unique=%d skipped=%d canon=%.2fs atlas_hits=%d misses=%d",
            dedup.n_total,
            dedup.n_unique,
            dedup.n_skipped,
            dedup.canon_time_total,
            dedup.atlas_hits,
            dedup.atlas_misses,
        )
        if ledger.enabled:
            log.info("FallbackLedger: %s", ledger.to_dict())
        self.last_ledger: FallbackLedger = ledger
        # Stage-D D2: flush the trace and write the four derived artefacts.
        # No-op unless ISALSR_STAGE_D_TRACE was set for this cell.
        dedup.tracer.close(
            ledger=ledger,
            run={
                "method": "udfs",
                "variant": self.variant,
                "key_mode": self.KEY_MODE,
                "problem": config.get("problem_name"),
                "seed": seed,
            },
        )
        self.last_shadow = dedup.shadow_counts()
        if self.last_shadow:
            log.info(
                "Shadow cardinalities: %s (serialisation failures=%d)",
                self.last_shadow,
                dedup.n_shadow_failures,
            )

        return UDFSRawResult(
            wall_clock_s=wall_clock,
            seed=seed,
            best_sympy=best_sympy,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            total_evals=total_evals,
            best_loss=best_loss,
            n_top_graphs=n_top,
            trajectory_snapshots=dedup.snapshots,
            n_total_dags=dedup.n_total,
            n_unique_canonical=dedup.n_unique,
            n_skipped=dedup.n_skipped,
            n_nonstructural=dedup.n_nonstructural,
            canonicalization_time_s=dedup.canon_time_total,
            search_only_time_s=search_only,
            conversion_time_s=dedup.conversion_time_total,
            shadow_time_s=dedup.shadow_time_total,
            atlas_hits=dedup.atlas_hits,
            atlas_misses=dedup.atlas_misses,
            atlas_lookup_time_s=dedup.atlas_lookup_time,
            canon_fallback_time_s=dedup.canon_fallback_time,
        )


class HashUDFSRunner(IsalSRUDFSRunner):
    """Runs UDFS with naive host-native-serialisation deduplication.

    The ``hash`` arm of reviewer comment R1.4. It differs from
    :class:`IsalSRUDFSRunner` in exactly one thing: the deduplication key is the
    hash of the host's own ``CompGraph``, serialised in the host's own
    ``node_dict`` key order — sound but incomplete — instead of the hash of the
    complete canonical string. The patched ``evaluate_cgraph`` wrapper, the
    duplicate return value, the counters and the trajectory snapshots are
    inherited unchanged, which is what makes the paired comparison valid.

    Set ``KEY_MODE = "hash"`` on a subclass to key on a fixed order over the
    *adapter's* output instead; that rung concedes IsalSR's own renumbering and
    is therefore a steel-manned, not a naive, baseline.
    """

    KEY_MODE = HASH_ARM_KEY_MODE

    @property
    def variant(self) -> str:
        return "hash"
