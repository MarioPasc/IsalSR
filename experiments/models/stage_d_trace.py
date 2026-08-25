"""Stage-D detailed candidate-stream tracer (EXECUTION-PLAN §4.4, deliverable D2).

The tracer persists a *replayable* slice of the candidate stream that the IsalSR
deduplication wrapper sees, on exactly one ``(method, problem, seed)`` cell of the
Stage-D certification.  It is driven entirely by three environment variables
exported by ``slurm/c2_stage_d/worker.sh``:

``ISALSR_STAGE_D_TRACE``
    ``1`` enables the tracer.  Anything falsy (unset, ``0``, ``false``, ``no``)
    leaves every hook a single attribute read.
``ISALSR_STAGE_D_TRACE_DIR``
    Directory the five artefacts are written into.
``ISALSR_STAGE_D_TRACE_SAMPLE_RATE``
    Deterministic 1-in-N sampling rate (default 100).  Candidate index ``i`` is
    sampled iff ``i % N == 0``; no RNG is involved, so the persisted stream is a
    reproducible function of the run.

Five artefacts
--------------
``candidates.jsonl``
    One JSON object per sampled candidate.  Carries the ``FixedOrder.INSERTION``
    serialisation, which :func:`isalsr.baselines.fixed_order_hash.deserialise`
    inverts losslessly, so the stream can be replayed offline (deliverable D3).
``canon_cost_hist.json``
    Canonicalisation-cost histogram stratified by ``k`` (feeds T10).
``fallback_ledger.md``
    The five T06 fallback rates sourced from
    :meth:`~experiments.models.fallback_ledger.FallbackLedger.to_dict`, plus
    worked examples of every observed residual post-normalisation violation.
``spot_check.json``
    20 candidates reservoir-sampled from the persisted stream, re-canonicalised
    in **pure Python** and compared byte-exact against the string the production
    engine emitted during the run.
``stream_size.md``
    Measured bytes/candidate and bytes/run, the 8,400-run projection, and the
    check against the FSCRATCH inode budget.

Persisted keys are **stable digests**, never ``hash()``
-------------------------------------------------------
:func:`isalsr.baselines.fixed_order_hash.fixed_order_hash` and
:func:`isalsr.baselines.host_native.host_native_hash` wrap CPython's builtin
``hash()``, which is SipHash keyed by ``PYTHONHASHSEED`` and therefore differs
between processes.  Those values are the *live* deduplication keys and must not
be persisted or replayed.  Everything this module writes to disk uses
:func:`isalsr.baselines.fixed_order_hash.fixed_order_digest` (BLAKE2b-64) or
:func:`canonical_digest` below, both of which are reproducible across processes
and machines.  ``canonical_hash`` is recorded alongside purely as a diagnostic
of what the live set held during *this* process; D3 never keys on it.

Sampling arithmetic (recorded here because the choice is a scientific decision)
------------------------------------------------------------------------------
EXECUTION-PLAN §11.1 (B9 row, 2026-08-03) measured 711,419 candidates in a 900 s
Bingo cell, i.e. 790.5 candidates/s.  A 12 h (43,200 s) Stage-D cell is therefore
of order ``790.5 × 43_200 ≈ 3.4e7`` candidates.  This schema was measured at
**571.7 B/record** over 8,180 records of a live 25 s Bingo run on the production
operator set (mean ``k`` 8.53, max 30), so a full-rate stream would be
``3.4e7 × 571.7 B ≈ 19.4 GB`` for a single cell, and ``json.dumps`` of such a
record costs ~7 µs, i.e. ``3.4e7 × 7 µs ≈ 240 s`` — 0.55 % of the 12 h budget.
That last figure is the disqualifying one: this same cell is the one that measures
per-DAG ``T_canon``/``T_eval`` for D1.7, so the instrumentation must not consume a
measurable share of the budget it measures.  At the worker's default rate of 100
the stream is ``3.4e5`` records ≈ 195 MB and ≈2.4 s of serialisation (0.006 % of
budget), which is why 100 is the default.  ``stream_size.md`` recomputes all of
this from the bytes the run actually wrote, rather than from these estimates.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.models.fallback_ledger import count_nonvar
from experiments.models.structural_scope import recorded_key
from isalsr.baselines.fixed_order_hash import (
    FixedOrder,
    SerialisationError,
    deserialise,
    fixed_order_digest,
    serialise,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

log = logging.getLogger(__name__)

__all__ = [
    "CANDIDATES_FILE",
    "CANON_COST_FILE",
    "FALLBACK_MD_FILE",
    "FALLBACK_PATHS",
    "SPOT_CHECK_FILE",
    "STREAM_SIZE_FILE",
    "StageDTraceConfig",
    "StageDTracer",
    "StorageBudget",
    "canonical_digest",
    "unreachable_nonvar",
]

# --------------------------------------------------------------------------- #
# Environment contract — must match slurm/c2_stage_d/worker.sh
# --------------------------------------------------------------------------- #
ENV_ENABLED = "ISALSR_STAGE_D_TRACE"
ENV_DIR = "ISALSR_STAGE_D_TRACE_DIR"
ENV_RATE = "ISALSR_STAGE_D_TRACE_SAMPLE_RATE"

DEFAULT_SAMPLE_RATE = 100

CANDIDATES_FILE = "candidates.jsonl"
CANON_COST_FILE = "canon_cost_hist.json"
FALLBACK_MD_FILE = "fallback_ledger.md"
SPOT_CHECK_FILE = "spot_check.json"
STREAM_SIZE_FILE = "stream_size.md"

#: The five T06 fallback paths, in the order they are reported.
FALLBACK_PATHS: tuple[str, ...] = (
    "violated_post",
    "timeout",
    "conversion_failure",
    "canon_raised",
    "atlas_hit",
)

#: Values accepted in ``CandidateRecord.fallback``.  ``violated_post`` is not a
#: control-flow path -- it is a property of the DAG -- so it is carried as its own
#: boolean field and only appears here for the worked-example index.
_CONTROL_FLOW_PATHS: frozenset[str] = frozenset(
    {"none", "timeout", "conversion_failure", "canon_raised", "atlas_hit"}
)

_FALSY = frozenset({"", "0", "false", "False", "no", "NO", "off", "OFF"})

#: Log-spaced histogram edges, 1 µs .. 1 s at four bins per decade.
_BIN_EDGES: tuple[float, ...] = tuple(1e-6 * (10.0 ** (i / 4.0)) for i in range(25))

#: Flush cadence.  Bounds the tail lost to a SLURM kill to ~1 000 records
#: (~550 kB) while costing 342 flushes over a 12 h cell.
_FLUSH_EVERY = 1_000

#: Worked examples retained per fallback path.  The deliverable asks for "at
#: least one"; three gives room to see whether a path has a single cause.
_MAX_EXAMPLES = 3

_SPOT_CHECK_SIZE = 20
_SPOT_CHECK_SEED = 0xD2


def canonical_digest(text: str) -> int:
    """Return a process-stable 64-bit BLAKE2b digest of *text*.

    Mirrors :func:`isalsr.baselines.fixed_order_hash.fixed_order_digest` so the
    canonical string and the fixed-order serialisations are keyed the same way in
    the persisted stream.

    Args:
        text: The string to digest.

    Returns:
        An unsigned 64-bit integer.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def unreachable_nonvar(dag: LabeledDAG) -> list[int]:
    """Return the non-VAR nodes of *dag* not reachable from any VAR node.

    The Round-Trip Fidelity precondition holds iff this list is empty.  This is
    the witness-producing counterpart of
    :func:`experiments.models.fallback_ledger.violates_precondition`, used to
    build the worked examples in ``fallback_ledger.md``.

    Args:
        dag: The DAG to inspect.

    Returns:
        Node identifiers, ascending.
    """
    n = dag.node_count
    visited = bytearray(n)
    queue: list[int] = []
    for i in range(n):
        if dag.node_label_unchecked(i) == NodeType.VAR:
            visited[i] = 1
            queue.append(i)
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nb in dag.out_neighbors_raw(node):
            if not visited[nb]:
                visited[nb] = 1
                queue.append(nb)
    return [i for i in range(n) if not visited[i] and dag.node_label_unchecked(i) != NodeType.VAR]


def _label_multiset(dag: LabeledDAG) -> dict[str, int]:
    """Return the node-label multiset of *dag* keyed by ``NodeType`` name.

    Args:
        dag: The DAG to inspect.

    Returns:
        Mapping from label name to count, sorted by label name.
    """
    counts = Counter(dag.node_label_unchecked(i).name for i in range(dag.node_count))
    return dict(sorted(counts.items()))


def _bin_index(value: float) -> int:
    """Return the histogram bin of *value* seconds.

    Bin 0 is the underflow (< 1 µs) and bin ``len(_BIN_EDGES)`` the overflow
    (>= 1 s); bin ``i`` for ``1 <= i <= len(_BIN_EDGES)-1`` covers
    ``[_BIN_EDGES[i-1], _BIN_EDGES[i])``.

    Args:
        value: Duration in seconds.

    Returns:
        The bin index.
    """
    if value < _BIN_EDGES[0]:
        return 0
    if value >= _BIN_EDGES[-1]:
        return len(_BIN_EDGES)
    return int(math.floor(4.0 * math.log10(value / _BIN_EDGES[0]))) + 1


@dataclass
class StorageBudget:
    """FSCRATCH headroom the stream-size projection is checked against.

    Attributes:
        inode_headroom: Free file slots on FSCRATCH.  Default from the live
            ``quota`` re-read of 2026-08-04 (155.4k used / 250.0k soft).
        space_headroom_gb: Free space on FSCRATCH in GB, or ``None`` when the
            figure has not been measured.  The 2026-08-04 capture records inodes
            only; the projection then reports the requirement without asserting
            on it.
        campaign_runs: Number of runs the campaign-wide projection multiplies by.
        files_per_trace: Artefacts this tracer writes per traced cell.
        provenance: Where the numbers came from, copied into the artefact.
    """

    inode_headroom: int = 94_600
    space_headroom_gb: float | None = None
    campaign_runs: int = 8_400
    files_per_trace: int = 5
    provenance: str = (
        "FSCRATCH 155.4k/250.0k files (94.6k headroom), live `quota` re-read "
        "2026-08-04, .claude/notes/review/tasks/audit.md §6.5.  No FSCRATCH "
        "space figure was captured; re-read live before Stage D."
    )


@dataclass(frozen=True)
class StageDTraceConfig:
    """Immutable configuration of a :class:`StageDTracer`.

    Attributes:
        enabled: Whether any artefact is produced.
        out_dir: Destination directory, or ``None`` when disabled.
        sample_rate: Deterministic 1-in-N rate; ``1`` persists every candidate.
        spot_check_size: Reservoir size for ``spot_check.json``.
        spot_check_seed: Seed of the reservoir RNG, recorded in the artefact so
            the draw is reproducible.
        max_examples: Worked examples retained per fallback path.
        budget: Storage headroom the projection is checked against.
    """

    enabled: bool = False
    out_dir: Path | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    spot_check_size: int = _SPOT_CHECK_SIZE
    spot_check_seed: int = _SPOT_CHECK_SEED
    max_examples: int = _MAX_EXAMPLES
    budget: StorageBudget = field(default_factory=StorageBudget)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> StageDTraceConfig:
        """Build a configuration from the Stage-D environment contract.

        A missing or falsy ``ISALSR_STAGE_D_TRACE``, or a missing
        ``ISALSR_STAGE_D_TRACE_DIR``, yields a disabled configuration.  An
        unparseable or non-positive sample rate falls back to the default with a
        warning rather than aborting the run.

        Args:
            env: Environment mapping.  Defaults to :data:`os.environ`.

        Returns:
            The configuration.
        """
        source = os.environ if env is None else env
        if source.get(ENV_ENABLED, "0").strip() in _FALSY:
            return cls()
        raw_dir = source.get(ENV_DIR, "").strip()
        if not raw_dir:
            log.warning("%s is set but %s is empty; Stage-D trace disabled", ENV_ENABLED, ENV_DIR)
            return cls()
        raw_rate = source.get(ENV_RATE, str(DEFAULT_SAMPLE_RATE)).strip()
        try:
            rate = int(raw_rate)
        except ValueError:
            log.warning(
                "%s=%r is not an integer; using %d", ENV_RATE, raw_rate, DEFAULT_SAMPLE_RATE
            )
            rate = DEFAULT_SAMPLE_RATE
        if rate < 1:
            log.warning("%s=%d is not positive; using %d", ENV_RATE, rate, DEFAULT_SAMPLE_RATE)
            rate = DEFAULT_SAMPLE_RATE
        return cls(enabled=True, out_dir=Path(raw_dir), sample_rate=rate)


class _KStats:
    """Per-``k`` accumulator for the canonicalisation-cost histogram."""

    __slots__ = ("n", "n_dedup_hits", "canon", "evaluation")

    def __init__(self) -> None:
        self.n: int = 0
        self.n_dedup_hits: int = 0
        self.canon: _Series = _Series()
        self.evaluation: _Series = _Series()


class _Series:
    """Streaming summary plus a log-spaced histogram of a non-negative duration."""

    __slots__ = ("n", "total", "minimum", "maximum", "bins")

    def __init__(self) -> None:
        self.n: int = 0
        self.total: float = 0.0
        self.minimum: float = math.inf
        self.maximum: float = 0.0
        self.bins: list[int] = [0] * (len(_BIN_EDGES) + 1)

    def add(self, value: float) -> None:
        """Fold one observation into the summary.

        Args:
            value: Duration in seconds; non-positive values are ignored so that
                "not measured" never contaminates the minimum.
        """
        if value <= 0.0:
            return
        self.n += 1
        self.total += value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value
        self.bins[_bin_index(value)] += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary.

        Returns:
            Count, sum, mean, min, max and the raw bin counts.
        """
        return {
            "n": self.n,
            "sum_s": self.total,
            "mean_s": (self.total / self.n) if self.n else 0.0,
            "min_s": (self.minimum if self.n else 0.0),
            "max_s": self.maximum,
            "bins": list(self.bins),
        }


class StageDTracer:
    """Persists a deterministic 1-in-N slice of the candidate stream.

    Construct one per run via :meth:`from_env` and hang it off the host's
    deduplicator.  When disabled every hook returns after a single attribute
    read and no file is created.

    The call protocol per candidate is:

    1. :meth:`begin` -- exactly once, as the candidate enters the wrapper.
    2. :meth:`note_eval_time` -- zero or one time, around the host's fitness call.
    3. :meth:`record` -- exactly once, on every exit path of the wrapper.

    Attributes:
        cfg: The immutable configuration.
        sampling: Whether the candidate currently being processed is sampled.
            Read directly by the hosts to skip timing work on unsampled
            candidates; always ``False`` when the tracer is disabled.
    """

    __slots__ = (
        "cfg",
        "sampling",
        "_dir",
        "_fh",
        "_index",
        "_n_sampled",
        "_eval_time",
        "_hist",
        "_reservoir",
        "_rng",
        "_examples",
        "_bytes",
        "_closed",
        "_write_failures",
        "_t_start",
        "_n_eligible",
    )

    def __init__(self, cfg: StageDTraceConfig) -> None:
        self.cfg = cfg
        self.sampling: bool = False
        self._dir: Path | None = None
        self._fh: Any = None
        self._index: int = 0
        self._n_sampled: int = 0
        self._eval_time: float = 0.0
        self._hist: dict[int, _KStats] = {}
        self._reservoir: list[dict[str, Any]] = []
        self._rng = random.Random(cfg.spot_check_seed)
        self._examples: dict[str, list[dict[str, Any]]] = {p: [] for p in FALLBACK_PATHS}
        self._bytes: int = 0
        self._closed: bool = False
        self._write_failures: int = 0
        self._t_start: float = time.time()
        self._n_eligible: int = 0
        if cfg.enabled and cfg.out_dir is not None:
            self._open(cfg.out_dir)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> StageDTracer:
        """Build a tracer from the Stage-D environment contract.

        Args:
            env: Environment mapping.  Defaults to :data:`os.environ`.

        Returns:
            An enabled tracer when ``ISALSR_STAGE_D_TRACE`` is truthy and a
            destination directory is given, otherwise an inert one.
        """
        return cls(StageDTraceConfig.from_env(env))

    def _open(self, out_dir: Path) -> None:
        """Create the output directory and open the JSONL handle.

        A failure here disables the tracer rather than aborting the run:
        instrumentation must never cost a 12 h cell.

        Args:
            out_dir: Destination directory.
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (out_dir / CANDIDATES_FILE).open("w", encoding="utf-8")
        except OSError:
            log.exception("Stage-D trace could not open %s; tracing disabled", out_dir)
            self.cfg = StageDTraceConfig()
            self._fh = None
            return
        self._dir = out_dir
        atexit.register(self.close)
        log.info(
            "Stage-D trace enabled: dir=%s sample_rate=1-in-%d",
            out_dir,
            self.cfg.sample_rate,
        )

    @property
    def enabled(self) -> bool:
        """Whether the tracer writes anything."""
        return self.cfg.enabled and self._fh is not None

    @property
    def n_seen(self) -> int:
        """Candidates that reached :meth:`begin`."""
        return self._index

    @property
    def n_sampled(self) -> int:
        """Candidates persisted to ``candidates.jsonl``."""
        return self._n_sampled

    # ------------------------------------------------------------------ #
    # Per-candidate hooks
    # ------------------------------------------------------------------ #

    def begin(self) -> bool:
        """Open a new candidate and decide whether it is sampled.

        Deterministic: candidate ``i`` is sampled iff ``i % sample_rate == 0``.

        Returns:
            ``True`` when the candidate will be persisted.
        """
        if self._fh is None:
            return False
        self.sampling = self._index % self.cfg.sample_rate == 0
        self._index += 1
        self._eval_time = 0.0
        return self.sampling

    def note_eval_time(self, seconds: float) -> None:
        """Attribute *seconds* of host fitness evaluation to the open candidate.

        Args:
            seconds: Wall-clock duration of the host's evaluation call.
        """
        if self.sampling:
            self._eval_time += seconds

    def record(
        self,
        *,
        dag: LabeledDAG | None = None,
        representation: str | None = None,
        representation_kind: str = "canonical",
        representation_hash: int | None = None,
        t_canon: float = 0.0,
        fallback: str = "none",
        dedup_hit: bool = False,
    ) -> None:
        """Close the open candidate, persisting it when it was sampled.

        Args:
            dag: The adapter's ``LabeledDAG``, or ``None`` when the conversion
                itself failed.
            representation: The arm's deduplication key string.  For the
                ``canonical`` arm this is the canonical string; for the ``hash``
                and ``host_native`` arms it is a fixed-order serialisation.
            representation_kind: The arm's ``key_mode``.
            representation_hash: The live, process-local ``hash()`` of
                *representation*.  Recorded as a diagnostic only; never a replay
                key.
            t_canon: Seconds spent resolving the deduplication key.
            fallback: Control-flow path taken, one of ``none``, ``timeout``,
                ``conversion_failure``, ``canon_raised``, ``atlas_hit``.
            dedup_hit: Whether the candidate was suppressed as a duplicate.
        """
        if not self.sampling:
            return
        self.sampling = False
        if fallback not in _CONTROL_FLOW_PATHS:
            log.warning("Stage-D trace: unknown fallback path %r recorded as 'none'", fallback)
            fallback = "none"
        try:
            row = self._build_row(
                dag=dag,
                representation=representation,
                representation_kind=representation_kind,
                representation_hash=representation_hash,
                t_canon=t_canon,
                fallback=fallback,
                dedup_hit=dedup_hit,
            )
        except Exception:  # noqa: BLE001 -- instrumentation must not kill a run
            self._write_failures += 1
            log.debug("Stage-D trace: candidate %d could not be encoded", self._index - 1)
            return
        self._accumulate(row)
        self._emit(row)

    # ------------------------------------------------------------------ #
    # Record construction
    # ------------------------------------------------------------------ #

    def _build_row(
        self,
        *,
        dag: LabeledDAG | None,
        representation: str | None,
        representation_kind: str,
        representation_hash: int | None,
        t_canon: float,
        fallback: str,
        dedup_hit: bool,
    ) -> dict[str, Any]:
        """Assemble the JSON object for the open candidate.

        Args:
            dag: The candidate DAG, or ``None``.
            representation: The arm's key string, or ``None``.
            representation_kind: The arm's ``key_mode``.
            representation_hash: Live ``hash()`` of *representation*.
            t_canon: Key-resolution cost in seconds.
            fallback: Control-flow path taken.
            dedup_hit: Whether the candidate was suppressed.

        Returns:
            A JSON-safe record.
        """
        row: dict[str, Any] = {
            "i": self._index - 1,
            "k": None,
            "labels": None,
            "serialisation": None,
            "digest_insertion": None,
            "digest_topological": None,
            "digest_topological_commutative": None,
            "canonical": None,
            "canonical_digest": None,
            "representation_kind": representation_kind,
            "representation_hash": representation_hash,
            "t_canon_s": float(t_canon),
            "t_eval_s": float(self._eval_time),
            "fallback": fallback,
            "dedup_hit": bool(dedup_hit),
            "violated_post": None,
        }
        if representation is not None and representation_kind == "canonical":
            row["canonical"] = representation
            row["canonical_digest"] = canonical_digest(representation)
        if dag is None:
            return row

        row["k"] = count_nonvar(dag)
        row["labels"] = _label_multiset(dag)
        unreachable = unreachable_nonvar(dag)
        row["violated_post"] = bool(unreachable)
        if unreachable:
            self._add_example("violated_post", row, extra={"unreachable_nodes": unreachable})
        try:
            row["serialisation"] = serialise(dag, FixedOrder.INSERTION)
            row["digest_insertion"] = fixed_order_digest(dag, FixedOrder.INSERTION)
            row["digest_topological"] = fixed_order_digest(dag, FixedOrder.TOPOLOGICAL)
            row["digest_topological_commutative"] = fixed_order_digest(
                dag, FixedOrder.TOPOLOGICAL_COMMUTATIVE
            )
        except SerialisationError:
            # A DAG the baseline cannot encode is still worth its k, labels and
            # timings; it simply cannot take part in the D3 replay.
            self._write_failures += 1
        return row

    def _add_example(
        self,
        path: str,
        row: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Retain a worked example of *path* if the quota is not yet met.

        Args:
            path: One of :data:`FALLBACK_PATHS`.
            row: The partially built record.
            extra: Path-specific witness fields.
        """
        bucket = self._examples.setdefault(path, [])
        if len(bucket) >= self.cfg.max_examples:
            return
        example = {
            "i": row["i"],
            "k": row["k"],
            "labels": row["labels"],
            "serialisation": row.get("serialisation"),
        }
        if extra:
            example.update(extra)
        bucket.append(example)

    def _accumulate(self, row: dict[str, Any]) -> None:
        """Fold *row* into the per-``k`` histogram, examples and reservoir.

        Args:
            row: A completed record.
        """
        self._n_sampled += 1
        k = row["k"] if row["k"] is not None else -1
        stats = self._hist.get(k)
        if stats is None:
            stats = self._hist[k] = _KStats()
        stats.n += 1
        stats.n_dedup_hits += int(row["dedup_hit"])
        stats.canon.add(row["t_canon_s"])
        stats.evaluation.add(row["t_eval_s"])

        if row["fallback"] != "none":
            self._add_example(row["fallback"], row)

        if row["canonical"] is not None and row["serialisation"] is not None:
            self._reservoir_offer(row)

    def _reservoir_offer(self, row: dict[str, Any]) -> None:
        """Offer *row* to the seeded reservoir behind ``spot_check.json``.

        Vitter's Algorithm R over the *spot-check-eligible* subsequence (records
        carrying both a canonical string and an INSERTION serialisation), so
        every eligible record has inclusion probability ``size / n_eligible``.
        The RNG is seeded from :attr:`StageDTraceConfig.spot_check_seed`, which
        is written into the artefact: the draw is random but reproducible.

        Args:
            row: A completed record.
        """
        entry = {
            "i": row["i"],
            "k": row["k"],
            "serialisation": row["serialisation"],
            "canonical_recorded": row["canonical"],
        }
        self._n_eligible += 1
        size = self.cfg.spot_check_size
        if len(self._reservoir) < size:
            self._reservoir.append(entry)
            return
        j = self._rng.randrange(self._n_eligible)
        if j < size:
            self._reservoir[j] = entry

    def _emit(self, row: dict[str, Any]) -> None:
        """Write *row* to ``candidates.jsonl``.

        Args:
            row: A completed record.
        """
        if self._fh is None:
            return
        try:
            line = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
            self._fh.write(line)
            self._fh.write("\n")
            self._bytes += len(line.encode("utf-8")) + 1
            if self._n_sampled % _FLUSH_EVERY == 0:
                self._fh.flush()
        except (OSError, TypeError, ValueError):
            self._write_failures += 1
            log.debug("Stage-D trace: failed to write candidate %d", row["i"])

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #

    def close(
        self,
        *,
        ledger: Any = None,
        run: dict[str, Any] | None = None,
    ) -> None:
        """Flush the stream and write the four derived artefacts.

        Idempotent, and registered with :mod:`atexit` when the tracer opens, so
        a SLURM kill after the last generation still leaves the artefacts on
        disk.  Every failure is logged and swallowed: a 12 h cell must not die
        inside its own instrumentation.

        Args:
            ledger: The run's
                :class:`~experiments.models.fallback_ledger.FallbackLedger`, or
                ``None``.  Supplies the five T06 rates.
            run: Run identity (``method``, ``problem``, ``seed``, ``variant``,
                ...), copied verbatim into ``canon_cost_hist.json`` so the D3
                replay can attribute a stream to a cell.
        """
        if self._closed or self._fh is None or self._dir is None:
            return
        self._closed = True
        try:
            self._fh.flush()
            self._fh.close()
        except OSError:
            log.exception("Stage-D trace: closing %s failed", CANDIDATES_FILE)
        finally:
            self._fh = None
            self.sampling = False

        meta = dict(run or {})
        for name, writer in (
            (CANON_COST_FILE, lambda: self._write_canon_cost(meta)),
            (FALLBACK_MD_FILE, lambda: self._write_fallback_md(ledger)),
            (SPOT_CHECK_FILE, self._write_spot_check),
            (STREAM_SIZE_FILE, self._write_stream_size),
        ):
            try:
                writer()
            except Exception:  # noqa: BLE001 -- artefact failure must not kill a run
                log.exception("Stage-D trace: writing %s failed", name)

    def _path(self, name: str) -> Path:
        """Return the absolute path of artefact *name*.

        Args:
            name: Artefact file name.

        Returns:
            The path inside the configured output directory.

        Raises:
            RuntimeError: If the tracer has no output directory.
        """
        if self._dir is None:
            raise RuntimeError("Stage-D tracer has no output directory")
        return self._dir / name

    def _write_canon_cost(self, meta: dict[str, Any]) -> None:
        """Write ``canon_cost_hist.json``.

        Args:
            meta: Run identity to embed.
        """
        payload = {
            "schema": "stage_d_canon_cost_hist/1",
            "run": meta,
            "sample_rate": self.cfg.sample_rate,
            "n_candidates_seen": self._index,
            "n_sampled": self._n_sampled,
            "n_write_failures": self._write_failures,
            "bin_edges_s": list(_BIN_EDGES),
            "bin_semantics": (
                "bins[0] is the underflow (< 1e-6 s); bins[i] for 1 <= i <= 24 "
                "covers [bin_edges_s[i-1], bin_edges_s[i]); bins[25] is the "
                "overflow (>= 1 s)"
            ),
            "stratified_on": "sampled stream only, not the full candidate stream",
            "by_k": {
                str(k): {
                    "n": st.n,
                    "n_dedup_hits": st.n_dedup_hits,
                    "t_canon_s": st.canon.to_dict(),
                    "t_eval_s": st.evaluation.to_dict(),
                }
                for k, st in sorted(self._hist.items())
            },
        }
        self._path(CANON_COST_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_fallback_md(self, ledger: Any) -> None:
        """Write ``fallback_ledger.md``.

        Args:
            ledger: The run's ``FallbackLedger`` or ``None``.
        """
        snapshot: dict[str, Any] = {}
        if ledger is not None and hasattr(ledger, "to_dict"):
            snapshot = ledger.to_dict()
        lines = [
            "# Stage-D fallback ledger",
            "",
            "The five T06 fallback rates for this cell.  Counts and denominators are",
            "`FallbackLedger.to_dict()`; the worked examples come from the Stage-D",
            f"trace, which persists 1 candidate in {self.cfg.sample_rate}.",
            "",
            f"- `n_seen` (ledger): {snapshot.get('n_seen', 'n/a')}",
            f"- `n_sampled` (ledger): {snapshot.get('n_sampled', 'n/a')}",
            f"- ledger sample rate: {snapshot.get('sample_rate', 'n/a')}",
            f"- trace candidates seen: {self._index}",
            f"- trace candidates persisted: {self._n_sampled}",
            "",
            "## Rates",
            "",
            "| Path | Count | Denominator | Rate | Denominator meaning |",
            "|---|---:|---:|---:|---|",
        ]
        lines.extend(self._fallback_rate_rows(snapshot))
        lines.extend(
            [
                "",
                "`violated_pre` (pre-normalisation, the sixth counter) is "
                f"{snapshot.get('violated_pre', 'n/a')} and is reported for context only: "
                "it is expected to be large and is not a fallback path.",
                "",
                "## Worked examples",
                "",
                "`serialisation` is the `FixedOrder.INSERTION` encoding; "
                "`isalsr.baselines.fixed_order_hash.deserialise` inverts it exactly, "
                "so each example is a runnable reproduction.",
                "",
            ]
        )
        lines.extend(self._fallback_example_blocks())
        self._path(FALLBACK_MD_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _fallback_rate_rows(self, snapshot: dict[str, Any]) -> list[str]:
        """Return the Markdown table rows for the five T06 rates.

        Args:
            snapshot: ``FallbackLedger.to_dict()`` output, possibly empty.

        Returns:
            One Markdown row per path in :data:`FALLBACK_PATHS`.
        """
        sampled = snapshot.get("n_sampled", 0) or 0
        seen = snapshot.get("n_seen", 0) or 0
        # violated_post is counted on the ledger's sampled subset; the other four
        # are full-rate O(1) events counted on every candidate.
        denominators = {
            "violated_post": (sampled, "ledger-sampled candidates"),
            "timeout": (seen, "all candidates"),
            "conversion_failure": (seen, "all candidates"),
            "canon_raised": (seen, "all candidates"),
            "atlas_hit": (seen, "all candidates"),
        }
        rows: list[str] = []
        for path in FALLBACK_PATHS:
            count = snapshot.get(path)
            denom, meaning = denominators[path]
            if count is None:
                rows.append(f"| `{path}` | n/a | n/a | n/a | {meaning} |")
                continue
            rate = f"{count / denom:.6g}" if denom else "n/a"
            rows.append(f"| `{path}` | {count} | {denom} | {rate} | {meaning} |")
        return rows

    def _fallback_example_blocks(self) -> list[str]:
        """Return the Markdown blocks holding the retained worked examples.

        Returns:
            Markdown lines, one section per path in :data:`FALLBACK_PATHS`.
        """
        lines: list[str] = []
        for path in FALLBACK_PATHS:
            examples = self._examples.get(path, [])
            lines.append(f"### `{path}`")
            lines.append("")
            if not examples:
                lines.append(
                    "No event of this path was observed on the traced stream, so no "
                    "worked example exists."
                )
                lines.append("")
                continue
            for example in examples:
                lines.append(f"- candidate `i={example['i']}`, `k={example['k']}`")
                lines.append(f"  - labels: `{example['labels']}`")
                lines.append(f"  - serialisation: `{example['serialisation']}`")
                if "unreachable_nodes" in example:
                    lines.append(
                        f"  - non-VAR nodes unreachable from any VAR: "
                        f"`{example['unreachable_nodes']}`"
                    )
            lines.append("")
        return lines

    def _write_spot_check(self) -> None:
        """Write ``spot_check.json``.

        Re-canonicalises each reserved candidate in **pure Python**
        (``backend="python"``) from its lossless INSERTION serialisation and
        compares the result byte-exact against the string the production engine
        emitted during the run.  Any mismatch is logged at ``ERROR``: it means
        the engine used in production is not the engine the gate certified.
        """
        from isalsr.core import backends  # noqa: PLC0415 -- keep import cost off the hot path
        from isalsr.core.canonical import fast_canonical_string  # noqa: PLC0415

        production_engine = backends.engine()
        checks: list[dict[str, Any]] = []
        for entry in self._reservoir:
            checks.append(self._spot_check_one(entry, fast_canonical_string))
        n_mismatch = sum(1 for c in checks if c["status"] != "match")
        payload = {
            "schema": "stage_d_spot_check/1",
            "n_requested": self.cfg.spot_check_size,
            "n_checked": len(checks),
            "n_eligible_offered": self._n_eligible,
            "n_mismatch": n_mismatch,
            "clean": n_mismatch == 0 and len(checks) > 0,
            "production_engine": production_engine,
            "replay_engine": "python",
            "replay_backend_kwarg": "python",
            "draw": "reservoir (Vitter Algorithm R) over spot-check-eligible records",
            "draw_seed": self.cfg.spot_check_seed,
            "checks": checks,
        }
        self._path(SPOT_CHECK_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if n_mismatch:
            log.error(
                "Stage-D spot check FAILED: %d/%d candidates disagree between the "
                "production engine (%s) and a pure-Python re-canonicalisation",
                n_mismatch,
                len(checks),
                production_engine,
            )

    @staticmethod
    def _spot_check_one(entry: dict[str, Any], canonicalise: Any) -> dict[str, Any]:
        """Re-canonicalise one reserved candidate in pure Python.

        Args:
            entry: A reservoir entry.
            canonicalise: ``fast_canonical_string``.

        Returns:
            The check outcome, with ``status`` in ``{"match", "mismatch", "error"}``.
        """
        out: dict[str, Any] = {
            "i": entry["i"],
            "k": entry["k"],
            "serialisation": entry["serialisation"],
            "canonical_recorded": entry["canonical_recorded"],
            "canonical_replay": None,
            "status": "error",
        }
        try:
            dag = deserialise(entry["serialisation"])
            # The production runners substitute the non-structural key at k=0, and
            # that substituted value is what the stream recorded.  Replaying the
            # raw canonical string here would flag every bare-variable candidate
            # as an engine disagreement (both engines return "" at k=0).
            replay = recorded_key(dag, canonicalise(dag, backend="python"))
        except Exception as exc:  # noqa: BLE001 -- report, never raise
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        out["canonical_replay"] = replay
        out["status"] = "match" if replay == entry["canonical_recorded"] else "mismatch"
        return out

    def _write_stream_size(self) -> None:
        """Write ``stream_size.md``.

        Reports measured bytes/candidate and bytes/run at the chosen sampling
        rate, the full-rate counterfactual, and the campaign-wide (8,400-run)
        projection against the FSCRATCH inode budget.  The 8,400 multiplier is a
        counterfactual: D2 traces **one** cell, so the realised cost is one
        column of the table.
        """
        budget = self.cfg.budget
        measured = self._measured_bytes()
        per_candidate = measured / self._n_sampled if self._n_sampled else 0.0
        rate = self.cfg.sample_rate
        full_rate_bytes = per_candidate * self._index
        campaign_bytes = measured * budget.campaign_runs
        campaign_files = budget.files_per_trace * budget.campaign_runs
        inode_ok = campaign_files <= budget.inode_headroom

        lines = [
            "# Stage-D trace stream size",
            "",
            f"- sampling rule: deterministic, candidate `i` persisted iff `i % {rate} == 0`",
            f"- candidates seen: {self._index}",
            f"- candidates persisted: {self._n_sampled}",
            f"- encode failures: {self._write_failures}",
            "",
            "## Measured",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| `{CANDIDATES_FILE}` bytes | {measured:,} |",
            f"| bytes / persisted candidate | {per_candidate:,.1f} |",
            f"| bytes / run (this cell, rate 1-in-{rate}) | {measured:,} |",
            f"| files / traced run | {budget.files_per_trace} |",
            "",
            "## Full-rate counterfactual",
            "",
            f"Persisting every candidate would cost `{per_candidate:,.1f} B x "
            f"{self._index:,} = {full_rate_bytes:,.0f} B` "
            f"({full_rate_bytes / 1e9:.2f} GB) for this cell alone, "
            f"i.e. {rate}x the measured figure.  The disqualifying cost is not the "
            "bytes but the CPU: `json.dumps` of one record is ~7 us, so a full-rate "
            "12 h Bingo cell (order 3.4e7 candidates, EXECUTION-PLAN 11.1 B9 row) "
            "would spend ~240 s (0.55 % of budget) inside the instrumentation that "
            "is supposed to measure that budget's T_canon / T_eval split (D1.7).",
            "",
            "## Campaign counterfactual (8,400 runs) -- NOT what D2 does",
            "",
            "D2 traces **one** cell.  This section exists only to justify the gating:",
            "",
            "| Quantity | Value | Budget | Verdict |",
            "|---|---:|---:|---|",
            f"| files if enabled campaign-wide | {campaign_files:,} | "
            f"{budget.inode_headroom:,} inodes | {'FITS' if inode_ok else 'EXCEEDS'} |",
            f"| bytes if enabled campaign-wide | {campaign_bytes / 1e12:.3f} TB | "
            f"{self._space_budget_text()} | {self._space_verdict(campaign_bytes)} |",
            f"| files for D2 as specified (1 cell) | {budget.files_per_trace} | "
            f"{budget.inode_headroom:,} inodes | FITS |",
            f"| bytes for D2 as specified (1 cell) | {measured / 1e6:.1f} MB | "
            f"{self._space_budget_text()} | FITS |",
            "",
            f"Budget provenance: {budget.provenance}",
            "",
        ]
        self._path(STREAM_SIZE_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _measured_bytes(self) -> int:
        """Return the on-disk size of ``candidates.jsonl``.

        Falls back to the byte counter accumulated while writing when the file
        cannot be stat-ed.

        Returns:
            Size in bytes.
        """
        try:
            return self._path(CANDIDATES_FILE).stat().st_size
        except OSError:
            return self._bytes

    def _space_budget_text(self) -> str:
        """Return the space-budget cell text for ``stream_size.md``."""
        budget = self.cfg.budget
        if budget.space_headroom_gb is None:
            return "not measured"
        return f"{budget.space_headroom_gb:,.0f} GB"

    def _space_verdict(self, projected_bytes: float) -> str:
        """Return the space verdict for a projected byte count.

        Args:
            projected_bytes: Projected bytes.

        Returns:
            ``FITS``, ``EXCEEDS`` or ``UNCHECKED``.
        """
        budget = self.cfg.budget
        if budget.space_headroom_gb is None:
            return "UNCHECKED (re-read `quota`)"
        return "FITS" if projected_bytes <= budget.space_headroom_gb * 1e9 else "EXCEEDS"
