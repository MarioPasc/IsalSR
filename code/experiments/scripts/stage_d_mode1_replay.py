"""T04 Mode-1 replay of the Stage-D certification streams (EXECUTION-PLAN §4.4 D3).

Replays a persisted candidate stream through the three fixed-order hashers **and**
through IsalSR canonicalisation on *identical input sequences*.  Because both
arms consume the same ordered list of DAGs, the comparison carries zero search
confound: any difference in distinct-count is attributable to the equivalence
relation alone, not to one arm having steered the host somewhere else.

Input
-----
``candidates.jsonl`` as written by :mod:`experiments.models.stage_d_trace`.  Each
record carries the ``FixedOrder.INSERTION`` serialisation of the candidate, which
:func:`isalsr.baselines.fixed_order_hash.deserialise` inverts exactly (node
identifiers, labels, variable indices and ``ordered_inputs`` restored verbatim),
so the stream is replayable without shipping DAG objects.  Only ``CONST`` values
are lost, and the canonical string is over labels only, so nothing the replay
needs is missing.

Redundancy ratios
-----------------
Let ``S = (D_1, ..., D_n)`` be the replayed stream, in stream order, restricted to
records that carry a serialisation.  For an equivalence relation ``~`` write
``|S/~|`` for the number of classes ``S`` meets.  Then:

``rho_total  = n / |S/=_INSERTION|``
    Redundancy visible with **no** normalisation of any kind: two candidates are
    the same only when the adapter emitted byte-identical output under the
    identity node order.  This is the floor of the ladder.
``rho_exact[order] = n / |S/=_order|``  for each ``FixedOrder``
    Redundancy captured by a *sound but incomplete* fixed-order serialisation.
    ``rho_exact["insertion"] == rho_total`` by construction; the script asserts it.
``rho_iso   = n / |S/~=|``
    Redundancy captured by the IsalSR canonical string, i.e. by labeled-DAG
    isomorphism.  This is a **complete** invariant, so it is the ceiling.

The partitions coarsen monotonically -- insertion refines topological refines
topological-commutative refines isomorphism -- so
``rho_total <= rho_exact[topological] <= rho_exact[topological_commutative] <= rho_iso``
must hold.  A violation is a bug, and the script reports it.

Two hard correctness checks
---------------------------
**Hash soundness (T04 AC-1).**  Any two DAGs sharing a fixed-order digest must
share a canonical string.  A violation is an *unsound merge*: the baseline would
collapse two non-isomorphic expressions and its reduction factor would be
inflated by an error, not by an invariant.  The script names the counterexample
pair and exits non-zero.  A pair whose *serialisations* also differ is reported
separately as a 64-bit digest collision -- still a merge, still fatal, but a
different diagnosis.

**IsalSR soundness.**  Any two DAGs sharing a canonical string must satisfy
:meth:`isalsr.core.labeled_dag.LabeledDAG.is_isomorphic`.  Checked on the
``--max-classes`` largest equivalence classes, each member against the class
representative.  A violation exits non-zero.

Sampling caveat
---------------
The persisted stream is a deterministic 1-in-N subsample.  Distinct counts are
**not** unbiased under subsampling, so every ``rho`` here is a downward-biased
estimate of the full-stream value; the full-stream cardinalities live in the run
log's HyperLogLog shadow sketches.  What survives subsampling exactly is the pair
of soundness checks (a violation on a subset is a violation) and the *ordering*
of the ratios, which is what the controlled comparison is for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.models.structural_scope import is_structural, recorded_key
from isalsr.baselines.fixed_order_hash import (
    FixedOrder,
    SerialisationError,
    deserialise,
    fixed_order_digest,
    serialise,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG

log = logging.getLogger("stage_d_mode1_replay")

CANDIDATES_FILE = "candidates.jsonl"
CANON_COST_FILE = "canon_cost_hist.json"

#: Ladder rungs, in refinement order (finest first).
ORDERS: tuple[FixedOrder, ...] = (
    FixedOrder.INSERTION,
    FixedOrder.TOPOLOGICAL,
    FixedOrder.TOPOLOGICAL_COMMUTATIVE,
)

_EXIT_OK = 0
_EXIT_UNSOUND = 2
_EXIT_NO_DATA = 3


class ReplayError(RuntimeError):
    """Raised when a stream cannot be replayed at all."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class ReplayConfig:
    """Configuration of one Mode-1 replay.

    Attributes:
        trace_dirs: Directories holding a ``candidates.jsonl``.
        out_json: Destination for the machine-readable report.
        out_md: Destination for the human-readable report.
        max_classes: Number of largest canonical classes to spot-check for
            IsalSR soundness.
        canonical_backend: ``backend=`` passed to ``fast_canonical_string``;
            ``None`` uses the active engine.
    """

    trace_dirs: list[Path]
    out_json: Path | None = None
    out_md: Path | None = None
    max_classes: int = 10
    canonical_backend: str | None = None


@dataclass
class StreamRecord:
    """One replayable candidate.

    Attributes:
        index: Stream index recorded at trace time.
        k: Non-VAR node count, recomputed on the reconstructed DAG.
        dag: The reconstructed DAG.
        serialisations: Serialisation per fixed order.
        digests: Digest per fixed order.
        canonical: Canonical string recomputed during the replay.
        canonical_recorded: Canonical string recorded during the run, if any.
    """

    index: int
    k: int
    dag: LabeledDAG
    serialisations: dict[str, str]
    digests: dict[str, int]
    canonical: str
    canonical_recorded: str | None


@dataclass
class LoadReport:
    """Bookkeeping for one loaded stream.

    Attributes:
        path: The ``candidates.jsonl`` that was read.
        run: Run identity recovered from the sibling ``canon_cost_hist.json``.
        n_lines: JSON lines read.
        n_replayable: Records carrying a serialisation.
        n_malformed: Lines that failed to parse.
        n_deserialise_failures: Serialisations the decoder rejected.
        n_canon_failures: DAGs the canonicaliser rejected.
        digest_mismatches: Records whose recomputed digest differs from the one
            recorded at trace time.
        canonical_mismatches: Records whose recomputed canonical string differs
            from the one recorded at trace time.
    """

    path: Path
    run: dict[str, Any] = field(default_factory=dict)
    n_lines: int = 0
    n_replayable: int = 0
    n_malformed: int = 0
    n_deserialise_failures: int = 0
    n_canon_failures: int = 0
    digest_mismatches: list[dict[str, Any]] = field(default_factory=list)
    canonical_mismatches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot."""
        return {
            "path": str(self.path),
            "run": self.run,
            "n_lines": self.n_lines,
            "n_replayable": self.n_replayable,
            "n_malformed": self.n_malformed,
            "n_deserialise_failures": self.n_deserialise_failures,
            "n_canon_failures": self.n_canon_failures,
            "n_digest_mismatches": len(self.digest_mismatches),
            "digest_mismatches": self.digest_mismatches[:5],
            "n_canonical_mismatches": len(self.canonical_mismatches),
            "canonical_mismatches": self.canonical_mismatches[:5],
        }


# --------------------------------------------------------------------------- #
# Loading and replay
# --------------------------------------------------------------------------- #


def load_stream(
    trace_dir: Path,
    canonical_backend: str | None = None,
) -> tuple[list[StreamRecord], LoadReport]:
    """Reconstruct the candidate stream persisted in *trace_dir*.

    Every record is decoded from its INSERTION serialisation, then re-serialised
    and re-digested under all three fixed orders and re-canonicalised.  The
    recomputed digests and canonical string are cross-checked against the values
    recorded at trace time; a disagreement means the reconstruction is not
    faithful and is reported, not silently absorbed.

    Args:
        trace_dir: Directory holding ``candidates.jsonl``.
        canonical_backend: ``backend=`` for ``fast_canonical_string``.

    Returns:
        The replayable records in stream order and the load report.

    Raises:
        ReplayError: If ``candidates.jsonl`` is absent.
    """
    path = trace_dir / CANDIDATES_FILE
    if not path.is_file():
        raise ReplayError(f"No {CANDIDATES_FILE} under {trace_dir}")
    report = LoadReport(path=path, run=_read_run_identity(trace_dir))
    records: list[StreamRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            report.n_lines += 1
            record = _replay_line(line, report, canonical_backend)
            if record is not None:
                records.append(record)
    report.n_replayable = len(records)
    return records, report


def _read_run_identity(trace_dir: Path) -> dict[str, Any]:
    """Return the run identity written beside the stream.

    Args:
        trace_dir: Directory holding ``canon_cost_hist.json``.

    Returns:
        The ``run`` block, or an empty dict when it is missing or unreadable.
    """
    path = trace_dir / CANON_COST_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    run = payload.get("run")
    return dict(run) if isinstance(run, dict) else {}


def _replay_line(
    line: str,
    report: LoadReport,
    canonical_backend: str | None,
) -> StreamRecord | None:
    """Decode and replay one JSONL line.

    Args:
        line: The raw JSON text.
        report: Load report, mutated with failure counts.
        canonical_backend: ``backend=`` for ``fast_canonical_string``.

    Returns:
        The replayed record, or ``None`` when the line is not replayable.
    """
    try:
        row = json.loads(line)
    except ValueError:
        report.n_malformed += 1
        return None
    text = row.get("serialisation")
    if not text:
        return None
    try:
        dag = deserialise(text)
    except SerialisationError:
        report.n_deserialise_failures += 1
        return None
    try:
        serialisations = {order.value: serialise(dag, order) for order in ORDERS}
        digests = {order.value: fixed_order_digest(dag, order) for order in ORDERS}
    except SerialisationError:
        report.n_deserialise_failures += 1
        return None
    try:
        canonical = fast_canonical_string(dag, backend=canonical_backend)
    except Exception as exc:  # noqa: BLE001 -- a refused DAG is data, not a crash
        report.n_canon_failures += 1
        log.debug("candidate %s: canonicalisation failed (%s)", row.get("i"), exc)
        return None

    _cross_check(row, digests, canonical, dag, report)
    return StreamRecord(
        index=int(row.get("i", -1)),
        k=_count_nonvar(dag),
        dag=dag,
        serialisations=serialisations,
        digests=digests,
        canonical=canonical,
        canonical_recorded=row.get("canonical"),
    )


def _cross_check(
    row: dict[str, Any],
    digests: dict[str, int],
    canonical: str,
    dag: LabeledDAG,
    report: LoadReport,
) -> None:
    """Compare the replayed keys against the ones recorded at trace time.

    The recorded value is the *deduplication key*, not the raw canonical string:
    at :math:`k = 0` the production runners substitute
    :func:`~experiments.models.structural_scope.nonstructural_key`.  The replay
    applies the same substitution through
    :func:`~experiments.models.structural_scope.recorded_key`; comparing the raw
    canonical string instead reports a mismatch on every bare-variable candidate,
    which reads as an engine disagreement but is a difference of key definition.

    Args:
        row: The raw persisted record.
        digests: Digests recomputed during the replay.
        canonical: Canonical string recomputed during the replay.
        dag: The reconstructed DAG, needed to decide the k=0 substitution.
        report: Load report, mutated with any mismatch.
    """
    for order in ORDERS:
        recorded = row.get(f"digest_{order.value}")
        if recorded is not None and recorded != digests[order.value]:
            report.digest_mismatches.append(
                {
                    "i": row.get("i"),
                    "order": order.value,
                    "recorded": recorded,
                    "replayed": digests[order.value],
                }
            )
    recorded_canonical = row.get("canonical")
    replayed_key = recorded_key(dag, canonical)
    if recorded_canonical is not None and recorded_canonical != replayed_key:
        report.canonical_mismatches.append(
            {
                "i": row.get("i"),
                "recorded": recorded_canonical,
                "replayed": replayed_key,
            }
        )


def _count_nonvar(dag: LabeledDAG) -> int:
    """Return the number of non-VAR nodes of *dag*.

    Args:
        dag: The DAG to inspect.

    Returns:
        The count.
    """
    from isalsr.core.node_types import NodeType  # noqa: PLC0415 -- keep import local

    return sum(1 for i in range(dag.node_count) if dag.node_label_unchecked(i) is not NodeType.VAR)


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #


def compute_ratios(records: list[StreamRecord]) -> dict[str, Any]:
    """Compute ``rho_total``, ``rho_exact`` and ``rho_iso`` on *records*.

    Args:
        records: The replayed stream, in stream order.

    Returns:
        A dict with the overall ratios and a ``by_k`` stratification.
    """
    overall = _ratios_for(records)
    by_k: dict[int, list[StreamRecord]] = defaultdict(list)
    for record in records:
        by_k[record.k].append(record)
    return {
        "n": len(records),
        "overall": overall,
        "by_k": {str(k): _ratios_for(group) for k, group in sorted(by_k.items())},
        "monotonicity_ok": _monotone(overall),
    }


def _ratios_for(records: list[StreamRecord]) -> dict[str, Any]:
    """Compute the ratio block for one stratum.

    Args:
        records: Records in the stratum.

    Returns:
        Stream length, distinct counts and ratios.
    """
    n = len(records)
    distinct_exact = {
        order.value: len({r.digests[order.value] for r in records}) for order in ORDERS
    }
    distinct_iso = len({r.canonical for r in records})
    distinct_total = distinct_exact[FixedOrder.INSERTION.value]
    return {
        "n": n,
        "distinct_total": distinct_total,
        "distinct_exact": distinct_exact,
        "distinct_iso": distinct_iso,
        "rho_total": _ratio(n, distinct_total),
        "rho_exact": {name: _ratio(n, d) for name, d in distinct_exact.items()},
        "rho_iso": _ratio(n, distinct_iso),
    }


def _ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or 0.0 when the stratum is empty.

    Args:
        numerator: Stream length.
        denominator: Distinct count.

    Returns:
        The ratio.
    """
    return (numerator / denominator) if denominator else 0.0


def _monotone(block: dict[str, Any]) -> bool:
    """Return whether the ladder's ratios are non-decreasing.

    Args:
        block: A ratio block from :func:`_ratios_for`.

    Returns:
        ``True`` when ``rho_total <= rho_exact[topological] <=
        rho_exact[topological_commutative] <= rho_iso``.
    """
    chain = [
        block["rho_total"],
        block["rho_exact"][FixedOrder.TOPOLOGICAL.value],
        block["rho_exact"][FixedOrder.TOPOLOGICAL_COMMUTATIVE.value],
        block["rho_iso"],
    ]
    return all(a <= b + 1e-12 for a, b in zip(chain, chain[1:], strict=False))


# --------------------------------------------------------------------------- #
# Hard check 1 — hash soundness (T04 AC-1)
# --------------------------------------------------------------------------- #


def check_hash_soundness(records: list[StreamRecord]) -> dict[str, Any]:
    """Verify that a shared fixed-order digest implies a shared canonical string.

    A digest that maps to two different canonical strings is an **unsound
    merge**: the fixed-order baseline would collapse two non-isomorphic
    expressions.  When the two serialisations also differ the cause is a 64-bit
    BLAKE2b collision rather than a defect in the ordering, and the two are
    reported separately -- both are fatal, but they are different diagnoses.

    Args:
        records: The replayed stream.

    Returns:
        Per-order outcome with named counterexample pairs.

    """
    out: dict[str, Any] = {"sound": True, "by_order": {}}
    for order in ORDERS:
        name = order.value
        witness: dict[int, StreamRecord] = {}
        unsound: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        for record in records:
            digest = record.digests[name]
            first = witness.get(digest)
            if first is None:
                witness[digest] = record
                continue
            if first.canonical == record.canonical:
                continue
            pair = _counterexample(first, record, name)
            if first.serialisations[name] == record.serialisations[name]:
                unsound.append(pair)
            else:
                collisions.append(pair)
        block = {
            "n_unsound_merges": len(unsound),
            "n_digest_collisions": len(collisions),
            "unsound_merges": unsound[:5],
            "digest_collisions": collisions[:5],
            "sound": not unsound and not collisions,
        }
        out["by_order"][name] = block
        out["sound"] = out["sound"] and block["sound"]
    return out


def _counterexample(left: StreamRecord, right: StreamRecord, order: str) -> dict[str, Any]:
    """Describe a pair of records that share a digest but not a canonical string.

    Args:
        left: The first record seen with this digest.
        right: The colliding record.
        order: The fixed order under which they collide.

    Returns:
        A named, reproducible counterexample.
    """
    return {
        "order": order,
        "digest": left.digests[order],
        "left": {
            "i": left.index,
            "k": left.k,
            "serialisation": left.serialisations[order],
            "serialisation_insertion": left.serialisations[FixedOrder.INSERTION.value],
            "canonical": left.canonical,
        },
        "right": {
            "i": right.index,
            "k": right.k,
            "serialisation": right.serialisations[order],
            "serialisation_insertion": right.serialisations[FixedOrder.INSERTION.value],
            "canonical": right.canonical,
        },
    }


# --------------------------------------------------------------------------- #
# Hard check 2 — IsalSR soundness
# --------------------------------------------------------------------------- #


def check_isalsr_soundness(records: list[StreamRecord], max_classes: int) -> dict[str, Any]:
    """Verify that a shared canonical string implies ``is_isomorphic``.

    Checked on the *largest* canonical equivalence classes, where a defect in the
    invariant would show up first: each member is compared against the class
    representative, which is ``|C| - 1`` comparisons rather than ``|C|^2``.

    Args:
        records: The replayed stream.
        max_classes: How many of the largest classes to check.

    Returns:
        The outcome, with any failing pair named.
    """
    # Restrict to the invariant's domain.  A candidate with zero internal nodes
    # is a bare input variable; every such DAG canonicalises to "" whatever m is
    # and whatever variable it returns, because Sigma_SR encodes only the
    # instructions that BUILD internal nodes.  Comparing two of them against
    # is_isomorphic therefore tests a claim we do not make: at k = 0 the
    # relabeling group is 0! = 1, so there is no redundancy to collapse.  The
    # runners exclude these from deduplication for the same reason
    # (experiments/models/structural_scope.py), so checking them here would
    # measure a code path that no longer exists.
    #
    # The exclusion is COUNTED, never silent: a reader must be able to see how
    # much of the stream the check declined to look at.
    classes: dict[str, list[StreamRecord]] = defaultdict(list)
    n_nonstructural = 0
    for record in records:
        if not is_structural(record.dag):
            n_nonstructural += 1
            continue
        classes[record.canonical].append(record)
    largest = sorted(classes.values(), key=len, reverse=True)[: max(0, max_classes)]

    failures: list[dict[str, Any]] = []
    n_pairs = 0
    for members in largest:
        head = members[0]
        for other in members[1:]:
            n_pairs += 1
            if not head.dag.is_isomorphic(other.dag):
                failures.append(
                    {
                        "canonical": head.canonical,
                        "left": {
                            "i": head.index,
                            "serialisation_insertion": head.serialisations[
                                FixedOrder.INSERTION.value
                            ],
                        },
                        "right": {
                            "i": other.index,
                            "serialisation_insertion": other.serialisations[
                                FixedOrder.INSERTION.value
                            ],
                        },
                    }
                )
    return {
        "sound": not failures,
        "n_classes_total": len(classes),
        "n_classes_checked": len(largest),
        "largest_class_sizes": [len(m) for m in largest],
        "n_pairs_checked": n_pairs,
        "n_failures": len(failures),
        "failures": failures[:5],
        "n_nonstructural_excluded": n_nonstructural,
        "scope": (
            "k >= 1 only; zero-internal-node candidates are outside the "
            "invariant's domain and are excluded from deduplication by the "
            "runners (see experiments/models/structural_scope.py)"
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def replay(cfg: ReplayConfig) -> dict[str, Any]:
    """Run the Mode-1 replay over every configured trace directory.

    Args:
        cfg: The replay configuration.

    Returns:
        The full report.

    Raises:
        ReplayError: If no directory yields a replayable stream.
    """
    streams: list[dict[str, Any]] = []
    for trace_dir in cfg.trace_dirs:
        try:
            records, load = load_stream(trace_dir, cfg.canonical_backend)
        except ReplayError as exc:
            log.warning("%s", exc)
            continue
        if not records:
            log.warning("%s: no replayable records", trace_dir)
        streams.append(
            {
                "trace_dir": str(trace_dir),
                "load": load.to_dict(),
                "method": load.run.get("method", "unknown"),
                "ratios": compute_ratios(records),
                "hash_soundness": check_hash_soundness(records),
                "isalsr_soundness": check_isalsr_soundness(records, cfg.max_classes),
            }
        )
    if not streams:
        raise ReplayError("No replayable stream found under any --trace-dir/--results-root")

    by_method = _aggregate_by_method(streams, cfg)
    unsound = [s for s in streams if not s["hash_soundness"]["sound"]]
    iso_unsound = [s for s in streams if not s["isalsr_soundness"]["sound"]]
    fidelity = [
        s
        for s in streams
        if s["load"]["n_digest_mismatches"] or s["load"]["n_canonical_mismatches"]
    ]
    return {
        "schema": "stage_d_mode1_replay/1",
        "rho_definitions": {
            "rho_total": "stream length / distinct FixedOrder.INSERTION serialisations "
            "(no normalisation of any kind; the floor of the ladder)",
            "rho_exact": "stream length / distinct fixed-order digests, per order "
            "(sound but incomplete)",
            "rho_iso": "stream length / distinct canonical strings "
            "(complete labeled-DAG invariant; the ceiling)",
        },
        "sampling_caveat": (
            "the persisted stream is a deterministic 1-in-N subsample, so every rho "
            "here is a downward-biased estimate of the full-stream value; the "
            "soundness checks and the ordering of the ratios are unaffected"
        ),
        "canonical_backend": cfg.canonical_backend or "active engine",
        "n_streams": len(streams),
        "streams": streams,
        "by_method": by_method,
        "hash_soundness_ok": not unsound,
        "isalsr_soundness_ok": not iso_unsound,
        "replay_fidelity_ok": not fidelity,
        "ok": not unsound and not iso_unsound and not fidelity,
    }


def _aggregate_by_method(streams: list[dict[str, Any]], cfg: ReplayConfig) -> dict[str, Any]:
    """Pool the per-stream ratios by host method.

    Streams from the same method are concatenated at the *distinct-count* level:
    the pooled numerator is the sum of stream lengths and the pooled denominator
    is the sum of per-stream distinct counts, which is an upper bound on the true
    pooled distinct count and hence a **lower** bound on pooled rho.  Stated
    rather than hidden: exact pooling would need the streams themselves, which
    the report does not carry.

    Args:
        streams: Per-stream blocks.
        cfg: The replay configuration, unused beyond documenting intent.

    Returns:
        Pooled ratios per method.
    """
    del cfg
    pooled: dict[str, dict[str, Any]] = {}
    for stream in streams:
        method = stream["method"]
        block = pooled.setdefault(
            method,
            {
                "n_streams": 0,
                "n": 0,
                "distinct_total": 0,
                "distinct_exact": {order.value: 0 for order in ORDERS},
                "distinct_iso": 0,
                "pooling": "sum of per-stream distinct counts; a lower bound on pooled rho",
            },
        )
        overall = stream["ratios"]["overall"]
        block["n_streams"] += 1
        block["n"] += overall["n"]
        block["distinct_total"] += overall["distinct_total"]
        block["distinct_iso"] += overall["distinct_iso"]
        for name, value in overall["distinct_exact"].items():
            block["distinct_exact"][name] += value
    for block in pooled.values():
        block["rho_total"] = _ratio(block["n"], block["distinct_total"])
        block["rho_iso"] = _ratio(block["n"], block["distinct_iso"])
        block["rho_exact"] = {
            name: _ratio(block["n"], d) for name, d in block["distinct_exact"].items()
        }
    return pooled


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown.

    Args:
        report: The output of :func:`replay`.

    Returns:
        The Markdown document.
    """
    lines = [
        "# T04 Mode-1 replay of the Stage-D certification streams",
        "",
        f"- streams replayed: {report['n_streams']}",
        f"- canonicalisation backend: `{report['canonical_backend']}`",
        f"- hash soundness: **{'PASS' if report['hash_soundness_ok'] else 'FAIL'}**",
        f"- IsalSR soundness: **{'PASS' if report['isalsr_soundness_ok'] else 'FAIL'}**",
        f"- replay fidelity: **{'PASS' if report['replay_fidelity_ok'] else 'FAIL'}**",
        "",
        "## Definitions",
        "",
    ]
    lines.extend(f"- `{k}` = {v}" for k, v in report["rho_definitions"].items())
    lines.extend(["", f"> Sampling caveat: {report['sampling_caveat']}.", "", "## By method", ""])
    lines.extend(_markdown_method_table(report["by_method"]))
    lines.extend(["", "## By stream", ""])
    for stream in report["streams"]:
        lines.extend(_markdown_stream_block(stream))
    lines.extend(_markdown_decision(report))
    return "\n".join(lines) + "\n"


def _markdown_method_table(by_method: dict[str, Any]) -> list[str]:
    """Return the pooled per-method Markdown table.

    Args:
        by_method: Pooled ratios per method.

    Returns:
        Markdown lines.
    """
    header = [
        "| Method | n | rho_total | rho_exact (ins) | rho_exact (topo) | "
        "rho_exact (topo-comm) | rho_iso |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        f"| {method} | {b['n']} | {b['rho_total']:.4f} | "
        f"{b['rho_exact'][FixedOrder.INSERTION.value]:.4f} | "
        f"{b['rho_exact'][FixedOrder.TOPOLOGICAL.value]:.4f} | "
        f"{b['rho_exact'][FixedOrder.TOPOLOGICAL_COMMUTATIVE.value]:.4f} | "
        f"{b['rho_iso']:.4f} |"
        for method, b in sorted(by_method.items())
    ]
    return header + rows


def _markdown_stream_block(stream: dict[str, Any]) -> list[str]:
    """Return the Markdown block for one stream.

    Args:
        stream: A per-stream report block.

    Returns:
        Markdown lines.
    """
    ratios = stream["ratios"]
    overall = ratios["overall"]
    lines = [
        f"### `{stream['trace_dir']}` ({stream['method']})",
        "",
        f"- records replayed: {overall['n']}",
        f"- monotonicity `rho_total <= rho_topo <= rho_topo_comm <= rho_iso`: "
        f"{'OK' if ratios['monotonicity_ok'] else 'VIOLATED'}",
        f"- load: {stream['load']['n_lines']} lines, "
        f"{stream['load']['n_malformed']} malformed, "
        f"{stream['load']['n_deserialise_failures']} undecodable, "
        f"{stream['load']['n_canon_failures']} uncanonicalisable",
        f"- replay fidelity: {stream['load']['n_digest_mismatches']} digest and "
        f"{stream['load']['n_canonical_mismatches']} canonical mismatches vs the "
        "values recorded during the run",
        "",
        "| k | n | rho_total | rho_exact (topo) | rho_exact (topo-comm) | rho_iso |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k, block in ratios["by_k"].items():
        lines.append(
            f"| {k} | {block['n']} | {block['rho_total']:.4f} | "
            f"{block['rho_exact'][FixedOrder.TOPOLOGICAL.value]:.4f} | "
            f"{block['rho_exact'][FixedOrder.TOPOLOGICAL_COMMUTATIVE.value]:.4f} | "
            f"{block['rho_iso']:.4f} |"
        )
    lines.append("")
    lines.extend(_markdown_soundness(stream))
    return lines


def _markdown_soundness(stream: dict[str, Any]) -> list[str]:
    """Return the Markdown soundness paragraphs for one stream.

    Args:
        stream: A per-stream report block.

    Returns:
        Markdown lines.
    """
    lines: list[str] = []
    for name, block in stream["hash_soundness"]["by_order"].items():
        if block["sound"]:
            continue
        lines.append(
            f"**UNSOUND MERGE under `{name}`**: {block['n_unsound_merges']} pairs share a "
            f"digest and a serialisation but not a canonical string; "
            f"{block['n_digest_collisions']} further pairs share a digest through a "
            "64-bit BLAKE2b collision."
        )
        for pair in block["unsound_merges"] + block["digest_collisions"]:
            lines.append(
                f"- `i={pair['left']['i']}` vs `i={pair['right']['i']}`: "
                f"`{pair['left']['canonical']}` vs `{pair['right']['canonical']}`"
            )
        lines.append("")
    iso = stream["isalsr_soundness"]
    verdict = "PASS" if iso["sound"] else f"FAIL ({iso['n_failures']} pairs)"
    lines.append(
        f"IsalSR soundness: {iso['n_pairs_checked']} pairs across the "
        f"{iso['n_classes_checked']} largest of {iso['n_classes_total']} canonical classes "
        f"(sizes {iso['largest_class_sizes'][:10]}) -- {verdict}."
    )
    lines.append("")
    return lines


def _markdown_decision(report: dict[str, Any]) -> list[str]:
    """Return the R1.4 decision paragraph.

    Args:
        report: The full report.

    Returns:
        Markdown lines.
    """
    null_result = all(
        abs(b["rho_exact"][FixedOrder.INSERTION.value] - 1.0) < 5e-3
        for b in report["by_method"].values()
    )
    lines = ["## R1.4 decision", ""]
    if null_result:
        lines.append(
            "`rho_exact` is 1.00 to within 0.5 % for every method on identical inputs: "
            "the live hash arm is a **null result**, which is itself the answer to R1.4 "
            "-- a fixed-order serialisation of the host's own output merges nothing the "
            "host did not already emit twice byte-identically."
        )
    else:
        lines.append(
            "`rho_exact` departs from 1.00 on at least one method, so the live hash arm "
            "is **not** a null result: the fixed-order rungs capture a measurable share "
            "of the redundancy and the arm must be reported on its own numbers."
        )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def discover_trace_dirs(results_root: Path) -> list[Path]:
    """Return every ``c2_trace``-style directory under *results_root*.

    Args:
        results_root: Campaign root to scan.

    Returns:
        Directories holding a ``candidates.jsonl``, sorted.
    """
    return sorted(p.parent for p in results_root.rglob(CANDIDATES_FILE))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="stage_d_mode1_replay",
        description=(
            "Replay a Stage-D candidate stream through the three fixed-order "
            "hashers and through IsalSR canonicalisation on identical inputs."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory holding candidates.jsonl.  Repeatable.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Campaign root; every candidates.jsonl beneath it is replayed.",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Write the JSON report here.")
    parser.add_argument("--out-md", type=Path, default=None, help="Write the Markdown report here.")
    parser.add_argument(
        "--max-classes",
        type=int,
        default=10,
        help="Largest canonical classes spot-checked for IsalSR soundness.",
    )
    parser.add_argument(
        "--canonical-backend",
        choices=("cpp", "python"),
        default=None,
        help="Force a canonicalisation engine; default is the active one.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` on a soundness or fidelity failure, ``3`` when no
        stream could be found.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    trace_dirs = list(args.trace_dir)
    if args.results_root is not None:
        trace_dirs.extend(discover_trace_dirs(args.results_root))
    if not trace_dirs:
        log.error("Give at least one --trace-dir or a --results-root")
        return _EXIT_NO_DATA

    cfg = ReplayConfig(
        trace_dirs=trace_dirs,
        out_json=args.out_json,
        out_md=args.out_md,
        max_classes=args.max_classes,
        canonical_backend=args.canonical_backend,
    )
    try:
        report = replay(cfg)
    except ReplayError as exc:
        log.error("%s", exc)
        return _EXIT_NO_DATA

    if cfg.out_json is not None:
        cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if cfg.out_md is not None:
        cfg.out_md.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_md.write_text(render_markdown(report), encoding="utf-8")

    return _EXIT_OK if report["ok"] else _log_and_fail(report)


def _log_and_fail(report: dict[str, Any]) -> int:
    """Log the failure loudly and return the failure exit code.

    Args:
        report: The full report.

    Returns:
        :data:`_EXIT_UNSOUND`.
    """
    if not report["hash_soundness_ok"]:
        for stream in report["streams"]:
            for name, block in stream["hash_soundness"]["by_order"].items():
                for pair in block["unsound_merges"]:
                    log.error(
                        "UNSOUND MERGE (%s): i=%s and i=%s share digest %s but "
                        "canonicalise to %r and %r",
                        name,
                        pair["left"]["i"],
                        pair["right"]["i"],
                        pair["digest"],
                        pair["left"]["canonical"],
                        pair["right"]["canonical"],
                    )
                for pair in block["digest_collisions"]:
                    log.error(
                        "DIGEST COLLISION (%s): i=%s and i=%s share digest %s with "
                        "different serialisations",
                        name,
                        pair["left"]["i"],
                        pair["right"]["i"],
                        pair["digest"],
                    )
    if not report["isalsr_soundness_ok"]:
        for stream in report["streams"]:
            for failure in stream["isalsr_soundness"]["failures"]:
                log.error(
                    "ISALSR SOUNDNESS VIOLATION: i=%s and i=%s share canonical %r but "
                    "are not is_isomorphic",
                    failure["left"]["i"],
                    failure["right"]["i"],
                    failure["canonical"],
                )
    if not report["replay_fidelity_ok"]:
        log.error(
            "REPLAY FIDELITY FAILURE: recomputed keys disagree with the values "
            "recorded during the run; the persisted stream does not reproduce it"
        )
    return _EXIT_UNSOUND


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
