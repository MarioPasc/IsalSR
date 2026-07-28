"""Instrumentation ledger for IsalSR dedup fallback paths.

Counts six event categories on each candidate DAG arriving at the IsalSR
deduplication wrapper.  Two categories involve an O(V+E) reachability walk
and are counted on a deterministic 1-in-N sample (env var
``ISALSR_LEDGER_SAMPLE_RATE``, default 1).  The remaining four are
exception-path or cache events counted at full rate (O(1) per DAG).

Enable via env var ``ISALSR_LEDGER_ENABLED=1``.  When unset or 0, every
hook is a single attribute check with no allocation or DAG traversal.

Six counters
------------
violated_pre
    Sampled DAGs that violated the Round-Trip Fidelity precondition
    *before* ``_normalize_const_edges`` was applied.  Expected to be
    large (up to 100 %) when the search produces CONST-bearing DAGs.
violated_post
    Sampled DAGs still violating the precondition *after* normalisation.
    Expected to be 0 in production; a non-zero value indicates a repair
    gap.
timeout
    Full-rate count of ``CanonicalTimeoutError`` events.
conversion_failure
    Full-rate count of adapter exceptions (DAG unavailable, k unknown).
canon_raised
    Full-rate count of non-timeout exceptions from the canonicaliser.
atlas_hit
    Full-rate count of atlas cache hits.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers (also importable by runners)
# ---------------------------------------------------------------------------


def count_nonvar(dag: LabeledDAG) -> int:
    """Return the number of non-VAR nodes in *dag*.

    Args:
        dag: The labeled DAG to inspect.

    Returns:
        Count of nodes whose label is not ``NodeType.VAR``.
    """
    n = dag.node_count
    count = 0
    for i in range(n):
        if dag.node_label_unchecked(i) != NodeType.VAR:
            count += 1
    return count


def violates_precondition(dag: LabeledDAG) -> bool:
    """Return True if any non-VAR node is unreachable from VAR sources.

    The Round-Trip Fidelity precondition (Theorem 1 in the manuscript):
    every non-variable node of *D* must be reachable from at least one
    variable node via directed edges.  A BFS is launched from all VAR
    nodes following out-edges; any unvisited non-VAR node is a violation.

    Uses ``out_neighbors_raw`` (no frozenset copy) and
    ``node_label_unchecked`` (no bounds check) for minimal overhead.

    Args:
        dag: LabeledDAG to inspect.

    Returns:
        True iff the precondition is violated.
    """
    n = dag.node_count
    visited = bytearray(n)  # 0/1 flags; cheaper than list[bool]
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
    return any(not visited[i] and dag.node_label_unchecked(i) != NodeType.VAR for i in range(n))


# ---------------------------------------------------------------------------
# FallbackLedger
# ---------------------------------------------------------------------------


class FallbackLedger:
    """Instrumentation hook counting six fallback paths in the dedup wrapper.

    Instantiate once per run.  Runners pass the instance to adapters and
    call the ``record_*`` methods at the appropriate points.

    Attributes:
        enabled: Whether counting is active (from env var).
        n_seen: Total DAGs that reached the ``record_pre`` call.
        n_sampled: DAGs on which the O(V+E) reachability walk was run.
        violated_pre: Sampled violations before ``_normalize_const_edges``.
        violated_post: Sampled violations after normalisation (target: 0).
        timeout: Full-rate count of ``CanonicalTimeoutError`` events.
        conversion_failure: Full-rate count of adapter exceptions.
        canon_raised: Full-rate count of non-timeout canonical exceptions.
        atlas_hit: Full-rate count of atlas cache hits.
    """

    __slots__ = (
        "enabled",
        "_rate",
        "n_seen",
        "n_sampled",
        "violated_pre",
        "violated_post",
        "timeout",
        "conversion_failure",
        "canon_raised",
        "atlas_hit",
        "_hist_n_sampled",
        "_hist_violated_pre",
        "_hist_violated_post",
        "_hist_timeout",
        "_hist_conversion_failure",
        "_hist_canon_raised",
        "_hist_atlas_hit",
    )

    def __init__(self) -> None:
        raw = os.environ.get("ISALSR_LEDGER_ENABLED", "0").strip()
        self.enabled: bool = raw not in ("", "0", "false", "False", "no", "NO")
        try:
            self._rate: int = max(1, int(os.environ.get("ISALSR_LEDGER_SAMPLE_RATE", "1")))
        except ValueError:
            self._rate = 1
        self.n_seen: int = 0
        self.n_sampled: int = 0
        self.violated_pre: int = 0
        self.violated_post: int = 0
        self.timeout: int = 0
        self.conversion_failure: int = 0
        self.canon_raised: int = 0
        self.atlas_hit: int = 0
        # Per-k denominator: sampled DAGs per k regardless of violation outcome.
        # Divides violated_pre_hist and violated_post_hist to yield per-k rates.
        # record_post shares the identical sampling decision as record_pre
        # (same modular condition on the already-incremented n_seen), so this
        # single histogram is the correct denominator for both.
        self._hist_n_sampled: dict[int, int] = {}
        self._hist_violated_pre: dict[int, int] = {}
        self._hist_violated_post: dict[int, int] = {}
        self._hist_timeout: dict[int, int] = {}
        self._hist_conversion_failure: dict[int, int] = {}
        self._hist_canon_raised: dict[int, int] = {}
        self._hist_atlas_hit: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Sampled events — O(V+E) reachability walk
    # ------------------------------------------------------------------ #

    def record_pre(self, dag: LabeledDAG) -> None:
        """Record reachability BEFORE ``_normalize_const_edges``.

        Called inside the adapter immediately before CONST creation edges
        are repaired.  Increments ``n_seen``; runs the reachability walk
        only on every N-th call (``ISALSR_LEDGER_SAMPLE_RATE``).

        On every sampled call, ``k = count_nonvar(dag)`` is computed and
        ``_hist_n_sampled[k]`` is incremented unconditionally.  This is
        the per-k denominator for both ``violated_pre_hist`` and
        ``violated_post_hist`` (``record_post`` shares the identical
        sampling decision — see its docstring).

        Args:
            dag: DAG as built by the adapter, before normalisation.
        """
        if not self.enabled:
            return
        sampled = self.n_seen % self._rate == 0
        self.n_seen += 1
        if not sampled:
            return
        self.n_sampled += 1
        # k is computed unconditionally so the denominator histogram is always
        # populated, enabling per-k violation rates even at high sample rates.
        k = count_nonvar(dag)
        self._hist_n_sampled[k] = self._hist_n_sampled.get(k, 0) + 1
        if violates_precondition(dag):
            self.violated_pre += 1
            self._hist_violated_pre[k] = self._hist_violated_pre.get(k, 0) + 1

    def record_post(self, dag: LabeledDAG) -> None:
        """Record reachability AFTER normalisation, before canonicalisation.

        Called in the runner immediately after receiving the normalised DAG
        from the adapter (and before the atlas lookup / ``fast_canonical_string``
        call).  Shares the sampling decision with ``record_pre`` by checking
        the same modular condition on the already-incremented ``n_seen``.

        **Denominator**: ``_hist_n_sampled`` (populated by ``record_pre``) is
        the correct per-k denominator for ``violated_post_hist`` as well as
        ``violated_pre_hist``.  The two methods fire on the same DAG and the
        same sampling gate, so one denominator covers both.  No separate
        post-denominator is needed or maintained.

        Args:
            dag: Normalised DAG the canonicaliser will receive.
        """
        if not self.enabled:
            return
        # n_seen was incremented by record_pre for this DAG.
        # The call that just ran had index (n_seen - 1); it was sampled iff
        # (n_seen - 1) % rate == 0.
        if (self.n_seen - 1) % self._rate != 0:
            return
        if violates_precondition(dag):
            self.violated_post += 1
            k = count_nonvar(dag)
            self._hist_violated_post[k] = self._hist_violated_post.get(k, 0) + 1

    # ------------------------------------------------------------------ #
    # Full-rate events — O(1), always counted
    # ------------------------------------------------------------------ #

    def record_conversion_failure(self) -> None:
        """Count an adapter exception.

        No DAG is available so k is unknown; the histogram uses sentinel -1.
        """
        if not self.enabled:
            return
        self.conversion_failure += 1
        self._hist_conversion_failure[-1] = self._hist_conversion_failure.get(-1, 0) + 1

    def record_timeout(self, dag: LabeledDAG) -> None:
        """Count a ``CanonicalTimeoutError`` from ``fast_canonical_string``.

        Args:
            dag: The DAG that triggered the timeout.
        """
        if not self.enabled:
            return
        self.timeout += 1
        k = count_nonvar(dag)
        self._hist_timeout[k] = self._hist_timeout.get(k, 0) + 1

    def record_canon_raised(self, dag: LabeledDAG) -> None:
        """Count a non-timeout exception from ``fast_canonical_string``.

        Args:
            dag: The DAG that triggered the exception.
        """
        if not self.enabled:
            return
        self.canon_raised += 1
        k = count_nonvar(dag)
        self._hist_canon_raised[k] = self._hist_canon_raised.get(k, 0) + 1

    def record_atlas_hit(self, dag: LabeledDAG) -> None:
        """Count an atlas cache hit.

        Args:
            dag: The DAG whose canonical was found in the atlas.
        """
        if not self.enabled:
            return
        self.atlas_hit += 1
        k = count_nonvar(dag)
        self._hist_atlas_hit[k] = self._hist_atlas_hit.get(k, 0) + 1

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of all counters and histograms.

        Histogram keys are stringified so the result is directly
        ``json.dumps``-able.

        Returns:
            Dict with scalar counters and ``*_hist`` variants keyed by
            ``str(k)`` where k is the number of non-VAR nodes.
        """

        def _sk(d: dict[int, int]) -> dict[str, int]:
            return {str(k): v for k, v in sorted(d.items())}

        return {
            "enabled": self.enabled,
            "sample_rate": self._rate,
            "n_seen": self.n_seen,
            "n_sampled": self.n_sampled,
            # Per-k denominator.  Divides violated_pre_hist and violated_post_hist
            # to yield per-k violation rates.  record_post shares the identical
            # sampling gate as record_pre, so this single histogram denominates both.
            "n_sampled_hist": _sk(self._hist_n_sampled),
            "violated_pre": self.violated_pre,
            "violated_pre_hist": _sk(self._hist_violated_pre),
            "violated_post": self.violated_post,
            "violated_post_hist": _sk(self._hist_violated_post),
            "timeout": self.timeout,
            "timeout_hist": _sk(self._hist_timeout),
            "conversion_failure": self.conversion_failure,
            "conversion_failure_hist": _sk(self._hist_conversion_failure),
            "canon_raised": self.canon_raised,
            "canon_raised_hist": _sk(self._hist_canon_raised),
            "atlas_hit": self.atlas_hit,
            "atlas_hit_hist": _sk(self._hist_atlas_hit),
        }
