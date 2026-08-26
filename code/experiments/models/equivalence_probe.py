"""EquivalenceProbe: cross-engine canonicaliser observability for evolved DAGs.

Module-level ``ACTIVE_PROBE`` is ``None`` by default; the production code path
is byte-for-byte unchanged when no probe is installed.  Call sites add exactly
one ``is not None`` guard:

    import experiments.models.equivalence_probe as _ep
    if _ep.ACTIVE_PROBE is not None:
        _ep.ACTIVE_PROBE.record_bingo(dag, indv)

Design points:
- The probe is **observational**: it does not touch the canonical string that
  the runner uses.  It independently computes both ``backend='python'`` and
  ``backend='cpp'`` and compares them.
- Legacy k is derived by calling the adapter with ``decompose=False`` on the
  same host object that produced the split dag.
- Thread-safe via a ``threading.Lock`` (single-process runs only; UDFS uses
  ``processes=1`` in all production configs).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level active probe.  None → inert; set by the gate driver.
# ---------------------------------------------------------------------------

#: Active probe instance. ``None`` (the default) means the production code
#: path is completely unchanged.
ACTIVE_PROBE: EquivalenceProbe | None = None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MismatchCase:
    """One canonicalisation mismatch or error divergence between two engines.

    Attributes:
        host: ``"bingo"`` or ``"udfs"``.
        k_split: Non-VAR node count in the split (T16) encoding.
        canonical_a: Canonical string from engine A, or ``None`` if it raised.
        canonical_b: Canonical string from engine B, or ``None`` if it raised.
        error_a: Exception description from engine A, or ``None``.
        error_b: Exception description from engine B, or ``None``.
    """

    host: str
    k_split: int
    canonical_a: str | None = None
    canonical_b: str | None = None
    error_a: str | None = None
    error_b: str | None = None


def _k_distribution(vals: list[int]) -> dict[str, Any]:
    """Compute mean, median, p95 and histogram for a list of k values.

    Args:
        vals: Observed k values (non-negative integers).

    Returns:
        Dict with ``count``, ``mean``, ``median``, ``p95``, and ``histogram``
        (key = str(k), value = count).
    """
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "histogram": {},
        }
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    # Median: lower median for even n
    if n % 2 == 1:
        median_val: float = float(sorted_vals[n // 2])
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    # p95: 95th percentile (floor index)
    p95_idx = max(0, int(n * 0.95) - 1)
    p95_val = sorted_vals[p95_idx]
    hist: dict[str, int] = {}
    for v in vals:
        key = str(v)
        hist[key] = hist.get(key, 0) + 1
    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "p95": p95_val,
        "histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
    }


# ---------------------------------------------------------------------------
# EquivalenceProbe
# ---------------------------------------------------------------------------


class EquivalenceProbe:
    """Compare Python vs C++ canonicalisation on every evolved DAG seen.

    Installed as ``ACTIVE_PROBE`` by the gate driver; absent during normal
    production runs.

    Attributes:
        engine_a: Name of engine A (normally ``"python"``).
        engine_b: Name of engine B (normally ``"cpp"``).
    """

    def __init__(
        self,
        engine_a: str = "python",
        engine_b: str = "cpp",
    ) -> None:
        self.engine_a = engine_a
        self.engine_b = engine_b
        self._lock = threading.Lock()

        # Per-host counters
        self._bingo_count: int = 0
        self._udfs_count: int = 0
        self._mismatches: int = 0
        self._errors: int = 0

        # k values (non_var_nodes count) for each encoding
        self._bingo_k_split: list[int] = []
        self._bingo_k_legacy: list[int] = []
        self._udfs_k_split: list[int] = []
        self._udfs_k_legacy: list[int] = []

        # Alphabet violation counts in the split dag / canonical string
        self._sub_node_count: int = 0
        self._div_node_count: int = 0
        self._sub_char_count: int = 0  # '-' chars in split canonical (engine A)
        self._div_char_count: int = 0  # '/' chars in split canonical (engine A)

        # Mismatch details (capped at 20)
        self._mismatch_cases: list[MismatchCase] = []

    # ------------------------------------------------------------------
    # Public call sites — one per host
    # ------------------------------------------------------------------

    def record_bingo(self, dag: Any, indv: Any) -> None:
        """Record a DAG from Bingo's ``_serial_eval`` fast-canonical branch.

        Args:
            dag: ``LabeledDAG`` (split / T16 encoding) built by
                ``agraph_to_labeled_dag``.
            indv: Bingo ``AGraph`` individual used to build the legacy
                encoding for k comparison.
        """
        from experiments.models.bingo.adapter import agraph_to_labeled_dag

        try:
            legacy_dag = agraph_to_labeled_dag(indv, decompose=False)
        except Exception:  # noqa: BLE001
            legacy_dag = None

        self._record("bingo", dag, legacy_dag)

    def record_udfs(self, dag: Any, cgraph: Any) -> None:
        """Record a DAG from UDFS's ``wrapped()`` fast-canonical branch.

        Args:
            dag: ``LabeledDAG`` (split / T16 encoding) built by
                ``compgraph_to_labeled_dag``.
            cgraph: UDFS ``CompGraph`` used to build the legacy encoding.
        """
        from experiments.models.udfs.adapter import compgraph_to_labeled_dag

        try:
            legacy_dag = compgraph_to_labeled_dag(cgraph, decompose=False)
        except Exception:  # noqa: BLE001
            legacy_dag = None

        self._record("udfs", dag, legacy_dag)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _record(
        self,
        host: str,
        split_dag: Any,
        legacy_dag: Any | None,
    ) -> None:
        """Compute both engines, compare, and update stats.

        Args:
            host: ``"bingo"`` or ``"udfs"``.
            split_dag: ``LabeledDAG`` with split (T16) encoding.
            legacy_dag: ``LabeledDAG`` with legacy encoding, or ``None`` if
                the adapter raised.
        """
        from isalsr.core.canonical import fast_canonical_string  # type: ignore[attr-defined]
        from isalsr.core.node_types import NodeType

        # --- Engine A ---
        try:
            can_a: str | None = fast_canonical_string(  # type: ignore[call-arg]
                split_dag, backend=self.engine_a
            )
            err_a: str | None = None
        except Exception as exc:  # noqa: BLE001
            can_a = None
            err_a = f"{type(exc).__name__}: {exc}"

        # --- Engine B ---
        try:
            can_b: str | None = fast_canonical_string(  # type: ignore[call-arg]
                split_dag, backend=self.engine_b
            )
            err_b: str | None = None
        except Exception as exc:  # noqa: BLE001
            can_b = None
            err_b = f"{type(exc).__name__}: {exc}"

        # --- k counts ---
        k_split = len(split_dag.non_var_nodes())
        k_legacy = len(legacy_dag.non_var_nodes()) if legacy_dag is not None else -1

        # --- Alphabet check (NodeType.SUB / NodeType.DIV in split dag) ---
        sub_nodes = sum(
            1 for i in range(split_dag.node_count) if split_dag.node_label(i) == NodeType.SUB
        )
        div_nodes = sum(
            1 for i in range(split_dag.node_count) if split_dag.node_label(i) == NodeType.DIV
        )
        sub_chars = can_a.count("-") if can_a is not None else 0
        div_chars = can_a.count("/") if can_a is not None else 0

        # --- Mismatch detection ---
        # Both succeeded: strings must match. Otherwise: error messages must
        # match (catches "one raised, one didn't" and "different exceptions").
        is_mismatch = can_a != can_b if err_a is None and err_b is None else err_a != err_b

        with self._lock:
            if host == "bingo":
                self._bingo_count += 1
                self._bingo_k_split.append(k_split)
                if k_legacy >= 0:
                    self._bingo_k_legacy.append(k_legacy)
            else:
                self._udfs_count += 1
                self._udfs_k_split.append(k_split)
                if k_legacy >= 0:
                    self._udfs_k_legacy.append(k_legacy)

            self._sub_node_count += sub_nodes
            self._div_node_count += div_nodes
            self._sub_char_count += sub_chars
            self._div_char_count += div_chars

            if err_a is not None or err_b is not None:
                self._errors += 1

            if is_mismatch:
                self._mismatches += 1
                if len(self._mismatch_cases) < 20:
                    self._mismatch_cases.append(
                        MismatchCase(
                            host=host,
                            k_split=k_split,
                            canonical_a=can_a,
                            canonical_b=can_b,
                            error_a=err_a,
                            error_b=err_b,
                        )
                    )

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Assemble a serialisable summary of all observations so far.

        Returns:
            Dict containing per-host DAG counts, k-distribution blocks,
            alphabet violation counts, mismatch cases, and engine names.
        """
        b_split = _k_distribution(self._bingo_k_split)
        b_legacy = _k_distribution(self._bingo_k_legacy)
        u_split = _k_distribution(self._udfs_k_split)
        u_legacy = _k_distribution(self._udfs_k_legacy)

        def _delta(split: dict[str, Any], legacy: dict[str, Any]) -> float | None:
            """Paired mean delta: split_mean - legacy_mean."""
            if split["mean"] is None or legacy["mean"] is None:
                return None
            return float(split["mean"]) - float(legacy["mean"])

        return {
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "dags_compared_total": self._bingo_count + self._udfs_count,
            "dags_compared": {
                "bingo": self._bingo_count,
                "udfs": self._udfs_count,
            },
            "mismatches": self._mismatches,
            "errors": self._errors,
            "k_distributions": {
                "bingo": {
                    "split": b_split,
                    "legacy": b_legacy,
                    "delta_split_minus_legacy_mean": _delta(b_split, b_legacy),
                },
                "udfs": {
                    "split": u_split,
                    "legacy": u_legacy,
                    "delta_split_minus_legacy_mean": _delta(u_split, u_legacy),
                },
            },
            "alphabet_checks": {
                "sub_nodes_in_split_dag": self._sub_node_count,
                "div_nodes_in_split_dag": self._div_node_count,
                "sub_chars_in_canonical": self._sub_char_count,
                "div_chars_in_canonical": self._div_char_count,
            },
            "mismatch_cases": [
                {
                    "host": c.host,
                    "k_split": c.k_split,
                    "canonical_a": c.canonical_a,
                    "canonical_b": c.canonical_b,
                    "error_a": c.error_a,
                    "error_b": c.error_b,
                }
                for c in self._mismatch_cases
            ],
        }
