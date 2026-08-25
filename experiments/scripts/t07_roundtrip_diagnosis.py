"""T07 round-trip fidelity diagnosis: unified comparator + failure classification.

(a) Comparator fix
------------------
The original study used ``fast_canonical_string`` (fcs) for the *keep* arm
round-trip check but ``_structural_key`` for the *drop* arm.
``_structural_key`` compares absolute node IDs, which vary between different
DAG constructions of the same abstract graph (e.g. adapter ordering vs S2D
ordering). It is therefore **not a valid comparator** for round-trip fidelity
across different constructions of isomorphic DAGs.

Unified check (this script) for any DAG D, with m input variables::

    keep_cs  = fcs(D, backend="cpp")
    drop_cs  = fcs_raw(D)   (None if drop raises)

    keep RT: fcs(S2D(keep_cs, m)) == keep_cs
    drop RT: fcs(S2D(drop_cs, m)) == keep_cs   ← fcs, NOT structural_key

Both arms are compared under the same oracle (fcs).  This is sound because
fcs(D) == fcs(D') iff D ≅ D' as labeled DAGs (paper Theorem 3.13), and fcs
applies normalize_const_creation internally, making it normalisation-aware.

(b) Failure classification
--------------------------
For every DAG where the keep arm round-trip fails, records::

    has_sub, has_div       — non-commutative SUB / DIV present
    has_pow                — POW present
    has_const              — CONST leaf present
    k                      — number of internal (non-VAR) nodes
    binary_single_edge     — count of SUB/DIV/POW nodes with exactly 1 in-edge
                             (occurs when Bingo has src1 == src2 for a binary op)
    indegree_profile       — per-NodeType list of in-degrees

Minimal reproducers (lowest k) are serialised to JSON.

Usage (local, collect ~100-200 k adapter DAGs)
-----------------------------------------------
    python -m experiments.scripts.t07_roundtrip_diagnosis \\
        --max-time 300 \\
        --out /tmp/t07_diagnosis

Usage (single problem for quick smoke)
---------------------------------------
    python -m experiments.scripts.t07_roundtrip_diagnosis \\
        --problems nguyen:Nguyen-5 --max-time 60 \\
        --out /tmp/t07_diag_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NodeType
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger("t07.diagnosis")

# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

# NodeTypes that are strictly non-commutative (order matters)
_NONCOMM: frozenset[NodeType] = frozenset({NodeType.SUB, NodeType.DIV, NodeType.POW})


@dataclass
class FailureFeatures:
    """Structural features of a failing adapter DAG.

    Args:
        has_sub: Any SUB node present.
        has_div: Any DIV node present.
        has_pow: Any POW node present.
        has_const: Any CONST leaf present.
        k: Number of internal (non-VAR) nodes.
        binary_single_edge: Count of BINARY_OPS nodes with exactly 1 in-edge.
        indegree_profile: {NodeType name: [in-degrees]} for all internal nodes.
        m: Number of VAR nodes (input variables).
    """

    has_sub: bool
    has_div: bool
    has_pow: bool
    has_const: bool
    k: int
    binary_single_edge: int
    indegree_profile: dict[str, list[int]]
    m: int


def classify_failure_features(dag: LabeledDAG) -> FailureFeatures:
    """Extract structural failure-classification features from *dag*.

    Args:
        dag: The adapter-produced LabeledDAG to inspect.

    Returns:
        FailureFeatures populated from *dag*.
    """
    has_sub = has_div = has_pow = has_const = False
    k = 0
    binary_single_edge = 0
    m = 0
    profile: dict[str, list[int]] = defaultdict(list)

    for i in range(dag.node_count):
        lbl = dag.node_label(i)
        if lbl == NodeType.VAR:
            m += 1
            continue
        k += 1
        indeg = dag.in_degree(i)
        profile[lbl.name].append(indeg)

        if lbl == NodeType.SUB:
            has_sub = True
        elif lbl == NodeType.DIV:
            has_div = True
        elif lbl == NodeType.POW:
            has_pow = True
        elif lbl == NodeType.CONST:
            has_const = True

        # Binary op with only 1 in-edge: adapter src1 == src2 case
        if lbl in BINARY_OPS and indeg <= 1:
            binary_single_edge += 1

    return FailureFeatures(
        has_sub=has_sub,
        has_div=has_div,
        has_pow=has_pow,
        has_const=has_const,
        k=k,
        binary_single_edge=binary_single_edge,
        indegree_profile=dict(profile),
        m=m,
    )


# ---------------------------------------------------------------------------
# Unified round-trip comparator
# ---------------------------------------------------------------------------


def round_trip_keep(keep_cs: str, m: int, timeout: float) -> bool:
    """Round-trip check for the keep arm.

    Decodes *keep_cs* via S2D, re-canonicalises with fcs, and checks identity.

    Args:
        keep_cs: Canonical string from keep arm (fcs(D)).
        m: Number of input variables for S2D decoding.
        timeout: Time budget for re-canonicalisation.

    Returns:
        True iff fcs(S2D(keep_cs, m)) == keep_cs.
    """
    try:
        from isalsr.core.canonical import fast_canonical_string

        dag2 = StringToDAG(keep_cs, num_variables=m).run()
        cs2 = fast_canonical_string(dag2, timeout=timeout, backend="cpp")
        return bool(cs2 == keep_cs)
    except Exception:  # noqa: BLE001
        return False


def round_trip_drop_unified(drop_cs: str, keep_cs: str, m: int, timeout: float) -> bool:
    """Round-trip check for the drop arm using the unified fcs comparator.

    Decodes *drop_cs* via S2D, re-canonicalises with fcs (keep arm's oracle),
    and compares to *keep_cs*.  This replaces the invalid ``_structural_key``
    comparator used in the original T07 study.

    For adapter DAGs where the adapter has already applied ``normalize_const_creation``,
    drop_cs == keep_cs (both arms agree), so this check is equivalent to
    ``round_trip_keep``.  The key point is that BOTH arms now use the same oracle.

    Args:
        drop_cs: Canonical string from drop arm (fcs_raw(D)).
        keep_cs: Canonical string from keep arm (fcs(D)); the reference.
        m: Number of input variables for S2D decoding.
        timeout: Time budget for re-canonicalisation.

    Returns:
        True iff fcs(S2D(drop_cs, m)) == keep_cs.
    """
    try:
        from isalsr.core.canonical import fast_canonical_string

        dag2 = StringToDAG(drop_cs, num_variables=m).run()
        cs2 = fast_canonical_string(dag2, timeout=timeout, backend="cpp")
        return bool(cs2 == keep_cs)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Arm scorers (thin wrappers mirroring t07_norm_removal_study)
# ---------------------------------------------------------------------------


def _score_keep(dag: LabeledDAG, timeout: float) -> tuple[str | None, str]:
    """Compute keep arm canonical string.

    Args:
        dag: The DAG to canonicalise.
        timeout: Time budget in seconds.

    Returns:
        (keep_cs, status) where status is "ok", "raised", or "timeout".
        keep_cs is None when status != "ok".
    """
    from isalsr.core.canonical import CanonicalTimeoutError, fast_canonical_string

    try:
        cs = fast_canonical_string(dag, timeout=timeout, backend="cpp")
        return cs, "ok"
    except CanonicalTimeoutError:
        return None, "timeout"
    except Exception:  # noqa: BLE001
        return None, "raised"


def _score_drop(dag: LabeledDAG, timeout: float) -> tuple[str | None, str]:
    """Compute drop arm canonical string (no normalization).

    Args:
        dag: The DAG to canonicalise.
        timeout: Time budget in seconds.

    Returns:
        (drop_cs, status) where status is "ok", "raised", or "timeout".
        drop_cs is None when status != "ok".
    """
    try:
        from isalsr.core import _native
        from isalsr.core.canonical import _py_dag_to_native

        raw = _native.testing.fast_canonical_string_raw
        cs = raw(_py_dag_to_native(dag), timeout)
        return cs, "ok"
    except Exception as exc:  # noqa: BLE001
        if "timeout" in str(exc).lower():
            return None, "timeout"
        return None, "raised"


# ---------------------------------------------------------------------------
# DAG serialisation for minimal reproducers
# ---------------------------------------------------------------------------


def dag_to_json(
    dag: LabeledDAG,
    keep_cs: str | None = None,
    decoded_cs: str | None = None,
    features: FailureFeatures | None = None,
) -> dict[str, Any]:
    """Serialise *dag* to a JSON-compatible dict for minimal reproducer dumps.

    Args:
        dag: The LabeledDAG to serialise.
        keep_cs: The original keep-arm canonical string.
        decoded_cs: The canonical string produced after round-trip decode.
        features: Pre-computed FailureFeatures (optional).

    Returns:
        JSON-compatible dict with nodes, edges, input_order, and metadata.
    """
    nodes: list[dict[str, Any]] = []
    for i in range(dag.node_count):
        lbl = dag.node_label(i)
        entry: dict[str, Any] = {"id": i, "label": lbl.name}
        d = dag.node_data(i)
        if "var_index" in d:
            entry["var_index"] = int(d["var_index"])
        if "const_value" in d:
            entry["const_value"] = float(d["const_value"])
        entry["in_degree"] = dag.in_degree(i)
        entry["out_degree"] = dag.out_degree(i)
        nodes.append(entry)

    edges: list[list[int]] = []
    for i in range(dag.node_count):
        for j in sorted(dag.out_neighbors(i)):
            edges.append([i, j])

    input_order: dict[str, list[int]] = {}
    for i in range(dag.node_count):
        oi = dag.ordered_inputs(i)
        if oi:
            input_order[str(i)] = oi

    result: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "input_order": input_order,
    }
    if keep_cs is not None:
        result["keep_cs"] = keep_cs
    if decoded_cs is not None:
        result["decoded_cs"] = decoded_cs
    if features is not None:
        result["features"] = asdict(features)
    return result


# ---------------------------------------------------------------------------
# Diagnostic statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticStats:
    """Per-arm and failure-classification statistics.

    Args:
        comparator: Label for the comparator used (always "fast_canonical_string").
    """

    comparator: str = "fast_canonical_string"
    n_total: int = 0
    n_keep_ok: int = 0
    n_keep_rt_checked: int = 0
    n_keep_rt_ok: int = 0
    n_drop_ok: int = 0
    n_drop_rt_checked: int = 0
    n_drop_rt_ok: int = 0
    n_arms_agree: int = 0
    # Failure breakdown
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def keep_rt_rate(self) -> float:
        """Round-trip success rate for keep arm."""
        return (
            self.n_keep_rt_ok / self.n_keep_rt_checked if self.n_keep_rt_checked else float("nan")
        )

    @property
    def drop_rt_rate(self) -> float:
        """Round-trip success rate for drop arm (unified comparator)."""
        return (
            self.n_drop_rt_ok / self.n_drop_rt_checked if self.n_drop_rt_checked else float("nan")
        )

    def failure_table(self) -> dict[str, Any]:
        """Aggregate failure counts by structural feature.

        Returns:
            Dict with per-feature failure breakdowns.
        """
        if not self.failures:
            return {}
        counters: dict[str, int] = defaultdict(int)
        k_dist: list[int] = []
        for f in self.failures:
            feat = f.get("features", {})
            if feat.get("has_sub"):
                counters["has_sub"] += 1
            if feat.get("has_div"):
                counters["has_div"] += 1
            if feat.get("has_pow"):
                counters["has_pow"] += 1
            if feat.get("has_const"):
                counters["has_const"] += 1
            bse = feat.get("binary_single_edge", 0)
            if bse > 0:
                counters["binary_single_edge"] += 1
            k_dist.append(feat.get("k", -1))

        n_fail = len(self.failures)
        result: dict[str, Any] = {
            "n_failures": n_fail,
            "pct_has_sub": 100.0 * counters["has_sub"] / n_fail,
            "pct_has_div": 100.0 * counters["has_div"] / n_fail,
            "pct_has_pow": 100.0 * counters["has_pow"] / n_fail,
            "pct_has_const": 100.0 * counters["has_const"] / n_fail,
            "pct_binary_single_edge": 100.0 * counters["binary_single_edge"] / n_fail,
            "k_min": min(k_dist) if k_dist else -1,
            "k_max": max(k_dist) if k_dist else -1,
            "k_mean": sum(k_dist) / len(k_dist) if k_dist else float("nan"),
        }
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict (excluding full failures list)."""
        return {
            "comparator": self.comparator,
            "n_total": self.n_total,
            "keep": {
                "n_ok": self.n_keep_ok,
                "n_rt_checked": self.n_keep_rt_checked,
                "n_rt_ok": self.n_keep_rt_ok,
                "rt_rate": self.keep_rt_rate,
            },
            "drop": {
                "n_ok": self.n_drop_ok,
                "n_rt_checked": self.n_drop_rt_checked,
                "n_rt_ok": self.n_drop_rt_ok,
                "rt_rate": self.drop_rt_rate,
            },
            "n_arms_agree": self.n_arms_agree,
            "failure_table": self.failure_table(),
        }


# ---------------------------------------------------------------------------
# Diagnostic recorder (monkey-patch hook)
# ---------------------------------------------------------------------------


class DiagnosticRecorder:
    """Intercept fast_canonical_string calls to run both-arm unified RT checks.

    Installed as a monkey-patch during a bingo/udfs search.  Returns the keep
    arm result so the search proceeds normally.  The drop arm RT check is run
    in parallel using the unified fcs comparator.

    Args:
        original_fn: Saved reference to the real fast_canonical_string.
        timeout: Default canonicalisation time budget.
        max_failures_stored: Maximum number of failing DAGs to store in full.
        max_reproducers: Maximum minimal reproducers to collect.
    """

    def __init__(
        self,
        original_fn: Callable[..., str],
        timeout: float,
        max_failures_stored: int = 500,
        max_reproducers: int = 20,
    ) -> None:
        self._original = original_fn
        self.timeout = timeout
        self.max_failures_stored = max_failures_stored
        self.max_reproducers = max_reproducers
        self.stats = DiagnosticStats()
        self._call_count = 0
        # Minimal reproducers sorted by k (keep the smallest)
        self._reproducers: list[dict[str, Any]] = []
        # Load drop-arm function references once at init time so __call__ never
        # re-imports from canonical_mod (which is monkey-patched to self).
        self._raw_fn: Callable[..., str] | None = None
        self._py_dag_to_native: Callable[[LabeledDAG], Any] | None = None
        try:
            from isalsr.core import _native  # noqa: PLC0415
            from isalsr.core.canonical import _py_dag_to_native as _pdn

            self._raw_fn = _native.testing.fast_canonical_string_raw
            self._py_dag_to_native = _pdn
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Internal arm scorers — use stored references, NEVER re-import
    # canonical_mod to avoid infinite recursion when monkey-patched.
    # ------------------------------------------------------------------

    def _keep_cs(self, dag: LabeledDAG, budget: float) -> tuple[str | None, str]:
        """Score keep arm via the saved original (pre-patch) function.

        Args:
            dag: DAG to canonicalise.
            budget: Time limit in seconds.

        Returns:
            (canonical_string, status) where status is "ok", "raised", or "timeout".
        """
        from isalsr.core.canonical import CanonicalTimeoutError

        try:
            cs = self._original(dag, timeout=budget, backend="cpp")
            return cs, "ok"
        except CanonicalTimeoutError:
            return None, "timeout"
        except Exception:  # noqa: BLE001
            return None, "raised"

    def _drop_cs(self, dag: LabeledDAG, budget: float) -> tuple[str | None, str]:
        """Score drop arm via the stored raw function (no normalization).

        Args:
            dag: DAG to canonicalise.
            budget: Time limit in seconds.

        Returns:
            (canonical_string, status).
        """
        if self._raw_fn is None or self._py_dag_to_native is None:
            return None, "raised"
        try:
            cs = self._raw_fn(self._py_dag_to_native(dag), budget)
            return cs, "ok"
        except Exception as exc:  # noqa: BLE001
            if "timeout" in str(exc).lower():
                return None, "timeout"
            return None, "raised"

    def _rt_check(self, cs: str, m: int, budget: float) -> tuple[bool, str | None]:
        """Unified round-trip check: S2D(cs) then re-canonicalise via original.

        Args:
            cs: Canonical string to decode and re-canonicalise.
            m: Number of input variables for S2D.
            budget: Time limit in seconds.

        Returns:
            (ok, decoded_cs_or_None).
        """
        try:
            dag2 = StringToDAG(cs, num_variables=m).run()
            cs2 = self._original(dag2, timeout=budget, backend="cpp")
            return bool(cs2 == cs), cs2
        except Exception:  # noqa: BLE001
            return False, None

    def __call__(
        self,
        dag: LabeledDAG,
        timeout: float | None = None,
        mode: str = "wl_only",
        backend: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Score both arms and return keep result.

        IMPORTANT: all scoring uses ``self._original`` and ``self._raw_fn``
        which are stored references to the pre-patch functions.  This avoids
        infinite recursion when canonical_mod.fast_canonical_string is
        monkey-patched to this recorder.

        Args:
            dag: DAG arriving at the canonicaliser.
            timeout: Per-call budget override.
            mode: Ignored (always wl_only via C++).
            backend: Ignored (always cpp).
            **kwargs: Accepted for signature compatibility.

        Returns:
            Canonical string from the keep arm.

        Raises:
            RuntimeError: If the keep arm fails.
        """
        budget = float(timeout) if timeout is not None else self.timeout
        m = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)

        self.stats.n_total += 1
        self._call_count += 1

        # --- Keep arm (uses stored self._original, not canonical_mod) ---
        keep_cs, keep_status = self._keep_cs(dag, budget)

        if keep_status != "ok" or keep_cs is None:
            raise RuntimeError("Fast canonical D2S: no valid operation found")

        self.stats.n_keep_ok += 1

        # --- Drop arm (uses stored self._raw_fn, not canonical_mod) ---
        drop_cs, drop_status = self._drop_cs(dag, budget)

        if drop_status == "ok" and drop_cs is not None:
            self.stats.n_drop_ok += 1
            if keep_cs == drop_cs:
                self.stats.n_arms_agree += 1

        # --- Unified round-trip: keep arm ---
        self.stats.n_keep_rt_checked += 1
        keep_rt_ok, decoded_cs = self._rt_check(keep_cs, m, budget)
        if keep_rt_ok:
            self.stats.n_keep_rt_ok += 1
        else:
            features = classify_failure_features(dag)
            if len(self.stats.failures) < self.max_failures_stored:
                self.stats.failures.append(
                    dag_to_json(dag, keep_cs=keep_cs, decoded_cs=decoded_cs, features=features)
                )
            self._maybe_add_reproducer(dag, keep_cs, decoded_cs, features)

        # --- Unified round-trip: drop arm ---
        # Use self._original directly (NOT round_trip_drop_unified which re-imports
        # canonical_mod and would call the monkey-patched recorder recursively).
        if drop_status == "ok" and drop_cs is not None:
            self.stats.n_drop_rt_checked += 1
            try:
                _dag_decoded = StringToDAG(drop_cs, num_variables=m).run()
                _cs_decoded = self._original(_dag_decoded, timeout=budget, backend="cpp")
                drop_rt_ok = bool(_cs_decoded == keep_cs)
            except Exception:  # noqa: BLE001
                drop_rt_ok = False
            if drop_rt_ok:
                self.stats.n_drop_rt_ok += 1

        return keep_cs

    def _maybe_add_reproducer(
        self,
        dag: LabeledDAG,
        keep_cs: str,
        decoded_cs: str | None,
        features: FailureFeatures,
    ) -> None:
        """Maintain a sorted list of minimal (smallest k) failing reproducers.

        Args:
            dag: The failing DAG.
            keep_cs: Canonical string from keep arm.
            decoded_cs: Canonical string after decode.
            features: Structural features.
        """
        entry = dag_to_json(dag, keep_cs=keep_cs, decoded_cs=decoded_cs, features=features)
        self._reproducers.append(entry)
        # Sort by k ascending; keep only max_reproducers smallest
        self._reproducers.sort(key=lambda x: x.get("features", {}).get("k", 999))
        self._reproducers = self._reproducers[: self.max_reproducers]

    @property
    def reproducers(self) -> list[dict[str, Any]]:
        """Minimal reproducing DAGs (sorted by k ascending)."""
        return list(self._reproducers)


# ---------------------------------------------------------------------------
# Problem loader (reused from t07_norm_removal_study pattern)
# ---------------------------------------------------------------------------


def _load_problem(suite: str, problem: str, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load train/test data for a named benchmark problem.

    Args:
        suite: Benchmark suite name ('nguyen', 'feynman', etc.).
        problem: Problem name (e.g. 'Nguyen-5').
        seed: Random seed for data generation.

    Returns:
        (x_train, y_train, x_test, y_test).
    """
    import importlib
    from typing import cast

    # Suite key -> (defining module, benchmark-list constant). The key and the
    # module name coincide except for ``cherrypicked``, a legacy suite key kept
    # because it is baked into the on-disk result tree.
    bench_source_map = {
        "nguyen": ("nguyen", "NGUYEN_BENCHMARKS"),
        "feynman": ("feynman", "FEYNMAN_BENCHMARKS"),
        "hard": ("hard", "HARD_BENCHMARKS"),
        "cherrypicked": ("structural", "STRUCTURAL_BENCHMARKS"),
        "roundoff": ("roundoff", "ROUNDOFF_BENCHMARKS"),
    }
    module_name, const_name = bench_source_map[suite]
    module = importlib.import_module(f"benchmarks.datasets.{module_name}")
    benches = getattr(module, const_name)
    if isinstance(benches, dict):
        benches = list(benches.values())
    bench = next(b for b in benches if b["name"] == problem)
    if suite == "nguyen":
        return cast(
            tuple[Any, Any, Any, Any],
            module.generate_data(bench, n_train=240, n_test=1000, seed=seed),
        )
    return cast(
        tuple[Any, Any, Any, Any],
        module.generate_data(bench, n_samples=1250, train_ratio=0.8, seed=seed),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

_DEFAULT_PROBLEMS = [
    "nguyen:Nguyen-1",
    "nguyen:Nguyen-5",
    "nguyen:Nguyen-9",
    "nguyen:Nguyen-11",
    "nguyen:Nguyen-12",
]


def run_diagnosis(
    problems: list[str],
    seed: int,
    max_time_per_problem: float,
    canonical_timeout: float,
    out_dir: Path,
    max_total_dags: int = 300_000,
) -> DiagnosticStats:
    """Run bingo search over *problems* to collect adapter DAGs for diagnosis.

    Monkey-patches ``fast_canonical_string`` with ``DiagnosticRecorder`` for the
    duration of each search.  Returns accumulated statistics.

    Args:
        problems: List of "suite:problem" strings.
        seed: Random seed for bingo.
        max_time_per_problem: Wall-clock budget per problem in seconds.
        canonical_timeout: Canonicalisation time budget in seconds.
        out_dir: Output directory for JSON results.
        max_total_dags: Stop early if this many DAGs collected.

    Returns:
        Accumulated DiagnosticStats across all problems.
    """
    import isalsr.core.canonical as canonical_mod
    from experiments.models.orchestrator import create_runner

    original_fn = canonical_mod.fast_canonical_string
    recorder = DiagnosticRecorder(
        original_fn=original_fn,
        timeout=canonical_timeout,
    )

    bingo_cfg_base: dict[str, Any] = {
        "population_size": 500,
        "stack_size": 32,
        "operators": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
        "use_simplification": False,
        "crossover_prob": 0.4,
        "mutation_prob": 0.4,
        "metric": "mse",
        "clo_alg": "lm",
        "generations": 100_000_000,
        "fitness_threshold": 1.0e-16,
        "max_evals": 100_000_000,
        "snapshot_frequency": 10,
        "canonicalization_timeout": canonical_timeout,
        "use_fast_canonical": True,
    }

    for prob_spec in problems:
        if recorder.stats.n_total >= max_total_dags:
            log.info("Reached max_total_dags=%d, stopping early.", max_total_dags)
            break

        suite, problem = prob_spec.split(":", 1)
        log.info("Starting problem %s (collected=%d)", prob_spec, recorder.stats.n_total)

        try:
            x_train, y_train, x_test, y_test = _load_problem(suite, problem, seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load %s: %s", prob_spec, exc)
            continue

        cfg: dict[str, Any] = {
            "bingo": {**bingo_cfg_base, "max_time": max_time_per_problem},
            "isalsr": {
                "canonicalization_timeout": canonical_timeout,
                "use_fast_canonical": True,
            },
        }

        try:
            canonical_mod.fast_canonical_string = recorder  # noqa: PGH003
            runner = create_runner("bingo", "isalsr", cfg)
            runner.fit(x_train, y_train, x_test, y_test, seed, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Search error on %s: %s", prob_spec, exc)
        finally:
            canonical_mod.fast_canonical_string = original_fn

        log.info(
            "After %s: total=%d keep_rt=%.4f drop_rt=%.4f",
            prob_spec,
            recorder.stats.n_total,
            recorder.stats.keep_rt_rate,
            recorder.stats.drop_rt_rate,
        )

    return recorder.stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for T07 round-trip fidelity diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        nargs="+",
        default=_DEFAULT_PROBLEMS,
        help="Problems as 'suite:name'. Default: 5 Nguyen problems.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-time",
        type=float,
        default=120.0,
        help="Wall-clock budget per problem (seconds). Default 120.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Canonicalisation time budget (seconds). Default 10.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--max-dags",
        type=int,
        default=300_000,
        help="Stop collecting after this many adapter DAGs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    stats = run_diagnosis(
        problems=args.problems,
        seed=args.seed,
        max_time_per_problem=args.max_time,
        canonical_timeout=args.timeout,
        out_dir=out_dir,
        max_total_dags=args.max_dags,
    )
    elapsed = time.monotonic() - t_start

    # Aggregate reproducers from the recorder (rebuilt via run_diagnosis)
    # The recorder is local; re-run was done in place, stats has failures
    # Collect minimal reproducers from stats.failures (sorted by k)
    reproducers = sorted(
        stats.failures[:20],
        key=lambda x: x.get("features", {}).get("k", 999),
    )[:3]

    summary = {
        "elapsed_sec": elapsed,
        "stats": stats.to_dict(),
        "n_reproducers": len(reproducers),
    }

    # Write outputs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if reproducers:
        reproducer_path = out_dir / "minimal_reproducers.json"
        reproducer_path.write_text(json.dumps(reproducers, indent=2, default=str))
        log.info("Wrote %d minimal reproducers to %s", len(reproducers), reproducer_path)

    # Print summary
    d = stats.to_dict()
    print(f"\nT07 Round-Trip Diagnosis  elapsed={elapsed:.1f}s")
    print(f"Total adapter DAGs collected: {stats.n_total:,}")
    print()
    print(f"Comparator for BOTH arms: {d['comparator']}")
    print()
    print(f"{'arm':<6} {'n_ok':>8} {'rt_checked':>11} {'rt_ok':>8} {'rt_rate':>9}")
    print("-" * 47)
    print(
        f"{'keep':<6} {d['keep']['n_ok']:>8,} {d['keep']['n_rt_checked']:>11,} "
        f"{d['keep']['n_rt_ok']:>8,} {d['keep']['rt_rate']:>9.5f}"
    )
    print(
        f"{'drop':<6} {d['drop']['n_ok']:>8,} {d['drop']['n_rt_checked']:>11,} "
        f"{d['drop']['n_rt_ok']:>8,} {d['drop']['rt_rate']:>9.5f}"
    )

    ft = stats.failure_table()
    if ft:
        n_fail = ft["n_failures"]
        print(f"\nFailure classification ({n_fail} keep-arm failures):")
        print(f"  has_sub:             {ft['pct_has_sub']:.1f}%")
        print(f"  has_div:             {ft['pct_has_div']:.1f}%")
        print(f"  has_pow:             {ft['pct_has_pow']:.1f}%")
        print(f"  has_const:           {ft['pct_has_const']:.1f}%")
        print(f"  binary_single_edge:  {ft['pct_binary_single_edge']:.1f}%")
        print(f"  k range:             [{ft['k_min']}, {ft['k_max']}] mean={ft['k_mean']:.1f}")
    else:
        print("\nNo keep-arm round-trip failures detected.")

    print(f"\nArms agree on canonical string: {stats.n_arms_agree}/{stats.n_drop_ok}")
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
