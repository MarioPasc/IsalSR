"""Benchmark driver for IsalSR canonicalisation cost, stratified by k-bucket.

Measures per-DAG canonicalisation cost for both Python and C++ engines across
the three k-buckets used in the paper (k < 5, 5 ≤ k < 15, 15 ≤ k < 32),
where k = number of internal (non-variable) nodes.

Protocol (adapted from sibling equivalence_gate.py)
-----------------------------------------------------
- 3 warmup repetitions, discarded.
- best-of-9 raw passes per repetition.
- median of 4 repetitions reported.
- Both engines measured in the same thermal state: Python and C++ are timed
  within the same repetition (Python first, then C++), alternating start order
  every other repetition so neither engine systematically benefits from cache
  warm-up.
- Time a full batch of DAGs per pass (not one-at-a-time) to amortise
  ``time.perf_counter`` overhead.

Usage
-----
Quick mode (< 3 min, for iteration)::

    python -m experiments.scripts.bench_canonical --quick --out report.json

Full mode (< 20 min single-core)::

    python -m experiments.scripts.bench_canonical --out report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — allows running without ``pip install -e``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.scripts._dag_generators import make_random_sr_dag
from isalsr.core.backends import build_info, engine
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants — never hardcoded in CLI args so the report is always
# self-describing.
# ---------------------------------------------------------------------------

FIXED_SEED: int = 20260101
"""Fixed RNG seed for all corpus generation.  Recorded in every report."""

N_WARMUP: int = 3
"""Warmup repetitions discarded before measurement."""

N_REPS: int = 4
"""Measured repetitions whose per-DAG costs are aggregated via median."""

N_PASSES_PER_REP: int = 9
"""Raw timing passes per repetition; best (minimum) is kept."""

# k-bucket boundaries (inclusive low, inclusive high) matching the paper.
K_BUCKETS: list[tuple[str, int, int]] = [
    ("k_lt5", 1, 4),
    ("k_5_to_14", 5, 14),
    ("k_15_to_31", 15, 31),
]

CORPUS_PER_BUCKET_QUICK: int = 50
"""DAGs per bucket in quick mode."""

CORPUS_PER_BUCKET_FULL: int = 300
"""DAGs per bucket in full mode."""

# Published overhead reference for the projected-overhead calculation.
# Source: production Bingo runs on Picasso (CLAUDE.md, §Production Results).
_BINGO_OVERHEAD_PUB: float = 0.392
"""Published Bingo canonicalisation overhead fraction (39.2 %)."""

_BINGO_EVAL_FRAC: float = 1.0 - _BINGO_OVERHEAD_PUB  # 0.608

# ---------------------------------------------------------------------------
# Reachability filter
# ---------------------------------------------------------------------------


def _is_fully_reachable(dag: LabeledDAG) -> bool:
    """Return True when every node is reachable from node 0 via out-edges.

    ``fast_canonical_string`` / D2S starts at x_0 (node 0) and follows
    out-edges.  Nodes not reachable from x_0 produce ``RuntimeError`` in both
    engines.  Filter before timing so no exceptions propagate.

    Args:
        dag: Labeled DAG to inspect.

    Returns:
        True if every node index in ``range(dag.node_count)`` is reachable
        from node 0 following out-edges.
    """
    n = dag.node_count
    if n == 0:
        return True
    visited: set[int] = set()
    stack: list[int] = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in dag.out_neighbors(node):
            if nb not in visited:
                stack.append(nb)
    return len(visited) == n


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _build_bucket_corpus(
    *,
    k_min: int,
    k_max: int,
    target: int,
    seed: int,
) -> tuple[list[LabeledDAG], int, int]:
    """Build a corpus of reachable DAGs for a k-bucket.

    Uses ``make_random_sr_dag`` with ``num_vars=1`` (guarantees structural
    reachability from node 0 by construction) and ``num_internal`` sampled
    uniformly across ``[k_min, k_max]``.  A reachability filter is applied
    regardless to catch any edge-cases and to report the discard count
    accurately.

    ``num_vars=1`` is used deliberately: with a single variable all internal
    nodes must trace back to x_0, eliminating the 76-82 % discard rate that
    afflicts random-string-generated DAGs.  This makes corpus building fast
    and deterministic.

    Args:
        k_min: Minimum number of internal nodes (inclusive).
        k_max: Maximum number of internal nodes (inclusive).
        target: Number of reachable DAGs to collect.
        seed: RNG seed for reproducibility.

    Returns:
        ``(corpus, n_generated, n_discarded)`` where *corpus* contains exactly
        *target* DAGs, *n_generated* is the total number of DAGs attempted
        (after empty-graph rejection), and *n_discarded* is the count rejected
        by the reachability filter.
    """
    import random

    rng = random.Random(seed)
    corpus: list[LabeledDAG] = []
    n_generated: int = 0
    n_discarded: int = 0

    k_range = list(range(k_min, k_max + 1))
    attempt = 0

    while len(corpus) < target:
        k = rng.choice(k_range)
        dag_seed = seed ^ (attempt * 2654435761) & 0xFFFFFFFF
        attempt += 1
        try:
            dag = make_random_sr_dag(num_vars=1, num_internal=k, seed=dag_seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("Corpus generation failed for k=%d seed=%d: %s", k, dag_seed, exc)
            continue

        n_generated += 1

        if not _is_fully_reachable(dag):
            n_discarded += 1
            continue

        corpus.append(dag)

    return corpus, n_generated, n_discarded


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _canonicalize_batch(dags: list[LabeledDAG], backend: str) -> None:
    """Canonicalise all DAGs in *dags* using *backend*.  Return value discarded.

    Args:
        dags: Batch of labeled DAGs.
        backend: ``"python"`` or ``"cpp"``.
    """
    for dag in dags:
        fast_canonical_string(dag, mode="wl_only", backend=backend)


def _time_one_rep(
    dags: list[LabeledDAG],
    backend: str,
) -> float:
    """Time one repetition: best of N_PASSES_PER_REP full-batch passes.

    Args:
        dags: Batch of labeled DAGs.
        backend: ``"python"`` or ``"cpp"``.

    Returns:
        Per-DAG cost in **milliseconds** (best-of-N_PASSES_PER_REP).
    """
    best_sec: float = float("inf")
    n = len(dags)
    for _ in range(N_PASSES_PER_REP):
        t0 = time.perf_counter()
        _canonicalize_batch(dags, backend)
        t1 = time.perf_counter()
        best_sec = min(best_sec, t1 - t0)
    return (best_sec / n) * 1_000.0  # convert to ms per DAG


def _mad(values: list[float]) -> float:
    """Median-absolute-deviation of *values*.

    Args:
        values: List of floats (length ≥ 1).

    Returns:
        MAD in the same units as *values*.
    """
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    mid = n // 2
    med: float = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
    deviations = sorted([abs(v - med) for v in values])
    mid2 = len(deviations) // 2
    if len(deviations) % 2:
        return deviations[mid2]
    return (deviations[mid2 - 1] + deviations[mid2]) / 2.0


def _median(values: list[float]) -> float:
    """Median of *values*.

    Args:
        values: Non-empty list of floats.

    Returns:
        Median value.
    """
    n = len(values)
    sorted_v = sorted(values)
    mid = n // 2
    if n % 2:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def _benchmark_bucket(
    dags: list[LabeledDAG],
    *,
    cpp_available: bool,
) -> dict[str, Any]:
    """Benchmark both engines on *dags*.

    Runs N_WARMUP + N_REPS repetitions.  In each repetition the Python engine
    is timed first in odd-numbered repetitions and second in even-numbered
    repetitions, so neither engine systematically benefits from CPU cache
    warm-up effects.

    Warmup repetitions are discarded.  The N_REPS measured per-DAG costs
    (one per repetition) are aggregated via median and MAD.

    Args:
        dags: Corpus of reachable DAGs for this bucket.
        cpp_available: Whether the C++ backend is usable.

    Returns:
        Bucket timing result dict with keys ``python``, ``cpp`` (or None),
        ``speedup``, ``n_dags``.
    """
    py_times: list[float] = []
    cpp_times: list[float] = []
    backends_to_run: list[str] = ["python"]
    if cpp_available:
        backends_to_run.append("cpp")

    total_reps = N_WARMUP + N_REPS

    for rep_idx in range(total_reps):
        is_warmup = rep_idx < N_WARMUP
        # Alternate order to equalise thermal state.
        order = backends_to_run if rep_idx % 2 == 0 else list(reversed(backends_to_run))

        rep_times: dict[str, float] = {}
        for be in order:
            t_ms = _time_one_rep(dags, be)
            rep_times[be] = t_ms

        if not is_warmup:
            py_times.append(rep_times["python"])
            if cpp_available and "cpp" in rep_times:
                cpp_times.append(rep_times["cpp"])

        label = "warmup" if is_warmup else f"rep {rep_idx - N_WARMUP}"
        py_t = rep_times.get("python", float("nan"))
        cpp_t = rep_times.get("cpp", float("nan"))
        log.debug(
            "  [%s] python=%.4f ms/DAG  cpp=%.4f ms/DAG",
            label,
            py_t,
            cpp_t,
        )

    py_median = _median(py_times)
    py_mad = _mad(py_times)

    result: dict[str, Any] = {
        "n_dags": len(dags),
        "python": {
            "median_ms_per_dag": round(py_median, 6),
            "mad_ms_per_dag": round(py_mad, 6),
            "rep_times_ms": [round(t, 6) for t in py_times],
        },
    }

    if cpp_available and cpp_times:
        cpp_median = _median(cpp_times)
        cpp_mad = _mad(cpp_times)
        speedup = py_median / cpp_median if cpp_median > 0.0 else float("inf")
        result["cpp"] = {
            "median_ms_per_dag": round(cpp_median, 6),
            "mad_ms_per_dag": round(cpp_mad, 6),
            "rep_times_ms": [round(t, 6) for t in cpp_times],
        }
        result["speedup"] = round(speedup, 4)
    else:
        result["cpp"] = None
        result["speedup"] = None

    return result


# ---------------------------------------------------------------------------
# Projected overhead
# ---------------------------------------------------------------------------


def _projected_overhead(speedup: float) -> dict[str, Any]:
    """Compute projected Bingo overhead after C++ speedup.

    Given the measured speedup *r* = t_python / t_cpp, the original Bingo
    overhead was ``OH = T_canon / (T_eval + T_canon)``.  With C++ the canon
    cost drops to ``T_canon / r``, giving::

        OH' = (T_canon / r) / (T_eval + T_canon / r)
            = (OH / r) / ((1 - OH) + OH / r)
            = (0.392 / r) / (0.608 + 0.392 / r)

    Args:
        speedup: Python/C++ per-DAG time ratio.

    Returns:
        Dict with the projection value and all input assumptions labelled.
    """
    oh_pub = _BINGO_OVERHEAD_PUB
    ef = _BINGO_EVAL_FRAC
    if speedup <= 0.0 or speedup != speedup:  # guard NaN / zero
        projected = float("nan")
    else:
        projected = (oh_pub / speedup) / (ef + oh_pub / speedup)
    return {
        "projected_overhead_bingo": round(projected, 6),
        "note": "projection only — not a measurement",
        "formula": "OH' = (OH_pub / r) / ((1 - OH_pub) + OH_pub / r)",
        "inputs": {
            "OH_pub": oh_pub,
            "OH_pub_source": (
                "published Bingo overhead (39.2%) from production Picasso runs; "
                "see CLAUDE.md §Production Results"
            ),
            "eval_fraction": ef,
            "speedup_r": round(speedup, 4),
        },
    }


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current git HEAD short hash, or ``"unknown"`` on failure.

    Returns:
        Short git commit hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _cpu_model() -> str:
    """Read CPU model from ``/proc/cpuinfo``.

    Returns:
        CPU model name string, or ``"unknown"`` if ``/proc/cpuinfo`` is absent.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _probe_cpp() -> tuple[bool, str]:
    """Probe whether the C++ canonicaliser is available.

    Calls ``fast_canonical_string`` with ``backend='cpp'`` on a trivial
    one-variable DAG.

    Returns:
        ``(available, detail)`` tuple.
    """
    from isalsr.core.node_types import NodeType

    probe = LabeledDAG(max_nodes=1)
    probe.add_node(NodeType.VAR, var_index=0)
    try:
        fast_canonical_string(probe, mode="wl_only", backend="cpp")
        return True, "fast_canonical_string(backend='cpp') OK on 1-VAR probe DAG"
    except TypeError as exc:
        return False, f"TypeError: {exc} — backend= keyword absent"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _assemble_report(
    *,
    bucket_results: dict[str, dict[str, Any]],
    corpus_meta: dict[str, dict[str, Any]],
    cpp_available: bool,
    quick: bool,
    start_time: float,
    end_time: float,
    build_info_dict: dict[str, str],
    capability_probe: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report from per-bucket timing results.

    Args:
        bucket_results: Dict keyed by bucket name, each a timing result.
        corpus_meta: Dict keyed by bucket name with corpus generation metadata.
        cpp_available: Whether C++ engine was benchmarked.
        quick: Whether quick mode was used.
        start_time: Unix timestamp of run start.
        end_time: Unix timestamp of run end.
        build_info_dict: Output of ``backends.build_info()``.
        capability_probe: Dict describing how C++ availability was determined.

    Returns:
        Complete report dict for JSON serialisation.
    """
    # Overall aggregation: collect per-DAG costs weighted by corpus size.
    all_py: list[float] = []
    all_cpp: list[float] = []
    total_discarded = 0

    for bname, bmeta in corpus_meta.items():
        br = bucket_results[bname]
        n = br["n_dags"]
        py_med = br["python"]["median_ms_per_dag"]
        all_py.extend([py_med] * n)
        if cpp_available and br["cpp"] is not None:
            cpp_med = br["cpp"]["median_ms_per_dag"]
            all_cpp.extend([cpp_med] * n)
        total_discarded += bmeta["n_discarded"]

    if all_py:
        overall_py_med = _median(all_py)
        overall_py_mad = _mad(all_py)
    else:
        overall_py_med = float("nan")
        overall_py_mad = float("nan")

    if all_cpp:
        overall_cpp_med = _median(all_cpp)
        overall_cpp_mad = _mad(all_cpp)
        overall_speedup = overall_py_med / overall_cpp_med if overall_cpp_med > 0 else float("nan")
    else:
        overall_cpp_med = float("nan")
        overall_cpp_mad = float("nan")
        overall_speedup = float("nan")

    # Build per-bucket section with discard metadata merged in.
    buckets_out: dict[str, Any] = {}
    for bname, br in bucket_results.items():
        bmeta = corpus_meta[bname]
        buckets_out[bname] = {
            **br,
            "k_min": bmeta["k_min"],
            "k_max": bmeta["k_max"],
            "n_generated": bmeta["n_generated"],
            "n_discarded": bmeta["n_discarded"],
            "discard_rate": bmeta["discard_rate"],
        }

    # Projected overhead (uses overall speedup).
    proj = (
        _projected_overhead(overall_speedup)
        if not _isnan(overall_speedup)
        else {
            "projected_overhead_bingo": float("nan"),
            "note": "projection unavailable — C++ engine not benchmarked",
        }
    )

    overall: dict[str, Any] = {
        "n_dags_total": sum(bucket_results[b]["n_dags"] for b in bucket_results),
        "n_discarded_total": total_discarded,
        "python": {
            "median_ms_per_dag": round(overall_py_med, 6),
            "mad_ms_per_dag": round(overall_py_mad, 6),
        },
    }
    if cpp_available:
        overall["cpp"] = {
            "median_ms_per_dag": round(overall_cpp_med, 6),
            "mad_ms_per_dag": round(overall_cpp_mad, 6),
        }
        overall["speedup"] = round(overall_speedup, 4)
    else:
        overall["cpp"] = None
        overall["speedup"] = None
    overall.update(proj)

    return {
        "schema_version": "1.0",
        "provenance": {
            "git_commit": _git_commit(),
            "build_info": build_info_dict,
            "engine_active": engine(),
            "python_version": sys.version,
            "cpu_model": _cpu_model(),
            "seed": FIXED_SEED,
            "quick_mode": quick,
            "mode": "quick" if quick else "full",
            "protocol": {
                "n_warmup": N_WARMUP,
                "n_reps": N_REPS,
                "n_passes_per_rep": N_PASSES_PER_REP,
                "aggregation": "median of per-rep best-of-N_PASSES_PER_REP times",
                "dispersion": "median-absolute-deviation (MAD)",
                "alternation": "engines alternate start order each repetition",
            },
            "capability_probe": capability_probe,
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
            "elapsed_seconds": round(end_time - start_time, 2),
        },
        "k_buckets": buckets_out,
        "overall": overall,
    }


def _isnan(x: float) -> bool:
    """Return True if *x* is NaN."""
    return x != x


# ---------------------------------------------------------------------------
# Human-readable table
# ---------------------------------------------------------------------------


def _print_table(report: dict[str, Any]) -> None:
    """Print a compact human-readable table to stdout.

    Args:
        report: Assembled report dict.
    """
    prov = report["provenance"]
    print()
    print("=" * 72)
    print(f"  IsalSR Canonicalisation Benchmark  [{prov['mode'].upper()}]")
    print("=" * 72)
    print(f"  git commit   : {prov['git_commit']}")
    print(f"  cpu          : {prov['cpu_model']}")
    print(f"  engine active: {prov['engine_active']}")
    print(f"  seed         : {prov['seed']}")
    print(f"  elapsed      : {prov['elapsed_seconds']} s")
    print(f"  protocol     : {N_WARMUP}W + {N_REPS}R × best-of-{N_PASSES_PER_REP}")
    print()
    print(
        f"  {'Bucket':<14}  {'n DAGs':>7}  {'Python (ms)':>12}  {'C++ (ms)':>10}"
        f"  {'Speedup':>8}  {'Discarded':>9}"
    )
    print("  " + "-" * 68)

    for bname, bd in report["k_buckets"].items():
        n = bd["n_dags"]
        py_ms = bd["python"]["median_ms_per_dag"]
        py_mad = bd["python"]["mad_ms_per_dag"]
        cpp_info = bd.get("cpp")
        if cpp_info:
            cpp_ms = cpp_info["median_ms_per_dag"]
            cpp_mad = cpp_info["mad_ms_per_dag"]
            speedup = bd.get("speedup", float("nan"))
            cpp_str = f"{cpp_ms:.4f}±{cpp_mad:.4f}"
            spd_str = f"{speedup:.2f}×"
        else:
            cpp_str = "n/a"
            spd_str = "n/a"
        disc = bd["n_discarded"]
        print(
            f"  {bname:<14}  {n:>7}  {py_ms:.4f}±{py_mad:.4f}  "
            f"{cpp_str:>14}  {spd_str:>8}  {disc:>9}"
        )

    print("  " + "-" * 68)
    ov = report["overall"]
    py_ms = ov["python"]["median_ms_per_dag"]
    py_mad = ov["python"]["mad_ms_per_dag"]
    cpp_ov = ov.get("cpp")
    if cpp_ov:
        cpp_ms = cpp_ov["median_ms_per_dag"]
        cpp_mad = cpp_ov["mad_ms_per_dag"]
        spd = ov.get("speedup", float("nan"))
        cpp_str = f"{cpp_ms:.4f}±{cpp_mad:.4f}"
        spd_str = f"{spd:.2f}×"
    else:
        cpp_str = "n/a"
        spd_str = "n/a"
    disc_total = ov.get("n_discarded_total", 0)
    print(
        f"  {'OVERALL':<14}  {ov['n_dags_total']:>7}  {py_ms:.4f}±{py_mad:.4f}  "
        f"{cpp_str:>14}  {spd_str:>8}  {disc_total:>9}"
    )

    proj_oh = ov.get("projected_overhead_bingo")
    if proj_oh is not None and not _isnan(float(proj_oh)):
        print()
        print(f"  Projected Bingo overhead after C++ speedup: {float(proj_oh) * 100:.1f}%")
        inp = ov.get("inputs", {})
        print(
            f"  (OH_pub={_BINGO_OVERHEAD_PUB:.1%}, r={inp.get('speedup_r', '?')}, "
            f"OH'=(OH_pub/r)/((1-OH_pub)+OH_pub/r))"
        )
        print("  NOTE: projection only — not a measurement")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the canonicalisation benchmark and write a JSON report.

    Args:
        argv: Command-line arguments (uses ``sys.argv`` when None).

    Returns:
        Exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark IsalSR canonicalisation cost per k-bucket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default="bench_canonical_report.json",
        help="Output path for the JSON report.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Use {CORPUS_PER_BUCKET_QUICK} DAGs/bucket instead of "
            f"{CORPUS_PER_BUCKET_FULL}. Targets < 3 min."
        ),
    )
    args = parser.parse_args(argv)

    corpus_size = CORPUS_PER_BUCKET_QUICK if args.quick else CORPUS_PER_BUCKET_FULL

    # Probe C++ engine.
    cpp_available, probe_detail = _probe_cpp()
    capability_probe = {
        "method": "fast_canonical_string_backend_probe",
        "cpp_available": cpp_available,
        "detail": probe_detail,
    }
    log.info("C++ probe: %s", probe_detail)

    build_info_dict = build_info()
    log.info(
        "Benchmark start: quick=%s corpus_per_bucket=%d cpp=%s",
        args.quick,
        corpus_size,
        cpp_available,
    )

    start_time = time.time()

    # Build corpora per bucket.
    bucket_corpora: dict[str, list[LabeledDAG]] = {}
    corpus_meta: dict[str, dict[str, Any]] = {}

    for bname, k_min, k_max in K_BUCKETS:
        log.info(
            "Building corpus for bucket %s (k=%d..%d, target=%d)", bname, k_min, k_max, corpus_size
        )
        bucket_seed = FIXED_SEED ^ (k_min * 131071) ^ (k_max * 524287)
        dags, n_generated, n_discarded = _build_bucket_corpus(
            k_min=k_min,
            k_max=k_max,
            target=corpus_size,
            seed=bucket_seed,
        )
        discard_rate = round(n_discarded / n_generated, 4) if n_generated > 0 else 0.0
        log.info(
            "Bucket %s: %d DAGs accepted, %d generated, %d discarded (rate=%.1f%%)",
            bname,
            len(dags),
            n_generated,
            n_discarded,
            100.0 * discard_rate,
        )
        bucket_corpora[bname] = dags
        corpus_meta[bname] = {
            "k_min": k_min,
            "k_max": k_max,
            "n_generated": n_generated,
            "n_discarded": n_discarded,
            "discard_rate": discard_rate,
        }

    # Benchmark each bucket.
    bucket_results: dict[str, dict[str, Any]] = {}

    for bname, _k_min, _k_max in K_BUCKETS:
        dags = bucket_corpora[bname]
        log.info(
            "Benchmarking bucket %s: %d DAGs, %d warmup + %d reps × best-of-%d",
            bname,
            len(dags),
            N_WARMUP,
            N_REPS,
            N_PASSES_PER_REP,
        )
        result = _benchmark_bucket(dags, cpp_available=cpp_available)
        py_ms = result["python"]["median_ms_per_dag"]
        cpp_ms = result["cpp"]["median_ms_per_dag"] if result["cpp"] else float("nan")
        spd = result.get("speedup") or float("nan")
        log.info(
            "Bucket %s: python=%.4f ms/DAG, cpp=%.4f ms/DAG, speedup=%.2f×",
            bname,
            py_ms,
            cpp_ms,
            spd,
        )
        bucket_results[bname] = result

    end_time = time.time()

    report = _assemble_report(
        bucket_results=bucket_results,
        corpus_meta=corpus_meta,
        cpp_available=cpp_available,
        quick=args.quick,
        start_time=start_time,
        end_time=end_time,
        build_info_dict=build_info_dict,
        capability_probe=capability_probe,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written to %s", out_path)

    _print_table(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
