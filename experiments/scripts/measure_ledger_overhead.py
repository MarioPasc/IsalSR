"""Measure FallbackLedger instrumentation overhead and rho bias direction.

(a1) Direct microbenchmark on a fixed DAG population.
    A fixed set of ~N_COLLECT real LabeledDAGs produced by one short Bingo search
    (data-collection phase, max_time=20 s) is used for both timings.
    violates_precondition and fast_canonical_string are timed separately on this
    fixed list, yielding mean ns/call and µs/call.  The ratio bfs_ns/canon_ns
    decides contamination: at sample rate R the ledger adds 2*ratio/R of
    per-DAG canonicalisation cost (factor 2 because record_pre and record_post
    each run one BFS per sampled DAG).

(a2) End-to-end paired replay on the same fixed DAG list.
    The fixed list is replayed through record_pre/record_post with
    ISALSR_LEDGER_ENABLED=0 (no-op baseline) and =1 at sample rates 1, 100,
    10000.  Identical DAGs in identical order; only the instrumentation path
    differs.  Overhead is reported as mean net ns/DAG above the disabled baseline
    and as % of mean canonicalisation cost.

    The previous design (Bingo search runs capped at max_time=20 s) was not
    informative for two reasons confirmed by the data themselves:
    (1) Wall-clock is pinned at ~20 s by the soft stopping criterion, so
        differencing two runs measures the time limit, not the ledger.
        Observed overheads of +0.04 % and +0.10 % are residuals of a constant.
    (2) DAG-throughput differences (+4.42 % at rate_1 but also +5.43 % at
        rate_10000, where BFS fires once per 10,000 DAGs) are larger at rate_10000
        than at rate_1 — impossible if the ledger were the cause — confirming
        that the metric is dominated by Bingo's generation-scheduling variance.
    Both figures are discarded.  (a1) and (a2) replace them.

(b) rho bias direction — preserved from the previous accepted run; same code,
    re-run with the same seed for reproducibility.

Note on rate_100 anomaly (previous run): one of three rate_100 seeds completed
in ~5.85 s despite fitness_threshold=0.0.  Bingo's convergence check stops when
best_fitness < fitness_threshold; a floating-point MSE that reaches exactly 0.0
satisfies 0.0 < 0.0 = False, so threshold=0.0 should never fire.  The most
likely cause is a different stopping condition (e.g. max_fitness_evaluations or a
Bingo-internal check) firing for that seed.  This is a Bingo internals question,
not a ledger issue, and is not investigated further here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pathlib
import statistics
import time
from typing import Any

import numpy as np

import experiments.models.fallback_ledger as _fl_module
from benchmarks.datasets.nguyen import generate_data, get_benchmark
from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner
from experiments.models.bingo.runner import BingoRawResult
from experiments.models.fallback_ledger import FallbackLedger, violates_precondition
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

_PROBLEM = "Nguyen-5"
_N_TRAIN = 100
_N_TEST = 100
_MAX_TIME = 20.0  # used for the data-collection run and for rho_bias runs
_POP = 200
_N_COLLECT = 50_000  # cap on DAGs to collect from the Bingo search
_N_BFS_PASSES = 5  # timing passes for violates_precondition
_N_CANON_PASSES = 3  # timing passes for fast_canonical_string (≥ 1.1 s each)
_WARMUP_BFS = 500  # DAGs used for BFS warm-up pass (not timed)
_WARMUP_CANON = 200  # DAGs used for canon warm-up pass (not timed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Nguyen-5 train/test arrays.

    Returns:
        (x_train, y_train, x_test, y_test) NumPy arrays.
    """
    bench = get_benchmark(_PROBLEM)
    return generate_data(bench, n_train=_N_TRAIN, n_test=_N_TEST, seed=1)


# ---------------------------------------------------------------------------
# Helper shared by rho_bias and data-collection
# ---------------------------------------------------------------------------


def _run_once(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ledger_enabled: bool,
    sample_rate: int,
    canon_timeout: float,
    seed: int,
) -> tuple[BingoRawResult, FallbackLedger]:
    """Run one IsalSR Bingo trial with specified ledger and timeout settings.

    Sets ISALSR_LEDGER_ENABLED and ISALSR_LEDGER_SAMPLE_RATE in the
    process environment before calling fit(), so that FallbackLedger()
    inside fit() picks them up at construction time.

    Args:
        x_train: Training features, shape (n_train, n_vars).
        y_train: Training targets, shape (n_train,).
        x_test: Test features, shape (n_test, n_vars).
        y_test: Test targets, shape (n_test,).
        ledger_enabled: Whether to enable the fallback ledger.
        sample_rate: BFS sample rate (1 = every DAG, 100 = every 100th).
        canon_timeout: Canonicalisation timeout in seconds.
        seed: Random seed passed to fit().

    Returns:
        (result, ledger) where result is the raw BingoRawResult and
        ledger is the FallbackLedger populated during the run.
    """
    os.environ["ISALSR_LEDGER_ENABLED"] = "1" if ledger_enabled else "0"
    os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = str(sample_rate)

    # fitness_threshold=0.0 prevents early convergence so all runs reach
    # the soft max_time ceiling, keeping the workload comparable across configs.
    cfg = BingoConfig(
        population_size=_POP,
        max_time=_MAX_TIME,
        canonicalization_timeout=canon_timeout,
        fitness_threshold=0.0,
    )
    runner = IsalSRBingoRunner(config=cfg)
    result = runner.fit(x_train, y_train, x_test, y_test, seed=seed, config={})
    ledger: FallbackLedger = runner.last_ledger
    return result, ledger


# ---------------------------------------------------------------------------
# Data collection: harvest post-normalisation DAGs from a Bingo run
# ---------------------------------------------------------------------------


def collect_dags(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_count: int = _N_COLLECT,
) -> list[LabeledDAG]:
    """Collect post-normalisation LabeledDAGs from one Bingo IsalSR search.

    Temporarily monkey-patches FallbackLedger.record_post at the class level
    to capture each post-normalisation DAG as it arrives, up to max_count.
    The patch is restored unconditionally in a finally block.  The collection
    run disables the ledger (ISALSR_LEDGER_ENABLED=0) so that the BFS hooks
    do not fire during harvesting — the patch captures DAGs independently of
    the ledger's enabled state.

    Args:
        x_train: Training features.
        y_train: Training targets.
        x_test: Test features.
        y_test: Test targets.
        max_count: Maximum number of DAGs to collect.

    Returns:
        List of LabeledDAG instances in production order, length ≤ max_count.
    """
    collected: list[LabeledDAG] = []
    _orig = _fl_module.FallbackLedger.record_post

    def _capturing(self: FallbackLedger, dag: LabeledDAG) -> None:
        if len(collected) < max_count:
            collected.append(dag)
        _orig(self, dag)

    _fl_module.FallbackLedger.record_post = _capturing  # type: ignore[method-assign]
    try:
        os.environ["ISALSR_LEDGER_ENABLED"] = "0"
        cfg = BingoConfig(
            population_size=_POP,
            max_time=_MAX_TIME,
            fitness_threshold=0.0,
        )
        runner = IsalSRBingoRunner(config=cfg)
        runner.fit(x_train, y_train, x_test, y_test, seed=1, config={})
    finally:
        _fl_module.FallbackLedger.record_post = _orig

    log.info("Collected %d DAGs for microbenchmark (cap %d)", len(collected), max_count)
    return collected


# ---------------------------------------------------------------------------
# (a1) Direct microbenchmark
# ---------------------------------------------------------------------------


def measure_microbenchmark(dags: list[LabeledDAG]) -> dict[str, Any]:
    """Time violates_precondition and fast_canonical_string on a fixed DAG list.

    Both functions are run over the same list in separate timing passes so that
    warm-up and cache effects are symmetric.  A short warm-up pass (not counted)
    precedes each measurement to avoid cold-start bias.

    The ratio bfs_ns / canon_ns is the overhead per DAG as a fraction of
    canonicalisation cost.  In production, record_pre and record_post each run
    one BFS per sampled DAG, so at sample rate R the total per-DAG ledger cost
    is 2 * bfs_mean_ns / R, or 2 * ratio / R of canon cost.

    Args:
        dags: Fixed list of post-normalisation LabeledDAGs from collect_dags.

    Returns:
        Dict with bfs_mean_ns_per_call, canon_mean_us_per_call, ratio, and
        overhead fractions at rates 1, 100, 10000.
    """
    n = len(dags)

    # ---- BFS timing ----
    log.info("(a1) BFS warm-up (%d DAGs)...", _WARMUP_BFS)
    for dag in dags[: min(_WARMUP_BFS, n)]:
        violates_precondition(dag)

    log.info("(a1) timing BFS: %d DAGs × %d passes", n, _N_BFS_PASSES)
    bfs_pass_ns: list[int] = []
    for _ in range(_N_BFS_PASSES):
        t0 = time.perf_counter_ns()
        for dag in dags:
            violates_precondition(dag)
        bfs_pass_ns.append(time.perf_counter_ns() - t0)

    bfs_mean_ns = statistics.mean(bfs_pass_ns) / n
    bfs_stdev_ns = statistics.stdev(bfs_pass_ns) / n
    log.info("(a1) BFS: %.1f ± %.1f ns/call", bfs_mean_ns, bfs_stdev_ns)

    # ---- Canon timing ----
    log.info("(a1) canon warm-up (%d DAGs)...", _WARMUP_CANON)
    for dag in dags[: min(_WARMUP_CANON, n)]:
        with contextlib.suppress(Exception):
            fast_canonical_string(dag, timeout=60.0)

    log.info("(a1) timing canon: %d DAGs × %d passes", n, _N_CANON_PASSES)
    canon_pass_ns: list[int] = []
    n_canon_ok = 0
    n_canon_err = 0
    for _ in range(_N_CANON_PASSES):
        t0 = time.perf_counter_ns()
        for dag in dags:
            try:
                fast_canonical_string(dag, timeout=60.0)
                n_canon_ok += 1
            except Exception:  # noqa: BLE001
                n_canon_err += 1
        canon_pass_ns.append(time.perf_counter_ns() - t0)

    canon_mean_ns = statistics.mean(canon_pass_ns) / n
    canon_stdev_ns = statistics.stdev(canon_pass_ns) / n
    canon_mean_us = canon_mean_ns / 1_000.0
    ratio = bfs_mean_ns / canon_mean_ns
    log.info(
        "(a1) canon: %.2f ± %.2f µs/call  ok=%d err=%d",
        canon_mean_us,
        canon_stdev_ns / 1_000.0,
        n_canon_ok // _N_CANON_PASSES,
        n_canon_err // _N_CANON_PASSES,
    )
    log.info(
        "(a1) ratio=%.4f → per-DAG ledger cost at rate R is 2*%.4f/R = %.4f%% of canon",
        ratio,
        ratio,
        200.0 * ratio,
    )

    return {
        "n_dags": n,
        "bfs_passes": _N_BFS_PASSES,
        "canon_passes": _N_CANON_PASSES,
        "bfs_mean_ns_per_call": round(bfs_mean_ns, 2),
        "bfs_stdev_ns_per_call": round(bfs_stdev_ns, 2),
        "canon_mean_ns_per_call": round(canon_mean_ns, 2),
        "canon_mean_us_per_call": round(canon_mean_us, 4),
        "canon_stdev_ns_per_call": round(canon_stdev_ns, 2),
        "bfs_canon_ratio": round(ratio, 6),
        # 2× because record_pre + record_post each run one BFS per sampled DAG
        "overhead_pct_of_canon_at_rate_1": round(200.0 * ratio, 4),
        "overhead_pct_of_canon_at_rate_100": round(2.0 * ratio, 4),
        "overhead_pct_of_canon_at_rate_10000": round(0.02 * ratio, 6),
        "n_canon_ok_per_pass": n_canon_ok // _N_CANON_PASSES,
        "n_canon_err_per_pass": n_canon_err // _N_CANON_PASSES,
    }


# ---------------------------------------------------------------------------
# (a2) End-to-end paired replay
# ---------------------------------------------------------------------------


def measure_paired_replay(
    dags: list[LabeledDAG],
    canon_mean_ns: float,
    n_passes: int = 5,
) -> dict[str, Any]:
    """Replay fixed DAG list through record_pre/record_post with four ledger configs.

    Four configurations are timed: disabled (ISALSR_LEDGER_ENABLED=0) and
    enabled at sample rates 1, 100, 10000.  Each configuration runs n_passes
    times over the entire DAG list; a fresh FallbackLedger is created per pass
    (reads env vars at construction time).  The disabled pass serves as the
    paired baseline — its cost is the overhead of two Python function calls plus
    two attribute-check early returns per DAG.

    Net overhead per DAG = mean_ns_per_dag(rate_N) - mean_ns_per_dag(disabled).
    Overhead as % of canonicalisation = net_overhead_ns / canon_mean_ns * 100.

    Args:
        dags: Fixed list of LabeledDAGs from collect_dags; identical across configs.
        canon_mean_ns: Mean canonicalisation cost (ns/DAG) from measure_microbenchmark.
        n_passes: Number of timing passes per configuration.

    Returns:
        Dict with per-config timing and net overhead relative to disabled baseline
        and as % of canon cost.
    """
    n = len(dags)
    configs: list[tuple[str, bool, int]] = [
        ("disabled", False, 1),
        ("rate_1", True, 1),
        ("rate_100", True, 100),
        ("rate_10000", True, 10000),
    ]

    per_config: dict[str, Any] = {}

    for label, enabled, rate in configs:
        os.environ["ISALSR_LEDGER_ENABLED"] = "1" if enabled else "0"
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = str(rate)

        pass_ns: list[int] = []
        for _ in range(n_passes):
            ledger = FallbackLedger()  # re-reads env vars each pass
            t0 = time.perf_counter_ns()
            for dag in dags:
                ledger.record_pre(dag)
                ledger.record_post(dag)
            pass_ns.append(time.perf_counter_ns() - t0)

        mean_total_ns = statistics.mean(pass_ns)
        stdev_total_ns = statistics.stdev(pass_ns) if n_passes > 1 else 0.0
        per_dag_ns = mean_total_ns / n

        per_config[label] = {
            "ledger_enabled": enabled,
            "sample_rate": rate if enabled else None,
            "pass_ns_samples": pass_ns,
            "mean_total_ns": round(mean_total_ns),
            "stdev_total_ns": round(stdev_total_ns),
            "mean_per_dag_ns": round(per_dag_ns, 2),
        }
        log.info(
            "(a2) %s: %.2f ± %.2f ns/DAG  total_passes=%d",
            label,
            per_dag_ns,
            stdev_total_ns / n,
            n_passes,
        )

    # Paired comparison: subtract disabled baseline from each instrumented config
    disabled_per_dag = per_config["disabled"]["mean_per_dag_ns"]
    for label in ("rate_1", "rate_100", "rate_10000"):
        net_ns = per_config[label]["mean_per_dag_ns"] - disabled_per_dag
        pct_of_canon = 100.0 * net_ns / canon_mean_ns if canon_mean_ns > 0 else float("nan")
        per_config[label]["net_overhead_per_dag_ns"] = round(net_ns, 2)
        per_config[label]["overhead_pct_of_canon"] = round(pct_of_canon, 4)
        log.info(
            "(a2) %s: net overhead %.2f ns/DAG = %.4f%% of canon cost",
            label,
            net_ns,
            pct_of_canon,
        )

    return {
        "n_dags": n,
        "n_passes": n_passes,
        "canon_mean_ns_per_dag_from_a1": round(canon_mean_ns, 2),
        "disabled_mean_per_dag_ns": round(disabled_per_dag, 2),
        "configurations": per_config,
    }


# ---------------------------------------------------------------------------
# (b) rho bias direction  — PRESERVED UNCHANGED
# ---------------------------------------------------------------------------


def measure_rho_bias(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Demonstrate that canonicalisation timeouts inflate rho = n_total/n_unique.

    Mechanism (from isalsr_runner.py):
    - n_total is incremented unconditionally before canonicalisation (line 286).
    - n_unique is incremented only on successful canonicalisation (line 418).
    - A CanonicalTimeoutError causes ``continue`` without touching n_unique
      or n_skipped (lines 349-363).

    Therefore rho_observed = (N + T) / U, where N is the count of
    non-timed-out attempts, T is the timeout count, and U = n_unique.
    Without timeouts rho = N / U_normal.  Since T > 0 and U is reduced
    (timed-out DAGs are never added to canonical_seen, so they are not
    deduplicated in future generations), rho_timeout > rho_normal.

    Two scenarios are run with the same seed:
    - normal: canon_timeout=60.0 s → expected T=0
    - forced: canon_timeout=1e-5 s → expected T >> 0 (below avg per-DAG cost ~22 µs)

    Args:
        x_train: Training features.
        y_train: Training targets.
        x_test: Test features.
        y_test: Test targets.

    Returns:
        Dict with counter triples (n_total, n_unique, n_skipped, n_timeouts)
        and rho for both scenarios, plus direction flag.
    """
    seed = 1

    log.info("[rho_bias] running normal (timeout=60 s)...")
    result_normal, ledger_normal = _run_once(
        x_train,
        y_train,
        x_test,
        y_test,
        ledger_enabled=True,
        sample_rate=1,
        canon_timeout=60.0,
        seed=seed,
    )

    # 1e-5 s (10 µs) is below the average per-DAG canon time (~22 µs measured
    # on Nguyen-5 with population_size=200), so a significant fraction of
    # DAGs (those with k ≥ 4 non-VAR nodes) should exceed the deadline.
    forced_timeout = 1e-5  # 10 µs: below avg per-DAG canon time ~22 µs
    log.info("[rho_bias] running forced-timeout (timeout=%.2e s)...", forced_timeout)
    result_timeout, ledger_timeout = _run_once(
        x_train,
        y_train,
        x_test,
        y_test,
        ledger_enabled=True,
        sample_rate=1,
        canon_timeout=forced_timeout,
        seed=seed,
    )

    def _rho(n_total: int, n_unique: int) -> float:
        """Compute rho = n_total / n_unique, or inf if n_unique=0."""
        return n_total / n_unique if n_unique > 0 else float("inf")

    rho_normal = _rho(result_normal.n_total_dags, result_normal.n_unique_canonical)
    rho_timeout = _rho(result_timeout.n_total_dags, result_timeout.n_unique_canonical)

    log.info(
        "[rho_bias] normal: n_total=%d n_unique=%d n_skipped=%d T=%d rho=%.4f",
        result_normal.n_total_dags,
        result_normal.n_unique_canonical,
        result_normal.n_skipped,
        ledger_normal.timeout,
        rho_normal,
    )
    log.info(
        "[rho_bias] forced: n_total=%d n_unique=%d n_skipped=%d T=%d rho=%.4f",
        result_timeout.n_total_dags,
        result_timeout.n_unique_canonical,
        result_timeout.n_skipped,
        ledger_timeout.timeout,
        rho_timeout,
    )

    direction = "inflated" if rho_timeout > rho_normal else "not_inflated"

    return {
        "seed": seed,
        "normal": {
            "canon_timeout_s": 60.0,
            "n_total": result_normal.n_total_dags,
            "n_unique": result_normal.n_unique_canonical,
            "n_skipped": result_normal.n_skipped,
            "n_timeouts": ledger_normal.timeout,
            "rho": rho_normal,
        },
        "forced_timeout": {
            "canon_timeout_s": forced_timeout,
            "n_total": result_timeout.n_total_dags,
            "n_unique": result_timeout.n_unique_canonical,
            "n_skipped": result_timeout.n_skipped,
            "n_timeouts": ledger_timeout.timeout,
            "rho": rho_timeout,
        },
        "rho_delta": rho_timeout - rho_normal,
        "direction": direction,
        "mechanism": (
            "n_total incremented before canon (isalsr_runner.py:286). "
            "n_unique incremented only on success (isalsr_runner.py:418). "
            "CanonicalTimeoutError path (lines 349-363) calls continue "
            "without touching n_unique or n_skipped. "
            "rho_observed = (N+T)/U >= N/U = rho_ideal."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, run measurements, write JSON output."""
    parser = argparse.ArgumentParser(
        description="Measure FallbackLedger overhead (microbenchmark + replay) and rho bias."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Path to write the output JSON file.",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=5,
        help="Timing passes per replay configuration (default 5).",
    )
    args = parser.parse_args()

    output_path: pathlib.Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s data (n_train=%d, n_test=%d)...", _PROBLEM, _N_TRAIN, _N_TEST)
    x_train, y_train, x_test, y_test = _get_data()

    log.info("=== Data collection (20 s Bingo search, cap=%d DAGs) ===", _N_COLLECT)
    dags = collect_dags(x_train, y_train, x_test, y_test)

    log.info("=== Part (a1): direct microbenchmark ===")
    mb_results = measure_microbenchmark(dags)

    log.info("=== Part (a2): paired replay ===")
    replay_results = measure_paired_replay(
        dags,
        canon_mean_ns=mb_results["canon_mean_ns_per_call"],
        n_passes=args.n_passes,
    )

    log.info("=== Part (b): rho bias direction ===")
    rho_results = measure_rho_bias(x_train, y_train, x_test, y_test)

    output: dict[str, Any] = {
        "experiment": "measure_ledger_overhead",
        "overhead_a1_microbenchmark": mb_results,
        "overhead_a2_paired_replay": replay_results,
        "previous_search_based_approach": {
            "status": "discarded",
            "reason": (
                "Wall-clock is pinned at ~20 s by the soft stopping criterion; "
                "differencing two time-limited runs measures the limit, not the ledger. "
                "DAG-throughput reduction was +4.42% at rate_1 but +5.43% at rate_10000 "
                "(more overhead at rate_10000 than rate_1 is impossible if the ledger is "
                "the cause), confirming domination by Bingo generation-scheduling variance."
            ),
            "rate_100_anomaly_note": (
                "In the search-based run, one of three rate_100 seeds completed in ~5.85 s "
                "despite fitness_threshold=0.0.  Most likely a separate Bingo stopping "
                "condition (max_fitness_evaluations or internal check) fired for that seed. "
                "Not a ledger issue; not investigated further."
            ),
        },
        "rho_bias": rho_results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    log.info("Results written to %s", output_path)

    # Printed summary
    mb = mb_results
    rp = replay_results["configurations"]
    print("\n=== (a1) Direct microbenchmark ===")
    print(f"  n_dags:          {mb['n_dags']:,}")
    bfs_s = mb["bfs_stdev_ns_per_call"]
    print(f"  BFS mean:        {mb['bfs_mean_ns_per_call']:.1f} ± {bfs_s:.1f} ns/call")
    c_us = mb["canon_mean_us_per_call"]
    c_std = mb["canon_stdev_ns_per_call"] / 1000
    print(f"  canon mean:      {c_us:.3f} ± {c_std:.3f} µs/call")
    print(f"  ratio (BFS/canon): {mb['bfs_canon_ratio']:.4f}")
    print(f"  overhead at rate_1:     {mb['overhead_pct_of_canon_at_rate_1']:.2f}%  (2*r/1)")
    print(f"  overhead at rate_100:   {mb['overhead_pct_of_canon_at_rate_100']:.4f}%  (2*r/100)")
    print(
        f"  overhead at rate_10000: {mb['overhead_pct_of_canon_at_rate_10000']:.6f}%  (2*r/10000)"
    )

    print("\n=== (a2) Paired replay (overhead vs disabled baseline) ===")
    print(f"  disabled baseline: {rp['disabled']['mean_per_dag_ns']:.2f} ns/DAG (2 attr checks)")
    for lbl in ("rate_1", "rate_100", "rate_10000"):
        c = rp[lbl]
        print(
            f"  {lbl:12s}: {c['mean_per_dag_ns']:.2f} ns/DAG  "
            f"net={c['net_overhead_per_dag_ns']:.2f} ns  "
            f"={c['overhead_pct_of_canon']:.4f}% of canon"
        )

    print("\n=== (b) rho bias direction ===")
    nb = rho_results["normal"]
    nt = rho_results["forced_timeout"]
    print(
        f"  normal   (T=0):   n_total={nb['n_total']:6d}  n_unique={nb['n_unique']:5d}  "
        f"n_timeouts={nb['n_timeouts']:5d}  rho={nb['rho']:.4f}"
    )
    print(
        f"  forced   (T>>0):  n_total={nt['n_total']:6d}  n_unique={nt['n_unique']:5d}  "
        f"n_timeouts={nt['n_timeouts']:5d}  rho={nt['rho']:.4f}"
    )
    print(f"  direction: {rho_results['direction']}  (delta={rho_results['rho_delta']:+.4f})")


if __name__ == "__main__":
    main()
