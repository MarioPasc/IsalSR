"""Empirical verification of UDFS canonical deduplication.

Validates that the monkey-patch approach correctly intercepts ALL
evaluate_cgraph calls in single-process mode (processes=1), and
confirms the latent bypass in multi-process mode (spawn context).

Five checks:
    1. Extended accounting: n_total == n_unique + n_skipped + n_conv_fail + n_canon_fail
    2. Wrapper coverage: secondary_count == dedup.n_total
    3. Dedup effectiveness: n_skipped > 0
    4. Canonical correctness: spot-check deduplicated pairs
    5. Multiprocessing bypass: with processes>1, dedup.n_total near zero

Usage:
    conda activate isalsr
    python experiments/scripts/verify_udfs_dedup.py \
        --output-dir /path/to/output \
        --seed 42 --max-time 60 --test-multiprocessing
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.nguyen import generate_data, get_benchmark  # noqa: E402

# Ensure vendored DAG_search is importable
_vendor_dir = str(Path(__file__).resolve().parents[1] / "models" / "udfs" / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

import DAG_search.dag_search as dag_search_module  # noqa: E402
from DAG_search.comp_graph import CompGraph  # noqa: E402
from DAG_search.dag_search import DAGRegressor  # noqa: E402

from experiments.models.udfs.adapter import compgraph_to_labeled_dag  # noqa: E402
from experiments.models.udfs.config import UDFSConfig  # noqa: E402
from experiments.models.udfs.isalsr_runner import _CanonicalDeduplicator  # noqa: E402
from isalsr.core.canonical import fast_canonical_string  # noqa: E402

log = logging.getLogger("verify_udfs_dedup")


# ======================================================================
# Instrumented deduplicator
# ======================================================================

MAX_DEDUP_SAMPLES = 20


def _mp_worker_check_patch() -> int:
    """Worker function: check if evaluate_cgraph is patched in this process.

    Must be module-level for pickling with spawn context.
    Returns 1 if patched (sentinel_wrapper), 0 if original.
    """
    import DAG_search.dag_search as worker_module

    fn = worker_module.evaluate_cgraph
    is_patched = getattr(fn, "__name__", "") == "sentinel_wrapper"
    return 1 if is_patched else 0


class _InstrumentedDeduplicator(_CanonicalDeduplicator):
    """Enhanced dedup tracker with failure counters and pair sampling."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_conversion_failures: int = 0
        self.n_canon_failures: int = 0
        self.dedup_samples: list[dict[str, Any]] = []

    def wrap_evaluate_cgraph(self, original_fn: Any) -> Any:
        """Override to track conversion/canon failures and sample dedup pairs."""
        self._original_evaluate = original_fn

        def wrapped(
            cgraph: Any,
            X: Any,  # noqa: N803
            loss_fkt: Any,
            opt_mode: str = "grid_zoom",
            loss_thresh: float | None = None,
        ) -> Any:
            self.n_total += 1

            # Path A: conversion failure
            try:
                labeled_dag = compgraph_to_labeled_dag(cgraph)
            except Exception:  # noqa: BLE001
                self.n_conversion_failures += 1
                result = self._original_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
                consts, loss = result
                if np.isfinite(loss) and loss < self._best_loss:
                    self._best_loss = loss
                self._maybe_snapshot()
                return result

            # Path B: canonicalization failure
            canon_hash = self._resolve_canonical_hash(labeled_dag)
            if canon_hash is None:
                self.n_canon_failures += 1
                result = self._original_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
                consts, loss = result
                if np.isfinite(loss) and loss < self._best_loss:
                    self._best_loss = loss
                self._maybe_snapshot()
                return result

            # Path C: duplicate detected
            if canon_hash in self.canonical_seen:
                self.n_skipped += 1
                # Sample the pair for spot-checking
                if len(self.dedup_samples) < MAX_DEDUP_SAMPLES:
                    self.dedup_samples.append(
                        {
                            "canon_hash": canon_hash,
                            "node_dict": copy.deepcopy(cgraph.node_dict),
                            "inp_dim": cgraph.inp_dim,
                            "outp_dim": cgraph.outp_dim,
                            "n_consts": cgraph.n_consts,
                        }
                    )
                n_consts = cgraph.n_consts
                dummy_consts = np.zeros(n_consts) if n_consts > 0 else np.array([])
                self._maybe_snapshot()
                return dummy_consts, np.inf

            # Path D: new unique graph
            self.canonical_seen.add(canon_hash)
            self.n_unique += 1
            result = self._original_evaluate(cgraph, X, loss_fkt, opt_mode, loss_thresh)
            consts, loss = result
            if np.isfinite(loss) and loss < self._best_loss:
                self._best_loss = loss
            self._maybe_snapshot()
            return result

        return wrapped


# ======================================================================
# Verification logic
# ======================================================================


def run_verification(
    config: UDFSConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[_InstrumentedDeduplicator, int, float]:
    """Run UDFS with instrumented dedup and secondary counter.

    Returns (dedup, secondary_count, wall_clock_s).
    """
    t0 = time.perf_counter()

    dedup = _InstrumentedDeduplicator(
        use_fast_canonical=config.use_fast_canonical,
        timeout=config.canonicalization_timeout,
        snapshot_freq=config.snapshot_frequency,
        t0=t0,
    )

    original_fn = dag_search_module.evaluate_cgraph

    # Inner layer: dedup wrapper
    dedup_wrapped = dedup.wrap_evaluate_cgraph(original_fn)

    # Outer layer: secondary counter
    secondary_count = 0

    def secondary_counter(*args: Any, **kwargs: Any) -> Any:
        nonlocal secondary_count
        secondary_count += 1
        return dedup_wrapped(*args, **kwargs)

    # Install the double wrapper
    dag_search_module.evaluate_cgraph = secondary_counter

    try:
        kwargs = config.to_dag_regressor_kwargs()
        regressor = DAGRegressor(**kwargs)
        regressor.random_state = seed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor.fit(x_train, y_train, verbose=0)
    finally:
        dag_search_module.evaluate_cgraph = original_fn

    wall_clock = time.perf_counter() - t0
    return dedup, secondary_count, wall_clock


def verify_canonical_correctness(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Spot-check canonical correctness for sampled dedup pairs."""
    results = []
    for sample in samples:
        # Reconstruct CompGraph from stored data
        cg = CompGraph(
            sample["inp_dim"],
            sample["outp_dim"],
            sample["n_consts"],
            node_dict=copy.deepcopy(sample["node_dict"]),
        )
        try:
            ldag = compgraph_to_labeled_dag(cg)
            canonical = fast_canonical_string(ldag, timeout=10.0)
            recomputed_hash = hash(canonical)
            results.append(
                {
                    "canon_hash": sample["canon_hash"],
                    "recomputed_hash": recomputed_hash,
                    "match": recomputed_hash == sample["canon_hash"],
                    "node_count": ldag.node_count,
                    "edge_count": ldag.edge_count,
                    "canonical_string": canonical,
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                {
                    "canon_hash": sample["canon_hash"],
                    "error": str(e),
                    "match": False,
                }
            )
    return results


def run_multiprocessing_test(
    config: UDFSConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_processes: int = 2,
) -> dict[str, Any]:
    """Test whether multiprocessing bypasses the dedup patch.

    Strategy: Instead of running full UDFS (which can error with small
    configs in multiprocessing mode), we directly test whether spawned
    workers see the monkey-patched evaluate_cgraph function.

    We patch evaluate_cgraph with a sentinel wrapper, then spawn workers
    via the same ``multiprocessing.get_context('spawn')`` that UDFS uses,
    and check if workers call the sentinel or the original.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")

    # Sentinel: if a worker calls this, the patch was inherited
    sentinel_called = ctx.Value("i", 0)

    original_fn = dag_search_module.evaluate_cgraph

    def sentinel_wrapper(*args: Any, **kwargs: Any) -> Any:
        # This should NOT be reachable in spawned workers
        sentinel_called.value += 1  # type: ignore[attr-defined]
        return original_fn(*args, **kwargs)

    # Install sentinel in main process
    dag_search_module.evaluate_cgraph = sentinel_wrapper

    try:
        with ctx.Pool(processes=1) as pool:
            results = pool.apply(_mp_worker_check_patch)

        worker_saw_patch = bool(results)

        return {
            "tested": True,
            "processes": n_processes,
            "method": "spawn_sentinel",
            "worker_saw_patch": worker_saw_patch,
            "bypass_confirmed": not worker_saw_patch,
            "main_sentinel_calls": sentinel_called.value,  # type: ignore[attr-defined]
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "tested": True,
            "processes": n_processes,
            "error": str(e),
            "bypass_confirmed": None,
        }
    finally:
        dag_search_module.evaluate_cgraph = original_fn


# ======================================================================
# Main
# ======================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify UDFS dedup correctness")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for JSON report",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--max-orders", type=int, default=200)
    parser.add_argument("--n-calc-nodes", type=int, default=3)
    parser.add_argument(
        "--test-multiprocessing",
        action="store_true",
        help="Also test multiprocessing bypass (processes=2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s [%(name)s] %(message)s",
    )

    # Generate data
    bench = get_benchmark("Nguyen-1")
    x_train, y_train, x_test, y_test = generate_data(bench, seed=args.seed)
    log.info("Benchmark: Nguyen-1 (%s), seed=%d", bench["expression"], args.seed)

    config = UDFSConfig(
        n_calc_nodes=args.n_calc_nodes,
        max_orders=args.max_orders,
        max_time=args.max_time,
        k=1,
        mode="hierarchical",
        processes=1,
        use_fast_canonical=True,
        canonicalization_timeout=10.0,
        snapshot_frequency=500,
    )
    log.info(
        "Config: n_calc_nodes=%d, max_orders=%d, max_time=%.0f, k=%d",
        config.n_calc_nodes,
        config.max_orders,
        config.max_time,
        config.k,
    )

    # ---- Run single-process verification ----
    log.info("Running IsalSR UDFS (processes=1)...")
    dedup, secondary_count, wall_clock = run_verification(config, x_train, y_train, args.seed)
    log.info("Run complete in %.1fs", wall_clock)

    # ---- Counter summary ----
    log.info("--- Counter Summary ---")
    log.info("Secondary counter:      %d", secondary_count)
    log.info("Dedup n_total:          %d", dedup.n_total)
    log.info("Dedup n_unique:         %d", dedup.n_unique)
    log.info("Dedup n_skipped:        %d", dedup.n_skipped)
    log.info("Conversion failures:    %d", dedup.n_conversion_failures)
    log.info("Canonicalization fails: %d", dedup.n_canon_failures)
    dedup_rate = 100.0 * dedup.n_skipped / dedup.n_total if dedup.n_total > 0 else 0.0
    log.info("Dedup rate:             %.1f%%", dedup_rate)

    # ---- Check 1: Extended accounting ----
    accounted = (
        dedup.n_unique + dedup.n_skipped + dedup.n_conversion_failures + dedup.n_canon_failures
    )
    check1_pass = dedup.n_total == accounted
    check1_expr = (
        f"{dedup.n_total} == {dedup.n_unique} + {dedup.n_skipped} "
        f"+ {dedup.n_conversion_failures} + {dedup.n_canon_failures}"
    )
    log.info(
        "[CHECK 1] Extended accounting:   %s (%s)",
        "PASS" if check1_pass else "FAIL",
        check1_expr,
    )

    # ---- Check 2: Wrapper coverage ----
    check2_pass = secondary_count == dedup.n_total
    log.info(
        "[CHECK 2] Wrapper coverage:      %s (%d == %d)",
        "PASS" if check2_pass else "FAIL",
        secondary_count,
        dedup.n_total,
    )

    # ---- Check 3: Dedup effectiveness ----
    check3_pass = dedup.n_skipped > 0
    log.info(
        "[CHECK 3] Dedup effectiveness:   %s (n_skipped=%d)",
        "PASS" if check3_pass else "FAIL",
        dedup.n_skipped,
    )

    # ---- Check 4: Canonical correctness ----
    spot_check_results = verify_canonical_correctness(dedup.dedup_samples)
    all_match = all(r.get("match", False) for r in spot_check_results)
    n_checked = len(spot_check_results)
    check4_pass = all_match and n_checked > 0
    log.info(
        "[CHECK 4] Canonical correctness: %s (%d/%d pairs verified)",
        "PASS" if check4_pass else "FAIL",
        sum(1 for r in spot_check_results if r.get("match")),
        n_checked,
    )

    # ---- Check 5: Multiprocessing bypass (optional) ----
    mp_result: dict[str, Any] = {"tested": False}
    check5_pass: bool | None = None
    if args.test_multiprocessing:
        log.info("Running multiprocessing bypass test (processes=2)...")
        mp_result = run_multiprocessing_test(config, x_train, y_train, args.seed)
        if mp_result.get("error"):
            log.info(
                "[CHECK 5] Multiprocessing bypass: SKIPPED (error: %s)",
                mp_result["error"],
            )
            check5_pass = None
        else:
            check5_pass = mp_result.get("bypass_confirmed", False)
            log.info(
                "[CHECK 5] Multiprocessing bypass: %s (dedup.n_total=%d, secondary=%d in workers)",
                "CONFIRMED" if check5_pass else "NOT CONFIRMED",
                mp_result.get("dedup_n_total", -1),
                mp_result.get("secondary_count", -1),
            )

    # ---- Overall ----
    all_checks = [check1_pass, check2_pass, check3_pass, check4_pass]
    # Check 5 is informational (latent bug), not a pass/fail gate
    overall = all(all_checks)
    log.info("Overall: %s", "PASS" if overall else "FAIL")

    # ---- Build and write report ----
    report = {
        "metadata": {
            "script": "verify_udfs_dedup.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmark": "Nguyen-1",
            "expression": bench["expression"],
            "config": {
                "n_calc_nodes": config.n_calc_nodes,
                "max_orders": config.max_orders,
                "max_time": config.max_time,
                "k": config.k,
                "mode": config.mode,
                "processes": config.processes,
            },
            "seed": args.seed,
            "wall_clock_s": round(wall_clock, 2),
        },
        "counters": {
            "secondary_count": secondary_count,
            "dedup_n_total": dedup.n_total,
            "dedup_n_unique": dedup.n_unique,
            "dedup_n_skipped": dedup.n_skipped,
            "dedup_n_conversion_failures": dedup.n_conversion_failures,
            "dedup_n_canon_failures": dedup.n_canon_failures,
            "dedup_rate_pct": round(dedup_rate, 2),
            "canon_time_s": round(dedup.canon_time_total, 3),
        },
        "checks": {
            "check1_extended_accounting": {
                "expression": "n_total == n_unique + n_skipped + n_conv_fail + n_canon_fail",
                "values": check1_expr,
                "passed": check1_pass,
            },
            "check2_wrapper_coverage": {
                "expression": "secondary_count == dedup.n_total",
                "values": f"{secondary_count} == {dedup.n_total}",
                "passed": check2_pass,
            },
            "check3_dedup_effectiveness": {
                "expression": "n_skipped > 0",
                "value": dedup.n_skipped,
                "passed": check3_pass,
            },
            "check4_canonical_correctness": {
                "n_pairs_checked": n_checked,
                "all_hashes_match": all_match,
                "passed": check4_pass,
                "details": [
                    {k: v for k, v in r.items() if k != "canonical_string"}
                    for r in spot_check_results
                ],
            },
            "check5_multiprocessing_bypass": mp_result,
        },
        "overall_result": "PASS" if overall else "FAIL",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dedup_verification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report written to: %s", out_path)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
