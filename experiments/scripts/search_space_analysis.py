"""Search space reduction analysis -- THE KEY EXPERIMENT for the IsalSR paper.

Measures the empirical O(k!) search space reduction achieved by canonicalization.
For each benchmark: generate N random IsalSR strings, count unique canonical forms,
and compute the reduction factor = N_valid / N_unique.

This directly validates the paper's central claim: the canonical representation
collapses O(k!) equivalent expression DAGs into one.

Usage:
    python experiments/scripts/search_space_analysis.py \
        --n-strings 1000 --max-tokens 15 --seed 42 \
        --output results/search_space_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from typing import Any

import numpy as np

# Add project root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS  # noqa: E402
from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string  # noqa: E402
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.search.random_search import random_isalsr_string  # noqa: E402

# Default timeout (seconds) per canonicalization call.
_CANON_TIMEOUT: float = 5.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/media/mpascual/Sandisk2TB/research/isalsr/results/search_space_analysis.csv"


def analyze_benchmark(
    benchmark: dict[str, Any],
    n_strings: int,
    max_tokens: int,
    allowed_ops: OperationSet,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze search space reduction for a single benchmark."""
    rng = np.random.default_rng(seed)
    nv = benchmark["num_variables"]

    canonical_set: set[str] = set()
    total_valid = 0
    n_timeouts = 0
    node_counts: list[int] = []

    for _ in range(n_strings):
        raw = random_isalsr_string(nv, max_tokens, allowed_ops, rng)
        try:
            dag = StringToDAG(raw, nv, allowed_ops).run()
            if dag.node_count <= nv:
                continue
            total_valid += 1
            canon = pruned_canonical_string(dag, timeout=_CANON_TIMEOUT)
            canonical_set.add(canon)
            node_counts.append(dag.node_count - nv)
        except CanonicalTimeoutError:
            n_timeouts += 1
            continue
        except Exception:  # noqa: BLE001
            continue

    n_unique = len(canonical_set)
    avg_k = float(np.mean(node_counts)) if node_counts else 0.0
    reduction = total_valid / max(n_unique, 1)
    theoretical = math.factorial(max(1, round(avg_k)))

    if n_timeouts > 0:
        log.warning("%s: %d canonicalization timeouts", benchmark["name"], n_timeouts)

    log.info(
        "%s: %d valid / %d unique = %.1fx (k=%.1f, k!=%.0f)",
        benchmark["name"],
        total_valid,
        n_unique,
        reduction,
        avg_k,
        theoretical,
    )

    return {
        "benchmark": benchmark["name"],
        "n_generated": n_strings,
        "n_valid": total_valid,
        "n_unique_canonical": n_unique,
        "reduction_factor": round(reduction, 2),
        "avg_internal_nodes": round(avg_k, 2),
        "theoretical_k_factorial": theoretical,
    }


def main() -> None:
    """Run search space analysis on Nguyen benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR search space reduction analysis")
    parser.add_argument("--n-strings", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=15)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    log.info("Search space analysis: %d strings, max %d tokens", args.n_strings, args.max_tokens)

    results: list[dict[str, Any]] = []
    for bench in NGUYEN_BENCHMARKS:
        result = analyze_benchmark(bench, args.n_strings, args.max_tokens, allowed_ops, args.seed)
        results.append(result)

    # Write CSV.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    log.info("Results saved to %s", args.output)

    # Summary table.
    print("\n=== Search Space Reduction Analysis ===")
    print(f"{'Benchmark':<15} {'Valid':>8} {'Unique':>8} {'Reduction':>10} {'Avg k':>8} {'k!':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['benchmark']:<15} {r['n_valid']:>8} {r['n_unique_canonical']:>8} "
            f"{r['reduction_factor']:>10.1f}x {r['avg_internal_nodes']:>8.1f} "
            f"{r['theoretical_k_factorial']:>10}"
        )


if __name__ == "__main__":
    main()
