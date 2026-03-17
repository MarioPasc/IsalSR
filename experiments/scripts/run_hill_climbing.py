"""Run hill climbing experiment on SR benchmarks.

Multi-restart hill climbing with optional canonicalization on Nguyen benchmarks.
Supports --no-canon for WITH vs WITHOUT paired comparison.

Usage:
    python experiments/scripts/run_hill_climbing.py \
        --n-iterations 200 --n-restarts 5 --seed 42 \
        --output-dir results/hill_climbing/canon
    python experiments/scripts/run_hill_climbing.py --no-canon \
        --output-dir results/hill_climbing/nocanon
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS, generate_data  # noqa: E402
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.search.hill_climbing import hill_climbing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/hill_climbing"


def main() -> None:
    """Run hill climbing on Nguyen benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR hill climbing experiment")
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Run single benchmark (e.g., 'Nguyen-1') or 'all'",
    )
    parser.add_argument(
        "--no-canon", action="store_true", help="Disable canonicalization (baseline comparison)"
    )
    parser.add_argument(
        "--run-id", type=int, default=0, help="Run identifier (for multi-run experiments)"
    )
    args = parser.parse_args()

    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)
    use_canonical = not args.no_canon
    mode = "canonical" if use_canonical else "no-canonical"

    benchmarks = NGUYEN_BENCHMARKS
    if args.benchmark != "all":
        benchmarks = [b for b in NGUYEN_BENCHMARKS if b["name"] == args.benchmark]
        if not benchmarks:
            log.error("Unknown benchmark: %s", args.benchmark)
            return

    log.info("Mode: %s | Run: %d | Benchmarks: %d", mode, args.run_id, len(benchmarks))

    all_results: list[dict[str, Any]] = []
    for bench in benchmarks:
        log.info("Benchmark: %s", bench["name"])
        x_train, y_train, x_test, y_test = generate_data(bench, seed=args.seed)
        results = hill_climbing(
            x_train,
            y_train,
            bench["num_variables"],
            allowed_ops,
            n_iterations=args.n_iterations,
            max_tokens=args.max_tokens,
            n_restarts=args.n_restarts,
            seed=args.seed,
            use_canonical=use_canonical,
        )
        best = results[0] if results else {"r2": -1e10, "string": ""}
        all_results.append(
            {
                "benchmark": bench["name"],
                "mode": mode,
                "run_id": args.run_id,
                "seed": args.seed,
                "best_r2": best.get("r2", -1e10),
                "best_string": best.get("string", ""),
            }
        )
        log.info("  Best R^2: %.6f", best.get("r2", -1e10))

    out_path = os.path.join(args.output_dir, f"run_{args.run_id:02d}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
