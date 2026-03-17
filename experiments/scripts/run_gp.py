"""Run evolutionary (GP-style) experiment with IsalSR string crossover/mutation.

Uses the Population class with optional canonicalization.
Supports --no-canon for WITH vs WITHOUT paired comparison.

Usage:
    python experiments/scripts/run_gp.py \
        --pop-size 50 --n-generations 20 --seed 42 \
        --output-dir results/gp/canon
    python experiments/scripts/run_gp.py --no-canon \
        --output-dir results/gp/nocanon
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
from isalsr.search.population import Population  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/gp"


def main() -> None:
    """Run GP-style evolutionary search on Nguyen benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR evolutionary search experiment")
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--n-generations", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--crossover-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark", type=str, default="all", help="Run single benchmark or 'all'"
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

        pop = Population(args.pop_size, bench["num_variables"], allowed_ops)
        pop.initialize(x_train, y_train, max_tokens=args.max_tokens, seed=args.seed)
        result = pop.evolve(
            x_train,
            y_train,
            n_generations=args.n_generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            seed=args.seed,
            use_canonical=use_canonical,
        )
        all_results.append(
            {
                "benchmark": bench["name"],
                "mode": mode,
                "run_id": args.run_id,
                "seed": args.seed,
                "best_r2": result.get("best_r2", -1e10),
                "best_string": result.get("best_string", ""),
            }
        )
        log.info("  Best R^2: %.6f", result.get("best_r2", -1e10))

    out_path = os.path.join(args.output_dir, f"run_{args.run_id:02d}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
