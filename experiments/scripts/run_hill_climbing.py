"""Run hill climbing experiment on SR benchmarks.

Multi-restart hill climbing with mandatory canonicalization on Nguyen benchmarks.

Usage:
    python experiments/scripts/run_hill_climbing.py \
        --n-iterations 200 --n-restarts 5 --seed 42 \
        --output results/hill_climbing.csv
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

DEFAULT_OUTPUT = "/media/mpascual/Sandisk2TB/research/isalsr/results/hill_climbing.csv"


def main() -> None:
    """Run hill climbing on all Nguyen benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR hill climbing experiment")
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    all_results: list[dict[str, Any]] = []
    for bench in NGUYEN_BENCHMARKS:
        log.info("Benchmark: %s", bench["name"])
        x_train, y_train, x_test, y_test = generate_data(bench)
        results = hill_climbing(
            x_train,
            y_train,
            bench["num_variables"],
            allowed_ops,
            n_iterations=args.n_iterations,
            max_tokens=args.max_tokens,
            n_restarts=args.n_restarts,
            seed=args.seed,
        )
        best = results[0] if results else {"r2": -1e10, "string": ""}
        all_results.append(
            {
                "benchmark": bench["name"],
                "best_r2": best.get("r2", -1e10),
                "best_string": best.get("string", ""),
            }
        )
        log.info("  Best R^2: %.6f", best.get("r2", -1e10))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
