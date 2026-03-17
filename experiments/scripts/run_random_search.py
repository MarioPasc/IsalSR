"""Run random search experiment on SR benchmarks.

Generates random IsalSR strings, canonicalizes, evaluates fitness on each
Nguyen benchmark. Saves results as CSV.

Usage:
    python experiments/scripts/run_random_search.py \
        --n-iterations 500 --max-tokens 30 --seed 42 \
        --output results/random_search.csv
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
from isalsr.search.random_search import random_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/media/mpascual/Sandisk2TB/research/isalsr/results/random_search.csv"


def main() -> None:
    """Run random search on all Nguyen benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR random search experiment")
    parser.add_argument("--n-iterations", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output CSV path (used when --output-dir is not set).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Combined with --run-id to form the path.")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run identifier (used with --output-dir).")
    parser.add_argument(
        "--no-canon",
        action="store_true",
        help="Disable canonicalization (baseline for WITH vs WITHOUT comparison).",
    )
    args = parser.parse_args()

    # Resolve output path: --output-dir + --run-id takes precedence over --output
    if args.output_dir is not None:
        run_tag = args.run_id if args.run_id is not None else 0
        output_path = os.path.join(args.output_dir, f"run_{run_tag}.csv")
    else:
        output_path = args.output

    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)
    use_canonical = not args.no_canon
    mode = "canonical" if use_canonical else "no-canonical"
    log.info("Mode: %s", mode)

    all_results: list[dict[str, Any]] = []
    for bench in NGUYEN_BENCHMARKS:
        log.info("Benchmark: %s", bench["name"])
        x_train, y_train, x_test, y_test = generate_data(bench)
        results = random_search(
            x_train,
            y_train,
            bench["num_variables"],
            allowed_ops,
            n_iterations=args.n_iterations,
            max_tokens=args.max_tokens,
            seed=args.seed,
            use_canonical=use_canonical,
        )
        best = results[0] if results else {"r2": -1e10, "canonical": ""}
        all_results.append(
            {
                "benchmark": bench["name"],
                "mode": mode,
                "best_r2": best.get("r2", -1e10),
                "best_canonical": best.get("canonical", ""),
                "n_unique": len(results),
            }
        )
        log.info("  Best R^2: %.6f, unique: %d", best.get("r2", -1e10), len(results))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
