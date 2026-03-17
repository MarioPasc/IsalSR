"""Analyze experiment results and generate summary tables.

Loads CSV results from search experiments, computes summary statistics
(mean, std, max, IQR of R^2), and prints LaTeX-ready tables.

Usage:
    python experiments/scripts/analyze_results.py \
        --results-dir results/ \
        --output results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_RESULTS = "/media/mpascual/Sandisk2TB/research/isalsr/results"


def load_csv(path: str) -> list[dict[str, str]]:
    """Load a CSV file as list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def analyze_file(path: str) -> dict[str, Any]:
    """Compute summary statistics for an experiment CSV."""
    rows = load_csv(path)
    r2_values = [float(r.get("best_r2", -1e10)) for r in rows if "best_r2" in r]
    if not r2_values:
        return {"file": os.path.basename(path), "n_benchmarks": 0}

    arr = np.array(r2_values)
    return {
        "file": os.path.basename(path),
        "n_benchmarks": len(arr),
        "mean_r2": round(float(np.mean(arr)), 6),
        "std_r2": round(float(np.std(arr)), 6),
        "max_r2": round(float(np.max(arr)), 6),
        "min_r2": round(float(np.min(arr)), 6),
        "iqr_r2": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 6),
    }


def main() -> None:
    """Analyze all result CSVs in the results directory."""
    parser = argparse.ArgumentParser(description="Analyze IsalSR experiment results")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=str, default=os.path.join(DEFAULT_RESULTS, "summary.csv"))
    args = parser.parse_args()

    csv_files = sorted(
        f for f in os.listdir(args.results_dir) if f.endswith(".csv") and f != "summary.csv"
    )

    if not csv_files:
        log.warning("No CSV files found in %s", args.results_dir)
        return

    summaries: list[dict[str, Any]] = []
    for f in csv_files:
        path = os.path.join(args.results_dir, f)
        summary = analyze_file(path)
        summaries.append(summary)
        log.info(
            "%s: mean R^2=%.4f, max=%.4f, IQR=%.4f",
            f,
            summary.get("mean_r2", 0),
            summary.get("max_r2", 0),
            summary.get("iqr_r2", 0),
        )

    # Print LaTeX-ready table.
    print("\n=== Summary Table (LaTeX-ready) ===")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\hline")
    print(r"Method & Mean $R^2$ & Max $R^2$ & Std & IQR \\")
    print(r"\hline")
    for s in summaries:
        print(
            f"{s['file'].replace('.csv', '')} & "
            f"{s.get('mean_r2', 0):.4f} & {s.get('max_r2', 0):.4f} & "
            f"{s.get('std_r2', 0):.4f} & {s.get('iqr_r2', 0):.4f} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")

    # Save summary.
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        log.info("Summary saved to %s", args.output)


if __name__ == "__main__":
    main()
