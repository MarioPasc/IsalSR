"""Post-processing and figure generation for arXiv experimental section.

Loads results from all arXiv experiments, generates publication-quality
figures and LaTeX tables, and produces a summary Markdown report.

Usage:
    python experiments/scripts/analyze_arxiv_results.py \
        --results-dir /media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking"


def _load_csv(path: str) -> list[dict[str, str]]:
    """Load a CSV file as a list of dicts."""
    if not os.path.exists(path):
        log.warning("CSV not found: %s", path)
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> dict:
    """Load a JSON file."""
    if not os.path.exists(path):
        log.warning("JSON not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


def generate_summary_report(results_dir: str) -> str:
    """Generate a Markdown summary of all experiment results.

    Args:
        results_dir: Base directory containing all experiment subdirectories.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("# IsalSR arXiv Experimental Results Summary\n")

    # One-to-One Property Validation (P1-P4)
    oto_path = os.path.join(results_dir, "onetoone_properties", "summary.csv")
    oto_rows = _load_csv(oto_path)
    if oto_rows:
        lines.append("## One-to-One Property Validation (P1-P4)\n")
        lines.append("| Property | m | Tested | Passed | Rate | 95% CI |")
        lines.append("|----------|---|--------|--------|------|--------|")
        for r in oto_rows:
            lines.append(
                f"| {r['property']} | {r['num_vars']} | {r['n_tested']} | "
                f"{r['n_passed']} | {r['pass_rate']} | "
                f"[{r['ci_lower']}, {r['ci_upper']}] |"
            )
        lines.append("")

    # Experiment 1: Shortest Path
    exp1_path = os.path.join(results_dir, "exp1_shortest_path", "shortest_path_examples.json")
    exp1_data = _load_json(exp1_path)
    if exp1_data:
        # Handle both list and dict formats
        pairs = exp1_data if isinstance(exp1_data, list) else exp1_data.get("pairs", [])
        lines.append("## Experiment 1: Shortest Path Between DAGs\n")
        lines.append(f"Number of pairs analyzed: {len(pairs)}\n")
        lines.append("| Pair | Expression A | Expression B | Distance |")
        lines.append("|------|-------------|-------------|----------|")
        for p in pairs:
            expr_a = p.get("expr_a_str", p.get("expr_a", "?"))
            expr_b = p.get("expr_b_str", p.get("expr_b", "?"))
            lines.append(f"| {p['pair_id']} | `{expr_a}` | `{expr_b}` | {p['distance']} |")
        lines.append("")

    # Experiment 2: Neighborhood
    exp2_path = os.path.join(results_dir, "exp2_neighborhood", "neighborhood_analysis.json")
    exp2 = _load_json(exp2_path)
    if exp2:
        lines.append("## Experiment 2: Distance-1 Neighborhood\n")
        # Handle both flat and nested formats
        metrics = exp2.get("metrics", exp2)
        lines.append(f"- Original expression: `{exp2.get('expression', 'N/A')}`")
        lines.append(f"- Canonical string: `{exp2.get('canonical_string', 'N/A')}`")
        lines.append(f"- Total neighbors: {metrics.get('total_generated', 'N/A')}")
        lines.append(f"- Valid DAGs: {metrics.get('n_valid', 'N/A')}")
        lines.append(f"- Unique canonical: {metrics.get('n_unique_canonical', 'N/A')}")
        rr = metrics.get("redundancy_rate", 0)
        lines.append(
            f"- Redundancy rate: {rr:.1%}"
            if isinstance(rr, (int, float))
            else f"- Redundancy rate: {rr}"
        )
        lines.append("")

    # Experiment 3: Canonicalization Time
    exp3_path = os.path.join(results_dir, "exp3_canonicalization_time", "timing.csv")
    exp3_rows = _load_csv(exp3_path)
    if exp3_rows:
        lines.append("## Experiment 3: Canonicalization Time vs Nodes\n")
        lines.append(f"Total measurements: {len(exp3_rows)}")
        # Summarize by n_internal
        from collections import defaultdict

        by_k: dict[int, list[float]] = defaultdict(list)
        by_k_pr: dict[int, list[float]] = defaultdict(list)
        for row in exp3_rows:
            k = int(row["n_internal"])
            if row.get("timed_out_exhaustive", "False") == "False":
                by_k[k].append(float(row["time_exhaustive_s"]))
            if row.get("timed_out_pruned", "False") == "False":
                by_k_pr[k].append(float(row["time_pruned_s"]))
        lines.append("\n| k | Median exhaustive (s) | Median pruned (s) | Speedup |")
        lines.append("|---|----------------------|-------------------|---------|")
        import statistics

        for k in sorted(by_k.keys()):
            if by_k[k] and by_k_pr.get(k):
                med_ex = statistics.median(by_k[k])
                med_pr = statistics.median(by_k_pr[k])
                speedup = med_ex / med_pr if med_pr > 0 else float("inf")
                lines.append(f"| {k} | {med_ex:.6f} | {med_pr:.6f} | {speedup:.1f}x |")
        lines.append("")

    # Experiment 4: Search Space Reduction
    exp4_path = os.path.join(results_dir, "exp4_search_space", "reduction.csv")
    if not os.path.exists(exp4_path):
        exp4_path = os.path.join(results_dir, "exp4_search_space", "reduction_basic.csv")
    exp4_rows = _load_csv(exp4_path)
    if exp4_rows:
        lines.append("## Experiment 4: Search Space Reduction\n")
        lines.append(f"Total bins: {len(exp4_rows)}")
        lines.append("")

    # Experiment 5: Pruning Accuracy
    exp5_path = os.path.join(results_dir, "exp5_pruning_accuracy", "summary.csv")
    exp5_rows = _load_csv(exp5_path)
    if exp5_rows:
        lines.append("## Experiment 5: Pruning Accuracy\n")
        total_exact = sum(int(r.get("n_exact", 0)) for r in exp5_rows)
        total_n = sum(int(r.get("n_total", 0)) for r in exp5_rows)
        if total_n > 0:
            lines.append(
                f"- Overall exact match rate: {total_exact}/{total_n} "
                f"({100 * total_exact / total_n:.1f}%)"
            )
        lines.append("")

    # Experiment 6: String Compression
    exp6_path = os.path.join(results_dir, "exp6_string_compression", "compression.csv")
    exp6_rows = _load_csv(exp6_path)
    if exp6_rows:
        lines.append("## Experiment 6: String Compression\n")
        ratios = [float(r["compression_canon"]) for r in exp6_rows if r.get("compression_canon")]
        if ratios:
            import statistics as st

            lines.append(f"- Samples: {len(ratios)}")
            lines.append(f"- Mean canonical/random ratio: {st.mean(ratios):.3f}")
            lines.append(f"- Median canonical/random ratio: {st.median(ratios):.3f}")
            shorter = sum(1 for r in ratios if r < 1.0)
            lines.append(
                f"- Canonical shorter: {shorter}/{len(ratios)} ({100 * shorter / len(ratios):.1f}%)"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate summary report for arXiv experiments."""
    parser = argparse.ArgumentParser(description="Analyze arXiv experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory containing experiment subdirectories",
    )
    args = parser.parse_args()

    log.info("Analyzing results in: %s", args.results_dir)

    # Check which experiments have results
    subdirs = [
        "onetoone_properties",
        "exp1_shortest_path",
        "exp2_neighborhood",
        "exp3_canonicalization_time",
        "exp4_search_space",
        "exp5_pruning_accuracy",
        "exp6_string_compression",
    ]
    for sd in subdirs:
        path = os.path.join(args.results_dir, sd)
        if os.path.exists(path):
            n_files = len(os.listdir(path))
            log.info("  %s: %d files", sd, n_files)
        else:
            log.warning("  %s: NOT FOUND", sd)

    # Generate report
    report = generate_summary_report(args.results_dir)

    report_path = os.path.join(args.results_dir, "summary_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Summary report saved to: %s", report_path)

    # Print report to stdout
    print(report)


if __name__ == "__main__":
    main()
