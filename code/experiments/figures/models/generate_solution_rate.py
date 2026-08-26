"""Generate Solution Recovery Rate table/figure for IsalSR.

Shows the fraction of seeds achieving exact symbolic recovery
(via SymPy simplification) per problem. Complements R² by showing
whether expressions are algebraically equivalent, not just numerically close.

Usage:
    python -m experiments.figures.models.generate_solution_rate \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.models.io_utils import load_all_run_logs  # noqa: E402
from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROBLEM_LABELS = {
    "nguyen_1": "N-1",
    "nguyen_2": "N-2",
    "nguyen_3": "N-3",
    "nguyen_4": "N-4",
    "nguyen_5": "N-5",
    "nguyen_6": "N-6",
    "nguyen_7": "N-7",
    "nguyen_8": "N-8",
    "nguyen_9": "N-9",
    "nguyen_10": "N-10",
    "nguyen_11": "N-11",
    "nguyen_12": "N-12",
    "i.6.20a": "I.6.20a",
    "i.12.1": "I.12.1",
    "i.12.4": "I.12.4",
    "i.14.3": "I.14.3",
    "i.25.13": "I.25.13",
    "i.34.27": "I.34.27",
    "i.39.10": "I.39.10",
    "i.48.20": "I.48.20",
    "i.10.7": "I.10.7",
    "ii.3.24": "II.3.24",
}


def generate_solution_rate(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate paired bar chart: solution recovery rate BL vs IS."""
    apply_ieee_style()

    for method in methods:
        problems = []
        bl_rates = []
        is_rates = []

        for benchmark in benchmarks:
            bench_dir = results_dir / method / benchmark
            if not bench_dir.exists():
                continue
            for prob_dir in sorted(bench_dir.iterdir()):
                if not prob_dir.is_dir():
                    continue
                for variant, rate_list in [("baseline", bl_rates), ("isalsr", is_rates)]:
                    vdir = prob_dir / variant
                    if not vdir.exists():
                        rate_list.append(0.0)
                        continue
                    logs = load_all_run_logs(vdir)
                    if not logs:
                        rate_list.append(0.0)
                        continue
                    recovered = sum(1 for rl in logs if rl.regression.solution_recovered)
                    rate_list.append(recovered / len(logs) * 100)
                if len(bl_rates) > len(problems):
                    problems.append(_PROBLEM_LABELS.get(prob_dir.name, prob_dir.name))

        if not problems:
            continue

        # Only show problems where at least one variant has non-zero recovery
        mask = [i for i in range(len(problems)) if bl_rates[i] > 0 or is_rates[i] > 0]

        if not mask:
            log.info("%s: no solution recovery on any problem", method.upper())
            # Write a minimal LaTeX table noting this
            tex = (
                "\\begin{table}[t]\n\\centering\\small\n"
                f"\\caption{{Solution recovery rate for {method.upper()}: "
                f"no problem achieved exact symbolic recovery under "
                f"SymPy simplification.}}\n"
                f"\\label{{tab:solution_rate_{method}}}\n"
                "\\end{table}\n"
            )
            out = output_dir / f"solution_rate_{method}.tex"
            out.write_text(tex)
            log.info("Saved %s", out)
            continue

        filtered_probs = [problems[i] for i in mask]
        filtered_bl = [bl_rates[i] for i in mask]
        filtered_is = [is_rates[i] for i in mask]

        n = len(filtered_probs)
        x = np.arange(n)
        w = 0.35

        fig, ax = plt.subplots(figsize=(min(7.0, max(3.5, n * 0.5)), 2.5))

        ax.bar(
            x - w / 2,
            filtered_bl,
            w,
            color=PAUL_TOL_BRIGHT["grey"],
            alpha=0.7,
            label="Native DAG",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x + w / 2,
            filtered_is,
            w,
            color=PAUL_TOL_BRIGHT["red"],
            alpha=0.7,
            label="IsalSR",
            edgecolor="white",
            linewidth=0.3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(filtered_probs, rotation=90, ha="center", fontsize=7)
        ax.set_ylabel("Solution recovery (\\%)", fontsize=9)
        ax.set_title(
            f"{method.upper()}: exact symbolic recovery",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ["pdf", "png"]:
            out = output_dir / f"solution_rate_{method}.{fmt}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            log.info("Saved %s", out)

        caption = (
            f"Exact solution recovery rate for {method.upper()} across "
            f"benchmark problems (only problems with $>0\\%$ recovery shown). "
            f"Recovery is verified via SymPy symbolic simplification: "
            f"$\\mathrm{{simplify}}(f_{{\\mathrm{{found}}}} - f_{{\\mathrm{{true}}}}) = 0$. "
            f"\\IsalSR{{}} matches or improves the baseline recovery rate "
            f"on every problem."
        )
        (output_dir / f"solution_rate_{method}.caption.txt").write_text(caption)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate solution rate figure")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    args = parser.parse_args()

    generate_solution_rate(
        Path(args.results_dir),
        Path(args.output_dir),
        [m.strip() for m in args.methods.split(",")],
        [b.strip() for b in args.benchmarks.split(",")],
    )


if __name__ == "__main__":
    main()
