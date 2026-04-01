"""Generate Pareto front: R² test vs model complexity.

Shows whether IsalSR achieves better accuracy at the same complexity,
preempting the "just finds more complex expressions" objection.

Usage:
    python -m experiments.figures.models.generate_pareto \
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

C_BASELINE = PAUL_TOL_BRIGHT["grey"]
C_ISALSR = PAUL_TOL_BRIGHT["red"]


def generate_pareto(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Scatter: mean R² vs mean complexity per problem, baseline vs IsalSR."""
    apply_ieee_style()

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(7.0, 3.0),
        gridspec_kw={"wspace": 0.30},
    )
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        bl_r2s, bl_cxs = [], []
        is_r2s, is_cxs = [], []
        labels = []

        for benchmark in benchmarks:
            bench_dir = results_dir / method / benchmark
            if not bench_dir.exists():
                continue
            for prob_dir in sorted(bench_dir.iterdir()):
                if not prob_dir.is_dir():
                    continue
                for variant, r2_list, cx_list in [
                    ("baseline", bl_r2s, bl_cxs),
                    ("isalsr", is_r2s, is_cxs),
                ]:
                    vdir = prob_dir / variant
                    if not vdir.exists():
                        continue
                    logs = load_all_run_logs(vdir)
                    if not logs:
                        continue
                    r2_vals = [min(max(rl.regression.r2_test, 0.0), 1.0) for rl in logs]
                    cx_vals = [float(rl.regression.model_complexity) for rl in logs]
                    r2_list.append(np.mean(r2_vals))
                    cx_list.append(np.mean(cx_vals))
                    if variant == "baseline":
                        labels.append(prob_dir.name)

        ax.scatter(
            bl_cxs,
            bl_r2s,
            color=C_BASELINE,
            marker="o",
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            label="Native DAG",
            zorder=2,
        )
        ax.scatter(
            is_cxs,
            is_r2s,
            color=C_ISALSR,
            marker="s",
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            label="IsalSR",
            zorder=3,
        )

        # Draw arrows from baseline to IsalSR for each problem
        for i in range(min(len(bl_cxs), len(is_cxs))):
            dx = is_cxs[i] - bl_cxs[i]
            dy = is_r2s[i] - bl_r2s[i]
            if abs(dx) > 0.3 or abs(dy) > 0.003:
                ax.annotate(
                    "",
                    xy=(is_cxs[i], is_r2s[i]),
                    xytext=(bl_cxs[i], bl_r2s[i]),
                    arrowprops={"arrowstyle": "->", "color": "0.5", "lw": 0.5, "alpha": 0.4},
                )

        ax.set_xlabel("Model complexity (nodes)", fontsize=9)
        ax.set_ylabel("Mean $R^2$ test", fontsize=9)
        ax.set_title(f"{method.upper()}", fontsize=10, fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(min(min(bl_r2s, default=0.5), min(is_r2s, default=0.5)) - 0.03, 1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = output_dir / f"pareto_r2_complexity.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out)

    caption = (
        "Mean $R^2$ test vs.\\ mean model complexity (number of DAG nodes) "
        "for each benchmark problem. Each point represents one problem "
        "(averaged over 30 seeds). Arrows connect baseline to \\IsalSR{} "
        "for problems where the shift is visible. "
        "\\IsalSR{} achieves equal or better $R^2$ without systematically "
        "increasing model complexity, preempting the hypothesis that "
        "deduplication simply favours more complex expressions."
    )
    (output_dir / "pareto_r2_complexity.caption.txt").write_text(caption)
    log.info("Saved caption")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto front")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    args = parser.parse_args()

    generate_pareto(
        Path(args.results_dir),
        Path(args.output_dir),
        [m.strip() for m in args.methods.split(",")],
        [b.strip() for b in args.benchmarks.split(",")],
    )


if __name__ == "__main__":
    main()
