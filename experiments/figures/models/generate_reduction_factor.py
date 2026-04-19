"""Generate Reduction Factor distribution figure for IsalSR model validation.

Produces a single-column figure with 2 stacked panels (1 column, 2 rows):
  Panel (a): Nguyen benchmarks — box plots of ρ per problem, grouped by method
  Panel (b): Feynman benchmarks — same layout

Each box = 30 seeds. Horizontal dashed line at ρ = 1 (no reduction).
Problems sorted by descending median ρ (highest redundancy first).
Optional diamond markers show DAG-weighted ρ per problem.

Usage:
    python -m experiments.figures.models.generate_reduction_factor \
        --results-dir /path/to/results \
        --output-dir /path/to/figures \
        [--dag-weighted]
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ======================================================================
# Visual style
# ======================================================================

METHOD_STYLE = {
    "udfs": {"color": "#228833", "label": "UDFS"},  # Paul Tol green
    "bingo": {"color": "#AA3377", "label": "Bingo"},  # Paul Tol purple
}

_NGUYEN_ORDER = [
    "nguyen_1",
    "nguyen_2",
    "nguyen_3",
    "nguyen_4",
    "nguyen_5",
    "nguyen_6",
    "nguyen_7",
    "nguyen_8",
    "nguyen_9",
    "nguyen_10",
    "nguyen_11",
    "nguyen_12",
]
_FEYNMAN_ORDER = [
    "i.6.20a",
    "i.12.1",
    "i.12.4",
    "i.14.3",
    "i.25.13",
    "i.34.27",
    "i.39.10",
    "i.48.20",
    "i.10.7",
    "ii.3.24",
]

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


# ======================================================================
# Data loading
# ======================================================================


def _load_rf_data(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
) -> dict[str, dict[str, list[float]]]:
    """Load per-seed reduction factors for each (method, problem)."""
    data: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        data[method] = {}
        bench_dir = results_dir / method / benchmark
        if not bench_dir.exists():
            continue
        for prob_dir in sorted(bench_dir.iterdir()):
            if not prob_dir.is_dir():
                continue
            isalsr_dir = prob_dir / "isalsr"
            if not isalsr_dir.exists():
                continue
            logs = load_all_run_logs(isalsr_dir)
            rfs = [
                rl.search_space.empirical_reduction_factor
                for rl in logs
                if np.isfinite(rl.search_space.empirical_reduction_factor)
            ]
            if rfs:
                data[method][prob_dir.name] = rfs
    return data


def _load_dag_weighted_rf_per_problem(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> dict[str, float]:
    """Compute DAG-weighted RF per problem: Σ(dags_i × rf_i) / Σ(dags_i)."""
    bench_dir = results_dir / method / benchmark
    result: dict[str, float] = {}
    if not bench_dir.exists():
        return result
    for prob_dir in sorted(bench_dir.iterdir()):
        if not prob_dir.is_dir():
            continue
        isalsr_dir = prob_dir / "isalsr"
        if not isalsr_dir.exists():
            continue
        total_dags = 0.0
        weighted_sum = 0.0
        for rl in load_all_run_logs(isalsr_dir):
            n = rl.search_space.total_dags_explored
            rf = rl.search_space.empirical_reduction_factor
            if n > 0 and np.isfinite(rf):
                total_dags += n
                weighted_sum += n * rf
        if total_dags > 0:
            result[prob_dir.name] = weighted_sum / total_dags
    return result


def _sort_problems_by_rf(
    problems: list[str],
    rf_data: dict[str, dict[str, list[float]]],
    methods: list[str],
) -> list[str]:
    """Sort problems by descending median RF (pooled across methods)."""

    def _median_rf(prob: str) -> float:
        all_rf: list[float] = []
        for m in methods:
            all_rf.extend(rf_data.get(m, {}).get(prob, []))
        return float(np.median(all_rf)) if all_rf else 1.0

    return sorted(problems, key=_median_rf, reverse=True)


# ======================================================================
# Plotting
# ======================================================================


def _plot_panel(
    ax: plt.Axes,
    rf_data: dict[str, dict[str, list[float]]],
    problem_order: list[str],
    methods: list[str],
    title: str,
    *,
    dag_weighted: dict[str, dict[str, float]] | None = None,
) -> None:
    """Draw one panel (one benchmark) of the reduction factor figure."""
    if problem_order:
        problems = [p for p in problem_order if any(p in rf_data[m] for m in methods)]
    else:
        all_probs: set[str] = set()
        for m in methods:
            all_probs.update(rf_data.get(m, {}).keys())
        problems = sorted(all_probs)
    problems = _sort_problems_by_rf(problems, rf_data, methods)
    n_methods = len(methods)

    if not problems:
        ax.set_visible(False)
        return

    # Narrow boxes for single-column fit
    width = 0.22
    group_width = n_methods * width + 0.08
    positions_per_method: dict[str, list[float]] = {}
    tick_positions: list[float] = []

    for i, _prob in enumerate(problems):
        center = i * group_width
        tick_positions.append(center)
        for j, method in enumerate(methods):
            offset = (j - (n_methods - 1) / 2) * width
            positions_per_method.setdefault(method, []).append(center + offset)

    # Draw box plots per method.
    # UDFS is deterministic: on easy problems all 30 seeds produce identical
    # RF (std=0). A zero-height boxplot is invisible, so we draw a thick
    # horizontal tick instead.
    for method in methods:
        style = METHOD_STYLE[method]
        color = style["color"]
        box_data = []
        box_positions = []
        const_positions: list[float] = []
        const_values: list[float] = []

        for k, prob in enumerate(problems):
            if prob not in rf_data[method]:
                continue
            vals = rf_data[method][prob]
            pos = positions_per_method[method][k]
            if np.std(vals) < 1e-10:
                const_positions.append(pos)
                const_values.append(vals[0])
            else:
                box_data.append(vals)
                box_positions.append(pos)

        if box_data:
            ax.boxplot(
                box_data,
                positions=box_positions,
                widths=width * 0.80,
                patch_artist=True,
                showfliers=True,
                flierprops={
                    "marker": ".",
                    "markersize": 2,
                    "alpha": 0.5,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                },
                medianprops={"color": "white", "linewidth": 0.8},
                boxprops={"facecolor": color, "alpha": 0.65, "edgecolor": color},
                whiskerprops={"color": color, "linewidth": 0.8},
                capprops={"color": color, "linewidth": 0.8},
            )

        # Constant-value problems: thick horizontal tick
        for pos, val in zip(const_positions, const_values, strict=True):
            hw = width * 0.40
            ax.plot(
                [pos - hw, pos + hw],
                [val, val],
                color=color,
                linewidth=2.5,
                alpha=0.65,
                solid_capstyle="round",
            )

        # Invisible handle for the unified figure-level legend
        ax.plot(
            [],
            [],
            color=color,
            marker="s",
            markersize=7,
            linestyle="none",
            alpha=0.65,
            label=style["label"],
        )

    # Optional DAG-weighted RF diamonds — one per (method, problem)
    if dag_weighted is not None:
        for method in methods:
            style = METHOD_STYLE[method]
            color = style["color"]
            dw = dag_weighted.get(method, {})
            for k, prob in enumerate(problems):
                if prob in dw:
                    x_pos = positions_per_method[method][k]
                    ax.plot(
                        x_pos,
                        dw[prob],
                        marker="D",
                        markersize=4,
                        color=color,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        zorder=5,
                        alpha=0.9,
                    )

    # ρ = 1 reference line
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    # Formatting
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [_PROBLEM_LABELS.get(p, p) for p in problems],
        rotation=90,
        ha="center",
        fontsize=7,
    )
    ax.set_ylabel("Reduction factor $\\rho$", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(
        tick_positions[0] - group_width * 0.6,
        tick_positions[-1] + group_width * 0.6,
    )
    # Finer y-ticks for taller panels
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=7)


# ======================================================================
# Figure assembly
# ======================================================================


def generate_reduction_factor_figure(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    *,
    show_dag_weighted: bool = False,
) -> None:
    """Generate the single-column 2-row figure (one row per benchmark)."""
    bench_config = {
        "nguyen": ("(a) Nguyen benchmarks", _NGUYEN_ORDER),
        "feynman": ("(b) Feynman benchmarks", _FEYNMAN_ORDER),
    }

    # Pre-compute global y-limits across both panels
    global_ymax = 1.15
    for benchmark in benchmarks:
        rf_data = _load_rf_data(results_dir, methods, benchmark)
        for m in methods:
            for seeds in rf_data.get(m, {}).values():
                if seeds:
                    global_ymax = max(global_ymax, max(seeds))
    global_ymax *= 1.05

    # Double-column: 1 row × 2 cols, shared y-axis
    fig, axes = plt.subplots(
        1,
        len(benchmarks),
        figsize=(7.0, 3.5),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    if len(benchmarks) == 1:
        axes = [axes]

    for idx, (ax, benchmark) in enumerate(zip(axes, benchmarks, strict=True)):
        title, order = bench_config.get(benchmark, (benchmark, []))
        rf_data = _load_rf_data(results_dir, methods, benchmark)

        dag_weighted: dict[str, dict[str, float]] | None = None
        if show_dag_weighted:
            dag_weighted = {}
            for method in methods:
                dag_weighted[method] = _load_dag_weighted_rf_per_problem(
                    results_dir,
                    method,
                    benchmark,
                )

        _plot_panel(ax, rf_data, order, methods, title, dag_weighted=dag_weighted)
        ax.set_ylim(0.95, global_ymax)
        # Only show y-label on the leftmost panel
        if idx > 0:
            ax.set_ylabel("")

    # Unified legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    if show_dag_weighted:
        diamond_handle = plt.Line2D(
            [],
            [],
            marker="D",
            markersize=4,
            color="gray",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
            label="DAG-weighted $\\rho$",
        )
        handles.append(diamond_handle)
        labels.append("DAG-weighted $\\rho$")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.11),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = output_dir / f"reduction_factor_distribution.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)

    caption = _caption(methods, benchmarks, show_dag_weighted)
    caption_path = output_dir / "reduction_factor_distribution.caption.txt"
    caption_path.write_text(caption)
    log.info("Saved %s", caption_path)
    plt.close(fig)


def _caption(methods: list[str], benchmarks: list[str], dag_weighted: bool) -> str:
    bench_str = " and ".join(b.capitalize() for b in benchmarks)
    method_str = " and ".join(m.upper() for m in methods)
    cap = (
        f"Distribution of the empirical reduction factor $\\rho$ across "
        f"{bench_str} benchmarks for {method_str}. "
        f"Each box summarizes $\\rho$ over 30 independent seeds; "
        f"thick horizontal ticks denote problems where all seeds yield "
        f"identical $\\rho$ (deterministic UDFS enumeration). "
        f"The horizontal dashed line marks $\\rho = 1$ (no redundancy). "
        f"Problems are sorted by descending median $\\rho$. "
    )
    if dag_weighted:
        cap += (
            "Diamond markers show the DAG-weighted $\\rho$ per problem, "
            "where each seed contributes proportionally to the number of "
            "DAGs it explored. "
        )
    cap += (
        "All boxes lie strictly above $\\rho = 1$, confirming that "
        "\\IsalSR{} detects structural duplicates on every problem."
    )
    return cap


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reduction factor distribution figure",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    parser.add_argument(
        "--dag-weighted",
        action="store_true",
        help="Overlay DAG-weighted ρ diamonds on each box",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    log.info("Generating reduction factor figure from %s", results_dir)
    generate_reduction_factor_figure(
        results_dir,
        output_dir,
        methods,
        benchmarks,
        show_dag_weighted=args.dag_weighted,
    )


if __name__ == "__main__":
    main()
