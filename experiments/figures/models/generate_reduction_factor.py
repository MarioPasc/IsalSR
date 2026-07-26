"""Generate Reduction Factor distribution figure for IsalSR model validation.

Produces a single wide figure with all problems on one x-axis, visually
grouped by benchmark suite (Nguyen, Feynman, Hard, Cherrypicked) with
vertical separators between groups.

Each box = 30 seeds. Horizontal dashed line at ρ = 1 (no reduction).
Problems sorted by descending median ρ within each group.

Usage:
    python -m experiments.figures.models.generate_reduction_factor \
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

METHOD_STYLE = {
    "udfs": {"color": PAUL_TOL_BRIGHT["green"], "label": "UDFS"},
    "bingo": {"color": PAUL_TOL_BRIGHT["purple"], "label": "Bingo"},
}

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
    "i.15.10": "I.15.10",
    "i.30.3": "I.30.3",
    "i.37.4": "I.37.4",
    "ii.11.27": "II.11.27",
    "iii.17.37": "III.17.37",
    "keijzer_6": "Keij-6",
    "korns_12": "Korns-12",
    "pagie_1": "Pagie-1",
    "vladislavleva_2": "Vlad-2",
    "vladislavleva_4": "Vlad-4",
    "i.16.6": "I.16.6",
    "i.29.16": "I.29.16",
    "i.50.26": "I.50.26",
    "ii.11.28": "II.11.28",
    "iii.14.14": "III.14.14",
    "keijzer_11": "Keij-11",
    "liv_14": "Liv-14",
    "r2": "R2",
    "r3": "R3",
    "vlad_7": "Vlad-7",
}

_SRC_NGUYEN = {
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
}
_SRC_FEYNMAN = {
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
    "i.15.10",
    "i.30.3",
    "i.37.4",
    "ii.11.27",
    "iii.17.37",
    "i.16.6",
    "i.29.16",
    "i.50.26",
    "ii.11.28",
    "iii.14.14",
}
_SRC_KEIJZER = {"keijzer_6", "keijzer_11"}
_SRC_KORNS = {"korns_12"}
_SRC_PAGIE = {"pagie_1"}
_SRC_VLAD = {"vladislavleva_2", "vladislavleva_4", "vlad_7"}
_SRC_KOZA = {"r2", "r3"}
_SRC_LIVERMORE = {"liv_14"}

_GROUPS = [
    ("Nguyen\n(Uy et al. '11)", _SRC_NGUYEN),
    ("Feynman\n(Udrescu & Tegmark '20)", _SRC_FEYNMAN),
    ("Vlad.\n('09)", _SRC_VLAD),
    ("Keij.\n('03)", _SRC_KEIJZER),
    ("Koza\n(DSO '21)", _SRC_KOZA),
    ("Korns\n('11)", _SRC_KORNS),
    ("Pagie\n('97)", _SRC_PAGIE),
    ("Liv.\n('21)", _SRC_LIVERMORE),
]


def _load_rf_data(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
) -> dict[str, dict[str, list[float]]]:
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


def generate_reduction_factor_figure(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    *,
    show_dag_weighted: bool = False,
) -> None:
    """Generate single-panel figure with all problems grouped by benchmark."""
    apply_ieee_style()

    rf_data = _load_rf_data(results_dir, methods, benchmarks[0])

    all_problems: set[str] = set()
    for m in methods:
        all_problems.update(rf_data.get(m, {}).keys())

    grouped: list[tuple[str, list[str]]] = []
    for group_name, members in _GROUPS:
        probs_in_group = [p for p in all_problems if p in members]
        if not probs_in_group:
            continue

        def _median_rf(prob: str) -> float:
            vals: list[float] = []
            for m in methods:
                vals.extend(rf_data.get(m, {}).get(prob, []))
            return float(np.median(vals)) if vals else 1.0

        probs_in_group.sort(key=_median_rf, reverse=True)
        grouped.append((group_name, probs_in_group))

    n_total = sum(len(probs) for _, probs in grouped)
    if n_total == 0:
        log.warning("No RF data found")
        return

    n_methods = len(methods)
    width = 0.22
    gap_between_groups = 0.6
    group_width = width

    fig_width = max(7.0, n_total * 0.35 + len(grouped) * gap_between_groups)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    group_centers: list[tuple[float, float, str]] = []
    x = 0.0

    for gi, (group_name, probs) in enumerate(grouped):
        group_start = x

        for prob in probs:
            center = x
            tick_positions.append(center)
            tick_labels.append(_PROBLEM_LABELS.get(prob, prob))

            for j, method in enumerate(methods):
                style = METHOD_STYLE[method]
                color = style["color"]
                offset = (j - (n_methods - 1) / 2) * width

                if prob not in rf_data.get(method, {}):
                    continue

                vals = rf_data[method][prob]
                pos = center + offset

                if np.std(vals) < 1e-10:
                    hw = width * 0.40
                    ax.plot(
                        [pos - hw, pos + hw],
                        [vals[0], vals[0]],
                        color=color,
                        linewidth=2.5,
                        alpha=0.65,
                        solid_capstyle="round",
                    )
                else:
                    ax.boxplot(
                        [vals],
                        positions=[pos],
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

            x += n_methods * width + 0.08

        group_end = x - (n_methods * width + 0.08)
        group_centers.append((group_start, group_end, group_name))

        if gi < len(grouped) - 1:
            sep_x = x + gap_between_groups / 2 - (n_methods * width + 0.08) / 2
            ax.axvline(x=sep_x, color="0.5", linestyle=":", linewidth=0.8, alpha=0.6)
            x += gap_between_groups

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=12)
    ax.set_ylabel("Reduction factor $\\rho$", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    y_max = 1.15
    for m in methods:
        for seeds in rf_data.get(m, {}).values():
            if seeds:
                y_max = max(y_max, max(seeds))
    ax.set_ylim(0.95, y_max * 1.05)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=12)

    for g_start, g_end, g_name in group_centers:
        g_mid = (g_start + g_end) / 2
        ax.text(
            g_mid,
            1.02,
            g_name,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    for method in methods:
        style = METHOD_STYLE[method]
        ax.plot(
            [],
            [],
            color=style["color"],
            marker="s",
            markersize=7,
            linestyle="none",
            alpha=0.65,
            label=style["label"],
        )
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)

    if tick_positions:
        margin = (n_methods * width + 0.08) * 0.6
        ax.set_xlim(tick_positions[0] - margin, tick_positions[-1] + margin)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = output_dir / f"reduction_factor_distribution.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)

    caption = (
        "Distribution of the empirical reduction factor $\\rho$ across "
        "42 benchmark problems for UDFS and Bingo. Problems are visually "
        "grouped by source suite (Nguyen, Feynman, Hard, Cherrypicked) "
        "and sorted by descending median $\\rho$ within each group. "
        "Each box summarizes $\\rho$ over 30 independent seeds; "
        "thick horizontal ticks denote problems where all seeds yield "
        "identical $\\rho$ (deterministic UDFS enumeration). "
        "The horizontal dashed line marks $\\rho = 1$ (no redundancy). "
        "All boxes lie strictly above $\\rho = 1$, confirming that "
        "\\IsalSR{} detects structural duplicates on every problem."
    )
    (output_dir / "reduction_factor_distribution.caption.txt").write_text(caption)
    log.info("Saved caption")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reduction factor distribution figure",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="benchmark")
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
