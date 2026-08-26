"""Draw the two figures the manuscript rebuilds from the campaign.

The critical-difference diagram already has a three-arm generator
(``experiments.figures.models.generate_critical_difference``); these two did
not, and are drawn here.

reduction_factor_distribution.pdf
    Per-problem distribution of the reduction factor across the suite, one
    panel per host, with the naive hash arm's reduction factor overlaid so that
    the vertical gap between the two is the completeness the canonical string
    buys.
rf_vs_overhead.pdf
    Reduction factor against key overhead, aggregated by the maximum internal
    node count a run reached, with a per-method Pareto frontier. Both
    deduplicating arms appear, so the horizontal displacement between them at
    matched k is what the cheaper key costs in reduction.

Usage
-----
    python -m experiments.scripts.review_campaign.figures [--analyses DIR]
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import METHODS, add_common_args  # noqa: E402

HOST_LABEL = {"udfs": "UDFS", "bingo": "Bingo"}

#: Colours, matching the critical-difference diagram's convention.
COLOUR = {
    "baseline": "#4477AA",
    "hash": "#228833",
    "isalsr": "#EE6677",
}
ARM_LABEL = {
    "baseline": "native DAG",
    "hash": "naive hash",
    "isalsr": r"\textsc{IsalSR}",
}


def read_cells(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"method", "suite", "problem", "arm", "git_hash", "engine", "config_sha256"}:
                continue
            row[key] = float(value) if value not in {"", "None"} else None
    return rows


def reduction_distribution(cells: list[dict[str, Any]], out: Path) -> None:
    """Per-problem box plots of rho, one panel per host."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 7.4))
    for ax, method in zip(axes, METHODS, strict=True):
        by_problem: dict[str, list[float]] = defaultdict(list)
        ser: dict[str, list[float]] = defaultdict(list)
        for row in cells:
            if row["method"] != method:
                continue
            if row["arm"] == "isalsr":
                by_problem[row["problem"]].append(row["rho"])
            elif row["arm"] == "hash":
                ser[row["problem"]].append(row["rho"])

        order = sorted(by_problem, key=lambda p: -st.median(by_problem[p]))
        positions = range(len(order))
        ax.boxplot(
            [by_problem[p] for p in order],
            positions=list(positions),
            widths=0.62,
            patch_artist=True,
            boxprops={
                "facecolor": COLOUR["isalsr"],
                "alpha": 0.55,
                "edgecolor": COLOUR["isalsr"],
                "linewidth": 0.7,
            },
            medianprops={"color": "black", "linewidth": 0.9},
            whiskerprops={"color": COLOUR["isalsr"], "linewidth": 0.7},
            capprops={"color": COLOUR["isalsr"], "linewidth": 0.7},
            flierprops={
                "marker": ".",
                "markersize": 2,
                "markerfacecolor": COLOUR["isalsr"],
                "markeredgecolor": "none",
            },
        )
        ax.scatter(
            list(positions),
            [st.median(ser[p]) for p in order],
            marker="_",
            s=90,
            linewidths=1.6,
            color=COLOUR["hash"],
            zorder=5,
            label="naive hash (median)",
        )
        ax.axhline(1.0, ls="--", lw=0.9, color="0.35")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(order, rotation=90, fontsize=6.0)
        ax.set_ylabel(r"reduction factor $\rho$", fontsize=9)
        ax.set_title(HOST_LABEL[method], fontsize=10, loc="left")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(-0.8, len(order) - 0.2)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D(
            [],
            [],
            color=COLOUR["isalsr"],
            lw=6,
            alpha=0.55,
            label="IsalSR, 30 seeds per problem",
        ),
        plt.Line2D(
            [],
            [],
            color=COLOUR["hash"],
            lw=1.6,
            marker="_",
            ls="none",
            markersize=10,
            label="naive hash, median",
        ),
        plt.Line2D([], [], color="0.35", ls="--", lw=0.9, label=r"$\rho = 1$"),
    ]
    axes[0].legend(
        handles=handles,
        fontsize=8,
        frameon=False,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
    )

    fig.tight_layout(h_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def rf_vs_overhead(cells: list[dict[str, Any]], out: Path) -> None:
    """Reduction factor against key overhead, aggregated by maximum k."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for ax, method in zip(axes, METHODS, strict=True):
        for arm in ("hash", "isalsr"):
            groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for row in cells:
                if row["method"] != method or row["arm"] != arm:
                    continue
                if row["overhead_pct"] is None or row["max_k"] is None:
                    continue
                groups[int(row["max_k"])].append((row["overhead_pct"], row["rho"]))

            points = []
            for k, values in sorted(groups.items()):
                if len(values) < 5:
                    continue
                oh = [v[0] for v in values]
                rho = [v[1] for v in values]
                points.append((k, st.fmean(oh), st.fmean(rho), st.pstdev(oh), st.pstdev(rho)))
            if not points:
                continue

            ks, ohs, rhos, oh_sd, rho_sd = zip(*points, strict=True)
            ax.errorbar(
                ohs,
                rhos,
                xerr=oh_sd,
                yerr=rho_sd,
                fmt="none",
                ecolor=COLOUR[arm],
                elinewidth=0.6,
                alpha=0.35,
            )
            sc = ax.scatter(
                ohs,
                rhos,
                c=ks,
                cmap="viridis",
                s=34,
                edgecolor=COLOUR[arm],
                linewidth=1.2,
                zorder=4,
            )

            # Pareto frontier: for each attainable rho, the least overhead.
            frontier: list[tuple[float, float]] = []
            for oh, rho in sorted(zip(ohs, rhos, strict=True), key=lambda t: t[0]):
                if not frontier or rho > frontier[-1][1]:
                    frontier.append((oh, rho))
            if len(frontier) > 1:
                xs, ys = zip(*frontier, strict=True)
                ax.step(
                    xs,
                    ys,
                    where="post",
                    color=COLOUR[arm],
                    lw=1.3,
                    alpha=0.85,
                    label=f"{ARM_LABEL[arm]} frontier",
                )

        ax.set_xscale("log")
        ax.set_xlabel("key overhead (% of wall clock)", fontsize=9)
        ax.set_ylabel(r"reduction factor $\rho$", fontsize=9)
        ax.set_title(HOST_LABEL[method], fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(lw=0.4, alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        handles = [
            plt.Line2D([], [], color=COLOUR["hash"], lw=1.4, label="naive hash"),
            plt.Line2D([], [], color=COLOUR["isalsr"], lw=1.4, label="IsalSR"),
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("maximum internal nodes $k$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )

    cells = read_cells(args.analyses / "data" / "cells.csv")
    out_dir = args.analyses / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduction_distribution(cells, out_dir / "reduction_factor_distribution.pdf")
    rf_vs_overhead(cells, out_dir / "rf_vs_overhead.pdf")


if __name__ == "__main__":
    main()
