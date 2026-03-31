"""Generate unified convergence trajectory figure.

Single-panel figure with all representative problems overlaid.
Color encodes the problem; linestyle encodes the representation
(solid = IsalSR, dashed = native DAG). Lines stop when R² >= 0.9999.

Uses Paul Tol colorblind-safe palette from plotting_styles.

Usage:
    python -m experiments.figures.models.generate_convergence \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Problem config: (method, benchmark, problem, color, label, t_max)
PROBLEMS = [
    (
        "udfs",
        "nguyen",
        "nguyen_1",
        PAUL_TOL_BRIGHT["blue"],
        "N-1: $x^3 + x^2 + x$",
        12.0,
    ),
    (
        "udfs",
        "nguyen",
        "nguyen_12",
        PAUL_TOL_BRIGHT["green"],
        "N-12: $x^4 - x^3 + y^2\\!/2 - y$",
        12.0,
    ),
    (
        "udfs",
        "feynman",
        "i.48.20",
        PAUL_TOL_BRIGHT["purple"],
        "I.48.20: $mc^2/\\!\\sqrt{1{-}(v/c)^2}$",
        12.0,
    ),
    (
        "udfs",
        "feynman",
        "i.12.4",
        PAUL_TOL_BRIGHT["red"],
        "I.12.4: $q_1/(4\\pi rc)$",
        12.0,
    ),
]

R2_CONVERGED = 0.9999  # Threshold to stop drawing lines


# ======================================================================
# Data loading
# ======================================================================


def _load_trajectories(
    prob_dir: Path,
    variant: str,
) -> list[list[tuple[float, float]]]:
    """Load (time_hours, best_r2) trajectories for all seeds."""
    vdir = prob_dir / variant
    trajs: list[list[tuple[float, float]]] = []
    if not vdir.exists():
        return trajs
    for sd in sorted(vdir.iterdir()):
        tp = sd / "trajectory.csv"
        if not tp.exists():
            continue
        points: list[tuple[float, float]] = []
        with open(tp) as f:
            for row in csv.DictReader(f):
                points.append(
                    (
                        float(row["timestamp_s"]) / 3600.0,
                        float(row["best_r2"]),
                    )
                )
        if points:
            trajs.append(points)
    return trajs


def _interpolate(
    trajs: list[list[tuple[float, float]]],
    t_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate onto common grid. Returns (n_seeds, len(t_grid))."""
    mat = np.full((len(trajs), len(t_grid)), np.nan)
    for i, traj in enumerate(trajs):
        ts = np.array([p[0] for p in traj])
        r2 = np.array([p[1] for p in traj])
        mat[i] = np.interp(t_grid, ts, r2, left=r2[0], right=r2[-1])
    return mat


def _truncate_at_convergence(
    t_grid: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    threshold: float = R2_CONVERGED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncate arrays at the first time the median reaches the threshold."""
    converged = np.where(median >= threshold)[0]
    if len(converged) == 0:
        return t_grid, median, q25, q75
    # Keep one point past convergence for visual continuity
    idx = min(converged[0] + 1, len(t_grid))
    return t_grid[:idx], median[:idx], q25[:idx], q75[:idx]


# ======================================================================
# Figure
# ======================================================================


def generate_convergence_figure(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate unified single-panel convergence figure."""
    apply_ieee_style()

    t_max = max(p[5] for p in PROBLEMS)
    t_grid = np.linspace(0, t_max, 400)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    legend_handles = []

    for method, bench, prob, color, label, _ in PROBLEMS:
        prob_dir = results_dir / method / bench / prob

        for variant, ls, alpha_band, lw in [
            ("baseline", "--", 0.07, 1.2),
            ("isalsr", "-", 0.15, 1.5),
        ]:
            trajs = _load_trajectories(prob_dir, variant)
            if not trajs:
                continue

            mat = _interpolate(trajs, t_grid)
            median = np.nanmedian(mat, axis=0)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)

            # Truncate at convergence
            t_plot, med_plot, q25_plot, q75_plot = _truncate_at_convergence(
                t_grid,
                median,
                q25,
                q75,
            )

            ax.plot(t_plot, med_plot, color=color, linestyle=ls, linewidth=lw)
            ax.fill_between(t_plot, q25_plot, q75_plot, color=color, alpha=alpha_band)

        # One colored legend entry per problem
        legend_handles.append(
            plt.Line2D([], [], color=color, linestyle="-", linewidth=1.5, label=label)
        )

    # Representation legend entries (black, style only)
    legend_handles.append(
        plt.Line2D([], [], color="black", linestyle="-", linewidth=1.2, label="IsalSR")
    )
    legend_handles.append(
        plt.Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Native DAG")
    )

    ax.set_xlabel("Wall-clock time (hours)")
    ax.set_ylabel("Median $R^2$ test")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.5, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # Legend outside below
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = output_dir / f"convergence_trajectories.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out)

    cap_path = output_dir / "convergence_trajectories.caption.txt"
    cap_path.write_text(_caption())
    log.info("Saved %s", cap_path)
    plt.close(fig)


def _caption() -> str:
    return (
        "Convergence trajectories for four representative UDFS problems. "
        "Solid lines: \\IsalSR; dashed lines: native DAG baseline. "
        "Each curve shows the median $R^2$ test over 30 seeds; "
        "shaded bands: interquartile range. "
        "Lines terminate when the median reaches $R^2 \\geq 0.9999$. "
        "Nguyen-1 (blue): easy polynomial---both representations "
        "converge identically, confirming no harm. "
        "Nguyen-12 (green): hard two-variable problem---neither variant "
        "reaches $R^2 = 1$, but \\IsalSR{} achieves consistently higher "
        "$R^2$ throughout the 12-hour budget. "
        "I.48.20 (purple): relativistic energy---baseline solves 1/30 "
        "seeds while \\IsalSR{} solves 30/30. "
        "I.12.4 (red): the baseline median plateaus at $R^2 \\approx 0.76$ "
        "while \\IsalSR{} reaches $R^2 = 1.0$, demonstrating that "
        "deduplication enables the search to escape local optima."
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate convergence figure")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_convergence_figure(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
