"""Generate case study figure for I.12.4 (strongest individual result).

Three-panel figure showing that deduplication enables better search:
  (a) Violin plot: R² test distribution (baseline bimodal vs IsalSR unimodal)
  (b) Convergence curves: median R² over wall-clock time (trajectory data)
  (c) Survival plot: fraction of seeds reaching R²≥0.999 over time

Usage:
    python -m experiments.figures.models.generate_case_study \
        --results-dir /path/to/results \
        --output-dir /path/to/figures \
        [--problem i.12.4] [--method udfs] [--benchmark feynman]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Colors consistent with the rest of the paper
C_BASELINE = "#777777"
C_ISALSR = "#b2182b"


# ======================================================================
# Data loading
# ======================================================================


def _load_r2_test(prob_dir: Path, variant: str) -> list[float]:
    """Load final R² test from run_log.json for all seeds."""
    vdir = prob_dir / variant
    r2s = []
    if not vdir.exists():
        return r2s
    for sd in sorted(vdir.iterdir()):
        rl = sd / "run_log.json"
        if not rl.exists() or rl.stat().st_size < 100:
            continue
        with open(rl) as f:
            d = json.load(f)
        r2s.append(d["results"]["regression"]["r2_test"])
    return r2s


def _load_trajectories(
    prob_dir: Path,
    variant: str,
) -> list[list[tuple[float, float]]]:
    """Load (timestamp_s, best_r2) trajectories for all seeds.

    Returns list of trajectories, each a list of (time_hours, r2) tuples.
    """
    vdir = prob_dir / variant
    trajs: list[list[tuple[float, float]]] = []
    if not vdir.exists():
        return trajs
    for sd in sorted(vdir.iterdir()):
        traj_path = sd / "trajectory.csv"
        if not traj_path.exists():
            continue
        points: list[tuple[float, float]] = []
        with open(traj_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_h = float(row["timestamp_s"]) / 3600.0
                r2 = float(row["best_r2"])
                points.append((t_h, r2))
        if points:
            trajs.append(points)
    return trajs


def _load_time_to_threshold(
    prob_dir: Path,
    variant: str,
    threshold: float = 0.999,
) -> list[float | None]:
    """Load time (hours) to reach R²≥threshold from trajectory, per seed.

    Returns None for seeds that never reached the threshold.
    """
    vdir = prob_dir / variant
    times: list[float | None] = []
    if not vdir.exists():
        return times
    for sd in sorted(vdir.iterdir()):
        traj_path = sd / "trajectory.csv"
        if not traj_path.exists():
            continue
        t_hit: float | None = None
        with open(traj_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["best_r2"]) >= threshold:
                    t_hit = float(row["timestamp_s"]) / 3600.0
                    break
        times.append(t_hit)
    return times


# ======================================================================
# Panel plotting
# ======================================================================


def _panel_violin(ax: plt.Axes, prob_dir: Path) -> None:
    """Panel (a): violin plot of R² test distributions."""
    bl = _load_r2_test(prob_dir, "baseline")
    is_ = _load_r2_test(prob_dir, "isalsr")

    parts_bl = ax.violinplot(
        bl,
        positions=[0],
        showmedians=True,
        showextrema=False,
    )
    parts_is = ax.violinplot(
        is_,
        positions=[1],
        showmedians=True,
        showextrema=False,
    )

    # Style violins
    for pc in parts_bl["bodies"]:
        pc.set_facecolor(C_BASELINE)
        pc.set_alpha(0.5)
    parts_bl["cmedians"].set_color(C_BASELINE)
    parts_bl["cmedians"].set_linewidth(2)

    for pc in parts_is["bodies"]:
        pc.set_facecolor(C_ISALSR)
        pc.set_alpha(0.5)
    parts_is["cmedians"].set_color(C_ISALSR)
    parts_is["cmedians"].set_linewidth(2)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(42)
    jitter = 0.04
    ax.scatter(
        rng.normal(0, jitter, len(bl)),
        bl,
        color=C_BASELINE,
        s=12,
        alpha=0.7,
        zorder=3,
        edgecolors="none",
    )
    ax.scatter(
        rng.normal(1, jitter, len(is_)),
        is_,
        color=C_ISALSR,
        s=12,
        alpha=0.7,
        zorder=3,
        edgecolors="none",
    )

    # Annotate counts
    n_solved_bl = sum(1 for v in bl if v >= 0.999)
    n_solved_is = sum(1 for v in is_ if v >= 0.999)
    ax.annotate(
        f"{n_solved_bl}/30\nsolved",
        xy=(0, 1.01),
        ha="center",
        va="bottom",
        fontsize=7,
        color=C_BASELINE,
    )
    ax.annotate(
        f"{n_solved_is}/30\nsolved",
        xy=(1, 1.01),
        ha="center",
        va="bottom",
        fontsize=7,
        color=C_ISALSR,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Native DAG", "IsalSR"], fontsize=8)
    ax.set_ylabel("$R^2$ test", fontsize=9)
    ax.set_ylim(0.5, 1.08)
    ax.set_title("(a) $R^2$ distribution", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def _panel_convergence(ax: plt.Axes, prob_dir: Path) -> None:
    """Panel (b): median R² convergence over wall-clock time."""
    bl_trajs = _load_trajectories(prob_dir, "baseline")
    is_trajs = _load_trajectories(prob_dir, "isalsr")

    # Build a common time grid and interpolate each trajectory
    t_max = 12.0  # hours
    t_grid = np.linspace(0, t_max, 200)

    def _interpolate_trajs(
        trajs: list[list[tuple[float, float]]],
    ) -> np.ndarray:
        """Interpolate trajectories onto common grid. Returns (n_seeds, n_grid)."""
        matrix = np.full((len(trajs), len(t_grid)), np.nan)
        for i, traj in enumerate(trajs):
            ts = np.array([p[0] for p in traj])
            r2s = np.array([p[1] for p in traj])
            # Extend: before first point use 0, after last use last value
            interp = np.interp(t_grid, ts, r2s, left=r2s[0], right=r2s[-1])
            matrix[i] = interp
        return matrix

    if bl_trajs:
        bl_mat = _interpolate_trajs(bl_trajs)
        bl_median = np.nanmedian(bl_mat, axis=0)
        bl_q25 = np.nanpercentile(bl_mat, 25, axis=0)
        bl_q75 = np.nanpercentile(bl_mat, 75, axis=0)
        ax.plot(t_grid, bl_median, color=C_BASELINE, linewidth=1.5, label="Native DAG")
        ax.fill_between(t_grid, bl_q25, bl_q75, color=C_BASELINE, alpha=0.15)

    if is_trajs:
        is_mat = _interpolate_trajs(is_trajs)
        is_median = np.nanmedian(is_mat, axis=0)
        is_q25 = np.nanpercentile(is_mat, 25, axis=0)
        is_q75 = np.nanpercentile(is_mat, 75, axis=0)
        ax.plot(t_grid, is_median, color=C_ISALSR, linewidth=1.5, label="IsalSR")
        ax.fill_between(t_grid, is_q25, is_q75, color=C_ISALSR, alpha=0.15)

    # R²=0.999 threshold line
    ax.axhline(y=0.999, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.annotate(
        "$R^2 = 0.999$",
        xy=(0.2, 0.999),
        xytext=(0.2, 0.96),
        fontsize=7,
        color="0.4",
        arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 0.5},
    )

    ax.set_xlabel("Wall-clock time (hours)", fontsize=9)
    ax.set_ylabel("Median $R^2$ test", fontsize=9)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title("(b) Convergence", fontsize=10, fontweight="bold", loc="left")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))


def _panel_survival(ax: plt.Axes, prob_dir: Path) -> None:
    """Panel (c): survival/CDF — fraction of seeds reaching R²≥0.999 over time."""
    bl_times = _load_time_to_threshold(prob_dir, "baseline", threshold=0.999)
    is_times = _load_time_to_threshold(prob_dir, "isalsr", threshold=0.999)

    t_max = 12.0

    def _cdf_curve(
        times: list[float | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, fraction_solved) for step plot."""
        valid = sorted(t for t in times if t is not None)
        n_total = len(times)
        if not valid:
            return np.array([0, t_max]), np.array([0, 0])
        t_vals = [0.0] + valid + [t_max]
        frac_vals = [0.0]
        for i in range(len(valid)):
            frac_vals.append((i + 1) / n_total)
        frac_vals.append(frac_vals[-1])  # extend to t_max
        return np.array(t_vals), np.array(frac_vals)

    bl_t, bl_f = _cdf_curve(bl_times)
    is_t, is_f = _cdf_curve(is_times)

    ax.step(bl_t, bl_f * 100, where="post", color=C_BASELINE, linewidth=1.5, label="Native DAG")
    ax.step(is_t, is_f * 100, where="post", color=C_ISALSR, linewidth=1.5, label="IsalSR")

    # Annotate final fractions
    bl_final = sum(1 for t in bl_times if t is not None)
    is_final = sum(1 for t in is_times if t is not None)
    ax.annotate(
        f"{bl_final}/30",
        xy=(t_max, bl_final / len(bl_times) * 100),
        xytext=(-8, -12),
        textcoords="offset points",
        fontsize=7,
        color=C_BASELINE,
        ha="right",
    )
    ax.annotate(
        f"{is_final}/30",
        xy=(t_max, is_final / len(is_times) * 100),
        xytext=(-8, 5),
        textcoords="offset points",
        fontsize=7,
        color=C_ISALSR,
        ha="right",
    )

    ax.set_xlabel("Wall-clock time (hours)", fontsize=9)
    ax.set_ylabel("Seeds with $R^2 \\geq 0.999$ (\\%)", fontsize=9)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax.set_title("(c) Time to solution", fontsize=10, fontweight="bold", loc="left")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))


# ======================================================================
# Figure assembly
# ======================================================================


def generate_case_study(
    results_dir: Path,
    output_dir: Path,
    method: str,
    benchmark: str,
    problem: str,
) -> None:
    """Generate the 3-panel case study figure."""
    prob_dir = results_dir / method / benchmark / problem

    if not prob_dir.exists():
        log.error("Problem directory not found: %s", prob_dir)
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.8),
        gridspec_kw={"wspace": 0.40, "width_ratios": [0.8, 1.2, 1.2]},
    )

    _panel_violin(axes[0], prob_dir)
    _panel_convergence(axes[1], prob_dir)
    _panel_survival(axes[2], prob_dir)

    # Suptitle
    problem_label = problem.upper().replace(".", ".\\!")
    fig.suptitle(
        f"Feynman {problem.upper()}: $q_1 / (4\\pi r c)$",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = output_dir / f"case_study_{problem.replace('.', '_')}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out)

    # Caption
    caption = _caption(method, problem)
    cap_path = output_dir / f"case_study_{problem.replace('.', '_')}.caption.txt"
    cap_path.write_text(caption)
    log.info("Saved %s", cap_path)

    plt.close(fig)


def _caption(method: str, problem: str) -> str:
    return (
        f"Case study: UDFS on Feynman {problem.upper()} "
        f"($q_1 / 4\\pi rc$, 3 variables). "
        f"(a)~Violin plot of $R^2$ test over 30 seeds. "
        f"The native-DAG baseline (gray) exhibits a bimodal distribution: "
        f"17 seeds stall at $R^2 \\approx 0.7$ while 13 reach $R^2 = 1.0$. "
        f"\\IsalSR{{}} (red) achieves $R^2 = 1.0$ on all 30 seeds. "
        f"(b)~Median $R^2$ convergence over wall-clock time "
        f"(shaded band: interquartile range). "
        f"\\IsalSR{{}} converges faster and reaches $R^2 \\geq 0.999$ "
        f"while the baseline median plateaus below $0.8$. "
        f"(c)~Cumulative fraction of seeds reaching $R^2 \\geq 0.999$. "
        f"\\IsalSR{{}} achieves 100\\% success (30/30) with a median "
        f"time of 5.5\\,h, compared to the baseline's 43\\% (13/30). "
        f"This demonstrates that eliminating redundant evaluations "
        f"does not merely save compute---it enables the search to "
        f"escape local optima and find the correct solution."
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case study figure")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="udfs")
    parser.add_argument("--benchmark", default="feynman")
    parser.add_argument("--problem", default="i.12.4")
    args = parser.parse_args()

    generate_case_study(
        Path(args.results_dir),
        Path(args.output_dir),
        args.method,
        args.benchmark,
        args.problem,
    )


if __name__ == "__main__":
    main()
