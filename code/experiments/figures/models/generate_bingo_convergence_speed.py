"""Bingo convergence speed: generation vs best R² for all hard-tier problems.

Single-column IEEE figure showing per-seed trajectories (thin, transparent)
and per-(problem, variant) mean curves. A marker highlights the generation at
which the mean R² reaches its maximum. A textbox reports the grand-mean
convergence generation for each representation with 95% CI.

Data source: ``convergence_log.npz`` files produced by Bingo runners.

Usage:
    python -m experiments.figures.models.generate_bingo_convergence_speed \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_MUTED,
    apply_ieee_style,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ======================================================================
# Problem configuration
# ======================================================================

PROBLEMS = [
    "i.15.10",
    "i.30.3",
    "i.37.4",
    "ii.11.27",
    "iii.17.37",
    "keijzer_6",
    "korns_12",
    "pagie_1",
    "vladislavleva_2",
    "vladislavleva_4",
]

PROBLEM_COLORS: dict[str, str] = {
    "i.15.10": PAUL_TOL_MUTED[0],  # rose
    "i.30.3": PAUL_TOL_MUTED[1],  # indigo
    "i.37.4": PAUL_TOL_MUTED[2],  # sand
    "ii.11.27": PAUL_TOL_MUTED[3],  # green
    "iii.17.37": PAUL_TOL_MUTED[4],  # cyan
    "keijzer_6": PAUL_TOL_MUTED[5],  # wine
    "korns_12": PAUL_TOL_MUTED[6],  # teal
    "pagie_1": PAUL_TOL_MUTED[7],  # olive
    "vladislavleva_2": PAUL_TOL_MUTED[8],  # purple
    "vladislavleva_4": "#4477AA",  # blue (Paul Tol Bright)
}

PROBLEM_LABELS: dict[str, str] = {
    "i.15.10": "I.15.10",
    "i.30.3": "I.30.3",
    "i.37.4": "I.37.4",
    "ii.11.27": "II.11.27",
    "iii.17.37": "III.17.37",
    "keijzer_6": "Keijzer-6",
    "korns_12": "Korns-12",
    "pagie_1": "Pagie-1",
    "vladislavleva_2": "Vlad.-2",
    "vladislavleva_4": "Vlad.-4",
}

VARIANTS = ["baseline", "isalsr"]
N_INTERP = 500
CI_Z = 1.96


# ======================================================================
# Data loading
# ======================================================================


def _load_convergence_logs(
    results_dir: Path,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load all convergence_log.npz files.

    Returns
    -------
    dict mapping (problem, variant) -> list of (generations, best_r2_train) tuples.
    """
    data: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    bingo_dir = results_dir / "bingo" / "hard"
    if not bingo_dir.exists():
        bingo_dir = results_dir / "bingo"
        if not bingo_dir.exists():
            log.warning("No bingo directory found under %s", results_dir)
            return data

    for prob in PROBLEMS:
        prob_dir = bingo_dir / prob
        if not prob_dir.exists():
            # Try without the benchmark subdirectory
            continue
        for variant in VARIANTS:
            var_dir = prob_dir / variant
            if not var_dir.exists():
                continue
            for seed_dir in sorted(var_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                npz = seed_dir / "convergence_log.npz"
                if not npz.exists():
                    continue
                try:
                    d = np.load(npz)
                    gens = d["generations"].astype(np.float64)
                    r2 = d["best_r2_train"].astype(np.float64)
                    data[(prob, variant)].append((gens, r2))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to load %s: %s", npz, exc)

    total = sum(len(v) for v in data.values())
    log.info("Loaded %d convergence logs across %d (problem, variant) groups", total, len(data))
    return data


def _load_r2_test(
    results_dir: Path,
) -> dict[tuple[str, str], list[float]]:
    """Load R² test from run_log.json files.

    Returns
    -------
    dict mapping (problem, variant) -> list of r2_test values (one per seed).
    """
    import json

    r2_test: dict[tuple[str, str], list[float]] = defaultdict(list)
    bingo_dir = results_dir / "bingo" / "hard"
    if not bingo_dir.exists():
        bingo_dir = results_dir / "bingo"
        if not bingo_dir.exists():
            return r2_test

    for prob in PROBLEMS:
        prob_dir = bingo_dir / prob
        if not prob_dir.exists():
            continue
        for variant in VARIANTS:
            var_dir = prob_dir / variant
            if not var_dir.exists():
                continue
            for seed_dir in sorted(var_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                rl_path = seed_dir / "run_log.json"
                if not rl_path.exists():
                    continue
                try:
                    with open(rl_path) as f:
                        rl = json.load(f)
                    val = rl["results"]["regression"]["r2_test"]
                    if val is not None and np.isfinite(val):
                        r2_test[(prob, variant)].append(float(val))
                except Exception:  # noqa: BLE001
                    pass

    return r2_test


def _convergence_generation(gens: np.ndarray, r2: np.ndarray) -> int:
    """Generation at which best_r2 first reaches its final maximum."""
    final_max = r2[-1]
    idx = int(np.argmax(r2 >= final_max))
    return int(gens[idx])


def _downsample(gens: np.ndarray, r2: np.ndarray, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a trajectory to at most *n* points for rendering."""
    if len(gens) <= n:
        return gens, r2
    idx = np.linspace(0, len(gens) - 1, n, dtype=int)
    return gens[idx], r2[idx]


# ======================================================================
# Figure
# ======================================================================


def _plot_trajectories(
    target_ax: plt.Axes,
    data: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
    *,
    r2_test: dict[tuple[str, str], list[float]] | None = None,
    show_markers: bool = True,
    seed_lw: float = 0.3,
    mean_lw: float = 1.2,
    seed_alpha: float = 0.2,
    marker_size: float = 6,
    test_marker_size: float = 45,
) -> dict[str, list[int]]:
    """Plot seed and mean trajectories on *target_ax*.

    Returns
    -------
    dict mapping variant name -> list of per-seed convergence generations.
    """
    conv_gens: dict[str, list[int]] = defaultdict(list)

    for prob in PROBLEMS:
        color = PROBLEM_COLORS[prob]
        for variant in VARIANTS:
            seeds = data.get((prob, variant), [])
            if not seeds:
                continue

            ls = "-" if variant == "isalsr" else "--"
            marker_style = "s" if variant == "isalsr" else "^"

            for gens, r2 in seeds:
                r2_clip = np.clip(r2, 0.0, None)
                g_ds, r2_ds = _downsample(gens, r2_clip)
                target_ax.plot(
                    g_ds,
                    r2_ds,
                    color=color,
                    linestyle=ls,
                    linewidth=seed_lw,
                    alpha=seed_alpha,
                    rasterized=True,
                )
                conv_gens[variant].append(_convergence_generation(gens, r2))

            max_gen = max(g[-1] for g, _ in seeds)
            gen_grid = np.linspace(0, max_gen, N_INTERP)
            mat = np.empty((len(seeds), N_INTERP))
            for i, (gens, r2) in enumerate(seeds):
                r2_clip = np.clip(r2, 0.0, None)
                mat[i] = np.interp(gen_grid, gens, r2_clip, left=r2_clip[0], right=r2_clip[-1])

            mean_r2 = np.mean(mat, axis=0)
            target_ax.plot(gen_grid, mean_r2, color=color, linestyle=ls, linewidth=mean_lw)

            if show_markers:
                best_idx = int(np.argmax(mean_r2))
                target_ax.plot(
                    gen_grid[best_idx],
                    mean_r2[best_idx],
                    marker=marker_style,
                    color=color,
                    markersize=marker_size,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    zorder=5,
                )

            # R² test marker (hatched, hollow) at the last generation
            if r2_test is not None:
                test_vals = r2_test.get((prob, variant), [])
                if test_vals:
                    mean_test = float(np.mean(np.clip(test_vals, 0.0, None)))
                    sc = target_ax.scatter(
                        [gen_grid[-1]],
                        [mean_test],
                        marker=marker_style,
                        s=test_marker_size,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.8,
                        zorder=6,
                    )
                    sc.set_hatch("///")

    return conv_gens


# Zoom inset configuration
_ZOOM_X = (5_000, 27_000)
_ZOOM_Y = (0.990, 1.002)
# Indicator box on the main axes (data coords) — height decoupled from inset
_INDICATOR_Y = (0.985, 1.01)
# Inset placement in axes-fraction coords [x0, y0, width, height]
_INSET_BOUNDS = [0.2, 0.25, 0.52, 0.35]


def generate_bingo_convergence_speed(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate the Bingo convergence speed figure."""
    apply_ieee_style()

    data = _load_convergence_logs(results_dir)
    if not data:
        log.error("No convergence data found. Skipping figure.")
        return

    r2_test_data = _load_r2_test(results_dir)
    log.info("Loaded R² test for %d (problem, variant) groups", len(r2_test_data))

    fig, ax = plt.subplots(figsize=(3.39, 4.5))

    # -- Main axes --
    conv_gens_by_variant = _plot_trajectories(ax, data, r2_test=r2_test_data)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best $R^2$ (train)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)

    # -- Zoom inset on the near-R²=1 region --
    axins = ax.inset_axes(_INSET_BOUNDS)
    _plot_trajectories(
        axins,
        data,
        r2_test=r2_test_data,
        show_markers=True,
        seed_lw=0.2,
        mean_lw=0.9,
        seed_alpha=0.15,
        marker_size=4,
        test_marker_size=25,
    )
    axins.set_xlim(*_ZOOM_X)
    axins.set_ylim(*_ZOOM_Y)
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.tick_params(labelsize=5.5, length=2, width=0.4)
    axins.grid(True, alpha=0.3, linewidth=0.3, linestyle=":")

    # Indicator rectangle (height controlled by _INDICATOR_Y, independent of inset zoom)
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch  # noqa: E402

    ix0, ix1 = _ZOOM_X
    iy0, iy1 = _INDICATOR_Y
    rect = FancyBboxPatch(
        (ix0, iy0),
        ix1 - ix0,
        iy1 - iy0,
        boxstyle="round,pad=0",
        edgecolor="0.4",
        facecolor="none",
        linewidth=0.6,
        alpha=0.8,
        zorder=4,
    )
    ax.add_patch(rect)

    # Connectors: left bottom-of-box → top-left of inset,
    #             right bottom-of-box → top-right of inset
    conn_kw: dict[str, object] = {
        "coordsA": "data",
        "coordsB": "axes fraction",
        "axesA": ax,
        "axesB": axins,
        "color": "0.4",
        "linewidth": 0.6,
        "alpha": 0.8,
    }
    for x_data, x_frac in [(ix0, 0.0), (ix1, 1.0)]:
        conn = ConnectionPatch(
            xyA=(x_data, iy0),
            xyB=(x_frac, 1.0),
            **conn_kw,
        )
        ax.add_artist(conn)

    # -- Legend: problems (colors) + representations (linestyle/marker) --
    problem_handles = []
    for prob in PROBLEMS:
        if (prob, "baseline") in data or (prob, "isalsr") in data:
            problem_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=PROBLEM_COLORS[prob],
                    linestyle="-",
                    linewidth=1.2,
                    label=PROBLEM_LABELS[prob],
                )
            )

    repr_handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            linestyle="-",
            marker="s",
            markersize=4,
            linewidth=1.0,
            label="IsalSR (train)",
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            marker="^",
            markersize=4,
            linewidth=1.0,
            label="Native DAG (train)",
        ),
    ]

    # Hatched hollow markers for R² test
    if r2_test_data:
        test_handles = [
            plt.Line2D(
                [],
                [],
                color="black",
                linestyle="none",
                marker="s",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="$R^2$ test (IsalSR)",
            ),
            plt.Line2D(
                [],
                [],
                color="black",
                linestyle="none",
                marker="^",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="$R^2$ test (Native)",
            ),
        ]
        repr_handles.extend(test_handles)

    all_handles = problem_handles + repr_handles
    ax.legend(
        handles=all_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=6.5,
        frameon=False,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.4,
    )

    # -- Textbox: grand-mean convergence generation --
    textbox_lines = ["\u25c6 Mean conv. gen"]
    for variant, label in [("isalsr", "IsalSR"), ("baseline", "Native")]:
        vals = conv_gens_by_variant.get(variant, [])
        if vals:
            arr = np.array(vals, dtype=float)
            m = float(np.mean(arr))
            ci = CI_Z * float(np.std(arr, ddof=1)) / np.sqrt(len(arr))
            textbox_lines.append(f"{label}: {m:,.0f} $\\pm$ {ci:,.0f}")

    textbox_str = "\n".join(textbox_lines)
    props = {"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"}
    ax.text(
        0.98,
        0.07,
        textbox_str,
        transform=ax.transAxes,
        fontsize=6.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    # -- Save figure --
    output_dir.mkdir(parents=True, exist_ok=True)
    base = str(output_dir / "bingo_convergence_speed")
    save_figure(fig, base, formats=("pdf", "png"))

    cap_path = output_dir / "bingo_convergence_speed.caption.txt"
    cap_path.write_text(_caption())
    log.info("Saved caption to %s", cap_path)
    plt.close(fig)

    # -- Save convergence summary table --
    _save_convergence_table(data, conv_gens_by_variant, r2_test_data, output_dir)


def _paired_wilcoxon_holm(
    baseline_vals: dict[str, list[float]],
    isalsr_vals: dict[str, list[float]],
    problems: list[str],
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Paired Wilcoxon signed-rank test with Holm-Bonferroni correction.

    Returns dict mapping problem -> significant (True/False).
    """
    from scipy.stats import wilcoxon

    raw_pvals: dict[str, float] = {}
    for prob in problems:
        bv = baseline_vals.get(prob, [])
        iv = isalsr_vals.get(prob, [])
        if not bv or not iv:
            continue
        n_paired = min(len(bv), len(iv))
        b_arr = np.array(bv[:n_paired])
        i_arr = np.array(iv[:n_paired])
        diff = i_arr - b_arr
        diff = diff[diff != 0.0]
        if len(diff) < 2:
            raw_pvals[prob] = 1.0
            continue
        _, p = wilcoxon(diff, alternative="greater")
        raw_pvals[prob] = float(p)

    sorted_probs = sorted(raw_pvals, key=lambda p: raw_pvals[p])
    m = len(sorted_probs)
    sig: dict[str, bool] = {p: False for p in problems}
    for rank, prob in enumerate(sorted_probs):
        adjusted_alpha = alpha / (m - rank)
        if raw_pvals[prob] <= adjusted_alpha:
            sig[prob] = True
        else:
            break
    return sig


def _save_convergence_table(
    data: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]],
    conv_gens_by_variant: dict[str, list[int]],
    r2_test: dict[tuple[str, str], list[float]],
    output_dir: Path,
) -> None:
    """Save a LaTeX table with Max R² train, R² test, Conv. Gen, and significance."""
    variant_labels = {"baseline": "Native DAG", "isalsr": "IsalSR"}

    # Per-(problem, variant) statistics for train R² and convergence gen
    stats: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    train_vals: dict[str, dict[str, list[float]]] = {"baseline": {}, "isalsr": {}}
    for prob in PROBLEMS:
        for variant in VARIANTS:
            seeds = data.get((prob, variant), [])
            if not seeds:
                continue
            best_r2s = [float(np.max(np.clip(r2, 0.0, None))) for _, r2 in seeds]
            conv_gs = [_convergence_generation(g, r2) for g, r2 in seeds]
            train_vals[variant][prob] = best_r2s
            stats[(prob, variant)] = (
                float(np.mean(best_r2s)),
                float(np.std(best_r2s, ddof=1)) if len(best_r2s) > 1 else 0.0,
                float(np.mean(conv_gs)),
                float(np.std(conv_gs, ddof=1)) if len(conv_gs) > 1 else 0.0,
            )

    # Per-(problem, variant) statistics for test R²
    test_stats: dict[tuple[str, str], tuple[float, float, int]] = {}
    test_vals_base: dict[str, list[float]] = {}
    test_vals_isal: dict[str, list[float]] = {}
    for prob in PROBLEMS:
        for variant in VARIANTS:
            vals = r2_test.get((prob, variant), [])
            if vals:
                test_stats[(prob, variant)] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    len(vals),
                )
                if variant == "baseline":
                    test_vals_base[prob] = vals
                else:
                    test_vals_isal[prob] = vals

    # Significance tests (Holm-Bonferroni corrected)
    sig_train = _paired_wilcoxon_holm(train_vals["baseline"], train_vals["isalsr"], PROBLEMS)
    sig_test = _paired_wilcoxon_holm(test_vals_base, test_vals_isal, PROBLEMS)

    # Build LaTeX
    prob_headers = " & ".join(f"\\textbf{{{PROBLEM_LABELS[p]}}}" for p in PROBLEMS)
    ncols = len(PROBLEMS)
    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Bingo convergence summary on the hard benchmark suite "
        "(mean $\\pm$ std across seeds). "
        "${}^{*}$\\,significant IsalSR improvement "
        "(Wilcoxon signed-rank, $\\alpha{=}0.05$, Holm--Bonferroni corrected). "
        "${}^{\\dagger}$\\,median reported due to catastrophic outliers.}",
        "\\label{tab:bingo_convergence}",
        f"\\begin{{tabular}}{{l{'c' * ncols}}}",
        "\\toprule",
        f" & {prob_headers} \\\\",
        "\\midrule",
    ]

    # Group 1: Max R² (train)
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Max $R^2$ (train)}}}} \\\\")
    for variant in VARIANTS:
        cells = []
        for prob in PROBLEMS:
            s = stats.get((prob, variant))
            if s is None:
                cells.append("--")
            else:
                m, sd = s[0], s[1]
                cell = f"${m:.3f} \\pm {sd:.3f}$"
                if variant == "isalsr" and sig_train.get(prob, False):
                    cell += "$^{*}$"
                cells.append(cell)
        row = " & ".join(cells)
        lines.append(f"\\quad {variant_labels[variant]} & {row} \\\\")

    # Group 2: R² (test)
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{$R^2$ (test)}}}} \\\\")
    for variant in VARIANTS:
        cells = []
        for prob in PROBLEMS:
            ts = test_stats.get((prob, variant))
            if ts is None:
                cells.append("--")
            else:
                m, sd, n = ts
                if abs(m) > 1.5 or sd > 1.0:
                    vals = r2_test.get((prob, variant), [])
                    med = float(np.median(vals))
                    cell = f"${med:.3f}^{{\\dagger}}$"
                else:
                    cell = f"${m:.3f} \\pm {sd:.3f}$"
                if variant == "isalsr" and sig_test.get(prob, False):
                    cell += "$^{*}$"
                cells.append(cell)
        row = " & ".join(cells)
        lines.append(f"\\quad {variant_labels[variant]} & {row} \\\\")

    # Group 3: Convergence generation
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Conv.\\ Gen}}}} \\\\")
    for variant in VARIANTS:
        cells = []
        for prob in PROBLEMS:
            s = stats.get((prob, variant))
            if s is None:
                cells.append("--")
            else:
                m, sd = s[2], s[3]
                cells.append(f"${m:,.0f} \\pm {sd:,.0f}$")
        row = " & ".join(cells)
        lines.append(f"\\quad {variant_labels[variant]} & {row} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    tex_path = output_dir / "table_bingo_convergence.tex"
    tex_path.write_text("\n".join(lines))
    log.info("Saved %s", tex_path)


def _caption() -> str:
    return (
        "Bingo convergence trajectories on the hard benchmark suite. "
        "Each thin transparent line shows one seed's cumulative-best $R^2$ (train) "
        "over generations; opaque lines are per-problem means across seeds. "
        "Solid lines with square markers: \\IsalSR; dashed lines with triangle "
        "markers: native DAG baseline. "
        "Markers indicate the generation at which the mean $R^2$ first reaches "
        "its maximum. "
        "The textbox reports the grand-mean convergence generation "
        "(across all 10 problems and all available seeds) with 95\\% confidence "
        "intervals. "
        "Problems where \\IsalSR{} achieves higher final $R^2$ (e.g.\\ III.17.37, "
        "I.37.4) demonstrate that the search-space reduction translates to "
        "regression quality gains when problems are sufficiently difficult."
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bingo convergence speed figure")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate_bingo_convergence_speed(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
