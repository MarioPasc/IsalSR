"""Generate diversity figure for Conjecture X.7 empirical validation.

Reads raw data from the diversity experiment and produces a multi-panel
figure: PCA scatter grids (baseline vs IsalSR) and time-series panels
(delta, d_bar, R^2) with 95% confidence interval bands.

Also outputs statistical test results (paired t-test, Wilcoxon,
Cohen's d) as a JSON file.

Usage:
    python -m experiments.scripts.diversity.generate_fig_diversity \
        --input-dir /media/.../diversity \
        --output-dir /media/.../diversity
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from experiments.plotting_styles import (
    COLOR_ISALSR,
    COLOR_NATIVE,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
)

log = logging.getLogger(__name__)

RESULTS_DIR = Path("/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/diversity")

COLOR_BASELINE = COLOR_NATIVE  # Re-export for downstream imports


# ======================================================================
# Data loading
# ======================================================================


def load_summary(input_dir: Path) -> pd.DataFrame:
    """Load summary.csv into a DataFrame."""
    path = input_dir / "summary.csv"
    df = pd.read_csv(path)
    return df


def load_all_trajectories(input_dir: Path) -> pd.DataFrame:
    """Load all per-generation trajectory CSVs and concatenate.

    Trajectory files have columns: generation, variant, seed, delta,
    n_unique, d_bar_lev, d_bar_bp, diameter_lev, diameter_bp,
    best_fitness, best_r2, n_failed.

    Returns combined DataFrame with dense per-generation data.
    """
    import glob

    pattern = str(input_dir / "seed_*" / "*" / "trajectory.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_snapshot(input_dir: Path, seed: int, variant: str, gen: int) -> dict:
    """Load a single snapshot .npz file."""
    path = input_dir / f"seed_{seed:02d}" / variant / f"snapshot_gen{gen:03d}.npz"
    data = np.load(path, allow_pickle=True)
    return dict(data)


# ======================================================================
# PCA with consistent axes across variants
# ======================================================================


def compute_joint_pca(
    input_dir: Path, seed: int, gen: int
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Compute PCA on the UNION of baseline + isalsr features at (seed, gen).

    This ensures comparable axes between the two variants.

    Returns:
        (baseline_coords (N,2), isalsr_coords (N,2), (var1, var2))
    """
    from sklearn.decomposition import PCA

    snap_b = load_snapshot(input_dir, seed, "baseline", gen)
    snap_i = load_snapshot(input_dir, seed, "isalsr", gen)

    feat_b = snap_b["wl_features"]
    feat_i = snap_i["wl_features"]

    # Ensure same number of columns (pad shorter one)
    max_cols = max(feat_b.shape[1], feat_i.shape[1])
    if feat_b.shape[1] < max_cols:
        feat_b = np.hstack([feat_b, np.zeros((feat_b.shape[0], max_cols - feat_b.shape[1]))])
    if feat_i.shape[1] < max_cols:
        feat_i = np.hstack([feat_i, np.zeros((feat_i.shape[0], max_cols - feat_i.shape[1]))])

    combined = np.vstack([feat_b, feat_i])
    n_b = feat_b.shape[0]

    n_components = min(2, combined.shape[0], combined.shape[1])
    if n_components < 2:
        coords = np.zeros((combined.shape[0], 2))
        return coords[:n_b], coords[n_b:], (0.0, 0.0)

    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(combined)
    var_ratios = pca.explained_variance_ratio_
    v1 = float(var_ratios[0]) if len(var_ratios) > 0 else 0.0
    v2 = float(var_ratios[1]) if len(var_ratios) > 1 else 0.0

    return all_coords[:n_b], all_coords[n_b:], (v1, v2)


# ======================================================================
# PCA scatter panels
# ======================================================================

# Use 9 representative generations for 3x3 grids
PCA_GENS = [0, 5, 10, 20, 50, 70, 100, 120, 150]


def draw_pca_panel(
    ax,
    coords: np.ndarray,
    n_unique: int,
    n_total: int,
    gen: int,
    color: str,
    is_left: bool = False,
    is_bottom: bool = False,
):
    """Draw one PCA scatter panel."""
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color,
        alpha=PLOT_SETTINGS["scatter_alpha"],
        s=PLOT_SETTINGS["scatter_size"],
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(f"$t={gen}$", fontsize=PLOT_SETTINGS["tick_labelsize"])

    delta = n_unique / n_total if n_total > 0 else 0
    ax.text(
        0.02,
        0.98,
        f"$\\delta$={delta:.2f}\n({n_unique}/{n_total})",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
        va="top",
        ha="left",
    )

    if is_left:
        ax.set_ylabel("PC2", fontsize=PLOT_SETTINGS["tick_labelsize"])
    else:
        ax.set_yticklabels([])
    if is_bottom:
        ax.set_xlabel("PC1", fontsize=PLOT_SETTINGS["tick_labelsize"])
    else:
        ax.set_xticklabels([])

    ax.tick_params(labelsize=PLOT_SETTINGS["annotation_fontsize"] - 1)


# ======================================================================
# Time-series panels with CI bands
# ======================================================================


def draw_timeseries_panel(ax, df, metric: str, ylabel: str, ylim=None):
    """Draw a time-series panel with mean and 95% CI bands."""
    for variant, color, label in [
        ("baseline", COLOR_BASELINE, "Baseline"),
        ("isalsr", COLOR_ISALSR, "IsalSR"),
    ]:
        sub = df[df["variant"] == variant].groupby("generation")[metric]
        means = sub.mean()
        sems = sub.sem()
        gens = means.index.values
        ci_lo = means.values - 1.96 * sems.values
        ci_hi = means.values + 1.96 * sems.values

        ax.plot(
            gens,
            means.values,
            color=color,
            linewidth=PLOT_SETTINGS["line_width"],
            label=label,
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            gens,
            ci_lo,
            ci_hi,
            color=color,
            alpha=PLOT_SETTINGS["error_band_alpha"],
        )

    ax.set_xlabel("Generation $t$", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel(ylabel, fontsize=PLOT_SETTINGS["axes_labelsize"])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=PLOT_SETTINGS["tick_labelsize"])


# ======================================================================
# Statistical tests
# ======================================================================


def compute_statistical_tests(df: pd.DataFrame) -> dict:
    """Compute paired statistical tests at each generation.

    For each generation, performs:
    - Paired t-test on delta and d_bar_bp
    - Wilcoxon signed-rank test
    - Cohen's d effect size

    Returns dict keyed by generation with test results.
    """
    results = {}
    generations = sorted(df["generation"].unique())

    for gen in generations:
        gen_data = {}
        sub = df[df["generation"] == gen]
        baseline = sub[sub["variant"] == "baseline"].sort_values("seed")
        isalsr = sub[sub["variant"] == "isalsr"].sort_values("seed")

        # Match seeds
        common_seeds = set(baseline["seed"]) & set(isalsr["seed"])
        if len(common_seeds) < 2:
            results[int(gen)] = {"n_seeds": len(common_seeds), "tests": {}}
            continue

        b = baseline[baseline["seed"].isin(common_seeds)].sort_values("seed")
        i = isalsr[isalsr["seed"].isin(common_seeds)].sort_values("seed")

        for metric in ["delta", "d_bar_lev", "d_bar_bp"]:
            b_vals = b[metric].values
            i_vals = i[metric].values
            diff = i_vals - b_vals

            # Paired t-test
            if np.std(diff) > 1e-15:
                t_stat, t_pval = stats.ttest_rel(i_vals, b_vals)
            else:
                t_stat, t_pval = 0.0, 1.0 if np.mean(diff) == 0 else 0.0

            # Wilcoxon (requires non-zero differences)
            nonzero_diff = diff[np.abs(diff) > 1e-15]
            if len(nonzero_diff) >= 5:
                w_stat, w_pval = stats.wilcoxon(nonzero_diff)
            else:
                w_stat, w_pval = float("nan"), float("nan")

            # Cohen's d
            pooled_sd = np.sqrt((np.var(b_vals) + np.var(i_vals)) / 2)
            if pooled_sd > 1e-15:
                cohens_d = float(np.mean(diff) / pooled_sd)
            else:
                cohens_d = 0.0

            gen_data[metric] = {
                "mean_baseline": float(np.mean(b_vals)),
                "mean_isalsr": float(np.mean(i_vals)),
                "mean_diff": float(np.mean(diff)),
                "t_stat": float(t_stat),
                "t_pval": float(t_pval),
                "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
                "w_pval": float(w_pval) if not np.isnan(w_pval) else None,
                "cohens_d": cohens_d,
            }

        results[int(gen)] = {"n_seeds": len(common_seeds), "tests": gen_data}

    return results


# ======================================================================
# Main figure
# ======================================================================


def generate_figure(input_dir: Path, output_dir: Path):
    """Generate the full diversity figure."""
    apply_ieee_style()

    df = load_summary(input_dir)
    seeds = sorted(df["seed"].unique())
    n_seeds = len(seeds)

    # Use first seed for PCA panels (representative)
    pca_seed = seeds[0]

    # Filter PCA_GENS to available generations
    available_gens = sorted(df["generation"].unique())
    pca_gens = [g for g in PCA_GENS if g in available_gens]
    if len(pca_gens) > 9:
        pca_gens = pca_gens[:9]

    n_pca = len(pca_gens)
    n_cols = 3
    n_rows_pca = (n_pca + n_cols - 1) // n_cols

    # Figure layout: 2 PCA blocks (baseline + isalsr) + 1 time-series row
    total_rows = 2 * n_rows_pca + 1
    fig_width = PLOT_SETTINGS["figure_width_double"]
    row_height = fig_width / n_cols * 0.85
    ts_height = fig_width / n_cols * 0.7
    fig_height = 2 * n_rows_pca * row_height + ts_height + 0.8
    fig_height = min(fig_height, PLOT_SETTINGS["figure_height_max"])

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        total_rows,
        n_cols,
        figure=fig,
        hspace=0.5,
        wspace=0.25,
        height_ratios=[1] * (2 * n_rows_pca) + [1.2],
    )

    # -- PCA panels for BASELINE (top block) --
    for idx, gen in enumerate(pca_gens):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        try:
            b_coords, i_coords, pca_var = compute_joint_pca(input_dir, pca_seed, gen)
            sub = df[
                (df["seed"] == pca_seed) & (df["variant"] == "baseline") & (df["generation"] == gen)
            ]
            n_unique = int(sub["n_unique_classes"].iloc[0]) if len(sub) > 0 else 0
            n_total = len(b_coords)
            draw_pca_panel(
                ax,
                b_coords,
                n_unique,
                n_total,
                gen,
                COLOR_BASELINE,
                is_left=(col == 0),
                is_bottom=(row == n_rows_pca - 1),
            )
        except Exception as e:
            log.warning("PCA panel baseline gen=%d failed: %s", gen, e)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")

    # Row label for baseline block
    fig.text(
        0.01,
        0.5 + 0.5 * (n_rows_pca * row_height) / fig_height,
        "Baseline",
        rotation=90,
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        va="center",
        ha="center",
        fontweight="bold",
    )

    # -- PCA panels for IsalSR (middle block) --
    for idx, gen in enumerate(pca_gens):
        row = n_rows_pca + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        try:
            b_coords, i_coords, pca_var = compute_joint_pca(input_dir, pca_seed, gen)
            sub = df[
                (df["seed"] == pca_seed) & (df["variant"] == "isalsr") & (df["generation"] == gen)
            ]
            n_unique = int(sub["n_unique_classes"].iloc[0]) if len(sub) > 0 else 0
            n_total = len(i_coords)
            draw_pca_panel(
                ax,
                i_coords,
                n_unique,
                n_total,
                gen,
                COLOR_ISALSR,
                is_left=(col == 0),
                is_bottom=(row == 2 * n_rows_pca - 1),
            )
        except Exception as e:
            log.warning("PCA panel isalsr gen=%d failed: %s", gen, e)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")

    fig.text(
        0.01,
        0.5 * (n_rows_pca * row_height) / fig_height,
        "IsalSR",
        rotation=90,
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        va="center",
        ha="center",
        fontweight="bold",
        color=COLOR_ISALSR,
    )

    # -- Time-series bottom panels --
    # Use trajectory data (per-generation) for dense curves if available
    traj_df = load_all_trajectories(input_dir)
    ts_data = traj_df if not traj_df.empty else df  # fallback to snapshot summary

    # Panel 1: delta(t) — from trajectory (dense) or snapshots
    ax_delta = fig.add_subplot(gs[2 * n_rows_pca, 0])
    draw_timeseries_panel(ax_delta, ts_data, "delta", r"$\delta(P_t)$", ylim=(-0.05, 1.1))
    ax_delta.axhline(1.0, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_delta.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="lower left")

    # Panel 2: d_bar_bp — from trajectory (dense) or snapshots
    ax_dbar = fig.add_subplot(gs[2 * n_rows_pca, 1])
    draw_timeseries_panel(ax_dbar, ts_data, "d_bar_bp", r"$\bar{d}_{\mathrm{BP}}(P_t)$")
    # Overlay exact GED from snapshots if available
    if "d_bar_exact" in df.columns and (df["d_bar_exact"] >= 0).any():
        for variant, color, ls in [
            ("baseline", COLOR_BASELINE, "x"),
            ("isalsr", COLOR_ISALSR, "x"),
        ]:
            sub = df[(df["variant"] == variant) & (df["d_bar_exact"] >= 0)]
            if not sub.empty:
                grouped = sub.groupby("generation")["d_bar_exact"]
                ax_dbar.scatter(
                    grouped.mean().index,
                    grouped.mean().values,
                    color=color,
                    marker=ls,
                    s=30,
                    zorder=5,
                    label=f"Exact GED ({variant})" if variant == "baseline" else None,
                )

    # Panel 3: R^2 — from trajectory (dense) or snapshots
    ax_r2 = fig.add_subplot(gs[2 * n_rows_pca, 2])
    draw_timeseries_panel(ax_r2, ts_data, "best_r2", r"$R^2$ (train)", ylim=(-0.1, 1.1))

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "fig_diversity_conjecture")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    # Statistical tests
    stat_results = compute_statistical_tests(df)
    stat_path = output_dir / "statistical_tests.json"
    with open(stat_path, "w") as f:
        json.dump(stat_results, f, indent=2)
    log.info("Saved statistical tests: %s", stat_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("STATISTICAL TEST SUMMARY")
    print("=" * 80)
    for gen, data in sorted(stat_results.items()):
        n = data["n_seeds"]
        print(f"\nGeneration t={gen} (n_seeds={n}):")
        for metric, tests in data.get("tests", {}).items():
            sig = "*" if tests["t_pval"] < 0.05 else " "
            print(
                f"  {metric:12s}: baseline={tests['mean_baseline']:.4f} "
                f"isalsr={tests['mean_isalsr']:.4f} "
                f"diff={tests['mean_diff']:+.4f} "
                f"t={tests['t_stat']:.2f} p={tests['t_pval']:.4f}{sig} "
                f"d={tests['cohens_d']:.2f}"
            )
    print("=" * 80)


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate diversity figure")
    parser.add_argument("--input-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    generate_figure(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
