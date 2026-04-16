"""Gaussian noise as a tunable difficulty axis for Bingo vs IsalSR.

Lightweight local test. Takes a single hard-but-solvable 1D benchmark
(Vladislavleva-2), corrupts training labels with isotropic Gaussian noise of
relative magnitude gamma * std(y_train_clean), and runs Bingo baseline and
Bingo+IsalSR for a small number of generations per (gamma, seed) pair.

Two-panel figure:
    Panel A  Final test R² (on clean y_test) vs. convergence generation,
             with triangle = native DAG baseline and square = IsalSR,
             colored by gamma.
    Panel B  Training data scatter (X_train, y_train_noisy) per gamma with
             the ground-truth curve overlaid.

Outputs under --output-dir:
    runs/<variant>_gamma<g>_seed<s>/{run_log.json, trajectory.csv}
    data/noise_gamma<g>_seed<s>.npz
    summary.csv
    config.yaml
    figure.{pdf,png}

Author: Mario Pascual Gonzalez (mpascual@uma.es)
Date:   2026-04-15
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.datasets import feynman as _feynman
from benchmarks.datasets import hard as _hard
from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner
from experiments.models.bingo.runner import BingoBaselineRunner
from experiments.models.bingo.translator import BingoTranslator
from experiments.models.schemas import (
    TRAJECTORY_COLUMNS,
    RunMetadata,
    TrajectoryRow,
)

log = logging.getLogger(__name__)

_SUITES: dict[str, Any] = {"hard": _hard, "feynman": _feynman}
METHOD_STYLE: dict[str, dict[str, Any]] = {
    "baseline": {"marker": "^", "label": "Native DAG (Bingo)"},
    "isalsr": {"marker": "s", "label": "IsalSR representation"},
}


# ----------------------------------------------------------------------
# Data & runners
# ----------------------------------------------------------------------


@dataclass
class RunResult:
    method: str
    gamma: float
    seed: int
    convergence_gen: int
    r2_train_plateau: float
    r2_test_clean: float
    n_generations: int
    wall_clock_s: float
    best_expr: str


def _add_gaussian_noise(y_clean: np.ndarray, gamma: float, noise_seed: int) -> np.ndarray:
    """Add i.i.d. Gaussian noise with std gamma * std(y_clean)."""
    if gamma <= 0.0:
        return y_clean.copy()
    sigma = float(gamma * np.std(y_clean))
    rng = np.random.default_rng(noise_seed)
    return y_clean + rng.normal(loc=0.0, scale=sigma, size=y_clean.shape)


def _noise_rng_seed(base_seed: int, gamma_idx: int) -> int:
    return int(base_seed) + 10_000 * int(gamma_idx)


def _build_runner(method: str, cfg: BingoConfig):
    if method == "baseline":
        return BingoBaselineRunner(config=cfg)
    if method == "isalsr":
        return IsalSRBingoRunner(config=cfg)
    raise ValueError(f"unknown method {method!r}")


def _convergence_generation(trajectory: list[TrajectoryRow], fallback: int) -> tuple[int, float]:
    """First iteration at which best-so-far training R² reaches its plateau.

    Uses only intermediate snapshot rows (training R²). Returns ``fallback``
    (typically the final generation count) when the trajectory has no
    intermediate rows.
    """
    intermediate = trajectory[:-1] if len(trajectory) >= 2 else trajectory
    if not intermediate:
        return fallback, float("nan")
    r2s = [row.best_r2 for row in intermediate]
    plateau = max(r2s)
    for row in intermediate:
        if row.best_r2 >= plateau - 1e-12:
            return int(row.iteration), float(plateau)
    return int(intermediate[-1].iteration), float(plateau)


def _save_trajectory(path: Path, rows: list[TrajectoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _run_one(
    method: str,
    gamma: float,
    seed: int,
    gamma_idx: int,
    X_train: np.ndarray,
    y_train_clean: np.ndarray,
    X_test: np.ndarray,
    y_test_clean: np.ndarray,
    bench: dict[str, Any],
    suite_key: str,
    cfg: BingoConfig,
    out_root: Path,
) -> RunResult:
    y_train_noisy = _add_gaussian_noise(y_train_clean, gamma, _noise_rng_seed(seed, gamma_idx))

    data_path = out_root / "data" / f"noise_gamma{gamma:.2f}_seed{seed}.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        np.savez(
            data_path,
            X_train=X_train,
            y_train_clean=y_train_clean,
            y_train_noisy=y_train_noisy,
            X_test=X_test,
            y_test_clean=y_test_clean,
            gamma=np.float64(gamma),
            seed=np.int64(seed),
        )

    cfg_dict = {
        "population_size": cfg.population_size,
        "stack_size": cfg.stack_size,
        "operators": cfg.operators,
        "use_simplification": cfg.use_simplification,
        "crossover_prob": cfg.crossover_prob,
        "mutation_prob": cfg.mutation_prob,
        "metric": cfg.metric,
        "clo_alg": cfg.clo_alg,
        "generations": cfg.generations,
        "fitness_threshold": cfg.fitness_threshold,
        "max_time": cfg.max_time,
        "max_evals": cfg.max_evals,
        "snapshot_frequency": cfg.snapshot_frequency,
        "use_fast_canonical": cfg.use_fast_canonical,
        "canonicalization_timeout": cfg.canonicalization_timeout,
        "enforce_population_dedup": cfg.enforce_population_dedup,
    }

    runner = _build_runner(method, cfg)
    log.info(
        "Running method=%s gamma=%.2f seed=%d (pop=%d stack=%d max_gen=%d max_time=%.0fs)",
        method,
        gamma,
        seed,
        cfg.population_size,
        cfg.stack_size,
        cfg.generations,
        cfg.max_time,
    )
    t0 = time.perf_counter()
    raw = runner.fit(X_train, y_train_noisy, X_test, y_test_clean, seed=seed, config=cfg_dict)
    wall = time.perf_counter() - t0

    translator = BingoTranslator(
        y_train=y_train_noisy,
        y_test=y_test_clean,
        ground_truth_expr=bench.get("sympy_expression"),
        ground_truth_variables=list(bench.get("sympy_variables", []) or []),
    )

    metadata = RunMetadata(
        method="bingo",
        representation=method,
        benchmark=suite_key,
        problem=bench["name"],
        seed=seed,
        hardware={},
        hyperparameters={**cfg_dict, "gamma": gamma},
    )

    run_log = translator.to_run_log(raw, metadata)
    trajectory = translator.to_trajectory(raw)

    run_dir = out_root / "runs" / f"{method}_gamma{gamma:.2f}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log.save_json(run_dir / "run_log.json")
    _save_trajectory(run_dir / "trajectory.csv", trajectory)

    conv_gen, r2_train_plateau = _convergence_generation(
        trajectory,
        fallback=raw.n_generations,
    )

    log.info(
        "  -> conv_gen=%d r2_test_clean=%.4f r2_train_plateau=%.4f wall=%.1fs",
        conv_gen,
        run_log.regression.r2_test,
        r2_train_plateau,
        wall,
    )

    return RunResult(
        method=method,
        gamma=gamma,
        seed=seed,
        convergence_gen=conv_gen,
        r2_train_plateau=r2_train_plateau,
        r2_test_clean=run_log.regression.r2_test,
        n_generations=raw.n_generations,
        wall_clock_s=wall,
        best_expr=run_log.best_expression.symbolic_form,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _plot(
    results: list[RunResult],
    gammas: list[float],
    bench: dict[str, Any],
    data_dir: Path,
    out_root: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    gamma_min, gamma_max = min(gammas), max(gammas)
    if gamma_min == gamma_max:
        gamma_max = gamma_min + 1e-6
    norm = Normalize(vmin=gamma_min, vmax=gamma_max)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax_conv, ax_data = axes

    # Panel A: convergence gen vs final R² (clean test).
    #   - individual seeds: small faint markers
    #   - per-group (method, gamma) mean: large solid marker
    #   - per-method line across gammas: dashed for baseline, solid for isalsr
    rng_jitter = np.random.default_rng(0)
    method_line_style = {"baseline": "--", "isalsr": "-"}
    for res in results:
        if not np.isfinite(res.r2_test_clean):
            continue
        style = METHOD_STYLE[res.method]
        color = cmap(norm(res.gamma))
        jitter = float(rng_jitter.uniform(-1.0, 1.0))
        ax_conv.scatter(
            res.convergence_gen + jitter,
            res.r2_test_clean,
            marker=style["marker"],
            s=36,
            facecolor=color,
            edgecolor="none",
            alpha=0.35,
            zorder=2,
        )

    # Per-(method, gamma) group aggregates — median over seeds for robustness.
    # Median ignores NaN/pathological R² from evolved expressions that happen
    # to evaluate non-finite on the test grid (e.g. log of a negative term).
    means: dict[tuple[str, float], tuple[float, float]] = {}
    for method in ("baseline", "isalsr"):
        for gamma in sorted(set(gammas)):
            rs = [
                r
                for r in results
                if r.method == method
                and abs(r.gamma - gamma) < 1e-9
                and np.isfinite(r.r2_test_clean)
            ]
            if not rs:
                continue
            cg = float(np.median([r.convergence_gen for r in rs]))
            r2 = float(np.median([r.r2_test_clean for r in rs]))
            means[(method, gamma)] = (cg, r2)

    # Connecting line per method through group means (ordered by gamma).
    for method in ("baseline", "isalsr"):
        ordered = sorted(
            [(g, *means[(method, g)]) for g in sorted(set(gammas)) if (method, g) in means]
        )
        if len(ordered) >= 2:
            xs = [p[1] for p in ordered]
            ys = [p[2] for p in ordered]
            ax_conv.plot(
                xs,
                ys,
                linestyle=method_line_style[method],
                color="0.35",
                linewidth=1.3,
                alpha=0.9,
                zorder=3,
            )

    # Group-mean markers on top.
    for (method, gamma), (cg, r2) in means.items():
        ax_conv.scatter(
            cg,
            r2,
            marker=METHOD_STYLE[method]["marker"],
            s=130,
            facecolor=cmap(norm(gamma)),
            edgecolor="black",
            linewidths=1.1,
            alpha=1.0,
            zorder=4,
        )

    ax_conv.set_xlabel("Generation at best training R$^2$ (plateau)")
    ax_conv.set_ylabel("Test R$^2$ on clean data")
    ax_conv.set_title(f"Convergence vs. recovered R$^2$ — {bench['name']}")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.axhline(0.0, color="k", linestyle=":", alpha=0.4, linewidth=0.8)

    # Adaptive y-range: zoom to the group-median envelope so that small
    # R² differences across γ are readable, while clipping pathological
    # negative-R² outliers. If medians are tightly packed (all > 0.9), we
    # zoom in; if they span a wider range, we fall back to [-0.3, 1.0].
    if means:
        med_r2 = np.array([v[1] for v in means.values()])
        lo = float(np.min(med_r2))
        hi = float(np.max(med_r2))
        if lo > 0.85:
            pad = max(0.005, 0.1 * (hi - lo) + 0.002)
            ax_conv.set_ylim(bottom=lo - pad, top=min(1.0 + pad * 0.5, 1.005))
        else:
            ax_conv.set_ylim(bottom=max(-0.3, lo - 0.1), top=1.02)
    else:
        ax_conv.set_ylim(bottom=-0.3, top=1.0)

    # Legend: shape = method, line style = method's across-γ mean line.
    method_legend = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[m]["marker"],
            linestyle=method_line_style[m],
            color="0.35",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"{METHOD_STYLE[m]['label']} (mean per γ)",
        )
        for m in ("baseline", "isalsr")
    ]
    ax_conv.legend(handles=method_legend, loc="lower left", framealpha=0.95)

    # Panel B: data scatter for each gamma (using the first seed), ground truth.
    first_seed = min({r.seed for r in results})
    for gamma in sorted(gammas):
        npz_path = data_dir / f"noise_gamma{gamma:.2f}_seed{first_seed}.npz"
        if not npz_path.exists():
            continue
        arr = np.load(npz_path)
        x = arr["X_train"].flatten()
        y = arr["y_train_noisy"].flatten()
        color = cmap(norm(gamma))
        ax_data.scatter(
            x,
            y,
            marker="o",
            s=24,
            facecolor=color,
            edgecolor="black",
            linewidths=0.3,
            alpha=0.85,
            zorder=3,
        )

    lo, hi = bench["var_ranges"][0]
    xs = np.linspace(lo, hi, 1000)
    ys = bench["target_fn"](xs)
    ax_data.plot(xs, ys, color="crimson", linewidth=1.8, label="Ground truth", zorder=2)

    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.set_title(f"Training data (seed={first_seed}) — {bench['name']}")
    ax_data.grid(True, alpha=0.3)
    ax_data.legend(loc="upper right", framealpha=0.95)

    # Shared colorbar for gamma (noise amplitude relative to std(y_clean)).
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.15,
        aspect=40,
    )
    cbar.set_label(r"Noise level $\gamma$ (std$_\mathrm{noise}$ / std$_y$)")

    fig.suptitle(
        "Gaussian noise as problem-difficulty axis — Bingo (150 gens, pop=100)",
        fontsize=12,
    )

    out_pdf = out_root / "figure.pdf"
    out_png = out_root / "figure.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    log.info("Saved figure: %s", out_pdf)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _write_summary(results: list[RunResult], path: Path) -> None:
    fields = [
        "method",
        "gamma",
        "seed",
        "convergence_gen",
        "r2_train_plateau",
        "r2_test_clean",
        "n_generations",
        "wall_clock_s",
        "best_expr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "method": r.method,
                    "gamma": f"{r.gamma:.4f}",
                    "seed": r.seed,
                    "convergence_gen": r.convergence_gen,
                    "r2_train_plateau": f"{r.r2_train_plateau:.6f}",
                    "r2_test_clean": f"{r.r2_test_clean:.6f}",
                    "n_generations": r.n_generations,
                    "wall_clock_s": f"{r.wall_clock_s:.3f}",
                    "best_expr": r.best_expr.replace("\n", " "),
                }
            )


def _write_config(
    path: Path,
    cfg: BingoConfig,
    gammas: list[float],
    seeds: list[int],
    suite_key: str,
    benchmark_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": {
            "name": "gaussian_noise_as_problem_difficulty",
            "suite": suite_key,
            "benchmark": benchmark_name,
            "gammas": gammas,
            "seeds": seeds,
        },
        "bingo": {
            "population_size": cfg.population_size,
            "stack_size": cfg.stack_size,
            "operators": list(cfg.operators),
            "use_simplification": cfg.use_simplification,
            "crossover_prob": cfg.crossover_prob,
            "mutation_prob": cfg.mutation_prob,
            "metric": cfg.metric,
            "clo_alg": cfg.clo_alg,
            "generations": cfg.generations,
            "fitness_threshold": cfg.fitness_threshold,
            "max_time": cfg.max_time,
            "max_evals": cfg.max_evals,
            "snapshot_frequency": cfg.snapshot_frequency,
            "use_fast_canonical": cfg.use_fast_canonical,
            "canonicalization_timeout": cfg.canonicalization_timeout,
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--benchmark-suite",
        type=str,
        choices=list(_SUITES.keys()),
        default="hard",
        help="Benchmark suite module.",
    )
    p.add_argument(
        "--benchmark-name",
        type=str,
        default="Vladislavleva-2",
        help="Benchmark name inside the suite (must be 1D).",
    )
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.40])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=list(METHOD_STYLE.keys()),
        default=["baseline", "isalsr"],
    )
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--stack-size", type=int, default=16)
    p.add_argument("--max-time", type=float, default=120.0)
    p.add_argument("--snapshot-frequency", type=int, default=5)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--plot-only", action="store_true", help="skip runs; re-plot from summary.csv")
    return p.parse_args()


def _load_existing_results(summary_path: Path) -> list[RunResult]:
    if not summary_path.exists():
        return []
    rows: list[RunResult] = []
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                RunResult(
                    method=r["method"],
                    gamma=float(r["gamma"]),
                    seed=int(r["seed"]),
                    convergence_gen=int(r["convergence_gen"]),
                    r2_train_plateau=float(r["r2_train_plateau"]),
                    r2_test_clean=float(r["r2_test_clean"]),
                    n_generations=int(r["n_generations"]),
                    wall_clock_s=float(r["wall_clock_s"]),
                    best_expr=r.get("best_expr", ""),
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    suite_mod = _SUITES[args.benchmark_suite]
    bench = suite_mod.get_benchmark(args.benchmark_name)
    if bench.get("num_variables", 0) != 1:
        log.error(
            "This script expects a 1D benchmark; %s has %d vars.",
            args.benchmark_name,
            bench["num_variables"],
        )
        return 1

    cfg = BingoConfig(
        population_size=args.population_size,
        stack_size=args.stack_size,
        operators=["+", "-", "*", "/", "sin", "cos", "exp", "log"],
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        generations=args.generations,
        fitness_threshold=1e-16,
        max_time=args.max_time,
        max_evals=10_000_000,
        snapshot_frequency=args.snapshot_frequency,
        use_fast_canonical=True,
        canonicalization_timeout=10.0,
    )
    _write_config(
        out_root / "config.yaml",
        cfg,
        args.gammas,
        args.seeds,
        args.benchmark_suite,
        args.benchmark_name,
    )

    summary_path = out_root / "summary.csv"

    if args.plot_only:
        results = _load_existing_results(summary_path)
        if not results:
            log.error("--plot-only but %s is empty/missing.", summary_path)
            return 1
        _plot(results, args.gammas, bench, out_root / "data", out_root)
        return 0

    results: list[RunResult] = []
    gamma_index = {g: i for i, g in enumerate(sorted(set(args.gammas)))}

    for seed in args.seeds:
        # Clean data is reproducible given seed (suite.generate_data is seeded).
        X_train, y_train_clean, X_test, y_test_clean = suite_mod.generate_data(bench, seed=seed)
        for gamma in args.gammas:
            for method in args.methods:
                res = _run_one(
                    method=method,
                    gamma=gamma,
                    seed=seed,
                    gamma_idx=gamma_index[gamma],
                    X_train=X_train,
                    y_train_clean=y_train_clean,
                    X_test=X_test,
                    y_test_clean=y_test_clean,
                    bench=bench,
                    suite_key=args.benchmark_suite,
                    cfg=cfg,
                    out_root=out_root,
                )
                results.append(res)
                # Persist summary incrementally so partial progress survives Ctrl-C.
                _write_summary(results, summary_path)

    _plot(results, args.gammas, bench, out_root / "data", out_root)
    log.info("Done. %d runs. Outputs in %s", len(results), out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
