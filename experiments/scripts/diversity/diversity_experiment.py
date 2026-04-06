"""Diversity Conjecture (X.7) empirical validation experiment.

Runs Bingo baseline vs IsalSR on Nguyen-1, capturing population snapshots
at sampled generations. Computes effective diversity ratio (delta), mean
pairwise Levenshtein distance, bipartite and exact GED, and WL-based
PCA features. Saves all raw data for figure generation.

Supports local execution and Picasso HPC (SLURM array jobs).

Usage:
    # Local dry-run (1 seed, fast bipartite GED)
    python -m experiments.scripts.diversity.diversity_experiment \
        --seeds 1 --snapshot-gens 0,10,50 --ged-mode bp

    # Full local run (10 seeds, bipartite GED)
    python -m experiments.scripts.diversity.diversity_experiment

    # Picasso HPC single task (decoded from SLURM_ARRAY_TASK_ID)
    python -m experiments.scripts.diversity.diversity_experiment \
        --task-id 1 --n-tasks 40 --ged-mode exact --ged-timeout 10 \
        --n-workers 8

    # Picasso HPC: all 20 seeds, exact GED, 8 CPUs
    python -m experiments.scripts.diversity.diversity_experiment \
        --seeds 20 --ged-mode exact --ged-timeout 10 --n-workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

# Bingo imports
from bingo.evaluation.evaluation import Evaluation

from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.isalsr_runner import (
    IsalSREvaluation,
    _CanonicalDeduplicator,
    purge_penalized,
)
from experiments.models.bingo.runner import build_bingo_pipeline
from experiments.scripts.diversity.diversity_metrics import (
    DiversitySummary,
    PopulationSnapshot,
    capture_population_snapshot,
    compute_delta,
)

log = logging.getLogger(__name__)

RESULTS_DIR = Path("/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/diversity")

DEFAULT_SNAPSHOT_GENS = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500]

# Available benchmarks
BENCHMARKS = {
    "Nguyen-1": {
        "formula": "x^3 + x^2 + x",
        "n_variables": 1,
    },
    "I.10.7": {
        "formula": "m0 / sqrt(1 - v^2/c^2)",
        "n_variables": 3,
    },
    "I.12.4": {
        "formula": "q1 / (4 * pi * r * c)",
        "n_variables": 3,
    },
    "I.48.20": {
        "formula": "m*c^2 / sqrt(1-(v/c)^2)",
        "n_variables": 3,
    },
}


# ======================================================================
# Benchmark data
# ======================================================================


def make_benchmark_data(name: str):
    """Generate train data for a benchmark problem.

    Returns (x_train, y_train).
    """
    if name == "Nguyen-1":
        rng = np.random.RandomState(42)
        x = rng.uniform(-1, 1, (20, 1))
        y = x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]
        return x, y
    else:
        from benchmarks.datasets.feynman import generate_data, get_benchmark

        bench = get_benchmark(name)
        x_train, y_train, _, _ = generate_data(bench, n_samples=200)
        return x_train, y_train


# ======================================================================
# Lightweight per-generation logging
# ======================================================================


def per_generation_log(population, x_train, y_train, n_workers: int = 1) -> dict:
    """Compute per-generation metrics including pairwise distance summaries.

    Computes delta, Levenshtein d_bar, bipartite GED d_bar, best R^2.
    Does NOT save full NxN matrices (those are at snapshot gens only).
    BP-GED is ~10s for N=200, Levenshtein ~200ms — tractable every generation.

    Returns dict with all summary stats needed for dense time-series curves.
    """
    from experiments.scripts.diversity.diversity_metrics import (
        _compute_best_r2,
        _dag_to_nx_data,
        _safe_agraph_to_dag,
        _safe_canonical,
        compute_bipartite_ged_matrix,
        compute_levenshtein_matrix,
    )

    dags = [_safe_agraph_to_dag(ag) for ag in population]
    n_failed = sum(1 for d in dags if d is None)
    canonical_strings = [_safe_canonical(d) for d in dags]
    delta, n_unique = compute_delta(canonical_strings)

    # Levenshtein summary (fast: ~200ms for N=200)
    lev_mat = compute_levenshtein_matrix(canonical_strings)
    n = len(population)
    upper_tri_lev = lev_mat[np.triu_indices(n, k=1)]
    d_bar_lev = float(np.mean(upper_tri_lev)) if len(upper_tri_lev) > 0 else 0.0
    diameter_lev = float(np.max(upper_tri_lev)) if len(upper_tri_lev) > 0 else 0.0

    # Bipartite GED summary (~10s for N=200 with 8 workers)
    graph_data = []
    for d in dags:
        if d is not None:
            graph_data.append(_dag_to_nx_data(d))
        else:
            graph_data.append({"n": 0, "labels": [], "in_deg": [], "out_deg": [], "edges": set()})
    bp_mat = compute_bipartite_ged_matrix(graph_data, n_workers=n_workers)
    upper_tri_bp = bp_mat[np.triu_indices(n, k=1)]
    d_bar_bp = float(np.mean(upper_tri_bp)) if len(upper_tri_bp) > 0 else 0.0
    diameter_bp = float(np.max(upper_tri_bp)) if len(upper_tri_bp) > 0 else 0.0

    fitnesses = np.array(
        [ag.fitness if hasattr(ag, "fitness") and ag.fit_set else np.inf for ag in population],
        dtype=np.float64,
    )
    best_fitness = float(np.nanmin(fitnesses)) if np.any(np.isfinite(fitnesses)) else float("inf")
    best_r2 = _compute_best_r2(fitnesses, population, x_train, y_train)

    return {
        "delta": delta,
        "n_unique": n_unique,
        "d_bar_lev": d_bar_lev,
        "d_bar_bp": d_bar_bp,
        "diameter_lev": diameter_lev,
        "diameter_bp": diameter_bp,
        "best_fitness": best_fitness,
        "best_r2": best_r2,
        "n_failed": n_failed,
    }


# ======================================================================
# Evolution with population snapshots
# ======================================================================


def run_single_variant(
    variant: str,
    seed: int,
    cfg: BingoConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    snapshot_gens: list[int],
    n_workers: int,
    max_gen: int = 500,
    trajectory_freq: int = 1,
    enforce_dedup: bool = False,
) -> tuple[list[tuple[PopulationSnapshot, DiversitySummary]], list[dict]]:
    """Run one (variant, seed) and capture population snapshots.

    Uses island.evolve() in chunks between snapshot generations instead
    of evolve_until_convergence(), so we hit exact generation counts.

    Args:
        variant: "baseline" or "isalsr".
        seed: Random seed.
        cfg: BingoConfig.
        x_train, y_train: Training data.
        snapshot_gens: Sorted list of generations to snapshot.
        n_workers: Workers for parallel GED computation.
        ged_mode: "bp" or "exact".
        ged_timeout: Per-pair timeout for exact GED (seconds).
        max_gen: Maximum generation for trajectory logging.
        trajectory_freq: Log lightweight metrics every N generations.

    Returns:
        (list of (Snapshot, Summary), list of trajectory dicts).
    """
    np.random.seed(seed)

    if variant == "isalsr":
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=60.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={
                "dedup": dedup,
                "snapshot_freq": 100_000,
                "t0": 0.0,
                "enforce_dedup": enforce_dedup,
            },
        )
        evaluation._fitness_counter = fitness_fn
    else:
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            cfg,
            evaluation_cls=Evaluation,
        )

    # Evaluate initial population (gen 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        island.evaluate_population()

    results = []
    trajectory = []
    sorted_gens = sorted(snapshot_gens)
    last_snapshot = max(sorted_gens)
    actual_max_gen = max(max_gen, last_snapshot)

    gen = 0
    snapshot_idx = 0

    while gen <= actual_max_gen:
        # Full snapshot at designated generations
        if snapshot_idx < len(sorted_gens) and gen == sorted_gens[snapshot_idx]:
            t0 = time.perf_counter()
            snap, summary = capture_population_snapshot(
                island.population,
                gen,
                variant,
                seed,
                x_train,
                y_train,
                n_workers,
            )
            dt = time.perf_counter() - t0
            results.append((snap, summary))

            log.info(
                "  [%s] seed=%d gen=%3d | delta=%.3f d_lev=%.1f d_bp=%.1f "
                "R2=%.4f unique=%d/%d (%.1fs)",
                variant,
                seed,
                gen,
                summary.delta,
                summary.d_bar_lev,
                summary.d_bar_bp,
                summary.best_r2,
                summary.n_unique_classes,
                cfg.population_size,
                dt,
            )
            snapshot_idx += 1

        elif gen % trajectory_freq == 0:
            # Per-generation metrics (includes BP-GED + Levenshtein summaries)
            lw = per_generation_log(island.population, x_train, y_train, n_workers)
            trajectory.append({"generation": gen, "variant": variant, "seed": seed, **lw})

            if gen % 10 == 0:
                log.info(
                    "  [%s] seed=%d gen=%3d | delta=%.3f d_lev=%.1f d_bp=%.1f R2=%.4f unique=%d/%d",
                    variant,
                    seed,
                    gen,
                    lw["delta"],
                    lw["d_bar_lev"],
                    lw["d_bar_bp"],
                    lw["best_r2"],
                    lw["n_unique"],
                    cfg.population_size,
                )

        # Evolve one generation
        if gen < actual_max_gen:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                island.evolve(1)
            # Purge age-penalized duplicates that survived selection.
            if variant == "isalsr" and enforce_dedup:
                purge_penalized(island.population)

        gen += 1

    return results, trajectory


# ======================================================================
# Data serialization
# ======================================================================


def save_snapshot(snap: PopulationSnapshot, output_dir: Path) -> None:
    """Save a single population snapshot as compressed npz."""
    gen_dir = output_dir / f"seed_{snap.seed:02d}" / snap.variant
    gen_dir.mkdir(parents=True, exist_ok=True)

    path = gen_dir / f"snapshot_gen{snap.generation:03d}.npz"

    # Pad command arrays to uniform shape for stacking
    max_rows = max(cmd.shape[0] for cmd in snap.command_arrays)
    padded_cmds = np.zeros((len(snap.command_arrays), max_rows, 3), dtype=np.int32)
    for i, cmd in enumerate(snap.command_arrays):
        padded_cmds[i, : cmd.shape[0]] = cmd

    # Pad constants to uniform length
    max_const = max(len(c) for c in snap.constants) if snap.constants else 0
    max_const = max(max_const, 1)
    padded_consts = np.zeros((len(snap.constants), max_const), dtype=np.float64)
    for i, c in enumerate(snap.constants):
        if len(c) > 0:
            padded_consts[i, : len(c)] = c

    save_dict = {
        "canonical_strings": np.array(snap.canonical_strings, dtype=object),
        "command_arrays": padded_cmds,
        "constants": padded_consts,
        "fitnesses": snap.fitnesses,
        "wl_features": snap.wl_features,
        "lev_distances": snap.lev_distances,
        "bp_ged_distances": snap.bp_ged_distances,
        "pca_coords": snap.pca_coords,
        "pca_explained_variance": np.array(snap.pca_explained_variance),
    }

    # Save subsampled exact GED data (values + pair indices + exact flags)
    exact_result = getattr(snap, "_exact_result", None)
    if exact_result is not None and exact_result.n_sampled > 0:
        save_dict["exact_ged_values"] = exact_result.values
        save_dict["exact_ged_pairs"] = exact_result.pair_indices
        save_dict["exact_ged_flags"] = exact_result.exact_flags

    np.savez_compressed(path, **save_dict)
    log.debug("  Saved snapshot: %s", path)


def save_summary_csv(summaries: list[DiversitySummary], output_dir: Path) -> None:
    """Save all summary rows to a flat CSV (incremental, dedup by key)."""
    path = output_dir / "summary.csv"
    fieldnames = [
        "seed",
        "variant",
        "generation",
        "delta",
        "d_bar_lev",
        "d_bar_bp",
        "d_bar_exact",
        "diameter_lev",
        "diameter_bp",
        "diameter_exact",
        "best_fitness",
        "best_r2",
        "n_unique_classes",
        "pca_var1",
        "pca_var2",
        "n_exact_pairs",
        "n_total_pairs",
        "cv_ged",
        "frac_zero_pairs",
        "d_bar_nonzero",
    ]

    existing_keys = set()
    if path.exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row["seed"]), row["variant"], int(row["generation"]))
                existing_keys.add(key)

    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for s in summaries:
            key = (s.seed, s.variant, s.generation)
            if key not in existing_keys:
                writer.writerow(
                    {
                        "seed": s.seed,
                        "variant": s.variant,
                        "generation": s.generation,
                        "delta": f"{s.delta:.6f}",
                        "d_bar_lev": f"{s.d_bar_lev:.4f}",
                        "d_bar_bp": f"{s.d_bar_bp:.4f}",
                        "d_bar_exact": f"{s.d_bar_exact:.4f}",
                        "diameter_lev": f"{s.diameter_lev:.1f}",
                        "diameter_bp": f"{s.diameter_bp:.1f}",
                        "diameter_exact": f"{s.diameter_exact:.1f}",
                        "best_fitness": f"{s.best_fitness:.8f}",
                        "best_r2": f"{s.best_r2:.6f}",
                        "n_unique_classes": s.n_unique_classes,
                        "pca_var1": f"{s.pca_var1:.6f}",
                        "pca_var2": f"{s.pca_var2:.6f}",
                        "n_exact_pairs": s.n_exact_pairs,
                        "n_total_pairs": s.n_total_pairs,
                        "cv_ged": f"{s.cv_ged:.6f}",
                        "frac_zero_pairs": f"{s.frac_zero_pairs:.6f}",
                        "d_bar_nonzero": f"{s.d_bar_nonzero:.4f}",
                    }
                )


def save_trajectory_csv(trajectory: list[dict], output_dir: Path, seed: int, variant: str) -> None:
    """Save per-generation lightweight trajectory to CSV."""
    traj_dir = output_dir / f"seed_{seed:02d}" / variant
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / "trajectory.csv"
    if not trajectory:
        return
    fieldnames = list(trajectory[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trajectory:
            writer.writerow(row)


def save_config(args: argparse.Namespace, cfg: BingoConfig, output_dir: Path) -> None:
    """Save experiment configuration for reproducibility."""
    bench_info = BENCHMARKS.get(args.benchmark, {})
    config = {
        "population_size": cfg.population_size,
        "stack_size": cfg.stack_size,
        "operators": cfg.operators,
        "crossover_prob": cfg.crossover_prob,
        "mutation_prob": cfg.mutation_prob,
        "metric": cfg.metric,
        "clo_alg": cfg.clo_alg,
        "max_time": cfg.max_time,
        "snapshot_gens": args.snapshot_gens,
        "seeds": args.seeds,
        "n_workers": args.n_workers,
        "trajectory_freq": args.trajectory_freq,
        "benchmark": args.benchmark,
        "formula": bench_info.get("formula", ""),
        "n_variables": bench_info.get("n_variables", 0),
        "experiment": "diversity_conjecture_x7",
    }
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


# ======================================================================
# Resume logic
# ======================================================================


def is_run_complete(output_dir: Path, seed: int, variant: str, snapshot_gens: list[int]) -> bool:
    """Check if a (seed, variant) run already has all snapshots saved."""
    for gen in snapshot_gens:
        path = output_dir / f"seed_{seed:02d}" / variant / f"snapshot_gen{gen:03d}.npz"
        if not path.exists():
            return False
    return True


# ======================================================================
# SLURM task-ID decoding
# ======================================================================


def decode_task_id(task_id: int, n_seeds: int) -> tuple[int, str]:
    """Decode a 1-based SLURM task ID into (seed, variant).

    Layout: tasks 1..n_seeds = baseline, n_seeds+1..2*n_seeds = isalsr.
    """
    idx = task_id - 1
    if idx < n_seeds:
        return idx, "baseline"
    else:
        return idx - n_seeds, "isalsr"


# ======================================================================
# Main
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diversity Conjecture X.7: Bingo baseline vs IsalSR"
    )
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--stack-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--benchmark",
        type=str,
        default="I.48.20",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark problem to use",
    )
    parser.add_argument(
        "--snapshot-gens",
        type=str,
        default=",".join(str(g) for g in DEFAULT_SNAPSHOT_GENS),
        help="Comma-separated generation numbers for full snapshots",
    )
    parser.add_argument("--n-workers", type=int, default=0, help="0 = cpu_count")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--single-seed", type=int, default=None, help="Run one seed only")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["both", "baseline", "isalsr"],
        default="both",
    )
    parser.add_argument("--max-time", type=float, default=3600.0, help="Max wall-clock per run (s)")

    # Per-generation trajectory logging
    parser.add_argument(
        "--trajectory-freq",
        type=int,
        default=1,
        help="Log lightweight metrics (delta, fitness) every N generations",
    )

    # SLURM array support
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_ID: 1-based, maps to (seed, variant)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Total SLURM array size (should be 2 * n_seeds)",
    )

    # Population-level dedup enforcement (IsalSR variant only)
    parser.add_argument(
        "--enforce-dedup",
        action="store_true",
        help="Enforce population-level canonical dedup (delta → 1.0)",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    snapshot_gens = sorted(int(g) for g in args.snapshot_gens.split(","))
    args.snapshot_gens = snapshot_gens
    max_gen = max(snapshot_gens)

    n_workers = args.n_workers if args.n_workers > 0 else os.cpu_count() or 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BingoConfig(
        population_size=args.pop_size,
        stack_size=args.stack_size,
        operators=["+", "-", "*", "/", "sin", "cos", "exp", "log"],
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        generations=10_000_000,
        fitness_threshold=1e-16,
        max_time=args.max_time,
        max_evals=10_000_000,
        snapshot_frequency=100_000,
    )

    # Determine (seed, variant) pairs to run
    if args.task_id is not None:
        # SLURM array mode: decode task ID into single (seed, variant)
        seed, variant = decode_task_id(args.task_id, args.seeds)
        run_pairs = [(seed, variant)]
    elif args.single_seed is not None:
        variants = ["baseline", "isalsr"] if args.variant == "both" else [args.variant]
        run_pairs = [(args.single_seed, v) for v in variants]
    else:
        seeds = list(range(args.seeds))
        variants = ["baseline", "isalsr"] if args.variant == "both" else [args.variant]
        run_pairs = [(s, v) for s in seeds for v in variants]

    log.info("=" * 72)
    log.info("Diversity Conjecture X.7 Experiment")
    log.info("=" * 72)
    log.info(
        "Benchmark: %s (%s)", args.benchmark, BENCHMARKS.get(args.benchmark, {}).get("formula", "")
    )
    log.info("Population size: %d", cfg.population_size)
    log.info("Stack size: %d", cfg.stack_size)
    log.info("Snapshot generations: %s", snapshot_gens)
    log.info("GED mode: bipartite (Riesen & Bunke 2009)")
    log.info("Trajectory freq: every %d generations", args.trajectory_freq)
    log.info("Run pairs: %d (seed, variant) combinations", len(run_pairs))
    log.info("Workers: %d", n_workers)
    log.info("Output: %s", output_dir)
    log.info("Max time per run: %.0fs", cfg.max_time)
    if args.task_id is not None:
        log.info(
            "SLURM task ID: %d -> seed=%d, variant=%s",
            args.task_id,
            run_pairs[0][0],
            run_pairs[0][1],
        )
    log.info("=" * 72)

    if args.dry_run:
        log.info("DRY RUN -- exiting")
        save_config(args, cfg, output_dir)
        return

    save_config(args, cfg, output_dir)

    # Generate training data (fixed seed=42 for reproducibility)
    x_train, y_train = make_benchmark_data(args.benchmark)
    log.info("Training data: x_train=%s, y_train=%s", x_train.shape, y_train.shape)

    total_t0 = time.perf_counter()

    for seed, variant in run_pairs:
        # Resume: skip if all snapshots exist
        if is_run_complete(output_dir, seed, variant, snapshot_gens):
            log.info("SKIP seed=%d variant=%s (already complete)", seed, variant)
            continue

        log.info("-" * 60)
        log.info("Running seed=%d variant=%s", seed, variant)
        log.info("-" * 60)

        t0 = time.perf_counter()
        results, trajectory = run_single_variant(
            variant=variant,
            seed=seed,
            cfg=cfg,
            x_train=x_train,
            y_train=y_train,
            snapshot_gens=snapshot_gens,
            n_workers=n_workers,
            max_gen=max_gen,
            trajectory_freq=args.trajectory_freq,
            enforce_dedup=args.enforce_dedup,
        )
        dt = time.perf_counter() - t0

        # Save snapshots, summaries, and trajectory
        for snap, _summary in results:
            save_snapshot(snap, output_dir)

        save_summary_csv([s for _, s in results], output_dir)
        save_trajectory_csv(trajectory, output_dir, seed, variant)
        log.info("  Completed in %.1fs (%.1f min)", dt, dt / 60)

    total_dt = time.perf_counter() - total_t0
    log.info("=" * 72)
    log.info("DONE. Total time: %.1fs (%.1f min)", total_dt, total_dt / 60)
    log.info("Results saved to: %s", output_dir)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
