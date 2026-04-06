"""Smoke test: validate delta approaches 1.0 with age penalty fix.

Runs short evolutionary experiments with different configurations and
measures delta per generation to confirm the age penalty fix eliminates
the theoretical vs empirical delta discrepancy.

Usage:
    # Run all configs (pre_fix, age_fix, legacy_age_fix)
    python -m experiments.scripts.diversity.dedup_smoke_test

    # Run a single config
    python -m experiments.scripts.diversity.dedup_smoke_test --config age_fix

    # Custom parameters
    python -m experiments.scripts.diversity.dedup_smoke_test \
        --pop-size 250 --max-gen 50 --seed 0

Results saved to:
    /media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/diversity/dedup_smoke/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from bingo.evaluation.evaluation import Evaluation

from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.isalsr_runner import (
    IsalSREvaluation,
    _CanonicalDeduplicator,
    purge_penalized,
)
from experiments.models.bingo.runner import build_bingo_pipeline
from experiments.scripts.diversity.diversity_metrics import (
    _safe_agraph_to_dag,
    _safe_canonical,
    compute_delta,
)

log = logging.getLogger(__name__)

RESULTS_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/diversity/dedup_smoke"
)


@dataclass
class SmokeConfig:
    """Configuration for one smoke test run."""

    name: str
    enforce_dedup: bool
    use_age_penalty: bool
    pop_size: int = 250
    max_gen: int = 50
    seed: int = 0
    stack_size: int = 32
    benchmark: str = "I.10.7"


@dataclass
class GenerationMetrics:
    """Per-generation metrics for the smoke test."""

    generation: int
    delta: float
    n_unique: int
    n_failed: int
    n_inf_fitness: int
    n_old_age: int
    best_fitness: float
    pop_size: int


SMOKE_CONFIGS = {
    "pre_fix": SmokeConfig(
        name="pre_fix",
        enforce_dedup=True,
        use_age_penalty=False,
    ),
    "age_fix": SmokeConfig(
        name="age_fix",
        enforce_dedup=True,
        use_age_penalty=True,
    ),
    "legacy_age_fix": SmokeConfig(
        name="legacy_age_fix",
        enforce_dedup=False,
        use_age_penalty=True,
    ),
}


def make_benchmark_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate training data for a benchmark problem."""
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


def measure_population(population: list) -> GenerationMetrics:
    """Compute diversity metrics on the current population.

    Re-computes canonical strings from scratch (independent of dedup tracker)
    to get ground-truth delta.
    """
    dags = [_safe_agraph_to_dag(ag) for ag in population]
    n_failed = sum(1 for d in dags if d is None)
    canonical_strings = [_safe_canonical(d) for d in dags]
    delta, n_unique = compute_delta(canonical_strings)

    n_inf = sum(
        1 for ag in population if hasattr(ag, "fitness") and ag.fit_set and ag.fitness == np.inf
    )
    n_old = sum(1 for ag in population if hasattr(ag, "genetic_age") and ag.genetic_age > 1000)

    fitnesses = np.array(
        [ag.fitness if hasattr(ag, "fitness") and ag.fit_set else np.inf for ag in population],
        dtype=np.float64,
    )
    best = float(np.nanmin(fitnesses)) if np.any(np.isfinite(fitnesses)) else float("inf")

    return GenerationMetrics(
        generation=-1,  # set by caller
        delta=delta,
        n_unique=n_unique,
        n_failed=n_failed,
        n_inf_fitness=n_inf,
        n_old_age=n_old,
        best_fitness=best,
        pop_size=len(population),
    )


def run_smoke_variant(
    cfg: SmokeConfig, x_train: np.ndarray, y_train: np.ndarray
) -> list[GenerationMetrics]:
    """Run one smoke test configuration and collect per-generation metrics."""
    np.random.seed(cfg.seed)

    bingo_cfg = BingoConfig(
        population_size=cfg.pop_size,
        stack_size=cfg.stack_size,
        operators=["+", "-", "*", "/", "sin", "cos", "exp", "log"],
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        generations=10_000_000,
        fitness_threshold=1e-16,
        max_time=3600.0,
        max_evals=10_000_000,
        snapshot_frequency=100_000,
    )

    if cfg.enforce_dedup or cfg.use_age_penalty:
        # IsalSR variant
        dedup = _CanonicalDeduplicator(use_fast_canonical=True, timeout=60.0)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            bingo_cfg,
            evaluation_cls=IsalSREvaluation,
            evaluation_kwargs={
                "dedup": dedup,
                "snapshot_freq": 100_000,
                "t0": 0.0,
                "enforce_dedup": cfg.enforce_dedup,
                "use_age_penalty": cfg.use_age_penalty,
            },
        )
        evaluation._fitness_counter = fitness_fn
    else:
        # Pure baseline (no IsalSR)
        island, fitness_fn, evaluation = build_bingo_pipeline(
            x_train,
            y_train,
            bingo_cfg,
            evaluation_cls=Evaluation,
        )

    # Evaluate initial population
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        island.evaluate_population()

    metrics: list[GenerationMetrics] = []

    for gen in range(cfg.max_gen + 1):
        m = measure_population(island.population)
        m.generation = gen
        metrics.append(m)

        if gen % 10 == 0:
            log.info(
                "  [%s] gen=%3d | delta=%.4f unique=%d/%d inf=%d old_age=%d failed=%d best=%.6f",
                cfg.name,
                gen,
                m.delta,
                m.n_unique,
                m.pop_size,
                m.n_inf_fitness,
                m.n_old_age,
                m.n_failed,
                m.best_fitness,
            )

        if gen < cfg.max_gen:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                island.evolve(1)
            # Purge any penalized duplicates that survived selection.
            if cfg.use_age_penalty:
                purge_penalized(island.population)

    return metrics


def save_metrics_csv(metrics: list[GenerationMetrics], path: Path) -> None:
    """Save per-generation metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GenerationMetrics.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def print_summary(all_results: dict[str, list[GenerationMetrics]]) -> None:
    """Print comparison summary across configurations."""
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(
        f"{'Config':<20} {'Gen 0 δ':>10} {'Gen 1 δ':>10} {'Gen 10 δ':>10} "
        f"{'Gen 50 δ':>10} {'Min δ':>10} {'Max δ':>10}"
    )
    print("-" * 80)

    for name, metrics in all_results.items():
        gen_map = {m.generation: m for m in metrics}
        g0 = gen_map.get(0)
        g1 = gen_map.get(1)
        g10 = gen_map.get(10)
        g50 = gen_map.get(50, gen_map.get(max(gen_map.keys())))
        min_d = min(m.delta for m in metrics)
        max_d = max(m.delta for m in metrics)

        print(
            f"{name:<20} "
            f"{g0.delta if g0 else 'N/A':>10.4f} "
            f"{g1.delta if g1 else 'N/A':>10.4f} "
            f"{g10.delta if g10 else 'N/A':>10.4f} "
            f"{g50.delta if g50 else 'N/A':>10.4f} "
            f"{min_d:>10.4f} "
            f"{max_d:>10.4f}"
        )

    print("=" * 80)

    # Check if age_fix achieves delta = 1.0 from gen 1+
    if "age_fix" in all_results:
        post_gen0 = [m for m in all_results["age_fix"] if m.generation >= 1]
        if post_gen0:
            min_delta = min(m.delta for m in post_gen0)
            if min_delta >= 0.99:
                print("PASS: age_fix achieves delta >= 0.99 from gen 1+")
            elif min_delta >= 0.95:
                print(f"PARTIAL: age_fix min delta = {min_delta:.4f} (>= 0.95 but < 0.99)")
            else:
                print(f"FAIL: age_fix min delta = {min_delta:.4f} (below 0.95)")

            # Check n_old_age (should be 0 after selection)
            max_old = max(m.n_old_age for m in post_gen0)
            if max_old == 0:
                print("PASS: No individuals with age > 1000 survive selection")
            else:
                print(f"WARNING: {max_old} individuals with age > 1000 found in population")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta fix smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(SMOKE_CONFIGS.keys()),
        help="Run only one config (default: all)",
    )
    parser.add_argument("--pop-size", type=int, default=250)
    parser.add_argument("--max-gen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", type=str, default="I.10.7")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override parameters in all configs
    configs_to_run = {}
    if args.config:
        configs_to_run[args.config] = SMOKE_CONFIGS[args.config]
    else:
        configs_to_run = dict(SMOKE_CONFIGS)

    for cfg in configs_to_run.values():
        cfg.pop_size = args.pop_size
        cfg.max_gen = args.max_gen
        cfg.seed = args.seed
        cfg.benchmark = args.benchmark

    # Generate training data
    x_train, y_train = make_benchmark_data(args.benchmark)
    log.info("Benchmark: %s, x_train=%s", args.benchmark, x_train.shape)

    all_results: dict[str, list[GenerationMetrics]] = {}

    for name, cfg in configs_to_run.items():
        log.info("=" * 60)
        log.info("Running config: %s", name)
        log.info("  enforce_dedup=%s, use_age_penalty=%s", cfg.enforce_dedup, cfg.use_age_penalty)
        log.info("  pop_size=%d, max_gen=%d, seed=%d", cfg.pop_size, cfg.max_gen, cfg.seed)
        log.info("=" * 60)

        t0 = time.perf_counter()
        metrics = run_smoke_variant(cfg, x_train, y_train)
        dt = time.perf_counter() - t0

        log.info("  Config '%s' completed in %.1fs", name, dt)

        # Save results
        csv_path = output_dir / f"{name}_seed{cfg.seed}_{cfg.benchmark}.csv"
        save_metrics_csv(metrics, csv_path)
        log.info("  Saved: %s", csv_path)

        all_results[name] = metrics

    # Save combined config
    config_path = output_dir / "smoke_test_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {name: asdict(cfg) for name, cfg in configs_to_run.items()},
            f,
            indent=2,
        )

    print_summary(all_results)


if __name__ == "__main__":
    main()
