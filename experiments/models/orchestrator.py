"""Experiment orchestrator.

Main entry point for running paired comparison experiments.
Iterates over (method, benchmark, problem, seed, variant) and
produces the full output folder structure.

Usage:
    python -m experiments.models.orchestrator \
        --config experiments/configs/udfs_nguyen.yaml \
        --output-dir /media/mpascual/Sandisk2TB/research/isalsr/results \
        --seeds 1-3 --problems Nguyen-1 --variants baseline,isalsr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.datasets.feynman import FEYNMAN_BENCHMARKS  # noqa: E402
from benchmarks.datasets.feynman import generate_data as feynman_generate_data  # noqa: E402
from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS  # noqa: E402
from benchmarks.datasets.nguyen import generate_data as nguyen_generate_data  # noqa: E402
from experiments.models.analyzer.aggregation import (  # noqa: E402
    aggregate_all_metrics,
    apply_holm_correction,
    compute_paired_stats,
)
from experiments.models.hardware_info import collect_hardware_info  # noqa: E402
from experiments.models.io_utils import (  # noqa: E402
    ensure_output_structure,
    load_all_run_logs,
    load_run_log,
    save_aggregate,
    save_metadata,
    save_paired_stats,
    save_run_log,
    save_trajectory,
    seed_dir,
)
from experiments.models.schemas import RunMetadata  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Benchmark registries: name -> (problem_list, generate_data_fn)
_BENCHMARK_REGISTRY: dict[str, tuple[list[dict[str, Any]], Any]] = {
    "nguyen": (NGUYEN_BENCHMARKS, nguyen_generate_data),
    "feynman": (FEYNMAN_BENCHMARKS, feynman_generate_data),
}


def parse_seeds(s: str) -> list[int]:
    """Parse seed specification like '1-30' or '1,5,10'."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",")]


def get_benchmarks(
    benchmark_name: str,
    problem_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Get benchmark problem list."""
    if benchmark_name not in _BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Available: {list(_BENCHMARK_REGISTRY.keys())}"
        )

    benchmarks = list(_BENCHMARK_REGISTRY[benchmark_name][0])

    if problem_filter and problem_filter != "all":
        names = [n.strip() for n in problem_filter.split(",")]
        benchmarks = [b for b in benchmarks if b["name"] in names]
        if not benchmarks:
            raise ValueError(f"No benchmarks match filter: {problem_filter}")

    return benchmarks


def _generate_benchmark_data(
    bench_name: str,
    bench: dict[str, Any],
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    """Dispatch to the correct generate_data function for the benchmark.

    Nguyen signature: generate_data(bench, n_train, n_test, seed)
    Feynman signature: generate_data(bench, n_samples, train_ratio, seed)
    """
    if bench_name == "nguyen":
        return nguyen_generate_data(
            bench,
            n_train=train_size,
            n_test=test_size,
            seed=seed,
        )
    elif bench_name == "feynman":
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return feynman_generate_data(
            bench,
            n_samples=n_samples,
            train_ratio=train_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Cannot generate data for benchmark: {bench_name}")


def create_runner(method: str, variant: str, config: dict[str, Any], atlas=None):
    """Factory for model runners.

    Runner imports are lazy: UDFS requires torch (vendored), Bingo requires
    mpi4py. These are only available on HPC or when the full [experiments]
    extras are installed.

    Args:
        method: SR method name ("udfs" or "bingo").
        variant: "baseline" or "isalsr".
        config: Full YAML config dict.
        atlas: Optional AtlasLookup for O(1) canonical lookup (isalsr only).
    """
    if method == "udfs":
        from experiments.models.udfs.config import UDFSConfig
        from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner
        from experiments.models.udfs.runner import UDFSBaselineRunner

        cfg = UDFSConfig.from_dict(config.get("udfs", {}))
        if variant == "baseline":
            return UDFSBaselineRunner(config=cfg)
        elif variant == "isalsr":
            return IsalSRUDFSRunner(config=cfg, atlas=atlas)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    elif method == "bingo":
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner
        from experiments.models.bingo.runner import BingoBaselineRunner

        cfg = BingoConfig.from_dict(config.get("bingo", {}))
        if variant == "baseline":
            return BingoBaselineRunner(config=cfg)
        elif variant == "isalsr":
            return IsalSRBingoRunner(config=cfg, atlas=atlas)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    else:
        raise ValueError(f"Unknown method: {method}")


def create_translator(method, y_train, y_test, gt_expr, gt_vars):
    """Factory for result translators.

    Translator imports are lazy for the same reason as runners.
    """
    if method == "udfs":
        from experiments.models.udfs.translator import UDFSTranslator

        return UDFSTranslator(
            y_train=y_train,
            y_test=y_test,
            ground_truth_expr=gt_expr,
            ground_truth_variables=gt_vars,
        )
    elif method == "bingo":
        from experiments.models.bingo.translator import BingoTranslator

        return BingoTranslator(
            y_train=y_train,
            y_test=y_test,
            ground_truth_expr=gt_expr,
            ground_truth_variables=gt_vars,
        )
    else:
        raise ValueError(f"Unknown method for translator: {method}")


def _resolve_atlas(atlas_dir: str | None, num_variables: int):
    """Find and load the atlas for a given num_variables.

    Scans *atlas_dir* subdirectories for ``cache_merged.h5`` files whose
    HDF5 root attribute ``num_variables`` matches. Returns ``None`` (with
    a logged warning) if no match is found or *atlas_dir* is ``None``.
    """
    if atlas_dir is None:
        return None

    atlas_path = Path(atlas_dir)
    if not atlas_path.is_dir():
        log.warning("Atlas dir does not exist: %s", atlas_dir)
        return None

    from isalsr.precomputed.atlas_lookup import AtlasLookup  # noqa: PLC0415

    # Scan subdirectories for cache_merged.h5
    for subdir in sorted(atlas_path.iterdir()):
        merged = subdir / "cache_merged.h5" if subdir.is_dir() else None
        if merged is None or not merged.exists():
            continue
        try:
            candidate = AtlasLookup.from_hdf5(merged)
            if candidate.num_variables == num_variables:
                return candidate
        except Exception:  # noqa: BLE001
            log.warning("Failed to load atlas %s", merged, exc_info=True)

    # Also check atlas_dir itself for .h5 files (flat layout)
    for h5_file in sorted(atlas_path.glob("*.h5")):
        try:
            candidate = AtlasLookup.from_hdf5(h5_file)
            if candidate.num_variables == num_variables:
                return candidate
        except Exception:  # noqa: BLE001
            log.warning("Failed to load atlas %s", h5_file, exc_info=True)

    log.warning(
        "No atlas found for num_variables=%d in %s",
        num_variables,
        atlas_dir,
    )
    return None


def run_experiment(config_path: str, args: argparse.Namespace) -> None:
    """Run the full experiment from a YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    method = exp["method"]
    n_seeds = exp.get("n_seeds", 30)

    seeds = parse_seeds(args.seeds) if args.seeds else list(range(1, n_seeds + 1))
    variants = args.variants.split(",") if args.variants else ["baseline", "isalsr"]

    output_base = Path(args.output_dir)
    hardware = collect_hardware_info()

    # Save global metadata
    save_metadata(
        {
            "config": config,
            "hardware": hardware,
            "seeds": seeds,
            "variants": variants,
        },
        output_base / "metadata.json",
    )

    # Get benchmarks
    benchmark_configs = config.get("benchmarks", {})
    for bench_name, bench_cfg in benchmark_configs.items():
        benchmarks = get_benchmarks(bench_name, args.problems)
        train_size = bench_cfg.get("train_size", 20)
        test_size = bench_cfg.get("test_size", 100)

        all_paired_stats = []

        # Resolve atlas once per (benchmark, num_variables)
        atlas_dir = getattr(args, "atlas_dir", None)
        _atlas_cache: dict[int, object] = {}

        for bench in benchmarks:
            problem_name = bench["name"]
            log.info("=== %s / %s ===", bench_name, problem_name)

            nv = bench["num_variables"]
            if nv not in _atlas_cache:
                _atlas_cache[nv] = _resolve_atlas(atlas_dir, nv)
            atlas = _atlas_cache[nv]

            paths = ensure_output_structure(output_base, method, bench_name, problem_name)

            for seed in seeds:
                for variant in variants:
                    run_key = f"{problem_name} seed={seed} variant={variant}"

                    # Check if already done (validate JSON, not just existence)
                    sd = seed_dir(paths[variant], seed)
                    run_log_path = sd / "run_log.json"
                    if run_log_path.exists():
                        try:
                            load_run_log(run_log_path)
                            log.info("  Skipping %s (already exists)", run_key)
                            continue
                        except (json.JSONDecodeError, KeyError, TypeError, OSError):
                            log.warning(
                                "  Corrupt run_log detected for %s, re-running",
                                run_key,
                            )
                            run_log_path.unlink(missing_ok=True)

                    log.info("  Running %s", run_key)

                    x_train, y_train, x_test, y_test = _generate_benchmark_data(
                        bench_name,
                        bench,
                        train_size,
                        test_size,
                        seed,
                    )

                    runner = create_runner(method, variant, config, atlas=atlas)
                    raw = runner.fit(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        seed=seed,
                        config=config.get(method, {}),
                    )

                    # Get ground truth for solution recovery
                    gt_expr = _get_ground_truth_sympy(bench)
                    gt_vars = _get_ground_truth_vars(bench)

                    translator = create_translator(
                        method,
                        y_train,
                        y_test,
                        gt_expr,
                        gt_vars,
                    )

                    metadata = RunMetadata(
                        method=method,
                        representation=variant,
                        benchmark=bench_name,
                        problem=problem_name,
                        seed=seed,
                        hardware=hardware,
                        hyperparameters=config.get(method, {}),
                    )

                    run_log = translator.to_run_log(raw, metadata)
                    trajectory = translator.to_trajectory(raw)

                    save_run_log(run_log, sd / "run_log.json")
                    save_trajectory(trajectory, sd / "trajectory.csv")

                    log.info(
                        "    R²=%.4f total_dags=%d unique=%d",
                        run_log.regression.r2_test,
                        run_log.search_space.total_dags_explored,
                        run_log.search_space.unique_canonical_dags,
                    )

            # After all seeds: aggregate + paired stats
            for variant in variants:
                logs = load_all_run_logs(paths[variant])
                if logs:
                    agg_rows = aggregate_all_metrics(logs)
                    save_aggregate(agg_rows, paths[variant] / "aggregate.csv")

            if "baseline" in variants and "isalsr" in variants:
                baseline_logs = load_all_run_logs(paths["baseline"])
                isalsr_logs = load_all_run_logs(paths["isalsr"])
                if baseline_logs and isalsr_logs:
                    paired = compute_paired_stats(baseline_logs, isalsr_logs)
                    save_paired_stats(paired, paths["problem"] / "paired_stats.json")
                    all_paired_stats.append(paired)

        # Holm correction across problems
        if all_paired_stats:
            apply_holm_correction(all_paired_stats)
            for ps in all_paired_stats:
                problem_slug = ps.problem.lower().replace("-", "_")
                ps_path = output_base / method / bench_name / problem_slug / "paired_stats.json"
                save_paired_stats(ps, ps_path)

    log.info("Experiment complete. Results in %s", output_base)


def _get_ground_truth_sympy(bench: dict[str, Any]):
    """Get ground truth as SymPy expression.

    NOTE: Currently only handles Nguyen-style benchmarks with 1-2 variables
    named x, y. For Feynman benchmarks (physics variable names, up to 3 vars),
    returns None gracefully. Solution recovery will report False for these.
    """
    try:
        import sympy

        expr_str = bench.get("expression", "")
        if not expr_str:
            return None

        # Create symbols
        nv = bench["num_variables"]
        if nv == 1:
            sympy.Symbol("x_0")
            expr_str = expr_str.replace("x", "x_0")
        elif nv == 2:
            sympy.Symbol("x_0")
            sympy.Symbol("x_1")
            expr_str = expr_str.replace("x", "x_0").replace("y", "x_1")
        else:
            # Feynman 3-variable problems: variable names don't match x/y pattern
            return None

        return sympy.sympify(expr_str)
    except Exception:  # noqa: BLE001
        return None


def _get_ground_truth_vars(bench: dict[str, Any]):
    """Get ground truth variables as SymPy symbols."""
    try:
        import sympy

        nv = bench["num_variables"]
        return [sympy.Symbol(f"x_{i}") for i in range(nv)]
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IsalSR experiment orchestrator",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalsr/results",
        help="Base output directory",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Seed specification (e.g., '1-30' or '1,5,10')",
    )
    parser.add_argument(
        "--problems",
        default=None,
        help="Problem filter (e.g., 'Nguyen-1,Nguyen-2' or 'all')",
    )
    parser.add_argument(
        "--variants",
        default="baseline,isalsr",
        help="Variants to run (e.g., 'baseline,isalsr' or 'baseline')",
    )
    parser.add_argument(
        "--atlas-dir",
        default=None,
        help="Directory with precomputed atlas HDF5 files for O(1) "
        "canonical lookup (isalsr variants only). If not set, "
        "canonicalization is computed online.",
    )
    args = parser.parse_args()
    run_experiment(args.config, args)


if __name__ == "__main__":
    main()
