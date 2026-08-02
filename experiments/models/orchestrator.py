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
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.datasets.cherrypicked import CHERRYPICKED_BENCHMARKS  # noqa: E402
from benchmarks.datasets.cherrypicked import (
    generate_data as cherrypicked_generate_data,  # noqa: E402
)
from benchmarks.datasets.feynman import FEYNMAN_BENCHMARKS  # noqa: E402
from benchmarks.datasets.feynman import generate_data as feynman_generate_data  # noqa: E402
from benchmarks.datasets.feynman_remainder import FEYNMAN_REMAINDER_BENCHMARKS  # noqa: E402
from benchmarks.datasets.feynman_remainder import (  # noqa: E402
    generate_data as feynman_remainder_generate_data,
)
from benchmarks.datasets.hard import HARD_BENCHMARKS  # noqa: E402
from benchmarks.datasets.hard import generate_data as hard_generate_data  # noqa: E402
from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS  # noqa: E402
from benchmarks.datasets.nguyen import generate_data as nguyen_generate_data  # noqa: E402
from benchmarks.datasets.roundoff import ROUNDOFF_BENCHMARKS  # noqa: E402
from benchmarks.datasets.roundoff import generate_data as roundoff_generate_data  # noqa: E402
from benchmarks.datasets.strogatz import STROGATZ_BENCHMARKS  # noqa: E402
from benchmarks.datasets.strogatz import generate_data as strogatz_generate_data  # noqa: E402
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
    "hard": (HARD_BENCHMARKS, hard_generate_data),
    "cherrypicked": (CHERRYPICKED_BENCHMARKS, cherrypicked_generate_data),
    "roundoff": (ROUNDOFF_BENCHMARKS, roundoff_generate_data),
    # D2, the R3.1 extension (T05): ODE-Strogatz plus the pre-registered
    # AI Feynman remainder.  See docs/md_files/changes/r31_extension_selection.md
    "strogatz": (STROGATZ_BENCHMARKS, strogatz_generate_data),
    "feynman_remainder": (FEYNMAN_REMAINDER_BENCHMARKS, feynman_remainder_generate_data),
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
    elif bench_name == "hard":
        # Same signature as feynman; per-problem ``sampling`` key inside
        # the bench dict overrides train/test sizes for grid-based protocols.
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return hard_generate_data(
            bench,
            n_samples=n_samples,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif bench_name == "cherrypicked":
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return cherrypicked_generate_data(
            bench,
            n_samples=n_samples,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif bench_name == "roundoff":
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return roundoff_generate_data(
            bench,
            n_samples=n_samples,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif bench_name == "feynman_remainder":
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return feynman_remainder_generate_data(
            bench,
            n_samples=n_samples,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif bench_name == "strogatz":
        # Published fixed datasets (PMLB / ODE-Strogatz): the per-problem
        # ``sampling`` overrides pin 300/100 and the seed selects the split,
        # so ``train_size``/``test_size`` from the YAML are advisory only.
        n_samples = train_size + test_size
        train_ratio = train_size / n_samples
        return strogatz_generate_data(
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
        variant: "baseline", "isalsr" or "hash".
        config: Full YAML config dict.
        atlas: Optional AtlasLookup for O(1) canonical lookup (isalsr only).
            Ignored by the "hash" arm, whose key is not the canonical hash.
    """
    if method == "udfs":
        from experiments.models.udfs.config import UDFSConfig
        from experiments.models.udfs.isalsr_runner import HashUDFSRunner, IsalSRUDFSRunner
        from experiments.models.udfs.runner import UDFSBaselineRunner

        cfg = UDFSConfig.from_dict(config.get("udfs", {}))
        if variant == "baseline":
            return UDFSBaselineRunner(config=cfg)
        elif variant == "isalsr":
            return IsalSRUDFSRunner(config=cfg, atlas=atlas)
        elif variant == "hash":
            return HashUDFSRunner(config=cfg, atlas=atlas)
        else:
            raise ValueError(f"Unknown variant: {variant}")
    elif method == "bingo":
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.bingo.isalsr_runner import HashBingoRunner, IsalSRBingoRunner
        from experiments.models.bingo.runner import BingoBaselineRunner

        cfg = BingoConfig.from_dict(config.get("bingo", {}))
        if variant == "baseline":
            return BingoBaselineRunner(config=cfg)
        elif variant == "isalsr":
            return IsalSRBingoRunner(config=cfg, atlas=atlas)
        elif variant == "hash":
            return HashBingoRunner(config=cfg, atlas=atlas)
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


def _resolve_atlas(
    atlas_dir: str | None,
    benchmark_name: str,
    num_variables: int,
):
    """Find and load the atlas for a given (benchmark, num_variables).

    Looks for subdirectories matching ``generate_cache_{benchmark}_{n}var``
    containing a ``cache_merged.h5`` file. Falls back to matching by
    ``num_variables`` alone if no benchmark-specific dir exists.

    Returns ``None`` (with a logged warning) if no match is found.
    """
    if atlas_dir is None:
        return None

    atlas_path = Path(atlas_dir)
    if not atlas_path.is_dir():
        log.warning("Atlas dir does not exist: %s", atlas_dir)
        return None

    from isalsr.precomputed.atlas_lookup import AtlasLookup  # noqa: PLC0415

    # Try exact match: generate_cache_{benchmark}_{n}var/cache_merged.h5
    exact = atlas_path / f"generate_cache_{benchmark_name}_{num_variables}var"
    merged = exact / "cache_merged.h5"
    if merged.exists():
        try:
            return AtlasLookup.from_hdf5(merged)
        except Exception:  # noqa: BLE001
            log.warning("Failed to load atlas %s", merged, exc_info=True)

    # Fallback: scan all subdirs, match by num_variables
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

    log.warning(
        "No atlas found for %s/%dvar in %s",
        benchmark_name,
        num_variables,
        atlas_dir,
    )
    return None


def apply_cli_overrides(
    config: dict[str, Any],
    max_time: float | None,
    no_shadow_hash: bool,
) -> dict[str, Any]:
    """Apply command-line overrides to the method section of a loaded config.

    Both overrides are optional and inert when unset, so an invocation without
    them reproduces the YAML exactly. The overrides are written into
    ``config[method]`` because every runner rebuilds its own dataclass config
    from that dict inside ``fit`` (``BingoConfig.from_dict`` /
    ``UDFSConfig.from_dict``), so the value reaches the host's own budget
    (Bingo's ``evolve_until_convergence(max_time=...)``, UDFS's
    ``DAGRegressor(max_time=...)``) rather than a wrapper timer. Writing it
    there also makes the deviation visible in ``metadata.json`` and in each
    ``run_log.json``'s ``hyperparameters``.

    Args:
        config: Parsed YAML config. Not mutated; a shallow copy with a fresh
            method section is returned.
        max_time: Wall-clock budget in seconds to substitute for the config's
            ``max_time``, or ``None`` to keep the config value.
        no_shadow_hash: If ``True``, set ``shadow_hash: false`` in the method
            section, disabling the fixed-order shadow cardinality counters. If
            ``False``, the key is left absent so the runner's own default
            (on for the canonical arm) applies.

    Returns:
        A new config dict with the overrides applied.

    Raises:
        KeyError: If ``config["experiment"]["method"]`` is missing.
    """
    if max_time is None and not no_shadow_hash:
        return config

    method = config["experiment"]["method"]
    section = dict(config.get(method) or {})

    if max_time is not None:
        log.info("CLI override: %s.max_time = %.1f s (config value overridden)", method, max_time)
        section["max_time"] = float(max_time)
    if no_shadow_hash:
        log.info("CLI override: %s.shadow_hash = False (shadow counters disabled)", method)
        section["shadow_hash"] = False

    return {**config, method: section}


def _positive_float(value: str) -> float:
    """Parse a strictly positive float for argparse.

    Args:
        value: Raw command-line token.

    Returns:
        The parsed value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive float.
    """
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return parsed


def run_experiment(config_path: str, args: argparse.Namespace) -> None:
    """Run the full experiment from a YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = apply_cli_overrides(
        config,
        max_time=getattr(args, "max_time", None),
        no_shadow_hash=bool(getattr(args, "no_shadow_hash", False)),
    )

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

        # Resolve atlas once per (benchmark_name, num_variables)
        atlas_dir = getattr(args, "atlas_dir", None)
        _atlas_cache: dict[tuple[str, int], object] = {}

        for bench in benchmarks:
            problem_name = bench["name"]
            log.info("=== %s / %s ===", bench_name, problem_name)

            nv = bench["num_variables"]
            cache_key = (bench_name, nv)
            if cache_key not in _atlas_cache:
                _atlas_cache[cache_key] = _resolve_atlas(atlas_dir, bench_name, nv)
            atlas = _atlas_cache[cache_key]

            paths = ensure_output_structure(
                output_base, method, bench_name, problem_name, variants=variants
            )

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

                    # Shadow fixed-order cardinality estimates, if the arm
                    # collected them.  Attached here rather than in the
                    # translator so the two host translators stay identical.
                    shadow = getattr(runner, "last_shadow", None)
                    if shadow:
                        run_log = dataclasses.replace(
                            run_log,
                            search_space=dataclasses.replace(run_log.search_space, **shadow),
                        )

                    save_run_log(run_log, sd / "run_log.json")
                    save_trajectory(trajectory, sd / "trajectory.csv")

                    # Dense per-generation convergence log (Bingo only)
                    if method == "bingo" and hasattr(translator, "save_convergence_log"):
                        translator.save_convergence_log(raw, sd / "convergence_log.npz")

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

            # Paired contrasts.  (reference, treatment, filename); the
            # isalsr-vs-baseline contrast keeps the historical filename and is
            # the only one carried into the across-problem Holm correction.
            contrasts = (
                ("baseline", "isalsr", "paired_stats.json"),
                ("baseline", "hash", "paired_stats_hash_vs_baseline.json"),
                ("hash", "isalsr", "paired_stats_isalsr_vs_hash.json"),
            )
            for ref, treat, fname in contrasts:
                if ref not in variants or treat not in variants:
                    continue
                ref_logs = load_all_run_logs(paths[ref])
                treat_logs = load_all_run_logs(paths[treat])
                if not ref_logs or not treat_logs:
                    continue
                paired = compute_paired_stats(ref_logs, treat_logs)
                save_paired_stats(paired, paths["problem"] / fname)
                if treat == "isalsr" and ref == "baseline":
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

    Resolution order:
        1. ``bench["sympy_expression"]`` — pre-built SymPy Expr (preferred,
           used by the hard-tier benchmarks; covers any variable count).
        2. String-parse ``bench["expression"]`` for 1-2 variable Nguyen-style
           problems (replacing ``x`` -> ``x_0`` and ``y`` -> ``x_1``).
        3. Returns None for Feynman 3+ variable problems with physics
           variable names (solution_recovery will report False for these).
    """
    expr = bench.get("sympy_expression")
    if expr is not None:
        return expr

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
    """Get ground truth variables as SymPy symbols.

    Prefers ``bench["sympy_variables"]`` (used by hard-tier benchmarks);
    falls back to constructing ``[x_0, ..., x_{nv-1}]`` for Nguyen/Feynman.
    """
    syms = bench.get("sympy_variables")
    if syms is not None:
        return list(syms)

    try:
        import sympy

        nv = bench["num_variables"]
        return [sympy.Symbol(f"x_{i}") for i in range(nv)]
    except Exception:  # noqa: BLE001
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator's command-line parser.

    Returns:
        The configured argument parser.
    """
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
        help=(
            "Search variants to run, comma-separated. One of 'baseline' "
            "(no dedup), 'isalsr' (canonical-string dedup) or 'hash' "
            "(naive fixed-order-serialisation dedup)."
        ),
    )
    parser.add_argument(
        "--atlas-dir",
        default=None,
        help="Directory with precomputed atlas HDF5 files for O(1) "
        "canonical lookup (isalsr variants only). If not set, "
        "canonicalization is computed online.",
    )
    parser.add_argument(
        "--max-time",
        type=_positive_float,
        default=None,
        help="Override the config's per-run wall-clock budget, in seconds. "
        "Reaches the host search's own budget (Bingo "
        "evolve_until_convergence(max_time=...), UDFS DAGRegressor(max_time=...)). "
        "If not set, the YAML value is used.",
    )
    parser.add_argument(
        "--no-shadow-hash",
        action="store_true",
        help="Disable the fixed-order shadow cardinality counters "
        "(shadow_distinct_* stay unset in run_log.json). If not set, "
        "shadow counting keeps its per-arm default (on for the isalsr arm).",
    )
    return parser


def main() -> None:
    """Parse command-line arguments and run the experiment."""
    args = build_parser().parse_args()
    run_experiment(args.config, args)


if __name__ == "__main__":
    main()
