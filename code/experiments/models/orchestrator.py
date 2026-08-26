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
import math
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
from benchmarks.datasets.structural import STRUCTURAL_BENCHMARKS  # noqa: E402
from benchmarks.datasets.structural import (
    generate_data as structural_generate_data,  # noqa: E402
)
from experiments.models.analyzer.aggregation import (  # noqa: E402
    aggregate_all_metrics,
    apply_holm_correction,
    compute_paired_stats,
)
from experiments.models.hardware_info import collect_hardware_info, peak_rss_gb  # noqa: E402
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
from experiments.models.provenance import config_sha256, data_fingerprint  # noqa: E402
from experiments.models.schemas import PairedStats, RunMetadata  # noqa: E402
from experiments.models.status_ledger import (  # noqa: E402
    RunStatus,
    collect_status_ledger,
    write_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Benchmark registries: name -> (problem_list, generate_data_fn)
# ``cherrypicked`` is a legacy suite key kept because it is baked into the
# on-disk result tree; its problems are defined in ``benchmarks.datasets.structural``.
_BENCHMARK_REGISTRY: dict[str, tuple[list[dict[str, Any]], Any]] = {
    "nguyen": (NGUYEN_BENCHMARKS, nguyen_generate_data),
    "feynman": (FEYNMAN_BENCHMARKS, feynman_generate_data),
    "hard": (HARD_BENCHMARKS, hard_generate_data),
    "cherrypicked": (STRUCTURAL_BENCHMARKS, structural_generate_data),
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
        return structural_generate_data(
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
        variant: "baseline", "isalsr", "hash" or "nodedup".
        config: Full YAML config dict.
        atlas: Optional AtlasLookup for O(1) canonical lookup (isalsr only).
            Ignored by the "hash" arm, whose key is not the canonical hash.

    Notes:
        The "nodedup" arm is the C3 control: the IsalSR runner with its
        duplicate suppression disabled. The wrapper, the adapter conversion and
        the canonicalisation all still run, so any difference from "baseline"
        measures the wrapper's own perturbation of the search rather than the
        effect of deduplication.
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
        elif variant == "nodedup":
            return IsalSRUDFSRunner(config=cfg, atlas=atlas, dedup_enabled=False)
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
        elif variant == "nodedup":
            return IsalSRBingoRunner(config=cfg, atlas=atlas, dedup_enabled=False)
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


# Every arm the post-run block knows about.  Ordering is fixed so that
# discovery from disk is deterministic.
_ALL_ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")

# Paired contrasts as ``(reference, treatment, filename)``; the isalsr-vs-baseline
# contrast keeps the historical filename and is the only one carried into the
# across-problem Holm correction.
_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("baseline", "isalsr", "paired_stats.json"),
    ("baseline", "hash", "paired_stats_hash_vs_baseline.json"),
    ("hash", "isalsr", "paired_stats_isalsr_vs_hash.json"),
)


def _problem_paths(
    output_base: Path,
    method: str,
    benchmark: str,
    problem: str,
    variants: Sequence[str] | None,
) -> dict[str, Path] | None:
    """Resolve the problem directory and its arm subdirectories.

    Args:
        output_base: Root of the results tree.
        method: SR method name.
        benchmark: Benchmark suite key.
        problem: Problem name.
        variants: Arms to allocate directories for, or ``None`` to discover the
            arms already present on disk. Discovery is what the standalone
            aggregation job needs: it runs after all arms have landed and must
            not depend on the ``--variants`` of any single array task.

    Returns:
        A dict mapping ``"problem"`` and each arm to its directory, or ``None``
        when discovery found no directory for this problem.
    """
    if variants is not None:
        return ensure_output_structure(output_base, method, benchmark, problem, variants=variants)

    problem_dir = output_base / method / benchmark / problem.lower().replace("-", "_")
    if not problem_dir.is_dir():
        return None
    paths = {"problem": problem_dir}
    for arm in _ALL_ARMS:
        arm_dir = problem_dir / arm
        if arm_dir.is_dir():
            paths[arm] = arm_dir
    return paths


def postprocess_output_root(
    output_base: Path,
    method: str,
    config: dict[str, Any],
    problem_filter: str | None = None,
    *,
    variants: Sequence[str] | None = None,
    write_ledger: bool = True,
) -> dict[str, int]:
    """Aggregate, compare and index whatever run logs are already on disk.

    Performs the per-cell post-run block over an entire output root: one
    ``aggregate.csv`` per ``(benchmark, problem, arm)``, the three paired
    contrasts written next to the problem, the across-problem Holm correction on
    the isalsr-vs-baseline contrast, and the campaign ``status_ledger.csv``.
    Reads only run logs and status records; it never runs a search.

    Args:
        output_base: Root of the results tree to post-process.
        method: SR method name, i.e. the first path component under the root.
        config: Parsed YAML config; its ``benchmarks`` section selects the
            suites and its ``experiment.method`` is expected to equal ``method``.
        problem_filter: Comma-separated problem names, ``"all"`` or ``None`` to
            take every problem in each suite's registry.
        variants: Arms to consider. ``None`` (the standalone-job case) discovers
            the arms present on disk and skips problems with no directory, so no
            empty tree is created for a problem that was never run.
        write_ledger: Whether to rebuild ``status_ledger.csv``. Everything else
            this function writes lives under ``method/benchmark/`` and is
            therefore disjoint between configs, but the ledger is a full
            recursive walk of the *whole* root writing one shared path. Running
            one aggregation job per config then means N identical walks — waste
            when serial, a torn file when concurrent. Pass ``False`` in the
            per-config tasks and run :func:`collect_status_ledger` once, from a
            single dependent job, afterwards.

    Returns:
        Counts of the artefacts produced: ``aggregates`` (aggregate.csv files
        written), ``paired_stats`` (paired-stats JSON files written, before the
        Holm re-save) and ``ledger_rows`` (rows in the status ledger, ``0`` when
        ``write_ledger`` is ``False``).
    """
    counts = {"aggregates": 0, "paired_stats": 0, "ledger_rows": 0}

    for bench_name in config.get("benchmarks", {}):
        benchmarks = get_benchmarks(bench_name, problem_filter)
        all_paired_stats: list[PairedStats] = []

        for bench in benchmarks:
            problem_name = bench["name"]
            paths = _problem_paths(output_base, method, bench_name, problem_name, variants)
            if paths is None:
                continue
            arms = [arm for arm in _ALL_ARMS if arm in paths]

            for variant in arms:
                logs = load_all_run_logs(paths[variant])
                if logs:
                    agg_rows = aggregate_all_metrics(logs)
                    save_aggregate(agg_rows, paths[variant] / "aggregate.csv")
                    counts["aggregates"] += 1

            for ref, treat, fname in _CONTRASTS:
                if ref not in paths or treat not in paths:
                    continue
                ref_logs = load_all_run_logs(paths[ref])
                treat_logs = load_all_run_logs(paths[treat])
                if not ref_logs or not treat_logs:
                    continue
                # A paired test needs >= 3 matched seeds (aggregation.py).  The
                # pre-flight smoke runs ONE seed by design -- seed 0, deliberately
                # outside the campaign seed set -- so without this guard every
                # one of Stage C's 420 tasks would raise here, after a complete
                # and correct run, and exit non-zero.  That would fail check
                # C1.1 universally while the artefacts it certifies were all
                # present.  Producing the artefacts and computing the statistic
                # are separate obligations; only the second needs seeds.
                n_paired = len(
                    {rl.metadata.seed for rl in ref_logs} & {rl.metadata.seed for rl in treat_logs}
                )
                if n_paired < 3:
                    log.info(
                        "  Skipping %s vs %s paired stats: %d matched seed(s), need 3",
                        treat,
                        ref,
                        n_paired,
                    )
                    continue
                paired = compute_paired_stats(ref_logs, treat_logs)
                save_paired_stats(paired, paths["problem"] / fname)
                counts["paired_stats"] += 1
                if treat == "isalsr" and ref == "baseline":
                    all_paired_stats.append(paired)

        # Holm correction across problems
        if all_paired_stats:
            apply_holm_correction(all_paired_stats)
            for ps in all_paired_stats:
                problem_slug = ps.problem.lower().replace("-", "_")
                ps_path = output_base / method / bench_name / problem_slug / "paired_stats.json"
                save_paired_stats(ps, ps_path)

    # P4: assemble the campaign status ledger from the per-run records.
    if write_ledger:
        counts["ledger_rows"] = write_status_ledger(output_base)
    return counts


def write_status_ledger(output_base: Path) -> int:
    """Rebuild ``status_ledger.csv`` from the per-run status records (P4).

    A full recursive walk of the whole output root. Kept separate from
    :func:`postprocess_output_root` so that a per-config aggregation array can
    skip it and one dependent job can run it exactly once.

    Args:
        output_base: Root of the results tree.

    Returns:
        Number of ledger rows written.
    """
    ledger_rows = collect_status_ledger(output_base, output_base / "status_ledger.csv")
    n_killed = sum(1 for r in ledger_rows if r.terminal_status == "started")
    log.info(
        "Status ledger: %d rows (%d completed, %d failed, %d killed) -> %s",
        len(ledger_rows),
        sum(1 for r in ledger_rows if r.terminal_status == "completed"),
        sum(1 for r in ledger_rows if r.terminal_status == "failed"),
        n_killed,
        output_base / "status_ledger.csv",
    )
    return len(ledger_rows)


def run_experiment(config_path: str, args: argparse.Namespace) -> int:
    """Run the full experiment from a YAML config.

    Returns:
        ``0`` if every cell completed, ``1`` if any cell's search raised. The
        code is propagated to the process exit status so a SLURM task that
        recorded a failure in the status ledger does not also report success
        (EXECUTION-PLAN P4).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mode = str(getattr(args, "postprocess", "auto") or "auto")
    if mode == "only":
        # No search, no data generation, no runner: just the post-run block over
        # whatever the arrays already wrote.
        output_root = Path(args.output_dir)
        log.info("Post-processing only over %s", output_root)
        try:
            counts = postprocess_output_root(
                output_root,
                config["experiment"]["method"],
                config,
                problem_filter=getattr(args, "problems", None),
                write_ledger=not bool(getattr(args, "no_status_ledger", False)),
            )
        except Exception:  # noqa: BLE001 -- reported through the exit code
            log.exception("Post-processing failed for %s", output_root)
            return 1
        log.info(
            "Post-processing complete: %d aggregate.csv, %d paired-stats file(s), %d ledger row(s)",
            counts["aggregates"],
            counts["paired_stats"],
            counts["ledger_rows"],
        )
        return 0

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

    _configure_ledger(args, variants)

    output_base = Path(args.output_dir)
    hardware = collect_hardware_info()
    cfg_sha = config_sha256(config_path)

    # Cells whose search raised.  Reported through the exit code so a SLURM
    # task that recorded a failure in the ledger does not also report success.
    n_failed_cells = 0

    # Save global metadata
    save_metadata(
        {
            "config": config,
            "config_sha256": cfg_sha,
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

                    # P3: commit to the data this run actually received, before
                    # any arm touches it.  Check C4 compares this string across
                    # the three arms of the cell.
                    fingerprint = data_fingerprint(x_train, y_train, x_test, y_test)

                    # P4, write-ahead: the record exists BEFORE the search, so an
                    # OOM SIGKILL -- which no handler observes, and which caused
                    # 36 of C1's 45 missing cells -- still leaves a row reading
                    # "started" rather than leaving a silent hole.
                    status = RunStatus(
                        method=method,
                        arm=variant,
                        benchmark=bench_name,
                        problem=problem_name,
                        seed=seed,
                        node_cpu_model=str(hardware.get("cpu_model", "")),
                        hostname=str(hardware.get("hostname", "")),
                        engine=str(hardware.get("engine", "")),
                        git_commit=str(hardware.get("git_hash", "")),
                        config_sha256=cfg_sha,
                        data_fingerprint=fingerprint,
                        slurm_job_id=str(hardware.get("slurm_job_id") or ""),
                        slurm_array_task_id=str(hardware.get("slurm_array_task_id") or ""),
                        started_at=datetime.now(tz=UTC).isoformat(),
                    )
                    write_status(status, sd)
                    started_at = time.monotonic()

                    try:
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
                            data_fingerprint=fingerprint,
                            config_sha256=cfg_sha,
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

                        # T06 reachability ledger: the five paths on which a
                        # candidate is evaluated without canonical dedup, plus
                        # their sampled denominator.  Attached here for the same
                        # reason as the shadow counts, and persisted because the
                        # population they describe exists only while the search
                        # runs -- no post-hoc pass can recover them (C1.9).
                        ledger = getattr(runner, "last_ledger", None)
                        if ledger is not None:
                            run_log = dataclasses.replace(
                                run_log,
                                search_space=dataclasses.replace(
                                    run_log.search_space,
                                    **ledger.to_search_space_fields(),
                                ),
                            )
                            save_metadata(ledger.to_dict(), sd / "fallback_ledger.json")

                        # T19 explored-DAG structural telemetry.  Attached here
                        # for the same reason as the two blocks above, and for
                        # the same reason persisted: the DAG population being
                        # described exists only while the search runs.  The flat
                        # scalars go into the run log so the analyzer's scalar
                        # METRIC_EXTRACTORS can reach them; the exact histograms
                        # go to a sidecar so the schema does not carry an array.
                        complexity = getattr(runner, "last_complexity", None)
                        if complexity is not None:
                            run_log = dataclasses.replace(
                                run_log,
                                search_space=dataclasses.replace(
                                    run_log.search_space,
                                    **complexity.scalars(),
                                ),
                            )
                            save_metadata(complexity.sidecar(), sd / "complexity.json")

                        save_run_log(run_log, sd / "run_log.json")
                        save_trajectory(trajectory, sd / "trajectory.csv")

                        # Dense per-generation convergence log (Bingo only)
                        if method == "bingo" and hasattr(translator, "save_convergence_log"):
                            translator.save_convergence_log(raw, sd / "convergence_log.npz")

                        nan_fields = _nonfinite_metric_fields(run_log)
                        status.terminal_status = "completed"
                        status.exit_code = 0
                        status.n_nan_metrics = len(nan_fields)
                        status.nan_fields = ";".join(nan_fields)

                        log.info(
                            "    R²=%.4f total_dags=%d unique=%d",
                            run_log.regression.r2_test,
                            run_log.search_space.total_dags_explored,
                            run_log.search_space.unique_canonical_dags,
                        )
                    except Exception as exc:  # noqa: BLE001 -- the ledger is the point
                        # §5.5 admits no third state: this cell now carries a
                        # named cause instead of becoming an unexplained gap.
                        # The failure is re-reported through the process exit
                        # code at the end of main(), so SLURM still sees it.
                        n_failed_cells += 1
                        status.terminal_status = "failed"
                        status.exit_code = 1
                        status.exception_class = type(exc).__name__
                        status.exception_message = str(exc)[:2000]
                        log.exception("  FAILED %s", run_key)
                    finally:
                        status.wall_clock_s = round(time.monotonic() - started_at, 3)
                        status.max_rss_gb = peak_rss_gb()
                        status.finished_at = datetime.now(tz=UTC).isoformat()
                        write_status(status, sd)

    # After every cell: aggregate, the three paired contrasts, the across-problem
    # Holm correction and the status ledger.  Suppressed under ``--postprocess
    # skip`` because inside a SLURM array every task would rewrite the same
    # shared files concurrently and re-walk the whole output tree.
    if mode == "skip":
        log.info(
            "Post-processing suppressed (--postprocess skip): no aggregate.csv, "
            "paired stats or status_ledger.csv written. Run --postprocess only "
            "over %s once every array has drained.",
            output_base,
        )
    else:
        postprocess_output_root(
            output_base, method, config, problem_filter=args.problems, variants=variants
        )

    log.info("Experiment complete. Results in %s", output_base)
    if n_failed_cells:
        log.error("%d cell(s) failed; see status_ledger.csv", n_failed_cells)
        return 1
    return 0


def _configure_ledger(args: argparse.Namespace, variants: list[str]) -> None:
    """Set the T06 ledger environment before any runner constructs one.

    ``FallbackLedger`` reads ``ISALSR_LEDGER_ENABLED`` at construction time, and
    the runners construct one inside ``fit``, so setting the variable here
    reaches every run. Routing it through a CLI flag rather than leaving it as
    an ambient environment variable makes the choice an auditable launch
    parameter: it is recorded in ``run_log.json`` as ``ledger_enabled`` and can
    be checked against the submission command afterwards.

    Warns loudly when a deduplicating arm runs with the counters off. That
    combination is the SP-6 trap in its exact form -- the run log then reports
    five rates of zero, which reads as "no fallbacks occurred" and actually
    means "nothing was counted". The distinction cannot be recovered later.
    """
    if getattr(args, "ledger", False):
        os.environ["ISALSR_LEDGER_ENABLED"] = "1"
    if getattr(args, "ledger_sample_rate", None):
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = str(args.ledger_sample_rate)

    enabled = os.environ.get("ISALSR_LEDGER_ENABLED", "0").strip() not in (
        "",
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    # "nodedup" suppresses nothing but still converts and canonicalises every
    # candidate, so it produces the same ledger evidence and needs the same flag.
    dedup_arms = [v for v in variants if v in ("isalsr", "hash", "nodedup")]
    if dedup_arms and not enabled:
        log.warning(
            "T06 fallback ledger is DISABLED for arm(s) %s. The five "
            "reachability rates will be recorded as zero and are NOT "
            "recoverable after the run. Pass --ledger to enable them.",
            ",".join(dedup_arms),
        )
    elif dedup_arms:
        log.info(
            "T06 fallback ledger ENABLED (sample rate %s) for arm(s) %s",
            os.environ.get("ISALSR_LEDGER_SAMPLE_RATE", "1"),
            ",".join(dedup_arms),
        )


def _nonfinite_metric_fields(run_log: Any) -> list[str]:
    """Return the names of regression metrics that are NaN or infinite.

    Feeds the status ledger's ``n_nan_metrics``/``nan_fields`` columns. Since
    T08 the runtime scores an expression undefined on part of the evaluation set
    as ``R2 = 0`` / ``NRMSE = 1`` and counts it in
    ``n_nonfinite_test_predictions``, so a NaN metric is once again an
    unambiguous defect signal rather than an expected extrapolation outcome --
    which is exactly why it is worth counting per run instead of discovering it
    during analysis (check C1.3).
    """
    offenders: list[str] = []
    for name in ("r2_train", "r2_test", "nrmse_train", "nrmse_test", "mse_test"):
        value = getattr(run_log.regression, name, None)
        if isinstance(value, float) and not math.isfinite(value):
            offenders.append(name)
    return offenders


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
        default=None,
        help="Path to YAML experiment config. Required for every mode except "
        "--postprocess ledger, which reads no config at all.",
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
            "(no wrapper), 'isalsr' (canonical-string dedup), 'hash' "
            "(naive fixed-order-serialisation dedup) or 'nodedup' (C3 control: "
            "wrapper and canonicalisation installed, suppression disabled)."
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
    parser.add_argument(
        "--postprocess",
        choices=("auto", "skip", "only", "ledger"),
        default="auto",
        help="When to run the post-run block (aggregate.csv, the three paired "
        "contrasts, the Holm correction, status_ledger.csv). Inside a SLURM "
        "array every task writes the same shared aggregate.csv/paired_stats.json "
        "and re-walks the whole output tree to rebuild one status_ledger.csv, "
        "concurrently, which tears those files and makes the contrasts depend on "
        "which arms happen to have landed. Use 'skip' in the array tasks and a "
        "single dependent 'only' job (no search, no data generation) over the "
        "whole output root afterwards; 'auto' (default) keeps the in-process "
        "behaviour for local runs. 'ledger' rebuilds status_ledger.csv alone, "
        "reading no config -- the once-per-root half of a per-config "
        "aggregation array.",
    )
    parser.add_argument(
        "--no-status-ledger",
        action="store_true",
        help="With --postprocess only: skip rebuilding status_ledger.csv. "
        "Everything else the step writes lives under method/benchmark/ and is "
        "disjoint between configs, but the ledger walks the WHOLE root and "
        "writes one shared path, so running one aggregation job per config "
        "would repeat the walk N times and race on the file. Pair this with a "
        "single dependent '--postprocess ledger' job.",
    )
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="Enable the T06 reachability/fallback counters for the dedup arms "
        "(sets ISALSR_LEDGER_ENABLED=1). The five fallback rates are the "
        "evidence base for R1.2 and CANNOT be recovered after the fact: the "
        "population they describe exists only while a search runs. The "
        "underlying env var defaults to OFF, so a campaign launched without "
        "this flag records zeros everywhere.",
    )
    parser.add_argument(
        "--ledger-sample-rate",
        type=int,
        default=None,
        help="Sampling rate for the two O(V+E) reachability counters "
        "(sets ISALSR_LEDGER_SAMPLE_RATE; 1 = every candidate). The other "
        "three paths are counted at full rate regardless.",
    )
    return parser


def main() -> int:
    """Parse command-line arguments and run the experiment."""
    parser = build_parser()
    args = parser.parse_args()

    if args.postprocess == "ledger":
        # The once-per-root half of a per-config aggregation array. Reads no
        # config: the ledger is assembled from status.json records, which carry
        # their own provenance.
        try:
            n_rows = write_status_ledger(Path(args.output_dir))
        except Exception:  # noqa: BLE001 -- reported through the exit code
            log.exception("Status ledger failed for %s", args.output_dir)
            return 1
        return 0 if n_rows else 1

    if not args.config:
        parser.error("--config is required unless --postprocess ledger is given")
    return run_experiment(args.config, args)


if __name__ == "__main__":
    sys.exit(main())
