"""Diagnostic script for debugging delta discrepancy on Picasso.

Run this ON THE CLUSTER to verify:
1. Code version: are the age penalty changes present?
2. Runtime behavior: is the age penalty actually being applied?
3. Delta measurement: does delta reach 1.0 with the fix?

Usage (on Picasso, from repo root):
    python -m experiments.scripts.diversity.picasso_diagnostic

Output written to stdout AND to a diagnostic log file.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def section(title: str) -> None:
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


def main() -> None:
    section("1. ENVIRONMENT")
    log.info("Python: %s", sys.executable)
    log.info("Version: %s", sys.version)
    log.info("CWD: %s", Path.cwd())
    log.info("PYTHONPATH: %s", sys.path[:5])

    # ---- 2. CODE VERSION CHECKS ----
    section("2. CODE VERSION CHECKS")

    # Check isalsr_runner.py
    from experiments.models.bingo import isalsr_runner

    runner_path = Path(inspect.getfile(isalsr_runner))
    log.info("isalsr_runner loaded from: %s", runner_path)

    # Check for _DUPLICATE_AGE_PENALTY
    has_penalty = hasattr(isalsr_runner, "_DUPLICATE_AGE_PENALTY")
    log.info("_DUPLICATE_AGE_PENALTY exists: %s", has_penalty)
    if has_penalty:
        log.info("_DUPLICATE_AGE_PENALTY value: %s", isalsr_runner._DUPLICATE_AGE_PENALTY)

    # Check for purge_penalized
    has_purge = hasattr(isalsr_runner, "purge_penalized")
    log.info("purge_penalized() exists: %s", has_purge)

    # Check IsalSREvaluation.__init__ signature for use_age_penalty
    sig = inspect.signature(isalsr_runner.IsalSREvaluation.__init__)
    has_age_param = "use_age_penalty" in sig.parameters
    log.info("IsalSREvaluation has use_age_penalty param: %s", has_age_param)

    # Check is_stale_dup line in _serial_eval source
    serial_src = inspect.getsource(isalsr_runner.IsalSREvaluation._serial_eval)
    has_isfinite_check = "np.isfinite" in serial_src
    has_old_inf_check = "indv.fitness == np.inf" in serial_src and "is_stale_dup" in serial_src
    log.info("_serial_eval uses np.isfinite for stale_dup: %s", has_isfinite_check)
    log.info("_serial_eval uses old == np.inf for stale_dup: %s", has_old_inf_check)

    # Hash key lines of _serial_eval to fingerprint code version
    src_hash = hashlib.md5(serial_src.encode()).hexdigest()[:12]
    log.info("_serial_eval source MD5: %s", src_hash)

    # Check diversity_experiment.py
    from experiments.scripts.diversity import diversity_experiment

    exp_path = Path(inspect.getfile(diversity_experiment))
    log.info("diversity_experiment loaded from: %s", exp_path)

    exp_src = inspect.getsource(diversity_experiment.run_single_variant)
    has_purge_call = "purge_penalized" in exp_src
    log.info("run_single_variant calls purge_penalized: %s", has_purge_call)

    # ---- VERDICT ----
    all_ok = has_penalty and has_purge and has_age_param and has_purge_call
    if all_ok:
        log.info("VERDICT: Code is UP TO DATE (all 4 checks pass)")
    else:
        log.info("VERDICT: Code is STALE — fix NOT present")
        log.info(
            "  Missing: %s",
            ", ".join(
                [
                    name
                    for name, check in [
                        ("_DUPLICATE_AGE_PENALTY", has_penalty),
                        ("purge_penalized", has_purge),
                        ("use_age_penalty param", has_age_param),
                        ("purge_penalized call", has_purge_call),
                    ]
                    if not check
                ]
            ),
        )
        log.info("")
        log.info("ACTION REQUIRED: git pull && pip install -e '.[dev]'")
        log.info("  Then re-run this diagnostic to confirm.")
        return

    # ---- 3. RUNTIME BEHAVIOR TEST ----
    section("3. RUNTIME BEHAVIOR TEST (I.10.7, N=200, 20 gens, seed=0)")

    # Generate I.10.7 data
    from benchmarks.datasets.feynman import generate_data, get_benchmark
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

    bench = get_benchmark("I.10.7")
    x_train, y_train, _, _ = generate_data(bench, n_samples=200)
    log.info("Training data: %s", x_train.shape)

    np.random.seed(0)
    cfg = BingoConfig(
        population_size=200,
        stack_size=32,
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
    )

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
            "enforce_dedup": True,
            "use_age_penalty": True,
        },
    )
    evaluation._fitness_counter = fitness_fn

    # Confirm the evaluation has the right flags
    log.info("enforce_dedup: %s", evaluation._enforce_dedup)
    log.info("use_age_penalty: %s", evaluation._use_age_penalty)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        island.evaluate_population()

    for gen in range(21):
        # Measure delta
        pop = island.population
        dags = [_safe_agraph_to_dag(ag) for ag in pop]
        n_failed = sum(1 for d in dags if d is None)
        canonicals = [_safe_canonical(d) for d in dags]
        delta, n_unique = compute_delta(canonicals)

        n_inf = sum(1 for ag in pop if ag.fit_set and not np.isfinite(ag.fitness))
        n_old = sum(1 for ag in pop if ag.genetic_age >= 1000)

        log.info(
            "gen=%2d | pop=%3d delta=%.4f unique=%d inf=%d old_age=%d failed=%d",
            gen,
            len(pop),
            delta,
            n_unique,
            n_inf,
            n_old,
            n_failed,
        )

        if gen < 20:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                island.evolve(1)
            purge_penalized(island.population)

    # ---- 4. FINAL VERDICT ----
    section("4. FINAL VERDICT")
    final_delta = delta
    if final_delta >= 0.99:
        log.info("PASS: delta = %.4f >= 0.99 at gen 20", final_delta)
        log.info("The fix is WORKING correctly on this machine.")
    else:
        log.info("FAIL: delta = %.4f < 0.99 at gen 20", final_delta)
        log.info("Something is still wrong. Check the per-generation logs above.")

    # Summary of key dedup stats
    log.info("")
    log.info("Dedup stats:")
    log.info("  n_total: %d", dedup.n_total)
    log.info("  n_unique: %d", dedup.n_unique)
    log.info("  n_skipped: %d", dedup.n_skipped)
    log.info("  n_rejected_duplicates: %d", dedup.n_rejected_duplicates)


if __name__ == "__main__":
    main()
