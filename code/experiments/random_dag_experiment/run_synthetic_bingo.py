"""Small-budget Bingo run on synthetic DAG-derived target functions.

For each chosen cell of the random-DAG screening grid:
  1. Generate the DAG (collapse + var-coverage via generate_dags.generate_one).
  2. Convert DAG -> SymPy expression -> numpy callable via lambdify.
  3. Sample (X_train, y_train) and (X_test, y_test) uniformly.
  4. Run BingoBaselineRunner + IsalSRBingoRunner with a low time budget.
  5. Persist results: search-trajectory RF (IsalSR only), wall-clock,
     evals, best fitness, R^2 on test, recovered SymPy expression.

This is a *sanity check* only -- not a production experiment. Use it
to verify the pipeline end-to-end on synthetic data before requesting
Picasso allocation. Total wall-clock target: <60 min.

Usage
-----
    python -m experiments.random_dag_experiment.run_synthetic_bingo \\
        --output-dir experiments/random_dag_experiment/outputs/bingo_synthetic \\
        --max-time 45 --seeds 2 --cells 5 6 9
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.models.bingo.config import BingoConfig
from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner
from experiments.models.bingo.runner import BingoBaselineRunner
from experiments.random_dag_experiment.generate_dags import (
    GRID,
    HARD_GRID,
    GridCell,
    dag_to_formula,
    generate_one,
)

_GRIDS = {"default": GRID, "hard": HARD_GRID}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("synthetic_bingo")


# Bingo operator string set per op_set (matches generate_dags.OP_SETS).
BINGO_OPS_BY_SET: dict[str, list[str]] = {
    "poly": ["+", "*", "-"],
    "poly_trig": ["+", "*", "-", "sin", "cos"],
    "full": ["+", "*", "-", "sin", "cos", "exp", "log"],
}


@dataclass
class SyntheticRunResult:
    """One Bingo run on one cell."""

    cell_id: int
    k_requested: int
    op_set: str
    formula: str
    variant: str
    seed: int
    max_time: float
    protocol: str  # "interp" or "extrap"
    train_bound: float
    test_bound: float
    wall_clock_s: float
    n_generations: int
    total_evals: int
    best_fitness: float
    r2_train: float
    r2_test: float
    solution_recovered: bool
    recovered_sympy: str
    # IsalSR-only:
    n_total_dags: int
    n_unique_canonical: int
    n_skipped: int
    rf_search_trajectory: float


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, NaN-safe."""
    if y_pred.size == 0 or np.any(np.isnan(y_pred)):
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _make_data(
    cell: GridCell,
    formula: str,
    n_train: int,
    n_test: int,
    seed: int,
    protocol: str = "interp",
    train_bound: float = 1.0,
    test_bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample inputs, evaluate the SymPy expression, return (X_tr, y_tr, X_te, y_te).

    Protocols
    ---------
    interp:
        Train and test drawn iid from Uniform([-train_bound, +train_bound])^m.
        Standard SR-benchmark convention (Nguyen, Feynman, SRBench).
    extrap:
        Train drawn from Uniform([-train_bound, +train_bound])^m.
        Test drawn from Uniform([-test_bound, +test_bound])^m, then any sample
        whose every coordinate falls inside the train hypercube is rejected
        and re-sampled. Guarantees test points lie at least partly outside
        the training support.
    """
    import sympy as sp

    expr = sp.sympify(formula)
    syms = [sp.Symbol(f"x_{i}") for i in range(cell.m)]
    fn = sp.lambdify(syms, expr, modules="numpy")

    rng = np.random.default_rng(seed)
    X_tr = rng.uniform(-train_bound, train_bound, size=(n_train, cell.m))

    if protocol == "interp":
        X_te = rng.uniform(-train_bound, train_bound, size=(n_test, cell.m))
    elif protocol == "extrap":
        if test_bound <= train_bound:
            raise ValueError(
                f"extrap requires test_bound>train_bound; got {test_bound}<={train_bound}"
            )
        X_te = np.empty((0, cell.m))
        # Rejection sampling: keep only points with at least one coordinate
        # outside the train hypercube.
        while X_te.shape[0] < n_test:
            cand = rng.uniform(-test_bound, test_bound, size=(n_test * 2, cell.m))
            outside = np.any(np.abs(cand) > train_bound, axis=1)
            X_te = np.vstack([X_te, cand[outside]])
        X_te = X_te[:n_test]
    else:
        raise ValueError(f"Unknown protocol '{protocol}'")

    def _eval(X: np.ndarray) -> np.ndarray:
        if cell.m == 1:
            y = fn(X[:, 0])
        else:
            y = fn(*X.T)
        # lambdify returns a Python scalar when expr is constant, even on
        # an array input. Broadcast back to the input length.
        y = np.asarray(y, dtype=np.float64)
        if y.shape == ():
            y = np.full(X.shape[0], float(y))
        return y.reshape(-1)

    y_tr = _eval(X_tr)
    y_te = _eval(X_te)
    if np.var(y_tr) < 1e-30:
        raise ValueError(
            f"cell {cell.cell_id}: target is (numerically) constant "
            f"(var(y_train)={np.var(y_tr):.3e}); skip"
        )
    mask_tr = np.isfinite(y_tr)
    mask_te = np.isfinite(y_te)
    return X_tr[mask_tr], y_tr[mask_tr], X_te[mask_te], y_te[mask_te]


def _check_solution_recovery(
    target_formula: str, recovered_formula: str, m: int, timeout_s: float = 5.0
) -> bool:
    """SymPy-based solution recovery test.

    Returns True iff (target - recovered) simplifies to 0. Uses a SIGALRM
    timeout because sympy.simplify is unbounded on pathological inputs.
    Constants in *recovered* are tolerated by treating numeric-only diffs
    < 1e-6 as equivalence.
    """
    if not recovered_formula:
        return False
    import signal

    import sympy as sp

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        target = sp.sympify(target_formula)
        rec = sp.sympify(recovered_formula)
        diff = sp.simplify(target - rec)
        if diff == 0:
            return True
        # Numerical fallback: if diff has free symbols, reject; if it is a
        # tiny constant from LM noise, accept.
        if not diff.free_symbols:
            try:
                return abs(float(diff)) < 1e-6
            except (TypeError, ValueError):
                return False
        return False
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _build_config(cell: GridCell, max_time: float) -> dict:
    """Bingo config matching the cell's operator set, low budget."""
    cfg = BingoConfig(
        population_size=200,
        stack_size=24,
        operators=BINGO_OPS_BY_SET[cell.op_set],
        use_simplification=False,
        crossover_prob=0.4,
        mutation_prob=0.4,
        metric="mse",
        clo_alg="lm",
        generations=10_000_000,
        fitness_threshold=1e-12,
        max_time=max_time,
        max_evals=10_000_000,
        canonicalization_timeout=15.0,
        use_fast_canonical=True,
        snapshot_frequency=20,
    )
    return {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}


def _run_one(
    cell: GridCell,
    formula: str,
    variant: str,
    seed: int,
    max_time: float,
    n_train: int,
    n_test: int,
    protocol: str = "interp",
    train_bound: float = 1.0,
    test_bound: float = 1.0,
) -> SyntheticRunResult:
    """Execute one (cell, variant, seed) Bingo run."""
    X_tr, y_tr, X_te, y_te = _make_data(
        cell,
        formula,
        n_train,
        n_test,
        seed,
        protocol=protocol,
        train_bound=train_bound,
        test_bound=test_bound,
    )
    cfg_dict = _build_config(cell, max_time)

    runner = BingoBaselineRunner() if variant == "baseline" else IsalSRBingoRunner()

    log.info(
        "  Running %s on cell %02d seed=%d (n_train=%d, n_test=%d, max_time=%.0fs)",
        variant,
        cell.cell_id,
        seed,
        len(y_tr),
        len(y_te),
        max_time,
    )
    t0 = time.perf_counter()
    raw = runner.fit(X_tr, y_tr, X_te, y_te, seed=seed, config=cfg_dict)
    wall = time.perf_counter() - t0

    r2_tr = _r2(y_tr, raw.y_pred_train) if raw.y_pred_train.size else float("nan")
    r2_te = _r2(y_te, raw.y_pred_test) if raw.y_pred_test.size else float("nan")

    rf = (
        float(raw.n_total_dags) / max(1, raw.n_unique_canonical)
        if raw.n_total_dags
        else float("nan")
    )

    recovered_str = str(raw.best_sympy) if raw.best_sympy is not None else ""
    sol_rec = _check_solution_recovery(formula, recovered_str, cell.m)

    return SyntheticRunResult(
        cell_id=cell.cell_id,
        k_requested=cell.k,
        op_set=cell.op_set,
        formula=formula,
        variant=variant,
        seed=seed,
        max_time=max_time,
        protocol=protocol,
        train_bound=train_bound,
        test_bound=test_bound,
        wall_clock_s=wall,
        n_generations=raw.n_generations,
        total_evals=raw.total_evals,
        best_fitness=float(raw.best_fitness),
        r2_train=r2_tr,
        r2_test=r2_te,
        solution_recovered=sol_rec,
        recovered_sympy=recovered_str,
        n_total_dags=int(raw.n_total_dags),
        n_unique_canonical=int(raw.n_unique_canonical),
        n_skipped=int(raw.n_skipped),
        rf_search_trajectory=rf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--grid",
        type=str,
        default="default",
        choices=list(_GRIDS.keys()),
        help="Which grid to draw cells from.",
    )
    parser.add_argument(
        "--cells",
        type=int,
        nargs="+",
        default=[1, 5, 6],
        help="Cell IDs to evaluate (must exist in chosen grid).",
    )
    parser.add_argument("--seeds", type=int, default=2, help="Seeds per (cell, variant).")
    parser.add_argument("--max-time", type=float, default=45.0, help="Bingo max wall time per run.")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--protocol",
        type=str,
        default="interp",
        choices=("interp", "extrap"),
        help="interp: train and test from same Uniform([-train_bound, train_bound])^m. "
        "extrap: train from [-train_bound, train_bound]^m, test from "
        "[-test_bound, test_bound]^m \\ [-train_bound, train_bound]^m.",
    )
    parser.add_argument(
        "--train-bound",
        type=float,
        default=1.0,
        help="Half-side of training hypercube (default 1.0 for interp; "
        "use 0.7 for extrap to leave headroom).",
    )
    parser.add_argument(
        "--test-bound",
        type=float,
        default=1.0,
        help="Half-side of test hypercube (must exceed --train-bound when "
        "--protocol=extrap; e.g. 1.2).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cells = [c for c in _GRIDS[args.grid] if c.cell_id in args.cells]
    if not cells:
        raise SystemExit(f"No cells matched ids {args.cells}")

    log.info(
        "Plan: %d cells x 2 variants x %d seeds = %d runs, max_time=%.0fs (~%.0fmin upper bound)",
        len(cells),
        args.seeds,
        len(cells) * 2 * args.seeds,
        args.max_time,
        len(cells) * 2 * args.seeds * args.max_time / 60.0,
    )

    all_results: list[SyntheticRunResult] = []
    t_start = time.perf_counter()

    for cell in cells:
        log.info("=== Cell %02d (k=%d, m=%d, op=%s) ===", cell.cell_id, cell.k, cell.m, cell.op_set)
        dag, _ = generate_one(cell, args.seed_base)
        sympy_str, _ = dag_to_formula(dag)
        if not sympy_str:
            log.warning("Skipping cell %02d: empty formula", cell.cell_id)
            continue
        log.info("  target f(x) = %s", sympy_str)

        for variant in ("baseline", "isalsr"):
            for s in range(args.seeds):
                seed = args.seed_base + 100 * cell.cell_id + s
                try:
                    res = _run_one(
                        cell,
                        sympy_str,
                        variant,
                        seed,
                        args.max_time,
                        args.n_train,
                        args.n_test,
                        protocol=args.protocol,
                        train_bound=args.train_bound,
                        test_bound=args.test_bound,
                    )
                except Exception as e:  # noqa: BLE001
                    log.exception("FAILED %s cell=%d seed=%d: %s", variant, cell.cell_id, seed, e)
                    continue

                all_results.append(res)
                log.info(
                    "    [%s seed=%d] R2_test=%.4f wall=%.1fs gens=%d evals=%d "
                    "RF_search=%s sol_rec=%s",
                    variant,
                    seed,
                    res.r2_test,
                    res.wall_clock_s,
                    res.n_generations,
                    res.total_evals,
                    f"{res.rf_search_trajectory:.2f}"
                    if not np.isnan(res.rf_search_trajectory)
                    else "n/a",
                    "Y" if res.solution_recovered else "N",
                )

    total_wall = time.perf_counter() - t_start

    out_json = os.path.join(args.output_dir, "synthetic_bingo_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "cells": args.cells,
                    "seeds": args.seeds,
                    "max_time_per_run_s": args.max_time,
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "protocol": args.protocol,
                    "train_bound": args.train_bound,
                    "test_bound": args.test_bound,
                    "total_wall_clock_s": total_wall,
                },
                "results": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )

    out_csv = os.path.join(args.output_dir, "synthetic_bingo_results.csv")
    if all_results:
        import csv

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            w.writeheader()
            for r in all_results:
                w.writerow(asdict(r))

    # Compact summary table to stdout.
    log.info("---- summary (RF_search and R^2_test by cell, variant, seed) ----")
    for r in all_results:
        log.info(
            "cell=%02d %s seed=%d  R2_test=%.4f  wall=%.1fs  RF_search=%s  recovered=%s",
            r.cell_id,
            r.variant,
            r.seed,
            r.r2_test,
            r.wall_clock_s,
            f"{r.rf_search_trajectory:.2f}" if not np.isnan(r.rf_search_trajectory) else "n/a",
            r.recovered_sympy[:60],
        )
    log.info("Total wall-clock: %.1fmin (%.1fs)", total_wall / 60.0, total_wall)
    log.info("Wrote: %s", out_json)
    log.info("Wrote: %s", out_csv)


if __name__ == "__main__":
    main()
