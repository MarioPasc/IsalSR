"""Run the full model-figure suite and place outputs in a given directory.

Usage
-----
python -m experiments.figures.models.generate_all \
    --results-dir /media/mpascual/Sandisk2TB/research/isalsr/results/model_validation \
    --output-dir  /media/mpascual/Sandisk2TB/research/isalsr/results/figures \
    [--methods udfs,bingo] [--benchmarks nguyen,feynman] [--dag-weighted]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from experiments.figures.models.generate_case_study import generate_case_study
from experiments.figures.models.generate_convergence import generate_convergence_figure
from experiments.figures.models.generate_critical_difference import (
    generate_cd_2d,
    generate_cd_nrmse,
    generate_cd_per_benchmark,
    generate_cd_r2,
)
from experiments.figures.models.generate_forest_plot import generate_forest_plot
from experiments.figures.models.generate_pareto import generate_pareto
from experiments.figures.models.generate_reduction_factor import (
    generate_reduction_factor_figure,
)
from experiments.figures.models.generate_solution_rate import generate_solution_rate
from experiments.figures.models.generate_tables import (
    generate_table1,
    generate_table2,
    generate_table_k_range,
    generate_table_supplementary,
)

log = logging.getLogger(__name__)

# Default case study problems (method, benchmark, problem_id).
_CASE_STUDIES: list[tuple[str, str, str]] = [
    ("udfs", "feynman", "i.12.4"),
]


def _run_step(name: str, fn: Callable[[], None]) -> tuple[bool, float]:
    """Execute *fn*, returning (success, elapsed_seconds)."""
    log.info("── %s", name)
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        log.info("   ✓ %s  (%.1fs)", name, elapsed)
        return True, elapsed
    except Exception:
        elapsed = time.perf_counter() - t0
        log.exception("   ✗ %s  FAILED (%.1fs)", name, elapsed)
        return False, elapsed


def generate_all(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    *,
    dag_weighted: bool = False,
    case_studies: list[tuple[str, str, str]] | None = None,
) -> dict[str, bool]:
    """Run every figure/table generator and return a name→success map."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if case_studies is None:
        case_studies = _CASE_STUDIES

    steps: list[tuple[str, Callable[[], None]]] = [
        # -- Tables ----------------------------------------------------------
        (
            "Table 1 – three-axis summary",
            lambda: generate_table1(results_dir, methods, benchmarks, output_dir),
        ),
        (
            "Table 2 – per-problem R²",
            lambda: generate_table2(results_dir, methods, benchmarks, output_dir),
        ),
        (
            "Table K – k-range overhead",
            lambda: generate_table_k_range(results_dir, methods, benchmarks, output_dir),
        ),
        (
            "Table S – supplementary per-problem",
            lambda: generate_table_supplementary(results_dir, methods, benchmarks, output_dir),
        ),
        # -- Figures ----------------------------------------------------------
        (
            "Reduction factor distribution",
            lambda: generate_reduction_factor_figure(
                results_dir,
                output_dir,
                methods,
                benchmarks,
                show_dag_weighted=dag_weighted,
            ),
        ),
        (
            "Forest plot (Cohen's d)",
            lambda: generate_forest_plot(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "Pareto (R² vs complexity)",
            lambda: generate_pareto(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "Solution rate",
            lambda: generate_solution_rate(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "Convergence trajectories",
            lambda: generate_convergence_figure(results_dir, output_dir),
        ),
        # -- Critical difference diagrams ------------------------------------
        (
            "CD diagram – R² test",
            lambda: generate_cd_r2(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "CD diagram – NRMSE test",
            lambda: generate_cd_nrmse(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "CD diagram – 2D (R² + RF)",
            lambda: generate_cd_2d(results_dir, output_dir, methods, benchmarks),
        ),
        (
            "CD diagrams – per benchmark",
            lambda: generate_cd_per_benchmark(results_dir, output_dir, methods, benchmarks),
        ),
    ]

    # -- Case studies (one step per configured problem) ----------------------
    for method, benchmark, problem in case_studies:
        steps.append(
            (
                f"Case study – {method}/{benchmark}/{problem}",
                lambda m=method, b=benchmark, p=problem: generate_case_study(
                    results_dir, output_dir, m, b, p
                ),
            )
        )

    results: dict[str, bool] = {}
    t_total = time.perf_counter()

    for name, fn in steps:
        ok, _ = _run_step(name, fn)
        results[name] = ok

    elapsed_total = time.perf_counter() - t_total

    # -- Summary -------------------------------------------------------------
    n_ok = sum(results.values())
    n_total = len(results)
    log.info("━" * 60)
    log.info("Done: %d/%d succeeded  (%.1fs total)", n_ok, n_total, elapsed_total)
    if n_ok < n_total:
        failed = [n for n, ok in results.items() if not ok]
        log.warning("Failed steps:\n  • %s", "\n  • ".join(failed))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the full model-figure suite.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root results directory (contains method/benchmark/problem sub-dirs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where all figures and tables will be written.",
    )
    parser.add_argument(
        "--methods",
        default="udfs,bingo",
        help="Comma-separated SR methods (default: udfs,bingo).",
    )
    parser.add_argument(
        "--benchmarks",
        default="nguyen,feynman",
        help="Comma-separated benchmark suites (default: nguyen,feynman).",
    )
    parser.add_argument(
        "--dag-weighted",
        action="store_true",
        help="Show DAG-weighted RF diamonds on the reduction-factor figure.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    results = generate_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        methods=methods,
        benchmarks=benchmarks,
        dag_weighted=args.dag_weighted,
    )

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
