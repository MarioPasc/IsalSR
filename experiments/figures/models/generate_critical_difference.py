"""Generate Critical Difference diagrams for IsalSR model validation.

Produces:
  1. CD diagram for R² test (4 groups: UDFS-BL, UDFS-IS, Bingo-BL, Bingo-IS)
  2. CD diagram for NRMSE test (same 4 groups)
  3. 2D CD diagram: R² test + Reduction Factor (joint view)

Uses critdd (Bunse, 2024): https://github.com/mirkobunse/critdd

Usage:
    python -m experiments.figures.models.generate_critical_difference \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from critdd import Diagram, Diagrams  # noqa: E402

from experiments.models.io_utils import load_all_run_logs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Data loading
# ======================================================================


def _load_problem_means(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    metric_extractor: callable,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the (n_problems x n_groups) matrix for Friedman/CD analysis.

    Returns:
        X: shape (n_problems, n_groups) — mean metric per problem per group.
        group_names: list of group labels (e.g., "UDFS baseline").
        problem_names: list of problem names.
    """
    # Collect per-problem means for each (method, variant)
    group_data: dict[str, dict[str, float]] = {}
    group_names: list[str] = []
    all_problems: set[str] = set()

    for method in methods:
        for variant in ["baseline", "isalsr"]:
            label = f"{method.upper()} {'native DAG' if variant == 'baseline' else 'IsalSR'}"
            group_names.append(label)
            group_data[label] = {}

            bench_dir = results_dir / method / benchmark
            if not bench_dir.exists():
                continue

            for prob_dir in sorted(bench_dir.iterdir()):
                if not prob_dir.is_dir():
                    continue
                variant_dir = prob_dir / variant
                if not variant_dir.exists():
                    continue

                logs = load_all_run_logs(variant_dir)
                if not logs:
                    continue

                values = []
                for rl in logs:
                    try:
                        v = metric_extractor(rl)
                        if np.isfinite(v):
                            values.append(v)
                    except Exception:  # noqa: BLE001
                        pass

                if values:
                    group_data[label][prob_dir.name] = float(np.mean(values))
                    all_problems.add(prob_dir.name)

    # Build matrix: only problems present in ALL groups
    problem_names = sorted(all_problems)
    complete_problems = [p for p in problem_names if all(p in group_data[g] for g in group_names)]

    if not complete_problems:
        log.warning("No problems with complete data across all groups for %s", benchmark)
        return np.array([]), group_names, []

    X = np.zeros((len(complete_problems), len(group_names)))
    for j, g in enumerate(group_names):
        for i, p in enumerate(complete_problems):
            X[i, j] = group_data[g][p]

    return X, group_names, complete_problems


def _load_reduction_factor_matrix(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build RF matrix. Baseline RF = 1.0 by definition."""
    group_data: dict[str, dict[str, float]] = {}
    group_names: list[str] = []
    all_problems: set[str] = set()

    for method in methods:
        for variant in ["baseline", "isalsr"]:
            label = f"{method.upper()} {'native DAG' if variant == 'baseline' else 'IsalSR'}"
            group_names.append(label)
            group_data[label] = {}

            bench_dir = results_dir / method / benchmark
            if not bench_dir.exists():
                continue

            for prob_dir in sorted(bench_dir.iterdir()):
                if not prob_dir.is_dir():
                    continue
                variant_dir = prob_dir / variant
                if not variant_dir.exists():
                    continue

                logs = load_all_run_logs(variant_dir)
                if not logs:
                    continue

                if variant == "baseline":
                    # Baseline has no deduplication: RF = 1.0
                    group_data[label][prob_dir.name] = 1.0
                else:
                    rfs = []
                    for rl in logs:
                        rf = rl.search_space.empirical_reduction_factor
                        if np.isfinite(rf):
                            rfs.append(rf)
                    if rfs:
                        group_data[label][prob_dir.name] = float(np.mean(rfs))

                all_problems.add(prob_dir.name)

    problem_names = sorted(all_problems)
    complete = [p for p in problem_names if all(p in group_data[g] for g in group_names)]

    if not complete:
        return np.array([]), group_names, []

    X = np.zeros((len(complete), len(group_names)))
    for j, g in enumerate(group_names):
        for i, p in enumerate(complete):
            X[i, j] = group_data[g][p]

    return X, group_names, complete


# ======================================================================
# TikZ → PDF rendering
# ======================================================================


def _tikz_to_pdf(tikz_str: str, output_path: Path) -> bool:
    """Compile TikZ string to PDF via pdflatex. Returns True on success."""
    tex_path = output_path.with_suffix(".tex")
    tex_path.write_text(tikz_str)

    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(output_path.parent),
                str(tex_path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        # Clean aux files
        for ext in [".aux", ".log"]:
            aux = output_path.with_suffix(ext)
            if aux.exists():
                aux.unlink()
        log.info("Compiled %s", output_path.with_suffix(".pdf"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("pdflatex failed (TikZ .tex still saved): %s", e)
        return False


# ======================================================================
# Figure generators
# ======================================================================


def generate_cd_r2(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate CD diagram for R² test across all benchmarks combined."""
    # Combine all benchmarks into one matrix
    all_X = []
    all_problems = []

    for benchmark in benchmarks:
        X, group_names, problems = _load_problem_means(
            results_dir,
            methods,
            benchmark,
            metric_extractor=lambda rl: min(max(rl.regression.r2_test, 0.0), 1.0),
        )
        if X.size > 0:
            all_X.append(X)
            all_problems.extend(f"{benchmark}/{p}" for p in problems)

    if not all_X:
        log.warning("No data for R² CD diagram")
        return

    X_combined = np.vstack(all_X)
    log.info("R² test CD: %d problems x %d groups", X_combined.shape[0], X_combined.shape[1])

    diagram = Diagram(X_combined, treatment_names=group_names, maximize_outcome=True)
    tikz_str = diagram.to_str(alpha=0.05, adjustment="holm", as_document=True)

    out_path = output_dir / "cd_r2_test"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".tex").write_text(tikz_str)
    log.info("Saved %s", out_path.with_suffix(".tex"))
    _tikz_to_pdf(tikz_str, out_path)


def generate_cd_nrmse(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate CD diagram for NRMSE test (lower is better)."""
    all_X = []

    for benchmark in benchmarks:
        X, group_names, problems = _load_problem_means(
            results_dir,
            methods,
            benchmark,
            metric_extractor=lambda rl: rl.regression.nrmse_test,
        )
        if X.size > 0:
            all_X.append(X)

    if not all_X:
        log.warning("No data for NRMSE CD diagram")
        return

    X_combined = np.vstack(all_X)
    log.info("NRMSE test CD: %d problems x %d groups", X_combined.shape[0], X_combined.shape[1])

    # NRMSE: lower is better → maximize_outcome=False
    diagram = Diagram(X_combined, treatment_names=group_names, maximize_outcome=False)
    tikz_str = diagram.to_str(alpha=0.05, adjustment="holm", as_document=True)

    out_path = output_dir / "cd_nrmse_test"
    out_path.with_suffix(".tex").write_text(tikz_str)
    log.info("Saved %s", out_path.with_suffix(".tex"))
    _tikz_to_pdf(tikz_str, out_path)


def _style_2d_tikz(tikz_str: str) -> str:
    """Inject custom PGFPlots styling into 2D CD diagram TikZ code.

    Design principles:
      - Same SHAPE  per algorithm:  UDFS = circle (o),  Bingo = triangle (triangle*)
      - Same COLOR  per representation: native DAG = gray (dim), IsalSR = red (vivid)
      - Mark size increased for readability in TPAMI double-column
    """
    # Define the custom cycle list BEFORE \\begin{axis}
    cycle_def = (
        "\\pgfplotscreateplotcyclelist{isalsr}{\n"
        "  {gray!70!black, mark=o, mark size=4pt, very thick},\n"  # UDFS native DAG
        "  {red!80!black, mark=o, mark size=4pt, very thick},\n"  # UDFS IsalSR
        "  {gray!70!black, mark=triangle*, mark size=5pt, very thick},\n"  # Bingo native DAG
        "  {red!80!black, mark=triangle*, mark size=5pt, very thick},\n"  # Bingo IsalSR
        "}\n"
    )
    tikz_str = tikz_str.replace(
        "\\begin{axis}[",
        cycle_def + "\\begin{axis}[\n  cycle list name=isalsr,",
    )
    return tikz_str


# Treatment label map: algorithm-first naming with representation qualifier
_TREATMENT_LABELS = {
    "UDFS native DAG": "UDFS native DAG",
    "UDFS IsalSR": "UDFS IsalSR",
    "BINGO native DAG": "Bingo native DAG",
    "BINGO IsalSR": "Bingo IsalSR",
}


def _caption_cd_2d(
    n_problems: int,
    methods: list[str],
    benchmarks: list[str],
) -> str:
    """Return a LaTeX-ready caption for the 2D critical difference diagram."""
    bench_str = " and ".join(b.capitalize() for b in benchmarks)
    method_str = " and ".join(methods)
    return (
        f"Two-dimensional critical difference diagram comparing four "
        f"configurations---{method_str}, each with its native DAG "
        f"representation and with \\IsalSR{{}} canonicalization---on "
        f"{n_problems} benchmark problems ({bench_str}). "
        f"Marker shape encodes the SR method "
        f"(\\tikz\\node[circle,draw=gray!70!black,inner sep=1.5pt]{{}}; "
        f"= UDFS, "
        f"\\tikz\\node[regular polygon,regular polygon sides=3,"
        f"fill=gray!70!black,inner sep=1pt]{{}}; = Bingo) "
        f"and colour encodes the representation "
        f"(gray = native DAG, red = \\IsalSR). "
        f"Top row: $R^2$ test (higher rank = better fit). "
        f"Bottom row: reduction factor $\\rho$ (higher rank = more "
        f"redundancy eliminated). "
        f"Thick horizontal bars connect groups that are not significantly "
        f"different (Nemenyi post-hoc, $\\alpha = 0.05$, Holm adjustment). "
        f"\\IsalSR{{}}-augmented variants (red) match or improve quality "
        f"rankings while achieving significantly higher $\\rho$ than their "
        f"native-DAG counterparts (gray)."
    )


def generate_cd_2d(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate 2D CD diagram: R² test (row 1) + Reduction Factor (row 2).

    This shows the joint trade-off: are the same groups that rank well on
    quality also ranking well on search space reduction?

    Visual encoding:
      Shape  → algorithm identity (circle = UDFS, triangle = Bingo)
      Colour → representation    (gray = native DAG, red = IsalSR)
    """
    all_r2: list[np.ndarray] = []
    all_rf: list[np.ndarray] = []
    group_names: list[str] = []

    for benchmark in benchmarks:
        X_r2, gn, problems_r2 = _load_problem_means(
            results_dir,
            methods,
            benchmark,
            metric_extractor=lambda rl: min(max(rl.regression.r2_test, 0.0), 1.0),
        )
        X_rf, _, problems_rf = _load_reduction_factor_matrix(
            results_dir,
            methods,
            benchmark,
        )
        if not group_names and gn:
            group_names = gn

        if X_r2.size > 0 and X_rf.size > 0:
            common = sorted(set(problems_r2) & set(problems_rf))
            idx_r2 = [problems_r2.index(p) for p in common]
            idx_rf = [problems_rf.index(p) for p in common]
            all_r2.append(X_r2[idx_r2])
            all_rf.append(X_rf[idx_rf])

    if not all_r2:
        log.warning("No data for 2D CD diagram")
        return

    X_r2_combined = np.vstack(all_r2)
    X_rf_combined = np.vstack(all_rf)
    log.info(
        "2D CD: %d problems x %d groups, 2 metrics",
        X_r2_combined.shape[0],
        X_r2_combined.shape[1],
    )

    # Apply label renaming
    display_names = [_TREATMENT_LABELS.get(g, g) for g in group_names]

    diagrams = Diagrams(
        [X_r2_combined, X_rf_combined],
        diagram_names=["$R^2$ test", "Reduction factor $\\rho$"],
        treatment_names=display_names,
        maximize_outcome=True,
    )
    tikz_str = diagrams.to_str(alpha=0.05, adjustment="holm", as_document=True)

    # Inject custom shape/color styling
    tikz_str = _style_2d_tikz(tikz_str)

    out_path = output_dir / "cd_2d_r2_rf"
    out_path.with_suffix(".tex").write_text(tikz_str)
    log.info("Saved %s", out_path.with_suffix(".tex"))
    _tikz_to_pdf(tikz_str, out_path)

    # Save caption
    caption = _caption_cd_2d(
        X_r2_combined.shape[0],
        [m.upper() for m in methods],
        benchmarks,
    )
    out_path.with_suffix(".caption.txt").write_text(caption)
    log.info("Saved %s", out_path.with_suffix(".caption.txt"))


def generate_cd_per_benchmark(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate separate CD diagrams per benchmark (for appendix/supplementary)."""
    for benchmark in benchmarks:
        X, group_names, problems = _load_problem_means(
            results_dir,
            methods,
            benchmark,
            metric_extractor=lambda rl: min(max(rl.regression.r2_test, 0.0), 1.0),
        )
        if X.size == 0:
            continue

        log.info("R² test CD (%s): %d problems x %d groups", benchmark, *X.shape)
        diagram = Diagram(X, treatment_names=group_names, maximize_outcome=True)
        tikz_str = diagram.to_str(alpha=0.05, adjustment="holm", as_document=True)

        out_path = output_dir / f"cd_r2_test_{benchmark}"
        out_path.with_suffix(".tex").write_text(tikz_str)
        log.info("Saved %s", out_path.with_suffix(".tex"))
        _tikz_to_pdf(tikz_str, out_path)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CD diagrams for IsalSR")
    parser.add_argument("--results-dir", required=True, help="Base results directory")
    parser.add_argument("--output-dir", required=True, help="Figure output directory")
    parser.add_argument("--methods", default="udfs,bingo", help="Comma-separated methods")
    parser.add_argument("--benchmarks", default="nguyen,feynman", help="Comma-separated benchmarks")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating CD diagrams from %s", results_dir)

    generate_cd_r2(results_dir, output_dir, methods, benchmarks)
    generate_cd_nrmse(results_dir, output_dir, methods, benchmarks)
    generate_cd_2d(results_dir, output_dir, methods, benchmarks)
    generate_cd_per_benchmark(results_dir, output_dir, methods, benchmarks)

    log.info("All CD diagrams saved to %s", output_dir)


if __name__ == "__main__":
    main()
