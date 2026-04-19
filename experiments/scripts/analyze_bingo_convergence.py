"""Analyze Bingo convergence logs for IsalSR hard-tier benchmarks.

Loads all convergence_log.npz files for the hard benchmark suite and reports:
  - Per-(problem, variant, seed): convergence generation and final best R².
  - Per-(problem, variant): mean/std/95%-CI of convergence generation, total generations, best R².
  - Global: mean convergence generation baseline vs isalsr with 95% CI and Wilcoxon test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation"
    "/wl_subtree_hard/models_hard/bingo/hard"
)
PROBLEMS = [
    "i.15.10",
    "i.30.3",
    "i.37.4",
    "ii.11.27",
    "iii.17.37",
    "keijzer_6",
    "korns_12",
    "pagie_1",
    "vladislavleva_2",
    "vladislavleva_4",
]
VARIANTS = ["baseline", "isalsr"]
N_SEEDS_EXPECTED = 30
CI_Z = 1.96  # 95% confidence interval multiplier


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Metrics for one (problem, variant, seed) triple."""

    problem: str
    variant: str
    seed: str
    convergence_gen: int  # generation at which best R² is first achieved
    total_gens: int  # last recorded generation
    final_best_r2: float  # best_r2_train[-1]
    n_logged_points: int  # length of the arrays


@dataclass
class VariantSummary:
    """Aggregated metrics across seeds for one (problem, variant)."""

    problem: str
    variant: str
    n_seeds: int
    mean_r2: float
    std_r2: float
    mean_conv_gen: float
    std_conv_gen: float
    ci95_conv_gen: float  # half-width of 95% CI
    mean_total_gen: float
    std_total_gen: float
    convergence_gens: list[int] = field(default_factory=list)  # raw per-seed values
    total_gens: list[int] = field(default_factory=list)
    final_r2s: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def convergence_generation(generations: npt.NDArray, best_r2: npt.NDArray) -> int:
    """Return the generation number at which best_r2_train is first maximised.

    Parameters
    ----------
    generations:
        Integer array of generation indices (shape ``(T,)``).
    best_r2:
        Cumulative-maximum best R² per generation (shape ``(T,)``).

    Returns
    -------
    int
        Generation number (not array index) of first occurrence of the
        overall maximum.
    """
    final_max = best_r2[-1]
    first_idx = int(np.argmax(best_r2 >= final_max))
    return int(generations[first_idx])


def load_seed(
    problem: str,
    variant: str,
    seed_dir: Path,
) -> SeedResult | None:
    """Load one convergence_log.npz and return a SeedResult.

    Returns None if the file is missing or unreadable.
    """
    npz_path = seed_dir / "convergence_log.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        generations: npt.NDArray = data["generations"]
        best_r2: npt.NDArray = data["best_r2_train"]

        return SeedResult(
            problem=problem,
            variant=variant,
            seed=seed_dir.name,
            convergence_gen=convergence_generation(generations, best_r2),
            total_gens=int(generations[-1]),
            final_best_r2=float(best_r2[-1]),
            n_logged_points=int(len(generations)),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Failed to load {npz_path}: {exc}")
        return None


def load_all() -> tuple[list[SeedResult], list[str]]:
    """Load all convergence logs and return (results, missing_files).

    Returns
    -------
    results:
        List of SeedResult for every successfully loaded file.
    missing:
        List of path strings for expected but absent/failed files.
    """
    results: list[SeedResult] = []
    missing: list[str] = []

    for problem in PROBLEMS:
        for variant in VARIANTS:
            variant_dir = ROOT / problem / variant
            if not variant_dir.exists():
                print(f"  [WARN] Missing variant directory: {variant_dir}")
                continue
            # Collect seed_* directories
            seed_dirs = sorted(
                d for d in variant_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
            )
            for seed_dir in seed_dirs:
                result = load_seed(problem, variant, seed_dir)
                if result is None:
                    missing.append(str(seed_dir / "convergence_log.npz"))
                else:
                    results.append(result)

    return results, missing


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise_variant(
    problem: str,
    variant: str,
    seed_results: list[SeedResult],
) -> VariantSummary:
    """Compute per-variant summary statistics from a list of SeedResult."""
    conv_gens = [r.convergence_gen for r in seed_results]
    total_gens = [r.total_gens for r in seed_results]
    r2s = [r.final_best_r2 for r in seed_results]
    n = len(seed_results)

    conv_arr = np.array(conv_gens, dtype=float)
    total_arr = np.array(total_gens, dtype=float)
    r2_arr = np.array(r2s, dtype=float)

    ci_half = CI_Z * float(np.std(conv_arr, ddof=1)) / np.sqrt(n) if n > 1 else 0.0

    return VariantSummary(
        problem=problem,
        variant=variant,
        n_seeds=n,
        mean_r2=float(np.mean(r2_arr)),
        std_r2=float(np.std(r2_arr, ddof=1)) if n > 1 else 0.0,
        mean_conv_gen=float(np.mean(conv_arr)),
        std_conv_gen=float(np.std(conv_arr, ddof=1)) if n > 1 else 0.0,
        ci95_conv_gen=ci_half,
        mean_total_gen=float(np.mean(total_arr)),
        std_total_gen=float(np.std(total_arr, ddof=1)) if n > 1 else 0.0,
        convergence_gens=conv_gens,
        total_gens=total_gens,
        final_r2s=r2s,
    )


def global_wilcoxon(
    baseline_summaries: dict[str, VariantSummary],
    isalsr_summaries: dict[str, VariantSummary],
) -> tuple[float, float]:
    """Run paired Wilcoxon signed-rank test on convergence generations.

    Pairs are matched within the same problem and (when counts match) by
    seed index order. Only complete pairs are used.

    Returns
    -------
    statistic, p_value
    """
    base_vals: list[float] = []
    isal_vals: list[float] = []

    for problem in PROBLEMS:
        bsumm = baseline_summaries.get(problem)
        isumm = isalsr_summaries.get(problem)
        if bsumm is None or isumm is None:
            continue
        n_pairs = min(len(bsumm.convergence_gens), len(isumm.convergence_gens))
        base_vals.extend(bsumm.convergence_gens[:n_pairs])
        isal_vals.extend(isumm.convergence_gens[:n_pairs])

    if len(base_vals) < 2:
        return float("nan"), float("nan")

    stat, p = stats.wilcoxon(base_vals, isal_vals, alternative="two-sided")
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def fmt_gen(mean: float, ci: float) -> str:
    return f"{mean:8.1f} ± {ci:6.1f}"


def fmt_r2(mean: float, std: float) -> str:
    return f"{mean:+.4f} ± {std:.4f}"


def print_separator(width: int = 130) -> None:
    print("-" * width)


def print_double_separator(width: int = 130) -> None:
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print_double_separator()
    print("BINGO CONVERGENCE ANALYSIS  —  Hard Benchmark Suite (IsalSR vs Baseline)")
    print_double_separator()
    print()

    # ------------------------------------------------------------------
    # 1. Load all files
    # ------------------------------------------------------------------
    print("Loading convergence logs ...")
    results, missing = load_all()
    print(f"  Loaded : {len(results)} files")
    print(f"  Missing: {len(missing)} files")
    if missing:
        print("  Missing files:")
        for m in missing:
            print(f"    {m}")
    print()

    # ------------------------------------------------------------------
    # 2. Group by (problem, variant)
    # ------------------------------------------------------------------
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[SeedResult]] = defaultdict(list)
    for r in results:
        grouped[(r.problem, r.variant)].append(r)

    # Sort seeds within group for deterministic pairing
    for key in grouped:
        grouped[key].sort(key=lambda r: r.seed)

    # ------------------------------------------------------------------
    # 3. Per-seed detail
    # ------------------------------------------------------------------
    print_double_separator()
    print("PER-SEED DETAIL")
    print_double_separator()
    hdr = f"{'Problem':<20} {'Variant':<10} {'Seed':<10} {'Conv.Gen':>10} {'TotalGen':>10} {'BestR²':>10} {'LogPts':>8}"
    print(hdr)
    print_separator()
    for problem in PROBLEMS:
        for variant in VARIANTS:
            seed_results = grouped.get((problem, variant), [])
            for sr in seed_results:
                print(
                    f"{problem:<20} {variant:<10} {sr.seed:<10} "
                    f"{sr.convergence_gen:>10d} {sr.total_gens:>10d} "
                    f"{sr.final_best_r2:>+10.4f} {sr.n_logged_points:>8d}"
                )
        print_separator()
    print()

    # ------------------------------------------------------------------
    # 4. Per-problem summary
    # ------------------------------------------------------------------
    baseline_summaries: dict[str, VariantSummary] = {}
    isalsr_summaries: dict[str, VariantSummary] = {}

    for problem in PROBLEMS:
        for variant in VARIANTS:
            seed_results = grouped.get((problem, variant), [])
            if not seed_results:
                continue
            summ = summarise_variant(problem, variant, seed_results)
            if variant == "baseline":
                baseline_summaries[problem] = summ
            else:
                isalsr_summaries[problem] = summ

    print_double_separator()
    print("PER-PROBLEM SUMMARY  (mean best R² and convergence generation, 95% CI)")
    print_double_separator()

    col_w = 130
    print(
        f"{'Problem':<20} "
        f"{'Variant':<10} "
        f"{'N':>3} "
        f"{'Mean Best R²':>22} "
        f"{'Conv.Gen (mean±95%CI)':>26} "
        f"{'Total Gen (mean±std)':>25} "
        f"{'Conv/Total %':>13}"
    )
    print_separator(col_w)

    for problem in PROBLEMS:
        for variant in VARIANTS:
            summ = (
                baseline_summaries.get(problem)
                if variant == "baseline"
                else isalsr_summaries.get(problem)
            )
            if summ is None:
                print(f"{problem:<20} {variant:<10} --- (no data)")
                continue
            conv_pct = (
                100.0 * summ.mean_conv_gen / summ.mean_total_gen
                if summ.mean_total_gen > 0
                else float("nan")
            )
            print(
                f"{problem:<20} {variant:<10} {summ.n_seeds:>3} "
                f"  {fmt_r2(summ.mean_r2, summ.std_r2):>22} "
                f"  {fmt_gen(summ.mean_conv_gen, summ.ci95_conv_gen):>26} "
                f"  {summ.mean_total_gen:>10.1f} ± {summ.std_total_gen:>8.1f} "
                f"  {conv_pct:>10.1f}%"
            )
        print_separator(col_w)
    print()

    # ------------------------------------------------------------------
    # 5. Convergence generation comparison (baseline vs isalsr, per problem)
    # ------------------------------------------------------------------
    print_double_separator()
    print("CONVERGENCE SPEED COMPARISON  (baseline vs isalsr, per problem)")
    print_double_separator()
    print(
        f"{'Problem':<20} "
        f"{'Base Conv.Gen':>18} "
        f"{'ISAL Conv.Gen':>18} "
        f"{'Δ (isal−base)':>16} "
        f"{'Faster?':>10} "
        f"{'Base Total':>12} "
        f"{'ISAL Total':>12} "
        f"{'Δ Total':>10}"
    )
    print_separator()

    for problem in PROBLEMS:
        bsumm = baseline_summaries.get(problem)
        isumm = isalsr_summaries.get(problem)
        if bsumm is None or isumm is None:
            print(f"{problem:<20}  (missing data for one variant)")
            continue
        delta_conv = isumm.mean_conv_gen - bsumm.mean_conv_gen
        delta_total = isumm.mean_total_gen - bsumm.mean_total_gen
        direction = (
            "IsalSR faster"
            if delta_conv < 0
            else ("Baseline faster" if delta_conv > 0 else "Equal")
        )
        print(
            f"{problem:<20} "
            f"  {fmt_gen(bsumm.mean_conv_gen, bsumm.ci95_conv_gen):>18} "
            f"  {fmt_gen(isumm.mean_conv_gen, isumm.ci95_conv_gen):>18} "
            f"  {delta_conv:>+14.1f} "
            f"  {direction:>14} "
            f"  {bsumm.mean_total_gen:>10.0f} "
            f"  {isumm.mean_total_gen:>10.0f} "
            f"  {delta_total:>+8.0f}"
        )
    print()

    # ------------------------------------------------------------------
    # 6. Global summary (pooled across all problems)
    # ------------------------------------------------------------------
    print_double_separator()
    print("GLOBAL SUMMARY  (pooled across all 10 problems, all seeds)")
    print_double_separator()

    all_base_conv: list[float] = []
    all_isal_conv: list[float] = []
    all_base_total: list[float] = []
    all_isal_total: list[float] = []
    all_base_r2: list[float] = []
    all_isal_r2: list[float] = []

    for problem in PROBLEMS:
        bsumm = baseline_summaries.get(problem)
        isumm = isalsr_summaries.get(problem)
        if bsumm:
            all_base_conv.extend(bsumm.convergence_gens)
            all_base_total.extend(bsumm.total_gens)
            all_base_r2.extend(bsumm.final_r2s)
        if isumm:
            all_isal_conv.extend(isumm.convergence_gens)
            all_isal_total.extend(isumm.total_gens)
            all_isal_r2.extend(isumm.final_r2s)

    def ci95(vals: list[float]) -> float:
        arr = np.array(vals, dtype=float)
        return CI_Z * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    def mean_std(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    bm_conv, bs_conv = mean_std(all_base_conv)
    im_conv, is_conv = mean_std(all_isal_conv)
    bm_total, bs_total = mean_std(all_base_total)
    im_total, is_total = mean_std(all_isal_total)
    bm_r2, bs_r2 = mean_std(all_base_r2)
    im_r2, is_r2 = mean_std(all_isal_r2)

    bc_ci = ci95(all_base_conv)
    ic_ci = ci95(all_isal_conv)

    print(f"  {'Metric':<30} {'Baseline':>25} {'IsalSR':>25}")
    print_separator(85)
    print(f"  {'N runs':<30} {len(all_base_conv):>25d} {len(all_isal_conv):>25d}")
    print(
        f"  {'Conv. Gen (mean ± 95%CI)':<30} {bm_conv:>15.1f} ± {bc_ci:>6.1f}   {im_conv:>15.1f} ± {ic_ci:>6.1f}"
    )
    print(f"  {'Conv. Gen (std)':<30} {bs_conv:>25.1f} {is_conv:>25.1f}")
    print(
        f"  {'Total Gen (mean ± std)':<30} {bm_total:>15.1f} ± {bs_total:>6.1f}   {im_total:>15.1f} ± {is_total:>6.1f}"
    )
    print(
        f"  {'Best R² (mean ± std)':<30} {bm_r2:>+14.4f} ± {bs_r2:>6.4f}   {im_r2:>+14.4f} ± {is_r2:>6.4f}"
    )
    print()

    # ------------------------------------------------------------------
    # 7. Wilcoxon test
    # ------------------------------------------------------------------
    stat, p = global_wilcoxon(baseline_summaries, isalsr_summaries)
    delta_global = im_conv - bm_conv
    direction_global = (
        "IsalSR converges earlier" if delta_global < 0 else "Baseline converges earlier"
    )
    print(f"  Δ convergence generation (IsalSR − Baseline): {delta_global:+.1f}")
    print(f"  Direction: {direction_global}")
    print()
    print("  Paired Wilcoxon signed-rank test (convergence generation):")
    print("    H₀: median(baseline conv.gen) = median(isalsr conv.gen)")
    print(f"    Statistic = {stat:.2f},  p = {p:.4f}")
    if not np.isnan(p):
        sig = "SIGNIFICANT (p < 0.05)" if p < 0.05 else "not significant (p ≥ 0.05)"
        print(f"    Result: {sig}")
    print()

    # ------------------------------------------------------------------
    # 8. Run-length analysis: do isalsr runs have more/fewer total generations?
    # ------------------------------------------------------------------
    print_double_separator()
    print("RUN-LENGTH ANALYSIS  (total generations: more = ran longer in wall-time budget)")
    print_double_separator()
    print(
        f"{'Problem':<20} "
        f"{'Base Total Gen':>18} "
        f"{'ISAL Total Gen':>18} "
        f"{'Δ (isal−base)':>16} "
        f"{'Interpretation':>30}"
    )
    print_separator()
    for problem in PROBLEMS:
        bsumm = baseline_summaries.get(problem)
        isumm = isalsr_summaries.get(problem)
        if bsumm is None or isumm is None:
            continue
        delta_t = isumm.mean_total_gen - bsumm.mean_total_gen
        if delta_t < -100:
            interp = "IsalSR ran fewer generations (overhead)"
        elif delta_t > 100:
            interp = "IsalSR ran more generations"
        else:
            interp = "Similar run length"
        print(
            f"{problem:<20} "
            f"  {bsumm.mean_total_gen:>16.0f} "
            f"  {isumm.mean_total_gen:>16.0f} "
            f"  {delta_t:>+14.0f} "
            f"  {interp:>30}"
        )
    print()
    print(f"  Global: Baseline mean total gen = {bm_total:.0f} ± {bs_total:.0f}")
    print(f"  Global: IsalSR   mean total gen = {im_total:.0f} ± {is_total:.0f}")
    delta_total_global = im_total - bm_total
    print(
        f"  Global: Δ = {delta_total_global:+.0f}  ({'IsalSR fewer gens' if delta_total_global < 0 else 'IsalSR more gens or equal'})"
    )
    print()
    print_double_separator()
    print("END OF REPORT")
    print_double_separator()


if __name__ == "__main__":
    main()
