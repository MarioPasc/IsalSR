"""Analysis script for IsalSR hard benchmark results.

Loads all run_log.json files from the hard benchmark experiment, computes
descriptive statistics (mean±std of r2_test, nrmse_test, model_complexity,
empirical_reduction_factor, redundancy_rate), paired Cohen's d with 95%
bootstrap CI, Wilcoxon signed-rank or paired t-test (Shapiro-Wilk guard)
with Holm-Bonferroni correction, and computational overhead percentages.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TypeAlias

import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RunData: TypeAlias = dict  # raw parsed run_log.json content


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "model_validation/wl_subtree_hard/models_hard"
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

METHODS = ["udfs", "bingo"]
VARIANTS = ["baseline", "isalsr"]

ALPHA = 0.05
N_BOOTSTRAP = 2000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_run_logs(results_dir: Path) -> dict[tuple[str, str, str], list[RunData]]:
    """Load all run_log.json files, keyed by (method, problem, variant).

    Parameters
    ----------
    results_dir : Path
        Root directory containing {method}/hard/{problem}/{variant}/seed_*/run_log.json.

    Returns
    -------
    dict[tuple[str, str, str], list[RunData]]
        Mapping from (method, problem, variant) to list of parsed JSONs.
    """
    data: dict[tuple[str, str, str], list[RunData]] = {}

    for json_path in sorted(results_dir.rglob("run_log.json")):
        parts = json_path.parts
        # Expected: .../{method}/hard/{problem}/{variant}/seed_XX/run_log.json
        try:
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed_"))
            variant = parts[seed_idx - 1]
            problem_raw = parts[seed_idx - 2]
            benchmark = parts[seed_idx - 3]
            method = parts[seed_idx - 4]
        except (StopIteration, IndexError):
            warnings.warn(f"Skipping unexpected path: {json_path}", stacklevel=2)
            continue

        if benchmark != "hard":
            continue

        # Normalise problem name to lowercase with underscores
        problem = problem_raw.lower()

        with json_path.open() as fh:
            run = json.load(fh)

        key = (method, problem, variant)
        data.setdefault(key, []).append(run)

    return data


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def safe_get(run: RunData, *keys: str, default: float = float("nan")) -> float:
    """Traverse nested dict safely, returning default on missing key/None."""
    node: dict | float | None = run
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if node is None:
            return default
    return float(node) if node is not None else default


def extract_metrics(runs: list[RunData]) -> dict[str, np.ndarray]:
    """Extract arrays of per-seed metrics from a list of run logs.

    Parameters
    ----------
    runs : list[RunData]
        Run logs for a single (method, problem, variant) group.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: r2_test, r2_train, nrmse_test, nrmse_train, model_complexity,
              solution_recovered, empirical_reduction_factor, redundancy_rate,
              wall_clock_total_s, overhead_time_s, evaluation_time_s.
    """
    keys_reg = ["r2_train", "r2_test", "nrmse_train", "nrmse_test", "model_complexity"]
    keys_ss = [
        "empirical_reduction_factor",
        "redundancy_rate",
        "total_dags_explored",
        "unique_canonical_dags",
    ]
    keys_time = [
        "wall_clock_total_s",
        "overhead_time_s",
        "evaluation_time_s",
        "canonicalization_runtime_s",
    ]

    metrics: dict[str, list[float]] = {
        k: [] for k in keys_reg + keys_ss + keys_time + ["solution_recovered"]
    }

    for run in runs:
        for k in keys_reg:
            metrics[k].append(safe_get(run, "results", "regression", k))
        metrics["solution_recovered"].append(
            float(safe_get(run, "results", "regression", "solution_recovered", default=False))
        )
        for k in keys_ss:
            metrics[k].append(safe_get(run, "results", "search_space", k))
        for k in keys_time:
            metrics[k].append(safe_get(run, "results", "time", k))

    return {k: np.array(v, dtype=float) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def cohens_d_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
    alpha: float = ALPHA,
) -> tuple[float, float, float]:
    """Compute paired Cohen's d with percentile bootstrap CI.

    Parameters
    ----------
    a, b : np.ndarray
        Paired observations (same length). d = mean(a-b) / std(a-b).
    n_boot : int
        Number of bootstrap replications.
    rng_seed : int
        Random seed for reproducibility.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    tuple[float, float, float]
        (d, ci_lower, ci_upper).
    """
    diffs = a - b
    d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))

    rng = np.random.default_rng(rng_seed)
    boot_d = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_d[i] = float(np.mean(sample) / (np.std(sample, ddof=1) + 1e-12))

    lo = float(np.percentile(boot_d, 100 * alpha / 2))
    hi = float(np.percentile(boot_d, 100 * (1 - alpha / 2)))
    return d, lo, hi


def paired_test(
    baseline: np.ndarray,
    isalsr: np.ndarray,
    alpha_shapiro: float = 0.05,
) -> tuple[str, float, float]:
    """Run paired test (Wilcoxon or t-test) after Shapiro-Wilk normality check.

    Parameters
    ----------
    baseline, isalsr : np.ndarray
        Paired samples (same length, same seed order).
    alpha_shapiro : float
        Significance level for normality test.

    Returns
    -------
    tuple[str, float, float]
        (test_name, statistic, p_value).
    """
    diffs = isalsr - baseline
    # Shapiro-Wilk requires n >= 3; degenerate diffs (all zeros) are normal.
    if len(diffs) < 3:
        return "n/a", float("nan"), float("nan")

    if np.all(diffs == 0):
        return "t-test", 0.0, 1.0

    _, p_shapiro = stats.shapiro(diffs)
    if p_shapiro >= alpha_shapiro:
        stat, p = stats.ttest_rel(isalsr, baseline)
        return "t-test", float(stat), float(p)
    else:
        # Wilcoxon requires non-zero differences
        nonzero = diffs[diffs != 0]
        if len(nonzero) == 0:
            return "Wilcoxon", float("nan"), 1.0
        stat, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return "Wilcoxon", float(stat), float(p)


def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Apply Holm-Bonferroni step-down correction.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values (one per hypothesis).
    alpha : float
        Family-wise error rate target.

    Returns
    -------
    list[bool]
        reject[i] is True if hypothesis i is rejected at corrected level.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            # All subsequent hypotheses are retained (monotone)
            break
    return reject


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def ms(mean: float, std: float, fmt: str = ".3f") -> str:
    """Format mean±std string."""
    if np.isnan(mean):
        return "   N/A    "
    return f"{mean:{fmt}} ± {std:{fmt}}"


def fmt_p(p: float) -> str:
    """Format p-value, highlighting significance."""
    if np.isnan(p):
        return "   N/A  "
    if p < 0.001:
        return f"{p:.2e}***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f}  *"
    return f"{p:.4f}   "


def fmt_d(d: float, lo: float, hi: float) -> str:
    """Format Cohen's d with CI."""
    if np.isnan(d):
        return "    N/A    "
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def section(title: str) -> None:
    """Print a section header."""
    width = 120
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full analysis and print formatted tables."""
    print("Loading run logs …")
    data = load_run_logs(RESULTS_DIR)

    # Inventory
    print(f"\nLoaded {sum(len(v) for v in data.values())} run_log.json files.")
    print(f"Groups: {len(data)}")
    print()
    print(f"{'Group (method, problem, variant)':<55} {'N runs':>7}")
    print("-" * 65)
    for (method, problem, variant), runs in sorted(data.items()):
        print(f"  ({method}, {problem}, {variant}){'':<5} {len(runs):>7}")

    # Pre-extract metrics for all groups
    metrics: dict[tuple[str, str, str], dict[str, np.ndarray]] = {
        key: extract_metrics(runs) for key, runs in data.items()
    }

    # -----------------------------------------------------------------------
    # TABLE 1: Regression quality — mean±std per (method, problem, variant)
    # -----------------------------------------------------------------------
    section("TABLE 1 — Regression Quality: mean ± std across seeds")

    col_w = 22
    hdr = (
        f"{'Method':<8} {'Problem':<18} {'Variant':<10}"
        f"{'N':>4}  "
        f"{'R² test':>{col_w}}  "
        f"{'NRMSE test':>{col_w}}  "
        f"{'Complexity':>{col_w}}  "
        f"{'Sol.Rec%':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for method in METHODS:
        for problem in PROBLEMS:
            for variant in VARIANTS:
                key = (method, problem, variant)
                if key not in metrics:
                    print(f"  {method:<8} {problem:<18} {variant:<10}  MISSING")
                    continue
                m = metrics[key]
                n = len(m["r2_test"])
                r2_m, r2_s = np.nanmean(m["r2_test"]), np.nanstd(m["r2_test"], ddof=1)
                nrmse_m, nrmse_s = np.nanmean(m["nrmse_test"]), np.nanstd(m["nrmse_test"], ddof=1)
                cplx_m, cplx_s = (
                    np.nanmean(m["model_complexity"]),
                    np.nanstd(m["model_complexity"], ddof=1),
                )
                sol_pct = 100.0 * np.nanmean(m["solution_recovered"])

                print(
                    f"  {method:<8} {problem:<18} {variant:<10}"
                    f"{n:>4}  "
                    f"{ms(r2_m, r2_s):>{col_w}}  "
                    f"{ms(nrmse_m, nrmse_s):>{col_w}}  "
                    f"{ms(cplx_m, cplx_s):>{col_w}}  "
                    f"{sol_pct:>8.1f}%"
                )
        print()

    # -----------------------------------------------------------------------
    # TABLE 2: Search space reduction (isalsr only)
    # -----------------------------------------------------------------------
    section("TABLE 2 — Search Space Reduction (IsalSR variant only): mean ± std")

    hdr2 = (
        f"{'Method':<8} {'Problem':<18}"
        f"{'N':>4}  "
        f"{'Emp. RF':>{col_w}}  "
        f"{'Redundancy Rate':>{col_w}}  "
        f"{'Total DAGs':>14}  "
        f"{'Unique DAGs':>14}"
    )
    print(hdr2)
    print("-" * len(hdr2))

    for method in METHODS:
        for problem in PROBLEMS:
            key = (method, problem, "isalsr")
            if key not in metrics:
                print(f"  {method:<8} {problem:<18}  MISSING")
                continue
            m = metrics[key]
            n = len(m["empirical_reduction_factor"])
            rf_m = np.nanmean(m["empirical_reduction_factor"])
            rf_s = np.nanstd(m["empirical_reduction_factor"], ddof=1)
            rr_m = np.nanmean(m["redundancy_rate"])
            rr_s = np.nanstd(m["redundancy_rate"], ddof=1)
            total_m = np.nanmean(m["total_dags_explored"])
            unique_m = np.nanmean(m["unique_canonical_dags"])

            print(
                f"  {method:<8} {problem:<18}"
                f"{n:>4}  "
                f"{ms(rf_m, rf_s):>{col_w}}  "
                f"{ms(rr_m, rr_s):>{col_w}}  "
                f"{total_m:>14.0f}  "
                f"{unique_m:>14.0f}"
            )
        print()

    # -----------------------------------------------------------------------
    # TABLE 3: Computational overhead (isalsr variant)
    # -----------------------------------------------------------------------
    section(
        "TABLE 3 — Computational Overhead (IsalSR variant): overhead_time_s / wall_clock_total_s"
    )

    hdr3 = (
        f"{'Method':<8} {'Problem':<18}"
        f"{'N':>4}  "
        f"{'Overhead %':>{col_w}}  "
        f"{'Canon. time (s)':>{col_w}}  "
        f"{'Wall clock (s)':>{col_w}}  "
        f"{'Eval time (s)':>{col_w}}"
    )
    print(hdr3)
    print("-" * len(hdr3))

    for method in METHODS:
        total_overhead_pct: list[float] = []
        for problem in PROBLEMS:
            key = (method, problem, "isalsr")
            if key not in metrics:
                print(f"  {method:<8} {problem:<18}  MISSING")
                continue
            m = metrics[key]
            n = len(m["overhead_time_s"])

            # Overhead % per seed, then average
            overhead_pct = 100.0 * m["overhead_time_s"] / (m["wall_clock_total_s"] + 1e-12)
            oh_m = np.nanmean(overhead_pct)
            oh_s = np.nanstd(overhead_pct, ddof=1)

            canon_m = np.nanmean(m["canonicalization_runtime_s"])
            wall_m = np.nanmean(m["wall_clock_total_s"])
            eval_m = np.nanmean(m["evaluation_time_s"])

            total_overhead_pct.extend(overhead_pct.tolist())

            print(
                f"  {method:<8} {problem:<18}"
                f"{n:>4}  "
                f"{ms(oh_m, oh_s):>{col_w}}  "
                f"{canon_m:>{col_w}.1f}  "
                f"{wall_m:>{col_w}.1f}  "
                f"{eval_m:>{col_w}.1f}"
            )

        grand_oh = np.nanmean(total_overhead_pct)
        print(f"\n  {method.upper()} GRAND MEAN OVERHEAD: {grand_oh:.2f}%\n")

    # -----------------------------------------------------------------------
    # TABLE 4: Statistical tests (paired, per method, Holm-Bonferroni)
    # -----------------------------------------------------------------------
    section("TABLE 4 — Paired Statistical Tests: IsalSR vs Baseline (r2_test)")
    print(
        "  Normality: Shapiro-Wilk on differences (α=0.05). Test: t-test if normal, Wilcoxon otherwise."
    )
    print("  Correction: Holm-Bonferroni within each method (10 problems).")
    print()

    hdr4 = (
        f"{'Method':<8} {'Problem':<18}"
        f"{'N_paired':>9}  "
        f"{'Test':>10}  "
        f"{'Stat':>10}  "
        f"{'p (raw)':>12}  "
        f"{'p (HB)':>12}  "
        f"{'Reject':>7}  "
        f"{'Cohen d [95% CI]':<35}  "
        f"{'Δ R² (mean)':>14}"
    )
    print(hdr4)
    print("-" * len(hdr4))

    for method in METHODS:
        p_values: list[float] = []
        test_results: list[tuple] = []
        problems_with_data: list[str] = []

        for problem in PROBLEMS:
            b_key = (method, problem, "baseline")
            i_key = (method, problem, "isalsr")

            if b_key not in metrics or i_key not in metrics:
                continue

            mb = metrics[b_key]
            mi = metrics[i_key]

            # Match seeds: find seeds present in both; use seed order from run metadata
            # Since runs are sorted by path (seed_XX) in load_run_logs, and seeds may differ,
            # we pair by index (assumes same seed ordering for both variants within a problem)
            n_b, n_i = len(mb["r2_test"]), len(mi["r2_test"])
            n_paired = min(n_b, n_i)

            if n_paired < 3:
                continue

            b_arr = mb["r2_test"][:n_paired]
            i_arr = mi["r2_test"][:n_paired]

            # Remove nan-pairs
            valid = ~(np.isnan(b_arr) | np.isnan(i_arr))
            b_arr, i_arr = b_arr[valid], i_arr[valid]

            if len(b_arr) < 3:
                continue

            test_name, stat, p = paired_test(b_arr, i_arr)
            d, lo, hi = cohens_d_bootstrap(i_arr, b_arr)  # isalsr - baseline

            p_values.append(p)
            test_results.append(
                (
                    problem,
                    n_paired,
                    test_name,
                    stat,
                    p,
                    d,
                    lo,
                    hi,
                    float(np.mean(i_arr) - np.mean(b_arr)),
                )
            )
            problems_with_data.append(problem)

        # Apply Holm-Bonferroni
        reject_flags = holm_bonferroni(p_values)

        for i, (problem, n_paired, test_name, stat, p, d, lo, hi, delta_r2) in enumerate(
            test_results
        ):
            reject_str = "YES *" if reject_flags[i] else "no"
            # Adjusted threshold for display
            rank = sorted(range(len(p_values)), key=lambda j: p_values[j]).index(i)
            p_adj_thresh = ALPHA / (len(p_values) - rank)
            # Compute adjusted p (Holm): p_adj = min(1, max over all j<=i of p_j*(m-rank_j+1))
            # Standard: present corrected threshold, not adjusted p, for clarity
            print(
                f"  {method:<8} {problem:<18}"
                f"{n_paired:>9}  "
                f"{test_name:>10}  "
                f"{stat:>10.4f}  "
                f"{fmt_p(p):>12}  "
                f"{p_adj_thresh:>12.4f}  "
                f"{reject_str:>7}  "
                f"{fmt_d(d, lo, hi):<35}  "
                f"{delta_r2:>+14.4f}"
            )
        print()

    # -----------------------------------------------------------------------
    # TABLE 5: Summary — per-method aggregates
    # -----------------------------------------------------------------------
    section("TABLE 5 — Method-Level Summary")

    for method in METHODS:
        print(f"\n  {method.upper()}")
        print(f"  {'Metric':<35} {'Value'}")
        print(f"  {'-' * 60}")

        all_r2_base, all_r2_isal = [], []
        all_rf, all_rr = [], []
        all_oh_pct = []
        n_sig_improve, n_sig_degrade, n_total_tests = 0, 0, 0

        for problem in PROBLEMS:
            b_key = (method, problem, "baseline")
            i_key = (method, problem, "isalsr")

            if b_key in metrics:
                all_r2_base.extend(metrics[b_key]["r2_test"].tolist())
            if i_key in metrics:
                all_r2_isal.extend(metrics[i_key]["r2_test"].tolist())
                rf_vals = metrics[i_key]["empirical_reduction_factor"]
                rr_vals = metrics[i_key]["redundancy_rate"]
                all_rf.extend(rf_vals[~np.isnan(rf_vals)].tolist())
                all_rr.extend(rr_vals[~np.isnan(rr_vals)].tolist())
                oh = (
                    100.0
                    * metrics[i_key]["overhead_time_s"]
                    / (metrics[i_key]["wall_clock_total_s"] + 1e-12)
                )
                all_oh_pct.extend(oh[~np.isnan(oh)].tolist())

            # Recompute tests for summary counts
            if b_key in metrics and i_key in metrics:
                mb = metrics[b_key]
                mi = metrics[i_key]
                n_paired = min(len(mb["r2_test"]), len(mi["r2_test"]))
                if n_paired >= 3:
                    b_arr = mb["r2_test"][:n_paired]
                    i_arr = mi["r2_test"][:n_paired]
                    valid = ~(np.isnan(b_arr) | np.isnan(i_arr))
                    b_arr, i_arr = b_arr[valid], i_arr[valid]
                    if len(b_arr) >= 3:
                        _, _, p = paired_test(b_arr, i_arr)
                        n_total_tests += 1
                        if p < ALPHA and np.mean(i_arr) > np.mean(b_arr):
                            n_sig_improve += 1
                        elif p < ALPHA and np.mean(i_arr) < np.mean(b_arr):
                            n_sig_degrade += 1

        def _ms(arr: list[float], fmt: str = ".4f") -> str:
            a = np.array(arr, dtype=float)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return "N/A"
            return f"{np.mean(a):{fmt}} ± {np.std(a, ddof=1):{fmt}}"

        print(f"  {'Mean R² test (baseline)':<35} {_ms(all_r2_base)}")
        print(f"  {'Mean R² test (isalsr)':<35} {_ms(all_r2_isal)}")
        print(f"  {'Mean Emp. RF (isalsr)':<35} {_ms(all_rf, '.4f')}")
        print(f"  {'Mean Redundancy Rate (isalsr)':<35} {_ms(all_rr, '.4f')}")
        print(f"  {'Mean Overhead % (isalsr)':<35} {_ms(all_oh_pct, '.2f')}")
        print(f"  {'Problems tested (n)':<35} {n_total_tests}")
        print(f"  {'Sig. improvements (raw α=0.05)':<35} {n_sig_improve}/{n_total_tests}")
        print(f"  {'Sig. degradations  (raw α=0.05)':<35} {n_sig_degrade}/{n_total_tests}")

    # -----------------------------------------------------------------------
    # TABLE 6: Solution recovery rates
    # -----------------------------------------------------------------------
    section("TABLE 6 — Solution Recovery Rate (% of seeds)")

    hdr6 = f"{'Problem':<20}" + "".join(f"  {m + '/' + v:<18}" for m in METHODS for v in VARIANTS)
    print(hdr6)
    print("-" * len(hdr6))

    for problem in PROBLEMS:
        row = f"  {problem:<18}"
        for method in METHODS:
            for variant in VARIANTS:
                key = (method, problem, variant)
                if key not in metrics:
                    row += f"  {'N/A':<18}"
                else:
                    pct = 100.0 * np.nanmean(metrics[key]["solution_recovered"])
                    n = len(metrics[key]["solution_recovered"])
                    row += f"  {pct:5.1f}% (n={n})      "
        print(row)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
