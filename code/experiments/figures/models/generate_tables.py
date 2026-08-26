"""Generate LaTeX tables for IsalSR model validation results.

Produces:
  Table 1: Unified three-axis summary + overhead (one row per method x benchmark)
  Table 2: Per-problem R² comparison (full 22-row breakdown, one per method)
  Table K: Bingo k-range overhead breakdown (for Discussion section)
  Table S: Supplementary per-problem statistics (one per method, TPAMI-ready)

Usage:
    python -m experiments.figures.models.generate_tables \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.models.io_utils import load_all_run_logs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Data extraction helpers
# ======================================================================

# Arm directory -> column prefix. ``hash`` is the Naive-Hash deduplication arm
# of the three-arm C2 campaign; a two-arm root simply has no such directory and
# every ``hs_*`` key stays absent, so the tables fall back to two columns.
_VARIANT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("baseline", "bl"),
    ("hash", "hs"),
    ("isalsr", "is"),
)

# Arms that deduplicate, and therefore carry search-space and overhead fields.
_DEDUP_PREFIXES: frozenset[str] = frozenset({"hs", "is"})


def _has_hash_arm(data: Mapping[str, Mapping[str, Mapping[int, float]]]) -> bool:
    """Report whether any problem in a benchmark has Naive-Hash run logs.

    Args:
        data: Output of :func:`_load_paired_metrics`.

    Returns:
        True when at least one problem carries hash-arm R^2 values.
    """
    return any(d.get("hs_r2_test") for d in data.values())


def _load_paired_metrics(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> dict[str, dict[str, list[float]]]:
    """Load per-seed metrics for every arm present, keyed by problem.

    Returns:
        {problem: {
            "bl_r2_test": [...], "is_r2_test": [...], "hs_r2_test": [...],
            "bl_r2_train": [...], "is_r2_train": [...], ...
            "bl_nrmse_test": [...], "is_nrmse_test": [...], ...
            "bl_wall": [...], "is_wall": [...], ...
            "bl_search": [...], "is_search": [...], ...
            "bl_complexity": [...], "is_complexity": [...], ...
            "bl_solution": [...], "is_solution": [...], ...
            "is_rf": [...], "is_redundancy": [...],
            "is_overhead_pct": [...], "is_per_dag_ms": [...],
            "is_canon_ms": [...], "is_eval_ms": [...],
            "is_max_k": [...],
        }}

        The ``hs_*`` keys are present only for a three-arm results root; the
        deduplication keys (``rf``, ``redundancy``, ``overhead_pct``,
        ``per_dag_ms``, ``canon_ms``, ``eval_ms``, ``max_k``) exist for both
        ``is_`` and ``hs_``.
    """
    bench_dir = results_dir / method / benchmark
    data: dict[str, dict[str, dict[int, float]]] = {}
    if not bench_dir.exists():
        return data

    def _clip01(x: float) -> float:
        """Clip R^2 to [0, 1], preserving NaN.

        ``min(max(nan, 0.0), 1.0)`` returns NaN in Python -- the clip is a no-op
        on a missing observation rather than mapping it to a bound. Made
        explicit here because the implicit behaviour hid the defect.
        """
        return float("nan") if not math.isfinite(x) else min(max(x, 0.0), 1.0)

    for prob_dir in sorted(bench_dir.iterdir()):
        if not prob_dir.is_dir():
            continue
        d: dict[str, dict[int, float]] = defaultdict(dict)

        for variant, prefix in _VARIANT_PREFIXES:
            vdir = prob_dir / variant
            if not vdir.exists():
                continue
            for rl in load_all_run_logs(vdir):
                # Key every metric by seed: pairing is by seed number, never by
                # list position. A campaign with missing cells has unequal and
                # non-aligned seed sets across the two arms (T08 / R2.7).
                s = rl.metadata.seed
                rec = rl.regression
                d[f"{prefix}_r2_test"][s] = _clip01(rec.r2_test)
                d[f"{prefix}_r2_train"][s] = _clip01(rec.r2_train)
                d[f"{prefix}_nrmse_test"][s] = rec.nrmse_test
                d[f"{prefix}_wall"][s] = rl.time.wall_clock_total_s
                d[f"{prefix}_search"][s] = rl.time.wall_clock_search_only_s
                d[f"{prefix}_complexity"][s] = float(rec.model_complexity)
                # solution_recovered is None when the SymPy equivalence check
                # exceeded its budget: undetermined, not "not recovered".
                d[f"{prefix}_solution"][s] = (
                    float("nan")
                    if rec.solution_recovered is None
                    else float(rec.solution_recovered)
                )

                if prefix in _DEDUP_PREFIXES:
                    ss = rl.search_space
                    t = rl.time
                    d[f"{prefix}_rf"][s] = ss.empirical_reduction_factor
                    d[f"{prefix}_redundancy"][s] = ss.redundancy_rate
                    w = t.wall_clock_total_s
                    n = ss.total_dags_explored
                    if w > 0:
                        d[f"{prefix}_overhead_pct"][s] = t.overhead_time_s / w * 100
                    if n > 0:
                        d[f"{prefix}_per_dag_ms"][s] = t.canonicalization_runtime_s / n * 1000
                        d[f"{prefix}_eval_ms"][s] = t.evaluation_time_s / n * 1000
                    d[f"{prefix}_canon_ms"][s] = t.canonicalization_runtime_s
                    d[f"{prefix}_max_k"][s] = float(ss.max_internal_nodes_seen)

        if d:
            data[prob_dir.name] = dict(d)

    return data


def _nanmean(values: Sequence[float]) -> float:
    """Mean over the finite entries, or NaN if there are none.

    A NaN or inf metric is a *missing observation* -- an expression that is
    well-defined on the training domain but undefined on part of the test
    domain (``log`` of a negative argument, ``exp`` overflow). Excluding it
    shrinks the effective seed count; it must never propagate into the cell,
    and it must never be silently replaced by 0.0, which would count an
    undefined result as a total failure.

    This is the same convention ``analyzer/aggregation.py`` applies via
    ``np.nanmean``. The two pipelines must agree.
    """
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _pair_by_seed(
    bl: Mapping[int, float],
    is_: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Align two per-seed metric maps into paired arrays.

    Pairing is by **seed number**, never by list position: a campaign with a
    missing cell has unequal, non-aligned seed sets, and truncating both lists
    to ``min(len(a), len(b))`` pairs different seeds with each other from the
    first gap onwards. Pairs where either side is non-finite are dropped
    (pairwise deletion), which is the policy reported in the manuscript.

    Returns:
        (baseline_values, isalsr_values), index-aligned and finite.
    """
    common = sorted(set(bl) & set(is_))
    b = np.array([bl[s] for s in common], dtype=float)
    i = np.array([is_[s] for s in common], dtype=float)
    keep = np.isfinite(b) & np.isfinite(i)
    return b[keep], i[keep]


def _paired_test(
    bl: Mapping[int, float],
    is_: Mapping[int, float],
) -> tuple[float, float]:
    """Run paired test (t-test or Wilcoxon) and return (cohens_d, p_value).

    Returns ``(nan, nan)`` when fewer than three seeds are paired. Returning
    ``(0.0, 1.0)`` there would be indistinguishable from a genuine null result.
    An exception raised by SciPy returns ``p = nan`` for the same reason; the
    NaN then propagates through the table formatter instead of masquerading as
    a decided null.

    A vector of identical paired values is a genuine null and returns
    ``(0.0, 1.0)`` explicitly, since no test is defined on it. Ties inside the
    signed-rank test are kept via ``zero_method="zsplit"`` rather than dropped,
    matching the CPDT tie policy (Pratt 1959; Demsar 2006).
    """
    bl_a, is_a = _pair_by_seed(bl, is_)
    if len(bl_a) < 3:
        return float("nan"), float("nan")
    diff = is_a - bl_a
    if np.all(diff == 0):
        return 0.0, 1.0
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 1e-10 else 0.0
    try:
        _, sw_p = sp_stats.shapiro(diff)
        if sw_p > 0.05:
            _, p = sp_stats.ttest_rel(bl_a, is_a)
        else:
            res = sp_stats.wilcoxon(bl_a, is_a, zero_method="zsplit")
            p = res.pvalue
    except Exception:  # noqa: BLE001
        p = float("nan")
    return d, float(p)


def _cohens_d_with_ci(
    bl: Mapping[int, float],
    is_: Mapping[int, float],
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Cohen's d (paired) with bootstrap 95% CI, pairing by seed."""
    bl_a, is_a = _pair_by_seed(bl, is_)
    n = len(bl_a)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    diff = is_a - bl_a
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 1e-10 else 0.0

    rng = np.random.default_rng(seed)
    boot_ds = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        s = np.std(sample, ddof=1)
        boot_ds[b] = np.mean(sample) / s if s > 1e-10 else 0.0
    ci_lo = float(np.percentile(boot_ds, 2.5))
    ci_hi = float(np.percentile(boot_ds, 97.5))
    return d, ci_lo, ci_hi


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


# ======================================================================
# Problem labels
# ======================================================================

_PROBLEM_LABELS = {
    # Nguyen
    "nguyen_1": "N-1",
    "nguyen_2": "N-2",
    "nguyen_3": "N-3",
    "nguyen_4": "N-4",
    "nguyen_5": "N-5",
    "nguyen_6": "N-6",
    "nguyen_7": "N-7",
    "nguyen_8": "N-8",
    "nguyen_9": "N-9",
    "nguyen_10": "N-10",
    "nguyen_11": "N-11",
    "nguyen_12": "N-12",
    # Feynman
    "i.6.20a": "I.6.20a",
    "i.12.1": "I.12.1",
    "i.12.4": "I.12.4",
    "i.14.3": "I.14.3",
    "i.25.13": "I.25.13",
    "i.34.27": "I.34.27",
    "i.39.10": "I.39.10",
    "i.48.20": "I.48.20",
    "i.10.7": "I.10.7",
    "ii.3.24": "II.3.24",
    # Hard
    "i.15.10": "I.15.10",
    "i.30.3": "I.30.3",
    "i.37.4": "I.37.4",
    "ii.11.27": "II.11.27",
    "iii.17.37": "III.17.37",
    "keijzer_6": "Keij-6",
    "korns_12": "Korns-12",
    "pagie_1": "Pagie-1",
    "vladislavleva_2": "Vlad-2",
    "vladislavleva_4": "Vlad-4",
    # Structural
    "i.16.6": "I.16.6",
    "i.29.16": "I.29.16",
    "i.50.26": "I.50.26",
    "ii.11.28": "II.11.28",
    "iii.14.14": "III.14.14",
    "keijzer_11": "Keij-11",
    "liv_14": "Liv-14",
    "r2": "R2",
    "r3": "R3",
    "vlad_7": "Vlad-7",
}


def _latex_escape(text: str) -> str:
    """Escape the LaTeX specials that appear in our identifiers.

    Only ``_`` occurs in practice (suite and problem directory names), but the
    others are cheap to cover and a suite named with one would fail the same way.

    Args:
        text: Raw identifier.

    Returns:
        The identifier safe to typeset outside math mode.
    """
    for char in ("\\", "_", "%", "&", "#", "$"):
        text = text.replace(char, "\\" + char) if char != "\\" else text
    return text


def _problem_label(problem: str) -> str:
    """Return the display label for a problem directory, LaTeX-safe.

    ``_PROBLEM_LABELS`` covers the D1 suites. Anything outside it -- the whole
    T05 D2 extension, whose directory names carry underscores (``strogatz_vdp1``,
    ``liv_19``) -- previously fell through raw, and a bare ``_`` outside math
    mode aborts pdflatex. The tables therefore *emitted* cleanly and then failed
    to compile, on exactly the 18 rows the coverage extension added. Check E4.

    Args:
        problem: Problem directory name.

    Returns:
        The mapped label, or the raw name with underscores escaped.
    """
    label = _PROBLEM_LABELS.get(problem)
    if label is not None:
        return label
    return _latex_escape(problem)


def _load_cpdt(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> dict | None:
    """Load CPDT results from analysis directory."""
    import json

    cpdt_path = results_dir / "analysis" / f"cross_problem_dominance_{method}_{benchmark}.json"
    if not cpdt_path.exists():
        return None
    with open(cpdt_path) as f:
        return json.load(f)


# Cell printed where a CPDT p-value does not exist. Two distinct reasons, both
# rendered as an em-dash rather than a number: the contrast is descriptive by
# policy (rho against a baseline that never merges), or the test could not run.
_NO_P_CELL = "---"

# Cell printed for a quantity that is undefined because too few observations
# remain. Matches the dagger convention used by the per-problem rows.
_UNDEFINED_CELL = "$\\dagger$"


def _fmt_cpdt_p(p: float) -> str:
    """Format a CPDT p-value for LaTeX with significance stars.

    A non-finite p is an absent test, not a number: it is rendered as a dagger
    so that ``nan`` can never reach the typeset table.

    Args:
        p: The p-value to render.

    Returns:
        A LaTeX cell.
    """
    if not math.isfinite(p):
        return _UNDEFINED_CELL
    if p < 0.001:
        return "$<$0.001$^{***}$"
    sig = ""
    if p < 0.01:
        sig = "$^{**}$"
    elif p < 0.05:
        sig = "$^{*}$"
    return f"${p:.3f}${sig}"


def _fmt_cpdt_d(d: float) -> str:
    """Format a CPDT Cohen's d, never emitting ``nan``.

    Args:
        d: Effect size.

    Returns:
        A LaTeX cell.
    """
    return f"${d:+.2f}$" if math.isfinite(d) else _UNDEFINED_CELL


def _cpdt_primary_p(entry: Mapping[str, object], *, table: str, method: str) -> float:
    """Holm-adjusted p of a primary-contrast CPDT entry, with a raw fallback.

    With three arms the CPDT runs three contrasts (isalsr vs baseline, hash vs
    baseline, isalsr vs hash) and Holm-corrects across them, so the headline
    tables must print the corrected value. A legacy two-arm payload carries no
    ``p_value_holm``; there the Holm family has size 1 and the corrected value
    equals the raw one-sided p by construction, so falling back is exact and is
    only logged.

    Args:
        entry: One metric block of a ``cross_problem_dominance_*.json`` payload,
            e.g. its ``r2_test`` entry.
        table: Table name, for the fallback log record.
        method: Method name, for the fallback log record.

    Returns:
        The Holm-adjusted p when the payload carries a finite one, otherwise the
        raw one-sided p, otherwise NaN.
    """
    holm = entry.get("p_value_holm")
    if isinstance(holm, (int, float)) and not isinstance(holm, bool) and math.isfinite(holm):
        return float(holm)

    log.warning(
        "%s (%s): the CPDT payload carries no Holm-adjusted p; printing the raw "
        "one-sided p instead. Expected only for a legacy two-arm input, where the "
        "two are equal by construction (Holm family of size 1).",
        table,
        method,
    )
    raw = entry.get("p_value_one_sided", float("nan"))
    return float(raw) if isinstance(raw, (int, float)) else float("nan")


def _cpdt_rho_cells(cpdt: Mapping[str, object]) -> tuple[str, str]:
    """Effect-size and p-value cells for the reduction-factor CPDT footer.

    Under the contrast policy decided 2026-08-04, rho is inferential only on
    the ``hash -> isalsr`` contrast: the baseline arm never merges, so rho is 1
    by construction there and its p-value is withheld. The footer therefore
    reports the tested contrast when a hash arm exists, and falls back to the
    descriptive primary contrast -- effect size only, no p -- when it does not.

    Args:
        cpdt: A decoded ``cross_problem_dominance_*.json`` payload.

    Returns:
        (Cohen's d cell, p-value cell). Neither ever contains ``nan``.
    """
    contrasts = cpdt.get("contrasts")
    tested = None
    if isinstance(contrasts, Mapping):
        candidate = contrasts.get("isalsr_vs_hash")
        if isinstance(candidate, Mapping):
            entry = candidate.get("empirical_reduction_factor")
            if isinstance(entry, Mapping) and "error" not in entry:
                tested = entry

    if tested is not None:
        holm = tested.get("p_value_holm")
        p = (
            holm
            if isinstance(holm, (int, float))
            else tested.get("p_value_one_sided", float("nan"))
        )
        return _fmt_cpdt_d(float(tested.get("cohens_d", float("nan")))), _fmt_cpdt_p(float(p))

    primary = cpdt.get("empirical_reduction_factor")
    if isinstance(primary, Mapping) and "error" not in primary:
        return _fmt_cpdt_d(float(primary.get("cohens_d", float("nan")))), _NO_P_CELL
    return _UNDEFINED_CELL, _NO_P_CELL


# ======================================================================
# Table 1: Unified Three-Axis + Overhead Summary
# ======================================================================


def generate_table1(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Unified summary table merging three-axis and overhead breakdown.

    One row per (method x benchmark). Columns cover search space reduction,
    regression quality, and computational overhead in a single view. On a
    three-arm results root the $R^2$ cell carries BL/HS/IS and the $\\rho$ cell
    carries IS/HS; on a two-arm root both cells are unchanged.
    """
    rows: list[str] = []
    loaded = {
        (method, benchmark): _load_paired_metrics(results_dir, method, benchmark)
        for method in methods
        for benchmark in benchmarks
    }
    show_hash = any(_has_hash_arm(d) for d in loaded.values())

    for method in methods:
        for benchmark in benchmarks:
            data = loaded[(method, benchmark)]
            if not data:
                continue

            # Aggregate across problems
            all_rf, all_rr, all_oh = [], [], []
            all_hs_rf: list[float] = []
            all_canon_ms, all_eval_ms = [], []
            bl_r2s, is_r2s = [], []
            hs_r2s: list[float] = []
            n_sig = 0
            n_prob = 0
            speedups = []
            p_values = []

            for prob, d in data.items():
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                n_prob += 1
                bl_r2s.append(_nanmean(d["bl_r2_test"].values()))
                is_r2s.append(_nanmean(d["is_r2_test"].values()))
                if d.get("hs_r2_test"):
                    hs_r2s.append(_nanmean(d["hs_r2_test"].values()))
                if d.get("hs_rf"):
                    all_hs_rf.extend(d["hs_rf"].values())
                _, p = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                p_values.append(p)

                if d.get("is_rf"):
                    all_rf.extend(d["is_rf"].values())
                if d.get("is_redundancy"):
                    all_rr.extend(d["is_redundancy"].values())
                if d.get("is_overhead_pct"):
                    all_oh.extend(d["is_overhead_pct"].values())
                if d.get("is_per_dag_ms"):
                    all_canon_ms.extend(d["is_per_dag_ms"].values())
                if d.get("is_eval_ms"):
                    all_eval_ms.extend(d["is_eval_ms"].values())
                if d.get("bl_search") and d.get("is_search"):
                    bl_s = _nanmean(d["bl_search"].values())
                    is_s = _nanmean(d["is_search"].values())
                    if is_s > 0:
                        speedups.append(bl_s / is_s)

            # CPDT (primary) — load pre-computed cross-problem dominance test
            cpdt = _load_cpdt(results_dir, method, benchmark)
            if cpdt and "r2_test" in cpdt and "error" not in cpdt["r2_test"]:
                cpdt_r2 = cpdt["r2_test"]
                cpdt_p = _cpdt_primary_p(cpdt_r2, table="Table 1", method=method)
                cpdt_d = cpdt_r2["cohens_d"]
            else:
                # Fallback: Holm-corrected count
                adjusted = _holm_bonferroni(p_values) if p_values else []
                n_sig_holm = sum(1 for p in adjusted if p < 0.05)
                cpdt_p = float("nan")
                cpdt_d = float("nan")

            rf_mean = _nanmean(all_rf) if all_rf else 1.0
            rf_std = float(np.nanstd(all_rf, ddof=1)) if len(all_rf) > 1 else 0.0
            rr_mean = _nanmean(all_rr) * 100 if all_rr else 0.0
            oh_mean = _nanmean(all_oh) if all_oh else 0.0
            s_mean = _nanmean(speedups) if speedups else 1.0
            bl_r2_mean = float(np.nanmean(bl_r2s)) if bl_r2s else 0.0
            is_r2_mean = float(np.nanmean(is_r2s)) if is_r2s else 0.0
            canon_mean = _nanmean(all_canon_ms) if all_canon_ms else 0.0
            eval_mean = _nanmean(all_eval_ms) if all_eval_ms else 0.0
            ratio = canon_mean / eval_mean if eval_mean > 0 else float("inf")
            ratio_str = f"{ratio:.1f}" if ratio < 100 else f"{ratio:.0f}"

            method_label = method.upper()
            bench_label = _latex_escape(benchmark.capitalize())

            if np.isfinite(cpdt_p):
                cpdt_p_str = _fmt_cpdt_p(cpdt_p)
                cpdt_d_str = _fmt_cpdt_d(cpdt_d)
            else:
                cpdt_p_str = f"{n_sig_holm}/{n_prob}"
                cpdt_d_str = "--"

            # Three-arm cells: the hash arm shares the R^2 and rho columns
            # rather than adding new ones, so the tabular spec is untouched.
            if show_hash:
                hs_r2_mean = float(np.nanmean(hs_r2s)) if hs_r2s else float("nan")
                hs_r2_cell = f"${hs_r2_mean:.3f}$" if math.isfinite(hs_r2_mean) else _UNDEFINED_CELL
                r2_cell = f"${bl_r2_mean:.3f}$ / {hs_r2_cell} / ${is_r2_mean:.3f}$"
                hs_rf_mean = _nanmean(all_hs_rf) if all_hs_rf else float("nan")
                hs_rf_cell = f"${hs_rf_mean:.2f}$" if math.isfinite(hs_rf_mean) else _UNDEFINED_CELL
                rho_cell = f"${rf_mean:.2f} \\pm {rf_std:.2f}$ / {hs_rf_cell}"
            else:
                r2_cell = f"${bl_r2_mean:.3f}$ / ${is_r2_mean:.3f}$"
                rho_cell = f"${rf_mean:.2f} \\pm {rf_std:.2f}$"

            rows.append(
                f"    {method_label:<6} & {bench_label:<12} & {n_prob:>2} "
                f"& {rho_cell} "
                f"& ${rr_mean:.1f}\\%$ "
                f"& {r2_cell} "
                f"& {cpdt_d_str} & {cpdt_p_str} "
                f"& ${canon_mean:.3f}$ & ${eval_mean:.2f}$ "
                f"& ${ratio_str}$ "
                f"& ${oh_mean:.1f}\\%$ "
                f"& ${s_mean:.2f}$ \\\\"
            )

    hash_note = (
        "HS: Naive-Hash deduplication arm; $\\rho$ and $R^2$ list IS then HS. " if show_hash else ""
    )
    rho_header = "$\\rho$ (IS/HS)" if show_hash else "$\\rho$"
    r2_header = "$R^2$ (BL/HS/IS)" if show_hash else "$R^2$ (BL/IS)"
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Unified three-axis summary of \\IsalSR{} integration. "
        "$\\rho$: empirical reduction factor. "
        f"{hash_note}"
        "$d_{\\mathrm{CPDT}}$/$p_{\\mathrm{CPDT}}$: Cross-Problem Dominance Test "
        "(one-sided paired test across problems, treating each problem's mean $R^2$ "
        "as one observation; $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$). "
        "$p_{\\mathrm{CPDT}}$ is Holm-adjusted across the three CPDT contrasts. "
        "$T_{\\mathrm{canon}}$/$T_{\\mathrm{eval}}$: per-DAG "
        "canonicalization/evaluation cost (ms). "
        "OH: overhead. $S$: speedup.}\n"
        "\\label{tab:three_axis}\n"
        "\\begin{tabular}{@{}llc cc cc c rrr cc@{}}\n"
        "\\toprule\n"
        " & & & \\multicolumn{2}{c}{Search Space} "
        "& \\multicolumn{3}{c}{Regression Quality (CPDT)} "
        "& \\multicolumn{5}{c}{Computational Cost} \\\\\n"
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-13}\n"
        "Method & Benchmark & $n$ "
        f"& {rho_header} & Red. "
        f"& {r2_header} & $d$ & $p$ "
        "& $T_{\\mathrm{canon}}$ & $T_{\\mathrm{eval}}$ & Ratio "
        "& OH & $S$ \\\\\n"
        "\\midrule\n"
    )
    tex += "\n".join(rows) + "\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

    out = output_dir / "table1_three_axis_summary.tex"
    out.write_text(tex)
    log.info("Saved %s", out)


# ======================================================================
# Table 2: Per-Problem R² Comparison
# ======================================================================


def generate_table2(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Per-problem R² comparison with Cohen's d and Holm p-value.

    A three-arm results root adds a Naive-Hash (HS) column between BL and IS;
    a two-arm root emits the original two columns unchanged. ``d``, ``Delta``
    and the per-problem p-value always describe the BL -> IS contrast.
    """

    for method in methods:
        rows: list[str] = []
        prev_bench = ""
        per_bench_data = {b: _load_paired_metrics(results_dir, method, b) for b in benchmarks}
        show_hash = any(_has_hash_arm(d) for d in per_bench_data.values())

        for benchmark in benchmarks:
            data = per_bench_data[benchmark]
            if not data:
                continue

            # Collect p-values for Holm correction within this benchmark
            prob_stats: list[tuple[str, float, float, float, float, float]] = []
            for prob in sorted(data.keys()):
                d = data[prob]
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                bl_mean = _nanmean(d["bl_r2_test"].values())
                is_mean = _nanmean(d["is_r2_test"].values())
                hs_mean = _nanmean(d.get("hs_r2_test", {}).values())
                cohens_d, p_raw = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                prob_stats.append((prob, bl_mean, hs_mean, is_mean, cohens_d, p_raw))

            # Holm correction
            raw_ps = [s[5] for s in prob_stats]
            adjusted = _holm_bonferroni(raw_ps) if raw_ps else []

            for i, (prob, bl_mean, hs_mean, is_mean, cohens_d, _) in enumerate(prob_stats):
                p_adj = adjusted[i] if i < len(adjusted) else 1.0
                sig = ""
                if p_adj < 0.001:
                    sig = "$^{***}$"
                elif p_adj < 0.01:
                    sig = "$^{**}$"
                elif p_adj < 0.05:
                    sig = "$^{*}$"

                label = _problem_label(prob)

                # Add midrule between benchmarks
                if benchmark != prev_bench and rows:
                    rows.append("    \\midrule")
                    prev_bench = benchmark
                elif not rows:
                    prev_bench = benchmark

                delta = is_mean - bl_mean
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                hs_cell = ""
                if show_hash:
                    hs_cell = (
                        f"& ${hs_mean:.4f}$ " if math.isfinite(hs_mean) else f"& {_UNDEFINED_CELL} "
                    )
                rows.append(
                    f"    {label:<8} "
                    f"& ${bl_mean:.4f}$ {hs_cell}& ${is_mean:.4f}$ "
                    f"& ${delta_str}$ "
                    f"& ${cohens_d:+.2f}$ "
                    f"& ${p_adj:.3f}${sig} \\\\"
                )

        # Add CPDT summary rows (one per benchmark, plus pooled "all")
        arm_span = 3 if show_hash else 2
        rows.append("    \\midrule")
        for cpdt_bench in benchmarks + ["all"]:
            cpdt = _load_cpdt(results_dir, method, cpdt_bench)
            if cpdt and "r2_test" in cpdt and "error" not in cpdt["r2_test"]:
                cr = cpdt["r2_test"]
                cpdt_label = (
                    f"CPDT ({_latex_escape(cpdt_bench)})"
                    if cpdt_bench != "all"
                    else "\\textbf{CPDT (all)}"
                )
                n_p = cr["n_problems"]
                w = cr["n_wins"]
                t = cr["n_ties"]
                lo = cr["n_losses"]
                d_val = float(cr["cohens_d"])
                # Supplementary detail table: the RAW one-sided p lives here by
                # policy; Tables 1 and S carry the Holm-adjusted value.
                p_val = float(cr["p_value_one_sided"])
                wt_str = f"W{w}/T{t}/L{lo}"
                rows.append(
                    f"    {cpdt_label} & \\multicolumn{{{arm_span}}}{{c}}{{{wt_str}}} "
                    f"& $n$={n_p} "
                    f"& {_fmt_cpdt_d(d_val)} "
                    f"& {_fmt_cpdt_p(p_val)} \\\\"
                )

        method_label = method.upper()
        arm_spec = "rrr" if show_hash else "rr"
        arm_header = "& BL $R^2$ & HS $R^2$ & IS $R^2$ " if show_hash else "& BL $R^2$ & IS $R^2$ "
        hash_note = (
            "HS: Naive-Hash deduplication arm. "
            "$\\Delta$, $d$ and $p$ describe the BL versus IS contrast. "
            if show_hash
            else ""
        )
        tex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            f"\\caption{{Per-problem $R^2$ test comparison for {method_label}. "
            f"{hash_note}"
            f"Cohen's $d$: per-problem paired effect size. "
            f"$p$: Holm-adjusted. "
            f"Bottom rows: Cross-Problem Dominance Test (CPDT) — one-sided paired "
            f"test across problems. W/T/L: wins/ties/losses. "
            f"The CPDT $p$ in those rows is the raw one-sided value, uncorrected; "
            f"the main tables report it Holm-adjusted across the three CPDT "
            f"contrasts. "
            f"$^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.}}\n"
            f"\\label{{tab:r2_per_problem_{method}}}\n"
            f"\\begin{{tabular}}{{@{{}}l {arm_spec} r r r@{{}}}}\n"
            "\\toprule\n"
            f"Problem {arm_header}& $\\Delta$ & $d$ & $p_{{\\mathrm{{Holm}}}}$ \\\\\n"
            "\\midrule\n"
        )
        tex += "\n".join(rows) + "\n"
        tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

        out = output_dir / f"table2_r2_per_problem_{method}.tex"
        out.write_text(tex)
        log.info("Saved %s", out)


# ======================================================================
# Table K: Bingo k-Range Overhead (Discussion)
# ======================================================================


def generate_table_k_range(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Bingo k-range computational overhead breakdown for Discussion section.

    Stratifies overhead by expression complexity (max internal nodes k)
    to show how canonicalization cost scales with DAG size.
    """
    k_data: list[tuple[int, float, float]] = []
    for benchmark in benchmarks:
        data = _load_paired_metrics(results_dir, "bingo", benchmark)
        for prob, d in data.items():
            if not d.get("is_max_k") or not d.get("is_overhead_pct"):
                continue
            # Align the three series by seed. is_overhead_pct and is_per_dag_ms
            # are only recorded when their denominators are positive, so the
            # three maps can have different seed sets; zipping them positionally
            # would pair one seed's k with another seed's cost.
            mk_by_seed = d.get("is_max_k", {})
            oh_by_seed = d.get("is_overhead_pct", {})
            pd_by_seed = d.get("is_per_dag_ms", {})
            for s in sorted(set(mk_by_seed) & set(oh_by_seed) & set(pd_by_seed)):
                mk, oh, pd = mk_by_seed[s], oh_by_seed[s], pd_by_seed[s]
                if np.isfinite(mk) and np.isfinite(oh) and np.isfinite(pd):
                    k_data.append((int(mk), oh, pd))

    if not k_data:
        log.warning("No Bingo k-range data available")
        return

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Bingo canonicalization overhead stratified by expression "
        "complexity $k$ (maximum internal nodes). Overhead increases with $k$ "
        "due to the combinatorial nature of the canonical form computation, "
        "but remains bounded by the greedy WL-hash algorithm.}\n"
        "\\label{tab:k_range_overhead}\n"
        "\\begin{tabular}{@{}l rrr@{}}\n"
        "\\toprule\n"
        "$k$-range & $N$ & OH (\\%) & $T_{\\mathrm{canon}}$ (ms) \\\\\n"
        "\\midrule\n"
    )
    for lo, hi in [(0, 5), (5, 15), (15, 32)]:
        subset = [(oh, pd) for mk, oh, pd in k_data if lo <= mk < hi]
        if subset:
            ohs, pds = zip(*subset)
            tex += (
                f"    $[{lo}, {hi})$ & {len(subset)} "
                f"& ${_nanmean(ohs):.1f}\\%$ "
                f"& ${_nanmean(pds):.3f}$ \\\\\n"
            )
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = output_dir / "table_k_range_overhead.tex"
    out.write_text(tex)
    log.info("Saved %s", out)


# ======================================================================
# Table S: Supplementary Per-Problem Statistics
# ======================================================================


def _fmt_bold_underline(
    bl_val: float,
    is_val: float,
    fmt: str,
    higher_is_better: bool = True,
) -> tuple[str, str]:
    """Format two values with bold (best) and underline (worst).

    Compares formatted strings to avoid spurious bold/underline when both
    values display identically (e.g., both "0.0000" despite fp noise).

    A non-finite value is a **missing observation, not a value that wins**.
    Both ``bl > is`` and ``is > bl`` are ``False`` when either side is NaN, so
    an unguarded comparison falls through to the "IsalSR is better" branch and
    typesets the NaN bold -- which is how the submitted appendix marked a real
    ``R^2 = 0.9385`` as worse than a missing observation (T08 / R2.7).
    Missing cells are rendered as an em dash and carry no mark; the surviving
    finite value takes the bold.

    Returns LaTeX strings for (bl_formatted, is_formatted).
    """
    bl_ok = math.isfinite(bl_val)
    is_ok = math.isfinite(is_val)

    # Both missing: nothing to compare, nothing to mark.
    if not bl_ok and not is_ok:
        return "$\\dagger$", "$\\dagger$"

    # Exactly one missing: the finite side wins by default, never the NaN.
    if not is_ok:
        return f"$\\mathbf{{{bl_val:{fmt}}}}$", "$\\dagger$"
    if not bl_ok:
        return "$\\dagger$", f"$\\mathbf{{{is_val:{fmt}}}}$"

    bl_str = f"{bl_val:{fmt}}"
    is_str = f"{is_val:{fmt}}"

    # If they display the same, no bold/underline
    if bl_str == is_str:
        return f"${bl_str}$", f"${is_str}$"

    if higher_is_better:
        bl_better = bl_val > is_val
    else:
        bl_better = bl_val < is_val

    if bl_better:
        return f"$\\mathbf{{{bl_str}}}$", f"$\\underline{{{is_str}}}$"
    else:
        return f"$\\underline{{{bl_str}}}$", f"$\\mathbf{{{is_str}}}$"


def generate_table_supplementary(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Comprehensive per-problem supplementary table (one per method).

    Designed for TPAMI supplementary material. Columns:
      Quality:      R² test (BL/IS), NRMSE test (BL/IS)
      Effect size:  Cohen's d [95% CI], p_Holm
      Search space: ρ (mean ± std), Redundancy rate
      Computation:  T_total BL/IS (s), Overhead %

    Bold = better of BL/IS per metric per problem.
    Underline = worse of BL/IS per metric per problem.
    """

    for method in methods:
        all_rows: list[str] = []
        prev_bench = ""

        for benchmark in benchmarks:
            data = _load_paired_metrics(results_dir, method, benchmark)
            if not data:
                continue

            # Collect all problems for Holm correction across this benchmark
            prob_stats: list[
                tuple[
                    str,  # problem name
                    dict[str, dict[int, float]],  # per-seed data
                    float,  # raw p-value
                ]
            ] = []
            for prob in sorted(data.keys()):
                d = data[prob]
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                _, p_raw = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                prob_stats.append((prob, d, p_raw))

            # Holm is applied over the problems that yielded a defined p only.
            # A problem with too few paired seeds must not enter the family and
            # must not be silently assigned p = 1.
            testable = [i for i, s in enumerate(prob_stats) if math.isfinite(s[2])]
            adj_by_index: dict[int, float] = {}
            if testable:
                adj_vals = _holm_bonferroni([prob_stats[i][2] for i in testable])
                adj_by_index = dict(zip(testable, adj_vals))

            for i, (prob, d, _) in enumerate(prob_stats):
                p_adj = adj_by_index.get(i, float("nan"))

                # Midrule between benchmarks
                if benchmark != prev_bench and all_rows:
                    all_rows.append("    \\midrule")
                    prev_bench = benchmark
                elif not all_rows:
                    prev_bench = benchmark

                label = _problem_label(prob)

                # R² test. Pairwise deletion: a seed whose expression is
                # undefined on part of the test domain is a missing
                # observation, excluded from the mean rather than propagated
                # through it (T08 / R2.7).
                bl_r2 = _nanmean(d["bl_r2_test"].values())
                is_r2 = _nanmean(d["is_r2_test"].values())
                bl_r2_f, is_r2_f = _fmt_bold_underline(bl_r2, is_r2, ".4f", higher_is_better=True)

                # NRMSE test
                bl_nrmse = _nanmean(d.get("bl_nrmse_test", {}).values())
                is_nrmse = _nanmean(d.get("is_nrmse_test", {}).values())
                bl_nrmse_f, is_nrmse_f = _fmt_bold_underline(
                    bl_nrmse, is_nrmse, ".4f", higher_is_better=False
                )

                # Cohen's d with bootstrap CI
                cd, ci_lo, ci_hi = _cohens_d_with_ci(d["bl_r2_test"], d["is_r2_test"])
                if math.isfinite(cd):
                    d_str = f"${cd:+.2f}$\\,[${ci_lo:+.2f}$, ${ci_hi:+.2f}$]"
                else:
                    d_str = "$\\dagger$"

                # Effective paired seed count, after pairwise deletion. Reported
                # per row because it is not uniformly S for every problem.
                n_paired = len(_pair_by_seed(d["bl_r2_test"], d["is_r2_test"])[0])

                # p-value with significance stars
                if not math.isfinite(p_adj):
                    p_str = "$\\dagger$"
                else:
                    sig = ""
                    if p_adj < 0.001:
                        sig = "$^{***}$"
                    elif p_adj < 0.01:
                        sig = "$^{**}$"
                    elif p_adj < 0.05:
                        sig = "$^{*}$"
                    if p_adj < 0.001:
                        p_str = f"$<$0.001{sig}"
                    else:
                        p_str = f"${p_adj:.3f}${sig}"

                # Reduction factor
                rf_vals = list(d.get("is_rf", {}).values()) or [1.0]
                rf_mean = _nanmean(rf_vals)
                rf_std = float(np.nanstd(rf_vals, ddof=1)) if len(rf_vals) > 1 else 0.0
                rf_str = f"${rf_mean:.2f} \\pm {rf_std:.2f}$"

                # Redundancy rate
                rr = _nanmean(d.get("is_redundancy", {}).values() or [0.0]) * 100
                rr_str = f"${rr:.1f}\\%$"

                # Wall-clock time (bold lower = better)
                bl_t = _nanmean(d.get("bl_wall", {}).values() or [0.0])
                is_t = _nanmean(d.get("is_wall", {}).values() or [0.0])
                bl_t_f, is_t_f = _fmt_bold_underline(bl_t, is_t, ".1f", higher_is_better=False)

                # Overhead
                oh = _nanmean(d.get("is_overhead_pct", {}).values() or [0.0])
                oh_str = f"${oh:.1f}\\%$"

                # Where the effective paired seed count falls short of the
                # campaign's nominal S, annotate the row rather than let it
                # imply a full complement (T08 / R2.7).
                nominal_s = max(len(d["bl_r2_test"]), len(d["is_r2_test"]))
                row_label = label if n_paired >= nominal_s else f"{label}$^{{[{n_paired}]}}$"

                all_rows.append(
                    f"    {row_label:<8} "
                    f"& {bl_r2_f} & {is_r2_f} "
                    f"& {bl_nrmse_f} & {is_nrmse_f} "
                    f"& {d_str} & {p_str} "
                    f"& {rf_str} & {rr_str} "
                    f"& {bl_t_f} & {is_t_f} & {oh_str} \\\\"
                )

        # CPDT summary rows
        all_rows.append("    \\midrule")
        for cpdt_bench in benchmarks + ["all"]:
            cpdt = _load_cpdt(results_dir, method, cpdt_bench)
            if cpdt and "r2_test" in cpdt and "error" not in cpdt["r2_test"]:
                cr = cpdt["r2_test"]
                cpdt_label = (
                    f"CPDT ({_latex_escape(cpdt_bench)})"
                    if cpdt_bench != "all"
                    else "\\textbf{CPDT (all)}"
                )
                n_p = cr["n_problems"]
                w, t_cnt, lo = cr["n_wins"], cr["n_ties"], cr["n_losses"]
                d_val = float(cr["cohens_d"])
                p_val = _cpdt_primary_p(cr, table="Table S", method=method)
                # CPDT for rho: inferential only on the hash -> isalsr
                # contrast, descriptive against a baseline that never merges.
                rf_d_cell, rf_p_cell = _cpdt_rho_cells(cpdt)

                all_rows.append(
                    f"    {cpdt_label} "
                    f"& \\multicolumn{{2}}{{c}}{{W{w}/T{t_cnt}/L{lo}}} "
                    f"& \\multicolumn{{2}}{{c}}{{$n$={n_p}}} "
                    f"& {_fmt_cpdt_d(d_val)} & {_fmt_cpdt_p(p_val)} "
                    f"& {rf_d_cell} & {rf_p_cell} "
                    f"& \\multicolumn{{3}}{{c}}{{}} \\\\"
                )

        if not all_rows:
            continue

        method_label = method.upper()
        tex = (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\scriptsize\n"
            f"\\caption{{Per-problem comparison of native DAG representation "
            f"(BL) versus \\IsalSR{{}} canonicalization (IS) for "
            f"{method_label}. "
            f"Cohen's $d$: per-problem paired effect size with 95\\% bootstrap CI. "
            f"$p$: Holm-adjusted. "
            f"Bottom rows: Cross-Problem Dominance Test (CPDT) — "
            f"one-sided paired test treating each problem's mean as one observation, "
            f"Holm-adjusted across the three CPDT contrasts "
            f"($^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$). "
            f"$\\rho$: empirical reduction factor. Its CPDT $p$ is the "
            f"Naive-Hash versus \\IsalSR{{}} contrast, the only one for which a "
            f"test of $\\rho$ is meaningful: the native arm never merges, so "
            f"$\\rho=1$ there by construction and that column is reported "
            f"descriptively (---). "
            f"\\textbf{{Bold}}: better of BL/IS. "
            f"\\underline{{Underline}}: worse of BL/IS. "
            f"All statistics use pairwise deletion: a seed is included only if "
            f"both arms produced a finite value for it. "
            f"A superscript $[n]$ on a problem name gives the effective number "
            f"of paired seeds where it is below the nominal count, either "
            f"because a run did not complete or because the recovered "
            f"expression is undefined on part of the test domain. "
            f"$\\dagger$ marks a quantity that is undefined because too few "
            f"seeds remain; it is never a value that wins a comparison.}}\n"
            f"\\label{{tab:supplementary_{method}}}\n"
            "\\begin{tabular}{@{}l "
            "rr "  # R² test BL/IS
            "rr "  # NRMSE test BL/IS
            "l r "  # d [CI], p
            "r r "  # ρ, Red.
            "rr r"  # T_BL, T_IS, OH
            "@{}}\n"
            "\\toprule\n"
            " & \\multicolumn{2}{c}{$R^2$ test} "
            "& \\multicolumn{2}{c}{NRMSE test} "
            "& \\multicolumn{2}{c}{Effect size} "
            "& \\multicolumn{2}{c}{Search space} "
            "& \\multicolumn{3}{c}{Computation} \\\\\n"
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} "
            "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9} "
            "\\cmidrule(lr){10-12}\n"
            "Problem "
            "& BL & IS "
            "& BL & IS "
            "& $d$\\,[95\\% CI] & $p_{\\mathrm{Holm}}$ "
            "& $\\rho$ & Red. "
            "& $T_{\\mathrm{BL}}$ (s) & $T_{\\mathrm{IS}}$ (s) & OH \\\\\n"
            "\\midrule\n"
        )
        tex += "\n".join(all_rows) + "\n"
        tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

        out = output_dir / f"table_supplementary_{method}.tex"
        out.write_text(tex)
        log.info("Saved %s", out)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for IsalSR")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating tables from %s", results_dir)
    generate_table1(results_dir, methods, benchmarks, output_dir)
    generate_table2(results_dir, methods, benchmarks, output_dir)
    generate_table_k_range(results_dir, methods, benchmarks, output_dir)
    generate_table_supplementary(results_dir, methods, benchmarks, output_dir)
    log.info("All tables saved to %s", output_dir)


if __name__ == "__main__":
    main()
