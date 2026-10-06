"""Per-dataset paired statistics and the paired test across problems.

Every statistic here is computed by the main campaign's own functions, called
unchanged:

* :func:`experiments.models.analyzer.aggregation.compute_paired_stats` pairs the
  seeds of two arms of one dataset (R^2 clipped to [0, 1], seed-matched);
* :func:`experiments.models.analyzer.aggregation.apply_holm_correction` adjusts
  the per-dataset p-values within each contrast (descriptive only);
* :func:`experiments.models.analyze.run_cross_problem_dominance_test` runs the
  paired test across problems for every contrast and metric, with the
  alternative of each pair taken from ``CPDT_CONTRAST_POLICY`` and the
  Shapiro-Wilk selection between the one-sample t-test and the signed-rank
  test, exactly as for Table 3 of the manuscript.

The only difference from the main campaign's pipeline is where the outputs go.
The main pipeline writes per-problem paired statistics into the corpus; this
corpus is read-only, so they are computed in memory and written under the
analysis tree instead.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments.models.analyze import (
    CPDT_CONTRASTS,
    CPDT_PRIMARY_CONTRAST,
    run_cross_problem_dominance_test,
)
from experiments.models.analyzer.aggregation import (
    _CPDT_TIE_THRESHOLD,
    apply_holm_correction,
    compute_paired_stats,
)
from experiments.models.io_utils import load_all_run_logs, save_paired_stats
from experiments.models.schemas import PairedStats
from experiments.scripts.blackbox.config import BENCHMARK, DATASET_DIR

#: Contrasts and metrics flattened for the tables (Table 3 of the manuscript).
CONTRASTS: tuple[str, ...] = ("isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline")
METRICS: tuple[str, ...] = (
    "r2_test",
    "r2_train",
    "nrmse_test",
    "empirical_reduction_factor",
    "redundancy_rate",
)

#: SciPy evaluates the exact signed-rank distribution up to this many pairs.
EXACT_WILCOXON_MAX_N = 50


def paired_stats_by_contrast(
    corpus: Path,
    method: str,
    datasets: list[str],
    out_dir: Path,
) -> dict[str, list[PairedStats]]:
    """Compute the per-dataset paired statistics of every contrast.

    Args:
        corpus: Corpus root (read-only).
        method: Host, ``"udfs"`` or ``"bingo"``.
        datasets: Datasets admitted to this host's analysis by the audit.
        out_dir: Directory receiving ``<method>/<dataset>/<contrast file>``.

    Returns:
        Contrast name -> one PairedStats per dataset, in ``datasets`` order.
    """
    bench_dir = corpus / method / BENCHMARK
    out: dict[str, list[PairedStats]] = {}
    for contrast, arm_a, arm_b, filename in CPDT_CONTRASTS:
        stats: list[PairedStats] = []
        for dataset in datasets:
            problem_dir = bench_dir / DATASET_DIR[dataset]
            stats.append(
                compute_paired_stats(
                    load_all_run_logs(problem_dir / arm_a),
                    load_all_run_logs(problem_dir / arm_b),
                )
            )
        apply_holm_correction(stats)
        for dataset, paired in zip(datasets, stats, strict=True):
            save_paired_stats(paired, out_dir / method / DATASET_DIR[dataset] / filename)
        out[contrast] = stats
    return out


def run_cpdt(corpus: Path, method: str, datasets: list[str], out_dir: Path) -> dict[str, Any]:
    """Run the main campaign's paired test across problems on one host.

    Args:
        corpus: Corpus root (read-only).
        method: Host name.
        datasets: Datasets admitted to this host's analysis.
        out_dir: Pipeline directory; receives the paired statistics and
            ``cross_problem_dominance_<method>_srbench_blackbox.json``.

    Returns:
        The saved test payload (primary contrast at the top level, every
        contrast under ``"contrasts"``).
    """
    by_contrast = paired_stats_by_contrast(corpus, method, datasets, out_dir / "paired")
    secondary = {k: v for k, v in by_contrast.items() if k != CPDT_PRIMARY_CONTRAST}
    return run_cross_problem_dominance_test(
        by_contrast[CPDT_PRIMARY_CONTRAST],
        method,
        BENCHMARK,
        out_dir,
        contrast_stats=secondary,
    )


def exact_wilcoxon(deltas: list[float]) -> bool:
    """Whether SciPy's ``method="auto"`` evaluates the exact null distribution.

    It does when there are at most 50 pairs, no difference is zero after the
    tie snap, and no two differences tie in magnitude; otherwise it uses the
    normal approximation.

    Args:
        deltas: Per-dataset differences as recorded by the test.

    Returns:
        True for the exact branch.
    """
    snapped = [0.0 if abs(d) <= _CPDT_TIE_THRESHOLD else d for d in deltas]
    magnitudes = [abs(d) for d in snapped]
    return (
        len(snapped) <= EXACT_WILCOXON_MAX_N
        and all(m > 0.0 for m in magnitudes)
        and len(set(magnitudes)) == len(magnitudes)
    )


def wilcoxon_floor(n: int, alternative: str) -> float:
    """Smallest exact signed-rank p-value at ``n`` pairs: all signs agree."""
    return 2.0 ** -(n - 1) if alternative == "two-sided" else 2.0**-n


def primary_p_value(record: dict[str, Any]) -> float:
    """The p-value the record's pre-registered alternative defines (NaN if none)."""
    if record["alternative"] == "two-sided":
        return float(record["p_value_two_sided"])
    return float(record["p_value_one_sided"])


def flatten_cpdt(doc: dict[str, Any], method: str) -> list[dict[str, Any]]:
    """Flatten one host's test payload into one row per contrast and metric.

    This is the black-box counterpart of ``review_campaign.derive.load_cpdt``,
    which is bound to the 70- and 50-problem views. It keeps that function's
    columns, adds the Shapiro-Wilk p, the primary p and whether the signed-rank
    branch was exact and at its floor.

    Args:
        doc: Payload returned by :func:`run_cpdt`.
        method: Host name.

    Returns:
        Rows in ``CONTRASTS`` x ``METRICS`` order; contrasts absent from the
        payload are skipped.
    """
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        block = doc.get("contrasts", {}).get(contrast, {})
        for metric in METRICS:
            rec = block.get(metric)
            if not rec or "error" in rec:
                continue
            p_primary = primary_p_value(rec)
            wilcoxon = rec["test_used"] == "wilcoxon_signed_rank"
            exact = wilcoxon and exact_wilcoxon(rec["problem_deltas"])
            floor = wilcoxon_floor(int(rec["n_problems"]), rec["alternative"])
            rows.append(
                {
                    "method": method,
                    "contrast": contrast,
                    "metric": metric,
                    "test": rec["test_used"],
                    "alternative": rec["alternative"],
                    "n_problems": rec["n_problems"],
                    "n_wins": rec["n_wins"],
                    "n_ties": rec["n_ties"],
                    "n_losses": rec["n_losses"],
                    "sw_p": rec["shapiro_wilk_p"],
                    "cohens_d": rec["cohens_d"],
                    "d_lo": rec["cohens_d_ci_lower"],
                    "d_hi": rec["cohens_d_ci_upper"],
                    "mean_delta": rec["mean_delta"],
                    "p_one_sided": rec["p_value_one_sided"],
                    "p_two_sided": rec["p_value_two_sided"],
                    "p_primary": p_primary,
                    "p_holm": rec.get("p_value_holm"),
                    "wilcoxon_exact": int(exact),
                    "at_exact_floor": int(
                        exact and math.isfinite(p_primary) and math.isclose(p_primary, floor)
                    ),
                }
            )
    return rows
