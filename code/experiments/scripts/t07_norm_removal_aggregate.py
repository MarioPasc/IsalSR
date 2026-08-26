"""T07 aggregate — pool per-task result JSONs into a single summary.

Recursively finds all ``results.json`` files under the results root, groups
them by population, and computes pooled arm statistics plus decisive comparisons.

Usage
-----
    python -m experiments.scripts.t07_norm_removal_aggregate \\
        --results-dir /tmp/t07_smoke \\
        --out /tmp/t07_smoke/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger("t07.aggregate")

ARMS: tuple[str, str] = ("keep", "drop")
POPULATIONS: tuple[str, ...] = ("synthetic", "adversarial", "bingo", "udfs")


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _pool_arm(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Pool per-task arm statistics for one population.

    Args:
        records: List of result dicts from individual tasks.
        arm: Arm name ('keep' or 'drop').

    Returns:
        Pooled arm statistics dict.
    """
    n_total = sum(r["arms"][arm]["n_total"] for r in records)
    n_ok = sum(r["arms"][arm]["n_ok"] for r in records)
    n_raised = sum(r["arms"][arm]["n_raised"] for r in records)
    n_timeout = sum(r["arms"][arm]["n_timeout"] for r in records)
    n_unique = sum(r["arms"][arm]["n_unique"] for r in records)
    n_reachable = sum(r["arms"][arm]["n_reachable"] for r in records)
    n_rt_ok = sum(r["arms"][arm]["n_round_trip_ok"] for r in records)
    n_rt_checked = sum(r["arms"][arm]["n_round_trip_checked"] for r in records)
    n_eq_samples = sum(r["arms"][arm]["n_equivariance_samples"] for r in records)
    n_eq_failures = sum(r["arms"][arm]["n_equivariance_failures"] for r in records)

    rho = n_ok / n_unique if n_unique else float("nan")
    rt_rate = n_rt_ok / n_rt_checked if n_rt_checked else float("nan")
    eq_failure_rate = n_eq_failures / n_eq_samples if n_eq_samples else float("nan")

    # Pool per-k (sum counters across tasks; n_unique sums are approximate
    # because distinct-string sets overlap across tasks)
    per_k_pooled: dict[str, dict[str, Any]] = {}
    for r in records:
        for k_str, pk in r["arms"][arm].get("per_k", {}).items():
            if k_str not in per_k_pooled:
                per_k_pooled[k_str] = {
                    "n_total": 0,
                    "n_ok": 0,
                    "n_raised": 0,
                    "n_timeout": 0,
                    "n_unique": 0,
                }
            dest = per_k_pooled[k_str]
            dest["n_total"] += pk.get("n_total", 0)
            dest["n_ok"] += pk.get("n_ok", 0)
            dest["n_raised"] += pk.get("n_raised", 0)
            dest["n_timeout"] += pk.get("n_timeout", 0)
            dest["n_unique"] += pk.get("n_unique", 0)
    for _k_str, pk in per_k_pooled.items():
        n_u = pk.get("n_unique", 0)
        n_o = pk.get("n_ok", 0)
        pk["rho"] = n_o / n_u if n_u else float("nan")

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_raised": n_raised,
        "n_timeout": n_timeout,
        "n_unique_lower_bound": n_unique,
        "rho_lower_bound": rho,
        "n_reachable": n_reachable,
        "n_round_trip_ok": n_rt_ok,
        "n_round_trip_checked": n_rt_checked,
        "round_trip_rate": rt_rate,
        "round_trip_comparator": records[0]["arms"][arm].get("round_trip_comparator", ""),
        "n_equivariance_samples": n_eq_samples,
        "n_equivariance_failures": n_eq_failures,
        "equivariance_failure_rate": eq_failure_rate,
        "per_k": per_k_pooled,
    }


def _pool_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool decisive cross-arm comparisons across tasks.

    Args:
        records: Result dicts from individual tasks.

    Returns:
        Pooled comparisons dict with decisive flags.
    """
    n_both_ok = sum(r["comparisons"].get("n_both_ok", 0) for r in records)
    n_agreement = sum(r["comparisons"].get("n_agreement", 0) for r in records)
    eq_fail_keep = sum(r["comparisons"].get("n_equivariance_failures_keep", 0) for r in records)
    eq_fail_drop = sum(r["comparisons"].get("n_equivariance_failures_drop", 0) for r in records)
    eq_samp_keep = sum(r["comparisons"].get("n_equivariance_samples_keep", 0) for r in records)
    eq_samp_drop = sum(r["comparisons"].get("n_equivariance_samples_drop", 0) for r in records)

    # Adversarial fields are None for non-adversarial tasks
    adv_silent = [
        r["comparisons"].get("adversarial_keep_silent_wrong")
        for r in records
        if r["comparisons"].get("adversarial_keep_silent_wrong") is not None
    ]
    adv_loud = [
        r["comparisons"].get("adversarial_drop_loud_refusal")
        for r in records
        if r["comparisons"].get("adversarial_drop_loud_refusal") is not None
    ]

    agr = n_agreement / n_both_ok if n_both_ok else float("nan")

    # Carry per-task rho values for cross-task comparison
    rho_keep_vals = [r["arms"]["keep"].get("rho", float("nan")) for r in records]
    rho_drop_vals = [r["arms"]["drop"].get("rho", float("nan")) for r in records]
    finite_keep = [v for v in rho_keep_vals if math.isfinite(v)]
    finite_drop = [v for v in rho_drop_vals if math.isfinite(v)]
    mean_rho_keep = sum(finite_keep) / len(finite_keep) if finite_keep else float("nan")
    mean_rho_drop = sum(finite_drop) / len(finite_drop) if finite_drop else float("nan")

    return {
        "n_both_ok": n_both_ok,
        "n_agreement": n_agreement,
        "per_dag_agreement_rate": agr,
        "n_equivariance_samples_keep": eq_samp_keep,
        "n_equivariance_failures_keep": eq_fail_keep,
        "n_equivariance_samples_drop": eq_samp_drop,
        "n_equivariance_failures_drop": eq_fail_drop,
        "mean_rho_keep": mean_rho_keep,
        "mean_rho_drop": mean_rho_drop,
        "rho_equal_across_tasks": all(
            abs(k - d) < 1e-6
            for k, d in zip(rho_keep_vals, rho_drop_vals, strict=False)
            if math.isfinite(k) and math.isfinite(d)
        ),
        "adversarial_keep_silent_wrong_total": sum(adv_silent) if adv_silent else None,
        "adversarial_drop_loud_refusal_total": sum(adv_loud) if adv_loud else None,
        "n_tasks": len(records),
        "n_tasks_with_errors": sum(1 for r in records if r.get("error")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for T07 result aggregation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root directory containing per-task results.json files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path for summary JSON output (default: <results-dir>/summary.json).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_root = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_root / "summary.json"

    # Collect all per-task result JSONs
    all_files = sorted(results_root.rglob("results.json"))
    log.info("Found %d result files under %s", len(all_files), results_root)

    if not all_files:
        log.error("No results.json files found under %s", results_root)
        return

    records: list[dict[str, Any]] = []
    for f in all_files:
        try:
            records.append(json.loads(f.read_text()))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", f, exc)

    # Group by population
    by_population: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        pop = r.get("population", "unknown")
        by_population.setdefault(pop, []).append(r)

    summary: dict[str, Any] = {"n_tasks_total": len(records), "populations": {}}

    for pop, pop_records in sorted(by_population.items()):
        valid = [r for r in pop_records if "arms" in r]
        log.info(
            "Population %s: %d tasks (%d with arm data)",
            pop,
            len(pop_records),
            len(valid),
        )
        if not valid:
            summary["populations"][pop] = {"error": "no valid records"}
            continue

        pop_summary: dict[str, Any] = {
            "n_tasks": len(pop_records),
            "n_tasks_valid": len(valid),
            "arms": {arm: _pool_arm(valid, arm) for arm in ARMS},
            "comparisons": _pool_comparisons(valid),
        }
        summary["populations"][pop] = pop_summary

        # Print per-population summary table
        print(f"\n=== Population: {pop} ({len(valid)} tasks) ===")
        print(
            f"{'arm':<6} {'n_total':>8} {'n_ok':>7} {'n_unique':>9} "
            f"{'rho':>7} {'rt_rate':>8} {'eq_fail_rate':>13}"
        )
        print("-" * 62)
        for arm in ARMS:
            a = pop_summary["arms"][arm]
            rt = a.get("round_trip_rate", float("nan"))
            eq = a.get("equivariance_failure_rate", float("nan"))
            rho = a.get("rho_lower_bound", float("nan"))
            print(
                f"{arm:<6} {a['n_total']:>8,} {a['n_ok']:>7,} "
                f"{a['n_unique_lower_bound']:>9,} {rho:>7.3f} "
                f"{rt:>8.4f} {eq:>13.6f}"
            )
        cmp = pop_summary["comparisons"]
        print(
            f"\nAgreement rate (both ok): {cmp['per_dag_agreement_rate']:.6f}  "
            f"rho_equal_across_tasks: {cmp['rho_equal_across_tasks']}"
        )
        if cmp.get("adversarial_keep_silent_wrong_total") is not None:
            print(
                f"Adversarial silent-wrong: {cmp['adversarial_keep_silent_wrong_total']}  "
                f"loud-refusal: {cmp['adversarial_drop_loud_refusal_total']}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote %s", out_path)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
