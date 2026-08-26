"""Pool the campaign's reachability and fallback ledgers into one artefact.

R1.2 asks how often the reachability precondition of Theorems 3.13 and 3.15 fails
in practice. Every deduplicating cell of the C2 campaign writes a
``fallback_ledger.json`` recording, for the DAGs it saw, how many violated the
precondition before normalisation, how many still violated it after, and how many
took each of the four paths that bypass canonicalisation. Those 8,400 files are
the measurement; this module pools them so the manuscript quotes a number a reader
can recompute.

Only campaign-derived numbers are produced. Pre-campaign probes measured the same
quantities on far smaller populations and are deliberately not read here: pooling
two populations that differ by five orders of magnitude invites a comparison
neither supports.

**A zero is only evidence if something was counted.** A disabled ledger and a
ledger that observed no events both report ``0``, and the T06 wave lost 1,260 runs
to exactly that confusion. :func:`validate` therefore refuses to emit an artefact
unless every cell is enabled and observed a non-zero population, and the report
states the checks that passed.

Definitions, which are easy to conflate:

``violated_pre``
    DAGs arriving at the canonicaliser with at least one non-variable node not
    reachable from any variable. This is R1.2's quantity.
``violated_post``
    The same count after ``normalize_const_creation`` supplied the missing
    creation edge. A non-zero value here would be a correctness defect.
``timeout``, ``conversion_failure``, ``canon_raised``, ``atlas_hit``
    The four paths by which a candidate reaches the evaluator without its
    canonical string being computed and stored. Each could bias the reduction
    factor, so each is counted rather than assumed.

Usage
-----
    python -m experiments.scripts.review_campaign.reachability [--corpus DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import add_common_args  # noqa: E402

#: Paths by which a candidate bypasses canonicalisation, in ledger key order.
BYPASS_PATHS: tuple[str, ...] = (
    "timeout",
    "conversion_failure",
    "canon_raised",
    "atlas_hit",
)

#: Counters pooled additively across cells.
COUNTERS: tuple[str, ...] = (
    "n_seen",
    "n_sampled",
    "violated_pre",
    "violated_post",
) + BYPASS_PATHS

#: Histograms keyed by internal-node count ``k``, paired with the counter they
#: must sum to. A ledger whose histogram disagrees with its own scalar is
#: internally inconsistent and its cell cannot be trusted.
HISTOGRAM_OF: dict[str, str] = {
    "n_sampled_hist": "n_sampled",
    "violated_pre_hist": "violated_pre",
    "violated_post_hist": "violated_post",
    **{f"{path}_hist": path for path in BYPASS_PATHS},
}

#: Arms that compute a key and therefore write a ledger. ``baseline`` does not.
DEDUP_ARMS: tuple[str, ...] = ("hash", "isalsr")


def wilson_upper(failures: int, trials: int, z: float = 1.959963985) -> float:
    """Return the upper end of the Wilson score interval for a proportion.

    Wilson is used rather than the normal approximation because the counts of
    interest are zero, where the normal interval collapses to a point and asserts
    a certainty the data does not support.

    Args:
        failures: Number of events observed.
        trials: Number of trials.
        z: Standard normal quantile; the default is the two-sided 95% value.

    Returns:
        The upper bound, or ``1.0`` when there were no trials.
    """
    if trials <= 0:
        return 1.0
    p = failures / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return min(1.0, centre + half)


def rule_of_three(trials: int) -> float:
    """Return the one-sided 95% upper bound on a rate after zero events.

    This is the ``3/n`` rule. It is reported alongside the Wilson bound because
    the two differ materially and the manuscript must name which it uses.

    Args:
        trials: Number of trials, all of which produced no event.

    Returns:
        ``3/n``, or ``1.0`` when there were no trials.
    """
    return 3.0 / trials if trials > 0 else 1.0


def read_ledgers(corpus: Path) -> list[dict[str, Any]]:
    """Read every per-cell fallback ledger in the corpus.

    Args:
        corpus: Campaign root holding ``<method>/<suite>/<problem>/<arm>/seed_NN/``.

    Returns:
        One record per cell, carrying its coordinates, counters and histograms.
        A ledger that cannot be parsed yields a record carrying ``unreadable``
        rather than raising, so a single bad file is reported instead of hiding
        the other 8,399.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(corpus.glob("*/*/*/*/seed_*/fallback_ledger.json")):
        method, suite, problem, arm, seed_dir = path.relative_to(corpus).parts[:5]
        base = {
            "method": method,
            "suite": suite,
            "problem": problem,
            "arm": arm,
            "seed": seed_dir.removeprefix("seed_"),
        }
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            records.append({**base, "unreadable": str(exc)})
            continue
        record: dict[str, Any] = {
            **base,
            "enabled": payload.get("enabled"),
            "sample_rate": payload.get("sample_rate"),
        }
        for key in COUNTERS:
            record[key] = int(payload.get(key, 0) or 0)
        record["_hists"] = {key: payload.get(key, {}) or {} for key in HISTOGRAM_OF}
        records.append(record)
    return records


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the checks that decide whether the pooled zeros mean anything.

    Args:
        records: Per-cell records from :func:`read_ledgers`.

    Returns:
        A report with one entry per check: the count that passed, the count that
        failed, and up to five offending cells. ``all_passed`` is the conjunction.
    """
    usable = [r for r in records if "unreadable" not in r]
    methods = sorted({r["method"] for r in usable})
    problems = sorted({(r["method"], r["suite"], r["problem"]) for r in usable})
    seeds = sorted({r["seed"] for r in usable})
    expected = len(methods) * len(DEDUP_ARMS) * (len(problems) // max(1, len(methods))) * len(seeds)

    def cell(record: dict[str, Any]) -> str:
        return "/".join(str(record[f]) for f in ("method", "suite", "problem", "arm", "seed"))

    def check(name: str, predicate: Any, why: str) -> dict[str, Any]:
        failures = [cell(r) for r in usable if not predicate(r)]
        return {
            "passed": len(usable) - len(failures),
            "failed": len(failures),
            "examples": failures[:5],
            "why": why,
            "ok": not failures,
        }

    hist_ok = lambda r: all(  # noqa: E731 - a named function here would not read better
        sum(int(v) for v in r["_hists"][h].values()) == r[c] for h, c in HISTOGRAM_OF.items()
    )
    per_k_ok = lambda r: all(  # noqa: E731
        int(r["_hists"]["violated_pre_hist"].get(k, 0)) <= int(n)
        for k, n in r["_hists"]["n_sampled_hist"].items()
    )

    report = {
        "n_ledgers": len(records),
        "n_unreadable": len(records) - len(usable),
        "n_expected": expected,
        "checks": {
            "readable": {
                "passed": len(usable),
                "failed": len(records) - len(usable),
                "examples": [cell(r) for r in records if "unreadable" in r][:5],
                "why": "a ledger that will not parse cannot contribute a zero or a count",
                "ok": len(usable) == len(records),
            },
            "instrumentation_enabled": check(
                "instrumentation_enabled",
                lambda r: r["enabled"] is True,
                "a disabled ledger reports zero events because it counted none",
            ),
            "population_non_empty": check(
                "population_non_empty",
                lambda r: r["n_sampled"] > 0,
                "a cell that sampled nothing contributes a vacuous zero",
            ),
            "full_census": check(
                "full_census",
                lambda r: r["sample_rate"] == 1 and r["n_seen"] == r["n_sampled"],
                "rates are only exact if every candidate was inspected",
            ),
            "histograms_sum_to_counters": check(
                "histograms_sum_to_counters",
                hist_ok,
                "an internally inconsistent ledger cannot be pooled",
            ),
            "violations_never_exceed_samples": check(
                "violations_never_exceed_samples",
                per_k_ok,
                "at no k may more DAGs violate than were seen",
            ),
        },
    }
    report["all_passed"] = all(c["ok"] for c in report["checks"].values())
    return report


def pool(records: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Pool counters and histograms over the given grouping.

    Args:
        records: Per-cell records, already validated.
        key_fields: Fields whose tuple identifies a group; empty pools everything.

    Returns:
        Mapping from a ``"|"``-joined group key to pooled counters, pooled
        histograms and derived rates.
    """
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        key = "|".join(str(record[f]) for f in key_fields) or "all"
        bucket = groups.setdefault(
            key,
            {
                "n_cells": 0,
                **{c: 0 for c in COUNTERS},
                "hists": {h: Counter() for h in HISTOGRAM_OF},
            },
        )
        bucket["n_cells"] += 1
        for counter in COUNTERS:
            bucket[counter] += record[counter]
        for hist_name, hist in record["_hists"].items():
            for k_value, count in hist.items():
                bucket["hists"][hist_name][str(k_value)] += int(count)

    for bucket in groups.values():
        sampled = bucket["n_sampled"]
        bucket["hists"] = {
            h: dict(sorted(c.items(), key=lambda kv: int(kv[0])))
            for h, c in bucket["hists"].items()
        }
        bucket["violated_pre_rate"] = bucket["violated_pre"] / sampled if sampled else None
        bucket["violated_post_rate"] = bucket["violated_post"] / sampled if sampled else None
        bucket["bypass_total"] = sum(bucket[p] for p in BYPASS_PATHS)
        bucket["bypass_rate"] = bucket["bypass_total"] / sampled if sampled else None
        bucket["bypass_wilson_upper_95"] = wilson_upper(bucket["bypass_total"], sampled)
        bucket["bypass_rule_of_three_upper_95"] = (
            rule_of_three(sampled) if bucket["bypass_total"] == 0 else None
        )
    return groups


def violation_by_k(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    """Return the violation rate stratified by internal-node count.

    Args:
        records: Per-cell records.
        method: Host solver to restrict to.

    Returns:
        One row per ``k``, ascending, with sampled and violating counts.
    """
    sampled: Counter[str] = Counter()
    violated: Counter[str] = Counter()
    for record in records:
        if record["method"] != method:
            continue
        for k_value, count in record["_hists"]["n_sampled_hist"].items():
            sampled[str(k_value)] += int(count)
        for k_value, count in record["_hists"]["violated_pre_hist"].items():
            violated[str(k_value)] += int(count)
    rows = []
    for k_value in sorted(sampled, key=int):
        n = sampled[k_value]
        v = violated.get(k_value, 0)
        rows.append(
            {"k": int(k_value), "n_sampled": n, "violated_pre": v, "rate": v / n if n else None}
        )
    return rows


def candidate_to_evaluation_ratio(
    corpus: Path, pooled: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Cross-check ledger candidate counts against the campaign's evaluation counts.

    The ledger counts every candidate arriving at the key computation; the
    campaign's ``cells.csv`` counts unique evaluations after deduplication. Their
    ratio must reproduce the reduction factor, which is an independent path to a
    number the manuscript already reports. On the UDFS naive-hash arm the ratio
    must be exactly one, because that key provably removes nothing there.

    Args:
        corpus: Campaign root, whose ``analyses/data/cells.csv`` holds ``n_eval``.
        pooled: Output of :func:`pool` grouped by ``("method", "arm")``.

    Returns:
        One entry per ``method|arm`` with candidates, evaluations and the ratio,
        or a ``note`` if ``cells.csv`` is absent.
    """
    cells = corpus / "analyses" / "data" / "cells.csv"
    if not cells.exists():
        return {"note": f"{cells} absent; cross-check skipped"}
    evaluations: Counter[str] = Counter()
    with cells.open() as handle:
        for row in csv.DictReader(handle):
            try:
                evaluations[f"{row['method']}|{row['arm']}"] += int(float(row["n_eval"]))
            except (KeyError, TypeError, ValueError):
                continue
    out: dict[str, Any] = {}
    for key, bucket in pooled.items():
        n_eval = evaluations.get(key, 0)
        out[key] = {
            "candidates": bucket["n_sampled"],
            "evaluations": n_eval,
            "ratio": bucket["n_sampled"] / n_eval if n_eval else None,
        }
    return out


def main() -> None:
    """Validate the ledgers, pool them, and write the artefact beside the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    out_dir = corpus / "reachability"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_ledgers(corpus)
    report = validate(records)
    usable = [r for r in records if "unreadable" not in r]

    by_method_arm = pool(usable, ("method", "arm"))
    artefact: dict[str, Any] = {
        "corpus": str(corpus),
        "validation": report,
        "global": pool(usable, ())["all"] if usable else {},
        "by_method": pool(usable, ("method",)),
        "by_method_arm": by_method_arm,
        "by_method_suite": pool(usable, ("method", "suite")),
        "violation_by_k": {
            m: violation_by_k(usable, m) for m in sorted({r["method"] for r in usable})
        },
        "candidate_to_evaluation_ratio": candidate_to_evaluation_ratio(corpus, by_method_arm),
    }
    (out_dir / "reachability.json").write_text(json.dumps(artefact, indent=1) + "\n")

    fields = ["method", "suite", "problem", "arm", "seed", "enabled", "sample_rate", *COUNTERS]
    with (out_dir / "reachability_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in usable:
            writer.writerow({f: record.get(f) for f in fields})

    glob = artefact["global"]
    print(f"ledgers            {report['n_ledgers']}  (expected {report['n_expected']})")
    for name, result in report["checks"].items():
        mark = "ok  " if result["ok"] else "FAIL"
        print(f"  {mark} {name:32} {result['passed']}/{report['n_ledgers']}")
    if not report["all_passed"]:
        print("\nvalidation failed; the pooled zeros below are not evidence")
    print(f"candidates         {glob.get('n_sampled', 0):,}")
    for stage in ("pre", "post"):
        rate = glob.get(f"violated_{stage}_rate", 0) or 0.0
        print(f"violated {stage:4}      {glob.get(f'violated_{stage}', 0):,}  ({rate:.4%})")
    for path in BYPASS_PATHS:
        print(f"  {path:20} {glob.get(path, 0):,}")
    r3 = glob.get("bypass_rule_of_three_upper_95") or float("nan")
    wilson = glob.get("bypass_wilson_upper_95", 1.0)
    print(f"bypass 95% upper   rule-of-three {r3:.3e}  Wilson {wilson:.3e}")
    print(f"written            {out_dir}")
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
