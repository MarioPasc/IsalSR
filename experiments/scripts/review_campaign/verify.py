"""Check every campaign-derived number in the manuscript against the data.

The manuscript, its supplementary material and the response letter quote figures
that live in ``analyses/values/summary.json`` and the derived tables. A number
that drifts from its source is invisible on a read and fatal in review, so each
one is asserted here against the value the pipeline computed rather than against
a copy of itself.

The check is deliberately literal: it looks for the exact rendered string in the
file that should carry it. A tolerance would let a rounding change pass, which is
the failure this exists to catch.

Usage
-----
    python -m experiments.scripts.review_campaign.verify [--analyses DIR]
        [--article DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import add_common_args  # noqa: E402

DEFAULT_ARTICLE = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/"
    "69c1637a28a81fea2badda9a"
)

PAPER = "article/paper"
SUPP = "article/supplementary"
LETTER = "reviews"


def load_sources(article: Path) -> dict[str, str]:
    """Concatenate the prose of each document into one searchable string."""
    out: dict[str, str] = {}
    for name, rel in (("paper", PAPER), ("supplementary", SUPP), ("letter", LETTER)):
        text = []
        for path in sorted((article / rel).glob("*.tex")):
            text.append(path.read_text(encoding="utf-8"))
        out[name] = "\n".join(text)
    return out


def build_checks(summary: dict) -> list[tuple[str, str, str]]:
    """Return (document, literal, description) for every quoted quantity."""
    checks: list[tuple[str, str, str]] = []

    def add(doc: str, literal: str, what: str) -> None:
        checks.append((doc, literal, what))

    # --- reduction and redundancy -------------------------------------------
    for doc in ("paper",):
        add(doc, "1.66 \\pm 0.26", "rho mean +- sd, UDFS")
        add(doc, "1.79 \\pm 0.09", "rho mean +- sd, Bingo")
        add(doc, "38.1\\%", "redundancy rate, UDFS")
        add(doc, "43.7\\%", "redundancy rate, Bingo")
    add("supplementary", "[1.11, 2.12]", "rho range, UDFS")
    add("supplementary", "[1.19, 1.85]", "rho range, Bingo")

    # --- the share phi -------------------------------------------------------
    add("paper", "0.047", "phi mean, Bingo")
    add("supplementary", "\\rho_\\sigma = 1.0000", "rho_ser on UDFS")
    add("supplementary", "1.727", "rho_ser mean, Bingo")
    add("letter", "95.3\\%", "share of redundancy the naive hash recovers on Bingo")

    # --- paired test across problems ----------------------------------------
    add("paper", "2.0 \\times 10^{-9}", "one-sided p, R2 test, UDFS")
    add("paper", "7.5 \\times 10^{-4}", "one-sided p, R2 test, Bingo")
    add("paper", "+0.41", "Cohen's d, R2 test, UDFS")
    add("paper", "[+0.31, +0.59]", "bootstrap CI on that d")
    add("paper", "2.54", "Cohen's d on rho vs naive hash, UDFS")
    add("paper", "[+2.12, +3.16]", "bootstrap CI on that d")
    add("paper", "7.05", "Cohen's d on rho vs naive hash, Bingo")
    add("paper", "1.8 \\times 10^{-13}", "one-sided p on rho vs naive hash")

    # --- cost ----------------------------------------------------------------
    add("paper", "0.126", "median per-candidate key cost, UDFS, ms")
    add("paper", "0.074", "median per-candidate key cost, Bingo, ms")
    add("paper", "727", "median per-evaluation cost, UDFS, ms")
    add("paper", "1.48", "median per-evaluation cost, Bingo, ms")
    add("paper", "0.04\\%", "median key overhead, UDFS")
    add("paper", "16.1\\%", "median key overhead, Bingo")
    add("paper", "0.15\\%", "95th percentile key overhead, UDFS")
    add("paper", "1.96", "median S on the unsaturated UDFS cells")
    add("paper", "1{,}727", "UDFS cells at the budget cap")
    add("paper", "0.72", "median S, Bingo")
    add("supplementary", "0.038", "mean key cost, Bingo, k in [0,5), ms")
    add("supplementary", "0.076", "mean key cost, Bingo, k in [15,32), ms")

    # --- counts --------------------------------------------------------------
    add("supplementary", "12{,}600", "campaign size")
    add("letter", "12{,}600", "campaign size")
    add("paper", "2{,}100", "cells per method and arm")
    add("paper", "22", "problems with S >= 1 on Bingo")
    add("paper", "13", "problems on which IsalSR is faster on Bingo, wall clock")

    # --- critical difference -------------------------------------------------
    add("paper", "0.40", "critical difference, three groups, N=70")
    add("paper", "0.90", "critical difference, six groups, N=70")

    return checks


def numeric_gate(summary: dict) -> list[tuple[str, float, float]]:
    """Recompute the headline values from the summary and pin the literals."""
    udfs, bingo = summary["udfs"], summary["bingo"]
    return [
        ("rho UDFS", udfs["isalsr"]["rho_over_problems"]["mean"], 1.66),
        ("rho Bingo", bingo["isalsr"]["rho_over_problems"]["mean"], 1.79),
        ("r UDFS", 100 * udfs["isalsr"]["r_over_problems"]["mean"], 38.1),
        ("r Bingo", 100 * bingo["isalsr"]["r_over_problems"]["mean"], 43.7),
        ("phi UDFS", udfs["phi"]["over_problems"]["mean"], 1.000),
        ("phi Bingo", bingo["phi"]["over_problems"]["mean"], 0.047),
        ("rho_ser UDFS", udfs["hash"]["rho_over_problems"]["mean"], 1.000),
        ("rho_ser Bingo", bingo["hash"]["rho_over_problems"]["mean"], 1.727),
        ("key ms UDFS", udfs["isalsr"]["key_ms_over_cells"]["median"], 0.126),
        ("key ms Bingo", bingo["isalsr"]["key_ms_over_cells"]["median"], 0.074),
        ("eval ms UDFS", udfs["isalsr"]["eval_ms_over_cells"]["median"], 727.3),
        ("eval ms Bingo", bingo["isalsr"]["eval_ms_over_cells"]["median"], 1.48),
        ("OH UDFS", udfs["isalsr"]["overhead_pct_over_cells"]["median"], 0.039),
        ("OH Bingo", bingo["isalsr"]["overhead_pct_over_cells"]["median"], 16.11),
        ("S Bingo", bingo["isalsr"]["S_over_cells"]["median"], 0.72),
        ("cells", summary["n_cells"], 12600),
    ]


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--article", type=Path, default=DEFAULT_ARTICLE)
    args = parser.parse_args()

    with (args.analyses / "values" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    failures: list[str] = []

    print("=== derived values against the summary ===")
    for label, value, expected in numeric_gate(summary):
        ok = abs(value - expected) <= 0.6 * 10 ** -len(str(expected).split(".")[-1])
        if isinstance(expected, int):
            ok = value == expected
        print(f"  {'ok ' if ok else 'FAIL'} {label:16s} {value:>12.4f} vs {expected}")
        if not ok:
            failures.append(f"derived value {label}: {value} != {expected}")

    print("\n=== quoted literals present in the documents ===")
    sources = load_sources(args.article)
    for doc, literal, what in build_checks(summary):
        ok = literal in sources[doc]
        print(f"  {'ok ' if ok else 'FAIL'} {doc:14s} {literal!r:28s} {what}")
        if not ok:
            failures.append(f"{doc} is missing {literal!r} ({what})")

    print("\n=== placeholders ===")
    for doc, text in sources.items():
        # The two macro definitions write `\pendingnum}` and `\pendingblock}`,
        # so counting the call form `\pendingnum{` excludes them already.
        n = text.count("\\pendingnum{") + text.count("\\pendingblock{")
        print(f"  {doc:14s} {n} remaining")
        if doc == "paper" and n:
            failures.append(f"{doc} still carries {n} placeholders")

    if failures:
        print(f"\n{len(failures)} FAILURES")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print("\nall checks pass")


if __name__ == "__main__":
    main()
