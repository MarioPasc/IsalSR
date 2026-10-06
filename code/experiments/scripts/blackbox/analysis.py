"""Rebuild every number and table of the black-box appendix from the corpus.

One command, run from ``code/``::

    python -m experiments.scripts.blackbox.analysis \\
        [--corpus DIR] [--c2-analyses DIR] [--out DIR] [--tables-dir DIR]

Steps, in order:

1. Audit completeness and provenance and apply the pre-declared inclusion rule
   (``audit.json``). The analysis refuses to continue if a provenance conflict
   is found.
2. Classify why every run stopped, on this corpus and on the main campaign's
   corpus (the parent of ``--c2-analyses``), from the counters the runs
   recorded (``values/stopping.json``, ``data/stopping_*.csv``).
3. Flatten the admitted cells with the main campaign's extractor
   (``data/cells.csv``, ``data/cells_extra.csv``).
4. Derive the per-dataset and per-host quantities with the main campaign's
   functions (``data/*.csv``, ``values/summary.json``).
5. Run the paired test across problems per host (``pipeline/``,
   ``data/cpdt.csv``).
6. Recompute the main campaign's governing quantities as the comparator
   (``values/c2_comparator.json``).
7. Write the LaTeX tables and the value macros (``tables/``, copied to
   ``--tables-dir`` when given).

Nothing is written inside either corpus or the main campaign's analysis tree.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from experiments.scripts.blackbox import latex, stopping
from experiments.scripts.blackbox.audit import run_audit
from experiments.scripts.blackbox.config import add_common_args
from experiments.scripts.blackbox.cpdt import flatten_cpdt, run_cpdt
from experiments.scripts.blackbox.measures import (
    EXTRA_COLUMNS,
    c2_comparator,
    derive_all,
    extract_cells,
    generation_ratios,
    governing,
    nrmse_max_cell,
    quality_by_arm,
    size_vs_speedup,
    termination_summary,
    write_csv,
)
from experiments.scripts.review_campaign import derive as c2
from experiments.scripts.review_campaign.config import METHODS
from experiments.scripts.review_campaign.extract_cells import COLUMNS

log = logging.getLogger(__name__)


class BlackboxAnalysisError(RuntimeError):
    """Raised when an input violates a precondition of the analysis."""


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=str)


def _check_paths(out: Path, *protected: Path) -> None:
    """Refuse an output tree inside any read-only input."""
    for root in protected:
        if out.resolve().is_relative_to(root.resolve()):
            raise BlackboxAnalysisError(f"--out {out} lies inside read-only {root}")


def _stop_reasons(
    corpus: Path, c2_corpus: Path, out: Path
) -> tuple[dict[str, Any], dict[tuple[str, str, str, int], str]]:
    """Classify the stop reason of every run of both campaigns.

    Returns:
        The summary of both campaigns, and the black-box reasons keyed by
        ``(method, problem directory, arm, seed)``.
    """
    summary: dict[str, Any] = {}
    reasons: dict[tuple[str, str, str, int], str] = {}
    for name, root in (("blackbox", corpus), ("c2", c2_corpus)):
        records = stopping.scan(root)
        write_csv(out / "data" / f"stopping_{name}.csv", records)
        summary[name] = stopping.summarise(records)
        if name == "blackbox":
            reasons = {
                (r["method"], r["problem"], r["arm"], r["seed"]): r["reason"] for r in records
            }
    _dump(out / "values" / "stopping.json", summary)
    return summary, reasons


def run(corpus: Path, c2_analyses: Path, out: Path, tables_dir: Path | None) -> dict[str, Any]:
    """Run the whole analysis.

    Args:
        corpus: Black-box corpus root (read-only).
        c2_analyses: Main campaign's analysis tree (read-only); its parent is
            the main campaign's corpus.
        out: Output tree.
        tables_dir: Optional directory receiving copies of the tables.

    Returns:
        The values payload the tables and the prose macros are built from.

    Raises:
        BlackboxAnalysisError: On a provenance conflict or a misplaced output.
    """
    c2_corpus = c2_analyses.parent
    _check_paths(out, corpus, c2_corpus)

    audit = run_audit(corpus)
    _dump(out / "audit.json", audit.to_dict())
    if audit.e6_e7["provenance"]["mixed"]:
        raise BlackboxAnalysisError(
            f"provenance conflict: {audit.e6_e7['provenance']['conflicts']}"
        )
    log.info("audit: %d/%d cells complete", audit.n_completed, audit.n_expected)

    stop_summary, reasons = _stop_reasons(corpus, c2_corpus, out)

    rows, extra = extract_cells(corpus, audit.included, reasons)
    write_csv(out / "data" / "cells.csv", rows, COLUMNS)
    write_csv(out / "data" / "cells_extra.csv", extra, EXTRA_COLUMNS)
    rows = c2.read_cells(out / "data" / "cells.csv")

    derived = derive_all(rows)
    write_csv(out / "data" / "per_problem.csv", derived["per_problem"])
    write_csv(out / "data" / "phi.csv", derived["phi"])
    summary = derived["summary"]
    _dump(out / "values" / "summary.json", summary)

    cpdt_rows: list[dict[str, Any]] = []
    for method in METHODS:
        doc = run_cpdt(corpus, method, audit.included[method], out / "pipeline")
        cpdt_rows.extend(flatten_cpdt(doc, method))
    write_csv(out / "data" / "cpdt.csv", cpdt_rows)

    comparator = c2_comparator(c2_analyses)
    _dump(out / "values" / "c2_comparator.json", comparator)

    values: dict[str, Any] = {
        "audit": audit.to_dict(),
        "governing": {m: governing(summary, m) for m in METHODS},
        "quality": quality_by_arm(derived["per_problem"]),
        "termination": termination_summary(extra),
        "nrmse_max_cell": nrmse_max_cell(rows),
        "stopping": stop_summary,
        "generations": generation_ratios(extra),
        "size_vs_speedup": size_vs_speedup(derived["speedup"], rows),
        "c2": comparator,
        "cpdt": cpdt_rows,
        "summary": summary,
    }
    _dump(out / "values" / "values.json", values)

    written = latex.write_tables(values, derived, out / "tables")
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        for path in written:
            shutil.copy2(path, tables_dir / path.name)
    return values


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    values = run(args.corpus, args.c2_analyses, args.out, args.tables_dir)
    audit = values["audit"]
    print(
        f"cells {audit['n_completed']}/{audit['n_expected']} complete; included",
        {m: len(v) for m, v in audit["included"].items()},
    )
    for method in METHODS:
        g = values["governing"][method]
        print(
            f"{method}: rho {g['rho']:.3f} rho_ser {g['rho_ser']:.3f} phi {g['phi']:.3f} "
            f"Teval/Tcanon {g['t_eval_over_t_canon']:.1f} OH {g['oh_pct']:.2f}% "
            f"S {g['s_median']:.3f} wall {g['wall_ratio']:.3f}"
        )
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
