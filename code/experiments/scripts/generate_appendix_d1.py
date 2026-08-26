"""Emit Appendix D.1 problem-documentation rows straight from the benchmark registry.

Appendix D.1 must give, for every problem in the suite, its expression,
dimensionality, variable ranges, sampling protocol and citation. Twenty-eight of
the submitted fifty were undocumented (reviewer comment R2.5, ticket T09) and the
R3.1 extension adds twenty more (ticket T05, AC-8).

Writing those rows by hand is how they drifted from the code in the first place.
This script derives them from `_BENCHMARK_REGISTRY`, so a row cannot disagree with
the definition it documents. T09 can run it over the whole suite; T05 uses the
`--suites strogatz,feynman_remainder` restriction for D2.

Usage
-----
    python -m experiments.scripts.generate_appendix_d1 --suites strogatz,feynman_remainder
    python -m experiments.scripts.generate_appendix_d1 --all --format latex
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.models.orchestrator import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    _get_ground_truth_sympy,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

#: Per-suite provenance. Every suite must appear here; a missing entry is a bug,
#: not a default, because an undocumented citation is exactly the R2.5 defect.
SUITE_CITATION: dict[str, str] = {
    "nguyen": r"\cite{uy2011}",
    "feynman": r"\cite{udrescu2020}",
    "hard": r"\cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003}",
    "cherrypicked": r"\cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021}",
    "roundoff": r"\cite{udrescu2020,pagie1997,mundhenk2021}",
    "strogatz": r"\cite{lacava2016,lacava2021,strogatz2014}",
    "feynman_remainder": r"\cite{udrescu2020,lacava2021}",
}

#: Human-readable sampling protocol per `sampling.type`.
SAMPLING_PROSE: dict[str, str] = {
    "uniform": "uniform i.i.d.",
    "published_fixed": "published simulation, random split",
    "grid_2d_skip_zero": "$26\\times26$ grid, zero skipped",
    "grid_1d_train_uniform_test_grid": "uniform train, grid test",
    "integer_grid": "integer grid (extrapolation)",
}


def _sampling_prose(bench: dict[str, Any]) -> str:
    """Describe a problem's sampling protocol.

    Parameters
    ----------
    bench
        A benchmark specification dict.

    Returns
    -------
    str
        A short human-readable protocol description.
    """
    sampling = bench.get("sampling")
    if not sampling:
        return SAMPLING_PROSE["uniform"]
    prose = SAMPLING_PROSE.get(sampling["type"], sampling["type"])
    n_tr = sampling.get("n_train_override")
    n_te = sampling.get("n_test_override")
    if n_tr is not None and n_te is not None:
        prose = f"{prose}, {n_tr}/{n_te}"
    return prose


def _ranges_prose(bench: dict[str, Any]) -> str:
    """Render variable ranges compactly, collapsing a common range to one entry.

    Parameters
    ----------
    bench
        A benchmark specification dict.

    Returns
    -------
    str
        A compact range description, e.g. ``$[1,5]^{4}$``.
    """
    ranges = bench.get("var_ranges") or []
    if not ranges:
        return "--"
    uniq = {(round(lo, 6), round(hi, 6)) for lo, hi in ranges}
    if len(uniq) == 1:
        lo, hi = next(iter(uniq))
        exp = f"^{{{len(ranges)}}}" if len(ranges) > 1 else ""
        return f"$[{lo:g},{hi:g}]{exp}$"
    return ", ".join(f"$[{lo:g},{hi:g}]$" for lo, hi in ranges)


def rows_for(suite: str) -> list[dict[str, str]]:
    """Build the Appendix D.1 rows for one suite.

    Parameters
    ----------
    suite
        Registry key.

    Returns
    -------
    list of dict
        One row per problem.

    Raises
    ------
    KeyError
        If the suite is unknown or has no declared citation.
    """
    if suite not in _BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown suite {suite!r}. Known: {sorted(_BENCHMARK_REGISTRY)}")
    if suite not in SUITE_CITATION:
        raise KeyError(f"Suite {suite!r} has no citation in SUITE_CITATION; add one.")

    rows = []
    for bench in _BENCHMARK_REGISTRY[suite][0]:
        rows.append(
            {
                "suite": suite,
                "name": bench["name"],
                "expression": bench.get("expression", "--"),
                "n_vars": str(bench["num_variables"]),
                "ranges": _ranges_prose(bench),
                "sampling": _sampling_prose(bench),
                "citation": SUITE_CITATION[suite],
                # Resolved through the orchestrator's own resolver, not the raw
                # dict key: a problem without `sympy_expression` may still be
                # covered by the string-parse fallback. What Stage C's C1.5
                # checks is whether `solution_recovered` is computable, which is
                # exactly what this call answers.
                "gt": "yes" if _get_ground_truth_sympy(bench) is not None else "NO",
            }
        )
    return rows


def _escape(text: str) -> str:
    """Escape the LaTeX-significant characters that appear in expression strings."""
    for a, b in (("_", r"\_"), ("^", r"\^{}"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(a, b)
    return text


def render(rows: list[dict[str, str]], fmt: str) -> str:
    """Render rows as a LaTeX tabular body or as Markdown.

    Parameters
    ----------
    rows
        Rows from :func:`rows_for`.
    fmt
        ``"latex"`` or ``"markdown"``.

    Returns
    -------
    str
        The rendered table.
    """
    if fmt == "latex":
        return "\n".join(
            " & ".join(
                (
                    _escape(r["name"]),
                    f"$\\mathtt{{{_escape(r['expression'])}}}$",
                    r["n_vars"],
                    r["ranges"],
                    r["sampling"],
                    r["citation"],
                )
            )
            + r" \\"
            for r in rows
        )
    head = "| Problem | Expression | $n$ | Range | Sampling | Source | GT |"
    sep = "|---|---|---|---|---|---|---|"
    body = [
        f"| {r['name']} | `{r['expression']}` | {r['n_vars']} | {r['ranges']} "
        f"| {r['sampling']} | {r['citation']} | {r['gt']} |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv
        Command-line arguments.

    Returns
    -------
    int
        Process exit status; non-zero if any problem lacks a ground-truth expression.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default="strogatz,feynman_remainder")
    parser.add_argument("--all", action="store_true", help="Every registered suite.")
    parser.add_argument("--format", choices=("latex", "markdown"), default="markdown")
    parser.add_argument("--output", default=None, help="Write here instead of stdout.")
    args = parser.parse_args(argv)

    suites = list(_BENCHMARK_REGISTRY) if args.all else args.suites.split(",")
    rows = [r for s in suites for r in rows_for(s.strip())]
    text = render(rows, args.format)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        log.info("wrote %d rows to %s", len(rows), args.output)
    else:
        log.info("%s", text)

    missing = [r["name"] for r in rows if r["gt"] == "NO"]
    if missing:
        log.warning(
            "%d problem(s) have no sympy ground truth, so solution_recovered is "
            "not computable for them: %s",
            len(missing),
            ", ".join(missing),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
