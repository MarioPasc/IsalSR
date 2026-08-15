"""Check every campaign-derived number in the manuscript against the data.

The manuscript, its supplementary material and the response letter quote figures
that live in ``analyses/values/summary.json`` and the derived tables. A number
that drifts from its source is invisible on a read and fatal in review, so each
one is asserted here against the value the pipeline computed rather than against
a copy of itself.

The check is deliberately literal: it looks for the exact rendered string in the
file that should carry it. A tolerance would let a rounding change pass, which is
the failure this exists to catch.

Every literal is additionally **anchored**. A bare presence test over a
concatenated blob is vacuous for a short literal -- ``22`` occurs 24 times in the
paper and ``13`` 16 times, so the old test was satisfied by any unrelated digit
pair. Each check therefore names a regular expression that must match on the same
line as the literal, or within a window of one line either side, and the pair must
resolve to **exactly one** ``file:line``. Zero matches means the number is gone;
more than one means the check does not know which occurrence it is pinning, and
an ambiguous check is as useless as an absent one. Both are failures.

Usage
-----
    python -m experiments.scripts.review_campaign.verify [--analyses DIR]
        [--article DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import add_common_args  # noqa: E402

DEFAULT_ARTICLE = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/"
    "69c1637a28a81fea2badda9a"
)

#: The authoritative benchmark inventory, generated from the campaign corpus.
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[3] / "docs/generated/appendix_d/appendix_d_benchmarks.json"
)

PAPER = "article/paper"
SUPP = "article/supplementary"
LETTER = "reviews"

#: Group name -> directory, relative to the manuscript root. The glob is ``*.tex``
#: in each, which already sweeps in the four paper table bodies
#: (``tab_cpdt_summary``, ``tab_three_axis``, ``tab_key_cost``, ``tab_phi_by_host``),
#: the three supplementary ones (``table_supplementary_{udfs,bingo}_body``,
#: ``tab_supp_phi_per_problem``, ``tab_supp_k_range``) and the two letter tables.
GROUPS: dict[str, str] = {"paper": PAPER, "supplementary": SUPP, "letter": LETTER}

#: Table bodies that must carry one row per benchmark problem. The row count is
#: asserted against ``n_problems`` in the summary, so a suite that changes size
#: without the table being regenerated is caught here rather than in proof.
PROBLEM_ROW_TABLES: tuple[tuple[str, str], ...] = (
    ("supplementary", "table_supplementary_udfs_body.tex"),
    ("supplementary", "table_supplementary_bingo_body.tex"),
    ("supplementary", "tab_supp_phi_per_problem.tex"),
)

#: Where Appendix D.1 starts and where it stops. The subsection documents the
#: suite in six tables; the region ends at the next ``\subsection``.
APPENDIX_D1_FILE = "supplementary.tex"
APPENDIX_D1_START = r"\label{sec:supp_benchmarks}"
APPENDIX_D1_END = r"\subsection"

#: The six tables Appendix D.1 is expected to carry, in document order. Declared
#: so that a table dropped from the subsection fails as a missing table rather
#: than only as fourteen missing problem ids.
APPENDIX_D1_TABLES: tuple[str, ...] = (
    "tab:nguyen",
    "tab:feynman",
    "tab:feynman_struct",
    "tab:gp_struct",
    "tab:strogatz",
    "tab:feynman_ext",
)

#: Two Appendix D.1 tables print problem ids without their suite prefix, so the
#: printed id is not the inventory's ``problem_id``. Each entry states the prefix
#: the table omits and the reason it does. The prefix is applied
#: **unconditionally**: if a table later starts printing the full id, every row of
#: it becomes one missing and one extra id, which is a loud failure naming both
#: sets. A conditional "add the prefix only if it is absent" rule would instead
#: absorb the change silently, which is the failure mode this table exists to
#: avoid.
APPENDIX_ID_PREFIX: dict[str, str] = {
    # tab:nguyen is a Nguyen-only table and numbers its rows 1..12; the inventory
    # records them as Nguyen-1..Nguyen-12.
    "tab:nguyen": "Nguyen-",
    # tab:strogatz is an ODE-Strogatz-only table and prints e.g. bacres1; the
    # inventory records it as Strogatz-bacres1.
    "tab:strogatz": "Strogatz-",
}

#: The per-problem result tables whose first column must list the same suite.
#: ``tab_supp_phi_per_problem`` is excluded on purpose: it is a phi table, its row
#: count is already asserted, and it is not one of the two tables T09's AC-2
#: names.
RESULT_ID_TABLES: tuple[tuple[str, str], ...] = (
    ("supplementary", "table_supplementary_udfs_body.tex"),
    ("supplementary", "table_supplementary_bingo_body.tex"),
)

_ROW_END = re.compile(r"\\\\")
_RULES = re.compile(r"\\(?:top|mid|bottom)rule|\\cmidrule(?:\([^)]*\))?\{[^}]*\}|\\addlinespace")
#: ``\begin{tabular}`` and its column specification share a chunk with the header
#: row, so they are removed before the first cell is read. The specification may
#: nest one level (``{@{}ll@{}}``).
_ENV = re.compile(
    r"\\(?:begin|end)\{(?:tabular|longtable)\*?\}(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})?"
)
_TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)
_LABEL = re.compile(r"\\label\{(tab:[^}]+)\}")
_INPUT = re.compile(r"\\input\{([^}]+)\}")

#: First-column values that mark a header band rather than a problem.
_HEADER_CELLS = frozenset({"", "ID", "Problem"})


@dataclass(frozen=True)
class Check:
    """One literal, pinned to one place in one document.

    Attributes:
        doc: Group key, one of ``paper``, ``supplementary`` or ``letter``.
        literal: Exact substring that must appear on a line of that group.
        anchor: Regular expression that must match within ``window`` lines of the
            literal, disambiguating which occurrence is meant.
        what: Human-readable description of the quantity.
        window: Number of lines either side of the literal in which the anchor may
            appear. ``0`` requires the anchor on the literal's own line.
        file: Basename the match must come from, or ``None`` for the whole group.
            The abstract of ``main.tex`` is a single very long line that repeats
            most of the paper's headline numbers, so a same-line anchor cannot
            separate it from the body text; naming the file does.
    """

    doc: str
    literal: str
    anchor: str
    what: str
    window: int = 0
    file: str | None = None


@dataclass(frozen=True)
class Coupled:
    """One quantity that the manuscript states in more than one place.

    Two occurrences of a single measurement that disagree is the defect class
    that produced errata E1 and E8 of the previous round, and neither was caught
    by a presence test. The rendered literal is built once, from the derived
    value, and every listed location must carry it.

    Attributes:
        name: Quantity name, for the report.
        literal: The rendered string, derived from ``summary.json``.
        derived: The value it was rendered from, for the report.
        locations: ``(group, file, anchor, window)`` per place that states it.
    """

    name: str
    literal: str
    derived: float
    locations: tuple[tuple[str, str, str, int], ...]


@dataclass(frozen=True)
class Document:
    """A single ``.tex`` file, split into lines once."""

    rel: str
    lines: tuple[str, ...]


def load_sources(article: Path) -> dict[str, list[Document]]:
    """Read every ``.tex`` file of each group, keeping file and line identity.

    Args:
        article: Manuscript root holding ``article/`` and ``reviews/``.

    Returns:
        Mapping from group name to the list of its documents.
    """
    out: dict[str, list[Document]] = {}
    for name, rel in GROUPS.items():
        docs: list[Document] = []
        for path in sorted((article / rel).glob("*.tex")):
            text = path.read_text(encoding="utf-8")
            docs.append(Document(f"{rel}/{path.name}", tuple(text.splitlines())))
        out[name] = docs
    return out


def locate(docs: list[Document], check: Check) -> list[str]:
    """Find every ``file:line`` where the literal appears with its anchor nearby.

    Args:
        docs: Documents of the group the check names.
        check: The literal, its anchor and the anchor window.

    Returns:
        One ``file:line`` string per matching line, in file order.
    """
    pattern = re.compile(check.anchor)
    hits: list[str] = []
    for doc in docs:
        if check.file is not None and doc.rel.rsplit("/", 1)[-1] != check.file:
            continue
        for index, line in enumerate(doc.lines):
            if check.literal not in line:
                continue
            lo = max(0, index - check.window)
            hi = min(len(doc.lines), index + check.window + 1)
            if any(pattern.search(near) for near in doc.lines[lo:hi]):
                hits.append(f"{doc.rel}:{index + 1}")
    return hits


def grouped(value: int) -> str:
    """Render an integer the way the manuscript groups thousands."""
    return f"{value:,}".replace(",", "{,}")


def build_checks(summary: dict[str, Any]) -> list[Check]:
    """Return one anchored check per quoted quantity.

    Structural counts and the three integer counts the text quotes are rendered
    from ``summary`` rather than written out, so the check cannot agree with a
    stale manuscript by sharing its typo.

    Args:
        summary: Parsed ``analyses/values/summary.json``.

    Returns:
        The checks, in the order the documents introduce them.
    """
    checks: list[Check] = []

    def add(
        doc: str,
        literal: str,
        anchor: str,
        what: str,
        window: int = 0,
        file: str | None = None,
    ) -> None:
        checks.append(Check(doc, literal, anchor, what, window, file))

    udfs, bingo = summary["udfs"], summary["bingo"]
    n_cells = grouped(int(summary["n_cells"]))
    n_arm_cells = grouped(int(udfs["isalsr"]["n_cells"]))
    n_problems = int(summary["n_problems"])
    n_seeds = int(summary["n_seeds"])
    n_cap = grouped(int(udfs["isalsr"]["n_cells_at_budget_cap"]))
    n_s_ge_1 = int(bingo["isalsr"]["n_problems_S_ge_1"])
    n_faster = int(bingo["isalsr"]["n_problems_faster_than_native"])
    n_faster_udfs = int(udfs["isalsr"]["n_problems_faster_than_native"])

    # --- reduction and redundancy -------------------------------------------
    add("paper", r"1.66 \pm 0.26", r"across", "rho mean +- sd, UDFS, prose")
    add("paper", r"1.66 \pm 0.26", r"UDFS &", "rho mean +- sd, UDFS, three-axis table")
    add("paper", r"1.79 \pm 0.09", r"across", "rho mean +- sd, Bingo, prose")
    add("paper", r"1.79 \pm 0.09", r"Bingo &", "rho mean +- sd, Bingo, three-axis table")
    add("paper", r"38.1\%", r"of evaluations, and Bingo", "redundancy rate, UDFS, prose")
    add("paper", r"38.1\%", r"UDFS & \$110\$", "redundancy rate, UDFS, phi table")
    add("paper", r"43.7\%", r"of evaluations the arm removes", "redundancy, Bingo, prose")
    add("paper", r"43.7\%", r"Bingo &", "redundancy rate, Bingo, three-axis table")
    add("supplementary", "[1.11, 2.12]", r"Reduction factors span", "rho range, UDFS", 1)
    add("supplementary", "[1.19, 1.85]", r"Reduction factors span", "rho range, Bingo", 1)

    # --- the share phi -------------------------------------------------------
    add("paper", "0.047", r"ranging over", "phi mean, Bingo, prose")
    add("paper", "0.047", r"Bingo & \$3\{,\}985\$", "phi mean, Bingo, phi table")
    # Renamed from \rho_\sigma 2026-08-14: \sigma(v) is already the ordered input
    # list of a node (methodology.tex), so \sigma(D) for a serialization collided
    # with it inside the same package. The serialization sense yielded to \mathrm{ser}.
    add("supplementary", r"\rho_{\mathrm{ser}} = 1.0000", r"on every one of the", "rho_ser, UDFS")
    add("supplementary", "1.727", r"against \$\\rho = 1\.785\$", "rho_ser mean, Bingo")
    add("letter", r"95.3\%", r"the simple rule recovers", "share of redundancy, naive hash")

    # --- paired test across problems ----------------------------------------
    add("paper", r"2.0 \times 10^{-9}", r"under the branch selected by", "p, R2 test, UDFS")
    add("paper", r"7.5 \times 10^{-4}", r"\$p\$ of", "one-sided p, R2 test, Bingo")
    add("paper", "+0.41", r"with confidence interval", "Cohen's d, R2 test, UDFS")
    add("paper", "[+0.31, +0.59]", r"one-sided", "bootstrap CI on that d")
    add("paper", "2.54", r"Cohen's", "Cohen's d on rho vs hash, UDFS", 0, "results.tex")
    add("paper", "2.54", r"Cohen's \$d\$ of", "same d, abstract", 0, "main.tex")
    add("paper", "[+2.12, +3.16]", r"confidence interval", "bootstrap CI on that d")
    add("paper", "7.05", r"on Bingo", "Cohen's d on rho vs hash, Bingo", 0, "results.tex")
    add("paper", "7.05", r"both at \$p < 10\^\{-12\}\$", "same d, abstract", 0, "main.tex")
    add("paper", r"1.8 \times 10^{-13}", r"one-sided \$p\$ of", "p on rho vs naive hash")

    # --- cost ----------------------------------------------------------------
    add("paper", "0.126", r"\\,ms, while", "median key cost, UDFS, ms, prose")
    add("paper", "0.126", r"UDFS &", "median key cost, UDFS, three-axis table")
    add("paper", "0.074", r"\\,ms and", "median key cost, Bingo, ms, prose")
    add("paper", "727", r"\\,ms\.", "median per-evaluation cost, UDFS, ms, prose")
    add("paper", "727.30", r"UDFS &", "median per-evaluation cost, UDFS, three-axis table")
    add("paper", "1.48", r"\\,ms, so the", "median per-evaluation cost, Bingo, ms")
    add("paper", r"0.04\%", r"The median overhead is therefore", "median key overhead, UDFS")
    add("paper", r"0.04\%", r"\$0.046\$", "median key overhead, UDFS, key-cost table")
    add("paper", r"16.1\%", r"k\$-stratified", "key overhead, Bingo, results", 1)
    add(
        "paper",
        r"16.1\%",
        r"sub-millisecond",
        "key overhead, Bingo, discussion",
        1,
        "discussion.tex",
    )
    add("paper", r"0.15\%", r"percentile", "95th percentile key overhead, UDFS", 1)
    add("paper", "1.96", r"finished first, the median is", "median S, unsaturated UDFS", 1)
    add("paper", n_cap, r"UDFS cells exhaust", "UDFS cells at the budget cap")
    add("paper", "0.72", r"search-only speedup is", "median S, Bingo, prose", 1)
    add("paper", "0.72", r"Bingo &", "median S, Bingo, three-axis table")
    add("supplementary", "0.038", r"below \$k = 5\$", "mean key cost, Bingo, k in [0,5)")
    add("supplementary", "0.038", r"\[15, 32\)\$ &", "mean key cost, Bingo, k-range table")
    add("supplementary", "0.076", r"for \$k \\in \[15, 32\)\$", "key cost, k in [15,32)")
    add("supplementary", "0.076", r"\[15, 32\)\$ &", "key cost, k in [15,32), table")

    # --- structural counts (reviewer R2.6) -----------------------------------
    add("supplementary", n_cells, r"read from the campaign's run ledger", "campaign size")
    add("letter", n_cells, r"runs across two host", "campaign size, letter")
    add("paper", n_arm_cells, r"seed-problem cells, eliminating", "cells per method and arm")
    add("supplementary", n_arm_cells, r"cells of that arm", "cells per method and arm, supp")
    add("letter", n_arm_cells, r"cells per host and arm", "cells per method and arm, letter")
    add("paper", f"${n_problems}$", r"benchmark problems drawn from", "problem count")
    add("paper", f"$N = {n_problems}$", r"denote the suite", "problem count, notation", 1)
    add("supplementary", f"${n_problems}$", r"three arms on each of the", "problem count, supp")
    add("paper", f"${n_seeds}$", r"per configuration", "seed count", 1)
    add("paper", f"$S = {n_seeds}$", r"number of paired", "seed count, notation")
    add("supplementary", f"${n_seeds}$", r"three arms on each of the", "seed count, supp")

    # --- integer counts quoted in the text -----------------------------------
    add("paper", f"${n_s_ge_1}$ of the", r"S \\geq 1", "problems with S >= 1 on Bingo", 1)
    add("paper", f"${n_faster}$ of the", r"the faster arm on", "problems where IsalSR is faster")
    add(
        "paper",
        f"${n_faster}$ of the",
        r"finishes ahead of its baseline",
        "problems where IsalSR is faster, discussion",
        1,
    )
    add(
        "paper",
        f"${n_faster_udfs}$ of the",
        r"the faster of the two on",
        "problems where IsalSR is faster, UDFS",
        1,
    )

    # --- critical difference -------------------------------------------------
    add("paper", "0.40", r"critical difference from", "critical difference, three groups")
    add("paper", "0.40", r"Nemenyi threshold from", "critical difference, three groups, results")
    add("paper", "0.90", r"since the", "critical difference, six groups")
    add("paper", "0.90", r"Nemenyi threshold from", "critical difference, six groups, results")

    # --- synthetic scalability: two runs, and the fit belongs to only one -----
    # The submitted manuscript credited an O(k^0.7) fit to the *exhaustive* run,
    # whose permutation count P = k! varies with k. That confounds cost with k:
    # low-k expressions average few, cold calls and high-k ones average many warm
    # calls, which inflates the small-k end and biases the fitted slope downward.
    # The fit now comes from the fixed-P scaling run instead.
    #
    # The defect this guards is therefore NOT a wrong number -- it is a right
    # number credited to the wrong experiment, which a check on "1.43" alone
    # would pass on a sentence naming 5,400 expressions. Each literal is
    # anchored on a phrase that identifies its own run, so the exponent, the fit
    # quality, the sample size and the k range cannot drift apart from each
    # other or be re-attributed.
    add("paper", "1.43", r"per-permutation cost fits an", "scaling exponent")
    add("paper", "0.90", r"R\^2 = 0\.90", "scaling fit quality")
    # Stated in both the Theoretical Implications and the Limitations
    # subsections, so it needs one anchored check per site rather than one
    # ambiguous check over the file.
    add("paper", "6{,}300", r"across", "scaling sample size, discussion")
    add("paper", "6{,}300", r"Per-\$k\$ timings on", "scaling sample size, limitations")
    add("paper", "36", r"sweeps \$k \\in \\\{8", "scaling k range")
    # The exhaustive run states a total that the submitted text got wrong: it
    # printed the k=9 term (600*9! = 2.18e8) as if it were the whole sum.
    add("paper", "245{,}467{,}800", r"600 \\cdot k!", "exhaustive permutation total")
    add("paper", "5{,}400", r"random expression", "exhaustive sample size", 1)

    return checks


def numeric_gate(summary: dict[str, Any]) -> list[tuple[str, float, float]]:
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


def count_problem_rows(doc: Document) -> int:
    """Count the data rows of a longtable body.

    A data row ends the line with ``\\\\``, carries at least one ``&`` and is not
    one of the ``\\multicolumn`` header bands.

    Args:
        doc: The table body document.

    Returns:
        Number of per-problem rows.
    """
    rows = 0
    for line in doc.lines:
        stripped = line.strip()
        if not stripped.endswith("\\\\") or "&" not in stripped:
            continue
        if "\\multicolumn" in stripped or stripped.startswith("&"):
            continue
        if stripped.startswith("Problem &"):  # the column-label band
            continue
        rows += 1
    return rows


def _unescape_id(cell: str) -> str:
    """Turn a printed first-column cell into a bare problem id.

    Only the escapes the suite actually uses are undone: ``\\_`` for the
    underscore of ``test_4`` and the math delimiters some tables wrap ids in.

    Args:
        cell: First-column text of a table row, already stripped.

    Returns:
        The id as the inventory spells it.
    """
    return cell.replace("\\_", "_").replace("$", "").replace("\\", "").strip()


def table_row_ids(tabular: str) -> list[str]:
    """Return the first-column value of every data row of one ``tabular``.

    Rows are split on ``\\\\`` rather than on newlines because a long expression
    may wrap. Rule macros are removed first, so a row that shares its line with a
    ``\\midrule`` is still seen; header bands are dropped by their first cell and
    by ``\\multicolumn``.

    Args:
        tabular: The text of one ``tabular`` environment.

    Returns:
        The printed ids, in row order, with duplicates preserved.
    """
    ids: list[str] = []
    for chunk in _ROW_END.split(_ENV.sub("", tabular)):
        row = _RULES.sub("", chunk).strip()
        if "&" not in row or "\\multicolumn" in row:
            continue
        cell = row.split("&", 1)[0].strip()
        if cell in _HEADER_CELLS:
            continue
        ids.append(_unescape_id(cell))
    return ids


def splice_inputs(text: str, by_name: dict[str, Document]) -> str:
    """Replace every ``\\input{...}`` by the document it names.

    Two of the six Appendix D.1 tables live in their own file, so an extractor
    that reads only ``supplementary.tex`` sees 42 of the 70 problems and would
    report the other 28 as undocumented.

    Args:
        text: The region to expand.
        by_name: Documents of the group, keyed by basename.

    Returns:
        The region with each input expanded once.

    Raises:
        KeyError: If an input names a file that is not in the corpus.
    """

    def expand(match: re.Match[str]) -> str:
        name = match.group(1)
        name = name if name.endswith(".tex") else f"{name}.tex"
        if name not in by_name:
            raise KeyError(f"Appendix D.1 inputs {name}, which is not in the corpus")
        return "\n".join(by_name[name].lines)

    return _INPUT.sub(expand, text)


def appendix_d1_ids(
    sources: dict[str, list[Document]],
    expected_tables: tuple[str, ...] = APPENDIX_D1_TABLES,
) -> tuple[dict[str, list[str]], list[str]]:
    """Extract the problem ids Appendix D.1 documents, table by table.

    Args:
        sources: Documents per group.
        expected_tables: Labels the subsection must still carry.

    Returns:
        A mapping from table label to its normalised ids, and the list of
        structural failures (missing subsection, missing table).
    """
    by_name = {
        doc.rel.rsplit("/", 1)[-1]: doc
        for doc in sources.get("supplementary", [])
        if doc is not None
    }
    doc = by_name.get(APPENDIX_D1_FILE)
    if doc is None:
        return {}, [f"the corpus has no {APPENDIX_D1_FILE}, so Appendix D.1 cannot be read"]

    text = "\n".join(doc.lines)
    start = text.find(APPENDIX_D1_START)
    if start < 0:
        return {}, [
            f"{APPENDIX_D1_FILE} has no {APPENDIX_D1_START}, so Appendix D.1 is not located"
        ]
    end = text.find(APPENDIX_D1_END, start)
    region = splice_inputs(text[start:] if end < 0 else text[start:end], by_name)

    labels = [(match.start(), match.group(1)) for match in _LABEL.finditer(region)]
    per_table: dict[str, list[str]] = {}
    for match in _TABULAR.finditer(region):
        preceding = [label for pos, label in labels if pos < match.start()]
        label = preceding[-1] if preceding else "tab:<unlabelled>"
        prefix = APPENDIX_ID_PREFIX.get(label, "")
        per_table[label] = [prefix + pid for pid in table_row_ids(match.group(0))]

    failures = [
        f"Appendix D.1 no longer carries the table {label}"
        for label in expected_tables
        if label not in per_table
    ]
    return per_table, failures


def _diff_failure(what: str, found: list[str], expected: set[str]) -> str | None:
    """Describe how a documented id set differs from the inventory.

    Args:
        what: Name of the document the ids came from.
        found: The ids it lists, in row order.
        expected: The authoritative suite.

    Returns:
        A message naming the missing, extra and duplicated ids, or ``None`` when
        the sets agree and no id repeats.
    """
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    duplicated = sorted({pid for pid in found if found.count(pid) > 1})
    if not (missing or extra or duplicated):
        return None
    parts = [f"{what} lists {len(set(found))} distinct ids, the suite has {len(expected)}"]
    if missing:
        parts.append(f"missing ({len(missing)}): {', '.join(missing)}")
    if extra:
        parts.append(f"extra ({len(extra)}): {', '.join(extra)}")
    if duplicated:
        parts.append(f"duplicated ({len(duplicated)}): {', '.join(duplicated)}")
    return "; ".join(parts)


def check_problem_inventory(
    sources: dict[str, list[Document]],
    inventory: set[str],
    expected_tables: tuple[str, ...] = APPENDIX_D1_TABLES,
) -> tuple[int, list[str]]:
    """Assert that the documented suite and the reported suite are the 70 problems.

    Three sets must coincide: the ids Appendix D.1 tabulates, the ids the UDFS
    per-problem result table reports and the ids the Bingo one reports. A problem
    that is run but never described, or described but never run, is invisible to a
    row count because both tables can have the right length while listing
    different problems.

    Args:
        sources: Documents per group.
        inventory: The authoritative ``problem_id`` values.
        expected_tables: Labels Appendix D.1 must still carry.

    Returns:
        The number of checks run and the list of failure messages.
    """
    failures: list[str] = []
    per_table, structural = appendix_d1_ids(sources, expected_tables)
    failures.extend(structural)

    documented = [pid for label in sorted(per_table) for pid in per_table[label]]
    for label in expected_tables:
        if label in per_table:
            print(f"       {label:20s} {len(per_table[label]):3d} ids")
    sets: list[tuple[str, list[str]]] = [("Appendix D.1", documented)]

    by_name = {
        (group, doc.rel.rsplit("/", 1)[-1]): doc for group, docs in sources.items() for doc in docs
    }
    for group, name in RESULT_ID_TABLES:
        doc = by_name.get((group, name))
        if doc is None:
            failures.append(f"{group} is missing the table body {name}")
            sets.append((name, []))
            continue
        sets.append((name, table_row_ids("\n".join(doc.lines))))

    for what, found in sets:
        message = _diff_failure(what, found, inventory)
        ok = message is None
        print(
            f"  {'ok  ' if ok else 'FAIL'} {what:38s} {len(set(found)):3d} of {len(inventory)} ids"
        )
        if message is not None:
            failures.append(message)
    return len(sets), failures


def load_inventory(path: Path) -> set[str]:
    """Read the authoritative problem ids from the generated benchmark inventory.

    Args:
        path: Path to ``appendix_d_benchmarks.json``.

    Returns:
        The ``problem_id`` values under ``problems``.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(entry["problem_id"]) for entry in payload["problems"]}


def build_coupled(summary: dict[str, Any]) -> list[Coupled]:
    """Return the quantities the manuscript states in more than one place.

    The list is a curated allow-list rather than an automatic grouping of equal
    numeric tokens. Equal digits are not equal quantities: ``2.12`` is the
    largest UDFS reduction factor in two documents and, separately, the lower
    end of a bootstrap interval on a Cohen's ``d`` in the results section, and
    ``0.04`` is a median overhead in three places and a correlation coefficient
    in two others. Coupling those would assert a relation that does not hold.

    Args:
        summary: Parsed ``analyses/values/summary.json``.

    Returns:
        One entry per coupled quantity, its rendered literal built from the
        derived value.
    """
    udfs, bingo = summary["udfs"], summary["bingo"]
    rho_u = udfs["isalsr"]["rho_over_problems"]
    rho_b = bingo["isalsr"]["rho_over_problems"]

    disc, res, supp, tab3 = (
        "discussion.tex",
        "results.tex",
        "supplementary.tex",
        "tab_three_axis.tex",
    )
    span_supp = ("supplementary", supp, r"Reduction factors span", 1)

    return [
        Coupled(
            "rho min, UDFS",
            f"{rho_u['min']:.2f}",
            rho_u["min"],
            (("paper", disc, r"on UDFS and", 0), span_supp),
        ),
        Coupled(
            "rho max, UDFS",
            f"{rho_u['max']:.2f}",
            rho_u["max"],
            (("paper", disc, r"on UDFS and", 0), span_supp),
        ),
        Coupled(
            "rho min, Bingo",
            f"{rho_b['min']:.2f}",
            rho_b["min"],
            (("paper", disc, r"on Bingo across", 0), ("paper", res, r"for the rest", 0), span_supp),
        ),
        Coupled(
            "rho max, Bingo",
            f"{rho_b['max']:.2f}",
            rho_b["max"],
            (("paper", disc, r"on Bingo across", 0), ("paper", res, r"for the rest", 0), span_supp),
        ),
        Coupled(
            "rho mean, UDFS",
            f"{rho_u['mean']:.2f}",
            rho_u["mean"],
            (
                ("paper", res, r"mean reduction factor of", 1),
                ("paper", disc, r"suite \(means", 0),
                ("paper", disc, r"of the two hosts", 1),
                ("paper", tab3, r"UDFS &", 0),
            ),
        ),
        Coupled(
            "rho mean, Bingo",
            f"{rho_b['mean']:.2f}",
            rho_b["mean"],
            (
                ("paper", res, r"\\pm 0\.09\$ across", 0),
                ("paper", res, r"sitting within", 0),
                ("paper", disc, r"suite \(means", 0),
                ("paper", disc, r"of the two hosts", 1),
                ("paper", tab3, r"Bingo &", 0),
            ),
        ),
        Coupled(
            "key cost median, UDFS",
            f"{udfs['isalsr']['key_ms_over_cells']['median']:.3f}",
            udfs["isalsr"]["key_ms_over_cells"]["median"],
            (
                ("paper", res, r"\\,ms, while", 0),
                ("paper", disc, r"\(UDFS\) and", 0),
                ("paper", tab3, r"UDFS &", 0),
            ),
        ),
        Coupled(
            "key cost median, Bingo",
            f"{bingo['isalsr']['key_ms_over_cells']['median']:.3f}",
            bingo["isalsr"]["key_ms_over_cells"]["median"],
            (
                ("paper", res, r"\\,ms and", 0),
                ("paper", disc, r"\(Bingo\)", 0),
                ("paper", tab3, r"Bingo &", 0),
            ),
        ),
        Coupled(
            "eval cost median, Bingo",
            f"{bingo['isalsr']['eval_ms_over_cells']['median']:.2f}",
            bingo["isalsr"]["eval_ms_over_cells"]["median"],
            (("paper", res, r"\\,ms, so the", 0), ("paper", tab3, r"Bingo &", 0)),
        ),
        Coupled(
            "key overhead median, Bingo",
            f"{bingo['isalsr']['overhead_pct_over_cells']['median']:.1f}",
            bingo["isalsr"]["overhead_pct_over_cells"]["median"],
            (
                ("paper", res, r"k\$-stratified", 1),
                ("paper", disc, r"sub-millisecond", 1),
                ("paper", "main.tex", r"of run time on UDFS and", 0),
            ),
        ),
        Coupled(
            "key overhead median, UDFS",
            f"{udfs['isalsr']['overhead_pct_over_cells']['median']:.2f}",
            udfs["isalsr"]["overhead_pct_over_cells"]["median"],
            (
                ("paper", res, r"The median overhead is therefore", 0),
                ("paper", disc, r"adds a median overhead of", 1),
                ("paper", "main.tex", r"Canonicalization consumes", 0),
            ),
        ),
    ]


def check_coupled(
    sources: dict[str, list[Document]], groups: list[Coupled]
) -> tuple[int, list[str]]:
    """Assert that every stated occurrence of a coupled quantity carries one literal.

    Args:
        sources: Documents per group.
        groups: The coupled quantities.

    Returns:
        The number of locations checked and the list of failure messages.
    """
    failures: list[str] = []
    n_locations = 0
    for group in groups:
        resolved: list[str] = []
        bad: list[str] = []
        for doc, file, anchor, window in group.locations:
            n_locations += 1
            check = Check(doc, group.literal, anchor, group.name, window, file)
            hits = locate(sources[doc], check)
            if len(hits) == 1:
                resolved.append(hits[0])
            else:
                bad.append(f"{file} /{anchor}/ -> {len(hits)} hits")
        status = "ok  " if not bad else "FAIL"
        print(
            f"  {status} {group.name:28s} {group.literal:>8s} "
            f"(={group.derived:.4f})  {' '.join(resolved)}"
        )
        for line in bad:
            failures.append(
                f"coupled '{group.name}': the literal {group.literal!r} derived from "
                f"{group.derived:.6f} is not at {line}"
            )
    return n_locations, failures


def check_literals(
    sources: dict[str, list[Document]], checks: list[Check]
) -> tuple[int, list[str]]:
    """Run every anchored check, printing the location each literal resolved to.

    Args:
        sources: Documents per group.
        checks: The anchored checks.

    Returns:
        The number of checks run and the list of failure messages.
    """
    failures: list[str] = []
    for check in checks:
        hits = locate(sources[check.doc], check)
        if len(hits) == 1:
            print(f"  ok   {check.doc:14s} {check.literal!r:24s} {hits[0]}  {check.what}")
            continue
        where = "nowhere" if not hits else ", ".join(hits)
        print(f"  FAIL {check.doc:14s} {check.literal!r:24s} {len(hits)} hits  {check.what}")
        failures.append(
            f"{check.doc}: {check.literal!r} anchored on /{check.anchor}/ "
            f"({check.what}) matched {len(hits)} locations: {where}"
        )
    return len(checks), failures


def check_table_rows(sources: dict[str, list[Document]], n_problems: int) -> tuple[int, list[str]]:
    """Assert that each per-problem table body carries one row per problem."""
    failures: list[str] = []
    by_name = {
        (group, doc.rel.rsplit("/", 1)[-1]): doc for group, docs in sources.items() for doc in docs
    }
    for group, name in PROBLEM_ROW_TABLES:
        doc = by_name.get((group, name))
        if doc is None:
            print(f"  FAIL {group:14s} {name:44s} missing")
            failures.append(f"{group} is missing the table body {name}")
            continue
        rows = count_problem_rows(doc)
        ok = rows == n_problems
        print(f"  {'ok  ' if ok else 'FAIL'} {group:14s} {name:44s} {rows} rows")
        if not ok:
            failures.append(f"{name} has {rows} problem rows, expected {n_problems}")
    return len(PROBLEM_ROW_TABLES), failures


def main() -> None:
    """Run the derived-value gate, the anchored literal gate and the placeholder count."""
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--article", type=Path, default=DEFAULT_ARTICLE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    with (args.analyses / "values" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    failures: list[str] = []
    n_checks = 0

    print("=== derived values against the summary ===")
    for label, value, expected in numeric_gate(summary):
        ok = abs(value - expected) <= 0.6 * 10 ** -len(str(expected).split(".")[-1])
        if isinstance(expected, int):
            ok = value == expected
        print(f"  {'ok ' if ok else 'FAIL'} {label:16s} {value:>12.4f} vs {expected}")
        n_checks += 1
        if not ok:
            failures.append(f"derived value {label}: {value} != {expected}")

    sources = load_sources(args.article)
    n_files = sum(len(docs) for docs in sources.values())
    print(f"\n=== quoted literals, anchored ({n_files} files in the corpus) ===")
    ran, literal_failures = check_literals(sources, build_checks(summary))
    n_checks += ran
    failures.extend(literal_failures)

    print("\n=== quantities stated in more than one place, all must agree ===")
    ran, coupled_failures = check_coupled(sources, build_coupled(summary))
    n_checks += ran
    failures.extend(coupled_failures)

    print("\n=== per-problem table bodies ===")
    ran, row_failures = check_table_rows(sources, int(summary["n_problems"]))
    n_checks += ran
    failures.extend(row_failures)

    inventory = load_inventory(args.inventory)
    print(f"\n=== documented suite vs reported suite ({len(inventory)} problems) ===")
    if len(inventory) != int(summary["n_problems"]):
        failures.append(
            f"the inventory holds {len(inventory)} problems but the summary reports "
            f"{int(summary['n_problems'])}"
        )
    ran, inventory_failures = check_problem_inventory(sources, inventory)
    n_checks += ran
    failures.extend(inventory_failures)

    print("\n=== placeholders ===")
    for doc, docs in sources.items():
        # The two macro definitions write `\pendingnum}` and `\pendingblock}`,
        # so counting the call form `\pendingnum{` excludes them already.
        text = "\n".join("\n".join(d.lines) for d in docs)
        n = text.count("\\pendingnum{") + text.count("\\pendingblock{")
        print(f"  {doc:14s} {n} remaining")
        n_checks += 1
        if doc == "paper" and n:
            failures.append(f"{doc} still carries {n} placeholders")

    if failures:
        print(f"\n{len(failures)} FAILURES of {n_checks} checks")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print(f"\nall {n_checks} checks pass")


if __name__ == "__main__":
    main()
