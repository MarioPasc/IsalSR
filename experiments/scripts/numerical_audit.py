"""Inventory, classify and cross-link every numeric literal in the manuscript.

The tool answers one question for a round-2 reviewer: *where does each number in
the paper come from, and is the same number stated twice?* It emits two
artefacts from a single pass over the six audited LaTeX files:

* ``numerical_audit.json`` -- one record per numeric literal, plus the
  cross-file duplicate report and the per-kind / bound-vs-unbound tallies.
* ``numerical_audit.md`` -- the same content rendered for a human reader.

Three design commitments make the output usable as evidence rather than as a
heuristic summary:

* **Completeness is positional.** Every literal is recorded with the exact
  ``(line, col_start, col_end)`` span it occupies, so a test can blank every
  recorded span and assert that no digit survives anywhere in the file. There
  is no "interesting numbers only" filter.
* **Classification is conservative.** ``kind`` is drawn from a closed enum and
  every rule that moves a literal *out of* ``measurement`` is an explicit,
  auditable pattern. Anything unmatched stays ``measurement``, because a
  measurement misfiled as bookkeeping is a number nobody re-checks.
* **Binding is never guessed.** ``source_artefact`` is either a concrete path /
  configuration key that produced the number, or the literal string
  ``"UNBOUND"``. The count of ``UNBOUND`` measurements is the work remaining.

The manuscript tree is a live Overleaf checkout and is opened read-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

log = logging.getLogger(__name__)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_MANUSCRIPT_ROOT: Final[str] = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/"
    "69c1637a28a81fea2badda9a/article"
)

#: The six audited files, relative to the manuscript root, in emission order.
AUDITED_FILES: Final[tuple[str, ...]] = (
    "paper/computational_experiments.tex",
    "paper/results.tex",
    "paper/discussion.tex",
    "supplementary/supplementary.tex",
    "supplementary/table_supplementary_udfs.tex",
    "supplementary/table_supplementary_bingo.tex",
)

#: Short, LaTeX-macro-safe tag per audited file, used to build macro names.
FILE_TAGS: Final[dict[str, str]] = {
    "paper/computational_experiments.tex": "CompExp",
    "paper/results.tex": "Res",
    "paper/discussion.tex": "Disc",
    "supplementary/supplementary.tex": "Supp",
    "supplementary/table_supplementary_udfs.tex": "TabUdfs",
    "supplementary/table_supplementary_bingo.tex": "TabBingo",
}

Kind = Literal[
    "measurement",
    "problem_id_fragment",
    "structural_count",
    "cross_reference",
    "year",
    "typography",
]

#: The closed classification enum, in report order.
KINDS: Final[tuple[Kind, ...]] = (
    "measurement",
    "problem_id_fragment",
    "structural_count",
    "cross_reference",
    "year",
    "typography",
)

UNBOUND: Final[str] = "UNBOUND"

BENCHMARK_ARTEFACT: Final[str] = "docs/generated/appendix_d/appendix_d_benchmarks.json"

MAX_SENTENCE_CHARS: Final[int] = 160


class NumericalAuditError(Exception):
    """Raised when the audit cannot be produced from the manuscript sources."""


# --------------------------------------------------------------------------
# Lexing
# --------------------------------------------------------------------------

#: Scientific-notation composites are lexed as a *single* literal. Reading
#: ``2.7 \times 10^{-22}`` as the three literals 2.7, 10 and 22 would flood the
#: duplicate report with spurious ``10`` collisions coming from exponents.
_SCI_PATTERN: Final[str] = (
    r"(?:\d+(?:\.\d+)?\s*(?:\\times|\{\\times\}|\\cdot|\{\\cdot\})\s*)?"
    r"10\^\{?\s*-?\s*\d+\s*\}?"
)

#: Plain decimals, with optional LaTeX thousands separators (``2{,}640``) and
#: optional plain-comma grouping (``10,000``). The leading-dot alternative is
#: guarded by a lookbehind: without it, ``I.48.20`` lexes as ``.48`` + ``.20``
#: and the problem-identifier token is destroyed before classification sees it.
_PLAIN_PATTERN: Final[str] = (
    r"\d{1,3}(?:(?:\{,\}|,)\d{3})+(?:\.\d+)?"  # grouped: 2{,}640, 10,000
    r"|\d+(?:\.\d+)?"  # 43200.6, 50
    r"|(?<![\w.])\.\d+"  # .5
)

_LITERAL_RE: Final[re.Pattern[str]] = re.compile(f"(?:{_SCI_PATTERN})|(?:{_PLAIN_PATTERN})")

_SCI_RE: Final[re.Pattern[str]] = re.compile(f"^{_SCI_PATTERN}$")

#: Characters that may form part of a problem identifier token.
_TOKEN_CHARS: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._\-]")


@dataclass(frozen=True)
class RawLiteral:
    """A numeric literal located in a source line.

    Attributes:
        line: 1-based line number.
        col_start: 0-based inclusive start column within the line.
        col_end: 0-based exclusive end column within the line.
        text: Verbatim slice of the line.
        in_comment: Whether the literal sits after an unescaped ``%``.
    """

    line: int
    col_start: int
    col_end: int
    text: str
    in_comment: bool


def comment_start(line: str) -> int:
    """Return the column of the first unescaped ``%``, or ``len(line)``.

    LaTeX escapes a literal percent sign as ``\\%``; a backslash that is itself
    escaped (``\\\\%``) does not protect the following ``%``.

    Args:
        line: A single source line, without its terminator.

    Returns:
        Index of the comment marker, or the line length when the line carries
        no comment.
    """
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char == "\\":
            idx += 2
            continue
        if char == "%":
            return idx
        idx += 1
    return len(line)


def extract_literals(text: str) -> list[RawLiteral]:
    """Lex every numeric literal of a LaTeX source, in reading order.

    Args:
        text: Full file contents.

    Returns:
        One :class:`RawLiteral` per literal, ordered by ``(line, col_start)``.
        Spans never overlap, so blanking every span removes every digit.
    """
    out: list[RawLiteral] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        cmt = comment_start(line)
        for match in _LITERAL_RE.finditer(line):
            out.append(
                RawLiteral(
                    line=lineno,
                    col_start=match.start(),
                    col_end=match.end(),
                    text=match.group(0),
                    in_comment=match.start() >= cmt,
                )
            )
    return out


def normalise(literal: str) -> str:
    """Reduce a literal to the canonical form used for duplicate matching.

    LaTeX thousands separators are removed (``2{,}640`` -> ``2640``) and
    scientific composites are folded to ``<mantissa>e<exponent>`` so that
    ``5.9{\\times}10^{-5}`` and ``5.9 \\times 10^{-5}`` collide. Trailing zeros
    are *not* stripped: ``0.050`` and ``0.05`` state different precisions and
    are different claims.

    Args:
        literal: Verbatim literal text.

    Returns:
        Canonical string form.
    """
    compact = re.sub(r"\s+", "", literal)
    if _SCI_RE.match(compact):
        mantissa = "1"
        body = compact
        # Braced forms first: ``\times`` is a substring of ``{\times}`` and
        # splitting on the bare form would leave the brace on the mantissa.
        for sep in ("{\\times}", "{\\cdot}", "\\times", "\\cdot"):
            if sep in body:
                mantissa, body = body.split(sep, 1)
                break
        exponent = re.sub(r"[^0-9\-]", "", body.split("^", 1)[1])
        return f"{mantissa}e{int(exponent)}"
    return compact.replace("{,}", "").replace(",", "")


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

#: Macros whose braced argument is bookkeeping, never a claim.
_CROSSREF_MACROS: Final[tuple[str, ...]] = (
    "ref",
    "cref",
    "Cref",
    "eqref",
    "autoref",
    "label",
    "cite",
    "citep",
    "citet",
    "input",
    "include",
    "bibliography",
    "bibliographystyle",
)

#: Macros whose arguments control layout, not content.
_TYPOGRAPHY_MACROS: Final[tuple[str, ...]] = (
    "documentclass",
    "usepackage",
    "includegraphics",
    "cmidrule",
    "multicolumn",
    "multirow",
    "hspace",
    "vspace",
    "setlength",
    "rule",
    "arraystretch",
    "addtolength",
    "columnwidth",
    "newtheorem",
    "markboth",
    "newcommand",
    "renewcommand",
    "url",
    "href",
    "IEEEauthorrefmark",
    "setcounter",
)

_MACRO_ARG_RE: Final[re.Pattern[str]] = re.compile(
    r"\\([A-Za-z]+)\s*(?:\([a-z]+\))?\s*(?:\[[^\]]*\]\s*)*\{"
)

#: In-text pointers to a numbered float, section or theorem environment.
_TEXT_CROSSREF_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Section|Sections|Appendix|Appendices|Table|Tables|Figure|Figures|"
    r"Definition|Theorem|Lemma|Proposition|Corollary|Remark|Algorithm|Step|"
    r"Equation|Eq\.|Eqs\.|Fig\.|Sec\.|Tab\.)~?\s*\(?[A-Z]?\.?\d"
)

#: Float placement specifiers, column specifications and dimensions.
_TYPOGRAPHY_CONTEXT_RES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\\begin\{tabular\*?\}"),
    re.compile(r"\\\\\s*\[[^\]]*\]$"),
    re.compile(r"\d+(?:pt|em|ex|cm|mm|in|bp|sp)\b"),
    re.compile(r"(?:width|height|scale|trim|angle)\s*="),
    re.compile(r"\\begin\{(?:table|figure|minipage|subfigure)\*?\}\s*\[[^\]]*\]"),
)

#: Problem-identifier grammars. Matched against the *whole enclosing token*, so
#: a bare ``13.12`` is never reclassified -- only ``I.13.12`` is.
_PROBLEM_ID_RES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^(?:I|II|III)\.\d+(?:\.\d+)?[a-z]?$"),
    re.compile(
        r"^(?:Nguyen|N|Vladislavleva|Vlad|Liv|Livermore|Pagie|Keijzer|Keij|"
        r"Korns|Strogatz|Jin|Neat|Feynman)-\d+[a-z]?$"
    ),
    re.compile(r"^R[0-9]$"),
    re.compile(r"^test_\d+$"),
)

#: Design constants. Each entry is ``(value, context regex, artefact)``; the
#: context regex is matched against a +/-80 character window around the literal.
#: Nothing is classified structurally without both a value and a context match.
_STRUCTURAL_RULES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "0.05",
        r"\\alpha\s*(?:=|\\in)|level\s+\$?\\alpha|p\s*<\s*0\.05",
        "design: significance level alpha",
    ),
    ("0.01", r"\^\{\*\*\}p\s*<", "design: significance-star legend"),
    ("0.001", r"\^\{\*\*\*\}p\s*<", "design: significance-star legend"),
    (
        "95",
        r"(?:bootstrap|confidence interval|\\%\$? (?:two-sided )?confidence|CI)",
        "design: confidence level",
    ),
    ("10000", r"resamples|bootstrap", "design: bootstrap resamples B"),
    ("30", r"seed|\\bS = 30\\b|independent seeds", "campaign design: 30 paired seeds"),
    (
        "50",
        r"(?:\$?N\$? ?= ?50|50\$?-problem|50\$? benchmark problems|the \$50\$ problems)",
        BENCHMARK_ARTEFACT,
    ),
    ("70", r"70\$?-problem", BENCHMARK_ARTEFACT),
    ("12", r"12\$? Nguyen", BENCHMARK_ARTEFACT),
    ("30", r"30\$? AI~?Feynman", BENCHMARK_ARTEFACT),
    ("14", r"14\$? ODE-Strogatz|14\$? use the fixed", BENCHMARK_ARTEFACT),
    ("3", r"3\$? Vladislavleva|3\$? DSO-Livermore|3\$? Koza|3\$? use grid", BENCHMARK_ARTEFACT),
    ("2", r"2\$? Pagie|2\$? Keijzer", BENCHMARK_ARTEFACT),
    ("1", r"1\$? Korns", BENCHMARK_ARTEFACT),
    ("53", r"53\$? problems draw", BENCHMARK_ARTEFACT),
    ("12", r"12\$?-hour budget", "experiments/configs/*.yaml: max_time=43200"),
    (
        "60",
        r"60\$?-second canonicalisation timeout|60\$?\\,s to resolve",
        "campaign config: canonicalisation timeout",
    ),
)

#: Realised dataset sizes, from the Appendix D.1 inventory. A literal matching
#: one of these *in a dataset-size context* binds to that artefact.
_BENCHMARK_SIZE_CONTEXT: Final[re.Pattern[str]] = re.compile(
    r"train|test|sample|point|set size|uniform|grid", re.IGNORECASE
)

_METRIC_MACRO_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"R\^2_\{\\mathrm\{test\}\}|R\^\{2\}_\{\\mathrm\{test\}\}"), "RSqTest"),
    (re.compile(r"R\^2_\{\\mathrm\{train\}\}|R\^\{2\}_\{\\mathrm\{train\}\}"), "RSqTrain"),
    (re.compile(r"\\mathrm\{NRMSE\}|NRMSE"), "Nrmse"),
    (re.compile(r"T_\{\\mathrm\{canon\}\}|canonicalisation time"), "TCanon"),
    (re.compile(r"T_\{\\mathrm\{eval\}\}|evaluation cost"), "TEval"),
    (re.compile(r"\\rho"), "Rho"),
    (re.compile(r"Cohen|\\bd = |\$d\$"), "CohenD"),
    (re.compile(r"overhead"), "Overhead"),
    (re.compile(r"speedup|\$S = |S \\geq"), "Speedup"),
    (re.compile(r"redundancy rate|Red\."), "Redundancy"),
    (re.compile(r"p_\{\\mathrm|p-value|p = |p <"), "PValue"),
    (re.compile(r"R\^2"), "RSq"),
)

_STAT_MACRO_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"median", re.IGNORECASE), "Median"),
    (re.compile(r"mean", re.IGNORECASE), "Mean"),
    (re.compile(r"percentile", re.IGNORECASE), "Pctile"),
)

_DIGIT_WORDS: Final[dict[str, str]] = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def digits_to_words(text: str) -> str:
    """Rewrite a string into letters only, spelling digits out.

    LaTeX control sequences accept letters only, so a macro name derived from a
    line number or a problem identifier cannot carry digits.

    Args:
        text: Arbitrary string.

    Returns:
        Letters-only string; non-alphanumeric characters are dropped.
    """
    out: list[str] = []
    for char in text:
        if char.isdigit():
            out.append(_DIGIT_WORDS[char])
        elif char.isalpha():
            out.append(char)
    return "".join(out)


def enclosing_token(line: str, col_start: int, col_end: int) -> str:
    """Return the maximal identifier-like token containing a literal span.

    Args:
        line: The source line.
        col_start: Inclusive start column of the literal.
        col_end: Exclusive end column of the literal.

    Returns:
        The token, expanded over ``[A-Za-z0-9._-]`` in both directions.
    """
    start = col_start
    while start > 0 and _TOKEN_CHARS.match(line[start - 1]):
        start -= 1
    end = col_end
    while end < len(line) and _TOKEN_CHARS.match(line[end]):
        end += 1
    return line[start:end]


def _open_macro_at(line: str, col: int) -> str | None:
    """Return the macro whose braced argument encloses ``col``, if any.

    Args:
        line: The source line.
        col: Column of interest.

    Returns:
        Macro name without the leading backslash, else ``None``.
    """
    stack: list[str | None] = []
    idx = 0
    while idx < col and idx < len(line):
        match = _MACRO_ARG_RE.match(line, idx)
        if match is not None:
            stack.append(match.group(1))
            idx = match.end()
            continue
        char = line[idx]
        if char == "\\":
            idx += 2
            continue
        if char == "{":
            stack.append(None)
        elif char == "}" and stack:
            stack.pop()
        idx += 1
    for name in reversed(stack):
        if name is not None:
            return name
    return None


def is_notation_script(line: str, col_start: int, col_end: int) -> bool:
    """Decide whether a literal is a single-digit script attached to a symbol.

    The rule is deliberately narrow, because it is the only mechanism that can
    remove a literal from the duplicate report: the literal must be exactly one
    digit, must be the *entire* sub/superscript group, and the base must be a
    single letter (``R^2``, ``L_2``, ``H_0``, ``x_1``) or a control sequence
    (``\\delta_1``). It therefore fires on symbol names and never on

    * multi-character scripts -- ``k^{0.7}`` is a fitted exponent and stays a
      measurement;
    * scripts whose base is a closing brace -- ``\\sum_{k=1}^{9}`` keeps its
      ``9``, which is the largest ``k`` of the synthetic study;
    * coefficients -- the ``2`` of ``2\\sin(x)`` is not a script.

    The classification ``kind`` is unaffected: a notation script stays a
    ``measurement``. Only the duplicate report filters on this flag.

    Args:
        line: The source line.
        col_start: Inclusive start column of the literal.
        col_end: Exclusive end column of the literal.

    Returns:
        ``True`` when the literal is symbol notation.
    """
    if col_end - col_start != 1 or not line[col_start].isdigit():
        return False
    idx = col_start - 1
    braced = idx >= 0 and line[idx] == "{"
    if braced:
        if col_end >= len(line) or line[col_end] != "}":
            return False
        idx -= 1
    if idx < 0 or line[idx] not in "^_":
        return False
    idx -= 1
    if idx < 0 or not line[idx].isalpha():
        return False
    end_of_base = idx
    while idx >= 0 and line[idx].isalpha():
        idx -= 1
    if idx >= 0 and line[idx] == "\\":
        return True
    return end_of_base - idx == 1


def context_window(lines: list[str], literal: RawLiteral, width: int = 80) -> str:
    """Return a raw +/-``width`` character window around a literal.

    The window is drawn from the literal's own line plus its immediate
    neighbours, because LaTeX prose wraps mid-sentence and the disambiguating
    words are routinely on the previous line.

    Args:
        lines: All source lines.
        literal: The literal to centre on.
        width: Half-width, in characters.

    Returns:
        The window, with newlines collapsed to single spaces.
    """
    idx = literal.line - 1
    prev = lines[idx - 1] if idx > 0 else ""
    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
    joined = f"{prev} {lines[idx]} {nxt}"
    centre = len(prev) + 1 + literal.col_start
    lo = max(0, centre - width)
    hi = min(len(joined), centre + len(literal.text) + width)
    return re.sub(r"\s+", " ", joined[lo:hi]).strip()


def surrounding_sentence(lines: list[str], literal: RawLiteral) -> str:
    """Return a trimmed, human-readable context string of at most 160 chars.

    Args:
        lines: All source lines.
        literal: The literal to centre on.

    Returns:
        Whitespace-normalised context, hard-truncated to
        :data:`MAX_SENTENCE_CHARS`.
    """
    window = context_window(lines, literal, width=MAX_SENTENCE_CHARS // 2)
    if len(window) <= MAX_SENTENCE_CHARS:
        return window
    return window[: MAX_SENTENCE_CHARS - 1] + "\u2026"


def classify(lines: list[str], literal: RawLiteral) -> tuple[Kind, str]:
    """Assign a ``kind`` to a literal, conservatively.

    Rule order is fixed: bookkeeping macros, in-text float pointers, problem
    identifiers, layout arguments, publication years, then the design-constant
    whitelist. A literal matching none of these stays ``measurement``.

    Args:
        lines: All source lines.
        literal: The literal to classify.

    Returns:
        ``(kind, rule)`` where ``rule`` names the pattern that fired, or the
        empty string for the default branch.
    """
    line = lines[literal.line - 1]
    window = context_window(lines, literal)

    if literal.in_comment:
        return "typography", "latex-comment (not rendered)"

    macro = _open_macro_at(line, literal.col_start)
    if macro in _CROSSREF_MACROS:
        return "cross_reference", f"argument of \\{macro}"
    if macro in _TYPOGRAPHY_MACROS:
        return "typography", f"argument of \\{macro}"

    prefix = line[: literal.col_end]
    tail = prefix[-60:]
    if _TEXT_CROSSREF_RE.search(tail):
        match = _TEXT_CROSSREF_RE.search(tail)
        assert match is not None
        if match.end() >= len(tail) - (literal.col_end - literal.col_start):
            return "cross_reference", "in-text float/section pointer"

    token = enclosing_token(line, literal.col_start, literal.col_end)
    for pattern in _PROBLEM_ID_RES:
        if pattern.match(token):
            return "problem_id_fragment", f"problem identifier {token!r}"

    for pattern in _TYPOGRAPHY_CONTEXT_RES:
        if pattern.search(line[max(0, literal.col_start - 40) : literal.col_end + 10]):
            return "typography", f"layout construct: {pattern.pattern}"

    normalised = normalise(literal.text)
    if (
        re.fullmatch(r"(?:19|20)\d{2}", literal.text)
        and token == literal.text
        and re.search(r"\bet al\.|\(\d{4}\)|\\cite", window)
    ):
        return "year", "publication year"

    for value, ctx, _artefact in _STRUCTURAL_RULES:
        if normalised == value and re.search(ctx, window):
            return "structural_count", f"design constant: {ctx}"

    return "measurement", ""


def bind_artefact(
    lines: list[str],
    literal: RawLiteral,
    kind: Kind,
    train_sizes: frozenset[str],
    test_sizes: frozenset[str],
) -> str:
    """Resolve the artefact a literal came from, or :data:`UNBOUND`.

    Only two binding sources are trusted: the design-constant whitelist and the
    Appendix D.1 benchmark inventory. Everything else -- in particular every
    campaign result -- is left ``UNBOUND`` rather than guessed.

    Args:
        lines: All source lines.
        literal: The literal to bind.
        kind: Its classification.
        train_sizes: Realised ``n_train`` values, as normalised strings.
        test_sizes: Realised ``n_test`` values, as normalised strings.

    Returns:
        An artefact path or description, or ``"UNBOUND"``.
    """
    window = context_window(lines, literal)
    normalised = normalise(literal.text)

    if kind == "structural_count":
        for value, ctx, artefact in _STRUCTURAL_RULES:
            if normalised == value and re.search(ctx, window):
                return artefact
        return UNBOUND

    if kind == "measurement":
        if (
            normalised in train_sizes or normalised in test_sizes
        ) and _BENCHMARK_SIZE_CONTEXT.search(window):
            return BENCHMARK_ARTEFACT
        return UNBOUND

    return UNBOUND


# --------------------------------------------------------------------------
# Macro proposals
# --------------------------------------------------------------------------

#: Column keys of the two per-problem supplementary tables, in column order.
_SUPP_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "Prob",
    "RSqTestBl",
    "RSqTestIs",
    "NrmseTestBl",
    "NrmseTestIs",
    "CohenD",
    "Rho",
    "Red",
    "TBl",
    "TIs",
    "Oh",
)

_SUFFIX_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def macro_suffix(index: int) -> str:
    """Encode a collision index as a letters-only, order-preserving suffix.

    A LaTeX control word admits catcode-11 characters only, so the suffix that
    disambiguates macros sharing a base may not carry digits. The encoding is
    bijective base-26 over ``A``--``Z`` (``0 -> A``, ``25 -> Z``, ``26 -> AA``,
    ``27 -> AB``, ...): every index maps to a distinct string, so uniqueness of
    ``base + suffix`` follows from uniqueness of the index within a base, and
    the map is a pure function of the index, so names are stable across runs.

    Args:
        index: Zero-based number of earlier entries sharing the same base.

    Returns:
        Letters-only suffix.

    Raises:
        NumericalAuditError: If ``index`` is negative.
    """
    if index < 0:
        raise NumericalAuditError(f"collision index must be non-negative, got {index}")
    out: list[str] = []
    n = index + 1
    while n > 0:
        n -= 1
        out.append(_SUFFIX_LETTERS[n % 26])
        n //= 26
    return "".join(reversed(out))


def _table_macro_base(line: str, literal: RawLiteral, tag: str) -> str | None:
    """Build a per-cell macro base for a row of a supplementary table.

    Args:
        line: The table row.
        literal: The literal inside the row.
        tag: File tag (``TabUdfs`` / ``TabBingo``).

    Returns:
        Macro base without a leading backslash, or ``None`` if the row does not
        parse as an 11-column data row.
    """
    body = line.rstrip()
    if not body.endswith("\\\\"):
        return None
    cells = body[:-2].split("&")
    if len(cells) != len(_SUPP_TABLE_COLUMNS):
        return None
    offset = 0
    col_index = 0
    for idx, cell in enumerate(cells):
        end = offset + len(cell)
        if offset <= literal.col_start < end:
            col_index = idx
            break
        offset = end + 1
    else:
        return None
    problem = digits_to_words(cells[0].strip())
    if not problem:
        return None
    return f"{tag}{problem}{_SUPP_TABLE_COLUMNS[col_index]}"


def _narrative_macro_base(lines: list[str], literal: RawLiteral, tag: str) -> str:
    """Build a metric-aware macro base for a literal in prose.

    Args:
        lines: All source lines.
        literal: The literal.
        tag: File tag.

    Returns:
        Macro base without a leading backslash.
    """
    window = context_window(lines, literal, width=70)
    metric = ""
    for pattern, name in _METRIC_MACRO_RULES:
        if pattern.search(window):
            metric = name
            break
    if not metric:
        return f"num{tag}L{digits_to_words(str(literal.line))}"
    stat = ""
    for pattern, name in _STAT_MACRO_RULES:
        if pattern.search(window):
            stat = name
            break
    method = ""
    if re.search(r"\bUDFS\b", window):
        method = "Udfs"
    if re.search(r"\bBingo\b", window):
        method = "Bingo" if not method else "Both"
    head = metric[0].lower() + metric[1:]
    return f"{head}{stat}{method}"


def propose_macros(
    entries: list[AuditEntry],
    lines_by_file: dict[str, list[str]],
) -> None:
    """Assign a unique LaTeX macro proposal to every measurement, in place.

    Macro names are letters-only (LaTeX control sequences admit nothing else)
    and are made unique by appending a bijective base-26 suffix (:func:`macro_suffix`)
    in file/line/column order, so re-running the tool reproduces the same names
    byte for byte. The suffix must not overflow into digits: a base with more
    than 26 occurrences continues ``AA``, ``AB``, ... rather than ``X26``.

    Args:
        entries: All audit entries, already ordered deterministically.
        lines_by_file: Source lines keyed by relative path.
    """
    used: Counter[str] = Counter()
    for entry in entries:
        if entry.kind not in ("measurement", "structural_count"):
            continue
        tag = FILE_TAGS[entry.file]
        lines = lines_by_file[entry.file]
        literal = RawLiteral(
            line=entry.line,
            col_start=entry.col_start,
            col_end=entry.col_end,
            text=entry.literal,
            in_comment=False,
        )
        base: str | None = None
        if tag in ("TabUdfs", "TabBingo"):
            base = _table_macro_base(lines[entry.line - 1], literal, tag)
        if base is None:
            base = _narrative_macro_base(lines, literal, tag)
        count = used[base]
        used[base] += 1
        entry.proposed_macro = "\\" + base + macro_suffix(count)


# --------------------------------------------------------------------------
# Entry schema
# --------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """One numeric literal in the manuscript.

    Attributes:
        file: Path relative to the manuscript root.
        line: 1-based line number.
        col_start: 0-based inclusive start column.
        col_end: 0-based exclusive end column.
        literal: Verbatim source slice.
        normalised: Canonical form used for duplicate matching.
        kind: Classification, from the closed enum :data:`KINDS`.
        classification_rule: The pattern that fired, empty for the default.
        surrounding_sentence: Trimmed context, at most 160 characters.
        source_artefact: Producing artefact, or ``"UNBOUND"``.
        notation_role: Whether the literal is a single-digit sub/superscript of
            a symbol (``R^2``). Orthogonal to ``kind``: such a literal is still
            a ``measurement``, but it is excluded from the headline duplicate
            report because it states nothing.
        proposed_macro: Suggested LaTeX macro name, empty for non-numbers-of-record.
    """

    file: str
    line: int
    col_start: int
    col_end: int
    literal: str
    normalised: str
    kind: Kind
    classification_rule: str
    surrounding_sentence: str
    source_artefact: str
    notation_role: bool = field(default=False)
    proposed_macro: str = field(default="")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def load_benchmark_sizes(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Read the realised train/test sizes from the Appendix D.1 inventory.

    Args:
        path: Path to ``appendix_d_benchmarks.json``.

    Returns:
        ``(n_train values, n_test values)`` as normalised strings. Both sets are
        empty when the inventory is absent, in which case dataset-size literals
        are reported ``UNBOUND`` rather than mis-bound.
    """
    if not path.exists():
        log.warning("Benchmark inventory %s not found; dataset sizes stay UNBOUND", path)
        return frozenset(), frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = payload.get("problems", [])
    train = frozenset(str(int(p["n_train"])) for p in problems)
    test = frozenset(str(int(p["n_test"])) for p in problems)
    return train, test


def audit_file(
    rel_path: str,
    text: str,
    train_sizes: frozenset[str],
    test_sizes: frozenset[str],
) -> list[AuditEntry]:
    """Produce the inventory of one manuscript file.

    Args:
        rel_path: Path relative to the manuscript root.
        text: File contents.
        train_sizes: Realised ``n_train`` values.
        test_sizes: Realised ``n_test`` values.

    Returns:
        Entries in ``(line, col_start)`` order.
    """
    lines = text.splitlines()
    entries: list[AuditEntry] = []
    for literal in extract_literals(text):
        kind, rule = classify(lines, literal)
        entries.append(
            AuditEntry(
                file=rel_path,
                line=literal.line,
                col_start=literal.col_start,
                col_end=literal.col_end,
                literal=literal.text,
                normalised=normalise(literal.text),
                kind=kind,
                classification_rule=rule,
                surrounding_sentence=surrounding_sentence(lines, literal),
                source_artefact=bind_artefact(lines, literal, kind, train_sizes, test_sizes),
                notation_role=is_notation_script(
                    lines[literal.line - 1], literal.col_start, literal.col_end
                ),
            )
        )
    return entries


#: The four narrative files. A value repeated between the two per-problem
#: tables is usually a coincidence (both hosts score 1.0000 on an easy
#: problem); a value repeated between two pieces of prose is a restated claim.
NARRATIVE_FILES: Final[frozenset[str]] = frozenset(
    {
        "paper/computational_experiments.tex",
        "paper/results.tex",
        "paper/discussion.tex",
        "supplementary/supplementary.tex",
    }
)


def find_cross_file_duplicates(
    entries: Iterable[AuditEntry],
    restrict_to: frozenset[str] | None = None,
    drop_notation: bool = False,
) -> list[dict[str, Any]]:
    """Group measurement literals that are stated in more than one file.

    This is the headline output: a number written out twice is a number that
    can silently diverge when one of the two sites is updated.

    Args:
        entries: All audit entries.
        restrict_to: When given, only literals occurring in at least two of
            these files form a group; occurrences outside the set are still
            listed, so the full blast radius of an edit is visible.
        drop_notation: When true, single-digit symbol scripts (``R^2``) are
            excluded. They are measurements by classification but state
            nothing, and there are enough of them to drown the report.

    Returns:
        Duplicate groups, sorted by descending file count then by value.
    """
    buckets: dict[str, list[AuditEntry]] = defaultdict(list)
    for entry in entries:
        if entry.kind != "measurement":
            continue
        if drop_notation and entry.notation_role:
            continue
        buckets[entry.normalised].append(entry)
    groups: list[dict[str, Any]] = []
    for value, group in buckets.items():
        files = sorted({e.file for e in group})
        keyed = files if restrict_to is None else [f for f in files if f in restrict_to]
        if len(keyed) < 2:
            continue
        ordered = sorted(group, key=lambda e: (AUDITED_FILES.index(e.file), e.line, e.col_start))
        groups.append(
            {
                "normalised": value,
                "n_files": len(files),
                "files": files,
                "n_occurrences": len(group),
                "occurrences": [
                    {
                        "file": e.file,
                        "line": e.line,
                        "col_start": e.col_start,
                        "literal": e.literal,
                        "source_artefact": e.source_artefact,
                        "proposed_macro": e.proposed_macro,
                        "surrounding_sentence": e.surrounding_sentence,
                    }
                    for e in ordered
                ],
            }
        )
    groups.sort(key=lambda g: (-int(g["n_files"]), -int(g["n_occurrences"]), str(g["normalised"])))
    return groups


def find_within_file_repeats(entries: Iterable[AuditEntry]) -> list[dict[str, Any]]:
    """Group measurement literals repeated inside a single file.

    Args:
        entries: All audit entries.

    Returns:
        Repeat groups keyed by ``(file, value)``, sorted deterministically.
    """
    buckets: dict[tuple[str, str], list[AuditEntry]] = defaultdict(list)
    for entry in entries:
        if entry.kind == "measurement":
            buckets[(entry.file, entry.normalised)].append(entry)
    groups: list[dict[str, Any]] = []
    for (path, value), group in buckets.items():
        if len(group) < 2:
            continue
        groups.append(
            {
                "file": path,
                "normalised": value,
                "n_occurrences": len(group),
                "lines": [e.line for e in group],
            }
        )
    groups.sort(key=lambda g: (str(g["file"]), -int(g["n_occurrences"]), str(g["normalised"])))
    return groups


def build_payload(
    entries: list[AuditEntry],
    manuscript_root: Path,
    benchmarks_json: Path,
) -> dict[str, Any]:
    """Assemble the machine-readable audit payload.

    Args:
        entries: All audit entries, deterministically ordered.
        manuscript_root: Root the audited files were read from.
        benchmarks_json: Path of the benchmark inventory used for binding.

    Returns:
        JSON-serialisable payload.
    """
    per_kind = Counter(e.kind for e in entries)
    per_file = Counter(e.file for e in entries)
    measurements = [e for e in entries if e.kind == "measurement"]
    unbound = [e for e in measurements if e.source_artefact == UNBOUND]
    per_file_kind: dict[str, dict[str, int]] = {
        path: {kind: 0 for kind in KINDS} for path in AUDITED_FILES
    }
    for entry in entries:
        per_file_kind[entry.file][entry.kind] += 1
    return {
        "schema_version": "numerical-audit.1",
        "provenance": {
            "generator": "experiments/scripts/numerical_audit.py",
            "manuscript_root": str(manuscript_root),
            "audited_files": list(AUDITED_FILES),
            "benchmark_inventory": str(benchmarks_json),
            "kinds": list(KINDS),
            "classification_policy": (
                "Conservative: a literal is moved out of 'measurement' only by an "
                "explicit named rule. Unmatched literals stay 'measurement'."
            ),
            "binding_policy": (
                "source_artefact is either a concrete artefact or the literal "
                "string 'UNBOUND'. Nothing is guessed."
            ),
        },
        "counts": {
            "n_entries": len(entries),
            "per_kind": {kind: per_kind.get(kind, 0) for kind in KINDS},
            "per_file": {path: per_file.get(path, 0) for path in AUDITED_FILES},
            "per_file_per_kind": per_file_kind,
            "n_measurements": len(measurements),
            "n_measurements_bound": len(measurements) - len(unbound),
            "n_measurements_unbound": len(unbound),
            "n_measurements_notation_role": sum(1 for e in measurements if e.notation_role),
        },
        "duplicates": find_cross_file_duplicates(entries),
        "narrative_duplicates": find_cross_file_duplicates(
            entries, restrict_to=NARRATIVE_FILES, drop_notation=True
        ),
        "within_file_repeats": find_within_file_repeats(entries),
        "entries": [asdict(e) for e in entries],
    }


def _md_escape(text: str) -> str:
    """Escape a context string for a Markdown table cell.

    Args:
        text: Raw context.

    Returns:
        Text with pipes and backticks neutralised.
    """
    return text.replace("|", "\\|").replace("`", "'")


def _duplicate_table(groups: list[dict[str, Any]]) -> list[str]:
    """Render a duplicate report as Markdown table rows.

    Args:
        groups: Duplicate groups from :func:`find_cross_file_duplicates`.

    Returns:
        Markdown lines.
    """
    if not groups:
        return ["_None._"]
    out = [
        "| value | files | occurrences | proposed macro | sites |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for group in groups:
        macro = group["occurrences"][0]["proposed_macro"] or ""
        sites = "; ".join(f"{o['file']}:{o['line']}" for o in group["occurrences"])
        out.append(
            f"| `{group['normalised']}` | {group['n_files']} | "
            f"{group['n_occurrences']} | `{macro}` | {_md_escape(sites)} |"
        )
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the human-readable audit report.

    Args:
        payload: The JSON payload.

    Returns:
        Markdown document, newline-terminated.
    """
    counts = payload["counts"]
    out: list[str] = [
        "# Numerical audit of the manuscript",
        "",
        "GENERATED FILE. Regenerate with:",
        "",
        "```",
        "python -m experiments.scripts.numerical_audit --out docs/generated/audit/",
        "```",
        "",
        f"Manuscript root: `{payload['provenance']['manuscript_root']}`",
        "",
        f"Total numeric literals inventoried: **{counts['n_entries']}**.",
        "",
        "## Tally by kind",
        "",
        "| kind | count |",
        "| --- | ---: |",
    ]
    for kind in KINDS:
        out.append(f"| `{kind}` | {counts['per_kind'][kind]} |")
    out += [
        "",
        "## Tally by file",
        "",
        "| file | total | " + " | ".join(f"`{k}`" for k in KINDS) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(KINDS)) + " |",
    ]
    for path in AUDITED_FILES:
        row = payload["counts"]["per_file_per_kind"][path]
        out.append(
            f"| `{path}` | {counts['per_file'][path]} | "
            + " | ".join(str(row[k]) for k in KINDS)
            + " |"
        )
    out += [
        "",
        "## Artefact binding",
        "",
        f"- measurements: **{counts['n_measurements']}**",
        f"- bound to an artefact: **{counts['n_measurements_bound']}**",
        f"- `UNBOUND`: **{counts['n_measurements_unbound']}** "
        "(the work remaining once the campaign results land)",
        "",
        "## Duplicated measurements across the narrative files (headline)",
        "",
        "A value restated in two pieces of prose is a value that can diverge "
        "silently when one site is edited. Each group below should collapse to "
        "a single macro. Single-digit symbol scripts (`R^2`, `L_2`, `H_0`) are "
        f"excluded: {counts['n_measurements_notation_role']} literals carry "
        "`notation_role: true` and are still listed in the full inventory.",
        "",
    ]
    out += _duplicate_table(payload["narrative_duplicates"])
    out += [
        "",
        "## Duplicated measurements across all six files",
        "",
        "Includes the two per-problem tables. A value shared between the UDFS "
        "and the Bingo table is often a coincidence rather than a restated "
        "claim, so this list is broader than the one above.",
        "",
    ]
    out += _duplicate_table(payload["duplicates"])
    out += ["", "## Within-file repeated measurements", ""]
    repeats = payload["within_file_repeats"]
    if not repeats:
        out.append("_None._")
    else:
        out += ["| file | value | occurrences | lines |", "| --- | --- | ---: | --- |"]
        for group in repeats:
            lines = ", ".join(str(n) for n in group["lines"])
            out.append(
                f"| `{group['file']}` | `{group['normalised']}` | "
                f"{group['n_occurrences']} | {lines} |"
            )
    out += ["", "## Full inventory", ""]
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in payload["entries"]:
        by_file[entry["file"]].append(entry)
    for path in AUDITED_FILES:
        out += [
            f"### `{path}`",
            "",
            "| line | literal | kind | source artefact | proposed macro | context |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
        for entry in by_file[path]:
            out.append(
                f"| {entry['line']} | `{entry['literal']}` | {entry['kind']} | "
                f"{entry['source_artefact']} | `{entry['proposed_macro']}` | "
                f"{_md_escape(entry['surrounding_sentence'])} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def run_audit(manuscript_root: Path, benchmarks_json: Path) -> dict[str, Any]:
    """Audit the six manuscript files and return the payload.

    Args:
        manuscript_root: Root of the (read-only) manuscript checkout.
        benchmarks_json: Path to the Appendix D.1 benchmark inventory.

    Returns:
        The JSON-serialisable audit payload.

    Raises:
        NumericalAuditError: If an audited file is missing or unreadable.
    """
    train_sizes, test_sizes = load_benchmark_sizes(benchmarks_json)
    entries: list[AuditEntry] = []
    lines_by_file: dict[str, list[str]] = {}
    for rel_path in AUDITED_FILES:
        path = manuscript_root / rel_path
        if not path.exists():
            raise NumericalAuditError(f"Audited file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NumericalAuditError(f"Cannot read {path}: {exc}") from exc
        lines_by_file[rel_path] = text.splitlines()
        entries.extend(audit_file(rel_path, text, train_sizes, test_sizes))
    entries.sort(key=lambda e: (AUDITED_FILES.index(e.file), e.line, e.col_start))
    propose_macros(entries, lines_by_file)
    return build_payload(entries, manuscript_root, benchmarks_json)


def generate(out_dir: Path, manuscript_root: Path, benchmarks_json: Path) -> list[Path]:
    """Write ``numerical_audit.json`` and ``numerical_audit.md``.

    Args:
        out_dir: Destination directory; created if absent.
        manuscript_root: Root of the manuscript checkout.
        benchmarks_json: Path to the benchmark inventory.

    Returns:
        The written paths, JSON first.
    """
    payload = run_audit(manuscript_root, benchmarks_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "numerical_audit.json"
    md_path = out_dir / "numerical_audit.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    counts = payload["counts"]
    log.info(
        "Inventoried %d literals across %d files; %d measurements (%d UNBOUND); "
        "%d cross-file duplicate groups",
        counts["n_entries"],
        len(AUDITED_FILES),
        counts["n_measurements"],
        counts["n_measurements_unbound"],
        len(payload["duplicates"]),
    )
    log.info("Wrote %s and %s", json_path, md_path)
    return [json_path, md_path]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Inventory and classify every numeric literal in the manuscript."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "generated" / "audit",
        help="Output directory.",
    )
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=Path(DEFAULT_MANUSCRIPT_ROOT),
        help="Root of the manuscript checkout (read-only).",
    )
    parser.add_argument(
        "--benchmarks-json",
        type=Path,
        default=_REPO_ROOT / "docs" / "generated" / "appendix_d" / "appendix_d_benchmarks.json",
        help="Appendix D.1 benchmark inventory, used to bind dataset sizes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    try:
        generate(args.out, args.manuscript_root, args.benchmarks_json)
    except NumericalAuditError:
        log.exception("Numerical audit failed")
        return 1
    return 0


def iter_kinds() -> Iterator[str]:
    """Yield the closed classification enum.

    Returns:
        Iterator over the kind names, in report order.
    """
    yield from KINDS


if __name__ == "__main__":
    raise SystemExit(main())
