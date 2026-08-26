"""Tests for the manuscript numerical audit tool.

The completeness test is deliberately *independent* of the tool's own lexer: it
blanks every span the tool recorded and asserts that no digit survives anywhere
in the source. A regex bug that drops a literal therefore fails the test rather
than silently agreeing with itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from experiments.scripts.docs_root import docs_root
from experiments.scripts.numerical_audit import (
    AUDITED_FILES,
    DEFAULT_MANUSCRIPT_ROOT,
    KINDS,
    UNBOUND,
    NumericalAuditError,
    audit_file,
    classify,
    comment_start,
    digits_to_words,
    enclosing_token,
    extract_literals,
    find_cross_file_duplicates,
    generate,
    is_notation_script,
    macro_suffix,
    normalise,
    propose_macros,
    run_audit,
)

#: Everything in this module reads the LaTeX manuscript, which is not part of
#: this repository (the audit compares the manuscript's numbers against the results tree).
#: Without it every test here can only skip, so the marker lets a run that has
#: no manuscript deselect the module outright -- `-m "not manuscript"` -- rather
#: than report dozens of skips that read as coverage gaps.
pytestmark = pytest.mark.manuscript

MANUSCRIPT_ROOT = Path(DEFAULT_MANUSCRIPT_ROOT)
BENCHMARKS_JSON = docs_root() / "docs" / "generated" / "appendix_d" / "appendix_d_benchmarks.json"

requires_manuscript = pytest.mark.skipif(
    not MANUSCRIPT_ROOT.exists(),
    reason="manuscript checkout not mounted",
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Run the audit once for the whole module.

    Returns:
        The audit payload.
    """
    if not MANUSCRIPT_ROOT.exists():
        pytest.skip("manuscript checkout not mounted")
    return run_audit(MANUSCRIPT_ROOT, BENCHMARKS_JSON)


# ---------------------------------------------------------------------------
# Lexer unit tests (no manuscript needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("no numbers here", []),
        ("$0.28$\\,ms", ["0.28"]),
        ("$2{,}640$ tasks", ["2{,}640"]),
        ("$5.9{\\times}10^{-5}$", ["5.9{\\times}10^{-5}"]),
        ("$2.7 \\times 10^{-22}$", ["2.7 \\times 10^{-22}"]),
        ("$<10^{-9}$", ["10^{-9}"]),
        ("$10! \\approx 3.6 \\times 10^6$", ["10", "3.6 \\times 10^6"]),
        ("Feynman~I.48.20", ["48.20"]),
        ("$k^{0.7}$", ["0.7"]),
        ("$43200.6$", ["43200.6"]),
        ("$1.71 \\pm 0.03$", ["1.71", "0.03"]),
    ],
)
def test_extract_literals_lexing(text: str, expected: list[str]) -> None:
    """The lexer recovers each expected literal exactly once, in order."""
    assert [lit.text for lit in extract_literals(text)] == expected


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("2{,}640", "2640"),
        ("10,000", "10000"),
        ("0.050", "0.050"),
        ("0.05", "0.05"),
        ("5.9{\\times}10^{-5}", "5.9e-5"),
        ("2.7 \\times 10^{-22}", "2.7e-22"),
        ("10^{-9}", "1e-9"),
        ("3.6 \\times 10^6", "3.6e6"),
    ],
)
def test_normalise(literal: str, expected: str) -> None:
    """Normalisation folds separators and scientific notation, not precision."""
    assert normalise(literal) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("plain text", len("plain text")),
        ("% a comment", 0),
        ("$34.2\\%$ eliminated", len("$34.2\\%$ eliminated")),
        ("value 3 % trailing note", len("value 3 ")),
    ],
)
def test_comment_start(line: str, expected: int) -> None:
    """An escaped percent sign is not a comment marker."""
    assert comment_start(line) == expected


def test_enclosing_token_recovers_problem_ids() -> None:
    """The token around a literal is what decides a problem-id classification."""
    line = "on Feynman~I.48.20 ($d = +4.68$)"
    assert enclosing_token(line, line.index("48.20"), line.index("48.20") + 5) == "I.48.20"
    assert enclosing_token(line, line.index("4.68"), line.index("4.68") + 4) == "4.68"


@pytest.mark.parametrize(
    ("line", "needle", "expected"),
    [
        ("$R^2$ on the test set", "2", True),
        ("$R^{2}_{\\mathrm{test}}$", "2", True),
        ("$L_2$-norms", "2", True),
        ("$H_0:\\; \\theta_m = 0$", "0", True),
        ("$\\delta_1^{(m)}$", "1", True),
        ("fits $O(k^{0.7})$", "0.7", False),
        ("$\\sum_{k=1}^{9} 600$", "9", False),
        ("$2\\sin(x)\\cos(y)$", "2", False),
        ("median of $0.28$ ms", "0.28", False),
        ("population of $500$ individuals", "500", False),
    ],
)
def test_is_notation_script(line: str, needle: str, expected: bool) -> None:
    """Only single-digit scripts attached to a single symbol are notation."""
    start = line.index(needle)
    assert is_notation_script(line, start, start + len(needle)) is expected


def test_notation_scripts_stay_measurements() -> None:
    """The notation flag is orthogonal to ``kind``; it never rewrites it."""
    lines = ["the $R^2$ value"]
    literal = extract_literals(lines[0])[0]
    kind, _ = classify(lines, literal)
    assert kind == "measurement"
    assert is_notation_script(lines[0], literal.col_start, literal.col_end) is True


def test_digits_to_words_is_letters_only() -> None:
    """Macro-name fragments admit letters only."""
    out = digits_to_words("I.10.7")
    assert out == "IOneZeroSeven"
    assert out.isalpha()


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (52, "BA")],
)
def test_macro_suffix_is_bijective_base_26(index: int, expected: str) -> None:
    """The collision suffix continues into two letters instead of into digits."""
    assert macro_suffix(index) == expected


def test_macro_suffix_is_letters_only_unique_and_stable() -> None:
    """Suffixes stay catcode-11, injective and a pure function of the index."""
    suffixes = [macro_suffix(i) for i in range(1000)]
    assert all(s.isalpha() and s.isupper() for s in suffixes)
    assert len(set(suffixes)) == len(suffixes), "suffixes must be unique"
    assert suffixes == [macro_suffix(i) for i in range(1000)], "suffixes must be stable"


def test_macro_suffix_rejects_negative_index() -> None:
    """A negative collision index is a programming error, not a silent name."""
    with pytest.raises(NumericalAuditError):
        macro_suffix(-1)


#: 30 rho measurements in one file: more than the 26 single-letter suffixes, so
#: the collision counter must roll over without emitting a digit.
FIXTURE_MANY_RHO = "\\subsection{Rho}\n" + "".join(
    f"The reduction factor is $\\rho = 1.{n:02d}$ on the suite.\n" for n in range(30)
)


def test_macro_proposals_survive_more_than_26_collisions() -> None:
    """A base shared by >26 measurements still yields letters-only unique names."""
    lines = FIXTURE_MANY_RHO.splitlines()
    entries = audit_file(AUDITED_FILES[0], FIXTURE_MANY_RHO, frozenset(), frozenset())
    propose_macros(entries, {AUDITED_FILES[0]: lines})
    macros = [e.proposed_macro for e in entries if e.proposed_macro]
    assert len(macros) >= 27, f"fixture must overflow the 26 letters, got {len(macros)}"
    offenders = [m for m in macros if not m.startswith("\\") or not m[1:].isalpha()]
    assert not offenders, f"macro names must be letters-only: {offenders}"
    assert len(set(macros)) == len(macros), "macro proposals must be unique"
    # Stability: a second pass over freshly parsed entries reproduces the names.
    again = audit_file(AUDITED_FILES[0], FIXTURE_MANY_RHO, frozenset(), frozenset())
    propose_macros(again, {AUDITED_FILES[0]: lines})
    assert [e.proposed_macro for e in again if e.proposed_macro] == macros


# ---------------------------------------------------------------------------
# Regression fixture: the tool must not be vacuous
# ---------------------------------------------------------------------------


FIXTURE_A = r"""
\subsection{Fixture A}
The median per-DAG canonicalisation time is $0.28$\,ms on UDFS, and the
empirical reduction factor is $\rho = 1.56 \pm 0.24$.
""".lstrip()

FIXTURE_B = r"""
\subsection{Fixture B}
As reported above, canonicalisation costs $0.28$\,ms per DAG and $\rho$
attains $1.56$ on the suite; unique to this file is $9.99$.
""".lstrip()


def _fixture_entries() -> list[Any]:
    """Audit the two in-memory fixture files.

    Returns:
        The combined entry list, with the audited-file names faked so the
        duplicate finder sees two distinct files.
    """
    entries = audit_file(AUDITED_FILES[0], FIXTURE_A, frozenset(), frozenset())
    entries += audit_file(AUDITED_FILES[1], FIXTURE_B, frozenset(), frozenset())
    propose_macros(
        entries,
        {
            AUDITED_FILES[0]: FIXTURE_A.splitlines(),
            AUDITED_FILES[1]: FIXTURE_B.splitlines(),
        },
    )
    return entries


def test_fixture_duplicates_are_reported() -> None:
    """A measurement duplicated across two files is reported with all sites."""
    groups = find_cross_file_duplicates(_fixture_entries())
    found = {g["normalised"]: g for g in groups}
    assert "0.28" in found, f"0.28 not reported; got {sorted(found)}"
    assert "1.56" in found, f"1.56 not reported; got {sorted(found)}"
    assert found["0.28"]["n_files"] == 2
    assert found["0.28"]["n_occurrences"] == 2
    assert sorted(o["file"] for o in found["0.28"]["occurrences"]) == sorted(AUDITED_FILES[:2])
    assert "9.99" not in found, "a single-file literal must not be reported as a duplicate"


def test_fixture_duplicate_occurrences_carry_context() -> None:
    """Every reported occurrence carries a file, a line and a context string."""
    groups = find_cross_file_duplicates(_fixture_entries())
    for group in groups:
        for occ in group["occurrences"]:
            assert occ["file"] in AUDITED_FILES
            assert occ["line"] >= 1
            assert occ["surrounding_sentence"]
            assert len(occ["surrounding_sentence"]) <= 160


# ---------------------------------------------------------------------------
# Manuscript-backed acceptance tests
# ---------------------------------------------------------------------------


@requires_manuscript
def test_every_literal_is_inventoried_exactly_once(payload: dict[str, Any]) -> None:
    """Blanking every recorded span must remove every digit from every file.

    This is the completeness check and it does not reuse the tool's lexer: it
    only trusts the recorded ``(line, col_start, col_end)`` spans.
    """
    spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for entry in payload["entries"]:
        spans.setdefault((entry["file"], entry["line"]), []).append(
            (entry["col_start"], entry["col_end"])
        )

    residue: list[str] = []
    for rel_path in AUDITED_FILES:
        lines = (MANUSCRIPT_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            chars = list(line)
            for start, end in spans.get((rel_path, lineno), []):
                for idx in range(start, end):
                    chars[idx] = " "
            leftover = "".join(chars)
            if any(char.isdigit() for char in leftover):
                residue.append(f"{rel_path}:{lineno}: {leftover.strip()[:120]}")
    assert not residue, "digits not covered by the inventory:\n" + "\n".join(residue[:20])


@requires_manuscript
def test_spans_do_not_overlap_and_match_the_source(payload: dict[str, Any]) -> None:
    """Each entry's span is disjoint from its neighbours and slices its literal."""
    seen: set[tuple[str, int, int]] = set()
    per_line: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for entry in payload["entries"]:
        key = (entry["file"], entry["line"], entry["col_start"])
        assert key not in seen, f"duplicate inventory position: {key}"
        seen.add(key)
        per_line.setdefault((entry["file"], entry["line"]), []).append(
            (entry["col_start"], entry["col_end"])
        )
    for (rel_path, lineno), spans in per_line.items():
        spans.sort()
        for (_, prev_end), (start, _) in zip(spans, spans[1:], strict=False):
            assert prev_end <= start, f"overlapping spans on {rel_path}:{lineno}"

    for rel_path in AUDITED_FILES:
        lines = (MANUSCRIPT_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
        for entry in payload["entries"]:
            if entry["file"] != rel_path:
                continue
            sliced = lines[entry["line"] - 1][entry["col_start"] : entry["col_end"]]
            assert sliced == entry["literal"], f"{rel_path}:{entry['line']} span/literal mismatch"


@requires_manuscript
def test_required_fields_present_and_sentence_bounded(payload: dict[str, Any]) -> None:
    """Every entry carries file, line, literal and a <=160-char context."""
    for entry in payload["entries"]:
        assert entry["file"] in AUDITED_FILES
        assert isinstance(entry["line"], int) and entry["line"] >= 1
        assert entry["literal"]
        assert isinstance(entry["surrounding_sentence"], str)
        assert len(entry["surrounding_sentence"]) <= 160


@requires_manuscript
def test_kind_enum_is_closed(payload: dict[str, Any]) -> None:
    """No entry escapes the closed classification enum."""
    observed = {entry["kind"] for entry in payload["entries"]}
    assert observed <= set(KINDS), f"kinds outside the enum: {observed - set(KINDS)}"
    assert set(payload["counts"]["per_kind"]) == set(KINDS)
    assert sum(payload["counts"]["per_kind"].values()) == payload["counts"]["n_entries"]


@requires_manuscript
def test_known_problem_id_fragments_are_not_measurements(payload: dict[str, Any]) -> None:
    """The confirmed false positives are classified as problem-id fragments.

    ``13.12``, ``16.6``, ``17.37``, ``37.4``, ``48.20`` and ``6.20`` are AI
    Feynman identifiers. Every occurrence whose enclosing token is a problem id
    must be classified as such.
    """
    offenders: list[str] = []
    for entry in payload["entries"]:
        lines = (MANUSCRIPT_ROOT / entry["file"]).read_text(encoding="utf-8").splitlines()
        token = enclosing_token(lines[entry["line"] - 1], entry["col_start"], entry["col_end"])
        if (
            re.fullmatch(r"(?:I|II|III)\.\d+(?:\.\d+)?[a-z]?", token)
            and entry["kind"] != "problem_id_fragment"
        ):
            offenders.append(f"{entry['file']}:{entry['line']} {token} -> {entry['kind']}")
    assert not offenders, "Feynman ids misclassified:\n" + "\n".join(offenders[:20])


@requires_manuscript
def test_hand_verified_cross_file_duplicates_are_rediscovered(payload: dict[str, Any]) -> None:
    """The tool independently finds the duplicates identified by hand.

    This test is a *canary*, not an invariant of the tool: its literals are the
    numbers the manuscript currently reports, so they move whenever the reported
    campaign is re-executed. When it fires, check whether the campaign changed
    before touching the tool.

    Pinned on 2026-08-14 to campaign **C2** (three arms, 70 problems, 30 seeds,
    12,600 runs), verified against
    ``results/review/c2_3arm/analyses/values/summary.json``. The previous pin
    (``0.28``/``0.82`` canonicalisation times, ``1.56``/``1.83`` per-host $\\rho$
    means, ``1.07`` speedup $S$) belonged to the submitted campaign and was
    legitimately superseded; the test fired correctly on that replacement.
    """
    # rho per host: mean (1.6637 / 1.7850), min and max over the 70 problems.
    expected = {"1.66", "1.79", "1.11", "2.12", "1.19", "1.85", "16.1"}
    for key in ("duplicates", "narrative_duplicates"):
        found = {g["normalised"] for g in payload[key]}
        missing = expected - found
        assert not missing, f"{key}: hand-verified duplicates not rediscovered: {sorted(missing)}"
    narrative = {g["normalised"]: g for g in payload["narrative_duplicates"]}
    # The discussion states both rho ranges inline; the supplementary restates
    # each as an interval in its per-method paragraph. Anchor on the file and on
    # the sentence text rather than on a line number: the manuscript is a live
    # checkout, so line numbers drift under edits that leave the claim intact.
    for value in ("1.11", "1.19"):
        occurrences = narrative[value]["occurrences"]
        files = {o["file"] for o in occurrences}
        assert {"paper/discussion.tex", "supplementary/supplementary.tex"} <= files, files
        supp = [o for o in occurrences if o["file"] == "supplementary/supplementary.tex"]
        assert any("Reduction factors span" in o["surrounding_sentence"] for o in supp), supp
        disc = [o for o in occurrences if o["file"] == "paper/discussion.tex"]
        assert any("The observed" in o["surrounding_sentence"] for o in disc), disc


@requires_manuscript
def test_duplicates_are_measurements_in_more_than_one_file(payload: dict[str, Any]) -> None:
    """The duplicate report is restricted to measurements spanning >1 file."""
    for group in payload["duplicates"]:
        assert group["n_files"] >= 2
        assert len({o["file"] for o in group["occurrences"]}) == group["n_files"]
        for occ in group["occurrences"]:
            match = next(
                e
                for e in payload["entries"]
                if e["file"] == occ["file"]
                and e["line"] == occ["line"]
                and e["col_start"] == occ["col_start"]
            )
            assert match["kind"] == "measurement"
            assert match["literal"] == occ["literal"]


@requires_manuscript
def test_measurements_carry_artefact_and_macro(payload: dict[str, Any]) -> None:
    """Every measurement has a source_artefact and a non-empty macro proposal."""
    macros: set[str] = set()
    for entry in payload["entries"]:
        if entry["kind"] != "measurement":
            continue
        assert entry["source_artefact"], "source_artefact must never be empty"
        assert entry["proposed_macro"].startswith("\\")
        body = entry["proposed_macro"][1:]
        assert body.isalpha(), f"macro name is not letters-only: {entry['proposed_macro']}"
        assert entry["proposed_macro"] not in macros, "macro proposals must be unique"
        macros.add(entry["proposed_macro"])


@requires_manuscript
def test_unbound_is_the_literal_sentinel(payload: dict[str, Any]) -> None:
    """Unknown provenance is the exact string UNBOUND, never a guess."""
    unbound = [e for e in payload["entries"] if e["source_artefact"] == UNBOUND]
    assert payload["counts"]["n_measurements_unbound"] == sum(
        1 for e in unbound if e["kind"] == "measurement"
    )
    assert payload["counts"]["n_measurements_unbound"] > 0, (
        "the C2 campaign has not landed, so unbound measurements must exist"
    )


@requires_manuscript
def test_benchmark_sizes_bind_to_the_appendix_inventory(payload: dict[str, Any]) -> None:
    """Dataset-size literals bind to the Appendix D.1 inventory, not UNBOUND."""
    if not BENCHMARKS_JSON.exists():
        pytest.skip("appendix D inventory not generated")
    bound = [
        e for e in payload["entries"] if e["source_artefact"].endswith("appendix_d_benchmarks.json")
    ]
    assert bound, "no literal bound to the benchmark inventory"


@requires_manuscript
def test_output_is_deterministic(tmp_path: Path) -> None:
    """Two consecutive runs produce byte-identical artefacts."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate(first, MANUSCRIPT_ROOT, BENCHMARKS_JSON)
    generate(second, MANUSCRIPT_ROOT, BENCHMARKS_JSON)
    for name in ("numerical_audit.json", "numerical_audit.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@requires_manuscript
def test_generate_writes_both_artefacts(tmp_path: Path) -> None:
    """The CLI target emits the JSON and the Markdown report."""
    written = generate(tmp_path / "out", MANUSCRIPT_ROOT, BENCHMARKS_JSON)
    assert [p.name for p in written] == ["numerical_audit.json", "numerical_audit.md"]
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    loaded = json.loads(written[0].read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "numerical-audit.1"


def test_missing_manuscript_raises(tmp_path: Path) -> None:
    """A missing audited file is an error, not a silent empty inventory."""
    with pytest.raises(NumericalAuditError):
        run_audit(tmp_path / "nonexistent", BENCHMARKS_JSON)


def test_classify_defaults_to_measurement() -> None:
    """An unrecognised literal falls through to ``measurement``."""
    lines = ["the widget emitted $7.31$ units of nothing in particular"]
    literals = extract_literals(lines[0])
    kind, rule = classify(lines, literals[0])
    assert kind == "measurement"
    assert rule == ""
