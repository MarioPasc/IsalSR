"""Unit tests for the anchored manuscript checker of the revision campaign.

The checker exists to catch a number that drifted from its source. Its own
failure mode is silent vacuity: a two-digit literal occurs by accident in any
long document, so a presence test over a concatenated blob passes on a
manuscript that no longer states the quantity at all. These tests pin the
anchoring, the ambiguity rule, the derived structural counts and the
cross-document agreement check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments.scripts.review_campaign.verify import (
    APPENDIX_ID_PREFIX,
    DEFAULT_INVENTORY,
    Check,
    Coupled,
    Document,
    appendix_d1_ids,
    build_checks,
    build_coupled,
    check_coupled,
    check_problem_inventory,
    check_table_rows,
    count_problem_rows,
    grouped,
    load_inventory,
    load_sources,
    locate,
    table_row_ids,
)

# The unrelated sentence that made the old presence test vacuous: the string
# "22" is present, but no statement of the count is.
BIBLIOGRAPHIC_22 = "Bingo~\\cite{randall2022bingo} evolves a population of graphs."
#: The statement that actually carries the count.
STATED_22 = "the search-only speedup reaches $S \\geq 1$ on\n$22$ of the $70$ problems."


def make_doc(*lines: str) -> Document:
    """Build a single-file document from literal lines."""
    return Document("article/paper/results.tex", tuple(lines))


def write_manuscript(root: Path, files: dict[str, str]) -> Path:
    """Lay out a synthetic manuscript tree and return its root."""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def stub_summary() -> dict[str, Any]:
    """A summary with the keys ``build_checks`` and ``build_coupled`` read."""

    def arm(**over: Any) -> dict[str, Any]:
        block: dict[str, Any] = {
            "n_cells": 2100,
            "rho_over_problems": {"mean": 1.66, "min": 1.11, "max": 2.12},
            "key_ms_over_cells": {"median": 0.126},
            "eval_ms_over_cells": {"median": 727.3},
            "overhead_pct_over_cells": {"median": 0.039},
            "n_cells_at_budget_cap": 1727,
            "n_problems_S_ge_1": 70,
            "n_problems_faster_than_native": 29,
        }
        block.update(over)
        return block

    return {
        "n_cells": 12600,
        "n_problems": 70,
        "n_seeds": 30,
        "udfs": {"isalsr": arm()},
        "bingo": {
            "isalsr": arm(
                rho_over_problems={"mean": 1.79, "min": 1.19, "max": 1.85},
                key_ms_over_cells={"median": 0.074},
                eval_ms_over_cells={"median": 1.48},
                overhead_pct_over_cells={"median": 16.11},
                n_problems_S_ge_1=22,
                n_problems_faster_than_native=13,
            )
        },
    }


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------


def test_unrelated_occurrence_is_not_a_match() -> None:
    """A citation key containing "22" must not satisfy the count check.

    This is the defect the anchoring exists to remove: the pre-change checker
    tested ``literal in blob`` and accepted this document.
    """
    doc = make_doc(BIBLIOGRAPHIC_22)
    assert "22" in "\n".join(doc.lines)  # what the old presence test asserted
    check = Check("paper", "$22$ of the", r"S \\geq 1", "problems with S >= 1", 1)
    assert locate([doc], check) == []


def test_stated_count_is_matched_at_its_line() -> None:
    """The same literal in its real sentence resolves to exactly one location."""
    doc = make_doc(BIBLIOGRAPHIC_22, *STATED_22.split("\n"))
    check = Check("paper", "$22$ of the", r"S \\geq 1", "problems with S >= 1", 1)
    assert locate([doc], check) == ["article/paper/results.tex:3"]


def test_ambiguous_match_is_reported_as_many_hits() -> None:
    """Two anchored occurrences are a failure, not a pass."""
    doc = make_doc(
        "the speedup reaches $S \\geq 1$ on $22$ of the $70$ problems.",
        "elsewhere $S \\geq 1$ holds on $22$ of the $70$ problems.",
    )
    check = Check("paper", "$22$ of the", r"S \\geq 1", "problems with S >= 1", 0)
    assert len(locate([doc], check)) == 2


@pytest.mark.parametrize(("window", "expected"), [(0, 0), (1, 1), (2, 1)])
def test_anchor_window_controls_reach(window: int, expected: int) -> None:
    """The anchor may sit on a neighbouring line only when the window allows it."""
    doc = make_doc("UDFS attains a mean reduction factor of", "$1.66 \\pm 0.26$ across")
    check = Check("paper", "1.66", r"mean reduction factor of", "rho mean", window)
    assert len(locate([doc], check)) == expected


def test_file_restriction_separates_the_abstract() -> None:
    """A one-line abstract repeating the numbers is separated by naming the file."""
    docs = [
        Document("article/paper/main.tex", ("a Cohen's $d$ of $2.54$ and $7.05$.",)),
        Document("article/paper/results.tex", ("Cohen's $d$ of $2.54$ with CI",)),
    ]
    loose = Check("paper", "2.54", r"Cohen's", "d", 0)
    pinned = Check("paper", "2.54", r"Cohen's", "d", 0, "results.tex")
    assert len(locate(docs, loose)) == 2
    assert locate(docs, pinned) == ["article/paper/results.tex:1"]


def test_empty_document_yields_no_hits() -> None:
    """An empty file is not a match and does not raise."""
    assert locate([Document("article/paper/empty.tex", ())], Check("paper", "1", "x", "")) == []


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_corpus_includes_the_table_bodies(tmp_path: Path) -> None:
    """The table bodies are part of the checked corpus, not only the prose."""
    root = write_manuscript(
        tmp_path,
        {
            "article/paper/results.tex": "prose\n",
            "article/paper/tab_three_axis.tex": "UDFS & $1.66 \\pm 0.26$ \\\\\n",
            "article/supplementary/supplementary.tex": "prose\n",
            "article/supplementary/table_supplementary_udfs_body.tex": "Nguyen-1 & $1$ \\\\\n",
            "reviews/response_to_reviewers.tex": "prose\n",
        },
    )
    sources = load_sources(root)
    names = {doc.rel for group in sources.values() for doc in group}
    assert "article/paper/tab_three_axis.tex" in names
    assert "article/supplementary/table_supplementary_udfs_body.tex" in names


# ---------------------------------------------------------------------------
# Structural counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [(0, "0"), (70, "70"), (2100, "2{,}100")])
def test_grouped_matches_the_manuscript_convention(value: int, expected: str) -> None:
    assert grouped(value) == expected


def test_structural_literals_come_from_the_summary() -> None:
    """Changing ``n_problems`` changes the literal the checker looks for."""
    summary = stub_summary()
    summary["n_problems"] = 71
    literals = {check.literal for check in build_checks(summary)}
    assert "$71$" in literals
    assert "$N = 71$" in literals
    assert "$70$" not in literals


def test_campaign_counts_are_asserted() -> None:
    """The four counts reviewer R2.6 raised are each checked somewhere."""
    checks = build_checks(stub_summary())
    literals = {check.literal for check in checks}
    assert {"12{,}600", "2{,}100", "$70$", "$30$"} <= literals


def test_count_problem_rows_ignores_the_header_bands() -> None:
    """Only per-problem rows count; header and multicolumn bands do not."""
    doc = make_doc(
        "\\begin{longtable}{lll}",
        " & \\multicolumn{3}{c}{$R^2$ test} \\\\",
        "Problem & NA & NH \\\\",
        "\\midrule",
        "Nguyen-1 & $0.997$ & $0.998$ \\\\",
        "Nguyen-2 & $0.998$ & $0.998$ \\\\",
        "\\end{longtable}",
    )
    assert count_problem_rows(doc) == 2


def test_count_problem_rows_on_an_empty_table() -> None:
    assert count_problem_rows(make_doc()) == 0


def test_table_row_count_failure_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """A body with the wrong number of rows fails against ``n_problems``."""
    body = "Problem & NA \\\\\nNguyen-1 & $1$ \\\\\n"
    sources = {
        "supplementary": [
            Document(f"article/supplementary/{name}", tuple(body.splitlines()))
            for name in (
                "table_supplementary_udfs_body.tex",
                "table_supplementary_bingo_body.tex",
                "tab_supp_phi_per_problem.tex",
            )
        ]
    }
    ran, failures = check_table_rows(sources, 70)
    capsys.readouterr()
    assert ran == 3
    assert len(failures) == 3
    assert "has 1 problem rows, expected 70" in failures[0]


# ---------------------------------------------------------------------------
# Cross-document agreement
# ---------------------------------------------------------------------------


def test_coupled_quantity_agrees_across_documents(capsys: pytest.CaptureFixture[str]) -> None:
    """Two documents stating the same value pass as one coupled group."""
    sources = {
        "paper": [
            Document("article/paper/results.tex", ("mean reduction factor of $1.66$ across",)),
            Document("article/paper/discussion.tex", ("suite (means $1.66$ and $1.79$),",)),
        ]
    }
    group = Coupled(
        "rho mean, UDFS",
        "1.66",
        1.6637,
        (
            ("paper", "results.tex", r"mean reduction factor of", 0),
            ("paper", "discussion.tex", r"suite \(means", 0),
        ),
    )
    ran, failures = check_coupled(sources, [group])
    capsys.readouterr()
    assert (ran, failures) == (2, [])


def test_coupled_quantity_disagreement_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """One document drifting to a different literal is a failure.

    This is errata E1 and E8 of the previous round: two statements of one
    measurement that disagree, both of which survived a full review.
    """
    sources = {
        "paper": [
            Document("article/paper/results.tex", ("mean reduction factor of $1.66$ across",)),
            Document("article/paper/discussion.tex", ("suite (means $1.45$ and $1.79$),",)),
        ]
    }
    group = Coupled(
        "rho mean, UDFS",
        "1.66",
        1.6637,
        (
            ("paper", "results.tex", r"mean reduction factor of", 0),
            ("paper", "discussion.tex", r"suite \(means", 0),
        ),
    )
    _, failures = check_coupled(sources, [group])
    capsys.readouterr()
    assert len(failures) == 1
    assert "discussion.tex" in failures[0]


def test_coupled_literals_are_derived_not_written_out() -> None:
    """The coupled literal follows the summary, so a drifted manuscript fails."""
    summary = stub_summary()
    summary["udfs"]["isalsr"]["rho_over_problems"]["mean"] = 1.45
    by_name = {group.name: group for group in build_coupled(summary)}
    assert by_name["rho mean, UDFS"].literal == "1.45"


def test_coupled_groups_have_at_least_two_locations() -> None:
    """A coupled group with one location would assert nothing about agreement."""
    for group in build_coupled(stub_summary()):
        assert len(group.locations) >= 2, group.name


# ---------------------------------------------------------------------------
# Documented suite vs reported suite (T09 AC-2)
# ---------------------------------------------------------------------------

#: The two labels the synthetic Appendix D.1 below carries. The real subsection
#: has six; the tests declare their own so that the fixture is small.
SYNTHETIC_TABLES = ("tab:nguyen", "tab:strogatz")

#: The suite the synthetic manuscript is supposed to document and report. It
#: exercises both id-shape normalisations: bare integers and a dropped
#: ``Strogatz-`` prefix.
SYNTHETIC_SUITE = {"Nguyen-1", "Nguyen-2", "Strogatz-bacres1", "Strogatz-bacres2"}

_STROGATZ_INPUT = (
    "\\begin{tabular}{@{}ll@{}}\n"
    "\\toprule\n"
    "ID & Expression \\\\\n"
    "\\midrule\n"
    "bacres1 & $y$ \\\\\n"
    "bacres2 & $z$ \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
)

_RESULT_BODY = (
    "\\cmidrule(lr){2-3}\nProblem & $R^2$ \\\\\n"
    "\\midrule\n"
    "Nguyen-1 & $0.9$ \\\\\n"
    "Nguyen-2 & $0.9$ \\\\\n"
    "Strogatz-bacres1 & $0.9$ \\\\\n"
    "Strogatz-bacres2 & $0.9$ \\\\\n"
)


def synthetic_sources(
    nguyen_rows: str = "1 & $x$ \\\\\n2 & $x^2$ \\\\\n",
    result_body: str = _RESULT_BODY,
) -> dict[str, list[Document]]:
    """Build a two-table Appendix D.1 plus the two per-problem result bodies.

    Args:
        nguyen_rows: Data rows of ``tab:nguyen``, as they are printed.
        result_body: Body shared by the UDFS and Bingo result tables.

    Returns:
        The document groups ``load_sources`` would have produced.
    """
    supplementary = (
        "\\subsection{Benchmark suite}\n"
        "\\label{sec:supp_benchmarks}\n"
        "The suite contains $4$ problems.\n"
        "\\begin{table}\n"
        "\\caption{Nguyen benchmarks.}\n"
        "\\label{tab:nguyen}\n"
        "\\begin{tabular}{@{}ll@{}}\n"
        "\\toprule\n"
        "ID & Expression \\\\\n"
        "\\midrule\n"
        f"{nguyen_rows}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\\begin{table}\n"
        "\\label{tab:strogatz}\n"
        "\\input{tab_supp_bench_struct_strogatz}\n"
        "\\end{table}\n"
        # Anything past the next subsection is outside D.1 and must be ignored.
        "\\subsection{Hyperparameters}\n"
        "\\label{tab:elsewhere}\n"
        "\\begin{tabular}{@{}ll@{}}\n"
        "Ghost-1 & $x$ \\\\\n"
        "\\end{tabular}\n"
    )
    files = {
        "supplementary.tex": supplementary,
        "tab_supp_bench_struct_strogatz.tex": _STROGATZ_INPUT,
        "table_supplementary_udfs_body.tex": result_body,
        "table_supplementary_bingo_body.tex": result_body,
    }
    return {
        "supplementary": [
            Document(f"article/supplementary/{name}", tuple(text.splitlines()))
            for name, text in files.items()
        ]
    }


def test_appendix_ids_expand_inputs_and_normalise_shapes() -> None:
    """Both tables are read, the ``\\input`` is expanded and prefixes are applied."""
    per_table, failures = appendix_d1_ids(synthetic_sources(), SYNTHETIC_TABLES)
    assert failures == []
    assert per_table["tab:nguyen"] == ["Nguyen-1", "Nguyen-2"]
    assert per_table["tab:strogatz"] == ["Strogatz-bacres1", "Strogatz-bacres2"]
    assert "tab:elsewhere" not in per_table  # past the next \subsection


def test_documented_and_reported_suites_agree(capsys: pytest.CaptureFixture[str]) -> None:
    """Appendix, UDFS body and Bingo body all list the suite: three checks, no failure."""
    ran, failures = check_problem_inventory(synthetic_sources(), SYNTHETIC_SUITE, SYNTHETIC_TABLES)
    capsys.readouterr()
    assert (ran, failures) == (3, [])


def test_problem_documented_nowhere_is_named(capsys: pytest.CaptureFixture[str]) -> None:
    """A problem run but never described fails, and the message names it.

    This is the defect a row count cannot see: the appendix and the result table
    can both have the right length while describing different problems.
    """
    sources = synthetic_sources(nguyen_rows="1 & $x$ \\\\\n3 & $x^3$ \\\\\n")
    _, failures = check_problem_inventory(sources, SYNTHETIC_SUITE, SYNTHETIC_TABLES)
    capsys.readouterr()
    assert len(failures) == 1
    assert "missing (1): Nguyen-2" in failures[0]
    assert "extra (1): Nguyen-3" in failures[0]


def test_extra_problem_in_a_result_table_is_named(capsys: pytest.CaptureFixture[str]) -> None:
    """A result row for a problem outside the suite is reported by id."""
    sources = synthetic_sources(result_body=_RESULT_BODY + "Nguyen-9 & $0.9$ \\\\\n")
    _, failures = check_problem_inventory(sources, SYNTHETIC_SUITE, SYNTHETIC_TABLES)
    capsys.readouterr()
    assert len(failures) == 2  # one per result body
    assert all("extra (1): Nguyen-9" in line for line in failures)


def test_duplicated_problem_row_is_named(capsys: pytest.CaptureFixture[str]) -> None:
    """A repeated row keeps the set equal but is still a defect, so it is reported."""
    sources = synthetic_sources(result_body=_RESULT_BODY + "Nguyen-1 & $0.9$ \\\\\n")
    _, failures = check_problem_inventory(sources, SYNTHETIC_SUITE, SYNTHETIC_TABLES)
    capsys.readouterr()
    assert len(failures) == 2
    assert all("duplicated (1): Nguyen-1" in line for line in failures)


def test_missing_appendix_table_is_reported() -> None:
    """A table dropped from D.1 fails as a missing table, not only as missing ids."""
    _, failures = appendix_d1_ids(synthetic_sources(), (*SYNTHETIC_TABLES, "tab:feynman"))
    assert failures == ["Appendix D.1 no longer carries the table tab:feynman"]


def test_normalisation_table_is_declared_for_exactly_two_tables() -> None:
    """The two id-shape exceptions are declared, and no others are.

    An undeclared normalisation would be a silent rewrite of an id; a declared
    one that is no longer needed shows up in the next test as a loud mismatch.
    """
    assert APPENDIX_ID_PREFIX == {"tab:nguyen": "Nguyen-", "tab:strogatz": "Strogatz-"}


def test_prefix_is_applied_unconditionally_so_a_shape_change_is_loud(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If ``tab:nguyen`` starts printing full ids, every row fails, not none of them."""
    sources = synthetic_sources(nguyen_rows="Nguyen-1 & $x$ \\\\\nNguyen-2 & $x^2$ \\\\\n")
    _, failures = check_problem_inventory(sources, SYNTHETIC_SUITE, SYNTHETIC_TABLES)
    capsys.readouterr()
    assert len(failures) == 1
    assert "missing (2): Nguyen-1, Nguyen-2" in failures[0]
    assert "extra (2): Nguyen-Nguyen-1, Nguyen-Nguyen-2" in failures[0]


def test_missing_supplementary_file_is_reported() -> None:
    """An absent ``supplementary.tex`` is a failure, not an empty documented set."""
    per_table, failures = appendix_d1_ids({"supplementary": []}, SYNTHETIC_TABLES)
    assert per_table == {}
    assert "no supplementary.tex" in failures[0]


def test_missing_subsection_label_is_reported() -> None:
    """A renamed anchor is a failure rather than a silently empty region."""
    sources = {"supplementary": [Document("article/supplementary/supplementary.tex", ("prose",))]}
    per_table, failures = appendix_d1_ids(sources, SYNTHETIC_TABLES)
    assert per_table == {}
    assert "sec:supp_benchmarks" in failures[0]


@pytest.mark.parametrize(
    ("tabular", "expected"),
    [
        ("\\begin{tabular}{ll}\n\\end{tabular}", []),
        ("ID & x \\\\\n\\midrule\nA & 1 \\\\", ["A"]),
        ("& \\multicolumn{2}{c}{band} \\\\\nProblem & x \\\\\nB & 1 \\\\", ["B"]),
        ("\\midrule A & 1 \\\\", ["A"]),  # rule sharing the row's line
        ("test\\_4 & 1 \\\\", ["test_4"]),  # the one escaped id in the suite
        ("A &\n1 \\\\\nB & 2 \\\\", ["A", "B"]),  # a row wrapped across lines
    ],
)
def test_table_row_ids_reads_data_rows_only(tabular: str, expected: list[str]) -> None:
    assert table_row_ids(tabular) == expected


def test_load_inventory_reads_the_seventy_problem_suite() -> None:
    """The authoritative suite is the generated inventory, not a literal in a test."""
    inventory = load_inventory(DEFAULT_INVENTORY)
    assert len(inventory) == 70
    assert {"Nguyen-1", "Strogatz-bacres1", "test_4", "I.6.20a"} <= inventory
