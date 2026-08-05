"""Unit tests for the Appendix D.1 benchmark documentation generator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from experiments.scripts import generate_appendix_d_tables as gen

REPO_ROOT = Path(gen.__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
BIB_PATH = Path(gen.DEFAULT_BIB_PATH)

EXPECTED_TIER_COUNTS: dict[str, int] = {
    "nguyen": 12,
    "feynman": 10,
    "hard": 10,
    "cherrypicked": 10,
    "roundoff": 8,
    "feynman_remainder": 6,
    "strogatz": 14,
}
EXPECTED_TOTAL = 70


@pytest.fixture(scope="module")
def bib_keys() -> set[str]:
    """Citation keys of the manuscript bibliography."""
    if not BIB_PATH.exists():
        pytest.skip(f"references.bib not reachable at {BIB_PATH}")
    return gen.parse_bib_keys(BIB_PATH)


@pytest.fixture(scope="module")
def rows(bib_keys: set[str]) -> list[gen.BenchmarkRow]:
    """All 70 assembled benchmark rows."""
    return gen.build_rows(CONFIG_DIR, bib_keys)


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate the artefacts once into a temporary directory."""
    if not BIB_PATH.exists():
        pytest.skip(f"references.bib not reachable at {BIB_PATH}")
    out = tmp_path_factory.mktemp("appendix_d")
    tex_path, json_path = gen.generate(out, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)
    return {
        "dir": out,
        "tex_path": tex_path,
        "json_path": json_path,
        "tex": tex_path.read_text(encoding="utf-8"),
        "payload": json.loads(json_path.read_text(encoding="utf-8")),
    }


# ----------------------------------------------------------------------
# 1. Exactly 70 rows, ID set equals the union of the seven suite lists.
# ----------------------------------------------------------------------


def test_seven_suites_declared() -> None:
    assert set(gen.SUITES) == set(EXPECTED_TIER_COUNTS)


def test_row_count_is_seventy(rows: list[gen.BenchmarkRow]) -> None:
    assert len(rows) == EXPECTED_TOTAL


@pytest.mark.parametrize("tier,expected", sorted(EXPECTED_TIER_COUNTS.items()))
def test_per_tier_counts(rows: list[gen.BenchmarkRow], tier: str, expected: int) -> None:
    assert sum(r.tier == tier for r in rows) == expected


def test_id_set_equals_union_of_suite_lists(rows: list[gen.BenchmarkRow]) -> None:
    union = {str(b["name"]) for lst in gen.SUITES.values() for b in lst}
    assert {r.problem_id for r in rows} == union
    assert len(union) == EXPECTED_TOTAL, "suite lists overlap or double-count"


def test_roundoff_sublists_are_not_double_counted(
    rows: list[gen.BenchmarkRow],
) -> None:
    """``_FEYNMAN_ROUNDOFF``/``_GP_ROUNDOFF`` are subsets of ``ROUNDOFF_BENCHMARKS``."""
    from benchmarks.datasets import roundoff

    sub = {b["name"] for b in roundoff._FEYNMAN_ROUNDOFF} | {
        b["name"] for b in roundoff._GP_ROUNDOFF
    }
    assert sub <= {b["name"] for b in roundoff.ROUNDOFF_BENCHMARKS}
    assert sum(r.tier == "roundoff" for r in rows) == 8


# ----------------------------------------------------------------------
# 2. Every mandatory field non-empty -- hard failure, not a warning.
# ----------------------------------------------------------------------


def test_all_mandatory_fields_non_empty(rows: list[gen.BenchmarkRow]) -> None:
    gen.validate_rows(rows)  # raises on any empty field
    for row in rows:
        for name, value in row.required_fields().items():
            assert value not in (None, "", [], 0), f"{row.problem_id}.{name} empty"


def test_validate_rows_rejects_an_empty_field(rows: list[gen.BenchmarkRow]) -> None:
    import dataclasses

    broken = dataclasses.replace(rows[0], sampling_protocol="")
    with pytest.raises(gen.AppendixGenerationError, match="sampling_protocol"):
        gen.validate_rows([broken])


def test_validate_rows_rejects_duplicate_ids(rows: list[gen.BenchmarkRow]) -> None:
    with pytest.raises(gen.AppendixGenerationError, match="Duplicate"):
        gen.validate_rows([rows[0], rows[0]])


def test_variable_ranges_length_matches_n_variables(
    rows: list[gen.BenchmarkRow],
) -> None:
    for row in rows:
        assert len(row.variable_ranges) == row.n_variables, row.problem_id
        for lo, hi in row.variable_ranges:
            assert lo < hi, f"{row.problem_id}: degenerate range [{lo}, {hi}]"


def test_expression_latex_present_for_all(rows: list[gen.BenchmarkRow]) -> None:
    for row in rows:
        assert row.expression_latex.strip(), row.problem_id


def test_sampling_protocols_are_known(rows: list[gen.BenchmarkRow]) -> None:
    assert {r.sampling_protocol for r in rows} <= set(gen.SAMPLING_LABELS)


# ----------------------------------------------------------------------
# 2b. Sizes must be the ones the campaign actually used.
# ----------------------------------------------------------------------


def test_campaign_sizes_agree_between_hosts() -> None:
    sizes = gen.load_campaign_sizes(CONFIG_DIR)
    assert sizes == {
        "nguyen": (240, 1000),
        "feynman": (1000, 250),
        "hard": (1000, 250),
        "cherrypicked": (1000, 250),
        "roundoff": (1000, 250),
        "feynman_remainder": (1000, 250),
        "strogatz": (300, 100),
    }


@pytest.mark.parametrize(
    "problem_id,n_train,n_test",
    [
        ("Nguyen-1", 240, 1000),
        ("I.6.20a", 1000, 250),
        ("Pagie-1", 676, 2500),
        ("Korns-12", 2000, 2000),
        ("Vladislavleva-4", 1024, 5000),
        ("Vladislavleva-2", 100, 221),
        ("Keijzer-6", 50, 120),
        ("Vlad-7", 300, 1200),
        ("Strogatz-vdp1", 300, 100),
    ],
)
def test_known_sample_sizes(
    rows: list[gen.BenchmarkRow], problem_id: str, n_train: int, n_test: int
) -> None:
    row = next(r for r in rows if r.problem_id == problem_id)
    assert (row.n_train, row.n_test) == (n_train, n_test)


def test_sample_sizes_are_seed_independent(bib_keys: set[str]) -> None:
    a = gen.build_rows(CONFIG_DIR, bib_keys, seed=1)
    b = gen.build_rows(CONFIG_DIR, bib_keys, seed=7)
    assert [(r.problem_id, r.n_train, r.n_test) for r in a] == [
        (r.problem_id, r.n_train, r.n_test) for r in b
    ]


# ----------------------------------------------------------------------
# 3. Citation keys resolve against references.bib.
# ----------------------------------------------------------------------


def test_citation_map_covers_exactly_the_campaign(rows: list[gen.BenchmarkRow]) -> None:
    assert set(gen.CITATION_MAP) == {r.problem_id for r in rows}


def test_unresolved_keys_are_exactly_the_declared_ones(
    rows: list[gen.BenchmarkRow], bib_keys: set[str]
) -> None:
    audit = gen.audit_citations(rows, bib_keys)
    assert set(audit["citation_keys_missing"]) == set(gen.EXPECTED_MISSING_CITATIONS)


def test_every_non_strogatz_citation_resolves(
    rows: list[gen.BenchmarkRow], bib_keys: set[str]
) -> None:
    unresolved = {r.problem_id: r.citation_key for r in rows if r.citation_key not in bib_keys}
    assert set(unresolved.values()) == {"strogatz1994"}
    assert len(unresolved) == 14


def test_missing_key_carries_a_bibtex_stub(
    rows: list[gen.BenchmarkRow], bib_keys: set[str]
) -> None:
    audit = gen.audit_citations(rows, bib_keys)
    for key in audit["citation_keys_missing"]:
        assert key in audit["suggested_bibtex"], key
        assert audit["suggested_bibtex"][key].startswith("@")


def test_generate_fails_on_an_undeclared_missing_key(
    tmp_path: Path, bib_keys: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(gen.CITATION_MAP, "Nguyen-1", "does_not_exist_2099")
    with pytest.raises(gen.AppendixGenerationError, match="does_not_exist_2099"):
        gen.generate(tmp_path, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)


def test_build_rows_rejects_an_unmapped_problem(
    bib_keys: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    trimmed = dict(gen.CITATION_MAP)
    trimmed.pop("Nguyen-1")
    monkeypatch.setattr(gen, "CITATION_MAP", trimmed)
    with pytest.raises(gen.AppendixGenerationError, match="No citation assignment"):
        gen.build_rows(CONFIG_DIR, bib_keys)


# ----------------------------------------------------------------------
# 4. The emitted LaTeX is a structurally valid fragment.
# ----------------------------------------------------------------------


def test_latex_fragment_validates(artefacts: dict[str, Any]) -> None:
    gen.validate_latex_fragment(artefacts["tex"])


def test_latex_has_one_table_per_tier(artefacts: dict[str, Any]) -> None:
    tex = artefacts["tex"]
    assert tex.count("\\begin{table}") == len(EXPECTED_TIER_COUNTS)
    assert tex.count("\\begin{table}") == tex.count("\\end{table}")
    assert tex.count("\\begin{tabular}") == tex.count("\\end{tabular}")


def test_latex_row_count_matches(artefacts: dict[str, Any]) -> None:
    body_rows = [
        line
        for line in artefacts["tex"].splitlines()
        if line.endswith(" \\\\") and "\\mathrm{train}" not in line
    ]
    assert len(body_rows) == EXPECTED_TOTAL


def test_latex_cites_every_problem(artefacts: dict[str, Any]) -> None:
    cites = re.findall(r"\\cite\{([^}]+)\}", artefacts["tex"])
    assert len(cites) == EXPECTED_TOTAL


def test_latex_escapes_underscore_in_problem_ids(artefacts: dict[str, Any]) -> None:
    """``test_4`` and the ``grid_2d_skip_zero`` label must not leak a raw ``_``."""
    assert "test\\_4" in artefacts["tex"]
    # Every non-comment text-mode segment is free of a bare underscore; this is
    # exactly what ``validate_latex_fragment`` enforces, asserted independently.
    for line in artefacts["tex"].splitlines():
        if line.lstrip().startswith("%"):
            continue
        for idx, segment in enumerate(line.split("$")):
            if idx % 2 == 0:
                assert not re.search(r"(?<!\\)_", segment), line


@pytest.mark.parametrize(
    "bad",
    [
        "\\begin{table}\n\\end{tabular}\n",
        "\\begin{table}\n",
        "\\end{table}\n",
        "raw_underscore in text mode\n",
        "unbalanced $math\n",
        "unbalanced {brace\n",
    ],
)
def test_validate_latex_fragment_rejects_defects(bad: str) -> None:
    with pytest.raises(gen.AppendixGenerationError):
        gen.validate_latex_fragment(bad)


def test_escape_latex_text_round_trip() -> None:
    assert gen.escape_latex_text("a_b") == "a\\_b"
    assert gen.escape_latex_text("50%") == "50\\%"
    assert gen.escape_latex_text("x^2") == "x\\^{}2"
    assert gen.escape_latex_text("a&b") == "a\\&b"


def test_format_variable_ranges_compacts_identical_boxes() -> None:
    assert gen.format_variable_ranges([[1.0, 5.0]]) == "[1, 5]"
    assert gen.format_variable_ranges([[1.0, 5.0]] * 3) == "[1, 5]^{3}"
    assert gen.format_variable_ranges([[0.0, 1.0], [2.0, 3.0]]) == "[0, 1],\\, [2, 3]"


# ----------------------------------------------------------------------
# 5. Re-running the generator is byte-identical.
# ----------------------------------------------------------------------


def test_regeneration_is_byte_identical(artefacts: dict[str, Any], tmp_path: Path) -> None:
    second_dir = tmp_path / "second"
    tex2, json2 = gen.generate(second_dir, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)
    assert tex2.read_bytes() == artefacts["tex_path"].read_bytes()
    assert json2.read_bytes() == artefacts["json_path"].read_bytes()


def test_row_order_is_deterministic(bib_keys: set[str]) -> None:
    a = [r.problem_id for r in gen.build_rows(CONFIG_DIR, bib_keys)]
    b = [r.problem_id for r in gen.build_rows(CONFIG_DIR, bib_keys)]
    assert a == b
    assert a[:2] == ["Nguyen-1", "Nguyen-2"]
    assert a[-1] == "Strogatz-vdp2"


# ----------------------------------------------------------------------
# JSON payload contract.
# ----------------------------------------------------------------------


def test_json_payload_shape(artefacts: dict[str, Any]) -> None:
    payload = artefacts["payload"]
    assert payload["counts"]["n_problems"] == EXPECTED_TOTAL
    assert payload["counts"]["per_tier"] == EXPECTED_TIER_COUNTS
    assert len(payload["problems"]) == EXPECTED_TOTAL
    assert payload["provenance"]["size_probe_seed"] == gen.SIZE_PROBE_SEED
    assert set(payload["provenance"]["operator_sets"]) == {"bingo", "udfs"}


def test_json_problems_carry_all_mandatory_keys(artefacts: dict[str, Any]) -> None:
    required = {
        "problem_id",
        "expression",
        "n_variables",
        "variable_ranges",
        "n_train",
        "n_test",
        "sampling_protocol",
        "citation_key",
        "tier",
    }
    for problem in artefacts["payload"]["problems"]:
        assert required <= set(problem), problem.get("problem_id")
        for key in required:
            assert problem[key] not in (None, "", [], 0), (problem["problem_id"], key)
