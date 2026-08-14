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
    written = gen.generate(out, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)
    tex_path, json_path = written[0], written[1]
    return {
        "dir": out,
        "written": written,
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
    written = gen.generate(second_dir, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)
    for path in written:
        assert path.read_bytes() == (artefacts["dir"] / path.name).read_bytes(), path.name


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


# ----------------------------------------------------------------------
# 6. Supplementary bodies for the 28 problems Appendix D.1 does not document.
# ----------------------------------------------------------------------

SUPP_FEYNMAN_FILE = "tab_supp_bench_struct_feynman.tex"
SUPP_OTHER_FILE = "tab_supp_bench_struct_other.tex"

#: Internal tier keys that must never surface in reviewer-facing LaTeX.
FORBIDDEN_TIER_WORDS = ("hard", "cherrypicked", "cherry-picked", "roundoff", "round-off")


def _data_rows(body: str) -> list[str]:
    """Return the tabular data lines between ``\\midrule`` and ``\\bottomrule``."""
    head, _, rest = body.partition("\\midrule")
    assert head and rest, "body has no \\midrule"
    payload, _, tail = rest.partition("\\bottomrule")
    assert tail is not None
    return [ln for ln in payload.splitlines() if ln.strip().endswith("\\\\")]


@pytest.fixture(scope="module")
def supp_bodies(artefacts: dict[str, Any]) -> dict[str, str]:
    """The two supplementary tabular bodies, read from the generated directory."""
    out = artefacts["dir"]
    bodies = {}
    for name in (SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE):
        path = out / name
        assert path.exists(), f"generator did not emit {name}"
        bodies[name] = path.read_text(encoding="utf-8")
    return bodies


def test_supplementary_selection_is_twenty_eight(rows: list[gen.BenchmarkRow]) -> None:
    supp = gen.select_supplementary_rows(rows)
    assert len(supp) == 28


def test_supplementary_split_is_fourteen_fourteen(rows: list[gen.BenchmarkRow]) -> None:
    feyn, other = gen.split_supplementary_rows(gen.select_supplementary_rows(rows))
    assert len(feyn) == 14
    assert len(other) == 14
    assert {r.citation_key for r in feyn} == {gen.FEYNMAN_CITATION_KEY}
    assert gen.FEYNMAN_CITATION_KEY not in {r.citation_key for r in other}


def test_supplementary_covers_exactly_the_undocumented_tiers(
    rows: list[gen.BenchmarkRow],
) -> None:
    expected = {r.problem_id for r in rows if r.tier in gen.SUPPLEMENTARY_TIERS}
    supp = gen.select_supplementary_rows(rows)
    assert {r.problem_id for r in supp} == expected
    assert len(expected) == 28


def test_supplementary_sources_are_five_keys_over_six_families(
    rows: list[gen.BenchmarkRow],
) -> None:
    """Six named families, five BibTeX keys: DSO ships Livermore and R-rationals."""
    _, other = gen.split_supplementary_rows(gen.select_supplementary_rows(rows))
    assert {r.citation_key for r in other} == {
        "keijzer2003",
        "korns2011",
        "pagie1997",
        "petersen2021",
        "vladislavleva2009",
    }
    assert gen.SUPP_OTHER_SOURCE_COUNT == 5


def test_supp_feynman_body_has_fourteen_data_rows(supp_bodies: dict[str, str]) -> None:
    assert len(_data_rows(supp_bodies[SUPP_FEYNMAN_FILE])) == 14


def test_supp_other_body_has_fourteen_data_rows(supp_bodies: dict[str, str]) -> None:
    assert len(_data_rows(supp_bodies[SUPP_OTHER_FILE])) == 14


@pytest.mark.parametrize("name", [SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE])
def test_supp_bodies_are_float_free(supp_bodies: dict[str, str], name: str) -> None:
    body = supp_bodies[name]
    for forbidden in ("\\begin{table}", "\\caption", "\\label"):
        assert forbidden not in body, f"{name} carries {forbidden}"
    assert body.count("\\begin{tabular}") == 1
    assert body.count("\\end{tabular}") == 1


@pytest.mark.parametrize("name", [SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE])
def test_supp_bodies_never_name_internal_tiers(supp_bodies: dict[str, str], name: str) -> None:
    lowered = supp_bodies[name].lower()
    for word in FORBIDDEN_TIER_WORDS:
        assert word not in lowered, f"{name} leaks internal tier word {word!r}"


@pytest.mark.parametrize("name", [SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE])
def test_supp_bodies_use_booktabs_rules(supp_bodies: dict[str, str], name: str) -> None:
    body = supp_bodies[name]
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert body.count(rule) == 1, f"{name}: {rule}"


def test_supp_feynman_mirrors_shipped_column_spec(supp_bodies: dict[str, str]) -> None:
    body = supp_bodies[SUPP_FEYNMAN_FILE]
    assert "\\begin{tabular}{@{}llcl@{}}" in body
    assert "ID & Expression & $m$ & Variable ranges \\\\" in body


def test_supp_other_extends_with_three_columns(supp_bodies: dict[str, str]) -> None:
    body = supp_bodies[SUPP_OTHER_FILE]
    assert "\\begin{tabular}{@{}llclrrl@{}}" in body
    assert "$n_{\\mathrm{train}}$" in body
    assert "$n_{\\mathrm{test}}$" in body
    assert "Sampling" in body


def test_supp_feynman_rows_are_uniform_1000_250(rows: list[gen.BenchmarkRow]) -> None:
    feyn, _ = gen.split_supplementary_rows(gen.select_supplementary_rows(rows))
    for row in feyn:
        assert (row.n_train, row.n_test, row.sampling_protocol) == (
            1000,
            250,
            "uniform",
        ), row.problem_id


def test_supp_other_rows_carry_their_own_sizes(supp_bodies: dict[str, str]) -> None:
    body = supp_bodies[SUPP_OTHER_FILE]
    for pid, n_train, n_test in (
        ("Pagie-1", 676, 2500),
        ("Korns-12", 2000, 2000),
        ("Vladislavleva-4", 1024, 5000),
        ("Keijzer-6", 50, 120),
    ):
        line = next(ln for ln in _data_rows(body) if ln.startswith(pid + " &"))
        cells = [c.strip() for c in line.removesuffix(" \\\\").split(" & ")]
        assert cells[4] == str(n_train), (pid, cells)
        assert cells[5] == str(n_test), (pid, cells)


def test_supp_every_inventory_id_appears_once(
    supp_bodies: dict[str, str], rows: list[gen.BenchmarkRow]
) -> None:
    joined = supp_bodies[SUPP_FEYNMAN_FILE] + supp_bodies[SUPP_OTHER_FILE]
    for row in gen.select_supplementary_rows(rows):
        assert joined.count(gen.escape_latex_text(row.problem_id) + " &") == 1, row.problem_id


@pytest.mark.parametrize("name", [SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE])
def test_supp_bodies_pass_the_latex_validator(supp_bodies: dict[str, str], name: str) -> None:
    gen.validate_latex_fragment(supp_bodies[name])


def test_supp_bodies_regenerate_byte_identical(artefacts: dict[str, Any], tmp_path: Path) -> None:
    second = tmp_path / "supp_second"
    gen.generate(second, CONFIG_DIR, BIB_PATH, gen.SIZE_PROBE_SEED)
    for name in (SUPP_FEYNMAN_FILE, SUPP_OTHER_FILE):
        assert (second / name).read_bytes() == (artefacts["dir"] / name).read_bytes()


def test_generate_returns_four_paths(artefacts: dict[str, Any]) -> None:
    names = [p.name for p in artefacts["written"]]
    assert names == [
        "appendix_d_tables.tex",
        "appendix_d_benchmarks.json",
        SUPP_FEYNMAN_FILE,
        SUPP_OTHER_FILE,
    ]


def test_supplementary_guard_rejects_non_uniform_feynman_row(
    rows: list[gen.BenchmarkRow],
) -> None:
    """The uniform-1000/250 premise is enforced, not assumed."""
    feyn, _ = gen.split_supplementary_rows(gen.select_supplementary_rows(rows))
    import dataclasses

    broken = [dataclasses.replace(feyn[0], n_train=999)] + list(feyn[1:])
    with pytest.raises(gen.AppendixGenerationError, match="1000"):
        gen.validate_supplementary_feynman_uniformity(broken)


def test_supplementary_guard_rejects_wrong_group_size(
    rows: list[gen.BenchmarkRow],
) -> None:
    feyn, other = gen.split_supplementary_rows(gen.select_supplementary_rows(rows))
    with pytest.raises(gen.AppendixGenerationError, match="14"):
        gen.validate_supplementary_split(feyn[:-1], other)


# ----------------------------------------------------------------------
# 7. Display overrides: printed expression must be the campaign's target.
# ----------------------------------------------------------------------

#: A decimal literal with more than this many significant figures is a raw
#: float dump, not a benchmark coefficient.
MAX_SIGNIFICANT_FIGURES = 4


def _long_decimals(text: str) -> list[str]:
    """Decimal literals in ``text`` carrying more than 4 significant figures."""
    return [
        m
        for m in re.findall(r"\d+\.\d+", text)
        if len(m.replace(".", "").lstrip("0")) > MAX_SIGNIFICANT_FIGURES
    ]


def test_keijzer6_cell_is_the_harmonic_sum_not_its_logarithm(
    supp_bodies: dict[str, str],
) -> None:
    """The campaign's y is H(x_0); log(x_0) + gamma is only its asymptotic form."""
    line = next(
        ln for ln in _data_rows(supp_bodies[SUPP_OTHER_FILE]) if ln.startswith("Keijzer-6 &")
    )
    cell = line.split(" & ")[1]
    assert "\\sum" in cell, cell
    assert "\\log" not in cell, cell
    assert "0.577" not in cell, cell


def test_keijzer6_row_carries_a_display_note(rows: list[gen.BenchmarkRow]) -> None:
    row = next(r for r in rows if r.problem_id == "Keijzer-6")
    assert row.expression_display_note
    assert "asymptotic" in row.expression_display_note


def test_display_overrides_are_declared_and_reachable(
    rows: list[gen.BenchmarkRow],
) -> None:
    ids = {r.problem_id for r in rows}
    assert set(gen.DISPLAY_OVERRIDES) <= ids, "stale override key"
    for pid, override in gen.DISPLAY_OVERRIDES.items():
        assert override.expression_latex.strip(), pid
        assert override.note.strip(), pid


def _expression_cells(text: str) -> list[str]:
    """The second column of every tabular data line in an emitted file."""
    cells = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or not stripped.endswith("\\\\") or " & " not in stripped:
            continue
        cells.append(stripped.removesuffix(" \\\\").split(" & ")[1])
    return cells


@pytest.mark.parametrize(
    "name",
    [
        "appendix_d_tables.tex",
        SUPP_FEYNMAN_FILE,
        SUPP_OTHER_FILE,
    ],
)
def test_no_emitted_expression_cell_dumps_a_long_float(
    artefacts: dict[str, Any], name: str
) -> None:
    text = (artefacts["dir"] / name).read_text(encoding="utf-8")
    cells = _expression_cells(text)
    assert cells, name
    offenders = {c: _long_decimals(c) for c in cells if _long_decimals(c)}
    assert offenders == {}, (name, offenders)


def test_strogatz_range_literals_are_kept_at_full_precision(
    artefacts: dict[str, Any],
) -> None:
    """Empirical PMLB data extents are not float dumps and must not be rounded."""
    rows = {p["problem_id"]: p for p in artefacts["payload"]["problems"]}
    ranges = gen.format_variable_ranges(rows["Strogatz-bacres1"]["variable_ranges"])
    assert "54.445" in ranges
    assert _long_decimals(ranges) == ["54.445"]


def test_json_expressions_carry_no_long_float(artefacts: dict[str, Any]) -> None:
    offenders = {
        p["problem_id"]: _long_decimals(p["expression_latex"])
        for p in artefacts["payload"]["problems"]
        if _long_decimals(p["expression_latex"])
    }
    assert offenders == {}, offenders


def test_genuine_benchmark_coefficients_are_not_rounded_away(
    artefacts: dict[str, Any],
) -> None:
    """Korns-12's 2.1, 1.3 and 9.8 are exact and must survive the float audit."""
    korns = next(p for p in artefacts["payload"]["problems"] if p["problem_id"] == "Korns-12")
    for coefficient in ("2.1", "1.3", "9.8"):
        assert coefficient in korns["expression_latex"], coefficient


def test_supp_other_body_emits_the_display_notes_as_comments(
    supp_bodies: dict[str, str],
) -> None:
    body = supp_bodies[SUPP_OTHER_FILE]
    head = body.split("\\setlength")[0]
    assert head.strip(), "note block missing"
    assert all(ln.lstrip().startswith("%") for ln in head.strip().splitlines())
    assert "Keijzer-6" in head


def test_supp_feynman_body_has_no_note_block(supp_bodies: dict[str, str]) -> None:
    """No AI Feynman row carries an override, so no comment block is emitted."""
    assert supp_bodies[SUPP_FEYNMAN_FILE].startswith("\\setlength")


def test_note_block_is_not_counted_as_a_data_row(supp_bodies: dict[str, str]) -> None:
    assert len(_data_rows(supp_bodies[SUPP_OTHER_FILE])) == 14


def test_long_decimal_validator_rejects_a_float_dump() -> None:
    with pytest.raises(gen.AppendixGenerationError, match="significant figures"):
        gen.validate_no_long_decimals("$x + 0.577215664901533$", "probe.tex")


def test_long_decimal_validator_accepts_benchmark_coefficients() -> None:
    gen.validate_no_long_decimals("$2.1 \\sin(1.3 x_4) + 9.8$", "probe.tex")
    gen.validate_no_long_decimals("$[0.05, 6.05]$", "probe.tex")


# ----------------------------------------------------------------------
# 8. Expression-versus-data agreement audit.
# ----------------------------------------------------------------------

#: Problems whose recorded expression is not the function their data came from.
EXPECTED_APPROXIMATIONS = {"Keijzer-6"}

#: Measured verdict counts. ``exact`` versus ``floating_point`` depends on the
#: NumPy/SymPy evaluation order and is pinned separately from the robust
#: ``69 agree / 1 does not`` split.
EXPECTED_AGREEMENT_COUNTS = {"exact": 36, "floating_point": 33, "approximation": 1}


@pytest.fixture(scope="module")
def agreement(artefacts: dict[str, Any]) -> dict[str, Any]:
    """The persisted ``data_agreement`` block."""
    return artefacts["payload"]["data_agreement"]


def test_agreement_block_covers_every_problem(agreement: dict[str, Any]) -> None:
    assert len(agreement["per_problem"]) == EXPECTED_TOTAL
    assert agreement["summary"]["n_problems"] == EXPECTED_TOTAL


def test_sixty_nine_of_seventy_agree_within_rounding(agreement: dict[str, Any]) -> None:
    """The robust claim: only one recorded expression disagrees with its data."""
    summary = agreement["summary"]
    assert summary["exact"] + summary["floating_point"] == 69
    assert summary["approximation"] == 1


def test_agreement_verdict_counts(agreement: dict[str, Any]) -> None:
    summary = agreement["summary"]
    assert {k: summary[k] for k in EXPECTED_AGREEMENT_COUNTS} == EXPECTED_AGREEMENT_COUNTS


def test_keijzer6_is_the_only_approximation_and_has_an_override(
    agreement: dict[str, Any],
) -> None:
    flagged = {
        r["problem_id"] for r in agreement["per_problem"] if r["verdict"] == "approximation"
    }
    assert flagged == EXPECTED_APPROXIMATIONS
    assert flagged <= set(gen.DISPLAY_OVERRIDES)
    assert agreement["problems_requiring_display_override"] == sorted(EXPECTED_APPROXIMATIONS)


def test_keijzer6_disagreement_is_large(agreement: dict[str, Any]) -> None:
    record = next(
        r for r in agreement["per_problem"] if r["problem_id"] == "Keijzer-6"
    )
    assert record["max_relative_error"] > 0.4
    assert record["n_points"] == 50


def test_nguyen11_is_floating_point_not_exact(agreement: dict[str, Any]) -> None:
    """``x^y`` reassociates between NumPy and lambdify; the error is ~1e-9."""
    record = next(r for r in agreement["per_problem"] if r["problem_id"] == "Nguyen-11")
    assert record["verdict"] == "floating_point"
    assert 0.0 < record["max_relative_error"] <= gen.FLOATING_POINT_TOLERANCE


def test_rounding_errors_are_orders_below_the_tolerance(
    agreement: dict[str, Any],
) -> None:
    """No measurement sits near the exact/approximation boundary."""
    worst = max(
        r["max_relative_error"]
        for r in agreement["per_problem"]
        if r["verdict"] != "approximation"
    )
    assert worst < gen.FLOATING_POINT_TOLERANCE / 100


def test_every_record_compared_at_least_one_point(agreement: dict[str, Any]) -> None:
    for record in agreement["per_problem"]:
        assert record["n_points"] > 0, record["problem_id"]
        assert record["n_points"] <= gen.DATA_AGREEMENT_MAX_POINTS, record["problem_id"]


def test_agreement_protocol_records_seed_and_sample_size(
    agreement: dict[str, Any],
) -> None:
    protocol = agreement["protocol"]
    assert protocol["seed"] == gen.DATA_AGREEMENT_SEED
    assert protocol["max_points_per_problem"] == gen.DATA_AGREEMENT_MAX_POINTS
    assert protocol["floating_point_tolerance"] == gen.FLOATING_POINT_TOLERANCE
    assert set(protocol["verdicts"]) == {"exact", "floating_point", "approximation"}


def test_agreement_is_deterministic() -> None:
    first = gen.audit_data_agreement(CONFIG_DIR)
    second = gen.audit_data_agreement(CONFIG_DIR)
    assert first == second


@pytest.mark.parametrize(
    "error,expected",
    [
        (0.0, "exact"),
        (1e-16, "floating_point"),
        (gen.FLOATING_POINT_TOLERANCE, "floating_point"),
        (gen.FLOATING_POINT_TOLERANCE * 10, "approximation"),
        (0.42, "approximation"),
    ],
)
def test_classify_agreement_boundaries(error: float, expected: str) -> None:
    assert gen.classify_agreement(error) == expected


def test_validator_rejects_undeclared_drift() -> None:
    """A benchmark whose expression drifts from its data must fail the build."""
    drifted = gen.DataAgreement(
        problem_id="Synthetic-Drift",
        tier="nguyen",
        max_relative_error=0.31,
        n_points=128,
        verdict="approximation",
    )
    with pytest.raises(gen.AppendixGenerationError, match="Synthetic-Drift"):
        gen.validate_data_agreement([drifted])


def test_validator_accepts_drift_that_carries_an_override() -> None:
    declared = gen.DataAgreement(
        problem_id="Keijzer-6",
        tier="hard",
        max_relative_error=0.42,
        n_points=50,
        verdict="approximation",
    )
    gen.validate_data_agreement([declared])


def test_validator_accepts_the_real_audit(agreement: dict[str, Any]) -> None:
    records = [gen.DataAgreement(**r) for r in agreement["per_problem"]]
    gen.validate_data_agreement(records)


def test_recorded_expression_ignores_display_overrides() -> None:
    """The audit must see the module's claim, not the corrected printed form."""
    bench = next(b for b in gen.SUITES["hard"] if b["name"] == "Keijzer-6")
    recorded = gen.recorded_sympy_expression(bench)
    assert "log" in str(recorded)
    assert "Sum" not in str(recorded)
