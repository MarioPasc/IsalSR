"""Unit tests for the AI Feynman catalogue and its Sigma_SR representability classifier.

The catalogue is the evidence artefact behind reviewer comment R3.1. These tests pin
both the data (120 rows, provenance, internal consistency) and the classifier verdicts
(116 syntactically representable, 117 semantically representable) so that neither can
drift silently.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
import sympy

from benchmarks.datasets.feynman_catalogue import (
    AIFEYNMAN_120,
    IN_SUITE_IDS,
    SEMANTICALLY_DERIVABLE,
    SIGMA_SR_FUNCTIONS,
    classification_table,
    classify_sigma_sr,
    eligible_extension_pool,
    get_equation,
)

ALL_IDS = [record["id"] for record in AIFEYNMAN_120]


def _parse(record: dict[str, object]) -> sympy.Expr:
    """Sympify a catalogue row with an explicit symbol table.

    Args:
        record: A catalogue record.

    Returns:
        The parsed right-hand side.
    """
    names = [variable["name"] for variable in record["variables"]]  # type: ignore[index,union-attr]
    local_dict: dict[str, object] = {name: sympy.Symbol(name) for name in names}
    local_dict["ln"] = sympy.log
    return sympy.sympify(record["formula"], locals=local_dict)


# --------------------------------------------------------------------------------------
# Catalogue shape and internal consistency
# --------------------------------------------------------------------------------------


def test_catalogue_length_is_120() -> None:
    assert len(AIFEYNMAN_120) == 120


def test_ids_are_unique() -> None:
    assert len(set(ALL_IDS)) == 120


def test_source_split_is_100_base_and_20_bonus() -> None:
    counts = {"base": 0, "bonus": 0}
    for record in AIFEYNMAN_120:
        counts[record["source"]] += 1
    assert counts == {"base": 100, "bonus": 20}


def test_catalogue_is_sorted_by_source_then_id() -> None:
    keys = [(record["source"], record["id"]) for record in AIFEYNMAN_120]
    assert keys == sorted(keys)


def test_record_schema_keys() -> None:
    expected = {
        "id",
        "source",
        "output",
        "formula",
        "num_variables",
        "variables",
        "pmlb_id",
    }
    for record in AIFEYNMAN_120:
        assert set(record) == expected, record["id"]


@pytest.mark.parametrize("record", AIFEYNMAN_120, ids=ALL_IDS)
def test_num_variables_matches_variable_list(record: dict[str, object]) -> None:
    assert record["num_variables"] == len(record["variables"])  # type: ignore[arg-type]
    assert record["num_variables"] >= 1  # type: ignore[operator]


@pytest.mark.parametrize("record", AIFEYNMAN_120, ids=ALL_IDS)
def test_variable_ranges_are_ordered_and_finite(record: dict[str, object]) -> None:
    for variable in record["variables"]:  # type: ignore[union-attr]
        assert isinstance(variable["low"], float)
        assert isinstance(variable["high"], float)
        assert variable["low"] < variable["high"], (record["id"], variable["name"])


@pytest.mark.parametrize("record", AIFEYNMAN_120, ids=ALL_IDS)
def test_every_declared_variable_appears_in_formula(record: dict[str, object]) -> None:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", record["formula"]))  # type: ignore[arg-type]
    for variable in record["variables"]:  # type: ignore[union-attr]
        assert variable["name"] in tokens, (record["id"], variable["name"])


@pytest.mark.parametrize("record", AIFEYNMAN_120, ids=ALL_IDS)
def test_every_formula_sympifies(record: dict[str, object]) -> None:
    expr = _parse(record)
    assert isinstance(expr, sympy.Basic)


@pytest.mark.parametrize("record", AIFEYNMAN_120, ids=ALL_IDS)
def test_free_symbols_equal_declared_variables(record: dict[str, object]) -> None:
    """Catch a variable name being swallowed by sympy's default namespace.

    Without an explicit ``locals`` mapping, ``E`` becomes Euler's number, ``I`` the
    imaginary unit, and ``beta``/``gamma``/``zeta``/``O``/``S``/``N`` become sympy
    special objects, none of which raises. Equality of free symbols with the declared
    names is the assertion that detects it.
    """
    declared = {variable["name"] for variable in record["variables"]}  # type: ignore[union-attr]
    observed = {symbol.name for symbol in _parse(record).free_symbols}
    assert observed == declared, record["id"]


def test_sympify_without_locals_would_swallow_a_variable() -> None:
    """The guard above is not vacuous: bare sympify does lose names in this catalogue."""
    offenders = []
    for record in AIFEYNMAN_120:
        declared = {variable["name"] for variable in record["variables"]}
        try:
            naive = sympy.sympify(record["formula"].replace("ln(", "log("))
        except Exception:  # noqa: BLE001 - a parse failure is also a loss
            offenders.append(record["id"])
            continue
        if {symbol.name for symbol in naive.free_symbols} != declared:
            offenders.append(record["id"])
    assert offenders, "expected at least one row to be corrupted by bare sympify"


# --------------------------------------------------------------------------------------
# Provenance: PMLB reconciliation
# --------------------------------------------------------------------------------------


def test_ii_11_17_is_the_only_row_without_a_pmlb_dataset() -> None:
    missing = [record["id"] for record in AIFEYNMAN_120 if record["pmlb_id"] is None]
    assert missing == ["II.11.17"]


def test_every_other_base_row_has_a_pmlb_id() -> None:
    for record in AIFEYNMAN_120:
        if record["source"] != "base" or record["id"] == "II.11.17":
            continue
        assert isinstance(record["pmlb_id"], str) and record["pmlb_id"]


def test_every_bonus_row_maps_to_a_feynman_test_dataset() -> None:
    bonus = [record for record in AIFEYNMAN_120 if record["source"] == "bonus"]
    assert len(bonus) == 20
    assert {record["pmlb_id"] for record in bonus} == {f"feynman_test_{i}" for i in range(1, 21)}


def test_pmlb_ids_are_unique() -> None:
    pmlb_ids = [r["pmlb_id"] for r in AIFEYNMAN_120 if r["pmlb_id"] is not None]
    assert len(pmlb_ids) == 119
    assert len(set(pmlb_ids)) == 119


@pytest.mark.parametrize(
    ("catalogue_id", "pmlb_id"),
    [
        ("I.6.20a", "feynman_I_6_2a"),
        ("I.6.20", "feynman_I_6_2"),
        ("I.6.20b", "feynman_I_6_2b"),
        ("I.15.10", "feynman_I_15_10"),
        ("I.39.10", "feynman_I_39_1"),
        ("I.48.20", "feynman_I_48_2"),
    ],
)
def test_trailing_zero_id_corrections(catalogue_id: str, pmlb_id: str) -> None:
    """The six identifiers where the distributed CSV truncates a trailing zero."""
    assert get_equation(catalogue_id)["pmlb_id"] == pmlb_id


def test_get_equation_raises_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        get_equation("not-an-equation")


# --------------------------------------------------------------------------------------
# Classifier unit cases with hand-computed answers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "variables", "syntactic", "semantic", "blocking_syn"),
    [
        ("x+y", ["x", "y"], True, True, ()),
        ("x - y", ["x", "y"], True, True, ()),
        ("x/y", ["x", "y"], True, True, ()),
        ("sin(x)/cos(x)", ["x"], True, True, ()),
        ("sqrt(x**2+y**2)", ["x", "y"], True, True, ()),
        ("exp(-x)*log(y)", ["x", "y"], True, True, ()),
        ("x**y", ["x", "y"], True, True, ()),
        ("pi*x", ["x"], True, True, ()),
        ("tan(x)", ["x"], False, True, ("tan",)),
        ("tanh(x)", ["x"], False, True, ("tanh",)),
        ("sinh(x)+cosh(x)", ["x"], False, True, ("cosh", "sinh")),
        ("asin(x)", ["x"], False, False, ("asin",)),
        ("acos(x)", ["x"], False, False, ("acos",)),
        ("atan(x)", ["x"], False, False, ("atan",)),
        ("sign(x)", ["x"], False, False, ("sign",)),
        ("arcsin(x)", ["x"], False, False, ("arcsin",)),
        ("Abs(x)", ["x"], True, True, ()),
        ("ln(x)", ["x"], True, True, ()),
    ],
)
def test_classify_sigma_sr_unit_cases(
    formula: str,
    variables: list[str],
    syntactic: bool,
    semantic: bool,
    blocking_syn: tuple[str, ...],
) -> None:
    verdict = classify_sigma_sr(formula, variables)
    assert verdict.representable_syntactic is syntactic, formula
    assert verdict.representable_semantic is semantic, formula
    assert verdict.blocking_ops_syntactic == blocking_syn, formula


def test_sub_and_div_are_not_misclassified_by_a_pow_rule() -> None:
    """``a-b`` and ``a/b`` become ``Add``/``Mul``/``Pow`` under sympy, all in Sigma_SR."""
    for formula in ("x - y", "x/y", "1/x", "x/(y-z)", "sqrt(x)"):
        verdict = classify_sigma_sr(formula, ["x", "y", "z"])
        assert verdict.representable_syntactic, formula
        assert verdict.functions_used == (), formula


def test_semantic_verdict_is_weaker_than_syntactic() -> None:
    for record in AIFEYNMAN_120:
        names = [variable["name"] for variable in record["variables"]]
        verdict = classify_sigma_sr(record["formula"], names)
        if verdict.representable_syntactic:
            assert verdict.representable_semantic, record["id"]
        assert set(verdict.blocking_ops_semantic) <= set(verdict.blocking_ops_syntactic)


def test_classification_is_frozen() -> None:
    verdict = classify_sigma_sr("x+y", ["x", "y"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.representable_syntactic = False  # type: ignore[misc]


def test_alphabet_sets_are_disjoint() -> None:
    assert not (SIGMA_SR_FUNCTIONS & SEMANTICALLY_DERIVABLE)
    for name in ("asin", "acos", "atan", "atan2", "arcsin", "arccos", "arctan", "sign"):
        assert name not in SIGMA_SR_FUNCTIONS
        assert name not in SEMANTICALLY_DERIVABLE


def test_classify_raises_on_unparseable_formula() -> None:
    with pytest.raises(sympy.SympifyError):
        classify_sigma_sr("x +* y", ["x", "y"])


# --------------------------------------------------------------------------------------
# The headline numbers for reviewer comment R3.1
# --------------------------------------------------------------------------------------


def test_classification_table_counts() -> None:
    table = classification_table()
    assert table["n_total"] == 120
    assert table["n_representable_syntactic"] == 116
    assert table["n_representable_semantic"] == 117


def test_syntactically_blocked_set_is_exactly_four_equations() -> None:
    table = classification_table()
    blocked = {entry["id"]: tuple(entry["blocking_ops"]) for entry in table["blocked_syntactic"]}
    # The bonus arccos equation is resolved from the catalogue, not hard-coded.
    arccos_ids = [
        record["id"]
        for record in AIFEYNMAN_120
        if record["source"] == "bonus" and "arccos" in record["formula"]
    ]
    assert len(arccos_ids) == 1
    assert set(blocked) == {"I.26.2", "I.30.5", "II.35.21", arccos_ids[0]}
    assert blocked["I.26.2"] == ("arcsin",)
    assert blocked["I.30.5"] == ("arcsin",)
    assert blocked["II.35.21"] == ("tanh",)
    assert blocked[arccos_ids[0]] == ("arccos",)


def test_semantically_blocked_set_drops_only_tanh() -> None:
    table = classification_table()
    syntactic = {entry["id"] for entry in table["blocked_syntactic"]}
    semantic = {entry["id"] for entry in table["blocked_semantic"]}
    assert syntactic - semantic == {"II.35.21"}
    assert len(semantic) == 3


def test_by_function_histogram() -> None:
    table = classification_table()
    assert table["by_function"] == {
        "arccos": 1,
        "arcsin": 2,
        "cos": 17,
        "exp": 9,
        "log": 1,
        "sin": 11,
        "tanh": 1,
    }


def test_blocked_entries_carry_their_formula() -> None:
    table = classification_table()
    for entry in table["blocked_syntactic"] + table["blocked_semantic"]:
        assert entry["formula"] == get_equation(entry["id"])["formula"]
        assert entry["blocking_ops"]


# --------------------------------------------------------------------------------------
# Eligible extension pool
# --------------------------------------------------------------------------------------


def test_in_suite_ids_are_all_catalogue_ids() -> None:
    assert set(ALL_IDS) >= IN_SUITE_IDS
    assert len(IN_SUITE_IDS) == 24


def test_eligible_extension_pool_size() -> None:
    assert len(eligible_extension_pool()) == 92


def test_eligible_extension_pool_is_disjoint_from_in_suite() -> None:
    assert not (set(eligible_extension_pool()) & IN_SUITE_IDS)


def test_eligible_extension_pool_members_are_representable() -> None:
    for eq_id in eligible_extension_pool():
        record = get_equation(eq_id)
        names = [variable["name"] for variable in record["variables"]]
        assert classify_sigma_sr(record["formula"], names).representable_syntactic


def test_eligible_extension_pool_is_a_pure_function() -> None:
    assert eligible_extension_pool() == eligible_extension_pool()


def test_eligible_extension_pool_is_sorted_and_unique() -> None:
    pool = eligible_extension_pool()
    assert pool == sorted(pool, key=lambda eq_id: (get_equation(eq_id)["source"], eq_id))
    assert len(set(pool)) == len(pool)


def test_pool_accounting_closes() -> None:
    """120 = 92 eligible + 24 already in suite + 4 blocked, with no overlap."""
    table = classification_table()
    blocked = {entry["id"] for entry in table["blocked_syntactic"]}
    pool = set(eligible_extension_pool())
    assert len(pool) + len(IN_SUITE_IDS) + len(blocked) == 120
    assert not (pool & blocked)
    assert not (IN_SUITE_IDS & blocked)
