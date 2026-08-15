"""Regression tests for the Holm-adjusted CPDT p-value policy in the LaTeX tables.

Decision 4 of the C2 fairness audit: with three arms the Cross-Problem Dominance
Test runs three contrasts (isalsr vs baseline, hash vs baseline, isalsr vs hash)
and Holm-corrects across them, so the headline tables must print the corrected
value. The policy the tests pin down:

P1. Table 1 and Table S (the main tables) print ``p_value_holm`` for the primary
    R^2 contrast.
P2. Table 2 (per-problem supplementary detail) keeps printing the RAW
    ``p_value_one_sided``; that is where the uncorrected value now lives.
P3. A legacy two-arm payload carries no ``p_value_holm``. There the Holm family
    has size 1, so the raw one-sided p *is* the corrected value: the main tables
    fall back to it and log a warning. The rendered p cell is exactly what the
    pre-change code emitted (the captions differ by design, so byte-identity is
    asserted on the p cells, not on the whole file).
P4. No generated table ever contains the literal ``nan``, whatever is missing
    from the payload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

# The synthetic two-arm results root is already built by the three-arm test
# module; reuse it rather than duplicating the RunLog fixture.
from test_three_arm_stats import ARMS_2, _build_root

from experiments.figures.models.generate_tables import (
    _cpdt_primary_p,
    _fmt_cpdt_p,
    generate_table1,
    generate_table2,
    generate_table_supplementary,
)

METHOD = "udfs"
BENCHMARK = "benchmark"

# Chosen so the two render differently under _fmt_cpdt_p:
#   raw  -> "$0.001$$^{**}$"   (p < 0.01)
#   holm -> "$0.018$$^{*}$"    (p < 0.05)
RAW_P = 0.001234
HOLM_P = 0.018


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _r2_entry(*, holm: float | None, include_holm: bool = True) -> dict[str, Any]:
    """One ``r2_test`` block of a CPDT payload.

    Args:
        holm: Value for ``p_value_holm``; ignored when ``include_holm`` is False.
        include_holm: Whether to emit the ``p_value_holm`` key at all.

    Returns:
        The metric block.
    """
    entry: dict[str, Any] = {
        "n_problems": 4,
        "n_wins": 4,
        "n_ties": 0,
        "n_losses": 0,
        "cohens_d": 0.303,
        "p_value_one_sided": RAW_P,
        "p_value_two_sided": 2 * RAW_P,
    }
    if include_holm:
        entry["p_value_holm"] = holm
    return entry


def _write_cpdt(root: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as the CPDT artefact the table generator reads."""
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    path = analysis / f"cross_problem_dominance_{METHOD}_{BENCHMARK}.json"
    path.write_text(json.dumps(payload, indent=2))


def _results_root(tmp_path: Path, payload: dict[str, Any] | None) -> Path:
    """Build a minimal two-arm results root carrying ``payload``.

    Args:
        tmp_path: Test-scoped directory.
        payload: CPDT payload to write, or None to omit the artefact entirely.

    Returns:
        The results root.
    """
    root = _build_root(tmp_path / "results", [METHOD], ARMS_2, benchmark=BENCHMARK)
    if payload is not None:
        _write_cpdt(root, payload)
    return root


def _emit_tables(root: Path, out_dir: Path) -> dict[str, str]:
    """Generate the three CPDT-carrying tables and return their sources.

    Args:
        root: Results root.
        out_dir: Directory to write the .tex files into.

    Returns:
        Table key ("t1", "t2", "ts") -> LaTeX source.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_table1(root, [METHOD], [BENCHMARK], out_dir)
    generate_table2(root, [METHOD], [BENCHMARK], out_dir)
    generate_table_supplementary(root, [METHOD], [BENCHMARK], out_dir)
    return {
        "t1": (out_dir / "table1_three_axis_summary.tex").read_text(),
        "t2": (out_dir / f"table2_r2_per_problem_{METHOD}.tex").read_text(),
        "ts": (out_dir / f"table_supplementary_{METHOD}.tex").read_text(),
    }


def _cpdt_rows(tex: str) -> str:
    """The CPDT footer rows of a table, excluding the caption that names them.

    Per-problem rows carry their own Holm-adjusted p and could coincidentally
    render the same string, so footer assertions are scoped to these lines.
    """
    return "\n".join(
        line for line in tex.splitlines() if "CPDT" in line and "\\caption{" not in line
    )


def _assert_no_nan(tex: str) -> None:
    """No typeset cell may read ``nan``.

    The caption is exempt from the substring scan because the word "dominance"
    contains it; captions carry no numbers.
    """
    for line in tex.splitlines():
        if "\\caption{" in line:
            continue
        assert "nan" not in line.lower(), line


# ----------------------------------------------------------------------
# (a) Three-contrast root: the main tables print the Holm-adjusted p
# ----------------------------------------------------------------------


def test_holm_and_raw_render_differently() -> None:
    """The fixture is only informative if the two p-values render apart."""
    assert _fmt_cpdt_p(HOLM_P) != _fmt_cpdt_p(RAW_P)


def test_main_tables_print_the_holm_p(tmp_path: Path) -> None:
    """Table 1 and Table S must show p_value_holm, never the raw one-sided p."""
    root = _results_root(tmp_path, {"r2_test": _r2_entry(holm=HOLM_P)})
    tex = _emit_tables(root, tmp_path / "figs")

    holm_cell, raw_cell = _fmt_cpdt_p(HOLM_P), _fmt_cpdt_p(RAW_P)

    assert holm_cell in tex["t1"]
    assert raw_cell not in tex["t1"]

    ts_footer = _cpdt_rows(tex["ts"])
    assert ts_footer, "the Table S CPDT footer row must be emitted"
    assert holm_cell in ts_footer
    assert raw_cell not in ts_footer


def test_table2_keeps_the_raw_one_sided_p(tmp_path: Path) -> None:
    """The per-problem supplementary table is where the uncorrected p lives."""
    root = _results_root(tmp_path, {"r2_test": _r2_entry(holm=HOLM_P)})
    tex = _emit_tables(root, tmp_path / "figs")

    t2_footer = _cpdt_rows(tex["t2"])
    assert t2_footer, "the Table 2 CPDT footer row must be emitted"
    assert _fmt_cpdt_p(RAW_P) in t2_footer
    assert _fmt_cpdt_p(HOLM_P) not in t2_footer


def test_captions_state_the_correction_policy(tmp_path: Path) -> None:
    """The reader must be told which p each table reports."""
    root = _results_root(tmp_path, {"r2_test": _r2_entry(holm=HOLM_P)})
    tex = _emit_tables(root, tmp_path / "figs")

    assert "Holm-adjusted across the three CPDT contrasts" in tex["t1"]
    assert "Holm-adjusted across the three CPDT contrasts" in tex["ts"]
    assert "raw one-sided value, uncorrected" in tex["t2"]


# ----------------------------------------------------------------------
# (b) Legacy root: fall back to the raw p and say so
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload_entry", "case"),
    [
        (_r2_entry(holm=None, include_holm=False), "key absent"),
        (_r2_entry(holm=None), "key present but null"),
        (_r2_entry(holm=float("nan")), "key present but NaN"),
    ],
)
def test_legacy_payload_prints_the_raw_p(
    tmp_path: Path,
    payload_entry: dict[str, Any],
    case: str,
) -> None:
    """Without a usable Holm p the tables reproduce the pre-change p cells."""
    root = _results_root(tmp_path, {"r2_test": payload_entry})
    tex = _emit_tables(root, tmp_path / "figs")

    raw_cell = _fmt_cpdt_p(RAW_P)
    assert raw_cell in tex["t1"], case
    assert raw_cell in _cpdt_rows(tex["ts"]), case
    assert raw_cell in _cpdt_rows(tex["t2"]), case
    assert _fmt_cpdt_p(HOLM_P) not in tex["t1"], case


def test_legacy_payload_logs_a_warning_naming_table_and_method(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback is announced once per table, naming the table and method."""
    root = _results_root(tmp_path, {"r2_test": _r2_entry(holm=None, include_holm=False)})
    with caplog.at_level(logging.WARNING, logger="experiments.figures.models.generate_tables"):
        _emit_tables(root, tmp_path / "figs")

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    holm_warnings = [m for m in warnings if "Holm-adjusted p" in m]
    assert len(holm_warnings) == 2, holm_warnings
    assert any(m.startswith(f"Table 1 ({METHOD})") for m in holm_warnings)
    assert any(m.startswith(f"Table S ({METHOD})") for m in holm_warnings)
    for message in holm_warnings:
        assert "raw" in message and "one-sided" in message


def test_no_warning_when_the_holm_p_is_present(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A three-contrast payload must not trigger the legacy path."""
    root = _results_root(tmp_path, {"r2_test": _r2_entry(holm=HOLM_P)})
    with caplog.at_level(logging.WARNING, logger="experiments.figures.models.generate_tables"):
        _emit_tables(root, tmp_path / "figs")

    assert not [r for r in caplog.records if "Holm-adjusted p" in r.getMessage()]


# ----------------------------------------------------------------------
# _cpdt_primary_p unit behaviour
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"p_value_holm": 0.018, "p_value_one_sided": RAW_P}, 0.018),
        ({"p_value_holm": 0.0, "p_value_one_sided": RAW_P}, 0.0),
        ({"p_value_holm": 1, "p_value_one_sided": RAW_P}, 1.0),
        ({"p_value_one_sided": RAW_P}, RAW_P),
        ({"p_value_holm": None, "p_value_one_sided": RAW_P}, RAW_P),
        ({"p_value_holm": float("nan"), "p_value_one_sided": RAW_P}, RAW_P),
        ({"p_value_holm": float("inf"), "p_value_one_sided": RAW_P}, RAW_P),
        ({"p_value_holm": True, "p_value_one_sided": RAW_P}, RAW_P),
        ({"p_value_holm": "0.018", "p_value_one_sided": RAW_P}, RAW_P),
    ],
)
def test_cpdt_primary_p_selection(entry: dict[str, Any], expected: float) -> None:
    """Only a finite, numeric, non-boolean Holm p is preferred."""
    got = _cpdt_primary_p(entry, table="Table 1", method=METHOD)
    assert got == pytest.approx(expected)


def test_cpdt_primary_p_is_nan_when_nothing_is_usable() -> None:
    """An empty entry yields NaN, which _fmt_cpdt_p renders as a dagger."""
    got = _cpdt_primary_p({}, table="Table S", method=METHOD)
    assert got != got  # NaN
    assert _fmt_cpdt_p(got) == "$\\dagger$"


# ----------------------------------------------------------------------
# (c) "nan" must never reach LaTeX
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "case"),
    [
        (None, "no CPDT artefact at all"),
        ({}, "empty payload"),
        ({"r2_test": {"error": "not enough problems"}}, "r2_test carries an error"),
        ({"empirical_reduction_factor": {"cohens_d": 0.5}}, "r2_test key missing"),
        (
            {
                "r2_test": _r2_entry(holm=float("nan")),
                "empirical_reduction_factor": {"error": "descriptive"},
            },
            "NaN Holm p and no rho contrast",
        ),
    ],
)
def test_no_table_ever_contains_nan(
    tmp_path: Path,
    payload: dict[str, Any] | None,
    case: str,
) -> None:
    """Whatever the payload is missing, no typeset cell may read ``nan``."""
    root = _results_root(tmp_path, payload)
    for key, tex in _emit_tables(root, tmp_path / "figs").items():
        assert "nan" not in tex.lower().replace("dominance", ""), f"{case} / {key}"
        _assert_no_nan(tex)
