"""Tests for the Stage E defects found in the table and figure generators.

Two defects, both invisible in the generators' own exit codes:

1. **E4.** Problem and suite identifiers were typeset raw. Every D2 name carries
   an underscore (``strogatz_vdp1``, ``feynman_remainder``), and a bare ``_``
   outside math mode aborts pdflatex -- so the tables emitted with exit 0 and
   then failed to compile, on exactly the rows the coverage extension added.
2. **E5.** The critical-difference generator iterated a hardcoded two-arm list,
   so a three-arm root produced four-group diagrams with the hash arm absent and
   nothing in the output saying so.
"""

from __future__ import annotations

import pytest

from experiments.figures.models.generate_critical_difference import (
    DEFAULT_CD_VARIANTS,
    _resolve_cd_variants,
    _variant_label,
)
from experiments.figures.models.generate_tables import _latex_escape, _problem_label

# ======================================================================
# E4 -- LaTeX-safe identifiers
# ======================================================================


@pytest.mark.parametrize(
    "raw",
    [
        "strogatz_vdp1",
        "strogatz_bacres1",
        "liv_19",
        "liv_4",
        "pagie_2",
        "feynman_remainder",
    ],
)
def test_d2_identifiers_are_escaped(raw: str) -> None:
    """Every D2 identifier must leave no bare underscore to abort pdflatex."""
    escaped = _latex_escape(raw)
    assert "_" in raw, "fixture must actually contain the hazard"
    assert "\\_" in escaped
    # No underscore survives without its escaping backslash.
    assert all(escaped[i - 1] == "\\" for i, char in enumerate(escaped) if char == "_" and i > 0)


def test_escape_is_idempotent_on_safe_text() -> None:
    """Text with nothing to escape is returned unchanged."""
    assert _latex_escape("Nguyen-1") == "Nguyen-1"


def test_mapped_labels_are_preferred_over_the_raw_name() -> None:
    """D1 problems keep their curated short labels."""
    assert _problem_label("nguyen_1") == "N-1"


def test_unmapped_labels_fall_back_escaped() -> None:
    """The fallback is the raw name made safe, never the raw name itself."""
    assert _problem_label("strogatz_vdp1") == "strogatz\\_vdp1"


def test_unknown_problem_never_emits_a_bare_underscore() -> None:
    """Regression: this is the exact string that aborted the Table S compile."""
    assert "_" not in _problem_label("liv_19").replace("\\_", "")


# ======================================================================
# E5 -- arms are a parameter, not a constant
# ======================================================================


def test_default_is_the_two_arm_campaign() -> None:
    """C1-era invocations must stay byte-identical."""
    assert _resolve_cd_variants(None) == list(DEFAULT_CD_VARIANTS)
    assert list(DEFAULT_CD_VARIANTS) == ["baseline", "isalsr"]


def test_three_arms_are_honoured_in_order() -> None:
    """A three-arm root must produce three arms, in the caller's group order."""
    assert _resolve_cd_variants(["baseline", "hash", "isalsr"]) == [
        "baseline",
        "hash",
        "isalsr",
    ]


def test_duplicate_arms_collapse() -> None:
    """A repeated arm must not double a CD group."""
    assert _resolve_cd_variants(["baseline", "isalsr", "baseline"]) == [
        "baseline",
        "isalsr",
    ]


def test_group_count_is_methods_times_arms() -> None:
    """Two methods and three arms give six groups -- the E5 criterion."""
    methods = ["udfs", "bingo"]
    variants = _resolve_cd_variants(["baseline", "hash", "isalsr"])
    labels = {_variant_label(m, v) for m in methods for v in variants}
    assert len(labels) == 6


def test_hash_arm_has_its_own_label() -> None:
    """The hash arm must be distinguishable, not folded into another arm."""
    assert _variant_label("bingo", "hash") == "BINGO Naive-Hash"
    assert _variant_label("udfs", "baseline") == "UDFS native DAG"
    assert _variant_label("udfs", "isalsr") == "UDFS IsalSR"


def test_unknown_arm_keeps_its_own_name() -> None:
    """An unrecognised arm is labelled by name, never silently relabelled."""
    assert _variant_label("udfs", "gray") == "UDFS gray"
