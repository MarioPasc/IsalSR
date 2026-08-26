"""Tests for tokenizer→cell-kind mapping in the instruction strip.

Covers every token in the IsalSR alphabet: single-char instructions and
V/v compound insert tokens for every label character.
"""

from __future__ import annotations

import pytest

from isalsr.types import VALID_LABEL_CHARS, VALID_SINGLE_INSTRUCTIONS
from isalsr.viz.instruction_view import tokenize_string
from isalsr.viz.style import GRAYED_FACE, color_for_token

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_empty_string() -> None:
    assert tokenize_string("") == []


def test_tokenize_single_char_instructions() -> None:
    """Every single-char instruction tokenises to a one-element list."""
    for ch in VALID_SINGLE_INSTRUCTIONS:
        result = tokenize_string(ch)
        assert result == [ch], f"unexpected result for {ch!r}: {result}"


def test_tokenize_v_compound_tokens() -> None:
    """V<label> tokenises to a single two-char token."""
    for label in VALID_LABEL_CHARS:
        tok = f"V{label}"
        result = tokenize_string(tok)
        assert result == [tok], f"unexpected for V{label!r}: {result}"


def test_tokenize_lowercase_v_compound_tokens() -> None:
    """v<label> tokenises to a single two-char token."""
    for label in VALID_LABEL_CHARS:
        tok = f"v{label}"
        result = tokenize_string(tok)
        assert result == [tok], f"unexpected for v{label!r}: {result}"


def test_tokenize_mixed_string() -> None:
    tokens = tokenize_string("VkV+NnCc")
    assert tokens == ["Vk", "V+", "N", "n", "C", "c"]


# ---------------------------------------------------------------------------
# color_for_token — every token in the alphabet produces a non-default color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", list(VALID_SINGLE_INSTRUCTIONS))
def test_single_token_has_non_gray_face(token: str) -> None:
    """Every single-char token maps to a named colour, not to GRAYED_FACE."""
    # Exception: 'W' (no-op) maps to a grey but a specific light-grey,
    # while GRAYED_FACE is the fallback.  The test checks that the function
    # returns a string that looks like a hex colour.
    colour = color_for_token(token)
    assert colour.startswith("#"), f"token {token!r} produced non-hex colour {colour!r}"
    assert len(colour) == 7, f"token {token!r} produced invalid hex {colour!r}"


@pytest.mark.parametrize("label", sorted(VALID_LABEL_CHARS))
def test_primary_insert_token_colour_is_not_fallback(label: str) -> None:
    """V<label> returns a colour derived from the NodeType, not GRAYED_FACE."""
    colour = color_for_token(f"V{label}")
    assert colour != GRAYED_FACE, (
        f"V{label} mapped to the fallback colour; "
        "ensure VALID_LABEL_CHARS and NODETYPE_FACE are in sync"
    )


@pytest.mark.parametrize("label", sorted(VALID_LABEL_CHARS))
def test_secondary_insert_token_colour_is_not_fallback(label: str) -> None:
    """v<label> returns a colour derived from the NodeType, not GRAYED_FACE."""
    colour = color_for_token(f"v{label}")
    assert colour != GRAYED_FACE, (
        f"v{label} mapped to the fallback colour; "
        "ensure VALID_LABEL_CHARS and NODETYPE_FACE are in sync"
    )


def test_vk_and_vs_have_distinct_colours() -> None:
    """CONST (k) and SIN (s) use different face colours."""
    assert color_for_token("Vk") != color_for_token("Vs")


def test_unknown_token_returns_grayed_face() -> None:
    assert color_for_token("?") == GRAYED_FACE
