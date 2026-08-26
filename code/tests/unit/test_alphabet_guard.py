"""Tests for the encoding-alphabet containment guard.

The invariant under test is that every operator a host solver can place in a
candidate expression has an image in the twelve labels of Definition 3.2, so
that no candidate is refused by the adapter at run time and evaluated without
deduplication.
"""

from __future__ import annotations

import pytest

from experiments.models.alphabet_guard import (
    SIGMA_SR_LABELS,
    AlphabetCoverageError,
    bingo_operator_image,
    validate_bingo_operators,
    validate_udfs_operators,
)
from isalsr.core.node_types import NodeType

#: The operator set every production Bingo configuration carries.
PRODUCTION_BINGO_OPERATORS = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"]


class TestSigmaSRLabels:
    def test_has_twelve_labels(self) -> None:
        assert len(SIGMA_SR_LABELS) == 12

    def test_excludes_sub_and_div(self) -> None:
        # Definition 3.2 carries no label for subtraction or division: both are
        # encoded through the commutative decomposition.
        assert NodeType.SUB not in SIGMA_SR_LABELS
        assert NodeType.DIV not in SIGMA_SR_LABELS

    def test_excludes_var(self) -> None:
        # Variables are pre-inserted by S2D, not created by an insertion token.
        assert NodeType.VAR not in SIGMA_SR_LABELS


class TestBingoOperatorImage:
    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            ("+", (NodeType.ADD,)),
            ("*", (NodeType.MUL,)),
            ("sin", (NodeType.SIN,)),
            ("cos", (NodeType.COS,)),
            ("exp", (NodeType.EXP,)),
            ("log", (NodeType.LOG,)),
            ("sqrt", (NodeType.SQRT,)),
            ("pow", (NodeType.POW,)),
        ],
    )
    def test_direct_labels(self, operator: str, expected: tuple[NodeType, ...]) -> None:
        assert bingo_operator_image(operator) == expected

    @pytest.mark.parametrize(
        ("operator", "expected"),
        [
            ("-", (NodeType.ADD, NodeType.NEG)),
            ("/", (NodeType.MUL, NodeType.INV)),
        ],
    )
    def test_decomposed_operators_map_to_two_labels(
        self,
        operator: str,
        expected: tuple[NodeType, ...],
    ) -> None:
        assert bingo_operator_image(operator) == expected

    @pytest.mark.parametrize("alias", ["addition", "ADD", " + ", "square root"])
    def test_aliases_and_whitespace_resolve(self, alias: str) -> None:
        assert bingo_operator_image(alias)

    @pytest.mark.parametrize(
        "operator",
        [
            "tanh",
            "tan",
            "sinh",
            "cosh",
            "arcsin",
            "arccos",
            "arctan",
            "square",
            "cube",
            "safe power",
        ],
    )
    def test_operators_outside_the_alphabet_are_refused(self, operator: str) -> None:
        with pytest.raises(AlphabetCoverageError, match="no image in the encoding alphabet"):
            bingo_operator_image(operator)

    def test_unknown_name_is_refused(self) -> None:
        with pytest.raises(AlphabetCoverageError, match="not a Bingo operator"):
            bingo_operator_image("frobnicate")

    def test_every_image_lies_in_the_alphabet(self) -> None:
        for operator in PRODUCTION_BINGO_OPERATORS:
            assert set(bingo_operator_image(operator)) <= SIGMA_SR_LABELS


class TestValidateBingoOperators:
    def test_production_set_passes(self) -> None:
        images = validate_bingo_operators(PRODUCTION_BINGO_OPERATORS)
        assert set(images) == set(PRODUCTION_BINGO_OPERATORS)

    def test_single_offender_is_named(self) -> None:
        with pytest.raises(AlphabetCoverageError, match="tanh"):
            validate_bingo_operators([*PRODUCTION_BINGO_OPERATORS, "tanh"])

    def test_all_offenders_reported_together(self) -> None:
        with pytest.raises(AlphabetCoverageError) as exc:
            validate_bingo_operators(["+", "tanh", "square"])
        assert "tanh" in str(exc.value)
        assert "square" in str(exc.value)

    def test_empty_set_passes(self) -> None:
        assert validate_bingo_operators([]) == {}


class TestBingoConfigEnforcesTheInvariant:
    def test_production_config_constructs(self) -> None:
        from experiments.models.bingo.config import BingoConfig

        assert BingoConfig(operators=list(PRODUCTION_BINGO_OPERATORS)).operators

    def test_uncoverable_operator_fails_at_construction(self) -> None:
        # The point of the guard: this must fail before a 12 h run starts, not
        # degrade deduplication silently once it is under way.
        from experiments.models.bingo.config import BingoConfig

        with pytest.raises(AlphabetCoverageError):
            BingoConfig(operators=["+", "*", "tanh"])

    def test_from_dict_is_guarded_too(self) -> None:
        from experiments.models.bingo.config import BingoConfig

        with pytest.raises(AlphabetCoverageError):
            BingoConfig.from_dict({"operators": ["+", "square"]})


class TestUDFSVendorTable:
    def test_vendor_table_is_fully_encodable(self) -> None:
        images = validate_udfs_operators()
        assert images

    def test_identity_is_absorbed_not_labelled(self) -> None:
        # UDFS emits identity nodes; the adapter contracts them onto their child,
        # so they need no label of their own.
        assert validate_udfs_operators()["="] == ()

    def test_every_image_lies_in_the_alphabet(self) -> None:
        for image in validate_udfs_operators().values():
            assert set(image) <= SIGMA_SR_LABELS

    def test_reversed_operand_orders_share_an_image(self) -> None:
        images = validate_udfs_operators()
        assert images["sub_l"] == images["sub_r"] == (NodeType.ADD, NodeType.NEG)
        assert images["div_l"] == images["div_r"] == (NodeType.MUL, NodeType.INV)


class TestAliasTableMatchesBingo:
    def test_no_drift_from_the_host_table(self) -> None:
        # The guard holds a static copy of Bingo's alias table so that validating
        # a configuration needs no host import. If the dependency changes its
        # table, this test fails rather than the guard silently accepting or
        # rejecting the wrong operators.
        bingo_definitions = pytest.importorskip(
            "bingo.symbolic_regression.agraph.operator_definitions",
        )
        from experiments.models.alphabet_guard import _BINGO_ALIAS_TO_OPCODE

        expected = {
            alias: opcode
            for opcode, aliases in bingo_definitions.OPERATOR_NAMES.items()
            for alias in aliases
        }
        assert expected == _BINGO_ALIAS_TO_OPCODE
