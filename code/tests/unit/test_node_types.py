"""Unit tests for NodeType enum and registry."""

from __future__ import annotations

from isalsr.core.node_types import (
    ALL_OPS,
    ARITY_MAP,
    BINARY_OPS,
    LABEL_CHAR_MAP,
    LEAF_TYPES,
    NODE_TYPE_TO_LABEL,
    UNARY_OPS,
    VALID_LABEL_CHARS,
    VARIADIC_OPS,
    NodeType,
    OperationSet,
)


class TestNodeTypeEnum:
    """NodeType enum completeness and uniqueness."""

    def test_all_types_exist(self) -> None:
        expected = {
            "VAR",
            "ADD",
            "MUL",
            "SUB",
            "DIV",
            "SIN",
            "COS",
            "EXP",
            "LOG",
            "SQRT",
            "POW",
            "ABS",
            "NEG",
            "INV",
            "CONST",
        }
        actual = {nt.name for nt in NodeType}
        assert actual == expected

    def test_values_are_unique(self) -> None:
        values = [nt.value for nt in NodeType]
        assert len(values) == len(set(values))

    def test_var_is_special(self) -> None:
        """VAR is not in LABEL_CHAR_MAP (pre-inserted, not created by V/v)."""
        assert NodeType.VAR not in LABEL_CHAR_MAP.values() or NodeType.VAR.value == "var"
        assert "var" not in LABEL_CHAR_MAP


class TestLabelCharMap:
    """LABEL_CHAR_MAP completeness and bijectivity."""

    def test_covers_all_non_var_types(self) -> None:
        """Every non-VAR NodeType has a label character."""
        for nt in NodeType:
            if nt == NodeType.VAR:
                continue
            assert nt in LABEL_CHAR_MAP.values(), f"{nt} missing from LABEL_CHAR_MAP"

    def test_bijective(self) -> None:
        """Map is bijective (no two chars map to same type, no type has two chars)."""
        assert len(LABEL_CHAR_MAP) == len(set(LABEL_CHAR_MAP.values()))

    def test_reverse_map_consistent(self) -> None:
        """NODE_TYPE_TO_LABEL is the inverse of LABEL_CHAR_MAP."""
        for char, ntype in LABEL_CHAR_MAP.items():
            assert NODE_TYPE_TO_LABEL[ntype] == char

    def test_all_labels_single_char(self) -> None:
        for char in LABEL_CHAR_MAP:
            assert len(char) == 1


class TestArityMap:
    """ARITY_MAP completeness."""

    def test_covers_all_types(self) -> None:
        for nt in NodeType:
            assert nt in ARITY_MAP, f"{nt} missing from ARITY_MAP"

    def test_var_arity_zero(self) -> None:
        assert ARITY_MAP[NodeType.VAR] == 0

    def test_const_arity_zero(self) -> None:
        assert ARITY_MAP[NodeType.CONST] == 0

    def test_unary_arity_one(self) -> None:
        for nt in UNARY_OPS:
            assert ARITY_MAP[nt] == 1, f"{nt} should have arity 1"

    def test_binary_arity_two(self) -> None:
        for nt in BINARY_OPS:
            assert ARITY_MAP[nt] == 2, f"{nt} should have arity 2"

    def test_variadic_arity_none(self) -> None:
        for nt in VARIADIC_OPS:
            assert ARITY_MAP[nt] is None, f"{nt} should have arity None (variadic)"


class TestCategorySets:
    """Category frozensets are disjoint and complete."""

    def test_disjoint(self) -> None:
        assert frozenset() == UNARY_OPS & BINARY_OPS
        assert frozenset() == UNARY_OPS & VARIADIC_OPS
        assert frozenset() == UNARY_OPS & LEAF_TYPES
        assert frozenset() == BINARY_OPS & VARIADIC_OPS
        assert frozenset() == BINARY_OPS & LEAF_TYPES
        assert frozenset() == VARIADIC_OPS & LEAF_TYPES

    def test_complete(self) -> None:
        """All NodeTypes are covered by exactly one category."""
        all_categorized = UNARY_OPS | BINARY_OPS | VARIADIC_OPS | LEAF_TYPES
        all_types = frozenset(NodeType)
        assert all_categorized == all_types

    def test_all_ops_excludes_leaves(self) -> None:
        assert ALL_OPS == UNARY_OPS | BINARY_OPS | VARIADIC_OPS
        assert frozenset() == ALL_OPS & LEAF_TYPES

    def test_valid_label_chars_matches_map(self) -> None:
        assert frozenset(LABEL_CHAR_MAP.keys()) == VALID_LABEL_CHARS


class TestOperationSet:
    """Configurable operation set."""

    def test_default_includes_all(self) -> None:
        opset = OperationSet()
        for nt in NodeType:
            assert nt in opset

    def test_custom_subset(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        assert NodeType.ADD in opset
        assert NodeType.SIN in opset
        assert NodeType.VAR in opset  # always included
        assert NodeType.CONST in opset  # always included
        assert NodeType.MUL not in opset

    def test_label_chars_filtered(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        assert "+" in opset.label_chars
        assert "s" in opset.label_chars
        assert "k" in opset.label_chars  # CONST always included
        assert "*" not in opset.label_chars

    def test_repr(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        r = repr(opset)
        assert "OperationSet" in r
        assert "ADD" in r
        assert "SIN" in r

    def test_len(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD}))
        # ADD + VAR + CONST = 3
        assert len(opset) == 3

    def test_commutative_factory(self) -> None:
        """OperationSet.commutative() includes NEG/INV, excludes SUB/DIV."""
        from isalsr.core.node_types import COMMUTATIVE_OPS

        opset = OperationSet.commutative()
        assert NodeType.NEG in opset
        assert NodeType.INV in opset
        assert NodeType.ADD in opset
        assert NodeType.MUL in opset
        assert NodeType.SUB not in opset
        assert NodeType.DIV not in opset
        assert NodeType.POW not in opset
        # All commutative ops are present.
        for op in COMMUTATIVE_OPS:
            assert op in opset

    def test_commutative_factory_with_pow(self) -> None:
        opset = OperationSet.commutative(include_pow=True)
        assert NodeType.POW in opset
        assert NodeType.NEG in opset
        assert NodeType.SUB not in opset

    def test_commutative_label_chars(self) -> None:
        opset = OperationSet.commutative()
        assert "g" in opset.label_chars  # NEG
        assert "i" in opset.label_chars  # INV
        assert "-" not in opset.label_chars  # SUB excluded
        assert "/" not in opset.label_chars  # DIV excluded
        assert "^" not in opset.label_chars  # POW excluded


class TestNegInvTypes:
    """NEG and INV are properly registered as unary operations."""

    def test_neg_is_unary(self) -> None:
        assert NodeType.NEG in UNARY_OPS
        assert ARITY_MAP[NodeType.NEG] == 1

    def test_inv_is_unary(self) -> None:
        assert NodeType.INV in UNARY_OPS
        assert ARITY_MAP[NodeType.INV] == 1

    def test_neg_label_char(self) -> None:
        assert LABEL_CHAR_MAP["g"] == NodeType.NEG
        assert NODE_TYPE_TO_LABEL[NodeType.NEG] == "g"

    def test_inv_label_char(self) -> None:
        assert LABEL_CHAR_MAP["i"] == NodeType.INV
        assert NODE_TYPE_TO_LABEL[NodeType.INV] == "i"

    def test_neg_inv_in_all_ops(self) -> None:
        assert NodeType.NEG in ALL_OPS
        assert NodeType.INV in ALL_OPS

    def test_neg_inv_not_binary(self) -> None:
        assert NodeType.NEG not in BINARY_OPS
        assert NodeType.INV not in BINARY_OPS

    def test_neg_inv_not_variadic(self) -> None:
        assert NodeType.NEG not in VARIADIC_OPS
        assert NodeType.INV not in VARIADIC_OPS

    def test_neg_inv_in_valid_labels(self) -> None:
        assert "g" in VALID_LABEL_CHARS
        assert "i" in VALID_LABEL_CHARS
