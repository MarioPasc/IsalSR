"""Comprehensive unit tests for fast_canonical_string — the PREFERRED canonical algorithm.

Tests cover all three modes: ``"wl_only"`` (default), ``"wl_tiebreak"``, ``"tuple_only"``.

Key properties verified:
- Completeness: all k! permutations → same canonical string (k=2..8)
- Statistical invariance: 100 random permutations → same canonical (k=10,12,15)
- Round-trip: S2D(fast_canonical(D)) isomorphic to D
- Evaluation preservation: numerical evaluation unchanged after canonicalization
- Discrimination: non-isomorphic DAGs → different canonical strings
- Mode consistency: all modes define the same equivalence classes
"""

from __future__ import annotations

import itertools
import math
import random
import warnings

import pytest

from isalsr.core.canonical import (
    CanonicalMode,
    CanonicalTimeoutError,
    fast_canonical_string,
)
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

ALL_MODES: list[CanonicalMode] = ["wl_only", "wl_tiebreak", "tuple_only"]

# Expressions for completeness tests (string, num_variables).
# Each is parsed via S2D; k is determined at runtime.
COMPLETENESS_EXPRS: list[tuple[str, int]] = [
    ("Vs", 1),  # k=1
    ("V+Vk", 1),  # k=2
    ("V+V*VkNvkPnC", 1),  # k=3-4
    ("VsV+VkNvk", 1),  # k=3-4
    ("V+VkVkPnC", 1),  # k=3
    ("V-VkNV*VkPnC", 1),  # k=3-4
    ("V+VkNV*VkPnCNv+Vk", 1),  # k=5-6
    ("V+V+VkNvkNv*VkPnCNV-Vk", 1),  # k=6-8
]

# Larger expressions for statistical tests.
LARGE_EXPRS: list[tuple[str, int]] = [
    ("V+V*V+VkNvkPnCNV/VkPnCNvsVkPnCNV-Vk", 1),  # k~10
    ("V*V+VkNV-VkNvcVkPnCPnCNV+VsVkPnCPnC", 1),  # k~12
]


def _make_dag(expr: str, num_vars: int) -> LabeledDAG:
    """Parse an IsalSR string into a LabeledDAG."""
    return StringToDAG(expr, num_variables=num_vars).run()


def _internal_k(dag: LabeledDAG) -> int:
    """Number of internal (non-variable) nodes."""
    return dag.node_count - len(dag.var_nodes())


# ======================================================================
# Basics
# ======================================================================


class TestFastCanonicalBasics:
    """Basic functionality of fast_canonical_string."""

    def test_empty_dag(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        assert fast_canonical_string(dag) == ""

    def test_var_only(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        assert fast_canonical_string(dag) == ""

    def test_two_vars_no_edges(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        assert fast_canonical_string(dag) == ""

    def test_sin_x_deterministic(self) -> None:
        """Same DAG always produces same canonical string."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        r1 = fast_canonical_string(dag)
        r2 = fast_canonical_string(dag)
        assert r1 == r2
        assert r1 != ""

    def test_default_mode_is_wl_only(self) -> None:
        """Calling with no mode defaults to wl_only."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        default = fast_canonical_string(dag)
        explicit = fast_canonical_string(dag, mode="wl_only")
        assert default == explicit


# ======================================================================
# Mode tests
# ======================================================================


class TestFastCanonicalModes:
    """All three modes produce valid results."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_mode_returns_nonempty_string(self, mode: CanonicalMode) -> None:
        dag = _make_dag("V+Vk", 1)
        r = fast_canonical_string(dag, mode=mode)
        assert isinstance(r, str)
        assert len(r) > 0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_mode_roundtrip(self, mode: CanonicalMode) -> None:
        """S2D(fast_canonical(D, mode=m)) isomorphic to D."""
        dag = _make_dag("V+V*VkNvkPnC", 1)
        canon = fast_canonical_string(dag, mode=mode)
        dag2 = StringToDAG(canon, num_variables=1).run()
        assert dag.is_isomorphic(dag2)

    def test_deprecated_use_wl_hash_true(self) -> None:
        dag = _make_dag("V+Vk", 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = fast_canonical_string(dag, use_wl_hash=True)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        assert r == fast_canonical_string(dag, mode="wl_tiebreak")

    def test_deprecated_use_wl_hash_false(self) -> None:
        dag = _make_dag("V+Vk", 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = fast_canonical_string(dag, use_wl_hash=False)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        assert r == fast_canonical_string(dag, mode="tuple_only")


# ======================================================================
# Invariance: isomorphic DAGs → same canonical
# ======================================================================


class TestFastCanonicalInvariance:
    """Isomorphic DAGs must produce identical fast_canonical_string."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_sin_cos_add_relabeled(self, mode: CanonicalMode) -> None:
        """sin(x)+cos(x) with nodes in different order → same canonical."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_node(NodeType.COS)
        dag1.add_node(NodeType.ADD)
        dag1.add_edge(0, 1)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 3)
        dag1.add_edge(2, 3)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.COS)
        dag2.add_node(NodeType.SIN)
        dag2.add_node(NodeType.ADD)
        dag2.add_edge(0, 1)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 3)
        dag2.add_edge(2, 3)

        assert fast_canonical_string(dag1, mode=mode) == fast_canonical_string(dag2, mode=mode)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_permuted_sin_x_mul_y(self, mode: CanonicalMode) -> None:
        dag = _make_dag("V*VkNvk", 1)
        k = _internal_k(dag)
        if k < 2:
            pytest.skip("Expression too simple for permutation test")
        perm = list(range(k))
        perm[0], perm[1] = perm[1], perm[0]
        dag_p = permute_internal_nodes(dag, perm)
        assert fast_canonical_string(dag, mode=mode) == fast_canonical_string(dag_p, mode=mode)


# ======================================================================
# Discrimination: non-isomorphic → different canonical
# ======================================================================


class TestFastCanonicalDiscrimination:
    """Different DAGs must produce different fast_canonical_string."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_sin_vs_cos(self, mode: CanonicalMode) -> None:
        dag_sin = LabeledDAG(max_nodes=5)
        dag_sin.add_node(NodeType.VAR, var_index=0)
        dag_sin.add_node(NodeType.SIN)
        dag_sin.add_edge(0, 1)

        dag_cos = LabeledDAG(max_nodes=5)
        dag_cos.add_node(NodeType.VAR, var_index=0)
        dag_cos.add_node(NodeType.COS)
        dag_cos.add_edge(0, 1)

        assert fast_canonical_string(dag_sin, mode=mode) != fast_canonical_string(
            dag_cos, mode=mode
        )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_add_vs_mul(self, mode: CanonicalMode) -> None:
        dag_add = _make_dag("V+Vk", 1)
        dag_mul = _make_dag("V*Vk", 1)
        assert fast_canonical_string(dag_add, mode=mode) != fast_canonical_string(
            dag_mul, mode=mode
        )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_chain_vs_fan(self, mode: CanonicalMode) -> None:
        """x→sin→exp (chain) vs x→sin, x→exp (fan)."""
        dag_chain = LabeledDAG(max_nodes=5)
        dag_chain.add_node(NodeType.VAR, var_index=0)
        dag_chain.add_node(NodeType.SIN)
        dag_chain.add_node(NodeType.EXP)
        dag_chain.add_edge(0, 1)
        dag_chain.add_edge(1, 2)

        dag_fan = LabeledDAG(max_nodes=5)
        dag_fan.add_node(NodeType.VAR, var_index=0)
        dag_fan.add_node(NodeType.SIN)
        dag_fan.add_node(NodeType.EXP)
        dag_fan.add_edge(0, 1)
        dag_fan.add_edge(0, 2)

        assert fast_canonical_string(dag_chain, mode=mode) != fast_canonical_string(
            dag_fan, mode=mode
        )


# ======================================================================
# Completeness: all k! permutations → same canonical
# ======================================================================


class TestFastCanonicalCompleteness:
    """Verify that ALL k! permutations of internal nodes produce the same canonical.

    This is the core completeness property of a labeled-DAG invariant.
    Tested exhaustively for k <= 8 (up to 40,320 permutations).
    """

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("expr,num_vars", COMPLETENESS_EXPRS)
    def test_all_perms_same_canonical(self, expr: str, num_vars: int, mode: CanonicalMode) -> None:
        dag = _make_dag(expr, num_vars)
        k = _internal_k(dag)
        if k < 1:
            pytest.skip("No internal nodes")
        if k > 8:
            pytest.skip(f"k={k} too large for exhaustive permutation test")

        reference = fast_canonical_string(dag, mode=mode, timeout=10.0)

        for perm in itertools.permutations(range(k)):
            dag_p = permute_internal_nodes(dag, list(perm))
            canon_p = fast_canonical_string(dag_p, mode=mode, timeout=10.0)
            assert canon_p == reference, (
                f"Completeness FAIL: mode={mode}, expr={expr}, k={k}, "
                f"perm={perm}: got {canon_p!r}, expected {reference!r}"
            )


# ======================================================================
# Larger DAGs: statistical invariance (k=10-15)
# ======================================================================


class TestFastCanonicalLargerDAGs:
    """Statistical invariance verification on k=10-15.

    Too large for exhaustive k!; test 100 random permutations instead.
    """

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("expr,num_vars", LARGE_EXPRS)
    def test_random_permutations_invariant(
        self, expr: str, num_vars: int, mode: CanonicalMode
    ) -> None:
        dag = _make_dag(expr, num_vars)
        k = _internal_k(dag)
        if k < 3:
            pytest.skip(f"k={k} too small for large-DAG test")

        reference = fast_canonical_string(dag, mode=mode, timeout=30.0)

        rng = random.Random(42)
        n_samples = 100
        for _ in range(n_samples):
            perm = list(range(k))
            rng.shuffle(perm)
            dag_p = permute_internal_nodes(dag, perm)
            canon_p = fast_canonical_string(dag_p, mode=mode, timeout=30.0)
            assert canon_p == reference, (
                f"Statistical invariance FAIL: mode={mode}, k={k}, perm={perm}: got {canon_p!r}"
            )


# ======================================================================
# Evaluation preservation
# ======================================================================


class TestFastCanonicalEvalPreservation:
    """Numerical evaluation is preserved through fast canonicalization."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_sin_x_evaluation(self, mode: CanonicalMode) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)

        val_before = evaluate_dag(dag, {0: math.pi / 2})
        canon = fast_canonical_string(dag, mode=mode)
        dag2 = StringToDAG(canon, num_variables=1).run()
        val_after = evaluate_dag(dag2, {0: math.pi / 2})
        assert val_before == pytest.approx(val_after)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_cos_x_evaluation(self, mode: CanonicalMode) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.COS)
        dag.add_edge(0, 1)
        val_before = evaluate_dag(dag, {0: 1.0})
        canon = fast_canonical_string(dag, mode=mode)
        dag2 = StringToDAG(canon, num_variables=1).run()
        val_after = evaluate_dag(dag2, {0: 1.0})
        assert val_before == pytest.approx(val_after)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_exp_x_evaluation(self, mode: CanonicalMode) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        val_before = evaluate_dag(dag, {0: 0.5})
        canon = fast_canonical_string(dag, mode=mode)
        dag2 = StringToDAG(canon, num_variables=1).run()
        val_after = evaluate_dag(dag2, {0: 0.5})
        assert val_before == pytest.approx(val_after)


# ======================================================================
# Round-trip
# ======================================================================


class TestFastCanonicalRoundTrip:
    """S2D(fast_canonical(D)) isomorphic to D for multiple expressions."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize(
        "expr,num_vars",
        [
            ("Vs", 1),
            ("V+Vk", 1),
            ("V+V*VkNvkPnC", 1),
            ("V+VkNV*VkPnCNv+Vk", 1),
        ],
    )
    def test_roundtrip(self, expr: str, num_vars: int, mode: CanonicalMode) -> None:
        dag = _make_dag(expr, num_vars)
        canon = fast_canonical_string(dag, mode=mode, timeout=10.0)
        dag2 = StringToDAG(canon, num_variables=num_vars).run()
        assert dag.is_isomorphic(dag2), (
            f"Round-trip FAIL: mode={mode}, expr={expr}: "
            f"canon={canon!r}, original has {dag.node_count} nodes, "
            f"reconstructed has {dag2.node_count} nodes"
        )


# ======================================================================
# Mode consistency (equivalence classes agree across modes)
# ======================================================================


class TestFastCanonicalModeConsistency:
    """All modes define the same equivalence classes.

    If mode A says D ~ D', then mode B must also say D ~ D' (and vice versa).
    The canonical strings themselves may differ, but the partition must be identical.
    """

    @pytest.mark.parametrize("expr,num_vars", COMPLETENESS_EXPRS[:5])
    def test_equivalence_classes_agree(self, expr: str, num_vars: int) -> None:
        dag = _make_dag(expr, num_vars)
        k = _internal_k(dag)
        if k < 2 or k > 6:
            pytest.skip(f"k={k} not in test range [2,6]")

        # Collect canonical strings for all permutations under each mode
        canonicals: dict[CanonicalMode, dict[str, set[tuple[int, ...]]]] = {}
        for mode in ALL_MODES:
            groups: dict[str, set[tuple[int, ...]]] = {}
            for perm in itertools.permutations(range(k)):
                dag_p = permute_internal_nodes(dag, list(perm))
                canon = fast_canonical_string(dag_p, mode=mode, timeout=10.0)
                groups.setdefault(canon, set()).add(perm)
            canonicals[mode] = groups

        # All modes should have exactly 1 equivalence class (= completeness)
        for mode in ALL_MODES:
            n_classes = len(canonicals[mode])
            assert n_classes == 1, (
                f"mode={mode} produced {n_classes} equivalence classes "
                f"(expected 1) for expr={expr}, k={k}"
            )


# ======================================================================
# Timeout
# ======================================================================


class TestFastCanonicalTimeout:
    """Timeout handling."""

    def test_timeout_raises_on_tiny_budget(self) -> None:
        dag = _make_dag("V+V+VkNvkNv*VkPnCNV-Vk", 1)
        with pytest.raises(CanonicalTimeoutError):
            fast_canonical_string(dag, timeout=1e-9)

    def test_no_timeout_completes(self) -> None:
        dag = _make_dag("V+VkNV*VkPnCNv+Vk", 1)
        result = fast_canonical_string(dag, timeout=30.0)
        assert isinstance(result, str) and len(result) > 0
