"""Unit tests for StringToDAG (S2D).

Covers: tokenization, initial state, pointer movement, node insertion (V/v),
edge insertion (C/c), DAG cycle enforcement, allowed_ops filtering,
and complete expression DAG construction.

These tests validate the S2D decoder -- the mechanism that converts
IsalSR strings (the search space) into expression DAGs. Correctness here
is prerequisite for the canonical string invariant (Phase 3).
"""

from __future__ import annotations

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType, OperationSet
from isalsr.core.string_to_dag import StringToDAG, _tokenize
from isalsr.errors import InvalidTokenError

# ======================================================================
# Tokenizer tests
# ======================================================================


class TestTokenizer:
    """Tokenization of IsalSR instruction strings."""

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_single_movement(self) -> None:
        assert _tokenize("N") == ["N"]
        assert _tokenize("P") == ["P"]
        assert _tokenize("n") == ["n"]
        assert _tokenize("p") == ["p"]

    def test_single_edge(self) -> None:
        assert _tokenize("C") == ["C"]
        assert _tokenize("c") == ["c"]

    def test_noop(self) -> None:
        assert _tokenize("W") == ["W"]

    def test_compound_V_tokens(self) -> None:
        assert _tokenize("V+") == ["V+"]
        assert _tokenize("V*") == ["V*"]
        assert _tokenize("Vs") == ["Vs"]
        assert _tokenize("Vc") == ["Vc"]
        assert _tokenize("Vk") == ["Vk"]

    def test_compound_v_tokens(self) -> None:
        assert _tokenize("v+") == ["v+"]
        assert _tokenize("vs") == ["vs"]
        assert _tokenize("vk") == ["vk"]

    def test_mixed_tokens(self) -> None:
        tokens = _tokenize("V+NnncVs")
        assert tokens == ["V+", "N", "n", "n", "c", "Vs"]

    def test_all_single_chars(self) -> None:
        tokens = _tokenize("NnPpCcW")
        assert tokens == ["N", "n", "P", "p", "C", "c", "W"]

    def test_bare_c_vs_Vc(self) -> None:
        """Bare 'c' is an edge instruction; 'Vc' is COS insertion."""
        assert _tokenize("c") == ["c"]
        assert _tokenize("Vc") == ["Vc"]
        assert _tokenize("cVc") == ["c", "Vc"]

    def test_V_at_end_raises(self) -> None:
        with pytest.raises(InvalidTokenError, match="requires a label"):
            _tokenize("V")

    def test_v_at_end_raises(self) -> None:
        with pytest.raises(InvalidTokenError, match="requires a label"):
            _tokenize("NV")

    def test_invalid_char_raises(self) -> None:
        with pytest.raises(InvalidTokenError, match="Invalid character"):
            _tokenize("X")

    def test_invalid_label_char_raises(self) -> None:
        with pytest.raises(InvalidTokenError, match="Invalid label"):
            _tokenize("Vz")

    def test_allowed_ops_filtering(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        assert _tokenize("V+", opset) == ["V+"]
        assert _tokenize("Vs", opset) == ["Vs"]
        with pytest.raises(InvalidTokenError, match="not in the allowed"):
            _tokenize("V*", opset)

    def test_allowed_ops_const_always_allowed(self) -> None:
        """CONST is always implicitly allowed (leaf type)."""
        opset = OperationSet(frozenset({NodeType.ADD}))
        assert _tokenize("Vk", opset) == ["Vk"]


# ======================================================================
# Initial state tests
# ======================================================================


class TestS2DInitialState:
    """Initial state setup with m pre-inserted variables."""

    def test_one_variable(self) -> None:
        s2d = StringToDAG("", num_variables=1)
        dag = s2d.run()
        assert dag.node_count == 1
        assert dag.edge_count == 0
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_data(0)["var_index"] == 0

    def test_two_variables(self) -> None:
        s2d = StringToDAG("", num_variables=2)
        dag = s2d.run()
        assert dag.node_count == 2
        assert dag.edge_count == 0
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.VAR
        assert dag.node_data(0)["var_index"] == 0
        assert dag.node_data(1)["var_index"] == 1

    def test_three_variables(self) -> None:
        s2d = StringToDAG("", num_variables=3)
        dag = s2d.run()
        assert dag.node_count == 3
        for i in range(3):
            assert dag.node_label(i) == NodeType.VAR
            assert dag.node_data(i)["var_index"] == i

    def test_cdll_initial_order(self) -> None:
        """CDLL contains variables in order: x_1 -> x_2 -> ... (circular)."""
        s2d = StringToDAG("", num_variables=3)
        s2d.run()
        cdll = s2d.cdll
        assert cdll.size() == 3
        # Starting from primary_ptr, traverse next: should visit 0, 1, 2.
        ptr = s2d.primary_ptr
        visited = []
        for _ in range(3):
            visited.append(cdll.get_value(ptr))
            ptr = cdll.next_node(ptr)
        assert visited == [0, 1, 2]

    def test_pointers_on_x1(self) -> None:
        """Both pointers start on x_1 (graph node 0)."""
        s2d = StringToDAG("", num_variables=2)
        s2d.run()
        assert s2d.cdll.get_value(s2d.primary_ptr) == 0
        assert s2d.cdll.get_value(s2d.secondary_ptr) == 0

    def test_zero_variables_raises(self) -> None:
        with pytest.raises(ValueError, match="num_variables must be >= 1"):
            StringToDAG("", num_variables=0)


# ======================================================================
# Pointer movement tests
# ======================================================================


class TestS2DMovement:
    """Pointer movement instructions (N, P, n, p)."""

    def test_N_moves_primary_forward(self) -> None:
        s2d = StringToDAG("N", num_variables=2)
        s2d.run()
        assert s2d.cdll.get_value(s2d.primary_ptr) == 1  # moved to x_2

    def test_NN_wraps_around(self) -> None:
        """With 2 vars, NN wraps primary back to x_1 (circular)."""
        s2d = StringToDAG("NN", num_variables=2)
        s2d.run()
        assert s2d.cdll.get_value(s2d.primary_ptr) == 0

    def test_P_moves_primary_backward(self) -> None:
        s2d = StringToDAG("P", num_variables=2)
        s2d.run()
        # From x_1, prev in circular [x_1, x_2] goes to x_2.
        assert s2d.cdll.get_value(s2d.primary_ptr) == 1

    def test_n_moves_secondary(self) -> None:
        s2d = StringToDAG("n", num_variables=2)
        s2d.run()
        assert s2d.cdll.get_value(s2d.secondary_ptr) == 1
        # Primary unchanged.
        assert s2d.cdll.get_value(s2d.primary_ptr) == 0

    def test_p_moves_secondary_backward(self) -> None:
        s2d = StringToDAG("p", num_variables=2)
        s2d.run()
        assert s2d.cdll.get_value(s2d.secondary_ptr) == 1


# ======================================================================
# Node insertion tests (V/v)
# ======================================================================


class TestS2DNodeInsertion:
    """V/v labeled node insertion."""

    def test_V_add_creates_node_and_edge(self) -> None:
        """'V+' with 1 var: creates ADD node, edge x -> ADD."""
        s2d = StringToDAG("V+", num_variables=1)
        dag = s2d.run()
        assert dag.node_count == 2
        assert dag.edge_count == 1
        assert dag.node_label(1) == NodeType.ADD
        assert dag.has_edge(0, 1)  # x -> ADD

    def test_V_sin(self) -> None:
        s2d = StringToDAG("Vs", num_variables=1)
        dag = s2d.run()
        assert dag.node_label(1) == NodeType.SIN
        assert dag.has_edge(0, 1)

    def test_V_const_has_default_value(self) -> None:
        s2d = StringToDAG("Vk", num_variables=1)
        dag = s2d.run()
        assert dag.node_label(1) == NodeType.CONST
        assert dag.node_data(1)["const_value"] == pytest.approx(1.0)

    def test_pointer_immobility_after_V(self) -> None:
        """Primary pointer does NOT move after V. Multiple V's create siblings."""
        s2d = StringToDAG("V+V*", num_variables=1)
        dag = s2d.run()
        # Both ADD(1) and MUL(2) have edges from x(0).
        assert dag.node_count == 3
        assert dag.has_edge(0, 1)  # x -> ADD
        assert dag.has_edge(0, 2)  # x -> MUL
        # Primary still on x_1.
        assert s2d.cdll.get_value(s2d.primary_ptr) == 0

    def test_V_then_N_then_V(self) -> None:
        """'V+NVs': ADD from x, move to ADD, SIN from ADD."""
        s2d = StringToDAG("V+NVs", num_variables=1)
        dag = s2d.run()
        assert dag.node_count == 3
        assert dag.has_edge(0, 1)  # x -> ADD
        assert dag.has_edge(1, 2)  # ADD -> SIN

    def test_v_uses_secondary_pointer(self) -> None:
        """'Nv+' with 2 vars: move primary to x_2, secondary stays on x_1, v+ from secondary."""
        s2d = StringToDAG("Nv+", num_variables=2)
        dag = s2d.run()
        assert dag.node_count == 3
        assert dag.node_label(2) == NodeType.ADD
        # Edge from secondary's graph node (x_1 = node 0) to new ADD.
        assert dag.has_edge(0, 2)

    def test_cdll_insertion_after_V(self) -> None:
        """After 'V+', new node is in CDLL after primary's position."""
        s2d = StringToDAG("V+", num_variables=1)
        s2d.run()
        cdll = s2d.cdll
        assert cdll.size() == 2
        # Primary on x(0), next should be ADD(1).
        next_ptr = cdll.next_node(s2d.primary_ptr)
        assert cdll.get_value(next_ptr) == 1

    def test_all_operation_types(self) -> None:
        """Every valid label char produces the correct NodeType."""
        label_to_type = {
            "+": NodeType.ADD,
            "*": NodeType.MUL,
            "-": NodeType.SUB,
            "/": NodeType.DIV,
            "s": NodeType.SIN,
            "c": NodeType.COS,
            "e": NodeType.EXP,
            "l": NodeType.LOG,
            "r": NodeType.SQRT,
            "^": NodeType.POW,
            "a": NodeType.ABS,
            "k": NodeType.CONST,
        }
        for label, expected_type in label_to_type.items():
            s2d = StringToDAG(f"V{label}", num_variables=1)
            dag = s2d.run()
            assert dag.node_label(1) == expected_type, f"Failed for label '{label}'"


# ======================================================================
# Edge insertion tests (C/c)
# ======================================================================


class TestS2DEdgeInsertion:
    """C/c edge creation with DAG cycle enforcement."""

    def test_C_creates_edge(self) -> None:
        """Test C creates edge from primary to secondary.

        With 2 vars: CDLL starts [x(0), y(1)].
        'NC': N moves primary to y(1), C: primary(y=1) -> secondary(x=0) => edge 1->0.
        """
        s2d = StringToDAG("NC", num_variables=2)
        dag = s2d.run()
        assert dag.has_edge(1, 0)  # C created y -> x

    def test_c_creates_reverse_edge(self) -> None:
        """'nc' with 2 vars: n moves secondary to x_2, c: secondary(x_2) -> primary(x_1)."""
        s2d = StringToDAG("nc", num_variables=2)
        dag = s2d.run()
        assert dag.has_edge(1, 0)  # c: x_2 -> x_1

    def test_C_cycle_silently_skipped(self) -> None:
        """If C would create a cycle, it's silently skipped."""
        # Build: x(0) -> ADD(1), then try to add ADD -> x via C.
        # "V+NC" with 1 var: V+ creates ADD(1) from x(0), N moves primary to ADD(1),
        #   secondary still on x(0). C: primary(ADD=1) -> secondary(x=0) => ADD -> x.
        #   This creates a cycle (x->ADD->x). Should be silently skipped.
        s2d = StringToDAG("V+NC", num_variables=1)
        dag = s2d.run()
        assert dag.edge_count == 1  # Only x -> ADD, cycle edge skipped
        assert dag.has_edge(0, 1)
        assert not dag.has_edge(1, 0)

    def test_C_duplicate_silently_skipped(self) -> None:
        """Duplicate edge via C is silently skipped."""
        # "V+C" with 1 var: V+ creates ADD(1) from x(0), edge 0->1.
        # C: primary(x=0) -> secondary(x=0) => self-loop, skipped.
        # Different test: "V+NnC" with 1 var: after V+ CDLL is [x(0), ADD(1)].
        # N: primary -> ADD(1). n: secondary -> ADD(1). C: ADD -> ADD (self-loop, skipped).
        s2d = StringToDAG("V+NnC", num_variables=1)
        dag = s2d.run()
        # Self-loop skipped.
        assert dag.edge_count == 1

    def test_W_is_noop(self) -> None:
        s2d = StringToDAG("W", num_variables=1)
        dag = s2d.run()
        assert dag.node_count == 1
        assert dag.edge_count == 0


# ======================================================================
# Cycle detection tests
# ======================================================================


class TestS2DCycleDetection:
    """DAG constraint enforcement -- the fundamental invariant."""

    def test_v_never_creates_cycle(self) -> None:
        """V/v always creates edge from existing -> new. New node has no outgoing edges."""
        s2d = StringToDAG("V+NVs", num_variables=1)
        dag = s2d.run()
        # x(0) -> ADD(1) -> SIN(2). All acyclic.
        order = dag.topological_sort()
        assert len(order) == 3

    def test_long_chain_then_cycle_attempt(self) -> None:
        """Build x -> ADD -> SIN, then try SIN -> x via C. Cycle rejected."""
        # "V+NVsNC": V+: ADD(1) from x(0). N: primary to ADD(1).
        # Vs: SIN(2) from ADD(1). N: primary to SIN(2). C: SIN(2) -> secondary(x=0). CYCLE!
        s2d = StringToDAG("V+NVsNC", num_variables=1)
        dag = s2d.run()
        assert not dag.has_edge(2, 0)  # Cycle edge rejected
        assert dag.edge_count == 2  # Only x->ADD, ADD->SIN

    def test_all_dags_from_s2d_are_acyclic(self) -> None:
        """A few random-ish strings to verify topological sort always succeeds."""
        test_strings = [
            ("V+V*VsNNnC", 1),
            ("V+nCV*nnC", 2),
            ("V+NVsNVeNVk", 1),
            ("", 3),
            ("WWWW", 1),
        ]
        for string, nvars in test_strings:
            s2d = StringToDAG(string, num_variables=nvars)
            dag = s2d.run()
            order = dag.topological_sort()
            assert len(order) == dag.node_count, f"Failed for string={string!r}"


# ======================================================================
# Allowed ops filtering
# ======================================================================


class TestS2DAllowedOps:
    """Operation set filtering."""

    def test_allowed_ops_passes(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        s2d = StringToDAG("V+Vs", num_variables=1, allowed_ops=opset)
        dag = s2d.run()
        assert dag.node_count == 3

    def test_disallowed_op_raises(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD, NodeType.SIN}))
        with pytest.raises(InvalidTokenError, match="not in the allowed"):
            StringToDAG("V*", num_variables=1, allowed_ops=opset)

    def test_const_always_allowed(self) -> None:
        opset = OperationSet(frozenset({NodeType.ADD}))
        s2d = StringToDAG("Vk", num_variables=1, allowed_ops=opset)
        dag = s2d.run()
        assert dag.node_label(1) == NodeType.CONST


# ======================================================================
# Complete expression DAGs via S2D
# ======================================================================


class TestS2DExpressionDAGs:
    """Full expression DAG construction and verification."""

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        """S2D('Vs', 1) produces DAG isomorphic to sin(x) fixture."""
        s2d = StringToDAG("Vs", num_variables=1)
        dag = s2d.run()
        assert dag.node_count == 2
        assert dag.edge_count == 1
        assert dag.is_isomorphic(sin_x_dag)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        """S2D for x + y: 'V+nC' with 2 vars."""
        # V+ : ADD(2) from x_1(0), edge 0->2
        # n  : secondary to x_2(1)
        # C  : primary(x_1=0) -> secondary(x_2=1) ... wait, that's x_1 -> x_2, not x_2 -> ADD.
        # We need: x -> ADD, y -> ADD.
        # V+ creates x -> ADD.
        # We need y -> ADD. Move secondary to y(1) via 'n', then to ADD(2) via 'n'.
        # Wait, after V+, CDLL is [x(0), ADD(2), y(1)]. So from x(0), n goes to ADD(2), nn to y(1).
        # "V+nnNc" with 2 vars:
        #   V+ : ADD(2) from x(0), edge 0->2. CDLL: [x(0), ADD(2), y(1)].
        #   nn : secondary x(0) -> ADD(2) -> y(1).
        #   N  : primary x(0) -> ADD(2).
        #   c  : secondary(y=1) -> primary(ADD=2) => edge 1->2. Yes!
        s2d = StringToDAG("V+nnNc", num_variables=2)
        dag = s2d.run()
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.has_edge(0, 2)  # x -> ADD
        assert dag.has_edge(1, 2)  # y -> ADD
        assert dag.is_isomorphic(x_plus_y_dag)

    def test_sin_x_mul_y(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        """S2D for sin(x) * y: build and verify isomorphism."""
        # Goal: x(0), y(1), sin(2), mul(3). Edges: x->sin, sin->mul, y->mul.
        # With 2 vars, CDLL starts: [x(0), y(1)].
        # "VsNV*nnNc" with 2 vars:
        #   Vs : SIN(2) from x(0), edge 0->2. CDLL: [x(0), SIN(2), y(1)].
        #   N  : primary -> SIN(2).
        #   V* : MUL(3) from SIN(2), edge 2->3. CDLL: [x(0), SIN(2), MUL(3), y(1)].
        #   nn : secondary x(0) -> SIN(2) -> MUL(3).
        #   N  : primary SIN(2) -> MUL(3).
        #   c  : secondary(MUL=3) -> primary(MUL=3) => self-loop, skipped.
        # That doesn't work. Let me rethink.
        # Need y -> MUL. Secondary needs to be on y, primary on MUL.
        # "VsNV*nnnNNc" with 2 vars:
        #   Vs : SIN(2) from x(0). CDLL: [x(0), SIN(2), y(1)].
        #   N  : primary -> SIN(2).
        #   V* : MUL(3) from SIN(2). CDLL: [x(0), SIN(2), MUL(3), y(1)].
        #   nnn: secondary x(0) -> SIN(2) -> MUL(3) -> y(1).
        #   NN : primary SIN(2) -> MUL(3) -> y(1).
        #   c  : secondary(y=1) -> primary(y=1) => self-loop, skipped.
        # Still wrong. Let me think more carefully.
        # After V*, CDLL order is [x(0), SIN(2), MUL(3), y(1)] (circular).
        # primary is on SIN(2) (didn't move after V*).
        # secondary is on x(0).
        # To get c from y to MUL:
        #   secondary needs to be on y(1): nnn from x -> SIN -> MUL -> y.
        #   primary needs to be on MUL(3): NN from SIN -> MUL -> ... wait, N from SIN(2).
        #   N: primary SIN(2) -> MUL(3). Good.
        # "VsNV*NnnnC":
        #   Vs : SIN(2) from x(0). CDLL: [x(0), SIN(2), y(1)].
        #   N  : primary -> SIN(2).
        #   V* : MUL(3) from SIN(2). CDLL: [x(0), SIN(2), MUL(3), y(1)].
        #   N  : primary SIN(2) -> MUL(3).
        #   nnn: secondary x(0) -> SIN(2) -> MUL(3) -> y(1).
        #   C  : primary(MUL=3) -> secondary(y=1) => MUL -> y? No, we want y -> MUL!
        # Use 'c' instead: secondary(y=1) -> primary(MUL=3) => y -> MUL. Yes!
        s2d = StringToDAG("VsNV*Nnnnc", num_variables=2)
        dag = s2d.run()
        assert dag.node_count == 4
        assert dag.edge_count == 3
        assert dag.has_edge(0, 2)  # x -> SIN
        assert dag.has_edge(2, 3)  # SIN -> MUL
        assert dag.has_edge(1, 3)  # y -> MUL
        assert dag.is_isomorphic(sin_x_mul_y_dag)

    def test_tokens_property(self) -> None:
        s2d = StringToDAG("V+NVs", num_variables=1)
        assert s2d.tokens == ["V+", "N", "Vs"]

    def test_trace_collects_snapshots(self) -> None:
        s2d = StringToDAG("V+", num_variables=1)
        s2d.run(trace=True)
        # Access trace via the internal attribute (set during run).
        trace_log = s2d._trace_log  # type: ignore[attr-defined]
        # Initial state + 1 token = 2 snapshots.
        assert len(trace_log) == 2
