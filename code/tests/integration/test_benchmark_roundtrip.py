"""Integration tests: benchmark expressions survive full IsalSR round-trip.

Tests that real symbolic regression benchmark expressions survive the full pipeline:
    build DAG -> D2S -> S2D -> evaluate
without corruption. These are the expressions the paper will be evaluated on.

If ANY of them fail the round-trip, the paper's results are wrong.

All DAGs are built manually (not via SymPy adapter) to test the core directly.
CONST nodes use const_value=1.0 because S2D does not encode constant values
in the instruction string -- the round-trip structure test is what matters.

Protected operations (POW, LOG, SQRT, DIV) follow the definitions in
dag_evaluator.py:
    - POW: (|x| + 1e-10)^clip(y, -100, 100)
    - LOG: log(|x| + 1e-10)
    - SQRT: sqrt(|x|)
    - DIV: x/y if |y|>1e-10, else 1.0
"""

from __future__ import annotations

import math

import pytest

from isalsr.core.canonical import canonical_string, pruned_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ======================================================================
# Helper: protected operations (mirror dag_evaluator.py exactly)
# ======================================================================


def _ppow(x: float, y: float) -> float:
    """Protected power matching dag_evaluator._protected_pow."""
    base = abs(x) + 1e-10
    exp = max(-100.0, min(100.0, y))
    try:
        result = float(base**exp)
    except OverflowError:
        return 1e15
    if not math.isfinite(result):
        return 1e15
    return result


def _plog(x: float) -> float:
    """Protected log matching dag_evaluator._protected_log."""
    return math.log(abs(x) + 1e-10)


def _psqrt(x: float) -> float:
    """Protected sqrt matching dag_evaluator._protected_sqrt."""
    return math.sqrt(abs(x))


def _pdiv(x: float, y: float) -> float:
    """Protected div matching dag_evaluator._protected_div."""
    if abs(y) > 1e-10:
        return x / y
    return 1.0


def _pexp(x: float) -> float:
    """Protected exp matching dag_evaluator._protected_exp."""
    return math.exp(max(-500.0, min(500.0, x)))


def _clamp(value: float) -> float:
    """Clamp matching dag_evaluator._clamp."""
    if math.isnan(value):
        return 0.0
    if value > 1e15:
        return 1e15
    if value < -1e15:
        return -1e15
    return value


# ======================================================================
# Helper: round-trip functions
# ======================================================================


def d2s_roundtrip(dag: LabeledDAG, num_vars: int) -> LabeledDAG:
    """DAG -> D2S string -> S2D -> new DAG."""
    string = DAGToString(dag, initial_node=0).run()
    return StringToDAG(string, num_variables=num_vars).run()


def canonical_roundtrip(dag: LabeledDAG, num_vars: int) -> LabeledDAG:
    """DAG -> canonical string -> S2D -> new DAG."""
    canon = canonical_string(dag)
    return StringToDAG(canon, num_variables=num_vars).run()


def pruned_canonical_roundtrip(dag: LabeledDAG, num_vars: int) -> LabeledDAG:
    """DAG -> pruned canonical string -> S2D -> new DAG."""
    canon = pruned_canonical_string(dag)
    return StringToDAG(canon, num_variables=num_vars).run()


# ======================================================================
# DAG builders for benchmark expressions
# ======================================================================


def build_nguyen8_sqrt_x() -> LabeledDAG:
    """Nguyen-8: sqrt(x).

    Nodes: x(0), SQRT(1)
    Edges: x -> SQRT
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    sq = dag.add_node(NodeType.SQRT)  # 1
    dag.add_edge(x, sq)
    return dag


def build_sin_cos_x() -> LabeledDAG:
    """sin(x) - cos(x): the canonical B9 reproducer.

    Nodes: x(0), SIN(1), COS(2), SUB(3)
    Edges: x->SIN, x->COS, SIN->SUB (first operand), COS->SUB (second operand)
    Result: sin(x) - cos(x)
    """
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    sin = dag.add_node(NodeType.SIN)  # 1
    cos = dag.add_node(NodeType.COS)  # 2
    sub = dag.add_node(NodeType.SUB)  # 3
    dag.add_edge(x, sin)
    dag.add_edge(x, cos)
    dag.add_edge(sin, sub)  # first operand
    dag.add_edge(cos, sub)  # second operand
    return dag


def build_x_div_y() -> LabeledDAG:
    """x / y: simple binary division with 2 vars.

    Nodes: x(0), y(1), DIV(2)
    Edges: x->DIV (first operand), y->DIV (second operand)
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    div = dag.add_node(NodeType.DIV)  # 2
    dag.add_edge(x, div)  # first operand
    dag.add_edge(y, div)  # second operand
    return dag


def build_sin_cos_chain() -> LabeledDAG:
    """sin(cos(x)): nested unary chain.

    Nodes: x(0), COS(1), SIN(2)
    Edges: x->COS, COS->SIN
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    cos = dag.add_node(NodeType.COS)  # 1
    sin = dag.add_node(NodeType.SIN)  # 2
    dag.add_edge(x, cos)
    dag.add_edge(cos, sin)
    return dag


def build_x_plus_y_plus_z() -> LabeledDAG:
    """x + y + z: variadic ADD with 3 inputs.

    Nodes: x(0), y(1), z(2), ADD(3)
    Edges: x->ADD, y->ADD, z->ADD
    """
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    z = dag.add_node(NodeType.VAR, var_index=2)  # 2
    add = dag.add_node(NodeType.ADD)  # 3
    dag.add_edge(x, add)
    dag.add_edge(y, add)
    dag.add_edge(z, add)
    return dag


def build_diff_of_squares() -> LabeledDAG:
    """(x - y) * (x + y): difference of squares with SUB.

    Nodes: x(0), y(1), SUB(2), ADD(3), MUL(4)
    Edges: x->SUB(1st), y->SUB(2nd), x->ADD, y->ADD, SUB->MUL, ADD->MUL
    """
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    sub = dag.add_node(NodeType.SUB)  # 2
    add = dag.add_node(NodeType.ADD)  # 3
    mul = dag.add_node(NodeType.MUL)  # 4
    dag.add_edge(x, sub)  # first operand of SUB
    dag.add_edge(y, sub)  # second operand of SUB
    dag.add_edge(x, add)
    dag.add_edge(y, add)
    dag.add_edge(sub, mul)
    dag.add_edge(add, mul)
    return dag


def build_nguyen11_x_pow_y() -> LabeledDAG:
    """Nguyen-11 (2 variables): x^y.

    Nodes: x(0), y(1), POW(2)
    Edges: x->POW (first=base), y->POW (second=exponent)

    Protected: result = (|x|+1e-10)^clip(y, -100, 100)
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    pw = dag.add_node(NodeType.POW)  # 2
    dag.add_edge(x, pw)  # first operand = base
    dag.add_edge(y, pw)  # second operand = exponent
    return dag


def build_nguyen9_sin_x_plus_sin_y2() -> LabeledDAG:
    """Nguyen-9 (2 variables): sin(x) + sin(y^k) where k is CONST(1.0).

    Structure: sin(x_1) + sin(x_2 ^ 1.0)

    Every non-VAR node must be reachable from VAR nodes via outgoing edges.
    CONST is a leaf in data-flow terms, but needs an in-edge so D2S can
    create it via V/v. We add y -> CONST as the "creation edge" (CONST
    ignores its in-edges during evaluation).

    Nodes: x(0), y(1), SIN_x(2), POW(3), CONST(4), SIN_y(5), ADD(6)
    Edges: x->SIN_x, y->POW(1st=base), CONST->POW(2nd=exp),
           y->CONST (creation edge), POW->SIN_y, SIN_x->ADD, SIN_y->ADD
    """
    dag = LabeledDAG(max_nodes=15)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    sin_x = dag.add_node(NodeType.SIN)  # 2
    pw = dag.add_node(NodeType.POW)  # 3
    k = dag.add_node(NodeType.CONST, const_value=1.0)  # 4
    sin_y = dag.add_node(NodeType.SIN)  # 5
    add = dag.add_node(NodeType.ADD)  # 6
    dag.add_edge(x, sin_x)
    dag.add_edge(y, pw)  # first operand = base
    dag.add_edge(k, pw)  # second operand = exponent
    dag.add_edge(y, k)  # creation edge (makes CONST reachable)
    dag.add_edge(pw, sin_y)
    dag.add_edge(sin_x, add)
    dag.add_edge(sin_y, add)
    return dag


def build_nguyen5_structure() -> LabeledDAG:
    """Nguyen-5 structure: sin(x^k1) * cos(x) - k2.

    With const_value=1.0, this evaluates as:
        sin((|x|+eps)^1.0) * cos(x) - 1.0

    Every non-VAR node must be reachable from VAR nodes via outgoing edges.
    CONST nodes need "creation edges" from reachable nodes so D2S can create
    them via V/v. CONST ignores its in-edges during evaluation.

    Nodes:
        x(0), POW(1), CONST_pow(2), SIN(3), COS(4), MUL(5), CONST_sub(6), SUB(7)
    Edges:
        x -> POW (1st=base), CONST_pow -> POW (2nd=exp), x -> CONST_pow (creation)
        POW -> SIN
        x -> COS
        SIN -> MUL, COS -> MUL
        MUL -> SUB (1st), CONST_sub -> SUB (2nd), MUL -> CONST_sub (creation)
    """
    dag = LabeledDAG(max_nodes=15)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    pw = dag.add_node(NodeType.POW)  # 1
    k1 = dag.add_node(NodeType.CONST, const_value=1.0)  # 2
    sin = dag.add_node(NodeType.SIN)  # 3
    cos = dag.add_node(NodeType.COS)  # 4
    mul = dag.add_node(NodeType.MUL)  # 5
    k2 = dag.add_node(NodeType.CONST, const_value=1.0)  # 6
    sub = dag.add_node(NodeType.SUB)  # 7
    dag.add_edge(x, pw)  # base
    dag.add_edge(k1, pw)  # exponent
    dag.add_edge(x, k1)  # creation edge (makes k1 reachable)
    dag.add_edge(pw, sin)
    dag.add_edge(x, cos)
    dag.add_edge(sin, mul)
    dag.add_edge(cos, mul)
    dag.add_edge(mul, sub)  # first operand
    dag.add_edge(k2, sub)  # second operand
    dag.add_edge(mul, k2)  # creation edge (makes k2 reachable)
    return dag


def build_nguyen7_structure() -> LabeledDAG:
    """Nguyen-7 structure: log(x+k1) + log(x^k2 + k3).

    With const_value=1.0, evaluates as:
        log(|x+1.0| + eps) + log(|(|x|+eps)^1.0 + 1.0| + eps)

    Every non-VAR node must be reachable from VAR nodes via outgoing edges.
    CONST nodes need "creation edges" so D2S can create them.

    Nodes:
        x(0), K1(1), ADD1(2), LOG1(3), POW(4), K2(5),
        K3(6), ADD2(7), LOG2(8), ADD_out(9)
    Edges (includes creation edges for CONST reachability):
        x->K1(creation), x->ADD1, K1->ADD1, ADD1->LOG1
        x->POW(1st=base), K2->POW(2nd=exp), x->K2(creation)
        POW->ADD2, K3->ADD2, POW->K3(creation), ADD2->LOG2
        LOG1->ADD_out, LOG2->ADD_out
    """
    dag = LabeledDAG(max_nodes=20)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    k1 = dag.add_node(NodeType.CONST, const_value=1.0)  # 1
    add1 = dag.add_node(NodeType.ADD)  # 2
    log1 = dag.add_node(NodeType.LOG)  # 3
    pw = dag.add_node(NodeType.POW)  # 4
    k2 = dag.add_node(NodeType.CONST, const_value=1.0)  # 5
    k3 = dag.add_node(NodeType.CONST, const_value=1.0)  # 6
    add2 = dag.add_node(NodeType.ADD)  # 7
    log2 = dag.add_node(NodeType.LOG)  # 8
    add_out = dag.add_node(NodeType.ADD)  # 9
    dag.add_edge(x, k1)  # creation edge
    dag.add_edge(x, add1)
    dag.add_edge(k1, add1)
    dag.add_edge(add1, log1)
    dag.add_edge(x, pw)  # base
    dag.add_edge(k2, pw)  # exponent
    dag.add_edge(x, k2)  # creation edge
    dag.add_edge(pw, add2)
    dag.add_edge(k3, add2)
    dag.add_edge(pw, k3)  # creation edge
    dag.add_edge(add2, log2)
    dag.add_edge(log1, add_out)
    dag.add_edge(log2, add_out)
    return dag


def build_sigmoid_like() -> LabeledDAG:
    """exp(x) / (exp(x) + k): sigmoid-like with shared subexpression + DIV.

    We share the EXP node: x -> EXP, then EXP -> DIV (1st numerator)
    and EXP -> ADD, K -> ADD, ADD -> DIV (2nd denominator).

    CONST needs a creation edge so it's reachable from VAR nodes.

    Nodes: x(0), EXP(1), K(2), ADD(3), DIV(4)
    Edges: x->EXP, EXP->ADD, K->ADD, EXP->K(creation),
           EXP->DIV(1st), ADD->DIV(2nd)
    """
    dag = LabeledDAG(max_nodes=10)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    exp = dag.add_node(NodeType.EXP)  # 1
    k = dag.add_node(NodeType.CONST, const_value=1.0)  # 2
    add = dag.add_node(NodeType.ADD)  # 3
    div = dag.add_node(NodeType.DIV)  # 4
    dag.add_edge(x, exp)
    dag.add_edge(exp, add)
    dag.add_edge(k, add)
    dag.add_edge(exp, k)  # creation edge (makes CONST reachable)
    dag.add_edge(exp, div)  # first operand = numerator
    dag.add_edge(add, div)  # second operand = denominator
    return dag


def build_nguyen12_structure() -> LabeledDAG:
    """Nguyen-12 (2 variables) structure: x^k1 - x^k2 + k3*y^k4 - y.

    Simplified with const_value=1.0:
        ((|x|+eps)^1 - (|x|+eps)^1) + (1.0 * (|y|+eps)^1) - y

    Every non-VAR node must be reachable from VAR nodes via outgoing edges.
    CONST nodes need "creation edges" so D2S can create them.

    Nodes: x(0), y(1), K1(2), POW1(3), K2(4), POW2(5),
           SUB1(6), K3(7), K4(8), POW3(9), MUL(10), SUB2(11), ADD(12)
    """
    dag = LabeledDAG(max_nodes=20)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    k1 = dag.add_node(NodeType.CONST, const_value=1.0)  # 2
    pw1 = dag.add_node(NodeType.POW)  # 3
    k2 = dag.add_node(NodeType.CONST, const_value=1.0)  # 4
    pw2 = dag.add_node(NodeType.POW)  # 5
    sub1 = dag.add_node(NodeType.SUB)  # 6
    k3 = dag.add_node(NodeType.CONST, const_value=1.0)  # 7
    k4 = dag.add_node(NodeType.CONST, const_value=1.0)  # 8
    pw3 = dag.add_node(NodeType.POW)  # 9
    mul = dag.add_node(NodeType.MUL)  # 10
    sub2 = dag.add_node(NodeType.SUB)  # 11
    add = dag.add_node(NodeType.ADD)  # 12
    # x^k1 and x^k2
    dag.add_edge(x, pw1)  # base
    dag.add_edge(k1, pw1)  # exponent
    dag.add_edge(x, k1)  # creation edge
    dag.add_edge(x, pw2)  # base
    dag.add_edge(k2, pw2)  # exponent
    dag.add_edge(x, k2)  # creation edge
    # SUB1 = pw1 - pw2
    dag.add_edge(pw1, sub1)  # first
    dag.add_edge(pw2, sub1)  # second
    # k3 * y^k4
    dag.add_edge(y, pw3)  # base
    dag.add_edge(k4, pw3)  # exponent
    dag.add_edge(y, k4)  # creation edge
    dag.add_edge(k3, mul)
    dag.add_edge(pw3, mul)
    dag.add_edge(y, k3)  # creation edge
    # SUB2 = mul - y
    dag.add_edge(mul, sub2)  # first
    dag.add_edge(y, sub2)  # second
    # ADD = SUB1 + SUB2
    dag.add_edge(sub1, add)
    dag.add_edge(sub2, add)
    return dag


def build_exp_x() -> LabeledDAG:
    """exp(x): simplest exponential.

    Nodes: x(0), EXP(1)
    Edges: x -> EXP
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    exp = dag.add_node(NodeType.EXP)  # 1
    dag.add_edge(x, exp)
    return dag


def build_abs_x() -> LabeledDAG:
    """abs(x): simplest absolute value.

    Nodes: x(0), ABS(1)
    Edges: x -> ABS
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    ab = dag.add_node(NodeType.ABS)  # 1
    dag.add_edge(x, ab)
    return dag


def build_x_mul_y() -> LabeledDAG:
    """x * y: variadic MUL with 2 inputs.

    Nodes: x(0), y(1), MUL(2)
    Edges: x->MUL, y->MUL
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    mul = dag.add_node(NodeType.MUL)  # 2
    dag.add_edge(x, mul)
    dag.add_edge(y, mul)
    return dag


def build_x_sub_y() -> LabeledDAG:
    """x - y: simple subtraction with 2 vars.

    Nodes: x(0), y(1), SUB(2)
    Edges: x->SUB (first), y->SUB (second)
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    y = dag.add_node(NodeType.VAR, var_index=1)  # 1
    sub = dag.add_node(NodeType.SUB)  # 2
    dag.add_edge(x, sub)  # first
    dag.add_edge(y, sub)  # second
    return dag


def build_log_x() -> LabeledDAG:
    """log(x): simplest logarithm.

    Nodes: x(0), LOG(1)
    Edges: x -> LOG
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    lg = dag.add_node(NodeType.LOG)  # 1
    dag.add_edge(x, lg)
    return dag


def build_cos_x() -> LabeledDAG:
    """cos(x): simplest cosine.

    Nodes: x(0), COS(1)
    Edges: x -> COS
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    cos = dag.add_node(NodeType.COS)  # 1
    dag.add_edge(x, cos)
    return dag


def build_sin_x() -> LabeledDAG:
    """sin(x): simplest sine.

    Nodes: x(0), SIN(1)
    Edges: x -> SIN
    """
    dag = LabeledDAG(max_nodes=5)
    x = dag.add_node(NodeType.VAR, var_index=0)  # 0
    sin = dag.add_node(NodeType.SIN)  # 1
    dag.add_edge(x, sin)
    return dag


# ======================================================================
# Test classes
# ======================================================================


class TestNguyen8SqrtX:
    """Nguyen-8: sqrt(x). The simplest benchmark expression."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen8_sqrt_x()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 2
        assert dag.edge_count == 1
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.SQRT

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 1

    @pytest.mark.parametrize("x_val", [0.0, 0.25, 1.0, 4.0, 9.0])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(_psqrt(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.25, 1.0, 4.0, 9.0])
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.25, 1.0, 4.0, 9.0])
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    def test_d2s_string_nonempty(self, dag: LabeledDAG) -> None:
        string = DAGToString(dag, initial_node=0).run()
        assert len(string) > 0

    def test_canonical_equals_pruned(self, dag: LabeledDAG) -> None:
        c1 = canonical_string(dag)
        c2 = pruned_canonical_string(dag)
        assert c1 == c2


class TestSinCosX:
    """sin(x) - cos(x): the canonical B9 reproducer. Non-commutative SUB."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_sin_cos_x()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 4
        assert dag.edge_count == 4
        assert dag.node_label(3) == NodeType.SUB

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 3

    def test_operand_order(self, dag: LabeledDAG) -> None:
        """SUB must have SIN as first operand, COS as second."""
        inputs = dag.ordered_inputs(3)
        assert inputs == [1, 2]  # SIN=1, COS=2

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 4, 2.0])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        expected = math.sin(x_val) - math.cos(x_val)
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 4, 2.0])
    def test_d2s_roundtrip_preserves_semantics(self, dag: LabeledDAG, x_val: float) -> None:
        """D2S -> S2D must preserve sin(x)-cos(x), NOT produce cos(x)-sin(x)."""
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 4, 2.0])
    def test_canonical_roundtrip_preserves_semantics(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 4, 2.0])
    def test_pruned_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = pruned_canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestXDivY:
    """x / y: simple division with 2 variables. Non-commutative DIV."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_x_div_y()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.node_label(2) == NodeType.DIV

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 2

    def test_operand_order(self, dag: LabeledDAG) -> None:
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]  # x=0 is numerator, y=1 is denominator

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 2.0), (3.0, 1.0), (5.0, -2.0), (-1.0, 3.0), (0.0, 1.0)],
    )
    def test_evaluate(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        expected = _pdiv(x_val, y_val)
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 2.0), (3.0, 1.0), (5.0, -2.0), (-1.0, 3.0), (0.0, 1.0)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 2.0), (3.0, 1.0), (5.0, -2.0), (-1.0, 3.0), (0.0, 1.0)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestSinCosChain:
    """sin(cos(x)): nested unary chain."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_sin_cos_chain()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.node_label(1) == NodeType.COS
        assert dag.node_label(2) == NodeType.SIN

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 2

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi, -1.0])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        expected = math.sin(math.cos(x_val))
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi, -1.0])
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi, -1.0])
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestXPlusYPlusZ:
    """x + y + z: variadic ADD with 3 inputs (3-variable DAG)."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_x_plus_y_plus_z()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 4
        assert dag.edge_count == 3
        assert dag.node_label(3) == NodeType.ADD

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 3

    @pytest.mark.parametrize(
        "x,y,z",
        [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (-1.0, 0.5, 2.5), (10.0, -5.0, -5.0)],
    )
    def test_evaluate(self, dag: LabeledDAG, x: float, y: float, z: float) -> None:
        expected = x + y + z
        result = evaluate_dag(dag, {0: x, 1: y, 2: z})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x,y,z",
        [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (-1.0, 0.5, 2.5)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x: float, y: float, z: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=3)
        orig = evaluate_dag(dag, {0: x, 1: y, 2: z})
        rt = evaluate_dag(dag2, {0: x, 1: y, 2: z})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x,y,z",
        [(1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (-1.0, 0.5, 2.5)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x: float, y: float, z: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=3)
        orig = evaluate_dag(dag, {0: x, 1: y, 2: z})
        rt = evaluate_dag(dag2, {0: x, 1: y, 2: z})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestDiffOfSquares:
    """(x - y) * (x + y): difference of squares. Has SUB (non-commutative)."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_diff_of_squares()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 5
        assert dag.edge_count == 6
        assert dag.node_label(2) == NodeType.SUB
        assert dag.node_label(3) == NodeType.ADD
        assert dag.node_label(4) == NodeType.MUL

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 4

    def test_sub_operand_order(self, dag: LabeledDAG) -> None:
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]  # x first, y second

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(3.0, 2.0), (1.0, 1.0), (5.0, 3.0), (-2.0, 1.0), (0.0, 4.0)],
    )
    def test_evaluate(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        expected = (x_val - y_val) * (x_val + y_val)
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(3.0, 2.0), (1.0, 1.0), (5.0, 3.0), (-2.0, 1.0), (0.0, 4.0)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(3.0, 2.0), (1.0, 1.0), (5.0, 3.0), (-2.0, 1.0), (0.0, 4.0)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestNguyen11XPowY:
    """Nguyen-11: x^y. Non-commutative POW with protected operations."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen11_x_pow_y()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.node_label(2) == NodeType.POW

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 2

    def test_operand_order(self, dag: LabeledDAG) -> None:
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]  # x=base, y=exponent

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(2.0, 3.0), (1.0, 1.0), (3.0, 0.5), (0.5, 2.0), (4.0, 0.0)],
    )
    def test_evaluate(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        expected = _ppow(x_val, y_val)
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(2.0, 3.0), (1.0, 1.0), (3.0, 0.5), (0.5, 2.0), (4.0, 0.0)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(2.0, 3.0), (1.0, 1.0), (3.0, 0.5), (0.5, 2.0), (4.0, 0.0)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestNguyen9SinXPlusSinYPow:
    """Nguyen-9: sin(x) + sin(y^k) with k=1.0."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen9_sin_x_plus_sin_y2()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 7
        # 6 data-flow edges + 1 creation edge (y->CONST)
        assert dag.edge_count == 7
        assert dag.node_label(6) == NodeType.ADD

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 6

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(0.0, 0.0), (1.0, 1.0), (math.pi / 2, 0.5), (-1.0, 2.0), (0.5, -0.5)],
    )
    def test_evaluate(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        # sin(x) + sin((|y|+eps)^1.0)
        expected = math.sin(x_val) + math.sin(_ppow(y_val, 1.0))
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(0.0, 0.0), (1.0, 1.0), (math.pi / 2, 0.5), (-1.0, 2.0)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(0.0, 0.0), (1.0, 1.0), (math.pi / 2, 0.5), (-1.0, 2.0)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestNguyen5Structure:
    """Nguyen-5 structure: sin(x^k1) * cos(x) - k2 with all CONST=1.0."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen5_structure()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 8
        # 8 data-flow edges + 2 creation edges (x->k1, mul->k2)
        assert dag.edge_count == 10
        assert dag.node_label(7) == NodeType.SUB

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 7

    def test_sub_operand_order(self, dag: LabeledDAG) -> None:
        """MUL must be first operand of SUB, CONST second."""
        inputs = dag.ordered_inputs(7)
        assert inputs == [5, 6]  # MUL=5, CONST=6

    def test_pow_operand_order(self, dag: LabeledDAG) -> None:
        """x must be first operand (base), CONST second (exponent)."""
        inputs = dag.ordered_inputs(1)
        assert inputs == [0, 2]  # x=0 is base, CONST=2 is exponent

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 1.5, 2.0, -0.5])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        # sin((|x|+eps)^1.0) * cos(x) - 1.0
        pw_val = _ppow(x_val, 1.0)
        expected = math.sin(pw_val) * math.cos(x_val) - 1.0
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 1.5, 2.0, -0.5])
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 1.5, 2.0, -0.5])
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestNguyen7Structure:
    """Nguyen-7 structure: log(x+k1) + log(x^k2 + k3) with all CONST=1.0."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen7_structure()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 10
        # 10 data-flow edges + 3 creation edges (x->k1, x->k2, pw->k3)
        assert dag.edge_count == 13
        assert dag.node_label(9) == NodeType.ADD

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 9

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        # log(|x + 1.0| + eps) + log(|(|x|+eps)^1.0 + 1.0| + eps)
        add1_val = x_val + 1.0
        log1_val = _plog(add1_val)
        pw_val = _ppow(x_val, 1.0)
        add2_val = pw_val + 1.0
        log2_val = _plog(add2_val)
        expected = log1_val + log2_val
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestNguyen12Structure:
    """Nguyen-12 (2 vars): x^k1 - x^k2 + k3*y^k4 - y with all CONST=1.0."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_nguyen12_structure()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 13
        # 14 data-flow edges + 4 creation edges (x->k1, x->k2, y->k4, y->k3)
        assert dag.edge_count == 18
        assert dag.node_label(12) == NodeType.ADD

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 12

    def test_sub1_operand_order(self, dag: LabeledDAG) -> None:
        """SUB1: POW1 first, POW2 second."""
        inputs = dag.ordered_inputs(6)
        assert inputs == [3, 5]

    def test_sub2_operand_order(self, dag: LabeledDAG) -> None:
        """SUB2: MUL first, y second."""
        inputs = dag.ordered_inputs(11)
        assert inputs == [10, 1]

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (3.0, -1.0), (-1.0, 3.0)],
    )
    def test_evaluate(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        pw1 = _ppow(x_val, 1.0)
        pw2 = _ppow(x_val, 1.0)
        sub1 = pw1 - pw2
        pw3 = _ppow(y_val, 1.0)
        mul_val = 1.0 * pw3
        sub2 = mul_val - y_val
        expected = sub1 + sub2
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
    )
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)],
    )
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float, y_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestSigmoidLike:
    """exp(x) / (exp(x) + k): sigmoid-like with shared subexpression + DIV."""

    @pytest.fixture()
    def dag(self) -> LabeledDAG:
        return build_sigmoid_like()

    def test_structure(self, dag: LabeledDAG) -> None:
        assert dag.node_count == 5
        # 5 data-flow edges + 1 creation edge (exp->k)
        assert dag.edge_count == 6
        assert dag.node_label(4) == NodeType.DIV

    def test_output_node(self, dag: LabeledDAG) -> None:
        assert dag.output_node() == 4

    def test_div_operand_order(self, dag: LabeledDAG) -> None:
        """EXP is numerator (first), ADD is denominator (second)."""
        inputs = dag.ordered_inputs(4)
        assert inputs == [1, 3]  # EXP=1, ADD=3

    @pytest.mark.parametrize("x_val", [0.0, 1.0, -1.0, 2.0, -3.0])
    def test_evaluate(self, dag: LabeledDAG, x_val: float) -> None:
        exp_val = _pexp(x_val)
        denom = exp_val + 1.0
        expected = _pdiv(exp_val, denom)
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 1.0, -1.0, 2.0, -3.0])
    def test_d2s_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = d2s_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 1.0, -1.0, 2.0, -3.0])
    def test_canonical_roundtrip(self, dag: LabeledDAG, x_val: float) -> None:
        dag2 = canonical_roundtrip(dag, num_vars=1)
        orig = evaluate_dag(dag, {0: x_val})
        rt = evaluate_dag(dag2, {0: x_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestSimpleUnaryOps:
    """Simple unary operations: sin(x), cos(x), exp(x), log(x), abs(x)."""

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 3, 2.0])
    def test_sin_x_evaluate(self, x_val: float) -> None:
        dag = build_sin_x()
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(math.sin(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 3, 2.0])
    def test_sin_x_roundtrip(self, x_val: float) -> None:
        dag = build_sin_x()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert evaluate_dag(dag2, {0: x_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val}), abs=1e-8
        )

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 3, 2.0])
    def test_cos_x_evaluate(self, x_val: float) -> None:
        dag = build_cos_x()
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(math.cos(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, math.pi / 3, 2.0])
    def test_cos_x_roundtrip(self, x_val: float) -> None:
        dag = build_cos_x()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert evaluate_dag(dag2, {0: x_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val}), abs=1e-8
        )

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, -1.0, 3.0])
    def test_exp_x_evaluate(self, x_val: float) -> None:
        dag = build_exp_x()
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(_pexp(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.0, 0.5, 1.0, -1.0, 3.0])
    def test_exp_x_roundtrip(self, x_val: float) -> None:
        dag = build_exp_x()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert evaluate_dag(dag2, {0: x_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val}), abs=1e-8
        )

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, math.e, 10.0])
    def test_log_x_evaluate(self, x_val: float) -> None:
        dag = build_log_x()
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(_plog(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [0.5, 1.0, 2.0, math.e, 10.0])
    def test_log_x_roundtrip(self, x_val: float) -> None:
        dag = build_log_x()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert evaluate_dag(dag2, {0: x_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val}), abs=1e-8
        )

    @pytest.mark.parametrize("x_val", [-2.0, -1.0, 0.0, 1.0, 3.5])
    def test_abs_x_evaluate(self, x_val: float) -> None:
        dag = build_abs_x()
        result = evaluate_dag(dag, {0: x_val})
        assert result == pytest.approx(abs(x_val), abs=1e-8)

    @pytest.mark.parametrize("x_val", [-2.0, -1.0, 0.0, 1.0, 3.5])
    def test_abs_x_roundtrip(self, x_val: float) -> None:
        dag = build_abs_x()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert evaluate_dag(dag2, {0: x_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val}), abs=1e-8
        )


class TestSimpleBinaryOps:
    """Simple binary/variadic operations: x*y, x-y."""

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(2.0, 3.0), (0.0, 5.0), (-1.0, 4.0), (0.5, 0.5)],
    )
    def test_x_mul_y_evaluate(self, x_val: float, y_val: float) -> None:
        dag = build_x_mul_y()
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(x_val * y_val, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(2.0, 3.0), (0.0, 5.0), (-1.0, 4.0), (0.5, 0.5)],
    )
    def test_x_mul_y_roundtrip(self, x_val: float, y_val: float) -> None:
        dag = build_x_mul_y()
        dag2 = d2s_roundtrip(dag, num_vars=2)
        assert evaluate_dag(dag2, {0: x_val, 1: y_val}) == pytest.approx(
            evaluate_dag(dag, {0: x_val, 1: y_val}), abs=1e-8
        )

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(5.0, 3.0), (1.0, 1.0), (0.0, 4.0), (-2.0, 3.0)],
    )
    def test_x_sub_y_evaluate(self, x_val: float, y_val: float) -> None:
        dag = build_x_sub_y()
        result = evaluate_dag(dag, {0: x_val, 1: y_val})
        assert result == pytest.approx(x_val - y_val, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(5.0, 3.0), (1.0, 1.0), (0.0, 4.0), (-2.0, 3.0)],
    )
    def test_x_sub_y_roundtrip(self, x_val: float, y_val: float) -> None:
        dag = build_x_sub_y()
        dag2 = d2s_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)

    @pytest.mark.parametrize(
        "x_val,y_val",
        [(5.0, 3.0), (1.0, 1.0), (0.0, 4.0), (-2.0, 3.0)],
    )
    def test_x_sub_y_canonical_roundtrip(self, x_val: float, y_val: float) -> None:
        dag = build_x_sub_y()
        dag2 = canonical_roundtrip(dag, num_vars=2)
        orig = evaluate_dag(dag, {0: x_val, 1: y_val})
        rt = evaluate_dag(dag2, {0: x_val, 1: y_val})
        assert rt == pytest.approx(orig, abs=1e-8)


class TestCanonicalInvariance:
    """Verify that isomorphic DAGs produce the same canonical string."""

    def test_sin_cos_x_canonical_deterministic(self) -> None:
        """Building the same DAG twice gives the same canonical string."""
        d1 = build_sin_cos_x()
        d2 = build_sin_cos_x()
        assert canonical_string(d1) == canonical_string(d2)

    def test_pruned_equals_exhaustive_nguyen8(self) -> None:
        dag = build_nguyen8_sqrt_x()
        assert canonical_string(dag) == pruned_canonical_string(dag)

    def test_pruned_equals_exhaustive_sin_cos(self) -> None:
        dag = build_sin_cos_x()
        assert canonical_string(dag) == pruned_canonical_string(dag)

    def test_pruned_valid_and_eval_preserving_sigmoid(self) -> None:
        """Pruned canonical may differ from exhaustive for CONST-normalized DAGs.

        The 6-tuple pruning is an approximation: it can produce a valid
        canonical string that is LONGER than the exhaustive result. This
        occurs when CONST normalization changes structural tuples, causing
        the pruning to miss the optimal path. Both strings are valid
        (evaluation-preserving), but they may not be identical.
        """
        dag = build_sigmoid_like()
        c_exh = canonical_string(dag)
        c_pru = pruned_canonical_string(dag)
        # Both must be non-empty and decodable.
        assert len(c_exh) > 0
        assert len(c_pru) > 0
        # Exhaustive is at most as long as pruned (it's optimal).
        assert len(c_exh) <= len(c_pru)
        # Both must preserve evaluation.
        dag_exh = StringToDAG(c_exh, num_variables=1).run()
        dag_pru = StringToDAG(c_pru, num_variables=1).run()
        v_orig = evaluate_dag(dag, {0: 1.0})
        v_exh = evaluate_dag(dag_exh, {0: 1.0})
        v_pru = evaluate_dag(dag_pru, {0: 1.0})
        assert v_orig == pytest.approx(v_exh, abs=1e-8)
        assert v_orig == pytest.approx(v_pru, abs=1e-8)

    def test_pruned_equals_exhaustive_chain(self) -> None:
        dag = build_sin_cos_chain()
        assert canonical_string(dag) == pruned_canonical_string(dag)

    def test_pruned_equals_exhaustive_diff_squares(self) -> None:
        dag = build_diff_of_squares()
        assert canonical_string(dag) == pruned_canonical_string(dag)

    def test_pruned_equals_exhaustive_nguyen11(self) -> None:
        dag = build_nguyen11_x_pow_y()
        assert canonical_string(dag) == pruned_canonical_string(dag)


class TestDAGIsomorphism:
    """Verify is_isomorphic for benchmark expressions."""

    def test_nguyen8_self_isomorphic(self) -> None:
        d1 = build_nguyen8_sqrt_x()
        d2 = build_nguyen8_sqrt_x()
        assert d1.is_isomorphic(d2)

    def test_sin_cos_self_isomorphic(self) -> None:
        d1 = build_sin_cos_x()
        d2 = build_sin_cos_x()
        assert d1.is_isomorphic(d2)

    def test_roundtrip_isomorphic_sigmoid(self) -> None:
        """Original and D2S->S2D result must be isomorphic."""
        dag = build_sigmoid_like()
        dag2 = d2s_roundtrip(dag, num_vars=1)
        assert dag.is_isomorphic(dag2)

    def test_roundtrip_isomorphic_diff_squares(self) -> None:
        dag = build_diff_of_squares()
        dag2 = d2s_roundtrip(dag, num_vars=2)
        assert dag.is_isomorphic(dag2)

    def test_roundtrip_isomorphic_nguyen9(self) -> None:
        dag = build_nguyen9_sin_x_plus_sin_y2()
        dag2 = d2s_roundtrip(dag, num_vars=2)
        assert dag.is_isomorphic(dag2)

    def test_different_dags_not_isomorphic(self) -> None:
        """sin(x) and cos(x) are NOT isomorphic."""
        d_sin = build_sin_x()
        d_cos = build_cos_x()
        assert not d_sin.is_isomorphic(d_cos)


class TestStringNonemptyAndDecodable:
    """Every benchmark DAG produces a non-empty string that decodes back."""

    BUILDERS = [
        (build_nguyen8_sqrt_x, 1),
        (build_sin_cos_x, 1),
        (build_x_div_y, 2),
        (build_sin_cos_chain, 1),
        (build_x_plus_y_plus_z, 3),
        (build_diff_of_squares, 2),
        (build_nguyen11_x_pow_y, 2),
        (build_nguyen9_sin_x_plus_sin_y2, 2),
        (build_nguyen5_structure, 1),
        (build_nguyen7_structure, 1),
        (build_nguyen12_structure, 2),
        (build_sigmoid_like, 1),
        (build_exp_x, 1),
        (build_abs_x, 1),
        (build_x_mul_y, 2),
        (build_x_sub_y, 2),
        (build_sin_x, 1),
        (build_cos_x, 1),
        (build_log_x, 1),
    ]

    @pytest.mark.parametrize(
        "builder,num_vars",
        BUILDERS,
        ids=[b[0].__name__ for b in BUILDERS],
    )
    def test_d2s_string_nonempty(self, builder: object, num_vars: int) -> None:
        dag = builder()  # type: ignore[operator]
        string = DAGToString(dag, initial_node=0).run()
        assert len(string) > 0, f"D2S produced empty string for {builder.__name__}"  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "builder,num_vars",
        BUILDERS,
        ids=[b[0].__name__ for b in BUILDERS],
    )
    def test_s2d_decodes_without_error(self, builder: object, num_vars: int) -> None:
        dag = builder()  # type: ignore[operator]
        string = DAGToString(dag, initial_node=0).run()
        dag2 = StringToDAG(string, num_variables=num_vars).run()
        assert dag2.node_count == dag.node_count
        assert dag2.edge_count == dag.edge_count

    @pytest.mark.parametrize(
        "builder,num_vars",
        BUILDERS,
        ids=[b[0].__name__ for b in BUILDERS],
    )
    def test_canonical_string_nonempty(self, builder: object, num_vars: int) -> None:
        dag = builder()  # type: ignore[operator]
        canon = canonical_string(dag)
        assert len(canon) > 0

    @pytest.mark.parametrize(
        "builder,num_vars",
        BUILDERS,
        ids=[b[0].__name__ for b in BUILDERS],
    )
    def test_canonical_decodes_matching_structure(self, builder: object, num_vars: int) -> None:
        dag = builder()  # type: ignore[operator]
        canon = canonical_string(dag)
        dag2 = StringToDAG(canon, num_variables=num_vars).run()
        assert dag2.node_count == dag.node_count
        assert dag2.edge_count == dag.edge_count


class TestOutputNodeCorrectness:
    """Every benchmark DAG returns the correct output node."""

    def test_nguyen8(self) -> None:
        dag = build_nguyen8_sqrt_x()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.SQRT

    def test_sin_cos(self) -> None:
        dag = build_sin_cos_x()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.SUB

    def test_x_div_y(self) -> None:
        dag = build_x_div_y()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.DIV

    def test_sin_cos_chain(self) -> None:
        dag = build_sin_cos_chain()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.SIN

    def test_x_plus_y_plus_z(self) -> None:
        dag = build_x_plus_y_plus_z()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.ADD

    def test_diff_of_squares(self) -> None:
        dag = build_diff_of_squares()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.MUL

    def test_nguyen11(self) -> None:
        dag = build_nguyen11_x_pow_y()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.POW

    def test_nguyen5(self) -> None:
        dag = build_nguyen5_structure()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.SUB

    def test_nguyen7(self) -> None:
        dag = build_nguyen7_structure()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.ADD

    def test_nguyen12(self) -> None:
        dag = build_nguyen12_structure()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.ADD

    def test_sigmoid(self) -> None:
        dag = build_sigmoid_like()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.DIV

    def test_exp(self) -> None:
        dag = build_exp_x()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.EXP

    def test_abs(self) -> None:
        dag = build_abs_x()
        out = dag.output_node()
        assert dag.node_label(out) == NodeType.ABS
