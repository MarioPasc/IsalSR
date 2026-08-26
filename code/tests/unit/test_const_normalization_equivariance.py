"""Unit tests for the normalize_const_creation (𝒩) equivariance contract.

After 2026-07-29: 𝒩 has been REMOVED from the canonicalisation path.
fast_canonical_string and is_isomorphic no longer call 𝒩 internally.

New contract (two parts):

  1. CANONICALISER: DAGs with orphan-CONST nodes are now *refused loudly*
     (RuntimeError) by fast_canonical_string.  The refusal is equivariant —
     both D and any permutation π(D) raise, so no silent wrong answer is
     possible.

  2. 𝒩 METHOD: normalize_const_creation() still exhibits the non-equivariance
     described in the original counterexample — 𝒩(D) and 𝒩(π(D)) can be
     non-isomorphic.  Tests at the method level preserve this knowledge so
     that 𝒩 is never reintroduced into the canonicalisation path.

The confirmed counterexample (brief, 2026-07-29):

    d  = 3 VARs, 2 CONSTs (c1=3, c2=4), 2 SIN nodes (A=5, B=6)
         edges: c2->A->x0,  c1->B->x1
    d2 = permute_internal_nodes(d, [1,0,2,3])  # swap c1 and c2

After the fix: fast_canonical_string raises on BOTH d and d2 (equivariant).
d.is_isomorphic(d2) now correctly returns True.
𝒩(d).is_isomorphic(𝒩(d2)) still returns False — the method-level defect.

Safe class C = C1 ∪ C2:
  C1: every non-VAR reachable from some VAR (RTF precondition)
  C2: no VAR has any in-edge

The counterexample is outside C: x0 and x1 have in-edges (outside C2), and
c1, c2 are unreachable from VARs (outside C1).
"""

from __future__ import annotations

import pytest

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_counterexample() -> tuple[LabeledDAG, LabeledDAG]:
    """Return (d, d2) from the brief's confirmed counterexample.

    d:  3 VARs (x0=0, x1=1, x2=2), c1=3, c2=4, A=5(SIN), B=6(SIN)
        edges: c2->A, A->x0,  c1->B, B->x1
    d2: permute_internal_nodes(d, [1, 0, 2, 3])  -- swaps c1 and c2
    """
    d = LabeledDAG(16)
    for i in range(3):
        d.add_node(NodeType.VAR, var_index=i)  # 0, 1, 2
    c1 = d.add_node(NodeType.CONST, const_value=1.0)  # 3
    c2 = d.add_node(NodeType.CONST, const_value=1.0)  # 4
    a = d.add_node(NodeType.SIN)  # 5
    b = d.add_node(NodeType.SIN)  # 6
    d.add_edge(c2, a)
    d.add_edge(a, 0)  # c2 -> A -> x0
    d.add_edge(c1, b)
    d.add_edge(b, 1)  # c1 -> B -> x1
    d2 = permute_internal_nodes(d, [1, 0, 2, 3])
    return d, d2


def _in_c1(dag: LabeledDAG) -> bool:
    """Every non-VAR node reachable from some VAR."""
    from collections import deque

    queue: deque[int] = deque()
    seen: set[int] = set()
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.VAR:
            seen.add(i)
            queue.append(i)
    while queue:
        node = queue.popleft()
        for succ in dag.out_neighbors(node):
            if succ not in seen:
                seen.add(succ)
                queue.append(succ)
    return all(i in seen for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)


def _in_c2(dag: LabeledDAG) -> bool:
    """No VAR has any in-edge."""
    return all(
        dag.in_degree(i) == 0 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR
    )


# ---------------------------------------------------------------------------
# Counterexample: structure
# ---------------------------------------------------------------------------


class TestCounterexampleStructure:
    """Verify structural properties of the confirmed counterexample DAG."""

    def test_node_labels(self) -> None:
        d, _ = _build_counterexample()
        assert d.node_label(0) == NodeType.VAR
        assert d.node_label(1) == NodeType.VAR
        assert d.node_label(2) == NodeType.VAR
        assert d.node_label(3) == NodeType.CONST
        assert d.node_label(4) == NodeType.CONST
        assert d.node_label(5) == NodeType.SIN
        assert d.node_label(6) == NodeType.SIN

    def test_edges_in_d(self) -> None:
        d, _ = _build_counterexample()
        # c2(4) -> A(5) -> x0(0)
        assert 4 in list(d.in_neighbors(5))
        assert 5 in list(d.in_neighbors(0))
        # c1(3) -> B(6) -> x1(1)
        assert 3 in list(d.in_neighbors(6))
        assert 6 in list(d.in_neighbors(1))

    def test_orphan_consts_in_d(self) -> None:
        d, _ = _build_counterexample()
        assert d.in_degree(3) == 0, "c1 must be orphan CONST in d"
        assert d.in_degree(4) == 0, "c2 must be orphan CONST in d"

    def test_vars_with_in_edges_in_d(self) -> None:
        d, _ = _build_counterexample()
        assert d.in_degree(0) > 0, "x0 must have in-edge in d"
        assert d.in_degree(1) > 0, "x1 must have in-edge in d"

    def test_d2_is_permutation_of_d(self) -> None:
        d, d2 = _build_counterexample()
        assert d2.node_count == d.node_count
        # In d2, nodes 3 and 4 have swapped labels.
        assert d2.node_label(3) == NodeType.CONST
        assert d2.node_label(4) == NodeType.CONST

    def test_counterexample_outside_c1(self) -> None:
        d, d2 = _build_counterexample()
        assert not _in_c1(d), "d must be outside C1"
        assert not _in_c1(d2), "d2 must be outside C1"

    def test_counterexample_outside_c2(self) -> None:
        d, d2 = _build_counterexample()
        assert not _in_c2(d), "d must be outside C2 (x0 and x1 have in-edges)"
        assert not _in_c2(d2), "d2 must be outside C2"

    def test_counterexample_outside_safe_class(self) -> None:
        d, d2 = _build_counterexample()
        in_safe_d = _in_c1(d) or _in_c2(d)
        in_safe_d2 = _in_c1(d2) or _in_c2(d2)
        assert not in_safe_d, "d must be outside safe class C"
        assert not in_safe_d2, "d2 must be outside safe class C"


# ---------------------------------------------------------------------------
# Counterexample: the known failure
# ---------------------------------------------------------------------------


class TestKnownEquivarianceFailure:
    """Pin the confirmed equivariance failure from the brief."""

    def test_canonical_strings_differ(self) -> None:
        # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
        # Orphan-CONST DAGs are now refused loudly (RuntimeError) by both arms.
        # Equivariance is restored: both d and d2 raise consistently.
        """fast_canonical_string raises on both d and d2 — equivariant refusal."""
        d, d2 = _build_counterexample()
        for dag, _label in [(d, "d"), (d2, "d2")]:
            with pytest.raises(RuntimeError, match="no valid operation found"):
                fast_canonical_string(dag, timeout=5.0)

    def test_canonical_strings_match_brief(self) -> None:
        # CONTRACT CHANGED 2026-07-29: The exact strings "VkpvknvsncNVsNpppC"
        # and "pvkpvknvsncPVsNppC" were produced by the now-removed 𝒩.
        # The new contract: canonicaliser refuses both DAGs with RuntimeError.
        # The strings are now only obtainable via fast_canonical_string(𝒩(d)).
        """fast_canonical_string raises on d and d2; old pinned strings from 𝒩 path."""
        d, d2 = _build_counterexample()
        with pytest.raises(RuntimeError):
            fast_canonical_string(d, timeout=5.0)
        with pytest.raises(RuntimeError):
            fast_canonical_string(d2, timeout=5.0)
        # The old pinned strings are still accessible at the 𝒩 method level:
        assert (
            fast_canonical_string(d.normalize_const_creation(), timeout=5.0) == "VkpvknvsncNVsNpppC"
        )
        assert (
            fast_canonical_string(d2.normalize_const_creation(), timeout=5.0)
            == "pvkpvknvsncPVsNppC"
        )

    def test_is_isomorphic_now_correct(self) -> None:
        # CONTRACT CHANGED 2026-07-29: is_isomorphic no longer applies 𝒩.
        # Previously returned False (wrong — d and d2 ARE isomorphic by construction).
        # Now correctly returns True.  The old False was the same root cause as
        # the canonical failure: 𝒩 inside is_isomorphic produced non-isomorphic
        # normalised DAGs, making the comparison conclude False.
        """d.is_isomorphic(d2) now correctly returns True (isomorphic by construction)."""
        d, d2 = _build_counterexample()
        assert d.is_isomorphic(d2), (
            "is_isomorphic(d, d2) must be True: d2 = permute_internal_nodes(d, π)"
        )


# ---------------------------------------------------------------------------
# Safe class: equivariance holds on C1 and C2
# ---------------------------------------------------------------------------


class TestSafeClassEquivariance:
    """Confirm equivariance holds on the safe class for hand-crafted examples."""

    def _check_equivariant(self, dag: LabeledDAG, perms: list[list[int]]) -> None:
        ref = fast_canonical_string(dag)
        for perm in perms:
            dag_pi = permute_internal_nodes(dag, perm)
            cs_pi = fast_canonical_string(dag_pi)
            assert cs_pi == ref, (
                f"Equivariance failure on safe DAG: perm={perm}, ref={ref!r}, got={cs_pi!r}"
            )

    def test_c1_dag_single_const_with_creation_edge(self) -> None:
        """DAG in C1: CONST already has an in-edge from a VAR."""
        dag = LabeledDAG(10)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        c = dag.add_node(NodeType.CONST, const_value=2.0)  # 2
        a = dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(0, c)  # x0 -> c  (in-edge for CONST: satisfies C1)
        dag.add_edge(c, a)  # c -> ADD
        dag.add_edge(1, a)  # x1 -> ADD
        assert _in_c1(dag), "DAG must be in C1"
        self._check_equivariant(dag, [[0, 1], [1, 0]])

    def test_c2_dag_vars_are_pure_sources(self) -> None:
        # CONTRACT CHANGED 2026-07-29: fast_canonical_string no longer applies 𝒩.
        # Orphan-CONST DAGs now raise at the canonicaliser level.
        # Equivariance on safe class C2 is now verified at the 𝒩 method level:
        # normalize_const_creation() must produce isomorphic DAGs for all
        # permutations, and their canonical strings (after repair) must agree.
        """𝒩 is equivariant on C2: normalized DAGs yield the same canonical string."""
        dag = LabeledDAG(16)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        c0 = dag.add_node(NodeType.CONST, const_value=1.0)  # 2
        c1 = dag.add_node(NodeType.CONST, const_value=1.0)  # 3
        add = dag.add_node(NodeType.ADD)  # 4
        dag.add_edge(c0, add)
        dag.add_edge(c1, add)
        dag.add_edge(0, add)
        # VARs 0 and 1 have in_degree 0 (var 1 is disconnected here, but still 0 in-deg)
        assert _in_c2(dag), "DAG must be in C2"
        # After repair by 𝒩, all permutations produce the same canonical string.
        ref = fast_canonical_string(dag.normalize_const_creation(), timeout=5.0)
        for perm in [[0, 1, 2], [1, 0, 2], [0, 2, 1]]:
            dag_pi = permute_internal_nodes(dag, perm)
            cs_pi = fast_canonical_string(dag_pi.normalize_const_creation(), timeout=5.0)
            assert cs_pi == ref, (
                f"𝒩 equivariance failure on C2 DAG: perm={perm}, ref={ref!r}, got={cs_pi!r}"
            )

    def test_c2_dag_multiple_consts_all_anchor_to_x0(self) -> None:
        # CONTRACT CHANGED 2026-07-29: fast_canonical_string no longer applies 𝒩.
        # Orphan-CONST DAGs raise at the canonicaliser. Knowledge preserved at the
        # 𝒩 method level: on C2 (no VAR has in-edges), every orphan CONST is
        # anchored to x_0 by normalize_const_creation, regardless of permutation.
        """𝒩 anchors every orphan CONST to x_0 on C2; equivariant across permutations."""
        dag = LabeledDAG(16)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.VAR, var_index=2)  # 2
        # 3 orphan CONSTs, no edges to VARs at all.
        dag.add_node(NodeType.CONST, const_value=1.0)  # 3
        dag.add_node(NodeType.CONST, const_value=1.0)  # 4
        dag.add_node(NodeType.CONST, const_value=1.0)  # 5
        assert _in_c2(dag), "All VARs must have in_degree 0"
        # 𝒩 anchors all orphan CONSTs to x_0 (no path from any CONST to x_0).
        normalized = dag.normalize_const_creation()
        for const_idx in [3, 4, 5]:
            assert normalized.has_edge(0, const_idx), (
                f"𝒩 must anchor CONST {const_idx} to x_0 on C2"
            )
        # Equivariance: permutations of the 3 internal CONSTs all anchor to x_0.
        ref = fast_canonical_string(normalized, timeout=5.0)
        for perm in [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1]]:
            dag_pi = permute_internal_nodes(dag, perm)
            cs_pi = fast_canonical_string(dag_pi.normalize_const_creation(), timeout=5.0)
            assert cs_pi == ref, f"𝒩 equivariance failure on C2 multi-CONST: perm={perm}"

    def test_no_const_nodes_trivially_equivariant(self) -> None:
        """DAG with no CONST nodes: normalize is no-op; equivariance trivial."""
        dag = LabeledDAG(10)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        sin_node = dag.add_node(NodeType.SIN)  # 2
        dag.add_edge(0, sin_node)
        add_node = dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(sin_node, add_node)
        dag.add_edge(1, add_node)
        assert not dag._has_const_nodes()
        self._check_equivariant(dag, [[0, 1], [1, 0]])


# ---------------------------------------------------------------------------
# Parametric: multiple adversarial instances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_vars,op_pair,var_targets",
    [
        # Targets must be the two lowest-indexed VARs (x0, x1) to trigger
        # anchor-path interference.  When chains point to non-adjacent VARs
        # (e.g. [0,2], [1,3]) the normalization steps do not interfere and no
        # failure arises — those configurations are tested separately in
        # TestNoFailureOnNonAdjacentTargets.
        (3, (NodeType.SIN, NodeType.SIN), [0, 1]),
        (3, (NodeType.COS, NodeType.SIN), [0, 1]),
        (4, (NodeType.SIN, NodeType.COS), [0, 1]),
        (4, (NodeType.EXP, NodeType.NEG), [0, 1]),
        (5, (NodeType.SIN, NodeType.SIN), [0, 1]),
    ],
)
def test_adversarial_equivariance_fails(
    n_vars: int,
    op_pair: tuple[NodeType, NodeType],
    var_targets: list[int],
) -> None:
    # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
    # Canonicaliser now raises on both dag and dag_swapped (equivariant refusal).
    # Knowledge preserved at the 𝒩 method level: normalize_const_creation()
    # is NOT isomorphism-equivariant for adjacent-target configurations —
    # 𝒩(D) and 𝒩(π(D)) are non-isomorphic.  This is why 𝒩 must never be
    # reintroduced into the canonicalisation path.
    """𝒩 is non-equivariant on adjacent-target fixtures; canonicaliser refuses both.

    Args:
        n_vars: Number of VAR nodes.
        op_pair: Two unary operators, one per CONST chain.
        var_targets: Indices of VARs to connect each chain to.
    """
    dag = LabeledDAG(n_vars + 10)
    for i in range(n_vars):
        dag.add_node(NodeType.VAR, var_index=i)
    c0 = dag.add_node(NodeType.CONST, const_value=1.0)
    c1 = dag.add_node(NodeType.CONST, const_value=1.0)
    op0 = dag.add_node(op_pair[0])
    op1 = dag.add_node(op_pair[1])
    dag.add_edge(c0, op0)
    dag.add_edge(op0, var_targets[0])
    dag.add_edge(c1, op1)
    dag.add_edge(op1, var_targets[1])

    assert not _in_c1(dag), "adversarial must be outside C1"
    assert not _in_c2(dag), "adversarial must be outside C2"

    # Swap the two CONST nodes (permutation [1, 0, 2, 3] over 4 internal nodes).
    dag_swapped = permute_internal_nodes(dag, [1, 0, 2, 3])

    # Canonicaliser refuses both loudly — equivariant (both raise, same outcome).
    for d in [dag, dag_swapped]:
        with pytest.raises(RuntimeError):
            fast_canonical_string(d, timeout=3.0)

    # 𝒩 directly produces non-isomorphic DAGs — the hidden defect preserved here.
    nd = dag.normalize_const_creation()
    nd_swapped = dag_swapped.normalize_const_creation()
    assert not nd.is_isomorphic(nd_swapped), (
        f"𝒩 must produce non-isomorphic results for n_vars={n_vars}, "
        f"ops={op_pair}, targets={var_targets}"
    )


# ---------------------------------------------------------------------------
# Non-adjacent targets: no interference, equivariance holds
# ---------------------------------------------------------------------------


class TestNoFailureOnNonAdjacentTargets:
    """Document that non-adjacent var-targets do NOT trigger the failure.

    When the two CONST chains point to VARs x_a and x_b where b > a+1 (i.e.
    x_{a+1} is not a target), anchoring the first CONST does not create a path
    that blocks x_a as an anchor for the second CONST.  Equivariance holds
    vacuously in these configurations despite the DAGs being outside C.
    """

    @pytest.mark.parametrize(
        "n_vars,op_pair,var_targets",
        [
            (3, (NodeType.COS, NodeType.SIN), [0, 2]),
            (4, (NodeType.SIN, NodeType.COS), [1, 3]),
            (4, (NodeType.EXP, NodeType.NEG), [0, 2]),
            (5, (NodeType.SIN, NodeType.SIN), [0, 4]),
        ],
    )
    def test_no_failure_non_adjacent(
        self,
        n_vars: int,
        op_pair: tuple[NodeType, NodeType],
        var_targets: list[int],
    ) -> None:
        # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
        # These DAGs (orphan CONSTs) now raise at the canonicaliser, same as adjacent.
        # Knowledge preserved at 𝒩 method level: non-adjacent targets do NOT trigger
        # the equivariance failure — 𝒩(D) and 𝒩(π(D)) ARE isomorphic here.
        # This documents the boundary of the failure mechanism in adjacent targets.
        """𝒩 IS equivariant for non-adjacent targets; both canonicaliser invocations raise.

        Args:
            n_vars: Number of VAR nodes.
            op_pair: Two unary operators, one per CONST chain.
            var_targets: Non-adjacent VAR indices for the two CONST chains.
        """
        dag = LabeledDAG(n_vars + 10)
        for i in range(n_vars):
            dag.add_node(NodeType.VAR, var_index=i)
        c0 = dag.add_node(NodeType.CONST, const_value=1.0)
        c1 = dag.add_node(NodeType.CONST, const_value=1.0)
        op0 = dag.add_node(op_pair[0])
        op1 = dag.add_node(op_pair[1])
        dag.add_edge(c0, op0)
        dag.add_edge(op0, var_targets[0])
        dag.add_edge(c1, op1)
        dag.add_edge(op1, var_targets[1])

        # These DAGs are outside C (VARs have in-edges, CONSTs unreachable).
        assert not _in_c1(dag)
        assert not _in_c2(dag)

        dag_swapped = permute_internal_nodes(dag, [1, 0, 2, 3])

        # Canonicaliser refuses both (orphan CONSTs) — equivariant loud refusal.
        for d in [dag, dag_swapped]:
            with pytest.raises(RuntimeError):
                fast_canonical_string(d, timeout=3.0)

        # 𝒩 directly: non-adjacent targets do NOT trigger non-equivariance.
        nd = dag.normalize_const_creation()
        nd_swapped = dag_swapped.normalize_const_creation()
        assert nd.is_isomorphic(nd_swapped), (
            f"Non-adjacent targets must not trigger 𝒩 non-equivariance: "
            f"n_vars={n_vars}, targets={var_targets}"
        )
