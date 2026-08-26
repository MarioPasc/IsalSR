# IsalSR Core -- Mathematical Foundation and Architecture

## 1. Introduction

IsalSR represents symbolic regression expressions as labeled Directed Acyclic Graphs (DAGs)
encoded in instruction strings. The canonical string is a complete labeled-DAG invariant,
enabling isomorphism-free search spaces for symbolic regression.

## 2. Instruction Set (Sigma_SR)

Two-tier encoding:
- Movement/structure: N, P, n, p, C, c, W (single-char)
- Labeled insertion: V[label], v[label] (two-char)
- Labels: +, *, -, /, s, c, e, l, r, ^, a, g, i, k

### 2.1 Commutative Encoding

Non-commutative binary ops can be decomposed into commutative variadic + unary pairs:
- SUB(x, y) = ADD(x, NEG(y)) where NEG (label 'g') is unary negation (-x)
- DIV(x, y) = MUL(x, INV(y)) where INV (label 'i') is unary inverse (1/x)

This eliminates operand-ordering requirements from the isomorphism definition
for all operations except POW (whose operand order is inherently semantic).
The commutative operation set is available via `OperationSet.commutative()`.

Inspired by the term-rewriting approach of GraphSR (Xiang et al.).

## 3. Initial State

For m input variables: m VAR nodes, no edges, CDLL in order, pointers on x_1.

## 4. Edge Semantics

Edge u -> v: "u provides input to v" (data flow direction).

## 5. DAG Constraint

C/c instructions check for cycles via DFS reachability before adding edges.
V/v instructions never create cycles (new node has no outgoing edges).

## 6. CONST Creation Edges and the Reachability Precondition

### 6.1 Why the precondition exists

Every insertion instruction of Sigma_SR creates a node **together with an edge
from the acting pointer**: `V[label]` / `v[label]` add a new node *and* the edge
(pointer node) -> (new node). The alphabet contains **no** instruction that
creates a node in isolation.

A non-VAR node of in-degree 0 is therefore **unencodable**: no string w satisfies
S2D(w, m) ~= D, and D2S cannot emit that node, because no pointer can ever be
placed on one of its in-neighbours. This is the Round-Trip Fidelity hypothesis
stated contrapositively:

> every non-VAR node of D is reachable from some VAR node via directed edges.

For every label except CONST the condition holds automatically -- operation nodes
consume operands, hence have in-edges. **CONST is the sole exception**: it is an
evaluation-neutral leaf that ignores its in-edges, so host SR systems emit
constants as expression-tree terminals with nothing pointing into them.

### 6.2 The operation

`LabeledDAG.normalize_const_creation()` supplies the missing *creation edge*.

```
N(D):
    D' <- copy(D)                      # nodes, labels, _input_order verbatim
    if D has no CONST node: return D'  # guard: no-op on CONST-free DAGs
    for c in sorted(CONST nodes of D'):        # increasing node index
        if in_degree(c) > 0: continue         # already encodable
        for i in 1..m:                        # lowest-indexed variable first
            if add_edge(x_i, c) succeeds:     # add_edge refuses cycle-closing
                break
    return D'
```

Complexity: at most `|CONST|` constants x `m` anchor attempts, each one
acyclicity check in O(|V|+|E|), so `O(|CONST| * m * (|V|+|E|))` worst case. On
host-adapter output `x_1` always succeeds (see 6.4 / E5), so it is `|CONST|`
checks.

**No edge is ever removed**, and CONST nodes of in-degree >= 1 are untouched.

### 6.3 Properties (all load-bearing)

| # | Property | Why |
|---|----------|-----|
| N1 | Edge-monotone: same nodes/labels, `E(D) subset E(N(D))` | only adds edges, so reachability is never destroyed |
| N2 | Idempotent: `N(N(D)) = N(D)` | anchored CONSTs are skipped; refused ones stay refused |
| N3 | **Identity on the hypothesis class** | if every non-VAR node is reachable from a VAR, no CONST has in-degree 0, so `N(D) = D`. In particular `N` is the identity on the entire image of S2D |
| N4 | Evaluation-preserving: `eval(N(D)) = eval(D)` | CONST ignores in-edges, and no out-degree changes, so the sink set (output node) is unchanged |
| N5 | Isomorphism-equivariance | **fails in general**; holds on `C = C1 ∪ C2` -- see 6.5 |

N3 is the key property: the normalization and the theorem hypothesis are the same
condition in constructive and declarative form. It is what keeps the canonical
string a *complete* labeled-DAG invariant rather than a complete invariant of some
coarser quotient.

### 6.4 Measured justification

| # | Experiment | Result |
|---|------------|--------|
| E1 | Incidence of the violation, 4 populations | S2D corpus 0/14,841 (0.00%); synthetic DAGs 0/49,980 (0.00%); **Bingo 132,746/154,568 (85.88%)**; **UDFS 3,890/3,890 (100.00%)**. After `N`: **0** everywhere |
| E2 | k-stratified Bingo profile | 0.00% at k=0, 27.31% at k=1, 50.91% at k=2, 85.19% at k=8, 98.22% at k=16, **100.00% for all k>=24**. Matches "violated iff >=1 CONST terminal"; residual 0 in all 37 (method,k) cells over 158,458 DAGs |
| E3 | N3 (identity on the hypothesis class) | 0 canonical-string disagreements and 0 edges removed on 10^5 random DAGs satisfying reachability |
| E4 | Policy-invariance of reported numbers | all normalization policies structurally identical on **12,176,790** Bingo and **234,865** UDFS DAGs; identical distinct-string counts and rho. Policies separate only on synthetic DAGs with an orphan CONST reaching a VAR (169 extra merged classes; synthetic rho 1.040 -> 1.042) |
| E5 | Order-independence on adapter output | `permute_internal_nodes`, K=8 per DAG: **0 failures / 123,240 tests** on 15,530 Bingo DAGs, all with every VAR a pure source. 0 failures on random S2D m=2,3 (3,200-4,000 tests each) |

Scripts: `experiments/models/fallback_ledger.py` (E1, E2),
`experiments/scripts/validate_const_repair_synthetic.py` (E3),
`measure_const_normalization_arms.py` (E4),
`experiments/scripts/validate_const_equivariance.py` (E5).
Tests: `tests/unit/test_const_normalization_repair.py`,
`tests/unit/test_const_normalization_equivariance.py`,
`tests/unit/test_fallback_ledger.py`.

### 6.5 Limits and where the step is applied

**N5 fails in general.** The tie-break "least index that does not close a cycle"
is stated over node indices, and node indices are exactly what isomorphism
permutes. Anchoring one orphan CONST creates paths that can make another orphan's
preferred anchor cycle-closing. Three conditions must hold simultaneously:

1. >= 2 CONST nodes with in-degree 0;
2. >= 1 VAR node that is the target of an edge;
3. a directed path from an orphan CONST to a VAR.

`N` is provably equivariant on `C = C1 ∪ C2`:

- `C1` = DAGs satisfying reachability -- `N = id` there (N3);
- `C2` = DAGs in which no VAR node is an edge target -- no orphan CONST can reach
  a variable, so `x_1` never closes a cycle, every orphan anchors to `x_1`, and
  the result is independent of processing order.

`C` is **sufficient, not tight**: some DAGs outside `C` do not fail. Do not
present it as a characterization of the failure set.

**Where the repair happens.** It is a *producer-side* step. The host adapters
call their own specialization (`_normalize_const_edges` in
`experiments/models/{bingo,udfs}/adapter.py`, which anchors unconditionally to
node 0 -- sound because adapter output is entirely inside `C2`). The
canonicalizer **assumes** the precondition and refuses loudly on an in-degree-0
CONST rather than repairing it silently: such a DAG has no encoding in Sigma_SR
at all. Keeping `N` out of the canonical map makes `fcs` a pure function of `D`,
so N5 cannot affect the canonical string.

### 6.6 What the operation is NOT

It is **not** "redirect all CONST creation edges to x_1". A relocation policy
(delete existing CONST in-edges, add `x_1 -> c`) is unsound on three counts:

1. **It can orphan the node.** `C`/`c` instructions can direct an edge *into* a
   variable, so `x_1` may lie downstream of the CONST; the replacement edge is
   then refused as cycle-closing while the original was already deleted.
   Measured: **48** canonicalization failures on 10^5 random DAGs, vs **0** for
   the additive repair.
2. **It is not evaluation-preserving.** On `x -> COS -> CONST(1.0)` it moves the
   output sink from CONST to COS: 1.0 becomes cos(1.5) = 0.0707.
3. **It is not injective on isomorphism classes.** It merges DAGs whose CONST
   nodes hang off different parents, which are different labeled DAGs under the
   isomorphism definition -- breaking the (=>) direction of completeness.

See `docs/md_files/changes/d2s_canonicalisation_failures.md`.

## 7. Canonical String

w*_D = lexmin{ w in argmin |D2S(D, x_1)| }

Computed from x_1 only (fixed, distinguished start node).
Complete labeled-DAG invariant: w*_D = w*_D' iff D ~ D'.
**Read section 7.3 before relying on that sentence**: `~` is labeled-DAG
isomorphism *plus* the first-operand designation on `{Sub, Div, Pow}`, and taking
`~` to mean anything finer produces spurious round-trip failures.

Three algorithm families:
- `canonical_string()`: Exhaustive search. Guaranteed optimal (lexmin of shortest). O(k!).
- `pruned_canonical_string()`: 6-tuple pruned exhaustive. Faster but ~0.03% suboptimal.
- `fast_canonical_string()`: **PREFERRED.** Greedy-invariant with three modes.

### 7.1 Fast Canonical (Greedy-Invariant)

At each V/v branch point, candidates are sorted by an isomorphism-invariant key.
If the best candidate is unique, it is taken greedily (no backtracking). Ties are
resolved by backtracking over tied candidates only (lexmin among tied).

Three modes control the invariant sort key:

| Mode | Sort key | Precomputation | Default? |
|------|----------|----------------|----------|
| `"wl_only"` | `(label_char, WL_hash)` | O(k) WL hash | **YES** |
| `"wl_tiebreak"` | `(label_char, 6-tuple↓, WL_hash)` | O(k²) tuple + O(k) WL | No |
| `"tuple_only"` | `(label_char, 6-tuple↓)` | O(k²) tuple | No (legacy) |

**WL-only is the default** because:

1. The 1-WL subtree hash h(v) = hash(label(v), multiset{h(c) : c in children(v)})
   captures the full rooted subtree isomorphism type. The 6-tuple tau(v) captures only
   depth-3 neighborhood cardinalities. Therefore h(v) = h(w) implies tau(v) = tau(w)
   (WL subsumes 6-tuple), but not conversely. WL is strictly more discriminative
   (Weisfeiler & Leman, 1968; Shervashidze et al., JMLR 2011).

2. Empirical speedup: 1.43x mean on evolved Bingo DAGs (k=6-14), range 1.09-1.73x.
   The speedup comes from (a) skipping O(k²) 6-tuple BFS and (b) simpler 2-component
   key comparison vs 3-component key in the D2S recursion hot path.

3. Completeness verified exhaustively for k=1..8 (all k! permutations, up to 40,320)
   and statistically for k=10-15 (100 random permutations each). All three modes
   produce valid complete invariants.

### 7.2 Pruning Limitation (Historical)

The 6-component structural tuple tau(v) = (|in_N1|, |out_N1|, ..., |out_N3|) captures
local neighborhood density at hops 1-3. It is automorphism-invariant but does not
account for the global pointer displacement cost (the number of N/P/n/p movement
instructions needed to reach a candidate in the CDLL). In rare cases, a candidate
with higher local connectivity (higher tuple) requires more movement tokens than a
candidate with lower connectivity but closer CDLL position, leading to a longer string.

Empirical measurement (28,890 entries across Nguyen and Feynman benchmarks):
- 99.88% agreement between pruned and exhaustive
- 0.09% same-length lexicographic differences (different tie-breaking)
- 0.03% length mismatches (pruned is longer than exhaustive)

### 7.3 What the invariant separates: the first-operand designation (T18)

Section 7's "complete labeled-DAG invariant" needs one qualifier, because a bare
labeled DAG is *not* the object Sigma_SR represents. Both `Sub(sin x, cos x)` and
`Sub(cos x, sin x)` have the same nodes, the same labels and the same edge set;
they are the same labeled DAG and different functions. What distinguishes them is
which in-edge is the **first operand**, and that is exactly the extra structure the
string carries.

**Definition (first-operand designation).** For a DAG `D` with in-edge insertion
order, let `phi_D(v) = ordered_inputs(v)[0]` for every node `v` whose label lies in
`BINARY_OPS = {Sub, Div, Pow}` with in-degree >= 1. Two labeled DAGs are
**Sigma_SR-equivalent** iff some label- and edge-preserving bijection `sigma`
(matching VAR nodes by `var_index`) also satisfies `sigma(phi_D(v)) = phi_D'(sigma(v))`
for every such `v`.

**Why `phi` and nothing more.** `V`/`v` may create a binary op only from its first
operand -- Critical Invariant 8 / B9, enforced in
`dag_to_string._find_new_out_neighbor`, at the four `ordered_inputs(c)[0] ==
ptr_in` admissibility sites in `canonical.py` (lines 648, 726, 1032, 1101) and at
their `input_order_[c][0] == ptr_in` counterparts in
`native/src/canonical.cpp`. Every *further* in-edge of that
node is emitted by `C`/`c` in canonical-traversal order. So the string determines
`phi_D` and the edge set, and determines nothing else about the in-edge ordering.
Any predicate that separates DAGs on more than that is strictly finer than the
canonical string and will report round-trip failures that are artefacts of the
predicate.

**Lemma (where the two readings can differ at all).** Let `~_full` require a
label- and edge-preserving bijection `sigma` (VAR matched by `var_index`) that
agrees with the **entire** `ordered_inputs` list on every binary node, and let
`~_first` require agreement on position 0 only. Then `~_full` implies `~_first`
always, and the converse holds whenever every binary node of `D` has in-degree
<= 2.

*Proof.* The forward direction is immediate. For the converse, let `sigma`
witness `~_first` and let `v` be binary with in-degree `d <= 2`. For `d = 0`
there is nothing to compare, and for `d = 1` position 0 is the whole list. For
`d = 2`, write `ordered_inputs(v) = [a1, a2]` and `ordered_inputs(sigma v) =
[b1, b2]`. Position 0 gives `sigma(a1) = b1`; `sigma` preserves edges, so
`{sigma(a1), sigma(a2)} = {b1, b2}`; and `sigma` is injective, so
`sigma(a2) = b2`. Hence `sigma` witnesses `~_full`. QED

So the two readings can diverge **only** at a binary node of in-degree >= 3. The
condition is weaker than well-formedness -- it tolerates the *under*-saturated
nodes that host adapters routinely emit, because `add_edge` refuses a duplicate
edge and `f(x, x)` therefore arrives with in-degree 1.

**On well-formed DAGs the qualifier is vacuous.** Call `D` *arity-well-formed* when
every unary op has in-degree 1, every binary op in-degree exactly 2, every variadic
op in-degree >= 2, every VAR in-degree 0 and every CONST in-degree <= 1. This is
precisely the class `dag_evaluator` accepts; outside it, evaluation raises
`EvaluationError`. On that class, fixing `phi_D(v)` leaves one in-edge, so the full
`ordered_inputs(v)` is forced and Sigma_SR-equivalence coincides with
"isomorphic and semantically identical". The two notions can differ only at an
*over-saturated* binary node (in-degree > 2), which denotes no expression at all.

**Consequence for `is_isomorphic`.** `LabeledDAG.is_isomorphic` implements
Sigma_SR-equivalence. Until 2026-08-03 it compared the whole `ordered_inputs` list
for binary ops, which made it finer than the canonical string on over-saturated
nodes -- and, inconsistently, still ignored the same surplus ordering on unary and
variadic nodes, where the evaluator reads `sorted(in_neighbors)`. That mismatch,
not the C++ port and not the canonicaliser, produced the five gate-3 round-trip
failures of ticket T18. See `tests/unit/test_t18_operand_order_completeness.py`.

## 8. Search Space Reduction

For k internal nodes, O(k!) equivalent labelings collapse to one canonical string.
Central contribution of the paper.

(Full mathematical details to be filled during implementation.)
