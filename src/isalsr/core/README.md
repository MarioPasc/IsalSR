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

## 6. Canonical String

w*_D = lexmin{ w in argmin |D2S(D, x_1)| }

Computed from x_1 only (fixed, distinguished start node).
Complete labeled-DAG invariant: w*_D = w*_D' iff D ~ D'.

Three algorithm families:
- `canonical_string()`: Exhaustive search. Guaranteed optimal (lexmin of shortest). O(k!).
- `pruned_canonical_string()`: 6-tuple pruned exhaustive. Faster but ~0.03% suboptimal.
- `fast_canonical_string()`: **PREFERRED.** Greedy-invariant with three modes.

### 6.1 Fast Canonical (Greedy-Invariant)

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

### 6.2 Pruning Limitation (Historical)

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

## 7. Search Space Reduction

For k internal nodes, O(k!) equivalent labelings collapse to one canonical string.
Central contribution of the paper.

(Full mathematical details to be filled during implementation.)
