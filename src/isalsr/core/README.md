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
6-component structural tuple for backtracking pruning.
Complete labeled-DAG invariant: w*_D = w*_D' iff D ~ D'.

Two variants:
- `canonical_string()`: Exhaustive search. Guaranteed optimal (lexmin of shortest).
- `pruned_canonical_string()`: 6-tuple pruned search. Much faster but may produce
  suboptimal results in rare cases (~0.03% empirically). Still a consistent invariant.

### 6.1 Pruning Limitation

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
