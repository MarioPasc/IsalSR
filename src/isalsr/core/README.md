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

## 7. Search Space Reduction

For k internal nodes, O(k!) equivalent labelings collapse to one canonical string.
Central contribution of the paper.

(Full mathematical details to be filled during implementation.)
