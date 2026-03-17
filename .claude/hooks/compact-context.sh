#!/usr/bin/env bash
# Compaction recovery hook -- re-injects critical context after /compact

cat <<'CONTEXT'
=== ISALSR COMPACTION RECOVERY ===

PROJECT: IsalSR -- Instruction Set and Language for Symbolic Regression
Labeled DAGs for SR with isomorphism-invariant canonical string representation.
Sibling project: IsalGraph (/home/mpascual/research/code/IsalGraph)

INSTRUCTION SET (two-tier encoding):
  Movement (single-char): N P n p C c W
  Labeled insertion (two-char): V[label] or v[label]
  Labels: + * - / s c e l r ^ a k
    +: ADD (variable-arity 2+)   *: MUL (variable-arity 2+)
    -: SUB (binary)              /: DIV (binary, protected)
    s: SIN (unary)               c: COS (unary)
    e: EXP (unary)               l: LOG (unary, protected)
    r: SQRT (unary, protected)   ^: POW (binary)
    a: ABS (unary)               k: CONST (leaf)

  Tokenization: V/v consume next char as label. Bare 'c' = edge instruction.
  C: directed edge primary->secondary (cycle check; no-op if cycle)
  c: directed edge secondary->primary (cycle check; no-op if cycle)

INITIAL STATE (m variables):
  DAG: m VAR nodes {x_1,...,x_m}, no edges
  CDLL: [x_1, x_2, ..., x_m] in order
  Both pointers on CDLL node for x_1

CRITICAL INVARIANTS:
  1. Pointers are CDLL node indices, NOT graph node indices.
     To get graph node: cdll.get_value(pointer)
     NEVER pass pointer directly to LabeledDAG.add_edge()
  2. Pointer does NOT move after V/v instruction.
  3. Edge u->v means "u provides input to v" (data flow direction).
  4. C/c must check for DAG cycles before adding edge (DFS reachability).
     V/v never creates cycles (new node has no outgoing edges).
  5. Variables are pre-inserted (NOT created by V/v). They are distinguishable.
  6. generate_pairs_sorted_by_sum sorts by |a|+|b|, not a+b.

CANONICAL STRING:
  - Computed from x_1 ONLY (fixed start, no isomorphism ambiguity over start node)
  - 6-component structural tuple: (in_N1, out_N1, in_N2, out_N2, in_N3, out_N3)
  - Complete labeled-DAG invariant: w*_D = w*_D' iff D ~ D'
  - Search space reduced by O(k!) for k internal nodes

DEPENDENCY RULE:
  isalsr.core = ZERO external deps (stdlib only)
  isalsr.adapters = optional (networkx, sympy)
  isalsr.evaluation = numpy, scipy
  isalsr.search = numpy

ENVIRONMENT:
  Conda env: isalsr
  Tests: python -m pytest tests/ -v
  Lint: python -m ruff check src/ tests/
  Types: python -m mypy src/isalsr/

KEY FILES:
  src/isalsr/core/         -- Core implementation (zero deps)
  src/isalsr/core/README.md -- Full mathematical + architectural spec
  .claude/CLAUDE.md        -- Project hub
  docs/ISALSR_AGENT_CONTEXT.md -- Full agent context

ROUND-TRIP PROPERTY:
  S2D(w) ~ S2D(D2S(S2D(w), x_1)) for all valid strings w
  ~ denotes labeled DAG isomorphism

=== END COMPACTION RECOVERY ===
CONTEXT
