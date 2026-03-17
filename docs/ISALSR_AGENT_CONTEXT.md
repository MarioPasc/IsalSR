# IsalSR Agent Context

## What is IsalSR?

IsalSR (Instruction Set and Language for Symbolic Regression) represents
mathematical expressions as labeled DAGs encoded in instruction strings.
The canonical string representation is a complete labeled-DAG invariant,
reducing the symbolic regression search space by O(k!) for k internal nodes.

## Key Differences from IsalGraph

| Aspect | IsalGraph | IsalSR |
|--------|-----------|--------|
| Graph type | Undirected, unlabeled | Directed (DAG), labeled |
| Nodes | Unlabeled, indistinguishable | Labeled: VAR, ADD, MUL, SIN, ... |
| Initial state | 1 node | m variable nodes (x_1, ..., x_m) |
| Edge constraint | None | DAG (acyclic): C/c check for cycles |
| Start node (canonical) | Try all nodes | x_1 only (distinguished) |
| Structural tuple | 3-component | 6-component (in/out at distances 1-3) |
| Instruction alphabet | {N,n,P,p,V,v,C,c,W} | Movement: {N,n,P,p,C,c,W} + labeled: V[label], v[label] |
| Application | General graphs | Symbolic regression expressions |

## Paper Strategy

- **Central claim**: Isomorphism-invariant representation reduces SR search space by O(k!)
- **Approach**: Show existing SR methods accelerate with IsalSR canonical strings
- **NOT**: Proposing a new SR method (avoid overreach)
- **Target**: IEEE TPAMI
- **Benchmarks**: Nguyen (12), Feynman (50+), SRBench

## Sibling Project

IsalGraph: /home/mpascual/research/code/IsalGraph
CDLL implementation reused verbatim. G2S/S2G algorithms adapted for labeled DAGs.
