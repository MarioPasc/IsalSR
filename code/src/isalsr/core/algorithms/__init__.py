"""D2S algorithm variants for IsalSR.

Provides a strategy pattern for DAG-to-string conversion algorithms:
    - GreedySingleD2S: Greedy from x_1 only (fastest)
    - GreedyMinD2S: Greedy from all variable nodes, pick lexmin shortest
    - ExhaustiveD2S: Full backtracking (true canonical, exponential)
    - PrunedExhaustiveD2S: Pruned with 6-component structural tuple

Restriction: ZERO external dependencies.
"""
