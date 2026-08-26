"""IsalSR search module -- string-level search operators and algorithms.

Dependencies: numpy.

Provides mutation, crossover, and search algorithms that operate directly
on IsalSR instruction strings. The key advantage: because strings are
canonicalized, every point in the search space corresponds to a structurally
unique expression, eliminating O(k!) redundancy.
"""
