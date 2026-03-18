# IsalSR -- Lessons Learned

Record failure modes, corrective patterns, and insights discovered during development.

## From IsalGraph (inherited lessons)

1. **B1-B9 bugs**: Index space confusion (CDLL vs graph node indices) is the #1 source
   of silent corruption. Always use `cdll.get_value(ptr)` to convert.
2. **Pair sorting**: Must sort by `|a|+|b|`, not `a+b`. The cost is total displacement.
3. **Loop termination**: Use `or` (nodes OR edges remaining), not `and`.
4. **Pointer updates**: After emitting movement instructions, update pointer fields.

## IsalSR-specific lessons

### Phase 2 (2026-03-17)

5. **Binary op input order vs canonicalization**: `evaluate_dag` uses `sorted(in_neighbors)` by
   node ID for SUB/DIV/POW. If Phase 3 canonical form reorders node IDs, evaluation results for
   non-commutative ops could change. Must address explicitly in D2S/canonical implementation.
6. **`output_node()` silent fallback**: Returns `sinks[-1]` when multiple non-VAR sinks exist
   instead of raising. This silently masks malformed DAGs. Add a test for multi-sink case.
7. **`__slots__` and dynamic attributes**: Cannot set attributes not in `__slots__` — the
   `_trace_log` attribute had to be added to `__slots__` and initialized in `__init__`.

### Phase 5: Cache Validation (2026-03-18)

8. **CONST normalization creates multi-sink DAGs**: `normalize_const_creation()` moves CONST
   in-edges to x_1, which can disconnect parent nodes whose only out-edge was to the CONST.
   The parent becomes an extra non-VAR sink, causing `output_node()` to fail. Fixed by making
   `output_node()` CONST-tolerant: when multiple sinks exist, CONST sinks are ignored if there
   is exactly one non-CONST sink. Degenerate CONST-output DAGs (where the entire expression
   is a constant) should be filtered during cache generation — they contribute no useful basis
   functions to symbolic regression search.

9. **6-tuple pruning is a heuristic, not optimal**: The 6-component structural tuple captures
   local neighborhood density at hops 1-3, but NOT the global pointer displacement cost
   (movement instructions needed to reach a candidate in the CDLL). In 0.028% of cases
   (8/28,890 in our benchmarks), the pruned canonical string is strictly longer than the
   exhaustive canonical. In 0.09% of cases (26/28,890), the pruned string has the same length
   but differs lexicographically. The pruned result remains a consistent labeled-DAG invariant
   (deterministic, same output for isomorphic inputs) but may not satisfy the lexmin-of-shortest
   criterion. **Use `canonical_string()` for all theoretical claims.** Present
   `pruned_canonical_string()` as a fast approximation with 99.97% optimality rate.
