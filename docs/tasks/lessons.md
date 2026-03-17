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
