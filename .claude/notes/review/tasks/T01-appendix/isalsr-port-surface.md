# IsalSR Core — C++ Port Surface Inventory

Generated: 2026-07-27. Profiling command used:
```
cProfile over 148 fast_canonical_string(mode="wl_only") calls on k=4..8 DAGs
(4 expressions × up to 50 permutations each), total 272,734 function calls, 0.067 s.
```

---

## 1. Hot Path (call graph + micro-profile)

Entry: `fast_canonical_string(dag, mode="wl_only")` → `_fast_canonical_d2s` → `_fast_step` (recursive)

| Function | File:Line | Calls | Cum% | Notes |
|---|---|---|---|---|
| `fast_canonical_string` | canonical.py:171 | 148 | 100% | entry, normalizes CONST, dispatches |
| `_fast_canonical_d2s` | canonical.py:741 | 148 | ~88% | setup + calls _fast_step |
| `_fast_step` | canonical.py:819 | 2510 (148 base) | ~80% | hot recursive loop |
| `_walk` | canonical.py:338 | 15,814 | ~9% | CDLL pointer traversal |
| `_invariant_candidate_key` | canonical.py:711 | 8,866 | ~8% | WL hash sort key |
| `list.sort` | stdlib | 1,723 | ~6% | candidate sort at V/v branch |
| `_compute_subtree_hashes` | canonical.py:670 | 148 | ~5% | O(k) WL precomputation |
| `normalize_const_creation` | labeled_dag.py:591 | 148 | ~4% | called once per invocation |
| `LabeledDAG.remove_edge` | labeled_dag.py:276 | 2,362 | ~4% | backtracking |
| `Enum.__hash__` | stdlib | 15,544 | ~2% | from hash() in WL computation |
| `LabeledDAG.add_edge` | labeled_dag.py:224 | 1,242 | ~3% | cycle-safe insert |
| `LabeledDAG.add_edge_unchecked` | labeled_dag.py:257 | 3,004 | ~2% | V/v insertion (no cycle check) |
| `CDLL.insert_after` | cdll.py:72 | 2,210 | ~3% | new node into CDLL |
| `LabeledDAG.has_edge_unchecked` | labeled_dag.py:170 | 17,900 | ~2% | C/c edge presence check |
| `LabeledDAG.undo_node` | labeled_dag.py:296 | 2,062 | ~2% | backtracking |
| `CDLL.remove` | cdll.py:100 | 2,062 | ~2% | backtracking |
| `generate_pairs_sorted_by_sum` | dag_to_string.py:34 | cached | <1% | lru_cache(maxsize=64), called per _fast_step |

`_fast_step` is the dominant cost: ~80% of wall time, called 17× per `fast_canonical_string` on average.

---

## 2. Port Surface Table

One row per function/class the C++ extension must implement.

| Name | File:Line | Signature | Arg types | Return type | Hot path | Expose to Python? |
|---|---|---|---|---|---|---|
| `fast_canonical_string` | canonical.py:171 | `(dag, *, timeout, mode)` | `LabeledDAG, float\|None, str` | `str` | Entry | YES — primary boundary |
| `_fast_canonical_d2s` | canonical.py:741 | `(input_dag, *, timeout, mode)` | `LabeledDAG, float\|None, str` | `str` | Yes | No |
| `_fast_step` | canonical.py:819 | `(ig,og,cdll,pri,sec,i2o,o2i,nleft,eleft,prefix,tuples,hashes,deadline)` | complex | `str` | Yes | No |
| `_compute_subtree_hashes` | canonical.py:670 | `(dag)` | `LabeledDAG` | `list[int]` | Yes | No — C++ internal |
| `_invariant_candidate_key` | canonical.py:711 | `(node,ig,tuples,hashes)` | `int,LabeledDAG,list\|None,list\|None` | `tuple` | Yes | No |
| `_walk` | canonical.py:338 | `(cdll, ptr, steps)` | `CDLL, int, int` | `int` | Yes | No |
| `generate_pairs_sorted_by_sum` | dag_to_string.py:34 | `(m)` | `int` | `tuple[tuple[int,int],...]` | Yes (cached) | No |
| `LabeledDAG.__init__` | labeled_dag.py:49 | `(max_nodes)` | `int` | — | Yes | YES |
| `LabeledDAG.add_node` | labeled_dag.py:195 | `(label, var_index, const_value)` | `NodeType, int\|None, float\|None` | `int` | Yes | YES |
| `LabeledDAG.add_edge` | labeled_dag.py:224 | `(source, target)` | `int, int` | `bool` | Yes | YES |
| `LabeledDAG.add_edge_unchecked` | labeled_dag.py:257 | `(source, target)` | `int, int` | `None` | Yes | YES (internal fast path) |
| `LabeledDAG.remove_edge` | labeled_dag.py:276 | `(source, target)` | `int, int` | `bool` | Yes (backtrack) | YES |
| `LabeledDAG.undo_node` | labeled_dag.py:296 | `()` | — | `None` | Yes (backtrack) | YES |
| `LabeledDAG.out_neighbors_raw` | labeled_dag.py:113 | `(node)` | `int` | `set[int]` | Yes | No (internal) |
| `LabeledDAG.in_neighbors_raw` | labeled_dag.py:127 | `(node)` | `int` | `set[int]` | Yes | No (internal) |
| `LabeledDAG.has_edge_unchecked` | labeled_dag.py:170 | `(source, target)` | `int, int` | `bool` | Yes | No (internal) |
| `LabeledDAG.node_label_unchecked` | labeled_dag.py:177 | `(node)` | `int` | `NodeType` | Yes | No |
| `LabeledDAG.node_data_unchecked` | labeled_dag.py:184 | `(node)` | `int` | `dict` | Yes | No |
| `LabeledDAG.ordered_inputs` | labeled_dag.py:150 | `(node)` | `int` | `list[int]` | Yes (B9) | YES |
| `LabeledDAG.normalize_const_creation` | labeled_dag.py:591 | `()` | — | `LabeledDAG` | Yes | YES |
| `LabeledDAG.var_nodes` | labeled_dag.py:654 | `()` | — | `list[int]` | Yes | YES |
| `LabeledDAG.node_data` | labeled_dag.py:90 | `(node)` | `int` | `dict` | Yes | YES |
| `CDLL.__init__` | cdll.py:31 | `(capacity)` | `int` | — | Yes | YES |
| `CDLL.insert_after` | cdll.py:72 | `(node, value)` | `int, int` | `int` | Yes | YES |
| `CDLL.remove` | cdll.py:100 | `(node)` | `int` | `None` | Yes | YES |
| `CDLL.next_node` | cdll.py:60 | `(node)` | `int` | `int` | Yes | No |
| `CDLL.prev_node` | cdll.py:63 | `(node)` | `int` | `int` | Yes | No |
| `CDLL.get_value` | cdll.py:52 | `(node)` | `int` | `int` | Yes | YES |
| `NodeType` enum | node_types.py:28 | — | — | uint8 in C++ | — | YES (must map to fixed ints) |

---

## 3. Boundary Types

### LabeledDAG (labeled_dag.py:38-47)
```python
__slots__ = (
    "_out_adj",       # list[set[int]]  — adjacency out; sets of integer node IDs
    "_in_adj",        # list[set[int]]  — adjacency in
    "_input_order",   # list[list[int]] — ORDERED insertion history per target node (B9)
    "_labels",        # list[NodeType|None] — length max_nodes, None for unallocated
    "_node_data",     # list[dict]      — {var_index: int, const_value: float} per node
    "_node_count",    # int
    "_edge_count",    # int
    "_max_nodes",     # int
)
```
Pre-allocated to `max_nodes`. Node IDs are contiguous 0-based integers.
Integer widths: Python `int` (unbounded), but values always fit in `int32_t` (max nodes ~1000 in practice).

**Minimal C++ struct:**
```cpp
struct LabeledDAG {
    int node_count, edge_count, max_nodes;
    std::vector<std::unordered_set<int>> out_adj, in_adj;
    std::vector<std::vector<int>> input_order;  // insertion-ordered, NOT a set
    std::vector<int8_t> labels;                 // NodeType -> uint8_t (14 values)
    std::vector<int> var_index;                 // -1 = absent
    std::vector<double> const_value;            // NaN = absent
};
```

### CDLL (cdll.py:29-37)
```python
__slots__ = (
    "_next",      # list[int] — size capacity
    "_prev",      # list[int] — size capacity
    "_data",      # list[int] — payloads (graph node IDs)
    "_free",      # list[int] — free-list stack (pop yields next free index)
    "_size",      # int — active node count
    "_capacity",  # int
)
```
All arrays pre-allocated to `capacity`. Free list is a stack (push/pop from end).

**Minimal C++ struct:**
```cpp
struct CDLL {
    int size, capacity;
    std::vector<int> next, prev, data;
    std::vector<int> free_stack;  // stack (back = top)
};
```

### NodeType (node_types.py:28-50)
14 values: VAR, ADD, MUL, SUB, DIV, SIN, COS, EXP, LOG, SQRT, POW, ABS, NEG, INV, CONST.
In C++: `enum class NodeType : uint8_t { VAR=0, ADD=1, ... }` with a fixed mapping.
`NODE_TYPE_TO_LABEL` (node_types.py:73) maps NodeType→single char; must be replicated as a constexpr array.

---

## 4. Six Invariants — Located

### Invariant 1: CDLL indices ≠ graph node indices
`cdll.py:8-10` (module docstring):
```python
# Critical invariant: CDLL indices != graph node indices.
#     Pointers are CDLL indices; payloads are graph node IDs.
#     Use get_value(ptr) to convert CDLL index -> graph node ID.
```
Enforced at every pointer dereference in `canonical.py:856,858`:
```python
tp_out = cdll.get_value(tp)   # CDLL index → graph node ID
tp_in = o2i[tp_out]           # graph node ID → input-DAG node ID
```

### Invariant 3: add_edge direction + _input_order tracking
`labeled_dag.py:251-254`:
```python
self._out_adj[source].add(target)
self._in_adj[target].add(source)
self._input_order[target].append(source)  # ← insertion ORDER, never a set
self._edge_count += 1
```
`add_edge_unchecked` (labeled_dag.py:257-274) has identical tracking. `remove_edge` (labeled_dag.py:276-294) removes from `_input_order` by value. The C++ port must maintain `input_order` as an ordered vector (NOT unordered_set).

### Invariant 5: |a|+|b| displacement sort (Bug Fix B2)
`dag_to_string.py:61`:
```python
# BUG FIX B2: was pair[0] + pair[1]. Must be |a| + |b|.
pairs.sort(key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]), pair))
```
The secondary sort `abs(pair[0])` (then `pair` lexicographic) is also required for determinism. C++ must replicate the full 3-key sort.

### Invariant 8: ordered_inputs() for binary op first-operand check
`canonical.py:479-484` (V branch, identical at v branch canonical.py:556-562 and fast canonical.py:862-868, 930-936):
```python
cands = [
    c for c in cands
    if ig.node_label_unchecked(c) not in BINARY_OPS
    or not ig.ordered_inputs(c)
    or ig.ordered_inputs(c)[0] == tp_in
]
```
`BINARY_OPS` = `{SUB, DIV, POW}` (node_types.py:107-113). C++ must check `input_order[c][0] == tp_in` before emitting V for binary ops.

### Invariant 9: normalize_const_creation before canonical
`canonical.py:231`:
```python
normalized = dag.normalize_const_creation() if dag._has_const_nodes() else dag
return _fast_canonical_d2s(normalized, timeout=timeout, mode=mode)
```
Also at `canonical.py:95` and `canonical.py:146`. The normalization moves all CONST creation edges to node 0 (x_1), called at `labeled_dag.py:641`:
```python
for c in sorted(const_nodes):
    new.add_edge(0, c)
```
Iteration order is `sorted(const_nodes)` — deterministic.

### Invariant 10: label-aware pruning (only in exhaustive/_canonical_d2s, not in _fast_step)
`canonical.py:488-498`:
```python
label_groups: dict[NodeType, list[int]] = {}
for c in cands:
    label_groups.setdefault(ig.node_label(c), []).append(c)
pruned: list[int] = []
for group in label_groups.values():
    max_tup = max(tuples[c] for c in group)
    pruned.extend(c for c in group if tuples[c] == max_tup)
```
In `_fast_step`, the sort key `_invariant_candidate_key` includes `label_char` as first component (canonical.py:730), achieving the same label-aware partition implicitly. C++ must preserve label as first component of the sort key.

---

## 5. Premise Verification

### Premise 1: "The 14,841-DAG unit-test corpus lives in tests/"
**REFUTED.**
- "14,841" does NOT appear in any file under `tests/`. Verified with `grep -rn "14841" tests/` → 0 results.
- No stored DAG artefact (JSON/pkl/npy) exists in `tests/`. The directory contains only `.py` files and `__pycache__`.
- The figure "14,841" appears only in `.claude/notes/review/` markdown files, attributed to `discussion.tex:38` — a manuscript file that does NOT exist in the repo (find returned no results).
- The test suite generates DAGs dynamically at test time from string expressions via `StringToDAG`. The completeness tests exercise 41,217 permutation instances across 8 expressions (k=1..8, k! permutations each, 3 modes × ~13,739 unique DAGs) — not 14,841.
- **To make an addressable corpus**: a script must parse the 8 COMPLETENESS_EXPRS + 2 LARGE_EXPRS, generate all k! (or 100 random) permutations, serialize each LabeledDAG to a file (e.g., via JSON adjacency list), and store the result. No such script exists.

### Premise 2: "Exhaustive completeness verified k=1..8 all k! perms, statistically k=10..15, 890 tests pass"
**PARTIAL.**
- Tests exist: `tests/unit/test_fast_canonical.py` — `TestFastCanonicalCompleteness` (lines 257-282) does exhaustive k! permutations for k≤8; `TestFastCanonicalLargerDAGs` (lines 290-317) does 100 random permutations.
- Current collected test count: **86 tests** in `test_fast_canonical.py`, **8 tests** in `test_fast_canonical_property.py`. Total: **94 fast-canonical tests** (from `pytest --collect-only -q`). The suite-wide total is **1,496 tests**.
- The claim "890 tests pass" is NOT matched: the file has 86 parametrized test FUNCTIONS, each running k! iterations internally. If 890 = the number of permutation evaluations within `TestFastCanonicalCompleteness` for mode="wl_only", then: 1+2+24+24+6+120+720 = 897 permutations for 7 of 8 exprs (skipping k=8 as k>8 guard would skip if k>8, but k=8 would be included). With 8 exprs × 3 modes = up to ~123,651 iterations total. The exact 890 does not correspond to any obvious count.
- The large-expr range: tests use k=10 and k=11 (LARGE_EXPRS computed above), NOT k=10..15 as claimed.
- Tests are present and exercisable; the claim "890" appears to be either an outdated count or a mode-specific iteration count rather than a pytest test function count.

### Premise 3: "≥100,000 evolved DAGs can be replayed from stored Bingo/UDFS trajectories in /media/.../wl_subtree_unified/"
**REFUTED on two independent grounds.**

(a) **Path is wrong**: The stated path `/media/mpascual/Sandisk2TB/research/isalsr/results/...` does NOT exist. Actual path: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified/`.

(b) **No DAG data stored**: `wl_subtree_unified/` contains 0 `run_log.json` files. It holds only aggregated analysis CSVs/JSONs (`analysis/` subdir) plus `bingo/benchmark/` and `udfs/benchmark/` with per-problem aggregate CSVs.

(c) **Run logs store aggregates only**: The 2,640 `run_log.json` files in `wl_subtree/` (the source that was merged into `unified/`) have structure:
```json
{"metadata": {...}, "results": {"regression": ..., "time": ..., "search_space": {"total_dags_explored": 692, "unique_canonical_dags": 369, "empirical_reduction_factor": 1.875, "max_internal_nodes_seen": 8, "theoretical_reduction_bound": 40320.0, "redundancy_rate": 0.467}}, "best_expression": {...}}
```
No individual DAG structures, canonical strings, or command arrays are persisted. `total_dags_explored` is a single integer per run. Replay is impossible from this data.

(d) **To enable replay**: each dedup interceptor would need to serialize the LabeledDAG (or its canonical string) at the time of canonicalization. No such serialization exists.

### Premise 4: `core/permutations.py` provides `permute_internal_nodes(dag, perm)` for all k! copies
**CONFIRMED.**
- `permutations.py:22`: `def permute_internal_nodes(dag: LabeledDAG, perm: Sequence[int]) -> LabeledDAG`
- Validates `sorted(perm) == list(range(k))` at line 54.
- Preserves operand order via `dag.ordered_inputs(old_target)` + `add_edge_unchecked` at lines 91-94.
- Actively used in `tests/unit/test_fast_canonical.py:277`: `dag_p = permute_internal_nodes(dag, list(perm))`
- Also used in `tests/unit/test_permutations.py:13`, `experiments/random_dag_experiment/generate_dags.py:50`, `experiments/synthetic_scalability/run_synthetic_scalability.py:62`, `experiments/scripts/search_space_permutation_analysis.py:65`.

---

## 6. External Call Sites

### isalsr.evaluation — 2 files
| Site | Import | Behavior change on engine switch? |
|---|---|---|
| `evaluation/constant_optimizer.py:22-24` | `evaluate_dag`, `LabeledDAG`, `NodeType` | No — does not call canonical |
| `evaluation/fitness.py:15-16` | `evaluate_dag`, `LabeledDAG` | No — does not call canonical |

### isalsr.search — 4 files
| Site | Import | Behavior change? |
|---|---|---|
| `search/hill_climbing.py:16-18,84,118` | `pruned_canonical_string`, `CanonicalTimeoutError`, `DAGToString`, `StringToDAG` | YES — uses `pruned_canonical_string` (not `fast_canonical_string`); engine switch changes dedup behavior |
| `search/population.py:17-19,151` | same + `StringToDAG` | YES |
| `search/random_search.py:16-18` | same | YES |
| `search/operators.py:14` | `OperationSet` | No — metadata only |

### isalsr.adapters — 3 files
| Site | Import | Behavior change? |
|---|---|---|
| `adapters/base.py:15-18` | `D2SAlgorithm`, `DAGToString`, `LabeledDAG`, `StringToDAG` | YES — DAGToString is a direct call into core |
| `adapters/networkx_adapter.py:11-12` | `LabeledDAG`, `NodeType` | No — structural conversion only |
| `adapters/sympy_adapter.py:28-29,55` | `LabeledDAG`, `NodeType`, `BINARY_OPS` | No — structural conversion only |

### experiments/models (production dedup — PRIMARY IMPACT SITES)
| Site | Import | Behavior change? |
|---|---|---|
| `models/bingo/isalsr_runner.py:321` | `fast_canonical_string` (inline import) | YES — this is the dedup guard; canonical string change changes reduction factor |
| `models/udfs/isalsr_runner.py:113` | `fast_canonical_string` (inline import) | YES |
| `models/bingo/translator.py:301` | `pruned_canonical_string` | YES — used to compute post-run canonical for reporting |
| `models/udfs/translator.py:245` | `pruned_canonical_string` | YES |
| `models/bingo/adapter.py:20-21` | `LabeledDAG`, `NodeType`, `BINARY_OPS` | Boundary type — must be compatible with C++ LabeledDAG |
| `models/udfs/adapter.py:24-25` | same | same |

### experiments/scripts (analysis and figure generation — not in production loop)
These import `canonical_string`, `pruned_canonical_string`, `fast_canonical_string`, `DAGToString`, `StringToDAG`, `LabeledDAG`, `NodeType` across 15+ scripts. An engine switch changes string values but not experiment validity if the C++ and Python strings agree.

---

## 7. Determinism Risks

### Risk 1 — CRITICAL: `hash()` on NodeType is PYTHONHASHSEED-dependent
`canonical.py:702`:
```python
node_hash[u] = hash((dag.node_label_unchecked(u), tuple(children_hashes)))
```
**Verified**: `hash(NodeType.SIN)` = -596,725,465,393,948,695 at SEED=0 vs 9,050,745,085,082,211,361 at SEED=42. The WL hash values change with every new Python process.

**Consequence**: `fast_canonical_string(mode="wl_only")` produces different canonical strings across Python sessions whenever two candidates have different WL hashes and the relative ordering changes. **Byte-exact equivalence between C++ and Python is impossible unless PYTHONHASHSEED=0 is fixed OR the hash function is replaced with a deterministic one** (e.g., map NodeType→uint8_t before hashing, use FNV-1a or xxHash with fixed seed).

**Fix for C++ port**: Replace `hash((NodeType, tuple(ints)))` with a deterministic function that maps NodeType to a fixed integer (e.g., its enum ordinal) then applies a non-randomized hash (e.g., boost::hash_combine or a simple polynomial hash).

### Risk 2 — MODERATE: set[int] iteration order affects tie-breaking
`labeled_dag.py:51-52`: `_out_adj: list[set[int]]`, `_in_adj: list[set[int]]`

In `canonical.py:861`:
```python
cands = [n for n in ig.out_neighbors_raw(tp_in) if n not in i2o]
```
Python's `set[int]` iteration order for small integers is deterministic in CPython (hash(n) = n for int), but NOT a language guarantee. Python's `list.sort` is stable, so tied candidates retain their set-iteration-derived order. If C++ uses `std::unordered_set<int>` with a different bucket layout, tied-candidate order may differ, producing a different (but still valid) canonical string.

**Fix**: Use a sorted enumeration of neighbors in C++ (`std::set<int>` or sort before filtering), OR add node ID as a final tiebreaker in `_invariant_candidate_key`.

### Risk 3 — MINOR: dict iteration order in label_groups (pruned mode only)
`canonical.py:491-497` (in `_step`, not `_fast_step`): `label_groups: dict[NodeType, list[int]]` uses insertion order (Python 3.7+). C++ `std::unordered_map` has undefined iteration order. Affects only `canonical_string`/`pruned_canonical_string`, not the production `fast_canonical_string`. In `_fast_step`, label is the first component of the sort key, so group enumeration order is not needed.

### Risk 4 — MINOR: float formatting for const_value
`canonical.py:511-512,886-887`:
```python
const_value=float(data["const_value"]) if "const_value" in data else None,
```
CONST values appear in node metadata but NOT in the canonical string (CONST nodes have label char "k", no value). No float-formatting risk in the canonical string output. Risk exists only if `best_expression` fields in run_logs are compared byte-for-byte.

### Summary
Risk 1 is a BLOCKER for byte-exact equivalence testing. The test harness must either set `PYTHONHASHSEED=0` or the C++ port must use a different, fixed-seed hash. Risks 2 and 3 are tie-breaking issues that produce valid-but-different canonical strings; they affect the acceptance gate only if the gate checks byte equality rather than isomorphism.

