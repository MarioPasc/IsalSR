# T18 — canonical-string completeness: the five counterexamples, in full

> **RESOLVED 2026-08-03.** The five pairs are **not** completeness failures. Every
> one is isomorphic as a labeled DAG *and* preserves the first-operand
> designation; the only difference is the order of the **surplus** in-edges of an
> over-saturated binary node (a `POW`/`DIV`/`SUB` with 3–4 in-edges), which
> Σ_SR does not encode and `dag_evaluator` refuses outright — all ten DAGs raise
> `EvaluationError`. The defect was in `LabeledDAG.is_isomorphic`, which compared
> the whole `ordered_inputs` list instead of position 0. After the fix, gate 3 is
> **10,000 DAGs / 20,000 comparisons / 0 mismatches / 0 errors**.
> Analysis and decision: `.claude/notes/review/tasks/T18-canonical-completeness-operand-order.md` §8.
> The per-case detail below is retained as the evidence trail and is unchanged.

**Commit**: `2365c823d5faa48751206c8bcaafa360d9844d8a`
**Reproduce**: `python experiments/scripts/equivalence_gate.py --gate 3 --backend-a python --backend-b cpp --out gate3.json`, then `python -m experiments.scripts.t18_completeness_counterexamples`

Each row is a pair `(D, D')` with `D' = S2D(fcs(D))` where **`fcs(D) == fcs(D')`**
(same dedup class) and **`D ≇ D'`** (not isomorphic). Two distinct labeled DAGs
therefore share one canonical string, which under-counts
`unique_canonical_dags` and over-states ρ.

## Summary

| # | k | vars | engines agree | same class | isomorphic | in-deg-0 CONST | VAR as target |
|---|---|---|---|---|---|---|---|
| 2166 | 19 | 2 | True | True | **False** | 0 | 1 |
| 2256 | 15 | 2 | True | True | **False** | 0 | 1 |
| 3687 | 17 | 1 | True | True | **False** | 0 | 0 |
| 7403 | 18 | 1 | True | True | **False** | 0 | 0 |
| 7771 | 13 | 1 | True | True | **False** | 0 | 0 |

`engines agree = True` on every row is what rules out the C++ port: Python and
C++ produce byte-identical strings and both fail. `in-deg-0 CONST = 0` on every
row rules out the `is_isomorphic` precondition (CLAUDE.md invariant 9); the rows
with `VAR as target = 0` additionally sit inside `𝒞₂`, where
`normalize_const_creation` is equivariant.

## Per-case detail

### Corpus index 2166 (k = 19, 2 variable(s))

```
source string   : pv+vgVav^VavlCNCvrV^WNvgviv*Vsv+vkV*Vsv*cncvlWvk
fcs(D)  python  : VaVapv*v*v+v+v^vgvgvivkvlvrCpv^nv*vsvsPVkVlCNNNNNnnnnCNc
fcs(D)  cpp     : VaVapv*v*v+v+v^vgvgvivkvlvrCpv^nv*vsvsPVkVlCNNNNNnnnnCNc
fcs(D') python  : VaVapv*v*v+v+v^vgvgvivkvlvrCpv^nv*vsvsPVkVlCNNNNNnnnnCNc
```

- `fcs(D) == fcs(D')`: **True**
- `D ≅ D'`: **False**
- labels equal: **True**
- `D`  : 21 nodes, 23 edges, {'ABS': 2, 'ADD': 2, 'CONST': 2, 'INV': 1, 'LOG': 2, 'MUL': 3, 'NEG': 2, 'POW': 2, 'SIN': 2, 'SQRT': 1, 'VAR': 2}
- `D'` : 21 nodes, 23 edges, {'ABS': 2, 'ADD': 2, 'CONST': 2, 'INV': 1, 'LOG': 2, 'MUL': 3, 'NEG': 2, 'POW': 2, 'SIN': 2, 'SQRT': 1, 'VAR': 2}

- edges `D` : `[(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (1, 15), (1, 18), (6, 1), (6, 9), (9, 13), (9, 16), (9, 17), (18, 9), (18, 19), (18, 20)]`
- edges `D'`: `[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (2, 1), (2, 15), (4, 15), (4, 19), (4, 20), (15, 16), (15, 17), (15, 18)]`
- labels `D` : `['VAR', 'VAR', 'ADD', 'NEG', 'ABS', 'POW', 'ABS', 'LOG', 'SQRT', 'POW', 'NEG', 'INV', 'MUL', 'SIN', 'ADD', 'CONST', 'MUL', 'SIN', 'MUL', 'LOG', 'CONST']`
- labels `D'`: `['VAR', 'VAR', 'ABS', 'ABS', 'MUL', 'MUL', 'ADD', 'ADD', 'POW', 'NEG', 'NEG', 'INV', 'CONST', 'LOG', 'SQRT', 'POW', 'MUL', 'SIN', 'SIN', 'CONST', 'LOG']`

- SymPy `D` : `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`
- SymPy `D'`: `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`

### Corpus index 2256 (k = 15, 2 variable(s))

```
source string   : vcvlvrCV/v^V^nCvgvcnCpWvgVaPVgV-cv-nCNv/v-C
fcs(D)  python  : V/V^V^VaVcVcVlVrpv-vgppv-vgvgnnnv-v/PCPPCpppcNNNNNNc
fcs(D)  cpp     : V/V^V^VaVcVcVlVrpv-vgppv-vgvgnnnv-v/PCPPCpppcNNNNNNc
fcs(D') python  : V/V^V^VaVcVcVlVrpv-vgppv-vgvgnnnv-v/PCPPCpppcNNNNNNc
```

- `fcs(D) == fcs(D')`: **True**
- `D ≅ D'`: **False**
- labels equal: **True**
- `D`  : 17 nodes, 19 edges, {'ABS': 1, 'COS': 2, 'DIV': 2, 'LOG': 1, 'NEG': 3, 'POW': 2, 'SQRT': 1, 'SUB': 3, 'VAR': 2}
- `D'` : 17 nodes, 19 edges, {'ABS': 1, 'COS': 2, 'DIV': 2, 'LOG': 1, 'NEG': 3, 'POW': 2, 'SQRT': 1, 'SUB': 3, 'VAR': 2}

- edges `D` : `[(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 9), (0, 11), (1, 12), (1, 13), (1, 14), (7, 1), (7, 8), (7, 9), (7, 10), (7, 14), (13, 14), (14, 15), (14, 16)]`
- edges `D'`: `[(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 10), (1, 11), (1, 12), (3, 1), (3, 7), (3, 12), (3, 13), (3, 14), (10, 12), (12, 15), (12, 16)]`
- labels `D` : `['VAR', 'VAR', 'COS', 'LOG', 'SQRT', 'DIV', 'POW', 'POW', 'NEG', 'COS', 'NEG', 'ABS', 'NEG', 'SUB', 'SUB', 'DIV', 'SUB']`
- labels `D'`: `['VAR', 'VAR', 'DIV', 'POW', 'POW', 'ABS', 'COS', 'COS', 'LOG', 'SQRT', 'SUB', 'NEG', 'SUB', 'NEG', 'NEG', 'SUB', 'DIV']`

- SymPy `D` : `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`
- SymPy `D'`: `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`

### Corpus index 3687 (k = 17, 1 variable(s))

```
source string   : CCVlcV/nvkVkV+viVsWPCVev/NVcVlvrCVavcvrpvavl
fcs(D)  python  : V+V/VkVlVsppv/vcvivkvrvrpvavlpvenvavcvlPPPPPPPPcpc
fcs(D)  cpp     : V+V/VkVlVsppv/vcvivkvrvrpvavlpvenvavcvlPPPPPPPPcpc
fcs(D') python  : V+V/VkVlVsppv/vcvivkvrvrpvavlpvenvavcvlPPPPPPPPcpc
```

- `fcs(D) == fcs(D')`: **True**
- `D ≅ D'`: **False**
- labels equal: **True**
- `D`  : 18 nodes, 19 edges, {'ABS': 2, 'ADD': 1, 'CONST': 2, 'COS': 2, 'DIV': 2, 'EXP': 1, 'INV': 1, 'LOG': 3, 'SIN': 1, 'SQRT': 2, 'VAR': 1}
- `D'` : 18 nodes, 19 edges, {'ABS': 2, 'ADD': 1, 'CONST': 2, 'COS': 2, 'DIV': 2, 'EXP': 1, 'INV': 1, 'LOG': 3, 'SIN': 1, 'SQRT': 2, 'VAR': 1}

- edges `D` : `[(0, 1), (0, 2), (0, 4), (0, 5), (0, 7), (1, 2), (1, 8), (2, 3), (2, 6), (2, 9), (2, 12), (2, 14), (2, 15), (4, 16), (4, 17), (8, 2), (8, 10), (8, 11), (8, 13)]`
- edges `D'`: `[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 12), (3, 13), (4, 2), (4, 14), (14, 2), (14, 15), (14, 16), (14, 17)]`
- labels `D` : `['VAR', 'LOG', 'DIV', 'CONST', 'CONST', 'ADD', 'INV', 'SIN', 'EXP', 'DIV', 'COS', 'LOG', 'SQRT', 'ABS', 'COS', 'SQRT', 'ABS', 'LOG']`
- labels `D'`: `['VAR', 'ADD', 'DIV', 'CONST', 'LOG', 'SIN', 'DIV', 'COS', 'INV', 'CONST', 'SQRT', 'SQRT', 'ABS', 'LOG', 'EXP', 'ABS', 'COS', 'LOG']`

- SymPy `D` : `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`
- SymPy `D'`: `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`

### Corpus index 7403 (k = 18, 1 variable(s))

```
source string   : ccv/VenWViviVlv*v-PcvgVkVgV-cVapv-vgccVlNVeWVaVa
fcs(D)  python  : V/VeViVlpv-vavgvkvlpv*v-vgvipv-vgPPPPPVaVaVePcnnnc
fcs(D)  cpp     : V/VeViVlpv-vavgvkvlpv*v-vgvipv-vgPPPPPVaVaVePcnnnc
fcs(D') python  : V/VeViVlpv-vavgvkvlpv*v-vgvipv-vgPPPPPVaVaVePcnnnc
```

- `fcs(D) == fcs(D')`: **True**
- `D ≅ D'`: **False**
- labels equal: **True**
- `D`  : 19 nodes, 20 edges, {'ABS': 3, 'CONST': 1, 'DIV': 1, 'EXP': 2, 'INV': 2, 'LOG': 2, 'MUL': 1, 'NEG': 3, 'SUB': 3, 'VAR': 1}
- `D'` : 19 nodes, 20 edges, {'ABS': 3, 'CONST': 1, 'DIV': 1, 'EXP': 2, 'INV': 2, 'LOG': 2, 'MUL': 1, 'NEG': 3, 'SUB': 3, 'VAR': 1}

- edges `D` : `[(0, 1), (0, 2), (0, 3), (0, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 15), (2, 1), (2, 4), (2, 6), (2, 7), (2, 8), (3, 1), (3, 13), (3, 14), (15, 16), (15, 17), (15, 18)]`
- edges `D'`: `[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 1), (2, 10), (2, 11), (2, 12), (2, 13), (3, 1), (3, 14), (3, 15), (9, 16), (9, 17), (9, 18)]`
- labels `D` : `['VAR', 'DIV', 'EXP', 'INV', 'INV', 'LOG', 'MUL', 'SUB', 'NEG', 'CONST', 'NEG', 'SUB', 'ABS', 'SUB', 'NEG', 'LOG', 'EXP', 'ABS', 'ABS']`
- labels `D'`: `['VAR', 'DIV', 'EXP', 'INV', 'LOG', 'SUB', 'ABS', 'NEG', 'CONST', 'LOG', 'MUL', 'SUB', 'NEG', 'INV', 'SUB', 'NEG', 'ABS', 'ABS', 'EXP']`

- SymPy `D` : `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`
- SymPy `D'`: `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`

### Corpus index 7771 (k = 13, 1 variable(s))

```
source string   : vapVgV^V+ppVkViVcv^ncNCVsNCVrvgNCvgVk
fcs(D)  python  : V+V^VaVcVgViVkppv^nvgvgpppvsnvrnvkPPPPcpcpc
fcs(D)  cpp     : V+V^VaVcVgViVkppv^nvgvgpppvsnvrnvkPPPPcpcpc
fcs(D') python  : V+V^VaVcVgViVkppv^nvgvgpppvsnvrnvkPPPPcpcpc
```

- `fcs(D) == fcs(D')`: **True**
- `D ≅ D'`: **False**
- labels equal: **True**
- `D`  : 14 nodes, 16 edges, {'ABS': 1, 'ADD': 1, 'CONST': 2, 'COS': 1, 'INV': 1, 'NEG': 3, 'POW': 2, 'SIN': 1, 'SQRT': 1, 'VAR': 1}
- `D'` : 14 nodes, 16 edges, {'ABS': 1, 'ADD': 1, 'CONST': 2, 'COS': 1, 'INV': 1, 'NEG': 3, 'POW': 2, 'SIN': 1, 'SQRT': 1, 'VAR': 1}

- edges `D` : `[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (3, 8), (7, 8), (7, 9), (8, 11), (8, 12), (9, 8), (9, 10), (10, 8), (10, 13)]`
- edges `D'`: `[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (2, 8), (4, 8), (4, 11), (8, 9), (8, 10), (11, 8), (11, 12), (12, 8), (12, 13)]`
- labels `D` : `['VAR', 'ABS', 'NEG', 'POW', 'ADD', 'CONST', 'INV', 'COS', 'POW', 'SIN', 'SQRT', 'NEG', 'NEG', 'CONST']`
- labels `D'`: `['VAR', 'ADD', 'POW', 'ABS', 'COS', 'NEG', 'INV', 'CONST', 'POW', 'NEG', 'NEG', 'SIN', 'SQRT', 'CONST']`

- SymPy `D` : `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`
- SymPy `D'`: `<unavailable: ImportError: cannot import name 'labeled_dag_to_sympy' from 'isalsr.adapters.sympy_adapter' (/home/mpascual/research/code/IsalSR/src/isalsr/adapters/sympy_adapter.py)>`

