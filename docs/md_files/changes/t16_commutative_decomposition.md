# T16 — Commutative decomposition at the adapter boundary, and the `Neg`/`Inv` sharing experiment

**Date**: 2026-07-30 · **Ticket**: `.claude/notes/review/tasks/T16-alphabet-mismatch.md`
**Reviewer comment**: R2.3 · **Decision owner**: Ezequiel (direction), Mario (sharing)
**Status**: implemented, validated, `SHARE_DECOMPOSED_UNARY = False` shipped

---

## 1. What changed and why

The paper's instruction set (Definition 3.2, `methodology.tex:93-119`) declares
twelve label characters `𝓛 = {+, *, g, i, s, c, e, l, r, ^, a, k}`. It contains
**no `-` and no `/`**. Subtraction and division are supposed to enter through a
commutative decomposition inspired by GraphSR:

```
x − y  =  Add(x, Neg(y))
x / y  =  Mul(x, Inv(y))
```

`Pow` is the only non-commutative operation with no exact commutative
decomposition, so it keeps a dedicated instruction. Under that alphabet `Pow` is
the sole operand-order-sensitive node, which is exactly what makes Definition
3.9(iv) sound as written.

The experiment adapters emitted `NodeType.SUB` and `NodeType.DIV` as primitive node
types anyway, and **61.1 % of production candidates contain them**. The implemented
alphabet was therefore 14 labels / 35 tokens against the stated 12 / 31.

The correction (T16 §5) rewrites `Sub` and `Div` **at the host → `LabeledDAG`
boundary only**, in the same layer as `_normalize_const_edges` (Critical Invariant
9). Three hard non-goals: the YAML host operator sets are untouched, `NodeType.SUB`
/ `DIV` and `BINARY_OPS` stay intact in `isalsr.core` (S2D must still decode legacy
`V-` / `V/` strings), and nothing is decomposed inside the canonicaliser.

**Implementation**: `experiments/models/commutative_encoding.py`, consumed inline by
`experiments/models/{bingo,udfs}/adapter.py`. Both adapters take keyword-only
`decompose: bool = True` and `share_unary: bool | None = None`.

---

## 2. The sharing question

When a host DAG shares a subexpression — say `a − b` and `c − b` both reading the
same `b` — the decomposition may either emit **one** `Neg(b)` with out-degree 2, or
**two** independent `Neg` nodes.

| | **Split** (no sharing) | **Shared** |
|---|---|---|
| Emitted nodes | one private `Neg`/`Inv` per `Sub`/`Div` | one per distinct operand |
| `k` growth | exactly `#Sub + #Div` | `≤ #Sub + #Div` |
| Map character | purely local node-splitting | partial common-subexpression elimination |
| Reverse adapter | `undecompose()` is total | `undecompose()` raises on out-degree > 1 |

Both are injective on host DAGs, so **dedup soundness does not discriminate between
them** — the discriminator has to be the measured effect on `k` and on the reduction
factor ρ, which is the paper's headline quantity.

T16 §5.1 initially *recommended* sharing, on the argument that `Neg(b)` denotes one
mathematical object and emitting it twice is a representational defect. Two
counter-considerations surfaced during implementation:

1. **The ticket's own `k` formula assumes non-sharing.** §5.1 states "each
   replacement adds exactly **one** node, so `k` increases by `#Sub + #Div`". That
   is the split formula; under sharing it is only an upper bound.
2. **Sharing is an inconsistent partial CSE.** The adapter is otherwise a faithful
   translation of the host's sharing structure: if Bingo's command array computes
   `sin(x)` in two rows, we emit two `SIN` nodes; if it computes `a − b` twice, we
   emit two `SUB` nodes. Merging only the `Neg`/`Inv` nodes *we* invent, while
   leaving the host's own duplicates alone, applies CSE to one node class and not
   the others.

So the choice was resolved by measurement rather than by argument.

---

## 3. Experimental protocol

**Script**: `experiments/scripts/measure_decomposition_impact.py`
**Invocation**: `--n 5000 --seed 42 --out <dir>` · engine `cpp` · runtime 5.0 s

Three encodings compared **paired, per DAG** (identical host graph, three
conversions):

| Key | Setting |
|---|---|
| `legacy` | `decompose=False` — the encoding the submitted results used |
| `split` | `decompose=True, share_unary=False` |
| `shared` | `decompose=True, share_unary=True` |

Hosts: Bingo (`AGraphGenerator`, `agraph_size=16`) and UDFS (`CompGraph`), both
with the production operator set. Canonicalisation via `fast_canonical_string`
(`mode="wl_only"`).

**Control**: the `legacy` Bingo run must reproduce T16 §2's 61.1 % of DAGs carrying
`Sub`/`Div`. Measured **59.40 %** — the generator matches production.

**Population caveat, which bounds every number below.** These are *randomly
generated* graphs, not evolved live-search populations. ρ here (Bingo 1.2960, UDFS
2.1505) is **not** production ρ (1.7931, 1.880), and `violated_pre` here (41.02 % /
75.06 %) is **not** T06's (85.88 % / 100 %), because rediscovery and orphan-`CONST`
rates are properties of evolution, not of random sampling. **This measurement
establishes direction and invariance, not magnitude.** Magnitudes come from Wave 1.

---

## 4. Results

### 4.1 The sharing comparison (the decision)

| Host | Quantity | `split` | `shared` | Δ |
|---|---|---|---|---|
| Bingo | mean `k` | 6.72 | 6.68 | −0.04 (0.6 %) |
| Bingo | ρ | 1.2960 | 1.2960 | **0.0000** |
| Bingo | distinct strings | 3,858 | 3,858 | 0 |
| UDFS | mean `k` | 3.99 | 3.98 | −0.01 |
| UDFS | ρ | 2.1805 | 2.1815 | +0.0010 (0.05 %) |
| UDFS | distinct strings | 2,293 | 2,292 | −1 |

Sharing buys **0.6 % on `k`** and **0.05 % on ρ**, and exactly nothing on Bingo ρ.

### 4.2 Decomposition vs legacy

| | Bingo legacy | Bingo split | UDFS legacy | UDFS split |
|---|---|---|---|---|
| mean `k` | 5.47 | **6.72** (+1.25, +22.9 %) | 3.27 | **3.99** (+0.72, +22.0 %) |
| mean canonical length | 21.2 | **26.9** (+27 %) | 11.1 | **13.5** (+22 %) |
| distinct strings | 3,858 | **3,858** | 2,325 | **2,293** |
| ρ | 1.2960 | **1.2960** | 2.1505 | **2.1805** (+1.4 %) |
| canon cost ms/DAG | 0.0079 | 0.0099 (**+24.6 %**) | 0.0059 | 0.0066 (**+10.8 %**) |
| `violated_pre` | 41.02 % | **41.02 %** | 75.06 % | **75.06 %** |
| `violated_post` | 0 | **0** | 0 | **0** |
| round-trip | 100 % | **100 %** | 100 % | **100 %** |
| completeness (20 perms × 100 DAGs) | — | **100/100** | — | **100/100** |
| false merges | 5 (artifact, §5) | 5 | 0 | **0** |

---

## 5. Findings that are not about sharing

**Bingo ρ is exactly encoding-invariant.** 3,858 distinct strings under all three
encodings. On a host with no native `neg`/`inv`, the decomposition is a bijection on
isomorphism equivalence classes.

**UDFS ρ increases 1.4 %, and the mechanism is identifiable.** UDFS's search always
samples `neg` and `inv` natively — its YAML `operator_set` is dead configuration
(`vendor/DAG_search/dag_search.py:1226-1227`). So `sub_l(a,b)` and `+(a, neg(b))`
previously produced **different** canonical strings and now produce the same one.
That merge is sound: the two denote the same expression, and M10 confirms **0 false
merges for UDFS**. The corrected alphabet removes a spurious distinction the old
encoding carried. This also explains the label-fraction gap: 52.00 % of UDFS DAGs
carry `Sub`/`Div` but 64.12 % carry `Neg`/`Inv`, the difference being host-native
ones.

**Reachability invariance holds at DAG level, not at node level.** Confusion matrix
vs legacy is `FP=0, FN=0` on both hosts and both decomposed encodings. But the
node-level violating count rises (Bingo 0.46 → 0.60, UDFS 1.52 → 1.67). The reason:
decomposing `Sub(a,b)` where `a` is reachable from a variable and `b` is not creates
a *new* violating `Neg(b)`, where `Sub` itself was not violating — but `b` is then
already a non-variable node with no variable ancestor, so the DAG was **already**
violating through `b`. T06's DAG-level headline survives; its k-stratification does
not and must be rebuilt from Wave 1.

**`Div` and `Mul(a, Inv(b))` are not equivalent in the guarded regime.** With
`|b| ≤ 1e-10`, `_protected_div` returns `1.0` while `Mul(a, Inv(b))` returns `a`;
they coincide only at `a == 1`, and **no multiplicative guard can reconcile them** —
the `Div` fallback discards `a` and the `Mul` form structurally cannot. Guarded
fraction **2.01 %**. Outside the guard, max absolute error **0.125**, ~1 ULP at the
`_MAX_VALUE` clamp of 1e15 — IEEE 754 rounding, not a semantic difference. This
touches **no reported number**: fitness is computed by the host on the host's own
representation and the runners cache `canon_hash → fitness` without ever calling
`evaluate_dag`. The false docstring claiming equivalence was corrected in
`src/isalsr/core/dag_evaluator.py`; the semantics were deliberately left alone.

**Decomposition reduces unevaluable DAGs by 19 points.** `x − x` and `x / x`
previously reached the DAG as a **one-input** `Sub`/`Div`, because
`LabeledDAG.add_edge` silently rejects the duplicate edge. Under decomposition they
become `Add(x, Neg(x))` and `Mul(x, Inv(x))` with two distinct in-edges, and now
evaluate to the correct `0` and `1`. Measured over n=2,000: `evaluate_dag` fails on
**52.1 %** of legacy DAGs and **33.3 %** of decomposed ones.
**Coverage caveat**: the semantic-equivalence block compares only where *both*
encodings evaluate, which systematically excludes the `x−x` cases decomposition
changes most. It is a biased subsample and must not be quoted as full coverage.

**The 5 Bingo "false merges" are a measurement artifact.**
`fast_canonical_string` returns `''` for every `k=0` DAG regardless of `m`, so two
variable-only DAGs with `m=1` and `m=2` collide. This is correct behaviour: the
canonical string encodes the *instruction sequence*, and `m` is a parameter of S2D,
not part of the string. `fcs` is a complete invariant **for fixed `m`**, and `m` is
fixed per problem in production. The count is identical (5/5/5) across all three
encodings, so decomposition introduces nothing. Filed for T07 as a scope
clarification for the completeness theorem.

---

## 6. Decision

**`SHARE_DECOMPOSED_UNARY = False`** — do not share. Taken by Mario, 2026-07-30, on
the §4.1 numbers.

Sharing's measured benefit (0.6 % on `k`, 0.05 % on ρ, 0.0000 on Bingo ρ) does not
justify:

- breaking `undecompose()`, which is total under split and raises on shared
  wrappers, and which the reverse Bingo adapter needs because Bingo's opcode table
  has no `Neg`/`Inv` entry;
- introducing a partial CSE that merges the `Neg`/`Inv` nodes the adapter invents
  while leaving the host's own duplicated `Sin`/`Sub` nodes untouched;
- diverging from T16 §5.1's own `k` formula, which the continuity table in T09/T10
  is written against.

The `shared` path is retained behind `share_unary=True` and is covered by the test
suite, so the experiment is reproducible and the decision is revisitable.

---

## 7. Validation

| Gate | Result |
|---|---|
| `pytest tests/unit/test_{bingo_adapter,udfs_adapter,commutative_encoding}.py` | 122 passed |
| Rest of `tests/unit/` | 4,662 passed, 5 skipped |
| `tests/integration/` | 458 passed |
| `tests/property/` | 16 passed |
| `ruff` + `mypy --strict` | clean |

**Mutation testing**, because a wrong operand order is invisible downstream — it
still canonicalises and still deduplicates, the run simply measures the wrong thing:

| Injected defect | Tests that caught it |
|---|---|
| Negate the **first** operand instead of the second | 35 / 122 |
| Drop UDFS's `REVERSED_OPS` (`sub_r`/`div_r` lose orientation) | 7 |
| Add `POW` to the decomposition table | 7 |

---

## 8. Gate G9 — confirmed on a Picasso compute node

Unit tests prove the adapters *can* decompose; they do not prove the code a SLURM
array executes *does*. `experiments/scripts/verify_alphabet_gate.py` instruments the
real path — orchestrator, runner, monkey-patched evaluation hook, deduplicator — and
reports the label histogram of every DAG handed to the canonicaliser.

**Local**: all 10 production configs pass, ~130,000 DAGs, zero `Sub`/`Div`, zero
`-`/`/` in any canonical string.

**Picasso** (job 1692451, `sd`-family CPU node, `COMPLETED`, 1 m 26 s, native C++
extension confirmed loaded — not a silent pure-Python fallback):

| Config | DAGs at canonicaliser | Order-sensitive binary present |
|---|---|---|
| `bingo_nguyen` / Nguyen-1 | 5,551 | NONE |
| **`bingo_hard` / Pagie-1** | 5,551 | **`['POW']`** — Pow survives |
| `udfs_nguyen` / Nguyen-1 | 225 | NONE |
| `udfs_hard` / Keijzer-6 | 461 | NONE |

**Verdict: PASS.**

Two environment failures preceded this and are recorded so they are not rediscovered:

1. `bingo` imports `mpi4py` at module scope (`bingo/util/log.py:14`), which
   `dlopen()`s libmpi at import. **libmpi is not in the conda env**, so the worker
   must `module load openmpi_gcc/4.1.5_gcc1520` (the production workers probe a
   module list for the same reason). Without it the job dies with
   `RuntimeError: cannot load MPI library`.
2. `conda activate isalsr` does not take on this cluster — it leaves the base py39
   interpreter on `PATH`. Use the absolute interpreter
   `${CONDA_PREFIX_ABS}/bin/python`.

Both failures made the gate report **FAIL**, correctly: it refuses to pass on zero
observed DAGs, so an environment break cannot be mistaken for a clean alphabet.

Reproduce: `bash slurm/alphabet_gate/launcher.sh` (add `--test-only` to probe,
`--dry-run` to preview). Runtime ~90 s; wallclock capped at 20 min.

---

## 9. Related

- `.claude/notes/review/tasks/T16-alphabet-mismatch.md` — the ticket, §10 work log
- `.claude/notes/review/tasks/T06-reachability-failure-rate.md` — DAG-level rates survive; k-stratification does not
- `.claude/notes/review/tasks/T07-theorem-foundation.md` — D-2 and D-3 evaporate under this branch
- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — gate G9, Wave 1 blocking
- `experiments/scripts/measure_decomposition_impact.py` — the measurement
