# T16 — Σ_SR as defined vs Σ_SR as implemented: `Sub` and `Div`

| Field | Value |
|---|---|
| Reviewer comments touched | **R2.3** (Σ_SR vs host operator set — currently filed under T09 as presentational; this ticket argues it is not) |
| Type | Specification/implementation mismatch → possible full re-execution |
| Owner | **Mario** (measurement, code) · **Ezequiel** (alphabet definition, Def 3.2 / Table 1 / Def 3.9(iv)) |
| Depends on | — (independent of compute; the *decision* gates a re-run) |
| Blocks | T07 (completeness (⇐)), T09 (R2.3), potentially **T02** (if the code is aligned to the paper, every number is recomputed) |
| Status | **OPEN** — mismatch confirmed and quantified 2026-07-29. Direction chosen by Mario (align code to paper), **pending Ezequiel's confirmation**. |
| Opened | 2026-07-29, by Mario, from a T07 finding |

---

## 1. The mismatch

**The paper defines an alphabet that excludes `Sub` and `Div`. The implementation
uses one that includes them, and 61.1 % of production candidate DAGs contain
them.**

`methodology.tex:93-119`, Definition 3.2 (IsalSR Instruction Set Σ_SR):

> Let `𝓛 = {+, *, g, i, s, c, e, l, r, ^, a, k}` be the set of label characters
> (matching the label column of Table 1). … The vocabulary contains 7
> single-character tokens and 24 compound tokens (2×|𝓛|), totaling **31 tokens**.

Twelve labels: `Add, Mul, Neg, Inv, Sin, Cos, Exp, Log, Sqrt, Pow, Abs, Const`.
**No `-`. No `/`.** `grep` for `texttt{-}` / `texttt{/}` in `methodology.tex`
returns **0 matches**.

The paper then argues the point explicitly (`methodology.tex:121-135`):

> The two unary operators `Neg` (label `g`, −x) and `Inv` (label `i`, 1/x)
> implement subtraction and division through a commutative decomposition
> inspired by GraphSR: `x − y = Add(x, Neg(y))`, `x / y = Mul(x, Inv(y))`.
> Both `Add` and `Mul` are commutative, so the inputs to these nodes are
> interchangeable and **no operand-order tracking is required**.

Under that alphabet, `Pow` is the only non-commutative operation, and
Definition 3.9(iv) — which constrains operand order for `Pow` alone — is
**correct as written**. So is Rule 1's prose.

**But the adapters emit `Sub` and `Div` as primitive node types.**
`experiments/models/bingo/adapter.py:27-40`, `BINGO_OP_TO_ISALSR`:
`3 → NodeType.SUB`, `5 → NodeType.DIV`. Same for UDFS
(`udfs/adapter.py:37-42`, which additionally distinguishes `sub_l`/`sub_r`
and `div_l`/`div_r`). And `BINARY_OPS = {SUB, DIV, POW}` in
`src/isalsr/core/node_types.py`, with label characters `-` and `/`.

So the implemented alphabet is **14 labels / 35 tokens** (7 + 2×14) against the
stated **12 labels / 31 tokens**.

---

## 2. Measurement

5,000 AGraphs generated with the production operator set verbatim from
`experiments/configs/bingo_nguyen.yaml`
(`["+", "-", "*", "/", "sin", "cos", "exp", "log"]`), converted through
`experiments/models/bingo/adapter.py`:

| Label | VAR | COS | SIN | ADD | EXP | **SUB** | **DIV** | LOG | MUL | CONST |
|---|---|---|---|---|---|---|---|---|---|---|
| count | 7,008 | 3,228 | 3,220 | 3,187 | 3,183 | **3,168** | **3,168** | 3,162 | 3,157 | 2,241 |

**DAGs containing at least one `Sub` or `Div`: 3,054 / 5,000 = 61.1 %.**
`Sub` and `Div` are as frequent as `Add` and `Mul`.

Every production config lists `-` and `/`:
`experiments/configs/{bingo,udfs}_{nguyen,feynman,hard,cherrypicked,roundoff}.yaml`.
`OperationSet.commutative()` — the factory that would realise the paper's
alphabet — is **not used by any production config**.

---

## 3. Why it matters

1. **The theorems quantify over the wrong space.** Theorems 3.13/3.14/3.15 range
   over labeled DAGs whose node labels lie in `𝓛`. 61.1 % of the DAGs the
   experiments canonicalise contain labels outside `𝓛`, so they are not elements
   of the space the theorems describe.
2. **Definition 3.9(iv) is unsound for the implemented alphabet.** For `Sub` and
   `Div` the identity bijection satisfies conditions (i)–(iii), and (iv) is
   vacuous because the node is not `Pow`. So `Sub(x₁,x₂) ≅ Sub(x₂,x₁)` per the
   definition, while the implementation correctly gives them different canonical
   strings. Measured, C++ backend, identical edge sets:

   | Op | `σ=(x₁,x₂)` | `σ=(x₂,x₁)` | strings differ |
   |---|---|---|---|
   | `Sub` | `V-PnC` | `pv-nC` | yes |
   | `Div` | `V/PnC` | `pv/nC` | yes |

   This falsifies the **(⇐)** direction of Theorem 3.15 *as stated*, on a
   three-node counterexample. **The code is right** (`x₁−x₂ ≠ x₂−x₁`); the
   definition is too coarse. Detail: `T07-appendix/const_normalization_equivariance.md` §2.
3. **The paper advertises a property the experiments do not exercise.** The
   commutative-encoding paragraph is a selling point — "no operand-order tracking
   is required" — and the reported runs did not use it. A reviewer who checks the
   configs finds this.
4. **The token count `31` is wrong** for the implementation (35).
5. **Rule 1's stated scope is wrong** for the implementation: prose says `Pow`,
   code covers `BINARY_OPS`. The Table 3 caption ("binary non-commutative node")
   already matches the code, so the manuscript is internally inconsistent too.

**No reported number is wrong because of this.** The implementation is
self-consistent and stricter than the definition; it distinguishes DAGs the
definition would merge, which is the safe direction. What is wrong is the
*description*.

---

## 4. The two directions

### 4a. Align the paper to the code — cheap, no recompute

Add `Sub` (`-`) and `Div` (`/`) to `𝓛`; correct "31 tokens" to 35 and Table 1;
widen Definition 3.9(iv) and Rule 1 to `{Sub, Div, Pow}`; demote the commutative
encoding to an available variant that the reported experiments did not use.

- **Cost**: manuscript only. **No number moves.** No re-execution.
- **Loss**: the "fully commutative alphabet" argument becomes an unused option
  rather than a described property of the runs. Weakens a selling point, and
  R2.3 gets answered with "we mis-described our alphabet".

### 4b. Align the code to the paper — expensive, changes every number

Make the adapters decompose: `a − b → Add(a, Neg(b))`, `a / b → Mul(a, Inv(b))`.

- **Restores every claim exactly as written.** `Pow` becomes the only
  non-commutative op, Definition 3.9(iv) and Rule 1's prose become correct
  unchanged, Theorem 3.15's (⇐) hole from §3.2 closes, and the commutative
  encoding becomes a real property of the experiments rather than an advertised
  option.
- **Cost: a full re-execution.** This is not a re-encoding — `x − y` becomes
  **two nodes instead of one**, so:
  - `k` (internal node count) changes for 61.1 % of candidates;
  - **ρ and the reduction factor change** (different node counts, different
    collision structure);
  - every **k-stratified** table changes (T06's violation-rate profile, T02's
    overhead-by-k, the bottleneck-type analysis);
  - canonicalisation cost per DAG changes (larger DAGs);
  - the search-space-reduction figures change.
- **Unknown, and worth knowing**: whether ρ goes up or down. More nodes means
  more structural variety (ρ down) but also more opportunities for isomorphic
  rediscovery (ρ up). **This is not predictable a priori and must be measured.**

**Decision by Mario, 2026-07-29: pursue 4b**, accepting a full re-run, on the
grounds that it makes the paper's central claims true as written rather than
patching the description. **Pending Ezequiel's confirmation** — it is his
alphabet definition and his theorems, and the re-run cost lands on the T02
campaign schedule.

---

## 5. Work specification

### 5.1 Decide (blocking)
Confirm 4a or 4b with Ezequiel. Everything below assumes 4b; under 4a the work
is a manuscript edit only and this ticket closes into T09/R2.3.

**What each branch costs inside T07 is enumerated in
`T07-theorem-foundation.md` §7bis.3** — a per-item table of which manuscript
edits become mandatory under 4a and which evaporate under 4b (Def 3.9(iv), Rule
1's prose scope, Def 3.2's label set and token count). T07 §7bis.2 lists the
proof work that is unaffected either way, so Ezequiel is not blocked on this
decision for the bulk of Lemma 3.14/A.2.

### 5.2 If 4b — implementation
- Adapters decompose `Sub`/`Div` into `Add`+`Neg` / `Mul`+`Inv`
  (`bingo/adapter.py`, `udfs/adapter.py`; UDFS additionally has `sub_l`/`sub_r`,
  `div_l`/`div_r`, so operand order must be preserved through the decomposition).
- Decide whether `NodeType.SUB`/`DIV` remain in the core at all. Keeping them
  costs nothing and preserves S2D's ability to decode legacy strings; removing
  them from the *adapters* is what matters.
- `BINARY_OPS` then reduces to `{POW}` **for adapter-produced DAGs only** —
  do not narrow the core constant, since S2D can still decode `V-`/`V/`.

### 5.3 If 4b — validation before any re-run
- Round-trip fidelity and completeness on the decomposed DAGs.
- **Semantic equivalence**: `eval(D) == eval(decompose(D))` on a large corpus,
  within floating-point tolerance. Protected `Inv` (1/x for |x|>ε, else 1) means
  `a / b` and `Mul(a, Inv(b))` are **not** numerically identical near zero —
  this must be quantified, not assumed. **It is the main scientific risk of 4b.**
- The k-distribution shift, reported explicitly so the continuity table can
  explain why numbers moved.

### 5.4 If 4b — re-execution scope
Feeds `EXECUTION-PLAN.md`. At minimum the IsalSR arm on the full 50-problem
suite; the baseline arm is unaffected (it never invokes the adapter). Coordinate
with T02 Wave 1 — **this changes what Wave 1 should run**, so it must be settled
before Wave 1 launches, not after.

---

## 6. Acceptance criteria

- **AC-0.** §7 work log filled as the work proceeds.
- **AC-1.** Direction 4a/4b confirmed with Ezequiel and recorded.
- **AC-2.** The mismatch is stated in the manuscript one way or the other:
  either `𝓛` gains `-` and `/` (4a) or the adapters stop emitting them (4b).
- **AC-3.** The token count in Definition 3.2 matches the implementation.
- **AC-4.** Definition 3.9(iv) and Rule 1's prose are sound for whichever
  alphabet is chosen.
- **AC-5.** *(4b only)* Semantic-equivalence measurement of the decomposition,
  including the protected-`Inv` near-zero regime, with a quantified error bound.
- **AC-6.** *(4b only)* Re-execution scope agreed and folded into
  `EXECUTION-PLAN.md`; the k-distribution shift documented for the continuity
  table.
- **AC-7.** R2.3's answer in T09 is consistent with this ticket.
- **AC-8.** §8 filled.

---

## 7. Work log

### 2026-07-29 — Opened from T07

Found while drafting the numbered definition of `normalize_const_creation` for
R1.3. The chain: an agent testing Rule 1 non-exclusion reported that the
implementation applies Rule 1 to all of `BINARY_OPS = {SUB, DIV, POW}` while the
manuscript prose says `Pow`. Chasing that divergence into Definition 3.9(iv)
produced the three-node `Sub`/`Div` counterexample to Theorem 3.15's (⇐)
direction. Chasing *that* into Definition 3.2 produced the real finding: the
paper's alphabet has no `-` and no `/` at all.

**The initial framing was wrong and is recorded so it is not repeated.** I first
reported this to Mario as "Definition 3.9(iv) is wrong — it omits Sub and Div".
That inverts the intent. Mario correctly pointed out that the commutative
encoding was the *design*, and that `Pow` was meant to be the only
order-sensitive operation. The definition is right for the alphabet the paper
declares; it is the **implementation** that silently uses a larger one. Same
observable symptom, opposite conclusion about what to fix — and the two
conclusions lead to very different amounts of work.

**Measured rather than assumed**: 61.1 % of production-configured Bingo
candidates contain `Sub` or `Div` (5,000 AGraphs, §2). This is not a corner case.

**Connection to R2.3.** The README maps R2.3 ("Σ_SR vs host operator set") to
T09 as a numerical-consistency item. On this evidence R2.3 is pointing at a real
specification gap, not a presentational one, and T09's answer must defer to
whichever direction this ticket takes.

---

## 8. Proposed answer

*(unfilled — blocked on AC-1)*
