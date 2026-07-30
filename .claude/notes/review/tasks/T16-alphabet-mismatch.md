# T16 — Σ_SR as defined vs Σ_SR as implemented: `Sub` and `Div`

| Field | Value |
|---|---|
| Reviewer comments touched | **R2.3** (Σ_SR vs host operator set) |
| Type | Specification/implementation mismatch → **confirmed full re-execution of the IsalSR arm** |
| Owner | **Mario** (implementation, validation, re-run) · **Ezequiel** (decision — **taken**, see §4) |
| Depends on | — |
| Blocks | **T02** (Wave 1 must run the corrected adapter), T04 (Naive-Hash must consume decomposed DAGs), T06 (k-stratification only), T07 (§7bis fork resolved), T09, T10 |
| Status | **IMPLEMENTED AND VALIDATED 2026-07-30.** Direction 4b. AC-0…AC-12 closed; AC-13 (R2.3 answer, T09) and AC-14 (§11) remain, both blocked on the Wave 1 re-run. Sharing decided: **do not share**. Write-up: `docs/md_files/changes/t16_commutative_decomposition.md`. **Wave 1 must launch against the corrected adapter — gate G9 in `EXECUTION-PLAN.md`.** |
| Opened | 2026-07-29, by Mario, from a T07 finding |
| Last updated | 2026-07-30 — Ezequiel's decision recorded; ticket rewritten as an implementation spec |

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

The paper argues the point explicitly (`methodology.tex:121-135`):

> The two unary operators `Neg` (label `g`, −x) and `Inv` (label `i`, 1/x)
> implement subtraction and division through a commutative decomposition
> inspired by GraphSR: `x − y = Add(x, Neg(y))`, `x / y = Mul(x, Inv(y))`.
> Both `Add` and `Mul` are commutative, so the inputs to these nodes are
> interchangeable and **no operand-order tracking is required**.

**The design intent, in Mario's words (2026-07-30), because it explains why `Inv`
is unary and why the alphabet is shaped this way:** every non-commutative
operation that *can* be expressed as a combination of commutative ones is
expressed that way. `Pow` is the only non-commutative operation with no exact
commutative decomposition, so it keeps a dedicated instruction. The alphabet is
therefore **minimal**, at the cost of needing more instructions per operation —
so canonical strings are **longer** and the instruction space differs.

Under that alphabet `Pow` is the only non-commutative operation, Definition
3.9(iv) is **correct as written**, and so is Rule 1's prose.

**But the adapters emit `Sub` and `Div` as primitive node types.**
`experiments/models/bingo/adapter.py:27-40`, `BINGO_OP_TO_ISALSR`:
`3 → NodeType.SUB`, `5 → NodeType.DIV`. Same for UDFS
(`udfs/adapter.py:37-42`, which additionally distinguishes `sub_l`/`sub_r` and
`div_l`/`div_r`). `BINARY_OPS = {SUB, DIV, POW}` in
`src/isalsr/core/node_types.py`, with label characters `-` and `/`.

So the implemented alphabet is **14 labels / 35 tokens** (7 + 2×14) against the
stated **12 labels / 31 tokens**, and the experiments emit **shorter strings than
the paper's alphabet would produce**.

**Root cause.** The commutative-decomposition design postdates the first
experiment campaign. The core was updated correctly; the Bingo and UDFS adapters
were left consuming the earlier encoding, in which `Sub` and `Div` had their own
instructions.

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
`OperationSet.commutative()` — the factory that realises the paper's alphabet —
is used by **no** production config.

---

## 3. Why it matters

1. **The theorems quantify over the wrong space.** Theorems 3.13/3.14/3.15 range
   over labeled DAGs whose labels lie in `𝓛`. 61.1 % of the DAGs the experiments
   canonicalise carry labels outside `𝓛`.
2. **Definition 3.9(iv) is unsound for the implemented alphabet**, falsifying the
   (⇐) direction of Theorem 3.15 *as stated*, on a three-node counterexample
   (§4 of `T07-appendix/const_normalization_equivariance.md`).
3. **The paper advertises a property the experiments do not exercise** — "no
   operand-order tracking is required" is a selling point, and the reported runs
   did not use the alphabet that delivers it.
4. **The token count `31` is wrong** for the implementation (35).
5. **Rule 1's stated scope is wrong** for the implementation.

**No reported number is *wrong* because of this.** The implementation is
self-consistent and stricter than the definition: it distinguishes DAGs the
definition would merge, which is the safe direction. What is wrong is the
*correspondence between the paper and the code.*

---

## 4. The decision — 4b, align the code to the paper

**Ezequiel, 2026-07-30, in reply to Mario's cost analysis:**

> Mi opinión es que debes relanzar los experimentos para que todos usen el
> alfabeto definido en el artículo. Es importante que esté alineada la teoría con
> el código. En una revista como ésta, habrá que publicar el código fuente cuando
> acepten el artículo. Lo siguiente que va a hacer todo el mundo es pasarle el
> artículo y el código a un LLM, el cual se dará cuenta en menos de un segundo de
> cualquier incoherencia entre ambos. Así que en esto no tenemos mucho margen.

The rationale is the one that matters and it is not about this review round:
**TPAMI acceptance obliges us to publish the source, and the first thing any
reader will do is hand the paper and the code to an LLM, which finds a
paper/code divergence immediately.** A mismatch that survives to publication is
worse than a re-run now.

Direction **4a** (widen `𝓛` to include `-` and `/`, keep the numbers) is
therefore **rejected** and is recorded only so it is not re-proposed.

**Ezequiel's availability**: travelling with low availability **3–8 August 2026**.
He starts T07 when he can. Nothing in this ticket is blocked on him.

---

## 5. Implementation specification

### 5.1 The transformation

At the adapter boundary, when a host expression DAG is converted to a
`LabeledDAG`, replace each non-commutative binary node other than `Pow`:

```
SUB(a, b)  ->  ADD(a, NEG(b))     with NEG(b) a unary node whose only input is b
DIV(a, b)  ->  MUL(a, INV(b))     with INV(b) a unary node whose only input is b
POW(a, b)  ->  unchanged
```

`a` is the **first** operand and `b` the **second**; the negation or inversion
applies to `b` only. Each replacement adds exactly **one** node, so
`k` increases by `#Sub + #Div` per DAG.

**Operand order is the silent-failure risk.** Swapping `a` and `b` produces a
different expression that still canonicalises cleanly and still deduplicates
consistently, so no error surfaces — the run simply measures the wrong thing.
UDFS makes this sharper: it distinguishes `sub_l`/`sub_r` and `div_l`/`div_r`
(`udfs/adapter.py:37-42`), which encode *which* operand is which, so both
orientations must be mapped and both must be tested.

**Sharing — an explicit decision, not an incidental one.** If the host DAG shares
a subexpression, e.g. `a − b` and `c − b` both reading the same `b`, the
decomposition may either create one `NEG(b)` with out-degree 2 or two
independent `NEG` nodes.

> **Recommendation: share.** `NEG(b)` denotes one mathematical object, and
> emitting it twice would represent one subexpression as two distinct nodes,
> which is itself a representational defect and would inflate `k` and depress
> structural collisions. Whichever is chosen, **record the choice in §10 and test
> it**, because it moves `ρ`.

### 5.2 Where to change it — and three hard non-goals

**Change**: `experiments/models/bingo/adapter.py` and
`experiments/models/udfs/adapter.py`, in the host→`LabeledDAG` conversion only.

**Do not change** — each of these would invalidate the comparison rather than fix it:

1. **Do not touch the host operator sets in the YAML configs.** The host still
   searches over `{+, −, ×, ÷, sin, cos, exp, log}`. IsalSR is a representation
   layer at the *evaluation boundary*; changing what the host searches over
   changes the search itself and breaks the paired design. This is the single
   biggest cost lever and the easiest thing to get wrong: editing the configs
   would silently force a baseline re-run too.
2. **Do not narrow `BINARY_OPS` or remove `NodeType.SUB` / `NodeType.DIV` from
   the core.** S2D must still decode legacy `V-` / `V/` strings, the property
   corpora and the atlas contain them, and Rule 1 must still cover them for those
   inputs. Only the *adapters* stop emitting them.
3. **Do not decompose inside the canonicaliser.** The canonical map stays a pure
   function of the DAG it is handed (T07). Decomposition is a producer-side
   translation, in the same layer as `_normalize_const_edges`.

**Also update**: `T04`'s Naive-Hash baseline must serialise the **decomposed**
DAGs, or its comparison against IsalSR is run on a different object. Confirm
whichever code path it uses receives the adapter output.

### 5.3 Manuscript edits — now almost none

This is the payoff of 4b. Under this branch:

| Item | Action |
|---|---|
| Definition 3.2, `𝓛`, "31 tokens" | **unchanged, and now correct** |
| Definition 3.9(iv) (`Pow` only) | **unchanged, and now correct** — T07 D-2 evaporates |
| Rule 1 prose (`Pow`) | **unchanged** for the runs; optional one-sentence remark that the implementation covers `BINARY_OPS` because S2D still decodes legacy strings — T07 D-3 evaporates |
| Commutative-encoding paragraph `:121-135` | **unchanged, and true of the experiments for the first time** |
| R2.3's answer (T09) | must state that the host operator set and Σ_SR are different objects, and that the host's `−` and `÷` enter the representation through the decomposition |

---

## 6. What changes, and what provably does not

The implementing agent must not re-measure what is invariant, and must not
assume invariance where there is none.

### 6.1 Provably unchanged

- **The DAG-level reachability violation rate (T06's headline numbers).**
  `Neg` and `Inv` each have exactly one in-edge, from their operand, so they
  inherit that operand's ancestry: if the operand has a variable ancestor so does
  the new node, and if it does not, the DAG already violated the condition
  through that operand. Decomposition therefore **creates no new violating DAG
  and removes none**. T06's 85.88 % (Bingo), 100 % (UDFS), 0 % (S2D corpus and
  synthetic) survive as stated. *Verify this argument empirically before relying
  on it — it is a proof sketch, not a measurement.*
- **Fitness, and every fitness-derived metric**: R², NRMSE, solution recovery.
  Fitness is computed by the host on the host's own representation; the runners
  cache `canon_hash → fitness` and never call `evaluate_dag`. Verified: no
  reference to `evaluate_dag` or `dag_evaluator` anywhere in
  `experiments/models/`.
- **The baseline arm**, which never invokes the adapter.

### 6.2 Changes, and must be re-measured

- `k` for the 61.1 % of candidates containing `Sub` or `Div`, by `#Sub + #Div`.
- **Canonical string length**, upward. Anything reported about string length or
  about Levenshtein as a DAG-distance proxy.
- **`ρ` and the reduction factor**, in an unknown direction: more nodes means
  more structural variety (pushes `ρ` down) but also more opportunity for
  isomorphic rediscovery (pushes `ρ` up). **Not predictable a priori. Measure it.**
- Distinct-string counts, and every **k-stratified** table: T06's violation
  profile against `k`, T02's overhead-by-`k`, the bottleneck-type analysis.
- **Canonicalisation cost per DAG**, upward with DAG size, so the computational
  overhead axis moves (the Bingo 51 % figure).
- The **CPDT** results for reduction factor, since `ρ` changes. CPDT for R² is
  unaffected per §6.1.

---

## 7. Validation gates — all of these before any re-run

1. **Semantic equivalence.** For a large sample of host candidates, compare the
   host's own evaluation against `evaluate_dag` on the decomposed `LabeledDAG`,
   on identical inputs. Report the error distribution, not a pass/fail.
   - Read the actual protected definitions in `src/isalsr/core/dag_evaluator.py`
     and each host's protected operators first, and report them. `Inv` is
     protected (`1/x` guarded near zero) and each host's protected division has
     its own guard, so `a/b` and `Mul(a, Inv(b))` are **not** bit-identical in
     the guarded regime. Quantify the disagreement and the fraction of
     evaluations that land in it.
   - **This is a validation concern, not a production risk** — it does not touch
     reported fitness, per §6.1. The previous version of this ticket called it
     "the main scientific risk of 4b"; that was wrong, and the correction matters
     because it changes how much this gate should delay the re-run.
2. **Operand-order correctness**, per host and per orientation, including UDFS's
   `sub_l`/`sub_r` and `div_l`/`div_r`. A wrong orientation is invisible
   downstream, so this needs a direct test against known expressions.
3. **Dedup soundness under decomposition.** The property "two host expressions
   with different fitness never share a canonical string" must be re-established,
   not inherited: decomposition is a new map into the representation and could in
   principle merge expressions the host distinguishes. Sample host candidate
   pairs that share a canonical string and check their host fitness agrees.
4. **Round-trip and completeness** on decomposed DAGs: `D ≅ S2D(fcs(D), m)`, and
   isomorphic ⇒ same string under `permute_internal_nodes`.
5. **Engine equivalence** (T01): Python and C++ must agree on the decomposed
   population.
6. **The `k`-distribution shift**, reported explicitly, so the continuity table
   in T09/T10 can explain why numbers moved.
7. **Reachability invariance** (§6.1) confirmed empirically on both hosts.

---

## 8. Re-execution scope

Mario's framing, with `D1` the problem set submitted to TPAMI and `D2` the
reviewer-requested extension:

| | Originally planned | With this fix |
|---|---|---|
| Baseline | `D2` | `D2` *(unchanged — the baseline never invokes the adapter)* |
| IsalSR | `D2` | **`D1 + D2`** |
| Naive-Hash | `D1 + D2` | `D1 + D2` *(on decomposed DAGs, §5.2)* |

The added cost is therefore **IsalSR on `D1`**, because those results were
produced under the wrong alphabet. The C++ core (T01) makes this materially
cheaper than the original campaign.

Fold into `EXECUTION-PLAN.md` and **settle before T02 Wave 1 launches**, or
Wave 1 runs the wrong adapter.

---

## 9. Acceptance criteria

- **AC-0.** §10 work log filled as the work proceeds.
- **AC-1.** ~~Direction confirmed~~ **CLOSED 2026-07-30**: 4b, Ezequiel (§4).
- **AC-2.** Adapters emit no `Sub` or `Div`; `NodeType.SUB`/`DIV` and
  `BINARY_OPS` remain intact in the core (§5.2 non-goal 2).
- **AC-3.** Host operator sets in every YAML config are **byte-identical** to
  before (§5.2 non-goal 1). Show the diff is empty.
- **AC-4.** The sharing decision (§5.1) is recorded with its measured effect on
  `k` and `ρ`.
- **AC-5.** Semantic-equivalence measurement complete, with the protected-regime
  disagreement quantified and the guarded-evaluation fraction reported (§7.1).
- **AC-6.** Operand-order tests pass for both hosts and both orientations,
  including UDFS's `*_l`/`*_r` variants (§7.2).
- **AC-7.** Dedup soundness re-established on decomposed DAGs (§7.3).
- **AC-8.** Round-trip, completeness and engine equivalence hold on the
  decomposed population (§7.4, §7.5).
- **AC-9.** Reachability invariance confirmed empirically; T06's headline rates
  restated as unchanged, or corrected if the argument fails (§6.1).
- **AC-10.** `k`-distribution shift documented for the continuity table.
- **AC-11.** Re-execution scope folded into `EXECUTION-PLAN.md`; T02 Wave 1
  configured against the corrected adapter.
- **AC-12.** Naive-Hash (T04) confirmed to consume decomposed DAGs.
- **AC-13.** R2.3's answer in T09 is consistent with this ticket, and states that
  Σ_SR and the host operator set are different objects.
- **AC-14.** §11 filled.

---

## 10. Work log

### 2026-07-29 — Opened from T07

Found while drafting the numbered definition of `normalize_const_creation` for
R1.3. The chain: an agent testing Rule 1 non-exclusion reported that the
implementation applies Rule 1 to all of `BINARY_OPS = {SUB, DIV, POW}` while the
manuscript prose says `Pow`. Chasing that divergence into Definition 3.9(iv)
produced the three-node `Sub`/`Div` counterexample to Theorem 3.15's (⇐)
direction. Chasing *that* into Definition 3.2 produced the real finding: the
paper's alphabet has no `-` and no `/` at all.

**The initial framing was wrong and is recorded so it is not repeated.** It was
first reported as "Definition 3.9(iv) is wrong — it omits Sub and Div". That
inverts the intent: the commutative encoding was the *design*, and `Pow` was
meant to be the only order-sensitive operation. The definition is right for the
alphabet the paper declares; the **implementation** silently uses a larger one.
Same symptom, opposite conclusion about what to fix, and very different amounts
of work.

**Measured rather than assumed**: 61.1 % of production-configured Bingo
candidates contain `Sub` or `Div` (5,000 AGraphs, §2).

### 2026-07-30 — Decision taken (4b). Ticket rewritten as an implementation spec.

Ezequiel confirmed 4b on the publication-coherence argument in §4. AC-1 closed.
The ticket was rewritten from "open decision" to "implementable spec" so it can
be handed to an agent: §5 the transformation and its three non-goals, §6 the
invariance analysis, §7 the validation gates, §8 the re-run scope.

**Three substantive additions made while rewriting, none of them in the original
ticket.**

1. **The decomposition belongs in the adapters, and the YAML operator sets must
   not be touched.** Promoted to a hard non-goal (§5.2). Editing the configs is
   the intuitive reading of "make the experiments use the paper's alphabet" and
   it is wrong: it would change the host's search space and force a baseline
   re-run, converting a one-arm re-execution into a full one.
2. **The reachability violation rate is invariant** (§6.1), with the argument:
   `Neg`/`Inv` inherit their operand's ancestry, so no violating DAG is created
   or removed. If it holds empirically, T06's headline numbers survive Branch B
   untouched and only its `k`-stratification needs re-measuring.
3. **The protected-`Inv` concern was mis-classified.** The original AC-5 called it
   "the main scientific risk of 4b". It is not: `grep` over `experiments/models/`
   finds no use of `evaluate_dag` or `dag_evaluator`, so IsalSR's evaluator is
   never in the fitness path — the runners cache `canon_hash → fitness` from the
   host. The protected regime therefore matters for *validating* the
   decomposition (§7.1) and for dedup soundness (§7.3), not for any reported
   number.

**Sharing of decomposed `Neg`/`Inv` nodes** was not previously identified as a
decision at all. It changes `k` and `ρ`, so it is now §5.1 with a recommendation
and a requirement to record it.

### 2026-07-30 — Implementation plan (written before any code was touched)

**Ownership check.** AC-1 is closed by Ezequiel (§4), so the only remaining
decision in the ticket is the sharing question (§5.1). That is escalated to Mario
with a measurement rather than settled by fiat — see S4/D-B below. Everything else
is implementation and is drivable.

**Prior art found in the core, not previously named in this ticket.**
`src/isalsr/core/commutative.py` already implements exactly the §5.1
transformation: `to_commutative()` rewrites `SUB(a,b) -> ADD(a, NEG(b))` and
`DIV(a,b) -> MUL(a, INV(b))` off `ordered_inputs`, and `from_commutative()`
inverts it by pattern-matching `ADD(a, NEG(b))` when `NEG` has out-degree 1. It is
**not** used by either adapter. Two reasons it cannot simply be bolted onto the end
of the adapters:

1. **It raises on self-referencing binary ops.** Bingo can emit `[3, r, r]`
   (`x - x`) and `[5, r, r]` (`x / x`). `LabeledDAG.add_edge` rejects a duplicate
   edge (`labeled_dag.py:239-241`), and the Bingo adapter's
   `elif src2 is not None and src2 == src1: pass` branch (`bingo/adapter.py:164`)
   deliberately does not retry, so such a `SUB` node reaches the DAG with
   **one** input. `to_commutative` then raises
   `ValueError: SUB node N has 1 inputs, expected 2` (`commutative.py:75`).
2. **It costs a full DAG copy** (topological sort + node-by-node rebuild) on a code
   path that runs once per candidate inside the Wave-1 hot loop.

So the decomposition goes **inline** in each adapter, at the point where the binary
node would have been created. That is strictly cheaper, it has the operand indices
already in hand, and it removes failure mode 1 by construction: `SUB(a,a)` becomes
`ADD(a, NEG(a))`, which has two *distinct* in-edges and is therefore representable,
where today it degenerates to a one-input `SUB`. **That is a behaviour change beyond
the letter of §5.1 and it is a fix, not a regression** — it is recorded here so it
is not mistaken for drift, and it gets its own test.

**Design decisions taken here (D-A … D-F).**

| # | Decision | Rationale |
|---|---|---|
| D-A | Decomposition lives **inside** `agraph_to_labeled_dag` / `compgraph_to_labeled_dag`, not in a post-pass and not in the canonicaliser | §5.2 non-goal 3; also makes every one of the 4 production call sites (`bingo/isalsr_runner.py:296`, `bingo/translator.py:304`, `udfs/isalsr_runner.py:151`, `diversity_metrics.py:627`) inherit it for free, which is how AC-12 gets satisfied without touching T04 |
| D-B | Sharing of emitted `NEG`/`INV` — **measured under both settings, then put to Mario** | §5.1 calls it an explicit decision. Note the ticket's own §5.1 k-formula ("adds exactly **one** node… `k` increases by `#Sub + #Div`") is the **non-sharing** formula; under sharing the increase is only an upper bound. Both are injective, so AC-7 is not the discriminator |
| D-C | `SUB(a,a)` / `DIV(a,a)` decompose to the full two-operand form | Forced by D-A; see above |
| D-D | `ADD(a,a)`, `MUL(a,a)`, `POW(a,a)` keep today's single-edge behaviour | Out of scope; changing it would move numbers this ticket does not own. **Filed, not absorbed** — noted in §11 residual risk |
| D-E | `max_nodes` capacity raised in both adapters | `add_node` raises at the cap (`labeled_dag.py:211`); decomposition adds up to one node per `Sub`/`Div` |
| D-F | Ordering `build (decomposed) → ledger.record_pre → _normalize_const_edges` is preserved | Decomposition never changes any `CONST`'s in-degree (a `CONST` is only ever an edge *source*), so it commutes with Invariant 9. This is also what makes AC-9 directly measurable: `record_pre` now sees the decomposed DAG |

**Subtasks.**

| # | Deliverable | Owner | Verified by |
|---|---|---|---|
| S1 | UDFS op-set expansion facts (`-` → `sub_l` and/or `sub_r`?), protected-operator semantics on all three sides, reverse-adapter call sites, T04 DAG provenance | investigator | returned as data, cited to `file:line` |
| S2 | `decompose_binary` helper + both adapters rewired + reverse direction repaired | me | AC-2, AC-3 |
| S3 | Test suite: operand order × 2 hosts × 2 orientations, self-reference, semantic equivalence, round-trip, completeness, dedup soundness, engine equivalence | implementer | AC-5…AC-8, re-run by me |
| S4 | Measurement: `k`-shift, ρ under share/no-share, reachability invariance, on ≥5,000 real candidates per host | implementer | AC-4, AC-9, AC-10 |
| S5 | `EXECUTION-PLAN.md` + ticket + T09/T04 notification | me | AC-11…AC-14 |

### 2026-07-30 — Implementation landed and independently verified (S2, S3)

**What was built.** `experiments/models/commutative_encoding.py` (new): `emit_binary`,
`emit_unary`, `new_unary_cache`, `undecompose`, `contains_decomposed_unary`,
`extra_node_budget`. Both adapters rewired inline, each gaining keyword-only
`decompose: bool = True` and `share_unary: bool | None = None`.
`labeled_dag_to_agraph` now calls `undecompose` first, because Bingo's opcode table
has no `Neg`/`Inv` entry and would otherwise raise.

**Verification I ran myself, not the subagents' claims.**

| Check | Result |
|---|---|
| `pytest tests/unit/{test_bingo_adapter,test_udfs_adapter,test_commutative_encoding}.py` | **122 passed** |
| Rest of `tests/unit/` | 4,662 passed, 5 skipped |
| `tests/integration/` | 458 passed |
| `tests/property/` | 16 passed |
| `ruff` + `mypy` on every changed file | clean |
| AC-3: `git status --porcelain experiments/configs/` | empty; all 15 configs still list `-` and `/` |

**Mutation testing, because a wrong orientation is invisible downstream (§5.1).**
A test suite that passes proves nothing about a defect it cannot see, so all three
plausible silent failures were injected and the suite re-run:

| Mutant | Tests that caught it |
|---|---|
| Negate the **first** operand instead of the second | 35 / 122 |
| Drop UDFS's `REVERSED_OPS`, so `sub_r`/`div_r` lose their orientation | 7 |
| Add `POW` to the decomposition table | 7 |

**Numbers I measured independently** (400–1,000 randomly generated Bingo AGraphs,
production operator set; random rather than evolved, so rho here is *not*
comparable to the production 1.7931 and is not quoted as such):

- Mean canonical string length 14.41 -> **18.40** (+3.98, ~28 %), the direction §6.2
  predicts.
- `k` delta mean **+0.915**, range [0, 6].
- Canonicalisation failures on decomposed DAGs: **0 / 400**.
- No `-` or `/` character in any decomposed canonical string.
- `violated_post` = 0 in **both** encodings, zero per-DAG disagreements over 1,000
  DAGs. Note this confirms only the *post*-normalisation half of AC-9; the
  `violated_pre` half needs the ledger hook and is measured in S4.

**Three findings that change what this ticket and its neighbours assert.**

1. **`_protected_inv`'s docstring stated a falsehood, and AC-5 is the reason it
   surfaced.** It claimed "``MUL(a, INV(b))`` evaluates identically to
   ``DIV(a, b)`` for all b". False: with `|b| <= 1e-10`, `_protected_div` returns
   `1.0` while `MUL(a, INV(b))` returns `a * 1.0 = a`. They coincide only at
   `a == 1`, and **no multiplicative guard can reconcile them** — the `Div`
   fallback discards `a` and the `Mul` form structurally cannot. Docstring
   corrected in `src/isalsr/core/dag_evaluator.py`; **semantics deliberately left
   unchanged**, because changing `_protected_div` would alter `isalsr.core` for
   legacy strings, the property corpora and the atlas, and because §6.1 already
   establishes `evaluate_dag` is not in the fitness path.
2. **UDFS's YAML `operator_set` is dead configuration.** `UDFSConfig.to_dag_regressor_kwargs()`
   never forwards it and `DAGRegressor.__init__` has no such parameter; the search
   always samples every key of `config.NODE_ARITY`
   (`vendor/DAG_search/dag_search.py:1226-1227`), so `bin_ops` is *always*
   `['+','*','sub_l','sub_r','div_l','div_r']` and `un_ops` *always* includes `neg`
   and `inv`. Two consequences: §5.2 non-goal 1 is vacuous for UDFS (there is
   nothing to preserve), and **the manuscript's description of the UDFS operator
   set is wrong** — filed for T09/T11, not absorbed here.
3. **§6.1's invariance argument is right but is stated one level too coarsely.**
   Decomposing `Sub(a,b)` where `a` is reachable from a variable and `b` is not
   creates a **new** violating node `Neg(b)` where `Sub` itself was not violating.
   The DAG-level rate still survives, because `b` is then a non-variable node with
   no variable ancestor and so the DAG was *already* violating through `b`. So:
   **DAG-level rate invariant; node-level violating count non-decreasing.** T06
   reports DAG-level, so its headline survives — but the two must not be conflated
   when the k-stratified table is rebuilt.

**Behaviour change beyond §5.1, recorded so it is not mistaken for drift.**
`x - x` and `x / x` previously reached the DAG as a **one-input** `Sub`/`Div`,
because `LabeledDAG.add_edge` silently rejects the duplicate edge. Under
decomposition they become `Add(x, Neg(x))` and `Mul(x, Inv(x))`, which have two
distinct in-edges and now evaluate to the correct `0` and `1`. This is a fix that
the decomposition forces rather than an option. The analogous degeneracy for
`Add(a,a)`, `Mul(a,a)` and `Pow(a,a)` is **left alone and filed** (§11) — changing
it would move numbers this ticket does not own.

### 2026-07-30 — Measurement (S4). AC-4 decided, AC-5/7/8/9/10 closed.

Script: `experiments/scripts/measure_decomposition_impact.py`. Re-run by me
independently: `--n 5000 --seed 42`, **engine = cpp**, 5.0 s, numbers identical to
the subagent's.

**Population caveat, stated first because it bounds everything below.** These are
*randomly generated* AGraphs/CompGraphs, not evolved live-search populations. The
legacy control lands at 59.40 % of Bingo DAGs carrying `Sub`/`Div` against §2's
61.1 %, so the generator matches production. But rho here (Bingo 1.2960, UDFS
2.1505) is **not** the production rho (1.7931, 1.880), and `violated_pre` here
(41.02 % / 75.06 %) is **not** T06's (85.88 % / 100 %), because rediscovery and
orphan-CONST rates are properties of evolution, not of random sampling. **This
measurement establishes direction and invariance, not magnitude.** The magnitudes
come from Wave 1.

| | Bingo legacy | Bingo split | UDFS legacy | UDFS split |
|---|---|---|---|---|
| mean `k` | 5.47 | **6.72** (+1.25) | 3.27 | **3.99** (+0.72) |
| mean canonical length | 21.2 | **26.9** (+27 %) | 11.1 | **13.5** (+22 %) |
| distinct strings | 3,858 | **3,858** | 2,325 | **2,293** |
| rho | 1.2960 | **1.2960** | 2.1505 | **2.1805** (+1.4 %) |
| canon cost ms/DAG | 0.0079 | 0.0099 (**+24.6 %**) | 0.0059 | 0.0066 (**+10.8 %**) |
| `violated_pre` | 41.02 % | **41.02 %** | 75.06 % | **75.06 %** |
| `violated_post` | 0 | **0** | 0 | **0** |
| round-trip (M8) | 100 % | **100 %** | 100 % | **100 %** |
| completeness (M9) | — | **100/100** | — | **100/100** |

**AC-9 — reachability invariance holds, exactly as refined.** Confusion matrix vs
legacy is `FP=0, FN=0` on both hosts and both decomposed encodings; `violated_post`
is 0 everywhere. And the refinement I derived earlier is confirmed numerically: the
**node-level** violating count *does* rise (Bingo 0.46 -> 0.60, UDFS 1.52 -> 1.67)
while the **DAG-level** verdict never flips. T06's headline rates survive; its
k-stratification does not and must be rebuilt from Wave 1.

**AC-4 — sharing: DO NOT SHARE. Decided by Mario 2026-07-30 on the measurement.**
Sharing moves `k` by 0.6 % (Bingo 6.72 -> 6.68) and rho by 0.05 % (UDFS 2.1805 ->
2.1815), and is exactly 0.0000 on Bingo rho. It is not worth breaking `undecompose`
(which raises on out-degree > 1 wrappers) or introducing a partial CSE that merges
the `Neg`/`Inv` nodes we invent while leaving the host's own duplicated `Sin`/`Sub`
nodes untouched. `SHARE_DECOMPOSED_UNARY = False` ships.

**The one result that is genuinely favourable, and the subagent buried it.** Bingo
rho is *exactly* encoding-invariant — 3,858 distinct strings under all three
encodings — so on a host with no native `neg`/`inv` the decomposition is a bijection
on isomorphism classes. UDFS rho *increases* 1.4 %, and the mechanism is
identifiable: UDFS's search always samples `neg` and `inv` natively, so `sub_l(a,b)`
and `+(a, neg(b))` previously produced **different** canonical strings and now
produce the same one. That merge is sound — the two denote the same expression — and
M10 confirms **0 false merges for UDFS**. The corrected alphabet therefore removes a
spurious distinction the old encoding was carrying. This also explains M1's UDFS
gap: 52.00 % of DAGs carry `Sub`/`Div` but 64.12 % carry `Neg`/`Inv`, the difference
being host-native ones.

**AC-5 — semantic equivalence, with the coverage hole stated rather than smoothed.**
Guarded-regime fraction **2.01 %** (934/46,380). Outside the guard, max absolute
error **0.125**, which is ~1 ULP at the `_MAX_VALUE` clamp of 1e15 — IEEE 754
rounding between `a/b` and `a*(1/b)`, not a semantic difference. Inside the guard the
error reaches 1e15, exactly as predicted: `Div(a,b) -> 1.0` while
`Mul(a, Inv(b)) -> a`. Per §6.1 this touches no reported number.
**However: 53.6 % of evaluation attempts were skipped**, and I verified the cause
myself rather than accepting the subagent's account — legacy DAGs fail
`evaluate_dag` at **52.1 %** and decomposed at **33.3 %** (n=2,000). So (i) the
decomposition *reduces* unevaluable DAGs by 19 points, a second unadvertised
improvement, and (ii) M7 compares only where **both** encodings evaluate, which
systematically excludes the `x-x` / `x/x` cases decomposition changes most. **M7 is
a biased subsample and must not be quoted as full coverage.**

**AC-10 — `k` shift for the continuity table:** Bingo **+1.25** mean (+22.9 %),
UDFS **+0.72** (+22.0 %); median +1.0 on both; Bingo p95 11 -> 15.

**M10's 5 Bingo "false merges" are a measurement artifact, not a defect, and not a
T16 regression.** I checked directly: `fast_canonical_string` returns `''` for every
`k=0` DAG regardless of `m`, so two variable-only DAGs with `m=1` and `m=2` collide.
That is correct behaviour — the canonical string encodes the *instruction sequence*,
and `m` is a parameter of S2D, not part of the string. `fcs` is a complete invariant
**for fixed `m`**, and `m` is fixed per problem in production. The count is identical
(5/5/5) across all three encodings, so decomposition introduces nothing. **Filed for
T07** as a scope clarification worth stating explicitly in the completeness theorem,
not as a bug.

---

## 11. Proposed answer

*(unfilled — R2.3, to be written once the re-run lands. The answer must state
that Σ_SR and the host operator set are different objects: the host searches over
`{+, −, ×, ÷, sin, cos, exp, log}`, and `−` and `÷` enter the representation
through the commutative decomposition of Definition 3.2, so Σ_SR needs no `-` or
`/` label.)*
