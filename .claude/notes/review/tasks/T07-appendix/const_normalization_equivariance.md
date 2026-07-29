# T07 appendix — `normalize_const_creation`: defects, options, and the Σ_SR question

**Opened** 2026-07-29, from T07 (§5.2, AC-5/AC-6) while drafting the numbered
definition of `normalize_const_creation` for reviewer comment R1.3.
**Owner** Mario (by email agreement 2026-07-28: *"haz tú lo de
normalize_const_creation"*). Theorem and lemma **proofs** remain Ezequiel's.
**Status** three manuscript defects confirmed; no submitted number affected;
decisions pending.

---

## 0. Executive summary

Three independent defects in the **manuscript's formal statements** surfaced
while writing the definition R1.3 asks for. In all three the *code is correct and
the paper is wrong*, so **no reported number moves**. All three sit in the area
Reviewer 2 already objected to (R2.1), which makes finding them ourselves worth
more than being told in round 2.

| # | Defect | Where | Consequence | Owner |
|---|---|---|---|---|
| D-1 | `𝒩` is not isomorphism-equivariant | `labeled_dag.py:591-676` | kills the *restatement* of Thm 3.15 on `𝒩(D)`; submitted Thm 3.15 unaffected | Mario |
| D-2 | Def 3.9(iv) constrains operand order for `Pow` only, but `Sub`/`Div` are non-commutative and appear in every production operator set | `methodology.tex:920-929`, Remark `:945-958` | **(⇐) direction of Thm 3.15 false as stated** | **Ezequiel** |
| D-3 | Rule 1 prose says "`Pow` node"; implementation applies it to all of `{Sub, Div, Pow}` | `methodology.tex:752-760` | Lemma 3.14/A.2 misstates the candidate filter | **Ezequiel** |

---

## 1. D-1 — `normalize_const_creation` is not isomorphism-equivariant

### 1.1 Statement

`LabeledDAG.normalize_const_creation` maps isomorphic labeled DAGs to
**non-isomorphic** labeled DAGs. There exist `D₁ ≅ D₂` (Definition 3.9 (i)–(iv))
with `𝒩(D₁) ≇ 𝒩(D₂)`, and therefore `fcs_{D₁} ≠ fcs_{D₂}` although `D₁ ≅ D₂`.

This is a property of the **current, repaired** policy (the minimal in-degree-0
repair introduced by T15 on 2026-07-27), not of the pre-T15 relocation policy.
It is a **distinct defect from the one T15 fixed**.

### 1.2 The counterexample

Three variables `x₁, x₂, x₃`; two `Const` leaves with no in-edge; two `Sin`
nodes. Node ids in parentheses.

```
D :   c₁(3) → B(6) → x₂(1)        c₂(4) → A(5) → x₁(0)        x₃(2) isolated
D₂ = permute_internal_nodes(D, [1, 0, 2, 3])          # swaps the two Const ids
```

`permute_internal_nodes` relabels internal nodes and nothing else, so `D₂` is an
isomorphic copy of `D` **by construction**: the edge multisets map onto each
other under the transposition `3 ↔ 4`, labels are preserved, and the variable
nodes are fixed pointwise.

| | anchors chosen by `𝒩` | `𝒩(D)` edge set | reachability holds on `𝒩(D)` |
|---|---|---|---|
| `D`  | `c₁→x₁`, `c₂→x₃` | `(0,3) (2,4) (3,6) (4,5) (5,0) (6,1)` | yes |
| `D₂` | `c₁→x₂`, `c₂→x₃` | `(1,3) (2,4) (3,5) (4,6) (5,0) (6,1)` | yes |

Canonical strings: `VkpvknvsncNVsNpppC` vs `pvkpvknvsncPVsNppC`.

`is_isomorphic(𝒩(D), 𝒩(D₂))` = **False**, and correctly so: one graph anchors a
`Const` to `x₁`, the other to `x₂`, and variable anchoring (condition (iii)) pins
`x₁` and `x₂`, so no bijection can reconcile them. The two normalised graphs
genuinely are different labeled DAGs.

Reproduction folded into `tests/unit/test_const_normalization_equivariance.py`
and `experiments/scripts/validate_const_equivariance.py`.

### 1.3 Mechanism

`normalize_const_creation` iterates `for c in sorted(const_nodes)` and, for each
orphan `Const`, takes the lowest-indexed variable whose edge does not close a
cycle. **Node-index order is exactly what isomorphism permutes**, and the
iterations are not independent: anchoring one orphan creates new directed paths
that can make another orphan's preferred anchor cycle-closing.

In the counterexample, anchoring `c₁` to `x₁` creates the path
`c₂ ⇝ x₁ → c₁ ⇝ x₂`, so `x₂` becomes cycle-closing for `c₂`, which then falls
through to `x₃`. Under the swapped numbering `c₂` is processed first and the
interference runs the other way.

Three conditions must hold simultaneously:

1. at least two `Const` nodes with in-degree 0;
2. at least one `Var` node that is the target of an edge;
3. a directed path from an orphan `Const` to a `Var`.

### 1.4 Not a pruning artefact, not a port artefact, not D2S

Same diagnostic shape as T15 §1. Every canonicalisation entry point separates the
two isomorphic inputs:

| Algorithm | Pruning | `fcs(D) == fcs(D₂)` |
|---|---|---|
| `fast_canonical_string(backend="cpp", mode="wl_only")` ← **production** | greedy on 1-WL | **False** |
| `fast_canonical_string(backend="cpp", mode="wl_tiebreak")` | 1-WL + 6-tuple | **False** |
| `fast_canonical_string(backend="cpp", mode="tuple_only")` | 6-tuple | **False** |
| `fast_canonical_string(backend="python", mode="wl_only")` | greedy on 1-WL | **False** |
| `pruned_canonical_string` | 6-tuple + backtracking | **False** |
| `canonical_string` — **exhaustive, no pruning, true lexmin** | none | **False** |

The exhaustive reference explores every branch and still separates them, so no
pruning rule is at fault. Both engines agree, so it is not a C++/Python
divergence. **D2S is innocent**: it is handed two structurally different graphs
and correctly returns two different strings.

### 1.5 Blast radius

**No submitted number is affected, and no currently published claim is false.**

- Theorem 3.15 as submitted quantifies over DAGs satisfying the reachability
  condition. The counterexample has orphan `Const` nodes, so it violates that
  hypothesis and lies outside the theorem's scope.
- On DAGs satisfying reachability, `𝒩` is the identity (T15, confirmed at 10⁵),
  hence trivially equivariant.
- Condition 2 of §1.3 (a `Var` with in-edges) is **impossible by construction**
  on host-adapter output: neither adapter ever makes a `Var` an edge target
  (`bingo/adapter.py:143-159`, `udfs/adapter.py:104-147`). No candidate produced
  during any reported experiment can trigger it.
- T15 AC-4 measured the three policies as structurally identical on 12,176,790
  Bingo and 234,865 UDFS DAGs, with 0 `repair` vs `none` disagreements.

**Latent, filed, not fixed:** `is_isomorphic` applies `𝒩` (Critical Invariant 9)
and therefore inherits the defect — it returns `False` on a pair produced by the
repo's own `permute_internal_nodes`. Unreachable in production for the same
adapter reason.

### 1.6 What it rules out

**Restating Theorems 3.13/3.14/3.15 on `𝒩(D)` is refuted as stated.** That
restatement extends the domain to exactly the DAGs that have orphan `Const`
nodes; the counterexample then sits *inside* the new hypothesis (both
normalisations satisfy reachability) while violating the conclusion. Any
restatement on `𝒩(D)` needs a further hypothesis, or a change to `𝒩`.

Recorded so the option is not re-proposed: it was chosen on the morning of
2026-07-29 and withdrawn the same day on this evidence.

### 1.7 The safe class

`𝒩` is provably equivariant on `𝒞 = 𝒞₁ ∪ 𝒞₂`:

- `𝒞₁ = {D : every non-Var node is reachable from some Var}` — the Round-Trip
  Fidelity hypothesis. Here no `Const` has in-degree 0, so `𝒩 = id`.
- `𝒞₂ = {D : no Var node is the target of any edge}`. Here no orphan `Const` can
  reach any variable, so `x₁` never closes a cycle, every orphan `Const` anchors
  to `x₁`, and the outcome is independent of processing order.

`𝒞₂` is the host-adapter image. On it, Ezequiel's design instinct — *"lo mejor es
que x₁ sea el padre de todos ellos"* — is **not a convention but a consequence**:
the `x₂`/`x₃` fallback branch is unreachable, so the algorithm provably always
chooses `x₁`. Where it genuinely is a choice: DAGs decoded from arbitrary IsalSR
strings, since `C`/`c` *can* direct an edge into a variable (31 % of random S2D
DAGs have one, per T15). That is the regime the fallback exists for, and the only
regime where D-1 lives.

---

## 2. D-2 — Definition 3.9(iv) is wrong for `Sub` and `Div`

**A second, independent falsification of Theorem 3.15, and the more serious of
the two because it needs no exotic input.**

`BINARY_OPS = {SUB, DIV, POW}` in `src/isalsr/core/node_types.py`, and **every
production config includes `-` and `/`** (`experiments/configs/bingo_*.yaml`,
`udfs_*.yaml`). The manuscript constrains operand order for `Pow` alone:

- `methodology.tex:920-929`, Definition 3.9 condition (iv): *"for every
  \textsc{Pow} node `v` … All other operation nodes are commutative and impose no
  ordering constraint."*
- `methodology.tex:955-957`, Remark: *"All other binary operations
  (\textsc{Add}, \textsc{Mul}) and all unary operations are commutative, so
  condition~(iv) applies only to \textsc{Pow} nodes."*

Measured, C++ backend, three-node DAGs with **identical edge sets**:

| Op | `σ = (x₁,x₂)` | `σ = (x₂,x₁)` | strings differ | `is_isomorphic` |
|---|---|---|---|---|
| `Sub` | `V-PnC` | `pv-nC` | yes | False |
| `Div` | `V/PnC` | `pv/nC` | yes | False |
| `Pow` | `V^PnC` | `pv^nC` | yes | False |

For `Sub` and `Div` the identity bijection satisfies Definition 3.9 (i), (ii) and
(iii), and (iv) is vacuous because the node is not `Pow`. So **`D₁ ≅ D₂` per the
definition while `fcs_{D₁} ≠ fcs_{D₂}`** — the (⇐) direction of Theorem 3.15
fails on a three-node counterexample.

**The code is right and the definition is wrong**: `x₁ − x₂ ≠ x₂ − x₁`, so these
DAGs *must* be distinguished. **No reported number is affected.**

**Fix (Ezequiel).** Condition (iv) must range over all non-commutative binary
operations `{Sub, Div, Pow}`, not `Pow` alone; the Remark's claim that all other
binaries are commutative must be deleted or restricted to `{Add, Mul}`. Note the
interaction with the commutative-alphabet option
(`OperationSet.commutative()` replaces `Sub`/`Div` with `Add`+`Neg` /
`Mul`+`Inv`): under *that* alphabet the current text would be correct, but it is
**not** the alphabet the experiments ran.

---

## 3. D-3 — Rule 1's stated scope does not match the implementation

`methodology.tex:752-760` states Rule 1 for a *"\textsc{Pow} node `c`"*. The
implementation applies it to every member of `BINARY_OPS`
(`canonical.py:611, 689, 995, 1064` — `if ig.node_label_unchecked(c) not in
BINARY_OPS`). The Table 3 caption (`methodology.tex:819-820`, *"binary
non-commutative node"*) already matches the implementation, so the manuscript is
internally inconsistent as well as wrong. Same fix family as D-2; make both in
one pass.

Found by the Rule 1 non-exclusion task (§7.1).

---

## 4. Consequences of abandoning `normalize_const_creation`

Asked directly, 2026-07-29. The answer depends on **which** of the two
application sites is removed; they are not equivalent.

| | What is removed | Effect on production | Verdict |
|---|---|---|---|
| **(b) only** | the call inside `canonical.py:95,146,231` + `labeled_dag.py:458` (`is_isomorphic`) | **none measurable** — T15: `none` ≡ `repair` structurally on 12,176,790 Bingo + 234,865 UDFS DAGs, 0 disagreements | viable |
| **(a) + (b)** | also `_normalize_const_edges` in both adapters (`bingo/adapter.py:171-175`, `udfs/adapter.py:150`) | canonicalisation **raises on 85.9 % of Bingo and 100 % of UDFS candidates** (T06) | **not viable** |

The second row fails in the *flattering* direction and must not be adopted by
accident. A candidate whose canonicalisation raises is evaluated, counted in
`n_total`, and never in `n_unique`, so ρ = `n_total`/`n_unique` inflates. T06
demonstrated the mechanism with forced timeouts (ρ 1.80 → 3.91 at a 57 % failure
rate). At an 86–100 % failure rate the reduction factor would become meaningless
*and* spectacular.

**Why the repair cannot be dropped outright.** Σ_SR has no instruction that
creates a node from nowhere: every `V`/`v` token creates a node **together with
an edge from the acting pointer**. A constant leaf with no incoming edge is
therefore unencodable — no string produces it. Host solvers emit constants as
exactly such leaves. Verified: with normalisation bypassed the canonicaliser
**raises** on the counterexample rather than returning a wrong string, which is
the correct failure mode.

**Reachability itself cannot be abandoned** for the same reason. It is a
structural necessity of the encoding, not a proof convenience.

### 4.1 Recommended direction (pending decision)

Move the repair out of canonicalisation and make reachability an *enforced
precondition* of it:

- producers (adapters, `from_sympy`) establish the precondition;
- the canonicaliser assumes it and raises loudly if violated, instead of silently
  repairing;
- Theorems 3.13/3.14/3.15 stay stated on `D` satisfying reachability — theory and
  code finally describe the same object;
- D-1 leaves the canonicalisation path entirely and `is_isomorphic` becomes a
  true isomorphism test again.

Loose ends to check first: `from_sympy` almost certainly produces orphan `Const`
leaves; the precomputed atlas path calls D2S **outside** both runners'
try/except (T06 filed this — it would crash rather than fall back); hand-built
DAGs in the test suite.

### 4.2 Option table

| # | Option | Code change | Engine/recompute cost | Equivariance |
|---|---|---|---|---|
| A | Scope the theory to `𝒞`; define `𝒩`; state the `𝒞₂` consequence | none | none | holds on `𝒞`, stated |
| B | Anchor to `x₁` unconditionally; raise if cycle-closing | Python + C++ | T01 equivalence gate re-run, ~30 tests | holds everywhere `𝒩` succeeds |
| C | Drop `𝒩` from `canonical.py`; reachability becomes a caller precondition | delete call sites | none measurable | vacuous — `𝒩` leaves the pipeline |

---

## 4bis. FINAL definition of the normalisation, and why

**Settled 2026-07-29.** This is the text the manuscript's numbered definition and
the R1.3 response are to be built from. It supersedes §3.2's submitted policy and
§4.1's "recommended direction".

### 4bis.1 The operation, as implemented

> **Definition (`normalize_const_creation`).** Let `D` be a labeled DAG with
> variable nodes `x₁, …, x_m`. For each `Const` node `c` of **in-degree 0**, taken
> in increasing node index, add the single edge `x_i → c` where `i` is the least
> index such that the addition does not close a directed cycle; if no such `i`
> exists, leave `c` unchanged. `Const` nodes of in-degree ≥ 1 are untouched, and
> **no edge is ever removed**.
>
> Complexity: `O(|Const|)` anchor attempts, each with an acyclicity check.
> Applied only when `D` contains at least one `Const` node.

Properties, all load-bearing:

| # | Property | Status |
|---|---|---|
| N1 | Edge-monotone: `E(D) ⊆ E(𝒩(D))`, same nodes and labels ⇒ reachability is never destroyed | holds by construction |
| N2 | Idempotent: `𝒩(𝒩(D)) = 𝒩(D)` | holds — edges only added, so refusals stay refusals |
| N3 | **Identity on the hypothesis class**: if every non-`Var` node is reachable from some variable, no `Const` has in-degree 0, so `𝒩(D) = D` | verified at 10⁵ (T15) |
| N4 | Evaluation-preserving: `eval(𝒩(D)) = eval(D)`, because `Const` ignores in-edges and out-degrees are unchanged (so the sink set is unchanged) | **false under the submitted policy**, true now |
| N5 | **Isomorphism-equivariant** | **FAILS in general** — see §1. Holds on `𝒞 = 𝒞₁ ∪ 𝒞₂` |

### 4bis.2 Why it exists — the justification, in one paragraph

Σ_SR has no instruction that creates a node from nowhere: every `V`/`v` token
creates a node **together with an edge from the acting pointer**. A constant leaf
with no incoming edge is therefore *unencodable* — no string in Σ_SR produces it.
Host solvers emit constants as exactly such leaves, so **85.9 % of Bingo and
100 % of UDFS candidates arrive violating the reachability precondition** (T06),
against **0 %** of S2D-produced DAGs. The step supplies the missing creation
edge. It is a precondition repair, not cosmetic preprocessing.

### 4bis.3 Why it was wrong before, twice

1. **Submitted policy** relocated *every* `Const` creation edge onto node 0.
   `add_edge` refuses a cycle-closing edge and returns `False`; the return value
   was discarded, so the `Const` lost its only in-edge and became unreachable.
   It also **merged non-isomorphic DAGs** (breaking completeness) and was **not
   evaluation-preserving** (`x → Cos → Const`: `1.0` became `cos(1.5) = 0.0707`).
2. **Current policy** fixes both but is **not isomorphism-equivariant** (§1),
   because orphan `Const` nodes are processed in node-index order and the
   anchorings interfere. Unreachable on adapter output, but real.

### 4bis.4 The resolution — remove it from the canonicaliser

`𝒩` is applied at two points: inside each **adapter**
(`bingo/adapter.py:171-175`, `udfs/adapter.py:150`) and inside **`canonical.py`**
(`:95,146,231`) plus `is_isomorphic` (`labeled_dag.py:458`). The adapters'
application is the one that does the work; the canonicaliser's is redundant.

**Removing it from `canonical.py` and `is_isomorphic`:**

- makes `fcs` a **pure function of `D`**, so equivariance failure (D-1) is
  impossible by construction rather than merely improbable;
- repairs `is_isomorphic`, which currently returns `False` on a pair built by the
  repo's own `permute_internal_nodes`;
- leaves **Theorems 3.13/3.14/3.15 stated exactly as submitted**, because the
  object the theorem describes and the object the code canonicalises finally
  coincide — this is what discharges AC-6 without amending anything;
- keeps the precondition established, because the **adapters** still do it;
- changes the failure mode on unencodable input from *silently wrong* to
  *loudly refused*, which is correct behaviour.

**What must not be removed**: the adapters' repair. Dropping that too would make
canonicalisation raise on 85.9–100 % of candidates, and since a raised candidate
lands in `n_total` but never `n_unique`, ρ would inflate spectacularly and
meaninglessly.

### 4bis.5 Evidence

| Claim | Evidence |
|---|---|
| `𝒩` is the identity on the hypothesis class | 0 disagreements, 10⁵ synthetic DAGs (T15 AC-3) |
| Policy change moves no submitted number | three policies structurally identical on 12,176,790 Bingo + 234,865 UDFS DAGs (T15 AC-4) |
| Precondition is violated on arrival, repaired to 0 | 85.9 % / 100 % → 0 % (T06) |
| Equivariance fails only outside `𝒞` | 0 failures in 79,864 samples on clean populations; 18–20 % on adversarial, all outside `𝒞` (§7.2) |
| Arms agree on real data | per-DAG agreement **1.000000**, ρ and `n_unique` identical (§7.3, re-running post-fix) |
| Round-trip holds under both arms, common comparator | **351,200 / 351,200 both arms** (§7.5) |

---

## 5. Paper prose for R1.3 (plain language, no jargon)

Agreed wording to build the numbered definition's justification around. It
deliberately avoids "adapter image", "equivariance" and "hypothesis class", none
of which should appear in the manuscript unless they are actually used:

> When a host solver hands us an expression, its constants arrive as leaves with
> nothing pointing into them, and such a node has no encoding in Σ_SR. We
> therefore add one incoming edge, from `x₁`. This is safe for two reasons.
> First, a constant ignores its incoming edges when the expression is evaluated,
> so the value the DAG represents is unchanged. Second, host solvers never direct
> an edge *into* a variable, so `x₁` is never downstream of the constant and the
> added edge cannot create a cycle.

Supporting measurement for the same paragraph (T06): the step repairs a
precondition violated on arrival by **85.9 %** of Bingo candidates
(132,746 / 154,568) and **100 %** of UDFS candidates (3,890 / 3,890), and by
**0 %** of S2D-produced DAGs (0 / 14,841) — which is why the property-validation
experiment never saw it. Residual violation after the step: **0 %** everywhere.

---

## 6. Future work — extending Σ_SR with an edgeless insertion instruction

Raised 2026-07-29. **Assessment: the right design, the wrong revision.**

### 6.1 The proposal

Add an instruction to Σ_SR that creates a `Const` node **without** a creation
edge — inserting it into the CDLL at the pointer position and leaving the pointer
immobile (Critical Invariant 4). `normalize_const_creation` would then be
unnecessary: an orphan `Const` becomes directly encodable.

### 6.2 Why it is the right design

- Today the language cannot express a constant leaf, so we **mutate the graph to
  fit the language**. Extending the language to express what the domain naturally
  produces is the correct direction.
- It removes `𝒩` outright, so **D-1 vanishes by construction** rather than by
  hypothesis, and `is_isomorphic` is repaired as a side effect.
- The reachability precondition weakens to something structurally honest — *every
  in-degree-0 non-variable node is a `Const`* — which expression DAGs satisfy
  automatically, rather than a condition 85.9–100 % of real inputs violate on
  arrival.
- The current constraint is **inherited baggage**. IsalGraph is topology-only,
  where every node is legitimately created from an existing node. SR constants
  are the one case that constraint does not fit.

### 6.3 Why not in this revision

1. **It changes Σ_SR** — Definition 3.2 and Table 1, the paper's core object.
   That is a new version of the method, not a revision. Ezequiel would have to
   re-prove 3.13/3.14/3.15 against a changed alphabet *on top of* the Lemma A.2
   rewrite he has already committed to, against a 2026-09-24 deadline.
2. **It is not a free win algorithmically.** A token that creates a node without
   an edge introduces a scheduling freedom D2S does not currently have: *when* to
   emit it relative to the other tokens. That is new string multiplicity the
   canonicaliser must resolve — a fresh canonicalisation sub-problem with its own
   proof obligations, traded for a known and bounded defect.
3. **The O(k!) reduction claim would need re-verification** on the new string
   space (advisor constraint 3).
4. **It buys nothing measurable on the reported populations.** On adapter output
   every `Const` is an orphan and anchors to `x₁`, so all constants already
   encode identically; the new token would also encode them identically.
   Bijective re-encoding ⇒ same equivalence classes ⇒ **ρ unchanged**. The defect
   it fixes is unreachable in production.
5. **Reviewer optics.** R1.3 asks what the undefined step *is*. Answering "we
   changed the instruction set" reads as *the submitted method was provisional*.
   Answering with a precise definition plus the 85.9–100 % measurement closes
   R1.2 and R1.3 together.

### 6.4 Recommendation

Two sentences in the Conclusion / Future Work naming this as the principled fix.
Cost: negligible. Benefit: pre-empts a reviewer proposing it, and establishes
priority for the follow-up. Draft:

> The creation edge that `normalize_const_creation` supplies is an artefact of
> Σ_SR rather than of the expression it encodes: every insertion instruction
> creates a node together with an edge from the acting pointer, so a constant
> leaf has no direct encoding. A dedicated edgeless insertion instruction would
> remove the preprocessing step entirely and weaken the reachability precondition
> to a condition expression DAGs satisfy by construction. It would also enlarge
> the string space, since the position of such an instruction is unconstrained,
> and we leave the resulting canonicalisation problem to future work.

**If it is ever taken up**, the validation must show: ρ and the distinct-string
count unchanged on the measured populations (the bijective-re-encoding argument
above, made empirical); equivariance under random internal-node permutations; and
round-trip fidelity on the enlarged string space.

---

## 7. Empirical status

### 7.1 Rule 1 non-exclusion (T07 §5.4 bullet 1, AC-7) — accepted for `Pow`

`experiments/scripts/validate_rule1_non_exclusion.py`,
`tests/unit/test_rule1_non_exclusion.py` (19 tests; **re-run and green in the
main tree**). `backend="cpp"` throughout.

| Quantity | seed 42 | seed 1337 (independent re-run by Mario) |
|---|---|---|
| `Pow`-containing DAGs tested | 16,392 | 16,440 |
| DAGs where Rule 1 actually excluded a candidate | **3,023 (18.4 %)** | **3,026 (18.4 %)** |
| Rule-1-attributable canonicalisation failures | **0** | **0** |
| Canonicalised / round-trip OK | 16,392 / 16,392 | 16,440 / 16,440 |
| Timeouts | **0** | **0** |

Stable across seeds; accepted. Supersedes a first attempt that was near-vacuous
(24 exercising DAGs of 13,394, 0.18 %) and whose 3 timeouts were Python-backend
backtracking cost, not Rule 1 — they vanish on the C++ backend.

**Residual gap, and it is a scoping error in the brief, not an agent failure.**
`n_nonpow_binary_tested = 2`: the population is dense for `Pow` and essentially
empty for `Sub` and `Div` — exactly the ops D-2 and D-3 concern. Since the
manuscript will be corrected to state Rule 1 over all of `BINARY_OPS`, the
non-exclusion claim must be carried by data for all three ops. Extension
requested: per op in `{Pow, Sub, Div}`, ≥ 5,000 DAGs with ≥ 1,000 exercising an
exclusion, reported **broken out per op** so `Pow`'s density cannot mask a gap.

D-3 was found by this task.

### 7.2 Equivariance failure rate — MEASURED, hypothesis confirmed

`experiments/scripts/validate_const_equivariance.py`,
`tests/unit/test_const_normalization_equivariance.py` (24 tests). K = 8
permutations per DAG via `permute_internal_nodes`, C++ engine, 10 s budget.
Two independent seeds (agent's, and Mario's re-run at `--seed 2718`).

| Population | N DAGs | perm. tests | failures | in `𝒞` |
|---|---|---|---|---|
| P1a random S2D, m = 2 | 400–500 | 3,200–4,000 | **0** | 100 % |
| P1b random S2D, m = 3 | 400–500 | 3,192–3,984 | **0** | 100 % |
| P2 `Const`-free synthetic | 20 | 160 | 0 — **vacuous** | 100 % |
| **P3 Bingo adapter output** | **15,530** | **123,240** | **0** | **100 % in `𝒞₂`** |
| P4 adversarial (built to violate `𝒞`) | 400–2,000 | 2,352–11,408 | **18.1–20.1 %** | **0 %** |

**Cross-tabulation, both seeds: failures inside `𝒞` = 0; every failure outside
`𝒞`.** P3's `frac_in_c2 = 1.0000` — every Bingo candidate has all variables as
pure sources, which is the structural reason D-1 is unreachable in production.
UDFS is 100 % in `𝒞₂` by the adapter argument (`udfs/adapter.py:108-151` adds
edges only into operator nodes) — a code argument, not a measurement, and
labelled as such in the script output.

P4's ~19 % rate is the point: the hypothesis is **not vacuous**, the defect is
real and reproducible on demand.

**Three caveats, recorded rather than buried.**

1. **`𝒞` is sufficient, not tight.** There exist DAGs outside `𝒞` that do *not*
   fail — the interference fires only when the two `Const` chains target the two
   lowest-indexed variables (`TestNoFailureOnNonAdjacentTargets`). `𝒞` is a
   usable theorem hypothesis; it does **not** characterise the failure set and
   must not be presented as if it did.
2. **P2 is vacuous and must not be cited as confirmation.** Its generator emits
   no `Const` nodes, so `𝒩` is a guaranteed no-op — the same trap T06 flagged
   for its corpus 3. Independently rediscovered here.
3. **P2 costs 83× more per DAG than P1** (456 ms vs 5.5 ms): `Const`-free random
   strings decode to deeper, larger-k DAGs. Any design using equal N across
   populations mis-budgets P2 by two orders of magnitude.

P1's N was scaled to 400–500 (not the 20,000 requested) to stay inside the
10-minute local ceiling; `--n-s2d 20000` is supported for a long run. **The P1
sample is therefore the weakest row in the table** and is the one the Picasso
study (§7.3) should enlarge.

Full suite after both new test files: **4,541 passed, 5 skipped** (4,478 + 24 +
39), `ruff` and `mypy --strict` clean. All re-run by Mario.

### 7.3 RESULT — the two-arm removal study (Picasso, 2026-07-29)

Jobs `1679357` (synthetic, 20), `1679404` (bingo, 15), `1679413` (udfs, 15),
`1679358` (adversarial, 1). **50/50 tasks COMPLETED, 0 FAILED, 0 OOM, 51/51
result files.** C++ engine throughout. All numbers below re-read by Mario from
`summary.json`, not taken from the agent.

`keep` = production (`𝒩` applied inside canonicalisation).
`drop` = `𝒩` not applied inside canonicalisation (`fast_canonical_string_raw`).
In both arms the host **adapters** still repair upstream, which is production
reality.

| Population | DAGs / arm | per-DAG agreement | `n_unique` keep = drop | ρ equal | equivariance failures keep / drop |
|---|---|---|---|---|---|
| **bingo** | **93,910,005** | **1.000000** | 325,486 = 325,486 | yes | **0 / 0** (14,809,528 samples) |
| **udfs** | **36,401,276** | **1.000000** | 111,553 = 111,553 | yes | **0 / 0** (11,657,120 samples) |
| synthetic | 500,000 | **1.000000** | 491,490 vs 491,491 | yes | 0 / 0 (79,864 samples) |
| adversarial | 12 | n/a (`n_both_ok`=0) | — | — | **53 / 0** (96 samples) |

> 🔴 **RETRACTED same day. The bingo and udfs rows are contaminated by recursive
> re-entry and must not be quoted.** `_round_trip_keep` calls `_score_keep`,
> which re-imports `isalsr.core.canonical.fast_canonical_string` at call time;
> during a live search that name is the monkey-patched `TwoArmRecorder`, and
> **neither the as-run nor the patched version has a re-entrancy guard**. Every
> round-trip check on a live-search DAG therefore re-entered the recorder and
> counted the S2D-decoded string as a fresh adapter DAG, inflating `n_total`,
> `n_unique`, ρ, equivariance sample counts and round-trip rates. Established by
> diffing the Picasso as-run copy against local (`asrun_study.py:371`), not from
> the agent's report — the agent attributed the artefact to the *drop* arm's
> comparator alone, which cannot explain a keep-arm figure measured with
> `fast_canonical_string`.
>
> **Synthetic and adversarial do not monkey-patch anything and are clean.**
> Bingo/udfs require a re-run with a re-entrancy guard and `keep_fn` threaded
> through the round-trip helpers. The 2026-07-29 partial fix did **not** address
> this: `_round_trip_keep` still recurses.

**SUPERSEDED by the v2 campaign — see §7.6 for the numbers that stand.**

**On the adversarial population the arms separate exactly as predicted**: `keep`
produces 53 equivariance failures in 96 samples (55.2 %) — isomorphic DAGs given
different canonical strings, **silently** — and refuses nothing. `drop` produces
0 failures and refuses all 12 loudly. `keep` silent-wrong 7, `drop` loud-refusal 12.

**Synthetic, the one place the arms behave differently on non-adversarial input.**
At k ≥ 21, `keep` records 359 *timeouts* and `drop` records 358 *raises*, on the
same k-distribution. Interpretation: `𝒩` gives an orphan `Const` a creation edge,
after which the DAG is canonicalisable in principle but expensive, so it burns the
budget; without `𝒩` the same DAG is refused immediately. `drop` ends with one
*more* successful canonicalisation (499,642 vs 499,641) and an identical ρ.

#### Three caveats — do not quote this study's numbers without them

1. **The ρ values here are NOT the paper's reduction factor and must never be
   quoted as such.** This study reports ρ ≈ 288 (bingo) and ρ ≈ 326–343 (udfs)
   against the paper's 1.793. The cause is aggregation: `n_total` is summed over
   15 independent tasks while `n_unique` is the **union** of their distinct-string
   sets, and independent seeds rediscover the same small structures, so the union
   barely grows. It is a pooled-union artefact. **It does not affect this study's
   conclusion**, because both arms receive the identical pooled value and the
   comparison is what is being tested — but the absolute number is meaningless
   outside this table. The field is honestly named `rho_lower_bound` in the JSON.
2. **The round-trip axis is not validly compared between arms.** `keep` uses
   `round_trip_comparator = fast_canonical_string`; `drop` uses `structural_key`.
   Different comparators. The synthetic `drop` figure of `round_trip_rate =
   0.0187` is therefore a **comparator artefact, not a failure** — `structural_key`
   is not a round-trip comparator for general DAGs. Round-trip fidelity under
   `drop` remains **unverified** and is the gap this study did not close.
3. **New, unrelated to normalisation, and worth its own investigation:** in the
   `keep` arm — the production configuration — round-trip succeeds on only
   **99.39 %** of real DAGs: 93,333,870 / 93,910,005 (bingo) and
   36,179,317 / 36,401,276 (udfs). That is **~576,000 and ~222,000 round-trip
   failures** respectively. It is ~0.6 % in *both* arms, so it is **not** caused by
   `𝒩`, and synthetic `keep` is 1.0000. A pre-existing property of real
   host-adapter DAGs that neither T15 nor T06 measured. Candidate explanation:
   the `Sub`/`Div` alphabet mismatch of T16. **Not concluded — flagged.**

#### What it settles

`𝒩` inside `canonical.py` can be removed with no measurable effect on production
and with the equivariance defect (D-1) eliminated by construction. That makes
`fcs` a pure function of `D`, so **Theorems 3.13/3.14/3.15 stand as written** and
AC-6 is discharged by deleting call sites rather than by amending any statement.

Conditional on caveat 2 being closed: round-trip fidelity under `drop` must be
re-measured with `fast_canonical_string` as the comparator in both arms before
the removal is committed.

### 7.5 The contamination, the fix, and the corrected numbers (2026-07-29, later)

**The defect.** `_round_trip_keep` called `_score_keep`, which re-imports
`isalsr.core.canonical.fast_canonical_string` **at call time**. During a live
search that name is the installed `TwoArmRecorder`, and no re-entrancy guard
existed, so every round-trip check re-entered the recorder and scored its own
S2D-decoded string as a fresh adapter DAG. Confirmed by diffing the Picasso
as-run copy against local (`asrun_study.py:371`).

**Scope — checked, not assumed.** Re-entry requires an installed monkey-patch:

| Source | Patches `fast_canonical_string`? | Re-enters? | Verdict |
|---|---|---|---|
| Production runners | **no** (`grep -rn` over `experiments/models/` → nothing) | no | **clean** |
| T15 `measure_const_normalization_arms.py` | yes | **no** — zero `StringToDAG`/round-trip | **clean** |
| T06 `fallback_ledger.py` | **no** | no | **clean** |
| `t07_norm_removal_study.py`, bingo/udfs only | yes | **yes** | contaminated |

**No submitted TPAMI number is affected**, and neither are T15's or T06's.
Synthetic and adversarial install no patch, so they were clean throughout.

**The fix.** (i) `keep_fn` threaded through `_round_trip_keep` and
`_round_trip_drop` so they use the unpatched function; (ii) a re-entrancy guard
on `TwoArmRecorder.__call__` that delegates to `self._original` and records
nothing; (iii) `n_reentrant_calls` emitted in `results.json` so the property is
provable from the output. **Measured: `n_reentrant_calls = 0` — the guard never
fires, so (i) is what actually fixed it and (ii) is a backstop.**

**Corrected numbers, local, C++ engine, both arms:**

| Problem | DAGs | Bingo's OWN counter | study `n_total` / `n_unique` | ρ | agreement | eq_fail | round-trip |
|---|---|---|---|---|---|---|---|
| Nguyen-5, 90 s | 10,520 | 10520 / 5936 | 10,520 / 5,936 | **1.772** | 1.000000 | 0 / 1,640 | 10,520/10,520 |
| I.6.20a, 90 s | 351,200 | 351200 / 197739 | 351,200 / 197,739 | **1.776** | 1.000000 | 0 / 56,008 | **351,200/351,200** |

**The cross-check that settles it**: the study's counters now reproduce Bingo's
own independent dedup counters **exactly**, and ρ lands on 1.772–1.776 against
the production 1.793. Under recursion they could not have matched. The earlier
ρ ≈ 288 is fully explained — and the "pooling artefact" hypothesis recorded in
the first draft of §7.3 was wrong.

**Both open caveats are closed.** Round-trip now uses `fast_canonical_string` in
**both** arms and is **100 % on 351,200 real DAGs in each** — so the drop arm's
round-trip is verified, and the 0.6 % failure rate was an artefact of the same
recursion, not a property of the representation.

Re-run of the bingo/udfs arrays into `t07_norm_removal_v2` (jobs `1679689`,
`1679698`); the 21 clean synthetic/adversarial results were carried over.

### 7.6 FINAL RESULT — v2 campaign on the fixed code (2026-07-29)

Jobs `1679689` (bingo, 15 tasks) and `1679698` (udfs, 15), **30/30 COMPLETED, 0
FAILED, 0 OOM**, into `~/execs/isalsr/t07_norm_removal_v2`. 51/51 result files.
C++ engine. All figures below re-read by Mario from `summary.json`.

| | bingo keep | bingo drop | udfs keep | udfs drop |
|---|---|---|---|---|
| `n_total` | 10,286,517 | 10,286,517 | 265,092 | 265,092 |
| `n_ok` | 10,286,517 | 10,286,517 | 265,092 | 265,092 |
| `n_raised` / `n_timeout` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `n_unique` | **5,736,798** | **5,736,798** | **141,009** | **141,009** |
| ρ | **1.7931** | **1.7931** | 1.8800 | 1.8800 |
| equivariance fails / samples | **0** / 1,639,064 | **0** / 1,639,064 | **0** / 42,480 | **0** / 42,480 |
| round-trip | **10,286,517 / 10,286,517** | **10,286,517 / 10,286,517** | **265,092 / 265,092** | **265,092 / 265,092** |
| comparator | `fast_canonical_string` | `fast_canonical_string` | `fast_canonical_string` | `fast_canonical_string` |
| per-DAG agreement | **1.000000** | | **1.000000** | |
| `n_reentrant_calls` | **0** (5 files spot-checked, bingo + udfs) | | | |

**Three things this establishes.**

1. **Removing `𝒩` from the canonicaliser is a no-op on real data.** 10,551,609
   DAGs across both hosts, per-DAG agreement exactly 1.000000, identical
   `n_unique` and ρ, zero raises and zero timeouts in either arm.
2. **Round-trip fidelity holds under both arms, same comparator, 100 %.** The
   earlier 0.6 % failure rate was entirely the recursion artefact. Caveats 2 and 3
   from §7.3 are **closed**.
3. **External validation of the whole measurement chain**: ρ = **1.7931** against
   the paper's independently produced production value **1.793** — four
   significant figures, from a different code path. Under recursion this figure
   was 288.5. That agreement is what makes the campaign trustworthy, and it is a
   cross-check the first run did not have.

**Carried-over populations, stated precisely.** The 21 `synthetic` and
`adversarial` result files were produced by the **pre-fix** run and copied
across. They are sound — neither installs a monkey-patch, so neither could
recurse — but their **drop-arm** rows still carry `round_trip_comparator =
structural_key`, so `synthetic drop rt = 9,338 / 499,642` and `adversarial drop
rt = 0 / 0` remain **comparator artefacts and must not be quoted**. Their
keep-arm figures, the agreement rate (1.000000) and the equivariance counts
(0 / 79,864 synthetic; **53 / 96 keep vs 0 / 96 drop** adversarial) are
unaffected by the comparator and stand.
Re-running synthetic would tidy this; no claim depends on it, since drop-arm
round-trip is now established far more strongly on 10.5 M real DAGs.

### 7.4 Superseded design note — the original three-arm plan

Settles §4 empirically.

1. `repair` — current behaviour (control)
2. `none-in-canonical` — adapters still repair, canonicaliser does not
3. `none-anywhere` — no repair at all (expected to fail catastrophically; this
   arm produces the failing examples for the R1.3 figure)

Per arm, per population (synthetic S2D 10⁵, Bingo live, UDFS subset):
reachability satisfied on arrival, canonicalisation failure rate, equivariance
under K random internal-node permutations, ρ, and distinct-string count.
**Arms 1 and 2 must agree on ρ and distinct-string count to 100 % for the drop to
be safe.**

---

## 8. Manuscript changes this document implies

| File | Location | Change | Owner |
|---|---|---|---|
| `methodology.tex` | before Def 3.8 (`:745`) | new numbered definition of `normalize_const_creation` (AC-5): §5's justification, complexity `O(\|Const\|)` plus cycle checks, and the `_has_const_nodes()` guard | Mario |
| `methodology.tex` | Remark `:960` | unnumber "Tightened automorphism bound" so Theorem 3.15 keeps its number | Mario |
| `methodology.tex` | `:830` (Table 3, inside `\begin{comment}`) | `// redirect all Const creation edges to x_1` → *"supply a creation edge to Const nodes that have none"* | Mario |
| `supplementary.tex` | near `:398` | same correction in the **rendered** pseudocode twin | Mario |
| `conclusion.tex` | future work | §6.4 draft paragraph | Mario |
| `methodology.tex` | `:920-929`, `:955-957` | **D-2** — condition (iv) must cover `{Sub, Div, Pow}` | **Ezequiel** |
| `methodology.tex` | `:752-760` | **D-3** — Rule 1 scope is `BINARY_OPS`, not `Pow` alone | **Ezequiel** |
| `supplementary.tex` | `:119-133` | Lemma A.2 proof; drop *"exactly the same candidate pool as D2S"* (AC-2) | **Ezequiel** |

**`double_blind/paper/methodology.tex` is a byte-identical copy, not a symlink —
every edit above must be mirrored.**

Manuscript root: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/`
`article/journal/69c1637a28a81fea2badda9a/` (Overleaf git remote, pulled clean
2026-07-29).
