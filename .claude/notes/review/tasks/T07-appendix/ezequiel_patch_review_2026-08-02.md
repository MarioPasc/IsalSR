# Review of Ezequiel's T07 patch (received 2026-08-02)

> ## ⚠️ SUPERSEDED IN PART — read this first (added later the same day)
>
> While drafting the manuscript edits, one check overturned **gap 2**, which had
> stood in this ticket since it was written. **D2S already applies Rule 1's exact
> predicate** (`dag_to_string.py:338–341` ≡ `canonical.py:647–649`/`:725–727`,
> same test, same `BINARY_OPS` scope — bug fix B9 / Critical Invariant 8), while
> the paper's D2S pseudocode has no such condition.
>
> So Rule 1 does **not** restrict the D2S pool; it restates the D2S rule. The
> submitted proof's sentence *"exactly the same candidate pool as D2S"* is **true
> of the two algorithms**. The defect is one level down: **Definition 3.5
> describes D2S without its first-operand restriction**, which makes `𝒲(D)` too
> large and **Theorem 3.13 false** under that reading. Verified counterexample:
> for `D = x₁^x₂`, the string `NV^Nc` places every node and every edge of `D` —
> so it qualifies under Definition 3.5 as written — yet decodes to `x₂^x₁`, and
> `is_isomorphic` returns `False`.
>
> **Consequences.** §1's B2 stands (the timing claim is false) but its framing of
> Rule 1 as restricting `𝒟_j` is wrong. Ezequiel's Step 2, and §2's S2, and the
> R2.1 letter paragraph conceding that the pool sentence is false, would all ship
> an **incorrect concession to R2**. The repair is to state the restriction in
> Definition 3.5 and Table 2, after which the pool identity is derivable and
> B4's widening of Theorem 3.13 becomes sound — it is not sound without it.
>
> The delivered `.tex` files implement this corrected version. What each file
> contains is in
> `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/T07_Ezequiel/CHANGES_MPG.md`.

Files reviewed, all in `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/T07_Ezequiel/`:
`methodology_T07.tex`, `supplementary_T07.tex`, `response_to_reviewers_T07.tex`
(+ `maindocument.pdf`, `appendices.pdf`).

Baseline for the diff: Overleaf checkout at `79157c4`. A read-only `git fetch`
shows one new remote commit, `8807b45 "Update on Overleaf."` (2026-07-31), whose
diff against `79157c4` is **empty** — so **Ezequiel has not yet integrated any of
this**. The timing for an argument by email is right.

**Verdict: do not integrate as-is.** The five-step proof is a real improvement and
its skeleton is the right one, but four points are blocking and four should be
fixed. Two of the blocking points are *coherence with the rest of the revision*,
which is exactly the question Ezequiel asked; two are defects inside the proof.

---

## 1. Blocking

### B0. The patch renumbers every theorem from 3.8 onward. Read straight off his own PDF.

The new definition is inserted *before* the Fast Canonical String definition, and the
2026-07-29 compensating step — unnumber Remark "Tightened automorphism bound" so the
count balances — was not applied. `pdftotext maindocument.pdf`:

| Environment | Submitted | In the patch |
|---|---|---|
| Round-Trip Fidelity | **Theorem 3.13** | Theorem 3.14 |
| FCS produces valid D2S strings | **Lemma 3.14** | Lemma 3.15 |
| Complete Labeled-DAG Invariant | **Theorem 3.15** | Theorem 3.16 |
| Labeled-DAG Isomorphism | Definition 3.9 | Definition 3.10 |
| Fast Canonical String | Definition 3.8 | Definition 3.9 |

These are the numbers **both reviewers used**: R2.1 is written as *"upgrades Conjecture
2.10/2.11 … to Theorem 3.13/3.15"*. The response letter carries **8 literal `3.13` and
9 literal `3.15`** (plus 2× `3.9`, 3× `3.5`), none of them `\ref`s; `methodology_T07.tex`
carries 4× `3.9`, 4× `3.8`; `supplementary_T07.tex` 3× `3.5`, 3× `3.9`. Every one of
them silently becomes wrong.

**Proposed fix, and it costs nothing:** place the new definition (and the lemma proposed
in §6) **after** Theorem 3.15, at the end of the section. Nothing before them moves,
3.13/3.14/3.15 keep their numbers, and the reading order — theorems first, then the
interface result that connects them to the pipeline — is if anything better. The
alternative (insert early + unnumber a remark) balances the count only for one insertion
and breaks again the moment a second environment is added.

### B1. The patch puts `𝒩` back inside canonicalisation. The code took it out, and the R1.3 answer in the same letter says it is out.

Three places in the patch:

| File | Text |
|---|---|
| `methodology_T07.tex`, Def. Fast Canonical String | *"The search is run on the normalised DAG `𝒩(D)` … and take `fcs_D := fcs_{𝒩(D)}`."* |
| `methodology_T07.tex` + `supplementary_T07.tex`, Thm 3.15 / A.3 | *"Let the fast canonical string be computed on the normalised DAG, `fcs_D := fcs_{𝒩(D)}`"* |
| `supplementary_T07.tex`, Table 3 (App. C) | keeps `D ← 𝒩(D)`, adding a **Precondition** line above it |

This contradicts three things that are already settled:

1. **The code.** `𝒩` was removed from `canonical.py` (3 sites), `labeled_dag.is_isomorphic`
   and `native/src/canonical.cpp` on 2026-07-29 (T07 work log; AC-6). Verified now,
   both engines:
   ```
   orphan CONST, backend=python: RuntimeError: Fast canonical D2S: no valid operation found.
   orphan CONST, backend=cpp   : RuntimeError: Fast canonical D2S: no valid operation found.
   ```
   `grep normalize_const_creation src/isalsr/core/canonical.py` returns only the three
   comments explaining why it is *not* called. So `fcs_D := fcs_{𝒩(D)}` describes
   behaviour the released code does not have. This is precisely the paper/code
   divergence Ezequiel himself made the deciding argument in T16:
   *"le pasarán el artículo y el código a un LLM, el cual se dará cuenta en menos de
   un segundo de cualquier incoherencia entre ambos."*

2. **The R1.3 answer, in the same letter, unchanged by this patch.**
   `response_to_reviewers_T07.tex:493–497`: *"The revision therefore presents `𝒩` as a
   step applied **at the interface between a host solver and the representation**, and
   states the canonical form on DAGs that satisfy the reachability condition, where
   `𝒩` is the identity. **Table 3 of the manuscript now states the precondition
   explicitly in place of the undefined call.**"* The patch's own `\changeref` for R1.3
   still promises *"Table 3 (Appendix C), first line replaced by an explicit statement
   of the precondition"* — while `supplementary_T07.tex` keeps the call. The
   `\changeref` and the patch disagree with each other.

3. **D-1.** `𝒩` is not isomorphism-equivariant (T07 §7 2026-07-29; adversarial
   population 18.1–20.1 % permutation failures, 0 inside the safe class `𝒞`).
   Defining `fcs_D := fcs_{𝒩(D)}` as a *total* construction extends the canonical
   string to exactly the DAGs with orphan `Const` nodes — the domain where it is
   **not** a complete invariant. Theorem 3.15 survives because its hypothesis still
   restricts to the reachability class, but the *definition* then defines a "canonical
   string" on inputs where it is not canonical. Restating on `𝒩(D)` was explicitly
   examined and **refuted** on 2026-07-29 for this reason; it should not come back
   without new evidence.

Also note the Table 3 line is internally incoherent on its own terms: it asserts a
**precondition** ("every non-`Var` node is reachable from a variable") and then applies
`𝒩` to *establish* that precondition. If it is assumed, `𝒩` is the identity and does
nothing; if `𝒩` establishes it, it is not a precondition.

**Proposed fix (small, and it costs no argument):**
- Keep the new numbered Definition of `𝒩` — it is good and it is what R1.3 asked for.
- Delete the sentence added to the Fast Canonical String definition; replace with one
  sentence stating that `𝒩` is applied by the producer at the host↔representation
  interface, and that canonicalisation *assumes* the precondition and refuses a DAG
  that violates it.
- Delete the inserted sentence from Theorem 3.15 and from A.3 entirely — it is
  unnecessary once `𝒩` is not in the path, and it is what AC-6 was discharged without.
- Table 3: keep the **Precondition** line, delete `D ← 𝒩(D)`, as the R1.3 `\changeref`
  already promises.

### B2. Step 3's load-bearing claim is false, and it is the same unproven assertion the submitted paper already had.

`supplementary_T07.tex`, Step 3:

> *"since a node is inserted only after a pointer has been placed on one of its
> in-neighbours, and the CDLL retains every inserted node, `σ(c)[0]` is already in the
> CDLL by the time `c` first becomes a candidate."*

The premise is about **some** in-neighbour of `c`; the conclusion is about the
**specific** in-neighbour `σ(c)[0]`. It does not follow.

Minimal counterexample, 4 nodes, `m = 2` (verified in code):

```
V = {x1, x2, Sin(a), Pow(p)},  E = {x1→a, a→p, x2→p},  σ(p) = (a, x2)
fcs = 'Vsnv^PnC'                                  # the algorithm handles it fine
```

A free-choice D2S run (Definition 3.5 permits any choice at a branch point) may place
the acting pointer on `x2` before `a` is inserted. There `p` is an uninserted
out-neighbour of `x2`, hence `p ∈ 𝒟_j`, while `σ(p)[0] = a` is **not** in the CDLL.
Rule 1 excludes `p` at that moment. The claim fails on a four-node DAG.

This matters more than an ordinary slip: it is verbatim the informal argument at
`methodology.tex:761–766` that §3.1 gap 3 asked to be *moved into the lemma **and made
rigorous***. It was moved; it was not made rigorous. R2 found the previous soft spot
in this proof themselves; they will read an unproven timing assertion the same way.

**Proposed fix — induction on a topological order of `D`.** For an excluded `c`, its
base `a = σ(c)[0]` is an in-neighbour, hence strictly earlier in every topological
order; by the reachability hypothesis `a` is a variable or reachable from one; by the
induction hypothesis `a` is *eventually* inserted; the CDLL retains it; and `𝒫_n`
enumerates every offset, so at some later step the acting pointer sits on `a` and `c`
becomes eligible. Conclusion: **Rule 1 defers `c`, it never strands it.** This is the
statement that is true, and it consumes the hypothesis just as visibly.

### B3. The non-emptiness corollary is stated per-position, where it is false; Step 4 then cites the version Step 3 never proved.

Step 3 closes with:

> *"In particular `𝒞_j ≠ ∅` whenever an out-neighbour of the tentative position is
> still uninserted."*

False, and B2's counterexample is again the witness: at pointer `x2` the only
uninserted out-neighbour is `p`, Rule 1 removes it, so `𝒞_j = ∅` **at that position**.
Step 4 then writes *"By Step 3 the loop over `𝒫_n` finds an admissible insertion
whenever a **reachable node** is uninserted"* — which is the correct statement, and is
not what Step 3 proved.

Two further consequences worth stating in the email:

- Non-emptiness must be quantified over the whole `𝒫_n` sweep, not over one pointer
  position. That is the only form the algorithm needs and the only form that is true.
- As written, Step 3 asserts a timing fact and Step 4 derives progress from Step 3;
  but *proving* Step 3's fact needs Step 4's progress. The clean repair is a **single
  well-founded induction carrying both**, i.e. Steps 3 and 4 merged, rather than two
  steps that lean on each other.

### B4. The sixth gap — the domain mismatch in the chain — is not addressed, and it breaks the proof's final inference.

Theorem 3.13 as stated (`methodology.tex:972–981`, `supplementary.tex:62–71`):

> *Let `w ∈ Σ_SR*` with `m ≥ 1` variables and **`D = S2D(w,m)`**. If every non-variable
> node of `D` is reachable … then `D ≅ S2D(**D2S(D, x₁)**, m)`.*

The new Lemma A.2 invokes it twice, and neither instantiation is licensed:

- Step 1: *"Under the reachability hypothesis `𝒲(D) ≠ ∅` … (Theorem 3.13)"* — applied to
  an arbitrary labeled DAG, but 3.13 quantifies only over the **image of S2D**.
- Step 5: *"applying Theorem 3.13 to `fcs_D` gives `D ≅ S2D(fcs_D, m)`"* — but 3.13's
  conclusion is about **one particular string**, `D2S(D, x₁)`, not about an arbitrary
  `w ∈ 𝒲(D)`. Having just spent five steps proving `fcs_D ∈ 𝒲(D)`, the theorem being
  invoked says nothing about members of `𝒲(D)`.

This was listed as item 5 of T07 §7bis.2 ("A round-2 reviewer checking the chain will
find it") and the patch does not touch Theorem 3.13.

**Proposed fix, and it is nearly free:** the *proof* of 3.13 in Appendix A already
establishes the stronger statement — it argues conditions (i)–(iv) for the bijection
`i2o` maintained by *the* D2S run and never uses `D = S2D(w,m)` nor the greedy choice.
So restate 3.13 as

> *Let `D` be a labeled DAG with `m ≥ 1` variables in which every non-variable node is
> reachable from a variable. Then `D ≅ S2D(w, m)` for **every** `w ∈ 𝒲(D)`.*

with the existing proof unchanged, or add it as a corollary. Lemma A.2's last line then
goes through verbatim.

---

## 2. Should fix

- **S1.** `𝒲(D) ≠ ∅` (Step 1) rests on the same over-broad citation of 3.13 as B4;
  it follows once 3.13 is restated.
- **S2.** Step 3 says *"Such an edge occurs in no `w ∈ 𝒲(D)`"*. Definition 3.5
  constrains **which node** is inserted next, not **from which in-neighbour**, and says
  nothing about `σ`. Under the definition as written a run may create a `Pow` node from
  its exponent; the decoded DAG then has `σ` reversed and is not isomorphic to `D` by
  condition (iv). So either Definition 3.5 must require the run to reproduce `D`
  *including operand order*, or Step 3 must argue that such runs are excluded from
  `𝒲(D)`. This also settles a framing question worth getting right: Rule 1 is a
  **correctness requirement**, not an optimisation — the paper already says so
  (*"ensures that each Pow node is created via its designated base source"*) — and the
  proof reads better if "non-exclusion" is stated as *Rule 1 removes no node from
  eventual insertability*, rather than *Rule 1 removes nothing*.
- **S3.** Arithmetic in Step 4: *"reaches 0 after exactly `(|V|−m)+|E|` operations"*.
  `(|V|−m)+|E|` is the **initial value of the measure**, not the number of operations: a
  `V`/`v` operation decrements it by 2 (one node **and** its creation edge), a `C`/`c`
  by 1. Each non-variable node has exactly one creation edge, so the run terminates
  after exactly **`|E|`** accepted operations. On B2's DAG: `(|V|−m)+|E| = 5`, actual
  operations in `Vsnv^PnC` = 3 = `|E|`. The wrong count is repeated verbatim in the
  response letter (Gap 1).
- **S4.** `methodology_T07.tex`, new definition paragraph: *"The added edge targets a
  variable, which is a pure source in every DAG produced by S2D or D2S."* False.
  Verified: `S2D("NC", m=2)` gives `x₁` in-degree 1 — `C` directs an edge into a
  variable. It also contradicts the paragraph's own next clause (*"the remaining case
  in which an arbitrary IsalSR string has directed an edge into a variable"*) and the
  R1.3 text at `:502–503` (*"An IsalSR string **can** direct an edge into a variable
  through the instructions `C` and `c`"*). The true statement is the adapter-scoped one
  already in R1.3 at `:473–475`: *neither host adapter* ever directs an edge into a
  variable. Scope the sentence to host-solver output.

---

## 3. Where Ezequiel is right, and where the patch is right

- **His three self-criticisms are all correct.** Trim the digressions; delete *"every
  checkable claim of the proof has been given an executable counterpart in the test
  suite"* (a theorem is not justified by a test suite, and a reviewer will read it as a
  substitute for the proof); delete *"the property this proof previously asserted is
  false as stated"* from the appendix. Note the asymmetry, though: both belong in the
  **response letter**, which may and should say the old sentence was false and may
  report the tests — it is the **article** that must not narrate its own review history.
- **The five-step skeleton is the right structure** and it does discharge gap 2 visibly
  (`𝒞_j ⊆ 𝒟_j`, with the false "same candidate pool" sentence named and removed). That
  is the single most important thing R2.1 asked for, and it is done.
- **Rule 1 is kept scoped to `Pow` alone** — correct under T16 Branch B, and consistent
  with Definition 3.9(iv) and Definition 3.2 as submitted.
- **Checked, and needs no change:** the `k = 0` scope worry recorded at the top of T07.
  `fcs` returns `''` for both `m = 1` and `m = 2` variable-only DAGs, but Theorem 3.15
  says *"Let `D₁` and `D₂` be labeled DAGs with `m ≥ 1` variables"* with one shared `m`,
  and Definition 3.9(iii) forces equal `m` anyway. No counterexample; no edit needed.
- **Process note:** the patch marks changes with `{\color{blue}…}`. The current
  manuscript has none, and README rule 4 requires the main PDF to be clean — colour
  goes in the separate Summary of Changes upload only.

---

## 4. Residual risk to record either way

Step 4 establishes that a **run** terminates; it does not bound the **search**. The
spiral sweep plus Rule 2 tie backtracking is not covered by the decreasing measure,
while `methodology.tex:679` advertises "near-`O(k²)`". 46 / 100,000 synthetic
hypothesis-satisfying DAGs at `k = 24–30` exceed a 10 s budget (T15 AC-3′). Production
sees none at 60 s. One sentence separating "terminates" from "terminates within the
advertised budget" would pre-empt the obvious round-2 follow-up.

---

## 5. Where `𝒩` belongs: proposal

Ezequiel is right that the normalisation must appear *somewhere* formal — it genuinely
runs, before canonicalisation, and the paper cannot stay silent about it. The
disagreement is only about **where**: not inside the definition of `fcs`, but as a
separate result at the host↔representation interface.

### 5.1 The fact that decides the form of the definition

**The operation that ran in every reported experiment is not
`LabeledDAG.normalize_const_creation`.** It is the adapters' `_normalize_const_edges`
(`bingo/adapter.py:212–216`, `udfs/adapter.py:216–224`), byte-identical in both:

```python
def _normalize_const_edges(dag: LabeledDAG) -> None:
    for i in range(dag.node_count):
        if dag.node_label(i) == NodeType.CONST and dag.in_degree(i) == 0:
            dag.add_edge(0, i)
```

Unconditional anchor to `x₁`. No least-index search, no acyclicity test.
`grep -rn normalize_const_creation experiments/models/` returns **nothing**: the general
routine is a library method used only by the measurement scripts. Three consequences:

1. **The interface form is isomorphism-equivariant by inspection.** It makes no
   node-index-ordered decision — each orphan `Const` independently receives `x₁`, and
   `φ(x₁) = x₁` by Definition 3.9(iii). **D-1 does not touch it.** The "we record one
   limitation" paragraph currently in R1.3 (`:483–497`) can be re-scoped to the general
   library variant instead of standing as a limitation of the pipeline.
2. **On the interface class the two forms coincide** — `x₁` is the least index and never
   closes a cycle there — so **no reported number depends on which one the paper
   defines.** There is no re-execution cost to this choice.
3. **The code's soundness silently depends on an unstated hypothesis.** `add_edge`
   returns `False` on a cycle-closing edge and `_normalize_const_edges` discards the
   return value — exactly the pattern that caused the T15 orphaning bug. It is safe here
   *only* because no `Var` is an edge target in adapter output. That hypothesis is
   load-bearing, unstated in the paper and unchecked at runtime. Stating it as a lemma
   is not bureaucracy; it is the precondition the released code relies on.

### 5.2 The proposal — one definition, one lemma, one corollary; the three theorems untouched

Place all three **after Theorem 3.15**, so nothing renumbers (B0).

> **Definition 3.16 (Constant creation edge).** Let `D` be a labeled DAG with variable
> nodes `x₁,…,x_m`. `𝒩(D)` is obtained from `D` by adding the edge `x₁ → c` for every
> node `c` with `ℓ(c) = Const` and in-degree 0. Nodes, labels, operand orders and all
> existing edges are preserved; no edge is removed. `𝒩(D) = D` when `D` has no `Const`
> node of in-degree 0.

> **Lemma 3.17 (The interface step establishes the precondition).** Let `D` be an
> expression DAG delivered by a host solver, i.e. a labeled DAG such that
> **(a)** every node of in-degree 0 is a `Var` or a `Const`, and
> **(b)** no `Var` node is the target of an edge. Then:
> 1. `𝒩(D)` is acyclic;
> 2. every non-`Var` node of `𝒩(D)` is reachable from a variable — i.e. `𝒩(D)` satisfies
>    the hypothesis of Theorem 3.13;
> 3. `𝒩` is isomorphism-equivariant on this class, and `𝒩(D₁) ≅ 𝒩(D₂) ⟺ D₁ ≅ D₂`;
> 4. `eval(𝒩(D)) = eval(D)` and the output node is unchanged.

> **Corollary 3.18.** For host DAGs `D₁, D₂` satisfying (a)–(b),
> `fcs_{𝒩(D₁)} = fcs_{𝒩(D₂)} ⟺ D₁ ≅ D₂`.

Proofs, all short enough for Appendix A:

1. `x₁` has no in-edges by (b), so a cycle through `x₁ → c` would need a path `c ⇝ x₁`,
   whose last edge makes `x₁` an edge target — contradiction with (b).
2. Induction on a topological order. After `𝒩`, every `Const` has `x₁` among its
   in-neighbours. Any other non-`Var` node `v` has in-degree ≥ 1 by (a); take an
   in-neighbour `u`, which is either a `Var` or, being topologically earlier, reachable
   by the induction hypothesis.
3. Let `φ` be an isomorphism `D₁ → D₂`. By Def. 3.9(ii)–(iii), `c` is a `Const` of
   in-degree 0 in `D₁` iff `φ(c)` is one in `D₂`, and `φ(x₁) = x₁`; so `φ` is also an
   isomorphism `𝒩(D₁) → 𝒩(D₂)`. Conversely, on this class every `Const` has in-degree 0
   in `D` and in-degree 1 in `𝒩(D)`, so `D` is recovered from `𝒩(D)` by deleting every
   in-edge of every `Const`, and that inverse commutes with isomorphisms.
4. A `Const` ignores its in-edges and no out-degree changes, so the sink set — hence the
   output node — is unchanged.

### 5.3 What this buys, against the alternatives already examined

| Route | Verdict |
|---|---|
| **Definition + interface lemma (this proposal)** | Theorems 3.13/3.14/3.15 stay **exactly as submitted**; `fcs` stays a pure function of `D`; the pipeline argument becomes explicit; Corollary 3.18 is the statement the deduplication experiments actually rely on and which the paper currently never makes; D-1 never enters. |
| Restate the theorems on `𝒩(D)` (the patch) | **Refuted 2026-07-29** by D-1, and contradicts the code (B1). |
| Prose-only scoping (T07 §7 option 3) | Cheapest in pages, weakest formally, and leaves hypothesis (b) — on which the code silently depends — unstated. |

Point 3 of the lemma is worth having on its own: it is what licenses using `fcs∘𝒩` as a
deduplication key on host output, which is the entire experimental contribution. Right
now that step is taken empirically (0 disagreements in 123,240 permutation tests on
15,530 Bingo DAGs, `frac_in_c2 = 1.0000`) and is nowhere justified.

Hypothesis (b) is also worth **enforcing in code**, not just asserting: one `assert` in
`_normalize_const_edges` on the return value of `add_edge` would turn the silent
dependency into a loud one. Small hardening item for Mario, not required for the paper.

---

## 6. Draft email to Ezequiel (Spanish)

> Asunto: T07 — revisión del arreglo del Lema A.2 y propuesta sobre `𝒩`

> **Nota (versión definitiva del correo).** Los puntos 3 y 4 de abajo se
> mantienen; el punto sobre los conjuntos de candidatos cambió de signo al
> comprobar el código, y va ahora como punto **2 bis**. Los ficheros `.tex`
> adjuntos ya implementan la versión corregida.

Hola Ezequiel,

Hemos revisado los tres ficheros. El esqueleto en cinco pasos nos parece el correcto y
resuelve lo esencial de R2.1: la frase falsa *"exactly the same candidate pool as D2S"*
se nombra explícitamente y se sustituye por la inclusión `𝒞_j ⊆ 𝒟_j`, que es justo la
contradicción que el revisor había localizado. También estamos de acuerdo con tus tres
recortes, con un matiz al final. Antes de integrarlo hay cinco puntos que creemos que
hay que corregir, y una propuesta sobre dónde debe ir la normalización.

**1. La numeración se desplaza.** Al insertar la definición nueva antes de la definición
de Fast Canonical String, todo lo que va después sube un número. En el PDF que has
enviado, Round-Trip Fidelity pasa a ser **Teorema 3.14**, el lema a **Lema 3.15** y el
teorema de invariante completa a **Teorema 3.16**. Son exactamente los números que usan
los revisores: R2.1 está escrito como *"upgrades Conjecture 2.10/2.11 to Theorem
3.13/3.15"*. En la carta de respuesta hay 8 apariciones literales de `3.13` y 9 de
`3.15`, ninguna con `\ref`. Propuesta: colocar la definición (y el lema del punto 6) **al
final de la sección, después del Teorema 3.15**. Así no se mueve nada y además se lee
mejor: primero los teoremas, y después el resultado de interfaz que los conecta con el
pipeline.

**2. `fcs_D := fcs_{𝒩(D)}` habría que quitarlo (tres sitios).** Desde el 29/07 el código
ya no aplica `𝒩` dentro de la canonicalización: lo verificamos hoy en ambos motores, el
canonicalizador lanza `RuntimeError` ante un `Const` de grado de entrada 0, y en
`canonical.py` sólo quedan los tres comentarios que explican por qué no se llama. Con el
enunciado del parche publicaríamos un artículo que describe un código que no tenemos, que
es justo el riesgo que señalaste en T16. Además la respuesta a R1.3, en esa misma carta y
sin tocar, dice que `𝒩` actúa *en la interfaz host↔representación* y que la primera línea
de la Tabla 3 se sustituye por la precondición — y el `\changeref` del parche sigue
prometiendo esa sustitución mientras el apéndice mantiene la llamada. Por último, `𝒩` no
es equivariante frente a isomorfismos en general (D-1), así que definir `fcs` sobre
`𝒩(D)` extiende la construcción precisamente al dominio donde no es invariante completa.

**2 bis. El hueco 2 va al revés de como lo teníamos, y esto es lo más
importante del correo.** Al comprobarlo contra el código: **D2S ya aplica el
predicado exacto de la Regla 1** — `dag_to_string.py:338–341` es literalmente el
mismo test que `canonical.py:647–649` y `:725–727`, con el mismo ámbito
`BINARY_OPS`; se añadió como corrección de orden de operandos. La Regla 1 no
restringe el conjunto de candidatos de D2S: lo reenuncia. Es decir, la frase de
la demostración enviada, *"exactly the same candidate pool as D2S"*, **es cierta
de los dos algoritmos**, y el paso 2 del parche —que concede al revisor que es
falsa— estaría concediendo algo incorrecto.

El defecto está un nivel más abajo: **la Definición 3.5 describe D2S sin su
restricción de primer operando**, y el pseudocódigo de D2S (Tabla 2) tampoco la
recoge. Con esa lectura `𝒲(D)` es estrictamente mayor que el conjunto de cadenas
que D2S puede producir, y **el Teorema 3.13 es falso**: para `D = x₁^x₂` la
cadena `NV^Nc` coloca todos los nodos y todas las aristas de `D` —luego
pertenece a `𝒲(D)` tal como está escrita la definición— pero decodifica a
`x₂^x₁`, y `is_isomorphic` devuelve `False`. Contraejemplo de tres nodos,
comprobado.

La reparación es enunciar la restricción en la Definición 3.5 y en la Tabla 2.
Hecho eso, la identidad de conjuntos `𝒞_j = 𝒟_j` se *deriva* en lugar de
afirmarse, y —esto es lo que hace falta para el punto 5— el Teorema 3.13 puede
ampliarse a todo `w ∈ 𝒲(D)`, cosa que sin este arreglo no sería lícita.

**3. El paso 3 tiene una afirmación falsa, y es la que sostiene el argumento.** Dice que
`σ(c)[0]` ya está en la CDLL cuando `c` aparece como candidato. La premisa habla de
*alguno* de los in-vecinos de `c` y la conclusión de *uno concreto*. Contraejemplo de
cuatro nodos, comprobado en código: `V = {x₁, x₂, Sin(a), Pow(p)}`,
`E = {x₁→a, a→p, x₂→p}`, `σ(p) = (a, x₂)`. Una ejecución con elección libre puede situar
el puntero en `x₂` antes de insertar `a`: allí `p ∈ 𝒟_j` y `σ(p)[0] = a` todavía no está
insertado. Es literalmente el argumento informal de `methodology.tex:761–766`, que era el
que había que trasladar al lema **y hacer riguroso**; se ha trasladado pero no reparado.
Se arregla con inducción sobre un orden topológico: `a` es in-vecino de `c`, luego es
anterior; por hipótesis de alcanzabilidad acaba insertándose; y `𝒫_n` recorre todos los
desplazamientos, así que en algún momento el puntero se sitúa sobre `a`. La conclusión
correcta es que **la Regla 1 aplaza `c`, nunca lo pierde.**

**4. La no-vacuidad está enunciada por posición, donde es falsa.** *"`𝒞_j ≠ ∅` siempre
que quede sin insertar un out-vecino de la posición tentativa"*: en el ejemplo anterior,
en `x₂` el único out-vecino sin insertar es `p`, la Regla 1 lo excluye y `𝒞_j = ∅` en esa
posición. El paso 4 cita después la versión buena (*"el bucle sobre `𝒫_n` encuentra una
inserción admisible siempre que quede un nodo alcanzable sin insertar"*), que es la que
hay que demostrar. Nota además que tal como está, el paso 3 asume un hecho temporal y el
paso 4 deriva el progreso del paso 3, pero demostrar el hecho del paso 3 requiere el
progreso del paso 4: lo limpio es una única inducción que lleve los dos.

**5. Falta el sexto hueco, y rompe la última inferencia.** El Teorema 3.13 está enunciado
sólo para `D = S2D(w,m)` y concluye sobre `D2S(D,x₁)`, una cadena concreta. El lema lo
invoca dos veces fuera de eso: para `𝒲(D) ≠ ∅` con `D` arbitrario, y al final para un
`w = fcs_D ∈ 𝒲(D)` cualquiera. Después de cinco pasos demostrando `fcs_D ∈ 𝒲(D)`, el
teorema que se invoca no dice nada sobre los elementos de `𝒲(D)`. La buena noticia es que
su demostración del Apéndice A ya prueba la versión fuerte — no usa en ningún momento que
`D` sea imagen de S2D ni que la elección sea la voraz — así que basta reenunciarlo: *"sea
`D` un DAG etiquetado con `m ≥ 1` variables en el que todo nodo no-variable es alcanzable
desde una variable; entonces `D ≅ S2D(w,m)` para **todo** `w ∈ 𝒲(D)`"*, con la
demostración intacta.

**Menores.** (i) El contador del paso 4: `(|V|−m)+|E|` es el valor inicial de la medida,
no el número de operaciones — una `V`/`v` la decrementa en 2 (nodo *y* arista de
creación), así que la ejecución termina tras exactamente `|E|` operaciones. En el DAG de
arriba: `(|V|−m)+|E| = 5`, operaciones reales en `Vsnv^PnC` = 3 = `|E|`. El número
equivocado se repite en la carta (Gap 1). (ii) *"the added edge targets a variable, which
is a pure source in every DAG produced by S2D or D2S"* no es cierto: `S2D("NC", m=2)` deja
`x₁` con grado de entrada 1, porque `C` puede dirigir una arista hacia una variable. Se
contradice además con la frase siguiente del propio párrafo y con la respuesta a R1.3.
Lo correcto es acotarlo a la salida de los adaptadores, como ya se dice en R1.3.

**6. Dónde poner la normalización — nuestra propuesta.** Tienes razón en que hay que
formalizarla: se ejecuta de verdad, antes de canonicalizar. El dato que nos parece que
decide la forma es que la operación que corrió en todos los experimentos publicados **no
es** `normalize_const_creation`, sino la de los adaptadores (`_normalize_const_edges`,
idéntica en Bingo y UDFS): añadir `x₁ → c` a todo `Const` de grado de entrada 0, sin
búsqueda de índice mínimo y sin test de aciclicidad. Sobre la clase de interfaz las dos
coinciden, así que no cambia ningún número; pero la forma de interfaz **no toma ninguna
decisión dependiente del índice de nodo, luego es equivariante por inspección y D-1 no la
afecta.**

Proponemos, todo colocado después del Teorema 3.15 para no renumerar:

- **Definición.** `𝒩(D)` añade `x₁ → c` a todo `Const` de grado de entrada 0; no elimina
  ninguna arista; es la identidad si no hay ninguno.
- **Lema (la etapa de interfaz establece la precondición).** Si `D` es un DAG de
  expresión entregado por un solver, es decir (a) todo nodo de grado de entrada 0 es
  `Var` o `Const` y (b) ninguna `Var` es destino de una arista, entonces: (1) `𝒩(D)` es
  acíclico; (2) todo nodo no-`Var` de `𝒩(D)` es alcanzable desde una variable, es decir
  `𝒩(D)` cumple la hipótesis del Teorema 3.13; (3) `𝒩` es equivariante en esa clase y
  `𝒩(D₁) ≅ 𝒩(D₂) ⟺ D₁ ≅ D₂`; (4) `eval(𝒩(D)) = eval(D)` y el nodo de salida no cambia.
- **Corolario.** Para DAGs de solver, `fcs_{𝒩(D₁)} = fcs_{𝒩(D₂)} ⟺ D₁ ≅ D₂`.

Las cuatro demostraciones son de tres o cuatro líneas cada una (te las mandamos escritas
si te sirve). Con esto los Teoremas 3.13, 3.14 y 3.15 se quedan **exactamente como se
enviaron**, `fcs` sigue siendo función pura de `D`, y el corolario dice por fin lo que los
experimentos usan en realidad —que la cadena canónica es una clave de deduplicación
correcta sobre la salida de los solvers— que hoy sólo está justificado empíricamente.
Añadimos que la hipótesis (b) es además la que hace correcto el código: el adaptador
descarta el valor de retorno de `add_edge`, que es el mismo patrón que causó el fallo de
T15, y sólo es seguro porque ninguna variable es destino de arista. Enunciarla como lema
no es formalismo: es la precondición de la que depende el código publicado.

**Sobre tus tres recortes, de acuerdo, con un matiz.** Los tres nos parecen correctos.
El matiz es que la asimetría importa: tanto la evidencia de los tests como el
reconocimiento de que la frase antigua era falsa **sí** deben quedarse en la carta de
respuesta —es el sitio donde se habla del proceso de revisión y donde admitirlo antes de
que lo encuentren juega a favor—; es el **artículo** el que no debe narrar su propia
historia de revisión. Y el marcado en azul habrá que quitarlo del PDF principal, que
según las normas de la revista va limpio; va en el "Summary of Changes".

Un saludo,
Mario

---

## 7. Verification script

`/tmp/claude-1000/…/scratchpad/check_t07_ez.py` — reproduces C1 (variable as edge
target under S2D), C2 (canonicaliser refuses orphan `Const`, both engines), C3 (`k = 0`
strings), C4 (`Pow` with base and exponent on different variable branches). The B0
numbering table comes from `pdftotext maindocument.pdf`; the adapter facts from
`grep -rn normalize_const_creation experiments/models/` (no hits).
