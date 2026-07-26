# Reviewer 1

**Overall rating: Good.** Technically sound: *"Appears to be - but didn't check completely."*
Tone is favourable and well-informed — this reviewer clearly read the method and understood the positioning. Four of the five comments are substantive; none disputes the core contribution.

## Opening assessment (verbatim)

> The paper addresses a real and underdressed problem: symbolic regression search wastes fitness evaluations on structurally isomorphic DAGs that differ only in node numbering. The canonical string approach is principled, the theoretical results are carefully stated and proved, and the drop-in integration design is practically sensible. The empirical protocol is rigorous (50 problems, 30 seeds, Demsar-style paired inference), and the result that deduplication fires on every problem in the suite is clean and credible.
>
> However, there are some issues to address:

("underdressed" is a typo for "underaddressed" in the original.)

Note what is already conceded: the problem is real, the theory is "carefully stated and proved", the protocol is "rigorous", and the every-problem result is "clean and credible". The 50-problem / 30-seed / Demšar protocol is explicitly endorsed by this reviewer — which sits against R3.1, who wants broader coverage.

---

## R1.1 — Bingo speedup framing

**Verbatim:**

> 1) The bingo search-only speedup is s = 0.93, a net loss under a fixed wall-clock budget, yet the paper characterizes the overhead as "approximately neutral." This needs a more honest framing. The claim holds only on the subset of problems where rho is large enough to compensate.

**Type**: framing / honesty of claims. No new experiment strictly required; the data to delimit the regime already exists.

**Where the objectionable phrasing is**: `article/paper/discussion.tex:66–68`

> When evaluation cost is comparable to canonicalisation cost, as in Bingo where both are sub-millisecond, the median overhead is $39\%$ and the wall-clock effect is **approximately neutral** after the search-time savings offset the canonicalisation cost.

**Where $S = 0.93$ is reported**: `article/paper/results.tex:58` (Table `tab:three_axis`, Bingo row, column $S$). UDFS is $S = 1.07$ in the same column.

**What the paper already says that supports the reviewer**: `article/paper/results.tex:180–185` already restricts the claim to a subset —

> Under a fixed wall-clock budget this overhead removes roughly the same fraction of search time, which the higher Bingo $\rho$ partially compensates: on the $30$ problems where Bingo exceeds $\rho = 1.85$ the search-only speedup recovers to $S \geq 0.95$, and on three problems---N-8, II.11.27, and Keijzer-11---it exceeds unity.

So the *results* section is already conditional; the *discussion* section overstates. The reviewer's complaint is precisely the mismatch between the two.

**Corroborating per-problem data**: `article/supplementary/table_supplementary_bingo.tex`. Only 4 of 50 rows have $T_{\mathrm{IS}} < T_{\mathrm{BL}}$ (I.6.20a, II.11.27, Keij-11, N-8). The other 46 rows are slower under IsalSR. The Nguyen rows carry the highest overheads (N-12 61.3%, N-5 60.5%, N-4 59.6%, N-3 58.2%, N-7 56.8%).

**Also relevant**: abstract (`main.tex:81`) says overhead is "at a median of $39\%$ in the sub-millisecond evaluation regime" — that phrasing is accurate; it is only the discussion that says "approximately neutral".

---

## R1.2 — Reachability condition failure rate

**Verbatim:**

> 2) The reachability condition in Theorems 3.13 and 3.15 gate the completeness guarantee, but the paper never reports how often this condition fails in practice.

**Type**: new measurement.

**What the condition is** (`article/paper/methodology.tex:972–981`, Theorem `thm:roundtrip`):

> If every non-variable node of $D$ is reachable from some variable via directed paths, then $D \cong \mathrm{S2D}(\mathrm{D2S}(D, x_1), m)$.

Theorem 3.15 (`thm:invariant`, `methodology.tex:1057–1067`) inherits it: "both satisfying the reachability condition of Theorem~\ref{thm:roundtrip}".
Lemma 3.14 / A.2 (`lem:fcs_valid`) also inherits it.
Rule 1 of the FCS definition additionally leans on it (`methodology.tex:762–766`): the argument that Rule 1 excludes no valid ordering is explicitly conditioned on "the reachability precondition (Theorem~\ref{thm:roundtrip})".

**Numbering cross-walk**: reviewer's "Theorems 3.13 and 3.15" = `thm:roundtrip` and `thm:invariant`. See `manuscript-map.md`.

**What the paper currently reports instead**: only a *collision* claim, not a *precondition-violation* claim — `article/paper/discussion.tex:36–40`:

> no false collision has been observed across the $14{,}841$ DAGs in the unit-test suite or the millions generated during the SR experiments

That is a different quantity from what R1 asks for. The reviewer wants the rate at which candidate DAGs arriving at the canonicaliser fail the precondition in the first place, and by implication what the deduplication wrapper does with them.

**Related unreported behaviour in the same area**: the 60-second canonicalisation timeout, `article/paper/discussion.tex:104–107` — "We set a $60$-second canonicalisation timeout and count timed-out DAGs as unique". Timeouts and reachability failures are two distinct fallback paths; only the first is documented.

**Where this would be measured**: see `codebase-pointers.md` (`src/isalsr/core/dag_to_string.py` and `labeled_dag.py` both contain reachability logic).

---

## R1.3 — Undefined `normalize_const_creation`

**Verbatim:**

> 3) The pseudocode (Table 3, Appendix C) opens with a call to normalize_const_creation(D), defined only as "redirect all CONST creation edges to x1," that appears nowhere else in the paper?

**Type**: missing definition. Confirmed.

**Only occurrence in the whole manuscript**: `article/supplementary/supplementary.tex:398–399`, first line of the FastCanonical pseudocode —

> $D \leftarrow \mathtt{normalize\_const\_creation}(D)$ \hfill *// redirect all \textsc{Const} creation edges to $x_1$*

Grep confirms no other occurrence in `article/paper/*.tex` or elsewhere in `supplementary.tex`. It also appears in the commented-out duplicate of the same pseudocode inside `methodology.tex:830–831`, which is inside a `\begin{comment}` block and therefore not rendered.

**What it actually does** — the implementation docstring, `src/isalsr/core/labeled_dag.py:591–608`:

> Return a new DAG with all CONST creation edges moved to x_1 (node 0).
>
> CONST nodes are evaluation-neutral leaves: they ignore in-edges and return `const_value` directly. But D2S requires every node to be reachable from a VAR via outgoing edges, so V/v creates a "creation edge" pointer → CONST. The choice of creation source is semantically irrelevant but produces different canonical strings.
>
> This normalization eliminates that redundancy by standardizing all CONST creation edges to come from node 0 (x_1). This is always valid because x_1 has no incoming edges (no cycle risk).
>
> The normalized DAG:
> - Computes the same function: eval(D) == eval(normalize(D))
> - Has deterministic CONST creation edges
> - Produces a unique canonical string for each equivalence class

**Why this matters beyond a definition gap**: the step is a *precondition of the invariance theorem*, not a cosmetic preprocessing pass. Without it, two isomorphic DAGs whose CONST nodes were created from different sources get different canonical strings. Theorem 3.15 as stated does not mention it. Call sites: `src/isalsr/core/canonical.py:95, 146, 231` (guarded by `dag._has_const_nodes()`), and `labeled_dag.py:458`.

**Related**: `\textsc{Const}` (label `k`) is in $\mathcal{T}$ per Table 1 (`methodology.tex:86`) and in $\Sigma_{\mathrm{SR}}$ per `computational_experiments.tex:64–66`, so CONST nodes are in scope for the benchmark suite.

---

## R1.4 — Missing naive-hash deduplication baseline

**Verbatim:**

> 4) There is no comparison against naive hash-based deduplication on a fixed-order DAG serialization. This is the obvious baseline and its absence makes it hard to assess how much of the benefit requires 1-WL machinery versus a much simpler approach.

**Type**: new experiment. This is the heaviest request in the whole review round and the one most likely to decide round 2.

**What is being asked for**: an ablation isolating the contribution of the 1-WL-guided canonicalisation against a cheap alternative — serialise each DAG in a fixed node order, hash the serialisation, deduplicate on the hash. The question is how many of the duplicates IsalSR catches would also be caught that way, and at what cost.

**Why the comparison is not trivially favourable to IsalSR**: a fixed-order serialisation hash is *sound but incomplete* — it never merges non-isomorphic DAGs, but it fails to merge isomorphic DAGs that differ in node numbering, which is exactly the redundancy IsalSR targets. The interesting quantity is therefore the *gap*, and the gap depends on how much of the observed $\rho$ comes from genuine node-renumbering versus from re-generation of byte-identical candidates.

**Context that bears on the answer**:
- `article/supplementary/supplementary.tex:689–693` already argues that UDFS's duplicates "arise only from the commutative symmetries of ADD and MUL", i.e. from operand permutation, which a fixed-order hash would *not* catch.
- `article/supplementary/supplementary.tex:782–786`: on 5,400 synthetic DAGs every one has trivial automorphism group and $\rho = k!$ exactly — a regime where a fixed-order hash would catch nothing beyond exact repeats.
- Existing per-DAG cost figures for the comparison's cost axis: $T_{\mathrm{canon}} = 0.296$ ms (UDFS), $0.817$ ms (Bingo), `results.tex:57–58`.

**Related existing baselines in the paper**: none. Related-work positions IsalSR against equality saturation (`defranca2023`, `defranca2025eggp`) and GraphSR (`xiang2025graphsr`) in `related_work.tex:43–56`, but no method is run as a comparator anywhere — the entire evaluation is baseline-vs-IsalSR *within* UDFS and Bingo.

---

## R1.5 — Writing pass

**Verbatim:**

> 5) Overall, please read through the paper to fix minor writing issues and improve overall readability of it.

**Type**: prose. Reinforced by this reviewer's structured answers: C3 (introduction) *"Could be improved"*, C5 (readability) *"Readable - but requires some effort to understand"*.

Overlaps R2.8 (duplicated abstract phrase, ISALSR/IsalSR, canonicalisation/canonicalization) and R3.2 (same abstract typo). See `verified-discrepancies.md` for the enumerated instances.

---

## Structured answers (verbatim)

- **A1. Category**: Research/Technology
- **A2. Relevance**: Relevant
- **B1. Significance**: Good
- **B3. Technically sound**: Appears to be - but didn't check completely
- **B4. Experimental validation**: Lacking in some respects; some cases of interest not tested
- **C1. Title/abstract/keywords appropriate**: Yes
- **C2. References**: References are sufficient and appropriate — suggested references: **NA**
- **C3. Introduction**: Could be improved
- **C4. Organization**: Satisfactory
- **C5. Readability**: Readable - but requires some effort to understand
- **C6. Length**: About right
- **C7. Supplemental material**: Yes, as part of the digital library for this submission if accepted
- **C8. Supplemental accept**: After revisions. Please include explanation under Public Comments below.
- **Overall**: Good

### B2 — significance statement (verbatim, complete)

> The paper fills a gap that prior SR work leaves open. Diversity-preservation mechanisms such as fitness sharing and age-fitness Pareto selection, and algebraic deduplication via equality saturation, target different sources of redundancy; neither addresses equivalent node orderings of the same DAG structure. ISALSR tackles this directly and is complementary to both.
>
> The canonical form itself is non-trivial to design for this setting. Prior instruction-based encodings handle unlabeled undirected graphs (IsalGraph) or chemistry-specific molecular graphs (IsalChem); neither accommodates operation-type labels, directed edges, or a non-commutative binary operator. The decision to absorb subtraction and division as unary NEG and INV operators is the key move that keeps ADD and MUL commutative and simplifies the isomorphism definition without narrowing the expression space.
>
> On the algorithmic side, the 1-WL-guided greedy canonicalization runs in near-O(k^2) on typical SR expressions and is provably complete under the reachability condition -- a meaningful middle ground between the O(k!) exhaustive search and hash-based approximations that offer no correctness guarantee.

This paragraph is useful material: the reviewer states the contribution back accurately, including "near-O(k^2)" — which matches `methodology.tex:885–887` and contradicts the "near-linear" wording used in `conclusion.tex:11` and `discussion.tex:97` (see `verified-discrepancies.md`, E3). It also pre-frames R1.4: the reviewer already characterises hash-based approaches as offering "no correctness guarantee", so the requested baseline is expected to lose on completeness and the open question is by how much and at what cost.
