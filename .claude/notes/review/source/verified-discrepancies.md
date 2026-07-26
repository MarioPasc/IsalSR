# Verified discrepancies

Every factual claim made by a reviewer, checked line by line against the `.tex` sources, plus discrepancies found during the same pass that no reviewer raised. Paths are relative to `journal/69c1637a28a81fea2badda9a/`.

**Verdict column**: CONFIRMED = the reviewer's description of the source is accurate. Nothing in this file proposes a fix.

---

## Part 1 — Reviewer-raised

### D1 · Total run count `2,640` vs `6,000` — CONFIRMED (R2.6)

| | |
|---|---|
| Where | `article/supplementary/supplementary.tex:560–561` and `:574` |
| Text | "yielding $2 \times 2 \times (12 + 10) \times 30 = 2{,}640$ total runs" / "All $2{,}640$ runs execute on CPU-only nodes of the Picasso supercomputer" |
| Fact | 2 methods × 2 variants × **50** problems × 30 seeds = **6,000**. The `(12 + 10)` factor is the 22-problem arXiv suite (12 Nguyen + 10 Feynman). |

**Complication the reviewer did not raise**: the cell counts in the main text do not reach 6,000 either. `article/paper/results.tex:68–70` reports ρ "across $1{,}500$ seed-problem cells" for UDFS (= 50 × 30 ✓) but "$1{,}465$ cells" for Bingo — **35 short**. Unexplained anywhere. Almost certainly the same root cause as D4.

---

### D2 · Feynman problem count: 20 vs 10 vs 24 — CONFIRMED (R2.5)

| Claim | Where | Value |
|---|---|---|
| "a 20-problem subset of AI Feynman" | `article/paper/computational_experiments.tex:79–81` | 20 |
| Table 5 (`tab:feynman`) | `article/supplementary/supplementary.tex:513–536`; caption/prose at `:469–471` says "10 physics equations" | 10 |
| Tables 6–7, Feynman rows | `table_supplementary_udfs.tex`, `table_supplementary_bingo.tex` | 24 |

**True composition of the 50-problem suite**, counted from the per-problem tables (both tables have exactly 50 data rows, identical problem sets):

| Family | n | IDs |
|---|---|---|
| Nguyen | 12 | N-1 … N-12 |
| AI Feynman | 24 | I.6.20a, I.10.7, I.12.1, I.12.4, I.13.12, I.14.3, I.15.10, I.16.6, I.25.13, I.29.16, I.30.3, I.34.27, I.37.4, I.39.10, I.44.4, I.48.20, I.50.26, II.3.24, II.11.3, II.11.27, II.11.28, III.10.19, III.14.14, III.17.37 |
| Vladislavleva | 3 | Vlad-2, Vlad-4, Vlad-7 |
| Livermore | 3 | Liv-4, Liv-14, Liv-19 |
| R (Koza rational) | 3 | R1, R2, R3 |
| Pagie | 2 | Pagie-1, Pagie-2 |
| Keijzer | 2 | Keij-6, Keij-11 |
| Korns | 1 | Korns-12 |
| **Total** | **50** | |

So the true split is **12 + 24 + 14**, not the "32 core (12 Nguyen + 20 Feynman) + 18 extension" described at `computational_experiments.tex:78–91`.

**Downstream reproducibility gap**: `computational_experiments.tex:52–54` and `:92–95` both promise "The benchmark tables in Appendix~D.1 list every problem with its expression, input dimensionality, sampling protocol, and source citation." Appendix D.1 documents 12 Nguyen + 10 Feynman = 22. **28 of the 50 problems have no expression, no variable range, and no sampling protocol anywhere in the submission.**

---

### D3 · "Table 4 of the main document" does not exist — CONFIRMED (R2.4)

| | |
|---|---|
| Where | `article/supplementary/supplementary.tex:120`, first line of the proof of Lemma A.2 |
| Text | "The fast canonical algorithm (Table~4 of the main document) uses the same insertion primitives as D2S" |
| Fact | Main paper has exactly 3 tables (`tab:operations`, `tab:three_axis`, `tab:cpdt_summary`). The FCS pseudocode is `tab:canon_pseudo` = **Table 3 of the supplementary, Appendix C**. |

See `manuscript-map.md` for the full inventory of hardcoded cross-document references, one more of which (`supplementary.tex:453`, "Section 5.1 of the main paper") is also wrong.

---

### D4 · `nan` results for Vlad-2 and Korns-12 — CONFIRMED (R2.7)

`article/supplementary/table_supplementary_bingo.tex`:

| Line | Problem | $R^2$ BL | $R^2$ IS | NRMSE BL | NRMSE IS | $d$ | ρ |
|---|---|---|---|---|---|---|---|
| 38 | Korns-12 | `\underline{0.0000}` | `\mathbf{nan}` | `\underline{1.0131}` | `\mathbf{nan}` | +0.00 [+0.00, +0.00] | 1.82 ± 0.00 |
| 60 | Vlad-2 | `\underline{0.9385}` | `\mathbf{nan}` | `\underline{0.1966}` | `\mathbf{nan}` | +0.00 [+0.00, +0.44] | 1.83 ± 0.00 |

Three distinct problems in these two rows:

1. **Undocumented failures.** No cause given anywhere in the manuscript.
2. **`nan` is typeset as the winner.** The table caption defines "**Bold**: better of BL/IS. <u>Underline</u>: worse of BL/IS." In both rows `nan` is bold and the finite baseline value is underlined. Vlad-2 marks a real $R^2 = 0.9385$ as *worse than* `nan`. R2 did not state this explicitly; it is visible in the rendered table.
3. **Handling in the paired test is unspecified.** The test (`computational_experiments.tex:170–180`) forms $\delta_i = \bar m_i^{IS} - \bar m_i^{BL}$ per problem over $N = 50$. A NaN per-problem mean makes $\delta_i$ undefined. $N = 50$ is asserted throughout — `computational_experiments.tex:160`, `:231`, and Table 3. The Wilcoxon description (`:229–230`) mentions only that "zero-valued differences are excluded"; NaN is never mentioned.

Cross-reference: `article/paper/discussion.tex:116–118` discusses Korns-12 ("problems whose primary difficulty is constant discovery … IsalSR therefore neither helps nor hurts") without mentioning its NaN.
Cross-reference: the 1,465-vs-1,500 Bingo cell shortfall in D1.

Note UDFS has no NaN: `table_supplementary_udfs.tex:38` gives Korns-12 as 0.0001/0.0001 and `:60` gives Vlad-2 as 0.3945/0.4021.

---

### D5 · Abstract duplication — CONFIRMED (R2.8a, R3.2)

`article/paper/main.tex:81`:

> A paired test across problems on the empirical  reduction factor returns Cohen's $d > 2$ at $p < 10^{-21}$ **on both methods on both methods**: canonicalisation eliminates a mean of $34\%$ …

Same line also has a **double space** in "empirical  reduction factor" (not reported by either reviewer).
The abstract is duplicated in `double_blind/paper/main_anonymous.tex`; the same headline numbers are restated in `previously_published_statement/main.tex:98–115`.

---

### D6 · Naming and spelling inconsistency — CONFIRMED (R2.8b, R2.8c)

**ISALSR vs IsalSR.** The macro `\IsalSR` = `\textsc{IsalSR}` (`main.tex:51`) renders as small-caps "ISALSR". Plain-text "IsalSR" bypassing the macro appears at `related_work.tex:23, 50, 76, 111` (and in `introduction.tex` prose). The rendered PDF therefore alternates. The defect is at the usage sites, not in the macro.

**canonicalisation vs canonicalization.** Both spellings in active use:
- `-isation`: `main.tex:81`; `computational_experiments.tex:33, 40`; `results.tex:6, 168, 193`; `discussion.tex:18, 68`; `conclusion.tex:9`
- `-ization`: `methodology.tex:679`; `supplementary.tex:485, 563, 567, 636, 645, 650, 675, 695, 710, 713, 741, 752, 768, 792, 800`

Roughly: main paper British, supplementary American, neither clean. Same split affects **neighbourhood/neighborhood**, **labelled/labeled** (`supplementary.tex:782` "labelled" vs `labeled` throughout `methodology.tex`), **colour/color**, **normalised/normalized**, **behaviour/behavior**.

---

### D7 · $\Sigma_{\mathrm{SR}}$ vs host operator set — CONFIRMED (R2.3)

| Source | Set |
|---|---|
| `article/paper/computational_experiments.tex:63–67` (Section IV.2, inclusion criterion ii) | $\Sigma_{\mathrm{SR}} = \{+, \times, \mathrm{Neg}, \mathrm{Inv}, \sin, \cos, \exp, \log, \sqrt{\,}, \lvert\cdot\rvert, \mathrm{Pow}, \mathrm{Const}\}$ |
| `article/supplementary/supplementary.tex:557–559` (Appendix D.2, "Common configuration") | $\{+, -, \times, \div, \sin, \cos, \exp, \log\}$ |

These are two different objects serving two different roles, and the manuscript never says so:
1. $\Sigma_{\mathrm{SR}}$ = the **encoding alphabet** of the representation, used as the benchmark *inclusion criterion*.
2. The D.2 set = the **host solvers' search primitives** (how UDFS and Bingo were configured).

**A third statement on the same question** exists: `methodology.tex:965–967` says the experiments "exclude \textsc{Pow}" ($k_\wedge = 0$), so the isomorphism reduces to conditions (i)–(iii).

**Nguyen-8 and Nguyen-11** (`supplementary.tex:504, 507`): N-8 $=\sqrt{x}$ on $[0,4]$; N-11 $= x^y$ on $[0,1]^2$. Both ranges non-negative, so both are expressible from $\{\exp,\log,\times\}$. Both solve empirically to $R^2 = 1.0000$ under Bingo (`table_supplementary_bingo.tex:52, 44`) and UDFS (`table_supplementary_udfs.tex:52, 44`). The results stand; only the description is incomplete.

Note also `supplementary.tex:459–461` describes Nguyen as including "$x^y$ (\textsc{Pow})", and `supplementary.tex:747–748` gives the *synthetic* benchmark operator set as $\{+, \times, \wedge, \sin, \cos, \exp, \log, \mathrm{neg}, \mathrm{inv}\}$ — a fourth set, which does include $\wedge$.

---

### D8 · Alphabet label characters across documents — CONFIRMED (R2.2), error is in the preprint

| Source | $\mathcal{L}$ |
|---|---|
| Journal Def 3.2, `article/paper/methodology.tex:95` | $\{+, *, \texttt{g}, \texttt{i}, s, c, e, l, r, \hat{}\,, a, k\}$ |
| Journal Table 1, `methodology.tex:77–78` | `g` → Neg, `i` → Inv |
| Preprint Def 2.2, `article/arxiv/69b91250e7e60fc6079dfd5d/methodology.tex:97` | $\{+, *, \texttt{-}, \texttt{/}, s, \ldots\}$ |
| Preprint Table 1, same file `:79–80` | `g` → Neg, `i` → Inv |
| Preprint prose, same file `:125–126` | "\textsc{Neg} (label \texttt{g}) … \textsc{Inv} (label \texttt{i})" |

**The journal manuscript is self-consistent.** The **preprint** is internally inconsistent: its Definition 2.2 contradicts its own Table 1 and its own prose. R2 saw both documents in the submitted package and read the pair as two definitions of one alphabet.

Related figure: journal Def 3.2 (`methodology.tex:116–117`) states "7 single-character tokens and 24 compound tokens ($2\times|\mathcal{L}|$), totaling 31 tokens", consistent with $|\mathcal{L}| = 12$. Separately `supplementary.tex:914` uses $|\mathcal{A}| = 17$ for a Lev-1 neighbourhood count over a reduced alphabet — a different quantity; do not conflate.

---

### D9 · `normalize_const_creation` undefined in the paper — CONFIRMED (R1.3)

Sole occurrence in the rendered manuscript: `article/supplementary/supplementary.tex:398–399`, first line of the FastCanonical pseudocode —

> $D \leftarrow \mathtt{normalize\_const\_creation}(D)$ · *// redirect all \textsc{Const} creation edges to $x_1$*

Also present at `methodology.tex:830–831`, but inside a `\begin{comment}` block, so not rendered. No definition, no justification, no complexity note anywhere.

**Implementation** — `src/isalsr/core/labeled_dag.py:591`, docstring at `:592–608`:

> Return a new DAG with all CONST creation edges moved to x_1 (node 0).
> CONST nodes are evaluation-neutral leaves: they ignore in-edges and return `const_value` directly. But D2S requires every node to be reachable from a VAR via outgoing edges, so V/v creates a "creation edge" pointer → CONST. The choice of creation source is semantically irrelevant but produces different canonical strings.
> This normalization eliminates that redundancy by standardizing all CONST creation edges to come from node 0 (x_1). This is always valid because x_1 has no incoming edges (no cycle risk).
> The normalized DAG:
> - Computes the same function: eval(D) == eval(normalize(D))
> - Has deterministic CONST creation edges
> - Produces a unique canonical string for each equivalence class

Call sites: `src/isalsr/core/canonical.py:95, 146, 231` (guarded by `dag._has_const_nodes()`), `src/isalsr/core/labeled_dag.py:458`.

**Why it is more than a definition gap**: the step is a *precondition of the invariance claim*, not cosmetic preprocessing — without it, two isomorphic DAGs whose CONST nodes were created from different sources produce different canonical strings. Theorem 3.15 as stated does not mention it. `\textsc{Const}` (label `k`) is in $\mathcal{T}$ (Table 1, `methodology.tex:86`) and in $\Sigma_{\mathrm{SR}}$ (`computational_experiments.tex:64–66`), so CONST nodes are in scope for the suite.

---

### D10 · Reachability-condition failure rate never reported — CONFIRMED (R1.2)

The condition, `methodology.tex:976–977` (Theorem 3.13): "If every non-variable node of $D$ is reachable from some variable via directed paths…". Inherited by Lemma 3.14/A.2 and Theorem 3.15 (`methodology.tex:1029–1030, 1059–1060`), and relied on by Rule 1's non-exclusion argument (`methodology.tex:762–766`).

Nothing in the manuscript reports how often candidate DAGs violate it, or what the deduplication wrapper does when they do.

Nearest reported quantity, which is a **different** claim — `article/paper/discussion.tex:36–40`:
> no false collision has been observed across the $14{,}841$ DAGs in the unit-test suite or the millions generated during the SR experiments

Also unquantified: the 60 s canonicalisation timeout fallback (`discussion.tex:104–107`, "timed-out DAGs are counted as unique"). Timeout and reachability failure are two distinct fallback paths; only the first is documented at all.

---

### D11 · Bingo $S = 0.93$ described as "approximately neutral" — CONFIRMED (R1.1)

| | |
|---|---|
| Value | $S = 0.93$ for Bingo, $S = 1.07$ for UDFS — `article/paper/results.tex:57–58`, Table 2 |
| Objectionable phrasing | `article/paper/discussion.tex:66–68`: "the median overhead is $39\%$ and the wall-clock effect is **approximately neutral** after the search-time savings offset the canonicalisation cost" |

The **results** section is already properly conditional — `results.tex:180–185`:
> on the $30$ problems where Bingo exceeds $\rho = 1.85$ the search-only speedup recovers to $S \geq 0.95$, and on three problems---N-8, II.11.27, and Keijzer-11---it exceeds unity.

Per-problem corroboration, `table_supplementary_bingo.tex`: only **4 of 50** rows have $T_{\mathrm{IS}} < T_{\mathrm{BL}}$ (I.6.20a, II.11.27, Keij-11, N-8); the other 46 are slower under IsalSR. Highest overheads are the Nguyen rows: N-12 61.3%, Keij-6 60.3%, N-5 60.5%, N-4 59.6%, N-3 58.2%, N-7 56.8%.

The abstract's own wording (`main.tex:81`, "at a median of $39\%$ in the sub-millisecond evaluation regime") is accurate. The overstatement is localised to the discussion.

---

## Part 2 — Not raised by any reviewer

Found during the same source pass. Each is the same class of defect the reviewers already penalised, so each is a live round-2 risk.

### E1 · $k$-stratified Bingo overhead: main text ≠ appendix

| Source | $k<5$ | $k\in[5,15)$ | $k\in[15,32)$ |
|---|---|---|---|
| `article/paper/results.tex:177–179` | 38.5% | **45.9%** | **41.6%** |
| `article/supplementary/supplementary.tex:720–722` (Table 8) | 38.5% | **47.0%** | **49.9%** |

Two of three buckets disagree, and the shapes differ qualitatively: the appendix trend is monotone increasing (38.5 → 47.0 → 49.9), the main text's is non-monotone (38.5 → 45.9 → 41.6). The main text cites Table 8 explicitly as its source. R2 checked every other number in these tables; this pair was missed only by chance.

### E2 · "35.5–56.0% total overhead reported for Bingo in the main text"

`article/supplementary/supplementary.tex:734` asserts a range the main text does not contain. `article/paper/results.tex:176` reports a **39.2% median**; Table 2 (`results.tex:58`) reports `39.2%`. No 35.5–56.0% range appears anywhere in the main paper.

### E3 · "near-linear" vs near-$O(k^2)$

| Source | Claim |
|---|---|
| `article/paper/methodology.tex:885–887` | "$O(k)$ insertion steps, each evaluating $O(k)$ candidates, yielding $O(k^2)$ total time" |
| `article/paper/conclusion.tex:9–11` | "computes the canonical string in **near-linear time** on typical SR expressions" |
| `article/paper/discussion.tex:97` | "runs in **near-linear time** on typical SR expressions" |
| `article/paper/introduction.tex:51–52` | "computes the canonical string in **near-$O(k^2)$** time" ✓ |
| `article/paper/related_work.tex:81–82` | "running in **near-quadratic** time" ✓ |

Introduction and related work are correct; conclusion and discussion overstate. **R1's own B2 statement uses "near-O(k^2)"**, so this reviewer has already registered the correct complexity — a "near-linear" claim in the revision reads as inconsistent with what they wrote back.

Related: `discussion.tex:22–24` infers "near-linear behaviour" from median per-DAG times; `supplementary.tex:736, 793` fits $O(k^{0.7})$ *per permutation*, which is a per-call cost, not the whole-DAG canonicalisation cost. Three different growth claims coexist ($O(k^{0.7})$, near-linear, near-$O(k^2)$) without a stated relationship.

### E4 · Bingo cell count 1,465 vs 1,500

`article/paper/results.tex:68–70`. UDFS uses 1,500 = 50 × 30; Bingo uses 1,465, unexplained. See D1 and D4.

### E5 · `previously_published_statement` cites "Section S.I"

`previously_published_statement/main.tex:77` refers to proofs in "the supplementary material (Section~S.I)". The supplementary uses Appendix A–G lettering; there is no Section S.I. Also `:142–148` describes the supplementary as "approximately thirty pages"; the actual `supplementary.pdf` is **10 pages**.

### E6 · Dead `\begin{comment}` blocks in `methodology.tex`

Five large blocks of commented-out content — full duplicates of all three pseudocodes and two TikZ trace figures: `methodology.tex:135–143, 194–251, 269–417, 460–522, 530–660, 812–879, 986–1023, 1039–1054, 1073–1138`. Roughly 600 of 1,143 lines. They compile to nothing but are the reason the same pseudocode exists in two files, which is how the Table 3/Table 4 confusion (D3) arose.

### E7 · Website URL is the anonymised one

`article/paper/computational_experiments.tex:2–4` footnote points at `https://little-manifold.github.io/isalsr-anon/` — the double-blind anonymised mirror — in the **non-anonymous** `article/` version.

### E8 · Reduction-factor range in discussion vs per-problem tables

`article/paper/discussion.tex:10–11` states "The observed $\rho$ values, $1.45$–$1.96$ across the $50$-problem suite". The per-problem tables give a wider span: UDFS ρ runs from **1.11** (I.12.1, I.25.13, I.34.27) to **1.98** (R3); Bingo from **1.57** to **1.96**. Union = [1.11, 1.98], not [1.45, 1.96]. The supplementary states the ranges correctly per method: "[1.11, 1.98]" for UDFS (`supplementary.tex:599`) and "[1.57, 1.96]" for Bingo (`:620–621`).

### E9 · Acknowledgements name a specific LLM version

`article/paper/main.tex:126` credits "Claude Opus 4.7 (Anthropic) \cite{anthropic_claude_opus_4_7}" for benchmark-suite discovery, Picasso parallelisation code, and the companion website. Fine to keep, but it is a citable claim in a revision where the benchmark suite composition (D2) is itself under challenge.

---

## Aggregate view

| Class | Items |
|---|---|
| Reviewer-raised, confirmed | D1–D11 (11 of 11 — **every** factual claim the reviewers made checks out) |
| Found independently | E1–E9 |
| Concentrated in | `supplementary.tex` (D1, D2, D3, D7, D9, E1, E2), per-problem tables (D2, D4), `discussion.tex` (D11, E3, E8) |
| Root cause of D1, D2, E4 | Appendix D.1–D.3 was written for the 22-problem arXiv configuration and never updated when the suite grew to 50 problems |
