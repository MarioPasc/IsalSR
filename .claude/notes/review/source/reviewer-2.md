# Reviewer 2

**Overall rating: Fair.** Technically sound: **Partially**. Title/abstract/keywords: **No**.
The harshest of the three, but the harshness is almost entirely bookkeeping: 6 of 8 comments are internal inconsistencies between the main text and the appendices, all verified as real. Only R2.1 challenges the science. This reviewer read the appendices line by line and cross-checked them against the embedded preprint — assume they will do so again in round 2.

## Summary and Strengths (verbatim)

> **Summary**
>
> This is an Arxiv-Extension work that introduces a representation framework called ISALSR. It encodes expression DAGs as canonical instruction strings over a two-tier alphabet, collapsing the Θ(k!) redundant node-ordering representations of each expression into a single canonical form. The canonical string is computed by a greedy algorithm guided by 1-Weisfeiler–Leman subtree hashing and is proven to be a complete labeled-DAG invariant under a reachability condition. Experimental validation on UDFS and Bingo over a 50-problem suite, showing 34–45% evaluation elimination while preserving regression quality.
>
> **Strengths**
>
> 1. The ISALSR encoding maps expression DAGs to canonical strings that provably serve as a complete labeled-DAG isomorphism invariant under a well-defined reachability condition.
> 2. The method integrates as a drop-in component at the evaluation boundary, requiring no modification to the host search algorithm.

Note "Arxiv-Extension work" — this reviewer read `previously_published_statement/main.tex` and treats the delta from the preprint as the object under review. That framing drives R2.1 and R2.2.

---

## R2.1 — Lemma A.2 proof is incomplete

**Verbatim:**

> 1. The extension version upgrades Conjecture 2.10/2.11 (left unproven in the preprint) to Theorem 3.13/3.15, but the proof of the key Lemma A.2 is unexpectedly terse and does not formally establish that κ-minimal candidate selection always yields valid D2S strings. A complete proof should be provided.

**Type**: theory. The only scientific objection in this review, and the reason B3 is "Partially" rather than "Yes".

**Cross-walk**: Lemma A.2 = `lem:fcs_valid`, "FCS produces valid D2S strings". Stated in `article/paper/methodology.tex:1027–1033`; proved in `article/supplementary/supplementary.tex:111–133`.

**The proof in full** (`supplementary.tex:119–133`) — 14 lines, the entirety of it:

> The fast canonical algorithm (Table~4 of the main document) uses the same insertion primitives as D2S: $\texttt{V}\ell$/$\texttt{v}\ell$ tokens create nodes and edges, $\texttt{C}$/$\texttt{c}$ tokens create edges, and pointer movements navigate the CDLL identically.
> The only difference is the *candidate selection rule*: where D2S greedily picks the first valid candidate in spiral order, FCS selects among $\kappa$-minimal candidates (Rule~2) with optional backtracking over ties.
> At each branch point, FCS chooses one of the uninserted out-neighbors of the current pointer node---exactly the same candidate pool as D2S.
> The resulting string therefore belongs to $\mathcal{W}(D)$ (Definition~3.5 of the main document).
> By Theorem~\ref{thm:roundtrip}, $D \cong \mathrm{S2D}(\fcs_D, m)$.

**Why the reviewer calls it terse**: the argument is "same candidate pool ⇒ same string set", asserted rather than derived. It does not address:
- **Termination.** $\mathcal{W}(D)$ (Def 3.5, `methodology.tex:682–688`) is the set of strings *producible by the D2S procedure*. Membership requires that the FCS run terminates having placed every node and edge. Rule 1 *removes* candidates from the pool (a Pow node $c$ with non-empty $\sigma(c)$ and $\sigma(c)[0] \neq u$ is excluded), so the pools are **not** identical — contradicting the proof's own "exactly the same candidate pool as D2S".
- **Rule 1's non-exclusion argument lives elsewhere and is informal.** It is in `methodology.tex:762–766`, inside Definition 3.8, not in the lemma: "Rule~1 does not exclude any valid insertion ordering: by the reachability precondition ..., the base $\sigma(c)[0]$ of every \textsc{Pow} node $c$ is reachable from some variable, so D2S will have inserted $\sigma(c)[0]$ into the CDLL before $c$ becomes a candidate, and some displacement in $\mathcal{P}_n$ will place the acting pointer on $\sigma(c)[0]$, making $c$ eligible."
- **Reachability of the C/c edge phase.** The lemma's hypothesis is the Thm 3.13 reachability condition, but the proof never uses it explicitly.
- **Definition 3.5 has no κ.** $\mathcal{W}(D)$ is defined as free choice among uninserted out-neighbours at each branch point; showing a κ-minimal choice is one of those choices is exactly the step the reviewer says is missing.

**Dependency**: Theorem 3.15's (⇒) completeness direction rests entirely on Lemma A.2 (`supplementary.tex:202–210`). If A.2 is not established, the completeness half of the headline theorem is not established. This is the mechanism by which one terse lemma downgrades B3 to "Partially".

**Preprint history**: the reviewer's "Conjecture 2.10/2.11 left unproven in the preprint" is confirmed by `previously_published_statement/main.tex:71–79`, which advertises exactly this upgrade as contribution #2.

---

## R2.2 — Alphabet label characters differ between documents

**Verbatim:**

> 2. Definition 3.2 uses label characters {g, i} for NEG and INV, while Definition 2.2 in the embedded preprint uses {−, /}. These two definitions of the same alphabet Σ_SR coexist in the submitted PDF and should be reconciled.

**Type**: cross-document inconsistency. **Confirmed, and the error is in the preprint, not the journal manuscript.**

- Journal, `article/paper/methodology.tex:95`:
  `\mathcal{L}=\{\texttt{+},\texttt{*},\texttt{g},\texttt{i},\texttt{s},\texttt{c},\texttt{e},\texttt{l},\texttt{r},\texttt{\^{}},\texttt{a},\texttt{k}\}` — uses `g`, `i`. Consistent with journal Table 1 (`methodology.tex:77–78`), which lists `g`→Neg and `i`→Inv.
- arXiv preprint, `arxiv/69b91250e7e60fc6079dfd5d/methodology.tex:97`:
  `\mathcal{L}=\{\texttt{+},\texttt{*},\texttt{-},\texttt{/},\texttt{s},…\}` — uses `-`, `/`.
- But that same preprint's Table 1 (`arxiv/.../methodology.tex:79–80`) already lists `g`→Neg and `i`→Inv, and its prose at lines 125–126 says "\textsc{Neg} (label \texttt{g}) and \textsc{Inv} (label \texttt{i})".

So the preprint is **internally** inconsistent: its Def 2.2 contradicts its own Table 1 and its own prose. The journal manuscript is self-consistent throughout. Whoever answers this needs both facts.

**Knock-on**: alphabet size. Journal Def 3.2 (`methodology.tex:116–117`) says "$7$ single-character tokens and $24$ compound tokens ($2\times|\mathcal{L}|$), totaling $31$ tokens" — consistent with $|\mathcal{L}| = 12$. Separately, `supplementary.tex:914` uses $|\mathcal{A}| = 17$ for the Lev-1 neighbourhood count on a different (reduced) alphabet; check before quoting either number.

---

## R2.3 — Σ_SR includes Pow and √; host operator set excludes them

**Verbatim:**

> 3. Section 4.2 defines Σ_SR as including Pow and √, but Appendix D.2 specifies the host operator set as {+, −, ×, ÷, sin, cos, exp, log}, which excludes both. Benchmark problems such as Nguyen-8 (√x) and Nguyen-11 (xʸ) require them. Please clarify this discrepancy.

**Type**: inconsistency. **Confirmed.** Both statements exist as quoted.

- Section 4.2 = `sec:benchmarks`, `article/paper/computational_experiments.tex:63–67`:
  > $\Sigma_{\mathrm{SR}} = \{+, \times, \mathrm{Neg}, \mathrm{Inv}, \sin, \cos, \exp, \log, \sqrt{\,}, |\cdot|, \mathrm{Pow}, \mathrm{Const}\}$, excluding problems that require operators outside $\Sigma_{\mathrm{SR}}$ (e.g., $\tanh$, $\arctan$, $\mathrm{sgn}$)
- Appendix D.2 = `sec:supp_baseline`, "Common configuration", `article/supplementary/supplementary.tex:557–559`:
  > Both methods share the operator set $\{+, -, \times, \div, \sin, \cos, \exp, \log\}$

**Context an answering agent needs — these are two different sets serving two different roles:**
1. $\Sigma_{\mathrm{SR}}$ is the **encoding alphabet** of the representation (what IsalSR *can* express), used in `computational_experiments.tex` as the *inclusion criterion* for admitting a benchmark problem into the suite.
2. The operator set in D.2 is the **host solvers' search primitive set** (what UDFS and Bingo are configured to search over).

Nothing in the paper states this distinction. R2 read them as the same object, which is a reasonable reading of the text as written.

**Nguyen-8 and Nguyen-11 specifically** (`supplementary.tex:504, 507`): N-8 = $\sqrt{x}$ on $[0,4]$; N-11 = $x^y$ on $[0,1]^2$. Both are recoverable from $\{\exp, \log, \times\}$ ($\sqrt{x} = \exp(\tfrac12\log x)$, $x^y = \exp(y\log x)$) on positive domains, and both ranges are non-negative. Empirically both solve: Bingo reaches $R^2 = 1.0000$ on N-8 and N-11 (`table_supplementary_bingo.tex:52, 44`), UDFS likewise (`table_supplementary_udfs.tex:52, 44`). So the results are not wrong; the description is incomplete.

**Related text that will need to stay consistent**: `supplementary.tex:459–461` says of Nguyen "four involve two variables, including $x^y$ (\textsc{Pow})", and the Pow operand-order machinery (Rule 1, condition (iv) of Def 3.9) is a substantial part of the theory. `methodology.tex:965–967` states the experiments "exclude \textsc{Pow}" ($k_\wedge = 0$) — a third statement on the same question, in a third place.

---

## R2.4 — Broken table cross-reference

**Verbatim:**

> 4. Appendix A cites “Table 4 of the main document,” but no Table 4 exists in the main text; the referenced FCS pseudocode is in Table 3 of Appendix C.

**Type**: cross-reference error. **Confirmed.**

`article/supplementary/supplementary.tex:120`, inside the proof of Lemma A.2:
> The fast canonical algorithm (**Table~4 of the main document**) uses the same insertion primitives as D2S

The main text has exactly three tables: Table 1 = `tab:operations` (`methodology.tex:59`), Table 2 = `tab:three_axis` (`results.tex:32`), Table 3 = `tab:cpdt_summary` (`results.tex:89`). The FCS pseudocode is `tab:canon_pseudo` at `supplementary.tex:382`, which is Table 3 of the supplementary (Appendix C).

**Same class of defect elsewhere** — the supplementary refers to main-document objects by hardcoded number in several places, because the two documents compile separately:
- `supplementary.tex:131` "Definition~3.5 of the main document"
- `supplementary.tex:170` "Definition~3.7 of the main document"
- `supplementary.tex:209, 235` "Definition~3.9 of the main document"
- `supplementary.tex:239` "Theorem~3.15 of the main document"
- `supplementary.tex:453` "Section~5.1 of the main paper"
- `supplementary.tex:1005` "Table~1 of the main document"

And symmetrically the main text refers into the supplementary by hardcoded number:
- `methodology.tex:175, 190` "Table~1 of Appendix~C"
- `methodology.tex:422` "Table~2 of Appendix~C"
- `methodology.tex:680, 809` "Table~3 of Appendix~C"
- `results.tex:176` "Table~8 in the appendices"
- `discussion.tex:28, 50, 101` "Appendix~E.2", "Appendix~E.1"
- `computational_experiments.tex:52, 92` "Appendix~D.1"

Every one of these is a manual number that has to be re-verified after any structural edit. Note `methodology.tex:175/190` calls the S2D pseudocode "Table 1 of Appendix C" while `supplementary.tex:453` calls the experimental-configuration section "Section 5.1 of the main paper" — the main paper's Section 5 is Results, and experimental configuration is Section 4. That reference is also wrong.

---

## R2.5 — Feynman problem counts do not agree

**Verbatim:**

> 5. Section 4.2 refers to a “20-problem subset of AI Feynman,” but Table 5 (in the Appendix) lists only 10 equations, while Tables 6–7 (in the Appendix) contain 24. These counts should be consistent and Table 5 should list all problems used.

**Type**: inconsistency. **Confirmed, all three numbers.**

- "20-problem subset": `article/paper/computational_experiments.tex:79–81`
  > The core ($32$ problems) consists of the $12$ Nguyen benchmarks~\cite{uy2011} and a $20$-problem subset of AI~Feynman following~\cite{liu2025}.
- Table 5 = `tab:feynman`, `article/supplementary/supplementary.tex:513–536`: **10 rows** (I.6.20a, I.12.1, I.25.13, I.34.27, I.39.10, II.3.24, I.14.3, I.12.4, I.10.7, I.48.20). Caption and surrounding text at `:469–471` say "10 physics equations".
- Tables 6–7 = `tab:supplementary_udfs` / `tab:supplementary_bingo`: **24 Feynman rows** each. Counted: I.10.7, I.12.1, I.12.4, I.13.12, I.14.3, I.15.10, I.16.6, I.25.13, I.29.16, I.30.3, I.34.27, I.37.4, I.39.10, I.44.4, I.48.20, I.50.26, I.6.20a, II.11.27, II.11.28, II.11.3, II.3.24, III.10.19, III.14.14, III.17.37.

**Actual composition of the 50-problem suite**, derived from the per-problem tables (both tables have exactly 50 data rows, and the problem sets are identical between them):

| Family | Count | IDs |
|--------|-------|-----|
| Nguyen | 12 | N-1 … N-12 |
| AI Feynman | 24 | the 24 listed above |
| Vladislavleva | 3 | Vlad-2, Vlad-4, Vlad-7 |
| Livermore | 3 | Liv-4, Liv-14, Liv-19 |
| R (Koza rational) | 3 | R1, R2, R3 |
| Pagie | 2 | Pagie-1, Pagie-2 |
| Keijzer | 2 | Keij-6, Keij-11 |
| Korns | 1 | Korns-12 |
| **Total** | **50** | |

So the true split is **12 Nguyen + 24 Feynman + 14 other**, not "32 core (12+20) + 18 extension". The appendix benchmark tables (`tab:nguyen`, `tab:feynman`) were never updated past the 22-problem arXiv configuration; `tab:feynman` documents ranges and sampling for only 10 of the 24 Feynman problems actually run, and no appendix table documents the 14 non-Nguyen/non-Feynman problems at all.

**Consequence for reproducibility**: `computational_experiments.tex:52–54` and `:92–95` both promise that "The benchmark tables in Appendix~D.1 list every problem with its expression, input dimensionality, sampling protocol, and source citation." They do not. 28 of 50 problems have no documented expression, range, or sampling protocol anywhere in the submission. This is the most substantive item in R2's list after R2.1.

---

## R2.6 — Total run count is wrong

**Verbatim:**

> 6. Appendix D.2 reports “2 × 2 × (12 + 10) × 30 = 2,640 total runs,” but the paper uses a 50 problem suite, so the correct count is 6,000. The same error recurs in Appendix D.3. Please reconcile and confirm all 50 problems were run with 30 seeds.

**Type**: inconsistency. **Confirmed. The reviewer's arithmetic is right.**

- `article/supplementary/supplementary.tex:560–561`:
  > $30$ independent seeds per (method, benchmark, problem, variant) configuration, yielding $2 \times 2 \times (12 + 10) \times 30 = 2{,}640$ total runs.
- Recurs in Appendix D.3 (`sec:supp_infra`), `supplementary.tex:574`:
  > All $2{,}640$ runs execute on CPU-only nodes of the Picasso supercomputer

$2 \times 2 \times 50 \times 30 = 6{,}000$. The `(12 + 10)` factor is the 22-problem arXiv suite.

**But note the cell counts reported in the main text do not reach 6,000 either** — `article/paper/results.tex:68–70`:
> UDFS attains $\rho = 1.56 \pm 0.24$ across $1{,}500$ seed-problem cells … and Bingo $\rho = 1.83 \pm 0.09$ across $1{,}465$ cells

$50 \times 30 = 1{,}500$ per (method, variant). UDFS is complete; **Bingo is 35 cells short of 1,500** and the shortfall is never explained. This connects directly to R2.7 (the `nan` entries). Anyone reconciling the run count must reconcile 1,465 as well, or R2 will catch it again.

---

## R2.7 — Undiscussed NaN results

**Verbatim:**

> 7. In the Appendix, Table 7 reports “nan” for Vlad-2 and Korns-12 under the Bingo–ISALSR variant. These failures are not discussed, and it is unclear how NaN values were handled in the paired statistical tests.

**Type**: unexplained result + statistical-handling question. **Confirmed.**

`article/supplementary/table_supplementary_bingo.tex`:
- Line 38, **Korns-12**: BL $R^2 = \underline{0.0000}$, IS $R^2 = \mathbf{nan}$; BL NRMSE $= \underline{1.0131}$, IS NRMSE $= \mathbf{nan}$; $d = +0.00\,[+0.00,+0.00]$; $\rho = 1.82 \pm 0.00$.
- Line 60, **Vlad-2**: BL $R^2 = \underline{0.9385}$, IS $R^2 = \mathbf{nan}$; BL NRMSE $= \underline{0.1966}$, IS NRMSE $= \mathbf{nan}$; $d = +0.00\,[+0.00,+0.44]$; $\rho = 1.83 \pm 0.00$.

**Two separate problems in one comment:**
1. The failures themselves are undocumented — no cause given anywhere in the manuscript.
2. **The `nan` is typeset as the *better* value.** The table's own caption defines "\textbf{Bold}: better of BL/IS. \underline{Underline}: worse of BL/IS." In both rows `nan` is bold and the finite baseline value is underlined. Vlad-2 in particular marks a real $R^2 = 0.9385$ as *worse than* `nan`. The bold/underline assignment is produced by whatever comparison the table generator applies, and it evidently treats NaN as winning. R2 did not say this explicitly — they will if it survives.

**Statistical handling**: the paired test (`computational_experiments.tex:170–180`) takes per-problem means $\bar{m}_i$ over $S = 30$ seeds and forms $\delta_i = \bar m_i^{IS} - \bar m_i^{BL}$ across $N = 50$ problems. If a problem's IS mean is NaN, $\delta_i$ is undefined and $N < 50$ for that metric. The manuscript states $N = 50$ throughout — `computational_experiments.tex:160` ("$N = 50$"), `:231` ("For $N = 50$ we evaluate $W_m^{+}$ via the continuity-corrected normal approximation"), and all of Table 3 (`tab:cpdt_summary`). The Wilcoxon description at `:229–230` mentions only that "zero-valued differences are excluded"; NaN exclusion is not described.

**Note also**: Korns-12 and Keijzer-6 are singled out in `article/paper/discussion.tex:116–118` as "problems whose primary difficulty is constant discovery" where "IsalSR therefore neither helps nor hurts the search" — the discussion touches Korns-12 without mentioning that its Bingo–IsalSR result is NaN.

---

## R2.8 — Abstract duplication and inconsistent naming

**Verbatim:**

> 8. The Abstract section contains a duplicated phrase: “…on both methods on both methods.” This should be corrected. Besides, the manuscript alternates between “ISALSR” and “IsalSR”, and between “canonicalisation” and “canonicalization.” These should be unified.

**Type**: copy-editing. **All three confirmed.** Also raised independently by R3.2 (the abstract phrase). Drives R2's C1 answer of **No** on title/abstract/keywords.

**(a) Duplicated phrase** — `article/paper/main.tex:81`:
> A paired test across problems on the empirical  reduction factor returns Cohen's $d > 2$ at $p < 10^{-21}$ **on both methods on both methods**: canonicalisation eliminates …

There is also a double space in "empirical  reduction factor" on the same line.

**(b) ISALSR vs IsalSR**: the macro `\IsalSR` is defined as `\textsc{IsalSR}` (`main.tex:51`), which renders as small-caps "ISALSR". Plain-text "IsalSR" (not via the macro) appears in `related_work.tex:23, 50, 76, 111` and `introduction.tex` prose. So the *rendered* PDF alternates between small-caps and mixed-case forms of the same name. The fix is at the usage level, not the macro level.

**(c) canonicalisation vs canonicalization**: both spellings are in active use, sometimes within one file.
- `-isation`: `computational_experiments.tex:33, 40`; `results.tex:6, 168, 193`; `discussion.tex:18, 68`; `conclusion.tex:9`; `main.tex:81`.
- `-ization`: `supplementary.tex:485, 563, 567, 636, 645, 650, 675, 695, 710, 713, 741, 752, 768, 792, 800`; `methodology.tex:679` ("canonicalization").
Broadly the main paper uses British `-isation` and the supplementary uses American `-ization`, but neither is clean. Related pairs to sweep at the same time: neighbourhood/neighborhood, labelled/labeled, colour/color, normalised/normalized, behaviour/behavior — the manuscript mixes these too (e.g. `labeled` throughout methodology vs `labelled` at `supplementary.tex:782`).

---

## Structured answers (verbatim)

- **A1. Category**: Research/Technology
- **A2. Relevance**: Relevant
- **B1. Significance**: Good
- **B3. Technically sound**: **Partially**
- **B4. Experimental validation**: Lacking in some respects; some cases of interest not tested
- **C1. Title/abstract/keywords appropriate**: **No**
- **C2. References**: References are sufficient and appropriate — suggested references: **NA.**
- **C3. Introduction**: Yes
- **C4. Organization**: **Could be improved**
- **C5. Readability**: Readable - but requires some effort to understand
- **C6. Length**: **Should be trimmed a bit**
- **C7. Supplemental material**: **Yes, as part of the main paper if accepted (cannot exceed the strict page limit)**
- **C8. Supplemental accept**: After revisions. Please include explanation under Public Comments below.
- **Overall**: **Fair**

C7 conflicts with R1 and R3, who both want digital-library supplementary. Main paper is already 12/12 pages and supplementary is 10 — merging is not possible under the limit, and R2 simultaneously asks (C6) for the paper to be trimmed.

### B2 — significance statement (verbatim, complete)

> ISALSR addresses a previously unexplored source of redundancy in symbolic regression: the Θ(k!) node-ordering permutations of expression DAGs. ISALSR targets structural isomorphism—a complementary and orthogonal source of redundancy. The canonical string is proven to be a complete labeled-DAG invariant under a reachability condition, and the drop-in design allows integration with any SR solver without modifying its search logic. Empirical validation on 50 problems across eight benchmark suites with 30 seeds and paired statistical testing demonstrates 34–45% evaluation elimination while preserving regression quality.

Despite the "Fair" overall and "Partially" on soundness, the significance statement is positive and accepts the contribution. The gap between B2 and the overall rating is accounted for by R2.1 plus the seven bookkeeping errors — i.e. the rating is recoverable.
