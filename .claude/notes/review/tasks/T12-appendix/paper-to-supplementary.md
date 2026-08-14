# Paper → supplementary: staged moves, and what was deleted instead

Produced by the T12 second length pass, 2026-08-14. The paper went **18 → 17 pages**
(clean `final` build) and **14,105 → 13,622** prose words.

`article/supplementary/` was **locked** throughout — the parallel session held it with 14 open
`\pendingnum`/`\pendingblock` placeholders and two Picasso jobs landing. Nothing under
`article/supplementary/**` or `double_blind/supplementary/**` was written. Every candidate was
therefore classified into one of two buckets:

- **(a) deleted from the paper** — because it is implementation trivia that needs no home, or
  because a supplementary appendix that *already exists and is already cited by the paper*
  carries it;
- **(b) staged** — left in the paper untouched, recorded here, because deleting it before its
  destination exists would lose content or leave a dangling pointer.

**14 items were deleted. 2 are staged.**

---

## The finding that made the page

`article/supplementary/supplementary.tex` already contains a full appendix on the naive-hash
comparator — `\section{The naive hash comparator}` at line 1077, which is **Appendix E**, the
appendix the paper already cites twice for per-problem $\phi$. Its subsections duplicate the
second half of the paper's §3.6 almost item for item:

| Paper §3.6 | Supplementary Appendix E | Verdict |
|---|---|---|
| Def 3.19, serialization key | E.1, `:1093–1116`, plus the host-specific record format | paper keeps the definition |
| Lemma 3.20 statement | Lemma `lem:hash_sound`, `:1129–1133` | paper keeps the statement |
| **Lemma 3.20 proof, inline** | **proof at `:1135–1144`, fuller** | **deleted → pointer** |
| $\pi_D$ hypothesis paragraph | `:1118–1124`, near-verbatim | paper keeps one shortened sentence |
| Eq (12), (13) for $\Delta$, $\phi$ | `:1185–1196`, identical | paper keeps the equations |
| "At $\phi=0$ … at $\phi=1$ …" | `:1197–1199`, verbatim | paper keeps (two lines, needed to read Table 4) |
| **Orbit-stabilizer bound on $\Delta$** | **`:1200–1207`, fuller** | **deleted → pointer** |
| **Per-candidate cost of the two keys** | **`:1245–1252`** | **deleted → pointer** |
| — | E.2 incompleteness counterexample `:1153–1173` | paper never had it |

Lemma 3.20's proof was **the only proof still typeset inline in the paper**; every other one
reads `\begin{proof}Please see Appendix~A.\end{proof}`. Replacing it with a pointer removes an
inconsistency and ~90 words, and the target already exists, so no staging was needed.

---

## (a) Deleted — implementation-level, no home required

| # | Text | Was at | Why it goes |
|---|---|---|---|
| 1 | "In the implementation, $\mathrm{hash}$ is Python's built-in hash on tuples of integers, which is deterministic within a process and produces 64-bit values." | `methodology.tex` §3.4 | Names the language and the standard-library function. Replaced by the specification the paper actually needs: "a deterministic $64$-bit hash on tuples of integers". **The 64-bit width was kept deliberately** — the very next sentence bounds collisions and that bound rests on the width. "Deterministic within a process" was also inviting a reproducibility question the paper has no reason to open. |
| 2 | "using in-place state mutation with explicit undo on backtrack" | `methodology.tex`, after the FCS definition | Data-structure choice, not method. Already stated in the Appendix C pseudocode (`supplementary.tex:563`, "\textsc{FCSstep} … uses in-place mutation of"). |
| 3 | "($\approx 200$ lines of Python for Bingo, $\approx 100$ for UDFS)" | `discussion.tex` §6.2 | Source line counts. Not a measurement of anything a reader can use, and not a claim a TPAMI methods section should make. |
| 4 | "so a configured operator without an image aborts the experiment instead of being discovered mid-search" | `computational_experiments.tex` §4.1 | Engineering practice. The clause that survives — "Containment is verified before each run, and Appendix D.2 reports the number of candidates refused at conversion" — keeps both the guarantee and the evidence. |

## (a) Deleted — already carried by an existing, already-cited appendix

| # | Text | Was at | Now points to |
|---|---|---|---|
| 5 | Lemma 3.20's 90-word proof body | `methodology.tex` §3.6 | "Please see Appendix~E, which also gives the smallest counterexample to the converse." The counterexample is a genuine addition the paper never had, so the pointer buys the reader something. |
| 6 | The orbit-stabilizer paragraph **and** the key-cost sentence, ~110 words | `methodology.tex` §3.6 | "The orbit-stabilizer bound on $\Delta$, the per-candidate cost of each key and the per-problem estimates are in Appendix~E." |
| 7 | "Written this way $\phi$ depends on the two streams only through their reduction factors, which allows it to be estimated from two runs …" | `methodology.tex` §3.6 | Duplicated in §4.1, which is where the estimation choice is actually made. |
| 8 | The $17.6\%$ instrumentation-cost argument, ~60 words | `computational_experiments.tex` §4.1 | Appendix E `:1219–1234` carries it in full, including "we measured that option and did not adopt it". **The confound itself is still stated in the body** — only our defence for not removing it moved. |
| 9 | "The benchmark tables in Appendix~D.1 list every problem with its expression, input dimensionality, sampling protocol, and source citation." | `computational_experiments.tex` §4.2 | The same sentence appeared again 34 words later in the same subsection, and "Appendix D.1" was named four times in it. One statement kept, merged with the training-set-size clause. |
| 10 | "which is scale-free and so does not require the two arms to emit streams of the same length" | `results.tex`, Table 4 caption | Third telling, after §3.6 and §4.1. |

## (a) Deleted — framing that favours the method without a measurement

| # | Before | After | Why |
|---|---|---|---|
| 11 | UDFS "represents a best-case scenario for low redundancy, since the enumeration tree is fixed" | "its enumeration order is fixed" | The fact is the same; "best-case scenario" is our label on it. Let the measured $\rho$ do the work. |
| 12 | Bingo's "stochastic search generates more structural duplicates, making it a realistic stress test for the deduplication mechanism" | "its stochastic operators regenerate previously visited structures" | "Realistic stress test for the deduplication mechanism" reads as a claim about how demanding our own evaluation is. It also sits awkwardly against the paper's own $\phi = 0.047$ on Bingo, which says Bingo is precisely where the *cheap* key does nearly as well. |
| 13 | "\IsalSR{} requires no modification to the underlying SR method's search operators, fitness function, or selection mechanism" | deleted | Fourth statement of "no modification required", after the abstract, §4.1 and the conclusion. Three survive. |
| 14 | "On methods that produce larger expressions, the redundancy that canonicalization eliminates would grow substantially" | deleted | An unmeasured extrapolation in our favour, and redundant: two paragraphs later the same section makes the point with numbers ($10! \to 20!$) and immediately concedes that observed $\rho$ will not approach $k!$. |

---

## (b) Staged — left in the paper, needs the supplementary unlocked

### B1. §4.4, "Statistical Analysis: Paired Test across Problems"

- **Where:** `article/paper/computational_experiments.tex`, `\subsection{Statistical Analysis: Paired Test across Problems}` through the end of the file (`\label{sec:cpdt}` to `\label{sec:cross_method}` inclusive).
- **Size:** ~1,100 words and five numbered equations — **the largest single block left in the paper, roughly one full column.**
- **Target:** a new supplementary appendix, "Statistical protocol", placed after Appendix D
  (Experimental Configuration) and before Appendix E. It must be a *new* section, so the
  supplementary's own appendix lettering shifts from E onwards — see "what breaks" below.
- **Why there:** it is a textbook-standard Demšar protocol. Notation, the sign constant $c_m$,
  the Hodges–Lehmann characterisation of the nonparametric location parameter, the
  Shapiro–Wilk branch, the $t$ and Wilcoxon statistics, the continuity-corrected normal
  approximation, and the bootstrap CI are all reproducible from `\cite{demsar2006}` plus a
  statement of the branch rule. No reviewer questioned any of it.
- **Replacement pointer the paper would need** (~8 sentences, so the net saving is ~900 words):
  the portfolio-level framing; the Demšar citation; that each problem contributes one paired
  observation; that the branch is selected by Shapiro–Wilk; that the effect size is Cohen's $d$
  with a bootstrap CI; that the two hosts are ranked separately with the $0.40 \to 0.90$
  figure; and a pointer.
- **What breaks if the move never happens:** nothing. The paper is correct as it stands, one
  column longer.
- **Why I did not do it:** it is a content decision about the paper's inferential backbone, not
  an editorial one. R2 rated soundness "Partially" and the project's own standing rule makes
  this test the primary significance metric for $R^2$ and the reduction factor. Thinning it to
  save a column is the authors' call. **This is the recommendation to take to the authors if
  16 pages is wanted.**

### B2. The condition-(iv) equivalence argument

- **Where:** `article/paper/methodology.tex`, the `\added{}` block inside
  `\begin{remark}[Necessity of condition (iv)]` — the passage beginning "On expression DAGs,
  matching the whole list is equivalent to matching the base alone" and ending "Such graphs
  denote no function, so nothing is lost by excluding them."
- **Size:** ~200 words plus one displayed equation.
- **Target:** Appendix A, beside the proof of Theorem 3.15, which already uses condition (iv) at
  `supplementary.tex:307`.
- **Why there:** it is a proof, and every other proof in the paper is in Appendix A. It is T18's
  material and it justifies why `is_isomorphic` compares position 0 only.
- **Replacement pointer:** one sentence — that on expression DAGs matching the whole ordered
  list is equivalent to matching the base alone, with the argument in Appendix A.
- **What breaks if the move never happens:** nothing; but note this argument is **single-sourced
  in the paper today**. The supplementary uses condition (iv) without proving the equivalence,
  so it must be *copied* before it is deleted, not deleted and rewritten.

---

## Considered and declined — read this as carefully as the cut list

| Candidate | Size | Why it stays |
|---|---|---|
| The κ-invariance paragraph in §3.4 | ~180 w | Overlaps supplementary Appendix B's remark [Greedy selection correctness] on the *first* of its two mechanisms only. It is the intuition behind Theorem 3.15, which is exactly what R2.1 attacked. Cutting the intuition for a theorem whose proof a reviewer already called terse is the wrong economy. |
| §4.3 "Runs undefined on part of an evaluation set" | ~130 w | Exists nowhere else in the package. It is the policy R2.7 asked about. |
| §4.3 "Pairing" | ~120 w | Overlaps the supplementary D.3 ledger only on the phrase "all 12,600"; the two answer different questions (D.3 answers R2.6 on run counts, this answers R2.7 on how a missing cell is handled). The supplementary session also flagged the D.2/D.3 ledger sentences as enumerating exactly the list R2.6 asked to have confirmed. |
| The §4.4 pooling justification vs the Fig. 1 caption | ~90 w across both | A genuine duplication, and **frozen by the number-verifier**: `0.40` is anchored to `computational_experiments.tex` on the phrase "critical difference from" and to `results.tex` on "Nemenyi threshold from", `0.90` likewise. Both were asserted deliberately in two places. Removing either deletes a check rather than re-points one, which the brief forbids. Reported rather than resolved. |
| Merging Table 4 (`tab:phi_by_host`) and Table 5 (`tab:key_cost`) | ~0.25 column | They are adjacent single-column tables on the same page about the same comparator, so merging is tempting. It renumbers Table 5, which breaks `\ref{tab:key_cost}`, a verifier anchor, and the response letter's cross-references. A presentation decision for the authors. |
| Deleting or shrinking a figure | up to ~0.5 page | Out of scope by instruction, and correctly so. `fig:cd` and `fig:reduction_factor_distribution` are both `figure*` and both land on page 13; together they are the most expensive page in the paper. Flagged, not touched. |
| The three `\begin{remark}` blocks in §3.5 | ~430 w | Variable anchoring explains why $|V|! \to k!$; necessity of (iv) is R2.1-adjacent; the automorphism bound carries $(k-k_\wedge)!$ and the UDFS/Bingo split. All load-bearing. |
| The `\begin{comment}` blocks in `methodology.tex` | ~700 lines | Zero page cost. E6/T13's. |
| The abstract | ~250 w | Every sentence is a result. Already restructured in the first pass. |
| The four inclusion criteria, suite composition, ODE-Strogatz caveat | ~450 w | R2.5 and R3.1 asked for exactly this and the submitted version was found wanting on it. |
| The honest-negative material — $S = 0.72$, $1.45\times$, 13 of 70, $\phi = 0.047$, Korns-12 | — | Off limits by instruction, and it should be. It is also the material most likely to earn back R2's "Partially". |

---

## Handover

When the supplementary is released:

1. Copy B2 into Appendix A before deleting it from the paper — it is single-sourced.
2. Decide B1 with the authors. It is worth roughly a page and it is the only remaining lever of
   that size that does not touch a figure.
3. Re-run both gates over the whole package afterwards:
   `python -m experiments.scripts.review_campaign.verify` and
   `scripts/check_manuscript_style.sh` on each tree.
