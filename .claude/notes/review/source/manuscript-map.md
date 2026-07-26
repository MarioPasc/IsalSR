# Manuscript map and numbering cross-walk

Everything a reviewer reference points to, resolved to a file and label.
All numbering below was derived from the source order of numbered environments and verified against the reviewers' own citations (R2 cites Def 3.2, Thm 3.13/3.15, Lemma A.2, Tables 3/5/6/7 — all resolve correctly under this map, so the map is confirmed against the compiled PDFs the reviewers read).

## On-disk layout

```
journal/69c1637a28a81fea2badda9a/
├── article/
│   ├── paper/          main.tex, introduction.tex, related_work.tex, methodology.tex,
│   │                   computational_experiments.tex, results.tex, discussion.tex,
│   │                   conclusion.tex, references.bib
│   │                   figures: fig_s2d.pdf, fig_d2s.pdf, cd_2d_r2_rf.pdf,
│   │                            reduction_factor_distribution.pdf
│   │                   photos: EzequielLopez.pdf, MarioPascual.jpg, KarlThurnhofer.pdf
│   └── supplementary/  supplementary.tex, table_supplementary_udfs.tex,
│                       table_supplementary_bingo.tex, references.bib
│                       figures: rf_vs_overhead.pdf, fig_synthetic_scalability.pdf,
│                                fig_shortest_path.pdf, fig_neighbourhood.pdf
├── double_blind/       main_anonymous.tex + supplementary_anonymous.tex
│                       (content verified identical to article/ versions)
├── cover_letter/
├── previously_published_statement/
├── reviews/            ← response letter skeleton
└── title_page.docx
```

Note: `article/CLAUDE.md` still describes a `mario/` subdirectory layout. That layout no longer exists on disk. Author ownership is unchanged (see README).

Sibling: the arXiv preprint lives at `article/arxiv/69b91250e7e60fc6079dfd5d/` (`arxiv_main.tex`, `methodology.tex`, …). Referenced by R2.2.

## Main paper — sections

| # | Title | File |
|---|-------|------|
| I | Introduction | `introduction.tex` |
| II | Related Work | `related_work.tex` |
| II.1 | Genetic Programming for Symbolic Regression | `related_work.tex:2` |
| II.2 | DAG Representations and Structural Redundancy | `related_work.tex:27` |
| II.3 | Graph Isomorphism and Canonical Forms | `related_work.tex:58` |
| **II.4** | **Benchmarking in Symbolic Regression** | `related_work.tex:84` ← **R3's "section 2.4"** |
| II.5 | SR Methods Beyond GP | `related_work.tex:96` |
| III | Methodology | `methodology.tex` |
| III.1 | The IsalSR Instruction Set | `methodology.tex:7` |
| III.2 | String-to-DAG Algorithm (S2D) | `methodology.tex:171` |
| III.3 | DAG-to-String Algorithm (D2S) | `methodology.tex:419` |
| III.4 | Fast Canonical String | `methodology.tex:662` |
| III.5 | Canonical String as a Graph Invariant | `methodology.tex:898` |
| IV | Computational Experiments | `computational_experiments.tex` |
| IV.1 | Baseline Methods and IsalSR Integration | `computational_experiments.tex:15` |
| **IV.2** | **Benchmark Problem Suite** | `computational_experiments.tex:45` ← **R2's "Section 4.2"** |
| IV.3 | Evaluation Framework | `computational_experiments.tex:97` |
| IV.4 | Statistical Analysis: Paired Test across Problems | `computational_experiments.tex:144` |
| V | Results | `results.tex` |
| V.1 | Search Space Reduction | `results.tex:63` |
| V.2 | Regression Quality | `results.tex:117` |
| V.3 | Computational Cost | `results.tex:164` |
| VI | Discussion | `discussion.tex` |
| VI.1 | Theoretical Implications | `discussion.tex:2` |
| VI.2 | Practical Impact | `discussion.tex:54` |
| VI.3 | Limitations | `discussion.tex:94` |
| VII | Conclusion | `conclusion.tex` |

## Main paper — numbered environments

`main.tex:30–42` declares one shared counter per section: `theorem`, `lemma`, `proposition`, `corollary`, `conjecture`, `definition`, `example`, `remark` all increment the same counter, numbered `[section]`. Hence the continuous 3.1 … 3.15 run below.

| Number | Kind | Label | Line |
|--------|------|-------|------|
| 3.1 | Definition — Labeled DAG | `def:ldag` | `methodology.tex:14` |
| **3.2** | **Definition — IsalSR Instruction Set $\Sigma_{\mathrm{SR}}$** | `def:alphabet` | `methodology.tex:93` ← **R2.2** |
| 3.3 | Definition — S2D Execution | `def:s2d` | `methodology.tex:184` |
| 3.4 | Definition — Spiral Displacement Set | `def:pairs` | `methodology.tex:433` |
| **3.5** | **Definition — Valid String Set $\mathcal{W}(D)$** | `def:valid_set` | `methodology.tex:682` ← cited from Lemma A.2 |
| 3.6 | Definition — Exhaustive Canonical String $w^*_D$ | `def:canonical` | `methodology.tex:690` |
| **3.7** | **Definition — 1-WL Subtree Hash** | `def:wl_hash` | `methodology.tex:707` |
| **3.8** | **Definition — Fast Canonical String $\hat w_D$** (Rules 1 & 2) | `def:fast_canonical` | `methodology.tex:745` |
| **3.9** | **Definition — Labeled-DAG Isomorphism** (conditions i–iv) | `def:isomorphism` | `methodology.tex:906` |
| 3.10 | Remark — Variable anchoring as a domain-specific simplification | (unlabeled) | `methodology.tex:933` |
| 3.11 | Remark — Necessity of condition (iv) | (unlabeled) | `methodology.tex:945` |
| 3.12 | Remark — Tightened automorphism bound | (unlabeled) | `methodology.tex:960` |
| **3.13** | **Theorem — Round-Trip Fidelity** (states the reachability condition) | `thm:roundtrip` | `methodology.tex:972` ← **R1.2** |
| **3.14** | **Lemma — FCS produces valid D2S strings** | `lem:fcs_valid` | `methodology.tex:1027` ← = **Lemma A.2**, R2.1 |
| **3.15** | **Theorem — FCS is a Complete Labeled-DAG Invariant** | `thm:invariant` | `methodology.tex:1057` ← **R1.2** |

All three proofs in the main text are stubs reading "Please see Appendix~A"; the real proofs are in the supplementary. The main text also carries full copies of all three pseudocodes and two trace figures inside `\begin{comment}` blocks (`methodology.tex:194–251, 269–417, 460–522, 530–660, 812–879`) — dead content that still occupies the file.

## Main paper — tables and figures

| # | Object | Label | Line |
|---|--------|-------|------|
| Table 1 | IsalSR operation types, labels, arities, semantics | `tab:operations` | `methodology.tex:59` |
| Table 2 | Three-axis summary (ρ, R², cost) | `tab:three_axis` | `results.tex:32` |
| Table 3 | Paired test across problems | `tab:cpdt_summary` | `results.tex:89` |
| Fig. 1 | S2D execution trace | `fig:s2d` | `methodology.tex:253` |
| Fig. 2 | D2S encoding | `fig:d2s` | `methodology.tex:451` |
| Fig. 3 | Critical-difference diagram | `fig:cd` | `results.tex:2` |
| Fig. 4 | Reduction-factor distribution | `fig:reduction_factor_distribution` | `results.tex:19` |

**There is no Table 4 in the main paper.** This is the basis of R2.4.

## Supplementary — appendices

`supplementary.tex:54` opens `\appendices`; sections letter as A…G.

| Letter | Title | Label | Line |
|--------|-------|-------|------|
| **A** | Proofs | `sec:supp_proofs` | `:56` |
| **B** | Remarks on canonicalisation | `sec:supp_remarks` | `:215` |
| **C** | Algorithm pseudocodes | `sec:supp_pseudocodes` | `:258` |
| **D** | Experimental Configuration | `sec:supp_config` | `:447` |
| D.1 | Benchmark Suites | `sec:supp_benchmarks` | `:455` |
| **D.2** | **Baseline Method Configuration** | `sec:supp_baseline` | `:538` ← **R2.3, R2.6** |
| **D.3** | **Computational Infrastructure** | `sec:supp_infra` | `:571` ← **R2.6** |
| D.4 | Per-Problem Results: UDFS | `sec:supp_udfs` | `:585` |
| D.5 | Per-Problem Results: Bingo | `sec:supp_bingo` | `:605` |
| **E** | Scalability across $k$ internal nodes | `sec:supp_scalability` | `:626` |
| E.1 | Empirical per-$k$ overhead and reachable reduction | `sec:supp_scalability_empirical` | `:640` |
| E.2 | Synthetic invariance and timing | `sec:supp_scalability_synthetic` | `:738` |
| **F** | Metric-space properties of canonical strings | `sec:supp_metric` | `:830` |
| F.1 | Shortest-path distances | `sec:supp_metric_sp` | `:842` |
| F.2 | Distance-1 neighbourhood | `sec:supp_metric_nbh` | `:906` |
| **G** | Discussion | `sec:supp_discussion` | `:989` |
| G.1 | Relationship to IsalGraph | `sec:disc_isalgraph` | `:992` |

## Supplementary — numbered environments

Same shared-counter scheme (`supplementary.tex:21–27`), numbered by appendix letter.

| Number | Kind | Label | Line |
|--------|------|-------|------|
| A.1 | Theorem — Round-Trip Fidelity (proof) | `thm:roundtrip` | `:62` |
| **A.2** | **Lemma — FCS produces valid D2S strings (proof)** | `lem:fcs_valid` | `:111` ← **R2.1** |
| A.3 | Theorem — Complete Labeled-DAG Invariant (proof) | `thm:invariant` | `:135` |
| B.1 | Remark — Greedy selection correctness | (unlabeled) | `:218` |
| B.2 | Remark — Injectivity and search-space reduction | (unlabeled) | `:238` |

A.1/A.2/A.3 restate 3.13/3.14/3.15 verbatim, so each theorem exists twice under two numbers. That duplication is why R2 has to say "Theorem 3.13/3.15" and "Lemma A.2" in the same sentence.

## Supplementary — tables and figures

| # | Object | Label | Line |
|---|--------|-------|------|
| Table 1 | S2D pseudocode | `tab:s2d_pseudo` | `:263` |
| Table 2 | D2S pseudocode | `tab:d2s_pseudo` | `:320` |
| **Table 3** | **Fast Canonical String pseudocode** | `tab:canon_pseudo` | `:382` ← **R1.3, R2.4** |
| Table 4 | Nguyen benchmarks (12 rows) | `tab:nguyen` | `:487` |
| **Table 5** | **AI Feynman subset (10 rows)** | `tab:feynman` | `:513` ← **R2.5** |
| **Table 6** | **Per-problem UDFS (50 rows)** | `tab:supplementary_udfs` | `table_supplementary_udfs.tex` |
| **Table 7** | **Per-problem Bingo (50 rows)** | `tab:supplementary_bingo` | `table_supplementary_bingo.tex` ← **R2.7** |
| Table 8 | Bingo overhead stratified by $k$ | `tab:k_range_overhead` | `:707` |
| Table 9 | Synthetic scalability | `tab:synthetic_scalability` | `:807` |
| Table 10 | Levenshtein shortest paths | `tab:supp_shortest_path` | `:863` |
| Table 11 | Distance-1 neighbourhood | `tab:supp_neighbourhood` | `:928` |
| Fig. 1 | Reduction factor vs overhead by $k$ | `fig:empirical_scalability` | `:656` |
| Fig. 2 | Synthetic scalability | `fig:synthetic_scalability` | `:759` |
| Fig. 3 | Shortest Levenshtein path | `fig:shortest_path` | `:890` |
| Fig. 4 | Neighbourhood structure | `fig:neighbourhood` | `:968` |

## Hardcoded cross-document references

The two documents compile separately, so every cross-document reference is a **manually typed number**, not a `\ref`. Any renumbering silently breaks them. Complete inventory:

**Supplementary → main:**

| Line | Text | Resolves to | OK? |
|------|------|-------------|-----|
| `:120` | "Table 4 of the main document" | *nothing* | **broken (R2.4)** |
| `:131` | "Definition 3.5 of the main document" | `def:valid_set` | ok |
| `:170` | "Definition 3.7 of the main document" | `def:wl_hash` | ok |
| `:209` | "Definition 3.9 of the main document" | `def:isomorphism` | ok |
| `:235` | "Definition 3.9 of the main document" | `def:isomorphism` | ok |
| `:239` | "Theorem 3.15 of the main document" | `thm:invariant` | ok |
| `:453` | "Section 5.1 of the main paper" | Section V.1 = *Search Space Reduction* | **wrong — intended Section IV (Computational Experiments)** |
| `:1005` | "Table 1 of the main document" | `tab:operations` | ok |

**Main → supplementary:**

| Line | Text | Resolves to | OK? |
|------|------|-------------|-----|
| `methodology.tex:175` | "Table 1 of Appendix C" | `tab:s2d_pseudo` | ok |
| `methodology.tex:190` | "Table 1 of Appendix C" | `tab:s2d_pseudo` | ok |
| `methodology.tex:422` | "Table 2 of Appendix C" | `tab:d2s_pseudo` | ok |
| `methodology.tex:680` | "Table 3 of Appendix C" | `tab:canon_pseudo` | ok |
| `methodology.tex:809` | "Table 3 of Appendix C" | `tab:canon_pseudo` | ok |
| `results.tex:176` | "Table 8 in the appendices" | `tab:k_range_overhead` | ok (numbers inside disagree — see `verified-discrepancies.md` E1) |
| `discussion.tex:28` | "Appendix E.2" | synthetic scalability | ok |
| `discussion.tex:50` | "Appendix E.1" | empirical per-$k$ | ok |
| `discussion.tex:101` | "Appendix E.1" | empirical per-$k$ | ok |
| `computational_experiments.tex:52` | "Appendix D.1" | benchmark suites | ok (content incomplete — R2.5) |
| `computational_experiments.tex:92` | "Appendix D.1" | benchmark suites | ok (content incomplete — R2.5) |
| `results.tex:146` | "the per-problem tables in the appendices" | Tables 6–7 | ok |

**Other document → supplementary:**

| Line | Text | Resolves to | OK? |
|------|------|-------------|-----|
| `previously_published_statement/main.tex:77` | "supplementary material (Section S.I)" | no such section | **broken — supplementary uses Appendix A–E lettering** |

## LaTeX macros (identical in both documents)

`main.tex:44–52` / `supplementary.tex:29–37`:

```latex
\newcommand{\GTS}{\ensuremath{\mathrm{D2S}}}
\newcommand{\STG}{\ensuremath{\mathrm{S2D}}}
\newcommand{\wstar}{\ensuremath{w^{*}}}
\newcommand{\Sig}{\ensuremath{\Sigma}}
\newcommand{\iso}{\ensuremath{\cong}}
\newcommand{\calG}{\ensuremath{\mathcal{G}}}
\newcommand{\RR}{\ensuremath{\mathbb{R}}}
\newcommand{\IsalSR}{\textsc{IsalSR}}     % renders "ISALSR" — see R2.8(b)
\newcommand{\fcs}{\ensuremath{\hat{w}}}   % the fast canonical string
```

Citation style is IEEEtran `\cite{}` (numeric). `\citep{}`/`\citet{}` are **not** available — natbib is not loaded.

## Compile commands

```bash
cd article/paper          && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd article/supplementary  && pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

Page counts as submitted: `main.pdf` 12, `supplementary.pdf` 10, `main_anonymous.pdf` 12.
