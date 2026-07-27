# Manuscript Baseline Numbers — T01 §8.1

Extracted from:
- `results.tex` = `.../article/paper/results.tex`
- `discussion.tex` = `.../article/paper/discussion.tex`
- `computational_experiments.tex` = `.../article/paper/computational_experiments.tex`
- `methodology.tex` = `.../article/paper/methodology.tex`

Root: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/article/paper/`

---

## 1. Directory Map

```
article/paper/
  computational_experiments.tex   methodology  + statistical analysis
  conclusion.tex
  discussion.tex
  introduction.tex
  main.tex / main.pdf / main.aux / main.bbl / main.blg / main.log / main.out
  methodology.tex
  results.tex
  cd_2d_r2_rf.pdf
  reduction_factor_distribution.pdf
  fig_d2s.pdf / fig_s2d.pdf
  EzequielLopez.pdf / KarlThurnhofer.pdf
reviews/   (not read — not needed)
```

---

## 2. Baseline Numbers Table

| Quantity | Value in manuscript | file:line | Ticket value | Agree? |
|---|---|---|---|---|
| T_canon UDFS | 0.296 ms (table) / 0.28 ms (text) | results.tex:57 / results.tex:169 | 0.296 ms | YES (table) |
| T_canon Bingo | 0.817 ms (table) / 0.82 ms (text) | results.tex:58 / results.tex:175 | 0.817 ms | YES (table) |
| T_eval Bingo | **1.29 ms** | results.tex:58 | ≈0.14 ms | **NO — 9× discrepancy** |
| T_eval UDFS | ~519 ms (table) / ~500 ms (text) | results.tex:57 / results.tex:169 | not quoted | N/A |
| Canon:eval ratio Bingo | 0.817/1.29 ≈ **0.63:1** (inferred) | results.tex:58 | 3.3:1 | **NO** |
| Canon:eval ratio UDFS | T_eval/T_canon **> 1,500** | results.tex:191 | 1:64 | **NO — >23× discrepancy** |
| S UDFS | 1.07 | results.tex:57 | 1.07 | YES |
| S Bingo | 0.93 | results.tex:58 | 0.93 | YES |
| ρ UDFS | 1.56 ± 0.24 | results.tex:57 | 1.56 | YES |
| ρ Bingo | 1.83 ± 0.09 | results.tex:58 | 1.83 | YES |
| Bingo median OH | 39.2% | results.tex:58 | 39% | YES |
| k<5 overhead | 38.5% | results.tex:178 | — | — |
| k∈[5,15) overhead | 45.9% | results.tex:178 | — | — |
| k∈[15,32) overhead | 41.6% | results.tex:178 | — | — |
| "14,841 DAGs" | present | discussion.tex:37 | "14,841" | PRESENT but unsupported |

**Root cause of T_eval / ratio discrepancies**: Tickets quote CLAUDE.md figures from the
22-problem production campaign (Bingo eval ≈0.14 ms, Bingo ratio 3.3:1, UDFS ratio 1:64).
The manuscript reports the final 50-problem campaign figures (Bingo eval 1.29 ms,
UDFS eval ~519 ms). The old CLAUDE.md figures are stale relative to the submitted manuscript.

---

## 3. Definition of S

**results.tex:49** (table caption): `$S$: median search-only speedup`

**computational_experiments.tex:124-127**:
> The *search-only speedup*
> $S = T_{\mathrm{search}}^{\mathrm{baseline}} / T_{\mathrm{search}}^{\mathrm{IsalSR}}$
> isolates the effect of deduplication on pure search time.

Where (computational_experiments.tex:116-120):
> For the \IsalSR{} variant, $T_{\mathrm{search}} = T_{\mathrm{total}} - T_{\mathrm{canon}}$;
> for the baseline, $T_{\mathrm{search}} = T_{\mathrm{total}}$.

**Verdict for D1**: S is a ratio of measured wall-clock search times (total minus canon cost),
NOT normalised by number of evaluations.

---

## 4. The 14,841 Verdict

**Verbatim sentence (discussion.tex:37-40)**:
> no false collision has been observed across the $14{,}841$ DAGs in
> the unit-test suite or the millions generated during the SR
> experiments, and each synthetic DAG attains the maximal
> $\rho = k!$ at every $k \in \{1, \ldots, 9\}$.

**Verdict: (a)** — The manuscript asserts a "unit-test suite" of 14,841 DAGs. The current
repo test suite has 887 test functions (no stored corpus); CLAUDE.md records 890 passing
tests and 41,217 permutation instances across k=1..8. Neither 14,841 nor any artifact
close to it appears anywhere in the repo source or tests. This is a numerical claim
in the manuscript that is **not backed by any identifiable artifact** in the current
codebase. A reviewer could request the corpus; it does not exist as described.

---

## 5. Literal Canonical Strings Printed in the Manuscript

| String | Context | file:line | At risk from determinism fix? |
|---|---|---|---|
| `VcVspv*pv+PpcnnC` | Figure caption: "canonical string" (S2D figure) | methodology.tex:256 | **YES** — explicitly labelled as canonical |
| `VgnV*C` | Running example body text: "w = Vg n V* C" | methodology.tex:272 | YES — presented as canonical |
| `Vs NV+ NVg …` (partial) | D2S figure caption / walkthrough | methodology.tex:644 | YES — partial canonical prefix |

The string at methodology.tex:256 is the highest-risk: it is explicitly labelled "the canonical
string" in a figure caption. If the determinism fix changes ~10% of strings, this 16-character
example may not match the post-fix output.

---

## 6. Implementation-Dependent Numbers at Risk from C++ Re-execution

All numbers in Table~\ref{tab:three_axis} (results.tex:57-58) depend on implementation:

| Number | Why implementation-dependent |
|---|---|
| T_canon: 0.296 ms (UDFS), 0.817 ms (Bingo) | Python `fast_canonical_string` wall-clock; C++ port will differ |
| T_eval: ~519 ms (UDFS), 1.29 ms (Bingo) | Host-method evaluation cost on same hardware |
| OH: 0.05% (UDFS), 39.2% (Bingo) | Ratio T_canon/T_total; changes if C++ changes T_canon |
| S: 1.07 (UDFS), 0.93 (Bingo) | Wall-clock ratio; changes with C++ T_canon |
| discussion.tex:23: "0.28 ms (UDFS) and 0.82 ms (Bingo)" | Same as above, text repetition |
| discussion.tex:61: "T_eval/T_canon > 1,500" | Computed ratio; changes if T_canon changes |
| results.tex:178: k-bucket overheads 38.5/45.9/41.6% | All overhead-derived |

ρ values (1.56, 1.83), S values, and R² values are **method-level**, not
implementation-dependent, and survive a C++ re-execution.
