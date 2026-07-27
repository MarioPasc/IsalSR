# TPAMI-2026-05-1699 — Revision task board

Tickets for the major revision of *Representation of Directed Acyclic Graphs by
Sequences of Instructions for Symbolic Regression* (IsalSR).

**Decision received** 2026-07-26 · **Revision due** 2026-09-24 · **Hard freeze on new numbers** 2026-09-10.

Source material for every ticket lives in `../source/`. Nothing in `../source/`
proposes a fix; the tickets do.

---

## Roles

| Person | Remit |
|---|---|
| **Mario** (+ Claude Code) | Implementation, methodology design, testing, computational experiments, figures, `computational_experiments.tex`, `results.tex`, `discussion.tex`. Corresponding author. |
| **Ezequiel** | Mathematics and Isal-family design. Owns `introduction.tex`, `related_work.tex`, `methodology.tex`, `conclusion.tex`, `main.tex`, and the arXiv preprint. |
| **Karl** | Written-article revision, methodology plot-holes, organisation, register, copy-editing. |

---

## Tickets

### Track A — Engine and representation (not reviewer-driven; internal decision 2026-07-27)

| # | Title | Owner | Depends on | Blocks |
|---|---|---|---|---|
| [T01](T01-cpp-core-port.md) | C++ core port + numerical-equivalence gate | Mario | — | T02, T03, T06 |
| [T02](T02-cpp-reexecution-campaign.md) | Full re-execution on the C++ engine + continuity table | Mario | T01 | T04, T08, T09, T10 |
| [T03](T03-gray-code-integration.md) | Gray-code integration: design-space analysis, implementation, ablation | Ezequiel + Mario | T01 | T13 |

### Track B — New experiments demanded by reviewers

| # | Title | Owner | Reviewer | Depends on |
|---|---|---|---|---|
| [T04](T04-naive-hash-dedup-baseline.md) | Naive fixed-order-serialisation hash dedup baseline | Mario | **R1.4** | T01, T02 |
| [T05](T05-benchmark-extension.md) | Bounded benchmark extension: Feynman remainder + ODE-Strogatz | Mario + Karl | **R3.1** | T02 |
| [T06](T06-reachability-failure-rate.md) | Reachability-condition failure rate + timeout fallback | Mario | **R1.2** | T01, T07 |

### Track C — Theory

| # | Title | Owner | Reviewer |
|---|---|---|---|
| [T07](T07-theorem-foundation.md) | Complete the formal foundation of Theorem 3.15 | Ezequiel + Mario | **R2.1**, **R1.3** |

### Track D — Numerical integrity

| # | Title | Owner | Reviewer | Depends on |
|---|---|---|---|---|
| [T08](T08-nan-and-paired-test-integrity.md) | NaN failures and paired-test integrity | Mario | **R2.7** (+E4) | T02 |
| [T09](T09-appendix-d-rebuild.md) | Appendix D rebuild and numerical consistency | Mario | **R2.5**, **R2.6**, **R2.3** (+E1, E2, E8) | T02, T05, T08 |

### Track E — Manuscript

| # | Title | Owner | Reviewer | Depends on |
|---|---|---|---|---|
| [T10](T10-claim-calibration.md) | Claim calibration in Discussion and Conclusion | Mario + Ezequiel | **R1.1** (+E3) | T02 |
| [T11](T11-cross-document-consistency.md) | Cross-document and package consistency | Karl + Ezequiel | **R2.2**, **R2.4** (+E5, E7) | — |
| [T12](T12-editorial-pass.md) | Editorial pass: abstract, naming, spelling, readability | Karl | **R1.5**, **R2.8**, **R3.2**, C1, C3 (+E9) | T13 |
| [T13](T13-page-budget-and-architecture.md) | Document architecture and the 12-page constraint | Karl + Mario | C4, C6, C7 | all content tickets |

### Track F — Delivery

| # | Title | Owner | Depends on |
|---|---|---|---|
| [T14](T14-response-letter-assembly.md) | Response letter and submission package | Mario | all |

---

## Reviewer comment → ticket map

Every one of the 15 numbered comments is covered exactly once.

| Comment | Ticket |
|---|---|
| R1.1 — Bingo `S = 0.93` framed as "approximately neutral" | T10 (evidence from T02) |
| R1.2 — reachability failure rate never reported | T06 |
| R1.3 — `normalize_const_creation` undefined | T07 |
| R1.4 — no naive hash-dedup baseline | T04 |
| R1.5 — writing pass | T12 |
| R2.1 — Lemma A.2 proof terse | T07 |
| R2.2 — `{g,i}` vs `{−,/}` across documents | T11 |
| R2.3 — Σ_SR vs host operator set | T09 |
| R2.4 — "Table 4 of the main document" does not exist | T11 |
| R2.5 — Feynman counts 20 / 10 / 24 | T09 |
| R2.6 — 2,640 vs 6,000 runs | T09 |
| R2.7 — `nan` for Vlad-2 and Korns-12 | T08 |
| R2.8 — abstract duplication, ISALSR/IsalSR, -isation/-ization | T12 |
| R3.1 — why only 50 problems | T05 |
| R3.2 — abstract typo (duplicate of R2.8a) | T12 |

Structured answers: C1, C3 → T12. C4, C6, C7 → T13. B3 ("Partially", R2) → T07. B4 (all three reviewers) → T04 + T05 + T06.

Unraised discrepancies from `../source/verified-discrepancies.md` Part 2:
E1, E2, E8 → T09. E3 → T10. E4 → T08. E5, E7 → T11. E6 → T13. E9 → T12.

---

## Dependency spine

```
T01 ──┬─> T02 ──┬─> T04 ──┐
      │         ├─> T08 ──┤
      ├─> T03   ├─> T09 <─┤── T05
      └─> T06   └─> T10   │
                          │
T07 ──> T06               │
T11 ────────────────────  ┤
                          ├──> T13 ──> T12 ──> T14
```

**Critical path**: T01 → T02 → T09 → T13 → T12 → T14.
T02 is a multi-week Picasso campaign. Anything that delays T01 delays submission.

---

## Standing rules for every ticket

1. **Fill §7 Work log as you go.** Decisions, dead ends, surprises, disagreements.
   A ticket with an empty work log is not complete regardless of its other criteria.
2. **Fill §8 Proposed answer only when §6 is fully met.** T14 pastes these verbatim
   into `reviews/response_to_reviewers.tex`; they must be written in that file's
   register, and every claim in them must be backed by a number the ticket produced.
3. **Never soften a negative result.** If an experiment shows IsalSR losing, say so
   in §8 and characterise the regime. R1 explicitly asked for honest framing; a
   second round of over-claiming is the fastest route to rejection.
4. **The main manuscript PDF must stay clean** — no colour, no highlighting. Annotated
   versions go in the separate "Summary of Changes" upload.
5. **Confirm file ownership before editing** (`../source/README.md`, bottom section).
   Several reviewer comments land in Ezequiel-owned files.

---

## Decisions already taken (2026-07-27, Mario)

| Question | Decision |
|---|---|
| Do the Gray code and the C++ port belong to this revision? | Yes, both. C++ first, Gray on top of it. |
| C++ re-run: replace or supplement the reported results? | **Replace.** The article reports only the C++ numbers and treats the engine as an implementation detail. A Python↔C++ continuity table is produced **for the response letter only**, so reviewers can see what moved. |
| Gray-code insertion point | **Deliberately left open.** T03 must find the best location by analysis; re-execution cost and re-proof cost are explicitly not constraints. |
| R3.1 posture | Justify the exclusions **and** run a bounded extension: AI Feynman remainder passing criterion (ii) + the 14 ODE-Strogatz problems (SRBench's ground-truth track). |
| Compute budget for ablation arms | Full 12 h budget everywhere. An internal saturation-based early-stopping mechanism may be used to save queue hours — **subject to the safety contract in T02 §5.4**, and not reported in the article. |
| Supplementary material (C7) | **Keep separate** as digital-library material, against R2's request. Argue from the 12-page limit and from R1/R3 agreeing. |
