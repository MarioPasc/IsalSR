# TPAMI-2026-05-1699 — Major Revision: source notes

Reference documentation for agents working the IsalSR revision.
These files record **what the reviewers said and what is verifiably true in the sources**.
They do not contain proposed answers — drafting responses is a separate task.

Manuscript: *Representation of Directed Acyclic Graphs by Sequences of Instructions for Symbolic Regression* (IsalSR).
Decision received **2026-07-26**. Revision due **2026-09-24**.

## Files in this directory

| File | Content |
|------|---------|
| `00-editor-and-decision.md` | Editor/AE comments, submission mechanics, page limit, deadline, required elements. |
| `reviewer-1.md` | R1 — Overall *Good*. 5 comments. |
| `reviewer-2.md` | R2 — Overall *Fair*. 8 comments. |
| `reviewer-3.md` | R3 — Overall *Excellent*. 2 comments. |
| `manuscript-map.md` | Where every section, theorem, definition, table and figure lives; numbering cross-walk between reviewer references and source labels. |
| `verified-discrepancies.md` | Every factual claim by a reviewer, checked against the `.tex` sources, with file:line. Plus discrepancies the reviewers did not catch. |
| `codebase-pointers.md` | Where in `IsalSR/` the relevant implementations and result files are, for anyone who has to re-run or re-measure something. |

Response-letter skeleton (verbatim comments + empty response blocks):
`…/article/journal/69c1637a28a81fea2badda9a/reviews/response_to_reviewers.tex`

## Reviewer ratings at a glance

| Item | R1 | R2 | R3 |
|------|----|----|----|
| Overall | Good | Fair | **Excellent** |
| Significance (B1) | Good | Good | Good |
| Technically sound (B3) | "Appears to be — didn't check completely" | **Partially** | Yes |
| Experimental validation (B4) | Lacking in some respects | Lacking in some respects | Lacking in some respects |
| Title/abstract/keywords (C1) | Yes | **No** | Yes |
| References (C2) | Sufficient | Sufficient | Sufficient |
| Introduction (C3) | **Could be improved** | Yes | Yes |
| Organization (C4) | Satisfactory | **Could be improved** | Satisfactory |
| Readability (C5) | Requires some effort | Requires some effort | Easy to read |
| Length (C6) | About right | **Should be trimmed a bit** | About right |
| Supplementary (C7) | Digital library | **Part of main paper** | Digital library |
| Supplementary accept (C8) | After revisions | After revisions | **As is** |

No reviewer suggested additional references (all answered NA to C2's citation box).
All three answered B4 identically: *"Lacking in some respects; some cases of interest not tested."*

## Comment taxonomy

Grouping the 15 comments by the kind of work each implies. Nothing here prescribes a response.

**New measurement / experiment required**
- R1.4 — no comparison against naive hash-based deduplication on a fixed-order DAG serialization.
- R1.2 — failure rate of the reachability condition (Thm 3.13 / 3.15) is never reported.
- R3.1 — why only 50 problems when Section 2.4 names larger databases (AI Feynman 120, SRBench 250+).

**Theory / formal writing**
- R2.1 — proof of Lemma A.2 is terse; does not formally establish that κ-minimal candidate selection always yields valid D2S strings.
- R1.3 — `normalize_const_creation(D)` is invoked in the FCS pseudocode and defined nowhere else in the paper.

**Framing / honesty of claims**
- R1.1 — Bingo search-only speedup $s = 0.93$ is a net loss under a fixed wall-clock budget, but is described as "approximately neutral".

**Factual corrections (all verified against source — see `verified-discrepancies.md`)**
- R2.2 — {g,i} vs {−,/} label characters for NEG/INV between journal Def 3.2 and preprint Def 2.2.
- R2.3 — $\Sigma_{\mathrm{SR}}$ includes Pow and $\sqrt{\cdot}$; host operator set in Appendix D.2 excludes both.
- R2.4 — Appendix A cites "Table 4 of the main document"; no Table 4 exists in the main text.
- R2.5 — "20-problem subset of AI Feynman" vs Table 5 (10 equations) vs Tables 6–7 (24 Feynman rows).
- R2.6 — run count `2,640` inconsistent with a 50-problem suite; recurs in Appendix D.3.
- R2.7 — `nan` for Vlad-2 and Korns-12 under Bingo–IsalSR, undiscussed; handling in paired tests unclear.
- R2.8 / R3.2 — abstract duplication "on both methods on both methods"; ISALSR/IsalSR and canonicalisation/canonicalization inconsistency.

**Prose quality**
- R1.5 — general pass for minor writing issues and readability.
- R1 C3 — introduction "could be improved".
- R2 C4 — organization "could be improved"; C6 — "should be trimmed a bit".

## Hard constraints on the revision

- **12 pages** for the main file, *including* references, biographies and photos. `article/paper/main.pdf` is currently **exactly 12 pages**; `article/supplementary/supplementary.pdf` is **10 pages**.
- R2 asks (C7) for the supplementary to become *part of the main paper* "if accepted (cannot exceed the strict page limit)". R1 and R3 both ask for it to stay as digital-library supplementary material, and R3 accepts it **as is**. The three answers are mutually exclusive under the page limit; this is a decision the revision has to take a position on.
- Author bios and photos must be added now. Photos already present in `article/paper/`: `EzequielLopez.pdf`, `MarioPascual.jpg`, `KarlThurnhofer.pdf`. Bios are already drafted in `article/paper/main.tex:147–172`.
- The main manuscript PDF must be **clean** — no colored or highlighted text. Any annotated/diff version is uploaded separately under the "Summary of Changes" designation.
- Any new experiment added in response to R1.2, R1.4 or R3.1 has to fit a manuscript that R2 wants *shorter*, not longer.

## Scope rules that still apply

From `article/CLAUDE.md`: in the journal project, only files owned by Mario may be edited.
The current on-disk layout differs from the layout described in that file — there is no `mario/` subdirectory any more. Actual layout:

```
journal/69c1637a28a81fea2badda9a/
├── article/paper/          main.tex, introduction, related_work, methodology,
│                           computational_experiments, results, discussion, conclusion
├── article/supplementary/  supplementary.tex + table_supplementary_{udfs,bingo}.tex
├── double_blind/           anonymised copies (verified identical content)
├── cover_letter/
├── previously_published_statement/
└── reviews/                ← new; response letter skeleton
```

Ownership by author is unchanged: `introduction.tex`, `related_work.tex`, `methodology.tex`,
`conclusion.tex` and `main.tex` are Ezequiel's; `computational_experiments.tex`, `results.tex`
and `discussion.tex` are Mario's. Several reviewer comments (R2.2, R2.4, R1.3, R2.1) land in
Ezequiel-owned files or in the supplementary — confirm ownership before editing.
