# T14 — Response letter and submission package

| Field | Value |
|---|---|
| Reviewer comments closed | all 15, plus the AE comment and every structured answer |
| Type | Assembly / delivery |
| Owner | **Mario** (primary — corresponding author) · **Karl** and **Ezequiel** review |
| Depends on | **every other ticket** |
| Blocks | — |
| Status | **STRUCTURAL SCAFFOLDING COMPLETE (2026-08-06).** Cover, AE response, both summary-of-changes tables, the supplementary-material position, both reviewer acknowledgements, C7 and the closing are written and verified. **AC-1 partially met** (5 `\todoblock` remain, all outside this pass's lane and all gated on the C2 campaign: R2.6, C4/C6, R3.1 paired-test, and two campaign-outcome markers). **AC-2 met except the R2.6 row.** AC-3, AC-5, AC-6, AC-7 outstanding. |
| Target | draft 2026-09-20 · **submit 2026-09-24** |

---

## 1. What this ticket is

The terminal ticket. It consumes the §"Proposed answer" block of every other ticket
and assembles them into the response letter, then verifies and uploads the package.

It exists as a ticket rather than a final step because the submission has
**mechanical requirements that fail silently** — file designations, a clean
manuscript PDF, a page limit that includes photographs — and because the response
letter is the document the editor reads first.

The skeleton already exists, with all reviewer comments reproduced verbatim and all
response blocks empty:

```
…/journal/69c1637a28a81fea2badda9a/reviews/response_to_reviewers.tex
```

---

## 2. Mandatory reading

- `.claude/notes/review/source/00-editor-and-decision.md` — **the whole file**;
  every mechanical requirement is quoted verbatim there
- `.claude/notes/review/source/README.md` — hard constraints and file ownership
- `reviews/response_to_reviewers.tex` — the skeleton and its environments
- The §"Proposed answer" of T01–T13
- `.claude/notes/review/tasks/T11-cross-document-consistency.md` — §5.3; the
  cross-reference walk must be re-run here, after all content has landed

---

## 3. Verbatim requirements from the decision letter

> If you should choose to revise your paper, please prepare a separate document
> describing how each of the reviewers' comments are responded to in your revision
> and send it to us by 24-Sep-2026.

> Important: Your main manuscript file cannot include colored or highlighted text.
> Please upload a clean, publication-ready version of your manuscript under the
> "Formatted (Double Column) Main File - PDF Document Only" file designation. If you
> would like to include an annotated version of your main manuscript file, please
> upload it under the "Summary of Changes" file designation.

> Because this is a revision, we request that you add your author bios and photos at
> this time. … (Please note that all materials - including references, bios, photos,
> etc. - must fit within the 12-page limit imposed by the Submission Guidelines.)

Required elements of the revised paper: abstract, index terms, author affiliation
information, main text, references, figure captions, table titles, and a brief
biography of each author.

Submission URL is recorded in `00-editor-and-decision.md`.

---

## 4. Work specification

### 4.1 Assemble the letter
Paste each ticket's §8.3 draft into the matching `\begin{response}` block. Delete
every `\todoblock`. Fill the AE response and the cover paragraph.

**Fill the "Summary of changes" table** in the skeleton (`response_to_reviewers.tex:111–133`)
— one row per comment: tag, type, change made, location. That table is what an
editor scans first, so the "Location" column must be specific (section and
subsection, not "throughout").

### 4.2 Attachments
- **Continuity table** from T02 §8.5 — Python-engine versus C++-engine numbers per
  axis, so a reviewer can map the numbers they reviewed onto the new ones. The
  article treats the engine as an implementation detail; this attachment is where
  the change is disclosed to the reviewers explicitly. Decision taken 2026-07-27.
- **Numerical audit list** from T09 §AC-7, if it is compact enough to be useful as
  an appendix to the letter.

### 4.3 Register
The letter is read by three people who wrote very different reviews. Rules:
- Address each reviewer by their own concerns; do not cross-reference so heavily
  that a reviewer has to read another reviewer's section to find their answer.
- **Concede plainly where the reviewers are right.** Eleven of eleven factual claims
  the reviewers made check out. A defensive letter against a record like that reads
  badly; an accurate one reads well.
- **Volunteer the defects we found ourselves** (E1–E9). It costs a few lines and it
  demonstrates the manuscript was genuinely re-audited rather than patched.
- No new claims in the letter that are not in the paper.

### 4.4 Reconcile the three-way disagreements
Two exist and both need a stated position in the letter:
- **C7** — supplementary placement. Position and argument come from T13 §9.3.
- **Coverage** — R3 wants more problems, R1 endorses the existing protocol, R2 wants
  the paper shorter. T05 resolves this by extending the evidence without extending
  the exposition; the letter should say so once, where R3 will read it.

### 4.5 Pre-submission verification
A mechanical checklist, run against the **final compiled PDFs**, not the sources:

| # | Check |
|---|---|
| 1 | `main.pdf` ≤ 12 pages, including references, bios and photos |
| 2 | No coloured or highlighted text anywhere in `main.pdf` |
| 3 | All required elements present (§3) |
| 4 | Author bios and photos render correctly for all three authors |
| 5 | All 21 hardcoded cross-document references re-verified (T11 §5.3) — **content has moved since T11 closed** |
| 6 | Every numeric claim matches its source artefact (T09 numerical audit list) |
| 7 | Abstract headline numbers match the final tables |
| 8 | Abstract identical across `article/`, `double_blind/`, and the previously-published statement |
| 9 | `double_blind/` recompiled, content-identical, and anonymous — no author names, no real website URL, no acknowledgements that deanonymise |
| 10 | `article/` carries the **real** website URL; `double_blind/` the anonymised one (E7) |
| 11 | Both websites reachable and current |
| 12 | Previously-published statement updated (E5). **No arXiv v3** — decided 2026-07-27; R2.2 is answered as a comment and the journal version supersedes the preprint |
| 13 | Bibliography compiles with no missing or duplicated entries |
| 14 | Every `\todoblock` removed from the response letter |
| 15 | Summary-of-changes table complete, with specific locations |
| 16 | File designations correct on the submission site (§3) |

### 4.6 Upload
- Clean main PDF → **"Formatted (Double Column) Main File - PDF Document Only"**
- Annotated main PDF (if produced) → **"Summary of Changes"**
- Response letter → **"Summary of Changes"**
- Supplementary → its own designation

---

## 5. Acceptance criteria

- **AC-0.** §6 Work log filled in as the work proceeds.
- **AC-1.** Every `\begin{response}` block filled; zero `\todoblock` remaining.
- **AC-2.** Summary-of-changes table complete with specific locations.
- **AC-3.** Continuity table attached.
- **AC-4.** C7 and the coverage disagreement each have a stated position.
- **AC-5.** All 16 pre-submission checks pass, verified against final PDFs.
- **AC-6.** Karl and Ezequiel have both reviewed the letter; sign-off recorded in §6.
- **AC-7.** Uploaded under the correct designations before 2026-09-24.
- **AC-8.** §7 filled — here, the retrospective rather than a reviewer answer.

---

## 6. Work log

### 2026-08-06 — structural scaffolding of the letter

Wrote the eight editor-facing and reviewer-facing blocks that carry no campaign
numbers: the cover paragraph, the AE response, both summary-of-changes tables,
the note on supplementary placement, the R1 and R2 acknowledgements, the C7
position, and the closing. Ran the American-orthography sweep over the whole
letter. Did not touch `article/`, the annotated manuscript copy, any code, or
any block owned by a parallel agent.

**Baseline recorded before starting** (manuscript repo at `e01a2c1`): five
modified files under `article/`, three untracked paths under `reviews/`. That
baseline is unchanged; the only addition is `M reviews/response_to_reviewers.tex`.

#### Decisions

1. **The cover paragraph asserts no number.** Five material changes are named as
   *actions* (the foundation is complete, the measurements were re-executed, a
   third arm was added, the suite went 50→70, an editorial pass ran), never as
   outcomes. Every sentence survives any shift in the campaign's results. The
   one sentence that cannot be written yet — the outcome of the re-executed
   campaign and the extension — is a `\todoblock`.

2. **The summary-of-changes table had to be split in two.** With ten rows filled
   it overflowed by 412.6 pt, which no font reduction recovers, and `longtable`
   would have required a preamble edit that this pass was forbidden. Split by
   reviewer: `tab:summary` (Reviewer 1) and `tab:summary2` (Reviewers 2 and 3),
   both `\small`, `booktabs`, caption above, both cited by `\ref`. The added
   float renumbers `tab:incidence`, `tab:operator_sets` and `tab:coverage`, which
   is harmless: every letter-internal table is referenced by `\ref`, and every
   literal `Table~N` in the letter denotes a *manuscript* table (checked).

3. **The "Note on the supplementary material" was moved above the tables.**
   Leaving it after them produced a page carrying three lines and nothing else,
   then a page carrying three lines again after the second table was split out.
   Moving it ahead of the floats gives prose on p. 2, Table 1 on p. 3, Table 2 on
   p. 4. Second table set to `[!htbp]` so it can take a float page.

4. **The supplementary position is argued once, in C7, and summarised at the
   front.** The front note states the position and the arithmetic in five
   sentences and points to C7; C7 carries the full argument. Neither repeats the
   other. The argument does **not** claim the reviewers agreed: it states R2's
   request accurately, gives the page arithmetic (12 + 10 into fewer than 12,
   with bios and photos still to be compiled in), notes that R2's own C6 asks for
   a shorter paper, records that R1 and R3 both answered C7 "digital library" and
   that R3 accepts it as is, and then says what was done to lower the cost of the
   separation.

5. **The R1 acknowledgement leads on B2, not on thanks.** R1's significance
   statement restates the contribution more precisely than the paper did,
   including the near-quadratic complexity. That is the specific thing worth
   acknowledging, and it doubles as an honest disclosure that two sections of the
   submitted paper said "near-linear".

6. **The R2 acknowledgement concedes all eight comments "on the facts" and then
   states the one place we change nothing** (comment 2, where the divergent
   definition is in the preprint). An earlier draft said "all eight comments are
   correct" and then announced a disagreement two paragraphs later; that was a
   self-contradiction and was rewritten.

#### Surprises

- **R1.1 landed mid-pass.** The parallel agent filled the R1.1 block while this
  work was in flight, and its `\changeref` explicitly records `"near-linear"
  replaced by near-$O(k^2)$` in Sections 6 and 7. That retroactively validated
  the cross-reference written into the R1 acknowledgement, and made the R1.1
  summary-table row fillable, so it was filled rather than left pending. Only
  **R2.6** remains pending in the tables.
- **E3 is still open in the manuscript sources.** `discussion.tex:97` and
  `conclusion.tex:11` still read "near-linear" in **both** `article/` and
  `reviews/internal_copy_reviewed_article/`. The letter now promises the
  correction in two places (the R1 acknowledgement and the R1.1 `\changeref`).
  **The manuscript edit has not been made.** Listed in the checklist below.
- **The style guard found one British form the hand-built word list missed**:
  `orbit-stabiliser` at what is now line 837 of the letter, inside the R1.4
  answer. Corrected to `orbit-stabilizer`.

#### The two handoffs from the previous wave

**Handoff 1 — the anonymised companion-site mirror is stale.** Both sites are
live and the mirror is correctly anonymised, with no page-level divergence, but
`little-manifold.github.io/isalsr-anon/` was last modified 2026-05-12 against
2026-07-28 for `mariopasc.github.io/IsalSR/` — roughly 2.5 months. Nothing was
changed here; the refresh is item 11 of the checklist below, and it becomes
load-bearing the moment the real site is updated to describe the revision.

**Handoff 2 — the letter's own orthography.** T12 measured 50 British-variant
occurrences outside the five blocks it owned and could not sweep them. This pass
did.

| Lane | Before | After | Note |
|---|---:|---:|---|
| Sweepable prose | 58 | 0 | replaced |
| Inside this pass's own placeholders | 3 | 0 | placeholders replaced wholesale |
| Object language (`\emph{}` list in the R2.8 answer) | 9 | 9 | must stay British or the sentence is meaningless |
| Preamble comment, `rcomment` bodies, C4/C6 placeholder | 3 | 3 | out of lane by instruction |
| **Raw total in scope of the count** | **73** | **12** | |

Plus `orbit-stabiliser → orbit-stabilizer`, found by the guard, not by the word
list: **59 replacements in total**.

Families replaced: `canonicalis`×14, `serialis`×12, `normalis`×9, `neighbour`×6,
`characteris`×4, `labelled`×4, `favour`×3, `labelling`×2, `summaris`×1,
`artefact`×1, `organis`×1, `behaviour`×1, `stabiliser`×1.

**`scripts/check_manuscript_style.sh` can be pointed at the letter.** Its
signature is `check_manuscript_style.sh <dir-or-file> [...]`; nothing is
hardcoded except the wordlist path, and explicit file arguments are accepted
verbatim. Confirmed working:

```bash
bash scripts/check_manuscript_style.sh \
  <manuscript-root>/reviews/response_to_reviewers.tex
```

It now exits 1 on twelve spelling hits and three naming hits, **all of them
expected**: the nine object-language words, the `rcomment` occurrence, the
C4/C6 placeholder, and three bare `IsalSR` occurrences not routed through the
`\IsalSR` macro (letter lines 727 and 776 in the R1.3/R1.4 answers, plus one
inside a `rcomment`, which must never be touched). Add the invocation to the
pre-submission checklist and read its output against this expected set rather
than expecting exit 0.

#### Summary-of-changes rows: what could and could not be filled

| Row | Filled? | Source |
|---|---|---|
| R1.1 | **yes** | its answer landed mid-pass with a concrete `\changeref` |
| R1.2, R1.3, R1.4, R2.1, R3.1 | already filled | previous wave |
| R1.5, R2.2, R2.3, R2.4, R2.5, R2.7, R2.8, R3.2 | **yes** | each row derived from the shipped answer's own `\changeref`, not from the tickets |
| **R2.6** | **no** | its answer is still an empty `\todoblock`; the row turns on the run ledger of the re-executed campaign (T09 numbers gated on C2) |

Five `\todoblock` remain in the letter, all outside this pass's lane: the
campaign-outcome sentence in the cover, the R2.6 table row, the R2.6 answer, the
C4/C6 answer, and the R3.1 paired-test figures.

#### Verification

| Gate | Before | After |
|---|---|---|
| `pdflatex` exit, both passes | 0 | 0 |
| `Overfull` | 0 | 0 |
| `LaTeX Warning` | 0 | 0 |
| unresolved `??` | 0 | 0 |
| `rcomment` integrity | — | `HEAD 18 tree 18 differing 0` |
| new changes under `article/` | — | none |

Pages 1–5, 17, 27 and 31 were rendered to PNG at 150 dpi and read. Two layout
defects were found this way and neither appears in the log: the table overflow
of decision 2 and the near-empty page of decision 3.

---

## 6b. Pre-submission checklist

Mechanical, run against the **final compiled PDFs**, immediately before upload.
Extends §4.5 with what the ticket work logs have since flagged.

| # | Check | Source |
|---|---|---|
| 1 | `main.pdf` ≤ 12 pages including references, bios and photos | §4.5 |
| 2 | No coloured or highlighted text anywhere in `main.pdf` | §4.5 |
| 3 | All required elements present (§3) | §4.5 |
| 4 | Author bios and photos render for all three authors | §4.5 |
| 5 | **Re-walk all 21 hardcoded cross-document references against the rebuilt PDFs**, not the sources. Breaks silently on any renumbering. Last walk: 17 pass, 4 fail (S1, S7, P1 as submitted; **M6 at `results.tex:176`, "Table 8" → now Table 10, broken by the revision and still open**). Re-walk **after T13**, since the appendix-letter list moves with it | T11 §5.3, §7.3 |
| 6 | Every numeric claim matches its source artefact | T09 |
| 7 | Abstract headline figures match the final tables — **currently the submitted campaign's; the C1 answer says so explicitly and that disclosure must be removed when the figures land** | T12, letter C1 |
| 8 | Abstract identical across `article/`, `double_blind/` and the previously-published statement | §4.5 |
| 9 | `double_blind/` recompiled, content-identical and anonymous. **Propagation was deferred by T11**: three files, and `supplementary_anonymous.tex` is *not* a copy, so its edits must be applied by hand | T11 §7.8 |
| 10 | `article/` carries the real website URL, `double_blind/` the anonymised one | §4.5, E7 |
| 11 | **Regenerate the anonymised companion-site mirror.** `little-manifold.github.io/isalsr-anon/` is ~2.5 months staler than `mariopasc.github.io/IsalSR/` (2026-05-12 vs 2026-07-28). Both live, anonymisation correct, no page-level divergence today — but any update to the real site describing the revision must be mirrored | T11 §7.7 |
| 12 | Previously-published statement updated. **No arXiv v3** | §4.5, E5 |
| 13 | Bibliography compiles with no missing or duplicated entries | §4.5 |
| 14 | Every `\todoblock` removed from the response letter | §4.5 |
| 15 | Both summary-of-changes tables complete, with specific locations | §4.5 |
| 16 | File designations correct on the submission site | §3 |
| 17 | **`"near-linear"` replaced by near-$O(k^2)$ in `discussion.tex` and `conclusion.tex`.** Still present in **both** `article/` and the annotated copy. The letter promises this correction in two places | E3, T10 |
| 18 | **The R3.1 `\changeref` cites "Sec. 5.1 (suite size 50→70)"; the correct section is 4.2.** Fix before submission | T12 §7.8 |
| 19 | **Reconcile "70 problems / nine sources" against `related_work.tex:93` (50) and `conclusion.tex:14` ("50 … eight").** Four sites | T12 §7.8 |
| 20 | Re-run the statement's page count: `pdfinfo <supplementary>.pdf \| grep Pages`; the word "thirteen" at statement `:142` goes stale. **Last action before upload** | T11 §7.6 |
| 21 | **Promote the annotated copy into `article/` with all colour stripped**, then verify zero `\color` and zero inline review notes. Separate, deliberate step | skill §Paths, rule 4 |
| 22 | Re-run `scripts/check_manuscript_style.sh` on the letter and on `article/`. On the letter, expect **exit 1** with exactly the twelve object-language/out-of-lane spelling hits and three naming hits documented above; anything else is a regression | this pass |
| 23 | Recompile the letter: exit 0 twice, `Overfull` 0, `LaTeX Warning` 0, no `??`, and `rcomment` integrity `18 / 18 / 0` | this pass |

---

## 7. Retrospective

> Not a reviewer response. Record, for the next round: which tickets ran late and
> why, which reviewer comments turned out to be deeper than they first read, which
> of E1–E9 the reviewers would have found anyway, and what the round-2 risk register
> looks like. If the paper comes back, this is the first thing to read.

### 7.1 Comment coverage ledger

| Comment | Ticket | Answered | Evidence produced | Round-2 risk |
|---|---|---|---|---|
State as of 2026-08-06. "Answered" means the letter block is written; it does
not mean every number in it is final.

| Comment | Ticket | Answered | Evidence produced | Round-2 risk |
|---|---|---|---|---|
| R1.1 | T10 | yes | claim recalibrated; growth-rate inference withdrawn | figures move with the campaign |
| R1.2 | T06 | yes | violation-rate measurement, bypass audit | — |
| R1.3 | T07 | yes | numbered definition, pseudocode, incidence table | — |
| R1.4 | T04 | yes, **numbers gated on C2** | third arm implemented and C2-ready | **highest**; this is the round's heaviest request |
| R1.5 | T12 | yes | editorial pass, style guard committed | — |
| R2.1 | T07 | yes | Def. 3.5 corrected, Lemma A.2 reproved | R2 will re-read the proof |
| R2.2 | T11 | yes (no change made) | consistency argued from the journal manuscript | R2 may press on the preprint |
| R2.3 | T16 | yes | per-solver operator table; alphabet stated | — |
| R2.4 | T11 | yes | 21-reference walk; number-free replacements | **breaks on renumbering — checklist item 5** |
| R2.5 | T09 | yes | database expressions printed | — |
| R2.6 | T09 | **no — empty `\todoblock`** | run ledger gated on C2 | blocks AC-1 and AC-2 |
| R2.7 | T08 | yes | scoring rule, effective paired-seed counts | AC-7 gated on C2 |
| R2.8 | T12 | yes | orthography and naming unified; guard added | — |
| R3.1 | T05 | yes, **paired-test figures gated on C2** | coverage table; suite 50→70 | seeds 30→20 will be questioned |
| R3.2 | T12 | yes | typo and double space removed | — |
| C1, C3, C5 | T12 | yes | abstract rewritten; introduction rewritten | abstract figures pending |
| C4, C6 | T13 | **no — empty `\todoblock`** | — | T13 NOT STARTED |
| C7 | T13 | **yes (this pass)** | page arithmetic; R1/R3 C7 answers; reference audit | R2 may not accept the refusal |
| AE | — | **yes (this pass)** | — | — |
| Cover, closing, summary tables | T14 | **yes (this pass)** | — | one campaign-outcome sentence pending |
| B3 (R2, "Partially") | T07 | yes | Lemma A.2 reproved | — |
| B4 (all three) | T04, T05, T06 | partly | two of three arms gated on C2 | — |

### 7.2 Aggregate round-2 risk register

> Collect the §"Residual risk" sections from every ticket here, ranked. This is the
> single most useful artefact if the paper returns.

Partial, from the letter-assembly pass of 2026-08-06. Ranked by what would cost
the most in a second round.

1. **R1.4, the hash comparator, is the round's decisive request.** R1's own B2
   already frames hashing as offering "no correctness guarantee", so the
   comparator is expected to lose on completeness; the reviewable quantity is the
   *gap* and its cost. The arm is implemented and C2-ready, but every number in
   the answer is gated on the campaign.
2. **Cross-document references break silently.** The last walk left one failure
   open (`results.tex:176`) and the appendix-letter list moves with T13. R2
   found the original instance by reading appendices against the main text and
   will do it again.
3. **C7 is a refusal.** R2 asked for the supplementary inside the main paper and
   we decline on page arithmetic. The argument is sound and R1 and R3 both
   support the current arrangement, but R2 rated the paper Fair and may not
   accept it. Nothing further can be offered without exceeding the page limit.
4. **R3.1's seed reduction, 30 → 20.** The extension buys twenty problems by
   spending seeds. R1 explicitly endorsed "50 problems, 30 seeds" in its opening
   assessment, so the trade is visible to a reviewer who liked the original
   protocol. The letter states the resulting change in $N$; whether R1 reads that
   as a strengthening or a weakening is not controllable from here.
5. **The abstract's figures are stated to be provisional.** Deliberate and
   disclosed, but it means the abstract is the last thing written and the
   easiest thing to leave stale. Checklist items 7 and 20.
