# T14 — Response letter and submission package

| Field | Value |
|---|---|
| Reviewer comments closed | all 15, plus the AE comment and every structured answer |
| Type | Assembly / delivery |
| Owner | **Mario** (primary — corresponding author) · **Karl** and **Ezequiel** review |
| Depends on | **every other ticket** |
| Blocks | — |
| Status | NOT STARTED |
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
| 12 | Previously-published statement updated (E5, and the arXiv v3 if T11 produced one) |
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

_(empty — to be filled by the implementing agent)_

---

## 7. Retrospective

> Not a reviewer response. Record, for the next round: which tickets ran late and
> why, which reviewer comments turned out to be deeper than they first read, which
> of E1–E9 the reviewers would have found anyway, and what the round-2 risk register
> looks like. If the paper comes back, this is the first thing to read.

### 7.1 Comment coverage ledger

| Comment | Ticket | Answered | Evidence produced | Round-2 risk |
|---|---|---|---|---|
| R1.1 | T10 | | | |
| R1.2 | T06 | | | |
| R1.3 | T07 | | | |
| R1.4 | T04 | | | |
| R1.5 | T12 | | | |
| R2.1 | T07 | | | |
| R2.2 | T11 | | | |
| R2.3 | T09 | | | |
| R2.4 | T11 | | | |
| R2.5 | T09 | | | |
| R2.6 | T09 | | | |
| R2.7 | T08 | | | |
| R2.8 | T12 | | | |
| R3.1 | T05 | | | |
| R3.2 | T12 | | | |
| C1, C3 | T12 | | | |
| C4, C6, C7 | T13 | | | |
| B3 (R2, "Partially") | T07 | | | |
| B4 (all three) | T04, T05, T06 | | | |

### 7.2 Aggregate round-2 risk register

> Collect the §"Residual risk" sections from every ticket here, ranked. This is the
> single most useful artefact if the paper returns.
