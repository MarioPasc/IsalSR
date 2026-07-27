# T13 — Document architecture and the 12-page constraint

| Field | Value |
|---|---|
| Reviewer comments closed | R2-**C4** (organisation), R2-**C6** (length), **C7** (supplementary placement — R1/R2/R3 disagree) · and E6 |
| Type | Coordination — **stays open for the whole revision** |
| Owner | **Karl** + **Mario** (co-owned) · Ezequiel consulted on `methodology.tex` |
| Depends on | page-cost declarations from T03, T04, T05, T06, T07, T09, T10 |
| Blocks | T12, T14 |
| Status | NOT STARTED |
| Target | continuous · **content freeze 2026-09-12** |

---

## 1. Why this is a ticket and not a note

The 12-page limit is the binding constraint of the entire revision, and it is
currently **violated before any new content is added**:

| Item | Pages |
|---|---|
| `article/paper/main.pdf` as submitted | **12 / 12** |
| Author bios and photos — *required to be added now* | not yet in the PDF |
| `article/supplementary/supplementary.pdf` | 10 |

The decision letter is explicit: *"all materials — including references, bios,
photos, etc. — must fit within the 12-page limit."* The bios are drafted at
`main.tex:147–172` and the three photos are on disk but not compiled in. So the
paper is over budget on day one, before a single reviewer request is honoured.

Meanwhile **seven tickets want to add text**, and R2 asks (C6) for the paper to be
*shorter*. Without a ticket that owns the arithmetic, every content ticket will
assume the space exists and the last one to land will lose. This ticket owns the
ledger and the allocations.

---

## 2. The C7 conflict — the three reviewers are mutually exclusive

| Reviewer | C7 answer | C8 |
|---|---|---|
| R1 | digital library | after revisions |
| R2 | **part of the main paper** if accepted (cannot exceed the strict page limit) | after revisions |
| R3 | digital library | **as is** |

R2's request is arithmetically impossible: 12 pages of main + 10 of supplementary
cannot become 12, and R2 simultaneously asks for the main paper to be trimmed. The
two reviewers who disagree with R2 are 2-of-3, and R3 accepts the supplementary
exactly as it stands.

**Decision taken 2026-07-27: keep the supplementary separate** as digital-library
material. The response must argue this from the page limit and from R1/R3's
agreement — not by dismissing R2. Note in passing that R2's own C6 answer
("should be trimmed a bit") is in tension with their C7 answer, and say so
courteously; it is the strongest available argument and it is R2's own.

---

## 3. Established facts

- **Main paper structure**: 7 sections, 3 tables, 4 figures. Full inventory in
  `manuscript-map.md`.
- **Supplementary structure**: Appendices A–G, 11 tables, 4 figures.
- **E6 — dead content.** `methodology.tex` carries five large `\begin{comment}`
  blocks: full duplicates of all three pseudocodes and two TikZ trace figures, at
  `:135–143, 194–251, 269–417, 460–522, 530–660, 812–879, 986–1023, 1039–1054,
  1073–1138` — roughly **600 of 1,143 lines**. They compile to nothing, so they cost
  no pages, but they are the reason the same pseudocode exists in two files, which is
  how the Table 3 / Table 4 confusion in R2.4 arose. Delete them.
- **Appendix A restates Theorems 3.13/3.14/3.15 verbatim**, so each exists twice
  under two numbers — which is why R2 had to write "Theorem 3.13/3.15" and
  "Lemma A.2" in one sentence. Replacing the restatements with references saves
  supplementary space and removes a genuine source of reader confusion. Coordinate
  with T07, which is rewriting those proofs anyway.

---

## 4. Mandatory reading

- `.claude/notes/review/source/00-editor-and-decision.md` — the page-limit and
  required-elements text, verbatim
- `.claude/notes/review/source/README.md` — the hard-constraints section
- `.claude/notes/review/source/reviewer-2.md` — C4, C6, C7 and the note on their conflict
- `.claude/notes/review/source/reviewer-1.md` / `reviewer-3.md` — their C7/C8 answers
- `.claude/notes/review/source/manuscript-map.md` — the full structural inventory
- `.claude/notes/review/source/verified-discrepancies.md` — E6
- The §8 "Proposed answer" and page-cost declaration of **every** content ticket

---

## 5. Work specification

### 5.1 Establish the baseline
Compile with bios and photos included and measure the true starting overrun. That
number, not 12, is the budget this ticket manages.

### 5.2 Run a page ledger
A live table in §6 with one row per content ticket: requested pages, allocated
pages, placement (main / supplementary / cut), and status. **Every content ticket
must declare its page cost to this ticket before its text is written**, not after.

Expected demands: T07 (new definition + full proof), T09 (all suite problems
documented + operator-set passage), T04 (third comparator: table columns, a figure,
statistical treatment), T05 (extension paragraph + revised N), T10 (break-even
condition), T06 (fallback ledger), T03 (Gray, if promoted).

### 5.3 Default placement rule
**Main paper**: the claim, the headline number, the statistical verdict.
**Supplementary**: per-problem detail, proofs, configuration, ablations, ledgers.

Under this rule most new content lands in the supplementary, which has no stated
page limit. The main paper's net change should be near zero — which is also the
answer to C6.

### 5.4 Recover space in the main paper
Candidates, in order of expected yield:
1. Delete the E6 comment blocks (source hygiene; no page yield, but do it).
2. Move per-problem and per-k detail from Results into the supplementary; keep the
   aggregate and the test.
3. Compress Table 2 / Table 3 or merge them if the revised metrics allow.
4. Tighten the Related Work section — R1 and R3 both rated the introduction and
   references adequate, so this is low-risk trimming.
5. Only then consider figures. All four are load-bearing; the critical-difference
   diagram in particular is what makes the Demšar protocol legible, and R1 praised
   the protocol explicitly.

### 5.5 Address C4 (organisation)
R2 rated organisation "Could be improved" without elaborating. Their eight comments
are the evidence: six of eight are bookkeeping errors caused by content that lives
in two places and drifts (see `verified-discrepancies.md`, Aggregate view). The
organisational fix is therefore **single-sourcing**, which T09 and T11 are already
doing. Frame the C4 answer around that rather than around section reordering — it
is what R2's own comments actually point at.

### 5.6 Guard the freeze
After 2026-09-12 no content ticket may add text without an explicit trade recorded
here. State what was traded away and by whom.

---

## 6. Page ledger

| Ticket | Requested | Allocated | Placement | Status |
|---|---|---|---|---|
| baseline (bios + photos) | | | main | |
| T03 Gray | | | | |
| T04 hash baseline | | | | |
| T05 extension | | | | |
| T06 fallback ledger | | | | |
| T07 definition + proof | | | | |
| T09 appendix rebuild | | | | |
| T10 break-even | | | | |
| **recovered** | — | | main | §5.4 |
| **net main** | — | | | must be ≤ 12 |

---

## 7. Acceptance criteria

- **AC-0.** §8 Work log filled in as the work proceeds, including every trade made
  and who agreed to it.
- **AC-1.** Baseline compile with bios and photos measured; the true overrun recorded.
- **AC-2.** Page ledger (§6) complete, with every content ticket having declared a cost.
- **AC-3.** Final `main.pdf` **≤ 12 pages** including references, bios and photos.
- **AC-4.** E6 comment blocks deleted from `methodology.tex`.
- **AC-5.** C7 position drafted, arguing from the page limit and from R1/R3 rather
  than dismissing R2.
- **AC-6.** C4 answer drafted around single-sourcing, with the evidence.
- **AC-7.** C6 answer drafted: what was cut or moved, and the net page change.
- **AC-8.** Content freeze enforced from 2026-09-12; any post-freeze addition recorded
  with its trade.
- **AC-9.** `double_blind/` recompiled and verified at the same page count.
- **AC-10.** §9 filled.

---

## 8. Work log

_(empty — to be filled by the implementing agent)_

---

## 9. Proposed answer

### 9.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| `main.pdf` pages | 12 | | AC-3 |
| Bios and photos included | **no** | yes | AC-1 |
| `supplementary.pdf` pages | 10 | | |
| Supplementary placement | digital library (proposed) | digital library | AC-5 |
| Dead comment blocks in `methodology.tex` | ~600 of 1,143 lines | 0 | AC-4 |
| Theorems restated verbatim in Appendix A | 3 | | §3 / T07 |
| Content moved main → supplementary | — | | AC-7 |
| Content cut | — | | AC-7 |
| Net main-paper page change | — | | AC-7 |
| R2 answer to C4 | "Could be improved" | — | AC-6 |
| R2 answer to C6 | "Should be trimmed a bit" | — | AC-7 |

### 9.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 9.3 Draft response text

```latex
%% --- R2, C7 (supplementary placement) ---
\begin{response}
%%  1. Take the position plainly: the supplementary remains separate digital-
%%     library material.
%%  2. Argue from arithmetic, not preference: 12 pages of main text and 10 of
%%     supplementary cannot be combined inside a 12-page limit that must also
%%     accommodate references, biographies and photographs.
%%  3. Note courteously that this is what Reviewers 1 and 3 asked for, that
%%     Reviewer 3 accepts the supplementary as is, and that Reviewer 2's own C6
%%     answer asks for the main paper to be shorter -- which points the same way.
%%  4. State what was done instead: the supplementary was made self-contained and
%%     every cross-document reference verified (T11), so it reads correctly as a
%%     separate document.
\changeref{}
\end{response}

%% --- R2, C4 (organisation) ---
\begin{response}
%%  Frame around single-sourcing: six of Reviewer 2's eight comments trace to the
%%  same organisational weakness -- the same content living in two places and
%%  drifting apart. Name the fix (generated tables, verified cross-references,
%%  one authoritative campaign manifest) rather than describing section moves.
\changeref{}
\end{response}

%% --- R2, C6 (length) ---
\begin{response}
%%  What moved, what was cut, and the net page change -- with the note that the
%%  paper absorbed new content required by R1.2, R1.4, R2.1, R2.5 and R3.1
%%  without growing.
\changeref{}
\end{response}
```

### 9.4 Residual risk

> Candidates: R2 reading the refusal to merge the supplementary as non-compliance
> (mitigated by arguing from the limit and from R1/R3, and by making the
> supplementary self-contained); a late content ticket forcing a post-freeze trade;
> the 12-page compile passing locally but failing under the publisher's template.
