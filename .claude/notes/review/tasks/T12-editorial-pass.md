# T12 — Editorial pass: abstract, naming, spelling, readability

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.5**, **R2.8**, **R3.2**, R2-C1, R1-C3, R1-C5, R2-C5 (and E9) |
| Type | Copy-editing / prose |
| Owner | **Karl** (primary) · each author signs off their own sections |
| Depends on | T13 (do not copy-edit text that is about to be cut or moved) |
| Blocks | T14 |
| Status | NOT STARTED |
| Target | 2026-09-14 |

---

## 1. Why these are grouped

Every one of these is surface-level text quality, and every one is cheapest as a
single pass over the whole manuscript. Two of the three reviewers raised the same
abstract defect independently (R2.8a and R3.2), which is a signal that the abstract
is being read closely: it is the first thing all three reviewers evaluated, and it
is the reason Reviewer 2 answered **No** to C1.

Structured answers folded in here because they are the same work:
- R1-C3 introduction *"Could be improved"*
- R1-C5 and R2-C5 readability *"Readable — but requires some effort to understand"*
- R2-C1 title/abstract/keywords **No**

R2-C4 (organisation) and R2-C6 (length) are **not** here — they are structural, not
editorial, and belong to T13. Do not attempt to trim by copy-editing.

**Sequencing matters.** This ticket runs *after* T13 has decided what stays, what
moves and what is cut, and after the content tickets have landed their text.
Copy-editing prose that is about to be deleted is wasted, and re-copy-editing after
a restructure is worse.

---

## 2. Verbatim comments

**R1.5:**
> 5) Overall, please read through the paper to fix minor writing issues and improve
> overall readability of it.

**R2.8:**
> 8. The Abstract section contains a duplicated phrase: "…on both methods on both
> methods." This should be corrected. Besides, the manuscript alternates between
> "ISALSR" and "IsalSR", and between "canonicalisation" and "canonicalization."
> These should be unified.

**R3.2:**
> There is a typos at the abstract:
> on both methods on both methods: -> on both methods:

---

## 3. Established facts

### 3.1 The abstract defect — `main.tex:81`
> A paired test across problems on the empirical␣␣reduction factor returns Cohen's
> d > 2 at p < 10⁻²¹ **on both methods on both methods**: canonicalisation eliminates
> a mean of 34 % …

R3 gives the exact correction: delete one occurrence. The **same line also carries a
double space** in "empirical␣␣reduction factor", which neither reviewer mentioned.

**The abstract is duplicated across documents.** `double_blind/paper/main_anonymous.tex`
carries the same text, and `previously_published_statement/main.tex:98–115` restates
the same headline numbers. Fixes must propagate to all three — and the headline
numbers themselves will have changed by then (T02, T05), so the propagation is not
purely cosmetic.

### 3.2 ISALSR vs IsalSR
The macro `\IsalSR` is defined as `\textsc{IsalSR}` (`main.tex:51`), which renders
as small-caps "ISALSR". Plain-text "IsalSR" bypassing the macro appears at
`related_work.tex:23, 50, 76, 111` and in `introduction.tex` prose. **The defect is
at the usage sites, not in the macro** — do not "fix" the macro. Decide the rendered
form once (the reviewers wrote both back at us; R2 uses "ISALSR" throughout their
review, R1 uses "ISALSR" in B2), then route every occurrence through the macro.

### 3.3 -isation vs -ization
Both spellings in active use, sometimes in the same file:
- `-isation`: `main.tex:81`; `computational_experiments.tex:33, 40`;
  `results.tex:6, 168, 193`; `discussion.tex:18, 68`; `conclusion.tex:9`
- `-ization`: `methodology.tex:679`; `supplementary.tex:485, 563, 567, 636, 645,
  650, 675, 695, 710, 713, 741, 752, 768, 792, 800`

Broadly the main paper is British and the supplementary is American, but neither is
clean. **Sweep the whole family at once**: neighbourhood/neighborhood,
labelled/labeled (`supplementary.tex:782` "labelled" vs `labeled` throughout
`methodology.tex`), colour/color, normalised/normalized, behaviour/behavior.
TPAMI does not mandate a variant; internal consistency is what is being asked for.
Pick one, state the choice in §6, and apply it to both documents.

### 3.4 E9 — the acknowledgements
`main.tex:126` credits "Claude Opus 4.7 (Anthropic)" for benchmark-suite discovery,
Picasso parallelisation code, and the companion website. Fine to keep. Worth a
second look for one reason only: the benchmark-suite composition is itself under
challenge in R2.5, so an acknowledgement that an LLM discovered it is a sentence a
hostile reader could use. Keep it — it is honest and correct — but ensure T09 has
made the suite's provenance and criteria airtight so the two do not interact badly.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-1.md` — §R1.5 and the C3/C5 answers
- `.claude/notes/review/source/reviewer-2.md` — §R2.8 and the C1/C5 answers
- `.claude/notes/review/source/reviewer-3.md` — §R3.2; note R3 rated readability
  *"Easy to read"*, so the readability complaint is 2-of-3, not unanimous
- `.claude/notes/review/source/verified-discrepancies.md` — D5, D6, E9
- `.claude/notes/review/source/manuscript-map.md` — LaTeX macro list
- `.claude/notes/review/tasks/T13-page-budget-and-architecture.md` — **must be
  settled before this ticket starts**
- The `humanizer` skill (v3.0.0) — apply in scientific mode; the author's own voice
  profile for computer science / symbolic regression is the reference register

---

## 5. Work specification

1. **Abstract.** Fix the duplication and the double space. Then re-read it against
   the revised results — the headline numbers will have moved (T02, T05, T04 adds an
   arm) and R2 answered **No** to C1, so the abstract needs more than the typo fix.
   Propagate to `double_blind/` and to the previously-published statement.
2. **Naming.** One rendered form; every occurrence through the macro; both documents.
3. **Spelling.** One variant across the whole family, both documents. A committed
   word-list check (e.g. a `codespell`/`aspell` config or a small script) prevents
   recurrence and is worth the ten minutes.
4. **Introduction (R1-C3).** R1 rated it "Could be improved" without elaborating.
   Read it against R1's own B2 statement — that paragraph is an unusually accurate
   restatement of the contribution and is effectively a model of what the reviewer
   wanted the introduction to say. Use it as the target.
5. **Readability (C5, two reviewers).** Concentrate on `methodology.tex`, which is
   where the density is. Worked examples and figure references carry more than
   sentence-level smoothing here.
6. **Full pass.** Whole manuscript and supplementary, including captions, footnotes
   and the bibliography. Then a final read of the compiled PDF, not the source —
   line breaks, orphans, table overflow and math spacing only show up there.
7. **Bios and photos.** The decision letter requires author biographies and photos in
   this revision. Bios are drafted at `main.tex:147–172`; photos are present
   (`EzequielLopez.pdf`, `MarioPascual.jpg`, `KarlThurnhofer.pdf`). Verify they
   compile, render, and fit — and note that they count against the 12 pages (T13).
8. **Clean PDF.** The main manuscript must contain **no coloured or highlighted
   text**. Any annotated version is a separate "Summary of Changes" upload. Verify on
   the final compiled PDF.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds, including the naming and
  spelling decisions taken and why.
- **AC-1.** Abstract duplication and double space fixed in all three documents.
- **AC-2.** Abstract updated to the revised results and re-read against R2's C1.
- **AC-3.** One rendered form of the name; zero occurrences bypassing the macro.
- **AC-4.** One spelling variant across both documents for the whole family; a
  committed automated check that would catch a regression.
- **AC-5.** Introduction revised against R1-C3.
- **AC-6.** Full pass complete over main + supplementary, including captions.
- **AC-7.** Bios and photos present, compiling, and rendering correctly.
- **AC-8.** Final compiled PDF verified free of colour and highlighting.
- **AC-9.** Every author has signed off their own sections; record in §7.
- **AC-10.** §8 filled.

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Abstract, "on both methods on both methods" | present | | AC-1 |
| Abstract, double space | present (unreported) | | AC-1 |
| Abstract propagated to `double_blind/` | same defect | | AC-1 |
| Abstract propagated to statement | same numbers | | AC-1 |
| Rendered name form | ISALSR **and** IsalSR | | AC-3 |
| Occurrences bypassing `\IsalSR` | ≥ 4 in `related_work.tex` + intro prose | 0 | AC-3 |
| canonicalisation / canonicalization | both, ~9 vs ~16 occurrences | | AC-4 |
| labelled / labeled | both | | AC-4 |
| neighbourhood / neighborhood | both | | AC-4 |
| normalised / normalized | both | | AC-4 |
| Automated spelling check | none | | AC-4 |
| R2 answer to C1 | **No** | — | AC-2 |
| Introduction (R1-C3) | "Could be improved" | | AC-5 |
| Readability (C5) | "requires some effort" ×2 | | AC-6 |
| Author bios and photos | absent from PDF | present | AC-7 |
| Colour / highlighting in main PDF | — | none | AC-8 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

```latex
%% --- R2.8 ---
\begin{response}
%%  1. Fix confirmed for all three items, with the note that the abstract fix
%%     propagated to the double-blind version and the previously-published
%%     statement, which carried the same text.
%%  2. State the naming decision and that every occurrence now routes through the
%%     macro -- the defect was at the usage sites, not in the macro.
%%  3. State the spelling variant chosen, that the whole family was swept (not
%%     only canonicalis/zation), and that an automated check now guards it.
%%  4. Mention the double space, which the reviewer did not raise but which is on
%%     the same line. Cheap, and it signals the line was actually re-read.
\changeref{}
\end{response}

%% --- R3.2 ---
\begin{response}
%%  Short. Thank R3, confirm the exact correction they supplied, and cross-
%%  reference R2.8 rather than repeating it.
\changeref{}
\end{response}

%% --- R1.5 ---
\begin{response}
%%  1. Confirm the full pass, and say what it covered rather than asserting that
%%     the paper is now well written.
%%  2. Address C3 (introduction) and C5 (readability) here, naming the sections
%%     that changed most.
%%  3. Note that R3 rated readability "Easy to read", so the pass targeted the
%%     sections the other two reviewers would have found dense -- primarily the
%%     methodology.
\changeref{}
\end{response}

%% --- R2, C1 ---
\begin{response}
%%  What changed in the title, abstract and keywords, and why the abstract needed
%%  more than the typo fix (the headline numbers moved).
\changeref{}
\end{response}
```

### 8.4 Residual risk

> Candidates: content landing after this pass and reintroducing the inconsistencies
> (hence the T13 dependency and T14's final check); the abstract's headline numbers
> drifting from the final tables; bios and photos pushing past 12 pages.
