---
name: review-answer
description: |
  Write one reviewer's comment answer into the TPAMI response letter
  (`reviews/response_to_reviewers.tex`), starting from a ticket work log in
  `.claude/notes/review/tasks/`. Decides with the user whether a figure or table
  is needed, builds it from live code if so, then writes the answer as continuous
  scientific prose following a fixed narrative spine and an explicit style
  contract. Carries the matching manuscript edits into the annotated draft at
  `reviews/internal_copy_reviewed_article/`, marked in blue, leaving the clean
  manuscript under `article/` untouched. Verifies every quoted number against the
  ticket, compiles, renders,
  and optionally pushes to Overleaf. Triggers on "write the response to R1.3",
  "answer reviewer comment R2.1", "draft the response letter entry", "fill the
  response block", "write the reviewer answer", "review-answer", "respond to the
  reviewer".
---

# review-answer — write one comment's answer into the response letter

`review-ticket` produces the *evidence*. This skill turns that evidence into the
*letter the reviewers read*. They checked every number in round 1; assume they
will do it again, with the code open.

Two failure modes dominate, and both are silent:

1. **Quoting a retracted number.** Ticket logs are append-only. They contain
   numbers that were later found contaminated, superseded, or vacuous, sitting in
   plain tables that look quotable. Step 1 exists entirely to stop this.
2. **Writing prose that reads as machine-generated.** An outline of labelled
   parts with bulleted sub-lists is the default failure. The reviewer wanted a
   scientific argument, not a status report.

---

## Non-negotiables

- **Never edit text inside an `rcomment` environment.** Reviewer comments are
  verbatim from the decision letter. Verify with `git diff` before committing.
- **Never write a number you have not traced to a ticket line or re-measured.**
  No estimates, no "approximately", no recomputing from memory.
- **No claim in the letter may be false or misleading by omission.** The letter
  describes the manuscript's revised state, not the project's development
  history, but the two must never conflict. See §5.3.
- **Render the compiled page and look at it.** Not the log, the page. Two real
  defects in the R1.3 figure (missing arrowheads on a directed graph, mathtext
  padding inside a token string) survived a clean compile and were only caught by
  looking.
- **The answer ships as prose.** If the draft has `\paragraph{}` headings,
  `enumerate` as its backbone, or `\textbf{E1 ---}` style labels, it is not
  finished.
- **One comment per invocation.** If asked for several, do them in sequence and
  re-verify each.

---

## Paths

| What | Where |
|---|---|
| Response letter | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/reviews/response_to_reviewers.tex` |
| Figures for the letter | same `reviews/` directory, committed alongside the `.tex` |
| Ticket work logs | `.claude/notes/review/tasks/T*.md` (+ `T*-appendix/`) |
| Verbatim reviewer text | `.claude/notes/review/source/reviewer-{1,2,3}.md` |
| Known discrepancies | `.claude/notes/review/source/verified-discrepancies.md` |
| Manuscript, clean | `…/69c1637a28a81fea2badda9a/article/paper/`, `…/article/supplementary/` |
| **Manuscript, annotated** | `…/69c1637a28a81fea2badda9a/reviews/internal_copy_reviewed_article/{paper,supplementary}/` |
| Overleaf token | `/home/mpascual/research/token-overleaf.txt` |
| Figure generators | `experiments/scripts/generate_fig_*.py` |

The manuscript root is a live Overleaf checkout. **`double_blind/paper/*.tex` is
a byte-identical copy, not a symlink** — every manuscript edit must be mirrored.

### The two manuscript copies, and which one you edit

There are two, and they are not interchangeable.

`article/` is the **clean** manuscript, the one that goes to the journal. Rule 4
of `.claude/notes/review/tasks/README.md` requires it to carry no colour and no
highlighting; annotated versions belong in the separate "Summary of Changes"
upload. It is also the live Overleaf checkout, so anything written there is one
`git push` away from being the submitted paper.

`reviews/internal_copy_reviewed_article/` is the **annotated draft**: the same
manuscript with the revision's changes applied and **wrapped in
`{\color{blue}…}`**. It is the working copy this skill and `review-ticket` write
to, and the artefact that becomes the Summary of Changes. It starts as a
byte-identical copy of the submitted manuscript, so a `diff` against `article/`
is exactly the set of changes the reviewers will be shown.

Working rules:

- **Write manuscript edits into the annotated copy, in blue, not into
  `article/`.** Promoting them into `article/` with the colour stripped is a
  separate, deliberate step, and pushing to Overleaf is another one on top
  (§6). Do neither unless asked.
- **Blue means "changed in this revision", and nothing else.** Do not use it for
  emphasis, and do not use a second colour. If an upstream patch arrives with its
  own colours (red for one author's edits over another's blue, say), collapse all
  of it to blue when integrating.
- **Strip review scaffolding before integrating.** Inline notes of the form
  `[MPG 2026-08-02 — removed: …]` explain a decision to a co-author; they are not
  manuscript content. Grep for them and for any leftover `\color{red}` and
  confirm zero hits after integration.
- **A passage deleted in the revision leaves nothing behind.** Do not ship a blue
  paragraph narrating what used to be there. That belongs in the response letter,
  which may say the old sentence was wrong; the article must not narrate its own
  review history (see §5.3).
- **Check the numbering after integrating anything numbered.** Inserting a
  `theorem`/`definition`/`lemma` renumbers everything after it, and the reviewers
  quote the submitted numbers as literals in their comments and in the letter.
  Prefer appending new environments at the end of a section. Verify from the
  rebuilt PDF, not from the source:
  ```bash
  pdftotext paper/main.pdf - | grep -oE "(Theorem|Lemma|Definition|Corollary) [0-9]+\.[0-9]+" | sort -u -V
  ```
- **Both documents must compile.** `pdflatex` three times in
  `internal_copy_reviewed_article/paper/` and `.../supplementary/`, then check
  the logs for `^! ` and for undefined references.
- The annotated copy is untracked in the Overleaf repo. Leave it that way unless
  the user asks otherwise.

---

## 1. Load, and build the quotable-numbers ledger

Read the ticket, its appendix directory, and the verbatim comment. Then, before
writing a single sentence, **audit the ticket's numbers.**

Grep the ticket for correction markers and read every hit in full:

```
RETRACT  SUPERSED  "must not be quoted"  artefact  contaminated
caveat  vacuous  "not accepted"  refuted  withdrawn  🔴  ⚠️
```

Rules:

- **Later entries win.** Work logs are append-only and chronological; a number in
  an early entry may be corrected three entries down with no edit to the original.
- **A number inside a retracted block is dead everywhere**, even if it also
  appears in a summary table that carries no warning.
- **A zero is only evidence if the experiment could have produced a non-zero.**
  Check the population can actually exhibit the phenomenon. A `Const`-free
  generator reporting "0 normalisation failures" measures nothing; two separate
  tickets fell into this and labelled the row *vacuous* only on re-reading.
- **An "N tested, 0 failed" row needs its exercising count.** 13,394 DAGs of
  which 24 actually triggered the rule under test cannot carry a universal claim.
  Quote the exercising count next to the total, or do not quote the row.
- **Cross-check one headline number against an independent source.** In T07,
  ρ = 1.7931 from the study matched the paper's independently produced 1.793 to
  four significant figures; the contaminated run gave 288.5. That agreement is
  what made the campaign quotable.

Output of this step, kept for §5: a short ledger of every number you intend to
use, each with its ticket line reference. If a number the answer needs does not
exist, say so to the user — do not estimate it.

---

## 2. Decide whether a figure or table earns its place — then interview

Ask yourself first, then ask the user. Do not build anything before the user picks.

A figure or table is justified when the answer rests on one of:

- **a structural or mechanical claim** the reader must see to accept (before/after
  on a graph, a counterexample);
- **a measurement across several populations** (≥ 3 rows × ≥ 3 columns of numbers
  belongs in a table, not in a sentence);
- **a procedure** the manuscript never stated (pseudocode);
- **a rate whose mechanism matters** (a stratification, e.g. rate against $k$).

It is *not* justified when the claim is a single number, a definition that reads
cleanly in one sentence, or an editorial correction.

Then call `AskUserQuestion` with **concrete, instantiated** candidates — not
archetypes. Name the actual object, the actual data, and the cost. Put your
recommendation first and mark it `(Recommended)`. Always include an honest
"prose only" option; most comments do not need a figure.

Candidate archetypes to instantiate from:

| Archetype | Use when | Cost |
|---|---|---|
| Before/after structural pair | a repair or transformation on a graph | build + verify script, ~30 min |
| Population/incidence table | a rate measured on several populations | minutes, numbers already exist |
| Stratification plot or inline series | the mechanism behind a rate | minutes if the histogram is in the ticket JSON |
| Pseudocode block | the manuscript invokes an undefined procedure | minutes |
| Worked numerical example | a claim about semantics or evaluation | build + verify script |
| Counterexample diagram | a definition is too coarse | build + verify script |
| Prose only | everything else | free |

**Hard rule for any figure: it is computed from live code, never drawn from
remembered numbers.** Write a generator under `experiments/scripts/`, have it
build the object through the real API, read the strings and values off the
implementation, assert the properties it claims, and print them. A schematic that
merely illustrates a number you typed is worse than no figure, because it cannot
drift-check itself.

---

## 3. Build the figure or table, if one was chosen

Follow `references/latex-and-build.md` for the generator pattern, float
conventions, and the caption rules. In short:

- Generator takes `--output <path>`, writes vector PDF into the `reviews/` dir,
  and prints every value it embedded so the numbers are auditable in the
  transcript.
- Verify the claim *numerically first*, in a throwaway script, on **both engines**
  where the claim touches core semantics. Only then draw.
- Size text for the figure **after** `\linewidth` scaling (typically ~65%), so
  in-figure fonts must be large in the source.
- **Tables: `\caption` above the body, `\centering`, `\label`, and referenced
  from the text.** Figures: caption below. Both must be cited by `\ref`, never
  as "the table below".
- Then render to PNG and look at it.

---

## 4. Write the answer

Read `references/narrative-and-style.md` in full before drafting. It carries the
narrative spine and the style contract; both are mandatory, not advisory.
`references/worked-example.md` is the R1.3 answer annotated against the spine —
read it if this is your first answer.

The spine, in one screen:

1. **Concede in the first sentence.** No preamble, no throat-clearing.
2. **State the stakes and signal the length** if the answer is long, so the
   reviewer knows why.
3. **Pre-announce the awkward part**, if there is one, and say you address it
   later.
4. **Root cause** — the structural or mathematical reason the object exists.
   Never a historical reason.
5. **The formal object** — definition, algorithm, theorem, as a display block,
   referenced from the prose.
6. **Cost or complexity**, if the object is a procedure.
7. **Properties**, in prose, ordered to end on the one the reviewer cares about.
8. **The measurement** — what was instrumented, which populations, what $N$,
   what result. Table if it is wide.
9. **The mechanism** — a stratification or secondary analysis showing the headline
   rate is not a coincidence.
10. **Supporting measurements**, in prose, not as a list.
11. **A limitation, volunteered** — the exact conditions under which the claim
    fails, and why they cannot arise in the reported setting.
12. **The awkward part**, delivered here, now that the evidence is on the table.
13. **"No reported number changes"**, plainly, if and only if a policy-invariance
    measurement supports it.
14. **Cross-link** to sibling comments that share this answer.
15. **`\changeref{}`** with concrete manuscript locations.

Steps 12 and 13 sit late **on purpose**. The logical order and the rhetorical
order differ: you cannot say "this changes no number" until the measurements are
in front of the reader.

Not every comment needs all fifteen. An editorial correction is steps 1, 13, 15.
Do not pad.

---

## 5. Verify before declaring done

### 5.1 Build gate

```bash
cd <reviews dir>
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error response_to_reviewers.tex; done
```

All four must hold:

- exit status 0 on both passes;
- `grep -c Overfull` on the log returns **0**;
- `pdftotext … | grep '??'` finds no unresolved reference;
- no new `LaTeX Warning` lines.

If the answer also touched the manuscript, the annotated copy has its own gate.
Three passes, because the numbered environments and `\ref`s need them:

```bash
cd <reviews dir>/internal_copy_reviewed_article/paper
for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex; done
grep -c "^! " main.log            # 0
grep -c "Reference.*undefined" main.log   # 0
# then the same in ../supplementary on supplementary.tex
```

Then, in the annotated sources:

- `grep -c "color{red}"` and a grep for inline review notes both return **0**;
- the numbering check of the Paths section reproduces the numbers the reviewers
  cite;
- `article/` is unchanged (`git status` shows nothing under it).

### 5.2 Content gate

- Every number in the answer appears in the §1 ledger.
- Every table: caption above, `\centering`, `\label`, cited by `\ref`.
- Every figure: caption below, cited by `\ref`.
- References to the *manuscript's* tables say "of the manuscript" when the letter
  now has numbered tables of its own, otherwise "Table 3" is ambiguous.
- `git diff` shows **no** change inside any `rcomment` block.
- Grep the answer for implementation jargon and remove it: `raises`, `returns
  False`, `no-op`, `monkey-patch`, `backend`, `assert`, bare function names,
  ticket IDs (`T0`, `AC-`), dates, agent or session references.
- Grep for the banned rhetorical moves listed in the style contract.
- Any generator script: `ruff check` and `ruff format --check` clean.

### 5.3 Honesty gate

Apply this test literally:

> If a round-2 reviewer diffed the submitted and revised manuscripts, read the
> code, and re-ran the scripts, would any sentence in this answer read as
> concealment?

**Disclose**, in your own voice, before they find it:

- anything the reviewer quoted that turned out to be wrong;
- any defect discoverable from the artefacts;
- any limitation of the current design.

**Do not narrate**: ticket IDs, dates, who found what, which agent ran what,
intermediate wrong turns that left no trace in the shipped artefacts. Those are
project history, not manuscript content.

The line between the two is not about comfort. In the R1.3 answer the submitted
policy's three defects were disclosed in full, because the reviewer had quoted
that policy's own description; the sequence of internal code revisions was not,
because nothing in the shipped artefacts records it and it changes no claim.

---

## 6. Commit and push, only if asked

Never push unprompted. When asked:

```bash
# askpass keeps the token out of argv and out of .git/config
printf '#!/bin/sh\nhead -n1 /home/mpascual/research/token-overleaf.txt\n' > "$SCRATCH/askpass.sh"
chmod +x "$SCRATCH/askpass.sh"
export GIT_ASKPASS="$SCRATCH/askpass.sh" GIT_TERMINAL_PROMPT=0
git fetch origin                                   # confirm sync BEFORE committing
git rev-list --left-right --count origin/master...master
git add <tex> <pdf> <figure>
git commit -F -                                    # conventional prefix, no Co-authored-by
git push origin master
git fetch -q origin && git log --oneline -1 origin/master   # confirm the remote moved
rm -f "$SCRATCH/askpass.sh"
```

Check the figure is tracked. The letter fails to compile on Overleaf if a
`\includegraphics` target was left untracked, and `git status` will not warn you
if the file is already committed at an older revision.

**Stage the letter and its figures, never the annotated manuscript copy.**
`reviews/internal_copy_reviewed_article/` is a working artefact; committing it
puts a blue-marked duplicate of the paper into the Overleaf project. Promoting
its content into `article/` is a separate request, and when it comes the colour
is stripped in the process: the clean manuscript carries no `\color`.

---

## References

- `references/narrative-and-style.md` — the spine expanded, and the positive and
  negative style rules with before/after examples. **Read before drafting.**
- `references/latex-and-build.md` — letter preamble, environments, float and
  caption conventions, figure-generator pattern, compile and push commands.
- `references/worked-example.md` — the R1.3 answer annotated against the spine.

Related: `review-ticket` (produces the evidence this skill consumes), and the
global `humanizer` skill v3+ — if a drafted passage exceeds ~200 words and still
reads mechanically after applying the style contract, run it through
`humanizer-pass`.
