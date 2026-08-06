# T12 — Editorial pass: abstract, naming, spelling, readability

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.5**, **R2.8**, **R3.2**, R2-C1, R1-C3, R1-C5, R2-C5 (and E9) |
| Type | Copy-editing / prose |
| Owner | **Karl** (primary) · each author signs off their own sections |
| Depends on | ~~T13~~ — **dissolved 2026-08-06**: all trimming and length reduction delegated to Karl; this ticket does not cut anything |
| Blocks | T14 |
| Status | **DONE except AC-2 and AC-9.** AC-0, AC-1 (paper half), AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-10 met and verified. **AC-2 deferred** — the abstract's *defects* are fixed and its suite description is current, but its headline effect figures wait on the re-execution campaign; no placeholder was written. **AC-9 not startable by an agent** — author sign-off is a human step; a per-file checklist is provided in §7.9. **AC-1 propagation to `double_blind/` and the previously-published statement is pending**, with exact line references in §7.7; both are outside this lane. AC-4/AC-6 cover `paper/` only; the supplementary half ran in the parallel T11 lane under the same conventions. |
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

### 2026-08-06 — editorial pass executed (lane: annotated `paper/` tree + five letter blocks)

**Lane discipline.** `git status` under the manuscript root was recorded before any work and
re-checked after. The five pre-existing dirty files under `article/`
(`paper/computational_experiments.tex`, `paper/discussion.tex`, `paper/references.bib`,
`paper/results.tex`, `supplementary/supplementary.tex`) are unchanged; **this ticket added
nothing to `article/`**. `previously_published_statement/main.{tex,pdf}` became dirty during
the run — that is the parallel T11 lane, not this one. The response letter was compiled into
a private scratch directory so the parallel lane's `.aux` was never clobbered. No file under
`src/`, `experiments/`, `benchmarks/`, `tests/` or `slurm/` was modified; the only new repo
files are the two listed in §7.4.

#### 7.1 The two scope decisions, and what they cost

*No trimming.* T13 is dissolved as a dependency: length reduction belongs to Karl. Nothing
was cut to save space, and the pass made the manuscript **longer** — the annotated paper went
from 14 to 15 pages, entirely from the introduction expansion (AC-5) and the worked example
(AC-6). Items noticed but deliberately left in place for Karl are in §7.8.

*No new headline numbers.* The abstract's defects are fixed; its effect figures are left at
the submitted campaign's values and flagged. No placeholder was written into the manuscript.

#### 7.2 AC-3 — the naming decision, and a correction to this ticket's own §3.2

**Decision: keep the macro, fix the usage sites, render as small-caps ISALSR.** The macro
`\IsalSR` = `\textsc{IsalSR}` is not the defect; it was already correct and already used at
46 sites. The rendered document alternated only because some occurrences were typed as
literal text. Both R1 (in B2) and R2 (throughout) write the name back to us as "ISALSR", so
the small-caps rendering is what the reviewers already read as the canonical form. Changing
the macro would have flipped 46 correct sites to fix 10 wrong ones.

**Measured, not assumed: 10 bypassing occurrences, not the "≥ 4 plus intro prose" this ticket
and `verified-discrepancies.md` D6 both assert.**

| File | Bypassing occurrences (submitted) | Where |
|---|---|---|
| `related_work.tex` | 4 | `:23, 50, 77, 111` (D6 says `:76`; the line is **77**) |
| `methodology.tex` | 5 | subsection title, table caption, definition title, S2D prose, pseudocode "Output:" row |
| `computational_experiments.tex` | 1 | a math-mode superscript $T^{\mathrm{IsalSR}}_{\mathrm{search}}$ |
| `introduction.tex` | **0** | — |
| **total** | **10** | |

**`introduction.tex` has zero.** All six of its occurrences already went through the macro.
Both this ticket's §3.2 and `verified-discrepancies.md` D6 state that plain-text occurrences
appear "in `introduction.tex` prose". They do not. Corrected here; D6 should be amended.

The math-mode case needed a decision of its own. `T_{\mathrm{search}}^{\mathrm{IsalSR}}$`
renders "IsalSR" upright and mixed-case, so it *is* a rendered inconsistency even though it
is notation rather than prose. It now reads `^{\text{\IsalSR}}`, which scales to script size
and matches the body text. **Final count: 0 bypassing occurrences in `paper/`**, the only
literals left being the `\newcommand` and the project URL, where the name is part of an
address and must not be re-typeset.

#### 7.3 AC-4 — the spelling decision

**Decision: American, and sweep the whole family, not the one pair R2 names.**
Three grounds, in order of weight:

1. TPAMI does not mandate a variant, so the requirement is *internal consistency* and nothing
   else. That makes the decision a coin-flip on principle and a cost question in practice.
2. The submitted package already leaned American once the supplementary is counted
   (16 `-ization` against 9 `-isation` in the figures this ticket recorded at §3.3), and the
   supplementary was already broadly American. Choosing British would have meant editing the
   larger half.
3. IEEE house style is American, and IEEE prints "Acknowledgment", which is why
   `\section*{Acknowledgements}` → `\section*{Acknowledgments}` went in with the rest.

**Counts, measured by `scripts/check_manuscript_style.sh` over the whole `paper/` tree:**

| | Before | After |
|---|---|---|
| British-variant occurrences | **44** | **0** |
| Name occurrences bypassing the macro | **10** | **0** |

The families actually present were `canonicalis` (27), `labelled` (1), `neighbour` (3),
`optimis` (6), `normalis` (4), `behaviour` (2), `colour` (2), `standardis` (2), `summaris`
(1), `parameteris` (1), plus `overparameterisation`, `regularised`, `vectorised`,
`visualisations`, `orbit-stabiliser` and `Deep Symbolic Optimisation` (the last is also the
method's correct proper name, which is American). **Sweeping only `canonicalis` would have
left 17 occurrences behind and R2 would have been entitled to raise the same comment again.**

Two of the parallel lane's blue passages carried `-isation` and `in-neighbour`; they were
normalised with the rest, since the point is the rendered document, not authorship of the
line.

#### 7.4 AC-4 — the automated guard, and how it is run

`codespell` is **not installed** in this environment and its `en-GB_to_en-US` builtin would
in any case not catch the naming half. A dependency-free checker was written instead:

| File | Role |
|---|---|
| `scripts/manuscript_style_wordlist.txt` | 45 rules, `<forbidden-regex>\t<replacement>`, with the decision and its rationale in the header |
| `scripts/check_manuscript_style.sh` | walks `*.tex` under each argument; fails on any British form, or on any `IsalSR` outside the macro |

```bash
ROOT=/media/.../journal/69c1637a28a81fea2badda9a
scripts/check_manuscript_style.sh "$ROOT/article" "$ROOT/double_blind" \
    "$ROOT/previously_published_statement" "$ROOT/reviews/internal_copy_reviewed_article"
```

Exit 0 clean, 1 on violation, 2 on usage error. It skips comment-only lines and `\url{}` /
`\href{}` arguments (both produced false positives on the first run and both are legitimate).
`analyse(?!s)` is deliberately negative-lookahead so the correct American noun *analyses*
passes.

**It earned its place immediately: it caught two regressions this pass introduced itself**
("serialisation"/"serialise" in the new introduction text) and one stale rule of its own.
Current state:

| Tree | Spelling | Naming |
|---|---|---|
| `reviews/internal_copy_reviewed_article/paper` (**this lane**) | **0** | **0** |
| `previously_published_statement` | 0 | 0 |
| `article/paper` | 44 | 10 |
| `article/supplementary` | 31 | 1 |
| `double_blind/paper` | 44 | 10 |
| `double_blind/supplementary` | 31 | 1 |
| `reviews/internal_copy_reviewed_article/supplementary` | 6 | 2 |

The `article/` and `double_blind/` figures are the promotion backlog, not defects introduced
here — see §7.7.

#### 7.5 AC-5 — the introduction, diagnosed against R1's B2

R1 rated the introduction "could be improved" and said nothing more, but their B2 statement
restates the contribution more accurately than the submitted introduction did. Treating B2 as
the specification, four things it says and our text did not:

1. **Positioning.** The introduction claimed "no current SR method deduplicates at this
   granularity" and left the reason in §2. A reader who knows fitness sharing and equality
   saturation would stop there. The revision names both families and says why neither
   removes node-ordering redundancy, in R1's own terms.
   *Restraint recorded:* the first draft explained the e-graph case mechanistically ("a
   relabelled copy hash-conses to the same e-class"). **That claim was cut** — hash-consing
   would arguably catch pure renumbering, so the mechanism is not obviously in our favour and
   is not something this ticket can verify. The shipped text asserts only what the paper's §2
   and R1's B2 both already assert.
2. **Why the setting is hard**, not merely different from IsalGraph/IsalChem. Three
   requirements now stated: a label on every internal node, directed edges, one
   operand-ordered operation. Verified against `experiments/models/commutative_encoding.py`,
   whose docstring independently states the twelve-label alphabet and that `Pow` is the only
   non-commutative operation.
3. **Expressiveness is not narrowed** by absorbing Sub/Div. The submitted text never said so.
   The revision states $x-y = x+(-y)$ and $x/y = x\cdot y^{-1}$ as identities — verified
   against the `DECOMPOSITION` table in the same module.
4. **The middle ground** between exhaustive minimization and hashing. Added, and stated
   *more precisely than R1 did*: R1 writes that hash-based approaches offer "no correctness
   guarantee", which is too strong. A fixed-order serialisation hash is **sound but
   incomplete**. The shipped text says that, which is both correct and the framing T04's
   comparator will actually measure.

Also de-staled: contribution 5 and the roadmap both said the evaluation ran "across the
Nguyen and AI Feynman benchmark suites". It has not for some time. Replaced with a
count-free description, since the count belongs to the campaign lane.

`introduction.tex`: 74 → 119 lines. **Longer, per instruction.**

#### 7.6 AC-6 — where the readability work went, and the one substantive addition

R3 rated the manuscript "easy to read", so the complaint is **2 of 3** and was treated as
pointing at specific dense passages rather than at the prose. §3.4 was the obvious target.

**The finding worth recording: §3.4 had neither a figure nor a worked example, unlike §3.2
and §3.3 which both have trace figures.** The paper's central algorithm was the only one a
reader met with no instance. Added as an unnumbered `\noindent\textbf{Worked example.}`
paragraph — **deliberately unnumbered**, because inserting an `example` environment before
Definition 3.9 would have renumbered Definition 3.9 and Theorems 3.13/3.15, which is exactly
what the reviewers cite as literals.

Every value in it was computed from the live API, on both engines, not written from memory
(`scratchpad/T12/probe_example.py`):

| | value |
|---|---|
| $D$ | $\sin(x_1)+\cos(x_1)$, $m=1$, $k=3$ |
| greedy D2S, Sin numbered first | `VsVcpv+Ppc` |
| greedy D2S, Cos numbered first | `VcVspv+Ppc` |
| $\hat w_D$, either numbering | `VcVspv+Ppc` (python and cpp identical) |
| WL hashes of the two candidates | differ, and swap with the numbering |
| why no backtracking | $\kappa=(\ell,h)$ compares $\ell$ first and `c` < `s`, so $h$ is never reached |

This is the smallest expression on which the numbering visibly changes the encoded string,
which is why it was chosen over a single-operator DAG.

Also in this pass: `\subsection{The IsalSR Instruction Set}` and the §3.1 opening sentence
("prove that such string is" → "this string"); one over-long source line in §3.2 split.

**Compiled-PDF read (not the source).** Pages 1, 6, 14, 15 rendered at 110 dpi and inspected.
Findings: the abstract renders correctly with the duplication gone; the worked example's
monospaced token strings typeset correctly; **all three biographies and photographs compile
and render** (AC-7), two on page 14 and Thurnhofer-Hemsi alone on page 15. Overfull boxes:
**0**, against 0 in the clean baseline. Underfull: 21 against 17 — four new loose lines, all
cosmetic.

#### 7.7 Pending propagations, with exact line references

**None of these is in this lane. All are required before submission.**

| Target | What | Exact reference |
|---|---|---|
| `double_blind/paper/main_anonymous.tex` | the identical abstract, carrying **both** defects ("on both methods on both methods" and the double space) plus `canonicalisation` ×2 | **line 71** |
| `double_blind/paper/*.tex` | 44 British-variant occurrences and 10 name occurrences bypassing the macro — byte-identical to `article/paper` | `related_work.tex`, `methodology.tex`, `computational_experiments.tex`, `results.tex`, `discussion.tex`, `conclusion.tex` |
| `previously_published_statement/main.tex` | **no typo and no spelling work needed** — it is already American and already routes the name through the macro. **Only the headline figures** need refreshing | lines **99–115** ($\rho=1.56/1.83$, $34.2\%/45.2\%$, $d=2.38$, $p=2.7\times10^{-22}$, $p=5.9\times10^{-5}$, $p=4.4\times10^{-4}$, $39\%$) |
| `article/paper/*.tex` | the whole of this pass, colour stripped | the promotion step; 44 + 10 violations remain there by design |

**This corrects §3.1 of this ticket**, which implies the statement carries the same textual
defect. It does not; it restates the numbers only.

**Deferred, AC-2.** The abstract's effect figures are the submitted campaign's. They are left
in place and disclosed as under revision in the C1 answer. Do **not** treat AC-2 as met until
they are regenerated.

#### 7.8 Flagged for Karl, not acted on

1. **Page count.** Annotated paper is now **15** pages (clean baseline 12; annotated 14 before
   this pass). The third biography sits alone on page 15; if it moved up ~10 lines the
   document would end at 14.
2. **Cross-file number inconsistency inside the annotated tree, introduced by another lane.**
   The abstract and `computational_experiments.tex` say **70** problems / **nine** sources
   (blue), while `related_work.tex:93` still says **50** and `conclusion.tex:14` says "$50$
   … drawn from eight". Numbers are not this ticket's; whoever owns the campaign must
   reconcile all four.
3. **Wrong section number in another block of the letter.** The R3.1 `\changeref` says
   "Sec.~5.1 (suite size $50\to70$)". The benchmark suite is **Sec. 4.2**
   (`sec:benchmarks` → 4.2 in `main.aux`); 5.1 is Search Space Reduction. T14 should fix it.
4. **The response letter's own orthography.** T11 flagged it and this pass measured it:
   **50** British-variant occurrences across the letter, outside this ticket's five blocks.
   The five blocks written here are American. **Not swept** — the file lane for this ticket
   forbids editing other agents' blocks, and they may still be in flight. **T14 must sweep
   the whole letter**, or the R2.8 answer announcing an American document will sit three
   pages from a block printing "canonicalisation".
5. **E3 ("near-linear" vs near-$O(k^2)$)** is still live at `conclusion.tex:11` and
   `discussion.tex:23`. That is T10's, and R1's own B2 uses "near-O(k^2)", so it will be
   noticed. Not touched here.
6. Five `\begin{comment}` blocks still occupy roughly half of `methodology.tex` (E6, T13).

#### 7.9 E9 — the acknowledgement

**Keep it.** `main.tex:126` credits Claude Opus 4.7 for benchmark-suite discovery, the Picasso
parallelisation code and the companion website, and cites the system card as reference [34].
It is honest, it is accurate, and removing it after the fact would be worse than the risk.

**The risk, recorded once so it is not rediscovered:** R2.5 challenges the benchmark suite's
composition, and the acknowledgement says an LLM discovered that suite. A hostile reader can
pair the two into "the suite was chosen by a model and the counts do not add up". The
mitigation is not in this ticket — it is that **T09 must make the suite's provenance and its
four inclusion criteria airtight**, so the acknowledgement reads as disclosure rather than as
explanation. Flagged to T09, no action here.

#### 7.10 T11 handoff, executed

T11 completed mid-run and handed over §7.5. All four items applied in this lane.

**Item 1 (required).** `results.tex` cited "Table~8 in the appendices" for the $k$-stratified
overhead table. T05's two new supplementary tables shifted everything by +2, so Table 8 is now
the per-problem UDFS table. **Verified from the rebuilt annotated `supplementary.pdf`, not the
source**: `tab:k_range_overhead` → **Table 10**, Table 8 → per-problem UDFS. Took T11's
*preferred* number-free form rather than `8 → 10`, since the number is what keeps breaking:
now "the $k$-stratified overhead table in the appendices". Confirmed in the rebuilt
`main.pdf`. **The R2.4 answer's claim that this citation was updated is now true.**

**Items 2–4 (optional AC-5 hardening), all applied.** Six live "Table~N of Appendix~C"
references in `methodology.tex` replaced with descriptive forms (S2D / D2S / fast canonical
pseudocode in Appendix~C). Two of the six were wrapped across source lines and are not
findable by a single-line grep — the same trap that hid M6 from T11's first sweep.

**Result: the rendered main paper now contains no cross-document table number at all.** The
only "Table N" strings left in `main.pdf` are Tables 1–3, its own, all reached by `\ref`.
Left alone per T11: the four "Please see Appendix~A" stubs, the Appendix D.1/D.2/E.1/E.2
references and `results.tex:146`, which was already number-free.

**Supplementary length.** T11 reports 13 pages. Confirmed independently (`pdfinfo` → 13).
**Nothing in `paper/` states the supplementary's length**, so no edit was needed on this side.

#### 7.11 Conflict rule — no ticket/code divergence found

Every mechanism restated in the introduction and §3.4 was checked against the implementation
before it was written: the twelve-label alphabet and the two decomposition identities against
`experiments/models/commutative_encoding.py` (2026-07-30); the default `mode="wl_only"` key
$(\ell, h)$, the greedy-with-backtracking-on-ties structure and the near-$O(k^2)$ bound
against `fast_canonical_string` in `src/isalsr/core/canonical.py`. All agree with the
manuscript. `node_types.py` still carries `-`→SUB and `/`→DIV, which is **not** a divergence:
T16 §5.2 requires the core registry to keep them so S2D can decode legacy strings while the
adapters decompose. No unresolved point, and nothing written here depends on one.

The only ticket-versus-source divergence found is documentary, not semantic: the
`introduction.tex` claim in §3.2 / D6, corrected in §7.2.

#### 7.12 Per-file sign-off checklist

Review the diff, do not re-do the work: `diff -u article/paper/<f> reviews/internal_copy_reviewed_article/paper/<f>`.

| File | Changed by this ticket | Reviewer | Signed |
|---|---|---|---|
| `main.tex` | abstract `:81` (duplication, double space, orthography); `\section*{Acknowledgments}` `:123` | Ezequiel | ☐ |
| `introduction.tex` | new positioning paragraph `:28–45`; IsalGraph/IsalChem paragraph rewritten `:55–68`; contributions 2–5 sharpened `:77–95`; middle-ground paragraph `:99–107`; roadmap de-staled `:115` | Ezequiel | ☐ |
| `related_work.tex` | 4 macro sites; 8 spelling sites | Ezequiel | ☐ |
| `methodology.tex` | 5 macro sites; 7 spelling sites; worked example `:858–876`; grammar `:4`; 7 pseudocode references made number-free; `Section~5` → `\ref{sec:results}` `:1328` | Ezequiel | ☐ |
| `computational_experiments.tex` | companion URL `:4` (E7); math-mode name `:194`; 8 spelling sites | Mario | ☐ |
| `results.tex` | k-stratified table reference `:176` (T11 M6); 11 spelling sites | Mario | ☐ |
| `discussion.tex` | 11 spelling sites | Mario | ☐ |
| `conclusion.tex` | 1 spelling site | Ezequiel | ☐ |
| `reviews/response_to_reviewers.tex` | R1.5, R1 C3/C5, R2.8, R2 C1, R3.2 | Mario | ☐ |
| `scripts/check_manuscript_style.sh`, `scripts/manuscript_style_wordlist.txt` | new | Mario | ☐ |

#### 7.13 Verification results

| Gate | Result |
|---|---|
| Letter, 2 × `pdflatex -halt-on-error` | exit **0**, **0** |
| Letter, Overfull | **0** |
| Letter, unresolved `??` | **none** |
| Letter, new `LaTeX Warning` | **none** |
| Annotated paper, 3 × `pdflatex` | `^!` **0**, undefined refs **0**, undefined citations **0**, Overfull **0** |
| Numbering preserved | Definition **3.2**, Theorem **3.13**, Lemma **3.14**, Theorem **3.15** — all unchanged; T07's additions are 3.16/3.17/3.18, appended after 3.15 |
| `color{red}` in annotated sources | **0** |
| `[MPG …` inline notes | **0** |
| `rcomment` bodies touched | **none** (diff-hunk overlap test against the post-edit line map) |
| `article/` beyond the recorded baseline | **unchanged** |
| Style guard on this lane | **exit 0**, 8 files |
| Guard regression test | **exit 1** on a seeded probe |
| Clean `article/paper/main.pdf` colour (AC-8) | **none** — the only `\color` in the clean sources is `methodology.tex:401–402`, at comment-nesting depth 1, so it is never rendered. Colour operators in the PDF binary come solely from the four colour figures. **12 pages.** |

---

## 8. Proposed answer

### 8.1 Before / after

All figures measured over the whole annotated `paper/` tree by
`scripts/check_manuscript_style.sh` unless stated otherwise.

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Abstract, "on both methods on both methods" | present, `main.tex:81` | corrected to "on both methods", blue | AC-1 |
| Abstract, double space | present, unreported; **source-only — \LaTeX{} collapsed it, so the typeset abstract never showed it** | removed | AC-1 |
| Abstract propagated to `double_blind/` | same defect, `main_anonymous.tex:71` | **pending** (out of lane, §7.7) | AC-1 |
| Abstract propagated to statement | **no typo there**; only the headline numbers are restated, `:99–115` | **pending** (numbers only, §7.7) | AC-1 |
| Rendered name form | ISALSR **and** IsalSR | ISALSR uniformly | AC-3 |
| Occurrences bypassing `\IsalSR` | **10** (related work 4, methodology 5, experiments 1; **introduction 0**, contra §3.2/D6) | **0** | AC-3 |
| British-variant occurrences, whole family | **44** | **0** | AC-4 |
| — of which `canonicalis` | 27 | 0 | AC-4 |
| — `optimis` / `normalis` / `neighbour` / `behaviour` / `colour` / `standardis` / `labelled` / `summaris` / `parameteris` | 6 / 4 / 3 / 2 / 2 / 2 / 1 / 1 / 1 | 0 | AC-4 |
| Automated spelling + naming check | none | `scripts/check_manuscript_style.sh` (+ 45-rule word list); **exit 0** on this lane, **exit 1** on a seeded regression | AC-4 |
| R2 answer to C1 | **No** | abstract defect fixed and suite description current; **effect figures deferred** to the campaign | AC-2 |
| Introduction (R1-C3) | "Could be improved"; 74 lines | revised against R1's B2 on four counts; **119 lines (longer, per instruction)** | AC-5 |
| Readability (C5) | "requires some effort" ×2 (R3: "easy to read") | worked example added to §3.4, which had neither figure nor instance; 7 cross-document numbers removed | AC-6 |
| Cross-document table numbers in `paper/` | 7 (6 × "Table~N of Appendix~C" + "Table~8 in the appendices") | **0** | AC-6 / T11 |
| Author bios and photos | absent from PDF | present and rendering, pages 14–15 | AC-7 |
| Colour / highlighting in clean `article/paper/main.pdf` | — | **none**; sole `\color` sits inside a `comment` block | AC-8 |
| Annotated paper length | 14 pages (before this pass) | **15 pages** | §7.8 |
| Overfull boxes, annotated paper | 0 (clean baseline) | **0** | §7.13 |

### 8.2 Changes made to the manuscript

All in `reviews/internal_copy_reviewed_article/paper/`; `article/` untouched. Line numbers
are post-edit.

| File | Lines (revised) | Change |
|---|---|---|
| `main.tex` | 81 | abstract: duplicated phrase removed (blue), double space removed, `canonicalisation`/`Canonicalisation` → `-ization` |
| `main.tex` | 123 | `\section*{Acknowledgements}` → `Acknowledgments` (IEEE house form) |
| `introduction.tex` | 28–41 | **new**: positioning against diversity preservation and equality saturation, and why neither removes node-ordering redundancy |
| `introduction.tex` | 55–69 | IsalGraph/IsalChem paragraph rewritten; ungrammatical "Other related model is IsalChem" removed; the three properties that make the labeled directed setting hard now stated |
| `introduction.tex` | 77–95 | contributions 2–5: expressiveness preserved by the commutative encoding; 1-WL named as the guide; reachability condition named; suite description de-staled |
| `introduction.tex` | 99–107 | **new**: exhaustive-versus-hashing framing, with the sound-but-incomplete characterisation of a fixed-order hash |
| `introduction.tex` | 115 | roadmap de-staled ("the Nguyen and AI Feynman benchmark suites" → "the benchmark suite") |
| `related_work.tex` | 23, 50, 77, 111 | 4 name occurrences routed through the macro |
| `related_work.tex` | 17, 45, 66, 69, 91, 103, 107, 111 | 8 orthography sites |
| `methodology.tex` | 4 | "prove that such string is" → "this string" |
| `methodology.tex` | 8, 75, 107, 189, 491 | 5 name occurrences routed through the macro (subsection title, table caption, definition title, prose, pseudocode row) |
| `methodology.tex` | 190, 206, 439, 698, 708, 802, 880 | 7 pseudocode references made number-free (T11 items 2–4) |
| `methodology.tex` | 858–876 | **new**: worked example on $\sin(x_1)+\cos(x_1)$, unnumbered so as not to renumber Definition 3.9 / Theorems 3.13–3.15 |
| `methodology.tex` | 807, 1251, 1257, 1261, 1279, 1328, 1340 | orthography sites (incl. two inside the parallel lane's blue text) |
| `methodology.tex` | 1328 | hardcoded `Section~5` → `Section~\ref{sec:results}` |
| `computational_experiments.tex` | 4 | companion URL: anonymised mirror → `https://mariopasc.github.io/IsalSR/` (E7, blue) |
| `computational_experiments.tex` | 194 | math-mode `^{\mathrm{IsalSR}}` → `^{\text{\IsalSR}}` |
| `computational_experiments.tex` | 3, 25, 26, 30, 34, 176, 181, 191, 309 | 9 orthography sites |
| `results.tex` | 176–177 | `Table~8 in the appendices` → "the $k$-stratified overhead table in the appendices" (T11 item 1, blue) |
| `results.tex` | 6, 7, 24, 46, 48, 121, 167, 169, 193, 195, 202 | 11 orthography sites |
| `discussion.tex` | 7, 15, 22, 23, 59, 65, 68, 70, 76, 104, 114, 115 | 12 orthography sites |
| `conclusion.tex` | 9 | 1 orthography site |

**Blue-marking policy, stated because it is a judgement call.** Blue marks the changes a
reviewer will look for: the abstract fix, the URL fix, the introduction revision, the worked
example and the two cross-reference corrections. The global orthography sweep and the macro
routing are **not** individually blue-wrapped: 54 inline colour spans would have buried the
substantive marks that the other tickets are landing in the same document, and R2.8(b)/(c)
are verified by a single search of the revised PDF rather than by hunting marks. Every such
change is enumerated above and reproducible with `diff -u` against `article/paper/`.
Annotated tree after the pass: **50 blue spans, 0 red, 0 inline review notes.**

### 8.3 Response text as shipped

**Already written into `reviews/response_to_reviewers.tex`** — this is not a draft for T14 to
paste, it is a record of what is in the file. Five blocks, all `\todoblock` placeholders
removed. Compiled and page-checked (§7.13).

| Block | Section heading in the letter | Letter page |
|---|---|---|
| R1.5 | "R1.5 — Writing and readability" | 11 |
| R1 C3/C5 | the unlabelled `response` following R1.5 | 12 |
| R2.8 | "R2.8 — Abstract duplication and inconsistent naming" | 21–22 |
| R2 C1 | first `response` under "R2 — structured answers requiring a reply" | 22–23 |
| R3.2 | "R3.2 — Abstract typo" | 26 |

**R2.8** opens "The reviewer is correct on all three counts, and all three are corrected",
then takes the three in order. The abstract paragraph states the correction, discloses the
double space **and states that \LaTeX{} collapsed it so it never appeared in the typeset
abstract** (the honest version; claiming a rendered defect would have been false), and names
the two documents carrying the same text. The naming paragraph puts the defect at the usage
sites, gives **ten** occurrences and their locations, and states that the only literals left
are the macro definition and the project URL. The spelling paragraph gives the variant, the
three grounds, the eight further word families swept, and the **44 → 0 / 10 → 0** counts. The
closing paragraph describes the guard and volunteers that it caught two regressions this
revision introduced itself.

**R3.2** is five sentences: confirmation in the reviewer's own words, brief thanks for giving
the substitution verbatim, the double space, and a cross-reference to R2.8 for the naming and
spelling half rather than a repetition of it.

**R1.5** refuses to assert that the paper now reads well and says what the pass covered
instead. It states in its second sentence that **R3 rated the manuscript "easy to read"**, so
the complaint is 2 of 3, and explains that the methodology was therefore treated as the place
the two readings diverge. The worked example is given with both greedy strings and the
canonical one. The material added to §3.4/§3.6 under R2.1 is **cross-referenced, not
claimed**, so the letter does not present one change as two.

**R1 C3/C5** uses R1's own B2 statement as the specification and walks the four things B2 says
that the submitted introduction did not. It corrects R1 on one point: their "hash-based
approximations that offer no correctness guarantee" is too strong, and the answer states the
sound-but-incomplete characterisation instead. It closes by volunteering that the
introduction is **longer**, which matters because R2 asked for trimming.

**R2 C1** states that the title and the seven index terms are unchanged *after
reconsideration*, gives the three abstract changes, and then discloses in its own voice that
the effect figures are still the submitted campaign's and are being regenerated — with the
reason no placeholder was written. This is the block most exposed to §5.3 of the
`review-answer` skill, and it is written to fail no diff a round-2 reviewer could run.

### 8.4 Residual risk

> Candidates: content landing after this pass and reintroducing the inconsistencies
> (hence the T13 dependency and T14's final check); the abstract's headline numbers
> drifting from the final tables; bios and photos pushing past 12 pages.
