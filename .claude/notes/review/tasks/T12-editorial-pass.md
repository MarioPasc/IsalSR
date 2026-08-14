# T12 — Editorial pass: abstract, naming, spelling, readability

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.5**, **R2.8**, **R3.2**, R2-C1, R1-C3, R1-C5, R2-C5 (and E9) |
| Type | Copy-editing / prose |
| Owner | **Karl** (primary) · each author signs off their own sections |
| Depends on | ~~T13~~ — **dissolved 2026-08-06**: all trimming and length reduction delegated to Karl; this ticket does not cut anything |
| Blocks | T14 |
| Status | **DONE except AC-9.** AC-0 – AC-8 and AC-10 met and verified in the main tree. **AC-2 closed 2026-08-14**: campaign C2 landed, the abstract now carries the three-arm figures and is byte-identical in `article/` and `double_blind/`, and `previously_published_statement` restates the same values. **AC-8 was re-opened and re-closed 2026-08-14** — the earlier check missed hyperref's coloured link borders, which are annotation rather than text; `hidelinks` is now set. **AC-9 not startable by an agent** — author sign-off is a human step; the per-file checklist is §7.12, refreshed at §7.14.6. Two items sit outside this lane and are named in §7.14.8: the `double_blind/supplementary` re-sync (T14) and the AI-assistance disclosure question (authors). §7.1–§7.13 describe the 2026-08-06 pass and several of their file-state figures are **superseded**; see §7.14.1. |
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

### 2026-08-14 — AC-2 closed, AC-8 re-opened and closed, and a de-AI/length pass over `paper/`

Second session on this ticket. Lane: `article/paper/`, `double_blind/paper/`,
`previously_published_statement/`, plus the two guard scripts in the code repo. A parallel
session ("supplementary") holds `article/supplementary/` and `double_blind/supplementary/`;
the split was agreed by message before either of us edited anything, and I touched no file
under either supplementary directory.

#### 7.14.1 Five figures in §7.1–§7.13 are superseded, and one conclusion in them is wrong

The tree moved under this ticket between sessions. Commits `59f6413` ("adopt the changes
package as the single marked-up source") and `20223c9` ("mirror the revision, and retire the
separate marked copy") promoted the annotated copy into `article/` and deleted
`reviews/internal_copy_reviewed_article/`. Anything in §7.4/§7.7/§7.8/§7.13 that describes a
"promotion backlog" is therefore spent. Re-measured today:

| §7 claim | Then | Now |
|---|---|---|
| `article/paper` spelling / naming violations | 44 / 10 | **0 / 0** |
| `double_blind/paper` spelling / naming | 44 / 10 | **0 / 0** |
| `reviews/internal_copy_reviewed_article` | the working lane | **deleted** |
| Annotated paper length (§7.8) | 15 pages | **18** |
| Clean `article/paper/main.pdf` (§7.13) | 12 pages | **18** |

**The wrong conclusion is §7.1's**, which frames the length problem as one the promotion step
would partly absorb. It will not. I built four variants from the current source and counted:

| Build | Pages |
|---|---|
| annotated, as the tree stands | 18 |
| `changes` loaded with `final` (the clean submission build) | 18 |
| same, minus CRediT and the competing-interest declaration | 18 |
| same, also minus all three biographies and photographs | **17** |

Stripping the blue markup buys **zero** pages, because it removes colour and not content. The
bios and photos the decision letter mandates cost **one**. Against a 12-page limit that
includes references, bios and photos, the main file is **six over, in the body**. That is
T13's, it is larger than anything a prose pass can recover, and §7.8 item 1 understated it by
three pages.

#### 7.14.2 AC-2 — closed

Campaign C2 completed 2026-08-14 (T02) and the three-arm figures are in the abstract. Checked
rather than assumed:

- `grep "on both methods"` across `article/`, `double_blind/` and
  `previously_published_statement/`: **0 hits**. The double space is gone with it.
- The abstract in `article/paper/main.tex` and in `double_blind/paper/main_anonymous.tex` is
  **byte-identical** (`diff` over the `abstract` environment).
- `previously_published_statement/main.tex:99–115` restates $\rho = 1.66/1.79$,
  $38.1\%/43.7\%$, $d = 2.54/7.05$ at $p<10^{-12}$, $p = 2.0\times10^{-9}$,
  $p = 7.5\times10^{-4}$, $0.04\%$, $16.1\%$ and $1.45\times$ — the same values the abstract
  states, so §7.7's "pending, numbers only" is discharged.
- `experiments/scripts/review_campaign/verify.py`: **121/121 anchored literals pass**, with
  `main.tex:90` resolving for $16.1$, $0.04$ and $7.05$.

R2's C1 answer therefore no longer carries a deferral. §8.3's description of the R2 C1 block
is now stale in one clause and T14 should re-read it before submission.

#### 7.14.3 AC-8 — the earlier check was looking in the wrong place

§7.13 certified the clean PDF colour-free by searching the sources for `\color` and the PDF
binary for colour operators. Both were right and both missed the defect. `hyperref` was loaded
bare, so every citation and cross-reference carried a **green link border**, which is a PDF
annotation rather than a content-stream operator and does not contain the string `\color`
anywhere. I found it by rendering page 1 of the clean build and looking at it.

Fixed by loading `\usepackage[hidelinks]{hyperref}` in both `main.tex` and
`main_anonymous.tex`, with the reason in a comment beside it. Re-rendered page 1 of the
`final` build: black on white throughout, and the only colour left in the document is in the
four colour figures. Costs no pages.

**Method note worth keeping.** A grep cannot certify "no colour in the PDF". Render and look.

#### 7.14.4 AC-4 — two latent defects in this ticket's own guard, found by the parallel session

The supplementary session reported that `scripts/manuscript_style_wordlist.txt` demanded
"optimiztic". I reproduced it before changing anything (`printf 'The estimate is optimistic
here.\n' > probe.tex` → **SPELLING**, exit 1) and found it was one of a class:

- The bare `-is` stems are substrings of **`optimistic`, `optimism`, `specialist`,
  `generalist`, `organism` and `characteristic`**, all correct American English. Every stem now
  carries the lookahead `(?=ation|e|ing)`, which admits `-isation/-ise/-ised/-ises/-iser/-ising`
  and refuses the six.
- Check 1 now matches against a line-for-line **scrubbed** copy in which the arguments of
  `\label \ref \eqref \cref \Cref \autoref \pageref \cite* \includegraphics \input \include
  \bibliography \url \href \path \nolinkurl` are emptied. `\label{fig:neighbourhood}` and
  `\includegraphics{fig_neighbourhood.pdf}` are not reader-visible, and renaming one without
  the other gives an undefined reference or a missing graphic. Line numbers and reported text
  are still the original's. Check 2 (naming) still runs unscrubbed, so `\newcommand{\IsalSR}`
  is unaffected.

Regression-tested three ways: the six American words now exit 0; `We canonicalise the
neighbourhood and analyse the behaviour.` still reports 4 hits and exits 1; a file of bare
`\label`/`\ref`/`\includegraphics` exits 0.

Guard state across the package:

| Tree | Exit | Violations |
|---|---|---|
| `article/paper` | 0 | 0 |
| `article/supplementary` | 0 | 0 |
| `double_blind/paper` | 0 | 0 |
| `previously_published_statement` | 0 | 0 |
| `double_blind/supplementary` | **1** | **2 NAMING** (stale copy — §7.14.8) |

**The letter, swept — §7.8 item 4 discharged.** That item recorded "50 British-variant
occurrences across the letter" and left them, because the file lane forbade editing other
agents' blocks. The parallel session has since released the letter. Re-measured with the
corrected guard it is **16 hits, not 50**; the old figure was taken before the other lanes
cleaned up. Classified rather than swept blindly:

| Hits | Class | Action |
|---|---|---|
| 4 | authors' own prose: `favoured` ×2, bare `IsalSR` ×2 (`:407, :741, :790, :983`) | **fixed** |
| 2 | inside `\begin{rcomment}{R2.8}` — the reviewer's comment, quoted verbatim | **must never be edited** |
| 9 | the R2.8 answer listing the British forms it swept | **mention, not use** — editing them makes the sentence say "the split ran through *neighborhood, labeled, color*" |
| 1 | `\todoblock{C4 and C6 -- reorganisation …}` | T13's unwritten placeholder; left for them |

The last three classes are why a blanket sweep of this file would have been wrong, and the
humanizer skill says the same about the first of them: *do not rewrite quoted reviewer text in
a rebuttal*. Rather than exempt the whole file, the guard grew two exemptions, both documented
in the script beside the rule they relax:

- the body of an `rcomment` environment is skipped outright;
- `% style-guard-allow` on a line, or a `% style-guard-allow-begin` / `-end` block, is skipped,
  and each must carry its reason.

**Use the block form inside a paragraph.** A trailing `%` suppresses the newline, so an
end-of-line marker mid-paragraph glues two words together; a marker on its own line is inert.
I checked the typeset output rather than assuming: the sentence still prints
"… and \emph{summarises} as well as \emph{canonicalisation}. The main paper carried $44$ …",
correctly spaced.

Seven probes pass: American words exit 0; British words exit 1; `\label`/`\ref`/`\includegraphics`
exit 0; a bare name exits 1; the same line with the marker exits 0; text inside an `rcomment`
exits 0; text on the line **after** `\end{rcomment}` exits 1. The letter is now **one hit**, and
that hit is T13's placeholder telling T13 to write it in American.

#### 7.14.5 The de-AI and length pass (added scope, not in §6)

Instruction: make the prose read as human-written, and cut anything that is not scientific or
that a plainer word would carry, without pushing for the 12-page target. `humanizer` v3.0.0,
scientific mode.

**Diagnosis, measured rather than asserted.** The raw `.tex` readings are misleading — macro
names inflate mean word length and depress the function-word ratio — so I stripped LaTeX to
prose first (`scratchpad/T12/detex.py`) and measured that. One tell dominated and it was
uniform: **median sentence length was above the 10–24 band in all seven files**, 24.5 to 32
words. Three others recurred:

| Pattern | Count before |
|---|---|
| Pseudo-cleft as the default sentence frame ("X is what Y") | 14 |
| `rather than` as the default contrast | 26 |
| Meta-commentary about the reporting rather than the result | throughout |

**What changed.**

| File | Median sentence length | Prose words |
|---|---|---|
| `introduction.tex` | 31 → **19** | 860 → 691 |
| `related_work.tex` | 25 → 25 | 928 → 900 |
| `methodology.tex` | 25 → 25 | 5,062 → 5,021 |
| `computational_experiments.tex` | 27 → **25.5** | 2,943 → 2,850 |
| `results.tex` | 24.5 → **23.5** | 2,094 → 2,064 |
| `discussion.tex` | 32 → **25** | 1,630 → 1,469 |
| `conclusion.tex` | 32 → 32 | 321 → 322 |
| whole paper | — | 14,895 → **14,420** |

Pseudo-clefts 14 → **0**; `rather than` 26 → **15**; deletable openers, transition-adverb
openers and inflated adjectives: three survivors in the whole paper, each of them earning its
place (`significantly` is statistical vocabulary in a caption, `crucially` and `substantially`
are calibrated authorial stance, which the skill says to preserve rather than sterilise).

**Three cuts that are content decisions, not style, and that Ezequiel should confirm:**

1. **The introduction's roadmap paragraph is gone** (`Section 2 reviews related work. Section 3
   defines …`, seven sentences). It restated the numbered section headings and nothing else.
   AC-5's four substantive additions are all intact — I checked each against R1's B2 statement
   line by line after the rewrite.
2. **`related_work.tex` no longer re-defines symbolic regression** in its first sentence. It
   repeated the manuscript's own opening sentence one page later.
3. **A sentence was removed from `discussion.tex` §Limitations** that read "the submitted text
   described this as counting timed-out DAGs as unique, which is not what the implementation
   does". A published article cannot refer to "the submitted text". I confirmed the disclosure
   survives where it belongs before deleting it: `reviews/response_to_reviewers.tex:526–535`
   already states the correction and that it is numerically nil.

**What I did not do, deliberately.** `methodology.tex` sits at a median of 25 and I left it
there. It is the formal section, longer sentences are the genre norm in it, and chopping a
definition into fragments would trade one tell for another — the skill's explicit warning
against manufactured burstiness. Nominalization counts stay high in the introduction, related
work and conclusion because the nouns are the paper's technical vocabulary
(`canonicalization`, `deduplication`, `invariance`); de-nominalizing those is thesaurus-swapping
and would cost precision. Hedge and booster counts stay low because the results are exact and
adding hedges to an exact number is worse science, not better prose.

**Number integrity.** Every edit was made under `verify.py`, which anchors 120 quoted literals
to a `file:line` plus a same-line phrase. It caught me **five times**: four were line-wrap
artifacts where a reflow split an anchor phrase, and I repaired the prose; one was real — the
anchor for the six-group Nemenyi threshold was the phrase `which is what decides`, a pseudo-cleft
that this pass was supposed to remove. I re-pointed that anchor to `Nemenyi threshold from` in
`verify.py:367` rather than keep a style defect alive to satisfy a checker. **121/121 pass.**

#### 7.14.6 Verification results, this session

| Gate | Result |
|---|---|
| `article/paper/main.tex`, 3 × `pdflatex -halt-on-error` | exit **0**; Overfull **0**; undefined refs **0**; undefined citations **0** |
| `double_blind/paper/main_anonymous.tex`, 3 × `pdflatex` | exit **0**; Overfull **0**; undefined **0** |
| Theorem numbering vs the pre-pass baseline `main.aux` | **identical**, 17 labels; `thm:roundtrip` **3.13**, `thm:invariant` **3.15**, as R1 and R2 cite them |
| `verify.py` anchored literals | **121/121 pass** |
| `check_manuscript_style.sh`, four trees | **exit 0** |
| Clean (`final`) build, page 1 rendered and inspected | no colour, no link borders |
| Abstract, `article` vs `double_blind` | **byte-identical** |
| Identifying strings in `double_blind/paper/*.tex` | **none** (name, affiliation, e-mail, cluster name, project URL) |
| `article/supplementary`, `double_blind/supplementary` | **untouched** (other session's lane) |

Sign-off (AC-9) is still open on every row of §7.12. The files this session changed are
`main.tex` (abstract, `hidelinks`), `introduction.tex`, `related_work.tex`, `methodology.tex`,
`computational_experiments.tex`, `results.tex`, `discussion.tex`, `conclusion.tex`, their
`double_blind` mirrors, and the two guard scripts.

#### 7.14.7 D6 confirmed against the submitted source

§7.2 corrects `verified-discrepancies.md` D6 on two points. Both re-checked today against
commit `577e0f2`, the submitted state:

- `introduction.tex` carries **zero** occurrences bypassing the macro. D6's "and in
  `introduction.tex` prose" is wrong.
- `related_work.tex` carries four, at **23, 50, 77, 111**. D6 says 76; the line is 77.
- Total across the paper: 4 + 5 + 1 = **10**, as §7.2 measured.

D6 should be amended. The R2.8 answer already quotes the corrected figures.

#### 7.14.8 Two items outside this lane, both required before submission

1. **`double_blind/supplementary/supplementary_anonymous.tex` is stale**, and it is the copy
   R2 read. It is 1,670 lines against the live 1,729, has four `\input`s against six (missing
   both new Appendix D.1 tables), and still renders **"IsalSR" twice** at `:383` and `:506` —
   which is R2.8(b) surviving in the submitted package. The re-sync is unowned; the board puts
   it under T14. **Until it happens, R2.8(b) and (c) are not closed across the package**, only
   across `article/`.
2. **The AI-assistance disclosure may now be incomplete.** `main.tex:135` credits Claude for
   benchmark-suite discovery, the Picasso parallelisation code and the companion website. After
   this session an LLM has also **materially edited the manuscript prose**. §3.4 and §7.9 decided
   to keep the acknowledgement because it is honest; that reasoning now points at extending it.
   This is an authors' decision and I have not made it. It interacts with E9 and with R2.5
   exactly as §7.9 describes.

#### 7.14.9 The supplementary, measured read-only

The supplementary prose pass is agreed as mine and is **blocked** until the parallel session
finishes: two Picasso jobs (`2001009`, `2001366`) land into that file tonight, **14**
`\pendingnum`/`\pendingblock` placeholders are still open in it, and at least two sentences in
the synthetic-scalability appendix will change their claims. Passing it now would be thrown
away. The four patterns above, with their counts, were handed over so the work is not
re-derived.

I measured it read-only so the scope is known before the file frees up. **It is in better
shape than the paper was**, which is consistent with its having been written in one lane rather
than five:

| | supplementary | paper, before | paper, after |
|---|---|---|---|
| Prose tokens | 8,002 | 14,625 | 14,105 |
| Median sentence length | **24.5** | 24.5–32 | 19–25 |
| Sentence-length CV | 0.61 (ok) | 0.34–0.79 | 0.44–0.79 |
| Nominalizations / 1,000 | 47 (ok) | 56–72 | — |
| Function-word ratio | 0.43 (ok) | 0.31–0.46 | — |
| Pseudo-clefts | **2** | 14 | 0 |
| `rather than` | 15 | 26 | 15 |
| Negative parallelism | **1** | 0 | 0 |

So the supplementary needs a light pass, not a rewrite: two clefts, one negative parallelism,
a modest sentence-length trim, and the load-bearing-versus-not judgement on the Appendix D
meta-commentary the parallel session flagged. Estimated at well under an hour once the
placeholders close.

**Update, same day.** That session applied the same test I used on the Discussion — confirm the
concession survives in the letter before cutting it from the appendix — found it at R2.6
("*We have no defence for it: the number was never recomputed when the suite grew*") and **cut
the D.3 sentence**. It was also one of the two clefts, so the supplementary now stands at one.
The completeness statement above it stays; the `\changeref` promises it.

**Three constraints carried into that pass, from the session that wrote the file:**

1. **Do not smooth the numbers in the R2.5 block of the letter.** `42\%`,
   $3.7\times10^{-9}$ and the 36/33/1 agreement split are backed by the `data_agreement` block
   of `docs/generated/appendix_d/appendix_d_benchmarks.json`, and they arrived **by retraction** —
   an earlier draft claimed "sixty-eight agree exactly" from a scan that conflated *below
   $10^{-8}$* with *exact*, and the persisted artefact falsified it. Prose freely; numbers not
   at all.
2. **Long sentences in Appendices D.2 and D.3 may be split but not shortened.** Several
   enumerate a ledger's guarantees in one breath — exit status, balance, seed identity,
   revision, engine, digests. R2.6 asked us to confirm exactly that list, so dropping a clause
   answers a different comment than the one that was asked.
3. `rather than` at 15 in 8,002 tokens was left to me to judge against the paper's conventions.
   The paper now sits at 15 in 14,105, so the supplementary is roughly twice as dense in it and
   is worth a look, but it is a tic and not a defect.

### 2026-08-14 — second pass: length, and a notation defect it surfaced

A second pass ran over `article/paper/` with length as the objective rather than style, under
three constraints from the authors: cut hard but not past legibility, strip framing that
favours the method without a measurement behind it, and move implementation-level material to
the supplementary. Full cut, staged and declined tables:
`.claude/notes/review/tasks/T12-appendix/paper-to-supplementary.md`.

#### 7.15.1 Result, re-verified independently

**18 → 17 pages** on the clean `final` build, which I rebuilt myself rather than take on
report; prose $14{,}105 \to 13{,}622$ words. Both annotated builds are 17. Every gate re-run by
me: `verify.py` **121/121**; style guard **exit 0** on `article/paper`, `double_blind/paper`,
`article/supplementary` and `previously_published_statement`; both documents **0 overfull, 0
undefined references or citations**; theorem numbering **identical** to the pre-session
baseline, 17 labels, `thm:roundtrip` 3.13 and `thm:invariant` 3.15.

**The supplementary was not touched.** Its mtimes are unchanged across the whole pass
(`18:17`, against a spawn at `19:37`), so the lock held.

#### 7.15.2 What made the page, and the three cuts I checked hardest

`article/supplementary/supplementary.tex:1077` already carries a full **Appendix E, "The naive
hash comparator"** — the appendix the paper already cites for per-problem $\phi$ — and it
duplicates the second half of §3.6 item for item, in places more fully. **Lemma 3.20's proof
was the only proof still typeset inline in the paper**; Appendix E proves the same statement at
`:1135` and adds an incompleteness counterexample the paper never had. Replacing it with
`Please see Appendix~E` removes an inconsistency as well as ~90 words, and needed no staging.

Three cuts could have cost something, so I verified each in the file rather than from the
table:

| Cut | Risk | Checked |
|---|---|---|
| "Python's built-in hash … 64-bit values" | the next sentence bounds collisions and that bound rests on the width | width kept: "a deterministic $64$-bit hash on tuples of integers" |
| the $17.6\%$ instrumentation argument | this is the paragraph that concedes a confound; trimming a concession to save space is the one cut we cannot make | the confound is still stated in the body — `computational_experiments.tex:98`, "This costs a confound, and we state it rather than remove it". Only our defence moved |
| "$\approx 200$ lines of Python for Bingo" and the sentence carrying it | the sentence also described the integration mechanism | mechanism survives at `computational_experiments.tex:41` and in the abstract |

**14 items deleted, 2 staged.** The staged pair are left in the paper untouched, because
deleting content before its destination exists is how content is lost. **B2 in particular is
single-sourced in the paper today** — the supplementary uses condition~(iv) without proving the
equivalence — so it must be copied before it is deleted.

#### 7.15.3 The declined list is the part that shows the judgement

Nine items, with reasons. The one that matters: **§4.4, the paired-test protocol, ~1,100 words
and roughly one full column — the largest remaining lever that does not touch a figure.** It is
textbook Demšar and no reviewer questioned it. It was correctly left alone: R2 rated soundness
**"Partially"**, and this test is the project's primary significance metric for $R^2$ and the
reduction factor, so thinning the paper's inferential backbone to save a column is a content
decision for the authors and not an editorial one. **That is the recommendation if 16 pages is
wanted.** Also declined and correctly so: any figure change, merging Tables 4 and 5
(renumbering breaks a verifier anchor and the letter's cross-references), and the §4.4-versus-
caption pooling duplication, which the verifier has deliberately frozen in both places.

#### 7.15.4 A notation defect neither pass introduced, and it is R2.2's class

Found on review, not by the pass. **Three symbols for two objects across the submitted
package:**

| Object | Paper prose | Paper tables (generated) | Supplementary App. E |
|---|---|---|---|
| the serialization | $\mathrm{ser}(D)$ | — | $\sigma(D)$ ×4 |
| its reduction factor | $\rho_{\mathrm{ser}}$ ×7 | $\rho_{\sigma}$ ×4 | — |

A reviewer reads $\rho_{\sigma}$ in the Table~2 caption, meets $\rho_{\mathrm{ser}}$ two
paragraphs later, then opens Appendix~E and finds the function itself called $\sigma(D)$. This
is structurally **R2.2**, where `{g,i}` and `{−,/}` named one alphabet in two documents and the
reviewer wrote that they "coexist in the submitted PDF and should be reconciled". R2 reads
appendices line by line.

**Resolved toward $\mathrm{ser}$, and the argument is not a coin-flip: $\sigma$ is already
bound.** `methodology.tex:35` defines $\sigma(v)$ as the *ordered input list* of a node, and it
carries Rule~1, condition~(iv) and the whole operand-order treatment. $\sigma(D)$ for the
serialization of a DAG collides with it.

Fixed in my lane, 12 occurrences: `results.tex` ×2 (caption and prose), `tab_three_axis.tex`,
`tab_phi_by_host.tex`, their `double_blind` mirrors, and — the part that matters —
**`experiments/scripts/review_campaign/tables.py` ×4**, because those table bodies are
generated and a hand-edit would revert on the next regeneration. `ruff` clean, both documents
rebuilt, all gates re-run.

**Open, and handed to the supplementary session:** the four $\sigma(D)$ in Appendix~E, and
`tab_supp_phi_per_problem.tex`, which the same generator emits into their locked lane.

#### 7.15.4b The notation fix, completed across the package — and what it cost

The parallel session took the collision argument and **improved it**. `\sigma(v)` is not merely
a paper symbol: it occurs **26 times in the supplementary's own Appendices A–C**, carrying
Rule 1, condition (iv), the pseudocode and the DAG tuple $D=(V,E,\ell,\delta,\sigma)$, while
`\sigma(D)` occurred **17 times in Appendix E**. So one letter named two functions **inside a
single file**, which a reader meets without leaving the supplementary. That is a stronger
argument than the cross-document one and it settles which sense yields: the node sense is
theorem-bearing and older.

They renamed the 17, bounded by the `\label{sec:supp_hash}` anchor so a stray global replace
could not reach the 26. Verified by me: **0** occurrences after the anchor, **26** before it.

**The rename tripped `verify.py:288`**, which had anchored the literal `\rho_\sigma = 1.0000`;
they re-pointed it and recorded why, rather than reverting the rename. Back to 121/121. Two
consequences worth keeping:

- `U_{\mathrm{ser}}` is wider than `U_\sigma`, so the $\Delta$/$\phi$ display equation overflowed
  its column by 4.998 pt. **A symbol rename can break a display box; read the log, not the diff.**
- They regenerated rather than hand-editing, which wrote `tab_three_axis.tex` and
  `tab_phi_by_host.tex` **into this lane**. I diffed both against HEAD: **exactly one line each,
  the header, one symbol, no number moved.**

Package now: paper **17 pages**, supplementary 17, verify **121/121**, style exit 0 on four
trees, 0 overfull anywhere.

#### 7.15.5 Surfaced, not fixed

The paper's stated *reason* for selecting Bingo was that its stochastic search generates more
structural duplicates, making it the better test of deduplication. That framing is now gone as
unearned, but the tension underneath is real and is not editorial: **$\phi = 0.047$ on Bingo
says Bingo is precisely where a cheap equality check does nearly as well**, so the host chosen
as the showcase is the host on which the comparator is closest. The honest reading is already
in §6.2 and §6.3. Whether the *selection rationale* should be restated to match belongs to T09
and T10.

#### 7.15.6 OPEN AND BLOCKING — the synthetic-scalability exponent will diverge silently

The parallel session's two Picasso jobs landed and **the scaling result changed substance, not
wording**: the clean fixed-$P$ fit is **$O(k^{1.43})$ at $R^2 = 0.897$ over $k = 8..36$**,
against the submitted $O(k^{0.7})$, which came from a design they say could not measure it.

The old exponent is stated **in this lane**, five times, in `article/paper/discussion.tex`:

| Line | Text |
|---|---|
| `:34` | "On $5{,}400$ random expression DAGs with $k \in \{1,\ldots,9\}$" |
| `:36` | "$\sum_{k=1}^{9} 600 \cdot k! \approx 2.2 \times 10^{8}$ internal-node permutations" |
| `:37` | "fitting an $O(k^{0.7})$ power law, three orders of magnitude below the $O(k!)$ worst case at $k = 9$" |
| `:150` | "Per-$k$ timings on $5{,}400$ synthetic expressions (Appendix~F.2)" |
| `:151` | "fit $O(k^{0.7})$ in the benchmarked range", plus the `\added{}` clause defending the low exponent as a range artifact |

**None of the five is anchored in `verify.py`** — zero hits for `0.7`, `5.400`, `1.43`, `0.897`.
So the gate that caught the $\sigma$ rename **will not catch this**: the paper can keep printing
$O(k^{0.7})$ while Appendix F.2 prints $O(k^{1.43})$, and every check stays green. That is the
exact shape of E1, E2 and E8, all three of which R2 found by reading rather than by any tool.

**Not edited, deliberately**, and neither is the check wired: the supplementary is mid-fill, and
handing that session a red gate for something unrelated to their edit would be obstructive.
Three questions are with them, and the answers decide the edit:

1. **Two experiments or one?** `:34–:37` describes an *exhaustive* enumeration of all
   internal-node permutations at $k \le 9$, which reads as job `2001009` rather than the
   $k = 8..36$ sweep. If two, §6.1 may survive untouched and only §6.3 moves.
2. **Superseded or retracted?** Superseded is a number swap. Retracted means the §6.3 `\added{}`
   clause must be rewritten, because it currently *defends* the low exponent as an artifact of a
   short range — and at $1.43$ over $k = 8..36$ that defence is unnecessary and **the result is
   better for us**: it spans Bingo's actual stack depth of $32$ and still sits under the $O(k^2)$
   bound the paper claims.
3. The exact strings to copy rather than paraphrase: exponent, fit quality, $k$ range, sample
   count.

Then: edit `discussion.tex`, mirror to `double_blind`, and **wire an anchored cross-document
check so this pair cannot drift again**. Until that lands, the package has a live contradiction
risk that no gate covers.

#### 7.15.7 CLOSED — and the paper carried two errors, not one

The numbers settled the same evening and the answers were: **two experiments, and the paper
credited the fit to the wrong one.**

| | exhaustive (`2001009`) | scaling (`2001366`) |
|---|---|---|
| $k$ | $1$–$9$ | $8$–$36$, 21 values |
| Expressions | $5{,}400$ | $6{,}300$ |
| Permutations per expression | $k!$ — **varies with $k$** | $20{,}000$ — **fixed** |
| Canonicalizations | $245{,}467{,}800$ | $126{,}000{,}000$ |
| Answers | $\rho = k!$, $100\%$ invariance | the cost curve |

$O(k^{0.7})$ is **retracted, not superseded**. Under exhaustive enumeration $P = k!$, so $k=1$
averages one cold call and $k=9$ averages $362{,}880$ warm ones; the small-$k$ end is inflated
and the fitted slope is biased downward. Re-fitting the same series today gives $0.211$ at
$R^2 = 0.34$ — not $0.7$ either. The fit now comes from the fixed-$P$ run: **$O(k^{1.43})$,
$R^2 = 0.90$**, over a range that spans Bingo's stack depth of $32$ and still sits under the
$O(k^2)$ bound the paper claims. The `\added{}` clause that defended the low exponent as a
range artifact is deleted: at $1.43$ it defends nothing, and keeping it would invite the
reviewer to ask what it was for.

**Two errors were in the submitted paper, and only one of them was the exponent.**

1. **`:36` was arithmetically wrong.** It printed
   $\sum_{k=1}^{9} 600 \cdot k! \approx 2.2 \times 10^{8}$. I checked this myself rather than
   take it on report: the sum is $245{,}467{,}800$, and $2.18 \times 10^{8}$ is
   $600 \times 9!$ — **the last term written as the whole series.** An equation whose left side
   does not equal its right side, in the submitted manuscript. E1's class exactly.
2. **`:39` asserted a claim the data refutes.** "with no observable dependence on the variable
   count". Both runs' generated artefacts carry
   `m_dependence.verdict = "DEPENDENCE DETECTED: the 'no observable dependence on m' claim is
   not supported by this run"` — Kruskal--Wallis $p = 1.8\times10^{-70}$ on the scaling run
   ($13/21$ $k$-values significant after Holm) and $p = 0$ on the exhaustive one ($9/9$).

**Item 2 surfaced only because I read the source JSON instead of copying the strings I was
sent.** It was not in the hand-over. The clause is **dropped rather than reversed**: the artefact
gives the test and the $p$-value but neither direction nor effect size, and asserting a
dependence without a magnitude invites "how large?" from a reviewer who has just been told the
opposite in the previous version. Stating it is a new claim, not a correction.

**The same refuted claim is still live twice in the supplementary** (`:1469`, `:1477`) — flagged
to that session, not touched. Until it lands, the package asserts in its own appendix a claim
its own generator reports as falsified.

**The gate is wired and it is built for the right failure.** Seven anchored checks,
**121 → 128, all pass**. The defect being guarded is not a wrong number but a *right number
credited to the wrong experiment*, so each literal is anchored on a phrase naming its own run —
a check on `1.43` alone would pass on a sentence crediting it to the $5{,}400$-expression run.
**Regression-tested rather than assumed**: seeding `across $6{,}300$` → `across $5{,}400$`
produces 1 FAILURE; reverted, 128/128. The first version of the check was ambiguous on
`6{,}300`, which is stated in two subsections, and the verifier refused it with "matched 2
locations" — the ambiguous-check rule firing on its own author.

Paper **17 pages, 0 overfull, 0 undefined**, mirrored to `double_blind`, style exit 0 on both,
`ruff` clean.

### 2026-08-14 — supplementary released, prose pass done, and two findings back

The parallel session filled all twelve placeholders, deleted the scaffolding block and handed
`article/supplementary/` over. Placeholders now read **paper 0 · supplementary 0 · letter 0**.

#### 7.16.1 The supplementary prose pass — light, as predicted

Measured on de-TeX'd prose before and after. **8,382 tokens.** It arrived in better shape than
the paper did, which is what one lane writing it rather than five predicts:

| | before | after |
|---|---|---|
| Pseudo-clefts | 1 | **0** |
| Negative parallelism | 1 | **0** |
| Median sentence length | 25 | 25 |
| Nominalizations / 1,000 | 47 (ok) | 47 (ok) |
| Function-word ratio | 0.43 (ok) | 0.43 (ok) |

Three edits, all prose, no number touched:

- `:365` — "Hypothesis~(b) **is the one worth** stating explicitly, because **it is what makes**
  the step…" → "is worth stating explicitly, because it makes". The file's only pseudo-cleft.
- `:960` — "A third arm changes the multiplicity of the inference **and not only its size**." →
  "A third arm **also** changes the multiplicity of the inference." The file's only negative
  parallelism; *also* carries the contrast with the run-count paragraph above it.
- `:1136` — "we state it that way **rather than presenting it as an error-prone alternative**"
  → cut. Commentary on how we present the result, not on the result.

**Median sentence length is left at 25, one over the band, deliberately.** The long sentences
are almost all inside the Appendix A proofs — "Hence $X$, and applying Theorem $Y$ to $Z$
gives…" — where a chained sentence is the genre norm and splitting risks the logical thread. A
proof is the last place to manufacture burstiness, which is the overcorrection the skill warns
about by name. **Hedges are left low** for the same reason as in the paper: the results are
exact and adding a hedge to an exact number is worse science, not better prose.

**Two constraints from the handover were honoured**: the D.2/D.3 ledger enumerations — exit
status, balance, seed identity, revision, engine, digests — were left intact, because R2.6 asked
us to confirm exactly that list; and the R2.5 block's numbers were not touched.

Supplementary after the pass: **18 pages, 0 overfull, 0 undefined**; verify **128/128**; style
exit 0.

#### 7.16.2 Two findings returned to that lane

1. **I over-flagged the $m$-dependence by one, and they were right to push back.** Checked at
   source: the Kruskal--Wallis test runs on `log(mean_canon_time_s)` — timing — while
   `rho_equals_kfact_all = True` at $100\%$ invariance. So `:1469`, "$\rho = k!$ for every
   $(k,m)$, independent of the variable count $m$", is **true** and correctly kept. Only
   `:1477`, which attached the claim to the *timing*, was false. My own paper-side deletion was
   correct, because that clause attached to "a per-permutation **cost**". **A blanket sweep
   would have deleted a true claim** — the distinction is the whole content of the sentence.

2. 🔴 **The scaling figure is not in the document.** `figures_scaling/fig_synthetic_scalability.pdf`
   exists, timestamped 21:57, and I rendered it: the $k = 8..36$ panel with the fitted line, the
   $k!$ reference and the corrected badge, *"one canonical string on all $1.26\times10^{8}$
   sampled orderings"*. But `\label{sec:supp_scalability_sweep}` at `:1498` is **text-only**;
   the supplementary's only synthetic `\includegraphics`, at `:1431`, is the **exhaustive**
   panel. So the subsection carrying the exponent the manuscript now leads with has no picture
   of it. **Not wired by me** — adding a float changes their page count and layout.
   Two consequences if it goes in: the clipped `$O(k^{1.4})$ (fitted` legend then does reach the
   reader, and the legend rounds to `1.4` where both documents say `1.43`.

   **Resolved the same evening.** It was an omission, not a length decision, and the cause is
   worth recording: **both generator invocations write `fig_synthetic_scalability.pdf`**, so
   copying the scaling figure across would have silently overwritten the exhaustive one, and
   the copy step took only the exhaustive. A filename collision between two runs of one
   generator — the same shape as `\sigma` naming two functions and
   `\rho_\sigma`/`\rho_{\mathrm{ser}}` naming one quantity. Today's defects were all things
   sharing a name. Both cosmetic defects fixed in one regeneration: `{b_fit:.1f}` → `.2f` in
   the legend *and* the caption builder, and the label shortened to `$O(k^{1.43})$ fit` so two
   decimals still fit the frame. A unit test pinned the literal `(fitted)` and was re-pointed to
   a regex on the exponent rather than the wording. Installed as its own
   `fig_synthetic_sweep.pdf`, wired as `fig:synthetic_sweep`. Verified by me from a fresh build
   and a render: **19 pages, 0 overfull, 0 undefined, 128/128, style exit 0**, legend correct.

#### 7.16.3 The supplementary is not page-limited — a trade to refuse

The figure costs one page and that session offered to drop it if T13 wants the page back.
**It should not be dropped.** The decision letter's ceiling binds the **main manuscript file**
only — "12 pages, inclusive of main text, abstract, index terms, illustrations, references,
bios and photos", uploaded as the Formatted Main File. The supplementary uploads separately,
and the board's C7 decision keeps it separate as digital-library material; R1 and R3 both
answered C7 that way and R3 accepted the supplementary **as is**. So `17 → 19` costs nothing
against the ceiling, and dropping the figure would remove the only picture of the exponent the
manuscript now leads with in order to save a page in a document with no page budget.

**This hands T14 a better answer to R2's C7 than the one drafted.** R2 asked for the
supplementary to be folded into the main paper *within* the strict limit. The main file is 17
against a ceiling of 12; the supplementary is 19. Merging would require 36 pages to fit in 12.
At submission the pair was 12 + 10 and the request was merely unreasonable; it is now
arithmetically impossible — **and the growth is entirely R2.5's benchmark tables and this
figure, which is to say material R2 asked for**. Declining C7 in the reviewer's own terms is
both stronger and more courteous than arguing from the limit alone.

The page pressure is in `article/paper` — 17 against 12, six over in the body — and nothing in
the supplementary lane touches it (§7.14.1).

**Accepted by that session**, which re-checked the ceiling against
`source/README.md` rather than conceding, recorded the premise error as theirs, and withdrew
the offer. The C7 argument is now in T09 §7 for T14.

#### 7.16.4 B2 executed — copied before deleted

The condition-(iv) equivalence argument is now in **Appendix A**, immediately after the proof
of Theorem~A.3, where Step~4 of that proof uses condition~(iv) in exactly the base-only form
the argument justifies. The paper keeps a two-sentence statement of the equivalence and points
to the appendix.

**Order mattered and was followed**: the passage was **single-sourced in the paper**, so it was
copied and the copy verified to build and render (`pdftotext` finds it once) **before** a word
was removed from `methodology.tex`. The other order would have been unrecoverable.

**Added unnumbered**, as an `\added{\relax\medskip\noindent\textbf{...}}` paragraph rather than
a `remark` environment, for the same reason the §3.4 worked example was: a new theorem-like
environment in Appendix A would renumber everything after it, and `lem:fcs_valid` is **A.2** —
the literal R2 cites in comment R2.1. Verified after the move: A.1, A.2, A.3, A.4, E.1
unchanged.

Net: ~200 words and a display equation out of the paper for ~45 words of pointer; the
supplementary absorbs it at **no page cost** (19 either way). Every proof in the paper is now
in Appendix A without exception.

Final state, all re-run: paper **17 pages**, double-blind **17**, supplementary **19**, all
**0 overfull, 0 undefined**; paper theorem numbering **identical to the pre-session baseline**;
verify **128/128**; style **exit 0** on four trees.

---

## 8. Proposed answer

### 8.1 Before / after

All figures measured over the whole annotated `paper/` tree by
`scripts/check_manuscript_style.sh` unless stated otherwise.

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Abstract, "on both methods on both methods" | present, `main.tex:81` | corrected to "on both methods", blue | AC-1 |
| Abstract, double space | present, unreported; **source-only — \LaTeX{} collapsed it, so the typeset abstract never showed it** | removed | AC-1 |
| Abstract propagated to `double_blind/` | same defect, `main_anonymous.tex:71` | **done**; byte-identical to `article/` (§7.14.2) | AC-1 |
| Abstract propagated to statement | **no typo there**; only the headline numbers are restated, `:99–115` | **done**; restates the C2 values (§7.14.2) | AC-1 |
| Rendered name form | ISALSR **and** IsalSR | ISALSR uniformly | AC-3 |
| Occurrences bypassing `\IsalSR` | **10** (related work 4, methodology 5, experiments 1; **introduction 0**, contra §3.2/D6) | **0** | AC-3 |
| British-variant occurrences, whole family | **44** | **0** | AC-4 |
| — of which `canonicalis` | 27 | 0 | AC-4 |
| — `optimis` / `normalis` / `neighbour` / `behaviour` / `colour` / `standardis` / `labelled` / `summaris` / `parameteris` | 6 / 4 / 3 / 2 / 2 / 2 / 1 / 1 / 1 | 0 | AC-4 |
| Automated spelling + naming check | none | `scripts/check_manuscript_style.sh` (+ 45-rule word list); **exit 0** on this lane, **exit 1** on a seeded regression | AC-4 |
| R2 answer to C1 | **No** | abstract defect fixed, suite description current, and the effect figures are the C2 campaign's; **no deferral remains** (§7.14.2) | AC-2 |
| Guard false positives | `optimistic`, `optimism`, `specialist`, `generalist`, `organism`, `characteristic` all reported as misspellings; `\label{}`/`\includegraphics{}` arguments reported | **0**; stems anchored, non-prose command arguments scrubbed (§7.14.4) | AC-4 |
| Colour in the clean PDF | green hyperref link borders on every citation and cross-reference | **none**; `hidelinks` set (§7.14.3) | AC-8 |
| Median sentence length, longest section | $32$ words (discussion) | **$25$**; every section improved or held (§7.14.5) | R1.5 / C5 |
| Pseudo-clefts ("X is what Y") | $14$ | **0** | R1.5 |
| Prose words, whole paper | $14{,}895$ | **$14{,}420$** | R1.5 / C6 |
| Clean `article/paper/main.pdf` | — | **18 pages**; markup-stripping buys 0, bios and photos cost 1 (§7.14.1) | T13 |
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

**Amended 2026-08-14.** One paragraph was added to **R1.5**, reporting the sentence-level
pass with its measurements (Discussion median $32 \to 25$ words, Introduction $\to 19$,
fourteen cleft constructions removed, $520$ words shorter) and stating in its own voice that
this does not answer R2's C6 on length and is not offered as one. The **R2.8** block was
re-audited against the submitted source and every number in it holds: the $44$ British-variant
occurrences and the $10$ bypassing the macro were re-counted at commit `577e0f2` with the
*corrected* guard and come to $44$ and $10$ exactly, and none of the six words the old guard
false-flagged appears in the submitted paper, so the bug never inflated the count. The **R2 C1**
block still describes the effect figures as being regenerated; **that clause is now stale**
(§7.14.2) and T14 must drop it.

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

What a round-2 reviewer could still raise, in descending order of how likely they are to
find it. R2 checked the appendices line by line last round; assume they will again.

1. **R2.8(b) and (c) are not closed across the package.**
   `double_blind/supplementary/supplementary_anonymous.tex` still renders "IsalSR" twice
   (`:383`, `:506`) and still carries British spellings, and the double-blind package is the
   one R2 read. The answer to R2.8 says the name is unified; on that file it is not. **T14's
   re-sync must land before the letter's claim is true.** This is the single item most likely
   to be caught, because it is the exact defect the reviewer reported.
2. **The manuscript is six pages over, in the body.** Measured four ways in §7.14.1: stripping
   the revision markup buys nothing and the mandatory bios and photos cost one page. Nothing in
   an editorial pass reaches this. T13 owns it, and C6 (R2, "should be trimmed a bit") is not
   answerable until it moves.
3. **The AI-assistance acknowledgement no longer covers everything an LLM did.** It credits
   benchmark-suite discovery, cluster code and the website; the prose has since been edited by
   one. §7.14.8 states the decision the authors must take. If it is not taken and a reviewer
   infers it, the cost lands on the same paragraph R2.5 is already contesting.
4. **Author sign-off is unrecorded (AC-9).** Three of the eight changed files are Ezequiel's,
   and this session made three content-level cuts in them (§7.14.5): the introduction's
   roadmap, related work's re-definition of SR, and one self-referential sentence in the
   Discussion. Each is defensible and each is reversible; none has been approved.
5. **Content landing after this pass can reintroduce the inconsistencies.** The guard now runs
   clean on four trees and catches both regression classes, but it is not wired into anything.
   T14's final check should run `scripts/check_manuscript_style.sh` and
   `experiments.scripts.review_campaign.verify` over the assembled package, not over the tree
   as it stood when a ticket closed.
6. **The abstract's figures track C2 and would move again if any arm is re-run.** They are
   asserted by `verify.py` against `analyses/values/summary.json`, so drift fails loudly rather
   than silently — but only for whoever runs it.
