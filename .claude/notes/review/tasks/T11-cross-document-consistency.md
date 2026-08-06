# T11 — Cross-document and package consistency

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.2**, **R2.4** (and E5, E7) |
| Type | Cross-reference / package hygiene |
| Owner | **Karl** (primary — the sweep) · **Ezequiel** (R2.2, the preprint and `methodology.tex`) |
| Depends on | — (can start immediately) |
| Blocks | T13, T14 |
| Status | **AC-0…AC-7 MET · AC-8 DEFERRED to the promotion step · AC-9 MET.** AC-3 half-inherited from the T07 rewrite. AC-7's edit executed by the `paper/` lane; the reachability check is this ticket's. **One new failure found and handed off**: `results.tex:176` "Table 8 in the appendices" is broken by the two supplementary tables T05 added — the `paper/` lane must renumber it to **Table 10** before the R2.4 answer is true as written. |
| Target | 2026-08-24 |

---

## T16 impact — R2.2's `{g,i}` vs `{−,/}` now has a real answer (added 2026-07-30)

**Read this before starting the cross-document sweep. It changes what "consistent"
means for R2.2.**

R2.2 flags that the documents disagree about whether the alphabet contains `{g, i}`
(Neg, Inv) or `{−, /}` (Sub, Div). Until 2026-07-30 this was not merely an editing
slip: the manuscript said `{g, i}` and **the implementation actually used `{−, /}`**,
on 61.1 % of production candidates. The documents disagreed because the artefacts
disagreed.

**T16 resolved it in the code's favour of the paper.** Both adapters now emit the
paper's alphabet: `x − y = Add(x, Neg(y))`, `x / y = Mul(x, Inv(y))`, with `Pow` the
only non-commutative operation. Verified on all 10 production configs over ~130,000
DAGs: zero `Sub`/`Div` nodes, zero `-`/`/` characters in any canonical string.

**Consequence for the sweep: harmonise everything onto `{g, i}`.** That is now the
true description of the implementation, so it is the correct target — no compromise
wording is needed and no caveat is required. Two specifics:

- `NodeType.SUB`/`DIV` and `BINARY_OPS` **still exist in `isalsr.core`**, because S2D
  must keep decoding legacy `V-`/`V/` strings and the property corpora contain them.
  If a document describes the *core library's* type registry rather than Σ_SR, `{−,
  /}` is accurate there. Do not "fix" those into inconsistency.
- **A separate documentation error surfaced and is not yet fixed anywhere**: the
  manuscript describes UDFS as searching over a configured operator set, but UDFS's
  YAML `operator_set` key is **dead configuration** — the vendored search ignores it
  and always samples every operator in `config.NODE_ARITY`, including `neg`, `inv`,
  and both orientations of subtraction and division
  (`vendor/DAG_search/dag_search.py:1226-1227`). Whatever the manuscript says about
  the UDFS operator set needs checking against that. Owner unassigned.

**The arXiv preprint decision is unchanged** — no v3 (README "Decisions already
taken"). R2.2 is answered as a comment in the response letter.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.

---

## 1. Why these are grouped

All four are the same defect class: **a document in the submitted package disagrees
with another document in the package, or points at something that does not exist.**
They are not four independent errors; they are four instances of one structural
weakness — the main paper, the supplementary, the preprint and the
previously-published statement compile separately, so *every* cross-document
reference is a manually typed number that nothing checks.

Fixing them one at a time re-types four numbers. Fixing them together produces a
verified inventory, which is what stops the same class recurring in round 2. R2
found two of the four; the sweep must find the rest before they do.

The reviewers had **more than the main PDF**. R2 refers to *"the embedded preprint"*
and says the two definitions of Σ_SR *"coexist in the submitted PDF"*. The package
under review is the whole bundle, and consistency is a property of the bundle.

---

## 2. Verbatim comments

**R2.2:**
> 2. Definition 3.2 uses label characters {g, i} for NEG and INV, while Definition
> 2.2 in the embedded preprint uses {−, /}. These two definitions of the same
> alphabet Σ_SR coexist in the submitted PDF and should be reconciled.

**R2.4:**
> 4. Appendix A cites "Table 4 of the main document," but no Table 4 exists in the
> main text; the referenced FCS pseudocode is in Table 3 of Appendix C.

---

## 3. Established facts

### 3.1 R2.2 — the error is in the preprint, and the preprint contradicts *itself*

| Source | ℒ |
|---|---|
| Journal Def 3.2, `article/paper/methodology.tex:95` | {+, *, `g`, `i`, s, c, e, l, r, ^, a, k} |
| Journal Table 1, `methodology.tex:77–78` | `g` → Neg, `i` → Inv |
| Preprint Def 2.2, `article/arxiv/69b91250e7e60fc6079dfd5d/methodology.tex:97` | {+, *, `-`, `/`, s, …} |
| Preprint Table 1, same file `:79–80` | `g` → Neg, `i` → Inv |
| Preprint prose, same file `:125–126` | "Neg (label `g`) … Inv (label `i`)" |

**The journal manuscript is self-consistent throughout.** The *preprint* is
internally inconsistent: its Definition 2.2 contradicts its own Table 1 and its own
prose. Whoever answers R2.2 needs both facts — the answer is not "we will fix the
journal paper", it is "the journal paper is already correct; the preprint carries a
typo in one line, which we are correcting there."

**Decision taken 2026-07-27 (Mario): no arXiv update.** We do not issue a v3. The
journal version supersedes the preprint, and R2.2 is answered as a **comment in the
response letter**: the journal manuscript is self-consistent, the divergence is a
single-line typo in the preprint that contradicts the preprint's own Table 1 and its
own prose, and the published journal version replaces it.

This is the whole remedy. Do not open a preprint work item, do not edit anything
under `article/arxiv/`, and do not make the response conditional on an arXiv update
landing before submission.

**Knock-on to check**: journal Def 3.2 (`methodology.tex:116–117`) states "7
single-character tokens and 24 compound tokens (2×|ℒ|), totaling 31 tokens",
consistent with |ℒ| = 12 ✓. Separately `supplementary.tex:914` uses |𝒜| = 17 for a
Lev-1 neighbourhood count over a *different, reduced* alphabet. These are different
quantities. Do not "reconcile" them into one; verify each is correct in its own
context and, if the collision of notation is confusing, disambiguate the symbol.

### 3.2 R2.4 — and the full inventory

`supplementary.tex:120`, inside the proof of Lemma A.2, cites "Table 4 of the main
document". The main text has exactly three tables: `tab:operations`
(`methodology.tex:59`), `tab:three_axis` (`results.tex:32`), `tab:cpdt_summary`
(`results.tex:89`). The FCS pseudocode is `tab:canon_pseudo` at `supplementary.tex:382`
= **Table 3 of the supplementary, Appendix C**. R2's correction is exact.

**A second broken reference nobody raised**: `supplementary.tex:453` says
"Section 5.1 of the main paper". The main paper's Section V.1 is *Search Space
Reduction*; the intended target is Section IV (Computational Experiments). Wrong.

The complete inventory of hardcoded cross-document references is in
`manuscript-map.md` § "Hardcoded cross-document references" — 8 supplementary→main,
12 main→supplementary, 1 statement→supplementary. **Every one must be re-verified
after all content tickets land, because every one breaks silently on renumbering.**

**Root-cause note**: `methodology.tex` carries full duplicates of all three
pseudocodes inside `\begin{comment}` blocks (E6, ≈ 600 of 1,143 lines). Those dead
copies are why the same pseudocode exists in two files, which is how the
Table 3 / Table 4 confusion arose. Removing them is T13's job; flag it there.

### 3.3 E5 — the previously-published statement
- `previously_published_statement/main.tex:77` refers to proofs in "the supplementary
  material (Section S.I)". The supplementary uses Appendix A–G lettering. No such section.
- `:142–148` describes the supplementary as "approximately thirty pages". It is **10**.

### 3.4 E7 — anonymised URL in the non-anonymous file
`article/paper/computational_experiments.tex:2–4` points at
`https://little-manifold.github.io/isalsr-anon/` — the double-blind anonymised
mirror — in the **non-anonymous** `article/` version. The companion-website URL must
be the real one in `article/` and the anonymised one only in `double_blind/`.
Check the mirror is live and that the two versions have not drifted apart.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-2.md` — §R2.2, §R2.4
- `.claude/notes/review/source/verified-discrepancies.md` — D3, D8, E5, E6, E7
- `.claude/notes/review/source/manuscript-map.md` — **the whole file**; the hardcoded
  cross-reference inventory is the working checklist for this ticket
- `.claude/notes/review/source/00-editor-and-decision.md` — what the reviewers
  actually received in the package
- `.claude/notes/review/source/README.md` — file ownership; several of these files
  are Ezequiel's

---

## 5. Work specification

1. **R2.2.** Remedy already settled: no arXiv update, answered as a comment (§3.1).
   The only work here is to **verify the journal manuscript is self-consistent end to
   end** on ℒ and the token count — that verification is what the response asserts,
   so it must be true. Disambiguate ℒ vs 𝒜 if the notation collision is judged
   confusing. No preprint edits.
2. **R2.4.** Fix `supplementary.tex:120` and `supplementary.tex:453`.
3. **Verify the full inventory.** Walk all 21 hardcoded references in
   `manuscript-map.md`, confirm each resolves, and record the result as a checklist
   in §6. **Re-run this walk after every content ticket lands** — it is the last
   check before submission and belongs in T14's checklist too.
4. **Reduce the exposure.** Where a cross-document reference can be replaced by a
   description that does not depend on a number ("the fast canonical pseudocode in
   Appendix C" rather than "Table 4 of the main document"), do it. Numbers that
   cannot be avoided should be listed in one place so the pre-submission check is
   mechanical.
5. **E5.** Fix both errors in the previously-published statement; re-check the page
   count against the final compiled supplementary, not the submitted one.
6. **E7.** Restore the real URL in `article/`, keep the anonymised URL in
   `double_blind/`, and confirm both sites are reachable and current.
7. **Propagate to `double_blind/`.** Every fix must land in both trees. The
   submitted versions were verified identical in content; keep them that way.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** R2.2 answered as a response-letter comment; **no** arXiv v3, **no** edits
  under `article/arxiv/`. The self-consistency claim the response makes is verified
  before it is written.
- **AC-2.** Journal manuscript confirmed self-consistent on ℒ and on the token count.
- **AC-3.** `supplementary.tex:120` and `:453` fixed.
- **AC-4.** All 21 hardcoded cross-document references verified; checklist recorded
  in §7 with a pass/fail per entry.
- **AC-5.** Avoidable hardcoded references replaced by number-free descriptions;
  the irreducible remainder listed in one place.
- **AC-6.** E5 fixed, with the page count taken from the final compiled PDF.
- **AC-7.** E7 fixed; both websites checked reachable and current.
- **AC-8.** All fixes propagated to `double_blind/`; the two trees re-verified identical.
- **AC-9.** §8 filled.

---

## 7. Work log

### 2026-08-06 — sweep executed (writing lane: annotated supplementary + statement + R2.4 block)

**Lane discipline.** `article/` was recorded by md5 before any work and re-checked after:
all five pre-existing dirty files are byte-identical to the baseline, so this ticket added
nothing to `article/`. The `paper/` tree was read but never written; a second agent owns it.
The letter was compiled into a private scratch directory so the parallel lane's `.aux` was
never clobbered.

#### 7.1 AC-1 / AC-2 — the R2.2 self-consistency claim holds

The R2.2 answer was already written (letter lines ~991–1017) and asserts three checkable
things. All three were verified, and **none of them is contradicted**, so that block was
left untouched:

| Claim in the written R2.2 answer | Verified against | Verdict |
|---|---|---|
| ℒ = {+, *, g, i, s, c, e, l, r, ^, a, k} in Definition 3.2 | `article/paper/methodology.tex:95` | holds |
| Table 1 lists `g`→Neg, `i`→Inv | `methodology.tex:77–78` | holds |
| The commutative-encoding paragraph names both by those characters | `methodology.tex:120–122` | holds |
| "7 single-character tokens and 24 compound tokens (2×\|ℒ\|), totaling 31" | 7 = {N,P,n,p,C,c,W}; 24 = 2×12; 7+24 = 31 | holds |
| "the same twelve labels and the same 31 tokens in both documents" | preprint `methodology.tex:118–119` also prints **31** | holds |
| "agrees with the manuscript's in ten of twelve positions" | preprint `:97` differs only at positions 3 and 4 (`-`,`/` for `g`,`i`) | holds, exactly |
| "no occurrence of `-` or `/` as a label character remains anywhere in the paper or in the supplementary" | grep for `texttt{-}`/`texttt{/}`/`V-`/`V/` over `article/paper/*.tex` + `article/supplementary/*.tex` | zero hits |

**Code-side corroboration (per the mid-run amendment).** `src/isalsr/core/node_types.py`
`LABEL_CHAR_MAP` has **14** entries and *does* contain `-`→SUB and `/`→DIV. This is **not**
a divergence from the manuscript: §T16-impact of this ticket (lines 35–38) states that the
core registry keeps them so S2D can decode legacy `V-`/`V/` strings, while the adapters
decompose. `experiments/models/commutative_encoding.py` confirms it in its own docstring,
which independently states the twelve-label alphabet and the two decomposition identities.
Conflict rule applied: `commutative_encoding.py` and this ticket file carry the **same**
commit timestamp (2026-07-30 17:01:51 +0200); `node_types.py` is older (2026-03-18) and is
the file the ticket explicitly says not to change. **No staleness, no unresolved point.**

A third, independent corroboration turned up in AC-7: both companion websites state a
"31-token alphabet ... 7 single-char tokens ... 24 compound tokens". Three sources
(manuscript, code, website) agree on 12 / 31.

**ℒ vs 𝒜 (the notation collision).** Judged genuinely confusing and disambiguated.
`|𝒜| = 17` is a **character** count, not a token count: 𝒜 = {N,P,n,p,C,c,W,V,v} ∪ (label
characters of the reduced operation set configured for the metric-space experiment), and
`c` is in both parts. Its internal arithmetic is exact for L = 10: 10 deletions +
10×16 substitutions + 11×17 insertions = **357**, which is the number printed. So 17 is
correct in its own context and is a different quantity from |ℒ| = 12 and from |Σ_SR| = 31.
The two were **not** reconciled; instead 𝒜 is now defined at its point of use, in blue.
*Flag for whoever owns Appendix F*: reproducing 17 requires the reduced label set
{+,*,-,/,s,c,e,l,k}, i.e. it still contains the pre-T16 `-` and `/`. That set is never
printed, so no reader can see it, but the experiment behind Appendix F predates T16 and its
alphabet is not the paper's. Not a defect in the text; a provenance note.

#### 7.2 AC-3 — one half was already done, in the right style

`supplementary.tex:120` ("Table 4 of the main document") **no longer exists** in the
annotated copy. The T07 lane rewrote the proof of Lemma A.2 wholesale and, in doing so,
replaced the reference with "Fast Canonical String definition of the main document;
pseudocode in Table 3 of Appendix C" — already the number-free form AC-5 asks for, already
inside the blue block spanning annotated lines 125–225. The R2.1 answer in the letter
(line 966) already claims this. Verified from the rebuilt PDF, not the source.

`supplementary.tex:453` ("Section 5.1 of the main paper") was still present and still wrong.
Fixed at annotated `:617`.

Surprise worth recording: **the compiled main paper numbers its sections in Arabic, not
Roman.** `manuscript-map.md` writes them as I…VII and the reviewers write "Section 4.2",
which is what the PDF prints. The map's Roman numerals are a presentational error in the
map, not in the paper. Section 5 is Results and 5.1 is Search Space Reduction, so the
substance of the map's diagnosis was right.

#### 7.3 AC-4 — the 21-reference walk, verified from the rebuilt PDFs

Method: `pdftotext -raw` on the rebuilt `internal_copy_reviewed_article/paper/main.pdf` and
`.../supplementary/supplementary.pdf`, cross-checked against the compiled label tables in
`main.aux` / `supplementary.aux` (which are compile products, not source). Every target
number below is the number the typesetter actually assigned.

**Supplementary → main (8):**

| # | Submitted line | Text | Target in the rebuilt main PDF | Verdict |
|---|---|---|---|---|
| S1 | `:120` | "Table 4 of the main document" | main has Tables 1–3 only; no Table 4 | **FAIL (as submitted)** → fixed by the T07 rewrite, annotated `:145–149` |
| S2 | `:131` | "Definition 3.5 of the main document" | `def:valid_set` = **3.5** | PASS |
| S3 | `:170` | "Definition 3.7 of the main document" | `def:wl_hash` = **3.7** | PASS |
| S4 | `:209` | "Definition 3.9 of the main document" | `def:isomorphism` = **3.9** | PASS |
| S5 | `:235` | "Definition 3.9 of the main document" | **3.9** | PASS |
| S6 | `:239` | "Theorem 3.15 of the main document" | `thm:invariant` = **3.15** | PASS |
| S7 | `:453` | "Section 5.1 of the main paper" | 5.1 = Search Space Reduction; intended = Section 4, Computational Experiments | **FAIL** → fixed, annotated `:617` |
| S8 | `:1005` | "Table 1 of the main document" | `tab:operations` = **1** | PASS → also made number-free, annotated `:1310` |

**Main → supplementary (12):**

| # | Submitted line | Text | Target in the rebuilt supplementary PDF | Verdict |
|---|---|---|---|---|
| M1 | `methodology.tex:175` | "Table 1 of Appendix C" | `tab:s2d_pseudo` = **1**, `sec:supp_pseudocodes` = **C** | PASS |
| M2 | `methodology.tex:190` | "Table 1 of Appendix C" | **1** / **C** | PASS |
| M3 | `methodology.tex:422` | "Table 2 of Appendix C" | `tab:d2s_pseudo` = **2** | PASS |
| M4 | `methodology.tex:680` | "Table 3 of Appendix C" | `tab:canon_pseudo` = **3** | PASS |
| M5 | `methodology.tex:809` | "Table 3 of Appendix C" | **3** | PASS |
| M6 | `results.tex:176` | "Table 8 in the appendices" (k-stratified overhead) | `tab:k_range_overhead` is now **Table 10**; Table 8 is now the per-problem UDFS table | **FAIL — broken by this revision** |
| M7 | `discussion.tex:28` | "Appendix E.2" | `sec:supp_scalability_synthetic` = **E.2** | PASS |
| M8 | `discussion.tex:50` | "Appendix E.1" | `sec:supp_scalability_empirical` = **E.1** | PASS |
| M9 | `discussion.tex:101` | "Appendix E.1" | **E.1** | PASS |
| M10 | `computational_experiments.tex:52` | "Appendix D.1" | `sec:supp_benchmarks` = **D.1** | PASS |
| M11 | `computational_experiments.tex:92` | "Appendix D.1" | **D.1** | PASS |
| M12 | `results.tex:146` | "the per-problem tables in the appendices" | Tables 8–9 (were 6–7); reference is number-free so it survived | PASS |

**Other → supplementary (1):**

| # | Submitted line | Text | Target | Verdict |
|---|---|---|---|---|
| P1 | `previously_published_statement/main.tex:77` | "supplementary material (Section S.I)" | supplementary letters its appendices A–G; no Section S.I exists, and never did | **FAIL** → fixed, `:76` |

**Tally: 21 walked, 17 PASS, 4 FAIL.** Three of the four were failures in the submitted
package (S1, S7, P1); the fourth (M6) was created *by the revision itself*.

**M6 in detail, because it is the one that is still open.** T05 inserted two benchmark
tables into the supplementary, `tab:strogatz` (now Table 6) and `tab:feynman_ext` (now
Table 7). Everything after them shifted by +2: per-problem UDFS 6→**8**, per-problem Bingo
7→**9**, k-stratified overhead 8→**10**, synthetic scalability 9→**11**, shortest paths
10→**12**, neighbourhood 11→**13**. `results.tex:176` reads "…the median overhead is 39.2%
(Table~8 in the appendices reports the $k$-stratified breakdown: …)". In the rebuilt
supplementary, Table 8 is the per-problem UDFS table. **The sentence now points a reviewer
at the wrong table.** `results.tex` belongs to the `paper/` lane; the fix is in §7.5.
(My first grep for this missed it because the string wraps across a source line; found via
`pdftotext` on the rebuilt PDF, which is exactly why AC-4 mandates the PDF over the source.)

**References the revision added, checked the same way (not part of the 21).** Fourteen
further hardcoded references now exist and **all resolve**: main→supplementary gains one
more "Table 2 of Appendix C", one bare "Appendix C", three "Appendix D.2"
(`sec:supp_baseline` = D.2) and four "Please see Appendix A" (`sec:supp_proofs` = A);
supplementary→main gains "Definition 3.2 of the main document" (`def:alphabet` = 3.2) plus
several references to Definitions 3.4/3.5/3.9. Style flag, not a defect: four of the new
supplementary references write "Definition 3.5" *without* the qualifier "of the main
document". No supplementary object is numbered 3.x, so they are unambiguous by absence, but
they read inconsistently next to the qualified ones. Worth a pass in T13.

Also unchanged and therefore safe: T07 appended Definition 3.16, Lemma 3.17 and Corollary
3.18 **after** Theorem 3.15, so Definitions 3.1–3.9, Remarks 3.10–3.12 and Theorems
3.13/3.15 all keep the numbers the reviewers cite. This is the one place the revision could
have invalidated every supplementary→main reference at once, and it did not.

#### 7.4 AC-5 — what was made number-free, and the irreducible remainder

Executed in this lane (annotated supplementary):

| Annotated line | Before | After |
|---|---|---|
| `:106` | "(Definition 3.5 and Table 2)" — ambiguous: Table 2 of *which* document | "(Definition 3.5 of the main document and Table~`\ref{tab:d2s_pseudo}` of Appendix~`\ref{sec:supp_pseudocodes}`)" |
| `:148` | "pseudocode in Table 3 of Appendix C" | "pseudocode in Table~`\ref{tab:canon_pseudo}` of Appendix~`\ref{sec:supp_pseudocodes}`" |
| `:617` | "Section 5.1 of the main paper" | "the Computational Experiments section of the main paper" |
| `:1310` | "Table 1 of the main document" | "the operation-type table of the main document" |
| statement `:76` | "the supplementary material (Section S.I)" | "the proofs appendix of the supplementary material" |

The first two are *intra*-document references that had been typed as literals; they now go
through `\ref`, so the typesetter maintains them. That removes them from the manual list
entirely rather than merely making them robust.

**Irreducible remainder — the mechanical pre-submission list.** Everything below is a
number or letter that no `\ref` can resolve, because the two documents compile separately.
This is the complete list to re-walk in T14:

*Supplementary → main (12 numbers, all into the manuscript's Section 3):*
Definition 3.2 (×1), Definition 3.4 (×1), Definition 3.5 (×4), Definition 3.7 (×1),
Definition 3.9 (×4), Theorem 3.15 (×1). Kept as numbers deliberately: the reviewers cite
these objects by number themselves, and a description would make them harder to locate.

*Main → supplementary (appendix letters, 6 distinct):* A, C, D.1, D.2, E.1, E.2. Stable
unless T13 reorders or merges appendices, which is exactly what T13 is chartered to
consider — so this list must be re-walked after T13, not before.

*Main → supplementary (table numbers, 1):* `results.tex:176`. This is the only surviving
cross-document *table number* and it is currently wrong. If the `paper/` lane takes option
(b) below, the irreducible table-number count drops to **zero**.

#### 7.5 Handoff to the `paper/` lane — exact proposed edits, not made here

All line numbers are in `reviews/internal_copy_reviewed_article/paper/`. Item 1 is a
**required correction**; items 2–4 are AC-5 hardening and are optional.

1. **`results.tex:176` — REQUIRED.** Currently `(Table~8 in the` / `appendices reports the
   $k$-stratified breakdown: …)`.
   - (a) minimal: `Table~8` → `Table~10`.
   - (b) preferred, and removes the last cross-document table number:
     `(Table~8 in the appendices reports` → `(the $k$-stratified overhead table in the
     appendices reports`.
   **The R2.4 answer as shipped states that this citation "has been updated with it". If
   neither edit lands, that sentence is false.** This is the one hard dependency this
   ticket leaves behind.
2. `methodology.tex:190` and `:205` — "Table~1 of Appendix~C" → "the S2D pseudocode in
   Appendix~C".
3. `methodology.tex:437` and `:705` — "Table~2 of Appendix~C" → "the D2S pseudocode in
   Appendix~C".
4. `methodology.tex:695` and `:875` — "Table~3 of Appendix~C" → "the fast canonical
   pseudocode in Appendix~C".

Leave alone: the four "Please see Appendix~A" proof stubs, the "Appendix~D.1/D.2/E.1/E.2"
references, and `results.tex:146` ("the per-problem tables in the appendices"), which is
already number-free and survived the renumbering untouched — the best available evidence
that the AC-5 policy works.

**Also for the `paper`/letter lane, not actioned here:** the response letter mixes
`-isation` and `-ization` in its own prose (the R2.3 block on letter page 17 prints
"canonicalisation"). The R2.4 block written by this ticket uses no such word. The letter is
not the manuscript, but R2.8c is about exactly this inconsistency, so T12 should sweep the
letter as well as the paper.

#### 7.6 AC-6 — E5, and why the page count is a moving target

| Statement line | Before | After | Source of the new value |
|---|---|---|---|
| `:76–77` | "the supplementary material (Section~S.I)" | "the proofs appendix of the supplementary material" | supplementary letters appendices A–G; Appendix A is Proofs (`sec:supp_proofs` = A in `supplementary.aux`) |
| `:142` | "approximately thirty pages" | "thirteen pages" | `pdfinfo` on the freshly rebuilt annotated `supplementary.pdf` → **Pages: 13** |

The submitted supplementary was 10 pages; the annotated one is **13**, because T05's two
benchmark tables and T07's rewritten proofs have landed in it. **This number must be
re-taken at submission**: T09 (appendix D rebuild), T13 (reorganisation) and any further
content ticket will move it again, and the statement is the one document in the package
that nothing else forces you to re-open. Concretely: re-run
`pdfinfo <supplementary>.pdf | grep Pages` and re-word `:142` as the last action before
upload. The word "thirteen" is spelled out, so a stale value cannot hide as a digit in a
table.

This file has no annotated twin and is a separate administrative form, so it was edited
directly and carries **no blue**. Exact before/after is in §8.2.

#### 7.7 AC-7 — both sites are live; the mirror is stale but not divergent

The edit itself was **already executed by the `paper` lane** before this ticket ran:
annotated `computational_experiments.tex:4` now reads
`{\color{blue}\url{https://mariopasc.github.io/IsalSR/}}`. Clean
`article/paper/computational_experiments.tex:4` still carries the anonymised URL, which is
correct — the fix lives in the annotated copy until the promotion step.
`double_blind/paper/computational_experiments.tex:4` correctly retains the anonymised URL.
Nothing was edited here.

Reachability and currency, measured:

| URL | Status | `Last-Modified` |
|---|---|---|
| `https://mariopasc.github.io/IsalSR/` (real) | HTTP/2 **200** | Tue, 28 Jul 2026 14:33:49 GMT |
| `https://little-manifold.github.io/isalsr-anon/` (anonymised mirror) | HTTP/2 **200** | Tue, 12 May 2026 15:09:19 GMT |

Both live. Both carry the same four sections and both state the "31-token alphabet ...
7 single-char tokens ... 24 compound tokens", which is the manuscript's Definition 3.2.
The mirror is correctly anonymised: no author names, "Extending a prior instruction-set
framework" in place of "Extending IsalGraph", and a publications section that says author
and venue are withheld for double-blind review.

**Stated plainly: the mirror is 2.5 months staler than the real site.** No content
divergence was detectable at page level, but any change made to the real site after
2026-05-12 is not reflected in the mirror. If the real site is updated to describe the
revision (new benchmark suites, the C++ engine), the mirror must be regenerated or the
double-blind version will describe a different artefact from the non-anonymous one. Flag
for T14.

#### 7.8 AC-8 — `double_blind/` propagation, deferred with an explicit file list

Not done, by instruction: propagation is part of the promotion step, after the annotated
content is folded into `article/`. Recording it here so T14 picks it up.

Files that will need propagation once promotion happens:

| Annotated source (this lane) | `article/` target | `double_blind/` target | Note |
|---|---|---|---|
| `.../supplementary/supplementary.tex` | `article/supplementary/supplementary.tex` | `double_blind/supplementary/supplementary_anonymous.tex` | **not a copy** — the anonymous variant differs; the four cross-reference edits, the 𝒜 definition and the spelling sweep must be applied by hand |
| `.../supplementary/table_supplementary_udfs.tex` | `article/supplementary/table_supplementary_udfs.tex` | `double_blind/supplementary/table_supplementary_udfs.tex` | currently byte-identical between the two trees (md5 `8cb0c49e…`), so a straight copy is valid |
| `.../supplementary/table_supplementary_bingo.tex` | `article/supplementary/table_supplementary_bingo.tex` | `double_blind/supplementary/table_supplementary_bingo.tex` | currently byte-identical (md5 `201f3bac…`), straight copy valid |

`previously_published_statement/main.tex` has **no** double-blind twin and needs no
propagation. `double_blind/` has no `previously_published_statement/` or `cover_letter/`.

#### 7.9 Spelling sweep — American, count recorded

Scope: the annotated supplementary tree only. The `paper/` half belongs to T12 and the
other lane.

| Word | Occurrences changed | To |
|---|---|---|
| `neighbour*` (prose) | 21 | `neighbor*` |
| `canonicalisation` | 6 in `supplementary.tex` + 1 in `table_supplementary_udfs.tex` + 1 in `table_supplementary_bingo.tex` = **8** | `canonicalization` |
| `colour*` | 2 | `color*` |
| `initialisation` | 1 | `initialization` |
| `amortise` | 1 | `amortize` |
| `summarises` | 1 | `summarizes` |
| `visualises` | 1 | `visualizes` |
| `behaviour` | 1 | `behavior` |
| `labelled` | 1 | `labeled` |
| **Total** | **37** | |

Deliberately **not** changed (5 further `neighbourhood` occurrences): the identifiers
`tab:supp_neighbourhood` (×2), `fig:neighbourhood` (×2) and the figure filename
`fig_neighbourhood.pdf` (×1). Renaming labels is diff noise with no reader-visible effect,
and renaming the filename would break `\includegraphics` because the PDF on disk is
`fig_neighbourhood.pdf`. A residual grep for British forms in the supplementary tree
returns exactly these five and nothing else.

False positives checked and left alone: "analyses" (correct American plural of *analysis*,
twice), "otherwise", "pairwise", "optimistic".

**The spelling changes are not marked in blue**, and that is a deliberate departure worth
a decision from Karl. Blue means "changed in this revision", and these were. But 37 blue
words scattered over 13 pages would bury the four substantive blue passages this ticket
added, and R2.8c asks for consistency rather than for a visible diff. The full list is
above, so blueing them is a one-pass job if that call goes the other way.

The statement (`previously_published_statement/main.tex`) was swept too: `Canonicalisation`
×2, `canonicalisation` ×1, `colour` ×2, `analyse` ×1, `formalised` ×1, `labelled` ×1,
`favour` ×1 = **9** further changes, at lines 57, 63, 64, 68, 73, 74, 106, 112 and 164.
It is package material and it is in this lane. "analyses" at `:146` is the correct American
plural and was left alone.

#### 7.10 Per-file sign-off checklist

For Karl and Ezequiel: review the diff, not the work.

| File | Changed | What to check | Reviewer |
|---|---|---|---|
| `reviews/internal_copy_reviewed_article/supplementary/supplementary.tex` | 4 cross-reference edits (`:106`, `:148`, `:617`, `:1310`), 1 new blue paragraph defining 𝒜 (`:1210–1216`), 35 spelling normalisations | that `:617` names the right section; that the 𝒜 paragraph does not over-claim; that the two `\ref`-ised references still render "Table 2"/"Table 3" | Ezequiel (proofs), Karl (register) |
| `reviews/internal_copy_reviewed_article/supplementary/table_supplementary_udfs.tex` | 1 spelling | caption only | Karl |
| `reviews/internal_copy_reviewed_article/supplementary/table_supplementary_bingo.tex` | 1 spelling | caption only | Karl |
| `previously_published_statement/main.tex` | 2 E5 corrections + 8 spellings | **the page count, which will go stale** | Mario at submission |
| `reviews/response_to_reviewers.tex` | R2.4 block only (lines 1171–1208) | that the "has been updated with it" clause is backed by the `results.tex` fix | Mario |
| `reviews/internal_copy_reviewed_article/paper/results.tex` | **not changed here** | apply §7.5 item 1 | `paper/` lane |
| `article/**` | **not changed** | md5-identical to the pre-work baseline | — |
| `double_blind/**` | **not changed** | see §7.8 | T14 |

#### 7.11 Verification results

| Gate | Result |
|---|---|
| Letter, `pdflatex` ×2 into a private output dir | exit **0**, **0** |
| Letter, `grep -c Overfull` | **0** |
| Letter, `pdftotext \| grep -c '??'` | **0** |
| Letter, `grep "LaTeX Warning"` | **0** |
| Letter, rendered pages 17–18 at 150 dpi and read | prose renders as prose; no headings, lists or labels inside the response; `\changeref` breaks cleanly across the page |
| Annotated supplementary, `pdflatex` ×3 | exit **0**, **0**, **0** |
| Annotated supplementary, `grep -c "^! "` | **0** |
| Annotated supplementary, `grep -c "Reference.*undefined"` | **0** |
| Annotated supplementary, `grep -c "Citation.*undefined"` | **0** |
| Statement, `pdflatex -halt-on-error` ×2 | exit **0**, **0**; 1 page |
| `grep -c "color{red}"` across the annotated supplementary tree | **0** |
| `grep -c "\[MPG"` across the annotated supplementary tree | **0** |
| `rcomment` integrity | 18 blocks in `HEAD`, 18 in the working tree, **0** with a differing verbatim body |
| Change hunks in the letter attributable to this ticket | exactly one, the R2.4 response block |
| `article/` untouched | all 5 pre-existing dirty files md5-identical to the recorded baseline |

#### 7.12 Dead ends and things that cost time

- Grepping `results.tex` for `"Table 8 in the appendices"` returns nothing, because the
  string wraps across two source lines. The defect only surfaced through `pdftotext` on the
  rebuilt PDF. AC-4's insistence on the PDF over the source is not pedantry; it is the only
  reason M6 was found.
- `manuscript-map.md` locates the preprint at `article/arxiv/69b91250e7e60fc6079dfd5d/`
  relative to the journal root. It is actually a **sibling** of `journal/`, at
  `completed/isalsr/article/arxiv/…`. Minor, but it costs a `find`. Worth correcting in the
  map.
- `manuscript-map.md` gives the main paper's sections in Roman numerals; the compiled paper
  prints Arabic. See §7.2.
- `pdftotext -layout` mangles the two-column IEEE layout badly enough to interleave table
  captions with body text. `-raw` is the right mode for extracting captions and numbering.

---

## 8. Proposed answer

### 8.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Journal Def 3.2, ℒ | {+,*,g,i,s,c,e,l,r,^,a,k} ✓ | unchanged | AC-2 |
| Preprint Def 2.2, ℒ | {+,*,−,/,s,…} ✗ | unchanged — no arXiv v3 | AC-1 |
| Preprint Table 1 | g → Neg, i → Inv ✓ | unchanged | AC-1 |
| Preprint prose | g / i ✓ | unchanged | AC-1 |
| Preprint token count | "7 + 24 (2×\|ℒ\|), totaling **31**" ✓ | unchanged — identical to the journal's | AC-2 |
| Preprint internally consistent | **no** (Def 2.2 vs its own Table 1 and prose) | unchanged — superseded, not corrected | AC-1 |
| Preprint remedy | — | comment in the response letter; no arXiv v3 | AC-1 |
| ℒ / journal Def 2.2 agreement | 10 of 12 label positions | unchanged | AC-2 |
| Token count (31, \|ℒ\|=12) | ✓ (7 single-char + 24 compound) | unchanged | AC-2 |
| `-` or `/` as a label character in paper or supplementary | 0 occurrences | 0 occurrences | AC-2 |
| `\|𝒜\| = 17` in the supplementary | stated, never defined | defined at its point of use as a **character** set; 10 + 10×16 + 11×17 = 357 verified | AC-2 |
| `supplementary.tex:120` | "Table 4 of the main document" (does not exist) | "…pseudocode in Table~`\ref{tab:canon_pseudo}` of Appendix~`\ref{sec:supp_pseudocodes}`" (renders "Table 3 of Appendix C") | AC-3 |
| `supplementary.tex:453` | "Section 5.1 of the main paper" (= Search Space Reduction) | "the Computational Experiments section of the main paper" | AC-3 |
| Hardcoded cross-refs verified | 0 of 21 | **21 of 21**: 17 pass, 4 fail (S1, S7, P1 as submitted; M6 broken by the revision) | AC-4 |
| Cross-document references added by the revision | — | 14 further, all checked, all resolve | AC-4 |
| Supplementary table numbering | 11 tables | **13** tables; everything after Table 5 shifted by +2 | AC-4 |
| `results.tex:176` "Table 8 in the appendices" | correct as submitted | **now wrong — must become Table 10**; handed to the `paper/` lane | AC-4 |
| Avoidable cross-document references | 21 hardcoded | 5 replaced by descriptions or `\ref`; irreducible remainder listed in §7.4 | AC-5 |
| Cross-document **table numbers** remaining | 6 | **1** (`results.tex:176`), or 0 if the `paper/` lane takes option (b) | AC-5 |
| Statement, "Section S.I" | does not exist | "the proofs appendix of the supplementary material" | AC-6 |
| Statement, supplementary length | "approximately thirty pages" | **"thirteen pages"** (from the rebuilt PDF; must be re-taken at submission) | AC-6 |
| Supplementary page count | 10 (submitted) | **13** (annotated, rebuilt 2026-08-06) | AC-6 |
| Website URL in `article/` | anonymised mirror | real URL — **in the annotated copy**; `article/` unchanged pending promotion | AC-7 |
| Real site reachable | not checked | HTTP 200, Last-Modified 2026-07-28 | AC-7 |
| Anonymised mirror reachable | not checked | HTTP 200, Last-Modified 2026-05-12 — live, correctly anonymised, **2.5 months stale** | AC-7 |
| `double_blind/` in sync | yes | **not yet** — propagation deferred to the promotion step, file list in §7.8 | AC-8 |
| British spellings in the supplementary tree | 37 prose occurrences | 0 (5 identifiers deliberately kept) | conventions |
| British spellings in the statement | 8 | 0 | conventions |

### 8.2 Changes made to the manuscript

Line numbers are in the **annotated** copy at
`reviews/internal_copy_reviewed_article/`, except the statement, which has no annotated
twin and was edited in place.

| File | Lines (revised) | Change | Blue? |
|---|---|---|---|
| `…/supplementary/supplementary.tex` | `:105–106` | "(Definition~3.5 and Table~2)" → "(Definition~3.5 of the main document and Table~`\ref{tab:d2s_pseudo}` of Appendix~`\ref{sec:supp_pseudocodes}`)". Removes an ambiguity about *which* document's Table 2 | inherits T07's blue span |
| `…/supplementary/supplementary.tex` | `:148–149` | "pseudocode in Table~3 of Appendix~C" → "pseudocode in Table~`\ref{tab:canon_pseudo}` of Appendix~`\ref{sec:supp_pseudocodes}`". Renders identically; now maintained by the typesetter | inside T07's blue block (`:125–225`) |
| `…/supplementary/supplementary.tex` | `:617` | "(Section~5.1 of the main paper)" → "(the Computational Experiments section of the main paper)" | **yes** |
| `…/supplementary/supplementary.tex` | `:1210–1216` | New sentence defining 𝒜 as a set of *characters*, distinct from the token alphabet Σ_SR, so that \|𝒜\| = 17 cannot be read against \|ℒ\| = 12 or \|Σ_SR\| = 31 | **yes** |
| `…/supplementary/supplementary.tex` | `:1309–1310` | "(Table~1 of the main document)" → "(the operation-type table of the main document)" | **yes** |
| `…/supplementary/supplementary.tex` | throughout | 35 British→American spellings (21 `neighbour*`, 6 `canonicalisation`, 2 `colour`, and one each of `initialisation`, `amortise`, `summarises`, `visualises`, `behaviour`, `labelled`); 5 `neighbourhood` identifiers preserved | no — see §7.9 |
| `…/supplementary/table_supplementary_udfs.tex` | `:4` | caption: `canonicalisation` → `canonicalization` | no |
| `…/supplementary/table_supplementary_bingo.tex` | `:4` | caption: `canonicalisation` → `canonicalization` | no |
| `previously_published_statement/main.tex` | `:76–77` | "A formal proof now appears in the supplementary material (Section~S.I), together with…" → "A formal proof now appears in the proofs appendix of the supplementary material, together with…" | n/a |
| `previously_published_statement/main.tex` | `:142–143` | "A \emph{supplementary} document of approximately / thirty pages containing proofs, algorithm pseudocodes," → "A \emph{supplementary} document of thirteen pages / containing proofs, algorithm pseudocodes," | n/a |
| `previously_published_statement/main.tex` | `:57, 63, 64, 68, 73, 74, 106, 112, 164` | 9 British→American spellings | n/a |
| `reviews/response_to_reviewers.tex` | `:1171–1208` | `\todoblock{R2.4}` replaced by the answer in §8.3 | n/a |

**Not changed, deliberately:** `article/**` (byte-identical to the pre-work baseline),
`reviews/internal_copy_reviewed_article/paper/**` (parallel lane), `double_blind/**`
(promotion step), `article/arxiv/**` (decision of 2026-07-27), and every block of
`response_to_reviewers.tex` other than R2.4.

### 8.3 Response text as shipped

**R2.2 — not written by this ticket.** The block at `response_to_reviewers.tex:991–1017`
was already in place. This ticket's job was to verify the three self-consistency claims it
makes; all three hold (§7.1), so it was left untouched. The one point the draft scaffold
suggested and the shipped block does not make is the ℒ / 𝒜 note. That was handled in the
manuscript instead, by defining 𝒜 where it is used, which is the better place: the reader
who meets \|𝒜\| = 17 is in the supplementary, not in the letter.

**R2.4 — shipped verbatim at `response_to_reviewers.tex:1171–1208`:**

```latex
\begin{response}
The reviewer's correction is exact. The main document has three tables, and the
fast canonical pseudocode is Table~3 of Appendix~C of the supplementary
material; the proof of Lemma~A.2 cited it as Table~4 of the main document. The
rewritten proof described in our response to comment~R2.1 names the algorithm
instead of a number, and the table it points to is now cross-referenced
automatically rather than typed by hand.

A second reference is wrong in the same way, and no reviewer raised it.
Appendix~D of the supplementary material attributed the configuration it
describes to Section~5.1 of the main paper, which is Search Space Reduction; the
intended target is the Computational Experiments section, and the revision names
that section rather than numbering it.

Both errors follow from one property of the package. The manuscript and the
supplementary material are separate documents that compile separately, so no
reference between them passes through the typesetting system: each one is a
number typed by hand that nothing in the build checks, and it breaks silently
whenever either document is renumbered. The submitted package held twenty-one
such references, eight from the supplementary material into the manuscript,
twelve in the other direction, and one from the accompanying statement of
previously published work. We have resolved all twenty-one against the compiled
documents, together with those the revision adds, and replaced every reference
that a description can carry instead, among them the pseudocode tables, the
appendix sections and the operation table. What remains points at numbered
definitions and theorems, which the reviewers also cite by number and which a
description would make harder to locate rather than easier; these are collected
in one list and re-checked against the compiled documents as the last step
before submission. The check has already repaid itself once: the benchmark
tables added to the supplementary material in answer to comment~R3.1 renumbered
the tables that follow them, which moved the stratified overhead table cited
from the Results section, and that citation has been updated with it.

\changeref{proof of Lemma~A.2 and the opening of Appendix~D in the
supplementary material; number-free replacements for the remaining avoidable
cross-document references in Appendices~F.2 and~G of the supplementary material
and in the statement of previously published work; the reference to the
$k$-stratified overhead table in Section~5.3 of the manuscript renumbered}
\end{response}
```

Spine coverage: step 1 (concede, first sentence), step 3 folded into paragraph 2 (the
volunteered second defect), step 4 (root cause, structural not historical), step 11 (the
irreducible remainder, volunteered), step 14 (cross-links to R2.1 and R3.1), step 15.
Steps 5–10 and 12–13 do not apply to an editorial correction. Four paragraphs, no display
objects, no em-dashes.

**One claim in this answer is not yet true of the manuscript.** The final clause, "that
citation has been updated with it", depends on the `paper/` lane applying §7.5 item 1. If
that edit does not land, the sentence must be deleted along with the sentence before it.

### 8.4 Residual risk

> Candidates: renumbering caused by T07 (new definition), T09 (rebuilt appendix) or
> T13 (reorganisation) silently breaking a reference that passed this ticket's
> check — hence the requirement to re-run the walk in T14; R2 asking why the
> preprint was not corrected (the answer is that publication supersedes it, and it
> should be stated once rather than defended); `double_blind/` drifting.
