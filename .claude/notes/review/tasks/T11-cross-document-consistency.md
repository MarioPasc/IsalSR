# T11 — Cross-document and package consistency

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.2**, **R2.4** (and E5, E7) |
| Type | Cross-reference / package hygiene |
| Owner | **Karl** (primary — the sweep) · **Ezequiel** (R2.2, the preprint and `methodology.tex`) |
| Depends on | — (can start immediately) |
| Blocks | T13, T14 |
| Status | NOT STARTED |
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

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Journal Def 3.2, ℒ | {+,*,g,i,s,c,e,l,r,^,a,k} ✓ | unchanged | AC-2 |
| Preprint Def 2.2, ℒ | {+,*,−,/,s,…} ✗ | | AC-1 |
| Preprint Table 1 | g → Neg, i → Inv ✓ | unchanged | |
| Preprint prose | g / i ✓ | unchanged | |
| Preprint internally consistent | **no** | unchanged — superseded, not corrected | AC-1 |
| Preprint remedy | — | comment in the response letter; no arXiv v3 | AC-1 |
| Token count (31, \|ℒ\|=12) | ✓ | unchanged | AC-2 |
| `supplementary.tex:120` | "Table 4 of the main document" (does not exist) | | AC-3 |
| `supplementary.tex:453` | "Section 5.1 of the main paper" (wrong section) | | AC-3 |
| Hardcoded cross-refs verified | 0 of 21 | 21 of 21 | AC-4 |
| Statement, "Section S.I" | does not exist | | AC-6 |
| Statement, supplementary length | "approximately thirty pages" | 10 (or final) | AC-6 |
| Website URL in `article/` | anonymised mirror | real URL | AC-7 |
| `double_blind/` in sync | yes | yes | AC-8 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

```latex
%% --- R2.2 ---
\begin{response}
%% Structure that works here:
%%  1. Confirm the observation, then give the fact the reviewer could not have
%%     had: the journal manuscript is self-consistent -- Definition 3.2, Table 1
%%     and the prose all use g and i. The divergence is a typo in one line of the
%%     preprint's Definition 2.2, which contradicts the preprint's own Table 1 and
%%     its own prose.
%%  2. State plainly that the published journal version supersedes the preprint,
%%     so the discrepancy resolves on publication and no preprint revision is
%%     issued. Keep this to one sentence -- it is a comment, not a commitment.
%%  3. Note the ell / A notation point if it was judged confusing, since the
%%     reviewer will meet |A| = 17 in the supplementary and may connect it to the
%%     alphabet-size discussion.
\changeref{}
\end{response}

%% --- R2.4 ---
\begin{response}
%%  1. Confirm and fix; the reviewer's correction is exact.
%%  2. Volunteer the second broken reference they did not find
%%     (supplementary.tex:453, "Section 5.1 of the main paper"), and state that
%%     all 21 hardcoded cross-document references have now been verified, with
%%     avoidable ones replaced by number-free descriptions.
%%  3. Naming the root cause -- the two documents compile separately, so every
%%     cross-reference was a manually typed number -- turns a typo into a
%%     process fix, which is what makes the answer credible.
\changeref{}
\end{response}
```

### 8.4 Residual risk

> Candidates: renumbering caused by T07 (new definition), T09 (rebuilt appendix) or
> T13 (reorganisation) silently breaking a reference that passed this ticket's
> check — hence the requirement to re-run the walk in T14; R2 asking why the
> preprint was not corrected (the answer is that publication supersedes it, and it
> should be stated once rather than defended); `double_blind/` drifting.
