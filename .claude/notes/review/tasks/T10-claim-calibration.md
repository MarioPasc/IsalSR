# T10 — Claim calibration in the Discussion and Conclusion

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.1** (and E3, the "near-linear" overstatement) |
| Type | Framing / honesty of claims |
| Owner | **Mario** (`discussion.tex`, `results.tex`) · **Ezequiel** (`conclusion.tex`) |
| Depends on | T02 (the new `S`), T01 (§8.1 projection), T03 (if Gray changes cost) |
| Blocks | T12, T13 |
| Status | **IN PROGRESS — text half done, numbers pending T02.** The R1.1 answer is written into `reviews/response_to_reviewers.tex` with every campaign-dependent value wrapped in a new `\pendingnum{}` macro (7 placeholders, listed in §8). E3 is **complete**: "near-linear" is gone from `discussion.tex` and `conclusion.tex` in the annotated copy, and the relation between the O(k^0.7) fit and the near-O(k²) bound is stated once. "Approximately neutral" is gone. AC-2, AC-4, AC-5, AC-6, AC-7, AC-8 met; AC-0 met; **AC-1 met with a negative result** (§8: the §4 formula does **not** follow from the code's `S`); **AC-3 blocked on T02** and re-scoped away from the formula. |
| Target | 2026-09-10 |

---

## T16 impact — the cost side of `S` moves against us (added 2026-07-30)

T16 aligned the adapters to the paper's alphabet, so `Sub`/`Div` are now emitted as
`Add`+`Neg` / `Mul`+`Inv`. **Canonicalisation gets more expensive**, because the DAGs
are bigger: `k` +22.9 % (Bingo) / +22.0 % (UDFS), and per-DAG canonicalisation cost
**+24.6 %** (Bingo) / **+10.8 %** (UDFS) on a random-population probe.

**Why this matters here specifically.** R1.1's complaint is about `S = T_BL / T_IS`
being framed as "approximately neutral" at 0.93. The alphabet correction pushes the
IsalSR arm's canonicalisation cost **up**, so the honest expectation is that `S` does
not improve on this account and may worsen. **Do not write the calibration text
against the old cost numbers, and do not assume the C++ port's speedup and the
alphabet's cost increase cancel** — they are independent and both land in Wave 1.
Wait for the measured `S`.

The offsetting consideration, which is real but must not be oversold: on Bingo the
reduction factor was **exactly** invariant under the change (3,858 distinct strings
in both encodings), and on UDFS it *rose* 1.4 % because host-native `neg`/`inv` now
unify with decomposed ones. So the representation got strictly better at its job
while costing more per DAG. That is a trade to state plainly, not a win to claim.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.

---

## 1. Why R1.1 and E3 are grouped

Same defect, same two files, same fix. In both cases **the Results and Introduction
state the claim correctly and the Discussion and Conclusion overstate it.** The
reviewer's complaint in R1.1 is precisely that mismatch, and E3 is the identical
mismatch on a different quantity that no reviewer happened to name.

| Claim | Stated correctly in | Overstated in |
|---|---|---|
| Bingo wall-clock effect | `results.tex:180–185` (conditional, per-problem) | `discussion.tex:66–68` ("approximately neutral") |
| Canonicalisation complexity | `introduction.tex:51–52`, `related_work.tex:81–82` (near-O(k²)) | `conclusion.tex:9–11`, `discussion.tex:97` ("near-linear") |

Fixing them together means one pass over the late sections with one rule: **no
section may claim more than the Results section supports.**

E3 carries an extra hazard. **Reviewer 1's own B2 significance statement uses
"near-O(k^2)"** — they have already written the correct complexity back to us. A
"near-linear" claim surviving into the revision reads as inconsistent with what the
reviewer themselves recorded, on a point they clearly understood.

---

## 2. Verbatim comment

> 1) The bingo search-only speedup is s = 0.93, a net loss under a fixed wall-clock
> budget, yet the paper characterizes the overhead as "approximately neutral." This
> needs a more honest framing. The claim holds only on the subset of problems where
> rho is large enough to compensate.

---

## 3. Established facts

### 3.1 R1.1
- `S = 0.93` for Bingo, `S = 1.07` for UDFS — `results.tex:57–58`, Table 2.
- The objectionable phrasing, `discussion.tex:66–68`:
  > When evaluation cost is comparable to canonicalisation cost, as in Bingo where
  > both are sub-millisecond, the median overhead is 39 % and the wall-clock effect
  > is **approximately neutral** after the search-time savings offset the
  > canonicalisation cost.
- The Results section is **already** properly conditional, `results.tex:180–185`:
  > on the 30 problems where Bingo exceeds ρ = 1.85 the search-only speedup recovers
  > to S ≥ 0.95, and on three problems — N-8, II.11.27, and Keijzer-11 — it exceeds
  > unity.
- Per-problem corroboration, `table_supplementary_bingo.tex`: only **4 of 50** rows
  have `T_IS < T_BL` (I.6.20a, II.11.27, Keij-11, N-8). The other 46 are slower under
  IsalSR. Highest overheads are Nguyen rows: N-12 61.3 %, N-5 60.5 %, N-4 59.6 %,
  Keij-6 60.3 %, N-3 58.2 %, N-7 56.8 %.
- The **abstract is accurate** (`main.tex:81`, "at a median of 39 % in the
  sub-millisecond evaluation regime"). The overstatement is localised to the discussion.

### 3.2 E3 — three coexisting growth claims
`methodology.tex:885–887` derives O(k) insertion steps × O(k) candidates = **O(k²)**.
`supplementary.tex:736, 793` fits **O(k^0.7)** *per permutation* — a per-call cost,
not the whole-DAG canonicalisation cost. `discussion.tex:22–24` infers "near-linear
behaviour" from median per-DAG times. Three different growth claims coexist with no
stated relationship between them. Fixing E3 means stating that relationship once,
not just deleting the word "near-linear".

---

## 4. Two possible answers — prepare both, choose on T02's data

**Answer A — the concession (if `S` stays below 1 for Bingo).**
Accept the criticism, delete "approximately neutral", and replace it with the
regime characterisation the Results section already contains: IsalSR's wall-clock
benefit is realised when `T_eval / T_canon` is large (UDFS: ≈ 64:1, `S = 1.07`) and
is a net cost when the two are comparable (Bingo: ≈ 1:3.3, `S = 0.93`); the
crossover is at ρ ≈ 1.85 given the measured cost ratio. **State the break-even
condition as a formula**, not as a list of problems. A reader can then predict
whether IsalSR pays on their own solver, which is more useful than the original
sentence and is a genuine contribution rather than a retreat.

**Answer B — the fix (if the C++ engine pushes Bingo's `S` above 1).**
Report the new `S`, state that the previous figure reflected a Python
implementation, and *still* give the break-even formula — because the regime
characterisation is what makes the result generalise, and because dropping it would
look like the concession was avoided rather than earned.

**In both cases the break-even analysis is the deliverable.** Do not write Answer B
without it.

Derive it explicitly. With `ρ` the reduction factor, `T_e` the mean evaluation cost
and `T_c` the mean canonicalisation cost, the search-only speedup is

```
S  =  ρ / (1 + ρ · T_c / T_e)          ⇒     S > 1  ⟺  ρ > 1 + ρ·T_c/T_e
```

Verify this against the definition of `S` actually used in the analysis code
(`models/analyzer/`) before publishing it — do not reproduce the formula above on
trust. Then report the measured `T_c/T_e` per method and the implied break-even ρ.

---

## 5. Mandatory reading

- `.claude/notes/review/source/reviewer-1.md` — §R1.1 and the **full B2 statement**
- `.claude/notes/review/source/verified-discrepancies.md` — D11, E3
- `.claude/notes/review/tasks/T01-cpp-core-port.md` — §8.1, the projected `S`
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — §8.1, the measured `S`
- `.claude/notes/review/tasks/T03-gray-code-integration.md` — if promoted, cost changes again
- `docs/md_files/changes/bottleneck_type_analysis.md` — the existing regime analysis;
  reuse its framing rather than inventing a second one
- Source: `article/paper/{results,discussion,conclusion}.tex`, `main.tex:81`

---

## 6. Work specification

1. Verify the `S` definition in the analysis code and derive the break-even
   condition from it.
2. Compute `T_c/T_e` and the break-even ρ per method under the C++ engine.
3. Rewrite `discussion.tex:66–68`. Choose Answer A or B on the data; keep the
   break-even statement either way.
4. Sweep for every other unconditional wall-clock claim in `discussion.tex` and
   `conclusion.tex`.
5. Fix E3: replace "near-linear" at `conclusion.tex:9–11` and `discussion.tex:97`
   with the derived near-O(k²); reconcile `discussion.tex:22–24`; state once how the
   per-permutation O(k^0.7) fit relates to the whole-DAG O(k²) bound.
6. Add the per-problem count (`x of 50` rows with `T_IS < T_BL`) to the discussion.
   It is the single most direct answer to "the claim holds only on the subset of
   problems where rho is large enough" and it is currently only inferable from the
   supplementary tables.

---

## 7. Acceptance criteria

- **AC-0.** §8 Work log filled in as the work proceeds.
- **AC-1.** `S` definition verified against code; break-even condition derived and
  checked, not assumed.
- **AC-2.** "Approximately neutral" is gone.
- **AC-3.** Break-even condition stated in the paper as a condition on ρ and
  `T_c/T_e`, with the measured values per method.
- **AC-4.** The count of problems where IsalSR is wall-clock faster appears in the
  main text.
- **AC-5.** No unconditional wall-clock claim survives in `discussion.tex` or
  `conclusion.tex`.
- **AC-6.** "Near-linear" is gone from both locations; the relationship between
  O(k^0.7), the observed medians, and the O(k²) bound is stated once.
- **AC-7.** The revised claims are checked against the abstract and the introduction
  for consistency — those two are currently correct and must stay correct.
- **AC-8.** §9 filled.

---

## 8. Work log

### 2026-08-06 — `S` verified against the code; §4's break-even formula does not survive it

**AC-1, and it came out negative.** The formula proposed in §4,

```
S  =  ρ / (1 + ρ · T_c / T_e)
```

is **not** what the code computes, and it must not be published. What the code
computes, and what the manuscript defines, is a ratio of two *measured* times:

| Where | Statement |
|---|---|
| `computational_experiments.tex:186–195` | `T_search = T_total − T_canon`; baseline has `T_canon = 0`; `S = T_search^BL / T_search^IS` |
| `experiments/models/bingo/translator.py:116–124` | `search_only = wall_clock − canonicalization_time − conversion_time − shadow_time` |
| `experiments/models/udfs/translator.py:116–124` | identical |
| `experiments/models/analyzer/aggregation.py:397–398` | `speedups.append(m.baseline_mean / m.isalsr_mean)`, guarded to time metrics |

So `S` is an empirical ratio of realised search times, not a model. Two
consequences, both of which changed what got written:

1. **Under an exhausted, equal wall-clock budget `S > 1` is forced.** If both arms
   run for the same `T`, then `T_search^BL = T` and `T_search^IS = T − T_canon`, so
   `S = T/(T − T_canon) ≥ 1` identically. Bingo's `S = 0.93` is therefore only
   possible because Bingo terminates on convergence rather than on the budget
   (`evolve_until_convergence(max_time=…)`), so the two arms stop at different wall
   clocks. The supplementary per-problem table corroborates this: Bingo's `T_BL`
   and `T_IS` differ per row and sit far below 43,200 s on many problems, while
   every UDFS row is pinned at the budget.
2. **The submitted discussion sentence was wrong in a second way**, beyond the
   adjective. "The wall-clock effect is approximately neutral *after the
   search-time savings offset the canonicalisation cost*" describes an end-to-end
   quantity, but `S` has the canonicalisation cost **already removed** from the
   IsalSR arm before the ratio is taken. `S < 1` therefore says something sharper
   than "the overhead was not recovered": with the overhead discounted, the
   deduplicated search still needed more time. That is the reading the letter now
   uses, and it is the strongest honest form of the concession.

Any correct break-even model would need an extra assumption about the termination
rule (budget-exhausted versus convergence-terminated), and would give a different
expression in each case. **§4's derivation and §6 step 1–2 are therefore
retired**; AC-3 is re-scoped from "state the formula" to "state the condition in
terms of the two measured quantities `ρ` and `T_eval/T_canon`, and report the
measured crossover", which is what the letter and the discussion now do. This is
exactly the risk §9.4 anticipated.

### 2026-08-06 — R1.1 written with visible placeholders; E3 closed

**R1.1.** Written into `reviews/response_to_reviewers.tex`, replacing
`\todoblock{R1.1}`. A macro was added to the letter preamble next to `\todoblock`:

```latex
\newcommand{\pendingnum}[1]{{\color{todored}\textsf{[PENDING: #1]}}}
```

Seven placeholders, all campaign-dependent, all rendered red:

| # | Awaits |
|---|---|
| 1 | median per-DAG `T_eval/T_canon`, Bingo |
| 2 | median canonicalisation overhead as % of total run time, Bingo |
| 3 | search-only speedup `S`, Bingo, suite-level value for Table 2 |
| 4 | median per-DAG `T_eval/T_canon`, UDFS |
| 5 | search-only speedup `S`, UDFS, suite-level value for Table 2 |
| 6 | the ρ range over which Bingo's `S ≥ 1`, **or** an explicit statement that no such range occurs |
| 7 | count of suite problems where Bingo's IsalSR arm finishes in less total wall-clock time, with the suite size |

Placeholder 6 is deliberately phrased so it cannot be filled with a
direction-assuming number. The answer contains **no table**: a table of pending
cells is worse than none.

Two values are quoted outright because they are facts about the **submitted**
manuscript rather than about the campaign, and both are traced to §3.1: `S = 0.93`
(which the reviewer quoted) and `4 of 50` Bingo problems with `T_IS < T_BL`. Both
stay true whichever way the re-execution moves.

The answer also discloses, unprompted, that the submitted numbers came from a
Python canonicaliser and from the pre-T16 encoding, that the compiled engine
lowers the per-canonicalisation cost while the commutative normal form raises it,
and that we do not predict the net effect. It volunteers the protocol limitation
(equal wall-clock budgets; an evaluation-count budget would report a different
number) without claiming which direction that would move.

**E3 — complete, no campaign number needed.** Three occurrences in the annotated
copy:

| Location | Was | Now |
|---|---|---|
| `conclusion.tex:11` | "in near-linear time" | "in near-$O(k^2)$ time", blue |
| `discussion.tex:97` | "runs in near-linear time" | "runs in near-$O(k^2)$ time", blue |
| `discussion.tex:21–26` | two median timings "indicate near-linear behavior" | the medians now evidence that the greedy path is taken almost always; growth attributed to the near-$O(k^2)$ analysis of `Section~\ref{sec:canonical}`, blue |

The relation is stated once, at `discussion.tex:100–107`: the `O(k^0.7)` power law
is a log--log fit over `k ≤ 9` on synthetic DAGs, so it characterises the
benchmarked range and not the asymptotics, and it sits below the `O(k^2)` bound
because backtracking seldom fires at those sizes.

`related_work.tex:70` also contains "near-linear", but it describes **1-WL
hashing's ability to distinguish graphs**, not our canonicalisation cost. It is
correct and was left alone. `introduction.tex:82` and `related_work.tex:81` were
already correct.

**AC-5 sweep.** The one surviving unconditional wall-clock claim was
`discussion.tex:72–74`, "at neutral wall-clock cost". It is gone. `conclusion.tex`
makes no wall-clock claim.

**Verification.** Letter: two `pdflatex` passes, exit 0 both, `Overfull` 0,
unresolved references 0. Annotated paper: three passes, `^!` 0, undefined
references 0, `color{red}` 0, review notes 0; Theorems 3.13/3.14/3.15 keep their
numbers. `article/` untouched.

**One warning is not ours.** The letter's log carries
`LaTeX Warning: Float too large for page by 3.64752pt` on the summary-of-changes
table, which a parallel agent was editing during this session. `git show HEAD` of
the letter compiles with zero warnings, and nothing in the R1.1 block or the
preamble macro enters that float. Owner of that table needs to shorten a cell.

---

## 9. Proposed answer

### 9.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| `S`, Bingo | **0.93** | `\pendingnum` 3 | T02 |
| `S`, UDFS | 1.07 | `\pendingnum` 5 | T02 |
| Discussion characterisation | "approximately neutral" | **removed**; net cost stated outright, regime given as a joint condition on ρ and `T_eval/T_canon` | AC-2 ✅ |
| Break-even condition stated | no | **yes, as a condition on the two measured ratios** — not as §4's formula, which the code refutes (§8) | AC-3, re-scoped |
| `T_c / T_e`, Bingo | 3.3 : 1 | `\pendingnum` 1 | T02 |
| `T_c / T_e`, UDFS | 1 : 64 | `\pendingnum` 4 | T02 |
| Break-even ρ, Bingo | ≈ 1.85 (implicit) | `\pendingnum` 6, phrased so "no such range" is a legal answer | AC-3 |
| Problems with `T_IS < T_BL`, Bingo | 4 / 50 (supplementary only) | **now in the main text**, `discussion.tex`; re-executed value is `\pendingnum` 7 | AC-4 ✅ |
| Median canon overhead, Bingo | 39.2 % | `\pendingnum` 2 | T02 |
| Complexity in conclusion | "near-linear" | near-$O(k^2)$ | AC-6 ✅ |
| Complexity in discussion (×2) | "near-linear" | near-$O(k^2)$ | AC-6 ✅ |
| Complexity in introduction | near-O(k²) ✓ | unchanged | AC-7 ✅ |
| Complexity in related work | near-quadratic ✓ | unchanged | AC-7 ✅ |
| Relationship of O(k^0.7) to O(k²) | unstated | stated once, `discussion.tex:100–107` | AC-6 ✅ |
| Answer chosen | — | **A**, and structured so it stays true if T02 delivers B | §4 |

Answer A was chosen not because the data favoured it but because the data does not
exist yet. The text concedes, characterises the regime, and defers only the
numbers; if T02 returns `S > 1` for Bingo, filling the placeholders converts it
into Answer B without rewriting a sentence.

### 9.2 Changes made to the manuscript

Applied to `reviews/internal_copy_reviewed_article/`, in blue. **Not** applied
under `article/` and **not** pushed to Overleaf.

| File | Lines (revised) | Change |
|---|---|---|
| `paper/discussion.tex` | 21–31 | the growth-rate inference from two median timings withdrawn; the medians reattributed to backtracking frequency; growth pointed at `Section~\ref{sec:canonical}` |
| `paper/discussion.tex` | 65–82 | "approximately neutral" removed; net wall-clock cost stated with `S = 0.93` and 4 of 50; sign of `S − 1` given as the joint condition on ρ and `T_eval/T_canon`; "at neutral wall-clock cost" replaced |
| `paper/discussion.tex` | 100–113 | "near-linear" → near-$O(k^2)$; the `O(k^{0.7})` fit qualified as a range description, with why it sits below the bound |
| `paper/conclusion.tex` | 10–11 | "near-linear" → near-$O(k^2)$ |
| `paper/results.tex` | — | untouched; it was already conditional, which is the whole point of the fix |
| `reviews/response_to_reviewers.tex` | preamble, R1.1 | `\pendingnum` macro added; R1.1 answer written |

### 9.3 Response text — WRITTEN

In `reviews/response_to_reviewers.tex`, replacing `\todoblock{R1.1}`. Nine
paragraphs of prose, no display objects, no table, seven `\pendingnum{}`
placeholders.

The sketch above was followed except on one point. It planned to "state the
break-even condition as a formula"; the formula is refuted by the code (§8), so
the letter states the condition in terms of the two quantities that are actually
measured, `ρ` and `T_eval/T_canon`, and defers the crossover to a placeholder.

R1.1 runs: concede at `S = 0.93` and remove the phrase; say what `S` actually
measures and show that the submitted sentence described a quantity `S` does not
report; locate the defect as the mismatch between our own results and discussion
sections; give the regime as a joint condition on `ρ` and `T_eval/T_canon`;
disclose that the numbers are being re-measured, why, and that the engine change
and the alphabet correction push the cost in opposite directions; give the count
of problems where IsalSR wins end to end, with the submitted 4 of 50 stated
outright; volunteer the equal-wall-clock-budget limitation; volunteer E3
unprompted, including that R1's own significance statement already used
near-$O(k^2)$; `\changeref` to Section 6 and Section 7.

The answer deliberately contains **no** "no reported number changes" sentence.
Numbers do change here, and claiming otherwise would be the same failure mode the
comment is about.

### 9.4 Residual risk

- **The placeholders must not ship.** Seven of them, all red. Whoever closes T02
  fills them from the campaign and re-reads §9.1 for the two hard-coded submitted
  values (`0.93`, `4 of 50`), which must then be restated as "in the submitted
  manuscript" or removed if the surrounding sentence no longer needs them.
- **Anticipated and realised**: "the break-even formula not matching the code's
  `S` definition" (§9.4 as originally written). It did not match. Handled in §8.
- If T02 returns `S > 1` for Bingo, a reviewer may ask whether the engine change
  rather than the method produced it. The letter already answers this: it states
  the engine change and the alphabet correction up front and declines to predict
  their net effect, so the improvement is never attributed to the representation.
- `discussion.tex:65–82` still carries `S = 0.93`, `39\%`, `1.07`, `34\%`, `1.83`,
  `1.56` and "the $50$ problems". These are the submitted campaign's numbers and
  are T02/T09's to refresh **as a block**; the calibration change is orthogonal to
  them and must not be read as having endorsed them.
- `conclusion.tex` is listed as Ezequiel's in the header. The E3 edit there is one
  hyphenated complexity claim, made to keep the letter's statement true; flag it to
  him rather than assuming silent consent.
