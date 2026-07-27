# T10 — Claim calibration in the Discussion and Conclusion

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.1** (and E3, the "near-linear" overstatement) |
| Type | Framing / honesty of claims |
| Owner | **Mario** (`discussion.tex`, `results.tex`) · **Ezequiel** (`conclusion.tex`) |
| Depends on | T02 (the new `S`), T01 (§8.1 projection), T03 (if Gray changes cost) |
| Blocks | T12, T13 |
| Status | NOT STARTED |
| Target | 2026-09-10 |

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

_(empty — to be filled by the implementing agent)_

---

## 9. Proposed answer

### 9.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| `S`, Bingo | **0.93** | | T02 |
| `S`, UDFS | 1.07 | | T02 |
| Discussion characterisation | "approximately neutral" | | AC-2 |
| Break-even condition stated | no | | AC-3 |
| `T_c / T_e`, Bingo | 3.3 : 1 | | |
| `T_c / T_e`, UDFS | 1 : 64 | | |
| Break-even ρ, Bingo | ≈ 1.85 (implicit) | | AC-3 |
| Problems with `T_IS < T_BL`, Bingo | 4 / 50 (supplementary only) | | AC-4 |
| Median canon overhead, Bingo | 39.2 % | | |
| Complexity in conclusion | "near-linear" | | AC-6 |
| Complexity in discussion | "near-linear" | | AC-6 |
| Complexity in introduction | near-O(k²) ✓ | unchanged | |
| Complexity in related work | near-quadratic ✓ | unchanged | |
| Relationship of O(k^0.7) to O(k²) | unstated | | AC-6 |
| Answer chosen | — | A / B | §4 |

### 9.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| `article/paper/discussion.tex` | | |
| `article/paper/conclusion.tex` | | |
| `article/paper/results.tex` | | |

### 9.3 Draft response text

```latex
%% --- R1.1 ---
\begin{response}
%% Structure that works here:
%%  1. Accept the criticism without qualification in the first sentence. The
%%     reviewer is right; "approximately neutral" was not supportable at S = 0.93.
%%  2. Answer A: state the regime characterisation and the break-even condition,
%%     and give the 4-of-50 count. Do not bury it in the supplementary.
%%     Answer B: give the new S, say plainly that the earlier figure reflected a
%%     Python implementation, and give the break-even condition anyway so the
%%     result generalises past our own engine.
%%  3. Note that the Results section was already conditional and that the fix
%%     brings the Discussion into line with it, rather than weakening a result.
%%  4. Volunteer E3 here as well: the same over-reach affected the complexity
%%     claim ("near-linear" in the discussion and conclusion versus near-O(k^2)
%%     everywhere else). The reviewer's own B2 statement uses near-O(k^2), so
%%     they will notice; conceding it unprompted is cheap and reads well.
\changeref{}
\end{response}
```

### 9.4 Residual risk

> Candidates: if Answer B is used, a reviewer asking whether the engine change is
> what produced the improvement rather than the method (it is — say so plainly, and
> keep the break-even formula so the claim is engine-independent); the break-even
> formula not matching the code's `S` definition; a remaining unconditional claim
> in a section not swept.
