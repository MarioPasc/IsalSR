# T05 — Bounded benchmark extension: Feynman remainder + ODE-Strogatz

| Field | Value |
|---|---|
| Reviewer comments closed | **R3.1** (and B4 for all three reviewers) |
| Type | Justification + new experiment |
| Owner | **Mario** (+ Claude Code) for the experiments · **Karl** for the written justification |
| Depends on | T02 (protocol and infrastructure) |
| Blocks | T09 (Appendix D tables), T13 (page budget) |
| Status | NOT STARTED |
| Target | problem definitions 2026-08-17 · results 2026-09-08 |

---

## 1. Why this is its own ticket

R3.1 is the only substantive comment from the reviewer who rated the paper
*Excellent*, and it is phrased as a question — *"is there a reason why…"* — which
invites a justification as much as new runs. **Decision taken 2026-07-27: do both.**
Justify the exclusions quantitatively, and run a bounded extension so the answer is
not purely rhetorical.

**Verbatim comment:**

> The authors conducted verification using 50 problems, but is there a reason why
> they did not use the other databases mentioned in section 2.4? I believe the paper
> would be better if verification results obtained using these databases were
> included.

Section 2.4 (`related_work.tex:84–94`) names exactly two other databases:
**AI Feynman** (120 equations; the suite uses 24) and **SRBench / PMLB** (250+
problems; the suite uses none).

---

## 2. The three-way tension this ticket must resolve

| Reviewer | Position | Implication |
|---|---|---|
| R3 | wants broader coverage | add problems |
| R1 | *"The empirical protocol is rigorous (50 problems, 30 seeds, Demsar-style paired inference)"* — explicitly endorses the current protocol | do not disturb it |
| R2 | C6: *"Should be trimmed a bit"* | the paper cannot get longer |

Resolution: **extend the evidence, not the exposition.** New problems go into the
existing tables and the existing statistical machinery; the main text gains one
paragraph and a revised `N`. The per-problem detail goes to the supplementary.
Coordinate the page cost with T13.

---

## 3. The justification half — what to establish quantitatively

The paper already has four pre-declared inclusion criteria
(`computational_experiments.tex:56–76`). The response must show they *bind*, with
counts, not merely restate them.

**SRBench / PMLB.** The 250+ figure is dominated by the **black-box track**:
real-world regression datasets with **no ground-truth expression at all**. Criterion
(i) (published provenance of the expression and sampling protocol) and criterion
(iii) (published evidence that difficulty is structural rather than constant-fitting)
are *undefined* on those datasets, and solution recovery — one of the paper's
reported metrics — cannot be computed. They are out of scope by construction, and
that is a clean, principled answer rather than an excuse.

But SRBench's **ground-truth track** is a different object: it is essentially
AI Feynman plus the 14 **ODE-Strogatz** datasets (La Cava et al., built from
Strogatz's nonlinear-dynamics problems). Strogatz passes all four criteria and is
not currently represented. **Including it lets the response say the paper now covers
SRBench's ground-truth track**, which is the part of SRBench the paper's metrics are
defined on.

**AI Feynman.** Criterion (ii) genuinely excludes a substantial fraction —
equations requiring `arcsin`, `arccos`, `arctan`, `tanh` are outside Σ_SR. The
agent must **count them**: of the 120 equations, how many are Σ_SR-representable,
how many of those are already in the 24, and how many remain. That count is the
quantitative core of the answer to R3.1 and it does not currently exist anywhere.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-3.md` — the whole file, including the note
  that R3's B2 credits the paper with the *preprint's* intrinsic-property experiments
- `.claude/notes/review/source/reviewer-1.md` — the opening assessment (protocol endorsement)
- `.claude/notes/review/source/reviewer-2.md` — C6, and R2.5 on the undocumented 28 problems
- `.claude/notes/review/source/codebase-pointers.md` — note on the existing
  `experiments/configs/srbench.yaml` and why it must **not** be used as-is
  (`n_runs: 10`, and an operator set that does not match Appendix D.2)
- `docs/md_files/changes/candidate_problem_screening.md` — the existing screening
  methodology across 8 suites; **reuse it, do not reinvent it**
- `docs/md_files/changes/hard_problem_selection_rationale.md`
- `docs/md_files/changes/roundoff_problem_selection.md` — the template for a
  problem-selection rationale document
- `docs/md_files/changes/cross_problem_dominance_test.md` — CPDT and its N-dependence
- `docs/md_files/design/experimental_design/data_benchmarking_design.md` — sampling protocols
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md`

---

## 5. Statistical hazard — pre-register the selection

CPDT is the paper's primary significance metric and its p-value **decreases
monotonically with N** when δ_i ≥ 0. Adding ~20 problems moves N from 50 to ~70 and
therefore strengthens the headline statistic *by construction*. That is a real
benefit and worth stating — **but only if the selection is defensible.**

Requirements:

1. **Pre-register the selection rule before any run.** Write the rule and the
   resulting problem list into
   `docs/md_files/changes/r31_extension_selection.md`, commit it, and record the
   commit hash in §7 **before** launching. The rule must be mechanical: "all
   Strogatz problems" + "all AI Feynman equations satisfying criterion (ii) and not
   already present, capped at K by <stated tie-break>".
2. **No screening for expected IsalSR advantage.** The cherrypicked tier already
   exists and is disclosed as such; this extension must be the opposite — coverage
   driven, outcome blind. Selecting for advantage here would be indefensible and a
   reviewer who reads `candidate_problem_screening.md` would see it.
3. **Report CPDT at both N = 50 and N ≈ 70.** If the extension weakens the result,
   report that. It is the honest outcome and it is far cheaper than being caught.

---

## 6. Work specification

1. **Count and classify** all 120 AI Feynman equations against criterion (ii).
   Produce a table: representable / not representable, with the blocking operator
   named for each exclusion. This table is the answer to R3.1 whatever else happens.
2. **Classify SRBench**: ground-truth track vs black-box track, with counts, and the
   criteria that fail on the black-box track.
3. **Define the extension set**: 14 ODE-Strogatz + the Feynman remainder, per the
   pre-registered rule. Target ≈ +20 problems.
4. **Implement** as a new suite `benchmarks/datasets/strogatz.py` (+ Feynman
   additions), following the structure of `roundoff.py` / `cherrypicked.py`:
   target functions, sampling protocol, `sympy_expression` for solution recovery,
   and a unit-test file mirroring `tests/unit/test_roundoff_benchmarks.py`.
5. **Configs and launcher** following the `roundoff_launch.sh` pattern. Use the
   `picasso-sbatch` skill. Do **not** reuse `srbench.yaml`.
6. **Run** at the same protocol as T02 (12 h budget, 30 seeds, both methods, all
   variants active at that point — including the T04 `hash` arm if it exists).
7. **Analyse**: full pipeline, CPDT at both N values, and a per-tier breakdown so a
   reader can see the extension did not simply dilute the suite.

---

## 7. Acceptance criteria

- **AC-0.** §8 Work log filled in as the work proceeds.
- **AC-1.** AI Feynman criterion-(ii) classification table complete for all 120
  equations, with the blocking operator named per exclusion.
- **AC-2.** SRBench track classification with counts and the criteria that fail.
- **AC-3.** Selection rule pre-registered and committed **before launch**; commit
  hash recorded in §8.
- **AC-4.** New benchmark module(s) implemented with unit tests passing; ground-truth
  sympy expressions present so solution recovery is computable for every added problem.
- **AC-5.** Campaign complete or every missing run accounted for.
- **AC-6.** CPDT reported at N = 50 and N ≈ 70, per method, per metric.
- **AC-7.** Per-tier breakdown produced; the extension's effect on the headline is
  stated without softening, including if it is negative.
- **AC-8.** Every added problem is documented in the revised Appendix D.1 with
  expression, dimensionality, range, sampling protocol and citation — coordinate
  with T09, which is fixing exactly this gap for the existing 28 undocumented problems.
- **AC-9.** §9 filled.

---

## 8. Work log

_(empty — to be filled by the implementing agent)_

---

## 9. Proposed answer

### 9.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Problems in suite | 50 | | |
| AI Feynman equations used | 24 of 120 | | |
| AI Feynman Σ_SR-representable (of 120) | not reported | | AC-1 |
| AI Feynman excluded by criterion (ii) | not reported | | AC-1 |
| SRBench ground-truth track covered | 0 | | AC-2 |
| SRBench black-box track covered | 0 | 0 (out of scope, criteria i & iii) | AC-2 |
| ODE-Strogatz problems | 0 | | |
| Total runs | 6,000 | | |
| CPDT N | 50 | | |
| CPDT R² test p, UDFS | 0.00018 | | at N = 50 and N ≈ 70 |
| CPDT R² test p, Bingo | 0.0013 | | at N = 50 and N ≈ 70 |
| CPDT reduction-factor p, UDFS | ≈ 0 | | |
| CPDT reduction-factor p, Bingo | ≈ 0 | | |
| ρ, UDFS | 1.56 ± 0.24 | | |
| ρ, Bingo | 1.83 ± 0.09 | | |
| Problems where dedup fires | 50 / 50 | | R1 called this "clean and credible" |
| Problems documented in Appendix D.1 | 22 of 50 | | with T09 |

### 9.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 9.3 Draft response text

```latex
%% --- R3.1 ---
\begin{response}
%% Structure that works here:
%%  1. Thank R3 and answer the literal question first: yes, there is a reason, and
%%     here it is with numbers -- the criterion-(ii) count over all 120 Feynman
%%     equations, and the fact that SRBench's 250+ is dominated by a black-box
%%     track on which criteria (i) and (iii) and the solution-recovery metric are
%%     undefined.
%%  2. Then say we did not stop at the justification: the suite now includes the
%%     Feynman remainder and the 14 ODE-Strogatz problems, i.e. SRBench's
%%     ground-truth track.
%%  3. Give the new N and the CPDT at both N values. Note that the selection rule
%%     was pre-registered and outcome-blind -- state this, it pre-empts the
%%     obvious objection.
%%  4. One sentence acknowledging R1's endorsement of the existing protocol and
%%     that the extension preserves it rather than replacing it.
\changeref{}
\end{response}
```

### 9.4 Residual risk

> Candidates: the appearance of selecting problems to strengthen CPDT (mitigated by
> pre-registration — cite it); R3 asking why the black-box track is still absent;
> the extension diluting per-problem effect sizes; page cost colliding with R2's C6.
