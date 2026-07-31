# T05 — Bounded benchmark extension: Feynman remainder + ODE-Strogatz

| Field | Value |
|---|---|
| Reviewer comments closed | **R3.1** (and B4 for all three reviewers) |
| Type | Justification + new experiment |
| Owner | **Mario** (+ Claude Code) for the experiments · **Karl** for the written justification |
| Depends on | T02 (protocol and infrastructure) |
| Blocks | **Campaign C2 — this ticket gates the launch** · T09 (Appendix D tables), T13 (page budget) |
| Status | NOT STARTED |
| Target | **problem definitions 2026-08-17 — this is a hard launch gate, every day late costs ≈7,200 core-hours of headroom** · results 2026-09-03 |

---

## ⛔ Amendment 2026-07-31 — read before doing anything on this ticket

**There is no "Wave 2". D2 launches simultaneously with D1, and this ticket gates
that launch.**

`EXECUTION-PLAN.md` was rewritten on 2026-07-31 and is authoritative. Campaign C2 is
a single gated launch over the union of both suites:

```
{ baseline , hash , isalsr } × { UDFS , Bingo } × ( D1 ∪ D2 ) × 20 seeds
```

| Was | Is |
|---|---|
| Wave 2, `EXT`, 2 arms × 2 methods × ≈20 problems × 30 seeds = 2,400 runs, launches after Wave 1 and must not delay it | **D2 is part of C2**: **3** arms × 2 methods × ≈20 problems × **20 seeds = 2,400 runs**, launched together with D1 |
| "Wave 2 must not delay Wave 1. If the problem definitions are not ready when the C++ gate passes, Wave 1 launches on S50 alone" | **Superseded. This ticket now delays everything.** The gated-launch decision (§0.4b) was taken deliberately: one commit, one build, one node pool, one alphabet across all 70 problems |
| 30 seeds | **20 seeds.** §6.3 and the boxed note in §0.4 |
| CPDT at N = 50 and N ≈ 70 | unchanged, and now the *only* seed-count-sensitive claim you need to think about — see below |

**Why the schedule pressure is real:** `EXECUTION-PLAN.md` §8.2. A gated launch at
2026-08-20 needs ≈300 concurrent cores to finish by 2026-09-03. Every day this ticket
slips is ≈7,200 core-hours of headroom the campaign does not get back. If D2 cannot
be ready, the trade is **"Strogatz only, 14 problems"** (§8.3 item 2) — *not* delaying
the launch.

**Interaction with §5's statistical hazard, and it cuts your way.** C2 drops from 30
to 20 seeds, which weakens the *per-problem* supplementary tests, while this ticket
raises `N` from 50 to ≈70, which strengthens **CPDT**, the primary metric. The
response letter must present these together, in one paragraph, as a deliberate trade:
*more problems, fewer seeds per problem, primary metric strengthened, supplementary
tests weakened, both reported.* Do not let the two changes be announced in separate
places where a reviewer can read the seed reduction as a concession and the problem
increase as unrelated. The pre-registration requirement in §5.1 becomes *more*
important, not less, for exactly this reason.

### 🚫 This ticket does not submit the campaign

**`EXECUTION-PLAN.md` §4.0 SP-0 is binding.** No agent working this ticket submits C2
or any D2 production array. Everything submitted here is a **probe**: `max_time
≤ 1,800 s` (30 min), ≤ 60 tasks, **seed 0 only**, output to `~/execs/isalsr/t05_*/`,
never the campaign root.

**Before trusting any Picasso result from this ticket, establish SP-1…SP-6**
(`EXECUTION-PLAN.md` §4.0) and report them as a six-row table in the work log:
provenance; **installation freshness** (site-packages `.so` mtime post-dates the last
C++ edit; `pip install -e . --force-reinstall --no-deps`, **never**
`--no-build-isolation`); engine `native` **with the forced-Python negative control**;
alphabet clean on the probe's own candidate stream; **UDFS and Bingo both**; T06's
five fallback counters live and finite.

**SP-7 for this ticket** — what a T05 probe must establish, on a ≤30-minute run, on
**both hosts**, for **every** D2 problem:

1. The dataset **loads on Picasso** — path resolvable from a compute node, not just
   locally — with the **expected train/test shapes** asserted against the benchmark
   registry. Per-problem sampling protocols are not typos; do not "fix" them.
2. A `sympy_expression` ground truth is present, so `solution_recovered` is actually
   **computable**. This is the historic gap (AC-4) and it is invisible until analysis.
3. The run completes 30 minutes on both hosts **without crashing**, and produces a
   `run_log.json` that parses and validates against the full RunLog schema.
4. The **declared operator set** is what actually ran (check A4b: for a fixed
   `(method, problem)` the operator set must be identical across all three arms).
5. No NaN or inf in any regression metric on any D2 problem.

These five are precisely Stage C's criteria C1.1–C1.5 restricted to D2, so passing
them here is not duplicated work — it de-risks Stage C before 420 tasks depend on it.

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

- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — **read first.** §0.4 (campaign
  shape and the 20-seed decision), §4.0 (SP-0…SP-7 Picasso discipline), §4.3 Stage C
  (whose criteria C1.1–C1.5 this ticket must satisfy for D2), §6.2–6.3 (CPDT at both
  N, and the seed trade), §8.3 (the Strogatz-only escape hatch)
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
6. **Run as part of Campaign C2** — not as a separate wave. D2 is simply the second
   half of the problem list in all six arrays: same 12 h budget, **20 seeds**, both
   methods, **all three arms** (`baseline`, `hash`, `isalsr`), one campaign root, one
   MANIFEST, one commit, one build. The `hash` arm covers these problems here, not in
   a later campaign.

   ≈20 problems × 3 arms × 2 methods × 20 seeds = **≈2,400 runs**, ≈28,800 core-hours,
   inside C2's 100,800.

   **This ticket gates the launch** (§0.4b). If D2 is not ready, the campaign does not
   go out. The escape hatch is scope, not schedule: **Strogatz only, 14 problems**
   (`EXECUTION-PLAN.md` §8.3 item 2), which preserves the "SRBench ground-truth track
   is now covered" claim and costs the Feynman-remainder half of the R3.1 answer.
   Take that trade rather than delaying, and record it in §8.
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
- **AC-4b.** SP-1…SP-6 reported as a six-row table for every Picasso probe this
  ticket ran, and SP-7's five statements established for **every** D2 problem on
  **both hosts** before the definitions were declared launch-ready.
- **AC-5.** D2's share of C2 complete (≈2,400 runs) or every missing run accounted
  for in the status ledger.
- **AC-6.** CPDT reported at N = 50 and N ≈ 70, per method, per metric — **at 20
  seeds throughout**. The seed reduction and the `N` increase are presented together
  as one deliberate trade (see the 2026-07-31 amendment), not in separate places.
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
