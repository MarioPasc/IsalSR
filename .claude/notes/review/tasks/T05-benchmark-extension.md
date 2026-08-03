# T05 — Bounded benchmark extension: Feynman remainder + ODE-Strogatz

| Field | Value |
|---|---|
| Reviewer comments closed | **R3.1** (and B4 for all three reviewers) |
| Type | Justification + new experiment |
| Owner | **Mario** (+ Claude Code) for the experiments · **Karl** for the written justification |
| Depends on | T02 (protocol and infrastructure) |
| Blocks | **Campaign C2 — this ticket gates the launch** · T09 (Appendix D tables), T13 (page budget) |
| Status | **D2 IS LAUNCH-READY. AC-0…AC-4, AC-4b, AC-8, AC-9 met; AC-5…AC-7 gated on C2 and cannot be met by this ticket.** `D2 = 20`: **all 14 ODE-Strogatz** + **6 AI Feynman** drawn by the pre-registered rule (`I.12.2, II.34.29a, II.34.29b, III.19.51, III.4.32, test_4`). Registry resolves **70/70**; `solution_recovered` computable on **70/70** (was 65/70 — five *submitted* problems were failing C1.5). **AC-3 ordering verifiable from git**: rule `d95e7d9`, draw `0e4a573`. **Picasso probe 40/40 COMPLETED** (arrays 1741991/1742002 at `fa41e2a`); SP-1…SP-6 + negative control all 40/40 PASS; ρ > 1 on 40/40; zero NaN; memory max 541 MB. Full suite **6591 passed, 5 skipped**. 🔴 **Criterion (ii) excludes only 4 of 120**, so the R3.1 answer is re-attributed to criterion (iv) + compute. 🔴 **The probe found a C2 blocker that is T02's, not T05's: C1.9 and C1.14 are *uncheckable* — the five T06 fallback rates and `engine` never reach the RunLog (0/40 each).** 🔴 **16/20 Bingo cells saturate at a 25-min budget, so the extension may WEAKEN CPDT** — pre-committed to reporting this. Five D1 definitions corrected (internal record only). |
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

### Status as of 2026-08-02

| AC | State | Evidence |
|---|---|---|
| AC-0 | **met** | §8, written as the work proceeded |
| AC-1 | **met** | `feynman_catalogue.py`; 120 equations, 116 syntactic / 117 semantic representable, 4 blocked with the operator named per exclusion; 652 tests |
| AC-2 | **met** | 130 = 116 + 14 counted from `groundtruth.csv`; black-box track's failing criteria named in §8. Re-measured 2026-08-03 through `pmlb_id`: all 30 suite Feynman ids lie inside the track (rows nested, not additive), and SRBench's own 3 omissions equal criterion (ii)'s semantic exclusion set |
| AC-3 | **met** | rule `d95e7d9`, draw `0e4a573`; ordering verifiable from git, seed content-derived |
| AC-4 | **met** | `strogatz.py` + `feynman_remainder.py`, 433 tests; `sympy_expression` on 20/20 D2 and, after the C1.5 fix, on 70/70 overall |
| AC-4b | **met** | Arrays 1741991/1742002 at `fa41e2a`, **40/40 COMPLETED**. SP-1…SP-6 + SP-3′ all 40/40 PASS as a six-row table; SP-7's five statements established for every D2 problem on both hosts. `T05-appendix/probe_results.md` |
| AC-5 | **gated** | needs C2 |
| AC-6 | **gated** | needs C2. The seed/N trade is written as one paragraph in §9.3 already, so the framing obligation is discharged in advance |
| AC-7 | **gated** | needs C2 |
| AC-8 | **met (T05's half)** | 70 rows generated by `experiments/scripts/generate_appendix_d1.py`; D2's 20 and all 70 committed under `docs/md_files/changes/t05_appendix/`. Hand-off to T09 for placement in the supplementary |
| AC-9 | **met** | §9 filled; rows needing C2 marked *pending* rather than guessed |

---

## 8. Work log

### 2026-08-02 — plan

Dependency note: T05 lists `Depends on T02`, and T02 is `NOT STARTED`. The
dependency is on the **campaign**, not on the definitions: AC-1…AC-4 and AC-8 are
entirely offline and are on C2's critical path (§0.4b — this ticket gates the
launch). AC-4b/AC-5/AC-6/AC-7 are gated and are **not** attempted here. T04's probe
is in flight (arrays 1737666/1737667); per SP-0 discipline no second probe array
goes out while it runs, and a T05 probe needs the D2 definitions that this session
produces. So this session is offline work only.

Decomposition:

| # | Deliverable | Kind | Lane |
|---|---|---|---|
| S1 | Canonical AI Feynman 120-equation list reconciled against PMLB's 119; our 24 in-suite Feynman problems mapped to PMLB ids by formula equivalence | investigate | read-only |
| S2 | `benchmarks/datasets/strogatz.py` + vendored PMLB data + unit tests | implement | benchmarks/, tests/unit/ |
| S3 | `benchmarks/datasets/feynman_catalogue.py`: all 120 equations + a **sympy-tree** Σ_SR classifier, with the criterion-(ii) table as its output | implement | benchmarks/, tests/unit/ |
| S4 | `docs/md_files/changes/r31_extension_selection.md` — the pre-registered rule, committed **before** the draw is executed | mine | docs/ |
| S5 | Orchestrator registry + `_generate_benchmark_data` dispatch; configs; launcher | mine | experiments/, slurm/ |
| S6 | Appendix D.1 rows for every added problem (hand-off to T09) | mine | docs/ |

AC-3 ordering is enforced through git history, not assertion: S4 is committed with
the rule and the draw **script** but no draw **output**; the drawn list lands in a
second commit. Both hashes are recorded below, so the ordering is verifiable by a
third party.

### 2026-08-02 — data provenance settled before any code was written

`lacava/ode-strogatz` is **GPL-3.0**; IsalSR is MIT (`pyproject.toml:29`), so the
upstream files are not vendorable. PMLB redistributes the identical data under
**MIT**, and PMLB is what SRBench actually consumes. Verified numerically:
`pmlb/datasets/strogatz_*` and `ode-strogatz/d_*.txt` agree to **≤ 7.1 × 10⁻¹⁵**
(pure decimal-representation difference) on all 14 datasets, 400 × 3 each, no NaN.
Vendored the PMLB `.tsv.gz` **byte-verbatim** (172 KB total, sha256 recorded in
`benchmarks/datasets/data/strogatz/PROVENANCE.md`) so the provenance chain is
checkable against PMLB's own LFS objects.

The 14 ground-truth ODEs were obtained from **two independent sources** and agree
exactly: `ode-strogatz/simulate_ode.m` (the MATLAB/Simulink generator, `outstr`
literals) and each PMLB `metadata.yaml`. They are recorded in §8's equation table
below rather than transcribed from memory.

### 2026-08-02 — 🔴 PREMISE-FALSE against §3: criterion (ii) is a *small* filter

§3 asserts "Criterion (ii) genuinely excludes a substantial fraction" of AI Feynman.
**It does not.** Parsing all 119 PMLB `feynman_*` `metadata.yaml` formulas (119/119
parsed, 0 unresolved symbols) gives this function inventory:

| Function | Equations | In Σ_SR? |
|---|---|---|
| `sqrt` | 26 | yes |
| `cos` | 16 | yes |
| `sin` | 11 | yes |
| `exp` | 9 | yes |
| `ln` | 1 | yes |
| `arcsin` | 2 | **no** |
| `arccos` | 1 | **no** |
| `tanh` | 1 | see below |

So **at most 4 of 119** are blocked, not "a substantial fraction":

- `feynman_I_26_2` — `theta1 = arcsin(n*sin(theta2))`
- `feynman_I_30_5` — `theta = arcsin(lambd/(n*d))`
- `feynman_test_10` — `theta1 = arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))`
- `feynman_II_35_21` — `M = n_rho*mom*tanh(mom*B/(kb*T))`

and `tanh` is **semantically representable** in Σ_SR —
`tanh(u) = (e^u − e^{−u})·(e^u + e^{−u})^{−1}` uses only `Exp, Neg, Add, Mul, Inv` —
so under a semantic reading the count is **3**, not 4. By the same argument `tan`
and `cot` are representable, which matters for Strogatz `shearflow1`. The paper's
own example list for criterion (ii) (`tanh, arctan, sgn`) is therefore
**syntactically** meant, not semantically.

Consequences, all of which must reach the response letter:

1. The R3.1 answer **cannot** rest on criterion (ii). Attributing the 24-of-120
   choice to (ii) would be a claim a reviewer can falsify in ten minutes with PMLB.
2. Both counts (syntactic 4, semantic 3) get reported, with the distinction named.
   The selection rule uses the **syntactic** reading — the paper's own, and the more
   exclusive of the two, so it cannot be accused of stretching to admit favourable
   problems.
3. The binding constraints are criteria (iii)/(iv) and compute, and the compute
   arithmetic is already exact (`source/reviewer-3.md`): all 120 Feynman + 250
   SRBench would be ≈45,840 runs, ≈7.6× the submitted campaign.

### 2026-08-02 — SRBench track classification (AC-2), from the primary artefact

Counted directly from `cavalab/srbench/docs/csv/groundtruth.csv` (the results table
that defines the track), not from the paper's prose: **130 unique ground-truth
datasets = 116 `feynman_*` + 14 `strogatz_*`.** With the 122 black-box datasets this
reproduces the "over 250 problems" of `related_work.tex:88`.

Two counts that must not be conflated, because they differ and a reviewer checking
PMLB will see it: PMLB ships **119** `feynman_*` datasets; SRBench's ground-truth
track uses **116** of them.

The black-box track is out of scope by construction and this is checkable rather
than rhetorical: those datasets have no ground-truth expression, so criterion (i)
(published provenance *of the expression*) and criterion (iii) (structural rather
than constant-fitting difficulty) have no referent, and `solution_recovered` —
a reported metric — is undefined, not merely hard.

### 2026-08-03 — AC-2 re-measured: no double counting, and criterion (ii) gets an external check

Asked while writing the R3.1 answer: do the "AI Feynman" and "SRBench
ground-truth track" rows of the coverage table report overlapping problems, and
could a reviewer read them as duplicated coverage? Re-fetched `groundtruth.csv`
and mapped every suite id through `feynman_catalogue`'s verbatim `pmlb_id`
field. **Do not map ids by string substitution** — PMLB truncates
(`I.6.20a → feynman_I_6_2a`, `I.48.20 → feynman_I_48_2`,
`I.39.10 → feynman_I_39_1`), and a naive `.replace('.','_')` reports three false
gaps.

Measured:

| Quantity | Result |
|---|---|
| unique datasets in `groundtruth.csv` | 130 = 116 `feynman_*` + 14 `strogatz_*`, 0 other |
| suite AI Feynman ids inside the track | **30 / 30** |
| the drawn six inside the track | **6 / 6** |
| PMLB `feynman_*` datasets in the catalogue | 119 (120 equations, `II.11.17` has no PMLB dataset) |
| datasets in `groundtruth.csv` absent from the catalogue | 0 |

So the rows are **nested, not additive**, and `24 → 44` is right: the suite's 70
are 44 SRBench ground-truth (30 Feynman + 14 Strogatz) + 12 Nguyen + 14 others,
with no problem counted twice. The letter's table now says so in the caption and
italicises *its* on the two component rows.

🟢 **Unplanned result, and it is the strongest thing in the R3.1 answer.**
SRBench omits exactly **3** of PMLB's 119: `I.26.2`, `I.30.5`, `test_10` — which
is **exactly** criterion (ii)'s exclusion set under the *semantic* reading. The
only equation the two curations disagree on is `II.35.21`, which SRBench keeps
and we exclude, and that disagreement *is* the syntactic/semantic `tanh`
distinction already disclosed. **Our operator criterion and SRBench's own
selection agree on 118 of 119 datasets.** This makes criterion (ii) externally
corroborated rather than self-declared, which is worth more than the §3 claim it
replaced. Written into the letter.

### 2026-08-02 — the 14 ODE-Strogatz targets (verified, dual-sourced)

`x` and `y` are the two state variables; the regression target is the named
derivative. `label` is the target column in the vendored files.

| Problem | Target | Σ_SR? |
|---|---|---|
| Strogatz-bacres1 | `x' = 20 − x − x·y/(1 + 0.5·x²)` | yes |
| Strogatz-bacres2 | `y' = 10 − x·y/(1 + 0.5·x²)` | yes |
| Strogatz-barmag1 | `x' = 0.5·sin(x − y) − sin(x)` | yes |
| Strogatz-barmag2 | `y' = 0.5·sin(y − x) − sin(y)` | yes |
| Strogatz-glider1 | `x' = −0.05·x² − sin(y)` | yes |
| Strogatz-glider2 | `y' = x − cos(y)/x` | yes |
| Strogatz-lv1 | `x' = 3x − 2xy − x²` | yes |
| Strogatz-lv2 | `y' = 2y − xy − y²` | yes |
| Strogatz-predprey1 | `x' = x·(4 − x − y/(1 + x))` | yes |
| Strogatz-predprey2 | `y' = y·(x/(1 + x) − 0.075·y)` | yes |
| Strogatz-shearflow1 | `x' = cot(y)·cos(x)` | yes, as `cos(y)·cos(x)·sin(y)⁻¹` |
| Strogatz-shearflow2 | `y' = (cos²(y) + 0.1·sin²(y))·sin(x)` | yes |
| Strogatz-vdp1 | `x' = 10·(y − (1/3)(x³ − x))` | yes |
| Strogatz-vdp2 | `y' = −(1/10)·x` | yes |

All 14 pass criterion (ii). `shearflow1` is the only one needing the semantic
reading, and only for `cot`, which the T16 alphabet supplies directly through `Inv`.

### 2026-08-02 — 🔴 second correction to §3: "covers SRBench's ground-truth track" overstates it

§3 says including Strogatz "lets the response say the paper now covers SRBench's
ground-truth track". Measured against `groundtruth.csv`, the revised suite covers
**44 of 130**, not 130 of 130. The defensible claim, which is still a strong one:

> the suite now contains the **entire** ODE-Strogatz component of SRBench's
> ground-truth track — 14 of 14, previously 0 — together with 30 of its 116
> AI Feynman equations.

Written the loose way, a reviewer with the SRBench table open catches it immediately.
Written the precise way it is unimpeachable and says the thing that matters: the
component that was **wholly absent** is now **wholly present**.

### 2026-08-02 — AC-3 closed: rule committed before the draw, verifiable from git

| Commit | Contents |
|---|---|
| **`d95e7d9`** | `r31_extension_selection.md`, `r31_draw_extension.py`, `feynman_catalogue.py`, `strogatz.py` + tests. **No drawn ids anywhere.** |
| **`0e4a573`** | `r31_extension_selection_draw.json` — the draw output, and nothing else |

The rule: all 14 Strogatz, plus `K = 6` drawn uniformly from the 92 ids that are
criterion-(ii) representable and not already in the suite. The seed is
`sha256(sorted eligible ids)[:16] mod 2³²` = **2547107438** — derived from the pool
itself, so it is not a free parameter and there is no knob to fish over. Re-running
the script reproduces the same six; confirmed after an unrelated lint fix touched the
file.

Drawn: **I.12.2, II.34.29a, II.34.29b, III.19.51, III.4.32, test_4** (bonus).

No stratification, deliberately. Every candidate stratifying variable — arity, `k`,
operator class — correlates with the structural-bottleneck axis along which IsalSR's
advantage was characterised, so stratifying on it would be selection wearing a
neutral name. Recorded in the rule document's non-goals table so it is not
"improved" later.

### 2026-08-02 — five D1 definitions corrected (decision: Mario)

Building the catalogue exposed that five of the suite's 24 AI Feynman problems did
not encode the equation their id names. Verified against two independent renderings
of the database that agree on all 99 shared equations with 0 mismatches.

Decision taken by Mario: **correct the target functions, keep the ids**, and keep the
record internal — nothing reviewer-facing. Applied in `baba7a4`; all five now
reproduce the database to a relative error below `10⁻¹²` and their arity matches.
Two hard-tier tests asserted the old values and were rewritten.

I flagged two consequences before the change was made and they stand:
`III.17.37` drops to 3 variables and no longer needs `sqrt`, so it is no longer hard
by the hard tier's own rationale; and `II.11.27` is now structurally close to
cherrypicked's `II.11.28`, in tension with criterion (iv). Also: the C1↔C2 continuity
table must exclude these five rows, and the bottleneck-type classification of the two
replaced problems was derived from the old targets and no longer applies. Full
record and open items: `docs/md_files/changes/feynman_definition_corrections.md`.

### 2026-08-02 — C1.5 was failing on D1, and is now 70/70

Auditing ground-truth coverage across the whole registry — a by-product of building
the Appendix D.1 generator — showed `solution_recovered` was **not computable for
five of the submitted fifty**: `I.14.3`, `I.12.4`, `II.3.24`, `I.10.7`, `I.48.20`.
The Feynman tier carried no `sympy_expression` at all and relied on the orchestrator's
string-parse fallback, which only handles one- and two-variable targets.

This is Stage C criterion **C1.5**, which demands 70/70, and it would have failed on
D1 rather than on the D2 additions everyone was watching. Fixed by giving all ten
Feynman-tier problems explicit `sympy_expression` and `sympy_variables`; each
verified against its `target_fn` to ≤ `1.5 × 10⁻¹⁶`. Measured after:

```
C1.5: solution_recovered computable on 70/70; missing=[]
```

### 2026-08-02 — local smoke, both hosts (skill §5.1 gate)

Strogatz, 2 problems × 2 seeds × 2 arms, `max_time = 45 s`, both hosts. All eight
runs produced a `run_log.json` that parses and carries every field the analyzer
reads; no NaN or inf in any regression metric; `trajectory.csv` non-empty throughout.

| host | arm | ρ | canon runtime |
|---|---|---|---|
| Bingo | baseline | 1.0000 | 0.00 s |
| Bingo | isalsr | 1.7629 / 1.7772 | 3.90 s |
| UDFS | baseline | 1.0000 | 0.00 s |
| UDFS | isalsr | 1.5167 / 1.5082 | 0.02 s |

Dedup is live on both hosts and the baseline arm is genuinely un-instrumented
(ρ = 1, zero canonicalisation time), which is Stage C's **C1.8**.

The orchestrator exits non-zero at the end of a ≤2-seed run: `compute_paired_stats`
raises on `len(common_seeds) < 3`. That is a pre-existing guard in
`aggregation.py:207`, unrelated to D2, and it aborts the remaining problems in the
run — worth knowing when designing the Picasso probe, which must use ≥3 seeds or
tolerate the raise.

### 2026-08-02 — second smoke: the Feynman remainder, both hosts, 3 seeds

`III.4.32` and `test_4`, 3 seeds × 2 arms × 2 hosts, `max_time = 40 s`. **Both hosts
exit 0** — the ≥3-seed threshold is cleared, so `compute_paired_stats` runs to
completion rather than raising. All **24** runs land. Verified
through `slurm/t05_probe/check_d2.py --verify-runs`, which is the same code the probe
will run on Picasso — so the checker itself is exercised end-to-end here rather than
first meeting real output on a compute node:

```
runs                         ok=True
SP-7 overall: PASS
```

| host | arm | ρ (3 seeds) | unique canonical |
|---|---|---|---|
| Bingo | baseline | 1.000, 1.000, 1.000 | ~0.9–1.0 M |
| Bingo | isalsr | 1.750, 1.776, 1.780 | ~61–64 k |
| UDFS | baseline | 1.000 ×3 | ~148 |
| UDFS | isalsr | 1.448, 1.448, 1.451 | ~143 |

Zero NaN or inf across every regression metric on every run. `ρ > 1` on every
`isalsr` cell and exactly `1.0` with zero canonicalisation time on every `baseline`
cell — Stage C's **C1.6** and **C1.8** both hold on D2 locally.

**C1.12 holds too, and it was worth checking.** UDFS looked slow enough to suspect it
was overrunning its budget, which is the known Bingo defect in `CLAUDE.md`. It is not:
every UDFS run terminated at **40.1–40.3 s against a 40 s budget**. The wall-clock
belongs to the orchestrator's between-run work — data generation, ground-truth setup,
translation — not to the search. Worth knowing when sizing the probe's SLURM
wallclock, since that per-run overhead is paid 40 times and is invisible in
`max_time`.

Final tally over both smokes: **32 runs** (8 Strogatz + 24 Feynman remainder), both
hosts, both arms, `SP-7 overall: PASS`, zero NaN.

### 2026-08-02 — status of the gated items

**AC-4b, AC-5, AC-6, AC-7 are not attempted and are not claimable.** They need
Picasso and C2 respectively. Specifically:

- **AC-4b** — no probe was submitted. T04's probe (arrays 1737666/1737667) is still
  in flight and SP-0 discipline is per-probe; two probe arrays with overlapping log
  directories is exactly the confusion the discipline exists to prevent. The
  definitions a T05 probe needs only came into existence in this session. SP-7's five
  statements are established **locally** for all 20 D2 problems (shapes asserted
  against the registry, `sympy_expression` present on 20/20, both hosts run without
  crashing, operator set identical across arms by construction since it lives in one
  YAML per method, no NaN/inf); the **on-Picasso** half is untouched.
- **AC-5, AC-6, AC-7** — gated on C2 itself.

### 2026-08-02 — the probe ran: 40/40 COMPLETED, AC-4b closed

Arrays **1741991** (Bingo 1–20) and **1742002** (UDFS 21–40) at commit **`fa41e2a`**,
seed 0, `max_time = 1500 s`. **40/40 COMPLETED**, zero failures, zero NaN, zero
SLURM time-kills. SP-1…SP-6 and SP-3′ all **40/40 PASS**; SP-7's five statements
established for **every** D2 problem on **both** hosts. Full tables:
`T05-appendix/probe_results.md`; generated artefact:
`T05-appendix/probe_summary_raw.md`.

ρ > 1 on 40/40 (C1.6). Bingo 1.76–1.81 on 19 of 20; UDFS **2.11** mean on Strogatz
against 1.36 on the Feynman remainder — UDFS reduces *more* than Bingo on this tier,
reversing the submitted campaign's ordering. Memory median 393 MB, max 541 MB.

**SP-1 failed the first attempt and that is the most useful thing the probe did.**
Jobs 1739900/1739901 died in 7 s with four file-hash mismatches. Two real causes:
the cluster's `.provenance.json` was still T04's (`a4206b8`), and my first `rsync`
sent the **working tree**, carrying another session's uncommitted `aggregation.py`,
`metrics.py`, `schemas.py`. Fixed by deploying from a clean detached worktree and
extending the stamp to 46 files including the 14 vendored `.tsv.gz`. Without SP-1
this probe returns 40 clean green results from partly-uncommitted code.

Two smaller results worth keeping: the 24 s task is the **resume logic** correctly
skipping the cell the single-task gate had already completed, which is pre-flight
check **B8** demonstrated on D2; and the 13–17 KB `.err` files are Python INFO
logging, not errors (0 matches for `Traceback|Error|FATAL|OOM`).

### 2026-08-02 — T04's shadow-hash confirmation, satisfied by this probe

T04 asked that the shadow counters be enabled in "the pending `slurm/t05_probe/`
submission" so the sketch and the three extractors are seen running together on
Picasso once before ≈2,800 runs depend on it. **That request predates the probe
running, and the probe already carried them** — `worker.sh` never passes
`--no-shadow-hash`.

All four shadow cardinalities present, finite and > 0 on **40/40** runs, both hosts,
20 problems. `serialisation failures = 0` on **38/38** tasks that ran a search
(19 Bingo, 19 UDFS). The two without the line are `1741991_1` / `1742002_21`, the
`I.12.2` cells the gate had already completed — zero search invocations, resume logic
skipped them; their run_logs come from the gate, which is why RunLog coverage is
40/40 while log coverage is 38/38.

🔴 **But T04's proposed Stage C assertion is not yet checkable from a RunLog.**
`n_shadow_failures` is tracked (`bingo/isalsr_runner.py:221,293,300`) and then only
**logged** (`:773`, INFO, stderr); no `shadow_fail`/`n_shadow`/`serialis`/`failure`
key exists anywhere in a run_log. The zero above was recovered by grepping 40 `.err`
files. Less severe than `C1.9-BUG` — the four cardinalities T04 actually needs *are*
persisted, and this is recoverable while logs live — but asserting it across 420
Stage C tasks means grepping 420 files rather than reading a field.

**Recommendation, filed not done:** add `n_shadow_failures` to `search_space` beside
the four shadow fields. Already computed; one field. Not done here because
`experiments/models/schemas.py` is being edited by the T08 session and a second
writer in that file is how a wrong number gets in. **T04's or T02's call.**

### 2026-08-02 — 🔴 the probe found a C2 launch blocker that is not T05's

**Stage C's C1.9 and C1.14 are uncheckable, not merely failing.** Neither quantity
reaches the RunLog: a walk of all 69 keys finds **0/40** runs carrying any of the
five T06 fallback rates, and **0/40** carrying `metadata.hardware.engine`.

C1.14 is the known `A7-BUG`, now reproduced on a live D2 probe. **C1.9 is new.**
`EXECUTION-PLAN.md` §3: *"anything measured during a run must be in the code before
launch… Getting this wrong means re-running 8,400 jobs to recover a counter."* The
reachability rates exist only while a search runs, so unlike `engine` they cannot be
recovered afterwards — and they are the evidence base for R1.2.

SP-6 passing does not contradict this. `sp_probe.sp6_counters` imports
`FallbackLedger` and lists its attributes; it never reads a live count. The probe's
own summariser now states SP-6 in those weaker terms so the row cannot be quoted as
the stronger claim, and reports C1.9/C1.14 checkability as its own table.

**Owner: T02** (C1.9 is its check; T06 supplies the threshold). Filed, not fixed
here. T05 does not claim it is fixed.

### 2026-08-02 — saturation is now measured, and it cuts against us

**16 of 20 Bingo cells reach R² ≥ 0.999 at a 25-minute budget**, against a
production budget of 12 hours; the local smoke showed `baseline` saturating too. If
both arms saturate, `δᵢ ≈ 0` and the added problems **weaken** CPDT by contributing
ties rather than strengthening it.

This is an inference — the probe ran only the `isalsr` arm, so no `δᵢ` exists yet —
and it is flagged as one. But §5.4 of the pre-registration commits us to reporting
it, and the extension's defensible claim was always **coverage**, not a smaller
p-value. The response letter should say so before analysis discovers it.

`Strogatz-vdp2` is degenerate: target `−x/10`, solved by Bingo in 0 s and UDFS in
7 s, ρ 1.17/1.14, the low outlier on both hosts. It stays — removing a problem after
seeing its result is exactly the post-hoc selection the pre-registration forbids —
but Appendix D.1 should name it.

### 2026-08-02 — the probe harness exists and is one command from going out

`slurm/t05_probe/` — **built, not submitted.** 40 tasks (20 D2 problems × 2 hosts,
`isalsr` arm, seed 0, `max_time = 1500 s`), inside SP-0's 60-task and 1,800 s caps.

`worker.sh` is derived from `slurm/t04_probe/worker.sh` almost verbatim, and that is
the point rather than laziness. That worker carries three environment fixes that are
invisible from the application code and each of which exists because something failed
on this cluster: the `openmpi_gcc/5.0.9` module load (mpi4py's ABI-probing import hook
`dlopen()`s `libmpi` at **import** time, so its absence kills a Bingo task in ~13 s,
before any search starts), the conda `LD_LIBRARY_PATH`, and `PYTHONMALLOC=malloc`. A
freshly authored worker would be missing all three, and the `picasso-sbatch` skill
names exactly this failure mode.

`check_d2.py` turns SP-7's five statements into a command with an exit status. Its
offline half already passes on all 20 D2 problems:

```
shapes_and_ground_truth      ok=True
operator_sets                ok=True
SP-7 overall: PASS
```

`tasks.txt` is **generated** from the benchmark registry, not typed — a hand-written
list could pass while covering a different problem set from the one that launches.

What the probe adds over the local run, and the reason it cannot be skipped: the
Strogatz data is **vendored in the repo tree**, so its paths have to survive the
`rsync` and resolve from a compute node. That is SP-7.1's real content and no local
check can establish it.

Sequence when T04's probe clears: `--dry-run` → `--test-only` → `--one` → the array.

### 2026-08-02 — one process note, recorded because it cost time twice

The `PostToolUse` ruff autofix removes an import that is not yet referenced. Adding an
import and its first use in the same turn therefore leaves the file with the use and
no import, and the failure surfaces later as a `NameError` at module import. It hit
`orchestrator.py` and `generate_appendix_d1.py`. Add the reference first, or re-check
the import block after any autofix.

---

## 9. Proposed answer

### 9.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Problems in suite | 50 | **70** | registry, measured |
| AI Feynman equations used | 24 of 120 | **30 of 120** | 24 + the 6 drawn |
| AI Feynman Σ_SR-representable (of 120) | not reported | **116** syntactic, **117** semantic | AC-1 |
| AI Feynman excluded by criterion (ii) | not reported | **4** syntactic, **3** semantic | AC-1 |
| SRBench ground-truth track, total | not reported | **130** = 116 Feynman + 14 Strogatz | AC-2, `groundtruth.csv` |
| SRBench ground-truth track covered | 24 of 130 | **44 of 130** | AC-2 |
| — its ODE-Strogatz component | **0 of 14** | **14 of 14** | AC-2 |
| SRBench black-box track covered | 0 | 0 (out of scope, criteria i & iii) | AC-2 |
| ODE-Strogatz problems | 0 | **14** | |
| Total runs | 6,000 | **8,400** (C2, three arms, 20 seeds) | `EXECUTION-PLAN.md` §1 |
| Seeds per problem | 30 | **20** | §0.4a — reported *with* the N increase |
| CPDT N | 50 | **70** | |
| CPDT R² test p, UDFS | 0.00018 | *pending C2* | at N = 50 and N = 70 |
| CPDT R² test p, Bingo | 0.0013 | *pending C2* | at N = 50 and N = 70 |
| CPDT reduction-factor p, UDFS | ≈ 0 | *pending C2* | |
| CPDT reduction-factor p, Bingo | ≈ 0 | *pending C2* | |
| ρ, UDFS | 1.56 ± 0.24 | *pending C2* | |
| ρ, Bingo | 1.83 ± 0.09 | *pending C2* | |
| Problems where dedup fires | 50 / 50 | *pending C2* | R1 called this "clean and credible" |
| `solution_recovered` computable | 45 of 50 (**unreported defect**) | **70 of 70** | measured; Stage C C1.5 |
| Problems documented in Appendix D.1 | 22 of 50 | **70 of 70 rows generated** | AC-8, with T09 |

Rows marked *pending C2* are the ones this ticket cannot produce: they need the
campaign, which this ticket gates rather than runs. Everything else is measured.

### 9.2 Changes made to the manuscript

**Applied 2026-08-03** to the annotated draft
(`reviews/internal_copy_reviewed_article/`), in blue; `article/` untouched. Both
documents compile clean (0 errors, 0 undefined refs/cites); numbered environments
the reviewers cite (Thm. 3.13/3.15, Def. 3.5) are unchanged.

| File | Applied |
|---|---|
| `paper/main.tex` | abstract: `50`→`70` problems, `eight`→`nine` suites. Results numbers left alone (they are C2's) |
| `paper/computational_experiments.tex` §5 opening | `50`→`70`, `eight`→`nine`, `30`→`20` seeds |
| `paper/computational_experiments.tex` `sec:benchmarks` | assembly now three stages; **core corrected `32`→`22` and its Feynman share `20`→`10`, extension `18`→`28`** (the submitted split did not match the registry: the suite is 12 Nguyen + 24 Feynman + 14 others); criterion (ii) gains the syntactic-reading clause; new blue paragraph for the coverage extension with the pre-registered rule, the 30/120 and 44/130 coverage, and the black-box exclusion; training sizes gain `300` |
| `paper/computational_experiments.tex` `sec:cpdt` | `N = 50`→`70`, `S = 30`→`20`, and the normal-approximation sentence |
| `supplementary/supplementary.tex` §D.1 | two new blue tables: 14 ODE-Strogatz and the 6 Feynman remainder, with the trajectory-split caveat and `vdp2` named |
| `paper/references.bib`, `supplementary/references.bib` | `strogatz2014`, `romano2021` added |

**Left for other tickets, deliberately:**
- `supplementary.tex:735–737` still reads `2 × 2 × (12+10) × 30 = 2,640 total runs`
  and `:750` `all 2,640 runs`. That sentence is already wrong against the submitted
  6,000 and is **R2.6's** to rewrite; editing `30`→`20` inside a wrong formula makes
  it wronger. R2.6 must produce it at three arms × 20 seeds × 70 problems = 8,400.
- `results.tex:24` and `supplementary.tex:767, 785, 824` still say `30 seeds` /
  `50-problem suite`; they describe **existing figures and tables** and refresh with
  the C2 regeneration.
- The core/extension correction above overlaps **R2.5** (the undocumented 28) and
  **T09**. T09 owns placing all 70 Appendix D.1 rows; this session added only D2's 20.

Transient inconsistency, known and accepted: the revised setup describes 70 problems
at 20 seeds while every reported result still comes from the 50-problem, 30-seed
campaign. It resolves when C2 lands.

Original page-cost estimate, for T13's ledger:

| File | Change | Page cost |
|---|---|---|
| `computational_experiments.tex` `sec:benchmarks` | `50` → `70` problems; one paragraph adding the ODE-Strogatz tier and the Feynman remainder, and stating that the selection rule was pre-registered and outcome-blind | ≈ ⅓ column |
| `computational_experiments.tex` | `30 seeds` → `20 seeds`, in the **same paragraph** as the `N` increase, with the power arithmetic | ≈ ¼ column |
| `results.tex` | CPDT reported at `N = 50` and `N = 70`; per-tier breakdown | table row + ≈ 2 sentences |
| `related_work.tex` §2.4 (Ezequiel's) | no change needed — it already names both databases correctly | 0 |
| Appendix D.1 (supplementary) | 70 rows, generated by `experiments/scripts/generate_appendix_d1.py --all` | T09 owns; ≈ 2 pages |
| Appendix D.2 (supplementary) | D2's operator set, per `EXECUTION-PLAN.md` A4b | ≈ 3 lines |

### 9.3 Draft response text

**Superseded 2026-08-03.** The R3.1 answer is written into
`reviews/response_to_reviewers.tex` (letter Table 4 carries the coverage counts;
the pending statistics sit in a red `\todoblock`). Two numbers in the draft below
were **not** used, because they do not survive checking:

- `≈45,840 runs, 7.6× the campaign` double-counts. SRBench's ground-truth track
  *contains* 116 of the 120 AI Feynman equations, so `120 + 250` is not a union.
  The shipped letter says: 256 distinct problems, `3 × 2 × 256 × 20 = 30,720` runs
  against C2's 8,400.
- `36 of 50 problems exhaust the ceiling` overstates the source, which measured a
  *mean wall-clock at* the ceiling. The shipped letter says the weaker, true thing.

No probe number reached the letter. The 25-minute ρ values and the 16/20 saturation
count are provisional until C2, so the saturation risk is stated qualitatively
(coverage, not a smaller p-value) with `Strogatz-vdp2` named as the concrete case.

Original draft, kept for the record:

```latex
%% --- R3.1 ---
\begin{response}
We thank the reviewer for this question, and we answer it in two parts: what the
selection criteria actually exclude, and what we have added.

We first quantified the criteria over the full AI~Feynman database. Of the $120$
equations, $116$ are representable in $\Sigma_{\mathrm{SR}}$; four are not, and we
name them: I.26.2 and I.30.5 require $\arcsin$, one bonus equation requires
$\arccos$, and II.35.21 is stated with $\tanh$. The last of these is representable
as a finite composition, $\tanh(u) = (e^{u}-e^{-u})(e^{u}+e^{-u})^{-1}$, so the
strictly semantic count is three. Operator compatibility is therefore a weak filter
on AI~Feynman, and we no longer rest the exclusion on it. What binds is criterion
(iv), which caps redundant coverage: the database contains many equations that
differ only in the number of multiplied factors and add no difficulty axis the suite
lacks. Cost binds as well. Each added problem costs $3 \times 2 \times 20 = 120$
runs at a $12$\,h budget; the full database together with SRBench would be
$\approx 45{,}840$ runs, $7.6\times$ the campaign reported here.

SRBench requires a different answer, because its $250+$ problems are two distinct
objects. Its ground-truth track holds $130$ datasets, $116$ from AI~Feynman and $14$
from ODE-Strogatz. Its black-box track holds $122$ real-world datasets with no
ground-truth expression, so criterion~(i) has no expression whose provenance can be
checked, criterion~(iii) has no structure whose difficulty can be assessed, and
solution recovery --- one of our reported metrics --- is undefined rather than
merely difficult. The black-box track is out of scope by construction.

We did not stop at the justification. The suite now includes all $14$ ODE-Strogatz
problems, so the component of SRBench's ground-truth track that was previously
absent is now covered in full, and six further AI~Feynman equations. The suite grows
from $50$ problems to $70$, and its coverage of SRBench's ground-truth track from
$24$ of $130$ to $44$ of $130$.

Because the Cross-Problem Dominance Test treats each problem as one paired
observation, raising $N$ lowers its $p$-value whenever the per-problem differences
are non-negative. We therefore fixed the selection rule before running anything: all
$14$ ODE-Strogatz problems with no filter, and six AI~Feynman equations drawn
uniformly from the $92$ that satisfy criterion~(ii) and were not already in the
suite. The draw seed is derived from the eligible set itself rather than chosen, so
the rule has no free parameter, and we applied no stratification, since every
candidate stratifying variable correlates with the structural difficulty along which
we had previously characterised the method's behaviour. The rule and the script that
executes it were committed before the draw; the drawn list was committed after.

We report the test at both sizes. At $N=50$ the $p$-values are
[\,$p_{\mathrm{UDFS}}$\,] and [\,$p_{\mathrm{Bingo}}$\,]; at $N=70$ they are
[\,$p_{\mathrm{UDFS}}$\,] and [\,$p_{\mathrm{Bingo}}$\,]. Table~[\,X\,] breaks the
result down by tier, so that the effect of the extension can be separated from the
effect of a larger denominator.

One change travels with this one and we state it here rather than elsewhere. The
revised campaign compares three arms instead of two, and at $30$ seeds that is
$151{,}200$ core-hours, which we cannot fund. We run $20$ seeds. This is a
deliberate trade of seeds for problems: the Cross-Problem Dominance Test pools over
problems, so it gains from $N=50 \to 70$ far more than it loses from the
per-problem standard error growing by $\sqrt{30/20} = 1.22$, while the supplementary
per-problem tests lose power by that same factor and their minimum attainable
two-sided $p$ rises from $2^{-29}$ to $2^{-19}$. The primary metric strengthens, the
supplementary tests weaken, and we report both. Reviewer~1's assessment of the
protocol is one we have tried to honour: the extension enlarges the evidence and
leaves the Demšar-style paired inference, the pre-declared inclusion criteria and
the per-problem reporting exactly as they were.
\changeref{}
\end{response}
```

### 9.4 Residual risk

What a round-2 reviewer can still object to, in descending order of how likely it is
to be raised:

1. **"You added 20 problems and your p-value fell. Convenient."** Mitigated, not
   eliminated. The mitigation is checkable rather than rhetorical: the rule and its
   script are in `d95e7d9`, the drawn list in `0e4a573`, and the seed is a function
   of the eligible pool so there is no parameter to have varied. Cite both hashes in
   the letter. A reviewer who re-runs the script gets the same six ids.
2. **"Six of 92 is a thin sample."** True, and the honest answer is the cost
   arithmetic: 1,440 core-hours per problem against a committed 100,800. The weak
   point is that a *different* six could give a different tier-level result; the
   per-tier breakdown (AC-7) is what exposes that rather than hiding it.
3. **"Why is the black-box track still absent?"** The answer is principled — no
   ground-truth expression means criteria (i) and (iii) have no referent and
   `solution_recovered` is undefined — but R3 may reasonably reply that R² and NRMSE
   are still computable there. If they do, the concession is real: the black-box
   track is excluded because *some* of our metrics are undefined on it, not all.
   Prepare that reply rather than improvise it.
4. **The seed reduction.** R1 praised "50 problems, 30 seeds". We are returning 70
   problems and 20 seeds. §6.3's arithmetic is sound and the draft states it in one
   paragraph, but this is the change most likely to draw a comment.
5. **Dilution — no longer a risk, a measurement.** The probe found **16 of 20 Bingo
   cells at R² ≥ 0.999 on a 25-minute budget**, against a production budget of 12
   hours, and the local smoke showed `baseline` saturating too. Saturated problems
   contribute δᵢ ≈ 0, and ties *weaken* CPDT. The probe ran only the `isalsr` arm so
   no δᵢ exists yet, but this is the honest reading and §5.4 of the rule document
   pre-commits us to reporting it. **The extension's defensible claim is coverage,
   not a smaller p-value** — write the letter that way from the start rather than
   rewriting it in September.
6. **Trajectory-data leakage.** SRBench's 75/25 random split of ODE trajectory data
   puts temporally adjacent, nearly identical points on both sides of the split. We
   follow the published protocol, and the leakage is identical across all three arms
   so it cannot bias the paired contrast — but it does inflate the absolute R² on
   these 14 problems and should be stated once in Appendix D.1 rather than left for a
   reviewer to notice.
7. **Page cost** colliding with R2's C6. Estimated in §9.2 for T13.
8. **Not visible to reviewers, but real:** five D1 definitions changed between the
   submitted and revised campaigns (internal record only, by decision). The C1↔C2
   continuity table must exclude those five rows or it will report a shift that is an
   artefact of the correction rather than of the engine.
