# Execution plan — Picasso campaign waves

Single source of truth for **what gets launched, in what order, and under what
gate**. Referenced by T02, T03, T04, T05. If a ticket and this file disagree about
a launch, this file wins; update the ticket.

**Decisions taken 2026-07-27 (Mario):**

- The **C++ re-execution is the priority and launches first.** It is the headline
  result of the revision. Everything else is scheduled around it.
- **Early stopping is abandoned.** Full 12 h budget on every run, every arm. See §4.
- **Gray is secondary** and reserves no queue capacity. See Wave 4.
- **Nothing launches as an array until the certification gate in §2 passes.**

**Amendment 2026-07-30 (T16) — the IsalSR arm needs a FULL re-run, on the corrected
alphabet.**

Every IsalSR number in the submitted paper was produced under the **wrong
alphabet**. The paper's Σ_SR (Definition 3.2) has twelve labels and **no `-` and no
`/`**: subtraction and division are supposed to enter through the commutative
decomposition `x − y = Add(x, Neg(y))`, `x / y = Mul(x, Inv(y))`, leaving `Pow` as
the only non-commutative operation. The adapters emitted `Sub` and `Div` as
primitive node types anyway, and **61.1 % of production candidates contain them**.
Ezequiel's decision (T16 §4) is to align the *code* to the paper, because TPAMI
acceptance obliges us to publish the source and a paper/code divergence is the
first thing any reader's LLM will find.

**Consequence for this plan: none on the budget, everything on the gate.** Wave 1
already re-runs IsalSR on all of `S50` and Wave 2 on `EXT`, so the corrected
alphabet adds **zero** additional runs — `S50 + EXT` was already the scope. What
changes is that **Wave 1 must launch against the corrected adapter**, and that the
IsalSR arm now has *two independent reasons* its submitted numbers are void: the
C++ engine change (T02) and the alphabet correction (T16). The `baseline` arm is
still not re-run on `S50` (§5) and that remains sound — **the baseline never invokes
the adapter**, so the alphabet correction cannot touch it.

The quantities that move are `k`, canonical string length, ρ and the reduction
factor, canonicalisation cost, and every k-stratified table. Fitness and every
fitness-derived metric (R², NRMSE, solution recovery) do **not** move: fitness is
computed by the host on the host's own representation and the runners cache
`canon_hash -> fitness` without ever calling `evaluate_dag`. See T16 §6.

---

## 1. Notation

| Symbol | Meaning | n |
|---|---|---|
| `S50` | the suite as submitted | 50 problems |
| `EXT` | Feynman remainder + ODE-Strogatz (T05) | ≈ 20 problems |
| `S70` | `S50` + `EXT` | ≈ 70 problems |

Arms: `baseline`, `isalsr`, `hash`, `gray`. Methods: UDFS, Bingo. Seeds: 30.
Budget: `max_time = 43,200 s` (12 h), 1 core per run.

This repo's launchers emit **one SLURM array per (method, arm)** — as documented
for `hard_launch.sh` ("4 arrays × 300 tasks"). A 2-method × 2-arm wave is therefore
4 arrays, not one.

---

## 2. Wave 0 — certification gate (no compute)

> **Do not submit an array to Picasso unless you are 100% sure the code is
> correct.** A 300-task array failing identically costs 300 allocations and a day of
> queue time; a *subtly wrong* array costs the deadline, because the error is found
> during analysis in September.

Every condition must hold, with evidence, before Wave 1:

| # | Condition | Evidence |
|---|---|---|
| G1 | T01 equivalence gate passed: byte-exact canonical strings vs Python, exhaustive k=1..8, the 14,841-DAG corpus, and ≥100,000 replayed evolved DAGs | T01 AC-3 report, zero mismatches |
| G2 | `pytest` + `ruff` + `mypy --strict` clean on the exact commit being synced | command output, not a claim |
| G3 | Local smoke completes and its `run_log.json` **parses** and carries every field the analyzer reads | parsed output, not file existence |
| G4 | Smoke on an `isalsr` arm shows `n_duplicates_eliminated > 0` | a zero here means the dedup hook is dead and the whole arm is a null result |
| G5 | Remote import check confirms the **native** engine loaded on a compute node, not a silent pure-Python fallback | `_ENGINE` value from a compute node |
| G6 | `sbatch --test-only` passes | exit 0 |
| G7 | **One** real task submitted (`--array=1-1`), completed, log read, output validated | the single most valuable 12 h of the campaign |
| G8 | `MANIFEST.json` writes correctly: git commit, build hash, compiler flags, node CPU, engine, config hash, seed | inspect the file from G7 |
| G9 | **T16 alphabet.** The adapter being synced emits **no** `NodeType.SUB` and **no** `NodeType.DIV`, and no canonical string contains `-` or `/`. Assert on the G7 task's own candidate stream, not only in unit tests. Run `bash slurm/alphabet_gate/launcher.sh` (~90 s) | a zero count from the real run; a non-zero count means Wave 1 is measuring the wrong alphabet and every ρ it produces is void. **PASSED on Picasso 2026-07-30, job 1692451**, CPU node, native C++ engine confirmed: 5,551 / 5,551 / 225 / 461 DAGs across `bingo_nguyen`, `bingo_hard`, `udfs_nguyen`, `udfs_hard`; `POW` present only in `bingo_hard`; zero forbidden labels. **Re-run against the exact Wave-1 commit** |

G7 is not optional and is not parallelisable with the full launch. Everything that
only appears on a compute node — module differences, dataset paths, permissions,
memory profile, env activation — appears there and nowhere else.

Procedure and commands: `.claude/skills/review-ticket/references/picasso-loop.md`.
SLURM directives: invoke the **`picasso-sbatch`** skill; it is the authority.

---

## 2b. What must land before Wave 1 launches

The dividing line is simple: **anything measured *during* a run must be in the code
before launch; anything computed *after* can land later.** Getting this wrong means
re-running 3,000 jobs to recover a counter.

### Blocking — Wave 1 cannot launch without these

| Ticket | What exactly | Why it blocks |
|---|---|---|
| **T01** | Whole ticket, equivalence gate passed | It is the engine. |
| **T06** | The **instrumentation half only** (§4.1 counters for the five fallback paths) — not the analysis, not the write-up | R1.2 wants violation rates on the DAGs *arriving at the canonicaliser during real searches*. That population exists only while Wave 1 runs. Miss it and the only way back is a second campaign. |
| **T08** | The **root-cause half only** (§5.1) — why Bingo–IsalSR produced NaN on Vlad-2 / Korns-12 and why 35 Bingo cells were missing. Plus any *runtime* fix it implies (memory profile, the B12 clone path) | If the cause is still live, Wave 1 reproduces it at full scale. The analyzer-side fixes (NaN-as-winner, NaN policy in the paired test) are post-hoc and are **not** blocking. |
| **T02** | §5.3 MANIFEST schema, frozen | Written by the runs themselves. A field added afterwards is a field you do not have. |
| **T16** | The **adapter decomposition** (`experiments/models/commutative_encoding.py` + both adapters). Implementation and validation complete 2026-07-30 | The alphabet is baked into every candidate the run canonicalises. Launching Wave 1 on the old adapter means re-running all 3,000 jobs, and — worse — the error is invisible in the logs, so it would only surface during September analysis. |

### Three engineering checks that are nobody's ticket but will bite

| # | Check | If it fails |
|---|---|---|
| P1 | **Do the per-run logs persist what T04 Mode 1 needs?** T04 §5.1 replays "the stored DAG streams from the T02 campaign". Confirm the DAG/canonical-hash stream is actually persisted — and at what sampling rate, since full persistence is millions of entries per run. | Add the logging (a hash stream or a fixed-rate sample is enough) before Wave 1, or Mode 1 can only ever replay Wave 3's own runs — losing the `isalsr`-arm decomposition that answers R1.4. |
| P2 | **Do the cost fields survive the C++ port?** `T_canon` and `T_eval` per DAG feed T10's break-even analysis and R1.1's whole answer. | Restore them before launch. |
| P3 | **Node-type pinning (§5 mitigation).** Run D2 first, then pin `--constraint` to match the March baseline's predominant node type. | Skipping this leaves the `S` confound unbounded, and the baseline is not being re-run. |

### Explicitly NOT blocking

- **T03 (Gray)** — Wave 4, spillover, reserves nothing.
- **T05** — Wave 2 only. Wave 1 launches on S50 without it.
- **T04** — Wave 3. Its *logging* dependency is P1 above; its code is not needed.
- **T07** — proofs cost no compute. Only constraint: T06's definition of a
  precondition violation must match T07's statement, so agree the definition first,
  then instrument.
- **T09, T10, T11, T12, T13** — all post-hoc or manuscript-side.

---

## 3. The waves

### Wave 1 — C++ headline (T02) · **priority, launches first**

| Array | Arm | Suite | Runs |
|---|---|---|---|
| UDFS-isalsr | `isalsr` | S50 | 1,500 |
| Bingo-isalsr | `isalsr` | S50 | 1,500 |

**3,000 runs, 2 arrays.** The `baseline` arm is **not** re-run here (§5) — its March
numbers stand and are paired against these.

**Runs the T16-corrected adapter.** Every candidate is canonicalised under the
paper's alphabet: no `Sub`, no `Div`, `Pow` the only non-commutative operation.
This is gate G9 and it is checked on the G7 task's real candidate stream, not only
in unit tests.

Carries the measurement add-ons required by T06 (reachability and fallback
counters). **Verify at G3 that they do not perturb the timings T10 depends on** — if
they do, they come out of Wave 1 and T06 gets a separate subsampled characterisation
run instead. A violation *rate* does not need the full campaign; a paired *timing*
does.

### Wave 2 — extension (T05)

| Array | Arm | Suite | Runs |
|---|---|---|---|
| UDFS-baseline | `baseline` | EXT | 600 |
| UDFS-isalsr | `isalsr` | EXT | 600 |
| Bingo-baseline | `baseline` | EXT | 600 |
| Bingo-isalsr | `isalsr` | EXT | 600 |

**2,400 runs.** Both arms run here regardless of §5 — `EXT` problems have no prior
data at all.

Launches when T05's problem definitions and unit tests land (target 2026-08-17),
**into the same campaign root and MANIFEST as Wave 1**. Do not let Wave 2 block
Wave 1; splitting the launch does not split the provenance as long as the root,
the engine build and the configs are identical. Record the split in the MANIFEST.

### Wave 3 — hash comparator (T04) · decided: **full live arm on all 70**

| Array | Arm | Suite | Runs |
|---|---|---|---|
| UDFS-hash | `hash` | S70 | 2,100 |
| Bingo-hash | `hash` | S70 | 2,100 |

**4,200 runs.** Launches when T04's serialisations pass their soundness tests on the
14,841-DAG corpus (T04 AC-1). Mode 1 (offline replay for the ρ_exact / ρ_iso
decomposition) costs no queue time and should be done **before** this wave — if the
replay shows the hash catches almost nothing, that is worth knowing before spending
50,000 core-hours.

Three arms changes the statistics: pairwise CPDT with Holm across three contrasts,
plus Friedman/Nemenyi. See T04 §5.3. Do not reuse the two-arm machinery.

### Wave 4 — Gray ablation (T03) · **spillover only**

| Array | Arm | Suite | Runs |
|---|---|---|---|
| UDFS-gray | `gray` | S70 | 2,100 |
| Bingo-gray | `gray` | S70 | 2,100 |

**4,200 runs. Reserves no capacity.** Launches only if Waves 1–3 are complete and
the queue is free.

**Go/no-go date: 2026-08-31.** A 12 h campaign launched after that cannot finish,
be analysed, and reach the 2026-09-10 number freeze. If the date passes, T03 ships
as design + implementation + theory with the ablation declared as characterised
future work, and the C++ results remain the headline. That is an acceptable
outcome, not a failure.

---

## 4. No early stopping

An internal saturation-based early stop was considered on 2026-07-27 and
**rejected**. Recorded here so it is not re-proposed.

Wall-clock `T` is a **reported** quantity: it produces `S`, the overhead
percentages, and the cost column of Table 2. R1.1's entire complaint is about `S`.
A stop rule that fires at different times on the `baseline` and `isalsr` arms of the
same (problem, seed) pair — and IsalSR saturating earlier is precisely our
hypothesis — would make `S` a measurement of the stop rule rather than of the
method. ρ is equally exposed: ρ = evaluations / unique canonical strings, and
truncating a run truncates both terms non-proportionally.

**Every arm runs the full 43,200 s budget.** The protocol's own convergence
criterion (exact solution recovery, already implemented in
`evolve_until_convergence`) continues to terminate runs as it always did; that is
not early stopping, it is the protocol.

---

## 5. SETTLED — the `baseline` arm is **not** re-run on S50

**Decision taken 2026-07-27 (Mario), reconfirmed 2026-07-30 against T16.** The
`baseline` code path is untouched by the C++ port **and by the T16 alphabet
correction** — it never invokes the adapter at all — so its S50 numbers do not
change and re-running them buys nothing worth 36,000 core-hours. Only two things get
fresh compute: **IsalSR** (full re-run of `S50 + EXT`, now for two independent
reasons: the C++ engine and the corrected alphabet) and the **naive hash dedup** arm
(fresh full run, on decomposed DAGs — T16 §5.2). Baseline runs only on `EXT`, where
no prior data exists.

### The residual confound — characterise it, do not ignore it

Reusing the March baseline leaves one exposure that must be **measured and
disclosed**, not assumed away. Both arms are compared under a *fixed wall-clock
budget*, which makes two quantities hardware-sensitive:

1. **`S = T_BL / T_IS` absorbs any node difference.** Picasso mixes Intel Xeon Gold
   6230R and AMD EPYC 7H12; single-core performance differs by roughly 10–25%. `S`
   needs to move about 7% (0.93 → >1.0).
2. **A faster node buys more generations under a fixed budget, hence a better R².**
   If the March baseline drew slower nodes it was disadvantaged, which would inflate
   the headline CPDT claim.

Both effects bite only on problems that finish inside the budget. Problems that
saturate the 12 h ceiling have `T_BL = T_IS = 43,200` and contribute nothing either
way — 36 of 50 UDFS problems are in that state, which materially limits the exposure.

### D1–D3 — now mandatory characterisation, not a decision gate

≈30 min, no queue time. Run before Wave 1 analysis; report the result.

| # | Check | What it settles |
|---|---|---|
| D1 | Read the definition of `S` in `experiments/models/analyzer/`. Is it a ratio of measured seconds, or normalised by evaluations? | If normalised, exposure 1 vanishes outright and can be stated in one sentence. |
| D2 | From the March campaign's run logs, tabulate node CPU model per run, per arm. | If the pool was homogeneous — or matches the new pool — the exposure collapses to a bound. |
| D3 | Compare per-problem `T_BL` across node types in the March campaign. | Measures the confound directly instead of assuming a magnitude. |

**Mitigation, applied at launch:** pin the new IsalSR runs to the same node type the
March baseline predominantly used (`--constraint`, per the `picasso-sbatch` skill),
once D2 identifies it. This removes most of the exposure for free and is far cheaper
than re-running 3,000 baseline runs.

**Disclosure:** whatever D1–D3 return goes into the paper as a stated limitation and
into T02 §8.4 as a residual risk. If R1 asks whether both arms ran on the same
nodes, the answer must be a number, not a shrug.

Record the outcome here and in T02 §7.

---

## 6. Budget

| Wave | Arms | Runs | Core-hours |
|---|---|---|---|
| 1 | isalsr, S50 | 3,000 | 36,000 |
| 2 | baseline + isalsr, EXT | 2,400 | 28,800 |
| 3 | hash, S70 | 4,200 | 50,400 |
| **Subtotal (committed)** | | **9,600** | **115,200** |
| 4 | gray, S70 — spillover only | 4,200 | 50,400 |

Launch ≈ 2026-08-10, number freeze 2026-09-10 → 744 hours. Sustaining the committed
campaign needs **≈155 concurrent cores** with no queue loss. Confirm the group
allocation supports this **before** Wave 1.

If it does not, trade in this order and no other:
1. Wave 3's suite (S70 → S50): −1,200 runs.
2. Wave 4: it already reserves nothing, so it simply does not run.
3. Wave 2's seed count, and only with the R3.1 statistical consequence stated.

**Never trade away Wave 1.** It is the headline and the answer to R1.1.

---

## 7. Launch ledger

Fill as waves go out. One row per array.

| Wave | Array | Job ID | Submitted | Tasks | Completed | Failed | Notes |
|---|---|---|---|---|---|---|---|
| 0 | G7 single task | | | 1 | | | certification |
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
