# Stage D — design, decisions and measurement method

**Date:** 2026-08-04
**Branch:** `feature/experiment-fairness-audit`
**Scope:** EXECUTION-PLAN §4.4 (D1.1–D1.8, D2, D3), locked by audit.md §7.

Stage D is the full-length certification: 12 tasks at the production 43,200 s
budget, 144 core-hours. It exists because **the 15-minute smoke proves nothing
about a 12-hour run**. Memory growth, heap fragmentation, dedup-set size,
timeout paths and convergence behaviour are all budget-dependent, and C1's
dominant failure — 36 `OUT_OF_MEMORY` cells, 29 of them at
`MaxRSS ≈ 127.7 GB` against a 128 GB request — occurred after *hours* of
evolution, in a state a 900 s run cannot reach.

---

## 1. What Stage D certifies, and what it does not

| Certified here | Left to campaign C2 |
|---|---|
| The pipeline survives 12 h on both hosts, all three arms | Statistics over 20–30 seeds |
| Production `--mem` per (method, arm), from measurement | Per-problem significance |
| The two C1 NaN cells are finite at full budget | The 70-problem cohort |
| Overhead under the corrected accounting | The paper's headline numbers |

Stage D runs **one seed** (101). That is deliberate: it certifies *behaviour at
budget*, not statistics. A 12-cell design with 30 seeds would cost 4,320
core-hours to answer a question Stage D answers with 144.

---

## 2. The 12 cells, and why these twelve

Locked by audit.md §7 row 2, implemented in
`experiments/scripts/stage_d_task_spec.py`.

| Group | Cells | Rationale |
|---|---|---|
| `udfs` | Pagie-1 × {baseline, hash, isalsr} | 3 |
| `bingo_std` | {Pagie-1, Korns-12, Vlad-2} × {baseline, hash} | 6 |
| `bingo_isalsr` | {Pagie-1, Korns-12, Vlad-2} × {isalsr} | 3 |

**Pagie-1 as the trace problem.** The 2026-04-19 bottleneck analysis found
IsalSR helps *if and only if* the primary difficulty is structural search.
Pagie-1 is in the `structural` class (k = 7–8, integer constants, 5/5
`sig_train`), so it is a problem where the mechanism under test is actually
exercised. A problem IsalSR cannot help would certify the plumbing and nothing
else.

**Korns-12 and Vladislavleva-2** are the two cells that were NaN in the
submission. They are the T08 AC-7 evidence, and D1.4 is the criterion that
would stop the campaign if the root cause is still live.

**Why UDFS runs only Pagie-1.** UDFS's per-candidate cost is ~64× its
canonicalisation cost, so its overhead axis is settled (C1: 0.6 %). Six more
UDFS cells would buy little. The one problem it does run is the one that also
has a Bingo counterpart, so the cross-host comparison is paired.

### The registry is the single source of truth

`stage_d_task_spec.py` holds the enumeration, and both consumers read it:

- `slurm/c2_stage_d/worker.sh` via `--index`, which emits `D_KEY='value'`
  shell assignments for `eval`;
- `experiments/scripts/stage_d_certify.py` via the Python API.

The shell interface is **key=value, not positional**. Stage C's first wave lost
265 tasks to a decode that silently shifted (`--export` ate a comma and
delivered `C2_SEEDS=0`), producing complete, plausible, wrong cells. A
positional contract makes adding a field a silent corruption; a key=value one
makes it a loud `unbound variable`.

---

## 3. Three arrays, not one

`sbatch --mem` is **per job**, not per array task, and the three memory classes
differ by 16×. A single 12-task array would have to request 256 GB for all
twelve; on `sr` (439 GB/node) that serialises the whole stage onto one task per
node for cells that need 16 GB.

| Group | `--mem` | Basis |
|---|---|---|
| `udfs` | 16 GB | No large dedup set. Stage C measured 0.67 GB peak at 900 s |
| `bingo_std` | 32 GB | Baseline holds no dedup set; hash holds 64-bit digests |
| `bingo_isalsr` | 256 GB | §3.3's evidence floor from C1's 127.7 GB ceiling |

The 256 GB figure is a **floor set from evidence, not a substitute for
measuring**. §3.3 is explicit that if D1.2 shows the 12 h peak comfortably under
128 GB under the C++ dedup set, the request may be revised *down* before launch,
with the measurement recorded. That revision is the point of §4.

---

## 4. Memory measurement — the method, and why sacct alone is not enough

Mario's addendum to audit.md §7 row 3: *measure memory properly so production
can request less without OOM risk.*

`sacct` reports **one number per job**. It answers "did it fit". It cannot
answer "how much of the headroom is real", because a single peak cannot
distinguish a brief allocation spike from a sustained plateau — and those two
shapes justify very different production requests.

### The sampler

`slurm/c2_stage_d/worker.sh` backgrounds the payload, captures its PID, and
runs a sidecar loop that reads `/proc/<pid>/status` every 60 s, appending
`timestamp_s,vmrss_kb,vmhwm_kb` to `<seed dir>/rss_timeseries.csv`.

**Why 60 s is sufficient despite being coarse.** `VmHWM` is the kernel's own
high-water mark and is **monotone**. The peak it reports does not depend on when
we sample — only the *shape* of the `VmRSS` curve does. So the last row's
`vmhwm_kb` is the exact peak up to that instant at any sampling rate, while the
`vmrss_kb` column supplies the growth curve that motivates the whole exercise.
A 12 h run yields ~720 rows, ~20 KB — negligible against the FSCRATCH budget.

**Verified, not assumed.** The sampler was run locally against a payload that
allocates 6 × 30 MB and then releases it. Over 16 samples: the header is
`timestamp_s,vmrss_kb,vmhwm_kb`, `vmhwm_kb` is monotone non-decreasing, and the
peak `VmHWM` of **190.2 MB** is retained while `VmRSS` falls to **10.4 MB** by
the final sample. Sampling `VmRSS` alone would have reported ~10 MB and missed
the peak by 18×. This is the concrete reason both columns are recorded and why
the recommendation is built from `VmHWM`.

**Scope difference, carried explicitly.** The sampler observes the *payload
process*; `sacct MaxRSS` accounts the *whole cgroup*. Neither dominates the
other in general, so D1.2 takes the **maximum of the two** rather than trusting
either alone, and reports which source won.

### The recommendation

D1.2 emits, per (method, arm): observed peak and its source, p50/p95 of the
`VmRSS` series, the request, realised headroom, the recommended production
`--mem` at peak + ≥30 % headroom, and the resulting margin. That is the number
Stage F item 4 requires, and it is derived from measurement rather than history.

### The two sacct traps, restated because both have already bitten

- **Never `sacct -X`.** It returns an **empty** `MaxRSS`; memory is accounted on
  the `.batch` step. The profile comes back silently blank — no error, just
  empty cells.
- **Never join on `JobID`.** For an array it reads `<array_id>_<task>` while
  `status.json` records the raw numeric id. Joining on `JobID` matched 42 of
  1,260 rows and still reported PASS.

---

## 5. D1.6 — what "a defensible neighbourhood" means

D1.6 compares the 12 h ρ and R² against C1 for the three Stage D problems.
Three intended changes stand between the campaigns, and the neighbourhood is
defined against them rather than as a tolerance band pulled from nowhere.

| Change | Expected effect | Not an explanation for |
|---|---|---|
| T16 decomposition (`k` +≈22 %) | ρ **rises** — more internal nodes to permute | any change in R² |
| Python → C++ canonicaliser | cost only | any change in ρ or R² |
| Third (hash) arm added | new; no C1 counterpart | anything about the isalsr arm |

**ρ:** expect C2 ≥ C1 in direction. A *drop* is the alarm, and it has exactly
two candidate causes: the decomposition is not reaching the canonicaliser, or
the dedup population changed. The first is falsifiable directly by SP-4 — count
SUB/DIV tokens in the run's own canonical strings — and C5 already falsified it
on the smoke wave (0 violations in 292 strings). The second is what D1.6, at
equal budget, is for.

**R²:** both campaigns are at 43,200 s here, so unlike C5 there is no budget
asymmetry to explain a difference away with. A material excess means the
dataset, the split or the metric changed, and the check against that is C4's
data fingerprint.

**Use medians, not means, for Pagie-1.** C1's Bingo–isalsr Pagie-1 arm carries a
catastrophic outlier seed: mean −209.9 against median 0.746. The mean is not a
usable comparator and there is a remediation script for exactly this
(`experiments/scripts/fix_pagie1_outliers.py`).

**A constraint on the reference discovered while drafting C5:** the
`wl_subtree_unified/` tree is a directory of **dangling symlinks** — the raw
data was relocated. Only `analysis/` holds real content there. Absolute
per-problem values must be recomputed from
`.../real_benchmarks/wl_subtree_hard/models_hard/`, which does resolve, and the
two sources agree on ρ to three decimals.

---

## 6. D1.7 — the overhead figure will rise, and that is the accounting

Under this branch's cost-attribution fix (F-7 / F-8):

```
overhead = canonicalisation + conversion        (shadow reported separately)
```

Pre-merge code booked untimed wrapper work as "search", understating the
wrapper cost by **1.6–2.4×**. So the Bingo overhead percentage is **expected to
come out above** the old canon-only projection of ≈7.4 %.

**This must be read as an accounting change, not a regression**, and the
certifier's report says so in those words. Only a *missing or zero* `T_canon` or
`T_eval` fails D1.7 — the percentage itself is a measurement, not a threshold.

Shadow is kept separate because audit.md §6 decision 3 defers the
keep-or-drop call on shadow sketches until `shadow_time_s` can be read from
these 12 h cells. Folding it into overhead would destroy the input to that
decision.

---

## 7. D2 — the trace, and the sampling decision

The D2 trace is enabled for **exactly one cell** — Bingo × Pagie-1 × isalsr ×
seed 101 — through `ISALSR_STAGE_D_TRACE*` environment variables that the
worker sets from the registry's own `trace` field. Gating on the registry rather
than on a launcher flag means the launcher cannot enable it for the wrong cell.

The tracer is a strict **no-op** when the variables are unset, so the other 11
cells and all 8,400 campaign runs pay nothing.

### Why sampling is not optional

A single 900 s Bingo cell sampled **711,419** candidates (§11.1, B9). A 12 h
cell is therefore order **3.4 × 10⁷** candidates. At ~400 bytes per JSONL
record, full-stream persistence is ≈14 GB **for one cell**.

The multiplier matters and is easy to get wrong: **D2 is one cell, not 8,400**.
The 8,400 multiplier applies to the campaign-wide per-candidate stream (P1),
not to the D2 trace. `stream_size.md` reports both so the two are not conflated.

Sampling must be **deterministic**, not an unseeded RNG, or the spot check and
the D3 replay are not reproducible.

### The five artefacts

1. `candidates.jsonl` — per sampled candidate: `k`, node-label multiset,
   canonical string and its hash, the three fixed-order **digests**, `T_canon`,
   `T_eval`, fallback path, dedup-hit flag.
2. `canon_cost_hist.json` — canonicalisation cost stratified by `k` (feeds T10).
3. `fallback_ledger.md` — the five T06 rates with a worked example of each
   residual post-normalisation violation.
4. `spot_check.json` — 20 candidates re-canonicalised in **pure Python**,
   matched **byte-exact** against the C++ output recorded during the run. This
   is the end-to-end check that the engine used *in production* is the engine
   the gate certified — SP-3's negative control, applied to a real stream.
5. `stream_size.md` — measured bytes/run at the chosen rate, multiplied out, and
   checked against FSCRATCH headroom (94.6k inodes).

**Digests, not hashes.** `fixed_order_hash` uses Python's `hash()` and is
PYTHONHASHSEED-salted, so it is **not replayable across processes**.
`fixed_order_digest` is stable and is what gets persisted. Persisting a salted
value would produce a stream that replays to different answers on every run —
a defect that would look like a soundness violation.

---

## 8. D3 — Mode 1 replay

Replays the persisted stream through the three fixed-order hashers **and**
through canonicalisation on identical input sequences: the controlled
comparison, same inputs, zero search confound. Produces ρ_exact, ρ_iso and
ρ_total per method, stratified by `k`.

Two checks only Mode 1 can make:

- **Hash soundness (T04 AC-1).** Any two DAGs sharing a fixed-order digest must
  share a canonical string. A violation is an **unsound merge** and kills the
  arm — it exits non-zero and names the counterexample pair. Not a warning.
- **IsalSR soundness.** Any two DAGs sharing a canonical string must satisfy
  `is_isomorphic`, spot-checked on the largest equivalence classes.

**A result already in hand from the smoke wave.** ρ_hash = **1.0000 on all 210
UDFS cells** (against ρ_isalsr = 1.6552), while on Bingo the hash arm does merge
(1.7247 against 1.7814). This is the live-search analogue of the outcome §4.4
anticipates: if ρ_exact ≈ 1.00 the live hash arm is a null result, *which is
itself the answer to R1.4*. The smoke says it is null on UDFS and emphatically
not on Bingo — a more informative answer than either uniform outcome, and one
§10.1 should record that we knew before the campaign ran.

---

## 9. What was decided, and what was left open

| Decision | Taken | Rationale |
|---|---|---|
| Registry location | `experiments/scripts/stage_d_task_spec.py`, single copy | Two copies drift; Stage C's decode bug cost 265 tasks |
| Worker↔registry contract | `D_KEY='value'` for `eval` | A positional contract makes a new field a silent shift |
| Submission topology | 3 arrays, one per memory group | `--mem` is per job; 16× spread |
| RSS sample period | 60 s (15 s for the SP-0 probe) | `VmHWM` is monotone, so the peak is exact at any rate |
| Peak definition | `max(sacct MaxRSS, timeseries VmHWM)` | Different scopes: cgroup vs payload process |
| Aggregation dependency | `afterany` | A cell that OOMs must still be certified — that is D1.2's purpose |
| Manifest strictness | `strict_campaign=False` | 12 cells / 1 seed / 3 groups; the strict validator rejects all of that by design |
| Trace gating | Registry field, not launcher flag | The launcher cannot enable it for the wrong cell |

**Left open, by design:** whether shadow sketches ship in C2 (audit.md §6
decision 3 — decided after reading `shadow_time_s` from these cells), and
whether Bingo–IsalSR's 256 GB request is revised down (§3.3 — decided from
D1.2's measurement).

---

## 10. Known limitation in local validation

The Stage D shell scripts pass `bash -n` and the registry CLI they invoke is
unit-tested and returns 0 standalone, but end-to-end local execution of
`launcher.sh` exits 1 with no output at the first external command under
`set -e`. This reproduces for a minimal prefix of the script and does not
reproduce for an equivalent standalone script, so it is an artefact of the local
sandboxed shell, not of the launcher.

This is not a gap worth closing locally: `sbatch` does not exist on the
workstation, so a launcher can only be validated where it runs. RUNBOOK.md
steps 3 and 4 do exactly that — `--dry-run`, then `--test-only`, then a single
SP-0-capped probe of the real worker — before anything is submitted.
