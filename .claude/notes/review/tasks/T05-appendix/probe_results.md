# T05 Appendix B — the D2 Picasso probe

Arrays **1741991** (Bingo, tasks 1–20) and **1742002** (UDFS, 21–40), submitted
2026-08-02 against commit **`fa41e2a`**. 40 tasks = 20 D2 problems × 2 hosts on the
`isalsr` arm, seed 0, `max_time = 1500 s`, output under `~/execs/isalsr/t05_probe/`.
Inside SP-0's caps (≤ 1800 s, ≤ 60 tasks, seed 0, never the campaign root).

**40/40 COMPLETED. Zero failures, zero NaN, zero SLURM time-kills.**

Generated table: `probe_summary_raw.md`, produced by
`slurm/t05_probe/summarise.py`. This document is the reading of it.

---

## B.1 SP-1…SP-6 (AC-4b)

| # | Property | Cells | Verdict |
|---|---|---|---|
| SP-1 | Provenance — running the commit we think we are | 40/40 | **PASS** |
| SP-2 | Installation freshness — the `.so` is the code we edited | 40/40 | **PASS** |
| SP-3 | Engine native | 40/40 | **PASS** |
| SP-3′ | **negative control** — forced Python actually reports `python` | 40/40 | **PASS** |
| SP-4 | Alphabet — no `Sub`/`Div`, no `-`/`/` in any canonical string | 40/40 | **PASS** |
| SP-5 | Both hosts — UDFS and Bingo | 40/40 | **PASS** |
| SP-6 | T06 fallback ledger importable, five paths exposed | 40/40 | **PASS** |

**Read SP-6's wording carefully.** `sp_probe.sp6_counters` imports `FallbackLedger`
and lists its attributes. It never reads a live count. That is strictly weaker than
SP-6 as specified in `EXECUTION-PLAN.md` §4.0, which wants the five paths *"present
and finite in the probe output, at the production sampling rate"*. See §B.4 — the
counters do not reach any persisted artefact at all, so this row must not be quoted
as the stronger claim.

**SP-1 failed the first attempt, and that is the headline.** Jobs 1739900/1739901
died in **7 seconds** on provenance, with four file-hash mismatches. Two causes,
both real: the `.provenance.json` on the cluster was still T04's (stamped at
`a4206b8`), and the first `rsync` sent the **working tree**, which carried another
session's uncommitted `aggregation.py`, `metrics.py` and `schemas.py`. Fixed by
deploying from a clean detached worktree and extending the stamp's tracked globs to
cover D2's files — previously it verified T04's 18 files and would have said nothing
about the vendored Strogatz data or the D2 definitions the probe exists to test. The
stamp now covers **46** files including all 14 `.tsv.gz`.

Had SP-1 not existed, this probe would have produced 40 plausible, clean, green
results from partly-uncommitted code.

---

## B.2 What the probe establishes — SP-7's five statements

| # | Statement | Evidence |
|---|---|---|
| 1 | Datasets load **on Picasso** with the expected shapes | 40/40 tasks ran `check_d2.py --pre` on their compute node before searching. The vendored Strogatz files resolve after `rsync` — the one thing no local check could establish |
| 2 | `sympy_expression` present, so `solution_recovered` is computable | 20/20 D2 problems |
| 3 | Runs 25 min on both hosts without crashing; `run_log.json` parses and validates | 40/40 |
| 4 | Declared operator set is what ran, identical across arms | config-level, checked offline; the three arms read one YAML block per method |
| 5 | No NaN, no inf in any regression metric | 40/40 |

---

## B.3 What it measured

| Host | Tier | n | ρ mean | ρ range | R² median |
|---|---|---|---|---|---|
| Bingo | Strogatz | 14 | 1.750 | 1.171 – 1.806 | 1.0000 |
| Bingo | Feynman rem. | 6 | 1.774 | 1.763 – 1.787 | 1.0000 |
| UDFS | Strogatz | 14 | **2.108** | 1.143 – 2.202 | 0.8036 |
| UDFS | Feynman rem. | 6 | 1.363 | 1.336 – 1.392 | 0.4138 |

Provisional, as every probe number is, until C2 reproduces it. Three things stand out.

**Dedup fires everywhere.** ρ > 1 on 40/40, and Bingo's is strikingly tight —
1.76–1.81 on 19 of 20 problems. C1.6 holds on D2.

**UDFS reduces more on Strogatz than Bingo does** (2.11 vs 1.75), which reverses the
submitted campaign's ordering (UDFS 1.56, Bingo 1.83). The 400-sample, 2-variable
Strogatz problems let UDFS's systematic enumeration revisit the same small skeletons
far more often. Worth watching in C2; it is not a defect, and it is a point in the
extension's favour — the new tier is not simply a copy of the old one.

**`Strogatz-vdp2` is degenerate.** Its target is `−x/10`, a linear function. Bingo
solves it in **0 s** with 427 unique DAGs (ρ = 1.171) and UDFS in **7 s** with 35
(ρ = 1.143). It is the low outlier in both hosts' ranges. It will contribute
`δᵢ = 0` to CPDT and nothing to any k-stratified table. It stays in — removing it
after seeing the result is exactly the post-hoc selection the pre-registration
forbids — but it should be named in Appendix D.1 rather than left for a reviewer.

**Budget honoured.** Bingo terminated at 1469–1471 s and UDFS at 1500–1501 s against
a 1500 s budget; no task was killed by the SLURM wall limit. C1.12 holds on D2.

**Memory, measured not assumed.** Median **393 MB**, max **541 MB** across all 40
`.batch` steps. The first draft of the launcher requested 48 G / 16 G, inherited
from the T04 probe; at 48 G only three tasks fit on a 182 GB Intel node, so a 10-wide
throttle would have spread over four nodes for nothing. Resized to 8 G / 4 G before
the array — still 15–20× the peak. C2 should size D2 from this, not from T04.

### The saturation risk is now measured, not hypothesised

**16 of 20 Bingo cells reach R² ≥ 0.999 at a 25-minute budget**, against a production
budget of 12 hours. The local smoke showed the `baseline` arm saturating too
(R² = 1.0000 on `Strogatz-predprey1`). If both arms saturate then `δᵢ ≈ 0`, and
problems contributing zero differences **weaken** CPDT by adding ties rather than
strengthening it.

This is an inference, and it is flagged as one: the probe ran only the `isalsr` arm,
so no `δᵢ` was computed here. But it is the honest reading, it was pre-committed to
in §5.4 of the selection rule, and it should be stated in the response letter rather
than discovered during analysis. The extension's defensible claim is **coverage** —
SRBench's ground-truth track — not a stronger p-value.

UDFS's low R² (median 0.41 on the Feynman remainder) is **not** a finding: it is a
25-minute budget against a 12-hour one. Nothing follows from it.

---

## B.4 🔴 What the probe found that blocks C2

Two Stage C criteria are **not merely failing — they are uncheckable**, because the
quantities they assert on never reach the RunLog:

| Criterion | Needs | Present in RunLog | Checkable? |
|---|---|---|---|
| **C1.9** | the five T06 fallback rates, on every `isalsr` task | **0 / 40** | **NO** |
| **C1.14** | `engine == native`, on every task | **0 / 40** | **NO** |

`metadata.hardware` carries `conda_env, cpu, cpu_count, git_hash, os, os_version,
platform, python_version, ram_gb, timestamp` — and nothing else. A full walk of all
69 keys in a probe `run_log.json` finds no fallback, violation, timeout, conversion
or reachability field, and no ledger file anywhere in the probe output.

**C1.14 is the known `A7-BUG`**, now reproduced independently on a live D2 probe.

**C1.9 is new and it is worse.** `EXECUTION-PLAN.md` §3 draws the line explicitly:
*"anything measured during a run must be in the code before launch; anything computed
after can land later. Getting this wrong means re-running 8,400 jobs to recover a
counter."* The five reachability rates only exist while a search is running, so
unlike `engine` they **cannot be recovered post hoc**. They are the evidence base for
R1.2's answer.

SP-6 passing is not a contradiction: it proves the ledger class imports, not that
anything counts. That gap between the check and the criterion is precisely why
`EXECUTION-PLAN.md` warns that *"a zero-everywhere ledger means the counters are
dead, not that the rates are zero"* — here we cannot even observe zero.

**Owner: T02** (C1.9 is its check to execute, T06 supplies the threshold). This is
not T05's to fix, and T05 does not claim it is fixed.

---

## B.5 Reproducing this

```bash
ssh picasso 'cd <repo> && bash slurm/t05_probe/launcher.sh --dry-run'   # then
                                                          # --test-only, --one
python slurm/t05_probe/make_tasks.py --out slurm/t05_probe/tasks.txt   # regenerate
python slurm/t04_probe/make_provenance.py                              # BEFORE rsync
python slurm/t05_probe/summarise.py --root ~/execs/isalsr/t05_probe
```

`make_provenance.py` refuses to stamp a dirty tree, which is what makes SP-1 mean
something. Deploy from a clean checkout, not from the working tree.
