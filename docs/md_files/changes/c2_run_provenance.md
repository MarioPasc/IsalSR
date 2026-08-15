# C2 run provenance: closing A7-BUG, C1.9-BUG, P3 and P4

**Date:** 2026-08-03
**Status:** landed, 6,247 unit tests pass (was 6,244 before; +60 new, −57 net
reflects one rewritten contract test)
**Plan items closed:** `EXECUTION-PLAN.md` A7, A7-BUG, C1.9-BUG, P3, P4
**Files:** `experiments/models/{provenance,status_ledger,hardware_info,orchestrator,schemas,fallback_ledger}.py`,
`tests/unit/test_run_provenance.py`, `tests/unit/test_orchestrator_flags.py`

---

## 1. Why these four and not others

`EXECUTION-PLAN.md` §3 draws the line that decides what blocks a launch:

> anything measured **during** a run must be in the code before launch; anything
> computed **after** can land later. Getting this wrong means re-running 8,400
> jobs to recover a counter.

Four items fell on the blocking side and were not implemented. Each measures a
population that exists only while a search is running, so no post-hoc pass over
C2's output could recover it — the campaign would have to be re-run.

| Item | State before | Consequence had C2 launched |
|---|---|---|
| **A7-BUG** | `collect_hardware_info()` returned 10 fields, none of them `engine` | C1.14 ("every task records `engine == native`", 420/420) and D2's pure-Python spot check were **not checkable from a run log**. A silent fallback to the Python canonicaliser produces correct numbers at ~24× the cost and looks like nothing at all |
| **C1.9-BUG** | Only `n_conversion_failures` reached `SearchSpaceResults`; four of the five T06 paths did not | The evidence base for **R1.2** would be absent from all 8,400 runs. A walk of all 69 keys of a live probe `run_log.json` found no reachability field |
| **P3** | No data fingerprint anywhere in the repo | The paired design's premise — three arms, same data — was **unverifiable**. Check C4 had nothing to compare |
| **P4** | No terminal-status record | C1's "35 missing cells, cause unknown" recurs at 8,400-run scale. 36 of C1's 45 missing cells were OOM `SIGKILL`s, which **no Python handler observes** |

Two further Stage-A items (**A6** the MANIFEST validator, **A8** the three-arm
analyzer) are *post-hoc* computations over run logs. They remain open and are
listed in the certification ticket; they do not gate the smoke, because the
smoke produces the run logs those items consume.

---

## 2. What landed

### 2.1 A7 — engine, node, code and allocation identity

`collect_hardware_info()` keeps its ten pre-C2 fields verbatim (C1 artefacts
still parse) and adds fifteen.

Engine identity is read through `backends.engine()`, i.e. from **actual
dispatch**, not from the compiled-in default. This is deliberate: until
2026-07-31 `fast_canonical_string` read `backends.DEFAULT_BACKEND` and bypassed
the `ISALSR_ENGINE` override, so a probe reported `native` whichever engine ran
— it *passed while proving nothing* (check B2). A regression test forces
`ISALSR_ENGINE=python` and asserts the capture follows it.

The vocabulary is normalised: the backend registry says `cpp`, the plan says
`native`. `collect_hardware_info()["engine"]` returns `native`, so C1.14 is a
single string equality.

`build_hash`, `isa_level` and `avx512f` ride along because "native" is not one
thing. An extension built with `-march=native` on the AVX-512 login node dies
with `SIGILL` on the `sr`/`bl` nodes that are most of the pool, and it fails as
a *fraction* of tasks — indistinguishable from flaky hardware unless the ISA
level is recorded per run (B6b).

Measured locally: `engine=native`, `build_hash=298fc1188bf1b051`,
`isa_level=x86-64-v3`, `avx512f=0` — an AVX2 build, portable across all four
node families.

### 2.2 C1.9 — the five fallback paths

`SearchSpaceResults` gains ten fields, exported by
`FallbackLedger.to_search_space_fields()`: the five paths
(`n_violations_pre`, `n_violations_post`, `n_canon_timeouts`,
`n_conversion_failures`, `n_canon_raised`), the `n_atlas_hits` partition, and —
the part that matters — `ledger_enabled`, `ledger_sample_rate`,
`n_ledger_seen`, `n_ledger_sampled`.

The denominators are not decoration. SP-6 warns that *a zero-everywhere ledger
means the counters are dead, not that the rates are zero*. Counts alone cannot
distinguish those two states; `ledger_enabled` plus a sampled denominator can.
Per-k histograms stay out of the run log and go to a sibling
`fallback_ledger.json`, because k-stratification is a post-hoc analysis over the
campaign, not a per-run field.

### 2.3 P3 — the data fingerprint

`provenance.data_fingerprint()` returns SHA-256 over the four arrays, committing
to name, shape and raw IEEE-754 bytes in fixed order. Arrays are cast to
`float64` and made contiguous first, so the certified object is the *sample*,
not the container carrying it.

Byte equality is stricter than numerical equality (`-0.0` ≠ `0.0`). That is
intended: every arm calls the same generator with the same seed, so bit-for-bit
agreement is the expectation, and anything less is exactly the confound C4
exists to catch.

`provenance.config_sha256()` digests the config file's bytes. Both are recorded
on `RunMetadata`, which now also documents `representation ∈ {baseline, hash,
isalsr}`.

### 2.4 P4 — the crash-safe status ledger

`RunStatus` is written **before** the search starts, with
`terminal_status="started"`, and rewritten on the way out. This is the whole
mechanism: an OOM arrives as `SIGKILL`, which no handler observes, so an
exception-based ledger records nothing for precisely the failure mode that
caused C1's shortfall. A row still reading `started` after an array has drained
*is* the report — that cell was killed from outside.

Writes are atomic (`os.replace`), so a process killed mid-write leaves the
previous complete record rather than a truncated one that would be
indistinguishable from corruption.

One `status.json` per seed directory, not one appended CSV: 1,400 concurrent
array tasks appending to a shared file interleave partial lines.
`collect_status_ledger()` assembles `status_ledger.csv` afterwards, when nothing
is writing, and sorts deterministically so two audits of one root are
byte-identical. `reconcile()` implements C1.15/E6 — it **names** missing,
killed and failed cells rather than reporting a count, which is why explaining
C1's shortfall needed a forensic pass.

The orchestrator now catches a cell's exception, records the cause, and
continues; `run_experiment` returns 1 if any cell failed, so SLURM still sees
the failure. §5.5 admits no third state.

---

## 3. Two defects found by doing this

### 3.1 A 1-seed run exited non-zero — Stage C would have failed C1.1 on all 420 tasks

`compute_paired_stats` raises `ValueError` below 3 matched seeds. Stage C runs
**one** seed by design. Every one of the 420 tasks would therefore have raised
*after a complete and correct run*, written no `status_ledger.csv`, and exited
1 — failing C1.1 ("every task exits 0", 420/420) universally, with every
artefact it certifies present on disk.

Fixed by guarding the contrast on the matched-seed count and logging the skip.
Producing the artefacts and computing the statistic are separate obligations;
only the second needs seeds.

### 3.2 🔴 The T06 ledger is OFF by default and no worker script enables it

`FallbackLedger.__init__` reads `ISALSR_LEDGER_ENABLED`, **default `"0"`**. A
repo-wide grep finds it set only in `measure_ledger_overhead.py` and in unit
tests — **in no SLURM worker, no launcher and no config**.

Had C2 launched as-is, all 8,400 runs would have recorded five reachability
rates of zero. That reads as "no fallbacks occurred" and actually means "nothing
was counted", and the difference is unrecoverable: the population exists only
while the search runs. This is SP-6's trap in its exact form.

It was invisible before this change, because `ledger_enabled` was not in the run
log at all — the counters would simply have been absent or zero with no way to
tell dead from measured.

Two things landed:

- `--ledger` / `--ledger-sample-rate` CLI flags, so the choice is an **auditable
  launch parameter** recorded in every run log rather than an ambient
  environment variable someone forgot to export.
- A loud warning whenever a deduplicating arm runs with the counters off.

**This does not decide the question.** Check B9 reserves the right to remove the
counters from C2 if their overhead is material under the C++ engine and the
decomposed alphabet — both changed underneath T06's original measurement. The
decision is Mario's, per T06 AC-10. What has changed is that the decision is now
*visible in the artefacts* either way, instead of defaulting silently to the
worst outcome.

Verified live (UDFS, Nguyen-1, 25 s): `n_seen=476, n_sampled=476,
violated_pre=476, violated_post=0` — finite and non-zero, so a live counter is
distinguishable from a dead one, and the pre/post pair reproduces the expected
CONST-normalisation repair.

### 3.3 A unit bug in this change, caught before it shipped

`peak_rss_gb()` divided `ru_maxrss` (kilobytes on Linux) by 1024, yielding MB
under a GB label — a fresh Python process reported "163 GB". Since this number
sizes production `--mem` under C1.11/D1.2, it would have inflated every memory
request by 1,024×. A regression test now asserts the reading is plausible.

---

## 4. Verification

| Check | Result |
|---|---|
| `pytest tests/unit/` | **6,247 passed, 5 skipped, 0 failed** |
| `tests/unit/test_run_provenance.py` | 60 passed |
| `ruff check` on all changed files | clean |
| `ruff format --check` | clean |
| `mypy --strict src/isalsr/` | clean, 55 files |
| Bingo 3-arm local run, 1 seed | exit **0**; 3 run logs, 3 status rows, 2 fallback ledgers |
| UDFS 3-arm local run, 1 seed | exit **0**; ledger live at 476 candidates |
| Cross-arm fingerprint identity (C4 rehearsal) | identical on 3/3 arms, both hosts |
| `ρ_hash ≤ ρ_isalsr` (C1.7 rehearsal, UDFS) | 1.0000 ≤ 1.6081 ✓ |
| Baseline un-instrumented (C1.8 rehearsal) | all ten ledger fields `None`, ρ = 1.0 ✓ |
| D1 ∪ D2 registry resolution | 70/70 problems, both hosts |

`None` versus `0` is load-bearing throughout: `None` means "this arm was never
asked", `0` means "asked, and none occurred". Collapsing them would reintroduce
exactly the ambiguity these fields exist to remove.

---

## 5. What is still open before submission

Neither blocks the Stage-C smoke; both block the full campaign. Tracked in the
certification ticket `T17-c2-submission-certification.md`.

| Item | Why it does not block the smoke |
|---|---|
| **A6** MANIFEST schema + validator (`experiments/models/manifest.py`, absent) | Written at submission time, consumes run logs |
| **A8** three-arm analyzer (`analyze.py` hardcodes `["baseline", "isalsr"]` in several places) | Stage E, consumes the smoke's output |
