# T04 — C2 launch readiness of the hash implementation

**Question**: is the hash implementation ready for campaign C2 as specified in
`EXECUTION-PLAN.md`, and is everything remaining only further testing or
response-letter writing?

**Answer (2026-08-02, updated after the fix)**: **yes — the hash implementation is
ready for C2.** Everything remaining on T04 is running the campaign, analysing its
output, and writing the reviewer answer. No further implementation is implied.

The one code-readiness gap found while answering this question —
`shadow_distinct_host_native` shipping untested — was closed the same day (§2).

---

## 1. Ready — the `hash` arm itself

What C2 runs as its third arm is production-verified end to end:

| Component | Evidence |
|---|---|
| `HASH_ARM_KEY_MODE = "host_native"`, both hosts | probe commit `a4206b8`, 28/28 cells |
| `host_native.py` serialiser + soundness | AC-1: 0 violations / 14,841 DAGs × 3 orders × both backends |
| End-to-end on Picasso, both hosts | 28/28 validating `run_log.json` |
| SP-1…SP-6 | **336/336** across 56 evidence files |
| SP-3 engine + negative control | `cpp_actually_invoked` true ×28, **false** ×28 under `ISALSR_ENGINE=python` |
| Memory (AC-10) | shadow ON/OFF Δ −0.24 % / −0.14 %; no OOM |
| Orchestrator plumbing | `hash` factory branches, `--max-time`, `--no-shadow-hash`, three paired contrasts |

Nothing here is outstanding. C2 can run the `hash` arm today.

## 1b. Correction (2026-08-02, later) — the "untested" finding was half wrong

§2 below claimed `shadow_distinct_host_native` shipped with **no unit test**.
**That was wrong.** `tests/unit/test_shadow_host_native.py` was committed in
`a24d73c`, the same commit that added the feature. The claim came from grepping
`tests/unit/test_hash_arm.py` alone and generalising from one file.

What was correct: it had never run **on Picasso** at the time of the T04 probe
(`a4206b8` predates `a24d73c`). The T05 agent has since closed that half —
verified independently here against 40 T05 probe run logs: all four shadow
fields **present, finite and > 0 on 40/40**, both hosts, 20 problems.

So the launch gate was never actually open. The §2 tests are largely duplicate
coverage and have been trimmed to the one case the dedicated file does not
cover (that a failure reaches the *persisted* field, not just the attribute).

---

## 2. RESOLVED 2026-08-02 — `shadow_distinct_host_native` now tested

**Six tests added** to `tests/unit/test_hash_arm.py`, parametrised over both
hosts. Suite: **28 passed** (22 pre-existing + 6 new); full unit suite **6,146
passed, 5 skipped, 0 failed**; ruff and format clean.

| Test | Pins |
|---|---|
| `test_host_native_shadow_records_distinct_hosts` | Field **present, non-null**, estimate within 5 % of the true distinct count over 5 hosts each fed twice, and `n_shadow_failures == 0` |
| `test_host_native_shadow_absent_when_no_host_offered` | DAG-only call sites leave the counter **undefined, not 0** — a spurious 0 is indistinguishable from "this host emitted no distinct representations" |
| `test_host_native_shadow_failure_is_counted_not_silent` | A broken extractor raises `n_shadow_failures`, pinning the one signal Stage C can rely on |

The stubs implement each host's documented record contract (`command_array` +
`get_utilized_commands()` for Bingo; `node_dict` for UDFS) and are driven through
the **real** `{bingo,udfs}_host_native_records` extractors, with op codes drawn
from the modules' own arity sets so the stubs cannot drift from the extractors'
branching.

**Coverage now stands as:** the extractors themselves are proven against *real*
hosts by the probe — the `hash` arm keys on them and ran 28/28 cells on Picasso.
What was untested was the **shadow plumbing** wrapping them, and that is what
these tests close. The residual gap is only that the two halves have never run
*together* on Picasso.

**Remaining action, now low-risk:** enable the counter in the pending
`slurm/t05_probe/` submission and assert `n_shadow_failures == 0` plus a non-null
`shadow_distinct_host_native` in the C2 Stage-C criteria. Formerly a launch
blocker; now a confirmation step.

Two notes recorded while doing this:

- An earlier open item claimed `test_shadow_counters_track_all_three_orders`
  "should expect four fields". **That was wrong** — it calls `record_shadow(dag)`
  with no host, so three fields is the correct expectation under the
  `_host_native_offered` guard. Closed as invalid.
- The new tests initially passed while `Any` was unimported: `from __future__
  import annotations` defers annotation evaluation, so `F821` surfaced only under
  `ruff`, not `pytest`. Lint is load-bearing here, not cosmetic.

---

## 2b. Original finding (superseded by §2, retained for the record)

The fourth shadow sketch, which measures the **host's own** representation on the
`isalsr` arm's candidate stream, has **never executed anywhere**:

- **No unit test.** `grep -c host_native tests/unit/test_hash_arm.py` → **0**.
- **Not on Picasso.** It landed in `a24d73c`, *after* the probe was submitted at
  `a4206b8`; at that commit `record_shadow` took no `host` argument.

This is not a nice-to-have. §5.2 of `naive_hash_baseline.md` showed the three
adapter-order sketches **inflate the naive baseline from 0 % to 94.6 % on UDFS**,
because the adapters renumber nodes before the sketch sees them. The host-native
sketch is the only same-stream measurement free of that bias — i.e. the only one
whose numbers could be quoted. If it silently misbehaves across ≈2,800 runs, the
same-stream half of the R1.4 answer is unusable and C2 would have to be re-run.

**Residual risk is bounded but real.** Failures increment `n_shadow_failures`, so
a total breakage is detectable — but `_host_native_offered` is set *before* the
`try`, so the field would still be emitted, carrying an implausibly low count
beside a high failure count. It fails quietly, not loudly.

**Fix, ~1 hour:** (a) unit test asserting `shadow_distinct_host_native` is present,
non-null and within HLL tolerance on a known stream, both hosts; (b) exercise it in
the pending `slurm/t05_probe/` submission, or a 2-task probe; (c) assert
`n_shadow_failures == 0` in the C2 Stage-C criteria.

## 3. Everything else is C2 execution or write-up

| AC | Nature of remaining work |
|---|---|
| AC-2 (dispersion, k-strata) | analysis of C2 output |
| AC-3 (≈2,800 runs) | running C2 |
| AC-4 (`S` for three arms) | analysis of C2 output |
| AC-5 (three-arm stats + CD diagram) | analysis of C2 output |
| AC-7 (sound-but-incomplete in the paper) | `/review-answer` |

No further implementation is implied by any of these.

## 4. Verdict

> **The hash implementation is ready for C2.** The `hash` arm is probe-verified
> end to end on both hosts; the `isalsr` arm's host-native shadow counter is now
> unit-tested on both hosts with a failure tripwire. Everything left on T04 —
> AC-2, AC-3, AC-4, AC-5, AC-7 — is campaign execution, analysis of its output,
> or the response letter.
>
> **The confirmation step is done** (T05 agent, independently verified here): the
> T05 probe carried the counters — `worker.sh` never passes `--no-shadow-hash`,
> so shadow hashing was live by default. All four fields present, finite and > 0
> on **40/40** run logs, both hosts, 20 problems; zero serialisation failures on
> the 38 tasks that ran a search (the other 2 were resume-skipped cells whose run
> logs came from the single-task gate).

---

## 5. `n_shadow_failures` is now persisted (2026-08-02)

The T05 agent flagged that `assert n_shadow_failures == 0` was **not checkable
from a RunLog** — the counter was tracked and then only logged at INFO to stderr.
Verified: absent from `SearchSpaceResults` and from both translators; the T05
zero came from grepping 40 `.err` files. Across 420 Stage-C tasks that is 420
greps, dependent on log retention and on the line staying at INFO.

**Fixed.** `n_shadow_failures: int | None = None` added to `SearchSpaceResults`,
and both runners' `shadow_counts()` now emit it. It reaches the RunLog through
the orchestrator's existing splat (`orchestrator.py:516`,
`dataclasses.replace(run_log.search_space, **shadow)`) — so **no translator
changes were needed**, which matters because `bingo/translator.py` and
`udfs/translator.py` are both currently being edited by the T08 session.
`schemas.py` is contended too, but T08's edits are in `RegressionResults` and
this one is in `SearchSpaceResults`; T08's `n_nonfinite_test_predictions` was
verified intact after the edit.

Semantics: emitted only when the sketches actually ran. A shadow-off run leaves
the field `None` rather than claiming zero failures for work it never did.

Tests: the field reaching `shadow_counts()` **and** surviving
`dataclasses.replace` into the frozen dataclass; empty dict when shadow is off;
a broken extractor surfacing in the persisted field. Two pre-existing exact-key
assertions needed updating — one in `test_hash_arm.py`, one in
`test_shadow_host_native.py`.

**Stage C can now assert `n_shadow_failures == 0` as a field read, not a grep.**
