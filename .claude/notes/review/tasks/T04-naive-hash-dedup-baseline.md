# T04 — Naive fixed-order-serialisation hash dedup baseline

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.4** (and materially improves B4 for all three reviewers) |
| Type | New experiment — full comparator |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (engine), T02 (campaign infrastructure) |
| Blocks | **Campaign C2 — this ticket gates the launch** · T09 (tables), T13 (page budget) |
| Status | **IN PROGRESS — launch-gate code complete, probe not yet submitted.** **AC-1 met** (0 soundness violations / 14,841 DAGs × 3 orders × both backends). Arm keys on the **host-native** representation on both hosts; 4,987 tests green, mypy/ruff clean, smokes exit 0. Orchestrator carries `--max-time` / `--no-shadow-hash`. Probe scripts written (`slurm/t04_probe/`), SP harness verified locally with a **real** SP-3 negative control. **Headline, provisional: UDFS 92.4 % of duplicates require 1-WL (ρ 1.0222→1.4029); Bingo 6.5 % (ρ 1.7013→1.7890).** Both §2 predictions confirmed, Bingo concession included. **Blocking C2:** add `shadow_distinct_host_native` (AC-2 at campaign scale). **Nothing submitted to Picasso.** AC-2b/AC-3/AC-5/AC-10 open |
| Target | **implementation 2026-08-17 — this is a hard launch gate, every day late costs ≈7,200 core-hours of headroom** · results 2026-09-03 |

---

## ⛔ Amendment 2026-07-31 — read before doing anything on this ticket

**There is no "Wave 3". The `hash` arm is not a follow-on campaign — it launches
simultaneously with the other two arms, and this ticket gates that launch.**

`EXECUTION-PLAN.md` was rewritten on 2026-07-31 and is authoritative. Campaign C2 is
a single gated launch:

```
{ baseline , hash , isalsr } × { UDFS , Bingo } × ( D1 ∪ D2 ) × 20 seeds
```

| Was | Is |
|---|---|
| Wave 3, `hash` arm, S70 × 30 seeds = 4,200 runs, launches after the C++ wave | **The `hash` arm of C2**: 2 methods × ≈70 problems × **20 seeds = 2,800 runs**, 33,600 core-hours, launched *with* `baseline` and `isalsr` |
| Mode 1 replays "the stored DAG streams from the T02 campaign" | **C1 never persisted a replayable stream, and its alphabet is wrong anyway.** Mode 1 now replays the **pre-flight Stage D streams** (`EXECUTION-PLAN.md` §4.4, D2/D3) |
| T04 is downstream of the headline campaign | **T04 gates it.** Nothing at all submits until this ticket's code lands |

**Why the schedule pressure is real:** `EXECUTION-PLAN.md` §8.2. A gated launch at
2026-08-20 needs ≈300 concurrent cores to finish by 2026-09-03. Every day this
ticket slips is ≈7,200 core-hours of headroom the campaign does not get back.

### 🚫 This ticket does not submit the campaign

**`EXECUTION-PLAN.md` §4.0 SP-0 is binding.** No agent working this ticket submits
C2, the `hash` arm, or anything resembling either. Everything submitted here is a
**probe**: `max_time ≤ 1,800 s` (30 min), ≤ 60 tasks, **seed 0 only**, output to
`~/execs/isalsr/t04_*/`, never the campaign root. Mode 2 is submitted once, by Mario,
as part of C2, after Stage F sign-off.

**Before trusting any Picasso result from this ticket, establish SP-1…SP-6**
(`EXECUTION-PLAN.md` §4.0) and report them as a six-row table in the work log:

| SP | What you must show |
|---|---|
| SP-1 | provenance — the commit on the node is the commit you synced, not `-dirty` |
| SP-2 | **installation freshness** — the site-packages `.so` mtime post-dates your last C++ edit. `pip install -e . --force-reinstall --no-deps`; **never** `--no-build-isolation` |
| SP-3 | engine is `native`, **with the forced-Python negative control** |
| SP-4 | alphabet: 0 `Sub`, 0 `Div`, 0 `-`, 0 `/` on the probe's own candidate stream |
| SP-5 | **UDFS and Bingo both** — a hash arm verified on one host is unverified |
| SP-6 | T06's five fallback counters live and finite |

**SP-7 for this ticket** — the falsifiable statement a T04 probe must establish, on a
≤30-minute run, on **both hosts**, with all three arms present:

1. The `TOPOLOGICAL` hash (the arm, §4.1) executes **inside the live search**, not
   only in unit tests — and the three-rung shadow counters execute inside the live
   `isalsr` search at **constant memory, no OOM** (AC-10).
2. The candidate stream persists at the chosen sampling rate (check P1) **and
   replays** — round-trip, not just written.
3. On **identical replayed input**, `ρ_hash ≤ ρ_isalsr` **without exception**. This
   is guaranteed by construction (a fixed-order hash is sound but incomplete); a
   single violation is a bug in one of the two arms, not a surprise.
4. **Hash soundness**: equal hash ⇒ equal canonical string, over the whole probe
   stream. An unsound merge kills the arm.
5. The serialisations are sized against the **post-T16 `k` distribution** (Bingo mean
   5.47 → 6.72, p95 11 → 15), not the old one.

---

## T16 impact — the hash must serialise **decomposed** DAGs (added 2026-07-30)

T16 moved the adapters to the paper's alphabet: `Sub` and `Div` are rewritten to
`Add(a, Neg(b))` and `Mul(a, Inv(b))` at the host→`LabeledDAG` boundary, leaving
`Pow` as the only non-commutative operation.

**Requirement: this arm must hash the same object IsalSR canonicalises.** If the
naive serialisation is built over undecomposed DAGs while the IsalSR arm
canonicalises decomposed ones, the comparison answering R1.4 is run on two different
representations and is meaningless.

**This is satisfied by construction, and there is nothing to do — provided you build
on the adapters.** The decomposition lives *inside* `agraph_to_labeled_dag` and
`compgraph_to_labeled_dag`, so every consumer inherits it. Do not construct
`LabeledDAG`s by any other route for this arm. Verified 2026-07-30: no T04 code
exists yet, so this is recorded as a constraint on the implementation rather than a
check already performed.

Note also that `k` grows ~22 % under decomposition, so any fixed-width or
capacity-bounded serialisation must be sized against the **new** `k` distribution
(Bingo mean 5.47 → 6.72, p95 11 → 15).

---

## 1. Why this is its own ticket

R1.4 is the heaviest single request in the round and the one most likely to decide
round 2. It is not grouped with anything: it is a self-contained new comparator,
and unlike R1.2 (a measurement on existing machinery) or R3.1 (more problems on
existing machinery), it introduces a *method* into the evaluation that the paper
currently does not have.

**Decision taken 2026-07-27**: this baseline is expected to appear in the final
article as a **full comparison**, not as a footnote — reviewers will look for it.
It must simultaneously be made clear that it does not have IsalSR's properties.

**Verbatim comment:**

> 4) There is no comparison against naive hash-based deduplication on a fixed-order
> DAG serialization. This is the obvious baseline and its absence makes it hard to
> assess how much of the benefit requires 1-WL machinery versus a much simpler
> approach.

---

## 2. The scientific shape of the answer

The comparison is **not** a horse race that IsalSR wins. It is a decomposition, and
saying so plainly is the strongest available response.

A fixed-order serialisation hash is **sound but incomplete**: it never merges
non-isomorphic DAGs, but it fails to merge isomorphic DAGs that differ in node
numbering — which is exactly the redundancy IsalSR targets. So the observed
reduction factor decomposes:

```
ρ_total  =  ρ_exact   ×   ρ_comm    ×   ρ_iso
            ^^^^^^^^      ^^^^^^^^      ^^^^^
            caught by     caught by a   caught ONLY by an
            a fixed-      naive local   isomorphism-complete
            order hash    ADD/MUL sort  invariant
```

The paper's job is to report all three factors, on real search trajectories, per
method. That answers R1.4 exactly as asked ("how much of the benefit requires 1-WL
machinery versus a much simpler approach") and it is a *better* result than a win,
because it quantifies the contribution instead of asserting it.

The middle factor was added 2026-07-31 (§4.2). It exists because the two-factor
split leaves the obvious follow-up question open: a reviewer who reads
`supplementary.tex:689–693` — "duplicates arise only from the commutative symmetries
of ADD and MUL" — will immediately ask what a scheme that simply sorts commutative
operands would achieve. `ρ_comm` is that number. It is obtained from a third
*serialisation*, offline, at zero campaign cost; it is **not** a third comparator and
nothing extra runs on Picasso because of it.

**What the existing evidence predicts.** Two pieces already in the submission
constrain the answer and should be cited in §8:
- `supplementary.tex:689–693`: UDFS duplicates "arise only from the commutative
  symmetries of ADD and MUL", i.e. from operand permutation — which a fixed-order
  hash does **not** catch. Predicts ρ_exact ≈ 1 for UDFS.
- `supplementary.tex:782–786`: on the 5,400 synthetic DAGs every one has trivial
  automorphism group and ρ = k! exactly — a regime where a fixed-order hash catches
  nothing beyond byte-identical repeats.

Bingo is the interesting case: stochastic GP re-generates byte-identical individuals
(the B12 note in `CLAUDE.md` records that VarAnd produces unmodified `parent.copy()`
offspring ~36 % of the time), so ρ_exact should be materially above 1 there. That
is the honest concession, and it should be stated before a reviewer extracts it.

**Reviewer 1 has already pre-framed the cost axis in their own B2 statement**:
1-WL canonicalisation is *"a meaningful middle ground between the O(k!) exhaustive
search and hash-based approaches that offer no correctness guarantee"*. They expect
the hash baseline to lose on completeness. The open questions they actually want
answered are **by how much** and **at what cost**.

---

## 3. Mandatory reading

- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — **read first.** §0.4 (campaign
  shape), §4.0 (SP-0…SP-7 Picasso discipline), §4.4 D2/D3 (where Mode 1's stream
  comes from), §6.1 (three-arm statistics), §8.3 (where this arm sits in the trade
  order)
- `.claude/notes/review/source/reviewer-1.md` — §R1.4 and the full B2 statement
- `.claude/notes/review/source/verified-discrepancies.md` — context on ρ (E8)
- `.claude/notes/review/source/codebase-pointers.md` — `sympy_adapter` is flagged as
  the likely place to build the fixed-order serialisation;
  `model_validation/diversity/dedup_smoke/` may already hold a related smoke test
- `CLAUDE.md` (repo root) — B12 VarAnd clone detection; the UDFS `processes: 1`
  constraint; dedup uses `set[int]` not `set[str]`
- `src/isalsr/core/README.md`
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — protocol to match
- `docs/md_files/design/experimental_design/isalsr_experimental_design.md`

---

## 4. Steel-man the baseline — **rescoped 2026-07-31, read this instead**

> **Superseded.** The original §4 required "at least three" fixed orders *as
> comparators*, the third being a DFS pre-order from `x_1` with children sorted by
> `(label, subtree size)`, and instructed us to report the best performer as *the*
> baseline. That is over-built and it misreads the comment. R1.4 asks for
> "**a** fixed-order DAG serialization", singular, "**naive**", "the obvious
> baseline", "a much simpler approach". A DFS with children sorted by a structural
> key — and *a fortiori* any recursive hash-consing of commutative operands — is not
> a fixed-order serialisation at all: it is a canonicalisation with a weaker
> invariant. Shipping one under the label "naive baseline" would answer a question
> the reviewer did not ask, and would manufacture a competitor to our own method out
> of nothing.
>
> The purpose clause — *"how much of the benefit requires 1-WL machinery versus a
> much simpler approach"* — is a request for a **number**, and it is answered by the
> offline decomposition in §5.1, not by adding a second method to the evaluation.

### 4.1 What ships as the arm — one naive hash

**`TOPOLOGICAL`**: topological order (edge `u→v` ⇒ `u` precedes `v`), ties broken by
`(label, in-degree, out-degree)`, residual ties broken by original node index.
Serialise node labels, the variable index of every `VAR`, and operand order via
`ordered_inputs()` (Critical Invariant 8). Hash with CPython's builtin `hash()` —
**the same 64-bit function the IsalSR dedup set uses**
(`bingo/isalsr_runner.py:375`, `udfs/isalsr_runner.py:139`), so the memory and
collision analysis is shared and the comparison is not confounded by hashing.

`TOPOLOGICAL` is the *generous* choice within the naive class: it strictly dominates
insertion order, because it re-sorts nodes the host numbered arbitrarily and can
therefore merge pairs that insertion order separates. Choosing the weaker rung as
the headline would be the "chosen to lose" failure §8.4 names.

**CONST values are not serialised.** The canonical string is over labels only, so
IsalSR merges DAGs differing only in constant values. A serialisation that encoded
them would be finer than IsalSR for a reason unrelated to isomorphism and would
confound ρ.

### 4.2 What ships offline — a three-rung ladder, not three arms

All three are node-index-dependent fixed orders. None is a canonical form. They cost
**no campaign compute**: they run in Mode 1 over a persisted stream.

| Rung | Order | Isolates |
|---|---|---|
| 1 | `INSERTION` — host's own node numbering | byte-identical regeneration |
| 2 | `TOPOLOGICAL` — **the arm** | + renumbering the topological sort absorbs |
| 3 | `TOPOLOGICAL_COMMUTATIVE` — as rung 2, but each `ADD`/`MUL` node's operand list sorted locally by emitted position. **Local, one node, non-recursive** | + plain ADD/MUL operand swaps |

```
ρ_total  =  ρ_exact   ×   ρ_comm    ×   ρ_iso
            ^^^^^^^^      ^^^^^^^^      ^^^^^
            rung 1        rung 2→3      what is left:
                                        genuinely needs the
                                        labeled-DAG invariant
```

Two things this buys that a single order cannot:

- **Rung 1 vs rung 2** measures how much the answer depends on *which* fixed order
  we picked. If they agree, the "your order was chosen to lose" objection (§8.4) is
  answered with evidence rather than assertion.
- **Rung 2 vs rung 3** measures how much of ρ is plain ADD/MUL commutativity.
  `supplementary.tex:689–693` already asserts UDFS duplicates "arise only from the
  commutative symmetries of ADD and MUL"; a reviewer can and will ask what happens
  when a naive scheme absorbs exactly that. Rung 3 answers it, and if ρ_iso turns
  out small on UDFS, **that is reported as-is** (AC-8).

**Hard non-goals.** No fourth order. No recursion, no hash-consing, no subtree-hash
sorting, no DFS with a structural sort key. Each of those crosses from "fixed order"
into "canonical form" and puts a method in the evaluation that nobody asked for.

---

## 5. Work specification

Two measurement modes, both required (decision 2026-07-27, "1+2").

### 5.1 Mode 1 — offline replay (mechanism decomposition)

Replay a stored candidate stream through the three fixed-order serialisations of
§4.2 **and** through IsalSR canonicalisation, on **identical input sequences**.
Produces `ρ_exact`, `ρ_comm`, `ρ_iso`, `ρ_total` per method stratified by `k`, plus
per-DAG cost for each scheme. This is the controlled comparison: identical inputs,
zero search confound.

Note the division of labour set in §5.5: the ρ **factors** come from the online
constant-memory counters over the *full* stream in every production `isalsr` run;
Mode 1's replay supplies the **soundness proofs**, the k-stratified cross-check
against those online counters, and the per-scheme cost. Do not compute ρ from the
sampled stream.

**Where the stream comes from — changed 2026-07-31.** Not the submitted campaign: C1
persisted only aggregate counts, so replaying it was never possible, and its alphabet
is wrong regardless. Mode 1 replays the **pre-flight Stage D streams**
(`EXECUTION-PLAN.md` §4.4): `c2_trace/candidates.jsonl` from the detailed
single-problem trace, plus the 12 full-length certification runs' streams. Real
searches, full 12 h budget, correct alphabet, native engine — and available **before**
the 100,800 core-hours are committed, which is the entire point.

This requires check **P1** (stream persistence, format and sampling rate) to land
before the certification runs. It is this ticket's dependency to chase.

Mode 1 also carries the two soundness checks nothing else can make: **equal hash ⇒
equal canonical string** (hash soundness, AC-1) and **equal canonical string ⇒
isomorphic** (spot-checked on the largest equivalence classes).

### 5.2 Mode 2 — live third arm (end-to-end comparison)

The `hash` variant runs as a **full arm of Campaign C2**, alongside `baseline` and
`isalsr`: 2 methods × ≈70 problems × **20 seeds = 2,800 runs**, 33,600 core-hours,
same protocol, full 12 h budget, same commit, same build, same node pool, same
campaign root.

This is what makes it "a full comparison" in the article: R², NRMSE, solution
recovery, wall-clock and `S` for the hash variant on exactly the same footing as the
other two arms, across every problem they cover. That footing is only achievable
because all three arms launch together — it was not achievable under the old
sequenced-wave design.

**Mode 1 is read before C2 launches, not after.** If the replay shows `ρ_exact ≈
1.00` for both methods, the live arm is expected to be a null result — *which is
itself the answer to R1.4*, obtained for ≈0 core-hours. The arm still runs
(`EXECUTION-PLAN.md` §0.4), but the framing in the paper changes and the fact that we
knew in advance is recorded in §7. It also becomes trade item 3 in
`EXECUTION-PLAN.md` §8.3 if capacity turns out short.

### 5.3 Statistical treatment
Three arms means the paired-test structure changes. Use CPDT pairwise
(`isalsr` vs `baseline`, `hash` vs `baseline`, `isalsr` vs `hash`) with Holm
correction across the three contrasts, plus a Friedman/Nemenyi over the three arms
per method for the critical-difference figure. Do **not** silently reuse the
two-arm machinery.

### 5.4 Completeness demonstration

Construct and report a small explicit family of isomorphic DAG pairs that the hash
baseline separates and IsalSR merges. One worked example in the paper is worth more
than a paragraph of assertion, and it makes the soundness/completeness distinction
concrete for a reader.

### 5.5 Shadow counters in the `isalsr` arm — constant memory (decided 2026-07-31)

**Decision (Mario): shadow counting is ON for all production `isalsr` runs**, not
just the twelve Stage D certification runs. That turns ρ_exact/ρ_comm/ρ_iso from a
12-run measurement into a **2,800-cell** one, stratifiable per problem with real
dispersion — which is what makes §8's decomposition table a result rather than an
anecdote.

**The memory objection, and why it does not apply.** Three extra `set[int]` at ~10⁷
entries per run is ~1–2 GB on top of an arm that already needed a 128 GB request from
heap fragmentation. That is a real OOM risk at 8,400-run scale.

**It is avoided by not storing the hashes at all.** ρ needs `|distinct|`, never
membership. `|distinct|` is estimated with a **HyperLogLog** sketch (Flajolet, Fusy,
Gandouet & Meunier, *AofA* 2007), `p = 14` → 16,384 registers ≈ **16 KB, constant in
stream length**, relative standard error `≈ 1.04/√m ≈ 0.81 %`. Three sketches ≈ 50 KB
per run against 1–2 GB, a ~10⁵× reduction, and the error is negligible against a ρ
reported to two decimal places.

| | Exact `set[int]` × 3 | HLL × 3 |
|---|---|---|
| Memory at 10⁷ candidates | ≈ 1–2 GB | **≈ 50 KB** |
| Grows with stream? | yes, linearly | **no** |
| Gives ρ | exact | ±0.81 % (1 s.e.) |
| Gives membership / soundness pairs | yes | no |

The soundness checks (equal hash ⇒ equal canonical string) need actual pairs, not
cardinalities — so they stay in **Mode 1**, on the persisted, sampled stream, where
memory is bounded by the sample. The division of labour is exact:

- **online, whole stream, constant memory** → the ρ factors;
- **offline, sampled stream** → soundness, the k-stratification cross-check, cost.

The `isalsr` arm's own dedup set stays an **exact** `set[int]` keyed on the canonical
string, unchanged. Nothing about the search is perturbed.

**Verification obligation: AC-10.** This is a memory claim on a 100,800 core-hour
campaign, so it is measured on Picasso with `sacct MaxRSS`, ON versus OFF, on both
hosts — not asserted from the arithmetic above.

**Sampling and ρ — the trap that was avoided.** ρ must be computed **online over the
full stream**, never from the sampled persisted stream. Uniform subsampling at rate
`p` does not scale distinct counts by `p`; recovering ρ from a subsample is a
species-richness estimation problem (Good–Turing / Chao) with substantial bias. The
persisted stream is for verification and stratification only.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** The three fixed-order serialisations of §4.2 implemented, unit-tested,
  and shown to be **sound** (never merges non-isomorphic DAGs) on the 14,841-DAG
  corpus, against **both** backends. **Incompleteness** demonstrated in the same
  suite: isomorphic pairs the serialisation separates and the canonical string
  merges. Rung monotonicity `|distinct(1)| ≥ |distinct(2)| ≥ |distinct(3)| ≥
  |distinct(canonical)|` holds on the corpus.
- **AC-2.** Mode 1 replay complete; `ρ_exact`, `ρ_comm`, `ρ_iso`, `ρ_total` reported
  per method, per problem, with dispersion, and stratified by k.
- **AC-2b.** SP-1…SP-6 reported as a six-row table for every Picasso probe this
  ticket ran, and SP-7's five statements established on **both hosts** before the
  implementation was declared launch-ready.
- **AC-3.** Mode 2 complete — the `hash` arm of C2, ≈2,800 runs across `D1 ∪ D2` at
  20 seeds — or every missing run accounted for in the status ledger. Mode 1 was run
  and **read** before C2 launched, and its result is recorded in §7 whichever way it
  came out.
- **AC-4.** Per-DAG cost of hash vs canonicalisation measured on Picasso hardware
  under the C++ engine and the decomposed alphabet, with the resulting `S` for all
  three arms. Note that `S` is defined on `T_search = wall_clock − canon_time_total`,
  so it is insensitive to canonicalisation speed by construction (T01 AC-6) — report
  the raw per-DAG costs and the overhead percentages, not a claimed `S` improvement.
- **AC-5.** Three-arm statistical comparison done correctly (§5.3), including the
  critical-difference diagram.
- **AC-6.** Worked isomorphic-pair example produced (§5.4).
- **AC-7.** The paper text states explicitly that the hash baseline is **sound but
  incomplete** and does not provide a labeled-DAG isomorphism invariant.
- **AC-8.** If the hash baseline turns out to be competitive on any axis, that is
  reported without softening. Record it in §7 first. This explicitly includes the
  case where rung 3 shows ρ_iso ≈ 1 on UDFS — i.e. that plain ADD/MUL commutativity
  accounts for essentially all of UDFS's redundancy and 1-WL buys nothing there.
- **AC-9.** §8 filled.
- **AC-10 (added 2026-07-31, Mario).** **Shadow counters must not OOM.** The `isalsr`
  arm carries the three fixed-order counters through the production campaign, so
  their memory cost is a launch risk, not a detail. Required evidence: a Picasso
  probe on **both** UDFS and Bingo with shadow counting **ON**, at production
  `--mem`, with `sacct` `MaxRSS` reported per task and compared against the same
  configuration with it OFF. Pass: no OOM kill, and the RSS delta is consistent with
  the constant-memory sketch (§5.5) rather than growing with stream length. If the
  delta grows with the stream, the implementation has fallen back to exact sets and
  must be fixed before launch.

---

## 7. Work log

### 2026-07-31 — plan of record (orchestrator)

**Scope of this session.** AC-3 and AC-5 need C2 to have *run*, and SP-0 forbids any
agent from submitting it. What this session can and must deliver is the **launch-gate
half** — the code without which C2 cannot be submitted at all (`EXECUTION-PLAN.md`
§3.1: *"T04 … Gates the whole launch"*), plus everything Mode 1 needs so that D3 can
be executed the moment Stage D produces a stream.

**Shared-schema ownership.** `EXECUTION-PLAN.md` §10.1 assigns checks **A7, P3, P4** to
whichever of T04/T05 starts first. T05 is NOT STARTED, so T04 lands them. **A6**
(MANIFEST) stays with T02 per §3.1.

**Work breakdown.**

| # | Deliverable | Verifies | Kind |
|---|---|---|---|
| W1 | `isalsr.baselines.fixed_order_hash` — three fixed orders, label + operand-order serialisation, same 64-bit hash as the IsalSR dedup set | AC-1, §4 | implementer |
| W2 | Soundness proof on the 14,841-DAG corpus: equal hash ⇒ equal canonical string; plus permutation-sensitivity tests showing **incompleteness** | AC-1 | implementer |
| W3 | `hash` arm runner for **both** hosts, built strictly on `agraph_to_labeled_dag` / `compgraph_to_labeled_dag` so T16 decomposition is inherited | AC-3, SP-7.1, SP-5 | implementer |
| W4 | Candidate-stream persistence (**P1**) + per-DAG cost fields (**P2**), sampled, round-trippable | AC-2, SP-7.2 | implementer |
| W5 | Mode 1 replay tool → `ρ_exact`, `ρ_iso`, `ρ_total` per method stratified by `k`, + the two soundness checks | AC-2, D3 | implementer |
| W6 | Worked isomorphic-pair family the hash separates and IsalSR merges | AC-6, §5.4 | implementer |
| W7 | Orchestrator + schema: `--variants hash`, `RunLog.representation == "hash"`, provenance fields, `data_fingerprint`, terminal-status ledger | A7, P3, P4 | implementer |
| W8 | Local smoke on both hosts, then the Picasso probe: SP-1…SP-6 table + SP-7's five statements, ≤60 tasks, seed 0, `max_time ≤ 1800 s`, `~/execs/isalsr/t04_probe/` | AC-2b | **orchestrator only** |

**W8 status 2026-07-31 — scripts written, NOT submitted.** `slurm/t04_probe/`
(`launcher.sh` pending, `worker.sh`, `tasks.txt`, `sp_probe.py`) built against the
`picasso-sbatch` skill: CPU-only (no `--gres`), `--constraint=intel` (sd nodes, for
timing comparability with T01's AC-5 table — *not* for AVX-512 safety; the extension
is built `-march=x86-64-v3`, `CMakeLists.txt:23`, which is AVX2 and portable across
sd/sr/bc/bl), `--account=tic_163_uma`, 28 tasks over two arrays (14 per host, so both
hosts are covered per SP-5), seed 0, `max_time = 1500 s`. Every SP-0 cap respected.

`sp_probe.py` **runs and works locally**: SP-2…SP-6 PASS, and **SP-1 correctly FAILS
on a dirty working tree**, which is the intended behaviour. Critically, **SP-3 passes
in both directions** — the forced-Python control asserts on *observed dispatch* (a
call counter on the C++ entry point), not on a reported string, which is what makes it
a real negative control after the `canonical.py:349` defect.

**Two blockers before submission, both orchestrator-side:**
1. **`--max-time` does not exist.** `max_time` is read only from the YAML, so a probe
   cannot be capped at 1,500 s without either duplicating four configs (which risks
   drift from production and undermines pre-flight check **A4**, config equivalence) or
   adding a CLI override. **Add the override** — it keeps one config and makes the
   deviation visible on the command line.
2. **No shadow on/off switch.** `orchestrator.py:412` reads `runner.last_shadow` and
   merges it via `dataclasses.replace`; there is no way to disable it. **AC-10's A/B
   OOM comparison is impossible until this exists.**

Both are small, but the shadow flag plumbs into the runner constructors, which the
host-native rework is editing concurrently — so they land *after* that rework, not
beside it.

### 2026-08-02 — Picasso submission: three blockers found before a single task ran

All three would have silently corrupted the probe, and two of them threaten C2.

**1. SP-1 provenance was structurally impossible.** `rsync` excludes `.git`, so the
Picasso checkout's `git rev-parse HEAD` reported **`b34cded`** while the synced files
were **`8814771`**. SP-1 would have stamped a commit unrelated to the code it ran —
the exact failure it exists to catch. Rewriting the node's git state
(`fetch` + `reset --hard`) was **rejected as destructive**, correctly. Fix:
`slurm/t04_probe/make_provenance.py` stamps the local commit *and the sha256 of all 18
sources a probe depends on* into `.provenance.json`, which travels with the rsync;
SP-1 verifies the stamp against the bytes on the node. Stronger than a commit id — it
proves the code running is the code committed, not that a `.git` once pointed
somewhere. Cleanliness is scoped to the probe's dependency set, because ticket
markdown is edited in parallel sessions and blocking on it would tempt committing
another session's half-written work.

**2. 🔴 The C++ extension on Picasso was two days stale — SP-2 FAILED.**
`.so` dated **2026-07-28 13:43**; last C++ commit `00a717e` **2026-07-30 10:20**.
Python resolves from the repo and the extension from site-packages, so the probe would
have run current Python against an old canonicaliser **with no error anywhere**.
**Anyone launching C2 without an explicit rebuild inherits this.**

**3. 🔴 The rebuild does not work on Picasso out of the box.** `pip install -e .
--force-reinstall --no-deps` **fails**: Picasso's system compiler is `g++ (SUSE)
7.5.0`, and `-march=x86-64-v3` — the portable baseline adopted precisely to avoid the
AVX-512 SIGILL trap — was only introduced in GCC 11:

```
cc1plus: error: bad value ('x86-64-v3') for '-march=' switch
```

Fix: `module load gcc/11.1.0` (the highest available) with `CXX`/`CC` exported, then
rebuild. Verified afterwards: `.so` mtime **2026-08-02 10:06**; it **imports with the
gcc module unloaded**, so workers need no runtime `module load`; and `build_info()`
reports `isa_level = x86-64-v3`, `avx512f = 0` — AVX2-only and therefore portable
across `sd`/`sr`/`bc`/`bl`, which is the outcome pre-flight **B6b** is looking for.
Recorded there as **B6b-PRE**.

**After the fixes, all six SP checks PASS on Picasso in both engine directions**
(`ISALSR_ENGINE=python` genuinely dispatches to Python, per the `canonical.py:349`
fix). `sbatch --test-only` exits 0 on both arrays.

**4. The single-task stage caught a fourth blocker — which is exactly its job.**
The first bingo task **died after 13 s**, *after* all twelve SP checks had passed on
the compute node. Cause: `bingo-nasa` imports `mpi4py`, whose ABI-probing meta-path
finder `dlopen()`s `libmpi` at **import** time and raises `RuntimeError: cannot load
MPI library` when no MPI module is loaded. It fires before any search begins, so no
amount of SP checking would have caught it.

The production worker already solves this
(`slurm/workers/models_experiment_slurm.sh:31–50`); `slurm/t04_probe/worker.sh` was
written fresh and failed to inherit three things:
`module load openmpi_gcc/5.0.9_gcc7` (with `_gcc15`/`_gcc14` fallbacks — the wrong
major version yields *"Please use mpi 5.0.9"*), `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`,
and **`PYTHONMALLOC=malloc`**. The last matters beyond mere startup: it is what keeps
Bingo+IsalSR off the OOM killer over 10k+ generations, so **AC-10's memory
measurement is only meaningful with it set**. Fixed and re-verified: bingo task
COMPLETED.

*Lesson for T02: do not author a fresh worker for C2. Extend the production one, or
diff against it line by line — SP-1…SP-6 all passing is not evidence that the job
will run.*

#### Probe submitted 2026-08-02

| | |
|---|---|
| Arrays | **1737666** (bingo, tasks 1–14) · **1737667** (udfs, tasks 15–28) |
| Shape | 28 tasks, 3 arms × 2 hosts × 4 problems + 4 shadow-OFF cells |
| Caps | `max_time` 1500 s, **seed 0**, `--constraint=intel`, `~/execs/isalsr/t04_probe/` — every SP-0 limit respected |
| Provenance | commit `a4206b8`, stamped and hash-verified on the node |
| Single-task gate | bingo `1737664_1` COMPLETED; `run_log.json` valid, `r2_train = 1.0`, `total_dags = unique_canonical = 80,201`, `canonicalization_runtime_s = 0.0` — confirms **C1.8**, the baseline arm is genuinely un-instrumented |
| 3-minute watch | 3 COMPLETED · 16 RUNNING · 2 PENDING · **0 failures**, no `ModuleNotFoundError` / `FileNotFoundError` / `oom-kill` / `Traceback` in any log |

**Open on return:** collect the SP-1…SP-6 six-row table per task (AC-2b), the ρ
comparison across all three arms on both hosts, and `sacct MaxRSS` for the shadow
ON/OFF pairs (AC-10).

### 2026-08-02 — `shadow_distinct_host_native` landed: **the C2 blocker is closed**

The `isalsr` arm now records a fourth `HyperLogLog(p=16)` sketch keyed on the
**host-native** serialisation, fed the host object (`indv` / `cgraph`) at the existing
`record_shadow` call site. Production therefore measures ρ against the baseline the
paper actually claims, not against the adapter's own normalisation.

Verified by the orchestrator: all four `shadow_distinct_*` fields present on
`SearchSpaceResults`; `mypy` clean (55 files); `ruff` clean.

| | total | unique_canon | insertion | topological | topo_comm | **host_native** |
|---|---|---|---|---|---|---|
| Bingo | 30,560 | 17,306 | 17,566.3 | 17,469.6 | 17,373.0 | **18,295.9** |
| UDFS | 2,294 | 1,135 | 1,712.2 | 1,280.4 | 1,238.6 | **2,294.7** |

`serialisation failures = 0` on both; `--no-shadow-hash` → all four `None`.

**These cross-validate the independent Mode-1-in-miniature measurements**, from a
completely separate code path:

| | ρ_exact from shadow counter | ρ_exact measured independently |
|---|---|---|
| UDFS | 2,294 / 2,294.7 = **1.000** | **1.0222** |
| Bingo | 30,560 / 18,295.9 = **1.671** | **1.7013** |

Different problems and configurations, so agreement to a few percent is the expected
outcome. UDFS's host-native count landing *on* the candidate total is the sharpest
statement of the result: **a naive fixed-order hash merges essentially nothing on
UDFS.**

**Two defects found, neither fixed, both worth acting on.**

1. 🔴 **`record_shadow`'s bare `except Exception` silently converts an import error
   into a counter of zero.** This was not theoretical: mid-implementation a formatter
   stripped the `host_native_hash` import while it was momentarily unused, and every
   single record was swallowed — 12/12 failures with the counter reading `0.0` and no
   error anywhere. It was caught only because the new test asserts
   `n_shadow_failures == 0`. **The same trap covers the three adapter-order sketches**,
   where an import regression would silently produce a plausible-looking zero. The
   handler should catch `HostNativeSerialisationError` specifically and let
   `NameError`/`ImportError` propagate.
2. `shadow_counts()` emits `shadow_distinct_host_native` **only if at least one host
   object was offered** to `record_shadow`. Forced by
   `tests/unit/test_hash_arm.py::test_shadow_counters_track_all_three_orders`, which
   asserts `set(counts) == {the three adapter fields}` while calling `record_shadow(dag)`
   with no host. Production call sites always pass the host, so the field is always
   populated there — but the conditional shape is fragile and that test should be
   updated to expect four fields.

**Baseline-count correction.** The brief quoted a regression baseline of "5,003 passed"
for `tests/unit/`; that figure was **unit + property combined**. Measured directly:
`tests/unit/` = 4,987 pre-existing (now 4,999 with 12 new), `tests/property/` = 16
separately. The implementer flagged the mismatch rather than quietly matching the
number, which is the right call — no regression exists.

**Defect for T02/A7, found in the probe's own output:** `run_log.json`'s
`metadata.hardware` has **no `engine` field** — it read `<none>`. Check **A7** requires
`engine` recorded per run, and **C1.14** requires asserting `engine == native` on
420/420 smoke tasks. As it stands that assertion cannot be made from a RunLog at all.
T04 is unaffected (its engine evidence lives in `sp_evidence.json`), but **C2 needs
this before Stage C.**

### 2026-07-31 — host-native rework landed, and the red state proves the diagnosis exactly

**Re-verified by the orchestrator in the main tree:** `pytest tests/unit/ -q` →
**4,966 passed, 5 skipped** (4,930 + 36 new, no regression); `mypy src/isalsr/` →
Success, 55 files; `ruff` clean; `HashBingoRunner.KEY_MODE == HashUDFSRunner.KEY_MODE
== "host_native"`. Both smokes exit 0 with `"representation": "hash"`.

**The red state is stronger evidence than the original 25.1 % diagnosis.** On the same
streams, with the *old* key:

| host | N | host-native | adapter `INSERTION` | adapter `TOPOLOGICAL` (old arm key) | canonical |
|---|---|---|---|---|---|
| UDFS | 1,500 | 1,468 | 1,298 | **1,298** | 1,294 |
| Bingo | 3,000 | 741 | 696 | **696** | 690 |

The old arm key reproduced the adapter-`INSERTION` distinct count **exactly**, on both
hosts. So the `TOPOLOGICAL` re-sort contributed **zero** merges beyond the adapter's
own VAR/CONST/topological layout: the arm was not "partly inheriting" the adapter's
normalisation, it was measuring **nothing but** the adapter's normalisation. Soundness
0 violations throughout (0/1,500 UDFS, 0/3,000 Bingo).

**Design.** `src/isalsr/baselines/host_native.py` is a generic serialiser over
`(host_key, host_tag, operand_keys)` records, stdlib-only, importing nothing from
`experiments/`; the host-specific extraction lives in each runner
(`udfs_host_native_records` iterates `cg.node_dict` in key order and **never**
`eval_order`; `bingo_host_native_records` takes utilised `command_array` rows in row
order). `key_mode="hash"` (adapter-order) remains selectable as the steel-manned second
rung. Shadow sketches raised to **p=16** per the earlier precision finding.

**Deviation from the brief, reported rather than hidden — and it is the right call.**
Bingo unary/terminal rows emit `param1` only. Measured: **1,679 of 2,595 utilised
unary rows (64.7 %) carry `param1 != param2`**, but Bingo never reads `param2` for
unary operators, so emitting it would split identical expressions on junk — exactly
the argument the brief itself used to exclude dead code. Terminals: 0/3,010 mismatch.
Binary rows emit both operands in stored order (Invariant 8, pinned by test).

**Consequence, and it moves the headline the right way.** The naive arm now merges
*less*: the UDFS smoke went to `skipped=0` (1,491 total, 1,491 unique) where the
adapter-order key had been merging ~13 % of the stream. That is consistent with the
corrected ρ_exact ≈ 1.02 for UDFS and with §2's prediction. **Every T04 delta quoted
before this entry is superseded.**

### 2026-07-31 — ✅ **Both hosts measured correctly. §2's two predictions both confirmed.**

Bingo re-measured on the **evaluation-event stream** (the candidates the dedup hook
actually converts), fixing the population-snapshot defect that voided the earlier
figure. Exact set cardinalities, not sketch estimates.

**Bingo — 120,741 candidates, 0 canonicalisation failures:**

| Scheme | distinct | ρ |
|---|---|---|
| host-native (**the arm**) | 70,970 | 1.7013 |
| adapter `INSERTION` | 68,168 | 1.7712 |
| adapter `TOPOLOGICAL` | 68,051 | 1.7743 |
| adapter `TOP_COMMUTATIVE` | 67,592 | 1.7863 |
| canonical (IsalSR) | 67,491 | **1.7890** |

**Sanity: ρ_total = 1.789 against the submitted campaign's Bingo ρ = 1.83.** The
method now reproduces a known-good reference; the earlier 8.03 was pure artefact.

**The headline, both hosts:**

| | ρ_exact | ρ_iso | ρ_total | **duplicates requiring 1-WL** |
|---|---|---|---|---|
| **UDFS** | 1.0222 | **1.3724** | 1.4029 | **92.4 %** |
| **Bingo** | **1.7013** | 1.0515 | 1.7890 | **6.5 %** |

**Both of §2's predictions are confirmed, including the uncomfortable one.**

- *"Predicts ρ_exact ≈ 1 for UDFS"* — measured **1.0222**. UDFS enumerates
  systematically, so it rarely regenerates a byte-identical candidate; nearly all of
  its redundancy is isomorphism redundancy, and a fixed-order hash cannot see it.
- *"Bingo is the interesting case: stochastic GP re-generates byte-identical
  individuals… so ρ_exact should be materially above 1 there. That is the honest
  concession, and it should be stated before a reviewer extracts it."* — measured
  **1.7013**. The mechanism is exactly the one §2 named: `VarAnd` emits unmodified
  `parent.copy()` offspring ~36 % of the time (B12).

**This is the answer to R1.4, and it is a decomposition rather than a win** — which
§2 argued is the stronger result. The two hosts sit at opposite ends: on UDFS the
1-WL machinery accounts for **92.4 %** of the deduplication and the naive baseline is
nearly inert; on Bingo it accounts for **6.5 %** and the naive baseline captures most
of the benefit. Neither number is softened, and the Bingo concession goes in the
response letter **before** a reviewer extracts it, per §2 and AC-8.

**Status of these numbers.** Local, short runs (Nguyen-7 for Bingo; a bounded UDFS
enumeration), single seed, one problem each — *provisional*, in SP-0's sense. They fix
the measurement method and the expected direction; the values of record come from
Mode 1 (D3) over the Stage D certification streams and from the C2 `hash` arm.

**Gap to close: the shadow counters do not include host-native.** They currently track
the three adapter-order serialisations only, so the production campaign would record
ρ against the *wrong* baseline. `shadow_distinct_host_native` must be added before C2
launches — this is now the last blocking item for AC-2 at campaign scale. Also note
that at p=16 the adapter `TOP_COMMUTATIVE` shadow estimate came in *below* the exact
canonical count (61,331 vs 61,540, −0.34 %), which is arithmetically impossible and is
therefore sketch error inside the 0.41 % standard error — the small contrasts remain
at the edge of sketch precision, so **report ρ_comm from exact replay, never from the
sketch**.

**Standing constraints carried into every brief.** SP-0 (probes only, never C2);
decomposed alphabet, `k` ≈ +22 % over any pre-T16 number in the repo; 20 seeds not 30;
C++ engine is the engine — every core-semantics check runs against **both** backends
and disagreement is the tell; `picasso-sbatch` before any SLURM file.

**Open question deferred to the user, not decided here:** whether the `hash` arm's
per-run cost accounting should reuse `canonicalization_runtime_s` (making Table 2
directly comparable but overloading the field name) or add a parallel field.

### 2026-07-31 — three decisions, and a rescope of §4 (Mario, via orchestrator)

**1. §4 was over-built and misread the comment. Rescoped — see §4.**

The orchestrator proposed adding a *fourth* order: root-down DFS with ADD/MUL
operands sorted recursively by child hash, on the argument that it is the strongest
naive baseline and that omitting it leaves a round-2 hole. **Mario rejected it**, on
the grounds that R1.4 says "**a** fixed-order DAG serialization" — singular, naive,
"the obvious baseline" — and that we should not manufacture a comparator capable of
competing with our own method when the reviewer asked for one simple one.

On review this is correct, and for a sharper reason than schedule risk: a recursive
hash-consing scheme with sorted commutative operands **is not a fixed-order
serialisation at all**. It is a canonicalisation with a weaker invariant. Shipping it
labelled "naive baseline" would have been a category error, and it would have put a
method into the evaluation that nobody requested.

The reviewer's *purpose* clause still needs a number, and it now gets one from the
**offline three-rung ladder** (§4.2) rather than from extra arms: `ρ_total = ρ_exact
× ρ_comm × ρ_iso`. Rung 1 vs 2 answers "was your order chosen to lose"; rung 2 vs 3
answers "how much is just ADD/MUL commutativity" — the question
`supplementary.tex:689–693` invites. Zero extra campaign compute. **The DFS-from-`x_1`
order originally listed as (iii) is dropped**, for the same reason as the fourth: its
`(label, subtree size)` sort key makes it a partial canonical form.

**2. Cost field: reuse `canonicalization_runtime_s`.** The `hash` arm writes its
serialise+hash cost there. Table 2, `S`, and the overhead percentages stay
structurally identical across all three arms with no analyzer changes and no per-arm
field selector to get silently wrong. `RunMetadata.representation == "hash"`
disambiguates every row; the overload is disclosed in Appendix D. This closes the
open question left above.

**3. Shadow counters ON in production, via a sketch — new AC-10.** Mario asked for
full-campaign shadow counting *if* an engineering fix avoided the OOM risk, and for
an explicit OOM probe either way. The fix: ρ needs `|distinct|`, not membership, so
the shadow counters are **HyperLogLog** registers (≈16 KB each, constant in stream
length) instead of `set[int]` (~1–2 GB at 10⁷ candidates). ≈10⁵× reduction, ~0.81 %
relative standard error against a ρ quoted to 2 dp. The `isalsr` arm's own dedup set
is untouched and stays exact. Details and the sampling trap this avoids: §5.5.
Verification is **AC-10**, on both hosts, with `sacct MaxRSS` ON vs OFF — a memory
claim on a 100,800 core-hour campaign gets measured, not asserted.

**Engine state established before any code was written.** The site-packages `.so`
was dated 2026-07-30 09:57 while the last commit touching `src/isalsr/core/native/`
(`00a717e`) was 10:20 — SP-2 was **not** satisfied. Rebuilt with
`pip install -e . --force-reinstall --no-deps` (exit 0 read directly, not through a
pipe); `.so` now 2026-07-31 18:05, `DEFAULT_BACKEND == "cpp"`, and C++/Python
`fast_canonical_string` agree byte-exact on a probe DAG. All T04 work in this session
runs against that build.

**Integration surface, verified independently of the mapping agent.** Two edits the
work breakdown did not anticipate:
- `experiments/models/io_utils.py:169–173` hard-codes `{"problem", "baseline",
  "isalsr"}`, so `paths["hash"]` raises `KeyError`.
- `experiments/models/orchestrator.py:422` gates paired stats on exactly
  `{"baseline", "isalsr"}`; a `hash` arm would silently produce `aggregate.csv` and
  no `paired_stats.json`.
- `--variants` (`orchestrator.py:524`) has **no** argparse `choices`; the only
  validation is the `ValueError` inside `create_runner` (`:189`, `:201`).
- The live dedup key is CPython builtin `hash(canonical)` (`bingo:375`, `udfs:139`) —
  SipHash, **PYTHONHASHSEED-randomised**, confirmed by observing two different values
  for the same string across processes. It therefore **cannot be persisted or
  replayed**. The arm keeps builtin `hash()` live (so the collision and memory
  analysis is genuinely shared, per §4.1), and the persisted stream carries a stable
  `blake2b(digest_size=8)` digest instead.

### 2026-07-31 — W1/W2 landed: AC-1 met, and ρ_comm is 1.00 on corpus 1

**Delivered.** `src/isalsr/baselines/{__init__,fixed_order_hash,cardinality}.py` +
`tests/unit/test_fixed_order_hash.py` (97 tests). Public surface: `FixedOrder`,
`serialise`, `deserialise`, `fixed_order_hash` (builtin `hash`, live key),
`fixed_order_digest` (blake2b-8, stable key), `HyperLogLog(p=14)`.

**Re-verified in the main tree by the orchestrator, not taken on report:**
`pytest tests/unit/test_fixed_order_hash.py -q` → **97 passed**;
`ruff check` → clean; `mypy src/isalsr/baselines/` → **Success, 3 files**.
Full `tests/unit/` → 4,908 passed, 5 skipped (no regression).

**AC-1 evidence, corpus 1 = 14,841 DAGs:**

| Rung | distinct |
|---|---|
| `INSERTION` | 13,472 |
| `TOPOLOGICAL` | 13,196 |
| `TOPOLOGICAL_COMMUTATIVE` | **13,196** |
| canonical string (C++) | 7,625 |

- **Soundness: 0 violations**, 3 orders × 2 backends. C++ and Python canonical
  strings agreed on all 14,841 — which also re-establishes SP-2/SP-3 locally, since a
  stale `.so` would have shown up here as disagreement.
- **Incompleteness: 3,305 / 7,625** canonical classes are split by every rung — i.e.
  43 % of the equivalence classes IsalSR merges, a fixed-order hash does not. Plus a
  deterministic witness, `sin(sin(x₀)) + sin(x₀)`.
- Round trip: 0 structural and 0 canonical mismatches; `is_isomorphic` true on all
  14,841.
- HLL: unbiased over 20 independent streams of 10⁶, mean **−0.20 %**, sd **0.884 %**
  against a theoretical 0.8125 %. Sizing for §5.5 confirmed.

**Finding, and it is an honest-negative one (AC-8 applies): ρ_comm = 1.00 on this
corpus.** `TOPOLOGICAL_COMMUTATIVE` bought **zero** additional merges over
`TOPOLOGICAL` — 13,196 both ways. The middle factor of §2's decomposition is
*empty here*.

**This was checked for the obvious bug and it is not one.** The orchestrator built a
hand-made pair differing only in the operand order of one `ADD` node
(`ordered_inputs` `[2,1]` vs `[1,2]`) and confirmed the rungs behave as designed:

| | merges the pair? |
|---|---|
| `INSERTION` | no |
| `TOPOLOGICAL` | no |
| `TOPOLOGICAL_COMMUTATIVE` | **yes** |
| canonical string | yes |

So rung 3 is functionally live and the zero delta is a property of **corpus 1**, not
of the implementation. The mechanism is Critical Invariant 8: S2D builds a binary op
via `V`/`v` from the *first* operand and closes it with `C`/`c`, so operand order is
fixed by construction and the corpus simply contains no operand-swapped pairs.

**Why this does not settle the question.** Corpus 1 is S2D-generated. The claim
`ρ_comm > 1` was never about S2D output — it is about **host-generated** candidates,
where `supplementary.tex:689–693` locates UDFS's duplicates in ADD/MUL commutative
symmetry. Bingo and UDFS build DAGs through their own genetic operators, not through
S2D, and have no reason to respect Invariant 8's construction discipline. **The
number that matters is ρ_comm on the Mode 1 replay of real Bingo/UDFS streams**, and
it is not yet in hand.

Recorded now, before the answer is known, so it cannot be quietly dropped if it comes
out at 1.00 there too. If it does, §8 says so plainly and the decomposition reverts
to two factors — that is a result, not a failure.

### 2026-07-31 — 🔴 SP-3's negative control was inoperative. Fixed. **This is T02's problem too.**

Found while building this ticket's probe harness, not by looking for it.

**The defect.** `fast_canonical_string` resolved `backend=None` by reading
`backends.DEFAULT_BACKEND` (`canonical.py:349`) — a compiled-in constant that is
`"cpp"` whenever the extension imports. It therefore **bypassed the `ISALSR_ENGINE`
override entirely**, in violation of `backends.py`'s own documented resolution order
(*"1. `ISALSR_ENGINE` environment variable, 2. `DEFAULT_BACKEND`"*, `backends.py:6–10`).

**Why it is worse than a plain bug.** The *reporting* surface honoured the override
while the *computation* did not:

| With `ISALSR_ENGINE=python` | Before fix | After fix |
|---|---|---|
| `backends.engine()` | `python` | `python` |
| `backends.build_info()["engine"]` | `python` | `python` |
| **C++ actually invoked by `fast_canonical_string`** | **`True`** | `False` |

Measured by monkey-patching `_cpp_ext.fast_canonical_string` with a call counter, so
this is dispatch observed, not inferred.

`EXECUTION-PLAN.md` §4.0 SP-3 and §4.2 **B2** both require re-running the probe "with
the Python path forced" and asserting it reports `python`, and both warn that a probe
reporting `native` in both directions "proves nothing and is itself a defect". The
actual state was the inverse and strictly more dangerous: the probe would have
reported **`python`**, passed B2, and been running **C++** the whole time. It would
have produced false confidence rather than a visible failure.

**Blast radius — this is the part T02 must act on.** Any "run the check against both
backends" sweep driven by the **environment variable** rather than an explicit
`backend=` keyword has been exercising **C++ twice** and never Python. `CLAUDE.md`
states the both-backends rule as a standing invariant ("disagreement is the tell"), so
the exposure is wherever that rule was applied via the env var.

- **Not affected:** T04 W1/W2, which passed `backend="cpp"` / `backend="python"`
  explicitly and got 0 disagreements over 14,841 DAGs.
- **T01's equivalence gate: checked, and it is SAFE.**
  `experiments/scripts/equivalence_gate_evolved.py:11–12` and
  `experiments/models/equivalence_probe.py:13–14` both select the backend by
  **explicit `backend=` kwarg**, and `ISALSR_ENGINE` appears in **no** gate harness
  (`equivalence_gate*.py`, `equivalence_probe.py`, `slurm/t01_close/`,
  `slurm/smoke_cpp/`). The 117,798-DAG AC-3 gate 3 result **stands and does not need
  re-running.** Recorded as `EXECUTION-PLAN.md` check **B2b**, closed, so it is not
  re-opened later on a vague memory of "there was a backend bug".
- **Real exposure** was therefore confined to probes **not yet written** — including
  the one this ticket was about to write, which is how it was found.

### 2026-07-31 — ⛔ **RETRACTED — the entry below is WRONG. Read the correction after it.**

> The ρ_iso ≈ 1.00 result recorded below is an **artefact of measuring the baseline on
> the adapter's output instead of the host's own representation**, and it is
> **reversed** by the correction that follows. Kept in place, not deleted, because the
> mistake is instructive and because a work log that quietly erases its wrong turns is
> not evidence of anything. **Do not cite any number from the entry below.**
>
> Caught by Mario, who pushed back that the deduplication was "not naive enough".
> That was correct, and the mechanism is named in the correction.

### 2026-07-31 — [RETRACTED] ρ_iso ≈ 1.00 on Bingo host DAGs. AC-8 fires. ESCALATED.

**This is the honest-negative branch the ticket was written to catch, and it arrived
before a single core-hour was committed — which is exactly what Mode 1 was for.**

Corpus 1 said nothing about ρ_comm because S2D fixes operand order by construction
(previous entry). So the orchestrator measured the decomposition directly on
**host-generated** DAGs, using `build_bingo_pipeline` so the candidates are precisely
what the production runner sees, T16 decomposition included.

**Run A** — default config (stack 16, pop 100, target `x³+x²+x`), 3 seeds × 40 gens,
12,000 candidates, 0 conversion failures:

| Scheme | distinct | ρ |
|---|---|---|
| `INSERTION` | 1636 | 7.3350 |
| `TOPOLOGICAL` | 1632 | 7.3529 |
| `TOPOLOGICAL_COMMUTATIVE` | 1587 | 7.5614 |
| canonical | **1587** | **7.5614** |

`TOPOLOGICAL_COMMUTATIVE` and the canonical string produced **identical distinct
counts**. Since rung 3 is sound (0 unsound merges ⇒ its partition refines the
canonical one), equal cardinality forces the partitions to be **identical**. On this
stream the naive scheme is not merely close to 1-WL — it *is* 1-WL.

**Run B** — **production operator set** `[+,-,*,/,sin,cos,exp,log,sqrt,pow]`,
stack 32, pop 300, 3 seeds × 35 gens, on a Pagie-1 target (a *structural-bottleneck*
problem, i.e. the regime `bottleneck_type_analysis.md` says IsalSR should help most).
**31,500 candidates, 0 conversion failures:**

| Scheme | distinct | ρ |
|---|---|---|
| `INSERTION` | 3966 | 7.9425 |
| `TOPOLOGICAL` (**the arm**) | 3964 | 7.9465 |
| `TOPOLOGICAL_COMMUTATIVE` | 3925 | 8.0255 |
| canonical | 3921 | 8.0337 |

```
ρ_exact = 7.9465     ρ_comm = 1.0099     ρ_iso = 1.0010     ρ_total = 8.0337
```

**k-stratified ρ_iso: 1.0000 at every k in 1…19**, except k=4 (1.0022) and k=5
(1.0036). k=0 shows 2.0 on 413 degenerate candidates with no internal nodes.
**Bingo's post-T16 p95 is k=15**, so the null result covers the entire production
range — this is not a small-k artefact.

**The number R1.4 asked for, on Bingo:**
> **fraction of duplicates requiring 1-WL ≈ 0.1 %.** The naive arm alone recovers
> **98.9 %** of IsalSR's deduplication (7.9465 / 8.0337).

**Bounds on the claim — stated so it is neither softened nor overstated:**
1. **Bingo only.** UDFS untested here, and UDFS is the host `supplementary.tex:689–693`
   speaks to. It must be measured before anything is written.
2. **Short local runs**, not 12 h. Mitigated but not removed by the uniform k-coverage
   to 19.
3. **Not comparable to the published ρ = 1.83.** That is measured on a dedup-*active*
   stream; this counts raw per-generation population snapshots. Different denominators.
   Do not put these two numbers in one table.
4. **The obvious confound was checked and it does NOT explain the result.**
   `TOPOLOGICAL` breaks ties on `(label, in-degree, out-degree)`, which is itself a
   partial structural normalisation, so its near-parity with 1-WL might have been an
   artefact of that rather than a real absence of value in the complete invariant.
   **`INSERTION` settles it: it performs no re-sorting whatsoever, and it is still
   within 1.15 % of the canonical string** (3966 vs 3921). Removing every trace of
   normalisation moves the answer by ~0.3 pp. The result is not a tie-break artefact.

**Why this is coherent rather than surprising — the mechanism.** Bingo's genome *is*
its numbering: a `command_array` is a topologically ordered stack program, so two
Bingo individuals encoding the same expression built the same way already carry the
same node order. Isomorphic-but-renumbered variants are therefore rare by
construction. Meanwhile `CLAUDE.md` B12 records that `VarAnd` emits unmodified
`parent.copy()` offspring ~36 % of the time — **exact** duplicates. So the redundancy
Bingo actually generates is overwhelmingly byte-identical regeneration (ρ_exact ≈ 7.9),
which the simplest possible hash catches, and almost none of it is the isomorphism
redundancy IsalSR targets. The two measurements agree; nothing here is anomalous.

**Escalated to Mario, not decided here.** [This escalation was withdrawn — see below.]

### 2026-07-31 — ✅ **CORRECTION: the baseline was measured on the wrong object. ρ_iso = 1.37, and 92.4 % of duplicates require 1-WL.**

**What was wrong.** Every rung above was computed on the **adapter's output**
(`compgraph_to_labeled_dag` / `agraph_to_labeled_dag`), not on the host's own
representation. The adapters do not merely translate — they **renumber into a
normalised layout**: `udfs/adapter.py:140–155` assigns node indices as *VARs first in
variable-index order, then CONSTs, then operators in topological/evaluation order*.
That is already a partial canonical form.

So the "naive" baseline was being handed **IsalSR's own normalisation infrastructure
for free**, then credited with the deduplication that normalisation performed. The
comparison was rigged against ourselves, which is why ρ_iso collapsed to ≈1.

**Measured, on the same 604,334 UDFS candidates:**

| Scheme | distinct | ρ |
|---|---|---|
| **host-native order — genuinely naive** | 591,190 | **1.0222** |
| adapter `INSERTION` | 442,843 | 1.3647 |
| adapter `TOPOLOGICAL` | 437,444 | 1.3815 |
| **canonical string (IsalSR, 1-WL)** | 430,773 | **1.4029** |

The adapter collapses **148,347 distinct host representations — 25.1 % of them —
before the naive hash ever runs.**

**The corrected decomposition, UDFS:**

```
ρ_total = 1.4029   =   ρ_exact 1.0222   ×   ρ_iso 1.3724
                       ^^^^^^^^^^^^^^       ^^^^^^^^^^^^
                       a naive fixed-order  requires the isomorphism-
                       hash catches this    complete invariant
```

**The number R1.4 asked for.** Total duplicates `604,334 − 430,773 = 173,561`.
Caught by the naive hash: `604,334 − 591,190 = 13,144`.

> **92.4 % of UDFS's duplicate candidates require 1-WL machinery.** A naive
> fixed-order serialisation hash catches **7.6 %**.

**This confirms the ticket's own prediction.** §2 read `supplementary.tex:689–693`
("duplicates arise only from the commutative symmetries of ADD and MUL") as predicting
**ρ_exact ≈ 1 for UDFS**. Measured: **1.0222**. The prediction was right, and the
earlier contradiction of it was a measurement error, not a finding.

**Consequences for the implementation — W1/W3 need rework.**
1. The fixed-order serialisation must operate on the **host's native representation**
   (Bingo `command_array`; UDFS `CompGraph.node_dict` in its own key order), **not** on
   adapter output. This is what "fixed-order DAG serialization" means when the order is
   *the host's* rather than one we chose.
2. **Report both rungs, labelled honestly.** Host-native (ρ = 1.02) is the genuine
   baseline R1.4 names. Adapter-order (ρ = 1.36) is a *steel-manned* variant that
   concedes our own preprocessing to the baseline. Reporting both is stronger than
   either alone: *even granting the baseline IsalSR's adapter normalisation, ρ still
   improves 1.36 → 1.40; without that concession it manages only 1.02.* That pre-empts
   both "your baseline was crippled" and "your baseline was secretly your own method".
3. The Bingo numbers above are void for a **second, independent** reason: they sampled
   `island.population` once per generation, so an individual surviving 20 generations
   counted as 20 exact duplicates. That inflated ρ_exact and mechanically suppressed
   ρ_iso. Bingo must be re-measured at the **evaluation-event** stream (`_serial_eval`),
   which is what the dedup hook actually sees. The UDFS figures do not have this defect
   — they were captured at `evaluate_cgraph`, the true stream.

**Process note.** Two independent errors pointed the same way and would have produced a
confidently wrong answer to the heaviest comment in the round. Both were caught by
Mario's objection that the baseline was "not naive enough", not by the verification
machinery, which had signed off on soundness, monotonicity and 97 green tests. Soundness
tests cannot detect *measuring the wrong object* — every number above was internally
consistent and wrong. Recorded for §8.4 residual risk.

### 2026-07-31 — W3/W7 landed (arm wired, both hosts) — but **built on the wrong object**

**Delivered and re-verified by the orchestrator in the main tree:**
`pytest tests/unit/` → **4,930 passed, 5 skipped** (4,908 baseline + 22 new, no
regression); `mypy src/isalsr/` → Success, 54 files; `ruff` clean. All six
`(method, variant)` pairs resolve to distinct runner classes — `HashBingoRunner` and
`HashUDFSRunner` exist and report `variant == "hash"`. Four smokes exit 0 and write
valid `run_log.json`. Test file was seen **red first** (19 failed / 3 passed) before
implementation.

Implementation shape is right: `_CanonicalDeduplicator` was **parameterised**
(`key_mode`, `shadow_hash`) rather than forked, so the arms differ in exactly the
equivalence relation and the `_parent_ids` clone-detection (Invariant 11) is shared
untouched. `io_utils.ensure_output_structure` is now variant-driven with the legacy
arms always present.

**🔴 Required rework, from the correction above.** Both hash runners key on
`fixed_order_hash(adapter_output, TOPOLOGICAL)`. Per the correction, that is **not the
naive baseline** — it inherits the adapter's VAR/CONST/topological renumbering, which
alone performs 25.1 % of the deduplication. The arm must key on a **host-native**
serialisation (Bingo `command_array`; UDFS `CompGraph.node_dict` in its own key
order), with the adapter-order variant retained as the *steel-manned* second rung.
The wiring, the parameterisation, the counters and the output tree all survive; only
the key function's input changes.

**Three findings from the implementer, all real, none fixed:**

1. **HLL p=14 is too coarse for the Bingo shadow contrast.** Bingo topological
   17,603.5 vs exact canonical 17,401 → +1.17 %, against a 0.81 % standard error:
   ~1.4 σ, i.e. **not resolvable**. UDFS's +11.5 % is ~14 σ and is fine. Since ρ_comm
   and ρ_iso are *small* ratios, the sketch must resolve well below them. **Raise to
   p = 16** (65,536 registers, 64 KB, s.e. ≈ 0.41 %) — still ~10⁴× cheaper than exact
   sets, so §5.5's argument is unaffected. Update AC-10's sizing accordingly.
2. **`compute_paired_stats` raises `ValueError` at < 3 paired seeds *after* every
   `run_log.json` is written**, so a multi-variant run with 1–2 seeds exits 1 with
   valid results on disk. Pre-existing (same path for baseline+isalsr) but the
   three-arm default makes it hit far more often. **This interacts badly with P4/A10
   and the resume logic**: a cell that is complete on disk but whose job exited
   non-zero is exactly the "35 unexplained cells" failure mode C2 exists to prevent.
   Belongs to T08's analyzer lane; filed, not absorbed.
3. **Live `ρ_hash` (1.7572) marginally exceeded `ρ_isalsr` (1.7562) on Bingo**, by
   0.06 %. **Not a violation of SP-7.3 or C1.7**, which are guarantees on *identical
   replayed input*; live, the two arms explore different trajectories (15,530 vs
   30,560 candidates), so the ordering is only "strongly expected". Worth restating in
   C1.7 that the live check is a smell test, not a proof — the guaranteed version is
   Mode 1's.

### 2026-07-31 — the baseline has a name, a citation, and an SR-specific precedent

Verified bibliography (fabricated citations were explicitly forbidden; one caveat
noted below).

**The baseline R1.4 names is hash-consing / Merkle-style bottom-up structural
hashing over a fixed traversal order**:
`H(n) = h(label(n), H(c₁), …, H(c_a))`, children in stored operand order, dedup on
`H(root)`.

| Role | Reference |
|---|---|
| **SR-specific instance — the one to cite** | Burlacu, Kammerer, Affenzeller & Kronberger, *Hash-based Tree Similarity and Simplification in Genetic Programming for Symbolic Regression*, EUROCAST 2019, LNCS 12013:361–369. DOI 10.1007/978-3-030-45093-9_44; arXiv:2107.10640 |
| Provenance | Ershov, *On programming of arithmetic operations*, CACM 1(8):3–6, 1958. DOI 10.1145/368892.368907 |
| Modern formulation | Filliâtre & Conchon, *Type-safe modular hash-consing*, ML '06:12–19. DOI 10.1145/1159876.1159880 |
| GP framing of the tradeoff | Burke, Gustafson & Kendall, IEEE TEC 8(1):47–62, 2004. DOI 10.1109/TEVC.2003.819263 |
| 1-WL is incomplete in general | Cai, Fürer & Immerman, *Combinatorica* 12(4):389–410, 1992. DOI 10.1007/BF01305232 |
| Refinement + individualisation = practical complete canonical form | McKay & Piperno, *Practical graph isomorphism, II*, JSC 60:94–112, 2014. DOI 10.1016/j.jsc.2013.09.003 |

*Caveat on Ershov 1958: the ACM page returns 403; volume/issue/pages/DOI were
confirmed against two independent secondary sources, not the publisher. Verify before
it enters the response letter.*

**The uncomfortable part, stated plainly.** Burlacu et al.'s Algorithm 1 **sorts child
hashes for commutative symbols**. So the literature-standard "naive" baseline in our
own field already absorbs ADD/MUL commutativity — and because a Merkle hash is
computed bottom-up over structure, **it is inherently invariant to node numbering**.
The entire host-native-vs-adapter-order distinction that dominated the earlier
correction simply *does not apply to it*. It is a much stronger comparator than the
serialisation rungs built so far.

**Where it still falls short of IsalSR, and this is the real answer to R1.4:**
1. **It is defined on trees, not DAGs.** The authors' own scope. Our objects have
   **shared subexpressions** — that is what the representation is *for*. A tree hash
   either expands the DAG (exponential) or ignores sharing, and either way it does not
   distinguish DAGs that differ only in their sharing pattern.
2. **The authors call it "inexact"** — collisions are accepted by design.
3. It handles commutativity but not general **automorphism** of the labeled DAG.

**Gift for the framing, from the GP literature itself.** Burke et al. 2004, p. 48,
verbatim: *"Graph isomorphism could be applied to genetic programming tree structures
as a measure of diversity. However, due to the nature of nodes used in genetic
programming, the properties (associativity, commutativity, etc.) would require
special, and possibly complex, implementations of isomorphism… determining graph
isomorphism would be computationally expensive for an entire population. However, a
measure of possible isomorphic trees could be found by noting simple properties."*
The field states the cheap-proxy-vs-complete-invariant tradeoff in its own words, and
IsalSR is precisely the "special implementation" it declines to build. This belongs in
§8.3 and probably in the introduction.

**Honesty item for AC-7 / §8.** McKay & Piperno 2014 is the correct framing of our own
method: WL refinement is the *cheap half*; completeness needs the individualisation
(backtracking) step. And Cai–Fürer–Immerman 1992 means **1-WL alone is not a complete
invariant in general** — our completeness claim rests on WL *plus* backtracking over
tied candidates, verified exhaustively for k ≤ 8. The response letter must not say
"1-WL is complete".

### 2026-07-31 — DECISION (Mario): plain fixed-order serialisation is the arm; Burlacu is addressed in prose

**Decided.** The C2 arm implements **one plain fixed-order serialisation hash on the
host-native representation** — literally what R1.4 names. The Burlacu et al. (2019)
Merkle hash is **not implemented and not run**.

**It is disclosed, not omitted.** The R1.4 answer explicitly cites Burlacu et al. and
states why it is not the comparator: it is defined on **trees**, whereas IsalSR's
objects are **DAGs with shared subexpressions**, which is the representation's whole
purpose; and its authors describe it as *inexact*. Paired with the Burke, Gustafson &
Kendall (2004, p. 48) passage, in which the GP literature itself names exact
isomorphism as the correct-but-expensive option and settles for cheap structural
proxies, this is a stronger position than silently shipping a comparator we did not
need to ship. **The reviewer is told the stronger baseline exists and why it does not
apply.** See §8.3.

**Scope limits on this decision, so it is not over-read later.**
- We have **not measured** a Burlacu-style hash, so the paper and the letter assert
  **nothing** about how it would perform. No speculation in either direction.
- The exploratory scratch measurements in the entries above (including
  `TOPOLOGICAL_COMMUTATIVE` landing 0.17 % from the canonical string on adapter
  output) stay in **this work log** and go **nowhere else**. They were taken on the
  wrong object, on one problem, at a non-production configuration, and they are not
  results of record. They are retained because an internal log that is pruned of its
  inconvenient measurements stops being usable as a record.
- If a round-2 reviewer constructs the Burlacu comparator themselves, the prepared
  answer is the tree-vs-DAG argument above — not a claim that we never considered it.

**Fix.** `canonical.py:349` now resolves through `_backends.engine()`. One line, with
the rationale in a comment so it is not "simplified" back. Verified: default still
dispatches to C++; `ISALSR_ENGINE=python` now genuinely does not.

**Regression:** `pytest tests/unit/ tests/property/ -q` → **4,946 passed, 5 skipped**;
`ruff` clean; `mypy src/isalsr/` → Success, 54 files.

---

## 8. Proposed answer

### 8.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Hash-dedup baseline present | **absent** | present, full comparator | §5.2 |
| Fixed order used by the arm | — | **host-native** | §4.1 |
| ρ_exact (hash-catchable), UDFS | not reported | *1.0222* | rung 1 |
| ρ_exact (hash-catchable), Bingo | not reported | *1.7013* | rung 1 |
| ρ_iso (needs 1-WL), UDFS | not reported | *1.3724* | residual |
| ρ_iso (needs 1-WL), Bingo | not reported | *1.0515* | residual |
| **Duplicates requiring 1-WL, UDFS** | not reported | ***92.4 %*** | **the number R1.4 asked for** |
| **Duplicates requiring 1-WL, Bingo** | not reported | ***6.5 %*** | **the number R1.4 asked for** |
| ρ_total, UDFS | 1.56 | *1.4029* | |
| ρ_total, Bingo | 1.83 | *1.7890* | |
| Order-choice sensitivity (host-native vs adapter order) | not reported | *UDFS 1.13×, Bingo 1.04×* | §4.2 |

*Italicised values are **provisional** — local single-seed runs that fix the method and
direction. Replace every one with the Mode 1 (D3) / C2 figure before T14.*
| ρ_total, UDFS | 1.56 | | |
| ρ_total, Bingo | 1.83 | | |
| Fraction of duplicates requiring 1-WL, UDFS | not reported | | **the number R1.4 asked for** |
| Fraction of duplicates requiring 1-WL, Bingo | not reported | | **the number R1.4 asked for** |
| Per-DAG cost, hash (ms) | — | | |
| Per-DAG cost, IsalSR canon (ms) | 0.817 (Bingo, Python) | | T01/T02 |
| `S`, hash arm, Bingo | — | | Mode 2 |
| `S`, IsalSR arm, Bingo | 0.93 | | T02 |
| R² test, hash vs baseline | — | | CPDT |
| R² test, IsalSR vs hash | — | | CPDT |
| Merges non-isomorphic DAGs? | — | hash: no · IsalSR: no | AC-1 |
| Merges isomorphic renumbered DAGs? | — | hash: **no** · IsalSR: yes | §5.4 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

**Numbers below are PLACEHOLDERS pending the host-native re-measurement.** The UDFS
figures quoted are from an exploratory local run on the corrected object and are
directionally reliable, but every one must be replaced by the Mode 1 / C2 value before
this reaches T14. Marked `\TODO` inline.

```latex
%% --- R1.4 ---
\begin{response}
We agree, and the comparison is now in the paper. We implement deduplication by
hashing a fixed-order serialisation of the candidate DAG: the graph is written out
in the order the host solver itself numbers its nodes, recording each node's label
and its operands in stored order, and the resulting string is hashed with the same
64-bit function used for the canonical string, so that the two arms differ only in
the equivalence relation and not in the hashing. This baseline runs as a full third
arm of the campaign, on the same problems, seeds, protocol and hardware as the other
two, and is reported throughout Section~\ref{sec:results} rather than as a footnote.

The comparison is best read as a decomposition rather than a ranking. A fixed-order
hash is \emph{sound} but \emph{incomplete}: it never merges two DAGs that are not
isomorphic, but it fails to merge isomorphic DAGs that differ in node numbering,
which is precisely the redundancy our representation targets. The observed reduction
therefore factors as $\rho_{\mathrm{total}} = \rho_{\mathrm{exact}} \times
\rho_{\mathrm{iso}}$, and we report both factors per method. For UDFS we measure
$\rho_{\mathrm{exact}} = \TODO{1.02}$ against $\rho_{\mathrm{total}} = \TODO{1.40}$,
so $\TODO{92.4}\%$ of the duplicate candidates encountered during search are
recoverable only by an isomorphism-complete invariant. This is the quantity the
comment asks for, and we consider reporting it more informative than asserting that
one method dominates the other.

We should be explicit about what the baseline is and is not. Hash-based
deduplication of expression structures is well established, from Ershov's original
common-subexpression collapsing~\cite{ershov1958} to modern hash-consing
formulations~\cite{filliatre2006}, and it has been applied specifically to genetic
programming for symbolic regression by Burlacu et al.~\cite{burlacu2019}, who hash
each node bottom-up from its children's hashes and sort those hashes for commutative
symbols. That construction is defined on \emph{trees}. Our candidates are directed
acyclic graphs in which a subexpression may be shared by several parents, and the
sharing pattern is part of the object being deduplicated; a tree hash either expands
the graph, at exponential cost, or discards the sharing that the representation
exists to express. The authors also describe their scheme as inexact. We therefore
compare against the fixed-order serialisation hash the comment names, and note the
tree-based construction here for completeness rather than adopting it.

We would add that the trade-off at issue is recognised in the genetic-programming
literature itself. Burke et al.~\cite{burke2004} observe that graph isomorphism
could serve as a diversity measure, but that ``the properties (associativity,
commutativity, etc.) would require special, and possibly complex, implementations of
isomorphism,'' and that ``determining graph isomorphism would be computationally
expensive for an entire population,'' concluding that cheap structural properties
are the practical substitute. Our contribution is the special implementation that
remark declines to construct, together with the demonstration that its cost is
affordable inside a live search.

Finally, on cost: the fixed-order hash is cheaper per candidate than
canonicalisation, and we report the per-DAG cost of both, measured on the campaign
hardware under the native engine (Table~\TODO{2}). We note also that our
completeness rests on Weisfeiler--Leman refinement \emph{together with} the
backtracking step over tied candidates; 1-WL refinement alone is not a complete
invariant for general graphs~\cite{cai1992}, and it is the combination of refinement
and individualisation that yields a canonical form, as in the design of practical
isomorphism solvers~\cite{mckay2014}.
\changeref{}
\end{response}
```

**Citations this answer requires** (verified 2026-07-31; add to the `.bib`):
`ershov1958` (CACM 1(8):3–6, DOI 10.1145/368892.368907 — **re-verify, publisher page
returned 403**), `filliatre2006` (ML '06:12–19, DOI 10.1145/1159876.1159880),
`burlacu2019` (LNCS 12013:361–369, DOI 10.1007/978-3-030-45093-9_44,
arXiv:2107.10640), `burke2004` (IEEE TEC 8(1):47–62, DOI 10.1109/TEVC.2003.819263),
`cai1992` (Combinatorica 12(4):389–410, DOI 10.1007/BF01305232), `mckay2014`
(JSC 60:94–112, DOI 10.1016/j.jsc.2013.09.003).

**Style check before T14**: this draft is ~430 words and should get a `humanizer`
scientific-mode pass. It currently uses "we" throughout, avoids significance
inflation, and quantifies its claims — but the Burke quotation needs its page number
(p. 48) in the final `\cite`.

### 8.4 Residual risk

> Candidates: a reviewer arguing our fixed orders were chosen to lose (mitigated by
> §4's three orders and by reporting the best); order (iii) approaching canonicality
> and blurring the contribution; whether the three-arm correction was applied
> correctly; whether ρ_iso is large enough on UDFS to carry the claim given UDFS's
> duplicates are commutative-symmetry duplicates.
