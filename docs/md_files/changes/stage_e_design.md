# Stage E — the analysis dry-run, and the four defects it found

**Date**: 2026-08-05
**Plan reference**: `EXECUTION-PLAN.md` §4.5 (checks E1–E7)
**Input**: `c2_smoke_v4/` — 1,260 runs, 3 arms × 3 seeds, 60 fields, 420 paired-stat files
**Verdict**: **GO — 7/7 checks pass**, wall clock 181 s

---

## 1. Why the stage exists, and why it ran locally

The analysis pipeline had never been run on three arms. §4.5 calls discovering
that in September "the single most expensive failure mode left", because by then
the 100,800 core-hours are spent and the freeze is two weeks away.

Stage E runs **locally**, not on Picasso, for four reasons:

1. Every check that failed needed an **analyzer code fix**, and defect 10 in
   `T17-HANDOFF.md` forbids deploying to Picasso while an array is running —
   Stage D's 13 cells were live throughout.
2. `analyze.py`, `generate_tables.py` and the figure suite are the **local**
   step in September. Certifying them locally certifies the thing that will run.
3. E3, E6 and E7 mutate copies of the root; E4 needs `pdflatex`.
4. Every step is a soft probe. The full stage is **181 s**; the longest single
   step is the analyzer at 49 s.

**Projected cost at campaign scale.** The analyzer is 49 s on 1,260 runs, so
C2's 8,400 runs project to **≈5.5 min** — unlike the orchestrator's
`--postprocess only` aggregation, which took 1 h 35 m on the same input and is
projected at ≈11 h (§11.1, 2026-08-03). The two must not be confused when
sizing the campaign's analysis wall limit.

Nothing in Stage E writes to a campaign root, and the pristine mirror of
`c2_smoke_v4` is only ever read (SP-0).

---

## 2. What the checks found

| # | Verdict | Finding |
|---|---|---|
| E1 | PASS | 103 artefacts, all 6 families, all 3 arms present in the aggregates. 49 s |
| E2 | PASS | 3 contrasts × 5 metrics × 16 files; audit §6.1 policy holds on real data |
| E3 | PASS | Injected NaN never bolded; paired N 3→2; table discloses `[2]` |
| E4 | 🔴 **2 defects, fixed** | Tables emitted with exit 0 and **failed to compile** |
| E5 | 🔴 **1 defect, fixed** | CD diagrams silently dropped the hash arm |
| E6 | 🔴 **gap, implemented** | The analyzer never reconciled cells at all |
| E7 | 🔴 **gap, implemented** | The analyzer never checked provenance at all |

Three of the four are defects a clean run would never have surfaced: the
pipeline exited 0 in every case. That is the argument for E3/E6/E7 as
adversarial checks rather than smoke tests.

### 2.1 E4 — the D2 extension breaks LaTeX compilation

`generate_tables.py` typeset problem and suite identifiers raw. Every T05 D2
name carries an underscore — `strogatz_vdp1`, `liv_19`, `pagie_2`,
`feynman_remainder` — and a bare `_` outside math mode aborts `pdflatex`.

**18 rows per table across 4 tables**, i.e. precisely the rows the coverage
extension added. The generator exited 0; the failure lives one step later, in a
compile nobody had run on D2 data.

Three emission sites, all fixed via a shared `_latex_escape`:

| Site | Symptom |
|---|---|
| `_PROBLEM_LABELS.get(prob, prob)` (2 call sites) | `liv_19 & …` |
| `cpdt_label` CPDT footer (2 sites) | `CPDT (feynman_remainder)` |
| `bench_label = benchmark.capitalize()` (Table 1) | `Feynman_remainder` |

The fallback now routes through `_problem_label`, which prefers the curated D1
label and escapes anything else. Choosing *display names* for the D2 problems
(`Liv-19` rather than `liv\_19`) is an editorial decision and is left to Mario;
the fix here is strictly about compiling.

### 2.2 E5 — the hash arm vanished from every CD diagram

`generate_critical_difference.py` iterated a **hardcoded**
`["baseline", "isalsr"]` at both loader sites, with no `--variants` plumbing.
`cross_method.py` had been extended for three arms; the figure generator had
not. On the three-arm root it produced **4 groups** where 2 methods × 3 arms
gives 6 — the hash arm absent from every critical-difference figure, with
nothing in the output saying so.

This is the arm R1.4 asks about. Fixed by threading `variants` through both
loaders, all four public generators, `generate_all.py` and both CLIs; the arm
count is now logged and asserted. Verified: **6 groups on all 70 problems.**

### 2.3 E6 / E7 — two checks the analyzer did not implement

Neither existed. `reconcile()` lived in `status_ledger.py` and `analyze.py`
never called it; there was no provenance check of any kind.

New module `experiments/models/analyzer/completeness.py`:

- **E6.** `infer_expected_cells` builds the grid as the cross product of
  observed problems, seeds and root-wide arms — deliberately *not* "whatever is
  on disk", because a grid inferred from survivors defines a missing cell away.
  A test asserts the expectation is unchanged by a deletion. Missing cells are
  **named**, never merely counted.
  Note the ledger alone would not have caught this: deleting a `run_log.json`
  leaves `status.json` intact, so reconciliation must run over run logs.
- **E7.** Root-wide keys (`git_describe`, `git_dirty`, `build_hash`) must be
  single-valued; `config_sha256` must be single-valued *per (method, suite)*,
  since it legitimately differs between suites.

Both fail closed: `--allow-incomplete` and `--allow-mixed-provenance` are the
only ways through, and the analyzer exits **2** otherwise.

---

## 3. Two findings worth carrying beyond Stage E

### 3.1 `git_commit` is `None` on all 1,260 runs — an SP-6 trap in waiting

A7 lists `git_commit` among the provenance fields, but `collect_hardware_info()`
never populates it. A provenance guard keyed on it would have seen **one value
on every run of every campaign** and passed vacuously — the SP-6 pattern
verbatim, where a zero-everywhere ledger means the counters are dead rather than
the rates zero.

The guard therefore keys on `git_describe`/`git_dirty`/`build_hash`, and reports
keys that are absent on every run as **non-informative** rather than as
agreement. A regression test locks this behaviour.

### 3.2 The guard independently confirms v4's dirty split — and would block C2

Run against the real mirror with no overrides, the analyzer **refuses**
`c2_smoke_v4`:

```
git_describe: 2 distinct values across the root -- 'a455d6c' x1099, 'a455d6c-dirty' x161
git_dirty:    2 distinct values across the root -- False x1099, True x161
```

That is the 161-cell split recorded in §11.1 (2026-08-04), rediscovered from the
data alone. Two consequences:

1. Stage E's E1 must pass `--allow-mixed-provenance`, and does so explicitly,
   because v4 is a known-heterogeneous smoke root.
2. **The owed clean Stage C wave on `00635ae` is now enforced by code.** Under
   §5.1 the campaign must launch on a commit *and* configuration a Stage C wave
   has certified; the analyzer will now refuse a campaign root that is not.
   `config_sha256` was clean at 1 value per (method, benchmark) across all 14,
   so the split is confined to the commit.

---

## 4. What landed

| File | Role |
|---|---|
| `experiments/models/analyzer/completeness.py` | **New.** E6 + E7 core; fails closed |
| `experiments/models/analyze.py` | Integrity gate before any statistic; `--allow-incomplete`, `--allow-mixed-provenance`; exit 2; writes `analysis/campaign_integrity.json` |
| `experiments/figures/models/generate_critical_difference.py` | E5: arms are a parameter; `--variants`; group count logged |
| `experiments/figures/models/generate_all.py` | `--variants`, threaded to the CD generators |
| `experiments/figures/models/generate_tables.py` | E4: `_latex_escape` + `_problem_label`, three emission sites |
| `experiments/scripts/stage_e_certify.py` | **New.** The E1–E7 certifier and its fixtures |
| `tests/unit/test_analyzer_completeness.py` | **New.** 16 tests |
| `tests/unit/test_stage_e_figures_tables.py` | **New.** 16 tests |

Fixtures use a two-suite subset (nguyen + feynman, 396 runs) so each adversarial
run stays inside the soft-probe budget while still crossing both methods and all
three arms. E1/E2/E4/E5 run on the full seven-suite root.

**Verification.** `tests/unit` **6,929 passed, 5 skipped**; the only 2 failures
in the tree are `test_appendix_d_generator.py`, the T09 agent's *untracked*
work — verified to have zero import coupling to any file touched here, and left
alone. `ruff src/ tests/` clean; `mypy --strict src/isalsr/` clean;
`experiments/` ruff count **24 → 22 on the four modified files**, so zero
introduced and two removed.

---

## 5. Re-running it

```bash
python -m experiments.scripts.stage_e_certify \
  --smoke-root /media/.../real_benchmarks/c2_smoke_v4 \
  --work-dir   /media/.../real_benchmarks/c2_stage_e \
  --reuse-main            # skip the 572 MB working-copy rebuild
  --only E3,E6            # optional: a subset of checks
```

Artefacts land in `c2_stage_e/artefacts/`:
`stage_e_certification.{json,md}`, `tables/`, `figures/`, `latex_build/`,
`logs/`. Exit 0 iff every requested check passes.

**Stage E must be re-run on the v5 root** once the clean Stage C wave lands on
`00635ae`. It should then pass E7 **without** `--allow-mixed-provenance`; if it
does not, the wave was not clean and the `campaign/c2` tag must not be cut.
