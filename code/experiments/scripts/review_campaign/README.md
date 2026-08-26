# `review_campaign` — analysis of the three-arm revision campaign

Everything the revised TPAMI manuscript, its supplementary material and the
response letter report about the three-arm campaign is produced here. One
command rebuilds the lot:

```bash
bash experiments/scripts/review_campaign/run_all.sh [CORPUS] [ANALYSES]
```

`CORPUS` defaults to the local backup of the campaign
(`…/results/review/c2_3arm`) and `ANALYSES` to the `analyses/` tree beside it.
Outputs live with the data; only code lives here.

## Pipeline

| Step | Module | Produces |
|---|---|---|
| 1 | `run_all.sh` (inline) | Two symlink views of the corpus: all 70 problems under one benchmark, and the 50 that predate the revision |
| 2 | `experiments.models.analyze` | Paired statistics and the paired test across problems, on both views and per suite |
| 3 | `extract_cells` | `data/cells.csv`, one row per (method, arm, problem, seed) — 12,600 rows |
| 4 | `derive` | Per-problem aggregates, the share phi, seed-matched speedups, k-strata, the flattened paired tests, `values/summary.json` |
| 5 | `values` | `values/pending_values.{json,md}`: one row per placeholder of the two pending ledgers |
| 6 | `tables` | Every `tabular` the two documents and the letter `\input` |
| 7 | `figures`, `cd_diagram` | The reduction-factor distribution, the reduction-versus-overhead plot, and one critical-difference diagram per host with a shared legend |
| 8 | `verify` | Asserts every quoted literal against the derived value; exits non-zero on drift |

Nothing writes into the corpus. The 420 `aggregate.csv` and 420
`paired_stats*.json` files it already carries are read and never recomputed, so
the certified digests of the campaign root stay valid.

## Decisions worth knowing before reading a number

**`--allow-mixed-provenance` is required and correct.** Flattening the seven
per-suite directories into one benchmark pools seven configuration digests per
host, which differ in the benchmark block and, on three suites, in the host
operator set. Commit, build hash and engine are uniform across all 12,600 cells,
which is the provenance guarantee that matters.

**Per-evaluation cost is never read from the native arm.** On Bingo that arm
counts candidates through a different mechanism and reports about nine times as
many of them. The two deduplicating arms agree closely on both hosts, which is
what makes the quantity trustworthy.

**The reduction factor has no p-value against the native arm.** ρ is 1 there by
construction, so the analyzer marks that contrast
`descriptive_definitional_baseline` and reports it as a count. The inferential
statement is IsalSR against the naive hash.

**φ is a cross-arm estimate.** It is computed from the two reduction factors
through the scale-free identity φ = 1 − r_σ/r, taking r from the IsalSR arm and
r_σ from the naive hash arm of the same problem and seed. The campaign ran with
the same-stream shadow sketches off; the manuscript states the resulting
trajectory-divergence confound rather than hiding it.

**The critical-difference diagram is per host.** Pooling both hosts lets the
difference between the solvers absorb the axis and widens the Nemenyi threshold
from 0.40 to 0.90 at N = 70.

## A correctness fix this work carried

`experiments/models/analyzer/statistical_tests.py` computed the Nemenyi critical
difference from the raw studentized range. Demšar (2006, §3.2.2) defines it from
the studentized range **divided by √2**, and the divided values reproduce his
Table 5 to three decimals for every k from 2 to 10. The threshold was therefore
41 % too wide, which makes a diagram declare differences indistinguishable that
the test separates. Fixed, with `tests/unit/test_nemenyi_critical_difference.py`
pinning the constant against the published table.
