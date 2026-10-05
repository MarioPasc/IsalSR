# SRBench black-box vendored data: provenance

## What is here

Twenty gzip'd TSV files, `<name>.tsv.gz`, plus `manifest.csv`:

```
1027_ESL 1029_LEV 1030_ERA 1096_FacultySalaries 192_vineyard 210_cloud
228_elusage 485_analcatdata_vehicle 519_vinnie 523_analcatdata_neavote
529_pollen 556_analcatdata_apnea2 557_analcatdata_apnea1 663_rabe_266
678_visualizing_environmental 687_sleuth_ex1605 690_visualizing_galaxy
712_chscase_geyser1 banana titanic
```

Each file has one header line and `n` data rows. The column `target` is the
response; every other column is a feature, used in file order. The datasets
range over n in [48, 5300] and d in [2, 5]. Together the files take 190,955
bytes on disk. `manifest.csv` gives, per file: n, d, the feature-type counts,
the feature names, the SHA-256, the size, the PMLB commit and the source URL.

## Source

The files were copied byte-verbatim from PMLB (Penn Machine Learning Benchmarks),
`datasets/<name>/<name>.tsv.gz`, at **commit
`8eec4f9d1578c7ff1cbfa3efd8338a301adebe52`** (tag `v1.0.1.post3`, 2020-09-10).
Nothing was resampled, rounded, reordered or regenerated. Download URL pattern:
`https://github.com/EpistasisLab/pmlb/raw/8eec4f9d1578c7ff1cbfa3efd8338a301adebe52/datasets/<name>/<name>.tsv.gz`.

- Romano, Le, La Cava, Gregg, Goldberg, Ray, Imran, Fu, Moore (2021).
  *PMLB v1.0: an open source dataset collection for benchmarking machine
  learning methods.* Bioinformatics 38(3):878-880.
  <https://github.com/EpistasisLab/pmlb>, MIT licence.
- La Cava, Orzechowski, Burlacu, de Franca, Virgolin, Jin, Kommenda, Moore
  (2021). *Contemporary Symbolic Regression Methods and their Relative
  Performance.* NeurIPS Datasets and Benchmarks. SRBench's black-box track is
  the set of PMLB regression datasets that have no ground-truth model. SRBench
  reports the synthetic Friedman datasets as a separate stratum. Each of its 10
  trials uses "a different random state that controlled both the train/test
  split and the seed of the algorithm" (as quoted in the revision plan, D7).

## Why this PMLB commit

SRBench v2.0 (tag `v2.0`, commit `e5ded4715ed5721703353d3500a2fdb99004faf1`,
2021-10-29) cloned PMLB without pinning a commit (README: `git clone
https://github.com/EpistasisLab/pmlb/`). Its `environment.yml` pins the PMLB
package at `pmlb==1.0.1.post3`. Its black-box results file
(`results/black-box_results.feather`) was committed on 2021-06-11 and updated on
2021-07-01. The pin is therefore the PMLB release SRBench declares, and the
following was checked against the LFS object IDs of all 20 files at six PMLB
commits:

| PMLB commit | date | role |
|---|---|---|
| `8eec4f9` | 2020-09-10 | tag v1.0.1.post3, **the pin** |
| `5e3d1bb` | 2020-10-09 | first commit after the pin that touches these files |
| `3bf620d` | 2021-05-28 | last commit touching `datasets/` before SRBench's results were committed |
| `1c72343` | 2021-09-16 | last commit touching `datasets/` before the SRBench v2.0 tag |
| `8df469e` | 2022-10-06 | "Banana dataset is now a classification problem" |
| `master` (`7c1f4bd`) | 2025-02-25 | current |

Findings:

- **18 of 20 files are byte-identical at every commit from the pin through the
  v2.0 tag**, so any PMLB clone SRBench could have used holds these bytes.
- `519_vinnie` changed bytes on 2020-09-29/2020-10-05 ("encode feature to int64";
  `85.0` became `85`). The table is **numerically identical**: `np.array_equal`
  holds over all 380 x 3 values, in the same row order. Manifest drift:
  `format-only`.
- `banana` is identical from the pin through the v2.0 tag. PMLB re-encoded it
  as a classification problem on 2022-10-06 (`8df469e`), after SRBench v2.0.
  The vendored file is the regression version SRBench ran. Manifest drift
  versus master: `content`.
- `titanic` was **replaced by a different dataset** upstream on 2022-09-30
  (commit `807d0d7`, "Replace titanic dataset (#165)"). The current master
  file has 2207 rows x 8 features with missing values and is a classification
  task. The file SRBench ran, which is the one vendored here, has 2201 rows x
  3 features (`Class`, `Age`, `Sex`). Its last change before the replacement
  was on 2020-08-21, before the pin. Manifest drift versus master: `content`.
  This is why the revision plan's first count, which used PMLB master summary
  statistics, gave 19 datasets instead of 20: under master's statistics titanic
  has 8 features.

## Selection

`experiments/scripts/blackbox/select_datasets.py` applies the pre-registered
rule (revision plan, decision D6):

1. Population: the unique `dataset` values of SRBench's
   `results/black-box_results.feather` at tag v2.0. The file has SHA-256
   `b29392e5b2104266e40bf4330323a57b5aa91642975b0ceac8181b95f95960ec` and holds
   122 datasets.
2. Drop the 62 datasets flagged `friedman_dataset`.
3. Keep `n_features <= 5`, read from PMLB `pmlb/all_summary_stats.tsv` at the
   pinned commit.

The script reproduces exactly the 20 names in
`benchmarks/datasets/srbench_blackbox.py`.

## Feature types

`n_binary`, `n_categorical` and `n_continuous` come from PMLB's summary
statistics **at the pinned commit**. PMLB later curated its feature-type labels.
For example, `1027_ESL` went from 4 continuous to 4 categorical. The curated
counts from master (`7c1f4bd`) are recorded in the `*_curated` columns, **only
where the master file still holds the vendored table** (drift `none` or
`format-only`). They are left blank for `banana` and `titanic`, whose current
statistics describe different files.

## Verification

Done by `select_datasets.py` at vendoring time (2026-10-05):

| check | result |
|---|---|
| rows x features vs PMLB summary stats at the pin | 20/20 equal |
| rows x features vs the table in `srbench_blackbox.py` | 20/20 equal |
| feature order vs SRBench v2.0 `read_file` (pandas, sniffed separator) | 20/20 equal |
| values: this loader vs SRBench `read_file` | 12/20 bit-identical; max relative difference 1.45e-13 (`banana`) |

This repository's loader (`np.loadtxt`) parses each decimal string to the
nearest double and agrees bit-for-bit with Python's `float()`. pandas' parser
is not always correctly rounded, so SRBench's arrays can differ in the last
bits. The per-file maximum is the manifest column `srbench_parse_max_rel_diff`.

Offline, every load checks the file's SHA-256 against `manifest.csv`
(`srbench_blackbox.load_published`), and
`tests/unit/test_srbench_blackbox.py` checks the manifest, the hashes and the
shapes.

## Licence

MIT (PMLB). Redistribution inside this repository is permitted. This file
reproduces the upstream licence and attribution.
