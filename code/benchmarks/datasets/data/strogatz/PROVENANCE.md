# ODE-Strogatz vendored data — provenance

## What is here

Fourteen gzip'd TSV files, `strogatz_<key>.tsv.gz`, for the keys

```
bacres1 bacres2 barmag1 barmag2 glider1 glider2 lv1 lv2
predprey1 predprey2 shearflow1 shearflow2 vdp1 vdp2
```

Each file is one header line `target\tx\ty` followed by exactly 400 data rows
(verified 400x3, no NaN, no Inf). `x` and `y` are the two state variables of a
two-dimensional ODE system; `target` is the time derivative of one of them.

## Source

Copied byte-verbatim from PMLB (Penn Machine Learning Benchmarks),
`datasets/strogatz_<key>/strogatz_<key>.tsv.gz`. Nothing was resampled,
rounded, reordered or regenerated.

- Romano, Le, La Cava, Gregg, Goldberg, Ray, Imran, Fu, Moore (2021).
  *PMLB v1.0: an open source dataset collection for benchmarking machine
  learning methods.* Bioinformatics 38(3):878-880.
  <https://github.com/EpistasisLab/pmlb> — MIT licence.
- The datasets originate from the ODE-Strogatz repository
  (`ode-strogatz/simulate_ode.m`), which integrates fourteen systems from
  Strogatz, *Nonlinear Dynamics and Chaos*.
- They form the non-Feynman half of SRBench's ground-truth track:
  La Cava, Orzechowski, Burlacu, de Franca, Virgolin, Jin, Kommenda, Moore
  (2021). *Contemporary Symbolic Regression Methods and their Relative
  Performance.* NeurIPS Datasets and Benchmarks.

## Licence

MIT (PMLB). Redistribution inside this repository is permitted; the upstream
licence and attribution are reproduced by this file.

## Ground-truth equations

The fourteen equations transcribed in `benchmarks/datasets/strogatz.py` were
taken from `simulate_ode.m` and independently cross-checked against each PMLB
`metadata.yaml`. `shearflow1` is published as `cot(y)*cos(x)` and is written
`cos(y)*cos(x)/sin(y)` in the module, because Sigma_SR has no `cot` label and
supplies the reciprocal through `Inv`. No equation is otherwise simplified.

## Verification

`tests/unit/test_strogatz_benchmarks.py::test_target_fn_reproduces_published_target`
evaluates each `target_fn` on the published `(x, y)` columns and compares
against the published `target` column, `rtol=1e-6`, `atol=1e-8`.

Observed on 2026-08-02, all 14 problems, 400 rows each:

| quantity | value |
|---|---|
| max absolute error, `target_fn` vs published target | 1.95e-13 (Strogatz-glider2) |
| max absolute error, `sympy_expression` vs published target | 1.95e-13 (Strogatz-glider2) |
| negative control: same check with `x`/`y` swapped | min over the 14 problems of the max absolute error = 0.104 |

The negative control shows the test discriminates the column mapping: a
swapped mapping misses by ~1e-1, twelve orders of magnitude above the
tolerance.

## Split protocol

`generate_data` does not sample. It permutes the 400 published rows with
`np.random.default_rng(seed)` and takes the first 300 as train and the next
100 as test (SRBench's 75/25). `n_samples` and `train_ratio` are ignored,
because every problem carries `n_train_override=300` / `n_test_override=100`.
