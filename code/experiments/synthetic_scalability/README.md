# Synthetic Scalability Experiment

Measures the theoretical search-space reduction factor (rho) and
canonicalization wall-clock time as a function of DAG size (k internal
nodes), using controlled synthetic expressions.

## Method

Random expression DAGs are generated via the Lample-Charton (2020)
random unary-binary tree method with operators drawn uniformly from
{+, *, ^, sin, cos, exp, log, neg, inv}. For each expression, all k!
(or up to 5000 sampled) internal-node permutations are canonicalized
with `fast_canonical_string(mode="wl_only")`.

## Local run (quick validation)

```bash
PYTHONHASHSEED=0 python experiments/synthetic_scalability/run_synthetic_scalability.py \
    --output-dir /tmp/synth_test \
    --n-expr 5 --max-perms 50 \
    --k-values "1,2,3,4" --m-values "1,2"
```

**Note:** `PYTHONHASHSEED=0` ensures deterministic WL hash ordering in
`fast_canonical_string`, making `canonical_len` reproducible across runs.
Without it, rho and all structural columns are still correct and
reproducible; only the specific canonical string (and its length) may vary.

## SLURM (full experiment)

```bash
sbatch experiments/synthetic_scalability/slurm_synthetic.sh
```

33 array tasks (11 k-values x 3 m-values). Each writes a fragment CSV.
Merge fragments:

```bash
OUT=results/synthetic_scalability
head -1 "$OUT/synth_k1_m1.csv" > "$OUT/synthetic_scalability_results.csv"
tail -n +2 -q "$OUT"/synth_k*_m*.csv >> "$OUT/synthetic_scalability_results.csv"
```

## Output

- `synthetic_scalability_results.csv` -- one row per expression
- `synthetic_scalability_metadata.json` -- operator set, params, hardware
