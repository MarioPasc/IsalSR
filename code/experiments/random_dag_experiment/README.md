# Random DAG Experiment — preview generator

Purpose: visualise the **boundary cells** of the proposed random-DAG factor
grid before committing Picasso budget to a full sweep. Generates one
representative DAG per cell of a 16-cell screening design over

| Factor      | Levels                                  | Rationale                                               |
|-------------|-----------------------------------------|---------------------------------------------------------|
| k           | {6, 9, 12, 15}                          | spans saturation threshold (≥7) and IsalSR sweet spot   |
| m           | {1, 2, 4}                               | controls VAR-sharing density                            |
| op_set      | {poly, poly_trig, full}                 | direct test of the polynomial-only justification        |

Constants are excluded by construction (no CONST nodes inserted): we are
isolating *structural* search, per the bottleneck analysis 2026-04-19.

## Operator subsets

- **poly**:      ADD, MUL, NEG          (pure polynomial — supports
  the "infinite arity via repeated MUL" argument)
- **poly_trig**: poly + SIN, COS        (Taylor-extension, falsifies
  the polynomial-only claim if IsalSR still helps)
- **full**:      poly + SIN, COS, EXP, LOG, INV (mirrors the production
  alphabet)

## Files

```
experiments/random_dag_experiment/
├── README.md                 — this file
├── generate_dags.py          — main script
└── outputs/                  — created at runtime
    ├── manifest.json         — full grid + per-cell metrics
    ├── dag_metadata.csv      — flat metrics table
    ├── gallery.md            — one-page review document with all PNGs
    └── figures/              — one PNG per generated DAG
```

## Run

```bash
conda activate isalsr
python -m experiments.random_dag_experiment.generate_dags \
    --output-dir experiments/random_dag_experiment/outputs \
    --seed 42
```

## Metrics reported per DAG

- `k`, `m`, `op_set`           — design point
- `n_total_nodes`, `n_edges`   — DAG size
- `n_vars_used`                — distinct VARs that have ≥1 outgoing edge
- `max_in_degree`              — controls "infinite-arity" lever
- `depth`                      — longest path length
- `n_unique_canon`             — distinct canonical strings over a sampled
                                 permutation set (proxy for k!/|Aut|)
- `estimated_RF`               — `n_perms / n_unique_canon` (rough |Aut|⁻¹)

These metrics *predict* where IsalSR should win:
high `n_unique_canon` ≈ k! ⇒ low automorphism count ⇒ high redundancy
under random-permutation isomorphism (the RF claim).
