#!/usr/bin/env bash
# SLURM compute worker: hill_climbing_nocanon
# Runs 30 independent experiments (seeds 42..71) across all Nguyen benchmarks.
set -euo pipefail

echo "=== hill_climbing_nocanon: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
BENCH_CFG=$(python3 -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['hill_climbing_nocanon'], __import__('sys').stdout)")

N_RUNS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_runs', 30))")
SEED_BASE=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed_base', 42))")
OUT_DIR="${RESULTS_DIR}/hill_climbing_nocanon"

mkdir -p "$OUT_DIR"
echo "Config: n_runs=$N_RUNS, seed_base=$SEED_BASE"
echo "Output: $OUT_DIR"

# Run 30 independent experiments sequentially within this job
for RUN_ID in $(seq 1 $N_RUNS); do
    SEED=$((SEED_BASE + RUN_ID - 1))
    echo ""
    echo "--- Run ${RUN_ID}/${N_RUNS} (seed=${SEED}) ---"
    python experiments/scripts/run_hill_climbing.py \
        --n-iterations 1000 --n-restarts 10 --max-tokens 30 \
        --seed "$SEED" \
        --run-id "$RUN_ID" \
        --output-dir "$OUT_DIR" \
        --no-canon \
        || echo "WARNING: Run $RUN_ID failed (seed=$SEED)"
done

echo ""
echo "=== All runs complete: $(date) ==="
