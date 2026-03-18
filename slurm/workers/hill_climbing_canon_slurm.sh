#!/usr/bin/env bash
# SLURM compute worker: hill_climbing_canon (ARRAY JOB)
# Each array task runs ONE independent experiment.
# Submit with: sbatch --array=1-30 hill_climbing_canon_slurm.sh
set -euo pipefail

echo "=== hill_climbing_canon: SLURM Array Task ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
BENCH_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['hill_climbing_canon'], __import__('sys').stdout)")

SEED_BASE=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed_base', 42))")
OUT_DIR="${RESULTS_DIR}/hill_climbing_canon"

mkdir -p "$OUT_DIR"

RUN_ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED=$((SEED_BASE + RUN_ID - 1))

echo "Run ${RUN_ID} (seed=${SEED})"
echo "Output: $OUT_DIR"

python experiments/scripts/run_hill_climbing.py \
    --n-iterations 1000 --n-restarts 10 --max-tokens 30 \
    --seed "$SEED" \
    --run-id "$RUN_ID" \
    --output-dir "$OUT_DIR"

echo ""
echo "=== Run ${RUN_ID} complete: $(date) ==="
