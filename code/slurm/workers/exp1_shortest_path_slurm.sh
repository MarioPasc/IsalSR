#!/usr/bin/env bash
# SLURM compute worker: Shortest Path Between DAGs (Experiment 1)
# Fast illustrative experiment (~10 seconds).
set -euo pipefail

echo "=== Experiment 1: Shortest Path - SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"
RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")

OUT_DIR="${RESULTS_DIR}/exp1_shortest_path"
mkdir -p "$OUT_DIR"

echo "Output: $OUT_DIR"

python experiments/scripts/exp1_shortest_path.py \
    --output-dir "$OUT_DIR"

echo "Finished: $(date)"
