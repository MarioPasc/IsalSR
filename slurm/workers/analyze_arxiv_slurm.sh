#!/usr/bin/env bash
# SLURM compute worker: Aggregate arXiv experiment results
set -euo pipefail

echo "=== arXiv Results Analysis - SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")

echo "Results dir: $RESULTS_DIR"

python experiments/scripts/analyze_arxiv_results.py \
    --results-dir "$RESULTS_DIR"

echo "Finished: $(date)"
