#!/usr/bin/env bash
# =============================================================================
# SLURM post-processing: Diversity Conjecture figure generation
# =============================================================================
# Runs after all diversity_experiment array tasks complete.
# Generates the multi-panel figure, statistical tests, and summary JSON.
#
set -euo pipefail

echo "=== Diversity Analysis: Figure Generation ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

CONFIG="${DIVERSITY_CONFIG:-${REPO_DIR}/slurm/diversity_config.yaml}"
RESULTS_DIR="${DIVERSITY_RESULTS_DIR:-$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")}"

echo "Results dir: $RESULTS_DIR"
echo ""

$PYTHON -m experiments.scripts.diversity.generate_fig_diversity \
    --input-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Analysis complete: $(date) ==="
