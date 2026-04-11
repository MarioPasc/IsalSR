#!/bin/bash
#SBATCH --job-name=isalsr_synth_k10
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=results/synthetic_scalability/logs/synth_k10p_%A_%a.log
#SBATCH --error=results/synthetic_scalability/logs/synth_k10p_%A_%a.err

# Synthetic Scalability — k=10+ tasks only (sampled permutations for timing)
# 24 tasks = 8 k-values x 3 m-values.
#
# k=1..9 ran exhaustively and showed rho = k! for 100% of expressions.
# For k=10+, we use 5,000 sampled permutations to measure the timing
# curve at polynomial cost (each task finishes in minutes).
#
# Submit: sbatch experiments/synthetic_scalability/slurm_synthetic_k10plus.sh

set -euo pipefail

# --- Environment ---
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-.}"

# --- Grid mapping ---
K_VALUES=(10 11 12 14 16 18 20 25)
M_VALUES=(1 2 3)

K_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
M_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

K=${K_VALUES[$K_IDX]}
M=${M_VALUES[$M_IDX]}

RESULTS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr"
OUT_DIR="${RESULTS_DIR:-results}/synthetic_scalability"
mkdir -p "$OUT_DIR/logs"

echo "Task ${SLURM_ARRAY_TASK_ID}: k=${K}, m=${M}, output=${OUT_DIR}"

# --- Run ---
# 5000 sampled perms: enough for stable timing stats, finishes in minutes.
# Invariance (n_unique=1) is still verified on these 5000 samples.
PYTHONHASHSEED=0 python experiments/synthetic_scalability/run_synthetic_scalability.py \
    --output-dir "$OUT_DIR" \
    --seed 42 \
    --n-expr 200 \
    --max-perms 5000 \
    --timeout 120 \
    --k-values "$K" \
    --m-values "$M"

echo "Task ${SLURM_ARRAY_TASK_ID} (k=${K}, m=${M}) completed."
