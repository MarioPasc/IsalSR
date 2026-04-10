#!/bin/bash
#SBATCH --job-name=isalsr_synth
#SBATCH --array=0-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=results/synthetic_scalability/logs/synth_%A_%a.log
#SBATCH --error=results/synthetic_scalability/logs/synth_%A_%a.err

# Synthetic Scalability Experiment -- SLURM Array Job
# 51 tasks = 17 k-values x 3 m-values.
# Each task processes one (k, m) cell and writes a fragment CSV.
#
# Submit: sbatch experiments/synthetic_scalability/slurm_synthetic.sh
# Merge:  head -1 results/synthetic_scalability/synth_k1_m1.csv > results/synthetic_scalability/synthetic_scalability_results.csv
#         tail -n +2 -q results/synthetic_scalability/synth_k*_m*.csv >> results/synthetic_scalability/synthetic_scalability_results.csv

set -euo pipefail

# --- Environment ---
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-.}"

# --- Grid mapping ---
K_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 25)
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
# PYTHONHASHSEED=0 ensures deterministic WL hash ordering in
# fast_canonical_string, making canonical_len reproducible across runs.
PYTHONHASHSEED=0 python experiments/synthetic_scalability/run_synthetic_scalability.py \
    --output-dir "$OUT_DIR" \
    --seed 42 \
    --n-expr 200 \
    --max-perms 50000000 \
    --timeout 120 \
    --k-values "$K" \
    --m-values "$M"

echo "Task ${SLURM_ARRAY_TASK_ID} (k=${K}, m=${M}) completed."
