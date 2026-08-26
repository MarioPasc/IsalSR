#!/usr/bin/env bash
# =============================================================================
# C4 cross-architecture reproducibility probe -- worker
# =============================================================================
# Environment block copied from aggregate_worker.sh: the mpi4py module probe and
# PYTHONMALLOC are load-bearing (mpi4py dlopen()s libmpi at *import*, so a job
# that never touches MPI still dies in ~13 s without it).
# =============================================================================
set -euo pipefail

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
OUT="${C4_OUT:?ERROR: C4_OUT not set}"

for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONMALLOC=malloc

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

echo "=========================================="
echo "C4 arch probe | job ${SLURM_JOB_ID:-local}"
echo "Node:  $(hostname)"
echo "CPU:   $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "Flags: avx512f=$(grep -qm1 avx512f /proc/cpuinfo && echo 1 || echo 0)"
echo "=========================================="

"${PYTHON}" -m experiments.scripts.c4_arch_reproducibility --out "${OUT}" --seed 0
echo "done: ${OUT}"
