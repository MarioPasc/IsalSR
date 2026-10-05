#!/usr/bin/env bash
# =============================================================================
# SRBench black-box campaign (R3.2) -- array worker: a GATE, then the C2 worker.
# =============================================================================
# Every byte that runs a cell is C2's certified worker,
# slurm/c2_smoke/worker.sh (byte-identical to tag campaign/c2): local scratch,
# per-cell copy-back, the deadline rule (a cell starts only if its full budget
# fits), SIGTERM handling, `--ledger --postprocess skip`, PYTHONMALLOC=malloc.
# Re-implementing any of it here would be uncertified code (C2 README §1).
#
# This wrapper adds only what the black-box deployment needs:
#   1. the import shim: experiments/benchmarks from THIS tree, isalsr + engine
#      from the C2 tree's editable install (pyboot/sitecustomize.py);
#   2. a gate (check_env.py) that prints the three resolved module paths and
#      the engine build hash into this task's log BEFORE the first cell, and
#      refuses to run anything unless they are C2 / bbx / bbx / 298fc1188bf1b051
#      and the deployed files still hash as recorded in BBX_DEPLOY.json.
#
# No #SBATCH directives: launcher.sh supplies every resource flag.
#
# Environment (exported by launcher.sh), in addition to the C2 worker's own
# ISALSR_REPO_DIR / C2_* variables:
#   ISALSR_REPO_DIR  - the black-box tree ($FSCRATCH/repos/IsalSR_bbx); the C2
#                      worker cd's here and decodes cells with ITS c2_task_spec
#   BBX_C2_TREE      - the C2 tree the editable install points to
# =============================================================================
set -euo pipefail

BBX_ROOT="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
C2_TREE="${BBX_C2_TREE:?ERROR: BBX_C2_TREE not set}"

export ISALSR_BBX_ROOT="${BBX_ROOT}"
export PYTHONPATH="${BBX_ROOT}/slurm/blackbox/pyboot${PYTHONPATH:+:${PYTHONPATH}}"

# Same environment set-up as the C2 worker (it repeats it; harmless), needed
# here only so the gate runs under the interpreter the cells will use.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_PREFIX}/bin/python"

echo "=========================================="
echo "BBX gate | job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-1} | $(hostname)"
# cwd is irrelevant to the gate (the shim pins both packages to BBX_ROOT); run
# it from the tree anyway so relative paths in its output read naturally.
cd "${BBX_ROOT}"
if ! PYTHONMALLOC=malloc "${PYTHON}" "${BBX_ROOT}/slurm/blackbox/check_env.py" \
        --c2-tree "${C2_TREE}" --bbx-root "${BBX_ROOT}"; then
    echo "[FATAL] BBX gate failed; no cell was started." >&2
    exit 1
fi
echo "=========================================="

exec bash "${BBX_ROOT}/slurm/c2_smoke/worker.sh"
