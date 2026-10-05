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
# This wrapper adds only a gate (check_env.py) that prints the three resolved
# module paths, the engine build hash, the interpreter and the library versions
# into this task's log BEFORE the first cell, and refuses to run anything unless
# they are the deployed tree / 298fc1188bf1b051 / C2's python / the pinned
# versions, and the deployed files still hash as recorded in BBX_DEPLOY.json.
#
# Layout (T01b): $FSCRATCH/repos/IsalSR is the whole repository, the project is
# under code/, and the env's editable install points at it.  No import shim.
#
# No #SBATCH directives: launcher.sh supplies every resource flag.
#
# Environment (exported by launcher.sh), in addition to the C2 worker's own
# C2_* variables:
#   ISALSR_REPO_DIR  - the deployed tree's code/ directory; the C2 worker cd's
#                      here, decodes cells with ITS c2_task_spec and reads SP-1
#                      from the enclosing git repository
# =============================================================================
set -euo pipefail

CODE_ROOT="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"

# Same environment set-up as the C2 worker (it repeats it; harmless), needed
# here so the gate runs under the interpreter, modules and PYTHONPATH the cells
# will use.
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
cd "${CODE_ROOT}"
if ! PYTHONMALLOC=malloc PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" "${CODE_ROOT}/slurm/blackbox/check_env.py" --code-root "${CODE_ROOT}"; then
    echo "[FATAL] BBX gate failed; no cell was started." >&2
    exit 1
fi
echo "=========================================="

exec bash "${CODE_ROOT}/slurm/c2_smoke/worker.sh"
