#!/usr/bin/env bash
# T16 alphabet certification gate (G9) — Picasso worker.
#
# Proves that a run executing on a COMPUTE NODE, through the real orchestrator and
# the real production configs, canonicalises the paper's alphabet: no Sub, no Div,
# and Pow as the only order-sensitive binary operation.
#
# CPU-only: no --gres, no GPU constraint.
set -euo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Repo:         ${REPO_DIR}"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT — absolute interpreter, because `conda activate isalsr` does not
# take on this cluster (it leaves the base py39 python on PATH).
# ============================================================================
PY="${CONDA_PREFIX_ABS}/bin/python"
[[ -x "${PY}" ]] || { echo "[FATAL] interpreter not found: ${PY}"; exit 1; }

# LOAD-BEARING. bingo-nasa imports mpi4py at module scope
# (bingo/util/log.py:14), and mpi4py dlopen()s libmpi at import time. libmpi is
# NOT in the conda env, so without an MPI module the import dies with
# "RuntimeError: cannot load MPI library" and the gate reports FAIL for reasons
# that have nothing to do with the alphabet. Probe the same way the production
# workers do (slurm/workers/models_analyze_slurm.sh), but with module names that
# exist on this cluster -- verified against `module avail` on 2026-07-30.
mpi_loaded=0
for mod in openmpi_gcc/4.1.5_gcc1520 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc7 openmpi_gcc/3.1.4_gcc750; do
    if module load "$mod" 2>/dev/null; then
        echo "[env] MPI module loaded: $mod"
        mpi_loaded=1
        break
    fi
done
[[ "${mpi_loaded}" -eq 0 ]] && echo "[env] WARNING: no MPI module loaded; bingo import will likely fail."

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# LOAD-BEARING. bingo pulls in mpi4py, which dlopen()s libmpi.so.40 from the
# conda-installed OpenMPI. Without this the import fails on the compute node with
# "libmpi.so.40: cannot open shared object file" even though the login node is
# fine. Matches slurm/workers/*.sh, which all set the same thing off CONDA_PREFIX.
export LD_LIBRARY_PATH="${CONDA_PREFIX_ABS}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[env] python: $(${PY} -V 2>&1)"

# ============================================================================
# BACKEND CHECK — a silent pure-Python fallback would invalidate the whole run.
# ============================================================================
${PY} - <<'PYEOF'
import sys
try:
    from isalsr.core import _native
    print(f"[backend] NATIVE C++ extension loaded: {_native.__file__}")
except Exception as exc:  # noqa: BLE001
    print(f"[backend] FATAL: native extension did NOT load: {exc}")
    sys.exit(1)
from experiments.models.commutative_encoding import DECOMPOSITION, SHARE_DECOMPOSED_UNARY
print(f"[t16] decomposes: {sorted(k.name for k in DECOMPOSITION)}")
print(f"[t16] SHARE_DECOMPOSED_UNARY: {SHARE_DECOMPOSED_UNARY}")
PYEOF

# ============================================================================
# SMOKE CONFIGS — derived from the production configs, shortening ONLY the time
# budget and population. Operator sets are copied verbatim: they are the thing
# under test, and editing them would invalidate the gate.
# ============================================================================
CFG_DIR="${WORK_DIR}/g9cfg"
mkdir -p "${CFG_DIR}" "${WORK_DIR}/g9out"

${PY} - <<PYEOF
import pathlib, yaml
src_dir = pathlib.Path("${REPO_DIR}/experiments/configs")
out_dir = pathlib.Path("${CFG_DIR}")
for name in "${GATE_CONFIGS}".split():
    d = yaml.safe_load((src_dir / f"{name}.yaml").read_text())
    for section in d.values():
        if isinstance(section, dict):
            if "max_time" in section:
                section["max_time"] = int("${GATE_MAX_TIME}")
            if "population_size" in section:
                section["population_size"] = min(section["population_size"], 60)
    d.setdefault("output", {})["base_dir"] = "${WORK_DIR}/g9out"
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    ops = next(
        (s.get("operators") or s.get("operator_set")
         for s in d.values()
         if isinstance(s, dict) and (s.get("operators") or s.get("operator_set"))),
        None,
    )
    print(f"[cfg] {name}: operators={ops}")
PYEOF

# ============================================================================
# THE GATE
# ============================================================================
RC=0
for pair in ${GATE_PAIRS}; do
    cfg="${pair%%:*}"
    prob="${pair##*:}"
    echo ""
    echo "############ G9: ${cfg} / ${prob} ############"
    if ${PY} -m experiments.scripts.verify_alphabet_gate \
            --config "${CFG_DIR}/${cfg}.yaml" \
            --problems "${prob}" \
            --seeds 1 \
            --output-dir "${WORK_DIR}/g9out/${cfg}" \
            --json-out "${WORK_DIR}/g9_${cfg}.json"; then
        echo "[gate] ${cfg}: PASS"
    else
        echo "[gate] ${cfg}: FAIL"
        RC=1
    fi
done

echo ""
echo "=========================================="
if [[ ${RC} -eq 0 ]]; then
    echo "G9 OVERALL: PASS — Wave 1 may launch against this commit."
else
    echo "G9 OVERALL: FAIL — DO NOT LAUNCH WAVE 1."
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "=========================================="
exit ${RC}
