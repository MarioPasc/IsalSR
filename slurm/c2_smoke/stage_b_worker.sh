#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage B micro-job worker (EXECUTION-PLAN.md §4.2)
# =============================================================================
# Runs, on ONE compute node, for ONE host (udfs | bingo):
#
#   B1  environment probe + all 70 D1uD2 dataset paths resolved with the
#       declared shapes, and a sympy ground truth on every one (gates C1.5)
#   B2  C++ capability probe WITH A NEGATIVE CONTROL -- sp_probe asserts on
#       OBSERVED DISPATCH (a spy on _cpp_ext.fast_canonical_string), then the
#       whole probe is re-run under ISALSR_ENGINE=python and must report
#       `python` AND not call C++.  A probe that says `native` in both
#       directions proves nothing; one that says `python` while running C++ is
#       worse, and that is exactly the defect that existed until 2026-07-31.
#   B3  alphabet gate on the frozen commit: 0 Sub, 0 Div, 0 '-', 0 '/'
#   B9  T06 counter overhead re-measured under the C++ engine and the
#       decomposed alphabet, both of which changed underneath T06's original
#       measurement.  Designed so a LIVE counter is distinguishable from a dead
#       one: 240 s of real search, then assert n_ledger_sampled > 0.
#
# B4 (equivalence gate on a compute node) is host-independent and runs as its
# own job -- see stage_b_launcher.sh.
#
# Environment (exported by stage_b_launcher.sh):
#   ISALSR_REPO_DIR, C2_METHOD, C2_EVIDENCE_DIR
# =============================================================================
set -uo pipefail

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
METHOD="${C2_METHOD:?ERROR: C2_METHOD not set}"
EVIDENCE="${C2_EVIDENCE_DIR:?ERROR: C2_EVIDENCE_DIR not set}"

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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export REPO_DIR
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

OUT="${EVIDENCE}/${METHOD}"
mkdir -p "${OUT}"
FAILS=0
note() { echo ""; echo "########## $* ##########"; }
record() { [[ $1 -ne 0 ]] && { echo "[FAIL] $2 (rc=$1)"; FAILS=$((FAILS+1)); } || echo "[ OK ] $2"; }

echo "=========================================="
echo "C2 Stage B | ${METHOD} | job ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:   $(date)"
echo "Commit:  $(git -C "${REPO_DIR}" describe --tags --always --dirty 2>/dev/null || echo n/a)"
echo "=========================================="

# --- B2 run 1: engine must be cpp AND observably call into C++ ---------------
note "B2 run 1 -- expect cpp, observed dispatch"
"${PYTHON}" slurm/t04_probe/sp_probe.py --out "${OUT}/b2_sp_probe_cpp.json" \
    --method "${METHOD}" --expect-engine cpp
record $? "B2 run 1 (cpp)"

# --- B2 run 2: the negative control -----------------------------------------
# ISALSR_ENGINE=python is the real variable; there is no ISALSR_FORCE_PYTHON.
note "B2 run 2 -- NEGATIVE CONTROL, expect python and NO C++ call"
ISALSR_ENGINE=python "${PYTHON}" slurm/t04_probe/sp_probe.py \
    --out "${OUT}/b2_sp_probe_python.json" --method "${METHOD}" --expect-engine python
record $? "B2 run 2 (python negative control)"

# The pair is what proves anything: identical verdicts in both directions is a
# FAILURE, not a pass.
"${PYTHON}" - "${OUT}/b2_sp_probe_cpp.json" "${OUT}/b2_sp_probe_python.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["SP-3_engine"]
b = json.load(open(sys.argv[2]))["SP-3_engine"]
ok = (a["reported_engine"] == "cpp" and a["cpp_actually_invoked"]
      and b["reported_engine"] == "python" and not b["cpp_actually_invoked"])
print(f"  run1 engine={a['reported_engine']} cpp_invoked={a['cpp_actually_invoked']}")
print(f"  run2 engine={b['reported_engine']} cpp_invoked={b['cpp_actually_invoked']}")
print(f"  B2 negative control: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
record $? "B2 negative control differentiates"

# --- B1: environment + all 70 datasets --------------------------------------
note "B1 -- environment and 70/70 dataset resolvability"
"${PYTHON}" slurm/c2_smoke/stage_b_probe.py --out "${OUT}/b1_environment.json"
record $? "B1 environment + datasets"

# --- B3: alphabet gate on the frozen commit ---------------------------------
# Two problems per host, matching the 2026-07-30 precedent (job 1692451): an
# easy one and a structurally hard one, so the candidate stream carries a wide
# label mix rather than only Add/Mul.
note "B3 -- alphabet gate (decomposed Sigma_SR)"
B3_RC=0
# Problem choice matters more than the budget here.  Bingo SOLVES Nguyen-1
# (x^3+x^2+x) inside the first generations, so under a bounded budget it stops
# before any candidate reaches the canonicaliser and the gate reports
# "DAGs observed: 0" -- a dead measurement, which the gate rightly refuses.
# (Observed: job 1751926.)  Both Bingo problems below are structurally hard
# enough that the search is still producing candidates when the budget expires.
# UDFS enumerates systematically and does not converge away, so Nguyen-1 is fine
# for it (verified PASS on job 1751925).
if [[ "${METHOD}" == "bingo" ]]; then
    B3_PAIRS="bingo_hard:Pagie-1 bingo_cherrypicked:I.29.16"
else
    B3_PAIRS="udfs_nguyen:Nguyen-1 udfs_hard:Keijzer-6"
fi
# verify_alphabet_gate.py has NO max_time handling -- it runs whatever the YAML
# says, and every production config says 43,200 s. Pointed at a production
# config it therefore runs for 12 h and the job dies on the SLURM wall with the
# gate half-finished. (Observed: jobs 1751916/1751917, cancelled at 14 min.)
# The pre-existing slurm/alphabet_gate/worker.sh solves this by rewriting the
# YAML; do the same, into a scratch copy, so the production configs are untouched.
B3_TMP="${OUT}/b3_configs"
mkdir -p "${B3_TMP}"
for PAIR in ${B3_PAIRS}; do
    CFG_NAME="${PAIR%%:*}"; PROB="${PAIR##*:}"
    "${PYTHON}" - "experiments/configs/${CFG_NAME}.yaml" "${B3_TMP}/${CFG_NAME}.yaml" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
for section in cfg.values():
    if isinstance(section, dict) and "max_time" in section:
        section["max_time"] = 120
yaml.safe_dump(cfg, open(dst, "w"))
PY
    "${PYTHON}" experiments/scripts/verify_alphabet_gate.py \
        --config "${B3_TMP}/${CFG_NAME}.yaml" \
        --problems "${PROB}" --seeds 0 \
        --output-dir "${OUT}/b3_scratch/${CFG_NAME}" \
        --json-out "${OUT}/b3_alphabet_gate_${CFG_NAME}.json"
    B3_RC=$((B3_RC + $?))
done
record ${B3_RC} "B3 alphabet gate (${B3_PAIRS})"

# --- B9: T06 counter overhead, live-vs-dead distinguishable ------------------
note "B9 -- T06 counter overhead under the C++ engine + decomposed alphabet"
B9_ROOT="${OUT}/b9"
rm -rf "${B9_ROOT}"
CFG="experiments/configs/${METHOD}_nguyen.yaml"
"${PYTHON}" -m experiments.models.orchestrator --config "${CFG}" \
    --output-dir "${B9_ROOT}/on" --seeds 0 --problems Nguyen-4 --variants isalsr \
    --max-time 240 --ledger --postprocess skip
R1=$?
"${PYTHON}" -m experiments.models.orchestrator --config "${CFG}" \
    --output-dir "${B9_ROOT}/off" --seeds 0 --problems Nguyen-4 --variants isalsr \
    --max-time 240 --postprocess skip
R2=$?
record $((R1 + R2)) "B9 paired runs completed"
"${PYTHON}" slurm/c2_smoke/stage_b_probe.py --out "${OUT}/b9_overhead.json" \
    --skip-datasets --ledger-on-dir "${B9_ROOT}/on" --ledger-off-dir "${B9_ROOT}/off"
record $? "B9 counters live"

echo ""
echo "=========================================="
echo "Stage B (${METHOD}) finished: $(date)   failures=${FAILS}"
echo "Evidence: ${OUT}"
echo "=========================================="
exit $(( FAILS > 0 ? 1 : 0 ))
