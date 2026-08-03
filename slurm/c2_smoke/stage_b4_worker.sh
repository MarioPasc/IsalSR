#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage B check B4: equivalence gate on a COMPUTE node
# =============================================================================
# T01's gate G1 passed on the workstation only.  A workstation pass does not
# certify a different compiler (gcc 13.2.0 on Picasso vs 12.2.0 locally), a
# different libstdc++, or a different CPU -- and the whole C++ port rests on
# byte-exact agreement with the Python reference.
#
# Pass: 0 mismatches, 0 errors, and `self_comparison == false`.  That last field
# is the one that matters: when the extension is unavailable the harness falls
# back to comparing Python against Python and would otherwise report a clean
# gate while proving nothing.
#
# Environment: ISALSR_REPO_DIR, C2_EVIDENCE_DIR
# =============================================================================
set -uo pipefail

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

OUT="${EVIDENCE}/b4"
mkdir -p "${OUT}"

echo "=========================================="
echo "C2 Stage B4 | job ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:   $(date)"
echo "Commit:  $(git -C "${REPO_DIR}" describe --tags --always --dirty 2>/dev/null || echo n/a)"
"${PYTHON}" -c "from isalsr.core import backends; print('Build:  ', backends.build_info())"
echo "=========================================="

"${PYTHON}" experiments/scripts/equivalence_gate.py \
    --gate all --backend-a python --backend-b cpp \
    --out "${OUT}/b4_equivalence_gate.json"
GATE_RC=$?

echo ""
"${PYTHON}" - "${OUT}/b4_equivalence_gate.json" <<'PY'
import json, sys
try:
    rep = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"[FATAL] cannot read gate report: {exc}"); sys.exit(1)

self_cmp = rep.get("self_comparison")
print(f"  pass:              {rep.get('pass')}")
print(f"  self_comparison:   {self_cmp}   ({rep.get('reason')})")
prov = rep.get("provenance", {})
print(f"  engine_a/engine_b: {prov.get('engine_a')} / {prov.get('engine_b')}")
print(f"  compiler:          {prov.get('build_info', {}).get('compiler')}")
print(f"  build_hash:        {prov.get('build_info', {}).get('build_hash')}")
print(f"  quick_mode:        {prov.get('quick_mode')}")

# B4 asks exactly one question: does the C++ engine agree, byte for byte, with
# the Python reference on THIS compiler and THIS CPU?  That is the cross-engine
# mismatch count.
#
# Gate 3 measures a different property -- round-trip isomorphism, S2D(fcs(D)) ~ D
# -- which fails on 5 of 10,000 generated DAGs IDENTICALLY on both engines, and
# therefore says nothing about the port.  It is a real and serious finding about
# canonical-string completeness (two non-isomorphic DAGs sharing a string), it is
# written up in docs/md_files/changes/canonical_completeness_counterexamples.md,
# and it is escalated to T07.  It is reported here as a tracked quantity and is
# deliberately NOT folded into B4's verdict: letting it fail B4 would bury the
# signal B4 exists to detect under a defect B4 cannot see.
bad, tracked = [], []
if self_cmp is not False:
    # A self-comparison is a silent pass that proves nothing.
    bad.append("gate ran python-vs-python; the C++ extension was not exercised")
if prov.get("quick_mode"):
    bad.append("quick_mode=true; B4 requires the full corpus")

for name, g in sorted(rep.items()):
    if not (name.startswith("gate") and isinstance(g, dict)):
        continue
    mm, inv, err = (g.get("mismatches_cross_engine"), g.get("mismatches_invariance"), g.get("errors"))
    rt_a, rt_b = g.get("mismatches_engine_a"), g.get("mismatches_engine_b")
    print(f"  {name}: compared={g.get('comparisons_made')} cross_engine={mm} "
          f"invariance={inv} errors={err} roundtrip_a/b={rt_a}/{rt_b} pass={g.get('pass')}")
    if mm or inv or err:
        bad.append(f"{name}: cross_engine={mm} invariance={inv} errors={err}")
    if rt_a or rt_b:
        tracked.append(f"{name}: round-trip failures a={rt_a} b={rt_b} "
                       f"of {g.get('dags_tested')} DAGs (engine-independent)")

for t in tracked:
    print(f"[TRACKED, not a B4 failure] {t}")
    print("           see docs/md_files/changes/canonical_completeness_counterexamples.md")
if bad:
    for b in bad:
        print(f"[FATAL] {b}")
    sys.exit(1)
print("  B4: PASS (cross-engine equivalence)")
PY
CHECK_RC=$?

echo ""
echo "Finished: $(date)   gate_rc=${GATE_RC} check_rc=${CHECK_RC}"
echo "Evidence: ${OUT}/b4_equivalence_gate.json"
exit $(( GATE_RC != 0 ? GATE_RC : CHECK_RC ))
