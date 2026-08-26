#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage D worker (EXECUTION-PLAN.md §4.4, audit.md §7)
# =============================================================================
# One array task = one of the 12 full-length certification cells.
#
# Derived from slurm/c2_smoke/worker.sh.  Every line of that worker's
# environment block was added because something failed without it; the whole
# block is carried over unchanged.  What is NEW here:
#
#   1. The array index decodes against the 12-cell Stage D registry
#      (experiments/scripts/stage_d_task_spec.py), not against a suite config.
#      The registry is the only place the cell list exists.
#   2. A sidecar RSS sampler writes a per-cell memory time series.  This is
#      Mario's addendum to audit.md §7 row 3: sacct MaxRSS gives one number per
#      job, which cannot answer "how low can production request without OOM
#      risk".  The time series shows the growth curve.
#   3. The D2 detailed candidate trace is enabled for exactly one cell, via an
#      env flag the worker sets from the registry's own `trace` field.
#
# NO #SBATCH directives here on purpose: the launcher supplies every resource
# flag on the sbatch command line, because the three groups differ in size and
# memory by 16x.
#
# Environment variables (exported by launcher.sh):
#   ISALSR_REPO_DIR   - repo checkout on Picasso
#   D_GROUP           - "udfs" | "bingo_std" | "bingo_isalsr"
#   D_RESULTS_DIR     - output root (c2_stage_d/)
#   D_RSS_INTERVAL    - optional; RSS sample period in seconds (default 60)
#   D_PROBE_MAX_TIME  - optional; overrides the 43,200 s budget for an SP-0
#                       probe.  Hard-capped at 1,800 s, and when set the worker
#                       refuses to write into a campaign root.
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
GROUP="${D_GROUP:?ERROR: D_GROUP not set}"
RESULTS_DIR="${D_RESULTS_DIR:?ERROR: D_RESULTS_DIR not set}"
RSS_INTERVAL="${D_RSS_INTERVAL:-60}"

# ---------------------------------------------------------------------------
# Environment.  Carried verbatim from slurm/c2_smoke/worker.sh -- do not prune.
# ---------------------------------------------------------------------------

# bingo-nasa imports mpi4py, whose ABI probe dlopen()s libmpi at IMPORT time.
# A single-process job that never touches MPI still dies in ~13 s without this.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Bypass CPython's pymalloc arena allocator.  pymalloc fragments the heap over
# 10k+ generations (256 KB arenas pinned by surviving objects); glibc malloc
# mmaps large allocations and malloc_trim(0) reclaims them.  This is what keeps
# Bingo-IsalSR off the OOM ceiling -- and at 12 h it matters far more than it
# did at the 900 s Stage C budget.
export PYTHONMALLOC=malloc

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# One core per task: keep BLAS from oversubscribing what we did not reserve.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

# ---------------------------------------------------------------------------
# Decode the array index against the registry.  The decode is echoed in full
# below: a silently wrong decode produces a complete, plausible, WRONG cell,
# which is the failure mode that cost the first Stage C wave 265 tasks.
# ---------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

SPEC="$("${PYTHON}" -m experiments.scripts.stage_d_task_spec \
            --group "${GROUP}" --index "${TASK_ID}")" || {
    echo "[FATAL] registry decode failed for group '${GROUP}' index ${TASK_ID}" >&2
    exit 1
}
# The registry emits `D_KEY='value'` lines, not positional fields, so adding a
# field cannot silently shift what the worker reads.
eval "${SPEC}"

: "${D_METHOD:?ERROR: decode produced no D_METHOD}"
: "${D_ARM:?ERROR: decode produced no D_ARM}"
: "${D_PROBLEM:?ERROR: decode produced no D_PROBLEM}"
: "${D_SEED:?ERROR: decode produced no D_SEED}"
: "${D_SUITE:?ERROR: decode produced no D_SUITE}"
: "${D_TRACE:?ERROR: decode produced no D_TRACE}"

CONFIG="${REPO_DIR}/experiments/configs/${D_CONFIG_NAME}"
[[ -f "${CONFIG}" ]] || { echo "[FATAL] no config at ${CONFIG}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Budget.  Stage D is the ONLY pre-flight stage that runs the production
# 43,200 s.  D_PROBE_MAX_TIME exists so the runbook's step-4 probe can exercise
# this exact worker under SP-0's caps; it is capped and refuses a campaign root
# so a probe can never be mistaken for a certification cell.
# ---------------------------------------------------------------------------
MAX_TIME="${D_MAX_TIME}"
if [[ -n "${D_PROBE_MAX_TIME:-}" ]]; then
    if [[ "${D_PROBE_MAX_TIME}" -gt 1800 ]]; then
        echo "[FATAL] D_PROBE_MAX_TIME=${D_PROBE_MAX_TIME} exceeds the SP-0 cap of 1800 s." >&2
        exit 1
    fi
    case "${RESULTS_DIR}" in
        */execs/isalsr/*) : ;;
        *) echo "[FATAL] probe mode must write under ~/execs/isalsr/, got '${RESULTS_DIR}'." >&2
           exit 1 ;;
    esac
    MAX_TIME="${D_PROBE_MAX_TIME}"
    echo "[WARN] SP-0 PROBE MODE: max_time=${MAX_TIME}s. This is NOT a certification cell."
fi

SEED_DIR="${RESULTS_DIR}/${D_METHOD}/${D_SUITE}/${D_PROBLEM_SLUG}/${D_ARM}/seed_$(printf '%02d' "${D_SEED}")"
mkdir -p "${SEED_DIR}"

# ---------------------------------------------------------------------------
# D2 detailed trace, for exactly one cell (audit.md §7 row 2:
# Bingo x Pagie-1 x isalsr x seed 101).  Gated on the registry's own flag so
# the launcher cannot enable it for the wrong cell by mistake.
# ---------------------------------------------------------------------------
if [[ "${D_TRACE}" == "1" ]]; then
    export ISALSR_STAGE_D_TRACE=1
    export ISALSR_STAGE_D_TRACE_DIR="${SEED_DIR}/c2_trace"
    export ISALSR_STAGE_D_TRACE_SAMPLE_RATE="${D_TRACE_SAMPLE_RATE:-100}"
    mkdir -p "${ISALSR_STAGE_D_TRACE_DIR}"
    echo "[D2] detailed trace ENABLED -> ${ISALSR_STAGE_D_TRACE_DIR}"
    echo "[D2] sampling 1 candidate in ${ISALSR_STAGE_D_TRACE_SAMPLE_RATE}"
fi

# ---------------------------------------------------------------------------
# SP-1..SP-3 provenance header.  Recorded per task, from the COMPUTE NODE, so a
# stale checkout or a stale .so is visible in the log of every run rather than
# inferred afterwards.
# ---------------------------------------------------------------------------
echo "=========================================="
echo "C2 Stage D | job ${SLURM_JOB_ID:-local} task ${TASK_ID}"
echo "Node:        $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:       $(date)"
echo "Cell:        ${D_INDEX}/12  (group ${D_GROUP} index ${D_GROUP_INDEX})"
echo "Method/Arm:  ${D_METHOD} / ${D_ARM}"
echo "Suite:       ${D_SUITE}"
echo "Problem:     ${D_PROBLEM}"
echo "Seed:        ${D_SEED}"
echo "Budget:      ${MAX_TIME} s"
echo "Mem request: ${D_MEM_GB} GB"
echo "Trace cell:  ${D_TRACE}"
echo "Config:      ${CONFIG}"
echo "Results:     ${RESULTS_DIR}"
# SP-1 from the compute node: report HEAD, which is the thing that matters and
# is a pure read.  Deliberately NOT `--dirty`: rsync rewrites mtimes, so git's
# index looks stale and `describe --dirty` appends "-dirty" on content that is
# byte-identical to HEAD.  Tree cleanliness is asserted once, from the login
# node, by deploy.sh before submission.
echo "SP-1 commit: $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "SP-1 tag:    $(git -C "${REPO_DIR}" describe --tags --always 2>/dev/null || echo n/a)"
"${PYTHON}" - <<'PROV'
import datetime, os, sys
try:
    import isalsr
    from isalsr.core import _native, backends
    print(f"SP-2 pkg:    {isalsr.__file__}")
    print(f"SP-2 native: {_native.__file__}")
    print(f"SP-2 mtime:  {datetime.datetime.fromtimestamp(os.path.getmtime(_native.__file__))}")
    print(f"SP-3 engine: {backends.engine()}")
    print(f"SP-3 build:  {backends.build_info()}")
except Exception as exc:  # a broken engine must be loud, not inferred later
    print(f"[FATAL] engine probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PROV
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# RSS sampler (Mario's addendum to audit.md §7 row 3).
#
# Why a sampler at all, when sacct already reports MaxRSS: sacct gives ONE
# number for the whole step, which answers "did it fit" but not "how much
# headroom is real".  D1.2 must recommend a production --mem, and a single peak
# cannot distinguish a brief allocation spike from a sustained plateau.
#
# Why 60 s is enough despite being coarse: VmHWM is the kernel's own
# high-water mark and is MONOTONE.  The peak it reports does not depend on when
# we sample -- only the shape of the VmRSS curve does.  So the last row's
# vmhwm_kb is the true peak up to that instant regardless of sampling rate, and
# the vmrss_kb column supplies the growth curve.  A 12 h run yields ~720 rows,
# ~20 KB: negligible against the FSCRATCH inode and space budget.
#
# Scope note carried into the report: this samples the PAYLOAD PROCESS, while
# sacct MaxRSS accounts the whole cgroup.  D1.2 therefore takes the MAX of the
# two rather than trusting either alone.
# ---------------------------------------------------------------------------
RSS_CSV="${SEED_DIR}/rss_timeseries.csv"

rss_sampler() {
    local pid="$1" out="$2" interval="$3"
    local t0 now rss hwm
    t0=$(date +%s)
    echo "timestamp_s,vmrss_kb,vmhwm_kb" > "${out}"
    while kill -0 "${pid}" 2>/dev/null; do
        if [[ -r "/proc/${pid}/status" ]]; then
            rss=$(awk '/^VmRSS:/{print $2; exit}' "/proc/${pid}/status" 2>/dev/null || true)
            hwm=$(awk '/^VmHWM:/{print $2; exit}' "/proc/${pid}/status" 2>/dev/null || true)
            if [[ -n "${rss}" && -n "${hwm}" ]]; then
                now=$(date +%s)
                echo "$((now - t0)),${rss},${hwm}" >> "${out}"
            fi
        fi
        sleep "${interval}"
    done
}

# ---------------------------------------------------------------------------
# Payload.
#
#   --ledger          T17 §2.1.  ISALSR_LEDGER_ENABLED defaults to "0" and is
#                     set in no config; without this flag every run records
#                     five reachability rates of zero, which reads as "no
#                     fallbacks occurred" and means "nothing was counted".
#                     Unrecoverable after the fact (SP-6).
#   --postprocess skip
#                     Aggregates, the paired contrasts and the status ledger
#                     are a campaign-level step; one dependent job does them
#                     once, afterwards.
#   shadow sketches   ON.  Left at the default (the arm-level opt-out is
#                     --no-shadow-hash, deliberately NOT passed) because
#                     decision 3 of audit.md §6 defers the keep/drop call until
#                     shadow_time_s can be read from these 12 h cells.
#
# Backgrounded so the sampler can follow its PID, then waited on.  `wait`
# returns the child's status, so the exit code still propagates.
# ---------------------------------------------------------------------------
set +e
"${PYTHON}" -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --output-dir "${RESULTS_DIR}" \
    --seeds "${D_SEED}" \
    --problems "${D_PROBLEM}" \
    --variants "${D_ARM}" \
    --max-time "${MAX_TIME}" \
    --ledger \
    --postprocess skip &
PAYLOAD_PID=$!

rss_sampler "${PAYLOAD_PID}" "${RSS_CSV}" "${RSS_INTERVAL}" &
SAMPLER_PID=$!

wait "${PAYLOAD_PID}"
RC=$?
set -e

# The sampler exits on its own once the payload is gone; give it one interval,
# then make sure it cannot outlive the task.
sleep 1
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

N_RSS_ROWS=$(( $(wc -l < "${RSS_CSV}" 2>/dev/null || echo 1) - 1 ))
PEAK_HWM_KB=$(awk -F, 'NR>1 && $3+0>m {m=$3+0} END {print m+0}' "${RSS_CSV}" 2>/dev/null || echo 0)

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "RSS rows:  ${N_RSS_ROWS} -> ${RSS_CSV}"
echo "RSS peak:  ${PEAK_HWM_KB} kB VmHWM ($((PEAK_HWM_KB / 1048576)) GB) vs ${D_MEM_GB} GB requested"
echo "=== Cell ${D_INDEX}/13 (${D_METHOD}/${D_ARM}/${D_PROBLEM}/seed=${D_SEED}) rc=${RC} ==="
exit ${RC}
