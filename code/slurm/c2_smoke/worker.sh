#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- array worker (EXECUTION-PLAN.md §4.3, ticket T17)
# =============================================================================
# One array task = a CHUNK of (problem, seed) cells for a fixed
# (method, arm, suite), executed sequentially, on local scratch, under a
# deadline.
#
# 🔴 Why a chunk and not a cell (SCBI request, 2026-08-07)
# -------------------------------------------------------
# Picasso's administrators asked us to group short jobs so that every submitted
# task runs for at least two hours: the scheduler spends longer placing a short
# job than the job spends running, and 12,600 placements saturate the queue for
# every user of the cluster.  Our own accounting agrees -- 51.5 % of the 23,058
# job records this account produced since 2026-07-01 ran for under two minutes.
#
# The chunk size comes from experiments/scripts/c2_slot_plan.py, which sizes it
# from the C1-measured per-suite cell duration.  It is makespan-neutral: an array
# of N cells of duration T under throttle K finishes at N*T/K whether it is
# submitted as N tasks or as N/B tasks of B cells, as long as the array still has
# at least K tasks (the slot plan caps the throttle at the task count to
# guarantee that).  Measured on the real plan: 12,600 tasks -> 3,474, quantised
# makespan 48.0 h -> 49.2 h.
#
# NO #SBATCH directives here on purpose: the launcher supplies every resource
# flag on the sbatch command line, because the 42 arrays differ in size, memory,
# wall and job name.  (Same pattern as slurm/t04_probe and slurm/t05_probe.)
#
# Environment variables (exported by launcher.sh):
#   ISALSR_REPO_DIR    - repo checkout on Picasso
#   C2_METHOD          - "udfs" | "bingo"
#   C2_ARM             - "baseline" | "hash" | "isalsr"
#   C2_SUITE           - benchmark suite key (nguyen, ..., strogatz)
#   C2_CONFIG          - absolute path to the YAML config
#   C2_SEEDS           - colon- or comma-separated seed list/range, e.g. "1-30"
#   C2_MAX_TIME        - per-CELL payload budget in seconds
#   C2_RESULTS_DIR     - durable output root on FSCRATCH
#   C2_BUNDLE          - cells per task (default 1)
#   C2_START_CUTOFF_S  - last elapsed second at which a new cell may START
#   C2_USE_LOCALSCRATCH- 1 (default) to stage output on the node's local disk
#
# Task id -> cell list decoding is delegated to
# experiments/scripts/c2_task_spec.py so the benchmark registry lives in exactly
# one place.  The previous bash worker hard-coded a five-suite table and would
# have silently refused strogatz and feynman_remainder -- 20 of the 70 problems.
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
METHOD="${C2_METHOD:?ERROR: C2_METHOD not set}"
ARM="${C2_ARM:?ERROR: C2_ARM not set}"
SUITE="${C2_SUITE:?ERROR: C2_SUITE not set}"
CONFIG="${C2_CONFIG:?ERROR: C2_CONFIG not set}"
SEEDS="${C2_SEEDS:?ERROR: C2_SEEDS not set}"
MAX_TIME="${C2_MAX_TIME:?ERROR: C2_MAX_TIME not set}"
RESULTS_DIR="${C2_RESULTS_DIR:?ERROR: C2_RESULTS_DIR not set}"
BUNDLE="${C2_BUNDLE:-1}"
USE_LOCAL="${C2_USE_LOCALSCRATCH:-1}"
# Optional problem subset.  EMPTY on every campaign path, which means the whole
# suite -- it exists so a probe can exercise this exact worker on two problems
# without forking a config, and forking configs is how the A4b operator-set
# invariant gets perturbed by accident.
#
# 🔴 Shipped COLON-separated, for the same reason C2_SEEDS is: `sbatch --export`
# is comma-separated, so a comma inside a VALUE begins the next variable.
# Exporting `C2_PROBLEMS=Nguyen-1,Nguyen-2` delivers `Nguyen-1` and silently
# drops the rest -- measured 2026-08-07, when it halved the cell count of a
# probe and every task index above the new maximum died with "index out of
# range". Accept either form; the launcher sends colons.
PROBLEMS="${C2_PROBLEMS:-}"
PROBLEMS="${PROBLEMS//:/,}"

# ---------------------------------------------------------------------------
# Environment.  Extended from slurm/workers/models_experiment_slurm.sh -- every
# line below was added because something failed without it; do not prune.
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
# Bingo-IsalSR off the OOM ceiling.
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
# Decode the array index into a CELL LIST.  Echo it: a silently wrong decode
# produces a complete, plausible, WRONG result set.
# ---------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# The launcher ships the seed list COLON-separated because sbatch --export is
# comma-separated and would otherwise truncate "0,101,102" to "0". Accept either
# form, then assert the count matches what the array was sized for -- a silently
# short seed list is the exact failure that made 2/3 of the first wave die with
# "index out of range" while the other 1/3 produced correct-looking cells.
SEEDS="${SEEDS//:/,}"

# 🔴 The count must EXPAND ranges.  `c2_task_spec` accepts both an explicit list
# ("0,101,102") and a range ("1-30"), and the campaign profile ships the range --
# so a plain field count sees ONE field and this guard killed all 12,600 campaign
# tasks in seconds on 2026-08-06, with "decoded to 1 seed(s): '1-30'".
#
# The guard itself is right and stays: a silently short seed list is what made
# 2/3 of an early wave die with "index out of range" while the other 1/3 produced
# correct-looking cells.  It was the ARITHMETIC that did not know about ranges.
N_SEEDS_DECODED=$(awk -F, '{
    n = 0
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+-[0-9]+$/) { split($i, r, "-"); n += r[2] - r[1] + 1 }
        else if ($i != "")          { n += 1 }
    }
    print n
}' <<<"${SEEDS}")
if [[ "${N_SEEDS_DECODED}" -lt 2 ]]; then
    echo "[FATAL] C2_SEEDS decoded to ${N_SEEDS_DECODED} seed(s): '${SEEDS}'." >&2
    echo "        At least two are required. Check the launcher's --export quoting;" >&2
    echo "        sbatch --export is comma-separated, so a comma inside a VALUE is" >&2
    echo "        parsed as the next variable and truncates the list." >&2
    exit 1
fi

# One line per cell, "<problem> <seed>".  With --bundle 1 this is exactly the
# single line the pre-chunking worker decoded, so the two paths agree by
# construction rather than by convention.
SPEC_ARGS=(--config "${CONFIG}" --seeds "${SEEDS}"
           --bundle "${BUNDLE}" --index "${TASK_ID}")
[[ -n "${PROBLEMS}" ]] && SPEC_ARGS+=(--problems "${PROBLEMS}")
CELLS="$("${PYTHON}" -m experiments.scripts.c2_task_spec "${SPEC_ARGS[@]}")"
N_CELLS=$(grep -c . <<<"${CELLS}" || true)
if [[ "${N_CELLS}" -lt 1 ]]; then
    echo "[FATAL] empty decode for task ${TASK_ID} (bundle ${BUNDLE}): '${CELLS}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Local scratch.  Manual §4.9 and the SCBI mail of 2026-08-07: generate on the
# compute node's own disk, copy back only the results.
#
# Every cell writes ~5.6 small files, and the campaign writes ~72,000 of them
# into one shared GPFS tree.  Staging locally turns that into one rsync per cell
# out of a local directory, and -- more importantly -- keeps anything the payload
# writes DURING its 12 h search off the shared filesystem entirely.
#
# `sr` nodes carry 800 GB of localscratch (sinfo, 2026-08-07) against a campaign
# footprint measured in megabytes, so there is no capacity question here.
# ---------------------------------------------------------------------------
#
# 🔴 The task's private area is SPLIT: results under `out/`, bytecode cache under
# `pycache/`.  That separation is what lets the final copy-back mirror the WHOLE
# results tree unconditionally instead of naming the files it expects -- see
# `stage_out_all` below and the defect it exists to prevent.
LOCAL_ROOT=""
if [[ "${USE_LOCAL}" == "1" && -n "${LOCALSCRATCH:-}" && -d "${LOCALSCRATCH}" ]]; then
    LOCAL_ROOT="${LOCALSCRATCH}/${USER}/c2_${SLURM_JOB_ID:-local}_${TASK_ID}"
    if ! mkdir -p "${LOCAL_ROOT}/out" 2>/dev/null; then
        echo "[WARN] cannot create ${LOCAL_ROOT}/out; falling back to FSCRATCH" >&2
        LOCAL_ROOT=""
    fi
fi
LOCAL_OUT="${LOCAL_ROOT:+${LOCAL_ROOT}/out}"
WORK_DIR="${LOCAL_OUT:-${RESULTS_DIR}}"

# 🔴 Give this task its own bytecode cache.
#
# Picasso's login profile exports a SHARED PYTHONPYCACHEPREFIX
# (`/localscratch/users//mpascual`, measured 2026-08-07), so every task of this
# user on a node writes the SAME .pyc paths.  `sr` nodes take 128 tasks, and the
# campaign runs 2,016 concurrently.
#
# The mock array of 2026-08-07 lost one cell in twelve to
# `ModuleNotFoundError: No module named 'experiments.models.complexity_telemetry'`
# -- a top-level import in the Bingo runner, on a file that is present on disk,
# failing for ONE cell while eleven others imported it fine.  An import that is
# wrong is wrong every time; an import that fails once in twelve is a race.
# Scoping the cache per task removes the shared write target altogether, and it
# costs one directory on a disk with 800 GB free.
if [[ -n "${LOCAL_ROOT}" ]]; then
    export PYTHONPYCACHEPREFIX="${LOCAL_ROOT}/pycache"
    mkdir -p "${PYTHONPYCACHEPREFIX}"
fi

# ---------------------------------------------------------------------------
# Copy-back.  🔴 THIS IS WHERE RESULTS ARE LOST IF IT IS WRONG.
#
# `stage_out` (further down) mirrors ONE CELL as soon as it finishes, so a node
# failure at hour 30 costs the cell in flight and nothing else.  It is not
# sufficient on its own, and the reason is a measured defect:
#
#   The orchestrator writes `metadata.json` at the ROOT of the output tree
#   (orchestrator.py:665), not inside the cell's `seed_NN/`.  A per-cell
#   copy-back never touches it, so on localscratch it died with the node --
#   silently, because every per-cell artefact did come back.  `c2_certify.py:842`
#   READS that file, so the campaign would have run for two days and then failed
#   certification on evidence that no longer existed anywhere.
#   Found by diffing a localscratch run against a direct one, 2026-08-07.
#
# The lesson is not "also copy metadata.json".  It is that a copy-back which
# NAMES the files it expects is a standing invitation for the next artefact
# somebody adds at the root to vanish.  `stage_out_all` therefore mirrors the
# ENTIRE tree, unconditionally, and knows nothing about what is in it.
stage_out_all() {
    [[ -n "${LOCAL_OUT}" && -d "${LOCAL_OUT}" ]] || return 0
    mkdir -p "${RESULTS_DIR}" || return 1

    # Subtrees first. Every cell path belongs to exactly one task (the slot plan
    # partitions them), so these can never collide between concurrent tasks.
    # `cp -a src/. dst/` merges into an existing tree and preserves timestamps;
    # rsync would be tidier but is not guaranteed on a compute node, and cp is.
    local entry name tmp
    for entry in "${LOCAL_OUT}"/*/; do
        [[ -d "${entry}" ]] || continue
        cp -a "${entry}" "${RESULTS_DIR}/" || return 1
    done

    # Root-level files are the exception: `metadata.json` is written by EVERY
    # task to the SAME durable path, so 3,474 tasks race on one file. `cp`
    # truncates before it writes, so an overlap leaves a short or empty file --
    # and `c2_certify.py:842` reads exactly this file. Write to a private name
    # and rename: `mv` within one directory is an atomic replace, so a concurrent
    # reader sees either the old file or the new one, never a partial one.
    for entry in "${LOCAL_OUT}"/*; do
        [[ -f "${entry}" ]] || continue
        name="$(basename "${entry}")"
        tmp="${RESULTS_DIR}/.${name}.${SLURM_JOB_ID:-local}_${TASK_ID}.tmp"
        cp -a "${entry}" "${tmp}" && mv -f "${tmp}" "${RESULTS_DIR}/${name}" \
            || { rm -f "${tmp}"; return 1; }
    done
    return 0
}

# The manual's own cleanup example has the test INVERTED
# (`if [ -z "$MYLOCALSCRATCH" ]`) -- it deletes only when the variable is empty,
# which is exactly when `rm -rf` would target the wrong thing. Hence `-n`.
#
# And nothing is deleted until the copy-back has been VERIFIED. A `cp` that
# fails on a full or unreachable FSCRATCH must leave the node's copy in place:
# a task that reports success having deleted the only copy of its results is the
# worst outcome available here, and it is worth leaving a directory behind on a
# compute node to avoid it.
FINALIZED=0
finalize() {
    local rc=$?
    [[ "${FINALIZED}" == "1" ]] && return "${rc}"
    FINALIZED=1
    [[ -n "${LOCAL_ROOT}" ]] || return "${rc}"

    # 🔴 `set -e` is OFF for the whole of this function, and `rc` is captured on
    # entry and returned unchanged.
    #
    # Measured 2026-08-07: one staged task in twelve was marked FAILED by SLURM
    # while its own log said `ok=2 fail=0` and every one of its files had reached
    # FSCRATCH. The verification below used to walk the SHARED results root, and
    # a concurrent task renaming its atomic temp file made `find` fail to stat a
    # vanished entry and exit non-zero. `pipefail` propagated that through
    # `| wc -l`, `set -e` aborted the trap mid-way, and the shell exited 1 --
    # so a healthy task reported failure AND skipped its own cleanup, leaving the
    # node's disk occupied. An EXIT trap that can itself fail turns cosmetic
    # errors into lost work and false alarms; this one cannot.
    set +e

    local missing=0 n_local=0 rel
    if ! stage_out_all; then
        n_local=$(find "${LOCAL_OUT}" -type f 2>/dev/null | wc -l)
        echo "[FATAL] copy-back FAILED. ${n_local} file(s) remain ONLY at" >&2
        echo "        ${LOCAL_OUT} on $(hostname). NOT deleting them." >&2
        echo "        Recover before the node's scratch is reclaimed." >&2
        return "${rc}"
    fi

    # Verify against the paths THIS TASK owns, not against a walk of the shared
    # tree: exact, cheap, and immune to what the other 3,473 tasks are doing.
    while IFS= read -r rel; do
        [[ -n "${rel}" ]] || continue
        n_local=$((n_local + 1))
        [[ -f "${RESULTS_DIR}/${rel}" ]] || { missing=$((missing + 1)); \
            echo "[FATAL] not copied back: ${rel}" >&2; }
    done < <(cd "${LOCAL_OUT}" 2>/dev/null && find . -type f -printf '%P\n' 2>/dev/null)

    if (( missing > 0 )); then
        echo "[FATAL] ${missing} of ${n_local} file(s) did NOT reach ${RESULTS_DIR}." >&2
        echo "        NOT deleting ${LOCAL_OUT} on $(hostname)." >&2
        return "${rc}"
    fi

    echo "Copy-back:  ${n_local} file(s) verified, ${LOCAL_OUT} -> ${RESULTS_DIR}"
    rm -rf --one-file-system "${LOCAL_ROOT}"
    return "${rc}"
}
trap finalize EXIT

# 🔴 SLURM sends SIGTERM and waits (KillWait, 30 s by default) before SIGKILL, so
# a wall-clock kill or an `scancel` gets one chance to save its work. Without
# these two lines the default disposition terminates the shell WITHOUT running
# the EXIT trap, and every finished cell of the chunk dies on the node.
# The handlers exit, which runs `finalize` through the EXIT trap.
trap 'echo "[WARN] SIGTERM received -- copying results back before exit" >&2; exit 143' TERM
trap 'echo "[WARN] SIGINT received -- copying results back before exit"  >&2; exit 130' INT

# ---------------------------------------------------------------------------
# SP-1..SP-3 provenance header.  Recorded per task, from the COMPUTE NODE, so a
# stale checkout or a stale .so is visible in the log of every run rather than
# inferred afterwards.
# ---------------------------------------------------------------------------
CUTOFF_S="${C2_START_CUTOFF_S:-0}"
echo "=========================================="
echo "C2 | job ${SLURM_JOB_ID:-local} task ${TASK_ID}"
echo "Node:        $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:       $(date)"
echo "Method/Arm:  ${METHOD} / ${ARM}"
echo "Suite:       ${SUITE}"
echo "Config:      ${CONFIG}"
echo "Cells:       ${N_CELLS} (bundle ${BUNDLE}), payload ${MAX_TIME}s each"
echo "Deadline:    start no new cell after ${CUTOFF_S}s elapsed"
echo "Staging:     ${LOCAL_ROOT:-<none: writing straight to FSCRATCH>}"
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
echo "Cell list:"
nl -ba <<<"${CELLS}" | sed 's/^/  /'
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Staging helpers.
#
# The output tree is <root>/<method>/<suite>/<problem_slug>/<arm>/seed_NN
# (experiments/models/io_utils.py:create_folder_structure).  Both helpers move
# exactly that leaf, so two concurrent tasks of the same array can never touch
# the same path: the slot plan gives every cell to exactly one task.
# ---------------------------------------------------------------------------
# The problem SLUG is supplied by c2_task_spec, not recomputed here.  It is
# io_utils' own `problem.lower().replace("-", "_")`, and a bash reimplementation
# would be a silent-divergence trap: a wrong slug does not fail, it copies
# nothing, and the cell is then lost on a node whose localscratch is wiped.
cell_relpath() {  # <seed> <slug> -> relative dir of that cell's artefacts
    printf '%s/%s/%s/%s/seed_%02d' "${METHOD}" "${SUITE}" "$2" "${ARM}" "$1"
}

stage_in() {  # seed localscratch so the orchestrator's OWN resume check applies
    local rel="$1"
    [[ -n "${LOCAL_OUT}" ]] || return 0
    [[ -d "${RESULTS_DIR}/${rel}" ]] || return 0
    mkdir -p "${LOCAL_OUT}/${rel}"
    cp -a "${RESULTS_DIR}/${rel}/." "${LOCAL_OUT}/${rel}/" 2>/dev/null || true
}

stage_out() {  # copy back after EACH cell, so a node loss costs one cell only
    local rel="$1"
    [[ -n "${LOCAL_OUT}" ]] || return 0
    [[ -d "${LOCAL_OUT}/${rel}" ]] || return 0
    mkdir -p "${RESULTS_DIR}/${rel}"
    # Best-effort: `finalize` mirrors the whole tree at the end regardless, so a
    # transient failure here costs promptness, not results. It must NOT abort the
    # chunk under `set -e`.
    cp -a "${LOCAL_OUT}/${rel}/." "${RESULTS_DIR}/${rel}/" \
        || echo "[WARN] per-cell copy-back of ${rel} failed; final sync will retry" >&2
}

# ---------------------------------------------------------------------------
# Payload loop.
#
#   --ledger          T17 §2.1.  ISALSR_LEDGER_ENABLED defaults to "0" and is
#                     set in no config; without this flag every run records five
#                     reachability rates of zero, which reads as "no fallbacks
#                     occurred" and means "nothing was counted".  Unrecoverable
#                     after the fact (SP-6).
#   --postprocess skip
#                     Aggregates, the three paired contrasts and the status
#                     ledger are a campaign-level step.  Left on, every task
#                     would walk the whole output tree and write the same three
#                     files concurrently.  aggregate_worker.sh does it once.
#
# 🔴 The deadline.  A cell is started only if the FULL payload budget still fits
# inside the wall.  Two consequences, both deliberate:
#   * A SLURM TIMEOUT becomes impossible by construction, so the standing "no
#     task was killed by SLURM" assertion stays a real signal rather than
#     expected noise from chunking.
#   * Every cell that runs gets its full, identical `--max-time`.  Trimming a
#     late cell's budget to "fit" would make its wall clock incomparable with
#     its paired cells in the other two arms -- and wall clock is a REPORTED
#     quantity (S, the overhead percentages, Table 2's cost column).
# Cells left unstarted are printed as DEFERRED and picked up by the sweep pass,
# which re-derives the identical partition from the same (n_cells, n_tasks).
# ---------------------------------------------------------------------------
N_OK=0
N_FAIL=0
N_DEFERRED=0
FAILED_CELLS=""
DEFERRED_CELLS=""

CELL_INDEX=0

while read -r PROBLEM_NAME SEED SLUG; do
    [[ -n "${PROBLEM_NAME}" && -n "${SEED}" && -n "${SLUG}" ]] || continue
    CELL_INDEX=$((CELL_INDEX + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))

    # 🔴 The FIRST cell always runs, deadline or not.  Without this exemption a
    # task whose start-up alone exceeded the cutoff would defer its whole chunk,
    # the sweep would re-derive the same partition and defer it again, and the
    # array would never progress -- a livelock that costs the wall of every pass
    # and completes nothing.  It is also sound: c2_slot_plan asserts
    # `start_cutoff_h > 0`, i.e. the wall always has room for one full cell, so
    # the exemption can never cause the TIMEOUT the deadline exists to prevent.
    # Cells 2..B are the ones the deadline governs, and they are the ones a
    # sweep can pick up.
    if (( CELL_INDEX > 1 && CUTOFF_S > 0 && ELAPSED >= CUTOFF_S )); then
        N_DEFERRED=$((N_DEFERRED + 1))
        DEFERRED_CELLS="${DEFERRED_CELLS} ${PROBLEM_NAME}/seed=${SEED}"
        continue
    fi

    REL="$(cell_relpath "${SEED}" "${SLUG}")"
    stage_in "${REL}"

    echo "--- cell ${CELL_INDEX}/${N_CELLS}: ${PROBLEM_NAME} seed=${SEED} "\
"(elapsed ${ELAPSED}s) ---"
    set +e
    "${PYTHON}" -m experiments.models.orchestrator \
        --config "${CONFIG}" \
        --output-dir "${WORK_DIR}" \
        --seeds "${SEED}" \
        --problems "${PROBLEM_NAME}" \
        --variants "${ARM}" \
        --max-time "${MAX_TIME}" \
        --ledger \
        --postprocess skip
    RC=$?
    set -e

    stage_out "${REL}"

    if (( RC == 0 )); then
        N_OK=$((N_OK + 1))
    else
        N_FAIL=$((N_FAIL + 1))
        FAILED_CELLS="${FAILED_CELLS} ${PROBLEM_NAME}/seed=${SEED}(rc=${RC})"
        # Keep going.  One failed cell must not cost the other B-1 cells of the
        # chunk: §5.5 drops whole (method, problem, seed) triples across all
        # three arms, never a partial chunk, and the status ledger is what makes
        # the gap provable.
        echo "[WARN] cell ${PROBLEM_NAME}/seed=${SEED} exited ${RC}; continuing" >&2
    fi
done <<< "${CELLS}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "Cells:     ${N_OK} ok, ${N_FAIL} failed, ${N_DEFERRED} deferred (of ${N_CELLS})"
[[ -n "${FAILED_CELLS}" ]]   && echo "FAILED:  ${FAILED_CELLS}"
[[ -n "${DEFERRED_CELLS}" ]] && echo "DEFERRED:${DEFERRED_CELLS}"
echo "=== Task ${TASK_ID} (${METHOD}/${ARM}/${SUITE}) ok=${N_OK} fail=${N_FAIL} defer=${N_DEFERRED} ==="

# A DEFERRED cell is not a failure: the sweep pass exists for it, and exiting
# non-zero would make the sacct census read a healthy campaign as broken.
#
# A FAILED cell is.  The pre-chunking worker propagated the payload's exit code,
# and that is what makes `sacct -X ... State` a usable first-line census; keeping
# it means a chunk with any bad cell is visibly FAILED and gets looked at.  The
# status ledger remains authoritative for WHICH cells are missing (§5.5) -- this
# code only says "something in this chunk needs attention".
if (( N_FAIL > 0 )); then
    exit 1
fi
exit 0
