#!/usr/bin/env bash
# =============================================================================
# Mock-job exercise of the CHUNKED C2 worker -- workstation, real payload, no SLURM
# =============================================================================
#   bash slurm/c2_smoke/mock_chunk_test.sh          # ~8 minutes
#
# The unit tests cover the partition arithmetic. This covers the thing they
# cannot: the worker script itself, running the real orchestrator, with real
# staging and a real deadline. Every check here corresponds to a failure mode
# that was either observed or is unrecoverable in production:
#
#   1. a chunk of B cells runs all B and writes B run_log.json
#   2. $LOCALSCRATCH staging leaves the artefacts on the DURABLE root, and the
#      task's scratch directory is removed on exit
#   3. the deadline DEFERS the tail rather than overrunning the wall
#   4. 🔴 the FIRST cell runs even past the cutoff -- without this exemption the
#      array livelocks, deferring the same chunk on every pass forever (this is
#      how that defect was found, 2026-08-07)
#   5. a sweep over the same partition completes the deferred cells, and the
#      resume logic makes it cheap
#   6. the tasks of an array cover its cells exactly once, disjointly
#   7. a failing cell does not abort the chunk, is named, and still makes the
#      task exit non-zero so `sacct -X ... State` stays a usable census
#
# The Picasso half -- real $LOCALSCRATCH, module load, conda, sbatch acceptance --
# is exercised by submitting a 3x4 array from the deployed tree; see
# CAMPAIGN_BRIEF.md §5.
# =============================================================================
set -uo pipefail

REPO="${ISALSR_REPO_LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH="${C2_MOCK_DIR:-${TMPDIR:-/tmp}/c2_mock_$$}"
OUT="${SCRATCH}/mock_results"
FAKE_LOCAL="${SCRATCH}/fake_localscratch"
CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mock_chunk_config.yaml"

PASS=0
FAIL=0
check() {  # <name> <expected> <actual>
    if [[ "$2" == "$3" ]]; then
        printf '  PASS  %-58s %s\n' "$1" "$3"; PASS=$((PASS + 1))
    else
        printf '  FAIL  %-58s expected %s, got %s\n' "$1" "$2" "$3"; FAIL=$((FAIL + 1))
    fi
}

RC=0
run_task() {  # <index> <bundle> <cutoff_s> <use_local> <logfile>
    ISALSR_REPO_DIR="${REPO}" \
    C2_METHOD=bingo C2_ARM=hash C2_SUITE=nguyen \
    C2_CONFIG="${CONFIG}" C2_SEEDS="1-3" C2_MAX_TIME=8 \
    C2_RESULTS_DIR="${OUT}" \
    C2_BUNDLE="$2" C2_START_CUTOFF_S="$3" C2_USE_LOCALSCRATCH="$4" \
    LOCALSCRATCH="${FAKE_LOCAL}" \
    SLURM_ARRAY_TASK_ID="$1" SLURM_JOB_ID="mock$1" \
        bash "${REPO}/slurm/c2_smoke/worker.sh" > "$5" 2>&1
    RC=$?
}
n_logs() { find "${OUT}" -name run_log.json 2>/dev/null | wc -l; }
n_seed_dirs() { find "${OUT}" -type d -name 'seed_*' 2>/dev/null | wc -l; }

mkdir -p "${SCRATCH}"
rm -rf "${OUT}" "${FAKE_LOCAL}"; mkdir -p "${OUT}" "${FAKE_LOCAL}"
trap 'rm -rf "${SCRATCH}"' EXIT

echo "=== 1. one 3-cell chunk, generous deadline, localscratch ON ==="
run_task 1 3 100000 1 "${SCRATCH}/t1.log"
check "worker exit status" 0 "${RC}"
check "run_log.json written by the chunk" 3 "$(n_logs)"
check "worker counted 3 ok / 0 failed / 0 deferred" \
      1 "$(grep -c 'Cells:     3 ok, 0 failed, 0 deferred' "${SCRATCH}/t1.log")"
check "localscratch was used for the payload" \
      1 "$(grep -c "Staging:     ${FAKE_LOCAL}" "${SCRATCH}/t1.log")"
check "the task's localscratch dir removed on exit" \
      0 "$(find "${FAKE_LOCAL}" -maxdepth 2 -name 'c2_mock*' | wc -l)"
check "artefacts copied back to the DURABLE root" \
      1 "$([[ -d "${OUT}/bingo/nguyen" ]] && echo 1 || echo 0)"

echo ""
echo "=== 2. deadline guard: a 1 s cutoff defers everything after the first cell ==="
rm -rf "${OUT}"; mkdir -p "${OUT}"
run_task 1 3 1 1 "${SCRATCH}/t2.log"
check "worker exit status (deferral is not a failure)" 0 "${RC}"
# Forward progress is the property under test: the first cell is exempt from the
# deadline, so an array can never livelock on a cutoff it cannot reach.
check "the first cell ran despite the cutoff (no livelock)" 1 "$(n_logs)"
check "the other two were deferred" \
      1 "$(grep -c 'Cells:     1 ok, 0 failed, 2 deferred' "${SCRATCH}/t2.log")"
check "DEFERRED list printed for the sweep" 1 "$(grep -c '^DEFERRED:' "${SCRATCH}/t2.log")"

echo ""
echo "=== 3. sweep over the SAME partition completes the deferred cells ==="
run_task 1 3 100000 1 "${SCRATCH}/t3.log"
check "all 3 cells present after the sweep" 3 "$(n_logs)"
check "sweep re-ran only what was missing (resume skipped 1)" \
      1 "$(grep -c 'Cells:     3 ok' "${SCRATCH}/t3.log")"

echo ""
echo "=== 4. the 3 tasks of the array cover its 9 cells, disjointly ==="
rm -rf "${OUT}"; mkdir -p "${OUT}"
for i in 1 2 3; do run_task "${i}" 3 100000 0 "${SCRATCH}/t4_${i}.log"; done
check "9 cells over 3 tasks" 9 "$(n_logs)"
check "9 distinct seed directories, i.e. no cell run twice" 9 "$(n_seed_dirs)"
check "localscratch OFF writes straight to the durable root" \
      1 "$(grep -c 'Staging:     <none' "${SCRATCH}/t4_1.log")"
CELLS_SEEN=$(for i in 1 2 3; do
    PYTHONPATH="${REPO}/src:${REPO}" python -m experiments.scripts.c2_task_spec \
        --config "${CONFIG}" --seeds 1-3 --bundle 3 --index "${i}" | awk '{print $1"_"$2}'
done | sort | tr '\n' ' ')
check "decoded cell set is the full 3x3 grid" \
      "Nguyen-1_1 Nguyen-1_2 Nguyen-1_3 Nguyen-2_1 Nguyen-2_2 Nguyen-2_3 Nguyen-3_1 Nguyen-3_2 Nguyen-3_3 " \
      "${CELLS_SEEN}"

echo ""
echo "=== 5. a failing cell does not abort the chunk ==="
rm -rf "${OUT}"; mkdir -p "${OUT}"
# An unknown arm makes the orchestrator exit 1 (verified rc=1 directly). The
# chunk must ATTEMPT all three cells rather than aborting at the first, and must
# then report the failure through its own exit status.
ISALSR_REPO_DIR="${REPO}" C2_METHOD=bingo C2_ARM=bogus_arm C2_SUITE=nguyen \
C2_CONFIG="${CONFIG}" C2_SEEDS="1-3" C2_MAX_TIME=8 C2_RESULTS_DIR="${OUT}" \
C2_BUNDLE=3 C2_START_CUTOFF_S=100000 C2_USE_LOCALSCRATCH=1 \
LOCALSCRATCH="${FAKE_LOCAL}" SLURM_ARRAY_TASK_ID=1 SLURM_JOB_ID=mockbad \
    bash "${REPO}/slurm/c2_smoke/worker.sh" > "${SCRATCH}/t5.log" 2>&1
RC=$?
check "all 3 cells attempted, none skipped after the first failure" \
      3 "$(grep -c '^--- cell ' "${SCRATCH}/t5.log")"
check "worker counted 0 ok / 3 failed" \
      1 "$(grep -c 'Cells:     0 ok, 3 failed, 0 deferred' "${SCRATCH}/t5.log")"
check "worker exits non-zero when a cell failed" 1 "${RC}"
check "FAILED list names the cells" 1 "$(grep -c '^FAILED:.*seed=' "${SCRATCH}/t5.log")"

echo ""
echo "=== 6. 🔴 localscratch loses NOTHING: full-tree diff against a direct run ==="
# The check that matters, and the one whose absence hid a real defect. Counting
# run_log.json proved only that per-cell artefacts came back; `metadata.json` is
# written at the ROOT of the output tree (orchestrator.py:665), was never in any
# per-cell path, and died with the node. c2_certify.py:842 reads it.
#
# So compare the two trees FILE BY FILE rather than spot-checking either.
DIRECT="${SCRATCH}/direct"; STAGED="${SCRATCH}/staged"
rm -rf "${DIRECT}" "${STAGED}"; mkdir -p "${DIRECT}" "${STAGED}"
OUT="${DIRECT}"; run_task 1 3 100000 0 "${SCRATCH}/t6_direct.log"
OUT="${STAGED}"; run_task 1 3 100000 1 "${SCRATCH}/t6_staged.log"
OUT="${SCRATCH}/mock_results"
MISSING="$( (cd "${DIRECT}" && find . -type f | sort) > "${SCRATCH}/d.txt"
            (cd "${STAGED}" && find . -type f | sort) > "${SCRATCH}/s.txt"
            comm -23 "${SCRATCH}/d.txt" "${SCRATCH}/s.txt" | tr '\n' ' ' )"
check "no file produced by a direct run is missing after staging" "" "${MISSING# }"
check "the root-level metadata.json specifically came back" \
      1 "$([[ -f "${STAGED}/metadata.json" ]] && echo 1 || echo 0)"
check "both runs produced the same file count" \
      "$(find "${DIRECT}" -type f | wc -l)" "$(find "${STAGED}" -type f | wc -l)"

echo ""
echo "=== 7. 🔴 a SIGTERM still copies results back before the node is lost ==="
# SLURM sends SIGTERM and waits KillWait (30 s) before SIGKILL. Without a TERM
# trap the shell dies WITHOUT running the EXIT trap and the whole chunk's
# finished cells die on the node with it.
rm -rf "${OUT}"; mkdir -p "${OUT}"
ISALSR_REPO_DIR="${REPO}" C2_METHOD=bingo C2_ARM=hash C2_SUITE=nguyen \
C2_CONFIG="${CONFIG}" C2_SEEDS="1-3" C2_MAX_TIME=8 C2_RESULTS_DIR="${OUT}" \
C2_BUNDLE=9 C2_START_CUTOFF_S=100000 C2_USE_LOCALSCRATCH=1 \
LOCALSCRATCH="${FAKE_LOCAL}" SLURM_ARRAY_TASK_ID=1 SLURM_JOB_ID=mockterm \
    bash "${REPO}/slurm/c2_smoke/worker.sh" > "${SCRATCH}/t7.log" 2>&1 &
TERM_PID=$!
# Let it finish at least one cell, then terminate it as SLURM would.
for _ in $(seq 1 60); do
    [[ "$(find "${FAKE_LOCAL}" -name run_log.json 2>/dev/null | wc -l)" -ge 1 ]] && break
    sleep 1
done
kill -TERM "${TERM_PID}" 2>/dev/null
wait "${TERM_PID}" 2>/dev/null; TERM_RC=$?
check "SIGTERM was caught and announced" \
      1 "$(grep -c 'SIGTERM received' "${SCRATCH}/t7.log")"
check "results reached the durable root despite the kill" \
      1 "$([[ "$(n_logs)" -ge 1 ]] && echo 1 || echo 0)"
check "the task's localscratch was cleaned after the copy" \
      0 "$(find "${FAKE_LOCAL}" -maxdepth 2 -name 'c2_mockterm*' | wc -l)"

echo ""
echo "=== summary: ${PASS} passed, ${FAIL} failed ==="
[[ "${FAIL}" -eq 0 ]]
