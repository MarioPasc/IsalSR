#!/usr/bin/env bash
# =============================================================================
# Stage F go/no-go gate for Campaign C2 (EXECUTION-PLAN §4.6)
# =============================================================================
# Ten checks against the live cluster and the live tree.  SUBMITS NOTHING.
# Exits 0 only when every blocking check passes; exits 1 otherwise.
#
# Run from the workstation, from the repo root, on the commit to be tagged:
#
#   bash slurm/c2_campaign/stage_f_preflight.sh
#   bash slurm/c2_campaign/stage_f_preflight.sh --skip-tag   # before the tag exists
#
# FAILS CLOSED.  Every check that cannot be evaluated is a FAIL, never a SKIP.
# That rule is not pedantry: Stage D's Picasso certifier reported GO with a
# BLOCKING criterion sitting at SKIP, because its C1 reference was a workstation
# path no compute node could reach, and the headline banked an unevaluated
# check (T17-appendix §14.5).  Same family as both C1.11 defects and as the
# first version of the inode guard.  A check that cannot run has not passed.
# =============================================================================
set -uo pipefail

REMOTE="${ISALSR_REMOTE:-picasso}"
REPO_REMOTE="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
CAMPAIGN_ROOT="${C2_CAMPAIGN_ROOT:-${FSCRATCH}/results/isalsr/c2_3arm}"
SMOKE_ROOT="${C2_CERTIFIED_SMOKE_ROOT:-${FSCRATCH}/results/isalsr/c2_smoke}"
LOCAL_BASE="${C2_LOCAL_BASE:-/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/real_benchmarks}"
TAG="${C2_TAG:-campaign/c2}"
EXPECTED_ARRAYS=42
EXPECTED_TASKS=12600

SKIP_TAG=false
[[ "${1:-}" == "--skip-tag" ]] && SKIP_TAG=true

PY="$(conda run -n isalsr which python 2>/dev/null || echo ~/.conda/envs/isalsr/bin/python)"
REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

N_PASS=0
N_FAIL=0
FAILED_IDS=()

# ---- reporting -------------------------------------------------------------
ok()   { printf '  \033[32mPASS\033[0m  %-8s %s\n' "$1" "$2"; N_PASS=$((N_PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %-8s %s\n' "$1" "$2"; N_FAIL=$((N_FAIL+1)); FAILED_IDS+=("$1"); }

echo "==========================================================================="
echo "Stage F pre-flight -- Campaign C2"
echo "  repo (local) : ${REPO_LOCAL}"
echo "  repo (remote): ${REMOTE}:${REPO_REMOTE}"
echo "  campaign root: ${CAMPAIGN_ROOT}"
echo "  smoke root   : ${SMOKE_ROOT}"
echo "==========================================================================="
echo ""

# ---------------------------------------------------------------------------
# G1 -- SP-1: the commit is what you think it is, verified from the remote side
# ---------------------------------------------------------------------------
LOCAL_HEAD="$(git -C "${REPO_LOCAL}" rev-parse HEAD 2>/dev/null || echo)"
LOCAL_DIRTY="$(git -C "${REPO_LOCAL}" status --porcelain 2>/dev/null | wc -l)"
REMOTE_STATE="$(ssh "${REMOTE}" "cd ${REPO_REMOTE} 2>/dev/null && \
    printf '%s|%s' \"\$(git rev-parse HEAD)\" \"\$(git status --porcelain | wc -l)\"" 2>/dev/null || echo)"
REMOTE_HEAD="${REMOTE_STATE%%|*}"
REMOTE_DIRTY="${REMOTE_STATE##*|}"

if [[ -z "${LOCAL_HEAD}" || -z "${REMOTE_STATE}" ]]; then
    bad "G1" "SP-1: could not read local or remote git state"
elif [[ "${LOCAL_DIRTY}" != "0" ]]; then
    bad "G1" "SP-1: local tree is dirty (${LOCAL_DIRTY} files) -- commit first"
elif [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
    bad "G1" "SP-1: remote HEAD ${REMOTE_HEAD:0:7} != local ${LOCAL_HEAD:0:7} -- redeploy"
elif [[ "${REMOTE_DIRTY}" != "0" ]]; then
    bad "G1" "SP-1: remote tree dirty (${REMOTE_DIRTY} files)"
else
    ok "G1" "SP-1: local == remote == ${LOCAL_HEAD:0:7}, both clean"
fi

# ---------------------------------------------------------------------------
# G2 -- SP-2: the native artefact on the cluster is fresh and is the C++ engine
# ---------------------------------------------------------------------------
BUILD_OUT="$(ssh "${REMOTE}" "bash -l -c '
    cd ${REPO_REMOTE} 2>/dev/null || exit 1
    eval \"\$(conda shell.bash hook)\" && conda activate isalsr 2>/dev/null || exit 1
    python slurm/c2_smoke/verify_build.py 2>&1'" 2>/dev/null || echo)"
if grep -q "SP-2 OK" <<<"${BUILD_OUT}" && grep -q "engine:.*cpp" <<<"${BUILD_OUT}"; then
    BH="$(grep -o "'build_hash': '[^']*'" <<<"${BUILD_OUT}" | head -1 | cut -d"'" -f4)"
    ok "G2" "SP-2: engine=cpp, build_hash=${BH:-?}, artefact post-dates sources"
else
    bad "G2" "SP-2: native build unverified on the cluster (rerun deploy.sh)"
fi

# ---------------------------------------------------------------------------
# G3 -- A1: the tag exists, is annotated, and sits on the commit that will run
# ---------------------------------------------------------------------------
if ${SKIP_TAG}; then
    bad "G3" "A1: --skip-tag given; the tag is a BLOCKING gate and is not signed"
else
    TAG_COMMIT="$(git -C "${REPO_LOCAL}" rev-parse "${TAG}^{commit}" 2>/dev/null || echo)"
    TAG_TYPE="$(git -C "${REPO_LOCAL}" cat-file -t "${TAG}" 2>/dev/null || echo)"
    if [[ -z "${TAG_COMMIT}" ]]; then
        bad "G3" "A1: tag ${TAG} does not exist"
    elif [[ "${TAG_TYPE}" != "tag" ]]; then
        bad "G3" "A1: ${TAG} is ${TAG_TYPE}, not an annotated tag"
    elif [[ "${TAG_COMMIT}" != "${LOCAL_HEAD}" ]]; then
        bad "G3" "A1: ${TAG} -> ${TAG_COMMIT:0:7} but HEAD is ${LOCAL_HEAD:0:7}"
    elif ! git -C "${REPO_LOCAL}" ls-remote --tags origin "${TAG}" 2>/dev/null | grep -q .; then
        bad "G3" "A1: ${TAG} exists locally but was never pushed"
    else
        ok "G3" "A1: ${TAG} annotated, on ${TAG_COMMIT:0:7}, pushed"
    fi
fi

# ---------------------------------------------------------------------------
# G4 -- A5: the seed set, re-signed at 1..30 (§11.2 A5 was REOPENED)
# ---------------------------------------------------------------------------
SEED_OUT="$(PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" - <<'PYEOF' 2>&1
import pathlib, sys, yaml

CAMPAIGN = {"nguyen","feynman","hard","cherrypicked","roundoff",
            "feynman_remainder","strogatz"}
seeds = set(range(1, 31))
problems = []

if 0 in seeds:
    problems.append("seed 0 is in the campaign set")
if seeds & {0, 101, 102}:
    problems.append("campaign seeds collide with Stage C's {0,101,102}")
if sorted(seeds) != list(range(1, 31)):
    problems.append("campaign seed set is not exactly 1..30")

# Name the 14 exactly.  A glob sweeps in debug_*.yaml and the trace config, and
# `n_seeds` lives under `experiment:`, not at the top level -- reading the wrong
# key returned None for all 18 and reported a failure that was the check's own.
root = pathlib.Path("experiments/configs")
cfgs = [root / f"{m}_{s}.yaml" for m in ("udfs", "bingo") for s in sorted(CAMPAIGN)]
missing_files = [str(p) for p in cfgs if not p.exists()]
if missing_files:
    problems.append(f"missing config files: {missing_files}")
cfgs = [p for p in cfgs if p.exists()]

declared = {}
for c in cfgs:
    d = yaml.safe_load(c.read_text()) or {}
    declared[c.name] = (d.get("experiment") or {}).get("n_seeds")

if len(cfgs) != 14:
    problems.append(f"expected 14 campaign configs, found {len(cfgs)}")
bad_n = {k: v for k, v in declared.items() if v != 30}
if bad_n:
    problems.append(f"configs not at n_seeds=30: {bad_n}")

print("N_CONFIGS=%d" % len(cfgs))
print("PROBLEMS=%d" % len(problems))
for p in problems:
    print("  -", p)
sys.exit(1 if problems else 0)
PYEOF
)"
if [[ $? -eq 0 ]]; then
    ok "G4" "A5: seeds 1..30, 0 excluded, disjoint from {0,101,102}, 14/14 configs at n_seeds=30"
else
    bad "G4" "A5: $(sed -n 's/^  - //p' <<<"${SEED_OUT}" | paste -sd'; ')"
fi

# ---------------------------------------------------------------------------
# G5 -- D3: T04 Mode 1 replay, the hash arm's soundness veto
# ---------------------------------------------------------------------------
D3_JSON="${LOCAL_BASE}/c2_stage_d/c2_preflight/stage_d3_mode1_replay.json"
D3_OUT="$("${PY}" - "${D3_JSON}" <<'PYEOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print("unreadable:", exc); sys.exit(1)
bad = [k for k in ("hash_soundness_ok", "isalsr_soundness_ok", "replay_fidelity_ok")
       if not d.get(k)]
n = sum(s["load"]["n_lines"] for s in d.get("streams", []))
print(f"{n} records, hash/isalsr/fidelity ok" if not bad else "failed: " + ",".join(bad))
sys.exit(1 if bad else 0)
PYEOF
)"
if [[ $? -eq 0 ]]; then
    ok "G5" "D3: ${D3_OUT}"
else
    bad "G5" "D3: ${D3_OUT:-report missing at ${D3_JSON}}"
fi

# ---------------------------------------------------------------------------
# G6 -- Stage C: a GO verdict on a SINGLE-provenance root
# ---------------------------------------------------------------------------
SC_OUT="$(ssh "${REMOTE}" "cat ${SMOKE_ROOT}/c2_preflight/stage_c_certification.json 2>/dev/null" \
          | "${PY}" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("no stage_c_certification.json"); sys.exit(1)
v = d.get("verdict") or d.get("overall_verdict")
nb = d.get("n_blocking_failures", -1)
print(f"verdict={v} blocking_failures={nb}")
sys.exit(0 if (v == "GO" and nb == 0) else 1)
' 2>/dev/null)"
if [[ $? -eq 0 ]]; then
    ok "G6" "Stage C: ${SC_OUT}"
else
    bad "G6" "Stage C: ${SC_OUT:-no GO verdict on ${SMOKE_ROOT}}"
fi

# ---------------------------------------------------------------------------
# G7 -- Stage E: the analyzer accepts the root WITHOUT --allow-mixed-provenance
# ---------------------------------------------------------------------------
# This is the obligation Stage E carried forward: it passed on v4 only with the
# override, because the provenance guard rediscovered v4's own dirty split.
LOCAL_SMOKE="${LOCAL_BASE}/$(basename "${SMOKE_ROOT}")"
if [[ ! -d "${LOCAL_SMOKE}" ]]; then
    bad "G7" "Stage E: ${LOCAL_SMOKE} not mirrored locally -- cannot check provenance"
else
    if PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" -m experiments.models.analyze \
           --results-dir "${LOCAL_SMOKE}" --methods udfs,bingo \
           --benchmarks nguyen,feynman,hard,cherrypicked,roundoff,feynman_remainder,strogatz \
           --variants baseline,hash,isalsr > /tmp/c2_stage_e_provenance.log 2>&1; then
        ok "G7" "Stage E: analyzer accepts the root with NO provenance override"
    else
        bad "G7" "Stage E: analyzer refused (see /tmp/c2_stage_e_provenance.log) -- do NOT tag"
    fi
fi

# ---------------------------------------------------------------------------
# G8 -- the campaign root is empty (never mix a re-launch into a live root)
# ---------------------------------------------------------------------------
N_EXISTING="$(ssh "${REMOTE}" "find ${CAMPAIGN_ROOT} -name run_log.json 2>/dev/null | wc -l" 2>/dev/null || echo -1)"
if [[ "${N_EXISTING}" == "-1" ]]; then
    bad "G8" "campaign root: could not be read on the cluster"
elif [[ "${N_EXISTING}" == "0" ]]; then
    ok "G8" "campaign root: empty (${CAMPAIGN_ROOT})"
else
    bad "G8" "campaign root holds ${N_EXISTING} run_log.json -- resume is additive, but confirm this is intended"
fi

# ---------------------------------------------------------------------------
# G9 -- the plan resolves to exactly 42 arrays / 12,600 tasks
# ---------------------------------------------------------------------------
PLAN="$(PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" -m experiments.scripts.c2_slot_plan \
        --config-dir "${REPO_LOCAL}/experiments/configs" --seeds 1-30 --budget 2016 --tsv 2>/dev/null)"
N_ROWS="$(wc -l <<<"${PLAN}")"
N_TASKS="$(awk -F'\t' '{s+=$4} END {print s+0}' <<<"${PLAN}")"
if [[ "${N_ROWS}" == "${EXPECTED_ARRAYS}" && "${N_TASKS}" == "${EXPECTED_TASKS}" ]]; then
    ok "G9" "plan: ${N_ROWS} arrays, ${N_TASKS} tasks"
else
    bad "G9" "plan: ${N_ROWS} arrays / ${N_TASKS} tasks, expected ${EXPECTED_ARRAYS} / ${EXPECTED_TASKS}"
fi

# ---------------------------------------------------------------------------
# G10 -- A13: live inode headroom, and sbatch accepts all 42 at campaign shape
# ---------------------------------------------------------------------------
# Runs the real launcher's own guard on the cluster, so this measures the same
# arithmetic that will refuse at submit time rather than a copy of it.
TESTONLY="$(ssh "${REMOTE}" "bash -l -c 'cd ${REPO_REMOTE} && \
    C2_PROFILE=campaign bash slurm/c2_smoke/launcher.sh --test-only 2>&1'" 2>/dev/null || echo)"
N_OK="$(grep -c ' OK$' <<<"${TESTONLY}" || true)"
INODE_LINE="$(grep 'inode preflight' <<<"${TESTONLY}" | head -1)"
if grep -q 'FATAL' <<<"${TESTONLY}"; then
    bad "G10" "A13: $(grep -m1 'FATAL' <<<"${TESTONLY}")"
elif [[ "${N_OK}" == "${EXPECTED_ARRAYS}" ]]; then
    ok "G10" "A13: ${N_OK}/${EXPECTED_ARRAYS} accepted;${INODE_LINE#*inode preflight:}"
else
    bad "G10" "A13: only ${N_OK}/${EXPECTED_ARRAYS} arrays accepted by sbatch --test-only"
fi

# ---------------------------------------------------------------------------
# G11 -- P2: HEAD differs from the certified wave only in NON-EXECUTABLE files
# ---------------------------------------------------------------------------
# The tag procedure's P2 demands the tag sit on the commit Stage C ran.  Taken
# literally that forces a fresh 1,260-task wave for a typo in a markdown file,
# which is why it gets quietly ignored the first time it is inconvenient -- and
# a rule that gets ignored protects nothing.
#
# So enforce P2's INTENT mechanically instead: the commit that runs must be
# byte-identical to the certified one in every path that can execute during a
# run.  Documentation may move; `src/`, `experiments/`, `benchmarks/` and the
# deployed SLURM scripts may not.  The certified commit is read from the RUN
# LOGS, not from a note, so it cannot be asserted into being true.
CERTIFIED_COMMIT="$(ssh "${REMOTE}" "find ${SMOKE_ROOT} -name run_log.json -print0 2>/dev/null \
    | xargs -0 grep -ho '\"git_describe\": \"[^\"]*\"' 2>/dev/null \
    | sort -u | head -2" 2>/dev/null | sed 's/.*: "//;s/"//' | tr '\n' ' ')"
N_DISTINCT="$(wc -w <<<"${CERTIFIED_COMMIT}")"

if [[ "${N_DISTINCT}" -eq 0 ]]; then
    bad "G11" "P2: could not read git_describe from the certified wave's run logs"
elif [[ "${N_DISTINCT}" -gt 1 ]]; then
    bad "G11" "P2: the wave spans ${N_DISTINCT} commits (${CERTIFIED_COMMIT}) -- certification void"
else
    CC="$(tr -d ' ' <<<"${CERTIFIED_COMMIT}")"
    if ! git -C "${REPO_LOCAL}" rev-parse --verify -q "${CC}^{commit}" >/dev/null; then
        bad "G11" "P2: certified commit ${CC} is not in this repository"
    else
        RUNTIME_DIFF="$(git -C "${REPO_LOCAL}" diff --name-only "${CC}" HEAD -- \
            src/ experiments/ benchmarks/ slurm/c2_smoke/ pyproject.toml 2>/dev/null)"
        if [[ -n "${RUNTIME_DIFF}" ]]; then
            bad "G11" "P2: HEAD changes executable paths since ${CC}: $(tr '\n' ' ' <<<"${RUNTIME_DIFF}")"
        else
            ALL_DIFF="$(git -C "${REPO_LOCAL}" diff --name-only "${CC}" HEAD 2>/dev/null | wc -l)"
            ok "G11" "P2: HEAD == ${CC} in every executable path (${ALL_DIFF} non-executable file(s) differ)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# G12 -- the CAMPAIGN payload decodes.  `--test-only` never runs the worker.
# ---------------------------------------------------------------------------
# 🔴 This gate exists because its absence cost the whole first submission.  On
# 2026-08-06 all 12,600 campaign tasks died within seconds of starting:
# the campaign profile ships `C2_SEEDS=1-30`, the worker counted
# comma-separated fields, saw ONE, and aborted.  Four Stage C waves had passed
# and G10 had reported 42/42 accepted -- because Stage C runs the SMOKE profile
# ("0,101,102") and `sbatch --test-only` validates a job's resource request
# without ever executing its payload.
#
# So: exercise the worker's own seed arithmetic against the CAMPAIGN spec, and
# decode real task indices through the real task-spec module.  No queue time.
CAMPAIGN_SEEDS="$(sed -n 's/^ *DEF_SEEDS="\(1-[0-9]*\)".*/\1/p' \
                  "${REPO_LOCAL}/slurm/c2_smoke/launcher.sh" | head -1)"
AWK_PROG="$(sed -n "/N_SEEDS_DECODED=\$(awk -F, '/,/^}' <<</p" \
            "${REPO_LOCAL}/slurm/c2_smoke/worker.sh" \
            | sed -e "s/^.*awk -F, '//" -e "s/'\s*<<<.*$//")"

if [[ -z "${CAMPAIGN_SEEDS}" || -z "${AWK_PROG}" ]]; then
    bad "G12" "payload: could not extract the campaign seed spec or the worker's guard"
else
    N_DECODED="$(awk -F, "${AWK_PROG}" <<<"${CAMPAIGN_SEEDS}" 2>/dev/null)"
    N_EXPECTED=30
    FIRST="$(PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" -m experiments.scripts.c2_task_spec \
             --config "${REPO_LOCAL}/experiments/configs/bingo_nguyen.yaml" \
             --seeds "${CAMPAIGN_SEEDS}" --index 1 2>/dev/null)"
    LAST="$(PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" -m experiments.scripts.c2_task_spec \
            --config "${REPO_LOCAL}/experiments/configs/bingo_nguyen.yaml" \
            --seeds "${CAMPAIGN_SEEDS}" --index 360 2>/dev/null)"
    if [[ "${N_DECODED}" != "${N_EXPECTED}" ]]; then
        bad "G12" "payload: worker guard decodes '${CAMPAIGN_SEEDS}' to ${N_DECODED}, expected ${N_EXPECTED} -- every task would abort"
    elif [[ -z "${FIRST}" || -z "${LAST}" ]]; then
        bad "G12" "payload: c2_task_spec could not decode index 1 or 360 at '${CAMPAIGN_SEEDS}'"
    else
        ok "G12" "payload: '${CAMPAIGN_SEEDS}' -> ${N_DECODED} seeds; idx 1='${FIRST}', idx 360='${LAST}'"
    fi
fi

# ---------------------------------------------------------------------------
echo ""
echo "==========================================================================="
printf 'Stage F pre-flight: %d PASS, %d FAIL\n' "${N_PASS}" "${N_FAIL}"
if (( N_FAIL > 0 )); then
    printf 'Blocking: %s\n' "${FAILED_IDS[*]}"
    echo "VERDICT: NO-GO -- nothing may be submitted."
    echo "==========================================================================="
    exit 1
fi
echo "VERDICT: GO -- every blocking check passed."
echo "Submit with: bash slurm/c2_campaign/launch.sh"
echo "==========================================================================="
exit 0
