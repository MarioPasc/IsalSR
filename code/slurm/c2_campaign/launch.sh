#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- the final review experiments.  42 arrays, 12,600 runs.
# =============================================================================
#   bash slurm/c2_campaign/launch.sh --dry-run     # print the 42 arrays
#   bash slurm/c2_campaign/launch.sh --test-only   # sbatch --test-only, ON Picasso
#   bash slurm/c2_campaign/launch.sh               # gate, then SUBMIT
#
# This script deliberately contains no #SBATCH header, no worker and no
# topology.  It is a gate plus a delegation to slurm/c2_smoke/launcher.sh under
# C2_PROFILE=campaign -- the same launcher, the same worker and the same slot
# plan the Stage C wave certified.  A self-contained copy here would be
# uncertified code at the moment 108,000 core-hours are committed.  See
# README.md §1 and EXECUTION-PLAN §10.2.
#
# SP-0 stands: C2 is submitted once, by Mario, after Stage F sign-off.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REMOTE="${ISALSR_REMOTE:-picasso}"
REPO_REMOTE="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          MODE="submit" ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac

# --- The gate. -------------------------------------------------------------
# Only a real submission is gated: --dry-run and --test-only exist precisely to
# be run BEFORE the gate can pass (the tag is cut last), and gating them would
# make the preview unusable during pre-flight.
if [[ "${MODE}" == "submit" ]]; then
    echo "Running Stage F pre-flight before submitting ..."
    echo ""
    if ! bash "${SCRIPT_DIR}/stage_f_preflight.sh"; then
        echo ""
        echo "REFUSING TO SUBMIT: the Stage F gate did not pass." >&2
        echo "Nothing was submitted.  Fix the blocking checks and re-run." >&2
        exit 1
    fi
    echo ""
    echo "Gate passed.  Submitting Campaign C2 ..."
    echo ""
fi

# --- Delegate. -------------------------------------------------------------
# --dry-run is the one mode meaningful on the workstation: it neither needs
# sbatch nor reads the cluster quota.  Everything else runs on Picasso, from
# the DEPLOYED tree, so that the worker path sbatch records is the deployed
# worker and not this checkout's (defect 15, and SP-1).
if [[ "${MODE}" == "dry" ]]; then
    C2_PROFILE=campaign bash "${REPO_LOCAL}/slurm/c2_smoke/launcher.sh" --dry-run
    exit 0
fi

FLAG=""
[[ "${MODE}" == "test" ]] && FLAG="--test-only"

ssh "${REMOTE}" "bash -l -c 'cd ${REPO_REMOTE} && \
    C2_PROFILE=campaign bash slurm/c2_smoke/launcher.sh ${FLAG}'"

if [[ "${MODE}" == "submit" ]]; then
    cat <<'EOF'

---------------------------------------------------------------------------
Campaign C2 submitted.  Next actions, in order:

  1. Record the 42 job ids in EXECUTION-PLAN §11.3 (the launcher wrote them to
     <logs>/job_ids.txt in submission order, which is §11.3's row order).

  2. Watch achieved concurrency for the first six hours.  The plan assumes the
     scheduler grants 2,016 slots; the `short` QOS granted 934, and C2 runs at
     4.1x lower priority with fairshare eroding as it burns.  Lowering
     C2_SLOT_BUDGET mid-campaign is safe -- it touches no config and no
     deployed file, so it is not defect 10.

  3. After ~24 h, read the measured Bingo wall clock and rebalance:
       python -m experiments.scripts.c2_slot_plan \
              --bingo-hours <measured> --rebalance <logs>/job_ids.txt
     Recovers the ~9 h the deliberately pessimistic T_bingo = 8 h weighting
     costs.  scontrol re-apportions a RUNNING array; not defect 10.

  4. Do NOT deploy anything to the repo while these arrays run.  A mid-wave
     redeploy splits provenance across two HEADs and marks every subsequent
     cell -dirty (defect 10; it cost v4 161 of its 1,260 cells).
---------------------------------------------------------------------------
EOF
fi
