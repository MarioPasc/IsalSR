# A13 — FSCRATCH inode headroom, second measurement

**Date**: 2026-08-04
**Check**: EXECUTION-PLAN §4.1 A13 (storage and file-count projection)
**Supersedes**: `c2_preflight/storage_projection.md` (2026-07-31 measurement)
**Operator**: agent, under the authorisation limits recorded in §4 below

---

## 1. Requirement

C2 writes ≥5 files per run × 8,400 runs ≈ **42,000 files**, plus the Stage C
certification wave (1,260 runs ≈ 7,600 files). Working figure: **≈45,000 files**
of FSCRATCH headroom needed before the first array.

A13's stated pass criterion is stricter than the raw requirement:

> `quota` shows **≥60,000 files** of FSCRATCH headroom **and HOME under its soft
> quota**, before Stage C.

Both halves are graded. They are reported separately below because they now have
different verdicts.

---

## 2. Quota captures

### 2.1 Before (2026-08-04, immediately prior to any action)

```
		 Space	 Limits	 			 File	 Limits
	 used	 quota	 limit	grace	  ║	 files	 quota	 limit	grace
home	 0.34TB	 0.28TB	 0.75TB	2 days	  ║	  14.4k	  35.0k	 150.0k	none
fscratc	 0.46TB	 1.40TB	 1.68TB	none	  ║	 248.6k	 250.0k	 400.0k	none

-> Notice: You have exceeded your storage quota on Home.
```

FSCRATCH headroom to the **soft** quota: 250.0k − 248.6k = **1.4k files**.
Headroom to the hard limit: 400.0k − 248.6k = 151.4k, but the soft quota is the
operative ceiling — crossing it starts a grace clock.

### 2.2 After (2026-08-04, same session, post-action)

```
		 Space	 Limits	 			 File	 Limits
	 used	 quota	 limit	grace	  ║	 files	 quota	 limit	grace
home	 0.34TB	 0.28TB	 0.75TB	2 days	  ║	  14.4k	  35.0k	 150.0k	none
fscratc	 0.44TB	 1.40TB	 1.68TB	none	  ║	 155.4k	 250.0k	 400.0k	none

-> Notice: You have exceeded your storage quota on Home.
```

FSCRATCH headroom to the soft quota: 250.0k − 155.4k = **94.6k files**.

---

## 3. Verdict

| Half of A13 | Requirement | Measured | Verdict |
|---|---|---|---|
| FSCRATCH file headroom | ≥60.0k (criterion); ≈45k (raw need) | **94.6k** | **PASS**, with the attribution caveat in §5 |
| HOME under soft quota | 0.28 TB | **0.34 TB, 2 days grace** | **FAIL — still a live blocker** |

**A13 is not closed.** The FSCRATCH half passes; the HOME half does not, and its
grace window is 2 days. HOME cleanup is explicitly outside this agent's
authorisation (§4) and is Mario's to perform.

---

## 4. Actions taken, and the limits they were taken under

Authorisation was restricted to: capture quota; archive-then-delete the two
superseded smoke roots; inventory (read-only) everything else. In particular
`~/execs/vena`, HOME, and all non-isalsr data were out of scope and **were not
touched**.

### 4.1 Archived and deleted (the only deletions performed)

`c2_smoke_v3` supersedes both earlier smoke roots — same topology, re-certified,
on the pinned pool, with the corrected `I.34.27`. Both predecessors were
archived to a single sibling file each, verified, and only then removed.

| Root | Archive | Archive size | Members listed | Verified | Deleted |
|---|---|---|---|---|---|
| `…/results/isalsr/c2_smoke` | `c2_smoke_v1.tar.gz` | 484,824,775 B | 7,932 | size > 0 and members > 1000 | yes |
| `…/results/isalsr/c2_smoke_v2` | `c2_smoke_v2.tar.gz` | 532,594,422 B | 7,932 | size > 0 and members > 1000 | yes |

Each root collapsed from 7,932 inodes to 1. **Inodes freed by this action:
15,862.** Both archives remain in place and are restorable.

`c2_smoke_v3` was **not** touched.

### 4.2 Not touched

`~/execs/vena`, HOME, `fscratch/conda_envs`, `fscratch/datasets`,
`fscratch/checkpoints`, and every non-isalsr path. No deletion outside §4.1.

---

## 5. Attribution caveat — read this before trusting the 94.6k

The measured drop is **93.2k inodes** (248.6k → 155.4k). The authorised deletion
accounts for **15.9k** of that. The remaining **≈77k was not caused by this
agent** and cannot be attributed from the evidence collected.

Two observations bearing on it:

- `fscratch/conda_pkgs` measured **39,310** inodes in the pre-action inventory
  and **21,799** afterwards — a 17.5k drop this agent did not cause. This is
  consistent with an external `conda clean`, a scheduled cache eviction, or
  concurrent activity in another session.
- That still leaves ≈60k unaccounted. The most likely explanation is that the
  filesystem's quota accounting is updated asynchronously and the 248.6k reading
  was stale, i.e. real usage was already below 248.6k before any action.

**Consequence**: the 94.6k figure should not be treated as a durable measurement.
This is not a new concern — Stage F item 6 already requires quota headroom to be
"re-read live on the day, not from an earlier capture". This capture reinforces
that requirement rather than satisfying it.

---

## 6. Inventory of remaining candidates — proposed, not deleted

Read-only `du --inodes`, post-action. **No action taken on any of these.**

| Path | Inodes | Assessment |
|---|---|---|
| `fscratch/conda_envs` | 103,614 | **Do not touch.** Live environments, including the one C2 runs from. |
| `fscratch/tools/synthseg_env` | 22,636 | Non-isalsr (VENA/SynthSeg). Mario's call; out of this agent's scope. |
| `fscratch/conda_pkgs` | 21,799 | **Best candidate.** Package cache, fully regenerable by conda. `conda clean --all` frees most of it with no loss. Was 39,310 pre-action. |
| `fscratch/results/isalsr/c2_smoke_v3` | 7,932 | **Archive after Stage C sign-off**, not before — it is the live certification root. Same tar-then-delete recipe as §4.1 frees 7,931. |
| `fscratch/repos/VENA-validation` | 5,921 | Non-isalsr. Not this agent's to propose deleting; listed for completeness. |
| `fscratch/repos/VENA` | 4,513 | Non-isalsr. As above. |
| `fscratch/repos/IsalSR` | 4,291 | **Keep.** C2 runs from this checkout. |
| `fscratch/repos/IsalHG` | 1,775 | Sibling project; low yield. |
| `fscratch/repos/slim-diff` | 1,196 | Low yield. |
| `fscratch/results/isalsr/c2_preflight` | 153 | Keep — pre-flight evidence. |
| `fscratch/datasets` | 152 | Keep. |
| `fscratch/checkpoints` | 28 | Keep. |

**Recommended next action for Mario, in order:**

1. **HOME under soft quota within 2 days.** This is the binding blocker, and it
   is not an inode problem — HOME is at 14.4k/35.0k files but 0.34 TB against a
   0.28 TB space quota. It is a *space* cleanup, not a file-count one.
2. `conda clean --all` on `fscratch/conda_pkgs` (≈21.8k inodes, zero risk).
3. Archive `c2_smoke_v3` (≈7.9k inodes) **after** Stage C sign-off.
4. Re-read `quota` live on the submission day (Stage F item 6) and record it in
   the sign-off, superseding this capture.

---

## 7. The ≥15,000-file rule

A13 also requires that, per the site's ≥15,000-file rule, C2 either consolidate
per-run output into one archive or **mail `soporte@scbi.uma.es` before the first
array**. C2 writes ≈42,000 files, so this applies regardless of headroom.
**Still outstanding — not addressed by this measurement**, and it is a
correspondence action, not a cleanup one.
