# Cutting the `campaign/c2` tag

**Status**: procedure only. **The tag is not created by this document, and not by
any agent.** SP-0 is explicit: only Mario submits C2 and only Mario tags it. This
file exists so that when the moment comes, nothing has to be reconstructed from
memory.

**Check**: EXECUTION-PLAN §4.1 A1 ("Freeze the commit").

---

## 1. Why the tag cannot be cut early

A1 requires the tag to sit on **the exact commit C2 will run**. Two consequences
that have already bitten this campaign:

- The tag is not a milestone marker. If any commit lands after the tag, the tag
  is wrong and must be moved — and a moved annotated tag that has been pushed is
  a rewrite that every clone must be told about.
- Stage C certification is only evidence for the commit it ran on. Cutting the
  tag on a commit Stage C never exercised means A1 passes while proving nothing.
  This is why §11.2 records A1 as deliberately OPEN, with Stage C running on
  recorded commits (`53a1c1c` → `5f282cc`) instead.

Per audit.md §6 decision 4, the audit branch merges **before** the tag, and the
Stage C wave is re-run on the merged commit (≈33 min at `%24` on the `sr` pool)
before the tag is cut.

---

## 2. Preconditions — all five, verified in order

Do not proceed past a failing step.

### P1 — the audit branch is merged
`feature/experiment-fairness-audit` is merged into the campaign branch, and T17
has closed C4. Nothing is left in flight.

```bash
git -C <repo> log --oneline -1
git -C <repo> branch --merged | grep experiment-fairness-audit
```

### P2 — Stage C re-certified ON THE MERGED COMMIT
Not on a predecessor. The certification wave must have run, drained, and passed
on the commit about to be tagged.

```bash
# the commit Stage C ran, read from the run logs, must equal HEAD
git -C <repo> rev-parse HEAD
grep -h "SP-1 commit" <stage_c_logs>/*.out | sort -u
```
One unique commit, equal to `HEAD`. More than one value means the wave spanned a
push and the certification is void.

### P3 — A2 suite green
```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/
python -m mypy --strict src/isalsr/
```
Expected: full suite passing with the recorded skip count; `mypy` clean; `ruff`
clean on `src/` and `tests/`. The **444 pre-existing** `experiments/models/`
violations (N806, E501) are identical at HEAD and are not a blocker — but check
the count has not grown.

### P4 — working tree clean
```bash
git -C <repo> status --porcelain
```
Must print nothing. Not "nothing important" — nothing.

### P5 — the MANIFEST validates
```bash
python -m experiments.models.manifest validate <campaign_root>/MANIFEST.json
```
Must exit `0`. The MANIFEST's `build.git_commit` must equal the commit being
tagged and its `build.git_tag` must read `campaign/c2` — so write the MANIFEST
**after** deciding the commit and **before** pushing the tag, then re-validate.
Also confirm at this point (Stage F item 6): quota headroom re-read live on the
day, not from `c2_preflight/storage_projection_v2.md`.

---

## 3. The commands

Run from the repository root, on the merged commit, with P1–P5 all green.

```bash
# 1. Confirm what is about to be frozen.
git rev-parse HEAD
git log --oneline -1
git status --porcelain          # must be empty

# 2. Create the ANNOTATED tag (never lightweight: the campaign needs the
#    tagger, the date and the message to be part of the object).
git tag -a campaign/c2 -m "Campaign C2: three-arm re-execution on the native engine.

Arms:      baseline, hash, isalsr
Seeds:     1..30
Alphabet:  decomposed (T16)
Engine:    native
Node pool: sr
Cohort:    70 problems (D1 50 + D2 20), 42 arrays, 12,600 runs
Scope:     canonical completeness claimed for k >= 1 (D3, 2026-08-06)
Certified: Stage C on this commit; MANIFEST validates."

# 3. Verify the tag resolves to the intended commit BEFORE pushing.
git rev-parse campaign/c2^{commit}     # must equal step 1's HEAD
git show --stat campaign/c2 | head -20

# 4. Push the tag.
git push origin campaign/c2

# 5. Confirm it landed.
git ls-remote --tags origin campaign/c2
```

Then record in §11.2 that A1 has moved from OPEN to PASS, with the commit SHA
and the date, and re-run P5 so the MANIFEST's `git_tag` is verified against a
tag that now exists.

---

## 4. If the tag has to move

It should not. If it must (a defect found between tagging and submission):

```bash
git tag -d campaign/c2
git push origin :refs/tags/campaign/c2
# fix, re-run Stage C on the new commit, re-run P1-P5, then re-tag
```

Moving the tag **invalidates the Stage C certification**. Re-run the wave. A tag
moved without re-certification is the failure mode A1 exists to prevent.

---

## 5. Out of scope for agents

Creating, pushing, moving or deleting this tag, and submitting C2. Agents may
verify preconditions and report; the tag itself is Mario's, after Stage F
sign-off (§4.6), which is signed by Mario.
