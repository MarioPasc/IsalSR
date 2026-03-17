---
name: test-and-verify
description: |
  Run the full IsalSR verification pipeline: pytest, ruff, mypy. Then check that
  test names and coverage relate to the central hypothesis (O(k!) search space
  reduction via canonical strings). Use after any code change.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Test and Verify Pipeline

Run the full IsalSR verification pipeline in the `isalsr` conda environment and report results.

## Step 1: Run pytest

```bash
cd /home/mpascual/research/code/IsalSR && ~/.conda/envs/isalsr/bin/python -m pytest tests/ -v --tb=short 2>&1
```

Report: total passed, failed, errors, skipped.

## Step 2: Run ruff

```bash
cd /home/mpascual/research/code/IsalSR && ~/.conda/envs/isalsr/bin/python -m ruff check src/ tests/ 2>&1
```

Report: number of issues, or "All checks passed".

## Step 3: Run mypy

```bash
cd /home/mpascual/research/code/IsalSR && ~/.conda/envs/isalsr/bin/python -m mypy src/isalsr/ 2>&1
```

Report: number of errors, or "Success".

## Step 4: Hypothesis Alignment Check

Grep through test files for keywords that indicate alignment with the central hypothesis:
- "canonical" -- tests for canonical string invariance
- "isomorphi" -- tests for isomorphism equivalence
- "cycle" -- tests for DAG acyclicity
- "roundtrip" or "round_trip" -- tests for S2D/D2S round-trip
- "topological" -- tests for valid evaluation order

Report which hypothesis-relevant test categories exist and which are still missing.

## Output Format

Present results as a summary table:

| Check | Status | Details |
|-------|--------|---------|
| pytest | PASS/FAIL | X passed, Y failed |
| ruff | PASS/FAIL | X issues |
| mypy | PASS/FAIL | X errors |
| Hypothesis coverage | X/5 | Which categories present |

If ALL pass, conclude with: "All verification checks passed."
If any fail, list the failures and suggest fixes.
