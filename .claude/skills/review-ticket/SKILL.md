---
name: review-ticket
description: |
  Drive one TPAMI revision ticket from `.claude/notes/review/tasks/` to completion.
  You are the orchestrator: you hold the plan, verify every result, run local smoke
  tests, submit and monitor Picasso jobs, and write the ticket's work log and
  proposed answer. Investigation and coding are delegated to at most two Opus
  subagents at a time, each given a pre-digested context brief so it never has to
  rediscover what you already know. Triggers on "work ticket T0x", "run T0x",
  "complete T0x", "drive the revision ticket", "review-ticket", "work the TPAMI
  ticket", "close T0x".
---

# review-ticket — drive one revision ticket to completion

You own one ticket at a time. The subagents hold the *work*; you hold the
*judgment*: whether a result is true, whether a job is safe to submit, whether a
decision belongs to a human. That split is the whole point — your context stays
clean enough to still be making good decisions on hour six.

**Everything in `.claude/notes/review/tasks/` is under an external deadline
(2026-09-24) and will be read by three reviewers who checked every number in the
last round. Correctness beats speed, and an honest negative result beats a
convenient one.**

---

## Non-negotiables

- **At most 2 subagents running at once.** No exceptions.
- **Never trust a subagent's closing claim.** Re-run its check yourself, in the
  main tree, before you believe a number.
- **Never let a subagent submit to Picasso, edit the ticket file, or decide scope.**
  Those three are yours.
- **Never submit to Picasso without a green local smoke run first** (§5).
- **Never write or edit a SLURM script without invoking the `picasso-sbatch`
  skill first** (§5.0). It is the authority on partitions, constraints and
  wallclock limits, and the values recorded elsewhere in this repo have gone stale
  before.
- **Never close a ticket owned by Ezequiel or Karl** (§2). Prepare the material,
  then stop and hand over.
- **Never poll in a loop.** Background agents and background Bash notify you on
  completion. To wait on an external condition, background a single `until` loop.
- **Never mark an acceptance criterion met without the evidence in hand.** "Tests
  pass" is not evidence that a canonical string is byte-identical.

---

## 1. Load the ticket

```bash
ls .claude/notes/review/tasks/
```

Read, in this order and nothing else yet:

1. `.claude/notes/review/tasks/README.md` — roles, dependency spine, decisions
   already taken.
2. The ticket itself, in full.
3. Only the files in the ticket's **Mandatory reading** section that you personally
   need to plan. The rest you will hand to subagents as *pointers*, not read yourself.

Then check the ticket's `Depends on` field against the other tickets' `Status`.
A dependency that is not `COMPLETE` gates you. If the ticket is gated, say what
gates it and stop — do not start work that will be invalidated.

---

## 2. Ownership gate — check before planning

| Ticket | Owner | What you may do |
|---|---|---|
| T01, T02, T04, T05, T06, T08, T09, T10 | Mario (+ you) | Drive to completion. |
| T03 | Ezequiel (design) + Mario (impl) | Phase 1 design doc needs Ezequiel's sign-off **before** Phase 2. Do Phases 2–5; stop at the gate. |
| T07 | Ezequiel | **Do not write the proofs.** Prepare the empirical half (§5.4 tests, counterexamples), then hand over. |
| T11 | Karl + Ezequiel | Run the cross-reference walk and produce the checklist; leave the preprint decision and the prose to them. |
| T12, T13 | Karl | Produce the page ledger inputs and the automated checks; leave the editorial judgement to Karl. |
| T14 | Mario | Assembly only; every input must already be complete. |

When you hit a hand-over gate: write what you produced into the ticket's Work log,
state plainly what remains and who owns it, and stop. Do not fill someone else's
§Proposed answer.

---

## 3. Plan — decompose into atomic subtasks

Write the plan into the ticket's Work log **before** spawning anything. It is the
first entry and it is what you will be judged against.

A subtask is atomic when it satisfies all four:

- **One deliverable.** A file, a number, a table, a passing test — not "investigate X".
- **Verifiable without re-reading the ticket.** You can state the check in one sentence.
- **Bounded file lane.** You can name every file it may write, and there are ≤ 3.
- **No decisions.** If completing it requires choosing what "correct" means, it is
  yours, not a subagent's.

Classify each subtask:

| Kind | Agent | Isolation |
|---|---|---|
| Read code / logs / results, return a compact finding | `revision-investigator` | none (read-only) |
| Write code, tests, configs, analysis scripts | `revision-implementer` | see §4.3 |
| Submit, monitor, verify, decide, escalate, write the ticket | **you** | — |

Order by dependency, then fill at most 2 slots. **Prefer one reader + one writer
concurrently** — they cannot collide.

---

## 4. Delegate

### 4.1 The context brief — this is the token lever

Subagents are expensive. The single biggest cost driver is a subagent
re-deriving context you already hold. **Pre-digest it.** Give a self-contained
brief, not a reading list, and explicitly forbid the expensive reads.

Use this template verbatim:

```
## Goal (one deliverable)
<what must exist when you are done, in one sentence>

## Acceptance check
<the exact command or observation that proves it, and its expected result>

## What is already established — do not re-derive
<3–10 bullets of digested fact: file:line pointers, measured numbers, decisions
already taken, invariants that apply. This is the part that saves tokens.>

## Files you may READ (do not read others without saying why)
<≤5 paths, each with one line on what to look for>

## Files you may WRITE
<≤3 paths>

## Constraints
- Conda env: ~/.conda/envs/isalsr/bin/python
- Do NOT read: the ticket file, .claude/notes/review/source/*, docs/md_files/**
  unless listed above. I have already read them and digested what you need.
- Do NOT run anything longer than 10 minutes. If a check needs more, stop and
  return BLOCKED with the command you would have run.
- Do NOT submit to SLURM, rsync to picasso, git commit, or git push.
- Do NOT edit any .claude/notes/review/tasks/*.md file.
- Applicable Critical Invariants from CLAUDE.md: <list only the numbers that apply>

## Return protocol
End your final message with exactly one of:
  STATUS: DONE — <what you produced, ≤5 lines, with the numbers>
  STATUS: QUESTION — <the single blocking question>
  STATUS: PREMISE-FALSE — <what in the brief is contradicted, with evidence>
  STATUS: BLOCKED — <what stopped you>
Return data, not prose. No summaries of what you read.
```

`STATUS: PREMISE-FALSE` is a **successful** outcome. A subagent that discovers the
brief is wrong has saved the revision from a wrong answer; treat it that way.

### 4.2 Spawning

```
Agent(
  subagent_type: "revision-investigator" | "revision-implementer",
  model: "opus",
  run_in_background: true,        # the default; keeps them visible in the agents view
  description: "<3-5 words>",
  prompt: "<the filled brief above>"
)
```

Both agent definitions pin `model: opus` and `effort: high` in their frontmatter;
passing `model: "opus"` here is belt-and-braces against an inherited override.

Report **one line per launch**, then go quiet.

### 4.3 Isolation rule

Default: **work in place, one writer at a time.** This session is configured to
edit the working directory directly, and `isalsr` is an editable install, so a
second writer in a worktree is a trap unless you handle the import path.

Use `isolation: "worktree"` **only** when you genuinely need two writers at once,
and only when:

- neither touches `src/isalsr/core/_native/` (a worktree cannot see the built
  extension without a rebuild, so its test results would be fiction); and
- you tell each agent to run with `PYTHONPATH=<its worktree>/src` prepended, because
  `pip install -e` resolves to the **main** checkout and its tests will otherwise
  silently exercise main's code.

If either condition fails, serialize. A wrong number costs more than an hour.

---

## 5. The Picasso loop — yours alone

Compute tickets are T02, T03 (Phase 4), T04, T05. Read
`.claude/notes/review/tasks/EXECUTION-PLAN.md` first — it is authoritative on which
wave you are launching, and its §2 certification gate (G1–G8) must pass before any
array goes out. **Nothing launches unless you are 100% sure the code is correct.**
A failing array is caught in minutes; a subtly wrong one is caught during analysis
in September, and that costs the deadline.

The sequence below is not optional and not reorderable.

### 5.0 Gate: invoke `picasso-sbatch` first

Before creating or editing **any** launcher or worker script:

```
Skill(skill: "picasso-sbatch")
```

It is the source of truth for partitions, `--constraint`, GPU/CPU selection flags
and wallclock limits. IsalSR runs are **CPU-only** — never request a `--gres`.
Follow its launcher/worker split and its defensive conda activation; the existing
`slurm/*_launch.sh` files in this repo are the shape to match, but the skill wins
on any conflict.

### 5.1 Local smoke — hard gate, ≤ 10 minutes

Nothing goes to the cluster until a real run completes locally.

```bash
~/.conda/envs/isalsr/bin/python -m experiments.models.orchestrator \
    --config experiments/configs/<cfg>.yaml --seeds 1 --problems Nguyen-1
```

Override `max_time` to ~120 s for the smoke. It must:

- exit 0,
- write a `run_log.json` that **parses and contains the fields the analyzer reads**
  (not merely exists — a truncated log is the exact failure mode the orchestrator's
  resume logic was hardened against), and
- for IsalSR variants, show a non-zero dedup count.

A smoke that only proves "it started" has proved nothing. If it exceeds 10 minutes,
shrink the problem, not the check.

### 5.2 Sync

```bash
rsync -avz --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude 'results' --exclude '.hypothesis' --exclude 'build' \
  ./ picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR/
```

Then verify remotely that the code actually landed and imports:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
  git rev-parse HEAD 2>/dev/null; ls slurm/workers/'
```

### 5.3 Dry-run, then one task, then the array

Three stages. Do not skip the middle one — it is what catches the errors that only
appear on a compute node.

```bash
ssh picasso 'sbatch --test-only <worker>.sh'        # 1. validates without queueing
ssh picasso 'sbatch --array=1-1 <worker>.sh'        # 2. ONE real task, cluster smoke
# ... wait for it, read its logs, only then:
ssh picasso 'sbatch --array=1-<N> <worker>.sh'      # 3. the campaign
```

Most launchers here already expose `--dry-run` and `--experiment <group>`; use them.

### 5.4 Monitor for early errors

The first minutes decide whether a 12-hour campaign is worth waiting for. Check for
import errors, missing files, OOM kills, and immediate exits — not for results.

```bash
ssh picasso 'squeue -u mpascual -o "%.10i %.9P %.20j %.2t %.10M %R"'
ssh picasso 'tail -n 40 ~/execs/isalsr/logs/*_%j.err'
```

Run this as a **backgrounded** `until` loop or via `Monitor`; never a polling loop
in the foreground. Escalate immediately on: `ModuleNotFoundError`, `FileNotFoundError`,
`oom-kill`, `CANCELLED`, or any task that exits in under a minute.

**Kill early and fix.** A 300-task array failing identically is 300 wasted
allocations and a day of queue time.

Detailed commands, log paths and failure signatures: `references/picasso-loop.md`.

---

## 6. Verify — the part that cannot be delegated

On every subagent return, in order:

1. **Non-DONE first.**
   - `QUESTION` → relay to the human with `AskUserQuestion`, then `SendMessage` the
     answer back to that agent id; its context is intact and re-spawning wastes it.
   - `PREMISE-FALSE` → read the evidence yourself. If it holds, this changes the
     plan. Record it in the Work log, surface it, and re-plan. Never push an agent
     to implement something you now know is wrong.
   - `BLOCKED` → decide whether you unblock it or the human does.

2. **Re-run the acceptance check yourself**, in the main tree:
   ```bash
   ~/.conda/envs/isalsr/bin/python -m pytest tests/ -q
   ~/.conda/envs/isalsr/bin/python -m ruff check src/ tests/
   ~/.conda/envs/isalsr/bin/python -m mypy src/isalsr/
   ```
   Compare every number to what the agent claimed. A mismatch is the agent's
   result, not yours — send it back with the exact diff.

3. **Judge against the ticket's own acceptance criteria**, clause by clause. Ask:
   was a new test shown to fail against the pre-fix code? Was out-of-scope work
   filed rather than absorbed? Was a premise checked or assumed?

4. **Write the Work log entry.** `### YYYY-MM-DD — <topic>`: what was decided, what
   broke, what surprised you. AC-0 on every ticket makes this mandatory, and it is
   the entry you will want in round 2.

### Iteration budget

Send specific, reproducible defects back with `SendMessage`. **Two rounds maximum.**
After a second unsuccessful round, stop and ask the human. An agent grinding on a
brief whose premise is wrong will never converge, and telling it to try harder is
how you lose a day.

---

## 7. Escalate to the human

Use `AskUserQuestion` — do not decide these yourself:

- Anything the ticket or `EXECUTION-PLAN.md` marks as a decision — currently T03's
  insertion point, the open baseline question (`EXECUTION-PLAN.md` §5), T03's
  2026-08-31 go/no-go, and T13's page trades.
- **Any array submission.** Report the certification-gate evidence (G1–G8) and get a
  go before `sbatch` on more than one task. Early stopping is abandoned and the
  arXiv v3 is settled — do not reopen either.
- A `PREMISE-FALSE` that invalidates a ticket's stated basis.
- A result that would change what the paper claims — especially a **negative** one.
  Tickets T03 §5 Phase 5, T04 AC-8 and T10 §4 all have honest-negative branches;
  surface them, never soften them.
- Any compute request above ~5,000 core-hours that is not already in the ticket.
- A second failed iteration round.

---

## 8. Close

A ticket is complete only when **all** of these hold:

- Every acceptance criterion is met, with evidence you personally re-ran.
- AC-0's Work log is filled — decisions, dead ends, surprises, disagreements.
- §Proposed answer is filled: the before/after table has real numbers in both
  columns, the manuscript-change table names files and lines, and the LaTeX draft
  is written in `response_to_reviewers.tex` register with every claim backed by a
  number the ticket produced.
- The Residual-risk subsection names what a round-2 reviewer could still object to.
- The ticket's `Status` line is updated, and any ticket it **Blocks** is notified in
  your final message.

Then report, and stop. Do not roll on to the next ticket without being asked.

---

## 9. Verbosity

Emit only:

- one line per launch — `▶ T04/investigate-dedup-streams launched (opus)`
- any `QUESTION` or `PREMISE-FALSE`, in full, immediately
- one short block per return: verdict, the checks you re-ran, ≤ 3 lines of
  substance, what you did with it
- every Picasso stage transition, one line, with the job id
- anything red, in full
- the close report

Say nothing else. Do not narrate waiting, polling, or your own scheduling
arithmetic.

---

## 10. Related skills

| Skill | When |
|---|---|
| `picasso-sbatch` | **Mandatory** before creating or editing any SLURM script (§5.0). |
| `research-rigor` | Before proposing a new metric, ablation, statistical test, or eval protocol inside a ticket. T04's three-arm correction and T05's pre-registration both warrant it. |
| `humanizer` | Any §Proposed answer draft over ~200 words, and any manuscript prose. Scientific mode. |
| `test-and-verify` | The full pytest + ruff + mypy + hypothesis-alignment pass after a code-bearing ticket. |
| `server3` | Only if a ticket needs the GPU workstation. No current ticket does — IsalSR is CPU-only. |
