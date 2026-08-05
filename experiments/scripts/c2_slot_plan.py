"""Plan the 42 C2 arrays: task counts, array throttles, memory and wall clock.

Single source of truth for the resource shape of campaign C2, so the launcher
never carries a hard-coded table and every number here is unit-testable.

Why this module exists — the allocation finding (2026-08-05)
-----------------------------------------------------------
C2 ships as 42 arrays, one per ``(method, arm, suite)`` (EXECUTION-PLAN §1).
The arrays carry very unequal work: ``udfs:*:strogatz`` is 280 tasks x 12.0 h
while ``bingo:*:feynman_remainder`` is 120 tasks x ~5 h — a 5.4x spread. Under a
single uniform ``%K`` every array gets the same number of slots, so the small
ones drain early and hand their share back to nobody: the remaining arrays are
still capped at ``K``. The campaign makespan is then set by the largest array
alone.

An array of ``N`` tasks of duration ``T`` under throttle ``K`` finishes at
``N*T/K``. Minimising ``max_i N_i*T_i/K_i`` subject to ``sum_i K_i <= C`` gives
``K_i proportional to N_i*T_i`` — every array then finishes at the same time,
``(sum_i N_i*T_i) / C``, which is the information-theoretic floor for a fixed
slot budget. Measured against the uniform allocation at the same total slots this
is a **1.9x** makespan reduction, at 20 or 30 seeds and at every budget level,
and it changes no configuration content whatsoever.

Reconstructing the Stage C ``%24`` waves from ``sacct`` showed why the uniform
allocation was never visible before: those waves have 1.25 tasks per slot, so
they are ramp-up and drain almost end to end (peak concurrency reached 93 % of
the ceiling but held it for 2.7 % of the span). At C2's 8.3 tasks per slot the
ramp disappears and the allocation is the whole story.

Sensitivity
-----------
The split needs ``T_bingo``, which is the uncertain input: F-19 raised three
suites' ``max_evals`` tenfold and the 20 D2 problems have no runtime data at all.
Planning at ``T_bingo = 8 h`` and evaluating against the truth gives 1.65x over
uniform at ``T_bingo = 4 h``, 1.65x at 5.15 h, 1.65x at 8 h, 1.33x at 10 h and
1.11x at 12 h. **It never loses to uniform anywhere in the plausible range**,
which is why an uncertain ``T_bingo`` is not a reason to keep the uniform split.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.orchestrator import parse_seeds  # noqa: E402
from experiments.scripts.c2_task_spec import TaskSpecError, load_problem_names  # noqa: E402

#: Submission order.  UDFS first, deliberately: it is the long pole at a flat
#: 12.00 h per run, and it is also the arm that benefits most from being queued
#: while our fairshare factor is at its highest (PriorityWeightFairShare = 50000
#: with a 14-day decay half-life, so priority erodes as the campaign burns).
METHODS: tuple[str, ...] = ("udfs", "bingo")
ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")
SUITES: tuple[str, ...] = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)

#: Per-run duration used to weight the slot allocation, in hours.
#:
#: ``udfs`` 12.0 — MEASURED, not assumed: C1 n=600 gives mean = median = max =
#: 12.00 h.  UDFS has no ``max_evals`` (its knobs are ``max_time`` and
#: ``max_orders``) and saturates the full budget on 100 % of runs.
#:
#: ``bingo`` 8.0 — a deliberate over-estimate of the 5.15 h measured on C1
#: (n=564, corroborated by Stage D's 4.57 h over 8 cells).  Bingo stops on
#: ``max_evals = 100M``, never on ``max_time``, and F-19 raised three suites from
#: 10M to 100M, so the C1 mean understates C2.  See the sensitivity note above:
#: planning high is the safe direction.
RUNTIME_HOURS: dict[str, float] = {"udfs": 12.0, "bingo": 8.0}

#: Per-run duration actually observed, used ONLY to report the expected outcome.
#:
#: Kept separate from :data:`RUNTIME_HOURS` on purpose. The allocation is
#: weighted with a conservative ``T_bingo`` because planning high is the safe
#: direction (§ sensitivity above); the *expectation* should be quoted at what
#: was measured, or the plan reads as a 63 h campaign when 54 h is what the data
#: supports. Reporting the pessimistic number as if it were the forecast is how
#: a schedule acquires invisible padding.
#:
#: ``bingo`` 5.15 h: C1 isalsr n=564 mean; Stage D's 8 completed cells give
#: 4.57 h. ⚠ F-19 raised `roundoff`/`strogatz`/`feynman_remainder` from 10M to
#: 100M ``max_evals`` and the 20 D2 problems have no runtime data, so the
#: expectation is a lower bound on those 28 of 70 problems.
MEASURED_RUNTIME_HOURS: dict[str, float] = {"udfs": 12.0, "bingo": 5.15}

#: ``--mem`` per ``(method, arm)``, in GiB.
#:
#: **Bingo-IsalSR is 32 GB, revised down from EXECUTION-PLAN §3.3's 256 GB on the
#: measurement §3.3 itself asks for** ("if D1.2 shows 12 h MaxRSS comfortably
#: under 128 GB ... the request may be revised down before launch, with the
#: measurement recorded").  The measurement, 2026-08-05:
#:
#: * Stage D, full-length dedup-arm cells that stopped on ``max_evals``:
#:   ``MaxRSS`` **1.05–1.16 GB** at 6.3–7.2 M unique canonical hashes.
#: * Stage D baseline-arm cells, which hold no dedup set at all: 0.39–0.42 GB.
#: * ``IsalSRDeduplicator.canonical_seen: set[int]`` is the **only** unbounded
#:   container in the arm (``_parent_ids`` and the per-k histograms are bounded
#:   by ``population_size`` and by the k range).
#: * A candidate cannot enter that set without being scored, so
#:   ``n_unique <= n_total <= max_evals = 100_000_000`` is a **hard ceiling**,
#:   independent of which problem runs.
#: * Measured cost of a ``set[int]`` under the production allocator
#:   (``PYTHONMALLOC=malloc``, ``worker.sh:59``): **81.5 bytes/entry**, flat from
#:   1 M to 32 M entries, with a 1.16x transient at each table resize.
#: * Hence worst-case peak RSS = 100M x 81.5 B x 1.16 + 0.42 GB ~= **9.4 GB**.
#:
#: 32 GB is 3.4x that ceiling and 28x the observed peak.  C1's 127.7 GB OOMs were
#: pymalloc arena fragmentation over 10k+ generations plus a ``set[str]`` at
#: ~150 B/entry; both causes are gone (``PYTHONMALLOC=malloc`` and ``set[int]``).
#: An OOM would in any case be *named* rather than silent, because P4 writes the
#: status record ahead of the search.
MEM_GB: dict[tuple[str, str], int] = {
    ("udfs", "baseline"): 16,
    ("udfs", "hash"): 16,
    ("udfs", "isalsr"): 16,
    ("bingo", "baseline"): 32,
    ("bingo", "hash"): 32,
    ("bingo", "isalsr"): 32,
}

#: SLURM ``--time`` per method.
#:
#: Deliberately generous, and that costs nothing.  Measured 2026-08-05 with
#: ``sbatch --test-only``: the scheduler's estimated start is identical for
#: 13 h, 16 h, 23 h, 2 d 23 h and 6 d 23 h.  The only cliff on this cluster is at
#: **2 h**, where ``short`` (priority 118,933) gives way to ``medium_uma``
#: (28,873) and the estimate jumps by three hours — and no C2 task can be that
#: short.  So there is nothing to win by trimming the wall, and a tight wall
#: risks killing a cell in the post-search SymPy tail (§11.1 2026-08-03: a cell
#: that finished its search correctly then spent 7+ further minutes in SymPy),
#: which would cost a cell under §5.5.  Stay under 3 days: above that the job
#: drops to ``long_uma`` at priority 500 and loses a further 5,000 points.
WALL: dict[str, str] = {"udfs": "0-16:00:00", "bingo": "0-16:00:00"}

#: Default total concurrent array slots across all 42 arrays.
#:
#: 2,016 = a mean of ``%48``.  Justification: the Stage C ``%24`` waves reached a
#: peak of 934 concurrent 1-core ``sr`` tasks within minutes, so ~1,000 is
#: demonstrated; ``sbatch --test-only`` accepts ``%24``/``%48``/``%96``/``%280``;
#: the QOS entitlement is ``cpu = 9000`` per user and ``sr``'s usable pool is
#: 13,312 cores.  2,016 is 22 % of the entitlement and 15 % of ``sr``.
#:
#: ⚠ The Stage C figures were measured under the ``short`` QOS at priority
#: 118,933; C2 runs at 28,873.  The grant at 2,016 is therefore *not* proven —
#: watch achieved concurrency over the campaign's first hours.  Lowering the
#: budget mid-campaign is safe: it touches no config and no deployed file, so it
#: is not the "deploy is a config edit" defect.
DEFAULT_SLOT_BUDGET = 2016


class SlotPlanError(Exception):
    """Raised when a plan cannot be built from the configs on disk."""


@dataclass(frozen=True)
class ArrayPlan:
    """Resource plan for one ``(method, arm, suite)`` SLURM array."""

    method: str
    arm: str
    suite: str
    n_tasks: int
    throttle: int
    mem_gb: int
    wall: str
    runtime_h: float

    @property
    def key(self) -> str:
        """Return the ``method:arm:suite`` identifier used by the launcher."""
        return f"{self.method}:{self.arm}:{self.suite}"

    @property
    def work_h(self) -> float:
        """Return the array's total core-hours, ``n_tasks * runtime_h``."""
        return self.n_tasks * self.runtime_h

    @property
    def finish_h(self) -> float:
        """Return the array's makespan under its throttle, ``work / throttle``."""
        return self.work_h / self.throttle


def allocate_throttles(works: list[float], caps: list[int], budget: int) -> list[int]:
    """Apportion ``budget`` slots across arrays in proportion to their work.

    Largest-remainder (Hamilton) apportionment with a floor of one slot per array
    and a ceiling of that array's own task count — more slots than tasks would be
    unusable, and zero slots would stall the array forever. Slots freed by a
    capped array are redistributed to whichever uncapped array currently has the
    worst finish time, which is exactly the quantity being minimised.

    Args:
        works: Total work per array, in arbitrary but consistent units.
        caps: Maximum useful slots per array, i.e. its task count.
        budget: Total slots to distribute.

    Returns:
        Slots per array, summing to ``min(budget, sum(caps))``.

    Raises:
        SlotPlanError: If the inputs disagree in length, are empty, or the budget
            cannot cover one slot per array.
    """
    if len(works) != len(caps):
        raise SlotPlanError(f"works/caps length mismatch: {len(works)} vs {len(caps)}")
    if not works:
        raise SlotPlanError("empty plan")
    if budget < len(works):
        raise SlotPlanError(f"budget {budget} cannot give one slot to each of {len(works)} arrays")

    total_work = sum(works)
    if total_work <= 0:
        raise SlotPlanError("total work is zero")

    exact = [budget * w / total_work for w in works]
    slots = [max(1, min(cap, int(e))) for e, cap in zip(exact, caps, strict=True)]

    # Hand out the remainder to the arrays that finish last, one slot at a time.
    # This is O(budget) in the worst case but budget is a few thousand and the
    # loop terminates as soon as every array is capped.
    while sum(slots) < budget:
        candidates = [i for i in range(len(slots)) if slots[i] < caps[i]]
        if not candidates:
            break
        worst = max(candidates, key=lambda i: (works[i] / slots[i], -i))
        slots[worst] += 1

    # Over-allocation can only come from the floor of one slot per array.
    while sum(slots) > budget:
        candidates = [i for i in range(len(slots)) if slots[i] > 1]
        if not candidates:
            break
        best = min(candidates, key=lambda i: (works[i] / slots[i], i))
        slots[best] -= 1

    return slots


def build_plan(
    config_dir: str,
    seeds: list[int],
    budget: int = DEFAULT_SLOT_BUDGET,
    *,
    uniform: int | None = None,
) -> list[ArrayPlan]:
    """Build the resource plan for all 42 arrays.

    Args:
        config_dir: Directory holding ``{method}_{suite}.yaml``.
        seeds: Campaign seeds; only the count is used for sizing.
        budget: Total concurrent array slots to apportion.
        uniform: If given, ignore ``budget`` and give every array this many
            slots. Reproduces the pre-2026-08-05 behaviour for A/B work.

    Returns:
        One :class:`ArrayPlan` per ``(method, arm, suite)``, in submission order.

    Raises:
        SlotPlanError: If a config is missing or declares no problems.
    """
    n_seeds = len(seeds)
    if n_seeds < 1:
        raise SlotPlanError("need at least one seed")

    sizes: dict[tuple[str, str], int] = {}
    for method in METHODS:
        for suite in SUITES:
            path = os.path.join(config_dir, f"{method}_{suite}.yaml")
            if not os.path.isfile(path):
                raise SlotPlanError(f"missing config: {path}")
            try:
                n_problems = len(load_problem_names(path))
            except TaskSpecError as exc:
                raise SlotPlanError(str(exc)) from exc
            if n_problems < 1:
                raise SlotPlanError(f"{path}: no problems in suite {suite}")
            sizes[(method, suite)] = n_problems

    triples = [(m, a, s) for m in METHODS for a in ARMS for s in SUITES]
    counts = [sizes[(m, s)] * n_seeds for m, _, s in triples]
    works = [c * RUNTIME_HOURS[m] for c, (m, _, _) in zip(counts, triples, strict=True)]

    if uniform is not None:
        slots = [max(1, min(cap, uniform)) for cap in counts]
    else:
        slots = allocate_throttles(works, counts, budget)

    return [
        ArrayPlan(
            method=m,
            arm=a,
            suite=s,
            n_tasks=n,
            throttle=k,
            mem_gb=MEM_GB[(m, a)],
            wall=WALL[m],
            runtime_h=RUNTIME_HOURS[m],
        )
        for (m, a, s), n, k in zip(triples, counts, slots, strict=True)
    ]


def format_plan_tsv(plan: list[ArrayPlan]) -> str:
    """Render the plan as tab-separated rows for the launcher to read.

    Args:
        plan: Arrays in submission order.

    Returns:
        One ``method\\tarm\\tsuite\\tn_tasks\\tthrottle\\tmem_gb\\twall`` row per
        array, without a header, so the shell can read it with a bare ``while
        read``.
    """
    return "\n".join(
        f"{p.method}\t{p.arm}\t{p.suite}\t{p.n_tasks}\t{p.throttle}\t{p.mem_gb}\t{p.wall}"
        for p in plan
    )


def format_plan_table(plan: list[ArrayPlan]) -> str:
    """Render the plan as a human-readable table with the makespan summary.

    Args:
        plan: Arrays in submission order.

    Returns:
        A table plus the derived campaign totals.
    """
    lines = [
        f"{'array':32s} {'tasks':>6s} {'%K':>5s} {'mem':>6s} {'wall':>11s} "
        f"{'work_h':>9s} {'finish_h':>9s}",
    ]
    for p in sorted(plan, key=lambda q: -q.finish_h):
        lines.append(
            f"{p.key:32s} {p.n_tasks:6d} {p.throttle:5d} {p.mem_gb:5d}G "
            f"{p.wall:>11s} {p.work_h:9.0f} {p.finish_h:9.1f}"
        )
    total_work = sum(p.work_h for p in plan)
    total_slots = sum(p.throttle for p in plan)
    makespan = max(p.finish_h for p in plan)

    # The same plan, scored at the runtimes actually observed.  The allocation is
    # weighted pessimistically on purpose; the forecast should not be.
    exp_work = sum(p.n_tasks * MEASURED_RUNTIME_HOURS[p.method] for p in plan)
    exp_makespan = max(p.n_tasks * MEASURED_RUNTIME_HOURS[p.method] / p.throttle for p in plan)
    lines += [
        "",
        f"  arrays              : {len(plan)}",
        f"  tasks               : {sum(p.n_tasks for p in plan):,}",
        f"  slots               : {total_slots:,}",
        "",
        "  -- allocation basis (T_bingo = "
        f"{RUNTIME_HOURS['bingo']:.2f} h, deliberately pessimistic) --",
        f"  core-hours          : {total_work:,.0f}",
        f"  makespan            : {makespan:.1f} h = {makespan / 24:.2f} days",
        f"  packing efficiency  : {100 * total_work / total_slots / makespan:.0f} % "
        f"(floor {total_work / total_slots:.1f} h)",
        "",
        "  -- EXPECTED, at measured runtimes (T_bingo = "
        f"{MEASURED_RUNTIME_HOURS['bingo']:.2f} h) --",
        f"  core-hours          : {exp_work:,.0f}",
        f"  makespan            : {exp_makespan:.1f} h = {exp_makespan / 24:.2f} days",
        f"  mean concurrency    : {exp_work / exp_makespan:,.0f} cores",
    ]
    return "\n".join(lines)


def format_rebalance(plan: list[ArrayPlan], job_ids: list[str]) -> str:
    """Emit ``scontrol update`` lines that re-apportion a RUNNING campaign.

    The allocation is weighted with a deliberately pessimistic ``T_bingo``
    because F-19 raised three suites' ``max_evals`` tenfold and the D2 suites
    have no runtime data. If the first day shows Bingo finishing near the 5.15 h
    that C1 measured, its arrays drain early and idle their share for the rest of
    the campaign — at 30 seeds that is the difference between a 63 h and a 54 h
    makespan.

    ``scontrol update JobId=<id> ArrayTaskThrottle=<n>`` fixes that in flight
    (verified available on Picasso's SLURM 25.05.1). It touches no config and no
    file in the deployed tree, so it is **not** the "a deploy is a config edit"
    defect and is safe to run mid-campaign.

    Args:
        plan: The plan as submitted, in submission order.
        job_ids: SLURM job ids in the same order, i.e. the contents of the
            launcher's ``job_ids.txt``.

    Returns:
        One ``scontrol`` line per array, plus a comment header.

    Raises:
        SlotPlanError: If the counts disagree — applying a shifted mapping would
            silently throttle the wrong arrays.
    """
    if len(job_ids) != len(plan):
        raise SlotPlanError(
            f"{len(job_ids)} job ids for {len(plan)} arrays; refusing to guess the mapping"
        )
    lines = [
        f"# Re-apportion {sum(p.throttle for p in plan)} slots at "
        f"T_bingo = {RUNTIME_HOURS['bingo']:.2f} h.",
        "# Safe mid-campaign: touches no config and no deployed file.",
    ]
    for p, jid in zip(plan, job_ids, strict=True):
        lines.append(
            f"scontrol update JobId={jid} ArrayTaskThrottle={p.throttle}"
            f"   # {p.key}, finishes {p.finish_h:.1f} h"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured argument parser.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Plan the 42 C2 arrays")
    parser.add_argument(
        "--config-dir",
        default=os.path.join(here, "..", "configs"),
        help="Directory holding {method}_{suite}.yaml",
    )
    parser.add_argument("--seeds", required=True, help="Seed spec, e.g. '1-20' or '0,101,102'")
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_SLOT_BUDGET,
        help=f"Total concurrent array slots (default {DEFAULT_SLOT_BUDGET})",
    )
    parser.add_argument(
        "--uniform",
        type=int,
        default=None,
        help="Give every array this many slots instead of apportioning (A/B only)",
    )
    parser.add_argument(
        "--bingo-hours",
        type=float,
        default=None,
        help="Override the assumed Bingo per-run duration used to WEIGHT the "
        "allocation. Use with --rebalance once the campaign's first day has "
        "shown what Bingo actually costs under the F-19 budget.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tsv", action="store_true", help="Emit tab-separated rows for the shell")
    group.add_argument("--table", action="store_true", help="Emit a human-readable table")
    group.add_argument(
        "--rebalance",
        metavar="JOB_IDS_TXT",
        help="Emit `scontrol update ArrayTaskThrottle` lines for a RUNNING "
        "campaign, given the launcher's job_ids.txt (submission order). Safe "
        "mid-campaign: touches no config and no deployed file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the array plan as TSV or as a table.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` if the plan could not be built.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.bingo_hours is not None:
            if args.bingo_hours <= 0:
                raise SlotPlanError(f"--bingo-hours must be positive, got {args.bingo_hours}")
            RUNTIME_HOURS["bingo"] = args.bingo_hours
        plan = build_plan(
            args.config_dir,
            parse_seeds(args.seeds),
            args.budget,
            uniform=args.uniform,
        )
        if args.rebalance:
            with open(args.rebalance) as handle:
                job_ids = [line.strip() for line in handle if line.strip()]
            payload = format_rebalance(plan, job_ids)
        elif args.tsv:
            payload = format_plan_tsv(plan)
        else:
            payload = format_plan_table(plan)
    except (SlotPlanError, ValueError, OSError) as exc:
        # stderr only: the launcher substitutes stdout into sbatch arguments, so
        # a partial line on the failure path would submit a malformed array.
        print(f"c2_slot_plan: {exc}", file=sys.stderr)
        return 1

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
