"""Plan the 42 C2 arrays: chunking, task counts, throttles, memory and wall clock.

Single source of truth for the resource shape of campaign C2, so the launcher
never carries a hard-coded table and every number here is unit-testable.

Why the arrays are chunked — the SCBI request (2026-08-07)
---------------------------------------------------------
Picasso's administrators asked us to stop submitting short jobs:

    "me da la impresión de que los trabajos apenas duran minutos. Por lo que el
    sistema pierde más tiempo buscándoles hueco para ejecutar que en la propia
    ejecución. Esto satura el sistema de colas y afecta a todos los usuarios.
    Para estos casos lo ideal es agrupar varios trabajos en uno solo, de forma
    que cada trabajo combinado dure al menos 2h."   -- Manuel, SCBI

They are right, and it is measurable in our own accounting. Of the 23,058 job
records this account produced between 2026-07-01 and 2026-08-07, **51.5 % ran
for under two minutes and only 0.9 % reached two hours**. Two things produced
that: the Stage C smoke waves run a 900 s payload, and the aborted 2026-08-06
campaign submission had all 12,600 tasks die in seconds on the seed-spec guard.

It is *also* true of the production payload, which is what matters here. From
the C1 campaign's own ``sacct`` record (COMPLETED tasks only):

===========================  =======  =========  =========  =============
array                          n      mean (h)   p50 (h)    frac < 2 h
===========================  =======  =========  =========  =============
``udfs`` hard / cherry / rd    1,680      12.01      12.01      0.0 %
``udfs`` feynman               4,924       0.82       0.03    ~93 %
``bingo`` nguyen               5,608       0.14       0.02     100 %
``bingo`` feynman                297       0.20       0.02     100 %
``bingo`` hard                   564       5.27       5.96      26 %
``bingo`` roundoff               479       0.69       0.73     100 %
===========================  =======  =========  =========  =============

So the campaign is genuinely bimodal: UDFS saturates its 12 h budget on the
suites it cannot solve exactly and exits in seconds on the ones it can, while
Bingo stops on ``max_evals`` and is fast on the easy suites. Roughly 43 % of the
12,600 planned tasks would have run for under two hours.

The fix is to make one array task run a **contiguous chunk of cells in
sequence** instead of a single cell. Three properties make this nearly free:

1. **It is makespan-neutral.** An array of ``N`` cells of duration ``T`` under
   throttle ``K`` finishes at ``N*T/K``. Grouping into ``N/B`` tasks of duration
   ``B*T`` finishes at ``(N/B)*(B*T)/K = N*T/K`` — the same, provided the array
   still has at least ``K`` tasks, which :func:`build_plan` enforces by capping
   the apportionment at the post-chunking task count. Simulated end to end
   against the measured per-suite distributions, total campaign wall-clock moves
   from ~160 h to ~170 h, i.e. within the noise of the throttle grant itself.
2. **Memory is unaffected.** Each cell is a separate ``orchestrator`` process, so
   peak RSS is still a per-cell quantity and ``--mem`` does not change.
3. **A lost task costs one cell, not a chunk.** Every cell writes its
   ``run_log.json`` as it completes, and the orchestrator's resume logic skips
   completed cells, so a task killed at hour 30 loses only the cell in flight.

The cost is a **deadline**: a chunk cannot be allowed to overrun its wall. The
worker therefore refuses to *start* a cell unless the full payload budget still
fits (:data:`CELL_RESERVE_H`), which makes a SLURM ``TIMEOUT`` impossible by
construction and keeps the "no task was killed by SLURM" assertion meaningful.
Cells that do not get started are picked up by a dependent sweep pass; simulated
sweep sizes are 5-20 tasks against a first pass of ~3,400.

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
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.orchestrator import parse_seeds  # noqa: E402
from experiments.scripts.c2_task_spec import (  # noqa: E402
    TaskSpecError,
    load_problem_names,
    n_tasks_for,
)

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

#: Mean per-cell duration in hours, per ``(method, suite)``, used to size chunks.
#:
#: Measured from the C1 campaign's ``sacct`` record (COMPLETED tasks, 2026-03-01
#: to 2026-07-01), adjusted where C2's configuration differs from C1's. This is
#: a *finer* table than :data:`RUNTIME_HOURS`, which is per-method only: the
#: per-method figure cannot size a chunk, because ``udfs`` ranges from 0.82 h per
#: cell on ``feynman`` to a flat 12.01 h on ``hard``.
#:
#: Provenance, suite by suite:
#:
#: * ``udfs`` ``hard`` / ``cherrypicked`` / ``roundoff`` — n = 1,680, mean = p50 =
#:   p99 = 12.01 h. UDFS has no ``max_evals``; it stops on ``stop_thresh = 1e-10``
#:   or saturates ``max_time``, and on these suites it never stops early.
#: * ``udfs`` ``feynman`` — n = 4,924, mean 0.82 h, p50 0.03 h, p95 6.88 h,
#:   p99 12.02 h. UDFS recovers most Feynman equations exactly and exits in
#:   seconds; the ~7 % it cannot recover saturate the budget. **Bimodal, and the
#:   reason the deadline guard exists.**
#: * ``udfs`` ``nguyen`` — n = 4,450, p95 = p99 = 12.00 h. The 1 h-budget runs
#:   (``*_atlas``, n = 150) hit their cap on 98 % of cells, so under C2's uniform
#:   43,200 s budget essentially every cell saturates.
#: * ``bingo`` ``nguyen`` / ``feynman`` — n = 5,905, mean 0.06-0.22 h, max 4.76 h.
#:   ``max_evals`` binds far below ``max_time``; these never approach the cap.
#: * ``bingo`` ``hard`` / ``cherrypicked`` — n = 1,160, p50 3.7-9.8 h, p90 11.8 h.
#:   Already at 100M ``max_evals`` in C1, so C2 does not shift them.
#: * ``bingo`` ``roundoff`` — C1 measured 0.51-0.87 h at 10M ``max_evals``; F-19
#:   raised it to 100M, a 10x budget increase, so the C2 estimate is scaled and
#:   then clipped by the 12 h ``max_time``.
#: * ``feynman_remainder`` / ``strogatz`` — the 20 D2 problems have **no runtime
#:   data at all**. Estimated conservatively (high for UDFS, mid for Bingo);
#:   :func:`bundle_size` under-chunks rather than over-chunks when the estimate is
#:   too high, so an error here costs task count, never correctness.
CELL_HOURS: dict[tuple[str, str], float] = {
    ("udfs", "nguyen"): 11.5,
    ("udfs", "feynman"): 0.9,
    ("udfs", "hard"): 12.0,
    ("udfs", "cherrypicked"): 12.0,
    ("udfs", "roundoff"): 12.0,
    ("udfs", "feynman_remainder"): 6.0,
    ("udfs", "strogatz"): 11.0,
    ("bingo", "nguyen"): 0.15,
    ("bingo", "feynman"): 0.20,
    ("bingo", "hard"): 5.3,
    ("bingo", "cherrypicked"): 7.5,
    ("bingo", "roundoff"): 5.0,
    ("bingo", "feynman_remainder"): 2.0,
    ("bingo", "strogatz"): 2.0,
}

#: Per-cell payload budget in hours: ``max_time: 43200`` in all fourteen configs.
PAYLOAD_CAP_H = 12.0

#: Hours a chunked task must have left before it may START another cell.
#:
#: The payload cap plus a teardown allowance. The allowance is not decoration:
#: §11.1 (2026-08-03) records a cell that finished its search correctly and then
#: spent seven further minutes in SymPy. Reserving the FULL cap — rather than an
#: observed mean — is what makes a SLURM ``TIMEOUT`` impossible by construction,
#: which in turn keeps the standing "no task was killed by SLURM" assertion a
#: real signal instead of expected noise.
CELL_RESERVE_H = PAYLOAD_CAP_H + 0.5

#: Chunk duration the planner aims for, in hours.
#:
#: The floor that matters is SCBI's 2 h. 36 h is chosen well above it because the
#: per-cell distributions are heavily right-skewed (``bingo:nguyen`` has mean
#: 0.06 h against p50 0.011 h), so sizing a chunk at its *mean* leaves the
#: *median* chunk far shorter. Targeting 36 h tolerates a 18x over-estimate of
#: the per-cell time before a chunk drops under the 2 h floor, and the measured
#: cost of the larger chunks is nil: simulated total wall-clock is flat at
#: ~170 h from a 24 h target to a 48 h one.
TARGET_TASK_HOURS = 36.0

#: The floor SCBI asked for. :func:`build_plan` asserts every array clears it.
#:
#: ⚠ Do not "optimise" the residual 2.5 % of quantised makespan by forcing a
#: minimum task count per array. It was tried (2026-08-07): bounding the chunk so
#: every array keeps at least ``m`` tasks moves the quantised makespan to 48.0 h
#: at ``m = 12`` — exactly the unchunked figure — but the curve is **jagged**,
#: because the cost is ``ceil(n_tasks / throttle)`` and depends on how each
#: array's task count happens to align with its slot share. Over ``m`` in
#: [9, 20] the makespan swings 48.0 / 48.6 / 56.7 / 54.0 / 48.0 h with no
#: monotone structure, so ``m = 12`` is a lucky point, not an optimum, and
#: picking it is overfitting to inputs we do not know that well: ``CELL_HOURS``
#: is an estimate on 20 of the 70 problems, and a ±50 % error there swamps a
#: 2.5 % effect. The unbounded plan also leaves the shortest task at 20 h rather
#: than 4.5 h, i.e. ten times the floor instead of two — which is the margin that
#: actually protects the commitment made to SCBI if the estimates are wrong.
MIN_TASK_HOURS = 2.0

#: Slack between a chunk's EXPECTED duration and the deadline it must fit in.
#:
#: A chunk of ``B`` cells whose expected duration exactly equalled the cutoff
#: would spill about half the time. 1.4 keeps the simulated first-pass spill to
#: 5-20 tasks in ~3,400 against the measured distributions, including
#: ``udfs:feynman``'s 7 % of full-budget cells.
DEADLINE_SAFETY = 1.4

#: Wall bounds, in hours.
#:
#: ``MIN`` 16 h is the certified unchunked wall: it covers the 12 h payload cap
#: with four hours of margin. ``MAX`` 47 h stays a full day inside ``medium_uma``'s
#: ``MaxWall = 3-00:00:00`` (measured 2026-08-07 with ``sacctmgr``), so a chunk
#: never risks demotion to ``long_uma`` at priority 500.
MIN_WALL_H = 16
MAX_WALL_H = 47

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

#: Realised **p90** per-cell wall clock in hours, measured on C2 itself.
#:
#: Source: ``experiments/scripts/measure_cell_hours.py`` over
#: ``$FSCRATCH/results/isalsr/c2_3arm``, 2026-08-09, n = 10,533 cells with a
#: recorded ``results.time.wall_clock_total_s``. These are *not* the C1-derived
#: estimates in :data:`CELL_HOURS`; they are what this campaign actually did.
#:
#: 🔴 Why a second table rather than a correction to :data:`CELL_HOURS`. The two
#: answer different questions. ``CELL_HOURS`` is a *central* estimate and sizes
#: the bundle for throughput; ``P90_CELL_HOURS`` is a *tail* estimate and sizes
#: it for the deadline. The 2026-08-09 audit showed that ``udfs:feynman`` needs
#: both at once: median 0.18 h against p90 12.00 h, a 67x spread inside one
#: suite. A bundle sized on the median is right for throughput and catastrophic
#: for the deadline, which is precisely how ~1,100 cells came to be deferred.
#:
#: Measured 2026-08-09 (p50 / p90 / max, hours):
#:
#: =========================  =====  ======  ======  ======
#: method:suite                   n     p50     p90     max
#: =========================  =====  ======  ======  ======
#: ``udfs`` (every suite)      4,783   12.00   12.00   12.00
#: ``udfs:feynman``              462    0.18   12.00   12.00
#: ``bingo:feynman``             740    0.00    0.05    6.69
#: ``bingo:nguyen``              919    0.03    0.43    3.04
#: ``bingo:strogatz``          1,260    0.09    1.01    3.30
#: ``bingo:feynman_remainder``   435    0.08    4.21    8.64
#: ``bingo:roundoff``            686    2.05    5.55    8.51
#: ``bingo:cherrypicked``        880    3.69    6.47   10.18
#: ``bingo:hard``                830    3.29    7.18   11.76
#: =========================  =====  ======  ======  ======
P90_CELL_HOURS: dict[tuple[str, str], float] = {
    ("udfs", "nguyen"): 12.0,
    ("udfs", "feynman"): 12.0,
    ("udfs", "hard"): 12.0,
    ("udfs", "cherrypicked"): 12.0,
    ("udfs", "roundoff"): 12.0,
    ("udfs", "feynman_remainder"): 12.0,
    ("udfs", "strogatz"): 12.0,
    ("bingo", "nguyen"): 0.43,
    ("bingo", "feynman"): 0.05,
    ("bingo", "hard"): 7.18,
    ("bingo", "cherrypicked"): 6.47,
    ("bingo", "roundoff"): 5.55,
    ("bingo", "feynman_remainder"): 4.21,
    ("bingo", "strogatz"): 1.01,
}

#: Teardown allowance added to a p90 cell when sizing a RECOVERY chunk.
#:
#: The same 0.5 h that :data:`CELL_RESERVE_H` adds to the payload cap, for the
#: same reason: §11.1 (2026-08-03) records a cell that finished its search
#: correctly and then spent seven further minutes in SymPy, and the worker's
#: elapsed clock also carries ``stage_in``/``stage_out`` and orchestrator
#: start-up, none of which appear in ``wall_clock_total_s``.
RECOVERY_TEARDOWN_H = 0.5

#: Recovery sizing modes. See :func:`recovery_allowance_h`.
RECOVERY_MODES: tuple[str, ...] = ("safe", "p90")


class SlotPlanError(Exception):
    """Raised when a plan cannot be built from the configs on disk."""


def cell_hours(method: str, suite: str) -> float:
    """Return the mean per-cell duration used to size chunks and weight slots.

    Args:
        method: ``"udfs"`` or ``"bingo"``.
        suite: Benchmark suite key.

    Returns:
        Hours per ``(problem, seed)`` cell, from :data:`CELL_HOURS`, falling back
        to the coarser per-method :data:`RUNTIME_HOURS` for an unknown suite.
    """
    return CELL_HOURS.get((method, suite), RUNTIME_HOURS[method])


def bundle_size(cell_h: float, n_cells: int, max_bundle: int | None = None) -> int:
    """Return how many cells one array task should run in sequence.

    Two bounds apply. The chunk should last :data:`TARGET_TASK_HOURS`, and it
    must fit under the largest deadline any wall can offer,
    ``MAX_WALL_H - CELL_RESERVE_H``, with :data:`DEADLINE_SAFETY` slack.

    Args:
        cell_h: Mean per-cell duration in hours.
        n_cells: Cells available in the array; a chunk can never exceed it.
        max_bundle: Hard ceiling on the chunk. Used by the Stage C smoke, whose
            wall is ``B x (payload + teardown)``: at the campaign's bundles that
            crosses the 2 h ``short``-QOS cliff (priority 118,933 against
            ``medium_uma``'s 28,873, measured 2026-08-05) and the smoke would
            queue like the campaign it exists to precede.

    Returns:
        Cells per task, at least one.

    Raises:
        SlotPlanError: If ``cell_h`` is not positive or the array is empty.
    """
    if cell_h <= 0:
        raise SlotPlanError(f"cell_hours must be positive, got {cell_h}")
    if n_cells < 1:
        raise SlotPlanError(f"n_cells must be positive, got {n_cells}")

    by_target = max(1, round(TARGET_TASK_HOURS / cell_h))
    by_deadline = max(1, int((MAX_WALL_H - CELL_RESERVE_H) / (DEADLINE_SAFETY * cell_h)))
    bounds = [by_target, by_deadline, n_cells]
    if max_bundle is not None and max_bundle >= 1:
        bounds.append(max_bundle)
    return max(1, min(bounds))


def wall_hours(bundle: int, cell_h: float) -> int:
    """Return the SLURM wall a chunked task should request, in whole hours.

    Sized from whichever bound is tighter:

    * ``bundle * CELL_RESERVE_H`` — the **no-spill** wall, i.e. enough for every
      cell in the chunk to consume its full payload budget. Exact for the suites
      where UDFS provably saturates, and the reason those arrays need no sweep.
    * ``DEADLINE_SAFETY * bundle * cell_h + CELL_RESERVE_H`` — the expected chunk
      plus slack plus one full cell in reserve. This binds for the short-cell
      suites, where the no-spill wall would be absurd (``udfs:feynman`` at
      ``B = 27`` would ask for 337 h to cover a chunk expected to run 24 h).

    Args:
        bundle: Cells per task.
        cell_h: Mean per-cell duration in hours.

    Returns:
        Whole hours, clamped to ``[MIN_WALL_H, MAX_WALL_H]``.
    """
    no_spill = bundle * CELL_RESERVE_H
    expected = DEADLINE_SAFETY * bundle * cell_h + CELL_RESERVE_H
    needed = math.ceil(min(no_spill, expected))
    return max(MIN_WALL_H, min(MAX_WALL_H, needed))


def format_wall(hours: int) -> str:
    """Render whole hours as SLURM's ``D-HH:MM:SS``.

    Args:
        hours: Wall clock in whole hours.

    Returns:
        The ``D-HH:MM:SS`` string.
    """
    return f"{hours // 24}-{hours % 24:02d}:00:00"


@dataclass(frozen=True)
class ArrayPlan:
    """Resource plan for one ``(method, arm, suite)`` SLURM array."""

    method: str
    arm: str
    suite: str
    n_cells: int
    bundle: int
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
        """Return the array's total core-hours, ``n_cells * runtime_h``."""
        return self.n_cells * self.runtime_h

    @property
    def task_h(self) -> float:
        """Return the expected duration of one array task, in hours."""
        return self.bundle * self.runtime_h

    @property
    def wall_h(self) -> int:
        """Return the requested wall in whole hours."""
        days, hms = self.wall.split("-")
        return int(days) * 24 + int(hms.split(":")[0])

    @property
    def start_cutoff_h(self) -> float:
        """Return the last elapsed time at which the worker may start a new cell."""
        return self.wall_h - CELL_RESERVE_H

    @property
    def finish_h(self) -> float:
        """Return the array's makespan under its throttle, ``work / throttle``."""
        return self.work_h / self.throttle

    @property
    def quantised_finish_h(self) -> float:
        """Return the makespan actually achievable, in whole rounds of tasks.

        ``finish_h`` is the continuous ideal. Tasks are indivisible, so an array
        of ``n_tasks`` under throttle ``K`` runs ``ceil(n_tasks / K)`` rounds and
        the last round is not necessarily full. Chunking makes each round longer
        and rarer, so this is where its cost -- if any -- becomes visible; the
        two numbers should be compared whenever :data:`TARGET_TASK_HOURS` moves.
        """
        return math.ceil(self.n_tasks / self.throttle) * self.task_h


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
    chunk: bool = True,
    max_bundle: int | None = None,
) -> list[ArrayPlan]:
    """Build the resource plan for all 42 arrays.

    Args:
        config_dir: Directory holding ``{method}_{suite}.yaml``.
        seeds: Campaign seeds; only the count is used for sizing.
        budget: Total concurrent array slots to apportion.
        uniform: If given, ignore ``budget`` and give every array this many
            slots. Reproduces the pre-2026-08-05 behaviour for A/B work.
        chunk: Group cells into multi-cell tasks (the SCBI request). ``False``
            restores one cell per task for A/B work; it is strictly worse for the
            cluster and produces ~43 % sub-two-hour tasks.
        max_bundle: Hard ceiling on cells per task; see :func:`bundle_size`.

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
    cells = [sizes[(m, s)] * n_seeds for m, _, s in triples]
    per_cell = [cell_hours(m, s) for m, _, s in triples]
    works = [n * c for n, c in zip(cells, per_cell, strict=True)]

    bundles = [
        bundle_size(c, n, max_bundle) if chunk else 1 for n, c in zip(cells, per_cell, strict=True)
    ]
    n_tasks = [n_tasks_for(n, b) for n, b in zip(cells, bundles, strict=True)]

    # 🔴 Apportion against the TASK count, not the cell count. A throttle above
    # an array's task count is unusable, and chunking cuts that count by up to
    # 100x -- so keeping the pre-chunking cap would hand slots to arrays that
    # cannot fill them and starve the ones that can. This is the single line that
    # makes chunking makespan-neutral.
    if uniform is not None:
        slots = [max(1, min(cap, uniform)) for cap in n_tasks]
    else:
        slots = allocate_throttles(works, n_tasks, budget)

    plan = [
        ArrayPlan(
            method=m,
            arm=a,
            suite=s,
            n_cells=n,
            bundle=b,
            n_tasks=t,
            throttle=k,
            mem_gb=MEM_GB[(m, a)],
            wall=format_wall(wall_hours(b, c) if chunk else MIN_WALL_H),
            runtime_h=c,
        )
        for (m, a, s), n, c, b, t, k in zip(
            triples, cells, per_cell, bundles, n_tasks, slots, strict=True
        )
    ]

    # Every cell must be reachable from exactly one task, and no task may request
    # a wall it can overrun. Both are cheap to assert and expensive to discover
    # on the cluster.
    for p in plan:
        if p.n_tasks * p.bundle < p.n_cells:
            raise SlotPlanError(f"{p.key}: {p.n_tasks} tasks x {p.bundle} < {p.n_cells} cells")
        if p.start_cutoff_h <= 0:
            raise SlotPlanError(f"{p.key}: wall {p.wall} leaves no room for one cell")
    return plan


def recovery_allowance_h(method: str, suite: str, mode: str = "safe") -> float:
    """Return the wall a recovery chunk must reserve for ONE of its cells.

    This is the single number the whole recovery design turns on. A chunked task
    starts cell *i* only if ``elapsed < wall - CELL_RESERVE_H``; so if every cell
    is charged an allowance ``a`` and the wall is sized as
    ``(B - 1) * a + CELL_RESERVE_H``, no cell of the chunk can be deferred as
    long as no cell overruns ``a``.

    Two modes, because they buy different guarantees:

    * ``"safe"`` — ``a = CELL_RESERVE_H``. Every cell is charged the **full
      payload cap plus teardown**, which is a hard upper bound on any cell in
      this campaign (``max_time: 43200`` in all fourteen configs). Deferral then
      becomes impossible *by construction*, with **no distributional assumption
      at all**. This is the default, and it is the only mode whose guarantee
      survives the 2026-08-09 finding that a single suite spans 67x.
    * ``"p90"`` — ``a = min(CELL_RESERVE_H, p90 + RECOVERY_TEARDOWN_H)``. Charges
      the measured tail rather than the cap, which buys far larger bundles on the
      suites where Bingo finishes in seconds (``bingo:feynman`` goes from B=3 to
      B=63) and therefore far fewer SLURM placements. The guarantee is only
      probabilistic: a chunk defers if more than a handful of its cells land
      above p90.

    Args:
        method: ``"udfs"`` or ``"bingo"``.
        suite: Benchmark suite key.
        mode: One of :data:`RECOVERY_MODES`.

    Returns:
        Hours to reserve per cell.

    Raises:
        SlotPlanError: If ``mode`` is not a known mode.
    """
    if mode not in RECOVERY_MODES:
        raise SlotPlanError(f"unknown recovery mode {mode!r}; expected one of {RECOVERY_MODES}")
    if mode == "safe":
        return CELL_RESERVE_H
    p90 = P90_CELL_HOURS.get((method, suite), PAYLOAD_CAP_H)
    return min(CELL_RESERVE_H, p90 + RECOVERY_TEARDOWN_H)


def recovery_wall_hours(bundle: int, allowance_h: float) -> int:
    """Return the wall a recovery chunk needs so its LAST cell still starts.

    ``floor(x) + 1`` rather than ``ceil(x)``: the deferral test in
    ``worker.sh:450`` is ``elapsed >= cutoff``, so a wall that makes the cutoff
    exactly equal to the worst-case elapsed time would defer. The extra hour is
    what makes the inequality strict.

    Args:
        bundle: Cells per task.
        allowance_h: Hours reserved per cell, from :func:`recovery_allowance_h`.

    Returns:
        Whole hours, clamped to ``[MIN_WALL_H, MAX_WALL_H]``.

    Raises:
        SlotPlanError: If either argument is not positive.
    """
    if bundle < 1:
        raise SlotPlanError(f"bundle must be positive, got {bundle}")
    if allowance_h <= 0:
        raise SlotPlanError(f"allowance must be positive, got {allowance_h}")
    span = (bundle - 1) * allowance_h + CELL_RESERVE_H
    return max(MIN_WALL_H, min(MAX_WALL_H, math.floor(span) + 1))


def defers_nothing(bundle: int, allowance_h: float, wall_h: int) -> bool:
    """Return whether a chunk of ``bundle`` cells can reach its last cell.

    Args:
        bundle: Cells per task.
        allowance_h: Hours reserved per cell.
        wall_h: The requested SLURM wall, in whole hours.

    Returns:
        ``True`` when the worker's start-deadline strictly exceeds the worst-case
        elapsed time at which the last cell of the chunk begins.
    """
    cutoff_h = wall_h - CELL_RESERVE_H
    return cutoff_h > (bundle - 1) * allowance_h


def recovery_bundle(allowance_h: float, n_cells: int, forced: int | None = None) -> int:
    """Return the largest chunk whose last cell is still guaranteed to start.

    Largest, not smallest: every cell of a recovery chunk that is already
    complete costs a resume check and nothing else, so the binding cost is the
    number of SLURM placements — which is exactly what SCBI asked us to reduce.
    The bundle is therefore pushed up until the no-deferral inequality or the
    47 h wall ceiling stops it.

    Args:
        allowance_h: Hours reserved per cell, from :func:`recovery_allowance_h`.
        n_cells: Cells in the array; a chunk can never exceed it.
        forced: Operator override. Still validated against the inequality; a
            forced bundle that cannot clear it is a caller error, not a warning.

    Returns:
        Cells per task, at least one.

    Raises:
        SlotPlanError: If the inputs are not positive, or ``forced`` is a bundle
            whose last cell could be deferred.
    """
    if allowance_h <= 0:
        raise SlotPlanError(f"allowance must be positive, got {allowance_h}")
    if n_cells < 1:
        raise SlotPlanError(f"n_cells must be positive, got {n_cells}")

    if forced is not None:
        if forced < 1:
            raise SlotPlanError(f"forced bundle must be positive, got {forced}")
        bundle = min(forced, n_cells)
        if not defers_nothing(bundle, allowance_h, recovery_wall_hours(bundle, allowance_h)):
            raise SlotPlanError(
                f"forced bundle {bundle} at {allowance_h:.2f} h/cell needs a wall of "
                f"{(bundle - 1) * allowance_h + CELL_RESERVE_H:.1f} h, above the "
                f"{MAX_WALL_H} h ceiling -- it would defer exactly what this pass exists to fix"
            )
        return bundle

    bundle = min(n_cells, int((MAX_WALL_H - CELL_RESERVE_H) // allowance_h) + 1)
    while bundle > 1 and not defers_nothing(
        bundle, allowance_h, recovery_wall_hours(bundle, allowance_h)
    ):
        bundle -= 1
    return bundle


def match_selector(key: str, patterns: list[str]) -> bool:
    """Return whether ``method:arm:suite`` matches any ``*``-globbed pattern.

    Args:
        key: An :attr:`ArrayPlan.key`, i.e. ``method:arm:suite``.
        patterns: Selectors such as ``udfs:*:feynman``. A field of ``*`` matches
            anything; a pattern with fewer than three fields is rejected by
            :func:`build_recovery_plan` before it reaches here.

    Returns:
        ``True`` if any pattern matches every field.
    """
    fields = key.split(":")
    for pattern in patterns:
        parts = pattern.split(":")
        if len(parts) == len(fields) and all(
            p == "*" or p == f for p, f in zip(parts, fields, strict=True)
        ):
            return True
    return False


def build_recovery_plan(
    config_dir: str,
    seeds: list[int],
    *,
    only: list[str],
    budget: int = DEFAULT_SLOT_BUDGET,
    mode: str = "safe",
    bundle: int | None = None,
) -> list[ArrayPlan]:
    """Build a plan for a RECOVERY pass over a subset of the 42 arrays.

    Differs from :func:`build_plan` in exactly three ways, and in nothing else —
    same configs, same seeds, same partition arithmetic, same memory:

    1. Only the selected ``(method, arm, suite)`` triples appear.
    2. The bundle comes from :func:`recovery_bundle`, i.e. from the **tail** of
       the per-cell distribution rather than its centre.
    3. The wall comes from :func:`recovery_wall_hours`, which is sized so the
       worker's own start-deadline cannot bite.

    The partition itself is untouched: ``decode_chunk`` is a pure function of
    ``(n_cells, n_tasks, index)`` and ``n_tasks = ceil(n_cells / bundle)``, so a
    pass at a different bundle still covers every cell of the array exactly once.
    What changes is *which* task owns a cell, and that is immaterial because
    completion is recorded per cell, not per task.

    Args:
        config_dir: Directory holding ``{method}_{suite}.yaml``.
        seeds: Campaign seeds; only the count is used for sizing.
        only: Selectors, e.g. ``["udfs:*:feynman", "bingo:baseline:nguyen"]``.
        budget: Total concurrent array slots to apportion over the SELECTED
            arrays.
        mode: See :func:`recovery_allowance_h`.
        bundle: Operator override for the chunk size, applied to every selected
            array.

    Returns:
        One :class:`ArrayPlan` per selected triple, in submission order.

    Raises:
        SlotPlanError: If a selector is malformed, matches nothing, a config is
            missing, or a resulting array could still defer.
    """
    if not only:
        raise SlotPlanError("a recovery pass must name the arrays it covers; --only is required")
    for pattern in only:
        if len(pattern.split(":")) != 3:
            raise SlotPlanError(
                f"selector {pattern!r} is not method:arm:suite (use '*' to wildcard a field)"
            )

    n_seeds = len(seeds)
    if n_seeds < 1:
        raise SlotPlanError("need at least one seed")

    triples = [
        (m, a, s)
        for m in METHODS
        for a in ARMS
        for s in SUITES
        if match_selector(f"{m}:{a}:{s}", only)
    ]
    if not triples:
        raise SlotPlanError(
            f"selectors {only} match none of the {len(METHODS) * len(ARMS) * len(SUITES)} arrays"
        )

    sizes: dict[tuple[str, str], int] = {}
    for method, _, suite in triples:
        if (method, suite) in sizes:
            continue
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

    cells = [sizes[(m, s)] * n_seeds for m, _, s in triples]
    allowances = [recovery_allowance_h(m, s, mode) for m, _, s in triples]
    bundles = [recovery_bundle(a, n, bundle) for a, n in zip(allowances, cells, strict=True)]
    walls = [recovery_wall_hours(b, a) for b, a in zip(bundles, allowances, strict=True)]
    n_tasks = [n_tasks_for(n, b) for n, b in zip(cells, bundles, strict=True)]

    # Weight the slot split by the CENTRAL estimate, as the main pass does: the
    # tail sizes the chunk, but the expected finishing time is what a throttle
    # should chase.
    per_cell = [cell_hours(m, s) for m, _, s in triples]
    works = [n * c for n, c in zip(cells, per_cell, strict=True)]
    slots = allocate_throttles(works, n_tasks, min(budget, sum(n_tasks)))

    plan = [
        ArrayPlan(
            method=m,
            arm=a,
            suite=s,
            n_cells=n,
            bundle=b,
            n_tasks=t,
            throttle=k,
            mem_gb=MEM_GB[(m, a)],
            wall=format_wall(w),
            runtime_h=c,
        )
        for (m, a, s), n, c, b, t, k, w in zip(
            triples, cells, per_cell, bundles, n_tasks, slots, walls, strict=True
        )
    ]

    for p, allowance in zip(plan, allowances, strict=True):
        if p.n_tasks * p.bundle < p.n_cells:
            raise SlotPlanError(f"{p.key}: {p.n_tasks} tasks x {p.bundle} < {p.n_cells} cells")
        if not defers_nothing(p.bundle, allowance, p.wall_h):
            raise SlotPlanError(
                f"{p.key}: B={p.bundle} at {allowance:.2f} h/cell against a {p.wall_h} h wall "
                f"leaves a {p.start_cutoff_h:.1f} h deadline -- the last cell could defer"
            )
    return plan


def format_recovery_notes(plan: list[ArrayPlan], mode: str) -> str:
    """Render the trade-offs a recovery plan makes, so they are not silent.

    Two of them matter and both are reported per array rather than in aggregate:
    the SCBI two-hour floor (a recovery chunk is short precisely because most of
    its cells are already complete), and the no-deferral margin.

    Args:
        plan: The recovery plan.
        mode: The sizing mode used.

    Returns:
        A human-readable block.
    """
    lines = [
        f"  recovery mode       : {mode}",
        f"  arrays              : {len(plan)}",
        f"  cells (re-walked)   : {sum(p.n_cells for p in plan):,}",
        f"  SLURM tasks         : {sum(p.n_tasks for p in plan):,}",
        "",
        f"{'array':32s} {'B':>4s} {'tasks':>6s} {'wall':>11s} {'cutoff_h':>9s} {'E[task_h]':>10s}",
    ]
    short = []
    for p in plan:
        lines.append(
            f"{p.key:32s} {p.bundle:4d} {p.n_tasks:6d} {p.wall:>11s} "
            f"{p.start_cutoff_h:9.1f} {p.task_h:10.2f}"
        )
        if p.task_h < MIN_TASK_HOURS:
            short.append(p)
    lines += [
        "",
        f"  SCBI floor ({MIN_TASK_HOURS:.0f} h) : "
        + (
            "cleared by every array"
            if not short
            else f"{len(short)} array(s) below it at the CENTRAL estimate: "
            + ", ".join(f"{p.key} ({p.task_h:.2f} h)" for p in short)
        ),
        "  ⚠ A recovery task is short BECAUSE most of its cells are already",
        "    complete and cost only a resume check. Sizing it up to clear the",
        "    floor would re-introduce the deadline that deferred them.",
    ]
    return "\n".join(lines)


def format_plan_tsv(plan: list[ArrayPlan]) -> str:
    """Render the plan as tab-separated rows for the launcher to read.

    Args:
        plan: Arrays in submission order.

    Returns:
        One ``method\\tarm\\tsuite\\tn_tasks\\tthrottle\\tmem_gb\\twall\\tbundle
        \\tstart_cutoff_s\\tn_cells`` row per array, without a header, so the
        shell can read it with a bare ``while read``.

        ``start_cutoff_s`` is the worker's deadline in seconds: the last elapsed
        time at which it may start another cell. It is emitted rather than
        recomputed in bash so that the wall and the deadline can never drift
        apart -- a deadline larger than ``wall - CELL_RESERVE`` would reintroduce
        exactly the ``TIMEOUT`` this design removes.
    """
    return "\n".join(
        f"{p.method}\t{p.arm}\t{p.suite}\t{p.n_tasks}\t{p.throttle}\t{p.mem_gb}\t"
        f"{p.wall}\t{p.bundle}\t{int(p.start_cutoff_h * 3600)}\t{p.n_cells}"
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
        f"{'array':32s} {'cells':>6s} {'B':>4s} {'tasks':>6s} {'%K':>5s} {'mem':>5s} "
        f"{'wall':>11s} {'task_h':>7s} {'work_h':>9s} {'finish_h':>9s}",
    ]
    for p in sorted(plan, key=lambda q: -q.finish_h):
        lines.append(
            f"{p.key:32s} {p.n_cells:6d} {p.bundle:4d} {p.n_tasks:6d} {p.throttle:5d} "
            f"{p.mem_gb:4d}G {p.wall:>11s} {p.task_h:7.1f} {p.work_h:9.0f} {p.finish_h:9.1f}"
        )
    total_work = sum(p.work_h for p in plan)
    total_slots = sum(p.throttle for p in plan)
    makespan = max(p.finish_h for p in plan)
    short = [p for p in plan if p.task_h < MIN_TASK_HOURS]

    quantised = max(p.quantised_finish_h for p in plan)
    lines += [
        "",
        f"  arrays              : {len(plan)}",
        f"  cells               : {sum(p.n_cells for p in plan):,}",
        f"  SLURM tasks         : {sum(p.n_tasks for p in plan):,}  "
        f"({sum(p.n_cells for p in plan) / max(1, sum(p.n_tasks for p in plan)):.1f} cells/task)",
        f"  slots               : {total_slots:,}",
        f"  shortest task       : {min(p.task_h for p in plan):.1f} h "
        f"(SCBI floor {MIN_TASK_HOURS:.0f} h -- "
        f"{'OK' if not short else 'VIOLATED by ' + ', '.join(p.key for p in short)})",
        f"  longest wall        : {max(p.wall_h for p in plan)} h (medium_uma MaxWall 72 h)",
        "",
        "  -- allocation basis (per-SUITE CELL_HOURS, measured on C1) --",
        f"  core-hours          : {total_work:,.0f}",
        f"  makespan            : {makespan:.1f} h = {makespan / 24:.2f} days",
        f"  packing efficiency  : {100 * total_work / total_slots / makespan:.0f} % "
        f"(floor {total_work / total_slots:.1f} h)",
        f"  quantised makespan  : {quantised:.1f} h = {quantised / 24:.2f} days",
        "        (whole rounds of indivisible tasks -- this is where chunking",
        "         would show a cost, and against the unchunked plan it does not)",
        "",
        "  -- sensitivity: CELL_HOURS is an estimate on 20 of 70 problems --",
        f"  mean concurrency    : {total_work / makespan:,.0f} cores",
        f"  if cells run 1.5x   : {1.5 * quantised:.1f} h = {1.5 * quantised / 24:.2f} days",
        f"  if cells run 0.5x   : {0.5 * quantised:.1f} h = {0.5 * quantised / 24:.2f} days",
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
        "--no-chunk",
        action="store_true",
        help="One cell per SLURM task, as before 2026-08-07. A/B only: it "
        "restores the 12,600-task shape SCBI asked us to stop submitting.",
    )
    parser.add_argument(
        "--max-bundle",
        type=int,
        default=None,
        help="Ceiling on cells per task. The Stage C smoke sets it so its "
        "B x (payload + teardown) wall stays under the 2 h `short` QOS cliff.",
    )
    parser.add_argument(
        "--bingo-hours",
        type=float,
        default=None,
        help="Override the assumed Bingo per-run duration used to WEIGHT the "
        "allocation. Use with --rebalance once the campaign's first day has "
        "shown what Bingo actually costs under the F-19 budget.",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Size the plan from the measured p90 TAIL rather than the central "
        "estimate, so the worker's start-deadline cannot defer a cell. Requires "
        "--only. Used by slurm/c2_campaign/submit_recovery.sh.",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=RECOVERY_MODES,
        default="safe",
        help="'safe' charges every cell the full 12.5 h payload reserve, making "
        "deferral impossible with no distributional assumption (default). 'p90' "
        "charges the measured p90 instead: bigger chunks, fewer placements, "
        "probabilistic guarantee.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated method:arm:suite selectors, '*' wildcards a field, "
        "e.g. 'udfs:*:feynman,bingo:baseline:nguyen'. Required with --recovery.",
    )
    parser.add_argument(
        "--bundle",
        type=int,
        default=None,
        help="Force the recovery chunk size. Validated against the no-deferral "
        "inequality; a bundle that cannot clear it is refused, not warned about.",
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
            # Rescale every Bingo suite by the same factor rather than flattening
            # them to one number: the suites differ by 35x per cell, and a flat
            # override would re-weight roundoff against nguyen for no reason.
            factor = args.bingo_hours / RUNTIME_HOURS["bingo"]
            RUNTIME_HOURS["bingo"] = args.bingo_hours
            for key in [k for k in CELL_HOURS if k[0] == "bingo"]:
                CELL_HOURS[key] *= factor
        if args.recovery:
            if args.rebalance:
                raise SlotPlanError("--recovery and --rebalance are different jobs; pick one")
            if not args.only:
                raise SlotPlanError("--recovery requires --only method:arm:suite[,...]")
            plan = build_recovery_plan(
                args.config_dir,
                parse_seeds(args.seeds),
                only=[s.strip() for s in args.only.split(",") if s.strip()],
                budget=args.budget,
                mode=args.recovery_mode,
                bundle=args.bundle,
            )
            payload = (
                format_plan_tsv(plan)
                if args.tsv
                else format_recovery_notes(plan, args.recovery_mode)
            )
            print(payload)
            return 0
        if args.only or args.bundle is not None:
            raise SlotPlanError("--only and --bundle apply to --recovery only")
        plan = build_plan(
            args.config_dir,
            parse_seeds(args.seeds),
            args.budget,
            uniform=args.uniform,
            chunk=not args.no_chunk,
            max_bundle=args.max_bundle,
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
