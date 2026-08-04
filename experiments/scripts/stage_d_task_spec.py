"""Stage D cell registry — the single source of truth for campaign C2's 12 full-length cells.

Stage D (EXECUTION-PLAN §4.4) runs 12 tasks at the full 43,200 s budget to
certify what the 15-minute Stage C smoke cannot: memory growth, heap
fragmentation, dedup-set size, timeout paths and convergence behaviour are all
budget-dependent.

The registry is locked by ``audit.md`` §7 (Mario, 2026-08-04) and is deliberately
enumerated here rather than derived from a YAML config, because Stage D is a
hand-picked 12-cell design, not a suite sweep: six cells on the trace problem
(Pagie-1 × 3 arms × 2 methods) and six on the two problems that were NaN in the
submission (Korns-12, Vladislavleva-2 × 3 arms × Bingo).

Cells are grouped for submission because SLURM's ``--mem`` is per job, not per
array task, and the three memory classes differ by 16×. One ``sbatch`` array is
submitted per group.

This module is imported by ``slurm/c2_stage_d/worker.sh`` (via the ``--index``
shell-eval interface) and by ``experiments/scripts/stage_d_certify.py`` (via the
Python API). Both must agree on the registry, so neither owns a second copy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class StageDSpecError(RuntimeError):
    """Raised when the registry is queried with an argument it cannot satisfy."""


# ----------------------------------------------------------------------
# Locked configuration (audit.md section 7). Changing any of these changes
# what Stage D certifies, so they live here and nowhere else.
# ----------------------------------------------------------------------

#: Every Stage D cell runs this seed. One seed, because Stage D certifies
#: behaviour at the full budget, not statistics: the 30-seed population is
#: campaign C2's job.
STAGE_D_SEED: int = 101

#: Search budget handed to the host, in seconds (12 h).
STAGE_D_MAX_TIME_S: int = 43_200

#: SLURM wall limit. 16 h against a 12 h budget leaves 4 h (33 %) for the
#: post-search SymPy tail, which ``max_time`` does not bound (§11.1, 2026-08-03).
STAGE_D_WALL: str = "0-16:00:00"

#: Node family. Pinned so wall-clock numbers are comparable across cells; ``sr``
#: hosts exactly one 256 GB task per node (§3.3).
STAGE_D_CONSTRAINT: str = "sr"

#: The benchmark suite all three Stage D problems belong to.
STAGE_D_SUITE: str = "hard"

ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")

#: The trace problem (audit.md §7 row 2): a structural-bottleneck problem where
#: IsalSR is predicted to help, per the 2026-04-19 bottleneck analysis.
TRACE_PROBLEM: str = "Pagie-1"

#: The two cells that were NaN in the submission (T08 AC-7 evidence).
NAN_PROBLEMS: tuple[str, ...] = ("Korns-12", "Vladislavleva-2")

#: The single (method, problem, arm) cell that additionally persists the D2
#: detailed candidate trace.
TRACE_CELL: tuple[str, str, str] = ("bingo", TRACE_PROBLEM, "isalsr")


@dataclass(frozen=True)
class StageDCell:
    """One Stage D certification cell.

    Attributes:
        index: 1-based index over the whole 12-cell registry. Stable, and used
            as the cell's identity in the certification report.
        group: Submission group name. One SLURM array is submitted per group.
        group_index: 1-based index within the group. This is the value
            ``SLURM_ARRAY_TASK_ID`` takes.
        method: Host search method, ``"udfs"`` or ``"bingo"``.
        arm: Deduplication arm, one of :data:`ARMS`.
        problem: Benchmark problem name as the registry spells it.
        suite: Benchmark suite key.
        mem_gb: Requested memory for the group, in GiB.
        trace: Whether this cell persists the D2 detailed candidate trace.
    """

    index: int
    group: str
    group_index: int
    method: str
    arm: str
    problem: str
    suite: str
    mem_gb: int
    trace: bool

    @property
    def problem_slug(self) -> str:
        """Return the on-disk directory name for the problem.

        Mirrors ``experiments.models.io_utils.ensure_output_structure``, which
        is the producer of the layout the certifier later walks.

        Returns:
            The lowercased, hyphen-to-underscore problem slug.
        """
        return self.problem.lower().replace("-", "_")

    @property
    def seed(self) -> int:
        """Return the seed this cell runs.

        Returns:
            :data:`STAGE_D_SEED`, identical for all cells by design.
        """
        return STAGE_D_SEED

    @property
    def config_name(self) -> str:
        """Return the basename of the experiment config this cell uses.

        Returns:
            The ``<method>_<suite>.yaml`` name, matching the convention
            ``slurm/c2_smoke/launcher.sh`` uses.
        """
        return f"{self.method}_{self.suite}.yaml"

    @property
    def label(self) -> str:
        """Return a compact human-readable cell identifier.

        Returns:
            A ``method/arm/problem`` string for logs and report tables.
        """
        return f"{self.method}/{self.arm}/{self.problem}"

    def run_dir(self, root: Path) -> Path:
        """Return the seed directory this cell writes into.

        Args:
            root: Stage D results root.

        Returns:
            ``root/<method>/<suite>/<problem_slug>/<arm>/seed_<NN>``.
        """
        return (
            root / self.method / self.suite / self.problem_slug / self.arm / f"seed_{self.seed:02d}"
        )


def _build_registry() -> tuple[StageDCell, ...]:
    """Enumerate the 12 Stage D cells in submission-group order.

    Memory groups follow audit.md §7: UDFS 3 × 16 GB, Bingo non-isalsr
    6 × 32 GB, Bingo isalsr 3 × 256 GB. The 256 GB figure is §3.3's
    evidence-based floor, taken from C1's 29 IsalSR cells that died at
    ``MaxRSS ≈ 127.7 GB`` against a 128 GB request.

    Returns:
        The registry, ordered by group then by group index.
    """
    cells: list[StageDCell] = []
    index = 0

    def add(group: str, method: str, arm: str, problem: str, mem_gb: int) -> None:
        nonlocal index
        index += 1
        group_index = sum(1 for c in cells if c.group == group) + 1
        cells.append(
            StageDCell(
                index=index,
                group=group,
                group_index=group_index,
                method=method,
                arm=arm,
                problem=problem,
                suite=STAGE_D_SUITE,
                mem_gb=mem_gb,
                trace=(method, problem, arm) == TRACE_CELL,
            )
        )

    # Group 1 — UDFS on the trace problem only. UDFS has no `pow` in its
    # vendored table, and its per-candidate cost is ~64x its canonicalisation
    # cost, so it is the cheap half of the overhead comparison.
    for arm in ARMS:
        add("udfs", "udfs", arm, TRACE_PROBLEM, 16)

    # Group 2 — Bingo, non-dedup-heavy arms. The baseline arm holds no dedup
    # set at all; the hash arm holds one set of 64-bit digests.
    for problem in (TRACE_PROBLEM, *NAN_PROBLEMS):
        for arm in ("baseline", "hash"):
            add("bingo_std", "bingo", arm, problem, 32)

    # Group 3 — Bingo IsalSR. The canonical-string dedup set is what hit C1's
    # ceiling; this group is the one D1.2 exists to size.
    for problem in (TRACE_PROBLEM, *NAN_PROBLEMS):
        add("bingo_isalsr", "bingo", "isalsr", problem, 256)

    return tuple(cells)


#: The locked 12-cell Stage D registry.
STAGE_D_CELLS: tuple[StageDCell, ...] = _build_registry()

#: Submission groups in the order the launcher submits them, with their memory.
STAGE_D_GROUPS: tuple[str, ...] = ("udfs", "bingo_std", "bingo_isalsr")


def cells_for_group(group: str) -> tuple[StageDCell, ...]:
    """Return the cells belonging to one submission group.

    Args:
        group: Group name, one of :data:`STAGE_D_GROUPS`.

    Returns:
        The group's cells, ordered by ``group_index``.

    Raises:
        StageDSpecError: If the group name is not in the registry.
    """
    if group not in STAGE_D_GROUPS:
        raise StageDSpecError(
            f"unknown group {group!r}; expected one of {', '.join(STAGE_D_GROUPS)}"
        )
    return tuple(c for c in STAGE_D_CELLS if c.group == group)


def decode_index(group: str, index: int) -> StageDCell:
    """Resolve a SLURM array task index to its cell.

    Args:
        group: Submission group name.
        index: 1-based ``SLURM_ARRAY_TASK_ID``.

    Returns:
        The cell that task should run.

    Raises:
        StageDSpecError: If the index falls outside the group.
    """
    group_cells = cells_for_group(group)
    if not 1 <= index <= len(group_cells):
        raise StageDSpecError(
            f"index {index} out of range for group {group!r} (valid range 1..{len(group_cells)})"
        )
    return group_cells[index - 1]


def group_mem_gb(group: str) -> int:
    """Return the memory request for a submission group.

    Args:
        group: Submission group name.

    Returns:
        The group's ``--mem`` in GiB.

    Raises:
        StageDSpecError: If the group is unknown or internally inconsistent.
    """
    mems = {c.mem_gb for c in cells_for_group(group)}
    if len(mems) != 1:
        raise StageDSpecError(f"group {group!r} mixes memory requests: {sorted(mems)}")
    return mems.pop()


def trace_cell() -> StageDCell:
    """Return the single cell that persists the D2 detailed trace.

    Returns:
        The Bingo × Pagie-1 × isalsr cell.

    Raises:
        StageDSpecError: If the registry does not contain exactly one trace cell.
    """
    matches = [c for c in STAGE_D_CELLS if c.trace]
    if len(matches) != 1:
        raise StageDSpecError(f"expected exactly 1 trace cell, found {len(matches)}")
    return matches[0]


def _emit_shell(cell: StageDCell) -> str:
    """Render a cell as shell variable assignments for ``eval``.

    Values are single-quoted because problem names contain ``.`` and ``-``.
    A key=value contract is used rather than positional fields so that adding a
    field cannot silently shift the worker's parsing.

    Args:
        cell: The cell to render.

    Returns:
        Newline-separated ``D_*='value'`` assignments.
    """
    fields = {
        "D_INDEX": cell.index,
        "D_GROUP": cell.group,
        "D_GROUP_INDEX": cell.group_index,
        "D_METHOD": cell.method,
        "D_ARM": cell.arm,
        "D_PROBLEM": cell.problem,
        "D_PROBLEM_SLUG": cell.problem_slug,
        "D_SUITE": cell.suite,
        "D_SEED": cell.seed,
        "D_MEM_GB": cell.mem_gb,
        "D_TRACE": 1 if cell.trace else 0,
        "D_CONFIG_NAME": cell.config_name,
        "D_MAX_TIME": STAGE_D_MAX_TIME_S,
    }
    return "\n".join(f"{k}='{v}'" for k, v in fields.items())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Stage D 12-cell registry: index -> cell resolution.",
    )
    parser.add_argument("--group", help="submission group name")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int, help="1-based SLURM array task index")
    mode.add_argument("--count", action="store_true", help="print the group's task count")
    mode.add_argument("--groups", action="store_true", help="print all group names")
    mode.add_argument("--mem-gb", action="store_true", help="print the group's --mem in GiB")
    mode.add_argument("--list", action="store_true", help="print the whole registry as a table")
    mode.add_argument("--json", action="store_true", help="print the whole registry as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the registry command-line interface.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on a registry error.
    """
    args = build_parser().parse_args(argv)

    try:
        if args.groups:
            print("\n".join(STAGE_D_GROUPS))
            return 0
        if args.list:
            for c in STAGE_D_CELLS:
                flag = " TRACE" if c.trace else ""
                print(
                    f"{c.index:2d}  {c.group:13s} {c.group_index}  "
                    f"{c.method:5s} {c.arm:8s} {c.problem:17s} "
                    f"{c.mem_gb:3d}G{flag}"
                )
            return 0
        if args.json:
            print(json.dumps([asdict(c) for c in STAGE_D_CELLS], indent=2))
            return 0

        if not args.group:
            raise StageDSpecError("--group is required for --index/--count/--mem-gb")
        if args.count:
            print(len(cells_for_group(args.group)))
            return 0
        if args.mem_gb:
            print(group_mem_gb(args.group))
            return 0
        print(_emit_shell(decode_index(args.group, args.index)))
        return 0
    except StageDSpecError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
