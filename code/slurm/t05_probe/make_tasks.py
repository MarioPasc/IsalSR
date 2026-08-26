"""Generate the T05 probe task list from the committed D2 definition.

The task list is derived, not typed. `D2` is whatever the benchmark registry says
it is, and the registry is what the campaign will run — so a probe built from a
hand-written list could pass while covering a different set of problems from the
one that launches. That is the failure this script exists to prevent.

Usage
-----
    python slurm/t05_probe/make_tasks.py --out slurm/t05_probe/tasks.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.models.orchestrator import get_benchmarks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

#: (registry suite, config stem). Both D2 tiers, both hosts.
SUITES: tuple[tuple[str, str], ...] = (
    ("strogatz", "strogatz"),
    ("feynman_remainder", "feynman_remainder"),
)

METHODS: tuple[str, ...] = ("udfs", "bingo")

HEADER = """\
# T05 D2 Picasso probe task list -- GENERATED, do not hand-edit.
# Regenerate with: python slurm/t05_probe/make_tasks.py --out slurm/t05_probe/tasks.txt
#
# Fields: method variant problem config suite
# One line per array task. Line N is read by SLURM_ARRAY_TASK_ID=N (1-indexed).
#
# SP-0 compliance: seed 0 only, max_time <= 1800 s, <= 60 tasks, output under
# ~/execs/isalsr/t05_probe/. This is a PROBE. It produces no number for the paper.
#
# Coverage: every D2 problem x both hosts, on the `isalsr` arm -- the arm that
# exercises the adapter, the alphabet and the dedup hook, and therefore the only
# one whose failure would be invisible in the other two. The operator-set
# identity across arms (SP-7.4) is a config-level property checked offline by
# check_d2.py, not something an array task can establish.
#
# One variant per task is deliberate: the orchestrator's post-run
# compute_paired_stats raises on fewer than 3 paired seeds, and a single-variant
# run never reaches that code path.
"""


def main(argv: list[str] | None = None) -> int:
    """Write the task list.

    Parameters
    ----------
    argv
        Command-line arguments.

    Returns
    -------
    int
        Process exit status; non-zero if the task count would breach SP-0's cap.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "tasks.txt"))
    args = parser.parse_args(argv)

    lines: list[str] = []
    for suite, cfg_stem in SUITES:
        for bench in get_benchmarks(suite):
            for method in METHODS:
                cfg = f"experiments/configs/{method}_{cfg_stem}.yaml"
                lines.append(f"{method} isalsr {bench['name']} {cfg} {suite}")

    lines.sort(key=lambda row: (row.split()[0], row.split()[4], row.split()[2]))

    Path(args.out).write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %d tasks to %s", len(lines), args.out)

    if len(lines) > 60:
        log.error("SP-0 caps a probe at 60 tasks; this list has %d.", len(lines))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
