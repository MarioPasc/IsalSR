"""The worker's seed guard must accept every format a profile actually ships.

Regression for 2026-08-06: the campaign profile ships ``C2_SEEDS=1-30`` and the
worker counted comma-separated fields, saw ONE, and killed all 12,600 tasks in
seconds with ``decoded to 1 seed(s): '1-30'``.

Why four Stage C waves missed it: Stage C runs the **smoke** profile, whose
seeds are ``0,101,102``.  The campaign's range was never executed by any of
them, and ``sbatch --test-only`` accepts a job without ever running the worker,
so "42/42 accepted" was evidence about the scheduler, not about the payload.

These tests run the guard's arithmetic exactly as ``worker.sh`` does -- the awk
program is extracted from the shell source rather than copied, so the test
cannot drift away from the thing it certifies.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "slurm" / "c2_smoke" / "worker.sh"

#: Every seed spec a profile in launcher.sh actually ships, plus the colon form
#: the launcher translates to (``--export`` is comma-separated, so a comma in a
#: value would truncate the list).
PROFILE_SPECS = [
    ("0,101,102", 3),  # smoke, comma form
    ("0:101:102", 3),  # smoke, as shipped over --export
    ("1-30", 30),  # campaign  <- the one that broke
    ("1-20", 20),  # campaign, pre-2026-08-05
]


def _awk_program() -> str:
    """Extract the guard's awk program from worker.sh itself."""
    src = WORKER.read_text()
    match = re.search(r"N_SEEDS_DECODED=\$\(awk -F, '(.*?)'\s*<<<", src, re.S)
    assert match, "could not find the seed-count awk program in worker.sh"
    return match.group(1)


def _count(spec: str) -> int:
    awk = shutil.which("awk") or shutil.which("gawk")
    if awk is None:  # pragma: no cover
        pytest.skip("awk not available")
    spec = spec.replace(":", ",")
    out = subprocess.run(
        [awk, "-F,", _awk_program()],
        input=spec,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


@pytest.mark.parametrize(("spec", "expected"), PROFILE_SPECS)
def test_every_shipped_seed_spec_decodes_to_its_true_count(spec: str, expected: int) -> None:
    assert _count(spec) == expected


@pytest.mark.parametrize(("spec", "expected"), PROFILE_SPECS)
def test_every_shipped_seed_spec_passes_the_guard(spec: str, expected: int) -> None:
    """The guard rejects < 2; no real profile may trip it."""
    assert _count(spec) >= 2, f"{spec!r} would abort the worker"


def test_the_guard_still_catches_a_truncated_export() -> None:
    """Its original purpose: `--export` truncating "0,101,102" to "0"."""
    assert _count("0") == 1


def test_mixed_ranges_and_literals() -> None:
    assert _count("1-3,7,10-12") == 7


def test_launcher_profiles_are_covered_by_this_test() -> None:
    """If a profile gains a new seed spec, this test must be updated.

    Guards against the exact hole that caused the incident: a profile shipping a
    format nothing ever executed.
    """
    launcher = (WORKER.parent / "launcher.sh").read_text()
    shipped = set(re.findall(r'DEF_SEEDS="([^"]+)"', launcher))
    covered = {spec for spec, _ in PROFILE_SPECS}
    missing = shipped - covered
    assert not missing, (
        f"launcher.sh ships seed spec(s) {sorted(missing)} that this test does "
        f"not cover. Add them to PROFILE_SPECS."
    )
