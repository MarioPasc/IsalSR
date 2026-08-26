"""The aggregation worker's two roles must not require each other's variables.

Regression for job 1783830 (Stage C v5, 2026-08-06): ``aggregate_worker.sh``
asserted ``C2_CONFIG_LIST`` at the top of the file, unconditionally.  The
launcher does not export it to the dependent ledger job -- correctly, since a
full-root walk needs no config -- so the job died in 2 s under ``set -u``,
before the role was even selected.  The wave was 1,260/1,260 clean and still
produced no ``status_ledger.csv`` and no Stage C verdict.

These tests are deliberately static: they read the shell sources rather than
executing them, because the failure is a variable-expansion abort that happens
before any observable side effect, and because executing the worker requires a
Picasso environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[2] / "slurm" / "c2_smoke"
WORKER = SLURM / "aggregate_worker.sh"
LAUNCHER = SLURM / "launcher.sh"


def _code_only(text: str) -> str:
    """Strip comment lines.

    These files document their own defects in prose, so a naive substring scan
    matches the explanation rather than the code. Analyse code, not comments.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def worker_src() -> str:
    return _code_only(WORKER.read_text())


@pytest.fixture(scope="module")
def launcher_src() -> str:
    return _code_only(LAUNCHER.read_text())


def test_config_list_is_not_asserted_unconditionally(worker_src: str) -> None:
    """``${C2_CONFIG_LIST:?...}`` at file scope is the exact defect."""
    assert "${C2_CONFIG_LIST:?" not in worker_src, (
        "C2_CONFIG_LIST must not use the `:?` abort form at file scope -- the "
        "ledger role does not receive it and would die before selecting a role."
    )


def test_config_list_is_still_required_by_the_array_role(worker_src: str) -> None:
    """Relaxing the assertion must not make a missing config list silent."""
    assert re.search(r'-n\s+"\$\{CONFIG_LIST\}"', worker_src), (
        "the aggregation array role must still fail loudly on an empty "
        "CONFIG_LIST; otherwise the fix trades a crash for a silent no-op."
    )


def test_variables_the_ledger_job_is_given_are_the_ones_it_needs(
    worker_src: str, launcher_src: str
) -> None:
    """Every `:?`-asserted variable must be exported to BOTH roles."""
    asserted = set(re.findall(r"\$\{([A-Z0-9_]+):\?", worker_src))

    # The launcher's ledger-job --export line.
    ledger_export = re.search(r"C2_LEDGER_ONLY=1[^\"]*", launcher_src)
    assert ledger_export is not None, "could not find the ledger job's --export"
    block_start = launcher_src.rfind('--export="', 0, ledger_export.start())
    block = launcher_src[block_start : ledger_export.end()]
    exported = set(re.findall(r"([A-Z0-9_]+)=", block))

    # SLURM_ARRAY_TASK_ID is asserted inside the array-only branch, so it is
    # legitimately absent from the ledger job's environment.
    asserted.discard("SLURM_ARRAY_TASK_ID")

    missing = asserted - exported
    assert not missing, (
        f"aggregate_worker.sh aborts on {sorted(missing)} but the launcher does "
        f"not export them to the ledger job. Exported: {sorted(exported)}"
    )


def test_ledger_job_receives_what_it_reads(launcher_src: str) -> None:
    """The ledger role's own inputs must actually be exported."""
    idx = launcher_src.find("C2_LEDGER_ONLY=1")
    assert idx != -1
    block = launcher_src[idx - 400 : idx + 300]
    for var in ("C2_RESULTS_DIR", "C2_EXPECTED_TASKS", "ISALSR_REPO_DIR"):
        assert var in block, f"{var} missing from the ledger job's --export"
