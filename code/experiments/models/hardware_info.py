"""Hardware, environment and engine provenance capture.

Collects the fields that let a completed campaign be audited without trusting
the launch script: which CPU ran a task, which commit and which compiled
extension produced its numbers, and which SLURM allocation it occupied.

**Why the engine fields exist (EXECUTION-PLAN A7-BUG, found 2026-08-02).**
Check C1.14 requires asserting ``engine == native`` on every task of the
420-task smoke, and D2's spot check re-canonicalises production candidates in
pure Python to confirm the engine used *in production* is the engine the
equivalence gate certified. Neither is possible from a run log that does not
record the engine. It read ``<none>`` in a live T04 probe. The failure this
guards against is silent by construction: the Python and C++ canonicalisers
agree on every DAG the gate covers, so a run that quietly fell back to Python
produces correct numbers at ~24x the cost, and the only visible symptom is a
cost axis nobody can explain in September.

``build_hash`` and ``isa_level`` are recorded alongside it because "native" is
not one thing: an extension built with ``-march=native`` on the AVX-512 login
node dies with SIGILL on the ``sr``/``bl`` nodes that are most of the pool
(B6b), and the ISA level is what distinguishes a portable build from one that
will fail on a fraction of tasks and look like flaky hardware.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

__all__ = ["collect_hardware_info", "peak_rss_gb"]


def collect_hardware_info() -> dict[str, Any]:
    """Collect hardware, environment, engine and allocation information.

    Every value is best-effort: a field that cannot be determined is reported
    as ``"unknown"`` or ``None`` rather than raising, because provenance
    capture must never be the reason a 12 h run does not start. A missing
    field is caught by the pre-flight validator, which is where it belongs.

    Returns:
        Flat dict of provenance fields. Keys ``cpu``, ``cpu_count``, ``ram_gb``,
        ``python_version``, ``platform``, ``os``, ``os_version``, ``conda_env``,
        ``git_hash`` and ``timestamp`` are the pre-C2 set and are retained
        verbatim so C1-era readers keep working; the rest were added to close
        EXECUTION-PLAN A7.
    """
    engine_info = _engine_info()
    cpu_model = _cpu_model()
    return {
        # --- pre-C2 fields, unchanged ---------------------------------- #
        "cpu": cpu_model,
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_version": platform.version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "git_hash": _git("rev-parse", "HEAD"),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        # --- A7: node identity ----------------------------------------- #
        # ``cpu_model`` duplicates ``cpu`` under the name the plan and the
        # status ledger use. Kept as an alias rather than a rename so that
        # neither C1 artefacts nor C2 tooling need a compatibility shim.
        "cpu_model": cpu_model,
        "hostname": platform.node() or "unknown",
        # --- A7: code identity ----------------------------------------- #
        # ``git_describe`` carries the ``campaign/c2`` tag and, critically, the
        # ``-dirty`` suffix: SP-1 treats a dirty tree as invalidating, and that
        # is only checkable if the suffix is recorded at run time.
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": _git_dirty(),
        # --- A7: engine identity (A7-BUG) ------------------------------- #
        "engine": engine_info["engine"],
        "build_hash": engine_info["build_hash"],
        "isa_level": engine_info["isa_level"],
        "avx512f": engine_info["avx512f"],
        "compiler": engine_info["compiler"],
        "native_module_path": engine_info["native_module_path"],
        "native_module_mtime": engine_info["native_module_mtime"],
        # --- A7: allocation identity ------------------------------------ #
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "mem_requested_gb": _mem_requested_gb(),
    }


def peak_rss_gb() -> float:
    """Return peak resident set size of this process and its children, in GB.

    Reported per run into the status ledger, which is what sizes production
    ``--mem`` (C1.11, D1.2). Taken from ``getrusage`` rather than from
    ``sacct`` because ``sacct -X`` returns an **empty** ``MaxRSS`` -- memory is
    accounted on the ``.batch`` step -- so a profile built the obvious way comes
    back silently blank. An in-process reading is also available for a run that
    fails, where there may be no completed step to account.

    Returns:
        Peak RSS in GB, or ``0.0`` on a platform without ``resource``.
    """
    try:
        import resource  # noqa: PLC0415 -- POSIX-only, imported at the call site
    except ImportError:
        return 0.0
    # ru_maxrss is kilobytes on Linux and bytes on macOS, so the divisor to GB
    # differs by 1024x. The campaign runs on Linux; the branch keeps a local
    # macOS reading from being reported 1024x too large.
    per_gb = 1024**3 if sys.platform == "darwin" else 1024**2
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    return round(peak / per_gb, 3)


def _engine_info() -> dict[str, Any]:
    """Return the active canonicalisation engine and its build identity.

    Reports what is *actually dispatched*, via ``backends.engine()``, which
    honours the ``ISALSR_ENGINE`` override. Reading the compiled-in default
    instead would report ``cpp`` whatever the run did -- the exact defect B2
    found in ``fast_canonical_string`` on 2026-07-31, where a probe passed
    while proving nothing.
    """
    unknown: dict[str, Any] = {
        "engine": "unknown",
        "build_hash": "",
        "isa_level": "",
        "avx512f": "",
        "compiler": "",
        "native_module_path": "",
        "native_module_mtime": "",
    }
    try:
        from isalsr.core import backends  # noqa: PLC0415
    except ImportError:
        return unknown

    try:
        info = backends.build_info()
        active = backends.engine()
    except (ImportError, ValueError, RuntimeError):
        return unknown

    module_path, module_mtime = "", ""
    try:
        from isalsr.core import _native  # noqa: PLC0415

        module_path = getattr(_native, "__file__", "") or ""
        if module_path:
            module_mtime = datetime.fromtimestamp(os.stat(module_path).st_mtime, tz=UTC).isoformat()
    except (ImportError, OSError):
        pass

    return {
        # ``native`` is the plan's vocabulary for the compiled engine; the
        # backend registry calls it ``cpp``. Normalised here so C1.14 can
        # assert one string.
        "engine": "native" if active == "cpp" else active,
        "build_hash": info.get("build_hash", ""),
        "isa_level": info.get("isa_level", ""),
        "avx512f": info.get("avx512f", ""),
        "compiler": info.get("compiler", ""),
        "native_module_path": module_path,
        "native_module_mtime": module_mtime,
    }


def _mem_requested_gb() -> float | None:
    """Return the memory this allocation requested, in GB, or None off SLURM.

    SLURM exports ``SLURM_MEM_PER_NODE`` (MB) for ``--mem`` and
    ``SLURM_MEM_PER_CPU`` (MB) for ``--mem-per-cpu``; the campaign uses the
    former, the latter is handled so a differently-launched probe still records
    something meaningful.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return round(int(per_node) / 1024, 2)
        except ValueError:
            return None
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu:
        try:
            cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
            return round(int(per_cpu) * cpus / 1024, 2)
        except ValueError:
            return None
    return None


def _cpu_model() -> str:
    """Get CPU model string."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except (FileNotFoundError, PermissionError):
        pass
    return 0.0


def _git(*args: str) -> str:
    """Run a read-only git command, returning ``"unknown"`` on any failure."""
    try:
        return (
            subprocess.check_output(  # noqa: S603, S607
                ["git", *args],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _git_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes.

    ``None`` when git is unavailable, which is distinguishable from a clean
    tree -- SP-1 must not read "no answer" as "clean".
    """
    status = _git("status", "--porcelain")
    if status == "unknown":
        return None
    return bool(status)
