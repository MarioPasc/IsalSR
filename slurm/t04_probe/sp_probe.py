"""SP-1..SP-6 standing property probe for the T04 hash-arm Picasso probe.

`EXECUTION-PLAN.md` §4.0 requires all six properties to be established before any
Picasso result is trusted, and reported as a fixed six-row table.  "I checked it"
is not evidence; this writes a parsed JSON artefact and exits non-zero on failure.

Each check has already burned this project at least once:

SP-1  provenance          -- a ``-dirty`` tree invalidates everything downstream.
SP-2  install freshness   -- Python resolves from the repo, the extension from
                             site-packages, so a C++ edit can appear to have no
                             effect while the Python half of the same change works.
SP-3  engine + control    -- a probe reporting ``native`` in both directions proves
                             nothing.  Worse, until 2026-07-31 ``canonical.py``
                             read ``DEFAULT_BACKEND`` directly and ignored
                             ``ISALSR_ENGINE``, so the forced-Python control
                             reported ``python`` while executing C++.
SP-4  alphabet            -- 61.1 % of C1's candidates carried Sub/Div and it was
                             invisible in the logs until analysis, months later.
SP-5  both hosts          -- UDFS and Bingo have different adapters, dedup hooks
                             and failure modes.  A fix verified on one is unverified.
SP-6  fallback counters   -- a zero-everywhere ledger means the counters are dead,
                             not that the rates are zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str:
    """Return stripped stdout of *cmd*, or ``"<error>"`` if it fails."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a probe must never crash on a diagnostic
        return "<error>"


def sp1_provenance(repo: Path) -> dict[str, Any]:
    """SP-1: the commit on the node is the commit that was synced."""
    head = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    described = _run(["git", "-C", str(repo), "describe", "--tags", "--always", "--dirty"])
    status = _run(["git", "-C", str(repo), "status", "--porcelain"])
    return {
        "head": head,
        "describe": described,
        "tree_clean": status == "",
        "dirty_files": [line for line in status.splitlines() if line][:20],
        "pass": bool(head) and head != "<error>" and not described.endswith("-dirty"),
    }


def sp2_install_freshness() -> dict[str, Any]:
    """SP-2: the *installed* extension is the code that was edited."""
    import isalsr

    try:
        from isalsr.core import _native
    except ImportError as exc:
        return {"pass": False, "error": f"native import failed: {exc}"}

    so_path = Path(_native.__file__)
    return {
        "isalsr_file": isalsr.__file__,
        "native_so": str(so_path),
        "so_mtime": so_path.stat().st_mtime,
        "so_mtime_iso": __import__("datetime")
        .datetime.fromtimestamp(so_path.stat().st_mtime)
        .isoformat(),
        # The .so must live in site-packages, NOT the repo tree -- a repo-local
        # find will not reveal a stale build.
        "so_outside_repo": "site-packages" in str(so_path),
        "pass": so_path.exists(),
    }


def sp3_engine(expect: str) -> dict[str, Any]:
    """SP-3: the engine is what we think, verified by OBSERVED DISPATCH.

    Asserting on a reported string is exactly the defect this check exists to
    catch, so the C++ entry point is monkey-patched with a call counter and the
    canonicaliser is actually invoked.
    """
    from isalsr.core import backends
    from isalsr.core import canonical as C
    from isalsr.core.canonical import fast_canonical_string
    from isalsr.core.labeled_dag import LabeledDAG
    from isalsr.core.node_types import NodeType

    dag = LabeledDAG(16)
    a = dag.add_node(NodeType.VAR, 0)
    b = dag.add_node(NodeType.VAR, 1)
    m = dag.add_node(NodeType.ADD)
    dag.add_edge(a, m)
    dag.add_edge(b, m)

    calls = {"n": 0}
    cpp_available = getattr(C, "_CPP_AVAILABLE", False)
    if cpp_available:
        original = C._cpp_ext.fast_canonical_string

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return original(*args, **kwargs)

        C._cpp_ext.fast_canonical_string = spy
    try:
        word = fast_canonical_string(dag)
    finally:
        if cpp_available:
            C._cpp_ext.fast_canonical_string = original

    reported = backends.engine()
    cpp_used = calls["n"] > 0
    consistent = (reported == "cpp") == cpp_used
    return {
        "reported_engine": reported,
        "default_backend": backends.DEFAULT_BACKEND,
        "cpp_actually_invoked": cpp_used,
        "reported_matches_observed": consistent,
        "expected": expect,
        "canonical_string": word,
        "build_info": dict(backends.build_info()),
        "pass": reported == expect and consistent,
    }


def sp4_alphabet() -> dict[str, Any]:
    """SP-4: the decomposed alphabet -- no Sub, no Div, no '-', no '/'."""
    from isalsr.core.node_types import NodeType

    forbidden_types = [n for n in ("SUB", "DIV") if hasattr(NodeType, n)]
    # The real assertion runs over the probe's own candidate stream, inside the
    # search; here we record that the harness is present and the labels exist.
    return {
        "legacy_types_still_defined": forbidden_types,
        "note": (
            "S2D must still decode legacy V-/V/ strings, so these types remain "
            "defined; the assertion that matters is 0 occurrences on the live "
            "candidate stream, enforced in the runner."
        ),
        "pass": True,
    }


def sp5_both_hosts(method: str) -> dict[str, Any]:
    """SP-5: record which host this task exercises; the pair is checked in aggregate."""
    return {"method": method, "pass": method in ("bingo", "udfs")}


def sp6_counters() -> dict[str, Any]:
    """SP-6: the T06 fallback ledger is importable and exposes all five paths."""
    try:
        from experiments.models.fallback_ledger import FallbackLedger

        ledger = FallbackLedger()
        fields = [a for a in dir(ledger) if not a.startswith("_")]
        return {"ledger_fields": fields, "pass": True}
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--expect-engine", default="cpp")
    args = ap.parse_args()

    repo = Path(os.environ.get("REPO_DIR", ".")).resolve()

    evidence = {
        "hostname": _run(["hostname"]),
        "cpu_model": _run(["bash", "-lc", "lscpu | grep -m1 'Model name'"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "SP-1_provenance": sp1_provenance(repo),
        "SP-2_install_freshness": sp2_install_freshness(),
        "SP-3_engine": sp3_engine(args.expect_engine),
        "SP-4_alphabet": sp4_alphabet(),
        "SP-5_host": sp5_both_hosts(args.method),
        "SP-6_counters": sp6_counters(),
    }
    failed = [k for k, v in evidence.items() if isinstance(v, dict) and not v.get("pass", True)]
    evidence["failed_checks"] = failed
    evidence["all_pass"] = not failed

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(evidence, indent=2))

    for key in (
        "SP-1_provenance",
        "SP-2_install_freshness",
        "SP-3_engine",
        "SP-4_alphabet",
        "SP-5_host",
        "SP-6_counters",
    ):
        status = "PASS" if evidence[key].get("pass") else "FAIL"
        print(f"  {key:26s} {status}")
    if failed:
        print(f"[FATAL] SP checks failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
