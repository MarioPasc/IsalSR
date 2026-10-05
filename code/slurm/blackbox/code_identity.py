"""Check that the deployed tree's arm-deciding files equal tag ``campaign/c2`` (gate P1).

The black-box campaign (R3.2) must run the main campaign's code. Since the tag,
the project moved under ``code/`` (``src/`` -> ``code/src/``), so the check maps
every path of the tag onto ``code/<path>`` and compares git blob ids, which are
content hashes: equal blob id means byte-identical file.

Scope (``SCOPE``): everything that can decide what an arm computes -- the whole
``isalsr`` package except the drawing package ``viz``, the C++ build file, the
whole of ``experiments/models`` (both hosts, the vendored UDFS, the shared
runner, schema and telemetry modules), the cell decoder, C2's worker, the two
C2 configurations the black-box configurations are copied from, and the
packaging metadata.

Every difference must be listed in ``EXPECTED`` with the reason it cannot
change an arm's computation; any other difference, a missing file or an added
file fails the check. Run on the workstation by ``deploy.sh`` (recorded in
``BBX_DEPLOY.json``) and on the Picasso login node by ``preflight.sh`` against
the deployed ``.git``.

Exit status: 0 when the check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

C2_REF = "campaign/c2"
C2_COMMIT = "2dd56fd76a9ace327c2fd949688a5c3b677c1bfe"
#: Prefix of the project inside the repository after the post-C2 move.
NEW_PREFIX = "code/"

#: Paths at ``campaign/c2`` (pre-move layout); directories end with ``/``.
SCOPE: tuple[str, ...] = (
    "src/isalsr/",
    "CMakeLists.txt",
    "pyproject.toml",
    "experiments/models/",
    "experiments/scripts/c2_task_spec.py",
    "experiments/configs/udfs_strogatz.yaml",
    "experiments/configs/bingo_strogatz.yaml",
    "slurm/c2_smoke/worker.sh",
)
#: Sub-trees inside ``SCOPE`` that no runner, adapter or translator imports.
EXCLUDED: tuple[str, ...] = ("src/isalsr/viz/",)

#: Known differences, each with the reason it cannot alter an arm's computation.
EXPECTED: dict[str, str] = {
    "src/isalsr/adapters/sympy_adapter.py": (
        "to_sympy() returns node_expressions(dag)[dag.output_node()], where "
        "node_expressions is the former loop; same value for every DAG. Used "
        "only for the reported best-expression string (T01 log §3.7)"
    ),
    "experiments/models/orchestrator.py": (
        "cherrypicked import path after the module rename, plus the "
        "srbench_blackbox registry entry and dispatch branch; neither touches "
        "an arm (T01 log §3.7)"
    ),
    "experiments/models/analyzer/statistical_tests.py": (
        "Nemenyi critical difference divided by sqrt(2) (Demsar 2006); "
        "post-hoc analysis only, never called during a run"
    ),
    "experiments/models/stage_d_trace.py": (
        "the offline spot-check replay applies recorded_key(); the tracer is "
        "inert unless ISALSR_STAGE_D_TRACE=1, which only "
        "slurm/c2_stage_d/worker.sh sets"
    ),
    "experiments/models/structural_scope.py": (
        "adds recorded_key() (used by the Stage-D replay above); the "
        "functions the runners call are unchanged"
    ),
    "pyproject.toml": (
        "version 0.1.0 -> 1.0.0, readme key removed, 'manuscript' pytest "
        "marker; wheel.packages and the build configuration unchanged"
    ),
}


class IdentityError(Exception):
    """Raised when git cannot answer a query the check depends on."""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise IdentityError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def _blobs(repo: Path, ref: str, paths: list[str]) -> dict[str, str]:
    """Return ``{path: blob_id}`` for every file under ``paths`` at ``ref``."""
    out = _git(repo, "ls-tree", "-r", "--full-tree", ref, "--", *paths)
    blobs: dict[str, str] = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, blob = meta.split()
        if kind == "blob":
            blobs[path] = blob
    return blobs


def _in_scope(path: str) -> bool:
    in_scope = any(path == s or (s.endswith("/") and path.startswith(s)) for s in SCOPE)
    return in_scope and not any(path.startswith(e) for e in EXCLUDED)


def compare(repo: Path, ref: str = C2_REF, head: str = "HEAD") -> dict[str, Any]:
    """Compare the in-scope files of ``ref`` with their moved copies at ``head``."""
    resolved = _git(repo, "rev-parse", f"{ref}^{{commit}}").strip()
    if resolved != C2_COMMIT:
        raise IdentityError(f"{ref} resolves to {resolved}, expected {C2_COMMIT}")
    old = {p: b for p, b in _blobs(repo, ref, list(SCOPE)).items() if _in_scope(p)}
    new_raw = _blobs(repo, head, [NEW_PREFIX + s for s in SCOPE])
    new = {p[len(NEW_PREFIX) :]: b for p, b in new_raw.items() if _in_scope(p[len(NEW_PREFIX) :])}

    identical = sorted(p for p in old if new.get(p) == old[p])
    changed = sorted(p for p in old if p in new and new[p] != old[p])
    missing = sorted(p for p in old if p not in new)
    added = sorted(p for p in new if p not in old)
    expected = {p: EXPECTED[p] for p in changed if p in EXPECTED}
    unexpected = [p for p in changed + missing + added if p not in EXPECTED]
    stat = _git(
        repo,
        "diff",
        "-M",
        "--stat=200",
        f"{ref}..{head}",
        "--",
        *SCOPE,
        *[NEW_PREFIX + s for s in SCOPE],
        *[f":(exclude){e}" for e in EXCLUDED],
        *[f":(exclude){NEW_PREFIX}{e}" for e in EXCLUDED],
    )
    return {
        "ref": ref,
        "ref_commit": resolved,
        "head": _git(repo, "rev-parse", head).strip(),
        "scope": list(SCOPE),
        "excluded": list(EXCLUDED),
        "n_files_at_ref": len(old),
        "n_identical": len(identical),
        "changed_expected": expected,
        "missing": missing,
        "added": added,
        "unexpected": unexpected,
        "pass": not unexpected,
        "diff_stat": stat.rstrip().splitlines(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparison, print a summary, optionally write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--ref", default=C2_REF)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    try:
        report = compare(args.repo, args.ref, args.head)
    except IdentityError as exc:
        print(f"[FATAL] P1: {exc}", file=sys.stderr)
        return 1
    print(
        f"P1 {report['ref']} ({report['ref_commit'][:7]}) vs {report['head'][:7]}: "
        f"{report['n_identical']}/{report['n_files_at_ref']} arm-deciding files byte-identical, "
        f"{len(report['changed_expected'])} expected differences"
    )
    for path, why in report["changed_expected"].items():
        print(f"    expected  {path}: {why}")
    for path in report["unexpected"]:
        print(f"[FATAL] P1 unexpected difference: {path}", file=sys.stderr)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
