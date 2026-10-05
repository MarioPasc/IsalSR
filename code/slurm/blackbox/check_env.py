"""Pre-flight / per-task gate for the SRBench black-box campaign (R3.2).

Run with the black-box import shim active (``PYTHONPATH`` holding
``slurm/blackbox/pyboot`` and ``ISALSR_BBX_ROOT`` set). It prints, then
enforces:

1. ``isalsr`` resolves into the C2 tree's ``src/isalsr`` (the existing editable
   install), so the canonicaliser is the C2 one by construction;
2. ``experiments.models.orchestrator`` and ``benchmarks.datasets.srbench_blackbox``
   resolve into the deployed black-box tree;
3. the C++ engine is native with build hash ``298fc1188bf1b051`` (C2's);
4. the deployed files still have the SHA-256s recorded by ``deploy.sh`` in
   ``BBX_DEPLOY.json`` (no edit since deployment).

With ``--write-stamp PATH`` it also writes the provenance stamp (deploy record,
resolved module paths, engine build info, library versions) as JSON.

Exit status: 0 if every check passes, 1 otherwise. Nothing is printed to the
results tree unless ``--write-stamp`` is given.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

C2_BUILD_HASH = "298fc1188bf1b051"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path: str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def collect(bbx_root: Path) -> dict[str, Any]:
    """Import the three modules and the engine; return what was resolved."""
    import numpy
    import sklearn

    import benchmarks.datasets.srbench_blackbox as bbx
    import experiments.models.orchestrator as orch

    # mypy cannot see through the editable install's redirecting finder.
    import isalsr  # type: ignore[import-untyped]
    from isalsr.core import _native, backends  # type: ignore[import-untyped]

    return {
        "isalsr_file": isalsr.__file__,
        "native_file": _native.__file__,
        "orchestrator_file": orch.__file__,
        "srbench_blackbox_file": bbx.__file__,
        "engine": backends.engine(),
        "build_info": dict(backends.build_info()),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "hostname": platform.node(),
        "bbx_root": str(bbx_root),
        "registry_has_suite": "srbench_blackbox" in orch._BENCHMARK_REGISTRY,
        "n_problems": len(bbx.SRBENCH_BLACKBOX_BENCHMARKS),
    }


def verify(
    info: dict[str, Any], c2_tree: Path, bbx_root: Path, deploy: dict[str, Any]
) -> list[str]:
    """Return a list of failed checks (empty when everything holds)."""
    bad: list[str] = []
    if not _under(info["isalsr_file"], c2_tree / "src" / "isalsr"):
        bad.append(f"isalsr resolves to {info['isalsr_file']}, not under {c2_tree}/src/isalsr")
    for key in ("orchestrator_file", "srbench_blackbox_file"):
        if not _under(info[key], bbx_root):
            bad.append(f"{key} = {info[key]} is not under the black-box root {bbx_root}")
    if info["engine"] != "cpp":
        bad.append(f"engine is {info['engine']!r}, expected 'cpp'")
    if info["build_info"].get("build_hash") != C2_BUILD_HASH:
        bad.append(f"build_hash {info['build_info'].get('build_hash')} != C2 {C2_BUILD_HASH}")
    if not info["registry_has_suite"]:
        bad.append("orchestrator registry lacks 'srbench_blackbox'")
    for rel, expected in deploy.get("sha256", {}).items():
        path = bbx_root / rel
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != expected:
            bad.append(f"{rel}: sha256 {actual} != deployed {expected}")
    if not deploy.get("sha256"):
        bad.append("BBX_DEPLOY.json has no sha256 table")
    return bad


def main(argv: list[str] | None = None) -> int:
    """Print the resolved environment, enforce the four checks, optionally stamp."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c2-tree", required=True, type=Path)
    parser.add_argument(
        "--bbx-root", type=Path, default=Path(os.environ.get("ISALSR_BBX_ROOT", "."))
    )
    parser.add_argument("--write-stamp", type=Path, default=None)
    args = parser.parse_args(argv)

    deploy_path = args.bbx_root / "BBX_DEPLOY.json"
    deploy: dict[str, Any] = json.loads(deploy_path.read_text()) if deploy_path.is_file() else {}
    info = collect(args.bbx_root)
    for key in ("isalsr_file", "orchestrator_file", "srbench_blackbox_file", "native_file"):
        print(f"BBX {key:22s} {info[key]}")
    build_hash = info["build_info"].get("build_hash")
    print(f"BBX engine                 {info['engine']}  build_hash={build_hash}")
    print(f"BBX deployed commit        {deploy.get('git_head', '<no BBX_DEPLOY.json>')}")
    print(f"BBX numpy/sklearn          {info['numpy']} / {info['sklearn']}")

    failures = verify(info, args.c2_tree, args.bbx_root, deploy)
    for f in failures:
        print(f"[FATAL] {f}", file=sys.stderr)
    if failures:
        return 1
    print(
        "BBX gate: PASS (isalsr -> C2 tree, experiments/benchmarks -> bbx root, "
        "engine C2, files as deployed)"
    )

    if args.write_stamp is not None:
        stamp = {
            "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "deploy": deploy,
            "resolved": info,
            "c2_tree": str(args.c2_tree),
        }
        args.write_stamp.parent.mkdir(parents=True, exist_ok=True)
        args.write_stamp.write_text(json.dumps(stamp, indent=2, sort_keys=True))
        print(f"BBX provenance stamp -> {args.write_stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
