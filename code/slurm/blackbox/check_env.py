"""Pre-flight / per-task gate for the SRBench black-box campaign (R3.2).

Layout (T01b): one deployed tree, ``$FSCRATCH/repos/IsalSR`` (the full
repository with ``.git``; the project lives under ``code/``), and the conda env
``$FSCRATCH/conda_envs/isalsr`` holding an editable install of that tree, built
with C2's recipe. ``isalsr``, ``experiments`` and ``benchmarks`` therefore all
resolve from the deployed tree, and the C++ engine from the env's
site-packages. The ``pyboot`` import shim is not used.

The gate prints, then enforces:

1. ``isalsr``, ``experiments.models.orchestrator`` and
   ``benchmarks.datasets.srbench_blackbox`` resolve inside ``<code>``
   (``isalsr`` inside ``<code>/src/isalsr``); the import shim is inactive;
2. the C++ engine is native, loaded from site-packages, with build hash
   ``298fc1188bf1b051`` (C2's);
3. the deployed files still have the SHA-256s recorded by ``deploy.sh`` in
   ``slurm/blackbox/BBX_DEPLOY.json``, whose P1 verdict (arm-deciding files
   identical to ``campaign/c2``) is a pass, and whose commit is the tree's HEAD
   when git can be asked;
4. every ``name==version`` pin of ``slurm/blackbox/env_requirements.txt`` and
   the interpreter version match the running environment.

With ``--write-stamp PATH`` it also writes the provenance stamp (deploy record,
resolved module paths, engine build info, library versions) as JSON.

Exit status: 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

C2_BUILD_HASH = "298fc1188bf1b051"
#: C2's interpreter: run_log ``hardware.python_version`` of every C2 cell.
C2_PYTHON = "3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0]"
DEPLOY_RECORD = Path("slurm/blackbox/BBX_DEPLOY.json")
PINS_FILE = Path("slurm/blackbox/env_requirements.txt")
PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)")
#: Installed by create_env.sh from the CPU wheel index (not in the pins file).
TORCH_PIN = "2.12.0+cpu"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path: str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def read_pins(path: Path) -> dict[str, str]:
    """Return ``{distribution: version}`` for every ``name==version`` line."""
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = PIN_RE.match(line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def installed_versions(names: list[str]) -> dict[str, str]:
    """Return the installed version of each distribution (``<absent>`` if missing)."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "<absent>"
    return out


def git_head(repo: Path) -> str | None:
    """Return HEAD of ``repo``, or None when git is unavailable (compute node)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def collect(code_root: Path) -> dict[str, Any]:
    """Import the three modules and the engine; return what was resolved."""
    import benchmarks.datasets.srbench_blackbox as bbx
    import experiments.models.orchestrator as orch

    # mypy cannot see through the editable install's redirecting finder.
    import isalsr  # type: ignore[import-untyped]
    from isalsr.core import _native, backends  # type: ignore[import-untyped]

    pins = read_pins(code_root / PINS_FILE)
    return {
        "isalsr_file": isalsr.__file__,
        "native_file": _native.__file__,
        "orchestrator_file": orch.__file__,
        "srbench_blackbox_file": bbx.__file__,
        "engine": backends.engine(),
        "build_info": dict(backends.build_info()),
        "python": sys.version,
        "executable": sys.executable,
        "pins": pins,
        "versions": installed_versions(sorted({*pins, "torch"})),
        "pyboot_loaded": "sitecustomize" in sys.modules
        and "pyboot" in str(getattr(sys.modules["sitecustomize"], "__file__", "")),
        "hostname": platform.node(),
        "code_root": str(code_root),
        "git_head": git_head(code_root),
        "registry_has_suite": "srbench_blackbox" in orch._BENCHMARK_REGISTRY,
        "n_problems": len(bbx.SRBENCH_BLACKBOX_BENCHMARKS),
    }


def _check_paths(info: dict[str, Any], code_root: Path) -> list[str]:
    bad: list[str] = []
    if not _under(info["isalsr_file"], code_root / "src" / "isalsr"):
        bad.append(f"isalsr resolves to {info['isalsr_file']}, not under {code_root}/src/isalsr")
    for key in ("orchestrator_file", "srbench_blackbox_file"):
        if not _under(info[key], code_root):
            bad.append(f"{key} = {info[key]} is not under the deployed tree {code_root}")
    if "site-packages" not in info["native_file"]:
        bad.append(f"_native is {info['native_file']}, not the installed site-packages build")
    if info["pyboot_loaded"]:
        bad.append("the pyboot import shim is active; it must not be in this layout")
    return bad


def _check_engine(info: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    if info["engine"] != "cpp":
        bad.append(f"engine is {info['engine']!r}, expected 'cpp'")
    if info["build_info"].get("build_hash") != C2_BUILD_HASH:
        bad.append(f"build_hash {info['build_info'].get('build_hash')} != C2 {C2_BUILD_HASH}")
    if not info["registry_has_suite"]:
        bad.append("orchestrator registry lacks 'srbench_blackbox'")
    return bad


def _check_deploy(info: dict[str, Any], code_root: Path, deploy: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    if not deploy.get("sha256"):
        bad.append(f"{DEPLOY_RECORD} missing or has no sha256 table")
    for rel, expected in deploy.get("sha256", {}).items():
        path = code_root / rel
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != expected:
            bad.append(f"{rel}: sha256 {actual} != deployed {expected}")
    if not deploy.get("code_identity", {}).get("pass"):
        bad.append("deploy record: P1 (arm-deciding files == campaign/c2) did not pass")
    head = info["git_head"]
    if head is not None and head != deploy.get("git_head"):
        bad.append(f"tree HEAD {head} != deployed commit {deploy.get('git_head')}")
    return bad


def _check_versions(info: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    if info["python"] != C2_PYTHON:
        bad.append(f"python {info['python']!r} != C2 {C2_PYTHON!r}")
    for name, want in info["pins"].items():
        have = info["versions"].get(name)
        if have != want:
            bad.append(f"{name} {have} != pinned {want}")
    if info["versions"].get("torch") != TORCH_PIN:
        bad.append(f"torch {info['versions'].get('torch')} != pinned {TORCH_PIN}")
    return bad


def verify(info: dict[str, Any], code_root: Path, deploy: dict[str, Any]) -> list[str]:
    """Return a list of failed checks (empty when everything holds)."""
    return [
        *_check_paths(info, code_root),
        *_check_engine(info),
        *_check_deploy(info, code_root, deploy),
        *_check_versions(info),
    ]


def main(argv: list[str] | None = None) -> int:
    """Print the resolved environment, enforce the checks, optionally stamp."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(os.environ.get("ISALSR_REPO_DIR", Path(__file__).resolve().parents[2])),
        help="the deployed tree's code/ directory",
    )
    parser.add_argument("--write-stamp", type=Path, default=None)
    args = parser.parse_args(argv)
    code_root: Path = args.code_root.resolve()

    deploy_path = code_root / DEPLOY_RECORD
    deploy: dict[str, Any] = json.loads(deploy_path.read_text()) if deploy_path.is_file() else {}
    info = collect(code_root)
    for key in ("isalsr_file", "orchestrator_file", "srbench_blackbox_file", "native_file"):
        print(f"BBX {key:22s} {info[key]}")
    build_hash = info["build_info"].get("build_hash")
    print(f"BBX engine                 {info['engine']}  build_hash={build_hash}")
    print(f"BBX deployed commit        {deploy.get('git_head', '<no BBX_DEPLOY.json>')}")
    print(f"BBX tree HEAD              {info['git_head'] or 'n/a (git unavailable)'}")
    print(f"BBX python                 {info['python']}")
    v = info["versions"]
    print(
        f"BBX bingo/numpy/scipy/sklearn/sympy  {v.get('bingo-nasa')} / {v.get('numpy')} / "
        f"{v.get('scipy')} / {v.get('scikit-learn')} / {v.get('sympy')}"
    )

    failures = verify(info, code_root, deploy)
    for f in failures:
        print(f"[FATAL] {f}", file=sys.stderr)
    if failures:
        return 1
    print(
        f"BBX gate: PASS (modules -> deployed tree, engine C2, files as deployed, "
        f"P1 pass, {len(info['pins'])} pins + python match)"
    )

    if args.write_stamp is not None:
        stamp = {
            "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "deploy": deploy,
            "resolved": info,
        }
        args.write_stamp.parent.mkdir(parents=True, exist_ok=True)
        args.write_stamp.write_text(json.dumps(stamp, indent=2, sort_keys=True))
        print(f"BBX provenance stamp -> {args.write_stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
