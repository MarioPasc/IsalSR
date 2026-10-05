"""The black-box import shim (slurm/blackbox/pyboot/sitecustomize.py).

The ``isalsr`` editable install also redirects ``experiments`` and
``benchmarks`` to the tree it was built from. The shim pins those two packages
to ``$ISALSR_BBX_ROOT`` and leaves ``isalsr`` (and its C++ engine) on the
install. Each case runs in a fresh interpreter, because the shim acts at start-up.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
PYBOOT = CODE_ROOT / "slurm" / "blackbox" / "pyboot"

PROBE = (
    "import isalsr, experiments.models.orchestrator as o, "
    "benchmarks.datasets.srbench_blackbox as s; "
    "print(isalsr.__file__); print(o.__file__); print(s.__file__)"
)


def _run(code: str, root: str | None, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ISALSR_BBX_ROOT")}
    env["PYTHONPATH"] = str(PYBOOT)
    if root is not None:
        env["ISALSR_BBX_ROOT"] = root
    # cwd outside every tree, so nothing resolves from the working directory.
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _isalsr_without_shim(tmp_path: Path) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ISALSR_BBX_ROOT")}
    out = subprocess.run(
        [sys.executable, "-c", "import isalsr; print(isalsr.__file__)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return out.stdout.strip()


def test_resolves_both_packages_from_the_root(tmp_path: Path) -> None:
    res = _run(PROBE, str(CODE_ROOT), tmp_path)
    assert res.returncode == 0, res.stderr
    isalsr_file, orch_file, bbx_file = res.stdout.strip().splitlines()[-3:]
    assert Path(orch_file).resolve().is_relative_to(CODE_ROOT)
    assert Path(bbx_file).resolve().is_relative_to(CODE_ROOT)
    # isalsr stays on the editable install, exactly as without the shim.
    assert isalsr_file == _isalsr_without_shim(tmp_path)


def test_unset_root_fails_loudly(tmp_path: Path) -> None:
    res = _run("import experiments.models.orchestrator", None, tmp_path)
    assert res.returncode != 0
    assert "ISALSR_BBX_ROOT is not set" in res.stderr


def test_missing_root_fails_loudly(tmp_path: Path) -> None:
    res = _run("import benchmarks.datasets", str(tmp_path / "nope"), tmp_path)
    assert res.returncode != 0
    assert "is not a directory" in res.stderr


def test_no_fallback_to_the_install_tree(tmp_path: Path) -> None:
    """A module absent from the root must fail, not resolve from the install."""
    root = tmp_path / "root"
    (root / "experiments").mkdir(parents=True)
    (root / "benchmarks").mkdir()
    res = _run("import benchmarks.datasets.strogatz", str(root), tmp_path)
    assert res.returncode != 0
    assert "refusing to fall back" in res.stderr


def test_isalsr_unaffected_when_root_missing(tmp_path: Path) -> None:
    res = _run("import isalsr; print(isalsr.__file__)", None, tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == _isalsr_without_shim(tmp_path)
