"""The installable package's contract: public names, version, and the README quick start."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import isalsr

README = Path(__file__).resolve().parents[2] / "README.md"


def test_every_public_name_resolves() -> None:
    missing = [name for name in isalsr.__all__ if not hasattr(isalsr, name)]
    assert not missing, f"names in isalsr.__all__ that do not exist: {missing}"


def test_version_is_single_sourced() -> None:
    """The wheel's metadata version is read from isalsr.__version__ (pyproject regex provider)."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", isalsr.__version__)
    assert importlib.metadata.version("isalsr") == isalsr.__version__


def test_engine_reports_a_known_engine() -> None:
    assert isalsr.engine() in {"cpp", "python"}
    assert isalsr.build_info()["engine"] == isalsr.engine()


def test_readme_quick_start_runs_as_written(capsys: object) -> None:
    """Execute the README's quick-start block verbatim and check what it prints."""
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    quick_start = next(b for b in blocks if "fast_canonical_string" in b)
    exec(compile(quick_start, str(README), "exec"), {})  # noqa: S102
    printed = capsys.readouterr().out.split()  # type: ignore[attr-defined]
    assert printed[:2] == ["VcVspv+Ppc", "VcVspv+Ppc"]
    assert printed[2].startswith("1.357008")
