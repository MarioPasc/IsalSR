"""Resolve ``experiments`` and ``benchmarks`` from the black-box tree, nothing else.

Why this file exists
--------------------
The ``isalsr`` editable install (scikit-build-core) ships
``wheel.packages = ["src/isalsr", "experiments", "benchmarks"]`` and installs a
``ScikitBuildRedirectingFinder`` at ``sys.meta_path[0]``. It therefore redirects
**all three** top-level packages to the tree the install was built from, ahead
of ``PYTHONPATH`` and of the working directory. Run from any other tree, the
orchestrator, the runners and the benchmark suites are silently the install
tree's copies, and a module that exists only in the new tree
(``benchmarks.datasets.srbench_blackbox``) cannot be imported at all.

The SRBench black-box campaign needs exactly the opposite split:

* ``isalsr`` (pure Python and the C++ engine) from the existing C2 install, so
  the canonicaliser is the one that produced the main campaign, by construction;
* ``experiments`` and ``benchmarks`` from the deployed black-box tree, which adds
  the new suite and its registry entry.

Python imports ``sitecustomize`` after every ``.pth`` file has run, so a finder
inserted here sits in front of the redirecting finder. It claims only the two
top-level names and their submodules and defers every other name, ``isalsr``
included, to the finders already installed.

Activation: put this directory on ``PYTHONPATH`` and set ``ISALSR_BBX_ROOT`` to
the directory that contains ``experiments/`` and ``benchmarks/``. If the
variable is unset or the directory is missing, any import of either package
raises ``ImportError`` naming the cause. It never falls back to the install
tree, because a silent fallback is the defect this file exists to remove.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence
from types import ModuleType

ROOT_ENV = "ISALSR_BBX_ROOT"
CLAIMED: tuple[str, ...] = ("experiments", "benchmarks")


class BbxTreeFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that pins ``experiments*``/``benchmarks*`` to one root."""

    def __init__(self, root: str | None) -> None:
        self.root = root
        self.error: str | None = None
        if not root:
            self.error = f"{ROOT_ENV} is not set"
        elif not os.path.isdir(root):
            self.error = f"{ROOT_ENV}={root!r} is not a directory"
        else:
            missing = [p for p in CLAIMED if not os.path.isdir(os.path.join(root, p))]
            if missing:
                self.error = f"{ROOT_ENV}={root!r} lacks {missing}"

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Resolve claimed names from the root only; defer every other name."""
        if fullname.partition(".")[0] not in CLAIMED:
            return None
        if self.error is not None or self.root is None:
            raise ImportError(f"bbx pyboot: cannot import {fullname!r}: {self.error}")
        # Top level: search the root alone. Submodules: the parent's __path__,
        # which this finder produced, so it lies inside the root as well.
        search: list[str] = [self.root] if "." not in fullname else list(path or [])
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ModuleNotFoundError(
                f"bbx pyboot: {fullname!r} not found under {self.root!r}; refusing to "
                "fall back to the editable-install tree",
                name=fullname,
            )
        return spec


def install() -> BbxTreeFinder:
    """Insert the finder at the head of ``sys.meta_path`` (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, BbxTreeFinder):
            return finder
    finder = BbxTreeFinder(os.environ.get(ROOT_ENV))
    sys.meta_path.insert(0, finder)
    return finder


install()
