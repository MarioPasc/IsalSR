"""Name-keyed registry for DAG visualisation backends.

Concrete backend modules register themselves at import time via
:func:`register_backend`. The :func:`get_backend` accessor lazily imports
the backend module on first request so optional library dependencies stay
out of the import path until used.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable

from isalsr.viz.base import DagVizBackend


class VizBackendNotFoundError(Exception):
    """Raised when a requested viz backend is not registered."""


_BACKENDS: dict[str, Callable[[], DagVizBackend]] = {}

# Map of backend name -> module path that will register it on import.
_LAZY_MODULES: dict[str, str] = {
    "matplotlib": "isalsr.viz.backends.matplotlib_dag",
}


def register_backend(
    name: str,
    factory: Callable[[], DagVizBackend],
) -> None:
    """Register ``factory`` under ``name`` (overwrites any prior entry)."""
    _BACKENDS[name] = factory


def get_backend(name: str) -> DagVizBackend:
    """Return a fresh instance of the backend registered under ``name``.

    Triggers a lazy import of the backend's module if it has not yet been
    imported in this session.

    Raises
    ------
    VizBackendNotFoundError
        If ``name`` is unknown after the lazy import attempt.
    """
    if name not in _BACKENDS and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])
    if name not in _BACKENDS:
        raise VizBackendNotFoundError(
            f"viz backend {name!r} is not registered (known: {sorted(_BACKENDS)})"
        )
    return _BACKENDS[name]()


def available_backends() -> tuple[str, ...]:
    """Return the sorted tuple of registered backend names.

    Triggers lazy import of every backend module first so the listing
    reflects the actually-installed set rather than only whatever has
    been requested earlier in the session.
    """
    for name, module_path in _LAZY_MODULES.items():
        if name not in _BACKENDS:
            with contextlib.suppress(ImportError):
                importlib.import_module(module_path)
    return tuple(sorted(_BACKENDS.keys()))
