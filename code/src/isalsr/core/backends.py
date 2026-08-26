"""Engine-selection layer for isalsr.core.

Exposes the active computation engine (C++ extension or pure-Python fallback),
a build-info dict, and a generic dispatcher used by user-facing functions.

Engine resolution order
-----------------------
1. ``ISALSR_ENGINE`` environment variable (``"cpp"`` | ``"python"``).
2. ``DEFAULT_BACKEND``: ``"cpp"`` when the native extension loaded successfully,
   ``"python"`` otherwise.

Adding a new backend
--------------------
Add the callable to the per-function ``registry`` dict passed to
:func:`resolve` and it becomes available as a ``backend=`` keyword.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, TypeVar

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the C++ extension — graceful fallback to Python.
# ---------------------------------------------------------------------------
try:
    from isalsr.core import _native as _cpp_ext  # type: ignore[attr-defined]

    _CPP_AVAILABLE: bool = True
    log.debug("isalsr.core._native loaded (C++ engine active)")
except ImportError as _err:
    _cpp_ext = None
    _CPP_AVAILABLE = False
    log.debug("isalsr.core._native not available (%s); using Python engine", _err)

# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------
DEFAULT_BACKEND: Literal["cpp", "python"] = "cpp" if _CPP_AVAILABLE else "python"
"""Active default backend.  ``"cpp"`` when the native extension is importable,
``"python"`` otherwise."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

Backend = Literal["cpp", "python"]
"""Type alias for the ``backend=`` / ``ISALSR_ENGINE`` values."""

C = TypeVar("C")


def engine() -> Literal["cpp", "python"]:
    """Return the name of the currently active engine.

    The env variable ``ISALSR_ENGINE`` overrides the compiled-in default.
    Requesting ``"cpp"`` when the extension is absent raises
    :class:`RuntimeError`.

    Returns
    -------
    Literal["cpp", "python"]
        Active engine name.

    Raises
    ------
    RuntimeError
        When ``ISALSR_ENGINE=cpp`` but the C++ extension is unavailable.
    """
    override = os.environ.get("ISALSR_ENGINE", "").strip().lower()
    if override == "cpp":
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "ISALSR_ENGINE=cpp requested but isalsr.core._native could not be "
                "imported.  Build the extension with: "
                "pip install -e '.[dev,native]'"
            )
        return "cpp"
    if override == "python":
        return "python"
    if override and override not in ("cpp", "python"):
        raise ValueError(f"ISALSR_ENGINE={override!r} is not valid; choose 'cpp' or 'python'.")
    return DEFAULT_BACKEND


def build_info() -> dict[str, str]:
    """Return build metadata from the active engine.

    When the C++ extension is active, metadata is read from the compiled
    probe TU (compiler, ISA level, flags, build hash).  When the Python
    engine is active, returns a minimal dict.

    Returns
    -------
    dict[str, str]
        Compiler/ISA/flag metadata.  Keys are always present; values are
        strings (empty string means not applicable).
    """
    active = engine()
    if active == "cpp" and _cpp_ext is not None:
        info: dict[str, str] = _cpp_ext.build_info()
        info["engine"] = "cpp"
        return info
    return {
        "engine": "python",
        "compiler": "",
        "cplusplus": "",
        "isa_level": "",
        "avx2": "",
        "fma": "",
        "avx512f": "",
        "ndebug": "",
        "build_hash": "",
    }


def resolve(backend: str | None, registry: dict[str, C]) -> C:
    """Return the callable registered under *backend*.

    Parameters
    ----------
    backend : str or None
        Backend name.  ``None`` resolves to :func:`engine` (respects
        ``ISALSR_ENGINE`` override).
    registry : dict[str, C]
        Per-function dispatch table mapping backend names to callables.

    Returns
    -------
    C
        The chosen implementation.

    Raises
    ------
    ValueError
        When *backend* is not a key of *registry*.
    RuntimeError
        When *backend* resolves to ``"cpp"`` but the extension is absent.
    """
    chosen: str = engine() if backend is None else backend
    # Validate "cpp" availability even when caller passes it explicitly.
    if chosen == "cpp" and not _CPP_AVAILABLE:
        raise RuntimeError(
            "backend='cpp' requested but isalsr.core._native is not available. "
            "Run: pip install -e '.[dev,native]'"
        )
    impl = registry.get(chosen)
    if impl is None:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unknown backend {chosen!r}; valid options: {valid}")
    return impl
