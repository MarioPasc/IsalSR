"""Locate the ``docs/`` tree, which lives outside the packaged project.

Why this module exists
----------------------
The repository root and the *project* root are not the same directory. Code
Ocean's git import only takes the contents of ``code/``, ``data/``,
``environment/`` and ``metadata/``, so the packaged project was moved under
``code/`` and the repository root now holds only capsule scaffolding plus the
companion website in ``docs/`` (GitHub Pages serves it from ``main:/docs``, so
it cannot move).

Every script here computes its own root as ``Path(__file__).parents[N]``, which
resolves to the *project* root and is correct for in-project paths such as
``experiments/configs``. It is one level short for ``docs/``. Bumping the index
would fix the docs paths and break the in-project ones, so the two roots are
kept distinct and ``docs/`` is resolved by searching upward for it rather than
by counting directories --- counting is what broke when the tree moved, and it
would break again on the next move.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["docs_dir", "docs_root"]


def docs_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor directory that contains a ``docs/`` tree.

    Args:
        start: Directory to search upward from. Defaults to this file's
            directory, which makes the answer independent of the caller's
            working directory.

    Returns:
        The directory holding ``docs/``. Falls back to the project root when no
        ancestor has one, so a checkout without the website still imports.
    """
    origin = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "docs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def docs_dir(*parts: str) -> Path:
    """Return a path inside the ``docs/`` tree.

    Args:
        *parts: Path components below ``docs/``.

    Returns:
        The resolved path, which is not required to exist.
    """
    return docs_root().joinpath("docs", *parts)
