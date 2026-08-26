"""Tests for the viz backend registry round-trip."""

from __future__ import annotations

import pytest

from isalsr.viz.registry import (
    VizBackendNotFoundError,
    available_backends,
    get_backend,
    register_backend,
)


def test_matplotlib_backend_registered_on_import() -> None:
    """The matplotlib backend registers itself on import."""
    # Importing available_backends triggers lazy import of all backends.
    backends = available_backends()
    assert "matplotlib" in backends


def test_get_backend_returns_instance() -> None:
    """get_backend returns a DagVizBackend with the correct name."""
    from isalsr.viz.base import DagVizBackend

    b = get_backend("matplotlib")
    assert isinstance(b, DagVizBackend)
    assert b.name == "matplotlib"


def test_get_backend_returns_fresh_instance_each_call() -> None:
    """Each call to get_backend returns a distinct object."""
    b1 = get_backend("matplotlib")
    b2 = get_backend("matplotlib")
    assert b1 is not b2


def test_register_and_retrieve_custom_backend() -> None:
    """A custom backend can be registered and retrieved by name."""
    from isalsr.core.labeled_dag import LabeledDAG
    from isalsr.viz.base import DagVizBackend, Position

    class DummyBackend(DagVizBackend):
        @property
        def name(self) -> str:
            return "dummy_test"

        def draw(
            self,
            dag: LabeledDAG,
            ax: object,
            *,
            node_colors: dict[int, str] | None = None,
            reachable: frozenset[int] = frozenset(),
            layout: dict[int, Position] | None = None,
        ) -> dict[int, Position]:
            return {}

    register_backend("dummy_test", DummyBackend)
    b = get_backend("dummy_test")
    assert b.name == "dummy_test"
    assert isinstance(b, DummyBackend)


def test_unknown_backend_raises() -> None:
    """Requesting an unregistered backend raises VizBackendNotFoundError."""
    with pytest.raises(VizBackendNotFoundError, match="not registered"):
        get_backend("__no_such_backend__")
