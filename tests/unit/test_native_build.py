"""Tests for the C++ native extension build and the backends dispatch layer.

Three test groups:
    1. FNV-1a 64-bit correctness — C++ output vs. pure-Python reference.
    2. backends.engine() / resolve() contract.
    3. Fallback path (ISALSR_ENGINE=python) verified via subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Pure-Python FNV-1a 64-bit reference (ticket §5.5)
# Must stay in sync with native/include/isalsr/fnv.hpp constants.
# ---------------------------------------------------------------------------
_FNV_OFFSET: int = 0xCBF29CE484222325
_FNV_PRIME: int = 0x00000100000001B3
_MASK64: int = 0xFFFFFFFFFFFFFFFF


def _py_fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return h


# ---------------------------------------------------------------------------
# FNV-1a 64-bit tests (parametrised)
# ---------------------------------------------------------------------------

# 20 test vectors: empty, ASCII, high-byte, known reference values.
_FNV_VECTORS: list[bytes] = [
    b"",
    b"\x00",
    b"\xff",
    b"\x80",
    b"a",
    b"foobar",
    b"hello world",
    b"IsalSR",
    b"\x00\x01\x02\x03",
    b"\xff\xfe\xfd\xfc",
    b"the quick brown fox",
    b"x" * 256,
    b"\x00" * 64,
    b"\xff" * 64,
    b"cpp engine",
    b"\x01\x80\xfe\x7f\x00\xff",
    b"canonical_string",
    b"0" * 1,
    b"labeled_dag",
    b"\xde\xad\xbe\xef\xca\xfe\xba\xbe",
]


@pytest.mark.skipif(
    "isalsr.core._native" not in sys.modules
    and not pytest.importorskip("isalsr.core._native", reason="C++ extension not built"),
    reason="C++ extension not built",
)
class TestFnv1a64:
    """Validate fnv1a64 output matches the pure-Python reference."""

    @pytest.fixture(autouse=True)
    def _import_native(self) -> None:
        self._native = pytest.importorskip("isalsr.core._native")

    @pytest.mark.parametrize("data", _FNV_VECTORS)
    def test_matches_python_reference(self, data: bytes) -> None:
        expected = _py_fnv1a64(data)
        got = self._native.fnv1a64(data)
        assert got == expected, (
            f"fnv1a64({data!r}): C++ returned {got:#018x}, Python returned {expected:#018x}"
        )

    def test_empty_bytes_is_offset_basis(self) -> None:
        assert self._native.fnv1a64(b"") == _FNV_OFFSET

    def test_return_is_unsigned_int(self) -> None:
        v = self._native.fnv1a64(b"\xff" * 8)
        assert isinstance(v, int)
        assert 0 <= v <= 0xFFFFFFFFFFFFFFFF

    def test_high_byte_no_sign_extension(self) -> None:
        # Ensures the C++ side casts bytes to uint8_t, not int8_t.
        v = self._native.fnv1a64(b"\x80")
        assert v == _py_fnv1a64(b"\x80")


# ---------------------------------------------------------------------------
# backends module contract
# ---------------------------------------------------------------------------


class TestBackendsContract:
    """Verify engine(), resolve(), and DEFAULT_BACKEND semantics."""

    def test_engine_returns_literal(self) -> None:
        from isalsr.core import backends

        result = backends.engine()
        assert result in ("cpp", "python")

    def test_default_backend_consistent_with_engine(self) -> None:
        from isalsr.core import backends

        # When no env override is set, engine() should equal DEFAULT_BACKEND.
        saved = os.environ.pop("ISALSR_ENGINE", None)
        try:
            assert backends.engine() == backends.DEFAULT_BACKEND
        finally:
            if saved is not None:
                os.environ["ISALSR_ENGINE"] = saved

    def test_resolve_returns_registered_callable(self) -> None:
        from isalsr.core import backends

        sentinel_py: dict[str, Any] = {}
        sentinel_cpp: dict[str, Any] = {}
        registry: dict[str, Any] = {"python": sentinel_py, "cpp": sentinel_cpp}

        # Force python backend to avoid dependency on extension availability.
        result = backends.resolve("python", registry)
        assert result is sentinel_py

    def test_resolve_unknown_raises_value_error(self) -> None:
        from isalsr.core import backends

        registry: dict[str, Any] = {"python": lambda: None}
        with pytest.raises(ValueError, match="Unknown backend"):
            backends.resolve("rust", registry)

    def test_resolve_none_uses_engine(self) -> None:
        from isalsr.core import backends

        # Build a registry that covers whichever engine is active.
        active = backends.engine()
        registry: dict[str, Any] = {active: object()}
        result = backends.resolve(None, registry)
        assert result is registry[active]

    def test_build_info_is_populated_dict(self) -> None:
        from isalsr.core import backends

        info = backends.build_info()
        assert isinstance(info, dict)
        assert "engine" in info
        assert info["engine"] in ("cpp", "python")
        # Required keys must always be present (may be empty string for Python).
        for key in ("compiler", "isa_level", "ndebug", "build_hash"):
            assert key in info, f"Missing key {key!r} in build_info()"

    def test_engine_override_cpp_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force ISALSR_ENGINE=cpp with _CPP_AVAILABLE=False → RuntimeError."""
        from isalsr.core import backends

        monkeypatch.setenv("ISALSR_ENGINE", "cpp")
        monkeypatch.setattr(backends, "_CPP_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="isalsr.core._native"):
            backends.engine()

    def test_engine_invalid_override_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISALSR_ENGINE", "rust")
        from isalsr.core import backends

        with pytest.raises(ValueError, match="ISALSR_ENGINE"):
            backends.engine()


# ---------------------------------------------------------------------------
# Fallback path (ISALSR_ENGINE=python) via subprocess
# ---------------------------------------------------------------------------


class TestFallbackPath:
    """Verify that the Python engine is selected when forced via environment."""

    def test_env_python_override_in_subprocess(self) -> None:
        """ISALSR_ENGINE=python must make engine() return 'python'."""
        env = {**os.environ, "ISALSR_ENGINE": "python"}
        result = subprocess.run(
            [sys.executable, "-c", "from isalsr.core import backends; print(backends.engine())"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "python"

    def test_env_python_build_info_engine_key(self) -> None:
        """build_info()['engine'] must be 'python' under the override."""
        env = {**os.environ, "ISALSR_ENGINE": "python"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from isalsr.core import backends; "
                "import json; print(json.dumps(backends.build_info()))",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        import json

        info = json.loads(result.stdout.strip())
        assert info.get("engine") == "python"
