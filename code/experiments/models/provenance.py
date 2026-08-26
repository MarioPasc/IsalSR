"""Run provenance: data fingerprints and configuration digests.

Two quantities that a paired three-arm campaign cannot be verified without,
and that no artefact of campaign C1 carries.

**Data fingerprint (EXECUTION-PLAN P3, check C4).** The paired design compares
``(baseline, hash, isalsr)`` on what is asserted to be the same data. That
assertion has never been checked. It is not idle: the three arms differ in how
much RNG the search consumes, and a generator that drew from a shared global
stream would hand each arm a different sample while every label in the output
still said the same problem and the same seed. The paired test would then be
comparing different data, and nothing downstream would say so.
:func:`data_fingerprint` reduces the four arrays to one digest, so identity
across arms becomes a string equality over the campaign's run logs.

**Config digest (EXECUTION-PLAN A4/A6).** Records *which* YAML produced a run,
by content rather than by path. A config edited between two arrays is otherwise
undetectable after the fact.

Both are cheap: the fingerprint hashes at memory bandwidth, and the largest
benchmark in ``D1 ∪ D2`` is 2,000 x 5 float64 (80 KB).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "config_sha256",
    "data_fingerprint",
]

#: Order in which the four arrays enter the digest. Fixed, because a digest
#: whose value depends on iteration order is not a fingerprint.
_ARRAY_ORDER: tuple[str, ...] = ("x_train", "y_train", "x_test", "y_test")


def data_fingerprint(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
) -> str:
    """Return a SHA-256 digest of a run's four data arrays.

    The digest commits to the name, shape and raw IEEE-754 bytes of each array,
    in the fixed order of :data:`_ARRAY_ORDER`. Arrays are cast to
    ``float64`` and made C-contiguous first, so a host that returns ``float32``
    or a transposed view of otherwise identical data still fingerprints
    identically -- the quantity being certified is the *sample*, not the
    container that carries it.

    Byte equality is deliberately stricter than numerical equality: ``-0.0`` and
    ``0.0`` compare equal but hash differently. That is the intended behaviour.
    Every arm calls the same generator with the same seed, so the arms are
    expected to agree bit for bit; anything less is the confound check C4
    exists to catch.

    Args:
        x_train: Training inputs, shape ``(n_train, n_features)``.
        y_train: Training targets, shape ``(n_train,)``.
        x_test: Test inputs, shape ``(n_test, n_features)``.
        y_test: Test targets, shape ``(n_test,)``.

    Returns:
        Lowercase hexadecimal SHA-256 digest, 64 characters.

    Raises:
        ValueError: If any array cannot be cast to a real ``float64`` array,
            which means the generator returned something the campaign cannot
            fingerprint and therefore cannot certify.
    """
    digest = hashlib.sha256()
    for name, array in zip(_ARRAY_ORDER, (x_train, y_train, x_test, y_test), strict=True):
        try:
            values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} cannot be cast to a float64 array, so the run cannot be "
                f"fingerprinted and cross-arm data identity (EXECUTION-PLAN C4) "
                f"cannot be verified for it.",
            ) from exc
        digest.update(name.encode("utf-8"))
        # Shape is committed separately: two arrays with the same bytes and
        # different shapes are different data, and tobytes() alone would not
        # distinguish a (1000, 1) from a (1000,).
        digest.update(repr(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def config_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest of a configuration file's bytes.

    Hashes the file as written rather than the parsed structure, so a comment
    change registers. That is the conservative direction: a spurious digest
    difference prompts a check, a missed one hides a hyperparameter edit made
    between two of the six arrays.

    Args:
        path: Path to the YAML configuration.

    Returns:
        Lowercase hexadecimal SHA-256 digest, or ``"unavailable"`` if the file
        cannot be read. Never raises: provenance capture must not be the thing
        that kills a 12 h run.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"
