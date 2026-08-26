"""Host-native serialisation: the naive rung of the deduplication ladder.

Motivation
----------
:mod:`isalsr.baselines.fixed_order_hash` serialises a
:class:`~isalsr.core.labeled_dag.LabeledDAG`, i.e. the *output of the IsalSR
adapter*.  That is not a naive baseline.  Both host adapters renumber: the UDFS
adapter emits variables first (by variable index), then constants, then operators
in evaluation order, and the Bingo adapter emits variables first, then the
utilised command rows.  That layout is itself a partial canonical form, and it
alone merged 148,347 of 591,190 distinct host representations (25.1 %) on a
stream of 604,334 real UDFS candidates -- before any hashing happened.

A naive baseline must therefore key on the *host's own* structure in the *host's
own* node order, with no renumbering, no re-sorting, no topological pass and no
canonical layout.  The order is given by the host; it is never computed from the
graph.

Definition
----------
A host-native record is a triple ``(key, tag, operands)``:

``key``
    The host's own identifier for the node (a UDFS ``node_dict`` key, a Bingo
    ``command_array`` row index).  Emitted verbatim.
``tag``
    The host's own label for the node (a UDFS operation string, a Bingo opcode).
``operands``
    The node's operand list **as host keys, in the host's stored order**.

:func:`host_native_serialise` emits the records in the order it receives them.
The caller is responsible for supplying that order from the host, and for
restricting the record list to the nodes that contribute to the host's output.

Soundness
---------
Equal host-native serialisation implies an identical host graph on the same key
set with the same labels and the same operand order.  The identity map on keys is
then a label- and operand-order-preserving isomorphism between the two host
graphs, so any deterministic function of the host graph -- in particular the
adapter followed by :func:`~isalsr.core.canonical.fast_canonical_string` --
returns the same value.  Hence

    ``host_native_serialise(A) == host_native_serialise(B)``
    ``=> fast_canonical_string(adapt(A)) == fast_canonical_string(adapt(B))``.

The converse fails, and fails more often than for the fixed orders of
:mod:`isalsr.baselines.fixed_order_hash`: host keys are exactly what a
relabelling permutes, and the host's key order is not isomorphism-equivariant.
The rung is therefore **sound and strictly more incomplete**, which is the
correct position for a naive baseline.

Restriction: this module depends only on the Python standard library.  It knows
nothing about any host and imports nothing from ``experiments``; host-specific
extraction of the record list belongs to the host's runner.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import TypeAlias

__all__ = [
    "HostNativeRecord",
    "HostNativeSerialisationError",
    "host_native_serialise",
    "host_native_hash",
    "host_native_digest",
]

#: ``(host key, host label, operand host keys in the host's stored order)``.
HostNativeRecord: TypeAlias = "tuple[int, str, Sequence[int]]"

# Structural separators.  A tag containing any of them is rejected, which is what
# makes the grammar unambiguously parseable and hence the encoding injective.
_HEADER_SEP = "|"
_RECORD_SEP = ";"
_KEY_SEP = ":"
_OPERANDS_OPEN = "<"
_OPERANDS_CLOSE = ">"
_OPERAND_SEP = ","

_FORBIDDEN_IN_TAG = frozenset(
    {_HEADER_SEP, _RECORD_SEP, _KEY_SEP, _OPERANDS_OPEN, _OPERANDS_CLOSE, _OPERAND_SEP}
)


class HostNativeSerialisationError(ValueError):
    """Raised when a host-native record list cannot be serialised."""


def host_native_serialise(records: Iterable[HostNativeRecord]) -> str:
    """Serialise host-native records in the order given.

    The grammar is ``<n>|<record>;<record>;...`` with one record per node, each
    record being ``<key>:<tag><o0,o1,...>`` where ``key`` and the ``o`` are the
    host's own identifiers, emitted verbatim.

    Args:
        records: The host's nodes as ``(key, tag, operands)`` triples, already in
            the host's own order.  The iterable is consumed once.

    Returns:
        The serialisation string.

    Raises:
        HostNativeSerialisationError: If a tag contains a structural separator or
            is empty, or a key or operand is not an integer.
    """
    encoded: list[str] = []
    for key, tag, operands in records:
        if not tag:
            raise HostNativeSerialisationError("Empty tag")
        if _FORBIDDEN_IN_TAG.intersection(tag):
            raise HostNativeSerialisationError(f"Tag {tag!r} contains a structural separator")
        try:
            key_int = int(key)
            operand_ints = [int(operand) for operand in operands]
        except (TypeError, ValueError) as exc:
            raise HostNativeSerialisationError(
                f"Non-integer key or operand in record ({key!r}, {tag!r})"
            ) from exc
        joined = _OPERAND_SEP.join(str(operand) for operand in operand_ints)
        encoded.append(f"{key_int}{_KEY_SEP}{tag}{_OPERANDS_OPEN}{joined}{_OPERANDS_CLOSE}")
    return f"{len(encoded)}{_HEADER_SEP}{_RECORD_SEP.join(encoded)}"


def host_native_hash(records: Iterable[HostNativeRecord]) -> int:
    """Return the CPython builtin hash of the host-native serialisation.

    This is the **live** deduplication key of the naive arm.  It uses
    ``hash(str)`` so that its memory footprint and collision profile match the
    IsalSR deduplication set exactly, which stores ``hash(canonical_string)``.
    Builtin ``hash`` is SipHash and is randomised per process by
    ``PYTHONHASHSEED``, so the value cannot be replayed across processes; use
    :func:`host_native_digest` for that.

    Args:
        records: The host's nodes as ``(key, tag, operands)`` triples, in the
            host's own order.

    Returns:
        A 64-bit signed process-local hash.

    Raises:
        HostNativeSerialisationError: If the records cannot be serialised.
    """
    return hash(host_native_serialise(records))


def host_native_digest(records: Iterable[HostNativeRecord]) -> int:
    """Return a stable 64-bit BLAKE2b digest of the host-native serialisation.

    Unlike :func:`host_native_hash` this value is reproducible across processes
    and machines, which makes it the correct key for a persisted stream and for
    offline replay.

    Args:
        records: The host's nodes as ``(key, tag, operands)`` triples, in the
            host's own order.

    Returns:
        An unsigned 64-bit integer digest.

    Raises:
        HostNativeSerialisationError: If the records cannot be serialised.
    """
    payload = host_native_serialise(records).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
