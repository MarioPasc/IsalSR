"""Deduplication baselines against which the IsalSR canonical string is compared.

Two independent pieces live here:

- :mod:`isalsr.baselines.fixed_order_hash` -- three fixed-order serialisations of
  a labeled DAG, their live and stable hashes, and a lossless decoder.  Each is
  sound (never merges non-isomorphic DAGs) and incomplete (fails to merge
  isomorphic DAGs whose node numbering differs).
- :mod:`isalsr.baselines.cardinality` -- a HyperLogLog sketch, so a run can keep
  several shadow distinct-counters in kilobytes rather than gigabytes.

Dependency layer: stdlib and :mod:`isalsr.core` only.  Nothing in
:mod:`isalsr.core` may import this package.
"""

from __future__ import annotations

from isalsr.baselines.cardinality import HyperLogLog
from isalsr.baselines.fixed_order_hash import (
    FixedOrder,
    SerialisationError,
    deserialise,
    fixed_order_digest,
    fixed_order_hash,
    node_order,
    serialise,
)

__all__ = [
    "FixedOrder",
    "HyperLogLog",
    "SerialisationError",
    "deserialise",
    "fixed_order_digest",
    "fixed_order_hash",
    "node_order",
    "serialise",
]
