"""Precomputed canonical string cache for IsalSR.

Precomputes and persists canonical string representations for all four
D2S algorithms, eliminating redundant canonicalization across seeds,
SR methods, and experiment reruns.

Optional dependency: h5py >= 3.8 (for HDF5 persistence).

Reference: docs/design/precomputed_cache_design.md
"""

from __future__ import annotations

from isalsr.precomputed.cache_entry import CacheEntry, CacheStats, dag_depth

try:
    import h5py as _h5py  # noqa: F401

    HAS_H5PY: bool = True
except ImportError:
    HAS_H5PY = False

__all__: list[str] = [
    "CacheEntry",
    "CacheStats",
    "HAS_H5PY",
    "dag_depth",
]
