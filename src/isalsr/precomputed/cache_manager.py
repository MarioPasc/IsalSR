"""Cache manager for precomputed canonical strings.

Provides generation, HDF5 persistence, and lookup of canonical string
representations for all four D2S algorithms.

Optional dependency: h5py >= 3.8 (lazy-imported in HDF5 methods).

Reference: docs/design/precomputed_cache_design.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from isalsr.core.canonical import (
    CanonicalTimeoutError,
    canonical_string,
    pruned_canonical_string,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import LEAF_TYPES, OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.precomputed.cache_entry import CacheEntry, CacheStats, dag_depth

log = logging.getLogger(__name__)


# ======================================================================
# CacheManager
# ======================================================================


class CacheManager:
    """Manages precomputed canonical string cache with HDF5 persistence.

    Two modes of operation:
    1. **Generation**: compute entries via ``compute_and_add``, then
       ``flush_hdf5`` to persist.
    2. **Lookup**: ``load_hdf5`` an existing file, then query by index
       or iterate.

    Args:
        num_variables: Number of input variables (m).
        operator_set: The allowed operation set.
        exhaustive_timeout: Timeout in seconds for exhaustive canonical
            computation. None for no limit.
    """

    __slots__ = (
        "_entries",
        "_num_variables",
        "_operator_set",
        "_exhaustive_timeout",
    )

    def __init__(
        self,
        num_variables: int,
        operator_set: OperationSet,
        exhaustive_timeout: float | None = 60.0,
    ) -> None:
        self._entries: list[CacheEntry] = []
        self._num_variables = num_variables
        self._operator_set = operator_set
        self._exhaustive_timeout = exhaustive_timeout

    # ------------------------------------------------------------------
    # Entry computation
    # ------------------------------------------------------------------

    def compute_entry(self, raw_string: str) -> CacheEntry | None:
        """Compute a cache entry for a raw IsalSR string.

        Runs S2D on the string, then all four D2S algorithms with timing.
        Returns None if the string produces a trivial (VAR-only) DAG
        or if S2D fails.

        Args:
            raw_string: The raw IsalSR instruction string.

        Returns:
            A frozen CacheEntry, or None if the DAG is trivial/invalid.
        """
        try:
            dag = StringToDAG(
                raw_string,
                self._num_variables,
                self._operator_set,
            ).run()
        except Exception:  # noqa: BLE001
            return None

        n_vars = len(dag.var_nodes())
        if dag.node_count <= n_vars:
            return None  # VAR-only DAG, nothing to encode.

        # Check for single output node (skip multi-sink DAGs).
        try:
            dag.output_node()
        except ValueError:
            return None

        return self._compute_entry_from_dag(dag, raw_string)

    def _compute_entry_from_dag(
        self,
        dag: LabeledDAG,
        raw_string: str,
    ) -> CacheEntry:
        """Compute all D2S variants and properties for a DAG."""
        from isalsr.core.algorithms.greedy_min import GreedyMinD2S
        from isalsr.core.algorithms.greedy_single import GreedySingleD2S

        nv = self._num_variables
        timeout = self._exhaustive_timeout

        # --- Greedy single ---
        t0 = time.perf_counter()
        greedy_single = GreedySingleD2S().encode(dag)
        t_gs = time.perf_counter() - t0

        # --- Greedy min ---
        t0 = time.perf_counter()
        greedy_min = GreedyMinD2S().encode(dag)
        t_gm = time.perf_counter() - t0

        # --- Pruned canonical (direct call for timeout support) ---
        t0 = time.perf_counter()
        try:
            pruned = pruned_canonical_string(dag, timeout=timeout)
            t_pru = time.perf_counter() - t0
        except CanonicalTimeoutError:
            t_pru = time.perf_counter() - t0
            pruned = greedy_min  # Fallback
            log.warning("Pruned canonical timed out for: %s", raw_string[:60])

        # --- Exhaustive canonical (direct call for timeout support) ---
        t0 = time.perf_counter()
        exhaustive_timed_out = False
        try:
            exhaustive = canonical_string(dag, timeout=timeout)
            t_exh = time.perf_counter() - t0
        except CanonicalTimeoutError:
            t_exh = -1.0
            exhaustive = ""
            exhaustive_timed_out = True
            log.warning("Exhaustive canonical timed out for: %s", raw_string[:60])

        # --- DAG properties ---
        depth = dag_depth(dag)
        n_vars = len(dag.var_nodes())

        # --- Correctness flags ---
        exh_eq_pru = (not exhaustive_timed_out) and (exhaustive == pruned)
        gs_eq_exh = (not exhaustive_timed_out) and (greedy_single == exhaustive)
        gm_eq_exh = (not exhaustive_timed_out) and (greedy_min == exhaustive)

        return CacheEntry(
            raw_string=raw_string,
            num_variables=nv,
            n_nodes=dag.node_count,
            n_internal=dag.node_count - n_vars,
            n_edges=dag.edge_count,
            n_var_nodes=n_vars,
            depth=depth,
            greedy_single=greedy_single,
            greedy_min=greedy_min,
            pruned=pruned,
            exhaustive=exhaustive,
            exhaustive_timed_out=exhaustive_timed_out,
            timing_greedy_single=t_gs,
            timing_greedy_min=t_gm,
            timing_pruned=t_pru,
            timing_exhaustive=t_exh,
            is_canonical=(raw_string == pruned),
            exhaustive_eq_pruned=exh_eq_pru,
            greedy_single_eq_exhaustive=gs_eq_exh,
            greedy_min_eq_exhaustive=gm_eq_exh,
        )

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(self, entry: CacheEntry) -> None:
        """Append a cache entry to the internal list."""
        self._entries.append(entry)

    def compute_and_add(self, raw_string: str) -> bool:
        """Compute and add a cache entry. Returns True if added."""
        entry = self.compute_entry(raw_string)
        if entry is not None:
            self._entries.append(entry)
            return True
        return False

    @property
    def entries(self) -> list[CacheEntry]:
        """Return the list of cache entries (read-only view)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Compute aggregate statistics from loaded entries."""
        entries = self._entries
        n = len(entries)
        if n == 0:
            return CacheStats(
                total_entries=0,
                unique_canonical_pruned=0,
                unique_canonical_exhaustive=0,
                exhaustive_timeout_count=0,
                exhaustive_eq_pruned_count=0,
                greedy_single_eq_exhaustive_count=0,
                greedy_min_eq_exhaustive_count=0,
                avg_depth=0.0,
                max_depth=0,
                avg_internal_nodes=0.0,
            )

        pruned_set: set[str] = set()
        exhaustive_set: set[str] = set()
        timeout_count = 0
        eq_pruned = 0
        gs_eq = 0
        gm_eq = 0
        total_depth = 0
        max_d = 0
        total_internal = 0

        for e in entries:
            pruned_set.add(e.pruned)
            if not e.exhaustive_timed_out:
                exhaustive_set.add(e.exhaustive)
            if e.exhaustive_timed_out:
                timeout_count += 1
            if e.exhaustive_eq_pruned:
                eq_pruned += 1
            if e.greedy_single_eq_exhaustive:
                gs_eq += 1
            if e.greedy_min_eq_exhaustive:
                gm_eq += 1
            total_depth += e.depth
            if e.depth > max_d:
                max_d = e.depth
            total_internal += e.n_internal

        return CacheStats(
            total_entries=n,
            unique_canonical_pruned=len(pruned_set),
            unique_canonical_exhaustive=len(exhaustive_set),
            exhaustive_timeout_count=timeout_count,
            exhaustive_eq_pruned_count=eq_pruned,
            greedy_single_eq_exhaustive_count=gs_eq,
            greedy_min_eq_exhaustive_count=gm_eq,
            avg_depth=total_depth / n,
            max_depth=max_d,
            avg_internal_nodes=total_internal / n,
        )

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------

    def flush_hdf5(self, path: Path | str) -> None:
        """Write all entries to an HDF5 file.

        Creates the file structure specified in the design doc Section 4.2.
        Overwrites if the file already exists.

        Args:
            path: Output HDF5 file path.
        """
        import h5py
        import numpy as np

        path = Path(path)
        entries = self._entries
        n = len(entries)
        if n == 0:
            log.warning("No entries to flush.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        vlen_str = h5py.special_dtype(vlen=str)

        with h5py.File(path, "w") as f:
            # --- Root attributes ---
            f.attrs["num_variables"] = self._num_variables
            f.attrs["operator_set"] = json.dumps(
                sorted(op.value for op in self._operator_set.ops if op not in LEAF_TYPES)
            )
            f.attrs["creation_timestamp"] = datetime.now(tz=UTC).isoformat()
            f.attrs["isalsr_version"] = "0.1.0"
            f.attrs["git_hash"] = _git_hash()
            f.attrs["total_entries"] = n

            # --- strings/ group ---
            g = f.create_group("strings")
            g.create_dataset("raw", data=[e.raw_string for e in entries], dtype=vlen_str)
            g.create_dataset(
                "greedy_single", data=[e.greedy_single for e in entries], dtype=vlen_str
            )
            g.create_dataset("greedy_min", data=[e.greedy_min for e in entries], dtype=vlen_str)
            g.create_dataset("pruned", data=[e.pruned for e in entries], dtype=vlen_str)
            g.create_dataset("exhaustive", data=[e.exhaustive for e in entries], dtype=vlen_str)
            g.create_dataset(
                "is_canonical",
                data=np.array([e.is_canonical for e in entries], dtype=bool),
            )

            # --- dag_properties/ group ---
            g = f.create_group("dag_properties")
            g.create_dataset("n_nodes", data=np.array([e.n_nodes for e in entries], dtype=np.int32))
            g.create_dataset(
                "n_internal",
                data=np.array([e.n_internal for e in entries], dtype=np.int32),
            )
            g.create_dataset("n_edges", data=np.array([e.n_edges for e in entries], dtype=np.int32))
            g.create_dataset(
                "n_var_nodes",
                data=np.array([e.n_var_nodes for e in entries], dtype=np.int32),
            )
            g.create_dataset("depth", data=np.array([e.depth for e in entries], dtype=np.int32))

            # --- timings/ group ---
            g = f.create_group("timings")
            g.create_dataset(
                "greedy_single",
                data=np.array([e.timing_greedy_single for e in entries], dtype=np.float64),
            )
            g.create_dataset(
                "greedy_min",
                data=np.array([e.timing_greedy_min for e in entries], dtype=np.float64),
            )
            g.create_dataset(
                "pruned",
                data=np.array([e.timing_pruned for e in entries], dtype=np.float64),
            )
            g.create_dataset(
                "exhaustive",
                data=np.array([e.timing_exhaustive for e in entries], dtype=np.float64),
            )

            # --- correctness/ group ---
            g = f.create_group("correctness")
            g.create_dataset(
                "exhaustive_timed_out",
                data=np.array([e.exhaustive_timed_out for e in entries], dtype=bool),
            )
            g.create_dataset(
                "exhaustive_eq_pruned",
                data=np.array([e.exhaustive_eq_pruned for e in entries], dtype=bool),
            )
            g.create_dataset(
                "greedy_single_eq_exhaustive",
                data=np.array([e.greedy_single_eq_exhaustive for e in entries], dtype=bool),
            )
            g.create_dataset(
                "greedy_min_eq_exhaustive",
                data=np.array([e.greedy_min_eq_exhaustive for e in entries], dtype=bool),
            )

            # --- canonical_index/ group (deduplicated by pruned canonical) ---
            pruned_to_indices: dict[str, list[int]] = {}
            for i, e in enumerate(entries):
                pruned_to_indices.setdefault(e.pruned, []).append(i)

            unique_pruned = sorted(pruned_to_indices.keys())
            g = f.create_group("canonical_index")
            g.create_dataset("unique_canonical_pruned", data=unique_pruned, dtype=vlen_str)
            g.create_dataset(
                "canonical_to_first",
                data=np.array([pruned_to_indices[c][0] for c in unique_pruned], dtype=np.int64),
            )
            g.create_dataset(
                "multiplicity",
                data=np.array([len(pruned_to_indices[c]) for c in unique_pruned], dtype=np.int64),
            )

        log.info("Flushed %d entries to %s", n, path)

    def load_hdf5(self, path: Path | str) -> None:
        """Load cache entries from an HDF5 file.

        Appends to the internal entry list (supports merging multiple files).

        Args:
            path: Input HDF5 file path.
        """
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            nv = int(f.attrs["num_variables"])
            n = int(f.attrs["total_entries"])

            raw = f["strings/raw"]
            gs = f["strings/greedy_single"]
            gm = f["strings/greedy_min"]
            pru = f["strings/pruned"]
            exh = f["strings/exhaustive"]
            is_can = f["strings/is_canonical"]

            nn = f["dag_properties/n_nodes"]
            ni = f["dag_properties/n_internal"]
            ne = f["dag_properties/n_edges"]
            nvn = f["dag_properties/n_var_nodes"]
            dep = f["dag_properties/depth"]

            t_gs = f["timings/greedy_single"]
            t_gm = f["timings/greedy_min"]
            t_pru = f["timings/pruned"]
            t_exh = f["timings/exhaustive"]

            exh_to = f["correctness/exhaustive_timed_out"]
            eq_pru = f["correctness/exhaustive_eq_pruned"]
            gs_eq = f["correctness/greedy_single_eq_exhaustive"]
            gm_eq = f["correctness/greedy_min_eq_exhaustive"]

            for i in range(n):
                entry = CacheEntry(
                    raw_string=raw[i].decode() if isinstance(raw[i], bytes) else str(raw[i]),
                    num_variables=nv,
                    n_nodes=int(nn[i]),
                    n_internal=int(ni[i]),
                    n_edges=int(ne[i]),
                    n_var_nodes=int(nvn[i]),
                    depth=int(dep[i]),
                    greedy_single=gs[i].decode() if isinstance(gs[i], bytes) else str(gs[i]),
                    greedy_min=gm[i].decode() if isinstance(gm[i], bytes) else str(gm[i]),
                    pruned=pru[i].decode() if isinstance(pru[i], bytes) else str(pru[i]),
                    exhaustive=exh[i].decode() if isinstance(exh[i], bytes) else str(exh[i]),
                    exhaustive_timed_out=bool(exh_to[i]),
                    timing_greedy_single=float(t_gs[i]),
                    timing_greedy_min=float(t_gm[i]),
                    timing_pruned=float(t_pru[i]),
                    timing_exhaustive=float(t_exh[i]),
                    is_canonical=bool(is_can[i]),
                    exhaustive_eq_pruned=bool(eq_pru[i]),
                    greedy_single_eq_exhaustive=bool(gs_eq[i]),
                    greedy_min_eq_exhaustive=bool(gm_eq[i]),
                )
                self._entries.append(entry)

        log.info("Loaded %d entries from %s", n, path)

    def write_metadata_json(self, path: Path | str) -> None:
        """Write a JSON sidecar with human-readable cache metadata."""
        path = Path(path)
        stats = self.stats

        metadata = {
            "num_variables": self._num_variables,
            "operator_set": sorted(
                op.value for op in self._operator_set.ops if op not in LEAF_TYPES
            ),
            "total_entries": stats.total_entries,
            "unique_canonical_pruned": stats.unique_canonical_pruned,
            "unique_canonical_exhaustive": stats.unique_canonical_exhaustive,
            "exhaustive_timeout_count": stats.exhaustive_timeout_count,
            "avg_depth": round(stats.avg_depth, 2),
            "max_depth": stats.max_depth,
            "avg_internal_nodes": round(stats.avg_internal_nodes, 2),
            "correctness_summary": {
                "exhaustive_eq_pruned_rate": (
                    round(stats.exhaustive_eq_pruned_count / max(stats.total_entries, 1), 4)
                ),
                "greedy_single_eq_exhaustive_rate": (
                    round(
                        stats.greedy_single_eq_exhaustive_count / max(stats.total_entries, 1),
                        4,
                    )
                ),
                "greedy_min_eq_exhaustive_rate": (
                    round(
                        stats.greedy_min_eq_exhaustive_count / max(stats.total_entries, 1),
                        4,
                    )
                ),
            },
            "creation_timestamp": datetime.now(tz=UTC).isoformat(),
            "isalsr_version": "0.1.0",
            "git_hash": _git_hash(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info("Wrote metadata to %s", path)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def ops_hash(operator_set: OperationSet) -> str:
        """Compute an 8-character hex hash of the operator set.

        Used in cache filenames to distinguish configurations.
        """
        ops_json = json.dumps(sorted(op.value for op in operator_set.ops if op not in LEAF_TYPES))
        return hashlib.md5(ops_json.encode()).hexdigest()[:8]  # noqa: S324

    @staticmethod
    def cache_filename(benchmark: str, num_variables: int, operator_set: OperationSet) -> str:
        """Generate the canonical cache filename.

        Returns:
            e.g. 'cache_nguyen_1vars_a3f8b2c1.h5'
        """
        h = CacheManager.ops_hash(operator_set)
        return f"cache_{benchmark}_{num_variables}vars_{h}.h5"


# ======================================================================
# Helpers
# ======================================================================


def _git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        return (
            subprocess.check_output(  # noqa: S603, S607
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
