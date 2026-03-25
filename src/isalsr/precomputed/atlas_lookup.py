"""Atlas lookup for O(1) canonical string resolution.

Loads a precomputed HDF5 atlas and builds an in-memory hash map::

    hash(greedy_single_from_x0) → hash(pruned_canonical)

At runtime, computes ``GreedySingleD2S(dag, x_0)`` (~0.01 ms, O(k²))
instead of ``pruned_canonical_string`` (~0.76 ms+, O(k!-pruned)),
then performs a dict lookup.

Memory model: stores only ``int`` hashes (Python 64-bit), not full strings.
For 5 M entries the map occupies ~80 MB.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AtlasLookup:
    """O(1) canonical lookup from a precomputed HDF5 atlas.

    Multiple ``greedy_single`` strings may map to the same canonical
    hash (isomorphic DAGs discovered via different random strings).
    This is correct: the dict maps many keys to one value.

    Hash collision risk: < 3×10⁻⁶ for 10 M entries (birthday bound
    n²/2⁶⁵), consistent with the existing dedup ``set[int]`` design.
    """

    __slots__ = (
        "_gs_to_canon",
        "_n_entries",
        "_n_unique_canon",
        "_num_variables",
        "_operator_set_json",
        "_load_time_s",
    )

    def __init__(self) -> None:
        self._gs_to_canon: dict[int, int] = {}
        self._n_entries: int = 0
        self._n_unique_canon: int = 0
        self._num_variables: int = 0
        self._operator_set_json: str = ""
        self._load_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_hdf5(cls, path: Path | str) -> AtlasLookup:
        """Load atlas from a merged HDF5 file.

        Reads ``strings/greedy_single`` and ``strings/pruned`` in bulk,
        computes ``hash()`` for each pair, and builds the lookup dict.

        Args:
            path: Path to the merged ``cache_*.h5`` file.

        Returns:
            A ready-to-query :class:`AtlasLookup`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If required HDF5 datasets are missing.
        """
        import h5py  # noqa: PLC0415

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        t0 = time.perf_counter()
        obj = cls()

        with h5py.File(path, "r") as f:
            obj._num_variables = int(f.attrs.get("num_variables", 0))
            obj._operator_set_json = str(f.attrs.get("operator_set", ""))
            gs_ds = f["strings"]["greedy_single"]
            pr_ds = f["strings"]["pruned"]
            n = len(gs_ds)

            # Read all strings in bulk (vlen-str datasets)
            gs_raw: Any = gs_ds[:]
            pr_raw: Any = pr_ds[:]

        # Build hash map
        gs_to_canon: dict[int, int] = {}
        canon_hashes: set[int] = set()

        for i in range(n):
            gs_str = gs_raw[i]
            pr_str = pr_raw[i]
            # h5py may return bytes; decode if needed
            if isinstance(gs_str, bytes):
                gs_str = gs_str.decode("utf-8")
            if isinstance(pr_str, bytes):
                pr_str = pr_str.decode("utf-8")
            gs_h = hash(gs_str)
            pr_h = hash(pr_str)
            gs_to_canon[gs_h] = pr_h
            canon_hashes.add(pr_h)

        obj._gs_to_canon = gs_to_canon
        obj._n_entries = len(gs_to_canon)
        obj._n_unique_canon = len(canon_hashes)
        obj._load_time_s = time.perf_counter() - t0

        log.info(
            "Atlas loaded: %d greedy_single keys → %d canonical classes (%.1f s, %d vars, %s)",
            obj._n_entries,
            obj._n_unique_canon,
            obj._load_time_s,
            obj._num_variables,
            path.name,
        )
        return obj

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_dag(self, dag: object) -> tuple[int | None, bool]:
        """Look up a DAG's canonical hash via greedy-single D2S from x_0.

        Computes ``GreedySingleD2S().encode(dag)`` (O(k²), ~0.01 ms),
        hashes the result, and performs a dict lookup.

        Args:
            dag: A :class:`~isalsr.core.labeled_dag.LabeledDAG`.

        Returns:
            ``(canonical_hash, True)`` on hit, ``(None, False)`` on miss.
        """
        from isalsr.core.algorithms.greedy_single import GreedySingleD2S  # noqa: PLC0415

        gs_string = GreedySingleD2S().encode(dag)
        return self.lookup_greedy_hash(hash(gs_string))

    def lookup_greedy_hash(self, gs_hash: int) -> tuple[int | None, bool]:
        """Direct lookup by pre-computed greedy-single hash.

        Args:
            gs_hash: ``hash(greedy_single_string)``.

        Returns:
            ``(canonical_hash, True)`` on hit, ``(None, False)`` on miss.
        """
        canon_hash = self._gs_to_canon.get(gs_hash)
        if canon_hash is not None:
            return canon_hash, True
        return None, False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_entries(self) -> int:
        """Number of greedy-single keys in the lookup map."""
        return self._n_entries

    @property
    def n_unique_canonical(self) -> int:
        """Number of distinct canonical classes covered."""
        return self._n_unique_canon

    @property
    def num_variables(self) -> int:
        """Number of input variables the atlas was generated for."""
        return self._num_variables

    @property
    def operator_set_json(self) -> str:
        """JSON string of sorted operator labels (from HDF5 root attrs)."""
        return self._operator_set_json

    @property
    def load_time_s(self) -> float:
        """Wall-clock time to load the atlas from HDF5."""
        return self._load_time_s

    @property
    def memory_bytes(self) -> int:
        """Estimated memory of the hash map (dict overhead + entries)."""
        # CPython dict: ~72 bytes base + ~56 bytes per entry (key+value+hash)
        return 72 + self._n_entries * 56
