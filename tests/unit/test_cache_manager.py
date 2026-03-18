"""Unit tests for CacheManager compute and stats logic.

Covers:
- compute_entry: valid strings (sin(x), x+y), trivial DAGs, invalid strings.
- ops_hash: determinism and distinctness for different OperationSets.
- cache_filename: format validation.
- add_entry / len: entry management.
- stats: aggregate statistics after adding entries.
- correctness_flags: exhaustive == pruned for simple DAGs.
"""

from __future__ import annotations

from isalsr.core.node_types import NodeType, OperationSet
from isalsr.precomputed.cache_manager import CacheManager

# ======================================================================
# Helpers
# ======================================================================


def _default_ops() -> OperationSet:
    """Return the default OperationSet (all ops allowed)."""
    return OperationSet()


def _manager(num_variables: int = 1) -> CacheManager:
    """Create a CacheManager with default ops and generous timeout."""
    return CacheManager(
        num_variables=num_variables,
        operator_set=_default_ops(),
        exhaustive_timeout=60.0,
    )


# ======================================================================
# compute_entry tests
# ======================================================================


class TestComputeEntry:
    """Tests for CacheManager.compute_entry."""

    def test_compute_entry_sin_x(self) -> None:
        """'Vs' with 1 variable: sin(x). Should produce a valid entry.

        Expected DAG: x (node 0) -> sin (node 1).
        Properties: n_nodes=2, n_internal=1, depth=1.
        All D2S outputs should be non-empty strings.
        All timings should be positive.
        """
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("Vs")

        assert entry is not None
        assert entry.raw_string == "Vs"
        assert entry.num_variables == 1
        assert entry.n_nodes == 2
        assert entry.n_internal == 1
        assert entry.n_var_nodes == 1
        assert entry.n_edges == 1
        assert entry.depth == 1

        # All D2S algorithms should produce non-empty output.
        assert len(entry.greedy_single) > 0
        assert len(entry.greedy_min) > 0
        assert len(entry.pruned) > 0
        assert len(entry.exhaustive) > 0

        # All timings should be non-negative (wall-clock seconds).
        assert entry.timing_greedy_single >= 0
        assert entry.timing_greedy_min >= 0
        assert entry.timing_pruned >= 0
        assert entry.timing_exhaustive >= 0

        # Should not have timed out for this trivial DAG.
        assert entry.exhaustive_timed_out is False

    def test_compute_entry_x_plus_y(self) -> None:
        """'V+nnNc' with 2 variables: x + y.

        String execution:
        - Initial: CDLL = [x(0), y(1)], primary=x, secondary=x
        - V+: create ADD node 2, edge x->add, insert after primary
        - nn: move secondary twice (x -> y -> add)
        - N: move primary next (x -> y)
        - c: edge secondary->primary = add->y ... wait, that is wrong direction.
          c = edge from secondary's graph node to primary's graph node.
          secondary is on add (node 2), primary is on y (node 1).
          So edge: node 2 -> node 1. But y is a VAR (leaf), so this is
          add -> y, which would make y a child of add? No: edge semantics
          is source provides input to target. So edge 2->1 means add provides
          input to y, which does not make sense for SR. But the DAG allows it.

        Actually, let us think about what we need: x + y means edges x->add, y->add.
        Better string: V+ creates edge x->add. Then we need edge y->add.
        With 2 vars, CDLL = [x, y], primary=x, secondary=x.
        V+: create add(2), edge 0->2, CDLL = [x, add, y]. primary=x, secondary=x.
        n: secondary moves next -> add. Then n: secondary -> y.
        N: primary moves next -> add. Then N: primary -> y.
        C: edge primary->secondary = y -> y (self-loop, no-op since same node).

        Let me use a simpler approach: "NV+" with 2 vars.
        CDLL=[x,y], primary=x, secondary=x.
        N: primary -> y.
        V+: create add(2), edge y->add. CDLL=[x, y, add].

        We need both x->add and y->add. Let me try:
        "V+NnC" with 2 vars:
        - V+: create add(2), edge x->add. CDLL=[x, add, y]. primary=x, secondary=x.
        - N: primary -> add.
        - n: secondary -> add.
        - C: edge primary->secondary = add->add (self-loop, no-op).

        Hmm. The trick is that C/c are the only way to add edges between
        existing nodes. Let me use: "V+nNC" with 2 vars:
        - V+: add(2), edge x(0)->add(2). CDLL=[x, add, y]. pri=x, sec=x.
        - n: sec -> add (CDLL next of x is add).
        - N: pri -> add.
        - C: edge pri->sec = add->add (self-loop).

        Still wrong. The edge we need is y->add. Let me try:
        "V+nnNC" with 2 vars:
        - V+: add(2), edge x(0)->add(2). CDLL=[x, add, y]. pri=x(cdll0), sec=x(cdll0).
        - n: sec -> add (cdll node for add).
        - n: sec -> y (cdll node for y).
        - N: pri -> add.
        - C: edge pri(add, graph=2)->sec(y, graph=1). Edge add->y? No, we want y->add.

        Use 'c' instead: c = edge sec->pri = y(graph 1) -> add(graph 2). Yes!
        So "V+nnNc" with 2 vars should work.
        """
        mgr = _manager(num_variables=2)
        entry = mgr.compute_entry("V+nnNc")

        assert entry is not None
        assert entry.raw_string == "V+nnNc"
        assert entry.num_variables == 2
        assert entry.n_nodes == 3
        assert entry.n_internal == 1
        assert entry.n_var_nodes == 2
        assert entry.n_edges == 2
        assert entry.depth == 1

        # All D2S algorithms should produce non-empty output.
        assert len(entry.greedy_single) > 0
        assert len(entry.greedy_min) > 0
        assert len(entry.pruned) > 0
        assert len(entry.exhaustive) > 0

        assert entry.exhaustive_timed_out is False

    def test_compute_entry_var_only_returns_none(self) -> None:
        """'NP' with 1 variable moves pointers but creates no new nodes.

        The resulting DAG has only the initial VAR node, so compute_entry
        returns None (trivial DAG, nothing to encode).
        """
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("NP")
        assert entry is None

    def test_compute_entry_invalid_string_returns_none(self) -> None:
        """'ZZZ' is not valid in the IsalSR alphabet: compute_entry returns None.

        The tokenizer raises InvalidTokenError, which is caught internally.
        """
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("ZZZ")
        assert entry is None

    def test_compute_entry_empty_string_returns_none(self) -> None:
        """An empty string with 1 variable produces a VAR-only DAG -> None."""
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("")
        assert entry is None


# ======================================================================
# ops_hash tests
# ======================================================================


class TestOpsHash:
    """Tests for CacheManager.ops_hash."""

    def test_ops_hash_deterministic(self) -> None:
        """Calling ops_hash twice with the same OperationSet gives the same hash."""
        ops = _default_ops()
        h1 = CacheManager.ops_hash(ops)
        h2 = CacheManager.ops_hash(ops)
        assert h1 == h2

    def test_ops_hash_is_8_hex_chars(self) -> None:
        """The hash should be an 8-character hex string."""
        ops = _default_ops()
        h = CacheManager.ops_hash(ops)
        assert len(h) == 8
        # All characters are valid hex digits.
        assert all(c in "0123456789abcdef" for c in h)

    def test_ops_hash_different_ops(self) -> None:
        """Different OperationSets produce different hashes."""
        ops_all = _default_ops()
        ops_small = OperationSet(frozenset({NodeType.ADD, NodeType.MUL}))
        h_all = CacheManager.ops_hash(ops_all)
        h_small = CacheManager.ops_hash(ops_small)
        assert h_all != h_small


# ======================================================================
# cache_filename tests
# ======================================================================


class TestCacheFilename:
    """Tests for CacheManager.cache_filename."""

    def test_cache_filename_format(self) -> None:
        """Filename contains benchmark name, num_vars, and ops hash."""
        ops = _default_ops()
        fname = CacheManager.cache_filename("nguyen", 1, ops)
        assert fname.startswith("cache_nguyen_")
        assert "1vars" in fname
        assert fname.endswith(".h5")
        # The hash portion should be 8 hex characters before ".h5".
        parts = fname.replace(".h5", "").split("_")
        hash_part = parts[-1]
        assert len(hash_part) == 8

    def test_cache_filename_different_benchmarks(self) -> None:
        """Different benchmark names produce different filenames."""
        ops = _default_ops()
        f1 = CacheManager.cache_filename("nguyen", 1, ops)
        f2 = CacheManager.cache_filename("keijzer", 1, ops)
        assert f1 != f2

    def test_cache_filename_different_num_vars(self) -> None:
        """Different num_variables produce different filenames."""
        ops = _default_ops()
        f1 = CacheManager.cache_filename("nguyen", 1, ops)
        f2 = CacheManager.cache_filename("nguyen", 2, ops)
        assert f1 != f2


# ======================================================================
# Entry management tests
# ======================================================================


class TestEntryManagement:
    """Tests for add_entry, compute_and_add, len, entries."""

    def test_add_and_len(self) -> None:
        """Adding entries increases len(manager)."""
        mgr = _manager(num_variables=1)
        assert len(mgr) == 0

        entry = mgr.compute_entry("Vs")
        assert entry is not None
        mgr.add_entry(entry)
        assert len(mgr) == 1

        # Add a second distinct entry.
        entry2 = mgr.compute_entry("Ve")
        assert entry2 is not None
        mgr.add_entry(entry2)
        assert len(mgr) == 2

    def test_compute_and_add(self) -> None:
        """compute_and_add returns True and adds when successful."""
        mgr = _manager(num_variables=1)
        assert mgr.compute_and_add("Vs") is True
        assert len(mgr) == 1
        # Invalid string returns False, length unchanged.
        assert mgr.compute_and_add("ZZZ") is False
        assert len(mgr) == 1

    def test_entries_returns_copy(self) -> None:
        """The entries property returns a copy, not the internal list."""
        mgr = _manager(num_variables=1)
        mgr.compute_and_add("Vs")
        entries = mgr.entries
        entries.clear()
        # Internal list should be unaffected.
        assert len(mgr) == 1


# ======================================================================
# Stats tests
# ======================================================================


class TestStats:
    """Tests for CacheManager.stats property."""

    def test_stats_empty(self) -> None:
        """Stats on an empty manager returns zeros."""
        mgr = _manager(num_variables=1)
        stats = mgr.stats
        assert stats.total_entries == 0
        assert stats.unique_canonical_pruned == 0
        assert stats.avg_depth == 0.0
        assert stats.max_depth == 0

    def test_stats_after_adding(self) -> None:
        """After adding 3 entries, stats.total_entries == 3."""
        mgr = _manager(num_variables=1)
        # sin(x), cos(x), exp(x): three distinct 1-variable DAGs.
        mgr.compute_and_add("Vs")
        mgr.compute_and_add("Vc")
        mgr.compute_and_add("Ve")
        stats = mgr.stats
        assert stats.total_entries == 3
        assert stats.max_depth >= 1
        assert stats.avg_depth >= 1.0
        assert stats.avg_internal_nodes >= 1.0

    def test_stats_unique_counts(self) -> None:
        """Adding the same string twice should still give correct unique counts."""
        mgr = _manager(num_variables=1)
        mgr.compute_and_add("Vs")
        mgr.compute_and_add("Vs")
        stats = mgr.stats
        assert stats.total_entries == 2
        # Both entries encode the same DAG, so unique canonical should be 1.
        assert stats.unique_canonical_pruned == 1


# ======================================================================
# Correctness flag tests
# ======================================================================


class TestCorrectnessFlags:
    """Tests for correctness flags on simple DAGs."""

    def test_exhaustive_equals_pruned_for_simple_dag(self) -> None:
        """For simple DAGs (sin(x)), exhaustive should equal pruned canonical.

        With only 1 internal node, the pruned search is already exhaustive,
        so both algorithms must produce the same canonical string.
        """
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("Vs")
        assert entry is not None
        assert entry.exhaustive_timed_out is False
        assert entry.exhaustive_eq_pruned is True
        # The exhaustive and pruned strings should literally be equal.
        assert entry.exhaustive == entry.pruned

    def test_greedy_agrees_on_trivial_case(self) -> None:
        """For a single unary op, all four algorithms should agree.

        sin(x) has only one possible D2S encoding, so greedy_single,
        greedy_min, pruned, and exhaustive must all be identical.
        """
        mgr = _manager(num_variables=1)
        entry = mgr.compute_entry("Vs")
        assert entry is not None
        assert entry.greedy_single_eq_exhaustive is True
        assert entry.greedy_min_eq_exhaustive is True
        assert entry.greedy_single == entry.exhaustive
        assert entry.greedy_min == entry.exhaustive
