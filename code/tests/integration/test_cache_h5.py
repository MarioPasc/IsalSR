"""Integration tests: HDF5 round-trip for CacheManager.

Tests the full cycle: generate CacheEntry objects via compute_and_add,
flush to HDF5 via flush_hdf5, load back via load_hdf5, and verify that
all fields survive the round-trip without corruption or truncation.

Requires: h5py >= 3.8 (skipped if not installed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from isalsr.core.node_types import OperationSet
from isalsr.precomputed.cache_entry import CacheEntry
from isalsr.precomputed.cache_manager import CacheManager

h5py = pytest.importorskip("h5py")


# ======================================================================
# Helpers
# ======================================================================

# Strings that produce non-trivial single-output DAGs with 1 variable.
# Each creates at least one internal node so compute_entry returns non-None.
# Note: consecutive V instructions without movement (e.g. "V+V*") create
# multi-sink DAGs that output_node() rejects, so we use N to advance the
# pointer between insertions to form chains.
BASIC_STRINGS: list[str] = ["Vs", "Vc", "Ve", "VsNVe", "V+NV*"]


def _build_manager_with_entries(
    strings: list[str],
    num_variables: int = 1,
) -> CacheManager:
    """Create a CacheManager and compute entries for the given strings."""
    mgr = CacheManager(
        num_variables=num_variables,
        operator_set=OperationSet(),
        exhaustive_timeout=30.0,
    )
    for s in strings:
        added = mgr.compute_and_add(s)
        assert added, f"compute_and_add failed for string '{s}'"
    return mgr


def _assert_entries_equal(original: CacheEntry, loaded: CacheEntry) -> None:
    """Assert all fields of two CacheEntry objects match."""
    assert original.raw_string == loaded.raw_string
    assert original.num_variables == loaded.num_variables
    assert original.n_nodes == loaded.n_nodes
    assert original.n_internal == loaded.n_internal
    assert original.n_edges == loaded.n_edges
    assert original.n_var_nodes == loaded.n_var_nodes
    assert original.depth == loaded.depth
    assert original.greedy_single == loaded.greedy_single
    assert original.greedy_min == loaded.greedy_min
    assert original.pruned == loaded.pruned
    assert original.exhaustive == loaded.exhaustive
    assert original.exhaustive_timed_out == loaded.exhaustive_timed_out
    assert original.is_canonical == loaded.is_canonical
    assert original.exhaustive_eq_pruned == loaded.exhaustive_eq_pruned
    assert original.greedy_single_eq_exhaustive == loaded.greedy_single_eq_exhaustive
    assert original.greedy_min_eq_exhaustive == loaded.greedy_min_eq_exhaustive
    # Timings: cannot compare exactly (they are wall-clock), but they must
    # survive the round-trip as the same float values that were flushed.
    assert original.timing_greedy_single == pytest.approx(loaded.timing_greedy_single)
    assert original.timing_greedy_min == pytest.approx(loaded.timing_greedy_min)
    assert original.timing_pruned == pytest.approx(loaded.timing_pruned)
    assert original.timing_exhaustive == pytest.approx(loaded.timing_exhaustive)


# ======================================================================
# Tests
# ======================================================================


class TestFlushAndLoadRoundtrip:
    """Verify that flush_hdf5 -> load_hdf5 preserves all CacheEntry fields."""

    def test_flush_and_load_roundtrip(self, tmp_path: Path) -> None:
        """All fields survive the HDF5 round-trip for multiple strings."""
        mgr_orig = _build_manager_with_entries(BASIC_STRINGS)
        h5_path = tmp_path / "cache.h5"

        mgr_orig.flush_hdf5(h5_path)
        assert h5_path.exists(), "HDF5 file was not created"

        mgr_loaded = CacheManager(
            num_variables=1,
            operator_set=OperationSet(),
        )
        mgr_loaded.load_hdf5(h5_path)

        orig_entries = mgr_orig.entries
        loaded_entries = mgr_loaded.entries
        assert len(loaded_entries) == len(orig_entries)

        for orig, loaded in zip(orig_entries, loaded_entries, strict=True):
            _assert_entries_equal(orig, loaded)

        # Sanity: timing values are positive (computation did happen).
        for entry in orig_entries:
            assert entry.timing_greedy_single > 0
            assert entry.timing_greedy_min > 0
            assert entry.timing_pruned > 0
            # Exhaustive timing is > 0 unless timed out (then -1.0).
            if not entry.exhaustive_timed_out:
                assert entry.timing_exhaustive > 0

        # Structural sanity for the original entries.
        for entry in orig_entries:
            assert entry.n_nodes >= 2  # At least 1 var + 1 internal
            assert entry.n_internal >= 1
            assert entry.depth >= 1


class TestHDF5GroupStructure:
    """Verify the HDF5 file has the expected groups, datasets, and attrs."""

    def test_hdf5_group_structure(self, tmp_path: Path) -> None:
        """The HDF5 file must match the design doc schema."""
        mgr = _build_manager_with_entries(BASIC_STRINGS[:3])
        h5_path = tmp_path / "cache.h5"
        mgr.flush_hdf5(h5_path)

        with h5py.File(h5_path, "r") as f:
            # Top-level groups.
            expected_groups = {
                "strings",
                "dag_properties",
                "timings",
                "correctness",
                "canonical_index",
            }
            actual_groups = set(f.keys())
            assert expected_groups == actual_groups, (
                f"Missing groups: {expected_groups - actual_groups}, "
                f"extra groups: {actual_groups - expected_groups}"
            )

            # Datasets in strings/.
            strings_datasets = set(f["strings"].keys())
            expected_string_ds = {
                "raw",
                "greedy_single",
                "greedy_min",
                "pruned",
                "exhaustive",
                "is_canonical",
            }
            assert expected_string_ds == strings_datasets

            # Datasets in dag_properties/.
            dp_datasets = set(f["dag_properties"].keys())
            assert {"n_nodes", "n_internal", "n_edges", "n_var_nodes", "depth"} == dp_datasets

            # Datasets in timings/.
            timing_datasets = set(f["timings"].keys())
            assert {"greedy_single", "greedy_min", "pruned", "exhaustive"} == timing_datasets

            # Datasets in correctness/.
            corr_datasets = set(f["correctness"].keys())
            assert {
                "exhaustive_timed_out",
                "exhaustive_eq_pruned",
                "greedy_single_eq_exhaustive",
                "greedy_min_eq_exhaustive",
            } == corr_datasets

            # Datasets in canonical_index/.
            ci_datasets = set(f["canonical_index"].keys())
            assert {"unique_canonical_pruned", "canonical_to_first", "multiplicity"} == ci_datasets

            # Root attributes.
            expected_attrs = {
                "num_variables",
                "operator_set",
                "creation_timestamp",
                "git_hash",
                "total_entries",
                "isalsr_version",
            }
            actual_attrs = set(f.attrs.keys())
            assert expected_attrs.issubset(actual_attrs), (
                f"Missing root attrs: {expected_attrs - actual_attrs}"
            )

            # Verify attribute values.
            assert int(f.attrs["num_variables"]) == 1
            assert int(f.attrs["total_entries"]) == 3


class TestMetadataJSON:
    """Verify that write_metadata_json produces a valid, complete JSON sidecar."""

    def test_metadata_json_written(self, tmp_path: Path) -> None:
        """The JSON metadata file must contain required keys and valid values."""
        mgr = _build_manager_with_entries(BASIC_STRINGS[:3])
        h5_path = tmp_path / "cache.h5"
        json_path = tmp_path / "cache_metadata.json"

        mgr.flush_hdf5(h5_path)
        mgr.write_metadata_json(json_path)

        assert json_path.exists(), "Metadata JSON was not created"

        with open(json_path) as fj:
            meta = json.load(fj)

        # Required top-level keys.
        required_keys = {
            "num_variables",
            "total_entries",
            "unique_canonical_pruned",
            "correctness_summary",
        }
        assert required_keys.issubset(meta.keys()), (
            f"Missing JSON keys: {required_keys - meta.keys()}"
        )

        assert meta["num_variables"] == 1
        assert meta["total_entries"] == 3
        assert isinstance(meta["unique_canonical_pruned"], int)
        assert meta["unique_canonical_pruned"] >= 1

        # Correctness summary is a nested dict with rate keys.
        cs = meta["correctness_summary"]
        assert "exhaustive_eq_pruned_rate" in cs
        assert "greedy_single_eq_exhaustive_rate" in cs
        assert "greedy_min_eq_exhaustive_rate" in cs


class TestCanonicalIndexDeduplication:
    """Verify the canonical_index group correctly deduplicates pruned canonicals."""

    def test_canonical_index_deduplication(self, tmp_path: Path) -> None:
        """Duplicate raw strings sharing a pruned canonical are deduplicated."""
        # "Vs" twice will produce the same DAG and hence the same pruned canonical.
        strings_with_dups = ["Vs", "Vs", "Vc", "Ve"]
        mgr = _build_manager_with_entries(strings_with_dups)
        h5_path = tmp_path / "cache.h5"
        mgr.flush_hdf5(h5_path)

        with h5py.File(h5_path, "r") as f:
            n_raw = len(f["strings/raw"])
            assert n_raw == 4

            unique_pruned = f["canonical_index/unique_canonical_pruned"]
            multiplicity = f["canonical_index/multiplicity"]

            n_unique = len(unique_pruned)
            # Two "Vs" entries share the same pruned canonical, so
            # unique count must be strictly less than total count.
            assert n_unique < n_raw, (
                f"Expected fewer unique canonicals ({n_unique}) than raw entries ({n_raw})"
            )

            # Multiplicity must sum to total entries.
            mult_sum = sum(int(multiplicity[i]) for i in range(n_unique))
            assert mult_sum == n_raw, f"Multiplicity sum {mult_sum} != total entries {n_raw}"


class TestVlenStringsSurviveRoundtrip:
    """Verify variable-length strings are not truncated through HDF5."""

    def test_vlen_strings_survive_roundtrip(self, tmp_path: Path) -> None:
        """Strings of varying lengths survive flush + load without truncation."""
        # Use strings that produce different-length canonical outputs.
        varied_strings = ["Vs", "VsNVe", "V+NV*NVsNVc"]
        mgr_orig = _build_manager_with_entries(varied_strings)
        h5_path = tmp_path / "cache.h5"
        mgr_orig.flush_hdf5(h5_path)

        mgr_loaded = CacheManager(
            num_variables=1,
            operator_set=OperationSet(),
        )
        mgr_loaded.load_hdf5(h5_path)

        orig_entries = mgr_orig.entries
        loaded_entries = mgr_loaded.entries

        for orig, loaded in zip(orig_entries, loaded_entries, strict=True):
            # Every string field must match exactly (no truncation).
            assert orig.raw_string == loaded.raw_string, (
                f"raw_string mismatch: {orig.raw_string!r} vs {loaded.raw_string!r}"
            )
            assert orig.greedy_single == loaded.greedy_single, (
                f"greedy_single truncated: {orig.greedy_single!r} vs {loaded.greedy_single!r}"
            )
            assert orig.greedy_min == loaded.greedy_min
            assert orig.pruned == loaded.pruned
            assert orig.exhaustive == loaded.exhaustive

        # Verify strings actually have different lengths (test is meaningful).
        raw_lengths = [len(e.raw_string) for e in orig_entries]
        assert len(set(raw_lengths)) > 1, "All raw strings had the same length"


class TestEmptyManagerWarns:
    """Verify that flushing with no entries logs a warning and skips file creation."""

    def test_empty_manager_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """flush_hdf5 with zero entries should log a warning and not create the file."""
        mgr = CacheManager(
            num_variables=1,
            operator_set=OperationSet(),
        )
        h5_path = tmp_path / "cache.h5"

        with caplog.at_level(logging.WARNING):
            mgr.flush_hdf5(h5_path)

        assert not h5_path.exists(), "HDF5 file should not be created when there are no entries"
        assert any("No entries" in record.message for record in caplog.records), (
            "Expected a warning about no entries to flush"
        )
