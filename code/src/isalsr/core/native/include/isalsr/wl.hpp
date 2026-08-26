/// wl.hpp — WL subtree hash computation for IsalSR canonical strings.
///
/// Implements 1-Weisfeiler-Leman (WL) hash used in fast_canonical_string
/// (wl_only mode). The hash is isomorphism-invariant and byte-stable across
/// sessions — identical to Python's _wl_node_hash in canonical.py.
///
/// References:
///   Weisfeiler & Leman (1968).
///   FNV-1a 64-bit (Noll et al., 1994).

#pragma once

#include <isalsr/fnv.hpp>
#include <isalsr/labeled_dag.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace isalsr {

/// Return the NodeType "value" string used in WL hashing.
/// Matches Python NodeType.value: VAR->"var", ADD->"+",...
const std::string& node_type_value_string(NodeType t);

/// Return the label character for canonical string emission after V/v.
/// Matches Python's NODE_TYPE_TO_LABEL: ADD->'+', MUL->'*', etc.
/// Returns '\0' for NodeType::VAR (pre-inserted; never emitted).
char node_type_label_char(NodeType t) noexcept;

/// Compute FNV-1a 64-bit WL node hash.
///
/// Byte-stable: hashes label_value UTF-8 bytes, then mixes each child hash
/// in little-endian 8-byte order. ``sorted_children`` must already be
/// sorted numerically by the caller (identical to Python's contract).
/// Matches Python's _wl_node_hash in canonical.py exactly.
uint64_t wl_node_hash(
    const std::string& label_value,
    const std::vector<uint64_t>& sorted_children
) noexcept;

/// Compute WL subtree hashes for all nodes in ``dag``.
///
/// Bottom-up BFS (leaves first): for each node u,
///   hash(u) = wl_node_hash(label_value(u), sorted(hash(c) for c in out_neighbors(u)))
/// Matches Python's _compute_subtree_hashes in canonical.py.
std::vector<uint64_t> compute_subtree_hashes(const NativeLabeledDAG& dag);

} // namespace isalsr
