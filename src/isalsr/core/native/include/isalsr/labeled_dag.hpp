/// labeled_dag.hpp — C++ port of isalsr.core.labeled_dag.LabeledDAG.
///
/// Directed acyclic graph with labeled nodes and dual adjacency lists.
///
/// Invariants preserved:
///   1. CDLL slot indices ≠ graph node indices (enforced by callers; not our concern).
///   3. add_edge(source, target): edge source→target, data flows source→target.
///      input_order_[target] records insertion order (NOT a set; vector, ordered).
///   8. ordered_inputs(node) returns sources in insertion order for binary ops.
///   9. normalize_const_creation() moves all CONST creation edges to node 0,
///      iterating const_nodes in sorted order.
///
/// Reference: labeled_dag.py, IsalSR (Lopez-Rubio / Pascual, 2025).

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace isalsr {

/// NodeType enum — fixed integer codes that form the C++/Python FFI boundary.
/// MUST match NODE_TYPE_INT in test_native_datastructures.py exactly.
/// Do NOT reorder these values.
enum class NodeType : uint8_t {
    VAR   = 0,
    ADD   = 1,
    MUL   = 2,
    SUB   = 3,
    DIV   = 4,
    SIN   = 5,
    COS   = 6,
    EXP   = 7,
    LOG   = 8,
    SQRT  = 9,
    POW   = 10,
    ABS   = 11,
    NEG   = 12,
    INV   = 13,
    CONST = 14,
    NONE  = 255,  ///< Sentinel for unallocated slots (never returned to callers).
};

/// Directed acyclic graph with labeled nodes.
///
/// - out_adj_[u] and in_adj_[u] are std::set<int32_t> (sorted, no duplicates).
///   Sorting provides deterministic iteration order (avoids Python hash-order dependence).
/// - input_order_[t] is a std::vector<int32_t> tracking insertion order of
///   source nodes into target t.  This is NOT a set; order matters for
///   non-commutative binary ops (SUB, DIV, POW) — Invariant 8 (B9).
/// - Cycle detection: BFS from target before adding source→target.
/// - add_edge_unchecked skips BFS (use only when acyclicity is guaranteed by construction).
class NativeLabeledDAG {
public:
    int32_t node_count_;
    int32_t edge_count_;
    int32_t max_nodes_;

    std::vector<std::set<int32_t>> out_adj_;
    std::vector<std::set<int32_t>> in_adj_;
    std::vector<std::vector<int32_t>> input_order_;  ///< Insertion-ordered, NOT a set.
    std::vector<NodeType>  labels_;
    std::vector<int32_t>   var_index_;    ///< -1 = absent
    std::vector<double>    const_value_;  ///< NaN = absent

    explicit NativeLabeledDAG(int32_t max_nodes);

    int32_t node_count() const noexcept { return node_count_; }
    int32_t edge_count() const noexcept { return edge_count_; }
    int32_t max_nodes()  const noexcept { return max_nodes_; }

    /// Add a new node with the given label.
    /// var_index = -1 means absent (non-VAR nodes).
    /// const_value = NaN means absent (non-CONST nodes).
    /// Returns the new node ID (= node_count_ before increment).
    /// Raises std::runtime_error (→ Python RuntimeError) when full.
    int32_t add_node(NodeType label, int32_t var_index, double const_value);

    /// Add directed edge source→target.  Returns true if added.
    /// Returns false if: duplicate, self-loop, or would create a cycle.
    /// Raises std::out_of_range (→ Python IndexError) for invalid node IDs.
    bool add_edge(int32_t source, int32_t target);

    /// Add directed edge without cycle/duplicate check.
    /// Use ONLY when acyclicity is guaranteed (e.g. target is a freshly added node).
    /// Still tracks input_order_ for operand ordering (Invariant 8).
    void add_edge_unchecked(int32_t source, int32_t target);

    /// Remove directed edge source→target.
    /// Returns true if the edge existed and was removed, false otherwise.
    bool remove_edge(int32_t source, int32_t target);

    /// Remove the last-added node and all its incident edges.
    /// Used for backtracking in canonical search.  No-op on empty DAG.
    void undo_node();

    bool has_edge(int32_t source, int32_t target) const;
    bool has_edge_unchecked(int32_t source, int32_t target) const noexcept;

    /// Return sorted out-neighbors of `node` as a vector.
    std::vector<int32_t> out_neighbors(int32_t node) const;

    /// Return sorted in-neighbors of `node` as a vector.
    std::vector<int32_t> in_neighbors(int32_t node) const;

    /// Return sources of `node`'s in-edges in INSERTION ORDER (Invariant 8).
    std::vector<int32_t> ordered_inputs(int32_t node) const;

    NodeType node_label(int32_t node) const;
    int32_t  node_var_index(int32_t node) const;    ///< -1 if absent
    double   node_const_value(int32_t node) const;  ///< NaN if absent

    std::vector<int32_t> topological_sort() const;
    std::vector<int32_t> var_nodes() const;
    std::vector<int32_t> non_var_nodes() const;

    /// Check if adding source→target would create a cycle.
    bool has_cycle_if_added(int32_t source, int32_t target) const;

    /// Return a new DAG with all CONST creation edges moved to node 0 (x_1).
    /// Copies non-CONST edges from input_order_ (preserves operand order, Inv. 8).
    /// Adds node_0 → const_node edges in sorted(const_nodes) order (Invariant 9).
    NativeLabeledDAG normalize_const_creation() const;

    bool has_const_nodes() const;

private:
    void validate_node(int32_t node) const;
    bool has_path(int32_t start, int32_t end) const;
};

} // namespace isalsr
