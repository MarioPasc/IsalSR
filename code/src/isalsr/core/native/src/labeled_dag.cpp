/// labeled_dag.cpp — NativeLabeledDAG implementation.
///
/// Matches Python's LabeledDAG (labeled_dag.py) exactly, including:
///   - BFS-based cycle detection on add_edge
///   - insertion-order tracking in input_order_ (Invariant 8 / B9)
///   - normalize_const_creation iterating const_nodes in sorted order (Invariant 9)
///   - remove_edge removing the FIRST occurrence from input_order_ (matches list.remove())
///   - undo_node handling both outgoing and incoming edge sets (Python-exact loop order)

#include <isalsr/labeled_dag.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace isalsr {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NativeLabeledDAG::NativeLabeledDAG(int32_t max_nodes)
    : node_count_(0), edge_count_(0), max_nodes_(max_nodes),
      out_adj_(static_cast<std::size_t>(max_nodes)),
      in_adj_(static_cast<std::size_t>(max_nodes)),
      input_order_(static_cast<std::size_t>(max_nodes)),
      labels_(static_cast<std::size_t>(max_nodes), NodeType::NONE),
      var_index_(static_cast<std::size_t>(max_nodes), -1),
      const_value_(static_cast<std::size_t>(max_nodes),
                   std::numeric_limits<double>::quiet_NaN())
{}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void NativeLabeledDAG::validate_node(int32_t node) const {
    if (node < 0 || node >= node_count_) {
        throw std::out_of_range(
            "Invalid node ID: " + std::to_string(node) +
            " (node_count=" + std::to_string(node_count_) + ")");
    }
}

// ---------------------------------------------------------------------------
// Mutation — add_node
// ---------------------------------------------------------------------------

int32_t NativeLabeledDAG::add_node(NodeType label, int32_t var_index, double const_value) {
    if (node_count_ >= max_nodes_) {
        throw std::runtime_error(
            "Maximum number of nodes reached: " + std::to_string(max_nodes_));
    }
    auto idx = static_cast<std::size_t>(node_count_);
    labels_[idx]      = label;
    var_index_[idx]   = var_index;
    const_value_[idx] = const_value;
    return node_count_++;
}

// ---------------------------------------------------------------------------
// Cycle detection — BFS from `start` checking reachability of `end`
// ---------------------------------------------------------------------------

bool NativeLabeledDAG::has_path(int32_t start, int32_t end) const {
    if (start == end) return true;

    std::vector<bool> visited(static_cast<std::size_t>(node_count_), false);
    visited[static_cast<std::size_t>(start)] = true;
    std::queue<int32_t> q;
    q.push(start);

    while (!q.empty()) {
        int32_t current = q.front();
        q.pop();
        for (int32_t neighbor : out_adj_[static_cast<std::size_t>(current)]) {
            if (neighbor == end) return true;
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                q.push(neighbor);
            }
        }
    }
    return false;
}

bool NativeLabeledDAG::has_cycle_if_added(int32_t source, int32_t target) const {
    if (source == target) return true;
    return has_path(target, source);
}

// ---------------------------------------------------------------------------
// Mutation — add_edge (with cycle + duplicate check)
// ---------------------------------------------------------------------------

bool NativeLabeledDAG::add_edge(int32_t source, int32_t target) {
    validate_node(source);
    validate_node(target);

    // Duplicate check.
    if (out_adj_[static_cast<std::size_t>(source)].count(target)) return false;

    // Self-loop always creates a cycle in a DAG.
    if (source == target) return false;

    // Cycle check: would adding source→target create a path target→…→source?
    if (has_path(target, source)) return false;

    out_adj_[static_cast<std::size_t>(source)].insert(target);
    in_adj_[static_cast<std::size_t>(target)].insert(source);
    input_order_[static_cast<std::size_t>(target)].push_back(source);  // Invariant 8
    ++edge_count_;
    return true;
}

// ---------------------------------------------------------------------------
// Mutation — add_edge_unchecked (no BFS; target must be cycle-free by construction)
// ---------------------------------------------------------------------------

void NativeLabeledDAG::add_edge_unchecked(int32_t source, int32_t target) {
    // Still tracks input_order_ for operand ordering (Invariant 8 / B9).
    out_adj_[static_cast<std::size_t>(source)].insert(target);
    in_adj_[static_cast<std::size_t>(target)].insert(source);
    input_order_[static_cast<std::size_t>(target)].push_back(source);
    ++edge_count_;
}

// ---------------------------------------------------------------------------
// Mutation — remove_edge
// ---------------------------------------------------------------------------

bool NativeLabeledDAG::remove_edge(int32_t source, int32_t target) {
    validate_node(source);
    validate_node(target);

    if (!out_adj_[static_cast<std::size_t>(source)].count(target)) return false;

    out_adj_[static_cast<std::size_t>(source)].erase(target);
    in_adj_[static_cast<std::size_t>(target)].erase(source);

    // Remove the FIRST occurrence of `source` from input_order_[target],
    // matching Python's list.remove() semantics.
    auto& io = input_order_[static_cast<std::size_t>(target)];
    auto it  = std::find(io.begin(), io.end(), source);
    if (it != io.end()) {
        io.erase(it);
    }

    --edge_count_;
    return true;
}

// ---------------------------------------------------------------------------
// Mutation — undo_node
// ---------------------------------------------------------------------------

void NativeLabeledDAG::undo_node() {
    if (node_count_ == 0) return;

    int32_t node_id = node_count_ - 1;
    auto    nidx    = static_cast<std::size_t>(node_id);

    // Copy adjacency lists to vectors before clearing (mirrors Python's list(...))
    // so that modifications to neighbour sets don't invalidate our iterator.
    std::vector<int32_t> out_copy(out_adj_[nidx].begin(), out_adj_[nidx].end());
    for (int32_t target : out_copy) {
        auto tidx = static_cast<std::size_t>(target);
        in_adj_[tidx].erase(node_id);

        // Remove first occurrence of node_id from input_order_[target].
        auto& io = input_order_[tidx];
        auto  it = std::find(io.begin(), io.end(), node_id);
        if (it != io.end()) {
            io.erase(it);
        }
        --edge_count_;
    }
    out_adj_[nidx].clear();

    std::vector<int32_t> in_copy(in_adj_[nidx].begin(), in_adj_[nidx].end());
    for (int32_t source : in_copy) {
        out_adj_[static_cast<std::size_t>(source)].erase(node_id);
        --edge_count_;
    }
    in_adj_[nidx].clear();

    labels_[nidx]      = NodeType::NONE;
    var_index_[nidx]   = -1;
    const_value_[nidx] = std::numeric_limits<double>::quiet_NaN();
    input_order_[nidx].clear();
    --node_count_;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool NativeLabeledDAG::has_edge(int32_t source, int32_t target) const {
    validate_node(source);
    validate_node(target);
    return out_adj_[static_cast<std::size_t>(source)].count(target) > 0;
}

bool NativeLabeledDAG::has_edge_unchecked(int32_t source, int32_t target) const noexcept {
    return out_adj_[static_cast<std::size_t>(source)].count(target) > 0;
}

std::vector<int32_t> NativeLabeledDAG::out_neighbors(int32_t node) const {
    validate_node(node);
    const auto& s = out_adj_[static_cast<std::size_t>(node)];
    // std::set iterates in sorted order — deterministic (avoids hash-order risk).
    return std::vector<int32_t>(s.begin(), s.end());
}

std::vector<int32_t> NativeLabeledDAG::in_neighbors(int32_t node) const {
    validate_node(node);
    const auto& s = in_adj_[static_cast<std::size_t>(node)];
    return std::vector<int32_t>(s.begin(), s.end());
}

std::vector<int32_t> NativeLabeledDAG::ordered_inputs(int32_t node) const {
    validate_node(node);
    return input_order_[static_cast<std::size_t>(node)];  // copy
}

NodeType NativeLabeledDAG::node_label(int32_t node) const {
    validate_node(node);
    return labels_[static_cast<std::size_t>(node)];
}

int32_t NativeLabeledDAG::node_var_index(int32_t node) const {
    validate_node(node);
    return var_index_[static_cast<std::size_t>(node)];
}

double NativeLabeledDAG::node_const_value(int32_t node) const {
    validate_node(node);
    return const_value_[static_cast<std::size_t>(node)];
}

// ---------------------------------------------------------------------------
// Graph algorithms
// ---------------------------------------------------------------------------

std::vector<int32_t> NativeLabeledDAG::topological_sort() const {
    // Kahn's algorithm with FIFO queue — mirrors Python's deque-based BFS.
    std::vector<int32_t> in_deg(static_cast<std::size_t>(node_count_));
    for (int32_t i = 0; i < node_count_; ++i) {
        in_deg[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(in_adj_[static_cast<std::size_t>(i)].size());
    }

    std::queue<int32_t> q;
    for (int32_t i = 0; i < node_count_; ++i) {
        if (in_deg[static_cast<std::size_t>(i)] == 0) {
            q.push(i);
        }
    }

    std::vector<int32_t> result;
    result.reserve(static_cast<std::size_t>(node_count_));
    while (!q.empty()) {
        int32_t node = q.front();
        q.pop();
        result.push_back(node);
        for (int32_t neighbor : out_adj_[static_cast<std::size_t>(node)]) {
            auto nidx = static_cast<std::size_t>(neighbor);
            if (--in_deg[nidx] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (static_cast<int32_t>(result.size()) != node_count_) {
        throw std::runtime_error(
            "Graph contains a cycle: topological sort produced " +
            std::to_string(result.size()) + "/" + std::to_string(node_count_) + " nodes");
    }
    return result;
}

std::vector<int32_t> NativeLabeledDAG::var_nodes() const {
    std::vector<int32_t> result;
    for (int32_t i = 0; i < node_count_; ++i) {
        if (labels_[static_cast<std::size_t>(i)] == NodeType::VAR) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<int32_t> NativeLabeledDAG::non_var_nodes() const {
    std::vector<int32_t> result;
    for (int32_t i = 0; i < node_count_; ++i) {
        if (labels_[static_cast<std::size_t>(i)] != NodeType::VAR) {
            result.push_back(i);
        }
    }
    return result;
}

bool NativeLabeledDAG::has_const_nodes() const {
    for (int32_t i = 0; i < node_count_; ++i) {
        if (labels_[static_cast<std::size_t>(i)] == NodeType::CONST) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// normalize_const_creation (Invariant 9)
//
// Minimal repair: give every CONST that has NO in-edge a single creation edge
// from the lowest-indexed variable that does not close a cycle.  Every existing
// edge is preserved verbatim.  Mirrors labeled_dag.py exactly.
//
//   1. Copy all nodes (labels, var_index, const_value).
//   2. Re-add EVERY in-edge in input_order_ order, for every target
//      (preserves operand order — Invariant 8 / B9).
//   3. For each CONST c with in_degree(c) == 0, in ascending node order, add
//      x_i → c for the first variable x_i whose edge add_edge accepts.
//
// On any DAG satisfying the Round-Trip Fidelity reachability hypothesis every
// CONST already has an in-edge, so this is the identity — that is what keeps
// the fast canonical string a complete labeled-DAG invariant.  The former
// behaviour (relocate ALL CONST in-edges onto node 0) was non-injective and
// could silently drop the relocated edge when add_edge refused it as
// cycle-closing, orphaning the CONST and making D2S fail.
// See docs/md_files/changes/d2s_canonicalisation_failures.md.
// ---------------------------------------------------------------------------

NativeLabeledDAG NativeLabeledDAG::normalize_const_creation() const {
    NativeLabeledDAG new_dag(max_nodes_);

    // Step 1 — copy nodes, recording CONST IDs in ascending order.
    std::vector<int32_t> const_nodes;
    for (int32_t i = 0; i < node_count_; ++i) {
        auto idx = static_cast<std::size_t>(i);
        new_dag.add_node(labels_[idx], var_index_[idx], const_value_[idx]);
        if (labels_[idx] == NodeType::CONST) {
            const_nodes.push_back(i);
        }
    }

    // Step 2 — copy every edge, preserving operand order via input_order_.
    for (int32_t tgt = 0; tgt < node_count_; ++tgt) {
        for (int32_t src : input_order_[static_cast<std::size_t>(tgt)]) {
            // add_edge performs cycle/duplicate check; safe since source DAG is acyclic.
            new_dag.add_edge(src, tgt);
        }
    }

    // Step 3 — repair orphan CONST nodes only.  add_edge returns false when the
    // edge would close a cycle, so the loop settles on the lowest-indexed
    // variable that keeps the graph acyclic.  A CONST that reaches every
    // variable is left orphaned: the input genuinely violates the reachability
    // precondition and the ensuing D2S failure is correct behaviour.
    for (int32_t c : const_nodes) {
        if (new_dag.in_adj_[static_cast<std::size_t>(c)].empty()) {
            for (int32_t anchor = 0; anchor < new_dag.node_count_; ++anchor) {
                if (new_dag.labels_[static_cast<std::size_t>(anchor)] != NodeType::VAR) {
                    continue;
                }
                if (new_dag.add_edge(anchor, c)) {
                    break;
                }
            }
        }
    }

    return new_dag;
}

} // namespace isalsr
