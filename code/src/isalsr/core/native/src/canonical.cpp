/// canonical.cpp — C++ port of _fast_canonical_d2s / _fast_step (wl_only mode).
///
/// Mirrors Python's fast_canonical_string(dag, mode="wl_only") in canonical.py
/// exactly, including:
///   - Pairs sorted by (|a|+|b|, |a|, a, b) — Invariant 5.
///   - Candidate key = (label_char, wl_hash) — wl_only mode (Invariant 10).
///   - All tied candidates explored; lexmin taken (Trap 1: no ID tiebreak).
///   - Binary op admissibility: input_order_[c][0] == ptr_in (Inv. 8 / B9).
///   - Const normalization at entry (Invariant 9).
///   - Timeout checked on every recursive call.
///
/// State management:
///   i2o[input_node]  = output_node, -1 if uninserted.
///   o2i[output_node] = input_node, set when added, stale-but-safe after undo.
///   Only i2o[c] is reset to -1 on backtrack; o2i[new_out] is overwritten
///   when a new node occupies that slot index on the next candidate attempt.

#include <isalsr/canonical.hpp>
#include <isalsr/cdll.hpp>
#include <isalsr/labeled_dag.hpp>
#include <isalsr/wl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace isalsr {

// ---------------------------------------------------------------------------
// Pair generation — mirrors generate_pairs_sorted_by_sum (dag_to_string.py)
//
// Sort key: (|a|+|b|, |a|, a, b)  — matches Python's
//   key=lambda p: (abs(p[0])+abs(p[1]), abs(p[0]), p)
// where `p` is a 2-tuple so the last term is (a, b) lexicographic.
// ---------------------------------------------------------------------------

static const std::vector<std::pair<int32_t, int32_t>>& get_pairs(int32_t m) {
    static std::unordered_map<int32_t, std::vector<std::pair<int32_t, int32_t>>> cache;
    auto it = cache.find(m);
    if (it != cache.end()) return it->second;

    auto& pairs = cache[m];
    const auto side = static_cast<std::size_t>(2 * m + 1);
    pairs.reserve(side * side);
    for (int32_t a = -m; a <= m; ++a)
        for (int32_t b = -m; b <= m; ++b)
            pairs.push_back({a, b});

    std::stable_sort(pairs.begin(), pairs.end(),
        [](const std::pair<int32_t, int32_t>& x,
           const std::pair<int32_t, int32_t>& y) {
            const int32_t cx = std::abs(x.first) + std::abs(x.second);
            const int32_t cy = std::abs(y.first) + std::abs(y.second);
            if (cx != cy) return cx < cy;
            const int32_t ax = std::abs(x.first), ay = std::abs(y.first);
            if (ax != ay) return ax < ay;
            if (x.first  != y.first)  return x.first  < y.first;
            return x.second < y.second;
        });
    return pairs;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int32_t cdll_walk(
    const NativeCDLL& cdll, int32_t ptr, int32_t steps) noexcept
{
    if (steps > 0)
        for (int32_t i = 0; i < steps; ++i) ptr = cdll.next_node(ptr);
    else
        for (int32_t i = 0; i < -steps; ++i) ptr = cdll.prev_node(ptr);
    return ptr;
}

static std::string primary_moves(int32_t a) {
    return a >= 0 ? std::string(static_cast<std::size_t>(a), 'N')
                  : std::string(static_cast<std::size_t>(-a), 'P');
}

static std::string secondary_moves(int32_t b) {
    return b >= 0 ? std::string(static_cast<std::size_t>(b), 'n')
                  : std::string(static_cast<std::size_t>(-b), 'p');
}

static bool is_binary_op(NodeType t) noexcept {
    return t == NodeType::SUB || t == NodeType::DIV || t == NodeType::POW;
}

// Invariant candidate key (wl_only): (label_char, wl_hash).
// label_char is the FIRST component — label-aware sorting (Invariant 10).
struct CandKey {
    char     label_char;
    uint64_t wl_hash;
    bool operator<(const CandKey& o) const noexcept {
        if (label_char != o.label_char) return label_char < o.label_char;
        return wl_hash < o.wl_hash;
    }
    bool operator==(const CandKey& o) const noexcept {
        return label_char == o.label_char && wl_hash == o.wl_hash;
    }
};

static CandKey make_key(
    int32_t node,
    const NativeLabeledDAG& ig,
    const std::vector<uint64_t>& wl)
{
    return { node_type_label_char(ig.labels_[static_cast<std::size_t>(node)]),
             wl[static_cast<std::size_t>(node)] };
}

// Python: best is None or (len(r), r) < (len(best), best)
static bool is_better(const std::string& a, const std::string& b) noexcept {
    if (a.size() != b.size()) return a.size() < b.size();
    return a < b;
}

// ---------------------------------------------------------------------------
// Recursive fast_step (wl_only)
//
// Mirrors Python's _fast_step with tuples=None, subtree_hashes=wl.
// Arguments match canonical.py _fast_step signature exactly.
// ---------------------------------------------------------------------------

static std::string fast_step(
    const NativeLabeledDAG& ig,
    NativeLabeledDAG&        og,
    NativeCDLL&              cdll,
    int32_t                  pri,   // CDLL slot index (not graph node ID)
    int32_t                  sec,   // CDLL slot index
    std::vector<int32_t>&    i2o,   // i2o[input_node]  = output_node, -1 if absent
    std::vector<int32_t>&    o2i,   // o2i[output_node] = input_node
    int32_t                  nleft,
    int32_t                  eleft,
    std::string              prefix, // passed by value; recursive calls extend it
    const std::vector<uint64_t>& wl,
    const std::chrono::steady_clock::time_point* deadline
) {
    if (nleft <= 0 && eleft <= 0) return prefix;

    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
        throw CanonicalTimeoutError(
            "Fast canonical string computation exceeded time budget");
    }

    const auto& pairs = get_pairs(og.node_count_);

    for (const auto& ab : pairs) {
        const int32_t a = ab.first;
        const int32_t b = ab.second;

        // ---- tentative primary ----
        const int32_t tp     = cdll_walk(cdll, pri, a);
        const int32_t tp_out = cdll.get_value(tp);
        const int32_t tp_in  = o2i[static_cast<std::size_t>(tp_out)];

        // -- V: primary has uninserted outgoing neighbor --
        if (nleft > 0) {
            std::vector<int32_t> cands;
            for (int32_t c : ig.out_adj_[static_cast<std::size_t>(tp_in)]) {
                if (i2o[static_cast<std::size_t>(c)] != -1) continue; // already inserted
                // Binary op admissibility (Invariant 8 / B9):
                // c is admissible if NOT a binary op, OR input_order_ is empty,
                // OR the first operand is tp_in.
                if (is_binary_op(ig.labels_[static_cast<std::size_t>(c)])) {
                    const auto& oi = ig.input_order_[static_cast<std::size_t>(c)];
                    if (!oi.empty() && oi[0] != tp_in) continue;
                }
                cands.push_back(c);
            }
            if (!cands.empty()) {
                // Sort candidates by (label_char, wl_hash) — wl_only key
                std::sort(cands.begin(), cands.end(), [&](int32_t x, int32_t y) {
                    return make_key(x, ig, wl) < make_key(y, ig, wl);
                });
                const CandKey best_key = make_key(cands[0], ig, wl);
                const std::string mov = primary_moves(a);
                std::string best;
                bool has_best = false;

                // Explore all tied candidates (lexmin backtracking, Trap 1)
                for (int32_t c : cands) {
                    if (!(make_key(c, ig, wl) == best_key)) break; // past tied group

                    const auto  cidx  = static_cast<std::size_t>(c);
                    const auto  label = ig.labels_[cidx];
                    const char  lchar = node_type_label_char(label);
                    const int32_t vidx = ig.var_index_[cidx];
                    const double  cval = ig.const_value_[cidx];

                    // Forward: add node and edge (acyclic by construction)
                    const int32_t new_out  = og.add_node(label, vidx, cval);
                    i2o[cidx]                               = new_out;
                    o2i[static_cast<std::size_t>(new_out)]  = c;
                    og.add_edge_unchecked(tp_out, new_out);
                    const int32_t new_cdll = cdll.insert_after(tp, new_out);

                    std::string r = fast_step(
                        ig, og, cdll, tp, sec, i2o, o2i,
                        nleft - 1, eleft - 1,
                        prefix + mov + "V" + lchar,
                        wl, deadline);

                    if (!has_best || is_better(r, best)) {
                        best     = std::move(r);
                        has_best = true;
                    }

                    // Backward: undo in reverse order
                    cdll.remove(new_cdll);
                    og.remove_edge(tp_out, new_out);
                    og.undo_node();
                    i2o[cidx] = -1; // o2i[new_out] will be overwritten on next attempt
                }
                return best;
            }
        }

        // ---- tentative secondary ----
        const int32_t ts     = cdll_walk(cdll, sec, b);
        const int32_t ts_out = cdll.get_value(ts);
        const int32_t ts_in  = o2i[static_cast<std::size_t>(ts_out)];

        // -- v: secondary has uninserted outgoing neighbor --
        if (nleft > 0) {
            std::vector<int32_t> cands;
            for (int32_t c : ig.out_adj_[static_cast<std::size_t>(ts_in)]) {
                if (i2o[static_cast<std::size_t>(c)] != -1) continue;
                if (is_binary_op(ig.labels_[static_cast<std::size_t>(c)])) {
                    const auto& oi = ig.input_order_[static_cast<std::size_t>(c)];
                    if (!oi.empty() && oi[0] != ts_in) continue;
                }
                cands.push_back(c);
            }
            if (!cands.empty()) {
                std::sort(cands.begin(), cands.end(), [&](int32_t x, int32_t y) {
                    return make_key(x, ig, wl) < make_key(y, ig, wl);
                });
                const CandKey best_key = make_key(cands[0], ig, wl);
                const std::string mov = secondary_moves(b);
                std::string best;
                bool has_best = false;

                for (int32_t c : cands) {
                    if (!(make_key(c, ig, wl) == best_key)) break;

                    const auto  cidx  = static_cast<std::size_t>(c);
                    const auto  label = ig.labels_[cidx];
                    const char  lchar = node_type_label_char(label);
                    const int32_t vidx = ig.var_index_[cidx];
                    const double  cval = ig.const_value_[cidx];

                    const int32_t new_out  = og.add_node(label, vidx, cval);
                    i2o[cidx]                               = new_out;
                    o2i[static_cast<std::size_t>(new_out)]  = c;
                    og.add_edge_unchecked(ts_out, new_out);
                    const int32_t new_cdll = cdll.insert_after(ts, new_out);

                    std::string r = fast_step(
                        ig, og, cdll, pri, ts, i2o, o2i,
                        nleft - 1, eleft - 1,
                        prefix + mov + "v" + lchar,
                        wl, deadline);

                    if (!has_best || is_better(r, best)) {
                        best     = std::move(r);
                        has_best = true;
                    }

                    cdll.remove(new_cdll);
                    og.remove_edge(ts_out, new_out);
                    og.undo_node();
                    i2o[cidx] = -1;
                }
                return best;
            }
        }

        // -- C: edge primary→secondary in input, not yet in output --
        if (ig.has_edge_unchecked(tp_in, ts_in) &&
            !og.has_edge_unchecked(tp_out, ts_out)) {
            og.add_edge(tp_out, ts_out);
            std::string r = fast_step(
                ig, og, cdll, tp, ts, i2o, o2i, nleft, eleft - 1,
                prefix + primary_moves(a) + secondary_moves(b) + "C",
                wl, deadline);
            og.remove_edge(tp_out, ts_out);
            return r;
        }

        // -- c: edge secondary→primary in input, not yet in output --
        if (ig.has_edge_unchecked(ts_in, tp_in) &&
            !og.has_edge_unchecked(ts_out, tp_out)) {
            og.add_edge(ts_out, tp_out);
            std::string r = fast_step(
                ig, og, cdll, tp, ts, i2o, o2i, nleft, eleft - 1,
                prefix + primary_moves(a) + secondary_moves(b) + "c",
                wl, deadline);
            og.remove_edge(ts_out, tp_out);
            return r;
        }
    }

    throw std::runtime_error(
        "Fast canonical D2S: no valid operation found. Remaining: " +
        std::to_string(nleft) + " nodes, " + std::to_string(eleft) + " edges.");
}

// ---------------------------------------------------------------------------
// fast_canonical_string_wl_only — assumes const normalization already applied
// ---------------------------------------------------------------------------

std::string fast_canonical_string_wl_only(
    const NativeLabeledDAG& dag,
    double                  timeout_sec
) {
    const int32_t n = dag.node_count_;
    NativeLabeledDAG og(n);
    NativeCDLL       cdll(n);

    // Precompute WL subtree hashes on the input DAG
    const std::vector<uint64_t> wl = compute_subtree_hashes(dag);

    // Timeout deadline
    std::chrono::steady_clock::time_point  deadline_tp;
    const std::chrono::steady_clock::time_point* deadline_ptr = nullptr;
    if (timeout_sec >= 0.0) {
        deadline_tp  = std::chrono::steady_clock::now()
            + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(timeout_sec));
        deadline_ptr = &deadline_tp;
    }

    // Initialize: map all VAR nodes in var_index order.
    // Matches Python: sorted(var_nodes, key=lambda v: data.get("var_index", v))
    // In C++: var_index_ == -1 means absent; fall back to node ID.
    std::vector<int32_t> varnodes = dag.var_nodes();
    std::stable_sort(varnodes.begin(), varnodes.end(),
        [&dag](int32_t a, int32_t b) {
            int32_t ai = dag.var_index_[static_cast<std::size_t>(a)];
            int32_t bi = dag.var_index_[static_cast<std::size_t>(b)];
            if (ai < 0) ai = a;  // Python: default to node ID if absent
            if (bi < 0) bi = b;
            return ai < bi;
        });

    // i2o[input_node] = output_node, -1 if not yet inserted.
    // o2i[output_node] = input_node.
    std::vector<int32_t> i2o(static_cast<std::size_t>(n), -1);
    std::vector<int32_t> o2i(static_cast<std::size_t>(n), -1);

    int32_t prev_cdll  = -1;
    int32_t first_cdll = -1;
    for (int32_t inp_node : varnodes) {
        const auto inidx = static_cast<std::size_t>(inp_node);
        // Python: og.add_node(NodeType.VAR, var_index=data.get("var_index", 0))
        int32_t vidx_out = dag.var_index_[inidx];
        if (vidx_out < 0) vidx_out = 0; // Python default: 0 (not node ID) for output

        const int32_t out_node = og.add_node(
            NodeType::VAR, vidx_out,
            std::numeric_limits<double>::quiet_NaN());
        i2o[inidx]                               = out_node;
        o2i[static_cast<std::size_t>(out_node)]  = inp_node;

        const int32_t cdll_node = cdll.insert_after(prev_cdll, out_node);
        if (first_cdll < 0) first_cdll = cdll_node;
        prev_cdll = cdll_node;
    }

    const int32_t num_vars = static_cast<int32_t>(varnodes.size());
    const int32_t nleft    = n - num_vars;
    const int32_t eleft    = dag.edge_count_;

    // Both pointers start at the first CDLL node (x_1)
    return fast_step(
        dag, og, cdll,
        first_cdll, first_cdll,
        i2o, o2i,
        nleft, eleft,
        std::string{},
        wl, deadline_ptr);
}

// ---------------------------------------------------------------------------
// fast_canonical_string — full entry point
// ---------------------------------------------------------------------------

std::string fast_canonical_string(
    const NativeLabeledDAG& dag,
    double                  timeout_sec
) {
    if (dag.node_count_ == 0) return {};

    // Only VAR nodes, no edges: empty canonical string
    const int32_t num_vars =
        static_cast<int32_t>(dag.var_nodes().size());
    if (dag.node_count_ == num_vars && dag.edge_count_ == 0) return {};

    // CONST normalisation is NOT applied here (removed 2026-07-29, T07).
    // It made the canonical string a function of normalize(D) rather than of D,
    // and normalize_const_creation is not isomorphism-equivariant (it anchors
    // orphan CONST nodes in node-index order, which is exactly what isomorphism
    // permutes), so two isomorphic DAGs could receive different strings.
    // Establishing the reachability precondition is the producer's job -- the
    // host adapters do it -- and the canonicaliser now assumes it, raising on an
    // unreachable node rather than silently repairing.  Kept byte-equivalent
    // with the Python reference (T01).
    return fast_canonical_string_wl_only(dag, timeout_sec);
}

} // namespace isalsr
