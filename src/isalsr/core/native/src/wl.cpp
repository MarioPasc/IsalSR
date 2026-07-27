/// wl.cpp — WL hash implementation (isalsr.core._native).
///
/// Mirrors Python's _wl_node_hash and _compute_subtree_hashes in canonical.py.
/// All arithmetic is modulo 2^64 (uint64_t wrapping), matching Python's
/// _MASK64 = 0xFFFFFFFFFFFFFFFF masking. The hash is byte-stable and
/// session-reproducible — PYTHONHASHSEED has no effect.

#include <isalsr/wl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

namespace isalsr {

// ---------------------------------------------------------------------------
// node_type_value_string — matches Python NodeType.value strings exactly
// ---------------------------------------------------------------------------

const std::string& node_type_value_string(NodeType t) {
    // Indexed by NodeType integer value (0..14, matching labeled_dag.hpp enum).
    static const std::array<std::string, 15> table = {
        "var",  // VAR   = 0
        "+",    // ADD   = 1
        "*",    // MUL   = 2
        "-",    // SUB   = 3
        "/",    // DIV   = 4
        "s",    // SIN   = 5
        "c",    // COS   = 6
        "e",    // EXP   = 7
        "l",    // LOG   = 8
        "r",    // SQRT  = 9
        "^",    // POW   = 10
        "a",    // ABS   = 11
        "g",    // NEG   = 12
        "i",    // INV   = 13
        "k",    // CONST = 14
    };
    return table[static_cast<uint8_t>(t)];
}

// ---------------------------------------------------------------------------
// node_type_label_char — matches Python NODE_TYPE_TO_LABEL
// ---------------------------------------------------------------------------

char node_type_label_char(NodeType t) noexcept {
    // Same order as the enum.  VAR is never emitted in canonical strings.
    static const char table[] = {
        '\0', // VAR   = 0  (pre-inserted; not emitted)
        '+',  // ADD   = 1
        '*',  // MUL   = 2
        '-',  // SUB   = 3
        '/',  // DIV   = 4
        's',  // SIN   = 5
        'c',  // COS   = 6
        'e',  // EXP   = 7
        'l',  // LOG   = 8
        'r',  // SQRT  = 9
        '^',  // POW   = 10
        'a',  // ABS   = 11
        'g',  // NEG   = 12
        'i',  // INV   = 13
        'k',  // CONST = 14
    };
    return table[static_cast<uint8_t>(t)];
}

// ---------------------------------------------------------------------------
// wl_node_hash — matches Python _wl_node_hash exactly
//
// Python:
//   h = _FNV_OFFSET
//   for byte in label_value.encode("utf-8"):
//       h = ((h ^ byte) * _FNV_PRIME) & _MASK64
//   for ch in children_hashes:
//       for shift in range(0, 64, 8):
//           h = ((h ^ ((ch >> shift) & 0xFF)) * _FNV_PRIME) & _MASK64
//   return h
// ---------------------------------------------------------------------------

uint64_t wl_node_hash(
    const std::string& label_value,
    const std::vector<uint64_t>& sorted_children
) noexcept {
    uint64_t h = FNV1A64_OFFSET;
    // Hash each byte of the label value string (ASCII/UTF-8; all chars are ASCII)
    for (unsigned char byte : label_value) {
        h ^= static_cast<uint64_t>(byte);
        h *= FNV1A64_PRIME;
    }
    // Mix each child hash in little-endian 8-byte order
    for (uint64_t ch : sorted_children) {
        for (unsigned int shift = 0; shift < 64; shift += 8) {
            h ^= (ch >> shift) & UINT64_C(0xFF);
            h *= FNV1A64_PRIME;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// compute_subtree_hashes — matches Python _compute_subtree_hashes exactly
//
// Python:
//   out_deg[u] = len(dag.out_neighbors_raw(u))
//   queue starts with all u where out_deg[u] == 0 (leaves)
//   processed[] prevents double-processing
//   children_hashes = sorted(node_hash[v] for v in dag.out_neighbors_raw(u))
//   node_hash[u] = _wl_node_hash(label.value, tuple(children_hashes))
//   for v in dag.in_neighbors_raw(u): decrement out_deg[v]; enqueue if 0
// ---------------------------------------------------------------------------

std::vector<uint64_t> compute_subtree_hashes(const NativeLabeledDAG& dag) {
    const int32_t n = dag.node_count_;
    const auto    N = static_cast<std::size_t>(n);

    std::vector<uint64_t> node_hash(N, UINT64_C(0));
    std::vector<int32_t>  out_deg(N, 0);

    for (int32_t u = 0; u < n; ++u) {
        out_deg[static_cast<std::size_t>(u)] =
            static_cast<int32_t>(dag.out_adj_[static_cast<std::size_t>(u)].size());
    }

    // Enqueue leaves (out_deg == 0): BFS processes bottom-up
    std::queue<int32_t> q;
    for (int32_t u = 0; u < n; ++u) {
        if (out_deg[static_cast<std::size_t>(u)] == 0) q.push(u);
    }

    std::vector<bool> processed(N, false);
    while (!q.empty()) {
        const int32_t u    = q.front(); q.pop();
        const auto    uidx = static_cast<std::size_t>(u);
        if (processed[uidx]) continue;
        processed[uidx] = true;

        // Collect out-neighbor hashes and sort by value (not by node ID)
        // Python: sorted(node_hash[v] for v in dag.out_neighbors_raw(u))
        std::vector<uint64_t> child_hashes;
        child_hashes.reserve(dag.out_adj_[uidx].size());
        for (int32_t v : dag.out_adj_[uidx]) {
            child_hashes.push_back(node_hash[static_cast<std::size_t>(v)]);
        }
        std::sort(child_hashes.begin(), child_hashes.end());

        node_hash[uidx] = wl_node_hash(
            node_type_value_string(dag.labels_[uidx]),
            child_hashes);

        // Decrement out-degree of in-neighbors; enqueue when it hits 0
        for (int32_t v : dag.in_adj_[uidx]) {
            const auto vidx = static_cast<std::size_t>(v);
            if (--out_deg[vidx] == 0 && !processed[vidx]) q.push(v);
        }
    }

    return node_hash;
}

} // namespace isalsr
