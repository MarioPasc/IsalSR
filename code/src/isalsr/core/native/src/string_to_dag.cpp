/// string_to_dag.cpp — NativeStringToDAG implementation.
///
/// Matches Python's StringToDAG (string_to_dag.py) token-for-token.
/// All Critical Invariants are documented at each decision point.

#include <isalsr/string_to_dag.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace isalsr {

// ============================================================================
// Tokenizer
// ============================================================================

std::vector<std::string> tokenize(const std::string& input_string) {
    std::vector<std::string> tokens;
    const std::size_t n = input_string.size();
    std::size_t i = 0;

    while (i < n) {
        const char ch = input_string[i];

        if (ch == 'V' || ch == 'v') {
            // Compound token: consume the next character as a label.
            // A trailing V/v with no following character is invalid.
            if (i + 1 >= n) {
                throw std::invalid_argument(
                    std::string("'") + ch +
                    "' at position " + std::to_string(i) +
                    " requires a label character, but the string ends"
                );
            }
            const char label = input_string[i + 1];
            if (!is_valid_label_char(label)) {
                throw std::invalid_argument(
                    std::string("Invalid label character '") + label +
                    "' at position " + std::to_string(i + 1) +
                    " after '" + ch + "'"
                );
            }
            // Two-character token, e.g. "V+" or "vc".
            tokens.push_back(std::string{ch, label});
            i += 2;

        } else if (is_single_instruction(ch)) {
            // Single-character instruction: N P n p C c W.
            tokens.push_back(std::string{ch});
            i += 1;

        } else {
            throw std::invalid_argument(
                std::string("Invalid character '") + ch +
                "' at position " + std::to_string(i)
            );
        }
    }

    return tokens;
}

// ============================================================================
// NativeStringToDAG
// ============================================================================

NativeStringToDAG::NativeStringToDAG(
    const std::string& input_string,
    int32_t            num_variables
)
    : input_string_(input_string)
    , num_variables_(num_variables)
    , tokens_(tokenize(input_string))
{
    if (num_variables < 1) {
        throw std::invalid_argument(
            "num_variables must be >= 1, got " + std::to_string(num_variables)
        );
    }
}

NativeLabeledDAG NativeStringToDAG::run() const {
    // Pre-compute capacity: m variables + count of V/v two-char insertion tokens.
    // Invariant 7: variables are pre-inserted, not created by V/v.
    int32_t insertion_count = 0;
    for (const auto& t : tokens_) {
        if (t.size() == 2) ++insertion_count;
    }
    const int32_t max_nodes = num_variables_ + insertion_count;

    NativeLabeledDAG dag(max_nodes);
    NativeCDLL       cdll(max_nodes);

    // ---- Initial state: m VAR nodes, both pointers on x_1's CDLL slot ----
    // Invariant 7: Insert m VAR nodes before executing any instruction.
    int32_t prev_cdll_node  = -1;   // -1 → CDLL is empty; insert_after ignores it
    int32_t first_cdll_node = -1;

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    for (int32_t i = 0; i < num_variables_; ++i) {
        // Graph node IDs are assigned sequentially by add_node.
        const int32_t graph_node = dag.add_node(NodeType::VAR, i, kNaN);
        const int32_t cdll_node  = cdll.insert_after(prev_cdll_node, graph_node);
        if (i == 0) first_cdll_node = cdll_node;
        prev_cdll_node = cdll_node;
    }

    // Both pointers start on x_1's CDLL node (Invariant 7).
    int32_t primary_ptr   = first_cdll_node;
    int32_t secondary_ptr = first_cdll_node;

    // ---- Execute each token ----
    for (const auto& token : tokens_) {
        execute_token(token, dag, cdll, primary_ptr, secondary_ptr);
    }

    return dag;   // returned by value; caller owns an independent copy
}

// ============================================================================
// Private helpers
// ============================================================================

void NativeStringToDAG::execute_token(
    const std::string& token,
    NativeLabeledDAG&  dag,
    NativeCDLL&        cdll,
    int32_t&           primary_ptr,
    int32_t&           secondary_ptr
) {
    if (token == "N") {
        primary_ptr = cdll.next_node(primary_ptr);

    } else if (token == "P") {
        primary_ptr = cdll.prev_node(primary_ptr);

    } else if (token == "n") {
        secondary_ptr = cdll.next_node(secondary_ptr);

    } else if (token == "p") {
        secondary_ptr = cdll.prev_node(secondary_ptr);

    } else if (token == "C") {
        // Edge: primary → secondary.  DAG cycle check via add_edge; silent no-op on cycle.
        // Invariant 6: if add_edge would create a cycle, it returns false and we ignore it.
        const int32_t src = cdll.get_value(primary_ptr);
        const int32_t tgt = cdll.get_value(secondary_ptr);
        dag.add_edge(src, tgt);

    } else if (token == "c") {
        // Edge: secondary → primary.  Same cycle-check semantics.
        const int32_t src = cdll.get_value(secondary_ptr);
        const int32_t tgt = cdll.get_value(primary_ptr);
        dag.add_edge(src, tgt);

    } else if (token == "W") {
        // No-op.

    } else if (token.size() == 2 && (token[0] == 'V' || token[0] == 'v')) {
        execute_insertion(token, dag, cdll, primary_ptr, secondary_ptr);

    } else {
        // Should not reach here if the tokenizer is correct.
        throw std::invalid_argument(
            std::string("Unknown token: ") + token
        );
    }
}

void NativeStringToDAG::execute_insertion(
    const std::string& token,
    NativeLabeledDAG&  dag,
    NativeCDLL&        cdll,
    int32_t&           primary_ptr,
    int32_t&           secondary_ptr
) {
    const char pointer_char = token[0];  // 'V' or 'v'
    const char label_char   = token[1];
    const NodeType node_type = label_char_to_node_type(label_char);

    static constexpr double kNaN      = std::numeric_limits<double>::quiet_NaN();
    static constexpr double kConstOne = 1.0;  // matches Python: const_value=1.0

    // Create the new node.
    // CONST nodes receive initial const_value=1.0 (Python: StringToDAG._execute_insertion).
    // All other non-VAR nodes receive no metadata (-1, NaN).
    const double  cv       = (node_type == NodeType::CONST) ? kConstOne : kNaN;
    const int32_t new_node = dag.add_node(node_type, -1, cv);

    if (pointer_char == 'V') {
        // Edge: primary's graph node → new node.  Invariant 3: V/v sets first edge.
        const int32_t primary_graph = cdll.get_value(primary_ptr);
        dag.add_edge(primary_graph, new_node);
        // Insert new node into CDLL after primary.
        // Invariant 4: primary_ptr does NOT advance after insertion.
        cdll.insert_after(primary_ptr, new_node);
    } else {
        // pointer_char == 'v': same semantics using secondary pointer.
        const int32_t secondary_graph = cdll.get_value(secondary_ptr);
        dag.add_edge(secondary_graph, new_node);
        // Invariant 4: secondary_ptr does NOT advance after insertion.
        cdll.insert_after(secondary_ptr, new_node);
    }
}

} // namespace isalsr
