/// string_to_dag.hpp — C++ port of isalsr.core.string_to_dag.StringToDAG.
///
/// Tokenizes an IsalSR instruction string and executes the tokens to build a
/// NativeLabeledDAG.  This is the core of the S2D (String-to-DAG) conversion.
///
/// Invariants preserved (CLAUDE.md Critical Invariants):
///   1. CDLL slot indices are NOT graph node indices.  Pointers are CDLL slots;
///      cdll.get_value(ptr) converts a slot to a graph node ID.
///   3. add_edge(source, target): edge means source→target.  input_order tracked.
///   4. Pointer immobility on V/v: the active pointer does NOT advance after
///      a V/v insertion.
///   6. Cycle check on C/c: add_edge returns false on cycle — silent no-op.
///   7. Variables are pre-inserted: m VAR nodes exist before any token executes.
///   8. Operand order for binary ops: V/v creates first edge, C/c second.
///      ordered_inputs() preserves insertion order.
///
/// Reference: string_to_dag.py (IsalSR, Lopez-Rubio / Pascual, 2025).

#pragma once

#include <isalsr/cdll.hpp>
#include <isalsr/labeled_dag.hpp>
#include <isalsr/node_types.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace isalsr {

/// Tokenize an IsalSR instruction string into a list of tokens.
///
/// Token rules (matching _tokenize() in string_to_dag.py exactly):
///   - If current char is 'V' or 'v', consume the next char as the label.
///     The token is the two chars together (e.g., "V+", "vs").
///   - If 'V'/'v' is at end of string with no following char, throw.
///   - Otherwise, the single char must be one of: N P n p C c W.
///   - Unknown characters throw.
///
/// Args:
///   input_string: Raw IsalSR instruction string.
///
/// Returns:
///   Vector of tokens (single-char or two-char strings).
///
/// Throws:
///   std::invalid_argument: On invalid characters, truncated V/v, or unknown labels.
std::vector<std::string> tokenize(const std::string& input_string);

/// C++ port of StringToDAG (string_to_dag.py).
///
/// Converts an IsalSR instruction string to a NativeLabeledDAG by executing
/// the tokenized instructions against a NativeCDLL state machine.
///
/// Args:
///   input_string: The instruction string to execute.
///   num_variables: Number of input variables m.  Must be >= 1.
///
/// Throws:
///   std::invalid_argument: On invalid tokens or num_variables < 1.
class NativeStringToDAG {
public:
    NativeStringToDAG(const std::string& input_string, int32_t num_variables);

    /// Execute the string-to-DAG conversion.
    ///
    /// Initializes m VAR nodes with both pointers on x_1's CDLL slot,
    /// then executes each token in sequence.
    ///
    /// Returns:
    ///   The resulting NativeLabeledDAG (by value — independent copy).
    NativeLabeledDAG run() const;

    /// Return the tokenized instruction list (single- or two-char strings).
    std::vector<std::string> tokens() const { return tokens_; }

private:
    std::string              input_string_;
    int32_t                  num_variables_;
    std::vector<std::string> tokens_;

    /// Execute one token, mutating dag/cdll/primary_ptr/secondary_ptr.
    static void execute_token(
        const std::string& token,
        NativeLabeledDAG&  dag,
        NativeCDLL&        cdll,
        int32_t&           primary_ptr,
        int32_t&           secondary_ptr
    );

    /// Execute a V[label] or v[label] insertion token.
    static void execute_insertion(
        const std::string& token,
        NativeLabeledDAG&  dag,
        NativeCDLL&        cdll,
        int32_t&           primary_ptr,
        int32_t&           secondary_ptr
    );
};

} // namespace isalsr
