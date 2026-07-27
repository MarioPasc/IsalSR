/// node_types.hpp — Token grammar constants and helpers for IsalSR.
///
/// Maps label characters (+, *, -, /, s, c, e, l, r, ^, a, g, i, k) to
/// NodeType enum values, and provides helpers for the tokenizer.
///
/// Disambiguation note:
///   'c' is BOTH a valid single-character instruction (edge secondary→primary)
///   AND a valid label character (COS, after V/v).  The tokenizer resolves
///   this purely by context — if it immediately follows 'V' or 'v', it is the
///   COS label; otherwise it is the edge instruction.
///
/// Reference: node_types.py, types.py (IsalSR, Lopez-Rubio / Pascual, 2025).

#pragma once

#include <isalsr/labeled_dag.hpp>

namespace isalsr {

/// Map a V/v label character to the corresponding NodeType.
///
/// Returns NodeType::NONE if `c` is not a valid label character.
/// Valid label chars: + * - / s c e l r ^ a g i k
inline NodeType label_char_to_node_type(char c) noexcept {
    switch (c) {
        case '+': return NodeType::ADD;
        case '*': return NodeType::MUL;
        case '-': return NodeType::SUB;
        case '/': return NodeType::DIV;
        case 's': return NodeType::SIN;
        case 'c': return NodeType::COS;
        case 'e': return NodeType::EXP;
        case 'l': return NodeType::LOG;
        case 'r': return NodeType::SQRT;
        case '^': return NodeType::POW;
        case 'a': return NodeType::ABS;
        case 'g': return NodeType::NEG;
        case 'i': return NodeType::INV;
        case 'k': return NodeType::CONST;
        default:  return NodeType::NONE;
    }
}

/// Return true if `c` is a valid V/v label character.
inline bool is_valid_label_char(char c) noexcept {
    return label_char_to_node_type(c) != NodeType::NONE;
}

/// Return true if `c` is a valid single-character instruction.
///
/// Single-character instructions: N P n p C c W
/// Note: 'c' here denotes the edge instruction (secondary→primary).
/// When 'c' follows 'V' or 'v', the tokenizer treats it as the COS label instead.
inline bool is_single_instruction(char c) noexcept {
    switch (c) {
        case 'N': case 'P': case 'n': case 'p':
        case 'C': case 'c': case 'W':
            return true;
        default:
            return false;
    }
}

} // namespace isalsr
