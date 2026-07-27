/// canonical.hpp — Fast canonical string (wl_only mode) for IsalSR.
///
/// C++ port of fast_canonical_string(dag, mode="wl_only") from canonical.py.
/// ONLY wl_only mode is ported; wl_tiebreak and tuple_only remain in Python.
///
/// Invariants implemented:
///   5  — pairs sorted by |a|+|b|, then |a|, then (a,b) lex.
///   8  — binary op admissibility via ordered_inputs()[0] (B9).
///   9  — normalize_const_creation() applied at entry when needed.
///   10 — label_char is the FIRST sort key (label-aware pruning).
///
/// Reference: canonical.py, IsalSR (Lopez-Rubio / Pascual, 2025).

#pragma once

#include <isalsr/labeled_dag.hpp>

#include <stdexcept>
#include <string>

namespace isalsr {

/// Raised when canonical computation exceeds its time budget.
/// Translated to Python as isalsr.core._native.CanonicalTimeoutError.
class CanonicalTimeoutError : public std::runtime_error {
public:
    explicit CanonicalTimeoutError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/// Compute fast canonical string (wl_only) — full entry point.
///
/// Applies normalize_const_creation() when CONST nodes are present (Inv. 9),
/// then delegates to fast_canonical_string_wl_only.
///
/// @param dag         Input labeled DAG.
/// @param timeout_sec Wall-clock limit in seconds; negative = unlimited.
/// @throws CanonicalTimeoutError if the budget is exceeded.
std::string fast_canonical_string(
    const NativeLabeledDAG& dag,
    double timeout_sec
);

/// Entry point that assumes const-creation is already normalized.
/// Exposed as a separate symbol for unit tests; prefer fast_canonical_string.
std::string fast_canonical_string_wl_only(
    const NativeLabeledDAG& dag,
    double timeout_sec
);

} // namespace isalsr
