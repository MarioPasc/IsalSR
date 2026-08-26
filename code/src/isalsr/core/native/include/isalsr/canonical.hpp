/// canonical.hpp — Fast canonical string (wl_only mode) for IsalSR.
///
/// C++ port of fast_canonical_string(dag, mode="wl_only") from canonical.py.
/// ONLY wl_only mode is ported; wl_tiebreak and tuple_only remain in Python.
///
/// Invariants implemented:
///   5  — pairs sorted by |a|+|b|, then |a|, then (a,b) lex.
///   8  — binary op admissibility via ordered_inputs()[0] (B9).
///   9  — normalize_const_creation() is NOT applied here (removed 2026-07-29,
///        T07).  The reachability precondition is the producer's job; this
///        entry point assumes it and raises on an unreachable node.
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

/// Compute fast canonical string (wl_only) — production entry point.
///
/// A pure function of @p dag: no CONST normalisation is applied (see the note on
/// Invariant 9 above).  Handles the two trivial cases (empty DAG, variables-only
/// DAG) and delegates to fast_canonical_string_wl_only.
///
/// The caller must supply a DAG whose non-VAR nodes are all reachable from some
/// variable.  A CONST node with in-degree 0 has no encoding in Sigma_SR at all
/// and raises std::runtime_error rather than being repaired silently.
///
/// @param dag         Input labeled DAG.
/// @param timeout_sec Wall-clock limit in seconds; negative = unlimited.
/// @throws CanonicalTimeoutError if the budget is exceeded.
std::string fast_canonical_string(
    const NativeLabeledDAG& dag,
    double timeout_sec
);

/// Core routine, without the trivial-case guards of fast_canonical_string.
/// Exposed as a separate symbol for unit tests; prefer fast_canonical_string.
std::string fast_canonical_string_wl_only(
    const NativeLabeledDAG& dag,
    double timeout_sec
);

} // namespace isalsr
