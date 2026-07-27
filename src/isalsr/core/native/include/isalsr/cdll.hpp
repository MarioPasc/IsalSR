/// cdll.hpp — C++ port of isalsr.core.cdll.CircularDoublyLinkedList.
///
/// Array-backed circular doubly linked list with integer payloads.
/// CRITICAL INVARIANT: CDLL slot indices are NOT graph node indices.
///   Each slot stores a graph node ID as its `data` payload.
///   Use get_value(slot) to convert a CDLL slot index to a graph node ID.
///   Pointers held by the S2D/D2S algorithms are CDLL slot indices.
///
/// Reference: cdll.py, IsalGraph (Lopez-Rubio, 2025, arXiv:2512.10429v2).

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace isalsr {

/// Array-backed circular doubly linked list with integer payloads.
///
/// Slot indices are internal to this structure and are NOT graph node IDs.
/// The data field at each slot holds the graph node ID (the "payload").
class NativeCDLL {
public:
    int32_t size_;
    int32_t capacity_;
    std::vector<int32_t> next_;   ///< next_[i] = successor slot index of slot i
    std::vector<int32_t> prev_;   ///< prev_[i] = predecessor slot index of slot i
    std::vector<int32_t> data_;   ///< data_[i] = graph node ID stored in slot i
    /// Free-list stack (back = top).  Pop from back yields lowest free index first,
    /// matching Python's `list(range(capacity-1, -1, -1))` popped from the right.
    std::vector<int32_t> free_stack_;

    explicit NativeCDLL(int32_t capacity);

    int32_t size()     const noexcept { return size_; }
    int32_t capacity() const noexcept { return capacity_; }

    /// Return the graph node ID stored in slot `node`.
    int32_t get_value(int32_t node) const { return data_.at(static_cast<std::size_t>(node)); }

    /// Overwrite the graph node ID stored in slot `node`.
    void set_value(int32_t node, int32_t value) {
        data_.at(static_cast<std::size_t>(node)) = value;
    }

    /// Return the successor slot index of `node`.
    int32_t next_node(int32_t node) const { return next_.at(static_cast<std::size_t>(node)); }

    /// Return the predecessor slot index of `node`.
    int32_t prev_node(int32_t node) const { return prev_.at(static_cast<std::size_t>(node)); }

    /// Insert a new slot with payload `value` after slot `node`.
    ///
    /// When the list is empty, `node` is ignored and the new slot becomes the
    /// sole element pointing to itself — matching Python's behaviour exactly.
    ///
    /// Returns the new slot index.
    /// Raises std::runtime_error (→ Python RuntimeError) when full.
    int32_t insert_after(int32_t node, int32_t value);

    /// Remove slot `node` from the list and return its index to the free list.
    /// No-op when the list is empty (matches Python behaviour).
    void remove(int32_t node);

private:
    int32_t allocate_node();
    void    free_node(int32_t index);
};

} // namespace isalsr
