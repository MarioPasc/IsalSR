/// cdll.cpp — NativeCDLL implementation.
///
/// Matches Python's CircularDoublyLinkedList (cdll.py) exactly,
/// including free-list allocation order and error messages.

#include <isalsr/cdll.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace isalsr {

NativeCDLL::NativeCDLL(int32_t capacity)
    : size_(0), capacity_(capacity),
      next_(static_cast<std::size_t>(capacity), -1),
      prev_(static_cast<std::size_t>(capacity), -1),
      data_(static_cast<std::size_t>(capacity), 0)
{
    // Free-list initialised as range(capacity-1, -1, -1) pushed to a stack
    // whose back is the top.  Popping the back yields index 0 first, then 1,
    // then 2, …  — identical to Python's list.pop() on the reversed list.
    free_stack_.reserve(static_cast<std::size_t>(capacity));
    for (int32_t i = capacity - 1; i >= 0; --i) {
        free_stack_.push_back(i);
    }
}

int32_t NativeCDLL::allocate_node() {
    if (free_stack_.empty()) {
        throw std::runtime_error("CircularDoublyLinkedList is full");
    }
    int32_t idx = free_stack_.back();
    free_stack_.pop_back();
    return idx;
}

void NativeCDLL::free_node(int32_t index) {
    free_stack_.push_back(index);
}

int32_t NativeCDLL::insert_after(int32_t node, int32_t value) {
    int32_t new_node = allocate_node();
    data_[static_cast<std::size_t>(new_node)] = value;

    if (size_ == 0) {
        // When empty, `node` arg is ignored — new slot is a self-loop.
        next_[static_cast<std::size_t>(new_node)] = new_node;
        prev_[static_cast<std::size_t>(new_node)] = new_node;
    } else {
        int32_t next_of_node = next_[static_cast<std::size_t>(node)];
        next_[static_cast<std::size_t>(node)]       = new_node;
        prev_[static_cast<std::size_t>(new_node)]   = node;
        next_[static_cast<std::size_t>(new_node)]   = next_of_node;
        prev_[static_cast<std::size_t>(next_of_node)] = new_node;
    }

    ++size_;
    return new_node;
}

void NativeCDLL::remove(int32_t node) {
    if (size_ == 0) {
        return;  // Python no-ops on empty list
    }

    if (size_ == 1) {
        free_node(node);
        size_ = 0;
        return;
    }

    int32_t prev_of_node = prev_[static_cast<std::size_t>(node)];
    int32_t next_of_node = next_[static_cast<std::size_t>(node)];
    next_[static_cast<std::size_t>(prev_of_node)] = next_of_node;
    prev_[static_cast<std::size_t>(next_of_node)] = prev_of_node;
    free_node(node);
    --size_;
}

} // namespace isalsr
