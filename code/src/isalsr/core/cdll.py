"""Array-backed Circular Doubly Linked List (CDLL).

Reused verbatim from IsalGraph. Provides O(1) insert_after, O(1) remove,
O(1) next/prev traversal. Uses a free-list stack for node allocation.

Reference: IsalGraph (Lopez-Rubio, 2025, arXiv:2512.10429v2).

Critical invariant: CDLL indices != graph node indices.
    Pointers are CDLL indices; payloads are graph node IDs.
    Use get_value(ptr) to convert CDLL index -> graph node ID.

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations


class CircularDoublyLinkedList:
    """Array-backed circular doubly linked list with integer payloads.

    Nodes are represented by integer indices in [0, capacity).  Internally
    the structure maintains ``_next``, ``_prev``, and ``_data`` arrays plus
    a free-list stack for O(1) allocation/deallocation.

    Args:
        capacity: Maximum number of nodes that can coexist in the list.
    """

    __slots__ = ("_next", "_prev", "_data", "_free", "_size", "_capacity")

    def __init__(self, capacity: int) -> None:
        self._capacity: int = capacity
        self._next: list[int] = [-1] * capacity
        self._prev: list[int] = [-1] * capacity
        self._data: list[int] = [0] * capacity
        # Free list: stack order so first pop yields index 0.
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self._size: int = 0

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of active nodes."""
        return self._size

    def capacity(self) -> int:
        """Return the maximum number of nodes."""
        return self._capacity

    def get_value(self, node: int) -> int:
        """Return the integer payload stored at *node*."""
        return self._data[node]

    def set_value(self, node: int, value: int) -> None:
        """Overwrite the payload stored at *node*."""
        self._data[node] = value

    def next_node(self, node: int) -> int:
        """Return the successor index of *node* in the circular list."""
        return self._next[node]

    def prev_node(self, node: int) -> int:
        """Return the predecessor index of *node* in the circular list."""
        return self._prev[node]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert_after(self, node: int, value: int) -> int:
        """Insert a new node with *value* after the node at index *node*.

        If the list is empty the *node* argument is ignored and the new
        node becomes the sole element (pointing to itself).

        Returns:
            The index of the newly inserted node.

        Raises:
            RuntimeError: If the list has reached its capacity.
        """
        new_node: int = self._allocate_node()
        self._data[new_node] = value

        if self._size == 0:
            self._next[new_node] = new_node
            self._prev[new_node] = new_node
        else:
            next_of_node: int = self._next[node]
            self._next[node] = new_node
            self._prev[new_node] = node
            self._next[new_node] = next_of_node
            self._prev[next_of_node] = new_node

        self._size += 1
        return new_node

    def remove(self, node: int) -> None:
        """Remove *node* from the list and return its index to the free list."""
        if self._size == 0:
            return

        if self._size == 1:
            self._free_node(node)
            self._size = 0
            return

        prev_of_node: int = self._prev[node]
        next_of_node: int = self._next[node]
        self._next[prev_of_node] = next_of_node
        self._prev[next_of_node] = prev_of_node
        self._free_node(node)
        self._size -= 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_node(self) -> int:
        if not self._free:
            raise RuntimeError("CircularDoublyLinkedList is full")
        return self._free.pop()

    def _free_node(self, index: int) -> None:
        self._free.append(index)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(capacity={self._capacity}, size={self._size})"

    def __len__(self) -> int:
        return self._size
