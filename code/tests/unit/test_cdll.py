"""Unit tests for CircularDoublyLinkedList.

Adapted from IsalGraph's test_cdll.py with imports changed to isalsr.
"""

from __future__ import annotations

import pytest

from isalsr.core.cdll import CircularDoublyLinkedList


class TestCdllBasics:
    """Basic CDLL operations."""

    def test_empty_list(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        assert cdll.size() == 0
        assert len(cdll) == 0
        assert cdll.capacity() == 5

    def test_insert_single(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        n = cdll.insert_after(-1, 42)
        assert cdll.size() == 1
        assert cdll.get_value(n) == 42
        # Single node is its own next and prev (circular).
        assert cdll.next_node(n) == n
        assert cdll.prev_node(n) == n

    def test_insert_two(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        a = cdll.insert_after(-1, 10)
        b = cdll.insert_after(a, 20)
        assert cdll.size() == 2
        assert cdll.next_node(a) == b
        assert cdll.next_node(b) == a  # circular
        assert cdll.prev_node(a) == b
        assert cdll.prev_node(b) == a

    def test_insert_three_order(self) -> None:
        """Insert a, b after a, c after b -> circular order a->b->c->a."""
        cdll = CircularDoublyLinkedList(5)
        a = cdll.insert_after(-1, 1)
        b = cdll.insert_after(a, 2)
        c = cdll.insert_after(b, 3)
        assert cdll.next_node(a) == b
        assert cdll.next_node(b) == c
        assert cdll.next_node(c) == a
        assert cdll.prev_node(a) == c
        assert cdll.prev_node(c) == b
        assert cdll.prev_node(b) == a

    def test_insert_middle(self) -> None:
        """Insert c between a and b: a->c->b->a."""
        cdll = CircularDoublyLinkedList(5)
        a = cdll.insert_after(-1, 1)
        b = cdll.insert_after(a, 2)
        c = cdll.insert_after(a, 3)  # insert after a, before b
        assert cdll.next_node(a) == c
        assert cdll.next_node(c) == b
        assert cdll.next_node(b) == a


class TestCdllRemove:
    """Removal operations."""

    def test_remove_only_node(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        n = cdll.insert_after(-1, 1)
        cdll.remove(n)
        assert cdll.size() == 0

    def test_remove_middle(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        a = cdll.insert_after(-1, 1)
        b = cdll.insert_after(a, 2)
        c = cdll.insert_after(b, 3)
        cdll.remove(b)
        assert cdll.size() == 2
        assert cdll.next_node(a) == c
        assert cdll.prev_node(c) == a

    def test_remove_empty_is_noop(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        cdll.remove(0)  # should not raise


class TestCdllCapacity:
    """Capacity and allocation."""

    def test_full_raises(self) -> None:
        cdll = CircularDoublyLinkedList(2)
        a = cdll.insert_after(-1, 1)
        cdll.insert_after(a, 2)
        with pytest.raises(RuntimeError, match="full"):
            cdll.insert_after(a, 3)

    def test_reuse_after_remove(self) -> None:
        cdll = CircularDoublyLinkedList(2)
        a = cdll.insert_after(-1, 1)
        b = cdll.insert_after(a, 2)
        cdll.remove(b)
        c = cdll.insert_after(a, 3)
        assert cdll.size() == 2
        assert cdll.get_value(c) == 3

    def test_allocation_order(self) -> None:
        """First allocation yields index 0 (free list is a stack ending with 0)."""
        cdll = CircularDoublyLinkedList(5)
        first = cdll.insert_after(-1, 99)
        assert first == 0


class TestCdllSetValue:
    """Payload mutation."""

    def test_set_value(self) -> None:
        cdll = CircularDoublyLinkedList(3)
        n = cdll.insert_after(-1, 10)
        cdll.set_value(n, 42)
        assert cdll.get_value(n) == 42


class TestCdllRepr:
    """String representation."""

    def test_repr_empty(self) -> None:
        cdll = CircularDoublyLinkedList(5)
        r = repr(cdll)
        assert "CircularDoublyLinkedList" in r
        assert "capacity=5" in r
        assert "size=0" in r

    def test_repr_nonempty(self) -> None:
        cdll = CircularDoublyLinkedList(10)
        cdll.insert_after(-1, 1)
        cdll.insert_after(0, 2)
        r = repr(cdll)
        assert "size=2" in r
        assert "capacity=10" in r
