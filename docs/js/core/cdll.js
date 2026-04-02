(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  /**
   * Array-backed Circular Doubly Linked List.
   *
   * Faithful port of isalgraph.core.cdll.CircularDoublyLinkedList.
   * Nodes are integer indices in [0, capacity). Internal arrays store
   * next/prev pointers and data payloads. A free-list stack provides
   * O(1) allocation.
   *
   * @param {number} capacity - Maximum number of nodes.
   */
  function CircularDoublyLinkedList(capacity) {
    this._capacity = capacity;
    this._next = new Int32Array(capacity).fill(-1);
    this._prev = new Int32Array(capacity).fill(-1);
    this._data = new Int32Array(capacity);
    // Free list: descending so first pop yields index 0
    this._free = [];
    for (var i = capacity - 1; i >= 0; i--) {
      this._free.push(i);
    }
    this._size = 0;
  }

  // ---- Public accessors ----

  CircularDoublyLinkedList.prototype.size = function () {
    return this._size;
  };

  CircularDoublyLinkedList.prototype.capacity = function () {
    return this._capacity;
  };

  CircularDoublyLinkedList.prototype.getValue = function (node) {
    return this._data[node];
  };

  CircularDoublyLinkedList.prototype.setValue = function (node, value) {
    this._data[node] = value;
  };

  CircularDoublyLinkedList.prototype.nextNode = function (node) {
    return this._next[node];
  };

  CircularDoublyLinkedList.prototype.prevNode = function (node) {
    return this._prev[node];
  };

  // ---- Mutation ----

  /**
   * Insert a new node with the given value after the specified node.
   * If the list is empty, the node argument is ignored and the new
   * node becomes the sole element (pointing to itself).
   *
   * @param {number} node - CDLL index to insert after (ignored if empty).
   * @param {number} value - Graph node index (payload).
   * @returns {number} The index of the newly inserted node.
   */
  CircularDoublyLinkedList.prototype.insertAfter = function (node, value) {
    var newNode = this._allocateNode();
    this._data[newNode] = value;

    if (this._size === 0) {
      this._next[newNode] = newNode;
      this._prev[newNode] = newNode;
    } else {
      var nextOfNode = this._next[node];
      this._next[node] = newNode;
      this._prev[newNode] = node;
      this._next[newNode] = nextOfNode;
      this._prev[nextOfNode] = newNode;
    }

    this._size += 1;
    return newNode;
  };

  /**
   * Remove a node from the list and return its index to the free list.
   *
   * @param {number} node - CDLL index to remove.
   */
  CircularDoublyLinkedList.prototype.remove = function (node) {
    if (this._size === 0) {
      return;
    }

    if (this._size === 1) {
      this._freeNode(node);
      this._size = 0;
      return;
    }

    var prevOfNode = this._prev[node];
    var nextOfNode = this._next[node];
    this._next[prevOfNode] = nextOfNode;
    this._prev[nextOfNode] = prevOfNode;
    this._freeNode(node);
    this._size -= 1;
  };

  // ---- Internal helpers ----

  CircularDoublyLinkedList.prototype._allocateNode = function () {
    if (this._free.length === 0) {
      throw new Error('CircularDoublyLinkedList is full');
    }
    return this._free.pop();
  };

  CircularDoublyLinkedList.prototype._freeNode = function (index) {
    this._free.push(index);
  };

  // ---- Clone for trace snapshots ----

  /**
   * Create a deep copy of this CDLL.
   * @returns {CircularDoublyLinkedList}
   */
  CircularDoublyLinkedList.prototype.clone = function () {
    var copy = new CircularDoublyLinkedList(this._capacity);
    copy._next = new Int32Array(this._next);
    copy._prev = new Int32Array(this._prev);
    copy._data = new Int32Array(this._data);
    copy._free = this._free.slice();
    copy._size = this._size;
    return copy;
  };

  // ---- Iteration helpers ----

  /**
   * Return an array of all active node indices in forward order,
   * starting from the given node.
   *
   * @param {number} startNode - CDLL index to start from.
   * @returns {number[]} Array of CDLL node indices.
   */
  CircularDoublyLinkedList.prototype.toArray = function (startNode) {
    if (this._size === 0) {
      return [];
    }
    var result = [];
    var current = startNode;
    for (var i = 0; i < this._size; i++) {
      result.push(current);
      current = this._next[current];
    }
    return result;
  };

  /**
   * Return an array of data payloads in forward order,
   * starting from the given node.
   *
   * @param {number} startNode - CDLL index to start from.
   * @returns {number[]} Array of graph node indices (payloads).
   */
  CircularDoublyLinkedList.prototype.toDataArray = function (startNode) {
    if (this._size === 0) {
      return [];
    }
    var result = [];
    var current = startNode;
    for (var i = 0; i < this._size; i++) {
      result.push(this._data[current]);
      current = this._next[current];
    }
    return result;
  };

  CircularDoublyLinkedList.prototype.toString = function () {
    return 'CircularDoublyLinkedList(capacity=' + this._capacity + ', size=' + this._size + ')';
  };

  // Export
  IsalSR.CircularDoublyLinkedList = CircularDoublyLinkedList;
})();
