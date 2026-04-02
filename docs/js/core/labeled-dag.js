/**
 * IsalSR — LabeledDAG Data Structure
 *
 * Directed acyclic graph with node labels, dual adjacency lists,
 * input-order tracking, and BFS-based cycle detection.
 * Port of isalsr.core.labeled_dag.LabeledDAG.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  /**
   * @param {number} maxNodes - Upper bound on node count.
   */
  function LabeledDAG(maxNodes) {
    this._maxNodes = maxNodes;
    this._nodeCount = 0;
    this._labels = [];        // label per node (e.g. 'ADD', 'SIN', 'VAR')
    this._metadata = [];      // metadata per node (e.g. {varIndex: 0} or {constValue: 1.0})
    this._outAdj = [];        // out-neighbors (node provides input to these)
    this._inAdj = [];         // in-neighbors (these provide input to node)
    this._inputOrder = [];    // ordered input list per node (for binary ops)
    for (var i = 0; i < maxNodes; i++) {
      this._outAdj.push([]);
      this._inAdj.push([]);
      this._inputOrder.push([]);
      this._labels.push(null);
      this._metadata.push(null);
    }
  }

  LabeledDAG.prototype.nodeCount = function () { return this._nodeCount; };
  LabeledDAG.prototype.maxNodes = function () { return this._maxNodes; };

  LabeledDAG.prototype.getLabel = function (node) { return this._labels[node]; };
  LabeledDAG.prototype.getMetadata = function (node) { return this._metadata[node]; };
  LabeledDAG.prototype.getInputOrder = function (node) { return this._inputOrder[node]; };
  LabeledDAG.prototype.outNeighbors = function (node) { return this._outAdj[node]; };
  LabeledDAG.prototype.inNeighbors = function (node) { return this._inAdj[node]; };

  /**
   * Add a new node with the given label and metadata.
   * @returns {number} The new node ID.
   */
  LabeledDAG.prototype.addNode = function (label, metadata) {
    if (this._nodeCount >= this._maxNodes) {
      throw new Error('Maximum nodes reached: ' + this._maxNodes);
    }
    var id = this._nodeCount;
    this._labels[id] = label || null;
    this._metadata[id] = metadata || null;
    this._nodeCount++;
    return id;
  };

  /**
   * Add a directed edge source → target.
   * Does NOT check for cycles (caller must check with hasPath first).
   * @returns {boolean} true if edge was added, false if duplicate.
   */
  LabeledDAG.prototype.addEdge = function (source, target) {
    // Duplicate check
    if (this._outAdj[source].indexOf(target) !== -1) return false;
    this._outAdj[source].push(target);
    this._inAdj[target].push(source);
    this._inputOrder[target].push(source);
    return true;
  };

  /**
   * Check if a directed path exists from `from` to `to` using BFS.
   * Used for cycle detection: before adding edge u→v, check hasPath(v, u).
   */
  LabeledDAG.prototype.hasPath = function (from, to) {
    if (from === to) return true;
    var visited = {};
    var queue = [from];
    visited[from] = true;
    while (queue.length > 0) {
      var current = queue.shift();
      var neighbors = this._outAdj[current];
      for (var i = 0; i < neighbors.length; i++) {
        var next = neighbors[i];
        if (next === to) return true;
        if (!visited[next]) {
          visited[next] = true;
          queue.push(next);
        }
      }
    }
    return false;
  };

  /**
   * Get edge list for D3 rendering.
   * @returns {Array<{source: number, target: number}>}
   */
  LabeledDAG.prototype.getEdgeList = function () {
    var edges = [];
    for (var u = 0; u < this._nodeCount; u++) {
      for (var i = 0; i < this._outAdj[u].length; i++) {
        edges.push({ source: u, target: this._outAdj[u][i] });
      }
    }
    return edges;
  };

  /**
   * Convert to D3-friendly data.
   * @returns {{nodes: Array, edges: Array}}
   */
  LabeledDAG.prototype.toD3Data = function () {
    var nodes = [];
    for (var i = 0; i < this._nodeCount; i++) {
      var label = this._labels[i];
      var meta = this._metadata[i];
      var displayLabel = label || '?';
      if (label === 'VAR' && meta && meta.varIndex !== undefined) {
        displayLabel = 'x\u2081'.substring(0, 1) + String.fromCharCode(0x2081 + meta.varIndex);
      } else if (label === 'CONST') {
        displayLabel = 'k';
      }
      nodes.push({
        id: i,
        label: label,
        displayLabel: displayLabel,
        metadata: meta
      });
    }
    return { nodes: nodes, edges: this.getEdgeList() };
  };

  /**
   * Compute topological depth (distance from leaves/variables) for layout.
   * @returns {Array<number>} depth per node (0 = leaf/variable).
   */
  LabeledDAG.prototype.computeDepths = function () {
    var depths = [];
    var i;
    for (i = 0; i < this._nodeCount; i++) depths.push(-1);

    // BFS from nodes with no in-edges (roots of reversed DAG = leaves of original)
    var queue = [];
    for (i = 0; i < this._nodeCount; i++) {
      if (this._inAdj[i].length === 0) {
        depths[i] = 0;
        queue.push(i);
      }
    }

    while (queue.length > 0) {
      var node = queue.shift();
      var nextDepth = depths[node] + 1;
      var targets = this._outAdj[node];
      for (var j = 0; j < targets.length; j++) {
        var t = targets[j];
        if (nextDepth > depths[t]) {
          depths[t] = nextDepth;
          queue.push(t);
        }
      }
    }

    // Handle unreachable nodes
    for (i = 0; i < this._nodeCount; i++) {
      if (depths[i] === -1) depths[i] = 0;
    }

    return depths;
  };

  /**
   * Deep copy.
   */
  LabeledDAG.prototype.clone = function () {
    var copy = new LabeledDAG(this._maxNodes);
    copy._nodeCount = this._nodeCount;
    for (var i = 0; i < this._maxNodes; i++) {
      copy._labels[i] = this._labels[i];
      copy._metadata[i] = this._metadata[i] ? JSON.parse(JSON.stringify(this._metadata[i])) : null;
      copy._outAdj[i] = this._outAdj[i].slice();
      copy._inAdj[i] = this._inAdj[i].slice();
      copy._inputOrder[i] = this._inputOrder[i].slice();
    }
    return copy;
  };

  IsalSR.LabeledDAG = LabeledDAG;
})();
