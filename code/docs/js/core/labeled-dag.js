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

  // Subscript digits for variable display
  var SUB_DIGITS = ['\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089'];

  // Display names for expression rendering
  var EXPR_DISPLAY = {
    'ADD': '+', 'MUL': '\u00B7', 'SUB': '\u2212', 'DIV': '/',
    'POW': '^', 'SIN': 'sin', 'COS': 'cos', 'EXP': 'exp',
    'LOG': 'log', 'SQRT': '\u221A', 'ABS': 'abs', 'NEG': '\u2212',
    'INV': '1/', 'CONST': 'k'
  };

  var BINARY_OPS = { 'ADD': true, 'MUL': true, 'SUB': true, 'DIV': true, 'POW': true };
  var UNARY_OPS = { 'SIN': true, 'COS': true, 'EXP': true, 'LOG': true, 'SQRT': true, 'ABS': true, 'NEG': true, 'INV': true };

  /**
   * Convert this DAG to a human-readable mathematical expression string.
   * Finds root nodes (nodes with out-edges but no in-edges from other internal nodes,
   * or more precisely, nodes that are not inputs to any other node) and renders them.
   * If multiple roots exist, joins them with " ; ".
   * @returns {string} The mathematical expression.
   */
  LabeledDAG.prototype.toExpression = function () {
    if (this._nodeCount === 0) return '';

    var self = this;
    var visited = {};

    function renderNode(nodeId) {
      if (visited[nodeId]) return '(...)'; // cycle guard
      visited[nodeId] = true;

      var label = self._labels[nodeId];
      var meta = self._metadata[nodeId];

      // Leaf nodes
      if (label === 'VAR') {
        var idx = meta ? meta.varIndex : nodeId;
        visited[nodeId] = false;
        return 'x' + (SUB_DIGITS[idx] || String(idx + 1));
      }
      if (label === 'CONST') {
        visited[nodeId] = false;
        return 'k';
      }

      // Get inputs in insertion order
      var inputs = self._inputOrder[nodeId];
      if (!inputs || inputs.length === 0) {
        // No inputs yet — just show the operation name
        visited[nodeId] = false;
        return (EXPR_DISPLAY[label] || label) + '(?)';
      }

      var result;
      var sym = EXPR_DISPLAY[label] || label;

      if (UNARY_OPS[label]) {
        var child = renderNode(inputs[0]);
        if (label === 'NEG') {
          result = '(\u2212' + child + ')';
        } else if (label === 'INV') {
          result = '(1/' + child + ')';
        } else if (label === 'SQRT') {
          result = '\u221A(' + child + ')';
        } else {
          result = sym + '(' + child + ')';
        }
      } else if (BINARY_OPS[label]) {
        if (inputs.length === 1) {
          // Partially connected binary/variadic op
          result = sym + '(' + renderNode(inputs[0]) + ', ?)';
        } else {
          var parts = [];
          for (var i = 0; i < inputs.length; i++) {
            parts.push(renderNode(inputs[i]));
          }
          if (label === 'POW') {
            result = '(' + parts[0] + '^' + parts[1] + ')';
          } else {
            result = '(' + parts.join(' ' + sym + ' ') + ')';
          }
        }
      } else {
        // Unknown — render generically
        var args = [];
        for (var j = 0; j < inputs.length; j++) {
          args.push(renderNode(inputs[j]));
        }
        result = label + '(' + args.join(', ') + ')';
      }

      visited[nodeId] = false;
      return result;
    }

    // Find root nodes: nodes with out-neighbors that are NOT inputs to any other node
    // (i.e., nodes with no entries in any other node's _inputOrder)
    var isInput = {};
    for (var n = 0; n < this._nodeCount; n++) {
      for (var k = 0; k < this._inputOrder[n].length; k++) {
        isInput[this._inputOrder[n][k]] = true;
      }
    }

    // Roots = non-VAR, non-CONST nodes that nobody uses as input
    // If no such node, find the deepest nodes
    var roots = [];
    for (var r = 0; r < this._nodeCount; r++) {
      if (!isInput[r] && this._labels[r] !== 'VAR' && this._labels[r] !== 'CONST') {
        roots.push(r);
      }
    }

    // Fallback: if only VAR/CONST nodes, just list them
    if (roots.length === 0) {
      var vars = [];
      for (var v = 0; v < this._nodeCount; v++) {
        vars.push(renderNode(v));
      }
      return vars.join(', ');
    }

    var exprs = [];
    for (var e = 0; e < roots.length; e++) {
      exprs.push(renderNode(roots[e]));
    }
    return exprs.join(' ; ');
  };

  IsalSR.LabeledDAG = LabeledDAG;
})();
