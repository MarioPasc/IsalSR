/**
 * IsalSR — String-to-DAG (S2D) Algorithm
 *
 * Decodes an IsalSR instruction string into a LabeledDAG.
 * Port of isalsr.core.string_to_dag.StringToDAG.
 *
 * Key differences from IsalGraph's StringToGraph:
 * - Two-char tokenization (V/v + label)
 * - Node labels assigned on insertion
 * - Cycle detection on C/c via BFS hasPath
 * - Initial state: m VAR nodes pre-inserted
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  var CDLL = IsalSR.CircularDoublyLinkedList;
  var LabeledDAG = IsalSR.LabeledDAG;

  // Subscript digits for variable display
  var SUBSCRIPTS = ['\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086', '\u2087', '\u2088', '\u2089'];

  function varName(idx) {
    return 'x' + (SUBSCRIPTS[idx] || String(idx + 1));
  }

  /**
   * @param {string} inputString - The IsalSR instruction string.
   * @param {number} numVars - Number of input variables (>= 1).
   */
  function StringToDAG(inputString, numVars) {
    if (!numVars || numVars < 1) numVars = 1;
    this._inputString = inputString;
    this._numVars = numVars;

    // Tokenize
    this._tokens = IsalSR.tokenize(inputString);

    // Count insert tokens to size the DAG
    var insertCount = 0;
    for (var i = 0; i < this._tokens.length; i++) {
      if (this._tokens[i].category === 'insert') insertCount++;
    }
    this._maxNodes = numVars + insertCount;
  }

  /**
   * Execute S2D conversion.
   * @param {Object} [options]
   * @param {boolean} [options.trace=false] - Collect trace snapshots.
   * @returns {{dag: LabeledDAG, traceSteps: Array}}
   */
  StringToDAG.prototype.run = function (options) {
    var trace = (options && options.trace) || false;

    var dag = new LabeledDAG(this._maxNodes + 1);
    var cdll = new CDLL(this._maxNodes + 1);

    // Initialize: m VAR nodes
    var firstCdllNode = -1;
    for (var v = 0; v < this._numVars; v++) {
      dag.addNode('VAR', { varIndex: v });
      var cdllNode = cdll.insertAfter(v === 0 ? -1 : firstCdllNode, v);
      if (v === 0) firstCdllNode = cdllNode;
      // Insert after previous node to maintain order
      if (v > 0) {
        // Already inserted in correct position via insertAfter chain
      }
    }

    var primaryPtr = firstCdllNode;
    var secondaryPtr = firstCdllNode;

    var traceSteps = [];
    if (trace) {
      traceSteps.push({
        dag: dag.clone(),
        cdll: cdll.clone(),
        primaryPtr: primaryPtr,
        secondaryPtr: secondaryPtr,
        token: null,
        action: 'init',
        description: 'Initial state: ' + this._numVars + ' variable' + (this._numVars > 1 ? 's' : '') +
          ' (' + this._varNames().join(', ') + '), both pointers at ' + varName(0)
      });
    }

    // Process tokens
    for (var t = 0; t < this._tokens.length; t++) {
      var tok = this._tokens[t];
      var desc = '';

      switch (tok.category) {
        case 'movement':
          if (tok.token === 'N') {
            primaryPtr = cdll.nextNode(primaryPtr);
            desc = 'Move primary pointer next \u2192 node ' + this._nodeDisplayName(dag, cdll.getValue(primaryPtr));
          } else if (tok.token === 'P') {
            primaryPtr = cdll.prevNode(primaryPtr);
            desc = 'Move primary pointer prev \u2190 node ' + this._nodeDisplayName(dag, cdll.getValue(primaryPtr));
          } else if (tok.token === 'n') {
            secondaryPtr = cdll.nextNode(secondaryPtr);
            desc = 'Move secondary pointer next \u2192 node ' + this._nodeDisplayName(dag, cdll.getValue(secondaryPtr));
          } else if (tok.token === 'p') {
            secondaryPtr = cdll.prevNode(secondaryPtr);
            desc = 'Move secondary pointer prev \u2190 node ' + this._nodeDisplayName(dag, cdll.getValue(secondaryPtr));
          }
          break;

        case 'edge': {
          var priNode = cdll.getValue(primaryPtr);
          var secNode = cdll.getValue(secondaryPtr);
          if (tok.token === 'C') {
            // Edge primary → secondary (if acyclic)
            if (priNode === secNode) {
              desc = 'Edge ' + this._nodeDisplayName(dag, priNode) + '\u2192' +
                this._nodeDisplayName(dag, secNode) + ' \u2014 self-loop, skipped';
            } else if (dag.hasPath(secNode, priNode)) {
              desc = 'Edge ' + this._nodeDisplayName(dag, priNode) + '\u2192' +
                this._nodeDisplayName(dag, secNode) + ' \u2014 would create cycle, skipped';
            } else {
              dag.addEdge(priNode, secNode);
              desc = 'Edge ' + this._nodeDisplayName(dag, priNode) + '\u2192' +
                this._nodeDisplayName(dag, secNode);
            }
          } else {
            // c: Edge secondary → primary
            if (priNode === secNode) {
              desc = 'Edge ' + this._nodeDisplayName(dag, secNode) + '\u2192' +
                this._nodeDisplayName(dag, priNode) + ' \u2014 self-loop, skipped';
            } else if (dag.hasPath(priNode, secNode)) {
              desc = 'Edge ' + this._nodeDisplayName(dag, secNode) + '\u2192' +
                this._nodeDisplayName(dag, priNode) + ' \u2014 would create cycle, skipped';
            } else {
              dag.addEdge(secNode, priNode);
              desc = 'Edge ' + this._nodeDisplayName(dag, secNode) + '\u2192' +
                this._nodeDisplayName(dag, priNode);
            }
          }
          break;
        }

        case 'insert': {
          var labelInfo = tok.label;
          var ptrNode = tok.pointer === 'primary'
            ? cdll.getValue(primaryPtr)
            : cdll.getValue(secondaryPtr);
          var ptrCdll = tok.pointer === 'primary' ? primaryPtr : secondaryPtr;

          var newNodeId = dag.addNode(labelInfo.type);
          dag.addEdge(ptrNode, newNodeId);
          cdll.insertAfter(ptrCdll, newNodeId);

          desc = 'Insert ' + labelInfo.type + ' node (id=' + newNodeId + '), edge ' +
            this._nodeDisplayName(dag, ptrNode) + '\u2192' + labelInfo.type;
          break;
        }

        case 'noop':
          desc = 'No-op (skip)';
          break;
      }

      if (trace) {
        traceSteps.push({
          dag: dag.clone(),
          cdll: cdll.clone(),
          primaryPtr: primaryPtr,
          secondaryPtr: secondaryPtr,
          token: tok,
          action: tok.category,
          description: desc
        });
      }
    }

    return { dag: dag, traceSteps: traceSteps };
  };

  StringToDAG.prototype._varNames = function () {
    var names = [];
    for (var i = 0; i < this._numVars; i++) names.push(varName(i));
    return names;
  };

  StringToDAG.prototype._nodeDisplayName = function (dag, nodeId) {
    var label = dag.getLabel(nodeId);
    if (label === 'VAR') {
      var meta = dag.getMetadata(nodeId);
      return varName(meta ? meta.varIndex : nodeId);
    }
    return label || ('node' + nodeId);
  };

  IsalSR.StringToDAG = StringToDAG;
})();
