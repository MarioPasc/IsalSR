/**
 * IsalSR — DAG-to-String (D2S) Algorithm
 *
 * Encodes a LabeledDAG as an IsalSR instruction string.
 * Greedy algorithm: iterates displacement pairs sorted by |a|+|b|,
 * trying V/v insertion, then C/c edge creation.
 *
 * Adapted from IsalGraph's GraphToString with:
 * - Labeled insertion tokens (V+label, v+label)
 * - Directed edge semantics (C = pri→sec, c = sec→pri)
 * - Input-order tracking for binary ops (B9)
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  var CDLL = IsalSR.CircularDoublyLinkedList;
  var LabeledDAG = IsalSR.LabeledDAG;

  // Label char for each NodeType
  var TYPE_TO_LABEL = {
    'ADD': '+', 'MUL': '*', 'SUB': '-', 'DIV': '/', 'POW': '^',
    'SIN': 's', 'COS': 'c', 'EXP': 'e', 'LOG': 'l', 'SQRT': 'r',
    'ABS': 'a', 'NEG': 'g', 'INV': 'i', 'CONST': 'k'
  };

  /**
   * Generate displacement pairs sorted by |a|+|b|.
   */
  function generatePairs(m) {
    var pairs = [];
    for (var a = -m; a <= m; a++) {
      for (var b = -m; b <= m; b++) {
        pairs.push([a, b]);
      }
    }
    pairs.sort(function (p1, p2) {
      var c1 = Math.abs(p1[0]) + Math.abs(p1[1]);
      var c2 = Math.abs(p2[0]) + Math.abs(p2[1]);
      if (c1 !== c2) return c1 - c2;
      if (Math.abs(p1[0]) !== Math.abs(p2[0])) return Math.abs(p1[0]) - Math.abs(p2[0]);
      if (p1[0] !== p2[0]) return p1[0] - p2[0];
      return p1[1] - p2[1];
    });
    return pairs;
  }

  /**
   * @param {LabeledDAG} inputDAG - The DAG to encode.
   * @param {number} startVar - Index of the starting variable node (default 0 = x₁).
   */
  function DAGToString(inputDAG, startVar) {
    this._input = inputDAG;
    this._startVar = startVar || 0;
    this._outputString = '';
    this._cdll = new CDLL(inputDAG.nodeCount() + 1);
    this._primaryPtr = -1;
    this._secondaryPtr = -1;
    // Maps: input node ID ↔ output node ID
    this._i2o = {};
    this._o2i = {};
    // Output DAG (tracks what's been built so far)
    this._outputDAG = new LabeledDAG(inputDAG.nodeCount() + 1);
  }

  /**
   * Execute D2S conversion.
   * @returns {{string: string}}
   */
  DAGToString.prototype.run = function () {
    var numVars = 0;
    for (var v = 0; v < this._input.nodeCount(); v++) {
      if (this._input.getLabel(v) === 'VAR') numVars++;
    }

    // Initialize: insert all VAR nodes into output in order
    var lastCdll = -1;
    for (var vi = 0; vi < numVars; vi++) {
      var outId = this._outputDAG.addNode('VAR', { varIndex: vi });
      var cdllNode = this._cdll.insertAfter(lastCdll, outId);
      if (vi === 0) this._primaryPtr = cdllNode;
      if (vi === 0) this._secondaryPtr = cdllNode;
      lastCdll = cdllNode;
      this._i2o[vi] = outId;
      this._o2i[outId] = vi;
    }

    // Count nodes and edges to insert
    var numNodesToInsert = this._input.nodeCount() - numVars;
    var numEdgesToInsert = 0;
    for (var n = 0; n < this._input.nodeCount(); n++) {
      numEdgesToInsert += this._input.outNeighbors(n).length;
    }

    // Main loop: greedy displacement pair search
    while (numNodesToInsert > 0 || numEdgesToInsert > 0) {
      var currentNodeCount = this._outputDAG.nodeCount();
      var pairs = generatePairs(currentNodeCount);
      var found = false;

      for (var pi = 0; pi < pairs.length; pi++) {
        var priMoves = pairs[pi][0];
        var secMoves = pairs[pi][1];

        var tentPriPtr = this._movePointer(this._primaryPtr, priMoves);
        var tentPriOut = this._cdll.getValue(tentPriPtr);
        var tentPriIn = this._o2i[tentPriOut];

        // V: insert new node via primary pointer?
        if (numNodesToInsert > 0) {
          var candidateV = this._findInsertableNeighbor(tentPriIn);
          if (candidateV !== null) {
            var label = this._input.getLabel(candidateV);
            var labelChar = TYPE_TO_LABEL[label];
            if (labelChar) {
              var newOut = this._outputDAG.addNode(label, this._input.getMetadata(candidateV));
              this._i2o[candidateV] = newOut;
              this._o2i[newOut] = candidateV;
              this._outputDAG.addEdge(tentPriOut, newOut);
              numNodesToInsert--;
              numEdgesToInsert--;
              this._cdll.insertAfter(tentPriPtr, newOut);
              this._emitPrimaryMoves(priMoves);
              this._outputString += 'V' + labelChar;
              this._primaryPtr = tentPriPtr;
              found = true;
              break;
            }
          }
        }

        var tentSecPtr = this._movePointer(this._secondaryPtr, secMoves);
        var tentSecOut = this._cdll.getValue(tentSecPtr);
        var tentSecIn = this._o2i[tentSecOut];

        // v: insert new node via secondary pointer?
        if (numNodesToInsert > 0) {
          var candidatev = this._findInsertableNeighbor(tentSecIn);
          if (candidatev !== null) {
            var labelv = this._input.getLabel(candidatev);
            var labelCharv = TYPE_TO_LABEL[labelv];
            if (labelCharv) {
              var newOutv = this._outputDAG.addNode(labelv, this._input.getMetadata(candidatev));
              this._i2o[candidatev] = newOutv;
              this._o2i[newOutv] = candidatev;
              this._outputDAG.addEdge(tentSecOut, newOutv);
              numNodesToInsert--;
              numEdgesToInsert--;
              this._cdll.insertAfter(tentSecPtr, newOutv);
              this._emitSecondaryMoves(secMoves);
              this._outputString += 'v' + labelCharv;
              this._secondaryPtr = tentSecPtr;
              found = true;
              break;
            }
          }
        }

        // C: edge primary → secondary?
        if (tentPriIn !== tentSecIn && this._hasInputEdge(tentPriIn, tentSecIn) &&
            !this._hasOutputEdge(tentPriOut, tentSecOut)) {
          this._outputDAG.addEdge(tentPriOut, tentSecOut);
          numEdgesToInsert--;
          this._emitPrimaryMoves(priMoves);
          this._emitSecondaryMoves(secMoves);
          this._outputString += 'C';
          this._primaryPtr = tentPriPtr;
          this._secondaryPtr = tentSecPtr;
          found = true;
          break;
        }

        // c: edge secondary → primary?
        if (tentPriIn !== tentSecIn && this._hasInputEdge(tentSecIn, tentPriIn) &&
            !this._hasOutputEdge(tentSecOut, tentPriOut)) {
          this._outputDAG.addEdge(tentSecOut, tentPriOut);
          numEdgesToInsert--;
          this._emitPrimaryMoves(priMoves);
          this._emitSecondaryMoves(secMoves);
          this._outputString += 'c';
          this._primaryPtr = tentPriPtr;
          this._secondaryPtr = tentSecPtr;
          found = true;
          break;
        }
      }

      if (!found) {
        break; // Could not encode further — partial string
      }
    }

    return { string: this._outputString };
  };

  DAGToString.prototype._movePointer = function (ptr, steps) {
    if (steps >= 0) {
      for (var i = 0; i < steps; i++) ptr = this._cdll.nextNode(ptr);
    } else {
      for (var j = 0; j < -steps; j++) ptr = this._cdll.prevNode(ptr);
    }
    return ptr;
  };

  /**
   * Find an out-neighbor of inputNode in the input DAG that hasn't been mapped yet.
   */
  DAGToString.prototype._findInsertableNeighbor = function (inputNode) {
    var outs = this._input.outNeighbors(inputNode);
    for (var i = 0; i < outs.length; i++) {
      if (!(outs[i] in this._i2o)) return outs[i];
    }
    return null;
  };

  DAGToString.prototype._hasInputEdge = function (src, tgt) {
    var outs = this._input.outNeighbors(src);
    return outs.indexOf(tgt) !== -1;
  };

  DAGToString.prototype._hasOutputEdge = function (src, tgt) {
    var outs = this._outputDAG.outNeighbors(src);
    return outs.indexOf(tgt) !== -1;
  };

  DAGToString.prototype._emitPrimaryMoves = function (steps) {
    if (steps >= 0) { for (var i = 0; i < steps; i++) this._outputString += 'N'; }
    else { for (var j = 0; j < -steps; j++) this._outputString += 'P'; }
  };

  DAGToString.prototype._emitSecondaryMoves = function (steps) {
    if (steps >= 0) { for (var i = 0; i < steps; i++) this._outputString += 'n'; }
    else { for (var j = 0; j < -steps; j++) this._outputString += 'p'; }
  };

  IsalSR.DAGToString = DAGToString;
})();
