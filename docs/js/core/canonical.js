/**
 * IsalSR — Fast Canonical String
 *
 * Definition 3.7 (1-WL subtree hash) and Definition 3.8 (fast canonical
 * string) of the paper, plus the internal-node permutation machinery the
 * playground uses to exhibit the collapse of Theta(k!) numberings onto one
 * string.
 *
 * The greedy search is the D2S search of d2s.js with the candidate rule
 * replaced:
 *   Rule 1 (first-operand eligibility) — a Pow node whose ordered input list
 *          is non-empty may be created only from its base, sigma(c)[0].
 *   Rule 2 (invariant-key greedy selection) — candidates are ordered by
 *          kappa(c) = (label character, 1-WL subtree hash); a unique minimum
 *          is taken greedily, ties are resolved by backtracking over the tied
 *          group and keeping the lexicographically minimal result.
 *
 * The hash is FNV-1a 64-bit, byte-for-byte the function the Python and C++
 * engines use, so a string computed here matches the one the library returns.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  var CDLL = IsalSR.CircularDoublyLinkedList;
  var LabeledDAG = IsalSR.LabeledDAG;

  // NodeType -> the value string the hash commits to. VAR hashes as "var";
  // every other type hashes as its label character (Table 1).
  var TYPE_VALUE = {
    'VAR': 'var', 'ADD': '+', 'MUL': '*', 'NEG': 'g', 'INV': 'i',
    'SIN': 's', 'COS': 'c', 'EXP': 'e', 'LOG': 'l', 'SQRT': 'r',
    'ABS': 'a', 'POW': '^', 'CONST': 'k'
  };

  // Operations whose operands are ordered. Pow is the only one (Table 1).
  var ORDERED_BINARY = { 'POW': true };

  // ------------------------------------------------------------------
  // 1-WL subtree hash (Definition 3.7)
  // ------------------------------------------------------------------

  var FNV_OFFSET = BigInt('0xCBF29CE484222325');
  var FNV_PRIME = BigInt('0x100000001B3');
  var MASK64 = BigInt('0xFFFFFFFFFFFFFFFF');
  var BYTE = BigInt(0xFF);

  function wlNodeHash(labelValue, childHashes) {
    var h = FNV_OFFSET;
    var i, shift;
    for (i = 0; i < labelValue.length; i++) {
      h = ((h ^ BigInt(labelValue.charCodeAt(i))) * FNV_PRIME) & MASK64;
    }
    for (i = 0; i < childHashes.length; i++) {
      for (shift = 0; shift < 64; shift += 8) {
        h = ((h ^ ((childHashes[i] >> BigInt(shift)) & BYTE)) * FNV_PRIME) & MASK64;
      }
    }
    return h;
  }

  /**
   * h(v) = hash(l(v), sort({h(u) : u in N+(v)})), computed in reverse
   * topological order (nodes with no out-edges first). O(k).
   *
   * @param {LabeledDAG} dag
   * @returns {Array<BigInt>} hash per node id.
   */
  IsalSR.wlSubtreeHashes = function (dag) {
    var n = dag.nodeCount();
    var hashes = new Array(n);
    var outDeg = new Array(n);
    var processed = new Array(n);
    var queue = [];
    var u, i;

    for (u = 0; u < n; u++) {
      outDeg[u] = dag.outNeighbors(u).length;
      processed[u] = false;
      if (outDeg[u] === 0) queue.push(u);
    }

    var head = 0;
    while (head < queue.length) {
      u = queue[head++];
      if (processed[u]) continue;
      processed[u] = true;

      var outs = dag.outNeighbors(u);
      var childHashes = [];
      for (i = 0; i < outs.length; i++) childHashes.push(hashes[outs[i]]);
      childHashes.sort(function (a, b) { return a < b ? -1 : (a > b ? 1 : 0); });
      hashes[u] = wlNodeHash(TYPE_VALUE[dag.getLabel(u)] || '?', childHashes);

      var ins = dag.inNeighbors(u);
      for (i = 0; i < ins.length; i++) {
        var v = ins[i];
        outDeg[v] -= 1;
        if (outDeg[v] === 0 && !processed[v]) queue.push(v);
      }
    }

    // A node left unprocessed would sit on a cycle, which S2D cannot produce.
    for (u = 0; u < n; u++) {
      if (hashes[u] === undefined) hashes[u] = FNV_OFFSET;
    }
    return hashes;
  };

  // ------------------------------------------------------------------
  // Spiral displacement set (Definition 3.4)
  // ------------------------------------------------------------------

  function generatePairs(m) {
    var pairs = [];
    for (var a = -m; a <= m; a++) {
      for (var b = -m; b <= m; b++) pairs.push([a, b]);
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

  function walk(cdll, ptr, steps) {
    var i;
    if (steps >= 0) { for (i = 0; i < steps; i++) ptr = cdll.nextNode(ptr); }
    else { for (i = 0; i < -steps; i++) ptr = cdll.prevNode(ptr); }
    return ptr;
  }

  function moves(steps, fwd, bwd) {
    var s = '', i;
    if (steps >= 0) { for (i = 0; i < steps; i++) s += fwd; }
    else { for (i = 0; i < -steps; i++) s += bwd; }
    return s;
  }

  // ------------------------------------------------------------------
  // Fast canonical string (Definition 3.8)
  // ------------------------------------------------------------------

  function cloneState(st) {
    return {
      og: st.og.clone(),
      cdll: st.cdll.clone(),
      p: st.p,
      q: st.q,
      i2o: Object.assign({}, st.i2o),
      o2i: Object.assign({}, st.o2i),
      nodesLeft: st.nodesLeft,
      edgesLeft: st.edgesLeft,
      prefix: st.prefix
    };
  }

  function hasEdge(dag, src, tgt) {
    return dag.outNeighbors(src).indexOf(tgt) !== -1;
  }

  /** Rule 1: is candidate `c` creatable from pointer node `u`? */
  function eligible(ig, u, c) {
    if (!ORDERED_BINARY[ig.getLabel(c)]) return true;
    var sigma = ig.getInputOrder(c);
    if (!sigma || sigma.length === 0) return true;
    return sigma[0] === u;
  }

  /** kappa(c) = (label char, 1-WL hash). Returns -1, 0 or +1. */
  function compareKappa(ig, hashes, a, b) {
    var la = IsalSR.labelCharOf(ig.getLabel(a)) || '';
    var lb = IsalSR.labelCharOf(ig.getLabel(b)) || '';
    if (la !== lb) return la < lb ? -1 : 1;
    var ha = hashes[a], hb = hashes[b];
    if (ha === hb) return 0;
    return ha < hb ? -1 : 1;
  }

  /** Shortest-then-lexicographic order, as Definition 3.8 resolves ties. */
  function better(candidate, incumbent) {
    if (incumbent === null) return true;
    if (candidate.length !== incumbent.length) return candidate.length < incumbent.length;
    return candidate < incumbent;
  }

  function insertNode(ctx, st, tentPtr, c, isPrimary, disp) {
    var tentOut = st.cdll.getValue(tentPtr);
    var label = ctx.ig.getLabel(c);
    var newOut = st.og.addNode(label, ctx.ig.getMetadata(c));
    st.i2o[c] = newOut;
    st.o2i[newOut] = c;
    st.og.addEdge(tentOut, newOut);
    st.cdll.insertAfter(tentPtr, newOut);
    st.nodesLeft -= 1;
    st.edgesLeft -= 1;
    if (isPrimary) {
      st.prefix += moves(disp, 'N', 'P') + 'V' + IsalSR.labelCharOf(label);
      st.p = tentPtr;
    } else {
      st.prefix += moves(disp, 'n', 'p') + 'v' + IsalSR.labelCharOf(label);
      st.q = tentPtr;
    }
  }

  function candidatesFrom(ctx, st, u) {
    var outs = ctx.ig.outNeighbors(u);
    var cands = [];
    for (var i = 0; i < outs.length; i++) {
      var c = outs[i];
      if (c in st.i2o) continue;
      if (!eligible(ctx.ig, u, c)) continue;
      cands.push(c);
    }
    return cands;
  }

  function bestKeyGroup(ctx, cands) {
    cands.sort(function (a, b) { return compareKappa(ctx.ig, ctx.hashes, a, b); });
    var tied = [cands[0]];
    for (var i = 1; i < cands.length; i++) {
      if (compareKappa(ctx.ig, ctx.hashes, cands[i], cands[0]) === 0) tied.push(cands[i]);
      else break;
    }
    return tied;
  }

  function step(ctx, st) {
    if (st.nodesLeft === 0 && st.edgesLeft === 0) return st.prefix;
    ctx.steps += 1;
    if (ctx.steps > ctx.stepLimit) throw new Error('canonicalization step limit reached');

    var pairs = generatePairs(st.og.nodeCount());

    for (var pi = 0; pi < pairs.length; pi++) {
      var a = pairs[pi][0], b = pairs[pi][1];

      var tp = walk(st.cdll, st.p, a);
      var up = st.o2i[st.cdll.getValue(tp)];

      // V: insert via the primary pointer.
      if (st.nodesLeft > 0) {
        var candsV = candidatesFrom(ctx, st, up);
        if (candsV.length > 0) {
          var tiedV = bestKeyGroup(ctx, candsV);
          if (tiedV.length === 1) {
            insertNode(ctx, st, tp, tiedV[0], true, a);
            return step(ctx, st);
          }
          ctx.ties += 1;
          var bestV = null;
          for (var iv = 0; iv < tiedV.length; iv++) {
            var stv = cloneState(st);
            insertNode(ctx, stv, tp, tiedV[iv], true, a);
            var rv = step(ctx, stv);
            if (rv !== null && better(rv, bestV)) bestV = rv;
          }
          return bestV;
        }
      }

      var tq = walk(st.cdll, st.q, b);
      var uq = st.o2i[st.cdll.getValue(tq)];

      // v: insert via the secondary pointer.
      if (st.nodesLeft > 0) {
        var candsv = candidatesFrom(ctx, st, uq);
        if (candsv.length > 0) {
          var tiedv = bestKeyGroup(ctx, candsv);
          if (tiedv.length === 1) {
            insertNode(ctx, st, tq, tiedv[0], false, b);
            return step(ctx, st);
          }
          ctx.ties += 1;
          var bestv = null;
          for (var iw = 0; iw < tiedv.length; iw++) {
            var stw = cloneState(st);
            insertNode(ctx, stw, tq, tiedv[iw], false, b);
            var rw = step(ctx, stw);
            if (rw !== null && better(rw, bestv)) bestv = rw;
          }
          return bestv;
        }
      }

      var tpOut = st.cdll.getValue(tp);
      var tqOut = st.cdll.getValue(tq);

      // C: edge primary -> secondary.
      if (up !== uq && hasEdge(ctx.ig, up, uq) && !hasEdge(st.og, tpOut, tqOut)) {
        st.og.addEdge(tpOut, tqOut);
        st.edgesLeft -= 1;
        st.prefix += moves(a, 'N', 'P') + moves(b, 'n', 'p') + 'C';
        st.p = tp; st.q = tq;
        return step(ctx, st);
      }

      // c: edge secondary -> primary.
      if (up !== uq && hasEdge(ctx.ig, uq, up) && !hasEdge(st.og, tqOut, tpOut)) {
        st.og.addEdge(tqOut, tpOut);
        st.edgesLeft -= 1;
        st.prefix += moves(a, 'N', 'P') + moves(b, 'n', 'p') + 'c';
        st.p = tp; st.q = tq;
        return step(ctx, st);
      }
    }

    return null; // no displacement encodes a remaining element
  }

  /**
   * Compute the fast canonical string of a labeled DAG.
   *
   * @param {LabeledDAG} dag - must satisfy the reachability condition of
   *   Theorem 3.13: every non-variable node reachable from some variable.
   * @returns {{string: string|null, ties: number, wlHashes: Array<BigInt>}}
   */
  IsalSR.fastCanonical = function (dag) {
    var n = dag.nodeCount();
    var numVars = 0, v;
    for (v = 0; v < n; v++) if (dag.getLabel(v) === 'VAR') numVars++;

    var hashes = IsalSR.wlSubtreeHashes(dag);
    var ctx = { ig: dag, hashes: hashes, ties: 0, steps: 0, stepLimit: 200000 };

    var og = new LabeledDAG(n + 1);
    var cdll = new CDLL(n + 1);
    var st = {
      og: og, cdll: cdll, p: -1, q: -1, i2o: {}, o2i: {},
      nodesLeft: n - numVars, edgesLeft: 0, prefix: ''
    };
    for (v = 0; v < n; v++) st.edgesLeft += dag.outNeighbors(v).length;

    var last = -1;
    for (var vi = 0; vi < numVars; vi++) {
      var outId = og.addNode('VAR', { varIndex: vi });
      var cn = cdll.insertAfter(last, outId);
      if (vi === 0) { st.p = cn; st.q = cn; }
      last = cn;
      st.i2o[vi] = outId;
      st.o2i[outId] = vi;
    }

    var result = step(ctx, st);
    return { string: result, ties: ctx.ties, wlHashes: hashes };
  };

  /** Convenience wrapper returning the string only. */
  IsalSR.fastCanonicalString = function (dag) {
    return IsalSR.fastCanonical(dag).string;
  };

  // ------------------------------------------------------------------
  // Internal-node permutations (the k! collapse the playground exhibits)
  // ------------------------------------------------------------------

  /**
   * Relabel the internal nodes of `dag` by `perm`, producing an isomorphic
   * copy. Variable nodes keep their identifiers, as isomorphism condition
   * (iii) requires. Ordered input lists are carried over, so operand order at
   * Pow nodes is preserved.
   *
   * @param {LabeledDAG} dag
   * @param {Array<number>} perm - permutation of [0, k), internal node i of
   *   `dag` becomes internal node perm[i] of the copy.
   * @returns {LabeledDAG}
   */
  IsalSR.permuteInternalNodes = function (dag, perm) {
    var n = dag.nodeCount();
    var numVars = 0, i, v;
    for (v = 0; v < n; v++) if (dag.getLabel(v) === 'VAR') numVars++;

    var oldOf = new Array(n);   // new id -> old id
    var newOf = new Array(n);   // old id -> new id
    for (v = 0; v < numVars; v++) { oldOf[v] = v; newOf[v] = v; }
    for (i = 0; i < perm.length; i++) {
      var oldId = numVars + i;
      var newId = numVars + perm[i];
      oldOf[newId] = oldId;
      newOf[oldId] = newId;
    }

    var copy = new LabeledDAG(n + 1);
    for (v = 0; v < n; v++) {
      var src = oldOf[v];
      copy.addNode(dag.getLabel(src), dag.getMetadata(src));
    }
    // Add each node's in-edges in its own ordered-input order, walking the
    // targets in increasing new identifier, so the adjacency layout is a
    // function of the numbering alone.
    for (v = 0; v < n; v++) {
      var sigma = dag.getInputOrder(oldOf[v]) || [];
      for (i = 0; i < sigma.length; i++) copy.addEdge(newOf[sigma[i]], v);
    }
    return copy;
  };

  /**
   * A numbering-dependent fingerprint: two permutations of one DAG agree on it
   * exactly when they differ by an automorphism, so counting distinct
   * fingerprints over all k! permutations gives k!/|Aut(D)|.
   */
  IsalSR.structuralFingerprint = function (dag) {
    var n = dag.nodeCount();
    var parts = [];
    for (var v = 0; v < n; v++) {
      var meta = dag.getMetadata(v);
      var tag = dag.getLabel(v) + (meta && meta.varIndex !== undefined ? ':' + meta.varIndex : '');
      parts.push(tag + '<' + (dag.getInputOrder(v) || []).join(',') + '>');
    }
    return parts.join('|');
  };

  /** All permutations of [0, k). Callers must keep k small. */
  IsalSR.permutations = function (k) {
    var result = [];
    var current = [];
    var used = new Array(k).fill(false);
    (function recur() {
      if (current.length === k) { result.push(current.slice()); return; }
      for (var i = 0; i < k; i++) {
        if (used[i]) continue;
        used[i] = true; current.push(i);
        recur();
        current.pop(); used[i] = false;
      }
    })();
    return result;
  };
})();
