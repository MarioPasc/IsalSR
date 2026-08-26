/**
 * IsalSR — D3 Labeled DAG Renderer
 *
 * Renders a LabeledDAG into an SVG element using deterministic
 * layered layout (nodes evenly spaced per depth layer).
 * Features: directed edges with arrowheads, labeled nodes colored by
 * operation category, pointer indicators.
 */
(function () {
  'use strict';
  window.IsalSR = window.IsalSR || {};

  // Node colors by operation category
  var NODE_COLORS = {
    'VAR':   { fill: '#f59e0b', stroke: '#d97706', text: '#000' },
    'ADD':   { fill: '#fb923c', stroke: '#ea580c', text: '#000' },
    'MUL':   { fill: '#fb923c', stroke: '#ea580c', text: '#000' },
    'SUB':   { fill: '#fbbf24', stroke: '#d97706', text: '#000' },
    'DIV':   { fill: '#fbbf24', stroke: '#d97706', text: '#000' },
    'POW':   { fill: '#fbbf24', stroke: '#d97706', text: '#000' },
    'SIN':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'COS':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'EXP':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'LOG':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'SQRT':  { fill: '#34d399', stroke: '#059669', text: '#000' },
    'ABS':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'NEG':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'INV':   { fill: '#34d399', stroke: '#059669', text: '#000' },
    'CONST': { fill: '#94a3b8', stroke: '#64748b', text: '#000' }
  };

  var DEFAULT_COLOR = { fill: '#60a5fa', stroke: '#3b82f6', text: '#000' };

  // Display labels for nodes
  var DISPLAY_LABELS = {
    'ADD': '+', 'MUL': '\u00D7', 'SUB': '\u2212', 'DIV': '\u00F7',
    'POW': '^', 'SIN': 'sin', 'COS': 'cos', 'EXP': 'exp',
    'LOG': 'log', 'SQRT': '\u221A', 'ABS': '|x|', 'NEG': '\u2212x',
    'INV': '1/x', 'CONST': 'k'
  };

  // Subscript digits
  var SUB_DIGITS = ['\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089'];

  function nodeDisplayLabel(node) {
    if (node.label === 'VAR') {
      var idx = node.metadata ? node.metadata.varIndex : node.id;
      return 'x' + (SUB_DIGITS[idx] || String(idx + 1));
    }
    return DISPLAY_LABELS[node.label] || node.label || '?';
  }

  function nodeColor(label) {
    return NODE_COLORS[label] || DEFAULT_COLOR;
  }

  function nodeRadius(label) {
    if (label === 'VAR' || label === 'CONST') return 18;
    var dl = DISPLAY_LABELS[label] || '';
    if (dl.length > 2) return 24;
    return 20;
  }

  /**
   * Render a LabeledDAG or D3 data object in an SVG element.
   * @param {string} svgId - ID of the target SVG element
   * @param {Object} d3Data - { nodes: [{id, label, metadata}], edges: [{source, target}] }
   * @param {Object} [options]
   */
  IsalSR.renderDAG = function (svgId, d3Data, options) {
    if (typeof d3 === 'undefined') return;

    options = options || {};
    var primaryNode = options.primaryNode;
    var secondaryNode = options.secondaryNode;

    var svgEl = document.getElementById(svgId);
    if (!svgEl) return;

    var svg = d3.select('#' + svgId);
    svg.selectAll('*').remove();

    var rect = svgEl.getBoundingClientRect();
    var width = rect.width || 400;
    var height = rect.height || 350;

    svg.attr('viewBox', '0 0 ' + width + ' ' + height);

    if (!d3Data.nodes || d3Data.nodes.length === 0) return;

    // Marker size constant — the arrowhead path spans 0..8 on x-axis
    var MARKER_LEN = 8;

    // Arrowhead marker — refX=8 so the TIP of the arrow aligns with line end
    svg.append('defs').append('marker')
      .attr('id', 'dag-arrow-' + svgId)
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', MARKER_LEN)
      .attr('refY', 0)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0,-3.5 L 8,0 L 0,3.5 Z')
      .attr('fill', '#64748b');

    // ---- Compute depth layers ----
    var outAdj = {};
    var inAdj = {};
    var nodeById = {};
    d3Data.nodes.forEach(function (n) {
      outAdj[n.id] = [];
      inAdj[n.id] = [];
      nodeById[n.id] = n;
    });
    d3Data.edges.forEach(function (e) {
      var s = typeof e.source === 'object' ? e.source.id : e.source;
      var t = typeof e.target === 'object' ? e.target.id : e.target;
      outAdj[s].push(t);
      inAdj[t].push(s);
    });

    // BFS from source nodes (no in-edges) to compute longest-path depth
    var depthMap = {};
    var maxDepth = 0;
    d3Data.nodes.forEach(function (n) { depthMap[n.id] = 0; });
    var queue = [];
    d3Data.nodes.forEach(function (n) {
      if (inAdj[n.id].length === 0) {
        depthMap[n.id] = 0;
        queue.push(n.id);
      }
    });
    while (queue.length > 0) {
      var cur = queue.shift();
      var nd = depthMap[cur] + 1;
      outAdj[cur].forEach(function (t) {
        if (nd > depthMap[t]) {
          depthMap[t] = nd;
          if (nd > maxDepth) maxDepth = nd;
          queue.push(t);
        }
      });
    }

    // ---- Deterministic layered layout ----
    // Group nodes by depth
    var layers = [];
    for (var li = 0; li <= maxDepth; li++) layers.push([]);
    d3Data.nodes.forEach(function (n) {
      layers[depthMap[n.id]].push(n);
    });

    var padY = 45;
    var layerSpacing = maxDepth > 0 ? (height - 2 * padY) / maxDepth : 0;
    var padX = 40;

    // Assign positions: each layer's nodes equally spaced horizontally
    // Depth 0 at bottom, maxDepth at top
    var posMap = {};
    for (var d = 0; d <= maxDepth; d++) {
      var row = layers[d];
      var count = row.length;
      var usableWidth = width - 2 * padX;
      var spacing = count > 1 ? usableWidth / (count - 1) : 0;
      var startX = count > 1 ? padX : width / 2;
      for (var j = 0; j < count; j++) {
        posMap[row[j].id] = {
          x: startX + j * spacing,
          y: height - padY - d * layerSpacing
        };
      }
    }

    // Build positioned node array
    var nodes = d3Data.nodes.map(function (n) {
      var p = posMap[n.id];
      return {
        id: n.id,
        label: n.label,
        metadata: n.metadata,
        displayLabel: nodeDisplayLabel(n),
        x: p.x,
        y: p.y
      };
    });

    var nodeMap = {};
    nodes.forEach(function (n) { nodeMap[n.id] = n; });

    var links = d3Data.edges.map(function (e) {
      var s = typeof e.source === 'object' ? e.source.id : e.source;
      var t = typeof e.target === 'object' ? e.target.id : e.target;
      return { source: nodeMap[s], target: nodeMap[t] };
    });

    var g = svg.append('g');

    // ---- Edges ----
    // Line endpoint shortened to node border; arrowhead tip sits exactly at border
    g.selectAll('.dag-edge')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'dag-edge')
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1.8)
      .attr('stroke-opacity', 0.75)
      .attr('marker-end', 'url(#dag-arrow-' + svgId + ')')
      .attr('x1', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return d.source.x + (dx / dist) * (nodeRadius(d.source.label) + 2);
      })
      .attr('y1', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return d.source.y + (dy / dist) * (nodeRadius(d.source.label) + 2);
      })
      .attr('x2', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        // Shorten by target radius + gap so arrowhead tip sits just outside
        return d.target.x - (dx / dist) * (nodeRadius(d.target.label) + 4);
      })
      .attr('y2', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return d.target.y - (dy / dist) * (nodeRadius(d.target.label) + 4);
      });

    // ---- Node groups ----
    var node = g.selectAll('.dag-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'dag-node')
      .attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; });

    // Pointer glow rings
    node.append('circle')
      .attr('r', function (d) {
        if (d.id === primaryNode || d.id === secondaryNode) return nodeRadius(d.label) + 6;
        return 0;
      })
      .attr('fill', 'none')
      .attr('stroke', function (d) {
        if (d.id === primaryNode && d.id === secondaryNode) return '#f59e0b';
        if (d.id === primaryNode) return '#a78bfa';
        if (d.id === secondaryNode) return '#60a5fa';
        return 'none';
      })
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', function (d) {
        if (d.id === primaryNode && d.id === secondaryNode) return 'none';
        return d.id === secondaryNode ? '4,3' : 'none';
      })
      .attr('opacity', 0.8);

    // Main node circle
    node.append('circle')
      .attr('r', function (d) { return nodeRadius(d.label); })
      .attr('fill', function (d) { return nodeColor(d.label).fill; })
      .attr('stroke', function (d) { return nodeColor(d.label).stroke; })
      .attr('stroke-width', 2);

    // Node label text
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', function (d) { return nodeColor(d.label).text; })
      .attr('font-family', "'JetBrains Mono', 'Space Mono', monospace")
      .attr('font-size', function (d) {
        var dl = d.displayLabel;
        if (dl.length > 2) return '10px';
        return '13px';
      })
      .attr('font-weight', '700')
      .text(function (d) { return d.displayLabel; });

    // ---- Pointer legend (bottom, side by side) ----
    if (primaryNode !== undefined || secondaryNode !== undefined) {
      var legend = svg.append('g')
        .attr('transform', 'translate(' + (width / 2 - 80) + ', ' + (height - 8) + ')');

      var lx = 0;
      if (primaryNode !== undefined) {
        legend.append('circle').attr('cx', lx, 0).attr('cy', 0).attr('r', 4)
          .attr('fill', 'none').attr('stroke', '#a78bfa').attr('stroke-width', 1.5);
        legend.append('text').attr('x', lx + 8).attr('y', 3.5)
          .attr('fill', '#a78bfa').attr('font-size', '9px').attr('font-family', "'Source Sans 3', sans-serif")
          .text('\u03C0 primary');
        lx += 80;
      }
      if (secondaryNode !== undefined) {
        legend.append('circle').attr('cx', lx).attr('cy', 0).attr('r', 4)
          .attr('fill', 'none').attr('stroke', '#60a5fa').attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '3,2');
        legend.append('text').attr('x', lx + 8).attr('y', 3.5)
          .attr('fill', '#60a5fa').attr('font-size', '9px').attr('font-family', "'Source Sans 3', sans-serif")
          .text('\u03C3 secondary');
      }
    }
  };
})();
