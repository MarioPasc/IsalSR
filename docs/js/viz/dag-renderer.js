/**
 * IsalSR — D3 Labeled DAG Renderer
 *
 * Renders a LabeledDAG into an SVG element using D3 force layout
 * with hierarchical (top-down) positioning.
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
   * @param {Object} [options] - highlight options
   * @param {number} [options.primaryNode] - graph node ID of primary pointer
   * @param {number} [options.secondaryNode] - graph node ID of secondary pointer
   * @param {Set} [options.highlightEdges] - set of "source-target" edge keys to highlight
   * @param {number} [options.newNode] - node ID just inserted (for animation)
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

    // Arrowhead marker
    svg.append('defs').append('marker')
      .attr('id', 'dag-arrow-' + svgId)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9)
      .attr('refY', 5)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto-start-auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#475569');

    // Compute depths for vertical layout
    // Build adjacency from d3Data
    var depthMap = {};
    var maxDepth = 0;
    var outAdj = {};
    var inAdj = {};
    d3Data.nodes.forEach(function (n) {
      outAdj[n.id] = [];
      inAdj[n.id] = [];
    });
    d3Data.edges.forEach(function (e) {
      var s = typeof e.source === 'object' ? e.source.id : e.source;
      var t = typeof e.target === 'object' ? e.target.id : e.target;
      outAdj[s].push(t);
      inAdj[t].push(s);
    });

    // BFS from leaves (no in-edges) to compute depth
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
      var next = depthMap[cur] + 1;
      outAdj[cur].forEach(function (t) {
        if (next > depthMap[t]) {
          depthMap[t] = next;
          if (next > maxDepth) maxDepth = next;
          queue.push(t);
        }
      });
    }

    // Deep copy for D3 mutation
    var nodes = d3Data.nodes.map(function (n) {
      var depth = depthMap[n.id] || 0;
      return {
        id: n.id,
        label: n.label,
        metadata: n.metadata,
        displayLabel: nodeDisplayLabel(n),
        depth: depth,
        // Initial position hint: leaves at bottom, targets at top
        x: width / 2 + (Math.random() - 0.5) * 100,
        y: height - 50 - (depth / Math.max(maxDepth, 1)) * (height - 100)
      };
    });

    var links = d3Data.edges.map(function (e) {
      return {
        source: typeof e.source === 'object' ? e.source.id : e.source,
        target: typeof e.target === 'object' ? e.target.id : e.target
      };
    });

    // Force simulation with hierarchical Y constraint
    var simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function (d) { return d.id; }).distance(70).strength(0.8))
      .force('charge', d3.forceManyBody().strength(-250))
      .force('centerX', d3.forceX(width / 2).strength(0.1))
      .force('depthY', d3.forceY(function (d) {
        // Leaves/variables at bottom, root operations at top
        return height - 50 - (d.depth / Math.max(maxDepth, 1)) * (height - 100);
      }).strength(1.5))
      .force('collision', d3.forceCollide(function (d) { return nodeRadius(d.label) + 8; }));

    var g = svg.append('g');

    // Edges (lines with arrowheads)
    var link = g.selectAll('.dag-edge')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'dag-edge')
      .attr('stroke', '#475569')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', 'url(#dag-arrow-' + svgId + ')');

    // Node groups
    var node = g.selectAll('.dag-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'dag-node');

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

    // Warm up simulation
    simulation.alpha(1).restart();
    for (var i = 0; i < 200; i++) simulation.tick();
    simulation.stop();

    // Clamp positions to viewport
    nodes.forEach(function (d) {
      var r = nodeRadius(d.label) + 5;
      d.x = Math.max(r, Math.min(width - r, d.x));
      d.y = Math.max(r, Math.min(height - r, d.y));
    });

    // Apply final positions with edge shortening for arrowheads
    link
      .attr('x1', function (d) { return d.source.x; })
      .attr('y1', function (d) { return d.source.y; })
      .attr('x2', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        var r = nodeRadius(d.target.label) + 2;
        return d.target.x - (dx / dist) * r;
      })
      .attr('y2', function (d) {
        var dx = d.target.x - d.source.x;
        var dy = d.target.y - d.source.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        var r = nodeRadius(d.target.label) + 2;
        return d.target.y - (dy / dist) * r;
      });

    node.attr('transform', function (d) {
      return 'translate(' + d.x + ',' + d.y + ')';
    });

    // Pointer labels
    if (primaryNode !== undefined || secondaryNode !== undefined) {
      var legend = svg.append('g')
        .attr('transform', 'translate(10, ' + (height - 30) + ')');

      if (primaryNode !== undefined) {
        legend.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 5)
          .attr('fill', 'none').attr('stroke', '#a78bfa').attr('stroke-width', 2);
        legend.append('text').attr('x', 12).attr('y', 4)
          .attr('fill', '#a78bfa').attr('font-size', '10px').attr('font-family', "'Source Sans 3', sans-serif")
          .text('\u03C0 primary');
      }
      if (secondaryNode !== undefined) {
        var yOff = primaryNode !== undefined ? 16 : 0;
        legend.append('circle').attr('cx', 0).attr('cy', yOff).attr('r', 5)
          .attr('fill', 'none').attr('stroke', '#60a5fa').attr('stroke-width', 2)
          .attr('stroke-dasharray', '3,2');
        legend.append('text').attr('x', 12).attr('y', yOff + 4)
          .attr('fill', '#60a5fa').attr('font-size', '10px').attr('font-family', "'Source Sans 3', sans-serif")
          .text('\u03C3 secondary');
      }
    }
  };
})();
