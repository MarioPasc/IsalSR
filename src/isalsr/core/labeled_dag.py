"""LabeledDAG -- Directed Acyclic Graph with node labels and cycle detection.

The central data structure of IsalSR. Extends IsalGraph's SparseGraph to:
1. Directed-only edges (expressions have data flow direction)
2. Node labels (NodeType: VAR, ADD, MUL, SIN, etc.)
3. Per-node metadata (var_index for VAR, const_value for CONST)
4. Dual adjacency lists (in + out) for efficient evaluation and cycle detection
5. Cycle detection on edge insertion (DFS reachability check)

Edge semantics: edge u -> v means "u provides input to v" (data flow direction).

Reference: IsalGraph's SparseGraph at
    /home/mpascual/research/code/IsalGraph/src/isalgraph/core/sparse_graph.py

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import logging
from collections import deque

from isalsr.core.node_types import NodeType

log = logging.getLogger(__name__)


class LabeledDAG:
    """Directed acyclic graph with labeled nodes and cycle detection.

    Optimized for O(1) average edge insertion and membership testing.
    Always directed. Acyclicity enforced by ``add_edge``.

    Args:
        max_nodes: Upper bound on node count (pre-allocates storage).
    """

    __slots__ = (
        "_out_adj",
        "_in_adj",
        "_input_order",
        "_labels",
        "_node_data",
        "_node_count",
        "_edge_count",
        "_max_nodes",
    )

    def __init__(self, max_nodes: int) -> None:
        self._max_nodes: int = max_nodes
        self._out_adj: list[set[int]] = [set() for _ in range(max_nodes)]
        self._in_adj: list[set[int]] = [set() for _ in range(max_nodes)]
        # Ordered input lists: tracks the order in which in-edges were added.
        # Critical for non-commutative binary ops (SUB, DIV, POW) where
        # operand order determines semantics. V/v creates the first edge,
        # C/c creates subsequent edges.
        self._input_order: list[list[int]] = [[] for _ in range(max_nodes)]
        self._labels: list[NodeType | None] = [None] * max_nodes
        self._node_data: list[dict[str, int | float]] = [{} for _ in range(max_nodes)]
        self._node_count: int = 0
        self._edge_count: int = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Return the current number of nodes."""
        return self._node_count

    @property
    def edge_count(self) -> int:
        """Return the number of directed edges."""
        return self._edge_count

    @property
    def max_nodes(self) -> int:
        """Return the pre-allocated maximum node capacity."""
        return self._max_nodes

    def node_label(self, node: int) -> NodeType:
        """Return the NodeType label of *node*."""
        self._validate_node(node)
        label = self._labels[node]
        if label is None:
            raise ValueError(f"Node {node} has no label (should not happen)")
        return label

    def node_data(self, node: int) -> dict[str, int | float]:
        """Return per-node metadata dict (var_index, const_value, etc.)."""
        self._validate_node(node)
        return self._node_data[node]

    def set_const_value(self, node: int, value: float) -> None:
        """Set the const_value metadata for a node.

        Encapsulated setter for constant optimization. Avoids direct
        access to ``_node_data`` from external modules.

        Args:
            node: The node ID.
            value: The new constant value.
        """
        self._validate_node(node)
        self._node_data[node]["const_value"] = value

    def out_neighbors(self, node: int) -> frozenset[int]:
        """Return nodes that *node* has outgoing edges to (node -> target)."""
        self._validate_node(node)
        return frozenset(self._out_adj[node])

    def out_neighbors_raw(self, node: int) -> set[int]:
        """Return the underlying out-adjacency set for *node* (read-only use).

        Skips frozenset copy. Callers MUST NOT mutate the returned set.
        For performance-critical internal use (canonical backtracking).
        """
        return self._out_adj[node]

    def in_neighbors(self, node: int) -> frozenset[int]:
        """Return nodes that have edges into *node* (source -> node)."""
        self._validate_node(node)
        return frozenset(self._in_adj[node])

    def in_neighbors_raw(self, node: int) -> set[int]:
        """Return the underlying in-adjacency set for *node* (read-only use).

        Skips frozenset copy. Callers MUST NOT mutate the returned set.
        For performance-critical internal use (canonical backtracking).
        """
        return self._in_adj[node]

    def out_degree(self, node: int) -> int:
        """Return the number of outgoing edges from *node*."""
        self._validate_node(node)
        return len(self._out_adj[node])

    def in_degree(self, node: int) -> int:
        """Return the number of incoming edges to *node*."""
        self._validate_node(node)
        return len(self._in_adj[node])

    def has_edge(self, source: int, target: int) -> bool:
        """Return whether the directed edge source -> target exists."""
        self._validate_node(source)
        self._validate_node(target)
        return target in self._out_adj[source]

    def ordered_inputs(self, node: int) -> list[int]:
        """Return the ordered list of input source nodes for *node*.

        For non-commutative binary ops (SUB, DIV, POW), the order determines
        evaluation semantics: first element = first operand, second = second.
        The order is set by edge insertion order: V/v creates the first edge,
        C/c creates subsequent edges.

        For commutative ops (ADD, MUL) and unary ops, order is irrelevant
        but tracked for consistency.

        Args:
            node: The target node ID.

        Returns:
            List of source node IDs in insertion order.
        """
        self._validate_node(node)
        return list(self._input_order[node])

    def has_edge_unchecked(self, source: int, target: int) -> bool:
        """Return whether edge source -> target exists (no bounds check).

        For performance-critical internal use where node IDs are guaranteed valid.
        """
        return target in self._out_adj[source]

    def node_label_unchecked(self, node: int) -> NodeType:
        """Return node label without bounds check.

        For performance-critical internal use where node ID is guaranteed valid.
        """
        return self._labels[node]  # type: ignore[return-value]

    def node_data_unchecked(self, node: int) -> dict[str, int | float]:
        """Return per-node metadata without bounds check.

        For performance-critical internal use where node ID is guaranteed valid.
        """
        return self._node_data[node]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        label: NodeType,
        var_index: int | None = None,
        const_value: float | None = None,
    ) -> int:
        """Add a new node with the given label and return its integer ID.

        Args:
            label: The NodeType for this node.
            var_index: Variable index for VAR nodes (0-based).
            const_value: Constant value for CONST nodes.

        Raises:
            RuntimeError: If the graph has reached ``max_nodes``.
        """
        if self._node_count >= self._max_nodes:
            raise RuntimeError(f"Maximum number of nodes reached: {self._max_nodes}")
        node_id: int = self._node_count
        self._labels[node_id] = label
        data: dict[str, int | float] = {}
        if var_index is not None:
            data["var_index"] = var_index
        if const_value is not None:
            data["const_value"] = const_value
        self._node_data[node_id] = data
        self._node_count += 1
        return node_id

    def add_edge(self, source: int, target: int) -> bool:
        """Add directed edge source -> target if it preserves acyclicity.

        Edge semantics: source provides input to target (data flow).

        Returns:
            True if the edge was added. False if it would create a cycle
            or is a duplicate of an existing edge.

        Raises:
            IndexError: If either node ID is out of range.
        """
        self._validate_node(source)
        self._validate_node(target)

        # Duplicate check.
        if target in self._out_adj[source]:
            return False

        # Self-loop always creates a cycle in a DAG.
        if source == target:
            return False

        # Cycle check: would adding source -> target create a path target -> ... -> source?
        if self._has_path(target, source):
            return False

        self._out_adj[source].add(target)
        self._in_adj[target].add(source)
        self._input_order[target].append(source)
        self._edge_count += 1
        return True

    def add_edge_unchecked(self, source: int, target: int) -> None:
        """Add directed edge source -> target WITHOUT cycle or duplicate checks.

        Use ONLY when acyclicity is guaranteed by construction, e.g., when
        *target* is a freshly created node with no outgoing edges (V/v in
        canonical backtracking). Skips BFS reachability check for O(1)
        insertion instead of O(V+E).

        IMPORTANT: Still tracks _input_order for operand ordering (B9).

        Args:
            source: Source node ID.
            target: Target node ID.
        """
        self._out_adj[source].add(target)
        self._in_adj[target].add(source)
        self._input_order[target].append(source)
        self._edge_count += 1

    def remove_edge(self, source: int, target: int) -> bool:
        """Remove directed edge source -> target.

        Returns:
            True if the edge existed and was removed, False otherwise.
            Needed for backtracking in canonical search.
        """
        self._validate_node(source)
        self._validate_node(target)

        if target not in self._out_adj[source]:
            return False

        self._out_adj[source].discard(target)
        self._in_adj[target].discard(source)
        if source in self._input_order[target]:
            self._input_order[target].remove(source)
        self._edge_count -= 1
        return True

    def undo_node(self) -> None:
        """Remove the last-added node and all its incident edges.

        Used for backtracking in canonical search. Defensively removes
        all incident edges before deleting the node.
        """
        if self._node_count == 0:
            return

        node_id = self._node_count - 1

        # Remove all outgoing edges.
        for target in list(self._out_adj[node_id]):
            self._in_adj[target].discard(node_id)
            if node_id in self._input_order[target]:
                self._input_order[target].remove(node_id)
            self._edge_count -= 1
        self._out_adj[node_id].clear()

        # Remove all incoming edges.
        for source in list(self._in_adj[node_id]):
            self._out_adj[source].discard(node_id)
            self._edge_count -= 1
        self._in_adj[node_id].clear()

        self._labels[node_id] = None
        self._node_data[node_id] = {}
        self._input_order[node_id] = []
        self._node_count -= 1

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def has_cycle_if_added(self, source: int, target: int) -> bool:
        """Check if adding edge source -> target would create a cycle.

        A cycle exists iff there is already a path from target to source
        in the current graph. Uses BFS from target.

        Complexity: O(V + E) per check.
        """
        if source == target:
            return True
        return self._has_path(target, source)

    def _has_path(self, start: int, end: int) -> bool:
        """Check if there is a directed path from *start* to *end* via BFS."""
        if start == end:
            return True

        visited: set[int] = {start}
        queue: deque[int] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self._out_adj[current]:
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[int]:
        """Return nodes in topological order using Kahn's algorithm.

        Raises:
            ValueError: If the graph contains a cycle (should not happen
                        if add_edge is used correctly).
        """
        n = self._node_count
        in_deg = [len(self._in_adj[i]) for i in range(n)]
        queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
        result: list[int] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self._out_adj[node]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != n:
            raise ValueError(
                f"Graph contains a cycle: topological sort produced {len(result)}/{n} nodes"
            )
        return result

    def output_node(self) -> int:
        """Return the unique non-VAR node with out_degree 0 (the expression root).

        For a valid expression DAG, this is the final output node whose value
        is the expression's result.

        Raises:
            ValueError: If there is no unique non-VAR sink node, or if
                there are multiple non-VAR sinks (ambiguous output).
        """
        sinks: list[int] = []
        for i in range(self._node_count):
            if len(self._out_adj[i]) == 0 and self._labels[i] != NodeType.VAR:
                sinks.append(i)

        if len(sinks) == 1:
            return sinks[0]
        if len(sinks) == 0:
            raise ValueError("No non-VAR sink nodes in graph")
        raise ValueError(
            f"Ambiguous output: {len(sinks)} non-VAR sink nodes "
            f"(IDs: {sinks}). A valid expression DAG must have "
            f"exactly one output node."
        )

    # ------------------------------------------------------------------
    # Isomorphism (label-aware backtracking)
    # ------------------------------------------------------------------

    def is_isomorphic(self, other: LabeledDAG) -> bool:
        """Test labeled DAG isomorphism with *other* via backtracking.

        Two labeled DAGs are isomorphic iff there exists a bijection between
        their node sets that preserves: (a) all directed edges, (b) all node
        labels, (c) operand order for non-commutative binary ops (SUB, DIV,
        POW). Variable nodes (VAR) are matched by their var_index, ensuring
        x_1 maps to x_1, x_2 maps to x_2, etc. (variables are distinguishable).

        Condition (c) is necessary because sin(x)-cos(x) and cos(x)-sin(x)
        have the same graph structure but different evaluation semantics.
        The canonical string encodes operand order (Bug Fix B9), so the
        isomorphism check must be consistent.
        """
        if not isinstance(other, LabeledDAG):
            return False

        # Normalize CONST creation edges before comparison so that DAGs
        # differing only in where CONST was created are considered isomorphic.
        self_dag = self.normalize_const_creation() if self._has_const_nodes() else self
        other_dag = other.normalize_const_creation() if other._has_const_nodes() else other

        if self_dag._node_count != other_dag._node_count:
            return False
        if self_dag._edge_count != other_dag._edge_count:
            return False

        n = self_dag._node_count
        if n == 0:
            return True

        # From here, use self_dag/other_dag (CONST-normalized if needed).
        sd, od = self_dag, other_dag

        # Check label multisets match (use .value for sorting since Enum lacks <).
        def _label_key(
            labels: list[NodeType | None], adj_out: list[set[int]], adj_in: list[set[int]], i: int
        ) -> tuple[str, int, int]:
            lbl = labels[i]
            return (lbl.value if lbl is not None else "", len(adj_out[i]), len(adj_in[i]))

        self_labels = sorted(_label_key(sd._labels, sd._out_adj, sd._in_adj, i) for i in range(n))
        other_labels = sorted(_label_key(od._labels, od._out_adj, od._in_adj, i) for i in range(n))
        if self_labels != other_labels:
            return False

        # Build initial mapping: VAR nodes are fixed (matched by var_index).
        mapping: dict[int, int] = {}
        used: set[int] = set()

        # Map VAR nodes by var_index (they are distinguishable).
        self_vars = {
            sd._node_data[i].get("var_index", -1): i
            for i in range(n)
            if sd._labels[i] == NodeType.VAR
        }
        other_vars = {
            od._node_data[i].get("var_index", -1): i
            for i in range(n)
            if od._labels[i] == NodeType.VAR
        }

        if self_vars.keys() != other_vars.keys():
            return False

        for var_idx in self_vars:
            u = self_vars[var_idx]
            v = other_vars[var_idx]
            mapping[u] = v
            used.add(v)

        # Order remaining nodes by (label, out_degree, in_degree) for pruning.
        remaining = [i for i in range(n) if i not in mapping]
        remaining.sort(
            key=lambda u: (
                str(sd._labels[u]),
                len(sd._out_adj[u]),
                len(sd._in_adj[u]),
            ),
            reverse=True,
        )

        other_remaining = [i for i in range(n) if i not in used]
        other_remaining.sort(
            key=lambda u: (
                str(od._labels[u]),
                len(od._out_adj[u]),
                len(od._in_adj[u]),
            ),
            reverse=True,
        )

        def _check_operand_order() -> bool:
            """Verify operand order for non-commutative binary ops."""
            from isalsr.core.node_types import BINARY_OPS

            for u, v in mapping.items():
                label = sd._labels[u]
                if label not in BINARY_OPS:
                    continue
                self_inputs = sd._input_order[u]
                other_inputs = od._input_order[v]
                if len(self_inputs) != len(other_inputs):
                    return False
                for si, oi in zip(self_inputs, other_inputs, strict=True):
                    if si in mapping and mapping[si] != oi:
                        return False
            return True

        def _backtrack(idx: int) -> bool:
            if idx == len(remaining):
                return _check_operand_order()
            u = remaining[idx]
            for v in other_remaining:
                if v in used:
                    continue
                # Labels must match.
                if sd._labels[u] != od._labels[v]:
                    continue
                # Degree must match.
                if len(sd._out_adj[u]) != len(od._out_adj[v]):
                    continue
                if len(sd._in_adj[u]) != len(od._in_adj[v]):
                    continue
                # Check edge consistency with already-mapped nodes.
                ok = True
                for u2, v2 in mapping.items():
                    # Check outgoing edges.
                    if (u2 in sd._out_adj[u]) != (v2 in od._out_adj[v]):
                        ok = False
                        break
                    # Check incoming edges.
                    if (u in sd._out_adj[u2]) != (v in od._out_adj[v2]):
                        ok = False
                        break
                if not ok:
                    continue

                mapping[u] = v
                used.add(v)
                if _backtrack(idx + 1):
                    return True
                del mapping[u]
                used.remove(v)
            return False

        return _backtrack(0)

    # ------------------------------------------------------------------
    # CONST creation edge normalization
    # ------------------------------------------------------------------

    def normalize_const_creation(self) -> LabeledDAG:
        """Return a new DAG with all CONST creation edges moved to x_1 (node 0).

        CONST nodes are evaluation-neutral leaves: they ignore in-edges and
        return ``const_value`` directly. But D2S requires every node to be
        reachable from a VAR via outgoing edges, so V/v creates a "creation
        edge" pointer → CONST. The choice of creation source is semantically
        irrelevant but produces different canonical strings.

        This normalization eliminates that redundancy by standardizing all
        CONST creation edges to come from node 0 (x_1). This is always valid
        because x_1 has no incoming edges (no cycle risk).

        The normalized DAG:
        - Computes the same function: eval(D) == eval(normalize(D))
        - Has deterministic CONST creation edges
        - Produces a unique canonical string for each equivalence class

        Returns:
            A new LabeledDAG with normalized CONST creation edges.
        """
        new = LabeledDAG(self._max_nodes)

        # Copy all nodes.
        const_nodes: set[int] = set()
        for i in range(self._node_count):
            label = self._labels[i]
            if label is None:
                continue
            data = self._node_data[i]
            new.add_node(
                label,
                var_index=int(data["var_index"]) if "var_index" in data else None,
                const_value=float(data["const_value"]) if "const_value" in data else None,
            )
            if label == NodeType.CONST:
                const_nodes.add(i)

        # Copy all edges EXCEPT creation edges (edges TO CONST nodes).
        for src in range(self._node_count):
            for tgt in self._out_adj[src]:
                if tgt in const_nodes:
                    continue  # Skip creation edges; will re-add from x_1.
                new.add_edge(src, tgt)

        # Add normalized creation edges: x_1 (node 0) -> each CONST.
        for c in sorted(const_nodes):
            new.add_edge(0, c)

        return new

    def _has_const_nodes(self) -> bool:
        """Return True if the DAG contains any CONST nodes."""
        return any(self._labels[i] == NodeType.CONST for i in range(self._node_count))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def var_nodes(self) -> list[int]:
        """Return sorted list of VAR node IDs."""
        return [i for i in range(self._node_count) if self._labels[i] == NodeType.VAR]

    def non_var_nodes(self) -> list[int]:
        """Return sorted list of non-VAR node IDs."""
        return [i for i in range(self._node_count) if self._labels[i] != NodeType.VAR]

    def _validate_node(self, node: int) -> None:
        """Raise IndexError if *node* is out of range."""
        if node < 0 or node >= self._node_count:
            raise IndexError(f"Invalid node ID: {node} (node_count={self._node_count})")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        label_counts: dict[str, int] = {}
        for i in range(self._node_count):
            lbl = self._labels[i]
            name = lbl.name if lbl is not None else "?"
            label_counts[name] = label_counts.get(name, 0) + 1
        labels_str = ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items()))
        return (
            f"{self.__class__.__name__}("
            f"nodes={self._node_count}, "
            f"edges={self._edge_count}, "
            f"labels={{{labels_str}}})"
        )
