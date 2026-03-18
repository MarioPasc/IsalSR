"""DAGToString (D2S) -- Convert a LabeledDAG to an IsalSR instruction string.

Greedy algorithm adapted from IsalGraph's GraphToString (Lopez-Rubio, 2025).
Key differences from IsalGraph:
1. Emits two-char labeled tokens (V+, Vs, etc.) instead of bare V/v.
2. Always directed: C is primary->secondary, c is secondary->primary.
3. Candidate selection for V/v considers outgoing edges only (data flow).
4. Initial state: m VAR nodes pre-inserted (not 1 unlabeled node).

Bug fixes from IsalGraph applied from the start:
- B2: Sort pairs by |a|+|b| (not a+b).
- B3: Loop while nodes OR edges remain (not AND).
- B4: Update pointers after emitting movement instructions.
- B7: insert_after takes CDLL ptr, not graph node ID.
- B8: V/v checks node not in _i2o, not edge existence.

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NODE_TYPE_TO_LABEL, NodeType

log = logging.getLogger(__name__)


def generate_pairs_sorted_by_sum(m: int) -> list[tuple[int, int]]:
    """Return all integer pairs (a, b) with a, b in [-m, m],
    sorted by |a| + |b| (total displacement cost).

    Within the same cost, pairs are further sorted by (|a|, |b|) for
    determinism, and then by (a, b) lexicographically for tie-breaking.

    This is the "spiral enumeration" of Z^2 around the origin from
    Lopez-Rubio (2025), Idea.pdf.

    Args:
        m: Positive integer defining the range bounds.

    Returns:
        Sorted list of (a, b) tuples.

    Raises:
        ValueError: If *m* is not positive.
    """
    if m <= 0:
        raise ValueError("m must be a positive integer.")

    pairs: list[tuple[int, int]] = [(a, b) for a in range(-m, m + 1) for b in range(-m, m + 1)]
    # BUG FIX B2: was pair[0] + pair[1]. Must be |a| + |b|.
    pairs.sort(key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]), pair))
    return pairs


class DAGToString:
    """Convert a ``LabeledDAG`` into an IsalSR instruction string.

    The greedy algorithm searches for the least-cost pointer displacement
    that enables either a V/v (labeled node+edge insertion) or C/c
    (edge-only insertion) at each step.

    Args:
        input_dag: The labeled DAG to encode.
        initial_node: Starting node in the input DAG (default 0 = x_1).
    """

    __slots__ = (
        "_input_dag",
        "_output_string",
        "_cdll",
        "_primary_ptr",
        "_secondary_ptr",
        "_output_dag",
        "_i2o",
        "_o2i",
        "_num_variables",
        "_trace_log",
    )

    def __init__(self, input_dag: LabeledDAG, initial_node: int = 0) -> None:
        if initial_node < 0 or initial_node >= input_dag.node_count:
            raise ValueError(
                f"initial_node={initial_node} out of range [0, {input_dag.node_count})"
            )

        self._input_dag: LabeledDAG = input_dag
        self._output_string: str = ""
        self._num_variables: int = len(input_dag.var_nodes())
        self._cdll: CircularDoublyLinkedList = CircularDoublyLinkedList(input_dag.node_count)
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1
        self._output_dag: LabeledDAG = LabeledDAG(input_dag.node_count)
        self._i2o: dict[int, int] = {}
        self._o2i: dict[int, int] = {}
        self._trace_log: list[tuple[LabeledDAG, CircularDoublyLinkedList, int, int, str]] = []

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def run(self, *, trace: bool = False) -> str:
        """Execute the DAG-to-string conversion.

        Args:
            trace: If ``True``, collect deep-copied snapshots.

        Returns:
            The IsalSR instruction string.
        """
        self._check_reachability()
        self._initialize_variables()

        num_nodes_to_insert: int = self._input_dag.node_count - self._num_variables
        num_edges_to_insert: int = self._input_dag.edge_count

        # BUG FIX B3: was ``and``; must continue while nodes OR edges remain.
        while num_nodes_to_insert > 0 or num_edges_to_insert > 0:
            if trace:
                self._trace_log.append(self._snapshot())

            current_node_count = self._output_dag.node_count
            pairs = generate_pairs_sorted_by_sum(current_node_count)

            found = False
            for num_primary_moves, num_secondary_moves in pairs:
                # ---- tentative primary position ----
                tent_pri_ptr = self._move_pointer(self._primary_ptr, num_primary_moves)
                tent_pri_out = self._cdll.get_value(tent_pri_ptr)
                tent_pri_in = self._o2i[tent_pri_out]

                # -- V: insert new node via primary's outgoing edge? --
                if num_nodes_to_insert > 0:
                    candidate = self._find_new_out_neighbor(tent_pri_in)
                    if candidate is not None:
                        new_out = self._add_mapped_node(candidate)
                        num_nodes_to_insert -= 1
                        self._output_dag.add_edge(tent_pri_out, new_out)
                        num_edges_to_insert -= 1
                        # BUG FIX B7: insert_after takes CDLL ptr, not graph node.
                        self._cdll.insert_after(tent_pri_ptr, new_out)
                        self._emit_primary_moves(num_primary_moves)
                        label_char = NODE_TYPE_TO_LABEL[self._input_dag.node_label(candidate)]
                        self._output_string += "V" + label_char
                        # BUG FIX B4: update pointer.
                        self._primary_ptr = tent_pri_ptr
                        found = True
                        break

                # ---- tentative secondary position ----
                tent_sec_ptr = self._move_pointer(self._secondary_ptr, num_secondary_moves)
                tent_sec_out = self._cdll.get_value(tent_sec_ptr)
                tent_sec_in = self._o2i[tent_sec_out]

                # -- v: insert new node via secondary's outgoing edge? --
                if num_nodes_to_insert > 0:
                    candidate = self._find_new_out_neighbor(tent_sec_in)
                    if candidate is not None:
                        new_out = self._add_mapped_node(candidate)
                        num_nodes_to_insert -= 1
                        self._output_dag.add_edge(tent_sec_out, new_out)
                        num_edges_to_insert -= 1
                        # BUG FIX B7: same fix for secondary.
                        self._cdll.insert_after(tent_sec_ptr, new_out)
                        self._emit_secondary_moves(num_secondary_moves)
                        label_char = NODE_TYPE_TO_LABEL[self._input_dag.node_label(candidate)]
                        self._output_string += "v" + label_char
                        # BUG FIX B4: update pointer.
                        self._secondary_ptr = tent_sec_ptr
                        found = True
                        break

                # -- C: edge primary -> secondary in input but not output? --
                if self._input_dag.has_edge(
                    tent_pri_in, tent_sec_in
                ) and not self._output_dag.has_edge(tent_pri_out, tent_sec_out):
                    self._output_dag.add_edge(tent_pri_out, tent_sec_out)
                    num_edges_to_insert -= 1
                    self._emit_primary_moves(num_primary_moves)
                    self._emit_secondary_moves(num_secondary_moves)
                    self._output_string += "C"
                    # BUG FIX B4: update both pointers.
                    self._primary_ptr = tent_pri_ptr
                    self._secondary_ptr = tent_sec_ptr
                    found = True
                    break

                # -- c: edge secondary -> primary in input but not output? --
                if self._input_dag.has_edge(
                    tent_sec_in, tent_pri_in
                ) and not self._output_dag.has_edge(tent_sec_out, tent_pri_out):
                    self._output_dag.add_edge(tent_sec_out, tent_pri_out)
                    num_edges_to_insert -= 1
                    self._emit_primary_moves(num_primary_moves)
                    self._emit_secondary_moves(num_secondary_moves)
                    self._output_string += "c"
                    # BUG FIX B4: update both pointers.
                    self._primary_ptr = tent_pri_ptr
                    self._secondary_ptr = tent_sec_ptr
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    "DAGToString: no valid operation found. "
                    f"Remaining: {num_nodes_to_insert} nodes, "
                    f"{num_edges_to_insert} edges. "
                    "This indicates an algorithmic error or unreachable nodes."
                )

        if trace:
            self._trace_log.append(self._snapshot())

        return self._output_string

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_variables(self) -> None:
        """Map all m VAR nodes from input to output and insert into CDLL.

        VAR nodes are inserted into the CDLL in var_index order, matching
        the S2D initial state convention.
        """
        # Sort VAR nodes by var_index for deterministic order.
        var_nodes = self._input_dag.var_nodes()
        var_nodes_sorted = sorted(
            var_nodes,
            key=lambda n: self._input_dag.node_data(n).get("var_index", n),
        )

        prev_cdll_node: int = -1
        first_cdll_node: int = -1

        for input_node in var_nodes_sorted:
            output_node = self._output_dag.add_node(
                NodeType.VAR,
                var_index=int(self._input_dag.node_data(input_node).get("var_index", 0)),
            )
            self._i2o[input_node] = output_node
            self._o2i[output_node] = input_node

            cdll_node = self._cdll.insert_after(prev_cdll_node, output_node)
            if first_cdll_node == -1:
                first_cdll_node = cdll_node
            prev_cdll_node = cdll_node

        # Both pointers on the first VAR node.
        self._primary_ptr = first_cdll_node
        self._secondary_ptr = first_cdll_node

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_mapped_node(self, input_node: int) -> int:
        """Add a new node to the output DAG, copying label and data from input."""
        label = self._input_dag.node_label(input_node)
        data = self._input_dag.node_data(input_node)

        var_index = data.get("var_index")
        const_value = data.get("const_value")

        output_node = self._output_dag.add_node(
            label,
            var_index=int(var_index) if var_index is not None else None,
            const_value=float(const_value) if const_value is not None else None,
        )
        self._i2o[input_node] = output_node
        self._o2i[output_node] = input_node
        return output_node

    def _find_new_out_neighbor(self, input_node: int) -> int | None:
        """Find an outgoing neighbor of *input_node* not yet in the output.

        BUG FIX B8: checks ``not in _i2o`` (node not yet created),
        not whether a specific edge exists in the output.

        BUG FIX B9 (operand order): For non-commutative binary ops (SUB,
        DIV, POW), only returns the neighbor if *input_node* is the first
        operand (ordered_inputs[0]). This ensures V/v creates the edge
        from the correct operand, preserving evaluation semantics through
        the round-trip. The second operand is added later via C/c.

        Returns:
            An input-DAG node ID, or ``None`` if all outgoing neighbors
            are already in the output (or none are valid for V/v).
        """
        for neighbor in self._input_dag.out_neighbors(input_node):
            if neighbor not in self._i2o:
                # For binary ops, V/v must come from the first operand.
                label = self._input_dag.node_label(neighbor)
                if label in BINARY_OPS:
                    ordered = self._input_dag.ordered_inputs(neighbor)
                    if ordered and ordered[0] != input_node:
                        continue
                return neighbor
        return None

    def _move_pointer(self, ptr: int, steps: int) -> int:
        """Walk *ptr* through the CDLL by *steps* (positive=next, negative=prev)."""
        if steps >= 0:
            for _ in range(steps):
                ptr = self._cdll.next_node(ptr)
        else:
            for _ in range(-steps):
                ptr = self._cdll.prev_node(ptr)
        return ptr

    def _emit_primary_moves(self, steps: int) -> None:
        """Append N or P instructions for primary pointer movements."""
        if steps >= 0:
            self._output_string += "N" * steps
        else:
            self._output_string += "P" * (-steps)

    def _emit_secondary_moves(self, steps: int) -> None:
        """Append n or p instructions for secondary pointer movements."""
        if steps >= 0:
            self._output_string += "n" * steps
        else:
            self._output_string += "p" * (-steps)

    def _check_reachability(self) -> None:
        """Verify all non-VAR nodes are reachable from some VAR node via outgoing edges.

        The D2S algorithm can only create edges from existing → new nodes,
        so all internal/leaf nodes must be reachable from the initial variable nodes
        through outgoing edges in the input DAG.

        Raises:
            ValueError: If unreachable nodes are detected.
        """
        n = self._input_dag.node_count
        if n <= self._num_variables:
            return  # Only VAR nodes, nothing to check.

        # BFS from all VAR nodes via outgoing edges.
        visited: set[int] = set()
        stack: list[int] = list(self._input_dag.var_nodes())
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._input_dag.out_neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(visited) != n:
            unreachable = set(range(n)) - visited
            raise ValueError(
                f"DAGToString: all nodes must be reachable from VAR nodes "
                f"via outgoing edges. Unreachable nodes: {unreachable}"
            )

    def _snapshot(
        self,
    ) -> tuple[LabeledDAG, CircularDoublyLinkedList, int, int, str]:
        """Create a deep-copied snapshot of the current state."""
        return (
            deepcopy(self._output_dag),
            deepcopy(self._cdll),
            self._primary_ptr,
            self._secondary_ptr,
            self._output_string,
        )
