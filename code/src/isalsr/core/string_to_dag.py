"""StringToDAG (S2D) -- Execute an IsalSR instruction string to produce a LabeledDAG.

Tokenizes the input string into IsalSR tokens and executes them sequentially,
building a LabeledDAG. Handles the two-tier encoding: single-char tokens for
movement/edge/no-op, and two-char compound tokens for labeled node insertion.

Reference: IsalGraph's StringToGraph at
    /home/mpascual/research/code/IsalGraph/src/isalgraph/core/string_to_graph.py

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import LABEL_CHAR_MAP, NodeType, OperationSet
from isalsr.errors import InvalidTokenError
from isalsr.types import VALID_LABEL_CHARS, VALID_SINGLE_INSTRUCTIONS, InstructionToken

log = logging.getLogger(__name__)

# Trace entry: (dag_snapshot, cdll_snapshot, primary_ptr, secondary_ptr, tokens_so_far)
TraceEntry = tuple[LabeledDAG, CircularDoublyLinkedList, int, int, list[InstructionToken]]


class StringToDAG:
    """Convert an IsalSR instruction string into a ``LabeledDAG``.

    Args:
        input_string: The instruction string using the IsalSR two-tier encoding.
        num_variables: Number of input variables (m). Creates m VAR nodes as initial state.
        allowed_ops: If provided, restricts which operations V/v can create.
            Tokens for disallowed operations raise ``InvalidTokenError``.

    Raises:
        InvalidTokenError: If the string contains invalid tokens or disallowed operations.
        ValueError: If ``num_variables`` < 1.
    """

    __slots__ = (
        "_input_string",
        "_num_variables",
        "_allowed_ops",
        "_tokens",
        "_max_nodes",
        "_output_dag",
        "_cdll",
        "_primary_ptr",
        "_secondary_ptr",
        "_trace_log",
    )

    def __init__(
        self,
        input_string: str,
        num_variables: int,
        allowed_ops: OperationSet | None = None,
    ) -> None:
        if num_variables < 1:
            raise ValueError(f"num_variables must be >= 1, got {num_variables}")

        self._input_string: str = input_string
        self._num_variables: int = num_variables
        self._allowed_ops: OperationSet | None = allowed_ops

        # Tokenize and validate.
        self._tokens: list[InstructionToken] = _tokenize(input_string, allowed_ops)

        # Pre-compute capacity: m variables + count of V/v insertion tokens.
        insertion_count = sum(1 for t in self._tokens if len(t) == 2)
        self._max_nodes: int = num_variables + insertion_count

        # Pre-allocate data structures.
        self._output_dag: LabeledDAG = LabeledDAG(self._max_nodes)
        self._cdll: CircularDoublyLinkedList = CircularDoublyLinkedList(self._max_nodes)
        self._primary_ptr: int = -1
        self._secondary_ptr: int = -1
        self._trace_log: list[TraceEntry] = []

    # ------------------------------------------------------------------
    # Public accessors (useful for trace / debugging)
    # ------------------------------------------------------------------

    @property
    def tokens(self) -> list[InstructionToken]:
        """The tokenized instruction list."""
        return list(self._tokens)

    @property
    def cdll(self) -> CircularDoublyLinkedList:
        """The CDLL after (or during) conversion."""
        return self._cdll

    @property
    def primary_ptr(self) -> int:
        """Current primary pointer (CDLL node index)."""
        return self._primary_ptr

    @property
    def secondary_ptr(self) -> int:
        """Current secondary pointer (CDLL node index)."""
        return self._secondary_ptr

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def run(self, *, trace: bool = False) -> LabeledDAG:
        """Execute the string-to-DAG conversion.

        Args:
            trace: If ``True``, collect deep-copied snapshots after each token.
                Snapshots are stored in ``self.trace_log``.

        Returns:
            The resulting LabeledDAG.
        """
        # ---- Initial state: m VAR nodes, both pointers on x_1 ----
        self._initialize_variables()

        trace_log: list[TraceEntry] = []
        if trace:
            trace_log.append(self._snapshot([]))

        # ---- Process each token ----
        for idx, token in enumerate(self._tokens):
            self._execute_token(token)
            if trace:
                trace_log.append(self._snapshot(self._tokens[: idx + 1]))

        # Store trace for external access.
        self._trace_log = trace_log
        return self._output_dag

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_variables(self) -> None:
        """Create m VAR nodes and insert them into the CDLL in order."""
        prev_cdll_node: int = -1
        first_cdll_node: int = -1

        for i in range(self._num_variables):
            graph_node = self._output_dag.add_node(NodeType.VAR, var_index=i)
            cdll_node = self._cdll.insert_after(prev_cdll_node, graph_node)
            if i == 0:
                first_cdll_node = cdll_node
            prev_cdll_node = cdll_node

        # Both pointers on the CDLL node for x_1 (first variable).
        self._primary_ptr = first_cdll_node
        self._secondary_ptr = first_cdll_node

    # ------------------------------------------------------------------
    # Instruction dispatch
    # ------------------------------------------------------------------

    def _execute_token(self, token: InstructionToken) -> None:
        """Execute a single IsalSR token, mutating internal state."""
        if token == "N":
            self._primary_ptr = self._cdll.next_node(self._primary_ptr)

        elif token == "P":
            self._primary_ptr = self._cdll.prev_node(self._primary_ptr)

        elif token == "n":
            self._secondary_ptr = self._cdll.next_node(self._secondary_ptr)

        elif token == "p":
            self._secondary_ptr = self._cdll.prev_node(self._secondary_ptr)

        elif token == "C":
            # Edge: primary -> secondary (with DAG cycle check).
            src = self._cdll.get_value(self._primary_ptr)
            tgt = self._cdll.get_value(self._secondary_ptr)
            # add_edge returns False if cycle/duplicate; silently skip.
            self._output_dag.add_edge(src, tgt)

        elif token == "c":
            # Edge: secondary -> primary (with DAG cycle check).
            src = self._cdll.get_value(self._secondary_ptr)
            tgt = self._cdll.get_value(self._primary_ptr)
            self._output_dag.add_edge(src, tgt)

        elif token == "W":
            pass  # No-op.

        elif len(token) == 2 and token[0] in "Vv":
            self._execute_insertion(token)

        else:
            # Should not reach here if tokenizer is correct, but be defensive.
            raise InvalidTokenError(f"Unknown token: {token!r}")

    def _execute_insertion(self, token: InstructionToken) -> None:
        """Execute a V[label] or v[label] insertion token."""
        pointer_char = token[0]
        label_char = token[1]
        node_type = LABEL_CHAR_MAP[label_char]

        # Create the new node.
        if node_type == NodeType.CONST:
            new_node = self._output_dag.add_node(node_type, const_value=1.0)
        else:
            new_node = self._output_dag.add_node(node_type)

        if pointer_char == "V":
            # Edge from primary's graph node to new node.
            primary_graph_node = self._cdll.get_value(self._primary_ptr)
            self._output_dag.add_edge(primary_graph_node, new_node)
            # Insert new node into CDLL after primary. Pointer does NOT move.
            self._cdll.insert_after(self._primary_ptr, new_node)
        else:
            # pointer_char == "v": same using secondary pointer.
            secondary_graph_node = self._cdll.get_value(self._secondary_ptr)
            self._output_dag.add_edge(secondary_graph_node, new_node)
            self._cdll.insert_after(self._secondary_ptr, new_node)

    # ------------------------------------------------------------------
    # Trace support
    # ------------------------------------------------------------------

    def _snapshot(self, tokens_so_far: list[InstructionToken]) -> TraceEntry:
        """Create a deep-copied snapshot of the current state."""
        return (
            deepcopy(self._output_dag),
            deepcopy(self._cdll),
            self._primary_ptr,
            self._secondary_ptr,
            list(tokens_so_far),
        )


# ======================================================================
# Module-level tokenizer
# ======================================================================


def _tokenize(
    input_string: str,
    allowed_ops: OperationSet | None = None,
) -> list[InstructionToken]:
    """Tokenize an IsalSR instruction string into a list of tokens.

    Tokenizer rules:
        - If current char is 'V' or 'v', consume the next char as the label.
          The compound token is the two chars together (e.g., 'V+', 'vs').
        - If 'V'/'v' is at end of string with no next char, raise InvalidTokenError.
        - Otherwise, the single char is a token (N, P, n, p, C, c, W).
        - Unknown characters raise InvalidTokenError.
        - If allowed_ops is set, validate label chars against allowed_ops.label_chars.

    Args:
        input_string: The raw IsalSR instruction string.
        allowed_ops: If provided, restricts which label chars are valid.

    Returns:
        List of InstructionToken (single-char or two-char strings).

    Raises:
        InvalidTokenError: On invalid characters, truncated V/v, or disallowed ops.
    """
    tokens: list[InstructionToken] = []
    i = 0
    n = len(input_string)

    while i < n:
        ch = input_string[i]

        if ch in "Vv":
            # Compound token: consume next char as label.
            if i + 1 >= n:
                raise InvalidTokenError(
                    f"'{ch}' at position {i} requires a label character, but the string ends"
                )
            label = input_string[i + 1]
            if label not in VALID_LABEL_CHARS:
                raise InvalidTokenError(
                    f"Invalid label character '{label}' at position {i + 1} "
                    f"after '{ch}'. Valid labels: {sorted(VALID_LABEL_CHARS)}"
                )
            if allowed_ops is not None and label not in allowed_ops.label_chars:
                raise InvalidTokenError(
                    f"Operation '{label}' (token '{ch}{label}') is not in the allowed operation set"
                )
            tokens.append(ch + label)
            i += 2

        elif ch in VALID_SINGLE_INSTRUCTIONS:
            tokens.append(ch)
            i += 1

        else:
            raise InvalidTokenError(
                f"Invalid character '{ch}' at position {i}. "
                f"Expected one of {sorted(VALID_SINGLE_INSTRUCTIONS)} or V/v"
            )

    return tokens
