"""IsalSR core module -- ZERO external dependencies.

Re-exports the public core API: the labeled DAG and its node types, the two
string converters (S2D, D2S), the canonical-string functions, the evaluator,
the commutative rewriting, isomorphic relabeling, and the engine probe.

Restriction: This module and ALL submodules must have ZERO external dependencies.
Only Python stdlib and typing are allowed.
"""

from isalsr.core.backends import build_info, engine
from isalsr.core.canonical import (
    CanonicalTimeoutError,
    canonical_string,
    dag_distance,
    fast_canonical_string,
    levenshtein,
    pruned_canonical_string,
)
from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.commutative import from_commutative, to_commutative
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType, OperationSet
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

__all__ = [
    "CanonicalTimeoutError",
    "CircularDoublyLinkedList",
    "DAGToString",
    "LabeledDAG",
    "NodeType",
    "OperationSet",
    "StringToDAG",
    "build_info",
    "canonical_string",
    "dag_distance",
    "engine",
    "evaluate_dag",
    "fast_canonical_string",
    "from_commutative",
    "levenshtein",
    "permute_internal_nodes",
    "pruned_canonical_string",
    "to_commutative",
]
