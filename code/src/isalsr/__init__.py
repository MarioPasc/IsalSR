"""IsalSR -- Instruction Set and Language for Symbolic Regression.

Represents symbolic regression expressions as labeled DAGs encoded in
isomorphism-invariant instruction strings. The canonical string representation
collapses O(k!) equivalent expression representations into one, reducing the
search space for symbolic regression by factorial factors.

The names below are the supported public API; everything else is reachable
through its submodule (``isalsr.core.<module>``).

Citation
--------
E. Lopez-Rubio, M. Pascual-Gonzalez and K. Thurnhofer-Hemsi, "Representation of
Directed Acyclic Graphs by Sequences of Instructions for Symbolic Regression",
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

Authors:
    Ezequiel Lopez-Rubio (ezeqlr@lcc.uma.es)
    Mario Pascual Gonzalez (mpascual@uma.es)
    Karl Thurnhofer-Hemsi (karlkhader@uma.es)

University of Malaga, 2025-2026.
"""

__version__ = "1.0.0"

from isalsr.core import (
    CanonicalTimeoutError,
    DAGToString,
    LabeledDAG,
    NodeType,
    OperationSet,
    StringToDAG,
    build_info,
    canonical_string,
    dag_distance,
    engine,
    evaluate_dag,
    fast_canonical_string,
    from_commutative,
    levenshtein,
    permute_internal_nodes,
    to_commutative,
)
from isalsr.errors import EvaluationError, InvalidTokenError, IsalSRError

__all__ = [
    "__version__",
    "CanonicalTimeoutError",
    "DAGToString",
    "EvaluationError",
    "InvalidTokenError",
    "IsalSRError",
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
    "to_commutative",
]
