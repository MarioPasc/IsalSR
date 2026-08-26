"""Custom exception hierarchy for IsalSR."""


class IsalSRError(Exception):
    """Base exception for all IsalSR errors."""


class CycleDetectedError(IsalSRError):
    """Raised when an edge would create a cycle in the DAG."""


class InvalidTokenError(IsalSRError):
    """Raised for unrecognized or disallowed instruction tokens."""


class InvalidDAGError(IsalSRError):
    """Raised when a DAG violates structural constraints."""


class EvaluationError(IsalSRError):
    """Raised during numerical evaluation failures."""
