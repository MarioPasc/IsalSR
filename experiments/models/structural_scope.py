"""Domain restriction of the canonical-string invariant to structural candidates.

Why this module exists
----------------------
The IsalSR canonical string encodes the **instruction sequence that builds the
internal nodes** of a labeled DAG.  The :math:`m` input variables are
pre-inserted into the initial state and are never created by a ``V``/``v``
instruction (``CLAUDE.md`` invariant 7), so a DAG with **zero internal nodes**
has an empty instruction sequence and canonicalises to ``""`` --- for every
:math:`m`, and for every choice of which variable is the output.

That is not a defect of the canonicaliser.  ``""`` is the faithful encoding of
*the initial state*, exactly as Definition 3.2 specifies it.  What it cannot do
is name an output: a :class:`~isalsr.core.labeled_dag.LabeledDAG` carries no
output marker, and for :math:`k \\ge 1` the output is recoverable as the unique
sink, while at :math:`k = 0` there are :math:`m` sinks and nothing distinguishes
them.  Two host candidates ``f(x) = x_0`` and ``f(x) = x_1`` therefore share the
key ``""`` while computing different functions.

Why the restriction is principled, not a workaround
---------------------------------------------------
IsalSR's claim is that the :math:`O(k!)` relabelings of the :math:`k` internal
nodes collapse to one canonical string.  At :math:`k = 0` that group is trivial
(:math:`0! = 1`): there is no redundancy to collapse, no equivalence class with
more than one member, and :math:`\\rho` --- evaluations per distinct *structural*
class --- is undefined on an object with no structure.  A bare terminal is a
well-formed SR hypothesis, and the host is right to score it; it is simply not
an object our invariant is about.

So the canonical string is a complete labeled-DAG invariant **for** :math:`k \\ge 1`,
and candidates with :math:`k = 0` are passed through: evaluated normally by the
host, never deduplicated, never entered into a fitness cache, and excluded from
the :math:`\\rho` accounting so they neither inflate nor deflate it.

Measured incidence (Stage D, Pagie-1, Bingo/isalsr, seed 102, 119,795 sampled
candidates): **0.0593 %** at :math:`k = 0`, i.e. ~7,100 in the full stream.  UDFS
cannot reach :math:`k = 0` at all --- all seven campaign configs pin
``n_calc_nodes = 5`` --- so this is a Bingo-only effect in practice.

Discovered by the T04 Mode 1 replay (EXECUTION-PLAN §4.4 D3), 2026-08-06, before
the campaign committed any core-hours.  That is what D3 is for.
"""

from __future__ import annotations

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

__all__ = ["STRUCTURAL_SCOPE_REASON", "count_internal_nodes", "is_structural"]

#: Recorded on the trace record of every skipped candidate, so the exclusion is
#: visible in the stream rather than inferred from a counter that did not move.
STRUCTURAL_SCOPE_REASON = "k0_nonstructural"


def count_internal_nodes(dag: LabeledDAG) -> int:
    """Return the number of non-``VAR`` nodes, i.e. :math:`k`.

    ``CONST`` counts as internal: it is created by the ``Vk`` instruction and is
    therefore encoded, which is why a bare constant canonicalises to ``"Vk"``
    and is perfectly distinguishable.

    Args:
        dag: The labeled DAG to measure.

    Returns:
        The count of nodes whose label is not :attr:`NodeType.VAR`.
    """
    return sum(1 for node in range(dag.node_count) if dag.node_label(node) is not NodeType.VAR)


def is_structural(dag: LabeledDAG) -> bool:
    """Return whether *dag* lies in the domain of the canonical-string invariant.

    A DAG is structural iff it carries at least one internal node. Non-structural
    DAGs (bare input variables) all canonicalise to ``""`` regardless of which
    variable they return, so their canonical string must never be used as a
    deduplication or fitness-cache key.

    Args:
        dag: The labeled DAG to test.

    Returns:
        ``True`` when :math:`k \\ge 1`, ``False`` for a bare-variable candidate.
    """
    return count_internal_nodes(dag) > 0
