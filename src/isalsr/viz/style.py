"""Visual style constants for IsalSR DAG and instruction-strip visualisation.

Single source of truth for node face/edge colours and token-cell colours.
The node colours match those used in ``experiments/scripts/generate_fig_const_normalization.py``
so all review figures read as one visual system.

Dependency rule: this module imports only :mod:`isalsr.core`. matplotlib is
imported inside function bodies.
"""

from __future__ import annotations

from isalsr.core.node_types import NodeType

# ---------------------------------------------------------------------------
# Node colours (face and border).  Must match generate_fig_const_normalization.
# ---------------------------------------------------------------------------

#: Face colours keyed by :class:`~isalsr.core.node_types.NodeType`.
NODETYPE_FACE: dict[NodeType, str] = {
    NodeType.VAR: "#cfe3f7",
    NodeType.CONST: "#fbe3c2",
    # All operator nodes share the same light-grey face.
    NodeType.ADD: "#e8e8e8",
    NodeType.MUL: "#e8e8e8",
    NodeType.SUB: "#e8e8e8",
    NodeType.DIV: "#e8e8e8",
    NodeType.SIN: "#e8e8e8",
    NodeType.COS: "#e8e8e8",
    NodeType.EXP: "#e8e8e8",
    NodeType.LOG: "#e8e8e8",
    NodeType.SQRT: "#e8e8e8",
    NodeType.POW: "#e8e8e8",
    NodeType.ABS: "#e8e8e8",
    NodeType.NEG: "#e8e8e8",
    NodeType.INV: "#e8e8e8",
}

#: Border colours keyed by :class:`~isalsr.core.node_types.NodeType`.
NODETYPE_BORDER: dict[NodeType, str] = {
    NodeType.VAR: "#2c6fad",
    NodeType.CONST: "#c17c26",
    NodeType.ADD: "#6b6b6b",
    NodeType.MUL: "#6b6b6b",
    NodeType.SUB: "#6b6b6b",
    NodeType.DIV: "#6b6b6b",
    NodeType.SIN: "#6b6b6b",
    NodeType.COS: "#6b6b6b",
    NodeType.EXP: "#6b6b6b",
    NodeType.LOG: "#6b6b6b",
    NodeType.SQRT: "#6b6b6b",
    NodeType.POW: "#6b6b6b",
    NodeType.ABS: "#6b6b6b",
    NodeType.NEG: "#6b6b6b",
    NodeType.INV: "#6b6b6b",
}

#: Accent applied to nodes marked as reachable from a variable.
REACHABLE_TINT: str = "#1b7f4b"

#: Arrow colour for ordinary edges.
ARROW_COLOR: str = "#4a4a4a"

#: Arrow colour for creation edges (x_i -> CONST).
CREATION_EDGE_COLOR: str = "#1b7f4b"

#: Alert colour for violating nodes (in-degree 0 CONST).
ALERT_COLOR: str = "#b02a2a"

# ---------------------------------------------------------------------------
# Token-strip colours.
# ---------------------------------------------------------------------------

#: Face colours for single-character instruction tokens.
SINGLE_TOKEN_FACE: dict[str, str] = {
    "N": "#b8d4e8",  # primary move forward  (muted blue)
    "P": "#b8d4e8",  # primary move backward (muted blue)
    "n": "#c8e8c8",  # secondary move forward  (muted green)
    "p": "#c8e8c8",  # secondary move backward (muted green)
    "C": "#f4c090",  # edge primary→secondary (muted orange)
    "c": "#f0b0b0",  # edge secondary→primary (muted red)
    "W": "#dddddd",  # no-op (grey)
}

#: Fallback face colour for tokens not in any palette.
GRAYED_FACE: str = "#dddddd"


def color_for_node_face(node_type: NodeType) -> str:
    """Return the face colour hex string for a node of ``node_type``.

    Parameters
    ----------
    node_type:
        The :class:`~isalsr.core.node_types.NodeType` of the node.

    Returns
    -------
    str
        Hex colour string.
    """
    return NODETYPE_FACE.get(node_type, GRAYED_FACE)


def color_for_node_border(node_type: NodeType) -> str:
    """Return the border colour hex string for a node of ``node_type``.

    Parameters
    ----------
    node_type:
        The :class:`~isalsr.core.node_types.NodeType` of the node.

    Returns
    -------
    str
        Hex colour string.
    """
    return NODETYPE_BORDER.get(node_type, "#6b6b6b")


def _label_char_to_nodetype(label: str) -> NodeType | None:
    """Return the :class:`NodeType` for a V/v label character, or ``None``."""
    for nt in NodeType:
        if nt.value == label:
            return nt
    return None


def color_for_token(token: str) -> str:
    """Return the face colour for one token cell in the instruction strip.

    Single-character tokens (``N``, ``P``, ``n``, ``p``, ``C``, ``c``, ``W``)
    are coloured by the static :data:`SINGLE_TOKEN_FACE` palette.  Two-character
    tokens (``V<label>``, ``v<label>``) are coloured by the face colour of the
    :class:`NodeType` they insert, so the strip and the drawn nodes share a
    mechanical colour correspondence.

    Parameters
    ----------
    token:
        Raw token string such as ``"N"``, ``"Vs"``, or ``"vk"``.

    Returns
    -------
    str
        Hex colour string.
    """
    if len(token) == 2 and token[0] in ("V", "v"):
        nt = _label_char_to_nodetype(token[1])
        if nt is not None:
            return NODETYPE_FACE.get(nt, GRAYED_FACE)
        return GRAYED_FACE
    return SINGLE_TOKEN_FACE.get(token, GRAYED_FACE)
