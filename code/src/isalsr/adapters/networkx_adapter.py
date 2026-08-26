"""NetworkX DiGraph <-> LabeledDAG adapter.

Optional dependency: networkx >= 3.0
"""

from __future__ import annotations

import networkx as nx

from isalsr.adapters.base import DAGAdapter
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


class NetworkXAdapter(DAGAdapter["nx.DiGraph"]):
    """Convert between NetworkX DiGraphs and LabeledDAGs.

    Node attributes in nx.DiGraph:
        - 'label' (str): NodeType name (e.g., 'VAR', 'ADD', 'SIN').
        - 'var_index' (int, optional): For VAR nodes.
        - 'const_value' (float, optional): For CONST nodes.
    """

    def from_external(self, obj: nx.DiGraph) -> LabeledDAG:
        """Convert nx.DiGraph to LabeledDAG.

        Nodes must have a 'label' attribute containing a NodeType name.
        Node IDs in nx.DiGraph are mapped to contiguous integers.
        """
        node_list = sorted(obj.nodes())
        n = len(node_list)
        ext_to_int: dict[object, int] = {ext: i for i, ext in enumerate(node_list)}

        dag = LabeledDAG(max_nodes=n)
        for ext_node in node_list:
            attrs = obj.nodes[ext_node]
            label_str = attrs.get("label", "VAR")
            label = NodeType[label_str] if isinstance(label_str, str) else label_str

            var_index = attrs.get("var_index")
            const_value = attrs.get("const_value")

            dag.add_node(
                label,
                var_index=int(var_index) if var_index is not None else None,
                const_value=float(const_value) if const_value is not None else None,
            )

        for u, v in obj.edges():
            dag.add_edge(ext_to_int[u], ext_to_int[v])

        return dag

    def to_external(self, dag: LabeledDAG) -> nx.DiGraph:
        """Convert LabeledDAG to nx.DiGraph with node attributes."""
        g = nx.DiGraph()

        for i in range(dag.node_count):
            attrs: dict[str, object] = {"label": dag.node_label(i).name}
            data = dag.node_data(i)
            if "var_index" in data:
                attrs["var_index"] = data["var_index"]
            if "const_value" in data:
                attrs["const_value"] = data["const_value"]
            g.add_node(i, **attrs)

        for i in range(dag.node_count):
            for j in dag.out_neighbors(i):
                g.add_edge(i, j)

        return g
