# IsalSR

**Instruction Set and Language for Symbolic Regression**

IsalSR represents symbolic regression expressions as labeled DAGs encoded in
isomorphism-invariant instruction strings. The canonical string representation
collapses O(k!) equivalent expression representations into one, reducing the
search space for symbolic regression by factorial factors.

## Authors

- Ezequiel Lopez-Rubio (University of Malaga)
- Mario Pascual Gonzalez (University of Malaga)

## Installation

```bash
conda activate isalsr
pip install -e ".[dev]"
```

## Quick Start

```python
from isalsr.core.string_to_dag import StringToDAG
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.canonical import canonical_string

# Decode: instruction string -> expression DAG
s2d = StringToDAG("V+NnncVs", num_variables=2)
dag = s2d.run()

# Encode: expression DAG -> instruction string
d2s = DAGToString(dag)
string = d2s.run()

# Canonical: isomorphism-invariant representation
canon = canonical_string(dag)
```

## References

- Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
- Liu et al. (2025). Neural Networks 187:107405. GraphDSR.

## License

MIT
