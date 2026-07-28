# IsalSR

**Instruction Set and Language for Symbolic Regression**

IsalSR represents symbolic regression expressions as labeled DAGs encoded in
isomorphism-invariant instruction strings. The canonical string representation
collapses O(k!) equivalent expression representations into one, reducing the
search space for symbolic regression by factorial factors.

## Authors

- Ezequiel Lopez-Rubio (University of Malaga)
- Mario Pascual Gonzalez (University of Malaga)
- Karl Khader Thurnhofer-Hemsi (University of Malaga)

## Installation

```bash
conda activate isalsr
pip install -e ".[dev]"          # add ".[experiments]" to run the SR campaigns
```

### The C++ engine

The core is implemented twice: a pure-Python reference and a C++ extension built
by scikit-build-core during `pip install`. The two are kept numerically equivalent,
and the Python engine is a complete fallback — so **a failed compile degrades
silently instead of raising**. Check which one is live before trusting a benchmark:

```bash
python -c "from isalsr.core.backends import build_info; print(build_info())"
# {'engine': 'cpp', 'isa_level': 'x86-64-v3', 'compiler': 'gcc 12.2.0', ...}
```

`engine` is `"cpp"` when the extension is active and `"python"` otherwise. The
test suite is the other tell: **4,436 passed / 5 skipped** with the extension,
**1,188 passed / 30 skipped** without it, the difference being the cross-engine
tests that skip with "C++ extension not built".

Building requires **GCC ≥ 11** — `CMakeLists.txt` targets `-march=x86-64-v3`, an
architecture value that did not exist before GCC 11. Set `ISALSR_NATIVE_MARCH=ON`
to use `-march=native` instead when the binary does not need to be portable.

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
