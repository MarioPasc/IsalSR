"""Certification gate G9: prove a real run canonicalises the paper's alphabet.

Ticket T16 aligned the implementation to Definition 3.2 of the manuscript, whose
instruction set has **twelve** label characters and no ``-`` and no ``/``:
subtraction and division enter through the commutative decomposition
``x - y = Add(x, Neg(y))`` and ``x / y = Mul(x, Inv(y))``.  ``Pow`` is the only
non-commutative operation left, which is what makes Definition 3.9(iv) sound.

Unit tests prove the adapters *can* decompose.  They do not prove that the code a
SLURM array actually executes *does*, because the production path runs through the
orchestrator, the runner, the monkey-patched host evaluation hook and the
deduplicator before a DAG ever reaches the canonicaliser.  This script instruments
that real path: it wraps both adapters, runs the genuine orchestrator on a genuine
production config, and reports the label histogram of every DAG that was handed to
the canonicaliser.

Gate G9 passes for a (method, config) pair when the observed stream contains

* zero ``NodeType.SUB`` and zero ``NodeType.DIV`` nodes,
* zero ``-`` and zero ``/`` characters in any canonical string,
* ``POW`` as the only order-sensitive binary operation present, and
* a non-zero number of observed DAGs (an empty stream proves nothing).

Run before every Wave-1 style launch, and again on the ``--array=1-1`` single task
on the compute node.  See ``.claude/notes/review/tasks/EXECUTION-PLAN.md`` section 2.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NodeType

log = logging.getLogger("verify_alphabet_gate")

#: Labels that must never appear in a decomposed stream.
FORBIDDEN_LABELS: frozenset[NodeType] = frozenset({NodeType.SUB, NodeType.DIV})

#: Label characters that must never appear in a canonical string.
FORBIDDEN_CHARS: tuple[str, ...] = ("-", "/")


class AlphabetObserver:
    """Accumulates the label histogram of every DAG produced by an adapter.

    Attributes:
        n_dags: Number of DAGs observed.
        labels: Histogram of ``NodeType`` name -> count over all observed DAGs.
        dags_with_forbidden: Number of DAGs carrying a forbidden label.
        max_k: Largest internal-node count observed.
    """

    __slots__ = ("n_dags", "labels", "dags_with_forbidden", "max_k", "_samples", "_keep")

    def __init__(self, keep_samples: int = 3) -> None:
        self.n_dags: int = 0
        self.labels: collections.Counter[str] = collections.Counter()
        self.dags_with_forbidden: int = 0
        self.max_k: int = 0
        self._samples: list[str] = []
        self._keep: int = keep_samples

    def observe(self, dag: LabeledDAG) -> None:
        """Record one DAG's label multiset."""
        self.n_dags += 1
        found_forbidden = False
        k = 0
        for i in range(dag.node_count):
            label = dag.node_label(i)
            if label is None:
                continue
            self.labels[label.name] += 1
            if label not in (NodeType.VAR,):
                k += 1
            if label in FORBIDDEN_LABELS:
                found_forbidden = True
        if found_forbidden:
            self.dags_with_forbidden += 1
            if len(self._samples) < self._keep:
                self._samples.append(repr(dag))
        self.max_k = max(self.max_k, k)

    @property
    def forbidden_samples(self) -> list[str]:
        """Return up to ``keep_samples`` reprs of offending DAGs."""
        return list(self._samples)

    def binary_ops_present(self) -> list[str]:
        """Return the order-sensitive binary operations seen, sorted."""
        return sorted(n for n in self.labels if NodeType[n] in BINARY_OPS)


def _install_adapter_probes(observer: AlphabetObserver) -> None:
    """Wrap both host adapters so every produced DAG is observed.

    Patches the adapter modules *and* the names already imported into the runner
    modules, because ``from ... import agraph_to_labeled_dag`` binds a separate
    reference that a module-level patch would not reach.
    """
    import experiments.models.bingo.adapter as b_ad
    import experiments.models.udfs.adapter as u_ad

    def wrap(fn: Any) -> Any:
        def probed(*args: Any, **kwargs: Any) -> LabeledDAG:
            dag = fn(*args, **kwargs)
            observer.observe(dag)
            return dag

        probed.__name__ = getattr(fn, "__name__", "probed")
        return probed

    b_probe = wrap(b_ad.agraph_to_labeled_dag)
    u_probe = wrap(u_ad.compgraph_to_labeled_dag)
    b_ad.agraph_to_labeled_dag = b_probe
    u_ad.compgraph_to_labeled_dag = u_probe

    # Rebind in every module that imported the symbol directly.
    for mod_name, attr, probe in (
        ("experiments.models.bingo.isalsr_runner", "agraph_to_labeled_dag", b_probe),
        ("experiments.models.bingo.translator", "agraph_to_labeled_dag", b_probe),
        ("experiments.models.udfs.isalsr_runner", "compgraph_to_labeled_dag", u_probe),
    ):
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except ImportError:  # pragma: no cover - optional host missing
            continue
        if hasattr(mod, attr):
            setattr(mod, attr, probe)


def _install_canonical_probe(strings: list[str]) -> None:
    """Record every canonical string the deduplicators compute."""
    import isalsr.core.canonical as canon

    original = canon.fast_canonical_string

    def probed(*args: Any, **kwargs: Any) -> str:
        result = original(*args, **kwargs)
        strings.append(result)
        return result

    canon.fast_canonical_string = probed
    for mod_name in (
        "experiments.models.bingo.isalsr_runner",
        "experiments.models.udfs.isalsr_runner",
    ):
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except ImportError:  # pragma: no cover
            continue
        if hasattr(mod, "fast_canonical_string"):
            mod.fast_canonical_string = probed


def run_gate(
    config: Path,
    problems: str,
    seeds: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the real orchestrator under instrumentation and evaluate gate G9.

    Args:
        config: Production YAML config, used verbatim.
        problems: Problem filter passed to the orchestrator.
        seeds: Seed specification passed to the orchestrator.
        output_dir: Scratch directory for run artefacts.

    Returns:
        A dict with the observed histogram and the pass/fail verdict.
    """
    observer = AlphabetObserver()
    canonical_strings: list[str] = []
    _install_adapter_probes(observer)
    _install_canonical_probe(canonical_strings)

    from experiments.models import orchestrator

    argv = [
        "orchestrator",
        "--config",
        str(config),
        "--problems",
        problems,
        "--seeds",
        seeds,
        "--variants",
        "isalsr",
        "--output-dir",
        str(output_dir),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        orchestrator.main()
    finally:
        sys.argv = old_argv

    bad_strings = [s for s in canonical_strings if any(c in s for c in FORBIDDEN_CHARS)]
    binary_present = observer.binary_ops_present()

    checks = {
        "observed_dags_nonzero": observer.n_dags > 0,
        "no_forbidden_labels": observer.dags_with_forbidden == 0,
        "no_forbidden_chars_in_canonical": len(bad_strings) == 0,
        "only_pow_is_binary": set(binary_present) <= {"POW"},
    }
    return {
        "config": str(config),
        "problems": problems,
        "n_dags_observed": observer.n_dags,
        "n_canonical_strings": len(canonical_strings),
        "label_histogram": dict(observer.labels.most_common()),
        "order_sensitive_binary_ops_present": binary_present,
        "n_dags_with_forbidden_label": observer.dags_with_forbidden,
        "forbidden_samples": observer.forbidden_samples,
        "n_canonical_with_forbidden_char": len(bad_strings),
        "example_bad_strings": bad_strings[:3],
        "max_k": observer.max_k,
        "checks": checks,
        "PASS": all(checks.values()),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="T16 alphabet certification gate (G9)")
    parser.add_argument("--config", required=True, type=Path, help="Production YAML config")
    parser.add_argument("--problems", default="Nguyen-1", help="Problem filter")
    parser.add_argument("--seeds", default="1", help="Seed specification")
    parser.add_argument("--output-dir", type=Path, default=None, help="Scratch dir")
    parser.add_argument("--json-out", type=Path, default=None, help="Write verdict JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="g9_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_gate(args.config, args.problems, args.seeds, out_dir)

    print("\n" + "=" * 70)
    print(f"G9 ALPHABET GATE  config={args.config.name}  problems={args.problems}")
    print("=" * 70)
    print(f"DAGs observed at the canonicaliser : {result['n_dags_observed']}")
    print(f"Canonical strings computed         : {result['n_canonical_strings']}")
    binary_present = result["order_sensitive_binary_ops_present"] or "NONE"
    print(f"Order-sensitive binary ops present : {binary_present}")
    print(f"Max k observed                     : {result['max_k']}")
    print(f"Label histogram                    : {result['label_histogram']}")
    for name, ok in result["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)
    print("VERDICT:", "PASS" if result["PASS"] else "FAIL")
    print("=" * 70)

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))

    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
