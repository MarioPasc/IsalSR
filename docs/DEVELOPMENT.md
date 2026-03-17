# IsalSR Development Guide

## Setup

```bash
conda activate isalsr
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/unit/ -v` | Unit tests (fast, no external deps) |
| `python -m pytest tests/integration/ -v` | Integration tests |
| `python -m pytest tests/property/ -v` | Property-based tests (hypothesis) |
| `python -m pytest tests/ -v --cov=isalsr` | Full suite with coverage |
| `python -m ruff check --fix src/ tests/` | Lint + autofix |
| `python -m ruff format src/ tests/` | Format |
| `python -m mypy src/isalsr/` | Type checking (strict) |

## Dependency Rules

- `isalsr.core`: ZERO external deps (stdlib only)
- `isalsr.adapters`: optional (networkx, sympy)
- `isalsr.evaluation`: numpy, scipy
- `isalsr.search`: numpy
- `experiments/`, `benchmarks/`: anything

## Testing Strategy

1. **Unit tests**: Fast, no external deps. Cover all core functionality.
2. **Integration tests**: Test adapter bridges and evaluation pipeline.
3. **Property tests**: Hypothesis-based. Round-trip, acyclicity, canonical invariance.
4. **Benchmark tests**: Full pipeline on standard SR benchmarks.

## Git Workflow

- Feature branches from `main`
- All tests must pass before merge
- Ruff + mypy clean
