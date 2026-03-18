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

## Experimental Framework

### Setup

```bash
pip install -e ".[experiments]"  # includes bingo-nasa, stopit, statsmodels, etc.
```

### Running Experiments

```bash
# UDFS on Nguyen benchmarks (all problems, seeds 1-30)
python -m experiments.models.orchestrator \
    --config experiments/configs/udfs_nguyen.yaml \
    --output-dir /media/mpascual/Sandisk2TB/research/isalsr/results \
    --seeds 1-30 --variants baseline,isalsr

# Bingo on Nguyen benchmarks
python -m experiments.models.orchestrator \
    --config experiments/configs/bingo_nguyen.yaml \
    --output-dir /media/mpascual/Sandisk2TB/research/isalsr/results \
    --seeds 1-30 --variants baseline,isalsr

# Single problem, single seed (smoke test)
python -m experiments.models.orchestrator \
    --config experiments/configs/udfs_nguyen.yaml \
    --seeds 1 --problems Nguyen-1 --variants baseline,isalsr \
    --output-dir /tmp/smoke_test
```

### Adding a New Comparison Model

1. Create `experiments/models/<method>/` with: `adapter.py`, `config.py`, `runner.py`,
   `isalsr_runner.py`, `translator.py`
2. Register in `orchestrator.py`: add branches in `create_runner()` and `create_translator()`
3. Create config YAML in `experiments/configs/`
4. Write tests in `tests/unit/` and `tests/integration/`

### Architecture

```
ModelRunner.fit() → RawRunResult → ResultTranslator.to_run_log() → RunLog (unified)
                                                                  → analyzer/ (model-agnostic stats)
```

Two interception patterns for IsalSR canonical deduplication:
- **Monkey-patch** (UDFS): patch module-level eval function via context manager
- **Subclass** (Bingo): override `Evaluation._serial_eval()` (Bingo's component-swap design)

## Git Workflow

- Feature branches from `main`
- All tests must pass before merge
- Ruff + mypy clean
