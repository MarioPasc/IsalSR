# Reproducing IsalSR

This repository doubles as a [Code Ocean](https://codeocean.com) compute capsule.
Two top-level folders and one script exist only for that purpose and are inert
everywhere else:

| Path | Contents |
|---|---|
| `environment/` | `Dockerfile`, `postInstall`, `requirements.txt` — the capsule image |
| `metadata/` | `metadata.yml` — capsule title, description, authors |
| `run` | the capsule's master script |

## Import from the `codeocean` branch, not from `main`

Code Ocean's git import recognises exactly four directory names — `code`,
`data`, `environment`, `metadata` — and files their *contents* into the matching
capsule directories. Anything outside them is neither mounted during a
Reproducible Run nor offered as a core file, so the whole project has to sit
inside `code/`.

Putting it there on `main` would move `pyproject.toml` out of the repository
root and break `pip install git+…`, the README paths and the docs site. So
`main` keeps the normal layout and the capsule layout is *derived* onto a
separate branch:

```
main                        codeocean
  pyproject.toml     ->       code/pyproject.toml
  src/               ->       code/src/
  tests/             ->       code/tests/
  run                ->       code/run
  environment/       ->       environment/     (unchanged)
  metadata/          ->       metadata/        (unchanged)
```

Regenerate it after any change to `main`:

```bash
bash tools/make_codeocean_branch.sh
git push -f origin codeocean
```

The branch is derived, never edited by hand — it is rebuilt from scratch each
time, so there is nothing to merge. Every change belongs on `main`.

## What the capsule reproduces

The verification suite, not the benchmark campaign.

`run` builds the C++ engine from source, then runs the full test suite
**twice** — once on the C++ engine and once on the pure-Python reference
implementation — plus `ruff` and `mypy --strict`. Running both engines is the
point: every reported number was produced with the C++ engine, and the claim
that the two are semantically interchangeable is only evidence if both are
exercised.

Measured on the capsule image built from
`registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04`
(24-core x86-64, `--network none`) — deliberately the oldest base in play, so
any newer starter environment is covered too:

| engine | tests | passed | failed | skipped | time |
|---|---:|---:|---:|---:|---:|
| cpp | 8348 | 8348 | 0 | 0 | ~195 s |
| python | 8348 | 8348 | 0 | 0 | ~295 s |

Artefacts land in `/results`: `summary.md`, `environment.md` (interpreter, engine
build hash, ISA level, compiler, full `pip freeze`), `pytest-{cpp,python}.{log,xml}`,
`ruff.log`, `mypy.log`.

The 12,600-run benchmark campaign is **not** reproduced here: it is 30 seeds ×
70 problems × 2 solvers × 3 arms at a 12 h budget per run, which needs a cluster.
`slurm/` holds the launchers; `experiments/models/orchestrator.py` is the entry
point for a single cell.

## Setting the capsule up

1. **New Capsule → Clone from Git**, pointing at this repository and selecting
   the **`codeocean`** branch. Importing `main` produces a capsule that cannot
   see its own sources — see above.
2. Open the **Environment** editor. The imported `environment/Dockerfile` is
   already complete. Code Ocean accepts base images only from its own registry,
   and which tags a given deployment offers varies, so if the build reports
   *Base Image Not Found*, pick any starter environment from the dropdown — it
   rewrites the `FROM` line and nothing else has to follow.

   The Dockerfile assumes almost nothing about the base. `postInstall` installs
   the system packages, Python 3.11 (the project requires `>=3.11`; the starter
   environments ship anything from 3.8 up) and the C++ toolchain, all at the
   private prefix `/opt/isalsr-conda` so it cannot collide with a base image
   that already carries conda. GCC 12 comes from conda-forge rather than the
   base, for two reasons: Ubuntu 20.04 ships GCC 9 and CMake 3.16, below the
   GCC 11 that `-march=x86-64-v3` needs and the CMake 3.18 that `pyproject`
   needs; and the compiler determines the extension's `build_hash`, so pinning
   it makes the engine reproducible instead of a function of the base image.
   `mpi4py` likewise comes from conda-forge with OpenMPI pinned — a pip build
   links the base's MPI and segfaults at interpreter teardown on older bases,
   *after* every test has passed, and left free the conda solver picks Intel
   MPI, which cannot initialise inside a container.
3. Set the master script to `run` in the Reproducibility panel. On the
   `codeocean` branch it arrives as `code/run`, i.e. `/code/run` in the capsule,
   beside `pyproject.toml`.
4. **Reproducible Run**.

### Why the package is built at run time

A Reproducible Run has no network, so everything must be installed at image
build time. The one exception is the project itself: `postInstall` runs during
the build, when Code Ocean has not yet mounted `/code`, so it cannot see the
source. `run` therefore compiles the extension during the run, offline,
which forces three pip flags:

```
--no-build-isolation   use the scikit-build-core pinned in the image instead of
                       fetching one into a throwaway environment
--no-index             resolve nothing from PyPI
--no-deps              every runtime dependency is already installed
```

`run` asserts `backends.engine() == "cpp"` immediately after and exits
non-zero otherwise. The check is not decorative: the extension resolves from
site-packages while the Python sources resolve from the repository, so a failed
compile is silent — the pure-Python fallback simply takes over and the run looks
healthy while measuring something else.

## Tests the capsule does not run

Two modules carry `pytestmark = pytest.mark.manuscript`:
`tests/unit/test_appendix_d_generator.py` and `tests/unit/test_numerical_audit.py`.
Both check the LaTeX manuscript against the code and read a manuscript checkout
that is not part of this repository, so outside the authors' workstation they can
only skip. `run` deselects them with `-m "not manuscript"` rather than
letting 170 skips accumulate in the report, where they would read as untested
behaviour instead of out-of-scope tooling. Run them locally with the manuscript
mounted; `python -m pytest tests/` still collects them by default.

`experiments/scripts/stage_d_certify.py` writes a campaign manifest whose
`build.git_commit` comes from `git rev-parse HEAD`. In a checkout without `.git`
that field is deliberately left empty and `validate_manifest` rejects it, so a
real certification run needs the git history present. Its *tests* pin the
provenance and are unaffected.

## Reproducing locally instead

```bash
docker build -t isalsr -f environment/Dockerfile environment/
docker run --rm --network none \
    -v "$PWD":/code -v "$PWD/results":/results \
    -w /code isalsr /code/run
```

`--network none` is the point of the exercise: it proves the image is
self-contained. Without Docker, `docs/engineering/CPP_BUILD.md` documents the
conda route.
