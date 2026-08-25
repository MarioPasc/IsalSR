# Reproducing IsalSR

This repository doubles as a [Code Ocean](https://codeocean.com) compute capsule.
Three top-level folders exist only for that purpose and are inert everywhere else:

| Folder | Contents |
|---|---|
| `environment/` | `Dockerfile`, `postInstall`, `requirements.txt` — the capsule image |
| `code/` | `run` — the capsule's master script |
| `metadata/` | `metadata.yml` — capsule title, description, authors |

Code Ocean's git import recognises these four names (`metadata`, `environment`,
`code`, `data`) and files them under the matching capsule directory; everything
else lands at the capsule root. `code/run` locates the project by searching for
`pyproject.toml`, so it works whether the source tree ends up at `/code` or at `/`.

## What the capsule reproduces

The verification suite, not the benchmark campaign.

`code/run` builds the C++ engine from source, then runs the full test suite
**twice** — once on the C++ engine and once on the pure-Python reference
implementation — plus `ruff` and `mypy --strict`. Running both engines is the
point: every reported number was produced with the C++ engine, and the claim
that the two are semantically interchangeable is only evidence if both are
exercised.

Measured on the capsule image (24-core x86-64, `--network none`):

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

1. **New Capsule → Clone from Git**, pointing at this repository's `main` branch.
2. Open the **Environment** editor. The imported `environment/Dockerfile` is
   already complete; confirm the base image resolves. It is
   `registry.codeocean.com/codeocean/ubuntu:22.04`, chosen because the C++ engine
   targets `-march=x86-64-v3`, which GCC did not accept before version 11 —
   Ubuntu 22.04 ships GCC 11.4. Any 22.04 or 24.04 base works; an older one
   does not. Python is **not** taken from the base image: `postInstall` installs
   Miniforge and pins Python 3.11, because the project requires `>=3.11` and
   Ubuntu 22.04 ships 3.10.
3. If the source tree landed under **Other Files** rather than `/code`, either
   leave it — `code/run` finds it either way — or drag it into `/code`.
4. Set the master script to `code/run` in the Reproducibility panel.
5. **Reproducible Run**.

### Why the package is built at run time

A Reproducible Run has no network, so everything must be installed at image
build time. The one exception is the project itself: `postInstall` runs during
the build, when Code Ocean has not yet mounted `/code`, so it cannot see the
source. `code/run` therefore compiles the extension during the run, offline,
which forces three pip flags:

```
--no-build-isolation   use the scikit-build-core pinned in the image instead of
                       fetching one into a throwaway environment
--no-index             resolve nothing from PyPI
--no-deps              every runtime dependency is already installed
```

`code/run` asserts `backends.engine() == "cpp"` immediately after and exits
non-zero otherwise. The check is not decorative: the extension resolves from
site-packages while the Python sources resolve from the repository, so a failed
compile is silent — the pure-Python fallback simply takes over and the run looks
healthy while measuring something else.

## Tests the capsule does not run

Two modules carry `pytestmark = pytest.mark.manuscript`:
`tests/unit/test_appendix_d_generator.py` and `tests/unit/test_numerical_audit.py`.
Both check the LaTeX manuscript against the code and read a manuscript checkout
that is not part of this repository, so outside the authors' workstation they can
only skip. `code/run` deselects them with `-m "not manuscript"` rather than
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
    -w /code isalsr /code/code/run
```

`--network none` is the point of the exercise: it proves the image is
self-contained. Without Docker, `docs/engineering/CPP_BUILD.md` documents the
conda route.
