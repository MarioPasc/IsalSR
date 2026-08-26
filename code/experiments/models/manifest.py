"""Frozen provenance manifest for the C2 re-execution campaign (check A6).

The manifest is the single artefact that lets a reader reconstruct exactly which
code, which build, which configuration files, which operator sets, which arms and
which seeds produced a campaign root. It is written once, at submission time, and
never edited afterwards.

The schema is deliberately strict: :func:`validate_manifest` fails on *any*
missing or empty field rather than filling a default, because a manifest that
silently omits a field is worse than no manifest at all -- it looks like
provenance while carrying none. The module is executable
(``python -m experiments.models.manifest validate <path>``) and exits non-zero on
a manifest that does not validate, which is the form check A6 is graded in.

References
----------
``.claude/notes/review/tasks/EXECUTION-PLAN.md`` sections 4.1 (A4b, A5, A6),
7 (continuity exclusions) and 11.3 (submission splits).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ALPHABET_VERSION",
    "ARMS",
    "CAMPAIGN_TAG",
    "EXPECTED_N_ARRAYS",
    "EXPECTED_TASKS_PER_SEED_BLOCK",
    "NODE_CONSTRAINT",
    "SEEDS",
    "SUITE_SIZES",
    "BuildProvenance",
    "CampaignManifest",
    "ConfigDigest",
    "ManifestValidationError",
    "OperatorSetPolicy",
    "SubmissionSplit",
    "build_submission_splits",
    "load_manifest",
    "main",
    "save_manifest",
    "sha256_file",
    "validate_manifest",
]

# --------------------------------------------------------------------------
# Frozen campaign constants (EXECUTION-PLAN sections 4.1 and 11.3)
# --------------------------------------------------------------------------

SCHEMA_VERSION = "c2.1"
CAMPAIGN_TAG = "campaign/c2"

#: The three arms C2 runs. Order is significant: it is the order used by the
#: launcher and by the submission ledger of EXECUTION-PLAN section 11.3.
ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")

#: Campaign seeds. Seed 0 is reserved for smoke probes (SP-0) and must never
#: appear here, or a probe cell becomes mistakable for a campaign cell.
SEEDS: tuple[int, ...] = tuple(range(1, 21))

#: Sigma_SR alphabet in force since T16: SUB/DIV are decomposed into ADD+NEG and
#: MUL+INV inside both host adapters.
ALPHABET_VERSION = "decomposed"

#: SLURM node-constraint string pinned for C2 (check B6).
NODE_CONSTRAINT = "sr"

#: Problem count per suite. The 70-problem cohort is D1 (42) union D2 (28).
SUITE_SIZES: dict[str, int] = {
    "nguyen": 12,
    "feynman": 10,
    "hard": 10,
    "cherrypicked": 10,
    "roundoff": 8,
    "feynman_remainder": 6,
    "strogatz": 14,
}

#: One array per (method, arm, suite): 2 methods x 3 arms x 7 suites.
EXPECTED_N_ARRAYS = 42

#: C2 runs 20 seeds per (method, arm, suite, problem) cell.
EXPECTED_TASKS_PER_SEED_BLOCK = len(SEEDS)

_METHODS: tuple[str, ...] = ("udfs", "bingo")
_SHA256_HEX_LEN = 64


class ManifestValidationError(Exception):
    """Raised when a manifest is missing a field or carries an invalid value.

    The exception message enumerates *every* problem found, not just the first,
    so a single validator run tells the operator everything that must be fixed.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        joined = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"MANIFEST failed validation ({len(self.problems)} problem(s)):\n{joined}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file.

    Parameters
    ----------
    path
        File to digest. Read in binary chunks so large configs do not need to
        be held in memory.

    Returns
    -------
    str
        Lowercase 64-character hex digest.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(container: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    """Raise if any key is absent from a decoded mapping.

    Parameters
    ----------
    container
        Decoded JSON object.
    keys
        Keys that must all be present.
    where
        Human-readable location used in the error message.

    Raises
    ------
    ManifestValidationError
        If one or more keys are missing.
    """
    missing = [k for k in keys if k not in container]
    if missing:
        raise ManifestValidationError([f"{where}: missing field(s) {sorted(missing)}"])


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildProvenance:
    """Identity of the code and the native extension that produced a campaign.

    Attributes
    ----------
    git_commit
        Full 40-character commit SHA the campaign ran from.
    git_tag
        Annotated tag on that commit; ``campaign/c2`` for the production run.
    git_dirty
        Whether the working tree carried uncommitted changes. Must be ``False``
        for a production campaign.
    native_build_hash
        Build hash reported by ``isalsr.core`` for the loaded C++ extension.
    engine
        Dispatch engine actually observed, ``native`` or ``python``.
    compiler
        Compiler identification string for the native extension.
    compiler_flags
        Flags the extension was compiled with.
    isa_level
        Instruction-set baseline of the build, e.g. ``x86-64-v3``.
    avx512f
        Whether AVX-512F was enabled; portability-relevant (check B6b).
    native_module_path
        Absolute path of the loaded ``.so``, resolved at run time.
    native_module_mtime
        Modification time of that ``.so``, used to detect a stale build.
    """

    git_commit: str
    git_tag: str
    git_dirty: bool
    native_build_hash: str
    engine: str
    compiler: str
    compiler_flags: str
    isa_level: str
    avx512f: bool
    native_module_path: str
    native_module_mtime: str

    _KEYS: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        out = asdict(self)
        out.pop("_KEYS", None)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildProvenance:
        """Rebuild from a decoded mapping, rejecting missing fields.

        Raises
        ------
        ManifestValidationError
            If any field is absent.
        """
        keys = (
            "git_commit",
            "git_tag",
            "git_dirty",
            "native_build_hash",
            "engine",
            "compiler",
            "compiler_flags",
            "isa_level",
            "avx512f",
            "native_module_path",
            "native_module_mtime",
        )
        _require(data, keys, "build")
        return cls(**{k: data[k] for k in keys})


@dataclass(frozen=True)
class ConfigDigest:
    """SHA-256 of one resolved ``(method, suite)`` configuration file.

    Attributes
    ----------
    method
        Host search method, ``udfs`` or ``bingo``.
    suite
        Benchmark suite key, one of :data:`SUITE_SIZES`.
    path
        Repository-relative path of the YAML config.
    sha256
        Digest of the file as submitted.
    """

    method: str
    suite: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigDigest:
        """Rebuild from a decoded mapping, rejecting missing fields."""
        keys = ("method", "suite", "path", "sha256")
        _require(data, keys, "configs[]")
        return cls(**{k: data[k] for k in keys})


@dataclass(frozen=True)
class OperatorSetPolicy:
    """The A4b uniform-operator-set decision and its continuity consequences.

    C1 gave Bingo different primitive sets per tier, confounding the operator set
    with the problem group. A4b makes the set uniform per method across every
    problem. The cost is continuity: the D1 problems whose Bingo set changed are
    no longer like-for-like against C1 and are excluded from the continuity table
    of EXECUTION-PLAN section 7.

    Attributes
    ----------
    policy
        Policy identifier, ``uniform_per_method``.
    statement
        The A4b statement in prose, reproduced verbatim in the manifest so a
        reader never has to resolve a cross-reference to interpret the campaign.
    bingo_operators
        Primitive set carried by every ``bingo_*.yaml``.
    udfs_operators
        Primitive set UDFS actually searches. Taken from the vendored
        ``NODE_ARITY`` table, not from YAML.
    udfs_operator_source
        Why the UDFS set is not ours to set from configuration.
    bingo_continuity_exclusion_problems
        The 22 D1 problems whose Bingo operator set changed between C1 and C2.
        Excluded from the C1-to-C2 continuity table for Bingo only.
    corrected_definition_exclusion_problems
        D1 problems whose target function was corrected, so C1 ran a different
        function under the same name.
    continuity_note
        Why both exclusions are mandatory.
    """

    policy: str
    statement: str
    bingo_operators: list[str]
    udfs_operators: list[str]
    udfs_operator_source: str
    bingo_continuity_exclusion_problems: list[str]
    corrected_definition_exclusion_problems: list[str]
    continuity_note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorSetPolicy:
        """Rebuild from a decoded mapping, rejecting missing fields."""
        keys = (
            "policy",
            "statement",
            "bingo_operators",
            "udfs_operators",
            "udfs_operator_source",
            "bingo_continuity_exclusion_problems",
            "corrected_definition_exclusion_problems",
            "continuity_note",
        )
        _require(data, keys, "operator_set_policy")
        return cls(**{k: data[k] for k in keys})


@dataclass(frozen=True)
class SubmissionSplit:
    """One SLURM array of the campaign: a ``(method, arm, suite)`` triple.

    Attributes
    ----------
    index
        1-based position in the submission order of section 11.3.
    method
        Host search method.
    arm
        One of :data:`ARMS`.
    suite
        Benchmark suite key.
    n_problems
        Problems in the suite.
    n_tasks
        Array size, ``n_problems * len(SEEDS)``.
    """

    index: int
    method: str
    arm: str
    suite: str
    n_problems: int
    n_tasks: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubmissionSplit:
        """Rebuild from a decoded mapping, rejecting missing fields."""
        keys = ("index", "method", "arm", "suite", "n_problems", "n_tasks")
        _require(data, keys, "submission_splits[]")
        return cls(**{k: data[k] for k in keys})


def build_submission_splits(
    methods: tuple[str, ...] = _METHODS,
    arms: tuple[str, ...] = ARMS,
    suites: dict[str, int] | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> list[SubmissionSplit]:
    """Enumerate the campaign's SLURM arrays in submission order.

    The order is method-major, then arm, then suite -- the order the launcher
    writes to ``job_ids.txt`` and the order of the section 11.3 ledger.

    Parameters
    ----------
    methods
        Host methods, in submission order.
    arms
        Arms, in submission order.
    suites
        Mapping of suite key to problem count. Defaults to :data:`SUITE_SIZES`.
    seeds
        Seed list; its length is the per-problem task multiplier.

    Returns
    -------
    list[SubmissionSplit]
        One entry per array, 1-indexed.
    """
    suite_sizes = SUITE_SIZES if suites is None else suites
    splits: list[SubmissionSplit] = []
    idx = 1
    for method in methods:
        for arm in arms:
            for suite, n_problems in suite_sizes.items():
                splits.append(
                    SubmissionSplit(
                        index=idx,
                        method=method,
                        arm=arm,
                        suite=suite,
                        n_problems=n_problems,
                        n_tasks=n_problems * len(seeds),
                    )
                )
                idx += 1
    return splits


@dataclass(frozen=True)
class CampaignManifest:
    """Complete provenance record for one campaign root.

    Attributes
    ----------
    schema_version
        Manifest schema identifier.
    campaign
        Campaign name, e.g. ``c2``.
    campaign_root
        Absolute path of the results root the campaign writes to.
    created_utc
        ISO-8601 UTC timestamp of manifest creation.
    build
        Code and native-build identity.
    configs
        One digest per ``(method, suite)``.
    operator_set_policy
        A4b policy and the continuity exclusions.
    arms
        Arm names, expected to equal :data:`ARMS`.
    seeds
        Campaign seeds, expected to equal :data:`SEEDS`.
    alphabet_version
        Sigma_SR alphabet version.
    node_constraint
        SLURM ``--constraint`` string.
    submission_splits
        The campaign's SLURM arrays.
    notes
        Free-form operator notes; may be empty but must be present.
    """

    schema_version: str
    campaign: str
    campaign_root: str
    created_utc: str
    build: BuildProvenance
    configs: list[ConfigDigest]
    operator_set_policy: OperatorSetPolicy
    arms: list[str]
    seeds: list[int]
    alphabet_version: str
    node_constraint: str
    submission_splits: list[SubmissionSplit]
    notes: str = ""

    _TOP_KEYS: tuple[str, ...] = (
        "schema_version",
        "campaign",
        "campaign_root",
        "created_utc",
        "build",
        "configs",
        "operator_set_policy",
        "arms",
        "seeds",
        "alphabet_version",
        "node_constraint",
        "submission_splits",
        "notes",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole manifest."""
        return {
            "schema_version": self.schema_version,
            "campaign": self.campaign,
            "campaign_root": self.campaign_root,
            "created_utc": self.created_utc,
            "build": self.build.to_dict(),
            "configs": [c.to_dict() for c in self.configs],
            "operator_set_policy": self.operator_set_policy.to_dict(),
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "alphabet_version": self.alphabet_version,
            "node_constraint": self.node_constraint,
            "submission_splits": [s.to_dict() for s in self.submission_splits],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignManifest:
        """Rebuild a manifest from a decoded mapping.

        Every top-level field must be present; nothing is defaulted. This is what
        makes a truncated manifest fail rather than silently round-trip.

        Raises
        ------
        ManifestValidationError
            If any top-level or nested field is absent.
        """
        _require(data, cls._TOP_KEYS, "manifest")
        return cls(
            schema_version=data["schema_version"],
            campaign=data["campaign"],
            campaign_root=data["campaign_root"],
            created_utc=data["created_utc"],
            build=BuildProvenance.from_dict(data["build"]),
            configs=[ConfigDigest.from_dict(c) for c in data["configs"]],
            operator_set_policy=OperatorSetPolicy.from_dict(data["operator_set_policy"]),
            arms=list(data["arms"]),
            seeds=list(data["seeds"]),
            alphabet_version=data["alphabet_version"],
            node_constraint=data["node_constraint"],
            submission_splits=[SubmissionSplit.from_dict(s) for s in data["submission_splits"]],
            notes=data["notes"],
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _check_non_empty_strings(manifest: CampaignManifest, problems: list[str]) -> None:
    """Append a problem for every field that must be a non-empty string."""
    scalar_fields: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "campaign": manifest.campaign,
        "campaign_root": manifest.campaign_root,
        "created_utc": manifest.created_utc,
        "alphabet_version": manifest.alphabet_version,
        "node_constraint": manifest.node_constraint,
    }
    build = manifest.build
    scalar_fields.update(
        {
            "build.git_commit": build.git_commit,
            "build.git_tag": build.git_tag,
            "build.native_build_hash": build.native_build_hash,
            "build.engine": build.engine,
            "build.compiler": build.compiler,
            "build.compiler_flags": build.compiler_flags,
            "build.isa_level": build.isa_level,
            "build.native_module_path": build.native_module_path,
            "build.native_module_mtime": build.native_module_mtime,
        }
    )
    for name, value in scalar_fields.items():
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name}: must be a non-empty string, got {value!r}")


def _check_build(manifest: CampaignManifest, problems: list[str], *, strict_campaign: bool) -> None:
    """Validate the build block."""
    build = manifest.build
    if isinstance(build.git_commit, str) and len(build.git_commit.strip()) != 40:
        n_chars = len(build.git_commit.strip())
        problems.append(f"build.git_commit: expected a full 40-char SHA, got {n_chars} chars")
    if build.engine not in ("native", "python"):
        problems.append(f"build.engine: expected 'native' or 'python', got {build.engine!r}")
    if not strict_campaign:
        return
    if build.git_dirty:
        problems.append("build.git_dirty: a production campaign must run from a clean tree")
    if build.engine != "native":
        problems.append(f"build.engine: C2 must run the native engine, got {build.engine!r}")
    if build.git_tag != CAMPAIGN_TAG:
        problems.append(f"build.git_tag: expected {CAMPAIGN_TAG!r}, got {build.git_tag!r}")


def _check_configs(manifest: CampaignManifest, problems: list[str]) -> None:
    """Validate that every config digest is well formed and unambiguous."""
    if not manifest.configs:
        problems.append("configs: at least one (method, suite) digest is required")
        return
    seen: set[tuple[str, str]] = set()
    for cfg in manifest.configs:
        tag = f"configs[{cfg.method}/{cfg.suite}]"
        if not cfg.method or not cfg.suite or not cfg.path:
            problems.append(f"{tag}: method, suite and path must all be non-empty")
        digest = cfg.sha256 or ""
        if len(digest) != _SHA256_HEX_LEN or any(
            c not in "0123456789abcdef" for c in digest.lower()
        ):
            problems.append(f"{tag}: sha256 must be {_SHA256_HEX_LEN} hex chars, got {digest!r}")
        key = (cfg.method, cfg.suite)
        if key in seen:
            problems.append(f"{tag}: duplicate (method, suite) entry")
        seen.add(key)


def _check_operator_policy(manifest: CampaignManifest, problems: list[str]) -> None:
    """Validate the A4b policy block and its continuity exclusions."""
    pol = manifest.operator_set_policy
    for name, value in (
        ("policy", pol.policy),
        ("statement", pol.statement),
        ("udfs_operator_source", pol.udfs_operator_source),
        ("continuity_note", pol.continuity_note),
    ):
        if not isinstance(value, str) or not value.strip():
            problems.append(f"operator_set_policy.{name}: must be a non-empty string")
    if not pol.bingo_operators:
        problems.append("operator_set_policy.bingo_operators: must be non-empty")
    if not pol.udfs_operators:
        problems.append("operator_set_policy.udfs_operators: must be non-empty")
    # The continuity exclusion is the whole point of recording the policy: a
    # manifest that claims a uniform set without naming what it broke is not a
    # provenance record.
    if not pol.bingo_continuity_exclusion_problems:
        problems.append(
            "operator_set_policy.bingo_continuity_exclusion_problems: "
            "the 22-problem Bingo continuity exclusion must be enumerated"
        )
    if not pol.corrected_definition_exclusion_problems:
        problems.append(
            "operator_set_policy.corrected_definition_exclusion_problems: must be enumerated"
        )


def _check_arms_and_seeds(
    manifest: CampaignManifest, problems: list[str], *, strict_campaign: bool
) -> None:
    """Validate the arm list and the seed declaration (A5)."""
    if not manifest.arms:
        problems.append("arms: must be non-empty")
    if 0 in manifest.seeds:
        problems.append("seeds: seed 0 is reserved for smoke probes and must not appear (SP-0)")
    if len(set(manifest.seeds)) != len(manifest.seeds):
        problems.append("seeds: duplicate seeds")
    if not manifest.seeds:
        problems.append("seeds: must be non-empty")
    if manifest.alphabet_version != ALPHABET_VERSION:
        problems.append(
            f"alphabet_version: expected {ALPHABET_VERSION!r}, got {manifest.alphabet_version!r}"
        )
    if not strict_campaign:
        return
    if tuple(manifest.arms) != ARMS:
        problems.append(f"arms: C2 declares {list(ARMS)}, got {manifest.arms}")
    if tuple(manifest.seeds) != SEEDS:
        problems.append(f"seeds: C2 declares seeds {SEEDS[0]}..{SEEDS[-1]}, got {manifest.seeds}")
    if manifest.node_constraint != NODE_CONSTRAINT:
        got = manifest.node_constraint
        problems.append(f"node_constraint: C2 is pinned to {NODE_CONSTRAINT!r}, got {got!r}")


def _check_splits(
    manifest: CampaignManifest, problems: list[str], *, strict_campaign: bool
) -> None:
    """Validate the submission splits against section 11.3."""
    splits = manifest.submission_splits
    if not splits:
        problems.append("submission_splits: must be non-empty")
        return
    n_seeds = len(manifest.seeds) or len(SEEDS)
    for sp in splits:
        tag = f"submission_splits[{sp.index}]"
        if sp.n_problems <= 0 or sp.n_tasks <= 0:
            problems.append(f"{tag}: n_problems and n_tasks must be positive")
            continue
        if sp.n_tasks != sp.n_problems * n_seeds:
            problems.append(
                f"{tag}: n_tasks={sp.n_tasks} != n_problems({sp.n_problems}) x seeds({n_seeds})"
            )
    indices = [sp.index for sp in splits]
    if sorted(indices) != list(range(1, len(splits) + 1)):
        problems.append("submission_splits: indices must be a contiguous 1-based range")
    if not strict_campaign:
        return
    if len(splits) != EXPECTED_N_ARRAYS:
        n_got = len(splits)
        problems.append(
            f"submission_splits: section 11.3 declares {EXPECTED_N_ARRAYS} arrays, got {n_got}"
        )
    declared_arms = {sp.arm for sp in splits}
    if declared_arms != set(ARMS):
        problems.append(f"submission_splits: arms {sorted(declared_arms)} != {sorted(ARMS)}")
    declared_suites = {sp.suite for sp in splits}
    if declared_suites != set(SUITE_SIZES):
        problems.append(
            f"submission_splits: suites {sorted(declared_suites)} != {sorted(SUITE_SIZES)}"
        )
    for sp in splits:
        expected = SUITE_SIZES.get(sp.suite)
        if expected is not None and sp.n_problems != expected:
            problems.append(
                f"submission_splits[{sp.index}]: suite {sp.suite!r} has "
                f"{expected} problems, manifest says {sp.n_problems}"
            )


def validate_manifest(manifest: CampaignManifest, *, strict_campaign: bool = True) -> None:
    """Validate a manifest, raising on the first *complete* set of problems.

    Parameters
    ----------
    manifest
        Manifest to check.
    strict_campaign
        If ``True`` (the default), additionally enforce the frozen C2 constants:
        three arms, seeds 1..20, the ``sr`` node constraint, the ``campaign/c2``
        tag, a clean tree, the native engine, and 42 submission arrays. Set to
        ``False`` to validate a smoke or probe manifest, which is structurally
        identical but does not carry the campaign's frozen values.

    Raises
    ------
    ManifestValidationError
        If any field is missing, empty or inconsistent. The message enumerates
        every problem found.
    """
    problems: list[str] = []
    _check_non_empty_strings(manifest, problems)
    _check_build(manifest, problems, strict_campaign=strict_campaign)
    _check_configs(manifest, problems)
    _check_operator_policy(manifest, problems)
    _check_arms_and_seeds(manifest, problems, strict_campaign=strict_campaign)
    _check_splits(manifest, problems, strict_campaign=strict_campaign)
    if problems:
        raise ManifestValidationError(problems)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def save_manifest(manifest: CampaignManifest, path: Path) -> None:
    """Write a manifest to disk as pretty-printed JSON.

    The write is atomic (temp file plus rename) so a killed job can never leave a
    half-written MANIFEST behind, which would be indistinguishable from a
    truncated one.

    Parameters
    ----------
    manifest
        Manifest to write.
    path
        Destination path, typically ``<campaign_root>/MANIFEST.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)


def load_manifest(path: Path) -> CampaignManifest:
    """Read a manifest from disk.

    Parameters
    ----------
    path
        Path to a ``MANIFEST.json``.

    Returns
    -------
    CampaignManifest
        The decoded manifest. Structural completeness is enforced here; value
        validation is :func:`validate_manifest`.

    Raises
    ------
    ManifestValidationError
        If the file is not valid JSON, or a field is missing.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError([f"{path}: not valid JSON ({exc})"]) from exc
    except OSError as exc:
        raise ManifestValidationError([f"{path}: cannot be read ({exc})"]) from exc
    if not isinstance(raw, dict):
        raise ManifestValidationError([f"{path}: top level must be a JSON object"])
    return CampaignManifest.from_dict(raw)


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Validate a MANIFEST from the command line.

    Returns
    -------
    int
        ``0`` if the manifest validates, ``1`` otherwise. Check A6 is graded on
        this exit status against a deliberately truncated manifest.
    """
    parser = argparse.ArgumentParser(
        prog="python -m experiments.models.manifest",
        description="Validate a campaign MANIFEST.json (EXECUTION-PLAN check A6).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate a MANIFEST.json")
    val.add_argument("path", type=Path, help="path to MANIFEST.json")
    val.add_argument(
        "--allow-non-campaign",
        action="store_true",
        help="relax the frozen C2 constants (for smoke/probe manifests)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.path)
        validate_manifest(manifest, strict_campaign=not args.allow_non_campaign)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"MANIFEST OK: {args.path} (campaign={manifest.campaign}, arms={manifest.arms})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
