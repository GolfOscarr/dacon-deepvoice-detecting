"""YAML for the EDA pass.

The rules are `models.config`'s and deliberately not a second set of them --
`_build` is imported rather than copied, so "unknown keys are an error" has one
implementation. `training.config` reaches for the same private name for the same
reason.

Critical: **the source table lives here, not in `scripts/sources.yaml`.** Two
reasons, and the second is the load-bearing one. The acquisition registry
records what we were allowed to fetch; this records what is on local disk and
which pool each source feeds, which is a different question. And
`fakemusiccaps` -- the whole of pool D, carrying 0.27 of the metric -- is **not
in `sources.yaml` at all** (docs/EDA/07 section 4), so a config derived from that
registry would silently omit it.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from eda.ids import under_any

from models.config import ConfigError, _build
from training.manifest import POOLS, ROW_KINDS
from training.spec import CELL_TABLE

__all__ = ["AnalysisConfig", "ConfigError", "EdaConfig", "GateConfig", "MIXED",
           "ProbeConfig", "SampleConfig", "SourceSpec", "VALID_POOLS",
           "eda_config_from_dict", "load_eda_config"]

#: An archive that spans pools and whose components are assigned per directory.
#: `training.manifest.POOL_LABELS` has no entry for it on purpose --
#: `eda.analyze.shortcut.head_labels` maps it to null and drops it from the
#: audit rather than guessing a label.
MIXED = "mixed"

#: Derived from `training.manifest.POOLS` rather than written out, so a pool
#: added there cannot drift out of this check -- the same reason
#: `manifest._pool_labels` derives itself from `ROLE_POOLS` instead of
#: restating the table.
VALID_POOLS = (*POOLS, MIXED)

#: Cells that cannot be a `whole_file` row, from `validate_manifest`'s own rule:
#: they hold one real and one fake component, which is exactly why they cannot
#: be scraped and must be composed.
UNSCRAPEABLE_CELLS = (6, 7)

#: How `eda.gates` turns a file into an allowlist id. See `SourceSpec.allowlist_key`.
ALLOWLIST_KEYS = ("stem", "int_stem", "relpath")

#: Where a source's audio actually lives. 🔴 Two shapes, and the difference is
#: invisible until it costs you: most sources ship as archives and land under
#: `--extract` (`interim`), but some are stored in S3 with their directory tree
#: already intact -- MLAAD is, deliberately, because flattening
#: `fake/<language>/<generator>/` would destroy the generator-disjoint split
#: axis. `extract_archives` finds no archive in such a payload, creates an empty
#: `interim/<name>/<version>/`, and returns 0.
STAGES = ("interim", "raw")


@dataclass
class SourceSpec:
    """One acquired source, as it sits on local disk."""

    #: Matches the S3 directory and becomes the `source_name` prefix of every
    #: `file_id`. Caveat: this is the *coarse* key only -- speaker/track/artist
    #: granularity is recovered by `eda.analyze.grouping`, and docs/data/08
    #: records that it cannot be retrofitted.
    name: str
    #: Relative to `EdaConfig.root`.
    root: str
    #: 🔴 Exactly one of `pool` and `cell` is meaningful, and which one is
    #: decided by `row_kind`. This is `validate_manifest`'s invariant, not a
    #: convenience: a **component** is drawn and composed, so it has a pool and
    #: no cell; a **whole_file** row is used as-is, so it has a cell and
    #: *"pool = null -- a whole file is used as-is, not drawn from a component
    #: pool"*.
    #:
    #: SONICS is why this exists. It was acquired as `pool: D`, and pool D
    #: asserts `voice_present = 0`; its own `fake_songs.csv` reports
    #: `no_vocal = False` for all 49,074 rows. As a pool-D component it would
    #: have told the model there is no voice in 49k AI songs that all contain
    #: singing, and routed sung vocals into the 0.27-weight music-fake head. It
    #: is `whole_file`, `cell: 8`.
    row_kind: str = "component"
    pool: str | None = None
    cell: int | None = None
    #: `interim` (unpacked from an archive) or `raw` (synced with its tree
    #: intact, so `root` includes the `payload/` segment). See `STAGES`.
    stage: str = "interim"
    #: Extensions to enumerate, lowercase, with the dot.
    suffixes: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    #: Subtrees excluded from enumeration. Critical: this is the only place a
    #: file is refused, and it exists for one reason -- docs/EDA/04 D1 forbids
    #: MusicCaps real audio from entering pool D at all. Every exclusion is a
    #: licence decision and must name its reason.
    exclude: tuple[str, ...] = ()
    exclude_reason: str = ""
    #: Filename globs; empty means every file. 🔴 This exists because `exclude`
    #: matches **path components** and one real directory distinguishes two
    #: kinds of material by filename alone: OpenSLR-28's
    #: `real_rirs_isotropic_noises/` holds 325 impulse responses (`*_rir_*`,
    #: `air_*binaural*`, median 1.25 s) beside 92 isotropic noise recordings
    #: (`*_noise_*`, median 30.0 s). The first are convolution kernels for the
    #: reverb transform and the second are pool-E audio, and no subtree
    #: separates them (docs/EDA/05 E0d).
    #:
    #: ⚠️ Narrow on purpose. This is not a general query language: it selects
    #: *which files a source is*, so two sources over one directory each state
    #: their own half and the census can name both. Every use needs
    #: `name_glob_reason` for the same purpose `exclude_reason` serves.
    name_glob: tuple[str, ...] = ()
    name_glob_reason: str = ""
    #: G-EDA1. A CSV of permitted track ids, relative to `EdaConfig.root`. FMA
    #: and MTG-Jamendo are single downloads whose audio is **not under a single
    #: licence**: docs/data/12 section 4 partitions them per track and says
    #: plainly that "the archive being cleared does not clear a track".
    allowlist: str = ""
    #: Column in that CSV holding the id. Both shipped allowlists call it `id`.
    allowlist_column: str = "id"
    #: How a file becomes an id, matched against `allowlist_column`:
    #:
    #: * `stem`     -- the filename without its suffix
    #: * `int_stem` -- the stem with zero padding stripped. FMA needs this:
    #:   `000002.mp3` is id `2`.
    #: * `relpath`  -- the whole path below the source root. MTG-Jamendo needs
    #:   this: its ids are **paths**, `14/214.mp3`, not bare track numbers, so
    #:   matching on the stem compares `214` against `14/214.mp3` and every row
    #:   fails -- a G-EDA1 FAIL for a reason that has nothing to do with licences.
    allowlist_key: str = "stem"
    #: Non-empty means **this source is known and deliberately not runnable**,
    #: and the value is why. `enumerate_source` raises it rather than returning
    #: files.
    #:
    #: Critical: this exists so a source we cannot yet label *correctly* fails
    #: loudly instead of being quietly deleted from the config and rediscovered
    #: later, or -- far worse -- left registered under a pool that mislabels it.
    #: `training.stages` makes `rank_polish` raise for the same reason: a knob
    #: that validates and silently does the wrong thing is its own defect.
    blocked: str = ""

    @property
    def partition(self) -> str:
        """The output subdirectory, and the unit `consolidate` works in.

        A component source partitions by pool; a whole_file source has no pool
        to partition by, so it partitions by cell. Deliberately not called a
        "group" -- `folds.grouping_atoms` already owns that word.
        """
        return self.pool if self.row_kind == "component" else f"cell{self.cell}"

    def selects_name(self, name: str) -> bool:
        """Does this source claim a file with this basename?

        Empty `name_glob` claims everything, which is what every source but the
        RIRS pair wants.
        """
        if not self.name_glob:
            return True
        return any(fnmatch.fnmatch(name, g) for g in self.name_glob)

    def excludes_path(self, rel: str) -> bool:
        """Is this path under one of the source's excluded subtrees?

        Critical: the source owns this, so `eda.driver` and `eda.gates` cannot
        disagree about what an exclusion means -- they had two copies of the
        same substring test, and it matched `real_half/` for `exclude: [real]`.
        """
        return under_any(rel, self.exclude)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ConfigError(
                f"source {self.name!r}: stage must be one of {list(STAGES)}, "
                f"got {self.stage!r}")
        if self.row_kind not in ROW_KINDS:
            raise ConfigError(
                f"source {self.name!r}: row_kind must be one of {list(ROW_KINDS)}, "
                f"got {self.row_kind!r}")
        # Critical: the two branches mirror `training.manifest.validate_manifest`
        # exactly. Accepting a shape here that it would reject moves the failure
        # from config-load time to manifest-build time, after the EDA tables have
        # been written and a fold table may already rest on them.
        if self.row_kind == "component":
            if self.cell is not None:
                raise ConfigError(
                    f"source {self.name!r}: a component source must have cell = null "
                    f"-- a component has no cell until the sampler composes one")
            if self.pool not in VALID_POOLS:
                raise ConfigError(
                    f"source {self.name!r}: pool must be one of {list(VALID_POOLS)}, "
                    f"got {self.pool!r}")
        else:
            if self.pool is not None:
                raise ConfigError(
                    f"source {self.name!r}: a whole_file source must have pool = null "
                    f"-- a whole file is used as-is, not drawn from a component pool")
            if self.cell not in CELL_TABLE:
                raise ConfigError(
                    f"source {self.name!r}: cell must be one of "
                    f"{sorted(CELL_TABLE)}, got {self.cell!r}")
            if self.cell in UNSCRAPEABLE_CELLS:
                raise ConfigError(
                    f"source {self.name!r}: cell {self.cell} cannot be a whole_file "
                    f"source -- cells {list(UNSCRAPEABLE_CELLS)} hold one real and "
                    f"one fake component, which is why they cannot be scraped")
        if self.allowlist_key not in ALLOWLIST_KEYS:
            raise ConfigError(
                f"source {self.name!r}: allowlist_key must be one of "
                f"{list(ALLOWLIST_KEYS)}, got {self.allowlist_key!r}")
        if self.exclude and not self.exclude_reason:
            raise ConfigError(
                f"source {self.name!r}: `exclude` without `exclude_reason`. An "
                f"excluded subtree is a licence or a label decision and the "
                f"reason travels with it (docs/EDA/07 G-EDA1)")
        if self.name_glob and not self.name_glob_reason:
            raise ConfigError(
                f"source {self.name!r}: `name_glob` without `name_glob_reason`. "
                f"Selecting part of a directory by filename says this source is "
                f"not the whole directory, and the reason travels with it")


@dataclass
class ProbeConfig:
    """The M tier: ffprobe over 100% of every pool, no decode."""

    #: ffprobe is ~20 ms of process spawn per file and the corpus is ~1.7 M
    #: files, so this is the difference between 16 minutes and 8 hours. Threads,
    #: not processes: `subprocess.run` releases the GIL and a thread pool skips
    #: the pickling entirely.
    workers: int = 32
    #: Files per part file. Parquet cannot be appended, so a part is the unit of
    #: resumability -- see `eda.driver`.
    shard_size: int = 20000
    ffprobe: str = "ffprobe"
    timeout_s: float = 20.0
    #: Bytes read from the head of each file for the mp3 Xing/LAME scan.
    header_bytes: int = 8192
    #: Read size for the streaming sha256.
    hash_chunk_bytes: int = 1 << 20
    #: Which registered M-tier extractors to run; empty means all of them.
    #:
    #: ⚠️ **`identity` reads every byte of the corpus.** The rest of the M tier
    #: touches only headers, so "no decode" is not the same as "cheap": with
    #: `identity` on, the pass is a full 291 GiB read, and without it E1 -- the
    #: duplicate sweep that caught the RIRS/MUSAN leak -- cannot run at all.
    #: Narrow this only for a fast first look, never for the pass whose output
    #: a fold table is built from.
    extractors: tuple[str, ...] = ()


@dataclass
class SampleConfig:
    """The S tier draw. Seeded and recorded, because a selection is part of the
    corpus definition -- the MLAAD cap is the precedent (docs/data/12)."""

    seed: int = 0
    per_stratum: int = 2000
    stratify_by: tuple[str, ...] = ("source_name",)
    #: Pools measured at 100% rather than sampled: D because five generator
    #: families are too few to sample from and it carries 0.27, E because it is
    #: small and its last surprise cost a corpus rebuild (docs/EDA/00 section 5).
    full_pools: tuple[str, ...] = ("D", "E")
    #: Files per S-tier part, and therefore the unit of resumability.
    #:
    #: ⚠️ Deliberately not `probe.shard_size`. That one is sized for ffprobe at
    #: ~20 ms a file, where 20,000 files is about seven minutes of work; the
    #: same count at decode speed is a run you cannot interrupt. Measured on
    #: pool E: the first signal pass ran as a single 14,102-file part, wrote
    #: nothing until it finished, and had no checkpoint if it had died.
    shard_size: int = 500


@dataclass
class AnalysisConfig:
    """Knobs for `eda.analyze`. Separate from `SampleConfig` on purpose: that
    seed picks *which files are measured* and is part of the corpus definition,
    while this one only picks a cross-validation shuffle. Sharing one field
    would make a re-seeded analysis look like a re-drawn corpus in the ledger."""

    seed: int = 0
    #: Folds in the shortcut audit's cross-validation.
    n_splits: int = 5
    #: One-hot width per categorical feature; the rest folds into `__other__`.
    #: `encoder` is effectively a corpus fingerprint and would otherwise explode
    #: the design.
    top_k: int = 20
    #: How deep `grouping.depth_profile` walks before a layout stops being a
    #: readable tree and starts being hashed subdirectories.
    max_depth: int = 8


@dataclass
class GateConfig:
    """Pre-committed thresholds (docs/EDA/07 section 2). Written out so a result
    cannot be re-read favourably after the fact."""

    #: G-EDA2. docs/data/07 E-S2 sets it; above this, neutralize before training.
    shortcut_auc_max: float = 0.60
    #: G-EDA3. Below this the fold count drops and the caveat travels.
    min_groups_per_role: int = 6
    #: G-EDA5 tolerates no duplicate whatsoever across sources.
    max_cross_source_duplicates: int = 0


@dataclass
class EdaConfig:
    #: Where `fetch_from_s3.py --extract` put the unpacked corpus.
    root: Path = Path(".")
    #: Where `fetch_from_s3.py --dest` put the synced payloads. Only sources
    #: with `stage: raw` read from here.
    raw: Path = Path(".")
    out: Path = Path("eda/out")
    sources: tuple[SourceSpec, ...] = ()
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.raw = Path(self.raw)
        self.out = Path(self.out)
        names = [s.name for s in self.sources]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"duplicate source name(s) {dupes}")

    def source_root(self, spec: SourceSpec) -> Path:
        """Where this source's audio is, given its stage."""
        return (self.root if spec.stage == "interim" else self.raw) / spec.root

    def source(self, name: str) -> SourceSpec:
        for s in self.sources:
            if s.name == name:
                return s
        raise ConfigError(f"no source named {name!r}; have {sorted(s.name for s in self.sources)}")

    def sources_in(self, partition: str | None) -> tuple[SourceSpec, ...]:
        """Sources writing into one output partition (`A`-`E`, `mixed`, `cellN`)."""
        if partition is None:
            return self.sources
        return tuple(s for s in self.sources if s.partition == partition)

    def partitions(self) -> list[str]:
        return sorted({s.partition for s in self.sources})


def eda_config_from_dict(d: Mapping[str, Any]) -> EdaConfig:
    """Build from a mapping. Unknown keys are an error, at every level."""
    d = dict(d)
    raw = d.pop("sources", None) or []
    if not isinstance(raw, list):
        raise ConfigError(f"sources: expected a list, got {type(raw).__name__}")
    if not raw:
        # Critical: a config with no sources loads cleanly and then does nothing
        # -- `probe` selects no source, `consolidate` finds no parts, and the run
        # looks like it succeeded. That is the same shape as a knob that
        # validates and silently does nothing, which `training.stages` makes
        # `rank_polish` raise over rather than tolerate.
        #
        # It also keeps the classification in tests/test_config.py honest: an
        # all-defaults `{}` file is accepted by the train and run loaders both,
        # and a third loader that accepted it too would widen a known tie rather
        # than being a config anybody meant to write.
        raise ConfigError(
            "eda: `sources` is empty. An EDA config with no sources probes "
            "nothing and reports success")
    specs = tuple(_build(SourceSpec, s, f"sources[{i}]") for i, s in enumerate(raw))
    # `replace` re-runs `__post_init__`, which is what re-checks the duplicate
    # source names once the list is actually attached.
    return dataclasses.replace(_build(EdaConfig, {**d, "sources": []}, "eda"),
                               sources=specs)


def load_eda_config(path: str | Path) -> EdaConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return eda_config_from_dict(yaml.safe_load(fh) or {})
