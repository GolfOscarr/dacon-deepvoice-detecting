"""The publisher's own grouping key, joined into the census.

`analyze/grouping.py` profiles **path depth** and says plainly that what it
returns is a suggestion: *"a real assignment needs the publisher's own key"*.
For seven of thirteen sources it reaches no depth at all and says so on every
run. This module is the other half -- one provider per source, each reading the
key from the publisher's own evidence, each naming that evidence in its
docstring.

🔴 **Three states, not two.** A source with no publisher key is not
automatically a source that needs one:

``publisher``
    The key was read from the publisher's metadata or from structure in the
    filename that the publisher put there. SONICS' `algorithm`, MUSAN's
    `ANNOTATIONS`, RIRS' room names.
``single``
    The source genuinely has **one** grouping atom and no key can change that.
    LJSpeech is 13,100 utterances from one speaker. Reporting this as "needs
    the publisher's own key" is wrong in the expensive direction: it sends the
    next reader looking for a file that does not exist, and it hides that the
    source can never be rotated in a fold table.
``path``
    No provider, so the key is the directory prefix at the depth
    `grouping_report` would have suggested. Still a real key -- `cfad-fake`'s
    eleven generators live in the path and nowhere else -- but a suggested one.

⚠️ **Coverage is total or the provider raises.** A key that covers 645 of 660
rows produces a fold table where some rows are grouped and some are not, and
the ungrouped ones leak without anything reporting a problem. A provider that
cannot cover every row must say what the uncovered rows are -- as a named
bucket, not as a null. MUSAN's music annotations are the case: 15 of 660 files
have no line, and they become `<collection>/_unannotated`.

⚠️ **When in doubt, merge.** Bucketing several unknown artists into one group
costs group count; splitting one artist across two groups costs a leak that no
gate can see. Every fallback here errs toward merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

import pandas as pd

from eda.config import EdaConfig, SourceSpec
from eda.ids import relpath_of

__all__ = ["GroupKeys", "PROVIDERS", "attach_group_keys", "group_keys_for",
           "key_report"]

#: The column the census carries, and the one `grouping_report` prefers.
KEY_COLUMN = "group_key"
#: Which of the three states above produced it. Kept beside the key rather than
#: inferred from it, because `single` and a one-valued `publisher` key look
#: identical in the data and mean different things.
KIND_COLUMN = "group_key_kind"

PUBLISHER = "publisher"
SINGLE = "single"
PATH = "path"


@dataclass(frozen=True)
class GroupKeys:
    """One source's keys, and the evidence they were read from."""

    source_name: str
    kind: str
    #: What was read, in one line, so `key_report` can print it.
    evidence: str
    #: `file_id` -> group key, covering every row. Globally namespaced by the
    #: provider, except where two sources deliberately share one (`LJ_SPEAKER`).
    keys: pd.Series

    @property
    def n_groups(self) -> int:
        return int(self.keys.nunique()) if len(self.keys) else 0


Provider = Callable[[EdaConfig, SourceSpec, pd.DataFrame], GroupKeys]


def _stems(files: pd.DataFrame) -> pd.Series:
    """`file_id` -> the filename without its suffix."""
    return pd.Series(
        [PurePosixPath(relpath_of(f)).stem for f in files["file_id"]],
        index=files["file_id"], dtype="object")


def _check_total(source_name: str, files: pd.DataFrame, keys: pd.Series,
                 what: str) -> pd.Series:
    """Refuse a key that does not cover every row. See the module docstring."""
    missing = [f for f in files["file_id"] if f not in keys.index
               or pd.isna(keys.get(f))]
    if missing:
        raise KeyError(
            f"source {source_name!r}: {what} covers {len(keys)} of "
            f"{len(files)} row(s); {len(missing)} have no key, e.g. "
            f"{missing[:3]}. A partial key leaks the uncovered rows without "
            f"reporting anything -- give them a named bucket instead")
    return keys.loc[files["file_id"]]


# ---------------------------------------------------------------------------
# The providers
# ---------------------------------------------------------------------------

def _sonics(cfg: EdaConfig, source: SourceSpec, files: pd.DataFrame) -> GroupKeys:
    """SONICS' `algorithm`, from the publisher's own `fake_songs.csv`.

    ⚠️ The CSV is in the **raw** tree, not the interim one: `hf-2025/payload/`
    was synced with its directory intact and only `fake_songs/` was unpacked, so
    `cfg.source_root(source)` does not reach it.

    Five generators against the 1 that path depth finds -- chirp-v3.5,
    udio-120s, udio-30s, chirp-v3, chirp-v2-xxl-alpha -- and the join is
    49,074 of 49,074 with the CSV's own `duration` agreeing with ffprobe to
    0.000 s.
    """
    csv = cfg.raw / source.root / "payload" / "fake_songs.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"source {source.name!r}: {csv} is not on disk. It is the "
            f"publisher's key and lives in the raw tree beside `fake_songs/`")
    meta = pd.read_csv(csv, usecols=["filename", "algorithm"], low_memory=False)
    by_stem = dict(zip(meta["filename"].astype(str), meta["algorithm"].astype(str)))
    keys = _stems(files).map(
        lambda stem: f"{source.name}/{by_stem[stem]}" if stem in by_stem else None)
    keys = _check_total(source.name, files, keys, f"`algorithm` in {csv.name}")
    return GroupKeys(source.name, PUBLISHER,
                     f"{csv.name}: `algorithm` joined on the filename stem", keys)


def _fakemusiccaps(cfg: EdaConfig, source: SourceSpec,
                   files: pd.DataFrame) -> GroupKeys:
    """The MusicCaps caption id -- the filename stem, **not** the generator.

    🔴 This is a correction, not a refinement. Path depth finds the five
    generator directories and reports 5 groups, below the floor; the real
    structure is **5,521 MusicCaps clips x 5 generators, exactly**, and the
    stem is identical across all five (measured: 5,521 files in each of the
    five directories, 5,521 distinct stems in their union).

    So grouping by generator puts the *same caption* in every group. Hold out
    `musicldm` and its 5,521 captions are still in the training fold four times
    over, rendered by the other four models. The caption is the atom that has
    to be held together; the generator is a stratification axis, not a
    grouping one.
    """
    keys = _stems(files).map(lambda stem: f"{source.name}/{stem}")
    keys = _check_total(source.name, files, keys, "the filename stem")
    return GroupKeys(source.name, PUBLISHER,
                     "the MusicCaps clip id: the filename stem, shared by all "
                     "five generators", keys)


def _musan_music(cfg: EdaConfig, source: SourceSpec,
                 files: pd.DataFrame) -> GroupKeys:
    """MUSAN's per-collection `ANNOTATIONS`: `<id> <genre> <vocals> <artist>`.

    The artist is the fourth field. 15 of 660 files have no line -- 6 in `fma`,
    9 in `jamendo` -- and they become `<collection>/_unannotated` rather than a
    null: merging a handful of unknown artists costs group count, splitting one
    artist across folds costs a leak (module docstring).

    `rfm` is one artist across 147 files by the publisher's own annotation, and
    that is a fact about the collection rather than a gap in it.
    """
    root = cfg.source_root(source)
    by_id: dict[str, str] = {}
    for ann in sorted(root.glob("*/ANNOTATIONS")):
        collection = ann.parent.name
        for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 4:
                by_id[fields[0]] = f"{collection}/{fields[3]}"

    def key(file_id: str) -> str:
        rel = PurePosixPath(relpath_of(file_id))
        return f"{source.name}/" + by_id.get(rel.stem,
                                             f"{rel.parts[0]}/_unannotated")

    keys = pd.Series([key(f) for f in files["file_id"]],
                     index=files["file_id"], dtype="object")
    keys = _check_total(source.name, files, keys, "MUSAN's ANNOTATIONS")
    return GroupKeys(source.name, PUBLISHER,
                     "ANNOTATIONS field 4 (artist), per collection; "
                     "unannotated files bucketed per collection", keys)


def _rirs_isotropic_noise(cfg: EdaConfig, source: SourceSpec,
                          files: pd.DataFrame) -> GroupKeys:
    """The recording room, from the name OpenSLR-28 gave the file.

    `RVB2014_type1_noise_largeroom1_*.wav` -- corpus, type, kind, **room**, then
    the channel. Ten rooms over the 92 files: nine RVB2014 rooms at ten
    channels each, and `RWCP_type5_noise_cirline` twice. Above the floor of 6,
    where path depth found 0 because the directory is flat.

    The channel is deliberately *not* in the key: ten channels of one room are
    ten microphones recording the same air, which is the definition of a thing
    that must not straddle a fold boundary.
    """
    def key(file_id: str) -> str:
        stem = PurePosixPath(relpath_of(file_id)).stem
        parts = stem.split("_")
        if len(parts) < 4:
            raise ValueError(
                f"source {source.name!r}: {stem!r} does not have the "
                f"`<corpus>_<type>_noise_<room>_...` shape this key reads")
        return f"{source.name}/" + "_".join(parts[:4])

    keys = pd.Series([key(f) for f in files["file_id"]],
                     index=files["file_id"], dtype="object")
    keys = _check_total(source.name, files, keys, "the room in the filename")
    return GroupKeys(source.name, PUBLISHER,
                     "filename field 4: the recording room, channels merged", keys)


#: 🔴 A **cross-source** key, and the only one here. `ljspeech` and the seven
#: LJSpeech halves of `wavefake` are the same speaker, so they share a key
#: literally rather than by coincidence: a fold builder grouping on
#: `group_key` puts the real utterance and its vocoded twin on the same side of
#: every boundary without being told about the pairing. That pairing is what
#: docs/EDA/02 B1 exists to measure, and measuring it requires that no model
#: ever trains on one half and validates on the other.
LJ_SPEAKER = "ljspeech/LJ"
#: JSUT's single Japanese speaker, on the same principle.
JSUT_SPEAKER = "jsut/BASIC5000"


def _ljspeech(cfg: EdaConfig, source: SourceSpec, files: pd.DataFrame) -> GroupKeys:
    """One speaker. 13,100 utterances, and no key exists because none can.

    Declared rather than left to the path profiler, which reports it as *"needs
    the publisher's own key"* -- sending the reader after a file that does not
    exist and hiding the thing that actually matters: this source cannot be
    rotated in a fold table at all, and WaveFake is ~117 k more files from the
    same single speaker (`grouping_report`'s own caveat).
    """
    keys = pd.Series([LJ_SPEAKER] * len(files),
                     index=files["file_id"], dtype="object")
    return GroupKeys(source.name, SINGLE,
                     "one speaker, by the publisher's description; no key can "
                     "split it", keys)


def _wavefake(cfg: EdaConfig, source: SourceSpec, files: pd.DataFrame) -> GroupKeys:
    """The **voice** behind each vocoder directory -- two speakers, not ten.

    Measured on disk, 134,266 files in ten directories:

    * `ljspeech_*` x 7, 13,100 each -- LJSpeech's 13,100 utterances re-vocoded,
      stems `LJ001-0001_gen`;
    * `common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech`, 32,566 --
      Common Voice *prompts* spoken in **LJSpeech's voice** (the directory name
      says so, and the stems `gen_0...` carry no utterance id at all);
    * `jsut_*` x 2, 5,000 each -- JSUT's single Japanese speaker, stems
      `BASIC5000_0001_gen`.

    🔴 So nine of the ten directories are one speaker, and the key they get is
    `ljspeech`'s own -- `LJ_SPEAKER`, shared across the source boundary. Group
    by the vocoder instead and `LJ001-0001` sits in seven groups at once, beside
    the real recording in `ljspeech`: the model validates on an utterance it
    trained on in eight other forms. The vocoder is a stratification axis, the
    speaker is the grouping one -- the same distinction FakeMusicCaps needed.

    ⚠️ 2 atoms from 134,266 files. That is the honest count, and it is why
    `wavefake` cannot supply fold rotation however large it is.
    """
    def key(file_id: str) -> str:
        rel = PurePosixPath(relpath_of(file_id))
        # `generated_audio/<vocoder>/<file>.wav`; the vocoder directory is the
        # one component that names the voice.
        vocoder = next((p for p in rel.parts if p.startswith(("ljspeech_", "jsut_"))
                        or "ljspeech" in p), None)
        if vocoder is None:
            raise ValueError(
                f"source {source.name!r}: {rel.as_posix()!r} is under no "
                f"recognised vocoder directory, so the voice behind it is "
                f"unknown. WaveFake's ten directories are named for the voice "
                f"they used; a new one needs deciding, not defaulting")
        return JSUT_SPEAKER if vocoder.startswith("jsut_") else LJ_SPEAKER

    keys = pd.Series([key(f) for f in files["file_id"]],
                     index=files["file_id"], dtype="object")
    keys = _check_total(source.name, files, keys, "the vocoder directory")
    return GroupKeys(source.name, PUBLISHER,
                     "the voice named by the vocoder directory: LJSpeech's "
                     "speaker (shared with `ljspeech`) or JSUT's", keys)


#: Source name -> provider. A source absent from this table falls back to path
#: depth and is reported with kind `path`, so nothing is silent either way.
PROVIDERS: dict[str, Provider] = {
    "sonics": _sonics,
    "fakemusiccaps": _fakemusiccaps,
    "musan-music": _musan_music,
    "rirs-isotropic-noise": _rirs_isotropic_noise,
    "ljspeech": _ljspeech,
    "wavefake": _wavefake,
}

#: Every runnable source without a provider, and why. 🔴 An entry here is a
#: **decision**, not a gap: either the publisher's key already *is* the path, or
#: the source is not on disk yet and the choice waits for it. A source in
#: neither this table nor `PROVIDERS` would fall back to path depth with nothing
#: anywhere saying that was chosen -- which is the state all thirteen were in
#: before docs/EDA/09 step 1. `tests/test_eda_groupkeys.py` refuses it.
NO_PROVIDER = {
    "cfad-fake": "the path is CFAD's key: `<split>/<generator>/`",
    "cfad-real": "the path is CFAD's key: `<split>/<corpus>/`",
    "mlaad": "MLAAD's path is `fake/<language>/<generator>/`",
    "zeroth-korean": "speaker directories",
    "fma": "FMA's numbered shard directories -- ⚠️ not artist; its own "
           "`tracks.csv` has `artist`/`album` and is the better key once the "
           "metadata archive is fetched",
    "compspoof-env-bonafide": "CompSpoof's environment directories",
    "musan-noise": "collection only (`free-sound`, `sound-bible`); MUSAN "
                   "publishes no per-recording key for noise, so 2 atoms is "
                   "the count rather than a gap",
    "musan-speech": "collection only (`librivox`, `us-gov`); the librivox "
                    "ANNOTATIONS give gender and language, never a speaker id",
    # ⚠️ Not on disk. Named here so the absence is visible rather than silent;
    # each needs deciding when it lands, and the first is not a small decision.
    "common-voice-en": "NOT FETCHED. Its key is `client_id` in `validated.tsv`, "
                       "never in the path (docs/EDA/01 A3)",
    "common-voice-ko": "NOT FETCHED. `client_id` in `validated.tsv`, as above",
    "libritts-r": "NOT FETCHED. Speaker directories, like Zeroth's",
    "mtg-jamendo": "NOT FETCHED. Its allowlist ids are paths `14/214.mp3`; the "
                   "artist key is in the publisher's TSV",
    "compspoof-v2": "NOT FETCHED. The `mixed` partition, unprobed",
}

def path_key(files: pd.DataFrame, min_groups: int) -> tuple[pd.Series, int | None]:
    """The fallback key: the directory prefix at the shallowest usable depth.

    🔴 A **prefix**, never the name at that depth. `analyze/grouping.py` states
    the trap and this is where it would be walked into: with `train/spk1` and
    `test/spk1`, keying on `spk1` merges two speakers the publisher kept apart.

    The depth is the shallowest whose distinct prefixes reach `min_groups`
    without being the leaf -- `grouping_report`'s own heuristic, applied to the
    same frame it profiles. When no depth qualifies, the deepest non-leaf one is
    used and the source simply reports few groups, which is the truth about it.
    """
    parts = [tuple(relpath_of(f).split("/")[:-1]) for f in files["file_id"]]
    deepest = max((len(p) for p in parts), default=0)
    n = len(files)

    def prefixes(d: int) -> pd.Series:
        return pd.Series(["/".join(p[:d + 1]) if p else "" for p in parts],
                         index=files["file_id"], dtype="object")

    chosen: int | None = None
    for d in range(deepest):
        if min_groups <= prefixes(d).nunique() < n:
            chosen = d
            break
    if chosen is None:
        # No depth reaches the floor. Take the deepest that is not the leaf, so
        # the reported count is the best the tree can do rather than 1.
        for d in reversed(range(deepest)):
            if prefixes(d).nunique() < n:
                chosen = d
                break
    if chosen is None:
        # A flat directory, or one file per directory. One group: merging is the
        # safe direction (module docstring), and the count says so plainly.
        return pd.Series([""] * n, index=files["file_id"], dtype="object"), None
    return prefixes(chosen), chosen


def group_keys_for(cfg: EdaConfig, source: SourceSpec, files: pd.DataFrame, *,
                   min_groups: int = 6) -> GroupKeys:
    """This source's keys: the publisher's where one exists, the path's otherwise.

    ⚠️ Every row gets a key either way. An earlier revision left the `path`
    sources null, which read as tidy and made the column unusable for the one
    thing it is for: `cfad-fake`'s eleven generators -- the case docs/EDA/09
    step 1 is *about* -- live in the path, so a key that only covers
    publisher-keyed sources cannot stratify the draw that motivated it.
    """
    provider = PROVIDERS.get(source.name)
    if provider is not None:
        return provider(cfg, source, files)
    keys, depth = path_key(files, min_groups)
    # 🔴 Namespaced. A bare `train/` or `melgan/` collides with the same
    # directory name under another publisher and merges two unrelated sets of
    # files into one fold group. Providers do their own namespacing, which is
    # what lets `wavefake` and `ljspeech` deliberately share `LJ_SPEAKER`.
    keys = keys.map(lambda k: f"{source.name}/{k}" if k else source.name)
    where = ("a flat directory: one group" if depth is None
             else f"path prefix at depth {depth}")
    return GroupKeys(source.name, PATH,
                     f"{where} -- {NO_PROVIDER.get(source.name, 'no provider')}",
                     keys)


def attach_group_keys(cfg: EdaConfig, files: pd.DataFrame) -> pd.DataFrame:
    """Add `group_key` and `group_key_kind` to a census frame.

    Rows from a source with no provider keep a null key and kind `path`; they
    are not dropped and not guessed at. Called by `consolidate`, so the column
    is refreshed by re-consolidating -- no re-probe, because nothing here reads
    audio.
    """
    out = files.copy()
    out[KEY_COLUMN] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out[KIND_COLUMN] = PATH
    for name, rows in out.groupby("source_name"):
        try:
            source = cfg.source(str(name))
        except Exception:
            # A census row whose source is no longer declared. Left unkeyed
            # rather than raising: `consolidate` already refuses the cases that
            # matter, and a stale row is the duplicate check's business.
            continue
        result = group_keys_for(cfg, source, rows,
                                min_groups=cfg.gates.min_groups_per_role)
        out.loc[rows.index, KEY_COLUMN] = (
            result.keys.loc[rows["file_id"]].to_numpy())
        out.loc[rows.index, KIND_COLUMN] = result.kind
    return out


def key_report(cfg: EdaConfig, files: pd.DataFrame) -> pd.DataFrame:
    """Per source: the kind, the key count, and the evidence it was read from.

    Runs the providers rather than reading the census columns, so the evidence
    string has exactly one home -- the provider that produced it, or
    `NO_PROVIDER` for a source that has none. Nothing here decodes.
    """
    rows = []
    for name, g in files.groupby("source_name"):
        try:
            source = cfg.source(str(name))
        except Exception:
            continue
        result = group_keys_for(cfg, source, g)
        rows.append({"source_name": name, "files": int(len(g)),
                     "group_key_kind": result.kind, "keys": result.n_groups,
                     "evidence": result.evidence})
    return pd.DataFrame(rows).sort_values("source_name").reset_index(drop=True)
