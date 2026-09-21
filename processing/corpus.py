"""OFF-1..OFF-4 -- the real corpus as one validated manifest, from the EDA's
tables (docs/processing/03 §3, §4.1).

Inputs are the M tier (``eda/out/<partition>/files.parquet``: every probed
file), the S tier where it exists (``signal.parquet`` + ``vad_extra.parquet``:
the usable-span evidence), and the EDA's shared artifacts (the G-EDA6 work
list, the byte-identical duplicate groups, the fma licence ledger + metadata).
Labels are never typed by hand: a component row's come from ``POOL_LABELS``,
a whole-file row's from ``CELL_TABLE``.

The per-source rules are docs/processing/03 §4.1's, each derived from the
path or the publisher's metadata and each asserted after the build
(``check_rules``). Three departures from the table as written, measured:

* ``fma``'s ``speaker_ref_id`` is the **artist id from ``tracks.csv``**
  (2,309 atoms), not the 156 path buckets the EDA's ``group_key`` used --
  ``fma_small/000/`` is a numbering bucket, not an artist.
* ``mlaad`` has **205** generator directories, not 175.
* WaveFake's JSUT and Common-Voice subsets get their own families
  (``wf_jsut_<vocoder>``, ``wf_cv_fastspeech2_pwg``), as D-15 says.
* ``fakemusiccaps``'s ``speaker_ref_id`` is the parent clip **scoped to the
  generator**: shared across generators it fused the five families into one
  fold atom.

Paths are written **relative to the corpus root** (``<root>/interim/...`` or
``<root>/raw/...``), because a source's stage is a property of the EDA config
and the renderer joins one ``root`` to every path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from training.manifest import POOL_IS_FAKE, POOL_LABELS, REQUIRED_COLUMNS, validate_manifest
from training.registries import FILTER
from training.spec import CELL_TABLE, is_fake_cell

__all__ = ["EXTRA_COLUMNS", "SCHEME_VERSION", "BuildInputs", "apply_worklist", "assign_keys",
           "build_manifest", "check_rules", "dup_groups", "licence_verdicts", "verdicts"]

SCHEME_VERSION = "strategy-v1"

#: Columns beyond ``training.manifest.REQUIRED_COLUMNS``. ``corpus`` is the
#: EDA's source name (``cfad-real``, ``wavefake``, ...): the manifest's own
#: ``source_name`` is the publisher's ATOM -- the family for a fake row, the
#: sub-corpus / speaker / artist for a real one -- because
#: ``training.folds`` unions every row sharing a ``source_name`` into one
#: indivisible fold atom (docs/validation/01 §1: "track / artist / speaker
#: granularity"), and at corpus granularity the 278k rows are 13 atoms and no
#: fold is feasible. ``noise_has_speech``
#: is D-14's `restrict_noise` flag (DRAW-5 reads it); ``licence_verdict`` is
#: the fma ledger's word for the row (`allow` | `derivatives_barred`); the
#: rest are provenance a reviewer wants beside the row.
EXTRA_COLUMNS: tuple[str, ...] = ("corpus", "noise_has_speech", "licence_verdict", "stage",
                                  "reassigned_from")

_LABELS = ("label_voice_present", "label_music_present", "label_voice_fake", "label_music_fake")

#: D-15: WaveFake's eight LJSpeech vocoder subsets fall into three measured
#: correlation families (02 §10).
WAVEFAKE_FAMILY = {
    "ljspeech_melgan": "wf_melgan", "ljspeech_melgan_large": "wf_melgan",
    "ljspeech_full_band_melgan": "wf_gan", "ljspeech_hifiGAN": "wf_gan",
    "ljspeech_parallel_wavegan": "wf_gan", "ljspeech_waveglow": "wf_gan",
    "ljspeech_multi_band_melgan": "wf_mb_melgan",
    "jsut_multi_band_melgan": "wf_jsut_multi_band_melgan",
    "jsut_parallel_wavegan": "wf_jsut_parallel_wavegan",
    "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech": "wf_cv_fastspeech2_pwg",
}

#: D-15: SONICS' five generators are two families.
SONICS_FAMILY = {"chirp-v2-xxl-alpha": "suno_chirp", "chirp-v3": "suno_chirp",
                 "chirp-v3.5": "suno_chirp", "udio-30s": "udio", "udio-120s": "udio"}


@dataclass(frozen=True)
class BuildInputs:
    """Everything the build reads, so a test can hand it frames."""

    files: pd.DataFrame                       # the M tier, every partition
    worklist: pd.DataFrame                    # G-EDA6's list (D-14)
    duplicates: pd.DataFrame                  # byte-identical rows (sha256 groups)
    signal: pd.DataFrame | None = None        # S tier + vad_extra: decode failures
    #: tracks.csv: index = track id, columns (artist, id) and (track, license)
    fma_tracks: pd.DataFrame | None = None
    fma_allow: set[int] = field(default_factory=set)
    stages: Mapping[str, str] = field(default_factory=dict)   # source_name -> interim | raw
    component_floor_s: float = 4.0            # OFF-4 usable_duration, the sampler's floor


# --------------------------------------------------------------------------- #
# §4.1 -- the split keys, per source


def _rx(series: pd.Series, pattern: str) -> pd.Series:
    return series.str.extract(pattern)[0]


def assign_keys(files: pd.DataFrame, fma_tracks: pd.DataFrame | None = None) -> pd.DataFrame:
    """``artifact_family``, ``speaker_ref_id``, ``domain_key``, ``pair_id`` per
    docs/processing/03 §4.1, from the path and the publisher's metadata."""
    df = files.copy()
    src, path, gk = df["source_name"], df["path"], df["group_key"]
    fam = pd.Series(pd.NA, index=df.index, dtype="object")
    spk = gk.astype("object").copy()
    dom = pd.Series(pd.NA, index=df.index, dtype="object")
    pair = pd.Series(pd.NA, index=df.index, dtype="object")

    m = src == "cfad-fake"
    voc = _rx(path[m], r"/fake_clean/([^/]+)/")
    ssb = _rx(path[m], r"/(SSB\d+)_")
    fam[m] = "cfad/" + voc
    spk[m] = ssb.str[:7]                                  # SSB0354 of SSB03540001
    dom[m] = "cfad-fake|cfad/" + voc
    pair[m] = ("cfad:" + ssb).where(ssb.notna())

    m = src == "cfad-real"
    ssb = _rx(path[m], r"/(SSB\d+)\.")
    pair[m] = ("cfad:" + ssb).where(ssb.notna())

    m = src == "ljspeech"
    lj = _rx(path[m], r"/(LJ\d{3}-\d{4})\.")
    spk[m] = "ljspeech_LJ"
    pair[m] = "lj:" + lj

    m = src == "wavefake"
    subset = _rx(path[m], r"/generated_audio/([^/]+)/")
    lj = _rx(path[m], r"/(LJ\d{3}-\d{4})")
    fam[m] = subset.map(WAVEFAKE_FAMILY)
    spk[m] = np.where(subset.str.startswith("ljspeech_"), "ljspeech_LJ",
                      np.where(subset.str.startswith("jsut_"), "jsut", "common_voice"))
    dom[m] = "wavefake|" + subset
    pair[m] = ("lj:" + lj).where(lj.notna() & subset.str.startswith("ljspeech_"))

    m = src == "mlaad"
    parts = path[m].str.extract(r"/payload/fake/([^/]+)/([^/]+)/")
    fam[m] = "mlaad/" + parts[1]
    spk[m] = parts[0] + "/" + parts[1]
    dom[m] = "mlaad|" + parts[1]

    m = src == "fakemusiccaps"
    gen = _rx(path[m], r"/zenodo-15063698/([^/]+)/")
    fam[m] = "fakemusiccaps/" + gen
    dom[m] = "fakemusiccaps|" + gen
    # Caveat: NOT the bare parent clip §4.1 names. Every generator renders every
    # caption, so a clip key shared across generators unions the five families
    # into one fold atom and no 5-fold is feasible for the music head. The
    # generators share a caption, not a recording; the key is scoped to the
    # generator.
    spk[m] = "fakemusiccaps/" + gen + "/" + gk[m].str.replace("fakemusiccaps/", "", regex=False)

    m = src == "sonics"
    gen = gk[m].str.replace("sonics/", "", regex=False)
    fam[m] = gen.map(SONICS_FAMILY)
    spk[m] = gk[m]
    dom[m] = "sonics|" + gen

    m = src == "fma"
    if fma_tracks is not None:
        tid = pd.to_numeric(_rx(path[m], r"/(\d+)\.mp3$"), errors="coerce")
        artist = fma_tracks[("artist", "id")]
        spk[m] = ("fma_artist_" + tid.map(artist).astype("Int64").astype(str)).where(
            tid.map(artist).notna())

    df["artifact_family"], df["speaker_ref_id"] = fam, spk
    df["domain_key"], df["pair_id"] = dom, _both_sides(pair, src)
    return df


def _both_sides(pair: pd.Series, source: pd.Series) -> pd.Series:
    """A pair is a claim that the same utterance exists on BOTH sides. cfad-fake
    has 16,553 utterance ids and cfad-real 7,900 of them; an id with one side
    only is not a pair (§4.1: "where the id exists in cfad-fake") -- and a
    twin a filter dropped leaves no pair either, so this runs again after the
    drops."""
    sides = pd.DataFrame({"pair": pair, "src": source}).dropna().groupby("pair")["src"].nunique()
    return pair.where(pair.map(sides).fillna(0) >= 2)


# --------------------------------------------------------------------------- #
# OFF-3 -- duplicate groups


def dup_groups(files: pd.DataFrame, duplicates: pd.DataFrame) -> pd.Series:
    """``file_id -> dup_group`` for the byte-identical groups (``sha:<hash>``),
    NA elsewhere. CompSpoof's 292 shared parents are already one
    ``speaker_ref_id`` (its parent recording), and WaveFake's nested copy of
    the Common-Voice subset is not in the M tier at all."""
    out = pd.Series(pd.NA, index=files["file_id"], dtype="object")
    if len(duplicates):
        by_sha = duplicates.groupby("sha256")["file_id"].apply(list)
        for sha, ids in by_sha.items():
            if len(ids) > 1:
                out.loc[out.index.intersection(ids)] = f"sha:{sha[:16]}"
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# OFF-4 -- the fma licence, and the verdict sidecar


def licence_verdicts(files: pd.DataFrame, fma_tracks: pd.DataFrame | None,
                     fma_allow: set[int]) -> pd.Series:
    """``allow`` (in the ledger), ``derivatives_barred`` (an ND grant),
    ``deny`` (anything else: the ledger's own rule is that unknown is deny);
    NA for every non-fma row."""
    from scripts.filter_track_licences import classify

    out = pd.Series(pd.NA, index=files.index, dtype="object")
    m = files["source_name"] == "fma"
    if not m.any():
        return out
    tid = pd.to_numeric(_rx(files.loc[m, "path"], r"/(\d+)\.mp3$"), errors="coerce")
    verdict = pd.Series("deny", index=tid.index, dtype="object")
    verdict[tid.isin(fma_allow)] = "allow"
    if fma_tracks is not None:
        title = tid.map(fma_tracks[("track", "license")])
        cls = title.map(lambda t: classify(t) if isinstance(t, str) else "deny")
        verdict[(verdict != "allow") & (cls == "derivatives_barred")] = "derivatives_barred"
    out[m] = verdict
    return out


def verdicts(files: pd.DataFrame, signal: pd.DataFrame | None, licence: pd.Series,
             worklist: pd.DataFrame, component_floor_s: float) -> pd.DataFrame:
    """The verdict sidecar: one row per (file, filter). Filters are the
    FILTER registry's, applied per file over its own evidence; the licence and
    the G-EDA6 `degenerate` action are the two offline verdicts beside them.

    Critical: `usable_duration` judges every row on the M tier's probed
    ``duration_s`` -- the quantity the sampler's floor reads -- and NOT on the
    S tier's valid span. The span exists for the 17 % of rows the draw
    sampled, and a rule that read it where it existed made a row's fate depend
    on whether the draw had measured it. The valid-span evidence is a separate
    question (F3-2, deferred) and is not a filter here.
    """
    decoded = signal.set_index("file_id")["signal_ok"].to_dict() if signal is not None else {}
    corruption = FILTER.build("corruption")
    usable = FILTER.build("usable_duration", {"min_seconds": float(component_floor_s)})
    degenerate = set(worklist.loc[worklist["action"] == "degenerate", "file_id"])
    rows = []
    for i, r in enumerate(files.to_dict(orient="records")):
        fid = r["file_id"]
        quality = {"decode_ok": bool(r.get("probe_ok", True)) and decoded.get(fid, True),
                   "duration_s": r["duration_s"]}
        for name, fn in (("corruption", corruption), ("usable_duration", usable)):
            v = fn(r, quality)
            rows.append({"file_id": fid, "filter": name, "verdict": v.action, "reason": v.reason,
                         "threshold_version": (f"usable_duration:min_seconds={component_floor_s}"
                                               if name == "usable_duration" else "corruption:v1")})
        lic = licence.iloc[i]
        if pd.notna(lic):
            rows.append({"file_id": fid, "filter": "licence",
                         "verdict": "drop" if lic == "deny" else "keep",
                         "reason": str(lic), "threshold_version": "fma_allow.csv:2026-09-08"})
        if fid in degenerate:
            rows.append({"file_id": fid, "filter": "label_evidence", "verdict": "drop",
                         "reason": "G-EDA6 degenerate: normal length, no speech at all",
                         "threshold_version": "reassign:VOICE_EVIDENCE_RATIO=0.20"})
    return pd.DataFrame(rows, columns=["file_id", "filter", "verdict", "reason",
                                       "threshold_version"])


# --------------------------------------------------------------------------- #
# OFF-2 -- D-14's actions


def apply_worklist(manifest: pd.DataFrame, worklist: pd.DataFrame) -> pd.DataFrame:
    """`reassign_cell` rows become whole-file rows of their target cell (a real
    target loses family and domain); `restrict_noise` rows stay in pool E
    flagged ``noise_has_speech``; `unevidenceable` / `sparse_real` rows stay
    with ``label_confidence = reported``. `degenerate` rows are the verdict
    sidecar's to drop, not this function's."""
    m = manifest.set_index("file_id")
    wl = worklist[worklist["file_id"].isin(m.index)].set_index("file_id")
    m["noise_has_speech"] = False
    m["reassigned_from"] = pd.NA
    for fid, row in wl[wl["action"] == "reassign_cell"].iterrows():
        cell = int(row["target_cell"])
        m.at[fid, "reassigned_from"] = m.at[fid, "pool"]
        m.at[fid, "row_kind"] = "whole_file"
        m.at[fid, "cell"] = cell
        m.at[fid, "pool"] = pd.NA
        for col, value in zip(_LABELS, CELL_TABLE[cell]):
            m.at[fid, col] = value
        if not is_fake_cell(cell):
            m.at[fid, "artifact_family"] = pd.NA
            m.at[fid, "domain_key"] = pd.NA
    noisy = wl.index[wl["action"] == "restrict_noise"]
    m.loc[noisy, "noise_has_speech"] = True
    weak = wl.index[wl["action"].isin(["unevidenceable", "sparse_real"])]
    m.loc[weak, "label_confidence"] = "reported"
    return m.reset_index()


# --------------------------------------------------------------------------- #
# the build


def build_manifest(inp: BuildInputs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """``(manifest, verdict sidecar, report)``. The manifest is validated and
    every §4.1 rule is asserted; rows any filter drops are absent from it and
    present in the sidecar."""
    files = inp.files.reset_index(drop=True).copy()
    keyed = assign_keys(files, inp.fma_tracks)
    licence = licence_verdicts(keyed, inp.fma_tracks, inp.fma_allow)
    side = verdicts(keyed, inp.signal, licence, inp.worklist, inp.component_floor_s)

    comp = keyed["row_kind"] == "component"
    cell = pd.to_numeric(keyed["cell"], errors="coerce")
    labels = [POOL_LABELS[p] if pd.notna(p) else CELL_TABLE[int(c)]
              for p, c in zip(keyed["pool"].where(comp), cell)]
    m = pd.DataFrame({
        "file_id": keyed["file_id"],
        "path": [f"{inp.stages.get(s, 'interim')}/{p}"
                 for s, p in zip(keyed["source_name"], keyed["path"])],
        "sha256": keyed["sha256"], "row_kind": keyed["row_kind"],
        "pool": keyed["pool"].where(comp),
        "cell": cell.where(~comp).astype("Int64"),
        "duration_s": keyed["duration_s"].astype(float),
        "orig_sr": keyed["orig_sr"], "orig_channels": keyed["orig_channels"],
        "container": keyed["container"],
    })
    for i, col in enumerate(_LABELS):
        m[col] = pd.array([lab[i] for lab in labels], dtype="Int64")
    fake_row = (comp & keyed["pool"].map(POOL_IS_FAKE).fillna(False).astype(bool)) | (
        ~comp & cell.map(lambda c: pd.notna(c) and is_fake_cell(int(c))).fillna(False))
    m["artifact_family"] = keyed["artifact_family"].where(fake_row)
    m["corpus"] = keyed["source_name"]
    # docs/validation/01 §1: source_name at the publisher's granularity. A fake
    # row's atom is its family; a real row's is its group key (cfad-real's
    # sub-corpus, zeroth's speaker, musan's partition, CompSpoof's parent) --
    # except fma, whose key is the artist (its path buckets are numbering).
    real_key = keyed["speaker_ref_id"].where(keyed["source_name"] == "fma", keyed["group_key"])
    m["source_name"] = keyed["artifact_family"].where(fake_row, real_key)
    m["speaker_ref_id"] = keyed["speaker_ref_id"]
    m["pair_id"] = keyed["pair_id"]
    m["dup_group"] = dup_groups(keyed, inp.duplicates).to_numpy()
    m["domain_key"] = keyed["domain_key"].where(fake_row)
    m["slice"] = "train"
    m["fold"] = pd.array([pd.NA] * len(m), dtype="Int64")
    m["scheme_version"] = SCHEME_VERSION
    m["validity_mask_ref"] = pd.NA
    m["label_confidence"] = np.where(fake_row, "exact", "reported")
    m["aug_strength"] = 1.0                                       # D-17
    m["licence_verdict"] = licence.to_numpy()
    m["stage"] = keyed["source_name"].map(lambda s: inp.stages.get(s, "interim"))

    m = apply_worklist(m, inp.worklist)
    dropped = side[side["verdict"] == "drop"].groupby("file_id")["filter"].apply(list)
    m = m[~m["file_id"].isin(dropped.index)].reset_index(drop=True)
    m["pair_id"] = _both_sides(m["pair_id"], m["corpus"])
    m["cell"] = m["cell"].astype("Int64")
    m = m[list(REQUIRED_COLUMNS) + list(EXTRA_COLUMNS)]
    validate_manifest(m)
    check_rules(m)
    report = {
        "rows": int(len(m)),
        "dropped": {k: int(v) for k, v in
                    side[side["verdict"] == "drop"]["filter"].value_counts().items()},
        "row_kind": m["row_kind"].value_counts().to_dict(),
        "pool": m["pool"].value_counts(dropna=True).to_dict(),
        "cell": {int(k): int(v) for k, v in m["cell"].value_counts(dropna=True).items()},
        "families": int(m["artifact_family"].nunique()),
        "hours": {k: round(float(v) / 3600, 1) for k, v in m.groupby(
            m["pool"].fillna("cell" + m["cell"].astype(str)))["duration_s"].sum().items()},
        "noise_has_speech": int(m["noise_has_speech"].sum()),
        "licence": m.loc[m["corpus"] == "fma", "licence_verdict"].value_counts().to_dict(),
        "source_atoms": int(m["source_name"].nunique()),
        "scheme_version": SCHEME_VERSION,
    }
    return m, side, report


def check_rules(m: pd.DataFrame) -> None:
    """Every §4.1 rule, as an assertion. ``validate_manifest`` proves the
    schema; this proves the corpus."""
    def bad(mask: pd.Series, what: str) -> None:
        if mask.any():
            raise AssertionError(f"{int(mask.sum())} row(s) {what}: "
                                 f"{m.loc[mask, 'file_id'].head(3).tolist()}")
    comp = m["row_kind"] == "component"
    for pool in ("A", "B", "C", "D", "E"):
        if not (m["pool"] == pool).any():
            raise AssertionError(f"pool {pool} is empty -- validate_manifest does not check this")
    real_comp = comp & ~m["pool"].map(POOL_IS_FAKE).fillna(False).astype(bool)
    bad(real_comp & m["artifact_family"].notna(), "real components carrying an artifact_family")
    bad(real_comp & m["domain_key"].notna(), "real components carrying a domain_key")
    bad(m["speaker_ref_id"].isna(), "without a speaker_ref_id")
    wf = m["corpus"] == "wavefake"
    bad(wf & ~m["artifact_family"].isin(set(WAVEFAKE_FAMILY.values())),
        "wavefake outside D-15's families")
    so = m["corpus"] == "sonics"
    bad(so & ~m["artifact_family"].isin({"suno_chirp", "udio"}), "sonics outside D-15's families")
    bad(so & (m["row_kind"] != "whole_file"), "sonics not whole_file")
    # pairs resolve on both sides
    pairs = m.dropna(subset=["pair_id"]).groupby("pair_id")["corpus"].nunique()
    lonely = pairs[pairs < 2]
    if len(lonely):
        raise AssertionError(f"{len(lonely)} pair_id(s) with one side only, e.g. "
                             f"{lonely.index[:3].tolist()}")
    bad(comp & m["noise_has_speech"] & (m["pool"] != "E"), "noise_has_speech outside pool E")
    bad(m["licence_verdict"].eq("deny"), "with a denied licence still present")
    bad((m["corpus"] == "fma") & m["licence_verdict"].isna(),
        "fma rows without a licence verdict")
    bad(m["source_name"].isna(), "without a source_name atom")
    fake = m["artifact_family"].notna()
    bad(fake & (m["source_name"] != m["artifact_family"]),
        "fake rows whose source_name atom is not their family")
    bad(m["aug_strength"] != 1.0, "with aug_strength != 1.0 (D-17)")


# --------------------------------------------------------------------------- #
# reading the EDA's outputs


def load_inputs(eda_out: Path, corpus_root: Path, eda_cfg: Any,
                component_floor_s: float) -> BuildInputs:
    """The real inputs, from the EDA config's partitions and the corpus tree."""
    from eda.driver import load_signal

    files = pd.concat([pd.read_parquet(eda_out / p / "files.parquet")
                       for p in eda_cfg.partitions()
                       if (eda_out / p / "files.parquet").exists()], ignore_index=True)
    shared = eda_out / "_shared"
    worklist = pd.read_parquet(shared / "reassignment_worklist.parquet")
    dup_path = shared / "duplicates.parquet"
    duplicates = pd.read_parquet(dup_path) if dup_path.exists() else pd.DataFrame(
        columns=["file_id", "sha256"])
    signal = load_signal(eda_cfg, with_extra=True)
    meta = corpus_root / "interim" / "fma" / "fma_small" / "fma_metadata" / "tracks.csv"
    tracks = pd.read_csv(meta, index_col=0, header=[0, 1]) if meta.exists() else None
    allow_path = corpus_root / "interim" / "_licences" / "fma_allow.csv"
    allow = set(pd.read_csv(allow_path)["id"].astype(int)) if allow_path.exists() else set()
    stages = {s.name: s.stage for s in eda_cfg.sources}
    return BuildInputs(files=files, worklist=worklist, duplicates=duplicates, signal=signal,
                       fma_tracks=tracks, fma_allow=allow, stages=stages,
                       component_floor_s=component_floor_s)


def write_outputs(out_dir: Path, manifest: pd.DataFrame, side: pd.DataFrame,
                  report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    side.to_parquet(out_dir / "verdict.parquet", index=False)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

