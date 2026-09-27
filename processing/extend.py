"""strategy-v3: the strategy-v2 manifest plus the first run's new corpora
(docs/training/07 §1), and a `lang` column on every row.

strategy-v2 was built from the EDA's tiers (`processing.corpus.build_manifest`),
which the new corpora do not have, and re-running the EDA would cost hours the
deadline does not have. So v3 is v2's rows **unchanged** (every key, label and
filter decision kept) plus rows for:

| corpus | pool | from |
|---|---|---|
| `ko-synth` | B (fake voice, ko) | `interim/ko-synth/<family>/metadata.csv` (07 D-a) |
| `sonics-sep` | D (fake music) | `interim/sonics-sep/metadata.csv` (D-b) |
| `realmusic-sep` | C (real music) | `interim/realmusic-sep/metadata.csv` (D-b) |
| `libritts-r` | A (real voice, en) | a directory listing (D-c) |
| `common-voice-ko` | A (real voice, ko) | the release's `validated.tsv` (D-c) |

The keys a fold atom is made of, and why:

* A separated real track keeps its SOURCE row's `speaker_ref_id` and
  `source_name`: the stem and the original must land in one fold.
* A separated SONICS stem is family `sonics-sep/<version>` with its own
  speaker key per version: SONICS' whole files are never drawn (f8 = 1), and
  sharing their key would fuse every version into two atoms.
* A Korean clone's speaker key is `<family>/<prompt speaker>`, NOT the Zeroth
  speaker: a family's files span many prompt speakers, so sharing the Zeroth
  key would union every cloning family with every Zeroth speaker into one
  indivisible atom. The prompt speaker is kept in `prompt_speaker`.

New rows below `component_floor_s` or that fail to open are dropped and
counted in the report; there is no EDA signal tier for them.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from processing.corpus import EXTRA_COLUMNS, check_rules
from training.manifest import POOL_LABELS, REQUIRED_COLUMNS, validate_manifest

SCHEME_VERSION = "strategy-v3"
NEW_COLUMNS = ("lang", "prompt_speaker")

__all__ = ["NEW_COLUMNS", "SCHEME_VERSION", "extend_manifest", "lang_of"]


# --------------------------------------------------------------------------- #
# lang for the v2 rows


def lang_of(m: pd.DataFrame) -> pd.Series:
    """ko / en / zh / ja / <iso> for voice rows (pools A, B), None elsewhere."""
    out = pd.Series(None, index=m.index, dtype="object")
    corpus, fam = m["corpus"], m["artifact_family"].astype("object")
    fixed = {"zeroth-korean": "ko", "common-voice-ko": "ko", "ko-synth": "ko",
             "cfad-real": "zh", "cfad-fake": "zh", "ljspeech": "en",
             "musan-speech": "en", "libritts-r": "en"}
    for c, lg in fixed.items():
        out[corpus == c] = lg
    wf = corpus == "wavefake"
    out[wf] = np.where(fam[wf].fillna("").str.startswith("wf_jsut"), "ja", "en")
    ml = corpus == "mlaad"
    out[ml] = m.loc[ml, "path"].str.extract(r"/fake/([a-z]{2,3})/")[0]
    voice = m["pool"].isin(["A", "B"])
    return out.where(voice, None)


# --------------------------------------------------------------------------- #
# probing new files


def _probe(path: Path) -> dict[str, Any]:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return {"ok": True, "duration_s": float(info.frames) / info.samplerate,
                "orig_sr": float(info.samplerate), "orig_channels": float(info.channels),
                "sha256": h.hexdigest()}
    except Exception as exc:                              # noqa: BLE001 -- counted
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _probe_all(root: Path, rel_paths: list[str], workers: int) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda p: _probe(root / p), rel_paths))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the new corpora


@dataclass(frozen=True)
class NewCorpus:
    name: str
    pool: str
    #: rows with at least `path` (relative to the corpus root, stage dir
    #: included) and the key columns; see `_frame`.
    reader: Callable[[Path, pd.DataFrame], pd.DataFrame]


def _ko_synth(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    base = root / "interim" / "ko-synth"
    frames = []
    for meta in sorted(base.glob("*/metadata.csv")):
        d = pd.read_csv(meta)
        fam = meta.parent.name
        d["path"] = f"interim/ko-synth/{fam}/" + d["file"].astype(str)
        d["family"] = "ko-synth/" + fam
        prompt = d.get("prompt_speaker", pd.Series("", index=d.index)).fillna("").astype(str)
        d["prompt_speaker"] = prompt.where(prompt != "", None)
        d["speaker_ref_id"] = np.where(prompt != "", "ko-synth/" + fam + "/" + prompt,
                                       "ko-synth/" + fam)
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["path"])
    d = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "file_id": "ko-synth:" + d["path"].str.replace("interim/ko-synth/", "", regex=False),
        "path": d["path"], "artifact_family": d["family"],
        "source_name": d["family"], "speaker_ref_id": d["speaker_ref_id"],
        "domain_key": "ko-synth|" + d["family"].str.replace("ko-synth/", "", regex=False),
        "prompt_speaker": d["prompt_speaker"]})


def _kept(d: pd.DataFrame) -> pd.DataFrame:
    if "kept" in d.columns:
        d = d[d["kept"].astype(str).str.lower().isin(["true", "1"])]
    return d


def _sonics_sep(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    meta = root / "interim" / "sonics-sep" / "metadata.csv"
    if not meta.exists():
        return pd.DataFrame(columns=["path"])
    d = _kept(pd.read_csv(meta))
    ver = d["generator_version"].astype(str)
    fam = "sonics-sep/" + ver
    return pd.DataFrame({
        "file_id": "sonics-sep:" + d["file"].astype(str),
        "path": "interim/sonics-sep/" + d["file"].astype(str),
        "artifact_family": fam, "source_name": fam, "speaker_ref_id": fam,
        "domain_key": "sonics-sep|" + ver, "prompt_speaker": None})


def _realmusic_sep(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    meta = root / "interim" / "realmusic-sep" / "metadata.csv"
    if not meta.exists():
        return pd.DataFrame(columns=["path"])
    d = _kept(pd.read_csv(meta))
    src = v2.set_index("file_id")
    sid = d["source_file_id"].astype(str)
    missing = sorted(set(sid) - set(src.index))
    if missing:
        raise ValueError(f"realmusic-sep: {len(missing)} source_file_id(s) not in the v2 "
                         f"manifest, e.g. {missing[:3]}")
    return pd.DataFrame({
        "file_id": "realmusic-sep:" + d["file"].astype(str),
        "path": "interim/realmusic-sep/" + d["file"].astype(str),
        "artifact_family": None,
        "source_name": src.loc[sid, "source_name"].to_numpy(),
        "speaker_ref_id": src.loc[sid, "speaker_ref_id"].to_numpy(),
        "domain_key": None, "prompt_speaker": None})


def _libritts_r(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """``<stage>/libritts-r/.../<split>/<speaker>/<chapter>/<utt>.wav``."""
    hits = []
    for stage in ("interim", "raw"):
        base = root / stage / "libritts-r"
        if base.exists():
            hits = sorted(p for p in base.rglob("*.wav"))
            break
    if not hits:
        return pd.DataFrame(columns=["path"])
    rel = [str(p.relative_to(root)) for p in hits]
    spk = [p.parent.parent.name for p in hits]
    return pd.DataFrame({
        "file_id": ["libritts-r:" + r.split("libritts-r/", 1)[1] for r in rel],
        "path": rel, "artifact_family": None,
        "source_name": ["libritts-r/" + s for s in spk],
        "speaker_ref_id": ["libritts-r/" + s for s in spk],
        "domain_key": None, "prompt_speaker": None})


def _common_voice_ko(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """The release's ``validated.tsv``: ``client_id`` is the speaker."""
    tsvs = []
    for stage in ("interim", "raw"):
        base = root / stage / "common-voice-ko"
        if base.exists():
            tsvs = sorted(base.rglob("validated.tsv"))
            break
    if not tsvs:
        return pd.DataFrame(columns=["path"])
    tsv = tsvs[0]
    d = pd.read_csv(tsv, sep="\t", usecols=["client_id", "path"], quoting=3)
    clips = tsv.parent / "clips"
    rel_dir = str(clips.relative_to(root))
    spk = "common-voice-ko/" + d["client_id"].astype(str).str[:16]
    return pd.DataFrame({
        "file_id": "common-voice-ko:" + d["path"].astype(str),
        "path": rel_dir + "/" + d["path"].astype(str),
        "artifact_family": None, "source_name": spk, "speaker_ref_id": spk,
        "domain_key": None, "prompt_speaker": None})


# --------------------------------------------------------------------------- #
# round 2 (docs/training/09): readers for the synthesis tracks' contracts


def _metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _kept(pd.read_csv(path))


def _synth_families(root: Path, corpus: str, lang: str) -> pd.DataFrame:
    """`interim/<corpus>/<family>/metadata.csv` in round 1's ko-synth contract (+ kept)."""
    frames = []
    for meta in sorted((root / "interim" / corpus).glob("*/metadata.csv")):
        d = _metadata(meta)
        if d.empty:
            continue
        fam = meta.parent.name
        d["path"] = f"interim/{corpus}/{fam}/" + d["file"].astype(str)
        d["family"] = f"{corpus}/{fam}"
        prompt = d.get("prompt_speaker", pd.Series("", index=d.index)).fillna("").astype(str)
        d["prompt_speaker"] = prompt.where(prompt != "", None)
        d["speaker_ref_id"] = np.where(prompt != "", f"{corpus}/{fam}/" + prompt,
                                       f"{corpus}/{fam}")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["path"])
    d = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "file_id": corpus + ":" + d["path"].str.replace(f"interim/{corpus}/", "", regex=False),
        "path": d["path"], "artifact_family": d["family"], "source_name": d["family"],
        "speaker_ref_id": d["speaker_ref_id"],
        "domain_key": corpus + "|" + d["family"].str.split("/").str[-1],
        "prompt_speaker": d["prompt_speaker"], "lang": lang})


def _ko_synth2(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    return _synth_families(root, "ko-synth2", "ko")


def _ko_synth3(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """Round-3 Korean cloner families (docs/training/15 N1-ko), same contract as ko-synth2."""
    return _synth_families(root, "ko-synth3", "ko")


def _ko_synth2_extra(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """N5 (docs/training/15 §4): more ko-synth2 maskgct/seedvc with new prompt speakers. Same
    generator families as ko-synth2, so family, speaker and draw-domain keys are ko-synth2's;
    only file_id/path say ko-synth2-extra."""
    d = _synth_families(root, "ko-synth2-extra", "ko")
    if d.empty:
        return d
    for c in ("artifact_family", "source_name", "speaker_ref_id", "domain_key"):
        d[c] = d[c].str.replace("ko-synth2-extra", "ko-synth2", n=1, regex=False)
    return d


def _zh_synth(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    return _synth_families(root, "zh-synth", "zh")


def _emilia(root: Path, lang: str) -> pd.DataFrame:
    """Real in-the-wild speech. `speaker_ref_id` is Emilia's source key."""
    corpus = f"emilia-{lang}"
    # metadata_yodas2.csv: N3's second YODAS sample (docs/training/15 §4), disjoint by video
    parts = [_metadata(root / "interim" / corpus / n) for n in ("metadata.csv", "metadata_yodas2.csv")]
    parts = [x for x in parts if not x.empty]
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if d.empty:
        return pd.DataFrame(columns=["path"])
    spk = d["speaker_ref_id"].astype(str)
    return pd.DataFrame({
        "file_id": f"{corpus}:" + d["file"].astype(str),
        "path": f"interim/{corpus}/" + d["file"].astype(str),
        "artifact_family": None, "source_name": spk, "speaker_ref_id": spk,
        "domain_key": None, "prompt_speaker": None, "lang": lang})


def _emilia_ko(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """S3: real conversational Korean."""
    return _emilia(root, "ko")


def _emilia_en(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """Many-speaker in-the-wild English real (run-1 diagnosis F1/F2)."""
    return _emilia(root, "en")


def _en_synth2(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """English clones of Emilia-EN / LibriTTS-R speakers by the round-2 cloners."""
    return _synth_families(root, "en-synth2", "en")


def _en_synth3(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """Round-3 English cloner families (docs/training/15 N1-en), same contract as en-synth2."""
    return _synth_families(root, "en-synth3", "en")


def _en_synth2_extra(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """N5 (docs/training/15 §4): more hours of en-synth2's maskgct / seedvc with new prompt
    speakers. Same model and settings, so the rows keep en-synth2's family, draw domain and
    speaker keys (one family atom with the round-2 files); only file_id and path differ."""
    d = _synth_families(root, "en-synth2-extra", "en")
    if d.empty:
        return d
    for col in ("artifact_family", "source_name", "speaker_ref_id", "domain_key"):
        d[col] = d[col].str.replace("en-synth2-extra", "en-synth2", n=1, regex=False)
    return d


def _proc(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """S1: processed audio, label-preserving (DACON #417333 A1). A processed file
    keeps its SOURCE row's pool, generator family, speaker atom and language, so
    it shares a fold with its source. Its draw domain names the processing
    family: `domain_key` on a fake row, `source_name` on a real one."""
    d = _metadata(root / "interim" / "proc" / "metadata.csv")
    if d.empty:
        return pd.DataFrame(columns=["path"])
    src = v2.set_index("file_id")
    sid = d["source_file_id"].astype(str)
    missing = sorted(set(sid) - set(src.index))
    if missing:
        raise ValueError(f"proc: {len(missing)} source_file_id(s) not in the base manifest, "
                         f"e.g. {missing[:3]}")
    s = src.loc[sid]
    fam = d["family"].astype(str).to_numpy()
    pool = s["pool"].to_numpy()
    fake = np.isin(pool, ["B", "D"])
    base_src = s["source_name"].astype(str).to_numpy()
    base_dom = s["domain_key"].where(s["domain_key"].notna(), s["source_name"]).astype(str)
    return pd.DataFrame({
        "file_id": "proc:" + d["file"].astype(str),
        "path": "interim/proc/" + d["file"].astype(str),
        "pool": pool,
        "artifact_family": np.where(fake, s["artifact_family"].to_numpy(), None),
        # a fake row's source atom IS its family (check_rules); its own draw
        # domain comes from domain_key. A real row's source atom names the
        # processing family, which is its DOSS domain.
        "source_name": np.where(fake, s["artifact_family"].to_numpy(),
                                [f"{f}/{b}" for f, b in zip(fam, base_src)]),
        "speaker_ref_id": s["speaker_ref_id"].to_numpy(),
        "domain_key": np.where(fake, [f"{f}|{b}" for f, b in zip(fam, base_dom)], None),
        "prompt_speaker": None, "lang": s["lang"].to_numpy()})


def _ctrsvdd(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """S5: singing. Bona fide -> pool A, deepfake -> pool B; one singer's real and
    fake takes share a speaker atom, so they share a fold."""
    d = _metadata(root / "interim" / "ctrsvdd" / "metadata.csv")
    if d.empty:
        return pd.DataFrame(columns=["path"])
    fake = d["label"].astype(str).eq("deepfake").to_numpy()
    attack = d.get("attack", pd.Series("", index=d.index)).fillna("").astype(str)
    spk = "ctrsvdd/" + d["speaker"].astype(str)
    fam = "ctrsvdd/" + attack
    return pd.DataFrame({
        "file_id": "ctrsvdd:" + d["file"].astype(str),
        "path": d["file"].astype(str),
        "pool": np.where(fake, "B", "A"),
        "artifact_family": np.where(fake, fam, None),
        "source_name": np.where(fake, fam, spk),
        "speaker_ref_id": spk.to_numpy(),
        "domain_key": np.where(fake, "ctrsvdd|" + attack, None),
        "prompt_speaker": None,
        "lang": d.get("lang", pd.Series(None, index=d.index)).to_numpy()})


def _sing(root: Path, v2: pd.DataFrame, side: str) -> pd.DataFrame:
    """N2 (docs/training/15 §3): htdemucs VOCAL stems of songs, cut into
    vocal-active chunks (scripts/synth3/sing_*.py). A fake stem (SONICS,
    ACE-Step) is family `sing-fake/<generator>` with the speaker key of its
    source songs (`sonics/<version>`, `aisong-kr-en/acestep15`), so a stem and
    the whole song it came from share a fold atom. A real stem keeps its SOURCE
    song's artist key and publisher atom, as a realmusic-sep stem does: from
    the base manifest when the song is there, else from the metadata
    (fma_medium tracks, keyed the way processing.corpus keys FMA)."""
    corpus = f"sing-{side}"
    d = _metadata(root / "interim" / corpus / "metadata.csv")
    if d.empty:
        return pd.DataFrame(columns=["path"])
    lang = d["lang"].where(d["lang"].notna(), None).to_numpy()
    if side == "fake":
        ver = d["generator_version"].astype(str)
        fam = corpus + "/" + ver
        # The speaker key is also the draw's tile bucket, so it is cut to the real
        # side's size (a real stem's key is its artist: ~4 stems): one key per 3
        # source songs (~4-5 stems), never per version (hundreds of files, so a
        # fake singing slot tiled from ~14 songs vs ~5 on the real side: audit
        # I1c's voice_n_files). The family (source_name) is still one fold atom.
        # A SONICS stem never takes SONICS' `sonics/<version>` key: the whole
        # files are sealed in PROBE (strategy-v3), and that key would pull it in.
        song = d["source_file_id"].astype(str)
        rank = song.groupby(ver).rank(method="dense").astype(int) - 1
        root = d["speaker_ref_id"].astype(str)
        root = root.where(~root.str.startswith("sonics/"), fam)
        spk = root + "/s" + (rank // 3).map("{:04d}".format)
        return pd.DataFrame({
            "file_id": corpus + ":" + d["file"].astype(str),
            "path": f"interim/{corpus}/" + d["file"].astype(str),
            "artifact_family": fam, "source_name": fam,
            "speaker_ref_id": spk,
            "domain_key": corpus + "|" + ver, "prompt_speaker": None, "lang": lang})
    src = v2.drop_duplicates("file_id").set_index("file_id")
    sid = d["source_file_id"].astype(str)
    hit = sid.isin(src.index).to_numpy()
    name = d["source_name"].astype(object).to_numpy().copy()
    spk = d["speaker_ref_id"].astype(object).to_numpy().copy()
    name[hit] = src.loc[sid[hit], "source_name"].to_numpy()
    spk[hit] = src.loc[sid[hit], "speaker_ref_id"].to_numpy()
    if pd.isna(spk).any() or pd.isna(name).any():
        raise ValueError(f"{corpus}: rows without a source key, e.g. "
                         f"{d.loc[pd.isna(spk) | pd.isna(name), 'file'].head(3).tolist()}")
    return pd.DataFrame({
        "file_id": corpus + ":" + d["file"].astype(str),
        "path": f"interim/{corpus}/" + d["file"].astype(str),
        "artifact_family": None, "source_name": name, "speaker_ref_id": spk,
        "domain_key": None, "prompt_speaker": None, "lang": lang})


def _sing_fake(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    return _sing(root, v2, "fake")


def _sing_real(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    return _sing(root, v2, "real")


def _whole_songs(root: Path, meta: str, v2: pd.DataFrame) -> pd.DataFrame:
    """Whole-file song rows (`cell` per row) from a list CSV: `path` relative
    to the corpus root plus the key columns. The audio is not copied."""
    d = _metadata(root / meta)
    if d.empty:
        return pd.DataFrame(columns=["path"])
    key = d["base_file_id"] if "base_file_id" in d.columns else d["file_id"]
    d = d[~key.isin(set(v2["file_id"])).to_numpy()]      # never twice
    fake = d["cell"].astype(int).eq(8).to_numpy()
    fam = d.get("artifact_family", pd.Series(None, index=d.index)).to_numpy()
    return pd.DataFrame({
        "file_id": d["file_id"].astype(str), "path": d["path"].astype(str),
        "cell": d["cell"].astype(int).to_numpy(),
        "artifact_family": np.where(fake, fam, None),
        "source_name": np.where(fake, fam, d["source_name"].astype(object).to_numpy()),
        "speaker_ref_id": d["speaker_ref_id"].astype(str).to_numpy(),
        "domain_key": np.where(fake, d.get("domain_key", pd.Series(None, index=d.index)).to_numpy(), None),
        "licence_verdict": d.get("licence_verdict", pd.Series(None, index=d.index)).to_numpy(),
        "prompt_speaker": None, "lang": d.get("lang", pd.Series(None, index=d.index)).to_numpy()})


def _aisong_kr_en(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """N2c: ACE-Step 1.5 songs with Korean / English lyrics, cell 8 as-is."""
    return _whole_songs(root, "interim/aisong-kr-en/metadata.csv", v2)


def _sing_real_whole(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """N2b: fma_medium songs whose vocal stem passed the N2 screen, cell 5 as-is
    (fma_small / MUSAN songs with vocals are already cell-5 rows of the base)."""
    return _whole_songs(root, "interim/sing-real/whole_songs.csv", v2)


def _mtg_jamendo(root: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """strategy-v6: MTG-Jamendo instrumentals -> pool C (real music components).
    30 s 16 kHz mono excerpts (scripts/data_extra/jamendo_prepare.py) that the vocal
    screen calls instrumental. Publisher atom `mtg-jamendo/<artist id>` (the draw's
    domain and its `domain_weights` prefix); speaker key = processing.corpus's artist
    atom (`artist:<name>`), the key FMA and MUSAN rows use."""
    d = _metadata(root / "interim" / "mtg-jamendo" / "metadata.csv")
    if d.empty:
        return pd.DataFrame(columns=["path"])
    return pd.DataFrame({
        "file_id": "mtg-jamendo:" + d["track_id"].astype(str),
        "path": "interim/mtg-jamendo/" + d["file"].astype(str),
        "artifact_family": None, "source_name": d["source_name"].astype(str),
        "speaker_ref_id": d["speaker_ref_id"].astype(str),
        "domain_key": None, "prompt_speaker": None, "lang": None,
        "licence_verdict": d["licence_verdict"].astype(str)})


NEW_CORPORA = (
    NewCorpus("ko-synth", "B", _ko_synth),
    NewCorpus("sonics-sep", "D", _sonics_sep),
    NewCorpus("realmusic-sep", "C", _realmusic_sep),
    NewCorpus("libritts-r", "A", _libritts_r),
    NewCorpus("common-voice-ko", "A", _common_voice_ko),
    # round 2 (docs/training/09); "*" = the reader gives a per-row `pool`
    NewCorpus("proc", "*", _proc),
    NewCorpus("emilia-ko", "A", _emilia_ko),
    NewCorpus("ko-synth2", "B", _ko_synth2),
    NewCorpus("ko-synth3", "B", _ko_synth3),
    NewCorpus("ko-synth2-extra", "B", _ko_synth2_extra),
    NewCorpus("emilia-en", "A", _emilia_en),
    NewCorpus("en-synth2", "B", _en_synth2),
    NewCorpus("en-synth3", "B", _en_synth3),
    NewCorpus("en-synth2-extra", "B", _en_synth2_extra),
    NewCorpus("zh-synth", "B", _zh_synth),
    NewCorpus("ctrsvdd", "*", _ctrsvdd),
    # N2 singing (docs/training/15 §3); "W" = whole-file rows, `cell` per row
    NewCorpus("sing-fake", "B", _sing_fake),
    NewCorpus("sing-real", "A", _sing_real),
    NewCorpus("aisong-kr-en", "W", _aisong_kr_en),
    NewCorpus("fma-medium-songs", "W", _sing_real_whole),
    # strategy-v6: real music diversity (polished real production, not only FMA/MUSAN)
    NewCorpus("mtg-jamendo", "C", _mtg_jamendo),
)


def _frame(rows: pd.DataFrame, probe: pd.DataFrame, nc: NewCorpus) -> pd.DataFrame:
    n = len(rows)
    pools = (rows["pool"].to_numpy() if "pool" in rows.columns
             else np.array([nc.pool] * n, dtype=object))
    if nc.pool == "*" and "pool" not in rows.columns:
        raise ValueError(f"{nc.name}: the reader must give a per-row pool")
    whole = nc.pool == "W"
    if whole:
        from training.spec import CELL_TABLE, is_fake_cell
        cells = rows["cell"].astype(int).to_numpy()
        pools = np.array([None] * n, dtype=object)
    fake = (np.array([is_fake_cell(c) for c in cells], dtype=bool) if whole
            else np.isin(pools, ["B", "D"]))
    m = pd.DataFrame({
        "file_id": rows["file_id"].to_numpy(), "path": rows["path"].to_numpy(),
        "sha256": probe["sha256"].to_numpy(), "row_kind": "component", "pool": nc.pool,
        "cell": pd.array([pd.NA] * n, dtype="Int64"),
        "duration_s": probe["duration_s"].to_numpy(dtype=float),
        "orig_sr": probe["orig_sr"].to_numpy(), "orig_channels": probe["orig_channels"].to_numpy(),
        "container": rows["path"].str.rsplit(".", n=1).str[-1].str.lower().to_numpy(),
    })
    m["pool"] = pools
    for i, col in enumerate(("label_voice_present", "label_music_present",
                             "label_voice_fake", "label_music_fake")):
        m[col] = pd.array([CELL_TABLE[c][i] for c in cells] if whole
                          else [POOL_LABELS[p][i] for p in pools], dtype="Int64")
    if whole:
        m["row_kind"] = "whole_file"
        m["cell"] = pd.array(cells, dtype="Int64")
    m["artifact_family"] = np.where(fake, rows["artifact_family"].to_numpy(), None)
    m["source_name"] = rows["source_name"].to_numpy()
    m["speaker_ref_id"] = rows["speaker_ref_id"].to_numpy()
    m["pair_id"] = None
    m["dup_group"] = None
    m["domain_key"] = np.where(fake, rows["domain_key"].to_numpy(), None)
    m["slice"] = "train"
    m["fold"] = pd.array([pd.NA] * n, dtype="Int64")
    m["validity_mask_ref"] = None
    m["label_confidence"] = np.where(fake, "exact", "reported")
    m["aug_strength"] = 1.0
    m["corpus"] = nc.name
    m["noise_has_speech"] = False
    m["licence_verdict"] = (rows["licence_verdict"].to_numpy()
                            if "licence_verdict" in rows.columns else None)
    m["stage"] = rows["path"].str.split("/", n=1).str[0].to_numpy()
    m["reassigned_from"] = None
    m["prompt_speaker"] = rows["prompt_speaker"].to_numpy()
    m["lang"] = rows["lang"].to_numpy() if "lang" in rows.columns else None
    return m


def extend_manifest(v2: pd.DataFrame, corpus_root: Path, *, component_floor_s: float,
                    only: tuple[str, ...] | None = None, append: tuple[str, ...] = (),
                    workers: int = 32, scheme_version: str = SCHEME_VERSION) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``(v3 manifest, report)``. v2's rows are copied unchanged but for
    `scheme_version`, `lang` and an empty `prompt_speaker`. A corpus already in
    the base is skipped unless named in `append`: then only its rows whose
    `file_id` the base lacks are added (e.g. emilia-ko's second YODAS sample)."""
    report: dict[str, Any] = {"scheme_version": scheme_version, "base_rows": int(len(v2)),
                              "new": {}}
    base = v2.copy()
    if "prompt_speaker" not in base.columns:
        base["prompt_speaker"] = None
    have = set(base["corpus"].astype(str))
    parts = [base]
    seen_sha = set(v2["sha256"].astype(str))
    for nc in NEW_CORPORA:
        if (only is not None and nc.name not in only) or (nc.name in have and nc.name not in append):
            continue
        rows = nc.reader(corpus_root, v2)
        if nc.name in have and not rows.empty:
            rows = rows[~rows["file_id"].isin(set(base["file_id"]))].reset_index(drop=True)
        if rows.empty:
            report["new"][nc.name] = {"rows": 0, "note": "no input found"}
            continue
        probe = _probe_all(corpus_root, rows["path"].tolist(), workers)
        ok = probe["ok"].to_numpy(dtype=bool)
        failed = int((~ok).sum())
        rows, probe = rows[ok].reset_index(drop=True), probe[ok].reset_index(drop=True)
        short = probe["duration_s"].to_numpy() < component_floor_s
        dup = probe["sha256"].isin(seen_sha).to_numpy() | probe["sha256"].duplicated().to_numpy()
        keep = ~short & ~dup
        new = _frame(rows[keep].reset_index(drop=True), probe[keep].reset_index(drop=True), nc)
        seen_sha |= set(new["sha256"])
        report["new"][nc.name] = {
            "rows": int(len(new)), "hours": round(float(new["duration_s"].sum()) / 3600, 2),
            "failed_to_open": failed, "below_floor": int(short.sum()),
            "duplicate_sha": int((dup & ~short).sum()),
            "families": int(new["artifact_family"].nunique()),
            "speakers": int(new["speaker_ref_id"].nunique())}
        parts.append(new)
    m = pd.concat(parts, ignore_index=True)
    m["scheme_version"] = scheme_version
    known = m["lang"] if "lang" in m.columns else pd.Series(None, index=m.index)
    m["lang"] = known.where(known.notna(), lang_of(m)).where(m["pool"].isin(["A", "B"]), None)
    m["cell"] = m["cell"].astype("Int64")
    m["fold"] = m["fold"].astype("Int64")
    m = m[list(REQUIRED_COLUMNS) + list(EXTRA_COLUMNS) + list(NEW_COLUMNS)]
    if m["file_id"].duplicated().any():
        raise ValueError(f"duplicate file_id(s): {m.loc[m.file_id.duplicated(), 'file_id'][:3]}")
    validate_manifest(m)
    check_rules(m)
    voice = m[m["pool"].isin(["A", "B"])]
    report["rows"] = int(len(m))
    report["voice_hours_by_lang"] = {
        f"{p}/{lg}": round(float(h) / 3600, 1) for (p, lg), h in
        voice.groupby(["pool", voice["lang"].fillna("?")])["duration_s"].sum().items()}
    report["hours_by_pool"] = {k: round(float(v) / 3600, 1) for k, v in
                               m[m.row_kind == "component"].groupby("pool")["duration_s"]
                               .sum().items()}
    return m, report
