"""strategy-v3: the strategy-v2 manifest plus the first run's new corpora
(docs/training/07 §1), and a `lang` column on every row.

strategy-v2 was built from the EDA's tiers (`processing.corpus.build_manifest`),
which the new corpora do not have, and re-running the EDA would cost hours the
deadline does not have. So v3 is v2's rows **unchanged** (every key, label and
filter decision kept) plus rows for:

| corpus | pool | from |
|---|---|---|
| `ko-synth` | B (fake voice, ko) | `interim/ko-synth/<family>/metadata.csv` (docs/training/07 D-a) |
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


NEW_CORPORA = (
    NewCorpus("ko-synth", "B", _ko_synth),
    NewCorpus("sonics-sep", "D", _sonics_sep),
    NewCorpus("realmusic-sep", "C", _realmusic_sep),
    NewCorpus("libritts-r", "A", _libritts_r),
    NewCorpus("common-voice-ko", "A", _common_voice_ko),
)


def _frame(rows: pd.DataFrame, probe: pd.DataFrame, nc: NewCorpus) -> pd.DataFrame:
    labels = POOL_LABELS[nc.pool]
    fake = nc.pool in ("B", "D")
    n = len(rows)
    m = pd.DataFrame({
        "file_id": rows["file_id"].to_numpy(), "path": rows["path"].to_numpy(),
        "sha256": probe["sha256"].to_numpy(), "row_kind": "component", "pool": nc.pool,
        "cell": pd.array([pd.NA] * n, dtype="Int64"),
        "duration_s": probe["duration_s"].to_numpy(dtype=float),
        "orig_sr": probe["orig_sr"].to_numpy(), "orig_channels": probe["orig_channels"].to_numpy(),
        "container": rows["path"].str.rsplit(".", n=1).str[-1].str.lower().to_numpy(),
    })
    for col, val in zip(("label_voice_present", "label_music_present",
                         "label_voice_fake", "label_music_fake"), labels):
        m[col] = pd.array([val] * n, dtype="Int64")
    m["artifact_family"] = rows["artifact_family"].to_numpy() if fake else None
    m["source_name"] = rows["source_name"].to_numpy()
    m["speaker_ref_id"] = rows["speaker_ref_id"].to_numpy()
    m["pair_id"] = None
    m["dup_group"] = None
    m["domain_key"] = rows["domain_key"].to_numpy() if fake else None
    m["slice"] = "train"
    m["fold"] = pd.array([pd.NA] * n, dtype="Int64")
    m["validity_mask_ref"] = None
    m["label_confidence"] = "exact" if fake else "reported"
    m["aug_strength"] = 1.0
    m["corpus"] = nc.name
    m["noise_has_speech"] = False
    m["licence_verdict"] = None
    m["stage"] = rows["path"].str.split("/", n=1).str[0].to_numpy()
    m["reassigned_from"] = None
    m["prompt_speaker"] = rows["prompt_speaker"].to_numpy()
    return m


def extend_manifest(v2: pd.DataFrame, corpus_root: Path, *, component_floor_s: float,
                    only: tuple[str, ...] | None = None,
                    workers: int = 32) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``(v3 manifest, report)``. v2's rows are copied unchanged but for
    `scheme_version`, `lang` and an empty `prompt_speaker`."""
    report: dict[str, Any] = {"scheme_version": SCHEME_VERSION, "v2_rows": int(len(v2)),
                              "new": {}}
    base = v2.copy()
    base["prompt_speaker"] = None
    parts = [base]
    seen_sha = set(v2["sha256"].astype(str))
    for nc in NEW_CORPORA:
        if only is not None and nc.name not in only:
            continue
        rows = nc.reader(corpus_root, v2)
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
    m["scheme_version"] = SCHEME_VERSION
    m["lang"] = lang_of(m)
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
