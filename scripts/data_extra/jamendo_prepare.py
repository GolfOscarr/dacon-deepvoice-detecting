#!/usr/bin/env python3
"""MTG-Jamendo real music for strategy-v6: instrumentals -> pool C (real music components).

    python scripts/data_extra/jamendo_prepare.py select            # licences, artist cap
    python scripts/data_extra/jamendo_prepare.py extract --workers 24
    python scripts/data_extra/jamendo_prepare.py tag --workers 24 --threads 4   # AST, CPU
    # calibration subset only: scripts/synth3/sing_separate.py --device cpu --tag jam \
    #   --selection <work>/selection.csv --out-base <work>/stems, then `measure`
    python scripts/data_extra/jamendo_prepare.py measure --workers 1
    python scripts/data_extra/jamendo_prepare.py finalize

Source: MTG-Jamendo `raw_30s` audio-low (96 kbps MP3, full tracks), tars
`raw/mtg-jamendo/tars/raw_30s_audio-low-NN.tar` (sha256 checked against the
dataset's list), metadata from the dataset repo (`raw/mtg-jamendo/meta_repo`).

select   every track in a verified tar whose per-track licence (the repo's
         `audio_licenses.txt`, classified by scripts/filter_track_licences.py) is not
         `deny`; artists whose atom (`artist:<name>`, processing.corpus) already
         names a base real-music row are left out (MUSAN's `music/jamendo` is
         Jamendo material: never the same song twice, never an atom split);
         at most PER_ARTIST tracks per artist and PER_ALBUM per album, seed 0.
         Writes <work>/selection_tracks.csv and interim/_licences/mtg_jamendo_licences.csv.
extract  one 30 s excerpt from the MIDDLE of each track (sing_select.py's rule),
         CANONICAL first: downmix to mono, resample to 16 kHz (sing_separate.py's
         load_canon), a peak above 0.99 scaled down to 0.99 (gain, never clipping),
         WAV PCM16 at interim/mtg-jamendo/clips/<NN>/<id>.wav. Measures per clip:
         rms_dbfs, peak_src, clip_frac (|x| >= 0.999 before the gain), the exact-zero
         run (ms) and fraction of the written PCM16, and rolloff99_hz. Drops the
         exact-zero rule's clips (run > 20 ms or > 5 %) and near-silent ones; writes
         <work>/clips.csv and <work>/selection.csv (sing_separate.py's schema).
tag      the vocal screen: GPUs were busy and full htdemucs on CPU (~4 core-s per
         audio second) slowed the training node, so every clip is scored by an AudioSet
         tagger (AST) instead, calibrated against htdemucs on the clips it separated.
measure  Silero on the htdemucs vocal chunks of that calibration subset (sing_screen's
         speech_ratio), so finalize can apply the v5 chunk conditions there.
finalize interim/mtg-jamendo/metadata.csv: the clips RULES calls instrumental, each cut
         into 10 s rows (see _segments: the music draw's tile share is a row count).
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
RAW = CORPUS / "raw/mtg-jamendo"
META = RAW / "meta_repo"
OUT = CORPUS / "interim/mtg-jamendo"
WORK = OUT / "_work"
LICENCES = CORPUS / "interim/_licences/mtg_jamendo_licences.csv"
BASE = CORPUS / "manifests/strategy-v5/manifest.parquet"
PER_ARTIST, PER_ALBUM, EXCERPT_S, SR = 4, 2, 30.0, 16000
ZERO_RUN_MS, ZERO_FRAC = 20.0, 0.05
RMS_MIN_CLIP = -40.0          # sing_screen's near-silent floor, on the whole clip
SEG_S = 10.0                  # pool-C row length (finalize: see _segments)
# RULES (finalize). Vocal presence is the ONLY uncertain label (every row is real music);
# a clip the screen cannot call instrumental is dropped, never labelled. Instrumental =
#   AST: every 10 s window's max P(vocal class) < AST_INSTR_MAX (calibrated on the clips
#        htdemucs separated: 1 of 203 below 0.03 had any htdemucs vocal chunk), and
#   htdemucs, where the clip was separated: no vocal chunk passes rms/vox_rel/Silero
#        (sing_screen's chunk conditions without Whisper -- stricter than v5's rule), and
#   no voice-type Jamendo tag (uploader tags: only ever used to exclude).
AST_INSTR_MAX = 0.03
VOICE_TAGS = ("instrument---voice", "genre---rap", "genre---hiphop", "genre---singersongwriter")


def _tars() -> list[int]:
    want = dict(line.split()[::-1] for line in
                (META / "data/download/raw_30s_audio-low_sha256_tars.txt").read_text().splitlines())
    ok = []
    for p in sorted((RAW / "tars").glob("raw_30s_audio-low-*.tar")):
        if not pathlib.Path(str(p) + ".ok").exists():
            continue
        mark = pathlib.Path(str(p) + ".sha")
        if not mark.exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
            if h.hexdigest() != want[p.name]:
                print(f"SHA MISMATCH {p.name}: skipped", flush=True)
                continue
            mark.write_text(h.hexdigest())
        ok.append(int(p.stem.rsplit("-", 1)[1]))
    return ok


def select(args) -> int:
    from filter_track_licences import classify, load_jamendo
    from processing.corpus import _artist_atom
    tars = _tars()
    print(f"verified tars: {tars}", flush=True)
    rows = [ln.rstrip("\n").split("\t") for ln in open(META / "data/raw_30s_cleantags.tsv")][1:]
    d = pd.DataFrame([(r[0], r[1], r[2], r[3], float(r[4]), "|".join(r[5:])) for r in rows],
                     columns=["track_id", "artist_id", "album_id", "path", "track_duration_s", "tags"])
    # raw.meta.tsv has unquoted tabs inside some names: the id is first, the URL last
    mr = [ln.rstrip("\n").split("\t") for ln in open(META / "data/raw.meta.tsv", encoding="utf-8")][1:]
    meta = pd.DataFrame([(r[0], " ".join(r[3:-4]).strip(), r[-4], r[-3], r[-1]) for r in mr], columns=["TRACK_ID", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME", "URL"])
    d = d.merge(meta[["TRACK_ID", "TRACK_NAME", "ARTIST_NAME", "ALBUM_NAME", "URL"]].rename(columns={
        "TRACK_ID": "track_id", "TRACK_NAME": "track_name", "ARTIST_NAME": "artist_name",
        "ALBUM_NAME": "album_name", "URL": "url"}), on="track_id", how="left")
    lic = load_jamendo(META).rename(columns={"id": "path"})
    lic["verdict"] = lic["licence"].map(classify)
    d = d.merge(lic, on="path", how="left")
    d["verdict"] = d["verdict"].fillna("deny")
    d["tar"] = d["path"].str.split("/").str[0].astype(int)
    d = d[d["tar"].isin(tars)].copy()
    atom = d["artist_name"].map(_artist_atom)
    d["speaker_ref_id"] = atom.where(atom.notna(), "jamendo/" + d["artist_id"]).astype(str)
    base = pd.read_parquet(BASE, columns=["speaker_ref_id"])["speaker_ref_id"].dropna().astype(str)
    clash = d["speaker_ref_id"].isin(set(base))
    d["select_reason"] = ""
    d.loc[d["verdict"].eq("deny"), "select_reason"] = "licence-deny"
    d.loc[clash & d["select_reason"].eq(""), "select_reason"] = "artist-in-base"
    rng = np.random.default_rng(0)
    d["_r"] = rng.permutation(len(d))
    ok = d[d["select_reason"].eq("")].sort_values("_r")
    ok = ok[ok.groupby("album_id").cumcount() < PER_ALBUM]
    ok = ok[ok.groupby("speaker_ref_id").cumcount() < PER_ARTIST]
    d.loc[d["select_reason"].eq("") & ~d.index.isin(ok.index), "select_reason"] = "cap"
    d["selected"] = d["select_reason"].eq("")
    WORK.mkdir(parents=True, exist_ok=True)
    d = d.drop(columns="_r").sort_values("track_id")
    d.to_csv(WORK / "selection_tracks.csv", index=False)
    s = d[d["selected"]]
    LICENCES.parent.mkdir(parents=True, exist_ok=True)
    s[["track_id", "path", "artist_name", "track_name", "url", "licence", "licence_url",
       "verdict"]].to_csv(LICENCES, index=False)
    print(d["select_reason"].replace("", "selected").value_counts().to_dict())
    print(f"selected {len(s)} tracks, {s['speaker_ref_id'].nunique()} artists, "
          f"{s['album_id'].nunique()} albums; licences {s['licence'].value_counts().to_dict()}")
    return 0


def _extract_one(r: dict) -> dict:
    import io

    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as AF
    torch.set_num_threads(1)
    out = {"track_id": r["track_id"], "ok": False}
    try:
        mp3 = RAW / "audio" / r["path"].replace(".mp3", ".low.mp3")   # tar member name
        if not mp3.exists():
            raise FileNotFoundError(f"{mp3} (untar raw_30s_audio-low-{int(r['tar']):02d}.tar first)")
        wav, sr = torchaudio.load(io.BytesIO(mp3.read_bytes()), format="mp3")
        dur = wav.shape[1] / sr
        off = max(0.0, (dur - EXCERPT_S) / 2)
        a, n = int(round(off * sr)), int(round(EXCERPT_S * sr))
        seg = wav[:, a:a + n].mean(0)
        if sr != SR:
            seg = AF.resample(seg, sr, SR)
        y = seg.numpy().astype(np.float64)
        peak = float(np.abs(y).max())
        clip_frac = float(np.mean(np.abs(y) >= 0.999))
        if peak > 0.99:
            y = y * (0.99 / peak)
        pcm = np.round(y * 32767).astype(np.int16)
        z = pcm == 0
        run = 0
        if z.any():
            dz = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
            run = int((np.flatnonzero(dz == -1) - np.flatnonzero(dz == 1)).max())
        spec = np.abs(np.fft.rfft(y[: len(y) // 2048 * 2048].reshape(-1, 2048) * np.hanning(2048), axis=1)) ** 2
        cum = np.cumsum(spec.sum(0))
        roll = float(np.searchsorted(cum, 0.99 * cum[-1]) * SR / 2048)
        rel = f"clips/{r['path'].split('/')[0]}/{r['track_id']}.wav"
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part.wav")
        sf.write(tmp, pcm, SR, subtype="PCM_16")
        tmp.replace(dst)
        rms = float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))
        out.update(ok=True, file=rel, src_sr=sr, src_channels=int(wav.shape[0]), track_s=round(dur, 2),
                   offset_s=round(off, 3), duration_s=round(len(pcm) / SR, 3),
                   rms_dbfs=round(20 * np.log10(rms + 1e-12), 2), peak_src=round(peak, 4),
                   clip_frac=round(clip_frac, 6), zero_run_ms=round(run / SR * 1000, 2),
                   zero_frac=round(float(z.mean()), 6), rolloff99_hz=round(roll, 1))
    except Exception as exc:                                  # noqa: BLE001 -- recorded
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def extract(args) -> int:
    s = pd.read_csv(WORK / "selection_tracks.csv")
    s = s[s["selected"]]
    if args.limit:
        s = s.head(args.limit)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_extract_one, s.to_dict("records"), chunksize=8))
    c = pd.DataFrame(res)
    c.to_csv(WORK / "clips.csv", index=False)
    ok = c[c["ok"]]
    print(f"{len(ok)}/{len(c)} clips, {ok['duration_s'].sum() / 3600:.2f} h; failures "
          f"{c.loc[~c['ok'], 'error'].value_counts().head(5).to_dict() if (~c['ok']).any() else {}}")
    zero = (ok["zero_run_ms"] > ZERO_RUN_MS) | (ok["zero_frac"] > ZERO_FRAC)
    print(f"exact-zero rule flags {int(zero.sum())} clips")
    quiet = ok["rms_dbfs"] < RMS_MIN_CLIP
    print(f"rms < {RMS_MIN_CLIP:g} dBFS: {int(quiet.sum())} clips")
    # seeded random order: the CPU screen may stop at a deadline, and what it has
    # screened by then must be a random subset, not the first tars
    sel = ok[~zero & ~quiet & (ok["duration_s"] >= 10)].sample(frac=1.0, random_state=0).reset_index(drop=True)
    pd.DataFrame({
        "sel_idx": np.arange(len(sel)), "side": "real", "group": "jamendo",
        "path": "interim/mtg-jamendo/" + sel["file"], "file_id": "mtg-jamendo:" + sel["track_id"],
        "offset_s": 0.0, "excerpt_s": EXCERPT_S,
        "out_prefix": "jamendo/" + sel["file"].str.split("/").str[1] + "/" + sel["track_id"],
    }).to_csv(WORK / "selection.csv", index=False)
    print(f"wrote {WORK / 'selection.csv'} ({len(sel)} rows)")
    return 0


def _silero_one(path: str) -> dict:
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO / "scripts/synth3"))
    from sing_screen import silero
    return silero(pathlib.Path(path))


def measure(args) -> int:
    """Silero speech_ratio on every htdemucs vocal chunk of the separated clips (the
    calibration subset) -> <work>/screen_cache.csv."""
    done = {int(p.stem) for p in (WORK / "done_jam").glob("*.done")}
    sep = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob("sep_jam_*.csv"))], ignore_index=True)
    sep = sep[sep["sel_idx"].isin(done) & (sep["sep_ok"].astype(str) == "True") & sep["file"].notna()]
    sep = sep.drop_duplicates(["side", "file"], keep="last")
    root = WORK / "stems/sing-real"
    with ProcessPoolExecutor(args.workers) as ex:
        v = list(ex.map(_silero_one, [str(root / f) for f in sep["file"]], chunksize=16))
    pd.DataFrame(v).assign(side="real", file=sep["file"].to_numpy()).to_csv(WORK / "screen_cache.csv", index=False)
    print(f"{len(sep)} chunks of {len(done)} separated clips measured")
    return 0


AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_HF_HOME = RAW / "_models/hf"
#: AudioSet classes that mean "a human voice is in this music"
VOCAL_CLASSES = ("Speech", "Singing", "Male singing", "Female singing", "Child singing", "Choir",
                 "Yodeling", "Chant", "Mantra", "Rapping", "Humming", "A capella", "Vocal music",
                 "Synthetic singing", "Male speech, man speaking", "Female speech, woman speaking",
                 "Narration, monologue", "Whispering", "Shout", "Beatboxing")


def _tag_part(part: list[str], threads: int) -> list[dict]:
    """AST AudioSet tagger on 10 s windows (3 per 30 s clip): per window the max sigmoid
    over VOCAL_CLASSES, plus Singing / Speech alone."""
    import os
    os.environ.setdefault("HF_HOME", str(AST_HF_HOME))
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    torch.set_num_threads(threads)
    fe = ASTFeatureExtractor.from_pretrained(AST)
    model = ASTForAudioClassification.from_pretrained(AST).eval()
    lab = model.config.label2id
    vid = [lab[c] for c in VOCAL_CLASSES]
    out = []
    for f in part:
        y, sr = sf.read(OUT / f, dtype="float32")
        wins = [y[k * 160000:(k + 1) * 160000] for k in range(max(1, len(y) // 160000))]
        feats = fe(wins, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            p = torch.sigmoid(model(feats).logits).numpy()
        v = p[:, vid].max(1)
        out.append({"file": f, "vox_win": ";".join(f"{x:.4f}" for x in v), "vox_max": float(v.max()),
                    "vox_mean": float(v.mean()), "sing_max": float(p[:, lab["Singing"]].max()),
                    "speech_max": float(p[:, lab["Speech"]].max()),
                    "music_min": float(p[:, lab["Music"]].min())})
    return out


def tag(args) -> int:
    """CPU vocal screen (the GPU-free replacement for the htdemucs screen, calibrated
    against it on the clips htdemucs separated): <work>/ast_tags.csv, resumable."""
    sel = pd.read_csv(WORK / "selection.csv")
    files = sel["path"].str.replace("interim/mtg-jamendo/", "", regex=False).tolist()
    if args.limit:
        files = files[:args.limit]
    path = WORK / "ast_tags.csv"
    have = set(pd.read_csv(path)["file"]) if path.exists() else set()
    todo = [f for f in files if f not in have]
    print(f"{len(files)} clips, {len(todo)} to tag", flush=True)
    procs = max(1, args.workers // args.threads)
    step = 60 * procs
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        parts = [chunk[k::procs] for k in range(procs)]
        with ProcessPoolExecutor(procs) as ex:
            res = [r for part in ex.map(_tag_part, parts, [args.threads] * procs) for r in part]
        pd.DataFrame(res).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
    return 0


def _segments(keep: pd.DataFrame) -> pd.DataFrame:
    """Each kept 30 s clip -> consecutive SEG_S rows (seg10/<NN>/<id>_<k>.wav, PCM16 16 kHz).
    Why: the music draw has ONE bucket per side and tiles it uniformly over the rows that
    fit (processing.sampler.bucket_keys / _tiles), so a corpus's share of music tiles is its
    row count -- draw.domain_weights cannot move it. 10 s is fakemusiccaps' row length.
    A near-silent segment (rms < RMS_MIN_CLIP) is dropped."""
    import soundfile as sf
    n = int(SEG_S * SR)
    rows = []
    for r in keep.to_dict("records"):
        y, sr = sf.read(OUT / r["file"], dtype="int16")
        assert sr == SR, (r["file"], sr)
        for k in range(len(y) // n):
            c = y[k * n:(k + 1) * n]
            rms = float(np.sqrt(np.mean((c / 32768.0) ** 2)))
            if 20 * np.log10(rms + 1e-12) < RMS_MIN_CLIP:
                continue
            rel = f"seg10/{r['file'].split('/')[1]}/{r['track_id']}_{k}.wav"
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                sf.write(dst.with_suffix(".part.wav"), c, SR, subtype="PCM_16")
                dst.with_suffix(".part.wav").replace(dst)
            rows.append({**r, "clip": r["file"], "seg": k, "file": rel, "duration_s": len(c) / SR,
                         "offset_s": r["offset_s"] + k * SEG_S})
    return pd.DataFrame(rows)


def finalize(args) -> int:
    sys.path.insert(0, str(REPO / "scripts/synth3"))
    from sing_screen import REL_MIN, RMS_MIN, SPEECH_MIN
    tracks = pd.read_csv(WORK / "selection_tracks.csv").set_index("track_id")
    clips = pd.read_csv(WORK / "clips.csv").set_index("track_id")
    s = pd.read_csv(WORK / "selection.csv").drop(columns=["offset_s", "excerpt_s"])
    s["file"] = s["path"].str.replace("interim/mtg-jamendo/", "", regex=False)
    s["track_id"] = s["file_id"].str.replace("mtg-jamendo:", "", regex=False)
    s = s.merge(pd.read_csv(WORK / "ast_tags.csv").drop_duplicates("file", keep="last"), on="file", how="left")
    # htdemucs on the calibration subset
    done = {int(p.stem) for p in (WORK / "done_jam").glob("*.done")}
    sep = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob("sep_jam_*.csv"))], ignore_index=True)
    sep = sep[sep["sel_idx"].isin(done) & sep["file"].notna()].drop_duplicates("file", keep="last")
    ch = sep.merge(pd.read_csv(WORK / "screen_cache.csv"), on=["side", "file"], how="left")
    pre = (ch["rms_dbfs"] >= RMS_MIN) & (ch["vox_rel_db"] >= REL_MIN) & (ch["speech_ratio"] >= SPEECH_MIN)
    s["separated"] = s["sel_idx"].isin(done)
    s["htdemucs_vocal_s"] = s["file_id"].map(ch[pre].groupby("source_file_id")["duration_s"].sum()).fillna(0.0)
    s.loc[~s["separated"], "htdemucs_vocal_s"] = np.nan
    s = s.join(clips.drop(columns=["file", "ok"]), on="track_id").join(
        tracks[["artist_id", "album_id", "speaker_ref_id", "tags", "licence", "verdict"]], on="track_id")
    s["voice_tag"] = s["tags"].fillna("").map(lambda t: any(v in t for v in VOICE_TAGS))
    why = np.select(
        [s["vox_max"].isna(), s["vox_max"] >= AST_INSTR_MAX, s["htdemucs_vocal_s"].fillna(0) > 0, s["voice_tag"]],
        ["untagged", "ast-vocal-or-ambiguous", "htdemucs-vocal", "voice-tag"], default="")
    s["drop_reason"], s["kept"] = why, why == ""
    s["source_name"] = "mtg-jamendo/" + s["artist_id"].astype(str)
    s["licence_verdict"] = s["verdict"]
    s.to_csv(WORK / "screen_result.csv", index=False)
    print(s.groupby("drop_reason")["duration_s"].agg(["size", "sum"]).assign(h=lambda x: x["sum"] / 3600)
          .round(2).to_string())
    cal = s[s["separated"] & s["vox_max"].notna()]
    ast_i = cal["vox_max"] < AST_INSTR_MAX
    print(f"calibration (htdemucs-separated, n={len(cal)}): AST instrumental {int(ast_i.sum())}, of which "
          f"htdemucs vocal chunk {int((ast_i & (cal['htdemucs_vocal_s'] > 0)).sum())}")
    keep = _segments(s[s["kept"]])
    cols = ["file", "clip", "seg", "track_id", "artist_id", "album_id", "speaker_ref_id", "source_name", "duration_s",
            "offset_s", "track_s", "src_sr", "src_channels", "rms_dbfs", "peak_src", "clip_frac",
            "zero_run_ms", "zero_frac", "rolloff99_hz", "vox_max", "htdemucs_vocal_s", "licence",
            "licence_verdict", "tags", "kept"]
    keep[cols].sort_values("track_id").to_csv(OUT / "metadata.csv", index=False)
    print(f"instrumental: {keep['clip'].nunique()} clips -> {len(keep)} {SEG_S:g} s rows, "
          f"{keep['duration_s'].sum() / 3600:.2f} h, "
          f"{keep['speaker_ref_id'].nunique()} artists, {keep['album_id'].nunique()} albums")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["select", "extract", "measure", "tag", "finalize"])
    ap.add_argument("--threads", type=int, default=4, help="tag: torch threads per AST process")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    return {"select": select, "extract": extract, "measure": measure, "tag": tag, "finalize": finalize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
