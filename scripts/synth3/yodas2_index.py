"""N3 (docs/training/15 §4): a second 40 h of real Korean YODAS speech, disjoint by video.

Input: the round-2 tars from scripts/synth3/yodas2_download.py in
interim/emilia-ko/_raw/Emilia-YODAS/KO/. Round 1 (scripts/synth2/ko_emilia_index.py) wrote
interim/emilia-ko/metadata.csv; this script NEVER modifies it. It writes
interim/emilia-ko/metadata_yodas2.csv (same 13 columns) and the audio under
interim/emilia-ko/yodas/<tar stem>/<id>.mp3 + <id>.json (original bytes, not re-encoded).

Selection (deterministic, seed 2026:yodas2):
  1. Candidates: segments whose json says 3 <= duration <= 30 s, language ko, Hangul in the
     transcript, from a video (speaker key `KO_<video>`) that does NOT occur in metadata.csv.
  2. Order: round-robin over seed-shuffled videos, at most --cap segments per video.
  3. Each segment in that order is decoded (from the tar, in memory) and screened:
       duration 3-30 s and finite   -> "duration" / "nonfinite"
       RMS >= -50 dBFS              -> "silent"
       speech ratio >= 0.5          -> "low_speech"  (fraction of 10 ms frames >= -50 dBFS;
                                        the same test as ko_qc.py's silence_frac <= 50 %)
       Hangul in transcript         -> "not_korean"
     The walk stops when the kept segments reach --hours. Rows in the walked prefix are written
     (kept True/False with drop_reason, as round 1 did); later candidates are not.
speaker_ref_id = "emilia-ko/yodas/KO_<video>" (the video is the atom), diar_speaker = the
diarised `<video>_SPEAKER_nn`.

  python scripts/synth3/yodas2_index.py --tars KO-B000013,... [--hours 40] [--cap 6] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/data/project/private/dacon-corpus/interim/emilia-ko")
RAW = ROOT / "_raw" / "Emilia-YODAS" / "KO"
OUT_NAME = "metadata_yodas2.csv"
LIC = "CC-BY-4.0"
COLS = ["file", "part", "speaker_ref_id", "text", "duration_s", "sample_rate", "licence", "kept",
        "drop_reason", "source_tar", "dnsmos", "orig_format", "diar_speaker"]
HANGUL = re.compile(r"[가-힣]")
FLOOR_DB = -50.0
SPEECH_MIN = 0.5


def screen(a: np.ndarray, sr: int, text: str) -> tuple[float, bool, str]:
    """(duration, kept, drop_reason) for one decoded mono clip."""
    dur = len(a) / sr if sr else 0.0
    why = []
    if not np.isfinite(a).all():
        why.append("nonfinite")
        a = np.nan_to_num(a)
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if len(a) else 0.0
    if 20 * np.log10(max(rms, 1e-9)) < FLOOR_DB:
        why.append("silent")
    hop = max(sr // 100, 1)
    n = len(a) // hop
    if n:
        fr = np.sqrt(np.mean(a[: n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1))
        speech = float(np.mean(20 * np.log10(np.maximum(fr, 1e-9)) >= FLOOR_DB))
    else:
        speech = 0.0
    if speech < SPEECH_MIN:
        why.append("low_speech")
    if not 3.0 <= dur <= 30.0:
        why.append("duration")
    if not HANGUL.search(text):
        why.append("not_korean")
    return dur, not why, "|".join(why)


def scan_tar(tar: Path) -> list[dict]:
    out = []
    with tarfile.open(tar) as t:
        for m in t:
            if m.name.endswith(".json"):
                j = json.load(t.extractfile(m))
                j["_tar"] = tar.name
                j["id"] = j.get("id") or j.get("_id") or Path(m.name).stem
                spk = j["speaker"]
                j["src"] = spk.split("_SPEAKER_")[0] if "_SPEAKER_" in spk else spk
                out.append(j)
    return out


def check_tar(args) -> dict[str, tuple[float, int, bool, str]]:
    """Decode the given ids from one tar in memory: id -> (dur, sr, kept, why)."""
    tar, ids = args
    res = {}
    with tarfile.open(tar) as t:
        for m in t:
            key = Path(m.name).stem
            if m.name.endswith(".mp3") and key in ids:
                text = ids[key]
                try:
                    a, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=False)
                except Exception as e:  # noqa: BLE001
                    res[key] = (0.0, 0, False, f"unreadable:{type(e).__name__}")
                    continue
                if a.ndim > 1:
                    a = a.mean(axis=1)
                dur, ok, why = screen(a, sr, text)
                res[key] = (dur, sr, ok, why)
    for key in ids:
        res.setdefault(key, (0.0, 0, False, "missing"))
    return res


def extract_tar(args) -> None:
    tar, ids = args
    dest = ROOT / "yodas" / tar.stem
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        for m in t:
            if Path(m.name).stem in ids and m.isfile():
                p = dest / Path(m.name).name
                if not p.exists():
                    tmp = p.with_suffix(p.suffix + ".tmp")
                    tmp.write_bytes(t.extractfile(m).read())
                    os.replace(tmp, p)


def round1_videos() -> tuple[set[str], int]:
    """Video keys (`KO_<video>`) of every YODAS row in metadata.csv, kept or not."""
    vids, n = set(), 0
    with (ROOT / "metadata.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["part"] == "yodas":
                n += 1
                vids.add(r["speaker_ref_id"].split("/")[-1])
    return vids, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", required=True, help="comma-separated tar stems, e.g. KO-B000013")
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--cap", type=int, default=6, help="max segments per video")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", default="2026:yodas2")
    a = ap.parse_args()
    tars = [RAW / f"{s.strip()}.tar" for s in a.tars.split(",") if s.strip()]
    r1_tars = {f"KO-B{round(k * 208 / 8):06d}.tar" for k in range(8)}
    assert not r1_tars & {t.name for t in tars}, "round-1 tar given"
    assert all(t.exists() for t in tars), [t for t in tars if not t.exists()]

    old, n_old = round1_videos()
    print(f"round 1: {n_old} yodas rows over {len(old)} videos", flush=True)
    target = a.hours * 3600
    with ProcessPoolExecutor(a.workers) as ex:
        segs = [j for js in ex.map(scan_tar, tars) for j in js]
        json_ok = [j for j in segs if 3.0 <= float(j["duration"]) <= 30.0
                   and HANGUL.search(j["text"]) and j.get("language", "ko") == "ko"]
        overlap = {j["src"] for j in json_ok} & old
        cand = [j for j in json_ok if j["src"] not in old]
        print(f"{len(tars)} tars: {len(segs)} segments ({sum(float(j['duration']) for j in segs)/3600:.1f} h) "
              f"over {len({j['src'] for j in segs})} videos; json-screen candidates {len(json_ok)} "
              f"({sum(float(j['duration']) for j in json_ok)/3600:.1f} h); videos also in round 1: "
              f"{len(overlap)} (removed)", flush=True)

        rng = random.Random(a.seed)
        by_src = defaultdict(list)
        for j in sorted(cand, key=lambda j: j["id"]):
            by_src[j["src"]].append(j)
        srcs = sorted(by_src)
        rng.shuffle(srcs)
        for s in srcs:
            rng.shuffle(by_src[s])
        order = [by_src[s][r] for r in range(a.cap) for s in srcs if r < len(by_src[s])]
        print(f"order: {len(order)} segments ({sum(float(j['duration']) for j in order)/3600:.1f} h) "
              f"over {len(srcs)} videos, cap {a.cap}", flush=True)

        # screen in chunks of the order until the kept hours reach the target
        res: dict[str, tuple] = {}
        pos, kept_s, walked = 0, 0.0, []
        while kept_s < target and pos < len(order):
            chunk_s = max((target - kept_s) * 1.15, 1800.0)
            chunk, acc = [], 0.0
            while pos < len(order) and acc < chunk_s:
                chunk.append(order[pos]); acc += float(order[pos]["duration"]); pos += 1
            per_tar = defaultdict(dict)
            for j in chunk:
                per_tar[j["_tar"]][j["id"]] = j["text"].strip()
            for r in ex.map(check_tar, [(RAW / tn, ids) for tn, ids in sorted(per_tar.items())]):
                res.update(r)
            for j in chunk:
                if kept_s >= target:
                    break
                walked.append(j)
                if res[j["id"]][2]:
                    kept_s += res[j["id"]][0]
            print(f"screened {pos}/{len(order)}; kept {kept_s/3600:.2f} h", flush=True)
        per_tar = defaultdict(set)
        for j in walked:
            per_tar[j["_tar"]].add(j["id"])
        list(ex.map(extract_tar, [(RAW / tn, ids) for tn, ids in sorted(per_tar.items())]))

    rows = []
    for j in walked:
        dur, sr, ok, why = res[j["id"]]
        stem = Path(j["_tar"]).stem
        p = ROOT / "yodas" / stem / f"{j['id']}.mp3"
        if ok and not p.exists():
            ok, why = False, "missing"
        rows.append({"file": f"yodas/{stem}/{j['id']}.mp3", "part": "yodas",
                     "speaker_ref_id": f"emilia-ko/yodas/{j['src']}", "text": j["text"].strip(),
                     "duration_s": f"{dur:.3f}", "sample_rate": sr, "licence": LIC, "kept": str(ok),
                     "drop_reason": why, "source_tar": f"Emilia-YODAS/KO/{j['_tar']}",
                     "dnsmos": j.get("dnsmos", ""), "orig_format": "mp3", "diar_speaker": j["speaker"]})

    new_vids = {r["speaker_ref_id"].split("/")[-1] for r in rows}
    assert not new_vids & old, f"video overlap with metadata.csv: {sorted(new_vids & old)[:5]}"
    assert len({r["file"] for r in rows}) == len(rows)
    tmp = ROOT / (OUT_NAME + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ROOT / OUT_NAME)
    k = [r for r in rows if r["kept"] == "True"]
    per_vid = Counter(r["speaker_ref_id"] for r in k)
    print(json.dumps({
        "rows": len(rows), "kept": len(k), "kept_h": round(sum(float(r["duration_s"]) for r in k) / 3600, 3),
        "kept_videos": len(per_vid), "max_per_video": max(per_vid.values()),
        "per_video_hist": dict(sorted(Counter(per_vid.values()).items())),
        "drops": dict(Counter(r["drop_reason"] for r in rows if r["kept"] != "True")),
        "overlap_checked": {"new_videos": len(new_vids), "round1_videos": len(old),
                            "overlap": len(new_vids & old), "removed_before_selection": len(overlap)},
        "per_tar_kept": dict(sorted(Counter(r["source_tar"].split("/")[-1] for r in k).items())),
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
