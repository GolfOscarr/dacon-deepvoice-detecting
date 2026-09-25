"""S3: build interim/emilia-ko/ (conversational real Korean) + metadata.csv.

Inputs: interim/emilia-ko/_raw/{Emilia,Emilia-YODAS}/KO/*.tar (ko_emilia_download.py) and the
FLEURS ko_kr audio already unpacked under interim/emilia-ko/fleurs/<split>/<split>/*.wav.

Selection (deterministic): per part, every segment with 3 <= duration <= 30 s and Hangul in its
transcript is a candidate; candidates are taken round-robin over source keys (Emilia `speaker`
= one source video/episode), in a seed-shuffled order, until the part reaches --hours-per-part.
So hours are spread over as many sources as the downloaded tars hold.

Output: <part>/<tar stem>/<id>.mp3 + <id>.json (original bytes, not re-encoded: libsndfile
>= 1.1 reads mp3, checked per file) and metadata.csv:
  file, part, speaker_ref_id, text, duration_s, sample_rate, licence, kept, drop_reason,
  source_tar, dnsmos, orig_format
speaker_ref_id: "emilia-ko/<part>/<source key>" (the part is in the key because Emilia and
Emilia-YODAS reuse the same KO_Bxxxxx_Sxxxxx names) or "fleurs-ko/<split>-<gender>" (FLEURS
publishes no speaker id; train speakers are disjoint from dev/test, so split x gender is a
conservative atom).
kept = 3 <= duration <= 30, finite, RMS >= -50 dBFS, transcript contains Hangul.

  python scripts/synth2/ko_emilia_index.py [--hours-per-part 40] [--workers 16]
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/data/project/private/dacon-corpus/interim/emilia-ko")
RAW = ROOT / "_raw"
PARTS = {"emilia": ("Emilia/KO", "CC-BY-NC-4.0"), "yodas": ("Emilia-YODAS/KO", "CC-BY-4.0")}
FLEURS_LIC = "CC-BY-4.0"
COLS = ["file", "part", "speaker_ref_id", "text", "duration_s", "sample_rate", "licence", "kept",
        "drop_reason", "source_tar", "dnsmos", "orig_format", "diar_speaker"]
HANGUL = re.compile(r"[가-힣]")


def check(path: Path, text: str) -> tuple[float, int, bool, str]:
    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        return 0.0, 0, False, f"unreadable:{type(e).__name__}"
    if a.ndim > 1:
        a = a.mean(axis=1)
    dur = len(a) / sr if sr else 0.0
    why = []
    if not np.isfinite(a).all():
        why.append("nonfinite")
    rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
    if 20 * np.log10(max(rms, 1e-9)) < -50:
        why.append("silent")
    if not 3.0 <= dur <= 30.0:
        why.append("duration")
    if not HANGUL.search(text):
        why.append("not_korean")
    return dur, sr, not why, "|".join(why)


def fleurs_rows() -> list[dict]:
    rows = []
    for split in ("train", "dev", "test"):
        tsv = ROOT / "_fleurs_raw" / "data" / "ko_kr" / f"{split}.tsv"
        for line in tsv.read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) < 7:
                continue
            rel = f"fleurs/{split}/{split}/{f[1]}"
            rows.append({"file": rel, "part": "fleurs", "speaker_ref_id": f"fleurs-ko/{split}-{f[6].strip().lower()}",
                         "text": f[2].strip(), "licence": FLEURS_LIC, "source_tar": f"{split}.tar.gz",
                         "dnsmos": "", "orig_format": "wav", "diar_speaker": ""})
    return rows


def scan_tar(tar: Path) -> list[dict]:
    out = []
    with tarfile.open(tar) as t:
        for m in t:
            if m.name.endswith(".json"):
                j = json.load(t.extractfile(m))
                j["_tar"] = tar.name
                j["id"] = j.get("id") or j.get("_id") or Path(m.name).stem
                # source key: Emilia `speaker` is one source file; YODAS `speaker` is
                # <video>_SPEAKER_nn (diarised), so the video id is the conservative atom
                spk = j["speaker"]
                j["src"] = spk.split("_SPEAKER_")[0] if "_SPEAKER_" in spk else spk
                out.append(j)
    return out


def extract(args) -> list[dict]:
    tar, part, lic, picks = args
    rows = []
    stem = tar.stem
    dest = ROOT / part / stem
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        for m in t:
            key = Path(m.name).stem
            if key not in picks:
                continue
            p = dest / Path(m.name).name
            if not p.exists():
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_bytes(t.extractfile(m).read())
                os.replace(tmp, p)
    for key, j in picks.items():
        mp3 = dest / f"{key}.mp3"
        text = j["text"].strip()
        dur, sr, ok, why = check(mp3, text) if mp3.exists() else (0.0, 0, False, "missing")
        rows.append({"file": f"{part}/{stem}/{key}.mp3", "part": part,
                     "speaker_ref_id": f"emilia-ko/{part}/{j['src']}", "text": text,
                     "diar_speaker": j["speaker"],
                     "duration_s": f"{dur:.3f}", "sample_rate": sr, "licence": lic, "kept": str(ok),
                     "drop_reason": why, "source_tar": f"{PARTS[part][0]}/{tar.name}",
                     "dnsmos": j.get("dnsmos", ""), "orig_format": "mp3"})
    return rows


def fleurs_check(r: dict) -> dict:
    dur, sr, ok, why = check(ROOT / r["file"], r["text"])
    return {**r, "duration_s": f"{dur:.3f}", "sample_rate": sr, "kept": str(ok), "drop_reason": why}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours-per-part", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fleurs-only", action="store_true")
    a = ap.parse_args()

    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(fleurs_check, fleurs_rows(), chunksize=64))
        print(f"fleurs: {len(rows)} rows, {sum(r['kept'] == 'True' for r in rows)} kept", flush=True)
        if not a.fleurs_only:
            for part, (sub, lic) in PARTS.items():
                tars = sorted((RAW / sub).glob("*.tar"))
                cand = [j for js in ex.map(scan_tar, tars) for j in js
                        if 3.0 <= float(j["duration"]) <= 30.0 and HANGUL.search(j["text"])]
                rng = random.Random(f"{a.seed}:{part}")
                by_src = defaultdict(list)
                for j in sorted(cand, key=lambda j: j["id"]):
                    by_src[j["src"]].append(j)
                srcs = sorted(by_src)
                rng.shuffle(srcs)
                for s in srcs:
                    rng.shuffle(by_src[s])
                picked, tot, rnd = [], 0.0, 0
                while tot < a.hours_per_part * 3600:
                    added = False
                    for s in srcs:
                        if rnd < len(by_src[s]):
                            j = by_src[s][rnd]
                            picked.append(j)
                            tot += float(j["duration"])
                            added = True
                            if tot >= a.hours_per_part * 3600:
                                break
                    if not added:
                        break
                    rnd += 1
                print(f"{part}: {len(tars)} tars, {len(cand)} candidates "
                      f"({sum(float(j['duration']) for j in cand)/3600:.1f} h) over {len(srcs)} sources; "
                      f"picked {len(picked)} = {tot/3600:.1f} h, {len({j['src'] for j in picked})} sources, "
                      f"{rnd + 1} rounds", flush=True)
                per_tar = defaultdict(dict)
                for j in picked:
                    per_tar[j["_tar"]][j["id"]] = j
                jobs = [(RAW / sub / tn, part, lic, picks) for tn, picks in sorted(per_tar.items())]
                for rs in ex.map(extract, jobs):
                    rows.extend(rs)
                # keep the per-segment json next to the audio (original bytes are in the tar;
                # the extractor wrote only picked members, json included)
    tmp = ROOT / "metadata.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, ROOT / "metadata.csv")
    from collections import Counter
    k = [r for r in rows if r["kept"] == "True"]
    print("kept by part:", {p: (n, round(sum(float(r['duration_s']) for r in k if r['part'] == p) / 3600, 2))
                            for p, n in Counter(r["part"] for r in k).items()},
          "speakers:", {p: len({r['speaker_ref_id'] for r in k if r['part'] == p}) for p in ("emilia", "yodas", "fleurs")},
          "drops:", Counter(r["drop_reason"] for r in rows if r["kept"] != "True"), flush=True)


if __name__ == "__main__":
    main()
