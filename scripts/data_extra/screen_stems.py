#!/usr/bin/env python3
"""Vocal-bleed and silence screen over the Demucs stems; writes metadata.csv.

    python scripts/data_extra/screen_stems.py --work <work> [--threads 16] [--apply]

For every stem logged in `<work>/sep_*.csv`: downmix to mono, resample 44.1 k
-> 16 k, run the repo's vendored Silero VAD (models/vendor/silero_vad, the same
model and 512-sample chunks eda/extract/content.py uses) and record

    speech_ratio = fraction of 32 ms chunks with P(speech) >= 0.5
                   (== eda's `vad_speech_ratio_50`)
    rms_dbfs     = 20 log10(rms of the mono mix)

The SAME rule on both sides:  kept = speech_ratio <= 0.10 and rms_dbfs >= -50.

Output: `<root>/metadata.csv` for sonics-sep and realmusic-sep with columns
  file, source_path, source_file_id, generator_version | corpus, offset_s,
  duration_s, sample_rate, speech_ratio, rms_dbfs, kept, drop_reason
plus extras: artifact_family, speaker_ref_id, src_sr, src_channels, peak,
speech_prob_max.

With `--apply`, dropped stems are MOVED to `<root>/_dropped/<same rel path>`
so a suffix walk of the root sees only kept files; `file` then points at the
moved location. Without it, nothing moves (dry run: metadata only, with
`file` at the current location). Rerunnable: measurements are cached in
`<work>/screen_cache.csv` keyed by (side, out_rel); files already under
`_dropped/` are found there.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.vendor.silero_vad import SAMPLE_RATE, speech_probabilities  # noqa: E402

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
ROOTS = {"fake": CORPUS / "interim/sonics-sep", "real": CORPUS / "interim/realmusic-sep"}
GROUP_COL = {"fake": "generator_version", "real": "corpus"}
SPEECH_MAX, RMS_MIN = 0.10, -50.0
COLS = ["file", "source_path", "source_file_id", "offset_s", "duration_s", "sample_rate",
        "speech_ratio", "rms_dbfs", "kept", "drop_reason", "artifact_family",
        "speaker_ref_id", "src_sr", "src_channels", "peak", "speech_prob_max"]

_print_lock = threading.Lock()


def locate(root: pathlib.Path, rel: str) -> pathlib.Path | None:
    for p in (root / rel, root / "_dropped" / rel):
        if p.exists():
            return p
    return None


def measure(path: pathlib.Path) -> dict:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)     # (T, C)
    mono = y.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
    x = torch.from_numpy(mono)
    if sr != SAMPLE_RATE:
        x = AF.resample(x, sr, SAMPLE_RATE)
    probs = speech_probabilities(x.numpy(), SAMPLE_RATE)
    return {"speech_ratio": float((probs >= 0.5).mean()) if probs.size else None,
            "speech_prob_max": float(probs.max()) if probs.size else None,
            "rms_dbfs": float(rms_dbfs), "vad_chunks": int(probs.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--apply", action="store_true", help="move dropped stems to _dropped/")
    args = ap.parse_args()
    torch.set_num_threads(1)

    logs = sorted(args.work.glob("sep_*.csv"))
    sep = pd.concat([pd.read_csv(p) for p in logs], ignore_index=True)
    sep = sep.drop_duplicates(["side", "out_rel"], keep="last")
    sel = pd.read_csv(args.work / "selection.csv").set_index("sel_idx")
    sep = sep.join(sel[["artifact_family", "speaker_ref_id"]], on="sel_idx")
    sep = sep[sep["sep_ok"] == True].copy()                           # noqa: E712
    print(f"{len(sep)} separated stems from {len(logs)} shard log(s)")

    cache_path = args.work / "screen_cache.csv"
    cache_cols = ["speech_ratio", "speech_prob_max", "rms_dbfs", "vad_chunks"]
    cache = pd.read_csv(cache_path).set_index(["side", "out_rel"]) if cache_path.exists() \
        else pd.DataFrame(columns=cache_cols,
                          index=pd.MultiIndex.from_tuples([], names=["side", "out_rel"]))
    todo = [r for r in sep.itertuples(index=False) if (r.side, r.out_rel) not in cache.index]
    print(f"{len(todo)} to measure, {len(sep) - len(todo)} cached")

    results, done = {}, [0]

    def work(r):
        p = locate(ROOTS[r.side], r.out_rel)
        if p is None:
            return (r.side, r.out_rel), {"speech_ratio": None, "speech_prob_max": None,
                                         "rms_dbfs": None, "vad_chunks": 0}
        out = measure(p)
        done[0] += 1
        if done[0] % 500 == 0:
            with _print_lock:
                print(f"  measured {done[0]}/{len(todo)}", flush=True)
        return (r.side, r.out_rel), out

    with ThreadPoolExecutor(args.threads) as ex:
        for key, out in ex.map(work, todo):
            results[key] = out
    if results:
        new = pd.DataFrame.from_dict(results, orient="index")
        new.index = pd.MultiIndex.from_tuples(new.index, names=["side", "out_rel"])
        cache = pd.concat([cache, new])
        cache.to_csv(cache_path)

    m = sep.join(cache, on=["side", "out_rel"])
    m["drop_reason"] = ""
    m.loc[m["speech_ratio"].isna(), "drop_reason"] = "missing_or_unmeasurable"
    m.loc[m["speech_ratio"] > SPEECH_MAX, "drop_reason"] = "speech_ratio>0.10"
    m.loc[m["rms_dbfs"] < RMS_MIN, "drop_reason"] += ";rms<-50dBFS"
    m["drop_reason"] = m["drop_reason"].str.strip(";")
    m["kept"] = m["drop_reason"] == ""

    for side, root in ROOTS.items():
        d = m[m["side"] == side].copy()
        if d.empty:
            continue
        files = []
        for r in d.itertuples(index=False):
            cur = locate(root, r.out_rel)
            want = root / r.out_rel if r.kept else root / "_dropped" / r.out_rel
            if args.apply and cur is not None and cur != want:
                want.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cur), str(want))
                cur = want
            files.append(str(cur.relative_to(root)) if cur else "")
        d["file"] = files
        d = d.rename(columns={"group": GROUP_COL[side]})
        cols = COLS[:3] + [GROUP_COL[side]] + COLS[3:]
        d = d.sort_values("file")[cols]
        d.to_csv(root / "metadata.csv", index=False)
        print(f"\n== {side}: {root}/metadata.csv ({len(d)} rows)")
        g = d.groupby(GROUP_COL[side])
        print(pd.DataFrame({
            "files": g.size(), "hours": g["duration_s"].sum() / 3600,
            "kept_files": g["kept"].sum(), "kept_hours": g.apply(lambda x: x.loc[x.kept, "duration_s"].sum() / 3600),
            "drop_speech": g.apply(lambda x: x.drop_reason.str.contains("speech").sum()),
            "drop_rms": g.apply(lambda x: x.drop_reason.str.contains("rms").sum()),
        }).round(2).to_string())
        print("speech_ratio quantiles:",
              d["speech_ratio"].quantile([.5, .9, .95, .99, 1.0]).round(3).to_dict())
        print("rms_dbfs quantiles:",
              d["rms_dbfs"].replace(-np.inf, np.nan).quantile([0, .01, .05, .5]).round(1).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
