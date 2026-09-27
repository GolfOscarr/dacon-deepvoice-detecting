#!/usr/bin/env python3
"""N2: the whole-song lists that go with the vocal stems (after `sing_screen.py apply`).

    python scripts/synth3/sing_finalize.py --work /data/project/private/dacon-corpus/interim/_sing_work

Writes
  interim/aisong-kr-en/metadata.csv   every ACE-Step song as a cell-8 whole-file
      candidate, with its seed, caption, lyrics and model revision. kept =
      at least MIN_VOCAL_S of its vocal chunks passed the N2 screen (a song with
      no audible voice would be cell 4, not 8);
  interim/sing-real/whole_songs.csv   every real source song whose vocal chunks
      passed (>= MIN_VOCAL_S) as a cell-5 whole-file candidate: path = the
      ORIGINAL file (no audio duplicated). `in_base` marks songs the v4 manifest
      already holds (under `base_file_id`); the ingester skips those. A base row
      that is a pool-C component (`base_pool` C) is an "instrumental" whose
      vocal stem passed: a pool-C label error for the lead to review.
The same MIN_VOCAL_S on both sides.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
MANIFEST = CORPUS / "manifests/strategy-v4/manifest.parquet"
MIN_VOCAL_S = 8.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    args = ap.parse_args()
    base = pd.read_parquet(MANIFEST, columns=["file_id", "row_kind", "pool", "cell"]).set_index("file_id")

    fk = pd.read_csv(CORPUS / "interim/sing-fake/metadata.csv")
    voc = fk[fk["kept"]].groupby("source_file_id")["duration_s"].sum()
    root = CORPUS / "interim/aisong-kr-en"
    log = root / "gen_log.jsonl"
    if log.exists():
        g = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines() if l.strip()])
        g = g.drop_duplicates("id", keep="last")
        g["file_id"] = "aisong-kr-en:" + g["file"]
        g["vocal_kept_s"] = g["file_id"].map(voc).fillna(0.0).round(2)
        g["kept"] = g["vocal_kept_s"] >= MIN_VOCAL_S
        g["drop_reason"] = g["kept"].map({True: "", False: f"vocal_kept<{MIN_VOCAL_S:g}s"})
        for k in ("repo", "hf_sha", "code", "dit", "lm"):
            g[f"model_{k}"] = g["model"].map(lambda d, k=k: d[k])
        out = pd.DataFrame({
            "file": g["file"], "file_id": g["file_id"], "path": "interim/aisong-kr-en/" + g["file"],
            "cell": 8, "artifact_family": "acestep15", "source_name": "acestep15",
            "speaker_ref_id": "aisong-kr-en/acestep15", "domain_key": "aisong-kr-en|acestep15",
            "lang": g["lang"], "duration_s": g["duration"], "kept": g["kept"],
            "drop_reason": g["drop_reason"], "vocal_kept_s": g["vocal_kept_s"], "seed": g["seed"],
            "caption": g["caption"], "lyrics": g["lyrics"], "master": g["master"],
            **{c: g[c] for c in g.columns if c.startswith("model_")}, "created": g["created"]})
        out.to_csv(root / "metadata.csv", index=False)
        print(f"aisong-kr-en: {len(out)} songs, kept {int(out.kept.sum())}; kept hours by lang:",
              (out[out.kept].groupby("lang")["duration_s"].sum() / 3600).round(2).to_dict())

    rl = pd.read_csv(CORPUS / "interim/sing-real/metadata.csv")
    k = rl[rl["kept"]]
    songs = k.groupby("source_file_id").agg(
        path=("source_path", "first"), corpus=("corpus", "first"),
        source_name=("source_name", "first"), speaker_ref_id=("speaker_ref_id", "first"),
        lang=("lang", "first"), vocal_kept_s=("duration_s", "sum")).reset_index()
    songs = songs[songs["vocal_kept_s"] >= MIN_VOCAL_S].rename(columns={"source_file_id": "file_id"})
    songs["cell"] = 5
    # an fma_medium pick can be an fma_small track (medium contains small): name
    # it by the base's file_id so it is recognised
    tid = songs["file_id"].str.extract(r"^fma-medium:(\d+)\.mp3$")[0]
    small = "fma:fma_small/" + tid.str[:3] + "/" + tid + ".mp3"
    songs["base_file_id"] = songs["file_id"].where(tid.isna(), small)
    songs["in_base"] = songs["base_file_id"].isin(base.index)
    b = base.reindex(songs["base_file_id"])
    songs["base_row_kind"] = b["row_kind"].to_numpy()
    songs["base_pool"] = b["pool"].to_numpy()
    songs["licence_verdict"] = "allow"                    # fma_allow.csv / MUSAN CC BY 4.0
    songs["kept"] = True
    songs.to_csv(CORPUS / "interim/sing-real/whole_songs.csv", index=False)
    print(f"sing-real whole songs: {len(songs)} (new, not in base: {int((~songs.in_base).sum())})")
    print("in base as:", songs[songs.in_base].groupby(["base_row_kind", "base_pool"], dropna=False)
          .size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
