#!/usr/bin/env python3
"""Vocal-activity score for every v4 music component (pools C and D), for the lead.

    python scripts/synth3/sing_components.py --work /data/project/private/dacon-corpus/interim/sing-real/_screen

Input: `sing_select.py --components` (whole file tiled in 30 s windows),
`sing_separate.py` (htdemucs vocals, vocal-active chunks) and
`sing_screen.py measure`, all with `--out-base <work>/stems`. The SAME code
path and thresholds on C (real) and D (fake).

A chunk counts as vocal when it passes the N2 chunk screen of sing_screen.py
(rms, vox_rel, Silero speech_ratio, Whisper lid_p). The language rule (zh/ja)
is not a vocal-presence test and is not applied here. Per component file:

  vocal_active_s        seconds of screened vocal chunks
  vocal_ratio           vocal_active_s / duration_s
  vocal_rms_dbfs        energy-weighted RMS of those chunks (NaN if none)
  n_windows / n_chunks  30 s windows separated / vocal-active chunks cut
  passes_vocal_screen   vocal_active_s >= MIN_VOCAL_S or vocal_ratio >= MIN_RATIO
                        (i.e. the component carries vocals: drop / relabel it)

Output: <work>/music_components_vocal.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sing_screen import LIDP_MIN, REL_MIN, RMS_MIN, SPEECH_MIN  # noqa: E402

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
MANIFEST = CORPUS / "manifests/strategy-v4/manifest.parquet"
MIN_VOCAL_S, MIN_RATIO = 8.0, 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    args = ap.parse_args()
    sel = pd.read_csv(args.work / "selection_components.csv")
    sep = pd.concat([pd.read_csv(p) for p in sorted(args.work.glob("sep_comp_*.csv"))])
    sep = sep.drop_duplicates(["sel_idx", "file"], keep="last")
    done = set(sep["sel_idx"])
    ch = sep[sep["file"].notna()].merge(
        pd.read_csv(args.work / "screen_cache.csv").drop_duplicates(["side", "file"], keep="last"),
        on=["side", "file"], how="left")
    ok = ((ch["rms_dbfs"] >= RMS_MIN) & (ch["vox_rel_db"] >= REL_MIN)
          & (ch["speech_ratio"] >= SPEECH_MIN) & (ch["lid_p"] >= LIDP_MIN))
    unmeasured = ch["speech_ratio"].isna()
    v = ch[ok].copy()
    v["e"] = v["duration_s"] * 10 ** (v["rms_dbfs"] / 10)
    per = v.groupby("source_file_id").agg(vocal_active_s=("duration_s", "sum"), e=("e", "sum"))
    per["vocal_rms_dbfs"] = 10 * np.log10(per["e"] / per["vocal_active_s"])
    m = pd.read_parquet(MANIFEST, columns=["file_id", "pool", "corpus", "duration_s"])
    m = m[m["file_id"].isin(set(sel["file_id"]))].drop_duplicates("file_id").set_index("file_id")
    out = m.join(per[["vocal_active_s", "vocal_rms_dbfs"]])
    out["vocal_active_s"] = out["vocal_active_s"].fillna(0.0).round(2)
    out["vocal_ratio"] = (out["vocal_active_s"] / out["duration_s"]).clip(upper=1).round(4)
    out["vocal_rms_dbfs"] = out["vocal_rms_dbfs"].round(2)
    wins = sel.groupby("file_id")["sel_idx"].agg(list)
    out["n_windows"] = wins.map(len)
    out["screened"] = wins.map(lambda ix: all(i in done for i in ix))
    out["n_chunks"] = ch.groupby("source_file_id").size().reindex(out.index).fillna(0).astype(int)
    out["n_unmeasured"] = ch[unmeasured].groupby("source_file_id").size().reindex(out.index).fillna(0).astype(int)
    out["passes_vocal_screen"] = (out["vocal_active_s"] >= MIN_VOCAL_S) | (out["vocal_ratio"] >= MIN_RATIO)
    out["passes_vocal_screen"] = out["passes_vocal_screen"].astype("boolean")
    out.loc[~out["screened"], "passes_vocal_screen"] = pd.NA
    out = out.reset_index()[["file_id", "pool", "corpus", "duration_s", "vocal_active_s", "vocal_ratio",
                             "vocal_rms_dbfs", "passes_vocal_screen", "n_windows", "n_chunks",
                             "n_unmeasured", "screened"]]
    out.to_csv(args.work / "music_components_vocal.csv", index=False)
    g = out.groupby(["pool", "corpus"])
    print(pd.DataFrame({
        "files": g.size(), "screened": g["screened"].sum(), "hours": g["duration_s"].sum() / 3600,
        "vocal_files": g["passes_vocal_screen"].sum(),
        "vocal_file_h": g.apply(lambda x: x.loc[x.passes_vocal_screen.fillna(False).astype(bool), "duration_s"].sum() / 3600),
        "vocal_active_h": g["vocal_active_s"].sum() / 3600,
    }).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
