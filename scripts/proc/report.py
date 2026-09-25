"""Summarise interim/proc/metadata.csv: per family x side files / hours kept, drops,
tools. Also normalises the `licence` string per tool (one wording per tool) under the lock.
  python scripts/proc/report.py [--fix-licence]
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import common as C

LICENCE = {
    "facebook/encodec_24khz": "MIT (facebookresearch/encodec: code MIT; the weights ship from that repo "
                              "with no separate licence; HF card licence field empty, so the weights' "
                              "licence is UNVERIFIED)",
    "kyutai/mimi": "CC-BY-4.0 (HF card kyutai/mimi)",
    "HKUSTAudio/xcodec2": "CC-BY-NC-4.0 (HF card HKUSTAudio/xcodec2; code MIT; its semantic encoder "
                          "facebook/w2v-bert-2.0 is MIT)",
}


def main():
    fix = "--fix-licence" in sys.argv
    with C._lock():
        d = pd.read_csv(C.META)
        if fix:
            for t, lic in LICENCE.items():
                d.loc[d.tool == t, "licence"] = lic
            tmp = C.META.with_suffix(".tmp.csv"); d.to_csv(tmp, index=False); os.replace(tmp, C.META)
    dup = d.file.duplicated(keep="last")
    if dup.any():
        print(f"note: {int(dup.sum())} duplicate rows (re-written files); keeping the last")
    d = d[~dup]
    k = d[d.kept.astype(str).str.lower().isin(["true", "1"])]
    t = (d.groupby(["family", "side"]).agg(files=("file", "size"), hours=("duration_s", "sum"))
         .join(k.groupby(["family", "side"]).agg(kept=("file", "size"), kept_h=("duration_s", "sum"))))
    t["hours"] /= 3600; t["kept_h"] /= 3600
    print(t.round(2).to_string())
    print("\nby tool x side (kept h):")
    print((k.groupby(["tool", "side"]).duration_s.sum() / 3600).round(2).unstack().to_string())
    print("\ndrops:"); print(d[d.drop_reason.fillna("") != ""].groupby(["family", "side", "drop_reason"]).size().to_string())
    print("\ntotal kept h by side:", (k.groupby("side").duration_s.sum() / 3600).round(2).to_dict())


if __name__ == "__main__":
    main()
