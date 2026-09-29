#!/usr/bin/env python3
"""Select the SONICS (fake) and pool-C (real) music that goes through Demucs.

    python scripts/data_extra/select_music_sep.py --out /data/project/private/dacon-corpus/interim/_music_sep_work

Plan 07 §1 D-b. One deterministic selection (seed 0) for BOTH sides, written to
`<out>/selection.csv`, which `separate_music.py` and `screen_stems.py` consume.

Rules, identical on both sides:
  * one excerpt per source file, at most `--max-excerpt-s` (90 s), taken from
    the MIDDLE of the file: offset = max(0, (duration - 90) / 2);
  * files are shuffled within their group with seed 0 and taken in that order
    until the group's hour target is met.

Groups:
  * fake  -- one group per SONICS generator VERSION (`speaker_ref_id`, e.g.
    sonics/chirp-v3.5), equal hours each (`--fake-hours` / 5);
  * real  -- pool-C component rows: musan-music (all of it, it is the scarce
    one: 660 files) then fma to fill `--real-hours`.

Durations come from the manifest (`duration_s`); the separator re-measures on
decode and records the actual excerpt length.
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
MANIFEST = CORPUS / "manifests/strategy-v2/manifest.parquet"


def take_hours(df: pd.DataFrame, hours: float, max_s: float, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle `df` and take rows until the excerpted hours reach `hours`."""
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    excerpt = np.minimum(df["duration_s"].to_numpy(), max_s)
    cum = np.cumsum(excerpt) / 3600.0
    n = int(np.searchsorted(cum, hours)) + 1
    return df.iloc[:min(n, len(df))].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    ap.add_argument("--fake-hours", type=float, default=63.0,
                    help="SONICS hours before the drop rule, split evenly over versions")
    ap.add_argument("--real-hours", type=float, default=42.0,
                    help="real music hours before the drop rule")
    ap.add_argument("--max-excerpt-s", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = pd.read_parquet(args.manifest)
    keep_cols = ["file_id", "path", "duration_s", "orig_sr", "orig_channels",
                 "corpus", "artifact_family", "speaker_ref_id"]
    rows = []

    fake = m[(m["corpus"] == "sonics")][keep_cols]
    versions = sorted(fake["speaker_ref_id"].unique())
    per_version = args.fake_hours / len(versions)
    for i, ver in enumerate(versions):
        rng = np.random.default_rng([args.seed, 1, i])
        sel = take_hours(fake[fake["speaker_ref_id"] == ver], per_version, args.max_excerpt_s, rng)
        sel["side"] = "fake"
        sel["group"] = ver.split("/", 1)[1]          # 'chirp-v3.5'
        rows.append(sel)

    real = m[m["corpus"].isin(["musan-music", "fma"]) & (m["pool"] == "C")][keep_cols]
    musan = real[real["corpus"] == "musan-music"]
    rng = np.random.default_rng([args.seed, 2, 0])
    sel = take_hours(musan, 1e9, args.max_excerpt_s, rng)  # all of it
    sel["side"] = "real"; sel["group"] = "musan-music"
    rows.append(sel)
    got = float(np.minimum(sel["duration_s"], args.max_excerpt_s).sum() / 3600)
    rng = np.random.default_rng([args.seed, 2, 1])
    sel = take_hours(real[real["corpus"] == "fma"], max(args.real_hours - got, 0), args.max_excerpt_s, rng)
    sel["side"] = "real"; sel["group"] = "fma"
    rows.append(sel)

    df = pd.concat(rows, ignore_index=True)
    df["offset_s"] = np.maximum(0.0, (df["duration_s"] - args.max_excerpt_s) / 2).round(3)
    df["excerpt_s"] = np.minimum(df["duration_s"], args.max_excerpt_s).round(3)
    # <id>: the source file's stem, unique within a group on both sides.
    df["out_id"] = df["path"].map(lambda p: pathlib.Path(p).stem)
    assert not df.duplicated(["side", "group", "out_id"]).any(), "output id collision"
    df["out_rel"] = df["group"] + "/" + df["out_id"] + ".wav"
    # Interleave sides so a shard cut anywhere holds both labels in proportion.
    rng = np.random.default_rng([args.seed, 3])
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.index.name = "sel_idx"

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "selection.csv")
    summary = df.groupby(["side", "group"]).agg(files=("path", "size"),
                                                hours=("excerpt_s", lambda x: x.sum() / 3600))
    print(summary.to_string())
    print(df.groupby("side")["excerpt_s"].sum().div(3600).rename("hours").to_string())
    print(f"wrote {args.out / 'selection.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
