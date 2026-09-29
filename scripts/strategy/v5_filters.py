#!/usr/bin/env python3
"""strategy-v5's row drops, between extend_manifest.py and build_folds_pinned.py
(docs/training/15). Reads <in>/manifest.parquet, writes <out>/manifest.parquet plus
dropped.parquet (file_id, reason, ...) and filter_report.json. Rules, each on every row
it names, base (v4) rows included:

  zeros   a cloner row (ko/en-synth2, -synth3, -synth2-extra) whose family's zeros.csv
          flags it under the relaxed rule (scripts/synth3/ko_zeros.py: an exact-zero
          run > 20 ms or exact zeros > 5 % inside the speech span), or that the
          zeros.csv never measured. Digital silence is a label shortcut.
  vocal   a pool C/D component the vocal screen passed (its "instrumental" carries a
          voice: interim/sing-real/_screen/music_components_vocal.csv), and a MUSAN
          music component annotated vocals=Y (ANNOTATIONS: id, genres, vocals Y|N, artist).
          Same rule on both pools. Real songs with vocals enter as whole-file cell-5 rows.
  lang    a new singing voice row (sing-real, sing-fake) not in Korean or English, and
          the Korean AI singing stems (sing-fake acestep-ko): real Korean singing is 2 files,
          so Korean singing would be fake-only -- a label shortcut against real K-pop. A
          domain weight of 0 does not remove them: they share the ACE-Step speaker bucket
          with acestep-en and are drawn as its tiles. The whole songs (cell 8) stay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/data/project/private/dacon-corpus")
SYNTH = ("ko-synth2", "en-synth2", "ko-synth3", "en-synth3", "ko-synth2-extra", "en-synth2-extra")
MAX_RUN_MS, MAX_FRAC = 20.0, 0.05


def zero_flags(root: Path) -> pd.Series:
    """path -> flagged (bool) over every <corpus>/<family>/zeros.csv."""
    out = {}
    for corpus in SYNTH:
        for z in sorted((root / "interim" / corpus).glob("*/zeros.csv")):
            d = pd.read_csv(z)
            run = d["max_zero_run_ms"] if "max_zero_run_ms" in d.columns else d["run_ms_max"]
            fl = (run.astype(float) > MAX_RUN_MS) | (d["zero_frac"].astype(float) > MAX_FRAC)
            pre = f"interim/{corpus}/{z.parent.name}/"
            out.update(dict(zip(pre + d["file"].astype(str), fl.to_numpy())))
    return pd.Series(out, dtype=bool)


def musan_vocal_ids(root: Path) -> set[str]:
    ids = set()
    for ann in sorted((root / "interim/musan/openslr-17/musan/music").glob("*/ANNOTATIONS")):
        for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
            f = line.split()
            if len(f) >= 3 and f[2] == "Y":
                ids.add(f"musan-music:{ann.parent.name}/{f[0]}.wav")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-root", default=str(ROOT))
    ap.add_argument("--base-dir", required=True, help="strategy-v4: tells base rows from new")
    a = ap.parse_args()
    root = Path(a.corpus_root)
    m = pd.read_parquet(Path(a.in_dir) / "manifest.parquet")
    base = set(pd.read_parquet(Path(a.base_dir) / "manifest.parquet", columns=["file_id"])["file_id"])
    reason = pd.Series(None, index=m.index, dtype=object)

    flags = zero_flags(root)
    synth = m["corpus"].isin(SYNTH).to_numpy()
    f = m["path"].map(flags)
    reason[synth & f.eq(True).to_numpy()] = "zeros"
    reason[synth & f.isna().to_numpy()] = "zeros-unmeasured"

    v = pd.read_csv(root / "interim/sing-real/_screen/music_components_vocal.csv")
    vocal = set(v.loc[v["passes_vocal_screen"].astype(str) == "True", "file_id"])
    cd = m["pool"].isin(["C", "D"]) & m["row_kind"].eq("component")
    reason[cd & m["file_id"].isin(vocal) & reason.isna()] = "vocal-screen"
    mus = cd & m["corpus"].eq("musan-music") & m["file_id"].isin(musan_vocal_ids(root))
    reason[mus & reason.isna()] = "musan-vocals-Y"

    sing = m["corpus"].isin(["sing-real", "sing-fake"]) & ~m["lang"].isin(["ko", "en"])
    reason[sing & reason.isna()] = "lang"
    ko_sing = m["corpus"].eq("sing-fake") & m["lang"].eq("ko")
    reason[ko_sing & reason.isna()] = "sing-ko-fake-only"

    drop = reason.notna()
    d = m.loc[drop, ["file_id", "corpus", "pool", "row_kind", "duration_s"]].assign(
        reason=reason[drop], base=m.loc[drop, "file_id"].isin(base))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m[~drop].reset_index(drop=True).to_parquet(out / "manifest.parquet", index=False)
    d.to_parquet(out / "dropped.parquet", index=False)
    g = d.groupby(["reason", "base", "pool", "corpus"])["duration_s"].agg(["size", "sum"])
    rep = {"rows_in": int(len(m)), "rows_out": int((~drop).sum()),
           "dropped": {"/".join(map(str, k)): {"rows": int(r["size"]), "hours": round(r["sum"] / 3600, 2)}
                       for k, r in g.iterrows()}}
    (out / "filter_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
