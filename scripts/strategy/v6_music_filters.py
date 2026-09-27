#!/usr/bin/env python3
"""strategy-v6's music row drops, after cut_music_rows.py (independent audit, 2026-09-28):

    $V scripts/strategy/v6_music_filters.py --in-dir <v6>/_cut --out <v6> [--acestep-rows 6900]

zeros   EVERY pool C / D music component (both sides, after the 10 s cut) with an
        exact-zero run > 20 ms or exact zeros > 5 % (docs/training/16 §5.1, the
        v5_filters.py rule) inside the drawable span: the file minus edge_margin_s
        (0.5 s) at each end, where no tile ever reads. Measured on the cache16k array
        (what training reads); a sample is zero when every channel is 0. 12.3 % of
        fakemusiccaps rows tripped it against <= 0.4 % of real music: a fake-only cue.
sonics  sonics-sep pieces are subsampled to SONICS_SHARE of the fake-music rows the
        draw sees (the music draw's tile share is a row count), counting the
        acestep-inst rows v6b will add (--acestep-rows): its stems' spectral tilt
        (95 % rolloff ~445 Hz) became a fake cue once the cut raised its share to .52.
        Whole parents are kept or dropped, in a fixed order (sha256 of seed + parent
        id); only train_val rows are candidates (PROBE is never touched).

Writes <out>/manifest.parquet, folds.parquet (+ caveats / vg1 / report), dropped.parquet,
music_filter_report.json, and copies cut_map.parquet (the provenance ledger). VG1 re-checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.cache import cache_path                       # noqa: E402
from processing.splits import write_outputs                   # noqa: E402
from training.foldcheck import check_split_integrity           # noqa: E402
from training.folds import FoldPlan                            # noqa: E402

CACHE = Path("/data/project/private/dacon-corpus/cache16k")
SR, EDGE_S, MAX_RUN_MS, MAX_FRAC = 16000, 0.5, 20.0, 0.05
SONICS_SHARE, SEED = 0.33, "strategy-v6-sonics-0"


def zero_stats(fid: str, cache_root: Path) -> tuple[float, float]:
    """(longest exact-zero run in ms, zero fraction) inside the drawable span."""
    a = np.load(cache_path(cache_root, fid), mmap_mode="r")
    e = int(EDGE_S * SR)
    z = (np.asarray(a[:, e:a.shape[-1] - e]) == 0).all(0)
    if not z.size or not z.any():
        return 0.0, 0.0
    dz = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    run = (np.flatnonzero(dz == -1) - np.flatnonzero(dz == 1)).max()
    return float(run) / SR * 1000, float(z.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", default=str(CACHE))
    ap.add_argument("--acestep-rows", type=int, default=6900,
                    help="acestep-inst train rows v6b adds (2,300 clips x 3), 0 = none")
    ap.add_argument("--sonics-share", type=float, default=SONICS_SHARE)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    src, out = Path(a.in_dir), Path(a.out)
    if (out / "manifest.parquet").exists() or (out / "folds.parquet").exists():
        raise SystemExit(f"FATAL: {out} already has a manifest / fold table")
    m = pd.read_parquet(src / "manifest.parquet")
    f = pd.read_parquet(src / "folds.parquet")
    music = (m["pool"].isin(["C", "D"]) & m["row_kind"].eq("component")).to_numpy()
    ids = m.loc[music, "file_id"].astype(str).tolist()
    with ThreadPoolExecutor(a.workers) as ex:
        st = list(ex.map(lambda x: zero_stats(x, Path(a.cache_root)), ids))
    z = pd.DataFrame(st, columns=["zero_run_ms", "zero_frac"], index=m.index[music])
    reason = pd.Series(None, index=m.index, dtype=object)
    flag = (z["zero_run_ms"] > MAX_RUN_MS) | (z["zero_frac"] > MAX_FRAC)
    reason[flag[flag].index] = "zeros"

    # sonics-sep: whole parents, fixed order, train_val only
    sl = m[["file_id"]].merge(f[["file_id", "slice"]], on="file_id", how="left")["slice"].to_numpy()
    live = reason.isna().to_numpy() & (sl == "train_val")
    fmc = int((live & m["corpus"].eq("fakemusiccaps").to_numpy()).sum())
    oth_d = int((live & m["pool"].eq("D").to_numpy() & m["row_kind"].eq("component").to_numpy()
                 & ~m["corpus"].isin(["fakemusiccaps", "sonics-sep"]).to_numpy()).sum())
    others = fmc + oth_d + a.acestep_rows
    target = int(round(a.sonics_share / (1 - a.sonics_share) * others))
    son = live & m["corpus"].eq("sonics-sep").to_numpy()
    parent = m["file_id"].str.replace(r"~\d+$", "", regex=True)
    sp = pd.DataFrame({"parent": parent[son]}).groupby("parent").size()
    order = sorted(sp.index, key=lambda p: hashlib.sha256(f"{SEED}:{p}".encode()).hexdigest())
    keep_par, n = set(), 0
    for p in order:
        if n >= target:
            break
        keep_par.add(p)
        n += int(sp[p])
    sub = son & ~parent.isin(keep_par).to_numpy()
    reason[sub] = "sonics-subsample"

    drop = reason.notna().to_numpy()
    d = m.loc[drop, ["file_id", "corpus", "pool", "duration_s"]].assign(reason=reason[drop])
    d = d.join(z, how="left")
    mm = m[~drop].reset_index(drop=True)
    ff = f[f["file_id"].isin(set(mm["file_id"]))].reset_index(drop=True)
    vg1 = check_split_integrity(ff)
    if not vg1.ok:
        raise SystemExit(f"FATAL: VG1 fails:\n{vg1}")
    out.mkdir(parents=True, exist_ok=True)
    mm.to_parquet(out / "manifest.parquet", index=False)
    d.to_parquet(out / "dropped.parquet", index=False)
    if (src / "cut_map.parquet").exists():
        shutil.copy2(src / "cut_map.parquet", out / "cut_map.parquet")
    zm = m.loc[music, ["corpus", "pool", "duration_s"]].join(z)
    per = {}
    for (p, c), g in zm.groupby(["pool", "corpus"]):
        fl = (g["zero_run_ms"] > MAX_RUN_MS) | (g["zero_frac"] > MAX_FRAC)
        per[f"{p}/{c}"] = {"rows": int(len(g)), "zeros_rows": int(fl.sum()),
                           "zeros_hours": round(float(g.loc[fl, "duration_s"].sum()) / 3600, 2)}
    rep = {"rule": {"edge_s": EDGE_S, "max_run_ms": MAX_RUN_MS, "max_frac": MAX_FRAC},
           "zeros_by_corpus": per,
           "sonics": {"share": a.sonics_share, "acestep_rows_assumed": a.acestep_rows,
                      "train_val_fakemusiccaps_rows": fmc, "target_rows": target,
                      "kept_rows": int(n), "kept_parents": len(keep_par),
                      "dropped_rows": int(sub.sum()),
                      "dropped_hours": round(float(m.loc[sub, "duration_s"].sum()) / 3600, 2)},
           "rows_in": int(len(m)), "rows_out": int(len(mm))}
    (out / "music_filter_report.json").write_text(json.dumps(rep, indent=1))
    write_outputs(out, FoldPlan(ff, (f"music filters on {src}: zeros + sonics-sep subsample; "
                                     f"{int(drop.sum())} rows dropped, no fold moved",)), vg1, rep)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
