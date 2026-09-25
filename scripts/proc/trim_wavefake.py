"""Lead's rule (2026-09-25 22:20): WaveFake (LJ) <= 25 % of the fake en side. MLAAD en (8.1 h in
train_val) is already fully drawn, so fake en shrinks to MLAAD + MLAAD/3 per processing family;
real en is subsampled per processing family to the same hours (languages stay matched).
A subset of the existing draw: nothing is re-processed. Excluded rows move to
metadata_excluded.csv and their WAVs to _excluded/ (reversible). The plan becomes the trimmed one
(draw.parquet); the original is kept as _plan/draw_v1.parquet. Idempotent.
"""
from __future__ import annotations
import os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import common as C

SEED, WF_SHARE = 20260925, 0.249   # 0.1 pt margin: processed durations drift from source


def rr(rows: pd.DataFrame, target_s: float, rng, strict: bool = False) -> pd.DataFrame:
    rows = rows.sample(frac=1.0, random_state=int(rng.integers(2**31)))
    q = {g: list(d.index) for g, d in rows.groupby("group", sort=True)}
    order = list(q); rng.shuffle(order)
    out, tot = [], 0.0
    while tot < target_s and any(q.values()):
        for g in order:
            if tot >= target_s: break
            if q[g]:
                i = q[g].pop()
                if strict and tot + rows.at[i, "duration_s"] > target_s:
                    continue                       # never exceed the cap
                out.append(i); tot += rows.at[i, "duration_s"]
    return rows.loc[out]


def main():
    pd_ = C.OUT / "_plan"
    v1 = pd_ / "draw_v1.parquet"
    if not v1.exists():
        shutil.copy2(pd_ / "draw.parquet", v1)
    d = pd.read_parquet(v1)
    rng = np.random.default_rng([SEED, 25])
    keep = [d[d.lang != "en"]]
    for fam in sorted(d.family.unique()):
        fe = d[(d.family == fam) & (d.side == "fake") & (d.lang == "en")]
        ml = fe[fe.corpus != "wavefake"]
        wf = rr(fe[fe.corpus == "wavefake"], ml.duration_s.sum() * WF_SHARE / (1 - WF_SHARE), rng,
                strict=True)
        fake_h = ml.duration_s.sum() + wf.duration_s.sum()
        re = rr(d[(d.family == fam) & (d.side == "real") & (d.lang == "en")], fake_h, rng)
        keep += [ml, wf, re]
    new = pd.concat(keep).sort_index()
    new.to_parquet(pd_ / "draw.parquet", index=False)
    want = set(new.family + "/" + new.side + "/" + new["name"] + ".wav")
    with C._lock():
        m = pd.read_csv(C.META)
        old = C.OUT / "metadata_excluded.csv"
        if old.exists():                           # restore anything the new plan wants back
            prev = pd.read_csv(old)
            back = prev[prev.file.isin(want) & ~prev.file.isin(m.file)]
            for f in back.file:
                src = C.OUT / "_excluded" / f
                if src.exists(): os.replace(src, C.OUT / f)
            m = pd.concat([m, back], ignore_index=True)
            rest = prev[~prev.file.isin(want)].drop_duplicates("file", keep="last")
            tmp = old.with_suffix(".tmp.csv"); rest.to_csv(tmp, index=False); os.replace(tmp, old)
        ex = m[~m.file.isin(want)]
        if True:
            ex.to_csv(old, mode="a", header=not old.exists(), index=False)
            for f in ex.file:
                src, dst = C.OUT / f, C.OUT / "_excluded" / f
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); os.replace(src, dst)
            tmp = C.META.with_suffix(".tmp.csv"); m[m.file.isin(want)].to_csv(tmp, index=False)
            os.replace(tmp, C.META)
    print(f"excluded {len(ex)} rows; plan {len(d)} -> {len(new)}")
    k = new
    print((k.groupby(["side", "lang"]).duration_s.sum() / 3600).round(2).to_dict())
    fe = k[(k.side == "fake") & (k.lang == "en")]
    print("wavefake share of fake en:", round(fe[fe.corpus == "wavefake"].duration_s.sum() / fe.duration_s.sum(), 4))


if __name__ == "__main__":
    main()
