"""S1 source draw (docs/training/09 §2 S1): real (pool A) and fake (pool B) voice
subsets with the same language mix, from strategy-v3 train_val only (never PROBE).

Round-robin over groups -> speakers -> files, so hours spread as evenly as the pool
allows: fake groups = artifact_family, real groups = speaker_ref_id. Speaker and group
shares are capped at `--cap` of the side when the pool allows it; where a language's
pool cannot fill its target under the cap (fake ko: 6 big families), the cap is lifted
by water-filling and the real max share is reported.

Each drawn file gets one processing family, assigned in rotation within
(side, lang, group) so every family sees every group on both sides.

Output: interim/proc/_plan/draw.parquet (+ draw_report.json). Deterministic in --seed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/project/private/dacon-corpus")
V3 = ROOT / "manifests" / "strategy-v3"
FAMILIES = ("proc-encodec", "proc-dac", "proc-codec-lm", "proc-enhance", "proc-dsp")


def rr_draw(pool: pd.DataFrame, group_col: str, target_s: float, cap_s: float,
            rng: np.random.Generator) -> pd.DataFrame:
    """Round-robin: groups in rotation, each group's speakers in rotation, one file per
    turn. A group/speaker at `cap_s` sits out while any uncapped one remains."""
    pool = pool.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    queues = {}
    for g, dg in pool.groupby(group_col, sort=True):
        spk = {s: list(ds.index) for s, ds in dg.groupby("speaker_ref_id", sort=True)}
        queues[g] = spk
    gh = {g: 0.0 for g in queues}; sh: dict[str, float] = {}
    order = list(queues); rng.shuffle(order)
    spk_order = {g: list(queues[g]) for g in queues}
    for g in spk_order: rng.shuffle(spk_order[g])
    ptr = {g: 0 for g in queues}
    picked, total = [], 0.0
    dur = pool["duration_s"].to_numpy()
    spk_of = pool["speaker_ref_id"].to_numpy()
    capped_mode = True
    while total < target_s:
        progressed = False
        for g in order:
            if total >= target_s: break
            if capped_mode and gh[g] >= cap_s: continue
            sp = [s for s in spk_order[g] if queues[g][s]
                  and (not capped_mode or sh.get(s, 0.0) < cap_s)]
            if not sp: continue
            s = sp[ptr[g] % len(sp)]; ptr[g] += 1
            i = queues[g][s].pop()
            picked.append(i); d = float(dur[i]); total += d
            gh[g] += d; sh[s] = sh.get(s, 0.0) + d
            progressed = True
        if not progressed:
            if capped_mode:
                capped_mode = False          # pool cannot fill under the cap: water-fill
                continue
            break                            # pool exhausted
    return pool.loc[picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--ko", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=0.03)
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=60.0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    m = pd.read_parquet(V3 / "manifest.parquet")
    f = pd.read_parquet(V3 / "folds.parquet")[["file_id", "slice"]].rename(columns={"slice": "fslice"})
    d = m.merge(f, on="file_id", how="inner")
    d = d[(d.fslice == "train_val") & d.pool.isin(["A", "B"]) & (d.row_kind == "component")
          & d.duration_s.between(args.min_s, args.max_s)]
    assert not (d.fslice == "probe").any()
    langs = {"ko": args.ko, "en": 1.0 - args.ko}
    side_s = args.hours * 3600
    cap_s = args.cap * side_s
    out, rep = [], {"seed": args.seed, "hours_per_side": args.hours, "lang_share": langs,
                    "cap_share": args.cap, "sides": {}}
    for side, pool_id in (("real", "A"), ("fake", "B")):
        gcol = "speaker_ref_id" if side == "real" else "artifact_family"
        for lg, share in langs.items():
            p = d[(d.pool == pool_id) & (d.lang == lg)].copy()
            p["group"] = p[gcol].astype(str)
            p["speaker_ref_id"] = p["speaker_ref_id"].astype(str)
            s = rr_draw(p, "group", share * side_s, cap_s, rng)
            s["side"] = side
            out.append(s)
    dr = pd.concat(out, ignore_index=True)
    # family assignment: rotation within (side, lang, group), random start per group
    fam = np.empty(len(dr), dtype=object)
    for _, idx in dr.groupby(["side", "lang", "group"], sort=True).groups.items():
        idx = np.asarray(sorted(idx)); rng.shuffle(idx)
        k0 = int(rng.integers(len(FAMILIES)))
        for j, i in enumerate(idx):
            fam[i] = FAMILIES[(k0 + j) % len(FAMILIES)]
    dr["family"] = fam
    # per-file transform seed, stable in (seed, file_id)
    dr["seed"] = [int.from_bytes(np.random.default_rng([args.seed, abs(hash_id(x))]).bytes(4), "little")
                  for x in dr.file_id]
    dr["name"] = [safe_name(x) for x in dr.file_id]
    assert not dr["name"].duplicated().any()
    for side, ds in dr.groupby("side"):
        tot = ds.duration_s.sum()
        rep["sides"][side] = {
            "files": int(len(ds)), "hours": round(tot / 3600, 2),
            "hours_by_lang": (ds.groupby("lang").duration_s.sum() / 3600).round(2).to_dict(),
            "hours_by_family": (ds.groupby("family").duration_s.sum() / 3600).round(2).to_dict(),
            "hours_by_corpus": (ds.groupby("corpus").duration_s.sum() / 3600).round(2).to_dict(),
            "speakers": int(ds.speaker_ref_id.nunique()),
            "artifact_families": int(ds.artifact_family.nunique()),
            "max_speaker_share": round(float(ds.groupby("speaker_ref_id").duration_s.sum().max() / tot), 4),
            "max_speaker": str(ds.groupby("speaker_ref_id").duration_s.sum().idxmax()),
            "max_family_share": (round(float(ds.groupby("artifact_family").duration_s.sum().max() / tot), 4)
                                 if side == "fake" else None),
            "family_share_top": ((ds.groupby("artifact_family").duration_s.sum() / tot).round(4)
                                 .sort_values(ascending=False).head(8).to_dict() if side == "fake" else None),
        }
    rep["lang_by_family_side_h"] = {f"{k[0]}/{k[1]}/{k[2]}": round(v / 3600, 2) for k, v in
                                    dr.groupby(["family", "side", "lang"]).duration_s.sum().items()}
    od = ROOT / "interim" / "proc" / "_plan"; od.mkdir(parents=True, exist_ok=True)
    cols = ["file_id", "path", "side", "lang", "family", "seed", "name", "duration_s", "orig_sr",
            "corpus", "artifact_family", "speaker_ref_id", "group"]
    dr[cols].to_parquet(od / "draw.parquet", index=False)
    (od / "draw_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def hash_id(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") >> 1


def safe_name(fid: str) -> str:
    import hashlib, re
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", fid)[-80:]
    return f"{base}__{hashlib.sha1(fid.encode()).hexdigest()[:8]}"


if __name__ == "__main__":
    main()
