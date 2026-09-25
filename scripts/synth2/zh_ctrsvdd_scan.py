"""S5 (docs/training/09 §1): measure what the local CtrSVDD 2024 copy holds, and write its
ingest metadata. The audio is not transformed (CC BY-NC-ND: train from the originals + code).

Input : raw/ctrsvdd/2024/payload/{train,dev}.txt (protocol: singer-set, speaker, utt id, -,
        -|A01..A08, bonafide|deepfake) and interim/ctrsvdd/2024/{train,dev,test}_set/*.flac
        (unpacked by scripts/fetch_from_s3.py --extract).
Output: interim/ctrsvdd/metadata.csv  file (corpus-relative), label, attack, singer_set,
        speaker, duration_s, sample_rate, lang, kept, split
        interim/ctrsvdd/_scan/summary.json (counts/hours per label, attack, singer set)

``kept``: the file exists, reads and lasts >= --min-s seconds, AND its singer set has bona fide
audio on disk (plan 09 §3.5: fakes alone would teach "singing => fake"; measured 2026-09-25:
the Japanese sets' bona fide takes are absent, rebuilt by the release's scripts from their own
corpora, so ja fakes are not kept). Bona fide rows are all kept; deepfake rows are kept up to
--fake-cap-h hours (plan 09 §3.1: "CtrSVDD fakes, capped at 40 h"), drawn with a fixed seed and
balanced over the attacks.
The test_set has no protocol here (labels unknown), so it is counted but not listed.

  /data/project/private/dacon-venvs/dacon311/bin/python scripts/synth2/zh_ctrsvdd_scan.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

CORPUS = Path("/data/project/private/dacon-corpus")
RAW = CORPUS / "raw/ctrsvdd/2024/payload"
AUD = CORPUS / "interim/ctrsvdd/2024"
OUT = CORPUS / "interim/ctrsvdd"
LANG = {"m4singer": "zh", "opencpop": "zh", "kiritan": "ja", "ofuton": "ja", "oniku": "ja",
        "jvsmusic": "ja", "jvs_music": "ja"}


def info(path: str):
    import soundfile as sf
    try:
        i = sf.info(path)
        return i.frames / i.samplerate, i.samplerate, i.channels
    except Exception:
        return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-cap-h", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-s", type=float, default=1.0)
    a = ap.parse_args()

    prot = []
    for split in ("train", "dev"):
        d = pd.read_csv(RAW / f"{split}.txt", sep=r"\s+", header=None,
                        names=["singer_set", "speaker", "utt", "x", "attack", "label"])
        d["split"] = split
        prot.append(d)
    p = pd.concat(prot, ignore_index=True)
    p["attack"] = p["attack"].where(p["attack"] != "-", "")
    p["abs"] = [str(AUD / f"{s}_set" / f"{u}.flac") for s, u in zip(p["split"], p["utt"])]
    p["exists"] = [Path(x).exists() for x in p["abs"]]
    on_disk = {s: {f.stem for f in (AUD / f"{s}_set").glob("*.flac")} for s in ("train", "dev", "test")}
    listed = {s: set(p.loc[p.split == s, "utt"]) for s in ("train", "dev")}

    ex = p[p.exists]
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(info, ex["abs"], chunksize=256))
    p.loc[ex.index, "duration_s"] = [r[0] for r in res]
    p.loc[ex.index, "sample_rate"] = [r[1] for r in res]
    p.loc[ex.index, "channels"] = [r[2] for r in res]
    p["readable"] = p["duration_s"].notna()
    p["lang"] = p["singer_set"].map(LANG).fillna("unknown")

    ok = p[p.readable]
    real_sets = set(ok.loc[ok.label == "bonafide", "singer_set"])
    elig = ok[ok.singer_set.isin(real_sets) & (ok.duration_s >= a.min_s)]
    kept = pd.Series(False, index=p.index)
    if real_sets:
        kept[elig.index[elig.label == "bonafide"]] = True
        fakes = elig[elig.label == "deepfake"]
        rng = random.Random(20260925)
        by_atk = {k: rng.sample(list(g.index), len(g)) for k, g in fakes.groupby("attack")}
        budget, used, ptr = a.fake_cap_h * 3600, 0.0, defaultdict(int)
        while used < budget and any(ptr[k] < len(v) for k, v in by_atk.items()):
            for k, v in sorted(by_atk.items()):      # round-robin: balanced over attacks
                if ptr[k] < len(v) and used < budget:
                    i = v[ptr[k]]
                    ptr[k] += 1
                    kept[i] = True
                    used += float(p.at[i, "duration_s"])
    p["kept"] = kept

    meta = pd.DataFrame({
        "file": [str(Path(x).relative_to(CORPUS)) for x in p["abs"]],
        "label": p["label"], "attack": p["attack"], "singer_set": p["singer_set"],
        "speaker": p["speaker"], "duration_s": p["duration_s"].round(3),
        "sample_rate": p["sample_rate"].astype("Int64"), "lang": p["lang"],
        "kept": p["kept"], "split": p["split"]})[p.exists.to_numpy()]
    meta.to_csv(OUT / "metadata.csv", index=False)

    def agg(d, keys):
        g = d.groupby(keys, dropna=False)
        return {" / ".join(map(str, k)) if isinstance(k, tuple) else str(k):
                {"files": int(len(x)), "hours": round(float(x.duration_s.sum()) / 3600, 2)}
                for k, x in g}
    summary = {
        "protocol_rows": {s: int((p.split == s).sum()) for s in ("train", "dev")},
        "flac_on_disk": {s: len(v) for s, v in on_disk.items()},
        "listed_and_present": {s: int(((p.split == s) & p.exists).sum()) for s in ("train", "dev")},
        "listed_missing": {s: int(((p.split == s) & ~p.exists).sum()) for s in ("train", "dev")},
        "missing_by_label": agg(p[~p.exists].assign(duration_s=0.0), ["split", "label"]),
        "on_disk_not_listed": {s: len(on_disk[s] - listed[s]) for s in ("train", "dev")},
        "unreadable": int((p.exists & ~p.readable).sum()),
        "by_label": agg(ok, ["label"]),
        "by_split_label": agg(ok, ["split", "label"]),
        "by_attack": agg(ok, ["attack"]),
        "by_singer_set_label": agg(ok, ["singer_set", "label"]),
        "by_lang_label": agg(ok, ["lang", "label"]),
        "speakers": {"bonafide": int(ok[ok.label == "bonafide"].speaker.nunique()),
                     "deepfake": int(ok[ok.label == "deepfake"].speaker.nunique())},
        "sample_rates": {str(k): int(v) for k, v in ok.sample_rate.value_counts().items()},
        "channels": {str(k): int(v) for k, v in ok.channels.value_counts().items()},
        "duration_s": {q: round(float(ok.duration_s.quantile(v)), 2)
                       for q, v in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1))},
        "kept": agg(p[p.kept], ["label"]),
        "kept_by_attack": agg(p[p.kept], ["attack"]),
        "fake_cap_h": a.fake_cap_h,
        "min_s": a.min_s,
        "singer_sets_with_bonafide": sorted(real_sets),
        "short_dropped": agg(ok[ok.singer_set.isin(real_sets) & (ok.duration_s < a.min_s)], ["label"]),
        "kept_speakers": {"bonafide": int(p[p.kept & (p.label == "bonafide")].speaker.nunique()),
                          "deepfake": int(p[p.kept & (p.label == "deepfake")].speaker.nunique()),
                          "shared": len(set(p[p.kept & (p.label == "bonafide")].speaker)
                                        & set(p[p.kept & (p.label == "deepfake")].speaker))},
        "kept_by_lang_label": agg(p[p.kept], ["lang", "label"]),
    }
    (OUT / "_scan").mkdir(exist_ok=True)
    (OUT / "_scan" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
