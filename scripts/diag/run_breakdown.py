"""Slice a run's fold validation predictions by what was rendered into each clip.

For every fold of a run, joins ``val_predictions.parquet`` to ``val_specs.json`` (by
sample_id) and each spec's components to the manifest, then reports, per head, the
pooled EER threshold's error on each slice:

* real slices (by voice corpus / language / music corpus): false-alarm rate
* fake slices (by artifact family): miss rate

A slice's rate at the pooled threshold says *which side* carries the error, which a
per-slice EER (one side vs the whole other side) hides.

    $V scripts/diag/run_breakdown.py --run /data/project/private/dacon-runs/first-v3 \
        --manifest /data/project/private/dacon-corpus/manifests/strategy-v3/manifest.parquet \
        --out runs/first-v3/diag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.breakdown import t3_gap  # noqa: E402
from metrics.dacon import dacon_score, eer  # noqa: E402

HEADS = {
    "voice": ("voice_present", "voice_fake", "VOICE_FAKE_PROB"),
    "music": ("music_present", "music_fake", "MUSIC_FAKE_PROB"),
    "file": (None, "file_fake", "FILE_FAKE_PROB"),
}


def eer_threshold(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    return float(thr[np.argmin(np.abs(fpr - (1 - tpr)))])


def features(specs: list[dict], man: pd.DataFrame,
             cloned_in_train: frozenset = frozenset()) -> pd.DataFrame:
    """`cloned_in_train`: speakers whose clones are TRAIN (fake) in this fold. A VAL real
    clip of such a speaker measures "the voice I was taught as fake", not F1 (doc 11 M1)."""
    rows = []
    for s in specs:
        r = {"sample_id": s["sample_id"], "structure": s["structure"],
             "n_transforms": len(s["transforms"])}
        for role in ("voice", "music"):
            comps = [c for c in s["components"] if c["role"] == role
                     and not c.get("is_mixup_partner")]
            r[f"{role}_n"] = len(comps)
            r[f"{role}_secs"] = sum(c["duration_s"] for c in comps)
            snrs = [c["snr_db"] for c in comps if c.get("snr_db") is not None]
            r[f"{role}_snr"] = float(np.mean(snrs)) if snrs else np.nan
            if comps:
                m = man.loc[comps[0]["file_id"]] if comps[0]["file_id"] in man.index else None
                for k in ("corpus", "lang", "artifact_family", "source_name"):
                    r[f"{role}_{k}"] = None if m is None else m[k]
                if role == "voice":
                    r["voice_spk_cloned_in_train"] = bool(
                        m is not None and str(m["speaker_ref_id"]) in cloned_in_train)
        rows.append(r)
    return pd.DataFrame(rows)


def slice_rates(df: pd.DataFrame, head: str, keys: list[str], min_n: int) -> pd.DataFrame:
    present, fake, prob = HEADS[head]
    d = df if present is None else df[df[present] == 1]
    y, s = d[fake].to_numpy(), d[prob].to_numpy()
    thr = eer_threshold(y, s)
    out = []
    for key in keys:
        for side, lab in (("real", 0), ("fake", 1)):
            part = d[d[fake] == lab]
            for val, g in part.groupby(key, dropna=False):
                if len(g) < min_n:
                    continue
                err = (g[prob] >= thr).mean() if lab == 0 else (g[prob] < thr).mean()
                out.append({"head": head, "key": key, "value": str(val), "side": side,
                            "n": len(g), "err_at_pooled_thr": round(float(err), 4),
                            "mean_prob": round(float(g[prob].mean()), 4)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    man = pd.read_parquet(a.manifest).set_index("file_id")
    folds_path = Path(a.manifest).parent / "folds.parquet"
    fold_of = (pd.read_parquet(folds_path).set_index("file_id")["fold"]
               if folds_path.exists() else None)
    clones = (man[man["prompt_speaker"].notna()] if "prompt_speaker" in man.columns
              else man.iloc[:0])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, summary = [], {}
    for fold_dir in sorted(Path(a.run).glob("fold[0-9]")):
        preds = pd.read_parquet(fold_dir / "val_predictions.parquet")
        specs = json.loads((fold_dir / "val_specs.json").read_text())
        preds["sample_id"] = preds["file_id"].str[1:].astype(int)
        k = int(fold_dir.name[4:])
        cloned = frozenset()
        if fold_of is not None and len(clones):
            f = fold_of.reindex(clones.index)
            cloned = frozenset(clones.loc[(f.notna() & (f != k)).to_numpy(),
                                          "prompt_speaker"].astype(str))
        df = preds.merge(features(specs, man, cloned), on="sample_id", how="left",
                         validate="1:1")
        df["fold_name"] = fold_dir.name
        frames.append(df)
        ms = dacon_score(df).as_dict()
        ms["t3_voice"] = t3_gap(df, head="voice")
        ms["cell_file_eer"] = {int(c): round(eer(g["file_fake"], g["FILE_FAKE_PROB"]), 4)
                               for c, g in df.groupby("cell") if g["file_fake"].nunique() == 2}
        summary[fold_dir.name] = ms
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(out / "joined.parquet")
    tabs = []
    for fold, g in all_df.groupby("fold_name"):
        for head, keys in (("voice", ["voice_corpus", "voice_lang", "voice_artifact_family",
                                      "cell", "structure", "voice_spk_cloned_in_train"]),
                           ("music", ["music_corpus", "music_artifact_family", "cell"]),
                           ("file", ["cell", "voice_lang"])):
            t = slice_rates(g, head, keys, a.min_n)
            t.insert(0, "fold", fold)
            tabs.append(t)
    rates = pd.concat(tabs, ignore_index=True)
    rates.to_csv(out / "slice_rates.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: {m: v for m, v in s.items() if m in
                          ("score", "eer_file", "eer_voice", "eer_music")} for k, s in
                      summary.items()}, indent=1))
    worst = rates[rates["n"] >= a.min_n].sort_values("err_at_pooled_thr", ascending=False)
    print(worst.head(60).to_string(index=False))


if __name__ == "__main__":
    main()
