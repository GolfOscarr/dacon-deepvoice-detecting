"""Label-symmetry of a draw, for tuning `draw.domain_weights` (docs/training/11 §1.1).

Per voice label (real / fake) over drawn specs: the language shares, the processed (S1)
share, the share of voice slots that repeat source audio (voice_unique_fraction < 0.99:
tiny speaker buckets such as Emilia's and the round-2 clones'), and the corpus mix.
Every one of these must match across the labels, or it is a label shortcut (the audit's
I1c reads the repetition). Faster than audit_fold.py; run that for the gates.

    $V scripts/strategy/draw_balance.py --manifest-dir DIR --processing CFG --fold 1 --n 8000 \
        [--weights '{"emilia-ko/": 0.5}']      # merged over the config's domain_weights
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from processing.audit import draw_features  # noqa: E402
from processing.config import load_processing_config  # noqa: E402
from processing.sampler import Sampler  # noqa: E402
from training.folds import apply_folds  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--processing", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-data", action="store_true")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", default=None, help="JSON prefix -> multiplier, merged over config")
    a = p.parse_args()
    d = pathlib.Path(a.manifest_dir)
    m, f = pd.read_parquet(d / "manifest.parquet"), pd.read_parquet(d / "folds.parquet")
    cfg = load_processing_config(a.processing).draw
    if a.weights:
        merged = dict(cfg.domain_weights or ())
        merged.update(json.loads(a.weights))
        cfg = dataclasses.replace(cfg, domain_weights=tuple(sorted(merged.items())))
    view = apply_folds(m, f, a.fold).copy()
    fold = a.fold
    if a.all_data:
        view.loc[view["slice"] == "val", "slice"] = "train"
        fold = None
    specs = list(Sampler(view, cfg, slice_="train", fold=fold).epoch_specs(a.n, seed=a.seed))
    feats = draw_features(specs, view)
    by_id = view.set_index("file_id")
    rows = []
    for sp, uf in zip(specs, feats["voice_unique_fraction"]):
        v = [c for c in sp.components if c.role == "voice" and not c.is_mixup_partner]
        if not v:
            continue
        r = by_id.loc[v[0].file_id]
        rows.append({"label": "fake" if r["pool"] == "B" else "real", "corpus": r["corpus"],
                     "lang": r["lang"], "repeats": float(uf < 0.99),
                     "processed": float(r["corpus"] == "proc")})
    t = pd.DataFrame(rows)
    out = {"n_voice_specs": len(t)}
    for lab, g in t.groupby("label"):
        out[lab] = {"n": len(g), "repeats": round(g["repeats"].mean(), 3),
                    "processed": round(g["processed"].mean(), 3),
                    "lang": g["lang"].value_counts(normalize=True).round(3).to_dict(),
                    "corpus": g["corpus"].value_counts(normalize=True).round(3).to_dict()}
    wf = t[(t.label == "fake") & (t.lang == "en")]
    out["wavefake_share_of_en_fake"] = round(float((wf.corpus == "wavefake").mean()), 3)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
