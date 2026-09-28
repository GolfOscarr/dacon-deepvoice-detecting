#!/usr/bin/env python3
"""Music-slot mix of a draw (draw_balance.py's view, music role): corpus shares of the
real (pool C) and fake (pool D) music components drawn, and music repetition per label.

    $V scripts/strategy/music_balance.py --manifest-dir <dir> --processing <cfg> --all-data --n 8000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from processing.audit import draw_features                  # noqa: E402
from processing.config import load_processing_config        # noqa: E402
from processing.sampler import Sampler                      # noqa: E402
from training.folds import apply_folds                      # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--processing", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-data", action="store_true")
    p.add_argument("--n", type=int, default=8000)
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
    specs = list(Sampler(view, cfg, slice_="train", fold=fold).epoch_specs(a.n, seed=0))
    feats = draw_features(specs, view)
    by = view.set_index("file_id")
    rows = []
    for sp, uf in zip(specs, feats["music_unique_fraction"]):
        for c in sp.components:
            if c.role == "music" and not c.is_mixup_partner:
                r = by.loc[c.file_id]
                rows.append({"label": "fake" if r["pool"] == "D" else "real", "corpus": r["corpus"], "uf": uf})
    t = pd.DataFrame(rows)
    out = {}
    for lab, g in t.groupby("label"):
        out[lab] = {"tiles": len(g), "repeats(uf<0.99)": round(float((g.uf < 0.99).mean()), 3),
                    "corpus": g.corpus.value_counts(normalize=True).round(3).to_dict()}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
