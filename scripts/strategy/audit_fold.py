#!/usr/bin/env python3
"""The processing audit over one fold view, plus the drawn language shares.

    python scripts/strategy/audit_fold.py \
        --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v3 \
        --processing configs/processing_first_run.yaml --fold 0 --n 20000

`processing.audit.run_audit` over `apply_folds(manifest, folds, k)`, slice
train (or `--all-data`: TRAIN + VAL of every fold, the shipping runs' view).
Then docs/training/07 D-d measured rather than assumed: the share of each
language among the voice components drawn from the real and the fake pool.
Exit 1 when the audit fails.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from processing.audit import run_audit  # noqa: E402
from processing.config import load_processing_config  # noqa: E402
from processing.sampler import Sampler  # noqa: E402
from training.folds import apply_folds  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--processing", default="configs/processing_first_run.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-data", action="store_true")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = pathlib.Path(args.manifest_dir)
    m, f = pd.read_parquet(d / "manifest.parquet"), pd.read_parquet(d / "folds.parquet")
    cfg = load_processing_config(args.processing)
    view = apply_folds(m, f, args.fold).copy()
    fold: int | None = args.fold
    if args.all_data:
        view.loc[view["slice"] == "val", "slice"] = "train"
        fold = None
    sampler = Sampler(view, cfg.draw, slice_="train", fold=fold)
    report = run_audit(sampler, view, n=args.n, seed=args.seed)
    for name, (ok, why) in sorted(report.results.items()):
        print(f"{'PASS' if ok else 'FAIL'}  {name:<6} {why}")
    print(f"audit ok: {report.ok}")

    if "lang" in view.columns:
        by_id = view.set_index("file_id")
        counts = {"A": Counter(), "B": Counter()}
        proc = {"A": Counter(), "B": Counter()}
        for spec in sampler.epoch_specs(min(args.n, 4000), seed=args.seed + 1):
            for c in spec.components:
                pool = by_id.at[c.file_id, "pool"]
                if c.role == "voice" and c.snr_db is None and pool in counts:
                    counts[pool][by_id.at[c.file_id, "lang"] or "?"] += 1
                    proc[pool][str(by_id.at[c.file_id, "corpus"]) == "proc"] += 1
        for pool, cnt in counts.items():
            tot = sum(cnt.values()) or 1
            shares = ", ".join(f"{k} {v / tot:.3f}" for k, v in cnt.most_common(8))
            print(f"drawn voice lang, pool {pool} ({'fake' if pool == 'B' else 'real'}): {shares}")
        # docs/training/09 S1: processing must not predict the label
        for pool, cnt in proc.items():
            tot = sum(cnt.values()) or 1
            print(f"drawn voice share processed (corpus proc), pool {pool}: "
                  f"{cnt[True] / tot:.3f}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
