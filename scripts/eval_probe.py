#!/usr/bin/env python3
"""Score finished models, and their uniform soup, on the sealed PROBE slice.

    python scripts/eval_probe.py --manifest-dir .../strategy-v3 \
        --scored runs/first-v3/all_data_seed0/scored.pt ... (repeat) --soup \
        --weights audio=.../beats,speech=.../xlsr-300m --eval-n 3000

docs/training/07 §6 / 09: the four all-data runs share their initialisation, so the
shipping model is planned as their uniform weight soup. PROBE is the one slice no
run trained or validated on. Every model is scored on ONE frozen PROBE draw (eval
mode, fixed seed), so the rows are comparable. Caveat, measured on strategy-v3:
PROBE's voice side is mostly Chinese (6.5 h real, 59.4 h CFAD fakes), a language
the train views drop -- a harsh out-of-distribution read, good for ranking these
models against each other, not as an estimate of the leaderboard.

Writes <out>/probe_eval.json (one row per model, plus the soup).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from metrics.dacon import PREDICTION_COLUMNS, dacon_score  # noqa: E402
from models.model import load_checkpoint  # noqa: E402
from processing.config import load_processing_config  # noqa: E402
from processing.render import ManifestIndex  # noqa: E402
from processing.sampler import Sampler  # noqa: E402
from training.checkpoint import checkpoint_soup  # noqa: E402
from training.dataset import SpecDataset, frozen_eval_specs  # noqa: E402
from training.folds import apply_folds  # noqa: E402
from training.validate import evaluate  # noqa: E402


def _weights(s: str) -> dict[str, str]:
    return {k.strip(): v.strip() for k, _, v in (p.partition("=") for p in s.split(","))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--processing", default="configs/processing_first_run.yaml")
    p.add_argument("--scored", action="append", required=True)
    p.add_argument("--soup", action="store_true", help="also score the uniform soup")
    p.add_argument("--ensemble", action="store_true",
                   help="also score the per-file PREDICTION average of the single models "
                        "(prob mean and logit mean) -- per file, so legal under rule 2.4")
    p.add_argument("--weights", required=True)
    p.add_argument("--eval-n", type=int, default=3000)
    p.add_argument("--eval-seed", type=int, default=4321)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_processing_config(args.processing)
    d = pathlib.Path(args.manifest_dir)
    view = apply_folds(pd.read_parquet(d / "manifest.parquet"),
                       pd.read_parquet(d / "folds.parquet"), 0)
    index = ManifestIndex.from_frame(view)
    specs = frozen_eval_specs(Sampler(view, cfg.draw.for_eval(), slice_="probe", fold=None),
                              args.eval_n, seed=args.eval_seed)
    ds = SpecDataset.frozen(specs, index, cfg.render, slice_="probe", fold=None,
                            ship=cfg.ship)
    w = _weights(args.weights)

    rows = []
    preds = {}
    candidates = [(str(s), [s]) for s in args.scored]
    if args.soup and len(args.scored) > 1:
        candidates.append((f"soup of {len(args.scored)}", args.scored))
    for name, paths in candidates:
        model = load_checkpoint(paths[0], weights=w)
        if len(paths) > 1:
            model.load_state_dict(checkpoint_soup(paths), strict=True)
        report = evaluate(model, ds, batch_size=8, device=args.device, precision="fp32")
        m = report.metrics
        if len(paths) == 1:
            preds[name] = report.predictions
        row = {"model": name, "eval_n": args.eval_n,
               **{k: float(v) for k, v in dataclasses.asdict(m).items()
                  if isinstance(v, (int, float))}}
        rows.append(row)
        print(f"{name:<70} score {row['score']:.4f}  file {row['eer_file']:.4f}  "
              f"voice {row['eer_voice']:.4f}  music {row['eer_music']:.4f}", flush=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, df) in enumerate(preds.items()):
        df.assign(model=name).to_parquet(out / f"probe_pred_{i}.parquet", index=False)
    if args.ensemble and len(preds) > 1:
        frames = list(preds.values())
        base = frames[0].copy()
        for how in ("prob_mean", "logit_mean"):
            ens = base.copy()
            for c in PREDICTION_COLUMNS:
                stack = np.stack([f.set_index("file_id").loc[base["file_id"], c].to_numpy()
                                  for f in frames])
                if how == "prob_mean":
                    ens[c] = stack.mean(0)
                else:
                    z = np.log(np.clip(stack, 1e-12, 1 - 1e-12) / np.clip(1 - stack, 1e-12, 1))
                    ens[c] = 1 / (1 + np.exp(-z.mean(0)))
            m = dacon_score(ens)
            row = {"model": f"ensemble {how} of {len(frames)}", "eval_n": args.eval_n,
                   **{k: float(v) for k, v in dataclasses.asdict(m).items()
                      if isinstance(v, (int, float))}}
            rows.append(row)
            print(f"{row['model']:<70} score {row['score']:.4f}  file {row['eer_file']:.4f}  "
                  f"voice {row['eer_voice']:.4f}  music {row['eer_music']:.4f}", flush=True)
    (out / "probe_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
