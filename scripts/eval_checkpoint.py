#!/usr/bin/env python3
"""Score one training checkpoint's EMA (or raw) weights on its fold's VAL.

    python scripts/eval_checkpoint.py --ckpt runs/first-v3/fold1/joint-pass0.pt --fold 1 \
        --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v3 \
        --weights audio=/data/.../beats,speech=/data/.../xlsr-300m --eval-n 2000

The mid-run view of a fold run (docs/training/07 §6): the same frozen eval draw
`scripts/train.py` scores at the end (`frozen_eval_specs`, eval mode, same
seed), so a pass checkpoint and the final weights are comparable. Prints the
score and the per-head EERs and appends one JSON line to
<ckpt dir>/midrun_eval.jsonl.

Run it inside the training task's own allocation as ONE task:
    srun --jobid=<job>_<task> --overlap -n 1 -c 2 python scripts/eval_checkpoint.py ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from processing.config import load_processing_config  # noqa: E402
from processing.render import ManifestIndex  # noqa: E402
from processing.sampler import Sampler  # noqa: E402
from scripts.train import build_model  # noqa: E402
from training.checkpoint import EMA, load_train_checkpoint  # noqa: E402
from training.dataset import SpecDataset, frozen_eval_specs  # noqa: E402
from training.folds import apply_folds  # noqa: E402
from training.validate import evaluate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--processing", default="configs/processing_first_run.yaml")
    p.add_argument("--model", default="configs/c_first_run.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--raw", action="store_true", help="score the raw weights, not the EMA")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=None,
                   help="also write <out-dir>/fold<k>/val_specs.json + val_predictions.parquet "
                        "(the files scripts/diag/run_breakdown.py reads), e.g. to score a run-2 "
                        "model on run 1's exact VAL specs (docs/training/11 §4)")
    p.add_argument("--gpu-fraction", type=float, default=0.15,
                   help="cap on this process's share of the GPU: it runs BESIDE a training "
                        "task, and an eval that took the memory killed fold 1 of first-v3 "
                        "(cuFFT alloc failure in the trainer). Launch with srun -n 1.")
    args = p.parse_args()
    import torch
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)

    run_cfg = load_processing_config(args.processing)
    d = pathlib.Path(args.manifest_dir)
    view = apply_folds(pd.read_parquet(d / "manifest.parquet"),
                       pd.read_parquet(d / "folds.parquet"), args.fold)
    index = ManifestIndex.from_frame(view)

    model = build_model(args.model, args.weights)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "global_step" not in blob:
        # a finished model (scored.pt: config + the selected weights), as train.py writes
        model.load_state_dict(blob["state_dict"], strict=True)
        blob, which = {"global_step": -1, "ema": None}, "scored.pt"
    else:
        blob = load_train_checkpoint(args.ckpt)
        model.load_state_dict(blob["state_dict"], strict=True)
        which = "raw"
    if which != "scored.pt" and not args.raw and blob.get("ema") is not None:
        ema = EMA(model, run_cfg.loop.ema_decay or 0.999)
        ema.load_state_dict(blob["ema"])
        model.load_state_dict(ema.state_dict_for(model), strict=True)
        which = f"ema ({ema.steps} updates)"

    specs = frozen_eval_specs(Sampler(view, run_cfg.draw.for_eval(), slice_="val",
                                      fold=args.fold), args.eval_n, seed=args.eval_seed)
    ds = SpecDataset.frozen(specs, index, run_cfg.render, slice_="val", fold=args.fold,
                            ship=run_cfg.ship)
    t0 = time.time()
    report = evaluate(model, ds, batch_size=8, device=args.device, precision="fp32",
                      fold=args.fold)
    m = report.metrics
    row = {"ckpt": str(args.ckpt), "step": int(blob["global_step"]), "weights": which,
           "fold": args.fold, "eval_n": args.eval_n, "seconds": round(time.time() - t0),
           **{k: float(v) for k, v in dataclasses.asdict(m).items()
              if isinstance(v, (int, float))}}
    print(json.dumps(row, indent=1))
    if args.out_dir:
        od = pathlib.Path(args.out_dir) / f"fold{args.fold}"
        od.mkdir(parents=True, exist_ok=True)
        (od / "val_specs.json").write_text(json.dumps([s.to_dict() for s in specs]),
                                           encoding="utf-8")
        report.predictions.to_parquet(od / "val_predictions.parquet", index=False)
        (od / "eval_row.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        print(f"wrote {od}")
        return 0
    out = pathlib.Path(args.ckpt).parent / "midrun_eval.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
