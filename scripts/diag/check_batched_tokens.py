"""A trained checkpoint scored with `batched_tokens` off (run 1's path) and on.

Loads a scored.pt, runs saved bench batches (bench_train_step.py --render)
through the shipped chain, and reports the largest per-column difference in the
submitted probabilities, in fp32 and under fp16 autocast (the inference
precision). Run on the GPU the model trains on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from models.model import load_checkpoint                                # noqa: E402
from processing.ship import ship                                        # noqa: E402

WEIGHTS = {"audio": "/data/project/private/dacon-weights/beats",
           "speech": "/data/project/private/dacon-weights/xlsr-300m"}


def set_batched(model, on: bool) -> None:
    for fe in model.frontends.values():
        if fe.cfg.name == "beats":
            fe.cfg = dataclasses.replace(fe.cfg, batched_tokens=on)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", default="/data/project/private/dacon-runs/first-v3/"
                                       "all_data_seed0/scored.pt")
    p.add_argument("--batches", default="/data/project/private/dacon-runs/_bench/batches.pt")
    p.add_argument("--n-batches", type=int, default=3)
    args = p.parse_args()
    dev = torch.device("cuda")
    model = load_checkpoint(args.scored, weights=WEIGHTS).to(dev).eval()
    blob = torch.load(args.batches, weights_only=False)
    batches, ship_cfg = blob["batches"][:args.n_batches], blob["ship"]
    report, loop_probs = {}, {}
    for precision in ("fp32", "fp16"):
        diffs: dict[str, float] = {}
        for batch in batches:
            wav = ship(batch["wav"].to(dev), ship_cfg, batch["lengths"].to(dev))
            lengths = batch["lengths"].to(dev)
            got = {}
            for on in (False, True):
                set_batched(model, on)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=precision == "fp16"):
                    got[on] = model.submission_probs(model(wav, lengths))
            loop_probs.setdefault(precision, []).append(got[False])
            for col in got[False]:
                d = (got[False][col] - got[True][col]).abs().max().item()
                diffs[col] = max(diffs.get(col, 0.0), d)
        report[precision] = diffs
    # The scale to read those against: what fp16 inference itself moves.
    report["fp16_vs_fp32_loop_path"] = {
        col: max((a[col] - b[col]).abs().max().item()
                 for a, b in zip(loop_probs["fp32"], loop_probs["fp16"]))
        for col in loop_probs["fp32"][0]}
    print(json.dumps({"scored": args.scored, "clips": sum(len(b["lengths"]) for b in batches),
                      "max_abs_prob_diff": report}, indent=1))


if __name__ == "__main__":
    main()
