"""Benchmark one training step of the run-1 model on real rendered batches.

Two modes, so every variant is timed on the same bytes:

  --render N   draw pass 0 of fold 0 exactly as scripts/train.py does, render the
               first N batches (in the loop's own batch order) and save them to
               --batches. CPU only.
  (default)    load --batches, build the model, and time the loop's inner step
               (shipped chain, autocast forward + loss, backward, clip, AdamW,
               EMA) over them. Reports s/step, samples/s, audio-s/s and peak
               memory as one JSON line; --profile writes a torch.profiler table.

Variants are the flags below; each mirrors a config/loop knob, so a number here
is a number for that knob in scripts/train.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import threading
import time

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = "/data/project/private/dacon-corpus/manifests/strategy-v3"
WEIGHTS = ("audio=/data/project/private/dacon-weights/beats,"
           "speech=/data/project/private/dacon-weights/xlsr-300m")


def render_batches(args) -> None:
    import multiprocessing as mp

    from processing.config import load_processing_config
    from processing.render import ManifestIndex
    from processing.sampler import Sampler
    from models.config import load_model_config, load_train_config
    from training.collate import collate
    from training.dataset import SpecDataset
    from training.folds import apply_folds
    from training.loop import _worker_init, _worker_render
    from training.manifest import load_manifest
    from training.stages import pass_plan, stage_plan

    corpus = pathlib.Path(args.manifest_dir)
    manifest = load_manifest(corpus / "manifest.parquet")
    folds_tbl = pd.read_parquet(corpus / "folds.parquet")
    run_cfg = load_processing_config(args.processing)
    train_cfg = load_train_config(args.train)
    view = apply_folds(manifest, folds_tbl, 0)
    index = ManifestIndex.from_frame(view)
    ds = SpecDataset.from_sampler(Sampler(view, run_cfg.draw, slice_="train", fold=0),
                                  args.draws, index, run_cfg.render,
                                  seed=train_cfg.seed, ship=run_cfg.ship)
    ds.set_epoch(0)
    plan = stage_plan("joint", load_model_config(args.model))
    specs, batches = pass_plan(ds.specs, plan, batch_size=args.batch_size,
                               n_buckets=run_cfg.loop.n_buckets, seed=train_cfg.seed,
                               pass_index=0)
    batches = batches[:args.render]
    flat = [specs[i] for b in batches for i in b]
    with mp.get_context("spawn").Pool(args.workers, _worker_init,
                                      (ds.index, ds.cfg)) as pool:
        rendered = pool.map(_worker_render, flat, chunksize=1)
    out, k = [], 0
    for b in batches:
        out.append(collate(rendered[k:k + len(b)]))
        k += len(b)
    torch.save({"batches": out, "ship": ds.ship}, args.batches)
    secs = [float(b["lengths"].sum()) / 16000 for b in out]
    print(json.dumps({"rendered": len(out), "audio_s_per_batch": sum(secs) / len(secs),
                      "path": str(args.batches)}))


def build(args):
    from models.config import load_model_config
    from models.model import DeepVoiceNet
    from scripts.train import parse_weights

    cfg = load_model_config(args.model)
    by_name = parse_weights(args.weights, cfg.frontends)
    fes = {}
    for k, v in cfg.frontends.items():
        v = dataclasses.replace(v, weights=by_name.get(k, v.weights))
        if args.batched_tokens is not None and v.name == "beats":
            v = dataclasses.replace(v, batched_tokens=bool(args.batched_tokens))
        fes[k] = v
    cfg = dataclasses.replace(cfg, frontends=fes)
    torch.manual_seed(cfg.seed)
    return DeepVoiceNet(cfg)


def bench(args) -> None:
    from models.config import load_train_config
    from models.losses import multitask_loss
    from processing.ship import ship
    from training.checkpoint import EMA
    from training.loop import _param_groups, LoopConfig
    from training.stages import (_stage_loss_config, autocast_for, stage_plan,
                                 trainable_parameters)

    blob = torch.load(args.batches, weights_only=False)
    batches, ship_cfg = blob["batches"], blob["ship"]
    if args.merge > 1:
        # Bigger batches from the same samples. Adjacent saved batches can come
        # from different length buckets, so a merged batch carries more padding
        # than the loop's own would: a conservative throughput number.
        merged = []
        for i in range(0, len(batches) - args.merge + 1, args.merge):
            group = batches[i:i + args.merge]
            s = max(b["wav"].shape[-1] for b in group)
            wav = torch.cat([torch.nn.functional.pad(b["wav"], (0, s - b["wav"].shape[-1]))
                             for b in group])
            merged.append({"wav": wav,
                           "lengths": torch.cat([b["lengths"] for b in group]),
                           "targets": {k: torch.cat([b["targets"][k] for b in group])
                                       for k in group[0]["targets"]}})
        batches = merged

    device = torch.device("cuda")
    model = build(args).to(device)
    if args.compile:
        for fe in model.frontends.values():
            for layer in fe._transformer_layers() or []:
                layer.forward = torch.compile(layer.forward, dynamic=True)
    if args.grad_checkpointing:
        for fe in model.frontends.values():
            fe.enable_grad_checkpointing()
    train_cfg = load_train_config(args.train)
    plan = stage_plan("joint", model.cfg)
    group = plan.branch_groups[0]
    loop_cfg = LoopConfig(frontend_lr_scale=0.2)
    opt = torch.optim.AdamW(_param_groups(model, trainable_parameters(model, plan, group),
                                          train_cfg.lr, loop_cfg),
                            lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    ema = EMA(model, 0.999)
    loss_cfg = _stage_loss_config(model.cfg, group)
    model.train()

    def step(batch):
        wav = ship(batch["wav"].to(device), ship_cfg, batch["lengths"].to(device))
        lengths = batch["lengths"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}
        with autocast_for(train_cfg.precision, device):
            out = model(wav, lengths)
            total, parts = multitask_loss(out, targets, loss_cfg, train_cfg.loss,
                                          tensor_parts=bool(args.tensor_parts))
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                       5.0)
        opt.step()
        ema.update(model)
        return parts

    n = min(args.steps, len(batches))
    for b in batches[:args.warmup]:
        step(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timed = batches[args.warmup:args.warmup + n]
    util, stop = [], threading.Event()

    def sample_util():                  # NVML "GPU busy" share, as nvidia-smi reads it
        import subprocess
        while not stop.wait(0.5):
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits", "-i",
                                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                               capture_output=True, text=True)
            if q.returncode == 0 and q.stdout.strip().isdigit():
                util.append(int(q.stdout.strip()))

    sampler = threading.Thread(target=sample_util, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    losses = []
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for b in timed:
                losses.append(step(b)["total"])
            torch.cuda.synchronize()
    else:
        for b in timed:
            losses.append(step(b)["total"])
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    stop.set()
    samples = sum(int(b["lengths"].numel()) for b in timed)
    audio_s = sum(float(b["lengths"].sum()) for b in timed) / 16000
    res = {"label": args.label, "steps": len(timed), "s_per_step": dt / len(timed),
           "samples_per_s": samples / dt, "audio_s_per_s": audio_s / dt,
           "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "gpu_util_pct": sum(util) / max(1, len(util)),
           "batch": int(timed[0]["lengths"].numel()),
           "grad_checkpointing": args.grad_checkpointing, "compile": args.compile,
           "tensor_parts": args.tensor_parts, "batched_tokens": args.batched_tokens,
           "loss_first": float(losses[0]), "loss_mean": float(sum(losses)) / len(losses)}
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    if args.profile:
        tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        cpu_tbl = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
        pathlib.Path(args.profile).write_text(tbl + "\n\n" + cpu_tbl)
        prof.export_chrome_trace(str(args.profile) + ".trace.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", default=MANIFEST)
    p.add_argument("--processing", default=str(REPO / "configs/processing_first_run.yaml"))
    p.add_argument("--model", default=str(REPO / "configs/c_first_run.yaml"))
    p.add_argument("--train", default=str(REPO / "configs/train_first_run.yaml"))
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--batches", type=pathlib.Path,
                   default=pathlib.Path("/data/project/private/dacon-runs/_bench/batches.pt"))
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--merge", type=int, default=1,
                   help="concatenate this many saved batches into one step")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false")
    p.add_argument("--batched-tokens", type=int, choices=(0, 1), default=None)
    p.add_argument("--tensor-parts", type=int, choices=(0, 1), default=1,
                   help="1 = the loop's sync-free loss parts; 0 = run 1's float() per part")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile each transformer layer (dynamic shapes)")
    p.add_argument("--profile", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.render:
        render_batches(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
