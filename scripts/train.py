#!/usr/bin/env python3
"""Train one or more folds and emit a ledger row. The entrypoint.

    python3 scripts/train.py --corpus /data/corpus/test-v1 --out runs/t1 \
        --model configs/a_shared_trunk.yaml --weights /data/weights/beats
    python3 scripts/train.py --corpus /data/corpus/test-v1 --out runs/t1 \
        --folds 0,1 --select ema
    python3 scripts/train.py --corpus /data/corpus/test-v1 --out runs/smoke \
        --stages joint --max-steps 20 --eval-n 600      # Replay speed, NOT QUOTABLE
    python3 scripts/train.py --corpus /data/corpus/test-v1 --out runs/t1 --dry-run

`training/` is a library and `run_schedule()` had no caller, so every run before
this one was driven from a script outside the repo -- which made the thing this
project measures most carefully the one thing it could not reproduce.

What this adds over calling `train_stage` by hand, in order of how easily each is
got wrong:

*The caveat union.* `aggregate_folds`' own docstring says caveats "do not travel
on their own": every stage carries its own, a caller that reports only the last
result silently drops S1's, and one caveat containing `NOT_QUOTABLE` must void
the run. This collects them across every stage of every fold and passes the union
in, so `--max-steps` reaches the `quotable` column rather than only the prose.

*The weight-selection step, which did not exist.* `EMA.state_dict_for()` and
`checkpoint_soup()` were implemented, unit-tested, and called from nowhere but
`tests/`. `LoopConfig.ema_decay` defaults to 0.999, so every run so far
maintained an EMA every step and then discarded it, and every number came from
raw weights while docs/architecture/05 calls the soup "free, do it by default".
`--select` is that choice, made explicitly and recorded in the ledger row,
because two runs that scored different weights are two runs.

*Gates and tripwires per fold, not per run.* `FoldResult.ok` is
`gates.ok and tripwires.ok`, and a fold that fails either still contributes its
metrics to the mean -- so the row has to say which folds were clean.

⚠️ One GPU per invocation by default. Folds are independent, so N folds is a job
array rather than a distributed run.

*Data parallel* (docs/training/14): under `torchrun --nproc_per_node N` the SAME
invocation trains one fold or `--all-data` on N GPUs. Every rank builds the same
draw and pass plan at the global batch (`--batch-size` per rank x N x
`loop.grad_accum`) and trains its slice, so the bitwise-resume guarantee carries
over. After training every rank meets at a barrier, ranks > 0 exit 0, and rank 0
leaves the process group and selects, saves, evaluates and writes the ledger
exactly as a one-GPU run does. Only rank 0's exit code carries `quotable`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.config import load_model_config, load_train_config        # noqa: E402
from models.model import DeepVoiceNet, save_checkpoint                  # noqa: E402
from processing.config import dump_processing_config, load_processing_config  # noqa: E402
from processing.render import ManifestIndex                           # noqa: E402
from processing.sampler import Sampler                                # noqa: E402
from training.checkpoint import checkpoint_soup                       # noqa: E402
from training.dataset import SpecDataset, frozen_eval_specs           # noqa: E402
from training.distributed import (barrier, current, init_from_env,     # noqa: E402
                                  teardown)
from training.folds import apply_folds                                # noqa: E402
from training.loop import check_chain, run_schedule                   # noqa: E402
from training.manifest import load_manifest                           # noqa: E402
from training.stages import STAGES                                    # noqa: E402
from training.validate import (FoldResult, aggregate_folds, evaluate,  # noqa: E402
                               leak_tripwires, measured_split_kind, run_gates)

SELECTIONS = ("raw", "ema", "soup")


def parse_weights(weights: str | None, frontends) -> dict[str, str]:
    """`--weights` -> {frontend name: checkpoint dir}.

    Two forms. A bare path applies to every frontend -- the single-trunk form
    every earlier run used. `name=DIR[,name=DIR...]` names each frontend, which a
    two-trunk config needs because BEATs and XLS-R read different checkpoints.

    🔴 A bare path on a multi-frontend config is refused rather than broadcast:
    handing the BEATs directory to XLS-R fails deep inside `transformers` with a
    missing-config.json error that does not name the flag. A name the config does
    not have is refused too; a frontend the mapping omits keeps its config value.
    """
    if weights is None:
        return {}
    names = list(frontends)
    if "=" not in weights:
        if len(names) > 1:
            raise SystemExit(
                f"--weights {weights!r} is one path but the model has {len(names)} "
                f"frontends {names}; pass --weights "
                + ",".join(f"{n}=DIR" for n in names))
        return {n: weights for n in names}
    out = {}
    for part in weights.split(","):
        name, sep, path = part.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            raise SystemExit(f"--weights: cannot parse {part!r}; expected name=DIR")
        if name not in frontends:
            raise SystemExit(f"--weights names frontend {name!r}; the model has {names}")
        if name in out:
            raise SystemExit(f"--weights names frontend {name!r} twice")
        out[name] = path
    return out


#: Parameters a run-2 checkpoint may lack: new heads' state that starts fresh.
#: `.oc_center` is the O1 one-class centre (docs/training/13 O1).
INIT_MAY_MISS = (".oc_center",)


def init_from(model: DeepVoiceNet, path: str) -> None:
    """Load a finished model's weights (scored.pt: config + state_dict) strictly.
    The architecture must match; the optimizer and the draw start fresh.

    The one exception is a missing key ending in `INIT_MAY_MISS`, which keeps
    its fresh init and is logged. Any other missing key, and every unexpected
    key, still fails."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(blob["state_dict"], strict=False)
    fresh = [k for k in res.missing_keys if k.endswith(INIT_MAY_MISS)]
    missing = [k for k in res.missing_keys if k not in fresh]
    if missing or res.unexpected_keys:
        raise RuntimeError(
            f"init_from {path}: state dict does not match the model -- missing "
            f"{missing}, unexpected {res.unexpected_keys}")
    if fresh:
        print(f"  left at their fresh init (not in {path}): {fresh}", flush=True)
    print(f"  initialised from {path}", flush=True)


def init_partial(model: DeepVoiceNet, path: str) -> dict[str, list[str]]:
    """docs/training/13 M3 / 14 §5.1: initialise what fits from a scored.pt of a
    DIFFERENT architecture (run-2 T7, 300M@12, into 1B@24 + fusion).

    A key loads when its name AND shape match. On top of that:

    * a head (``heads.<name>``) and a frontend (``frontends.<name>``) load
      all-or-nothing: one missing or mismatched key leaves the WHOLE module at
      its fresh init -- a head whose first layer is fresh and whose later layers
      are T7's is neither model. 🔴 For a frontend this is not tidiness: the
      300M and 1B CNN feature extractors have the SAME shapes (512 channels), so
      a key-by-key load would put 300M's CNN under 1B's transformer;
    * everything else that does not fit (the fusion) stays fresh, and is reported;
    * nothing of BEATs (a ``beats`` frontend) loading is refused: then this is
      not a partial init of the T7 model at all.

    Run on every rank before the DDP wrap, so every rank holds the same result.
    Returns {"loaded": [...], "fresh": [...], "unused": [...]} (keys).
    """
    from training.distributed import current as current_dist

    src = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    own = model.state_dict()
    fits = {k for k, v in own.items() if k in src and tuple(src[k].shape) == tuple(v.shape)}
    for prefix in ([f"heads.{n}." for n in model.heads]
                   + [f"frontends.{n}." for n in model.frontends]):
        keys = [k for k in own if k.startswith(prefix)]
        if not all(k in fits for k in keys):
            fits -= set(keys)
    beats = tuple(f"frontends.{n}." for n, fe in model.cfg.frontends.items()
                  if fe.name == "beats")
    if not beats or not any(k.startswith(beats) for k in fits):
        raise RuntimeError(f"init_partial {path}: no BEATs frontend key fits the model; "
                           f"refusing a partial init that loads nothing of it")
    res = model.load_state_dict({k: src[k] for k in fits}, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    report = {"loaded": sorted(fits), "fresh": sorted(set(own) - fits),
              "unused": sorted(set(src) - fits)}
    if current_dist().main:
        def modules(keys):
            out: dict[str, int] = {}
            for k in keys:
                parts = k.split(".")
                m = ".".join(parts[:2]) if parts[0] in ("heads", "frontends", "fusion") \
                    else parts[0]
                out[m] = out.get(m, 0) + 1
            return ", ".join(f"{m} ({n})" for m, n in sorted(out.items())) or "-"
        print(f"  partial init from {path}", flush=True)
        print(f"    loaded: {modules(report['loaded'])}", flush=True)
        print(f"    fresh:  {modules(report['fresh'])}", flush=True)
        print(f"    unused from the checkpoint: {modules(report['unused'])}", flush=True)
        lora = [k for k in report["fresh"] if ".lora_" in k]
        if lora:
            print(f"    fresh LoRA tensors: {len(lora)}", flush=True)
    return report


def build_model(model_cfg_path: str, weights: str | None) -> DeepVoiceNet:
    cfg = load_model_config(model_cfg_path)
    by_name = parse_weights(weights, cfg.frontends)
    if by_name:
        cfg = dataclasses.replace(cfg, frontends={
            k: (dataclasses.replace(v, weights=by_name[k]) if k in by_name else v)
            for k, v in cfg.frontends.items()})
    torch.manual_seed(cfg.seed)
    return DeepVoiceNet(cfg)


def select_weights(model: DeepVoiceNet, results, how: str) -> str:
    """Apply the chosen weights in place and return what to record in the ledger.

    🔴 `raw` is the last optimizer step's weights, which is what every run before
    this entrypoint scored -- not because it was chosen, but because nothing
    applied the alternatives.
    """
    if how == "raw":
        return "raw (final optimizer step)"
    if how == "ema":
        ema = next((r.ema for r in reversed(results) if r.ema is not None), None)
        if ema is None:
            raise SystemExit(
                "--select ema but no stage carried an EMA: LoopConfig.ema_decay is 0, "
                "which disables it entirely. Set a decay or select raw.")
        if ema.steps == 0:
            raise SystemExit(
                "--select ema but the EMA took no update (a zero-step run). "
                "Scoring an unmaterialised EMA would score the initialisation.")
        model.load_state_dict(ema.state_dict_for(model), strict=True)
        return f"ema (decay {ema.decay}, {ema.steps} updates, bias-corrected)"
    # 🔴 The LAST stage's checkpoints, not every stage's.
    # docs/architecture/05 §3: weight averaging only works between checkpoints in
    # the same loss basin, and is "for epochs and same-init runs". S1 trains one
    # branch at a time behind a frozen frontend, so its checkpoints sit far closer
    # to the initialisation than S3's; pooling all three stages averages the final
    # weights back toward an earlier regime. `--soup-all-stages` is there for
    # someone who wants to measure that rather than inherit it.
    stages_used = results if how == "soup-all" else results[-1:]
    paths = [c.path for r in stages_used for c in r.checkpoints]
    if len(paths) < 2:
        raise SystemExit(
            f"--select soup needs at least two checkpoints from stage "
            f"{stages_used[-1].stage!r}, found {len(paths)}. Set "
            f"LoopConfig.checkpoint_every, raise --epochs so more passes "
            f"checkpoint, or pass --soup-all-stages.")
    model.load_state_dict(checkpoint_soup(paths), strict=True)
    scope = "all stages" if how == "soup-all" else f"stage {stages_used[-1].stage!r}"
    return f"soup ({len(paths)} checkpoints from {scope}, uniform average)"


def _leave_group() -> bool:
    """After training: meet, leave the process group, and say whether this
    process carries on (rank 0, or a one-process run).

    Every rank leaves -- rank 0 too -- so rank 0's evaluation, which can take far
    longer than any collective timeout, runs as a plain single process with
    nobody waiting on it (docs/training/14 D4).
    """
    d = current()
    if not d.enabled:
        return True
    barrier(d)
    teardown(d)
    return d.main


def train_fold(fold: int, *, manifest, folds_tbl, args, run_cfg, train_cfg):
    # 06 P8: every sampler is built on the fold VIEW (05 A7); the raw manifest
    # carries no fold and `processing.sampler.Sampler` refuses it.
    view = apply_folds(manifest, folds_tbl, fold)
    index = ManifestIndex.from_frame(view)
    rcfg = run_cfg.render
    if args.corpus_root is not None:
        rcfg = dataclasses.replace(rcfg, root=pathlib.Path(args.corpus_root))
    out = pathlib.Path(args.out) / f"fold{fold}"

    n_train = (view["slice"] == "train").sum()
    n_val = (view["slice"] == "val").sum()
    print(f"\n=== fold {fold}: {n_train} train rows, {n_val} val rows ===", flush=True)

    train_sampler = Sampler(view, run_cfg.draw, slice_="train", fold=fold)
    ds = SpecDataset.from_sampler(train_sampler, args.draws, index, rcfg,
                                  seed=train_cfg.seed, ship=run_cfg.ship)
    # 🔴 Both overrides guard for None. `max_steps` did not, so invoking without
    # `--max-steps` overwrote a run config's `loop.max_steps: 200` with None: the
    # stage stopped being truncated, `StageResult.truncated` stayed False, the
    # NOT_QUOTABLE caveat never fired, and `quotable` came out True for a run the
    # config had asked to truncate. The line directly below always had the guard,
    # which is what made the omission easy to miss.
    loop_cfg = dataclasses.replace(
        run_cfg.loop, out_dir=out, device=args.device,
        max_steps=(args.max_steps if args.max_steps is not None
                   else run_cfg.loop.max_steps),
        checkpoint_every=(args.checkpoint_every if args.checkpoint_every is not None
                          else run_cfg.loop.checkpoint_every))

    model = build_model(args.model, args.weights)
    if args.init_weights:
        (init_partial if args.init_partial else init_from)(model, args.init_weights)
    check_chain(model, ds)                      # one shipped chain, once (05 C2)
    t0 = time.time()
    results = run_schedule(model, ds, train_cfg=train_cfg, loop_cfg=loop_cfg,
                           stages=args.stages, resume_from=args.resume)
    if not _leave_group():
        return None                     # rank > 0: rank 0 scores and writes
    print(f"  trained {sum(r.steps for r in results)} step(s) across "
          f"{len(results)} stage(s) in {time.time() - t0:.0f}s", flush=True)
    for r in results:
        print(f"    {r.stage:<13} steps={r.steps:<5} passes={r.passes_done} "
              f"ckpts={len(r.checkpoints)} truncated={r.truncated}", flush=True)

    how = "soup-all" if (args.select == "soup" and args.soup_all_stages) else args.select
    selection = select_weights(model, results, how)
    print(f"  weights scored: {selection}", flush=True)

    # the evaluation draw: no augments, the test chain kept (06 P8 / 05 C5)
    val_specs = frozen_eval_specs(Sampler(view, run_cfg.draw.for_eval(), slice_="val", fold=fold),
                                  args.eval_n, seed=args.eval_seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_specs.json").write_text(json.dumps([s.to_dict() for s in val_specs]),
                                        encoding="utf-8")
    val_ds = SpecDataset.frozen(val_specs, index, rcfg, slice_="val", fold=fold,
                                ship=run_cfg.ship)
    report = evaluate(model, val_ds, batch_size=model.cfg.runtime.batch_size,
                      device=args.device, precision=args.eval_precision, fold=fold)

    train_specs = list(train_sampler.epoch_specs(args.draws, seed=train_cfg.seed))
    kind, detail = measured_split_kind(train_specs, val_specs, index)
    gates = run_gates(report, folds=folds_tbl, eval_specs=val_specs, manifest=view,
                      slice_="val", fold=fold,
                      scheme_version=run_cfg.folds.scheme_version)
    tripwires = leak_tripwires(report.metrics, kind, detail)
    m = report.metrics
    print(f"  score={m.score:.4f} eer_file={m.eer_file:.4f} "
          f"eer_voice={m.eer_voice:.4f} eer_music={m.eer_music:.4f}", flush=True)
    print(f"  split={kind}  gates.ok={gates.ok}  tripwires.ok={tripwires.ok}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    # 🔴 `save_checkpoint`, not a bespoke dict. The weights that produced the
    # numbers are the one artifact that has to be reloadable, and every reader in
    # this repo wants `config` + `state_dict`: `models.model.load_checkpoint`,
    # `training.checkpoint.load_train_checkpoint` and `checkpoint_soup` each raise
    # without them. An earlier version wrote {"model", "selection"}, which no
    # loader accepts -- so the across-run soup docs/architecture/05 describes was
    # impossible from a finished run. `selection` rides alongside rather than
    # replacing the contract.
    save_checkpoint(model, out / "scored.pt")
    (out / "selection.txt").write_text(selection + "\n", encoding="utf-8")
    # the chain the weights were trained behind, for `processing.infer`
    (out / "processing.json").write_text(json.dumps(dump_processing_config(run_cfg), indent=2),
                                         encoding="utf-8")
    report.predictions.to_parquet(out / "val_predictions.parquet", index=False)

    caveats = [c for r in results for c in r.caveats]
    return FoldResult(fold=fold, validation=report, gates=gates,
                      tripwires=tripwires), caveats, selection


def train_all_data(*, manifest, folds_tbl, args, run_cfg, train_cfg) -> str:
    """docs/training/07 §3: train on TRAIN + VAL of every fold (PROBE stays
    sealed) and save the selected weights. No validation: nothing is held out,
    so the fold runs of the same recipe are what say how far to train."""
    view = apply_folds(manifest, folds_tbl, 0).copy()
    view.loc[view["slice"] == "val", "slice"] = "train"
    index = ManifestIndex.from_frame(view)
    rcfg = run_cfg.render
    if args.corpus_root is not None:
        rcfg = dataclasses.replace(rcfg, root=pathlib.Path(args.corpus_root))
    out = pathlib.Path(args.out) / f"all_data_seed{train_cfg.seed}"
    print(f"\n=== all data, seed {train_cfg.seed}: "
          f"{(view['slice'] == 'train').sum()} train rows ===", flush=True)
    sampler = Sampler(view, run_cfg.draw, slice_="train", fold=None)
    ds = SpecDataset.from_sampler(sampler, args.draws, index, rcfg,
                                  seed=train_cfg.seed, ship=run_cfg.ship)
    loop_cfg = dataclasses.replace(
        run_cfg.loop, out_dir=out, device=args.device,
        max_steps=(args.max_steps if args.max_steps is not None
                   else run_cfg.loop.max_steps),
        checkpoint_every=(args.checkpoint_every if args.checkpoint_every is not None
                          else run_cfg.loop.checkpoint_every))
    model = build_model(args.model, args.weights)
    if args.init_weights:
        (init_partial if args.init_partial else init_from)(model, args.init_weights)
    check_chain(model, ds)
    t0 = time.time()
    results = run_schedule(model, ds, train_cfg=train_cfg, loop_cfg=loop_cfg,
                           stages=args.stages, resume_from=args.resume)
    if not _leave_group():
        return None                     # rank > 0: rank 0 selects and writes
    print(f"  trained {sum(r.steps for r in results)} step(s) in {time.time() - t0:.0f}s",
          flush=True)
    how = "soup-all" if (args.select == "soup" and args.soup_all_stages) else args.select
    selection = select_weights(model, results, how)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out / "scored.pt")
    (out / "selection.txt").write_text(selection + "\n", encoding="utf-8")
    (out / "processing.json").write_text(json.dumps(dump_processing_config(run_cfg), indent=2),
                                         encoding="utf-8")
    print(f"  weights saved: {selection} -> {out / 'scored.pt'}", flush=True)
    return selection


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True,
                   help="directory holding manifest.parquet and folds.parquet "
                        "(e.g. /data/project/private/dacon-corpus/manifests/strategy-v2)")
    p.add_argument("--corpus-root", default=None,
                   help="the audio root manifest paths are relative to; default: "
                        "the processing config's render.root")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--model", default="configs/a_shared_trunk.yaml")
    p.add_argument("--train", default="configs/train_joint.yaml")
    p.add_argument("--processing", default="configs/processing_v1.yaml",
                   help="draw / render / ship / folds / loop (06 P8); the older "
                        "run_*.yaml is the training sampler's and is not read")
    p.add_argument("--weights",
                   help="frontend checkpoint dir, or name=DIR,name=DIR for a config with "
                        "several frontends (e.g. audio=.../beats,speech=.../xlsr-300m); "
                        "omit for a stub config")
    p.add_argument("--folds", default="0", help="comma-separated, or 'all'")
    p.add_argument("--all-data", action="store_true",
                   help="train on TRAIN+VAL of every fold (PROBE sealed), no validation; "
                        "writes <out>/all_data_seed<seed>/ (docs/training/07 §3)")
    p.add_argument("--seed", type=int, help="override TrainConfig.seed (draw and batch order)")
    p.add_argument("--init-weights",
                   help="a scored.pt whose weights initialise the model (run 2 from run 1's "
                        "soup); a fresh optimizer, schedule and draw -- unlike --resume")
    p.add_argument("--init-partial", action="store_true",
                   help="with --init-weights: load only the keys that match by name AND "
                        "shape; a head with any mismatch stays fresh whole; the speech "
                        "trunk, its LoRA and the fusion may be fresh (docs/training/13 M3). "
                        "Off: the strict init")
    p.add_argument("--resume", help="a training checkpoint (e.g. <out>/fold1/joint-pass0.pt) "
                        "to continue the first --stages stage from; the run's other "
                        "arguments must match the ones it was started with")
    p.add_argument("--stages", default=",".join(STAGES),
                   help=f"comma-separated subset of {STAGES}, in order")
    p.add_argument("--select", choices=SELECTIONS, default="raw",
                   help="which weights to score: raw | ema | soup")
    p.add_argument("--soup-all-stages", action="store_true",
                   help="soup across every stage rather than the last one. Off by "
                        "default: S1's checkpoints sit far closer to the init than "
                        "S3's, so pooling them averages the result backwards")
    p.add_argument("--draws", type=int, default=2400, help="specs drawn per pass")
    p.add_argument("--eval-n", type=int, default=6000,
                   help="frozen eval specs; VG1 A8/A9 needs ~6000 to clear its floors")
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--eval-precision", default="fp32",
                   help="fp32 keeps the submitted ranking resolution; training uses "
                        "TrainConfig.precision, which is a different field")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, help="override TrainConfig.epochs")
    p.add_argument("--batch-size", type=int, help="override TrainConfig.batch_size")
    p.add_argument("--max-steps", type=int,
                   help="truncate each stage: Replay speed, and NOT QUOTABLE")
    p.add_argument("--checkpoint-every", type=int,
                   help="override LoopConfig.checkpoint_every (soup needs >= 2)")
    p.add_argument("--min-pool", type=int,
                   help="drop folds whose masked pool is below this, and record it")
    p.add_argument("--allow-unquotable", action="store_true",
                   help="exit 0 even when the run is not quotable. For a deliberate "
                        "smoke run (--max-steps, a small --eval-n): without it Slurm "
                        "reports the task FAILED, which is right for a real run and "
                        "noise for a test")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()
    if args.init_partial and not args.init_weights:
        raise SystemExit("--init-partial needs --init-weights")

    # Under torchrun: join the group first, pin this rank's GPU, and keep the
    # other ranks' stdout out of the Slurm log (their errors still reach stderr).
    d = init_from_env(device=args.device)
    if d.enabled:
        if args.device.startswith("cuda"):
            args.device = f"cuda:{d.local_rank}"
        if not d.main:
            sys.stdout = open(os.devnull, "w")

    corpus = pathlib.Path(args.manifest_dir)
    manifest = load_manifest(corpus / "manifest.parquet")
    folds_tbl = pd.read_parquet(corpus / "folds.parquet")
    run_cfg = load_processing_config(args.processing)
    train_cfg = load_train_config(args.train)
    # 🔴 `run_default.yaml` leaves `folds.scheme_version` null, and a null one makes
    # VG1 A10 -- "the run's scheme_version matches folds.parquet's" -- report SKIP
    # instead of comparing anything. The manifest carries exactly one, enforced by
    # `validate_manifest`, so default to it rather than let a gate go quiet.
    if run_cfg.folds.scheme_version is None:
        run_cfg = dataclasses.replace(run_cfg, folds=dataclasses.replace(
            run_cfg.folds, scheme_version=str(manifest["scheme_version"].iloc[0])))
    for field, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                         ("seed", args.seed)):
        if value is not None:
            train_cfg = dataclasses.replace(train_cfg, **{field: value})

    args.stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected a subset of {STAGES}")
    available = sorted(int(f) for f in folds_tbl["fold"].dropna().unique())
    wanted = available if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    missing = [f for f in wanted if f not in available]
    if missing:
        raise SystemExit(f"fold(s) {missing} not in folds.parquet (has {available})")
    if d.enabled and len(args.stages) != 1:
        # one DDP wrapper per stage call; the runs here use `--stages joint`
        raise SystemExit(f"under DDP one invocation trains one stage; got {args.stages}")
    if d.enabled and not args.all_data and len(wanted) != 1:
        # after the first fold every rank has left the process group
        raise SystemExit(f"under DDP one invocation trains one fold or --all-data; "
                         f"got folds {wanted}")

    print(f"corpus   {corpus}  ({len(manifest)} rows, folds {available}, "
          f"scheme {run_cfg.folds.scheme_version})")
    print(f"model    {args.model}" + (f"  weights {args.weights}" if args.weights else ""))
    print(f"stages   {' -> '.join(args.stages)}")
    print(f"folds    {wanted}   select={args.select}   device={args.device}")
    print(f"draws    {args.draws}/pass   eval {args.eval_n} specs @ {args.eval_precision}")
    if args.max_steps:
        print(f"⚠️  max_steps={args.max_steps}: every stage is truncated and the run is "
              f"NOT QUOTABLE")
    if d.enabled:
        print(f"ddp      world {d.world} x batch {train_cfg.batch_size} x accum "
              f"{run_cfg.loop.grad_accum} = global batch "
              f"{d.world * train_cfg.batch_size * run_cfg.loop.grad_accum}")
    if args.dry_run:
        teardown(d)
        return 0

    if args.all_data:
        train_all_data(manifest=manifest, folds_tbl=folds_tbl, args=args,
                       run_cfg=run_cfg, train_cfg=train_cfg)
        return 0

    results, caveats, selections = [], [], set()
    for fold in wanted:
        ret = train_fold(fold, manifest=manifest, folds_tbl=folds_tbl,
                         args=args, run_cfg=run_cfg, train_cfg=train_cfg)
        if ret is None:                  # a DDP rank > 0 after training
            return 0
        fr, cav, sel = ret
        results.append(fr)
        caveats += cav
        selections.add(sel)

    # 🔴 The union, across every stage of every fold. A caveat that stops at the
    # StageResult is one nobody reads, and one carrying NOT_QUOTABLE must void
    # the run whatever the gates say.
    caveats.append(f"weights scored: {'; '.join(sorted(selections))}")
    report = aggregate_folds(results, min_pool=args.min_pool, caveats=tuple(caveats))

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"score_mean={report.score_mean:.4f}  score_sd={report.score_sd:.4f} "
          f"(a measurement: {report.sd_is_a_measurement})")
    print(f"QUOTABLE: {report.quotable}")
    for c in report.blocking_caveats:
        print(f"  BLOCKING: {c}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 🔴 Per-fold when a single fold was asked for, because two array tasks share
    # one --out: `sbatch --array=0-1` had both tasks write `ledger_row.json` and
    # fold 1 silently overwrote fold 0's row. Only a run that actually aggregated
    # several folds owns the unqualified name.
    row_path = (out / "ledger_row.json" if len(results) > 1
                else out / f"fold{results[0].fold}" / "ledger_row.json")
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(report.as_ledger_row(), indent=2, default=str),
                        encoding="utf-8")
    print(f"\nledger row -> {row_path}")
    if len(results) == 1 and args.folds not in ("all",):
        print("NOTE: one fold per invocation aggregates one fold. Score_sd is 0.0 "
              "because nothing varied. For a cross-fold number use --folds all in a "
              "single job; an array gives per-fold rows that nothing combines yet.")
    if report.quotable or args.allow_unquotable:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
