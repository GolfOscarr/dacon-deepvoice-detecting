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

⚠️ One GPU per invocation, by design. Folds are independent, so N folds is a job
array rather than a distributed run; the sampler's bitwise-resume guarantee is
keyed on `(sample_id, epoch, seed)` and sharding it across ranks would have to be
re-established from scratch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models.config import load_model_config, load_train_config        # noqa: E402
from models.model import DeepVoiceNet                                 # noqa: E402
from training.checkpoint import checkpoint_soup                       # noqa: E402
from training.config import load_run_config                           # noqa: E402
from training.dataset import SpecDataset, frozen_eval_specs           # noqa: E402
from training.folds import apply_folds                                # noqa: E402
from training.loop import LoopConfig, run_schedule                    # noqa: E402
from training.manifest import load_manifest                           # noqa: E402
from training.render import ManifestIndex, RenderConfig               # noqa: E402
from training.sampler import Sampler                                  # noqa: E402
from training.stages import STAGES                                    # noqa: E402
from training.validate import (FoldResult, aggregate_folds, evaluate,  # noqa: E402
                               leak_tripwires, measured_split_kind, run_gates)

SELECTIONS = ("raw", "ema", "soup")


def build_model(model_cfg_path: str, weights: str | None) -> DeepVoiceNet:
    cfg = load_model_config(model_cfg_path)
    if weights is not None:
        cfg = dataclasses.replace(cfg, frontends={
            k: dataclasses.replace(v, weights=weights) for k, v in cfg.frontends.items()})
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
    paths = [c.path for r in results for c in r.checkpoints]
    if len(paths) < 2:
        raise SystemExit(
            f"--select soup needs at least two checkpoints, found {len(paths)}. "
            "Set LoopConfig.checkpoint_every, or raise --epochs so passes checkpoint.")
    model.load_state_dict(checkpoint_soup(paths), strict=True)
    return f"soup ({len(paths)} checkpoints, uniform average)"


def train_fold(fold: int, *, manifest, folds_tbl, args, run_cfg, train_cfg):
    view = apply_folds(manifest, folds_tbl, fold)
    index = ManifestIndex.from_frame(view)
    rcfg = dataclasses.replace(run_cfg.render, root=pathlib.Path(args.corpus))
    out = pathlib.Path(args.out) / f"fold{fold}"

    n_train = (view["slice"] == "train").sum()
    n_val = (view["slice"] == "val").sum()
    print(f"\n=== fold {fold}: {n_train} train rows, {n_val} val rows ===", flush=True)

    train_sampler = Sampler(view, run_cfg.sampler, slice_="train")
    ds = SpecDataset.from_sampler(train_sampler, args.draws, index, rcfg,
                                  seed=train_cfg.seed)
    loop_cfg = dataclasses.replace(
        run_cfg.loop, out_dir=out, device=args.device,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every
        if args.checkpoint_every is not None else run_cfg.loop.checkpoint_every)

    model = build_model(args.model, args.weights)
    t0 = time.time()
    results = run_schedule(model, ds, train_cfg=train_cfg, loop_cfg=loop_cfg,
                           stages=args.stages)
    print(f"  trained {sum(r.steps for r in results)} step(s) across "
          f"{len(results)} stage(s) in {time.time() - t0:.0f}s", flush=True)
    for r in results:
        print(f"    {r.stage:<13} steps={r.steps:<5} passes={r.passes_done} "
              f"ckpts={len(r.checkpoints)} truncated={r.truncated}", flush=True)

    selection = select_weights(model, results, args.select)
    print(f"  weights scored: {selection}", flush=True)

    val_specs = frozen_eval_specs(Sampler(view, run_cfg.sampler, slice_="val"),
                                  args.eval_n, seed=args.eval_seed)
    val_ds = SpecDataset.frozen(val_specs, index, rcfg, slice_="val", fold=fold)
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
    torch.save({"model": model.state_dict(), "selection": selection}, out / "scored.pt")
    report.predictions.to_parquet(out / "val_predictions.parquet", index=False)

    caveats = [c for r in results for c in r.caveats]
    return FoldResult(fold=fold, validation=report, gates=gates,
                      tripwires=tripwires), caveats, selection


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True,
                   help="root holding manifest.parquet and folds.parquet")
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--model", default="configs/a_shared_trunk.yaml")
    p.add_argument("--train", default="configs/train_joint.yaml")
    p.add_argument("--run", default="configs/run_default.yaml")
    p.add_argument("--weights", help="frontend checkpoint dir; omit for a stub config")
    p.add_argument("--folds", default="0", help="comma-separated, or 'all'")
    p.add_argument("--stages", default=",".join(STAGES),
                   help=f"comma-separated subset of {STAGES}, in order")
    p.add_argument("--select", choices=SELECTIONS, default="raw",
                   help="which weights to score: raw | ema | soup")
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
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    corpus = pathlib.Path(args.corpus)
    manifest = load_manifest(corpus / "manifest.parquet")
    folds_tbl = pd.read_parquet(corpus / "folds.parquet")
    run_cfg = load_run_config(args.run)
    train_cfg = load_train_config(args.train)
    # 🔴 `run_default.yaml` leaves `folds.scheme_version` null, and a null one makes
    # VG1 A10 -- "the run's scheme_version matches folds.parquet's" -- report SKIP
    # instead of comparing anything. The manifest carries exactly one, enforced by
    # `validate_manifest`, so default to it rather than let a gate go quiet.
    if run_cfg.folds.scheme_version is None:
        run_cfg = dataclasses.replace(run_cfg, folds=dataclasses.replace(
            run_cfg.folds, scheme_version=str(manifest["scheme_version"].iloc[0])))
    for field, value in (("epochs", args.epochs), ("batch_size", args.batch_size)):
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

    print(f"corpus   {corpus}  ({len(manifest)} rows, folds {available}, scheme {run_cfg.folds.scheme_version})")
    print(f"model    {args.model}" + (f"  weights {args.weights}" if args.weights else ""))
    print(f"stages   {' -> '.join(args.stages)}")
    print(f"folds    {wanted}   select={args.select}   device={args.device}")
    print(f"draws    {args.draws}/pass   eval {args.eval_n} specs @ {args.eval_precision}")
    if args.max_steps:
        print(f"⚠️  max_steps={args.max_steps}: every stage is truncated and the run is "
              f"NOT QUOTABLE")
    if args.dry_run:
        return 0

    results, caveats, selections = [], [], set()
    for fold in wanted:
        fr, cav, sel = train_fold(fold, manifest=manifest, folds_tbl=folds_tbl,
                                  args=args, run_cfg=run_cfg, train_cfg=train_cfg)
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
    (out / "ledger_row.json").write_text(
        json.dumps(report.as_ledger_row(), indent=2, default=str), encoding="utf-8")
    print(f"\nledger row -> {out / 'ledger_row.json'}")
    return 0 if report.quotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
