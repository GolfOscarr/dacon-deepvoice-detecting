"""The stage runner: S1 -> S2 -> S3, EMA, checkpointing and the pass schedule.

Everything above `training.dataset`. The pipeline decides *what* a sample is and
renders it; this module decides *when* it is shown to the model and what the
model is allowed to learn from it at that point. Whether the number that comes
out is allowed to be quoted is `training.validate`'s question.

The one thing here that is load-bearing and is a decision rather than plumbing:

Critical: **a resumed run must be the run it claims to be.** Restoring weights
and the optimizer is not enough: the corpus is drawn, so a resume that forgets
*where in the draw it was* silently trains on a different corpus and reports it
under the same exp_id. `Sampler.epoch_specs` is keyed on epoch-local `i`,
`epoch` and `seed` for exactly this reason -- there is no hidden generator to
serialise, so the sampler's state is the tuple `(pass_index, epoch_seed,
n_specs, batch_seed, batch_index)` and it lives in the checkpoint
(`training.checkpoint`). `tests/test_loop.py` proves a resumed run reproduces an
uninterrupted one **bitwise**, and proves the check goes red when that tuple is
dropped.

The neighbours, all of which this module re-exports so imports written before
the split keep resolving:

| Module | Holds |
|---|---|
| `training.stages` | the S1/S2/S3 schedule, S3's codec menu, `autocast_for` |
| `training.checkpoint` | the EMA, `SamplerState`, save/load, the soup |
| `training.validate` | scoring a fold, the VG gates, the leak tripwires |
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from models.config import TrainConfig
from models.losses import multitask_loss
from models.model import DeepVoiceNet
from training.checkpoint import (EMA, SamplerState, TrainCheckpoint,
                                 _set_rng_state, checkpoint_soup,
                                 load_train_checkpoint, save_train_checkpoint)
from training.collate import bucket_batches, collate, spec_durations
from training.dataset import SpecDataset
from processing.render import ManifestIndex, RenderConfig, render
from processing.ship import ShipConfig, ship
from training.spec import SampleSpec
from training.stages import (CODEC_VARIANTS, STAGES, StagePlan, _stage_loss_config,
                             autocast_for, codec_variant_specs, pass_plan,
                             stage_plan,
                             trainable_parameters)

__all__ = [
    "CODEC_VARIANTS", "STAGES",
    "EMA", "FoldResult", "LoopConfig", "RunReport", "SamplerState", "StagePlan",
    "TrainCheckpoint", "ValidationReport",
    "aggregate_folds", "autocast_for", "checkpoint_soup", "codec_variant_specs",
    "evaluate", "generator_key", "leak_tripwires", "measured_split_kind",
    "output_sanity", "predict", "prediction_frame", "run_gates", "run_schedule",
    "spec_digest",
    "stage_plan", "train_stage", "trainable_parameters", "validate_fold",
]


# --------------------------------------------------------------------------- #
# 5. The loop
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoopConfig:
    """Run-level knobs. Everything model- or objective-shaped lives in the configs.

    Critical: **every field here is read by this module**, and
    `tests/test_loop.py::test_no_loop_config_field_is_silently_ignored` enforces
    it. A knob that validates and then does nothing is worse than a missing
    knob: it makes an ablation report a difference it never tested
    (`models.model.CONSUMED_ELSEWHERE` is the same guard one layer down).
    """

    out_dir: Path = Path("runs/dev")
    n_buckets: int = 4
    #: 0 disables the EMA entirely. Anything EMA- or checkpoint-related must
    #: skip Replay speed and start at Medium: Replay systematically favours ideas
    #: that help early in training (docs/validation/03 §2).
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    device: str = "cpu"
    #: 0 = checkpoint at pass boundaries only. Any other value also checkpoints
    #: every N optimizer steps, which is what makes a mid-epoch resume testable.
    checkpoint_every: int = 0
    #: Replay speed and the tests. A run with `max_steps` set is a truncated
    #: run -- `StageResult.truncated` says so, and it is not quotable.
    max_steps: int | None = None
    #: 0 renders inline. N > 0 renders in N worker processes, in order: a render
    #: is a pure function of its spec (processing.render pins one torch thread
    #: and keys its RNG on the spec), so the batches are the inline batches and
    #: the bitwise-resume guarantee is unchanged (docs/training/07 §3).
    render_workers: int = 0
    #: "constant" (the historical behaviour) or "cosine": linear warmup over
    #: `warmup_steps`, then cosine decay to `min_lr_ratio` of the base rate at
    #: the last step of the stage. A pure function of the global step, so a
    #: resume needs no scheduler state.
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    #: The frontends' trainable parameters (LoRA, GeM) run at this multiple of
    #: the base rate; the heads at the base rate.
    frontend_lr_scale: float = 1.0
    #: 0 = no per-step log. N > 0 appends the mean loss parts of the last N
    #: steps to `<out_dir>/train_log.jsonl` every N steps.
    log_every: int = 0

    def __post_init__(self) -> None:
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {self.n_buckets}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")
        if self.render_workers < 0:
            raise ValueError(f"render_workers must be >= 0, got {self.render_workers}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"lr_schedule must be constant|cosine, got {self.lr_schedule!r}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if not self.frontend_lr_scale > 0:
            raise ValueError(
                f"frontend_lr_scale must be > 0, got {self.frontend_lr_scale}")


@dataclass
class StageResult:
    """What one stage did. Raw material for the experiment ledger."""

    stage: str
    plan: StagePlan
    steps: int
    passes_done: int
    loss_history: list[dict[str, float]] = field(default_factory=list)
    #: `spec_digest` of the corpus each pass drew, in pass order. Critical: two equal
    #: digests mean two passes trained on the same samples, which under S1 would
    #: make any cross-branch comparison an artefact of the schedule.
    pass_digests: list[str] = field(default_factory=list)
    checkpoints: list[TrainCheckpoint] = field(default_factory=list)
    ema: EMA | None = None
    truncated: bool = False

    @property
    def caveats(self) -> tuple[str, ...]:
        """Everything about this stage a result must not be read without.

        Caveat: pass these into `aggregate_folds(..., caveats=...)` so they reach the
        ledger row -- a caveat that stops at the `StageResult` is a caveat
        nobody reads.
        """
        out = list(self.plan.caveats)
        if self.truncated:
            out.append(
                f"stage {self.stage!r} was truncated at {self.steps} step(s) by "
                f"`LoopConfig.max_steps`: this is a Replay-speed or test run and "
                f"is not quotable (docs/validation/03 §2)")
        return tuple(out)


def _schedule(plan: StagePlan, epochs: int) -> list[tuple[int, int]]:
    """The flat ``(group index, epoch within the group)`` pass list.

    Flat, so `pass_index` alone locates a resume *and* keys the draw. Under S1
    that means five branch passes of `epochs` epochs each, every one drawing its
    own corpus -- rather than the same epoch-0 corpus five times, which would
    make the branches' training sets identical and their comparison a coincidence.
    """
    return [(g, e) for g in range(len(plan.branch_groups)) for e in range(epochs)]


def spec_digest(specs: Sequence[SampleSpec]) -> str:
    """A stable fingerprint of a drawn corpus. Measured, not asserted.

    `SamplerState` claims `pass_index` "doubles as the epoch key for the draw",
    so that every branch pass and every epoch sees its own corpus rather than
    replaying epoch 0. That claim lived only in a docstring: a review mutated
    `set_epoch(pass_index)` to `set_epoch(0)` and the whole suite stayed green,
    because every test looked at losses, weights and checkpoint fields and none
    looked at *which specs were drawn*.

    So the drawn corpus is now recorded per pass, in `StageResult.pass_digests`.
    Two passes with the same digest drew the same samples; two runs that claim
    the same corpus can be compared without shipping the spec lists around. It
    covers the full `SampleSpec`, `normalize` included, so S3's codec variants
    change it too.

    Caveat: not a substitute for the assertion: `tests/test_loop.py` checks the drawn
    spec *contents* directly and cross-checks the digest against them, because a
    digest that happened to include a counter would differ every pass and let the
    same mutation through again.
    """
    h = hashlib.blake2b(digest_size=16)
    for spec in specs:
        h.update(repr(spec.to_dict()).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _render_batch(specs: Sequence[SampleSpec], indices: Sequence[int],
                  index: ManifestIndex, cfg: RenderConfig) -> dict[str, Any]:
    return collate([render(specs[i], index, cfg) for i in indices])


_WORKER: dict[str, Any] = {}


def _worker_init(index: ManifestIndex, cfg: RenderConfig) -> None:
    _WORKER["index"], _WORKER["cfg"] = index, cfg


def _worker_render(spec: SampleSpec):
    return render(spec, _WORKER["index"], _WORKER["cfg"])


def _iter_batches(specs: Sequence[SampleSpec], batches: Sequence[Sequence[int]],
                  first: int, index: ManifestIndex, cfg: RenderConfig,
                  pool) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(batch_index, collated batch)`` from ``first`` on, inline or pooled.

    Pooled: every sample of the pass is submitted in batch order, ``imap``
    returns them in that order, and they are regrouped into the same batches
    the inline path builds. Submission runs ahead in windows so the render
    queue cannot outgrow memory when the GPU is the slower side.
    """
    if pool is None:
        for b in range(first, len(batches)):
            yield b, _render_batch(specs, batches[b], index, cfg)
        return
    window = 64
    for lo in range(first, len(batches), window):
        chunk = range(lo, min(lo + window, len(batches)))
        flat = [specs[i] for b in chunk for i in batches[b]]
        rendered = pool.imap(_worker_render, flat, chunksize=1)
        for b in chunk:
            yield b, collate([next(rendered) for _ in batches[b]])


def lr_factor(step: int, total: int, cfg: LoopConfig) -> float:
    """The multiple of the base rate at optimizer step ``step`` (0-based)."""
    if cfg.lr_schedule == "constant":
        return 1.0
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    span = max(1, total - cfg.warmup_steps)
    t = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def _param_groups(model: DeepVoiceNet, params: Sequence[torch.nn.Parameter],
                  base_lr: float, cfg: LoopConfig) -> list[dict[str, Any]]:
    """Heads at the base rate, frontend parameters at ``frontend_lr_scale``."""
    fe = {id(p) for p in model.frontends.parameters()}
    heads = [p for p in params if id(p) not in fe]
    front = [p for p in params if id(p) in fe]
    groups = []
    if heads:
        groups.append({"params": heads, "lr": base_lr, "base_lr": base_lr})
    if front:
        lr = base_lr * cfg.frontend_lr_scale
        groups.append({"params": front, "lr": lr, "base_lr": lr})
    return groups


def check_chain(model: DeepVoiceNet, dataset: SpecDataset) -> None:
    """06 P8 / 05 C2: the shipped chain (`processing.ship`) is applied ONCE,
    between the collator and the model, in training, evaluation and
    inference alike. The model's own `AudioConfig.band_hz` must therefore be
    off, and the three sample rates must agree -- they are defined in three
    places with no other cross-check."""
    sr_render = int(dataset.cfg.audio.sample_rate)
    sr_ship = int(dataset.ship.sample_rate)
    sr_model = int(model.cfg.audio.sample_rate)
    if not sr_render == sr_ship == sr_model:
        raise ValueError(
            f"sample rates disagree: render {sr_render}, ship {sr_ship}, model {sr_model}")
    if model.cfg.audio.band_hz is not None:
        raise ValueError(
            f"model.cfg.audio.band_hz = {model.cfg.audio.band_hz}: the band limit is the "
            f"shipped chain's (ShipConfig.band_hz = {dataset.ship.band_hz}); applying it in "
            f"the model too would filter twice in training and once at test")


def shipped(batch: Mapping[str, Any], cfg: ShipConfig, device: torch.device) -> torch.Tensor:
    """The model input: the collated ``(B, C, S)`` through the shipped chain."""
    return ship(batch["wav"].to(device), cfg, batch["lengths"].to(device))


def train_stage(model: DeepVoiceNet, dataset: SpecDataset, *,
                train_cfg: TrainConfig, loop_cfg: LoopConfig | None = None,
                stage: str | None = None,
                resume_from: Path | str | None = None) -> StageResult:
    """Train one stage. Resumable, EMA'd, and checkpointed with its draw state.

    ``dataset`` must be a **training** `SpecDataset` (`from_sampler`): the loop
    calls `set_epoch`, and a frozen eval set refuses that on purpose.

    Caveat: rendering happens inline rather than through a `DataLoader` with workers.
    That is a deliberate limit, not an oversight: worker processes would put the
    draw behind a second, per-worker RNG and the bitwise-resume guarantee above
    would become a claim about `torch.utils.data`'s seeding. There is no corpus
    yet, so nothing is waiting on throughput; when there is, the change is a
    `DataLoader(dataset, batch_sampler=plan, collate_fn=collate)` **plus** a new
    resume test, in that order.
    """
    loop_cfg = loop_cfg or LoopConfig()
    stage = stage or train_cfg.stage
    plan = stage_plan(stage, model.cfg)
    device = torch.device(loop_cfg.device)
    model.to(device)

    if dataset.is_frozen:
        raise ValueError(
            "train_stage needs a training SpecDataset (SpecDataset.from_sampler): "
            "a frozen spec list is the evaluation set and refuses set_epoch")

    passes = _schedule(plan, train_cfg.epochs)
    ema = EMA(model, loop_cfg.ema_decay) if loop_cfg.ema_decay else None
    # fp16 only. bf16 has fp32's exponent range, so it needs no scaler, and an
    # enabled scaler under bf16 would add a stateful factor to a run that does
    # not need one -- one more thing a resume can silently drop. `precision`
    # defaults to bf16 for exactly this reason; fp16 is the *inference* precision.
    scaler = torch.amp.GradScaler(device.type, enabled=train_cfg.precision == "fp16")
    result = StageResult(stage=stage, plan=plan, steps=0, passes_done=0, ema=ema)

    start = SamplerState(pass_index=0, epoch_seed=dataset.seed,
                         n_specs=int(dataset.n_per_epoch),
                         batch_seed=train_cfg.seed, batch_index=0)
    optimizer_state = scaler_state = None
    if resume_from is not None:
        blob = load_train_checkpoint(resume_from, map_location=device)
        if blob["stage"] != stage:
            raise ValueError(
                f"checkpoint is stage {blob['stage']!r}, this call is {stage!r}: "
                "resuming across stages would restore an optimizer built over a "
                "different parameter set")
        model.load_state_dict(blob["state_dict"], strict=True)
        if blob.get("ema") is not None and ema is not None:
            ema.load_state_dict(blob["ema"])
        _set_rng_state(blob["rng"])
        start = blob["sampler"]
        # Critical: a resume that redraws under a different key is a different run under
        # the same name. The draw is `(i, epoch, seed)` and `n`, so a mismatch in
        # either is fatal rather than a warning: nothing downstream can see it.
        if (start.epoch_seed, start.n_specs) != (dataset.seed, int(dataset.n_per_epoch)):
            raise ValueError(
                f"checkpoint drew {start.n_specs} specs at seed {start.epoch_seed}; "
                f"this dataset draws {dataset.n_per_epoch} at seed {dataset.seed}. "
                "Resuming would train on a different corpus and report it as the "
                "same run")
        optimizer_state = blob.get("optimizer")
        scaler_state = blob.get("scaler")
        result.steps = int(blob["global_step"])

    group_index = -1
    optimizer: torch.optim.Optimizer | None = None
    # spawn, not fork: the parent may already hold a CUDA context, which a
    # forked child inherits and must never touch.
    pool = (mp.get_context("spawn").Pool(loop_cfg.render_workers, _worker_init,
                                          (dataset.index, dataset.cfg))
            if loop_cfg.render_workers else None)

    for pass_index in range(start.pass_index, len(passes)):
        g, _ = passes[pass_index]
        group = plan.branch_groups[g]
        if g != group_index:
            group_index = g
            optimizer = torch.optim.AdamW(
                _param_groups(model, trainable_parameters(model, plan, group),
                              train_cfg.lr, loop_cfg),
                lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                optimizer_state = None
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
                scaler_state = None
        loss_cfg_model = _stage_loss_config(model.cfg, group)

        # Critical: `pass_index` is the epoch key, so every pass draws its own corpus --
        # rather than five branch passes replaying epoch 0, which would make any
        # cross-branch comparison an artefact of the schedule. Recorded rather
        # than trusted: `pass_digests` is what a test can assert on.
        dataset.set_epoch(pass_index)
        # One constructor for the spec list and the batch plan, shared with the
        # audit (training/stages.py::pass_plan). Building it here and again in
        # `dataset.training_batches` is what made "audit the same order you
        # train in" false for every pass but pass 0 at seed 0.
        specs, batches = pass_plan(dataset.specs, plan,
                                   batch_size=train_cfg.batch_size,
                                   n_buckets=loop_cfg.n_buckets,
                                   seed=train_cfg.seed, pass_index=pass_index)
        result.pass_digests.append(spec_digest(specs))
        batch_seed = start.batch_seed if pass_index == start.pass_index \
            else train_cfg.seed + pass_index
        if batch_seed != train_cfg.seed + pass_index:
            # Resuming mid-pass: the plan must come from the checkpoint's seed,
            # not the schedule's, or the resumed pass steps through a different
            # batching than the one it was interrupted in.
            batches = bucket_batches(spec_durations(specs), train_cfg.batch_size,
                                     n_buckets=loop_cfg.n_buckets, seed=batch_seed)
        first_batch = start.batch_index if pass_index == start.pass_index else 0

        model.train()
        parts_acc: list[dict[str, float]] = []
        total_steps = len(passes) * len(batches)
        if loop_cfg.max_steps is not None:
            total_steps = min(total_steps, loop_cfg.max_steps)
        stream = _iter_batches(specs, batches, first_batch, dataset.index, dataset.cfg,
                               pool)
        for batch_index, batch in stream:
            state = SamplerState(pass_index, start.epoch_seed, start.n_specs,
                                 batch_seed, batch_index)
            if loop_cfg.max_steps is not None and result.steps >= loop_cfg.max_steps:
                result.truncated = True
                _checkpoint(result, model, optimizer, ema, stage, state, train_cfg,
                            loop_cfg, scaler=scaler, tag="truncated")
                _close(pool)
                return result

            factor = lr_factor(result.steps, total_steps, loop_cfg)
            for gp in optimizer.param_groups:
                gp["lr"] = gp.get("base_lr", train_cfg.lr) * factor
            # float32 in, and it stays float32: the model casts under autocast.
            # 06 P8: the shipped chain, once, here -- not `prepare_waveform`.
            wav = shipped(batch, dataset.ship, device)
            lengths = batch["lengths"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            with autocast_for(train_cfg.precision, device):
                out = model(wav, lengths)
                total, parts = multitask_loss(out, targets, loss_cfg_model,
                                              train_cfg.loss)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if loop_cfg.grad_clip:
                # Critical: unscale first. Clipping a *scaled* gradient clips at a
                # threshold that moves with the scaler's own state, so the clip
                # norm would mean something different on every step.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for gp in optimizer.param_groups for p in gp["params"]],
                    loop_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            result.steps += 1
            parts_acc.append(parts)
            if loop_cfg.log_every and result.steps % loop_cfg.log_every == 0:
                recent = _mean_parts(parts_acc[-loop_cfg.log_every:], stage, pass_index,
                                     group)
                _log_line(loop_cfg, result.steps, total_steps, factor, recent)

            if loop_cfg.checkpoint_every and result.steps % loop_cfg.checkpoint_every == 0:
                _checkpoint(result, model, optimizer, ema, stage,
                            SamplerState(pass_index, start.epoch_seed,
                                         start.n_specs, batch_seed, batch_index + 1),
                            train_cfg, loop_cfg, scaler=scaler,
                            tag=f"step{result.steps}")

        result.passes_done += 1
        result.loss_history.append(_mean_parts(parts_acc, stage, pass_index, group))
        _checkpoint(result, model, optimizer, ema, stage,
                    SamplerState(pass_index + 1, start.epoch_seed, start.n_specs,
                                 train_cfg.seed + pass_index + 1, 0),
                    train_cfg, loop_cfg, scaler=scaler, tag=f"pass{pass_index}")
    _close(pool)
    return result


def _close(pool) -> None:
    if pool is not None:
        pool.terminate()
        pool.join()


def _log_line(cfg: LoopConfig, step: int, total: int, factor: float,
              row: Mapping[str, Any]) -> None:
    """One JSON line per `log_every` steps in `<out_dir>/train_log.jsonl`."""
    import json
    import time
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"t": time.time(), "step": step, "total": total, "lr_factor": factor}
    rec.update({k: v for k, v in row.items() if isinstance(v, (int, float, str))})
    with open(out / "train_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _mean_parts(parts: Sequence[Mapping[str, float]], stage: str, pass_index: int,
                group: Sequence[str]) -> dict[str, float]:
    keys = sorted({k for p in parts for k in p})
    row: dict[str, Any] = {"stage": stage, "pass": pass_index,
                           "group": "+".join(group), "n_batches": len(parts)}
    for k in keys:
        vals = [p[k] for p in parts if k in p]
        row[k] = float(np.mean(vals)) if vals else float("nan")
    return row


def _checkpoint(result: StageResult, model, optimizer, ema, stage, state,
                train_cfg, loop_cfg, *, scaler, tag: str) -> None:
    """Write one checkpoint. Critical: ``scaler`` is required, not defaulted.

    It used to default to `None`, and the mid-epoch call site was the one that
    forgot it -- so every `checkpoint_every` file written under fp16 carried
    `scaler: None` and a resume from it replayed the loss scaler's warm-up. A
    default that means "no scaler" is indistinguishable from a caller that
    dropped the argument, so there is no default: an omission is a `TypeError`
    at the call site rather than a wrong number in a file nobody opens.
    """
    ck = save_train_checkpoint(
        Path(loop_cfg.out_dir) / f"{stage}-{tag}.pt", model=model,
        optimizer=optimizer, ema=ema, stage=stage, global_step=result.steps,
        sampler=state, train_cfg=train_cfg, scaler=scaler)
    result.checkpoints.append(ck)


def run_schedule(model: DeepVoiceNet, dataset: SpecDataset, *,
                 train_cfg: TrainConfig, loop_cfg: LoopConfig | None = None,
                 stages: Sequence[str] = STAGES) -> list[StageResult]:
    """S1 -> S2 -> S3 on one model, in order, carrying the weights forward.

    Our own inference: ``codec_aware`` last and never skipped when time is
    short. It is the **best-evidenced stage in the recipe** (ArtifactNet,
    primary source: hard-negative FPR 98.7% -> 8.0%) and it is a schedule
    rather than an architecture, so it is adopted regardless of how the
    frontend question resolves (docs/training/04 §3). Caveat: ``joint`` is kept
    on different grounds -- its EER delta is 0.5 pts, below our local
    resolution -- so it is cheap, not measured.

    Caveat: **the caveats do not travel on their own.** Every stage carries its
    own `StageResult.caveats`, and a caller that reports only the last result
    silently drops S1's -- which is the stage that has one, since S1 is the only
    stage that overrides `FrontendConfig.freeze`. Whatever finally scores the
    run owes the ledger the union: `aggregate_folds(..., caveats=tuple(c for r
    in results for c in r.caveats))`. Nothing here can do it, because scoring
    happens in `training.validate` against a frozen eval set this function never
    sees.
    """
    for s in stages:
        stage_plan(s, model.cfg)              # refuse rank_polish before any work
    return [train_stage(model, dataset, train_cfg=train_cfg, loop_cfg=loop_cfg,
                        stage=s) for s in stages]


# --------------------------------------------------------------------------- #
# 6. Re-exports -- validation lives in `training.validate`
# --------------------------------------------------------------------------- #

#: Imported last, and deliberately: `training.validate` is a leaf that never
#: imports this module, so the one-way dependency is what keeps the split real.
#: These names are here so `from training.loop import evaluate, run_gates, ...`
#: keeps resolving for callers written before the split.
from training.validate import (  # noqa: E402
    FoldResult, RunReport, ValidationReport, aggregate_folds, evaluate,
    generator_key, leak_tripwires, measured_split_kind, output_sanity, predict,
    prediction_frame, run_gates, validate_fold)
from training.validate import (  # noqa: E402
    MUSIC_UNSEEN_FLOOR, RESOLUTION_FLOOR, VOICE_UNSEEN_FLOOR)
