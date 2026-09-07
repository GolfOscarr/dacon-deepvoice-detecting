"""The stage runner: S1 -> S2 -> S3, checkpointing, EMA, souping and validation.

Everything above `training.dataset`. The pipeline decides *what* a sample is and
renders it; this module decides *when* it is shown to the model, what the model
is allowed to learn from it at that point, and whether the number that comes out
is allowed to be quoted.

Four things are load-bearing and each is a decision, not plumbing:

🔴 **A resumed run must be the run it claims to be.** Restoring weights and the
optimizer is not enough: the corpus is drawn, so a resume that forgets *where in
the draw it was* silently trains on a different corpus and reports it under the
same exp_id. `Sampler.epoch_specs` is keyed on epoch-local `i`, `epoch` and
`seed` for exactly this reason -- there is no hidden generator to serialise, so
the sampler's state is the tuple `(pass_index, epoch, seed, n, batch_seed,
batch_index)` and it lives in the checkpoint. `tests/test_loop.py` proves a
resumed run reproduces an uninterrupted one **bitwise**, and proves the check
goes red when that tuple is dropped.

🔴 **Never recompute a metric.** `metrics/` is verified end to end and three
details of the official EER are load-bearing (docs/architecture/08 §5). This
module builds a prediction frame and hands it to `metrics.dacon.dacon_score`,
`metrics.breakdown` and `metrics.aggregate.fold_mean`. It contains no ROC code
and must not grow any.

🔴 **The headline is the mean of per-fold metrics.** Pooling raw OOF scores
across folds measured **0.1705 against a true 0.100** because each fold is scored
by a different model. `aggregate_folds` delegates to `metrics.aggregate.fold_mean`
and nothing here concatenates score columns.

🔴 **The leak tripwires belong here** (docs/pipelines/05 §6): music-fake
unseen-generator < 3% EER, voice-fake < 1%, or perfect separation on a random
split. They fail the run loudly rather than being read off a dashboard. ⚠️ And
"unseen generator" is **measured** from the drawn streams rather than declared --
`measured_split_kind` compares the realised generator sets, in the house style of
`training.registries`, which measures time invariance instead of trusting a
field.

⚠️ **S4 is dropped** (docs/training/04 §4). `stage='rank_polish'` raises here
rather than running S2 again under a different name: `models.config` still
accepts the value, and a stage that validates and then quietly does something
else is how an ablation reports a difference it never tested.
"""

from __future__ import annotations

import dataclasses
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from metrics.aggregate import AggregateMetrics, fold_mean
from metrics.breakdown import breakdown_table, t3_gap
from metrics.dacon import PREDICTION_COLUMNS, MetricSet, dacon_score
from models.audio import prepare_waveform
from models.config import ModelConfig, TrainConfig, dump_config
from models.losses import multitask_loss
from models.model import DeepVoiceNet
from training.audit import AuditReport, audit_specs
from training.collate import bucket_batches, collate, spec_durations
from training.dataset import SpecDataset, eval_batches
from training.folds import check_split_integrity
from training.render import ManifestIndex, RenderConfig, render
from training.spec import SampleSpec

__all__ = [
    "CODEC_VARIANTS", "STAGES",
    "EMA", "FoldResult", "LoopConfig", "RunReport", "SamplerState", "StagePlan",
    "TrainCheckpoint", "ValidationReport",
    "aggregate_folds", "autocast_for", "checkpoint_soup", "codec_variant_specs",
    "evaluate", "generator_key", "leak_tripwires", "measured_split_kind",
    "output_sanity", "predict", "prediction_frame", "run_gates", "run_schedule",
    "stage_plan", "train_stage", "trainable_parameters", "validate_fold",
]


# --------------------------------------------------------------------------- #
# 1. The stage schedule -- docs/training/04 §1
# --------------------------------------------------------------------------- #

#: The three stages that survive. `rank_polish` is deliberately absent: S4 was
#: dropped (docs/training/04 §4) after ★ TFPARN's own ablation moved EER
#: 12.91 -> 12.92 and its second justification landed below our ≈1 pt resolution
#: threshold. `models.config` still *accepts* the value because a config file in
#: the wild may carry it; `stage_plan` refuses it.
STAGES = ("independent", "joint", "codec_aware")

#: S3's 4-way codec menu (docs/training/04 §3). ★ ArtifactNet P2->P3: hard-negative
#: FPR 98.7% -> 8.0%, cross-codec drift -83%; the best-evidenced stage in the
#: recipe, so this is where the schedule's remaining value is concentrated.
#:
#: 🔴 The four are applied to **every** spec, not drawn per spec. That is what
#: keeps `normalize` label-independent by construction: a per-spec draw is a
#: transform parameter, and `container = mp3 if fake else wav` scored AUC 1.000
#: on the I1b metadata probe while every other invariant stayed green
#: (docs/pipelines/05 §1). Expanding uniformly makes the balance structural
#: rather than a property the sampler has to remember.
#:
#: ⚠️ Short of the full A-S3 menu for the reason `render.CODEC_CONTAINERS` gives:
#: AAC/OPUS/AMR-NB/GSM each need their encoder delay verified the way MP3's is,
#: and an uncancelled encoder delay moves the audio while `frame_intervals` stay
#: put. The 8 kHz telephone leg is included because ★ ASVspoof 5's hardest
#: condition is codec-10 (speex, 8 kHz, low bitrate) and that is our telephone
#: slice, named as the worst case by an independent evaluation.
CODEC_VARIANTS: tuple[dict[str, Any], ...] = (
    {},                                              # as decoded
    {"container": "mp3", "bitrate": 64},             # ⚠️ 64 kbps cost MusicDET +37 EER pts
    {"container": "flac"},                           # lossless, but a different container
    {"telephone_hz": 8000, "companding": "ulaw"},    # A-S4, the telephone slice
)


@dataclass(frozen=True)
class StagePlan:
    """What one stage trains, and over what.

    ``branch_groups`` is the whole difference between S1 and S2/S3: S1 is one
    group per branch (each branch alone, in sequence), S2/S3 are a single group
    holding every branch. ⚠️ The component losses stay masked in both -- masking
    mirrors the metric, not the schedule (docs/architecture/08 §2).
    """

    stage: str
    branch_groups: tuple[tuple[str, ...], ...]
    codec_variants: tuple[Mapping[str, Any], ...]
    train_frontends: bool

    def __str__(self) -> str:
        groups = " | ".join("+".join(g) for g in self.branch_groups)
        return (f"{self.stage}: groups [{groups}], "
                f"{len(self.codec_variants)}-way codec, "
                f"frontends {'trainable' if self.train_frontends else 'frozen'}")


def stage_plan(stage: str, cfg: ModelConfig) -> StagePlan:
    """The schedule for one stage. Refuses the dropped one.

    | Stage | Branches | Frontends | Codec |
    |---|---|---|---|
    | ``independent`` (S1) | one at a time | **frozen** | 1-way |
    | ``joint`` (S2)       | all together  | per config | 1-way |
    | ``codec_aware`` (S3) | all together  | per config | **4-way** |

    ⚠️ S1 freezes the frontends *regardless of* ``FrontendConfig.freeze`` -- that
    is what "independent, frozen frontends + adapters" means, and reading the
    model config here instead would make S1 and S2 the same stage on the shipped
    configs (both stubs already say ``freeze: true``), so an S1-vs-S2 comparison
    would measure nothing.
    """
    branches = tuple(cfg.branches)
    if stage == "independent":
        return StagePlan(stage, tuple((b,) for b in branches), ({},), False)
    if stage == "joint":
        return StagePlan(stage, (branches,), ({},), True)
    if stage == "codec_aware":
        return StagePlan(stage, (branches,), CODEC_VARIANTS, True)
    if stage == "rank_polish":
        raise NotImplementedError(
            "stage='rank_polish' is S4, which is DROPPED (docs/training/04 §4): "
            "TFPARN's own ablation moves EER 12.91 -> 12.92 and every gain it "
            "buys is minDCF/Cllr/actDCF, which a ranking metric cannot read. "
            "`LossConfig.ranking_weight` stays 0. Running S2 under this name "
            "would report a stage that was never trained.")
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def codec_variant_specs(specs: Sequence[SampleSpec],
                        variants: Sequence[Mapping[str, Any]]) -> tuple[SampleSpec, ...]:
    """Every spec under every codec variant, grouped by variant.

    🔴 Uniform expansion, never a per-spec draw -- see ``CODEC_VARIANTS``. The
    resulting stream has ``P(normalize | label) = P(normalize)`` exactly, which
    `audit_specs`' **I1b** measures rather than assumes; `tests/test_loop.py`
    mutates the expansion to a label-conditional draw and watches I1b go red.

    ⚠️ The variant is *merged into* whatever the spec already carried, so a
    normalize draw the sampler made (it makes none today -- G1's
    ``signal_chain.yaml`` does not exist) is not silently discarded.
    """
    if not variants:
        raise ValueError("codec_variant_specs needs at least one variant")
    out: list[SampleSpec] = []
    for variant in variants:
        for spec in specs:
            merged = {**spec.normalize, **dict(variant)}
            out.append(dataclasses.replace(spec, normalize=merged))
    return tuple(out)


def trainable_parameters(model: DeepVoiceNet, plan: StagePlan,
                         group: Sequence[str]) -> list[torch.nn.Parameter]:
    """The parameters one pass optimizes. Everything else keeps its value.

    🔴 Selected by *module*, not by zeroing a loss term. Restricting the loss
    alone still lets weight decay and optimizer momentum move a branch nobody is
    training this pass, which is exactly the silent divergence S1 exists to
    avoid -- and it would make "each branch alone" a claim rather than a fact.
    `tests/test_loop.py` asserts the inactive branches come out bitwise unchanged.
    """
    wanted = set(group)
    params: list[torch.nn.Parameter] = []
    for name, head in model.heads.items():
        if name in wanted:
            params += [p for p in head.parameters() if p.requires_grad]
    if plan.train_frontends:
        params += [p for p in model.frontends.parameters() if p.requires_grad]
    for extra in (model.distill_head, model.separation_head):
        if extra is not None and plan.train_frontends:
            params += [p for p in extra.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            f"stage {plan.stage!r} group {tuple(group)} has no trainable parameter. "
            "On a fully frozen config that is a silently empty run, not a fast one")
    return params


def _stage_loss_config(cfg: ModelConfig, group: Sequence[str]) -> ModelConfig:
    """`cfg` restricted to one branch group, for `multitask_loss` to iterate.

    ⚠️ A restricted *config*, not a filtered *output* dict: `multitask_loss`
    walks `cfg.branches`, so this is the one place that decides which heads take
    a loss, and the per-head weights and the masks come along untouched.
    """
    return dataclasses.replace(
        cfg, branches={name: cfg.branches[name] for name in group})


# --------------------------------------------------------------------------- #
# 2. Precision -- docs/pipelines/04 §4
# --------------------------------------------------------------------------- #

_TORCH_DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def autocast_for(precision: str, device: torch.device | str):
    """An autocast context for `TrainConfig.precision`. **The batch is not cast.**

    🔴 The pipeline emits float32 and the *model* casts, under autocast, with
    fp32 master weights. Pre-casting the batch is how this repo produced NaN
    attention: GeM overflowed in fp16 above a feature scale of ~40
    (docs/pipelines/04 §4), and an fp16 batch also arrives at `bandpass`'s rFFT
    and at the loss in a precision neither was measured in.

    ⚠️ `fp32` returns a null context rather than an autocast with float32 -- the
    two are not the same thing, and the second one still routes ops through the
    autocast dispatcher.
    """
    if precision not in _TORCH_DTYPE:
        raise ValueError(f"precision must be one of {sorted(_TORCH_DTYPE)}, got {precision!r}")
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(torch.device(device).type, dtype=_TORCH_DTYPE[precision])


# --------------------------------------------------------------------------- #
# 3. EMA -- docs/architecture/08 §4
# --------------------------------------------------------------------------- #

class EMA:
    """Bias-corrected exponential moving average of the float parameters.

    🔴 Bias-corrected, like Adam's moments, and that is not decoration. A raw EMA
    is initialised at the *starting* weights, so at decay 0.999 it is still 63%
    initialisation after 1,000 steps -- on a short schedule the "EMA weights"
    would mostly be the random init, and the run would report a number for a
    model it never trained. With the correction the EMA after one update is
    exactly the current weights, which is a testable statement and is tested.

    ⚠️ Integer buffers (`num_batches_tracked` and friends) are carried, not
    averaged: an averaged counter is meaningless and `load_state_dict(strict=True)`
    would reject a float one.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.steps = 0
        self._shadow = {k: torch.zeros_like(v, dtype=torch.float32)
                        for k, v in model.state_dict().items()
                        if v.is_floating_point()}
        self._frozen = {k: v.clone() for k, v in model.state_dict().items()
                        if not v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, shadow in self._shadow.items():
            shadow.mul_(self.decay).add_(state[k].detach().to(torch.float32),
                                         alpha=1.0 - self.decay)
        for k in self._frozen:
            self._frozen[k] = state[k].clone()
        self.steps += 1

    def state_dict_for(self, model: torch.nn.Module) -> dict[str, Tensor]:
        """The bias-corrected weights, shaped to load into `model` strictly."""
        if self.steps == 0:
            raise RuntimeError("EMA has taken no update; there is nothing to materialise")
        correction = 1.0 - self.decay ** self.steps
        reference = model.state_dict()
        out = {k: (v / correction).to(reference[k].dtype)
               for k, v in self._shadow.items()}
        out.update({k: v.clone() for k, v in self._frozen.items()})
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "steps": self.steps,
                "shadow": {k: v.clone() for k, v in self._shadow.items()},
                "frozen": {k: v.clone() for k, v in self._frozen.items()}}

    def load_state_dict(self, blob: Mapping[str, Any]) -> None:
        self.decay = float(blob["decay"])
        self.steps = int(blob["steps"])
        self._shadow = {k: v.clone() for k, v in blob["shadow"].items()}
        self._frozen = {k: v.clone() for k, v in blob["frozen"].items()}


# --------------------------------------------------------------------------- #
# 4. Checkpoints, resume, and the soup
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SamplerState:
    """Where in the draw a run is. 🔴 The part a resume silently gets wrong.

    There is no hidden generator to serialise -- `Sampler.epoch_specs` is a pure
    function of `(i, epoch, seed)` and `bucket_batches` a pure function of
    `(durations, seed)` -- so the whole sampler state is these six numbers. That
    is the *reason* the sampler was built stateless (docs/pipelines/02 §6), and
    it is why forgetting them is so easy: nothing crashes, the loss curve looks
    fine, and the run trains on a corpus it never reports.

    ``pass_index`` indexes the flat ``(branch group, epoch)`` schedule, so it
    doubles as the epoch key for the draw: every S1 branch pass sees its own
    epoch of specs rather than four replays of epoch 0.
    """

    pass_index: int
    epoch_seed: int
    n_specs: int
    batch_seed: int
    batch_index: int

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TrainCheckpoint:
    """A resumable run, on disk. Rebuilt through `models.model`'s own contract.

    ⚠️ Carries `config` for the same reason `models.model.save_checkpoint` does:
    the model is rebuilt from the stored config before a **strict** load, so a
    mismatch raises instead of producing a well-shaped model full of noise.
    """

    path: Path
    stage: str
    global_step: int
    sampler: SamplerState


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_train_checkpoint(path: Path | str, *, model: DeepVoiceNet,
                          optimizer: torch.optim.Optimizer | None,
                          ema: EMA | None, stage: str, global_step: int,
                          sampler: SamplerState, train_cfg: TrainConfig,
                          scaler: "torch.amp.GradScaler | None" = None,
                          extra: Mapping[str, Any] | None = None) -> TrainCheckpoint:
    """Weights, optimizer, EMA, **sampler state** and the torch RNG state.

    🔴 The last two are the ones that get dropped. Dropout draws from the global
    torch generator, so without its state a resumed run diverges from an
    uninterrupted one on the very first step -- and without `sampler` it diverges
    on the very first *batch*, which is worse because the loss curve does not
    show it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": dump_config(model.cfg),
        "train_config": dump_config(train_cfg),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        # ⚠️ The fp16 loss scale is state too: it adapts over the run, so a resume
        # that rebuilds it at the default replays the scaler's warm-up and takes
        # different steps for the first few hundred batches.
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "stage": stage,
        "global_step": int(global_step),
        "sampler": sampler.as_dict(),
        "rng": _rng_state(),
        "extra": dict(extra or {}),
    }, path)
    return TrainCheckpoint(path, stage, int(global_step), sampler)


def load_train_checkpoint(path: Path | str, map_location="cpu") -> dict[str, Any]:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing = [k for k in ("state_dict", "sampler", "rng", "stage") if k not in blob]
    if missing:
        raise ValueError(
            f"{path}: not a training checkpoint (missing {missing}). A checkpoint "
            "without `sampler` cannot resume the draw and a resumed run would "
            "quietly train on a different corpus")
    blob["sampler"] = SamplerState(**blob["sampler"])
    return blob


def checkpoint_soup(paths: Sequence[Path | str]) -> dict[str, Tensor]:
    """Uniform average of several checkpoints' weights.

    ★ Free ensembling at zero inference cost (docs/architecture/05 §3), across
    **epochs and seeds** -- which is why this takes a list of files rather than
    an in-run buffer: the across-seed soup is assembled from separate runs.

    🔴 Refuses a mismatched set instead of averaging what it can. A soup of two
    architectures, or of one model with a differently-shaped head, is not a
    worse model -- it is a `load_state_dict` failure deferred to whoever ships
    it, or worse, a silent partial average.

    ⚠️ Non-float entries are taken from the first checkpoint rather than
    averaged, for the reason `EMA` gives.
    """
    if len(paths) < 2:
        raise ValueError(
            f"a soup of {len(paths)} checkpoint(s) is that checkpoint; pass two or more")
    states, configs = [], []
    for p in paths:
        blob = torch.load(Path(p), map_location="cpu", weights_only=False)
        if "state_dict" not in blob:
            raise ValueError(f"{p}: not a checkpoint (no 'state_dict')")
        states.append(blob["state_dict"])
        configs.append(blob.get("config"))

    reference = states[0]
    for p, state in zip(paths[1:], states[1:]):
        if set(state) != set(reference):
            diff = sorted(set(state) ^ set(reference))[:5]
            raise ValueError(f"{p}: key set differs from {paths[0]} (e.g. {diff})")
        for k in reference:
            if state[k].shape != reference[k].shape:
                raise ValueError(
                    f"{p}: {k} is {tuple(state[k].shape)}, "
                    f"{paths[0]} has {tuple(reference[k].shape)}")
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError(
            "refusing to soup checkpoints built from different configs: the "
            "average of two architectures is not a model")

    out: dict[str, Tensor] = {}
    for k, v in reference.items():
        if not v.is_floating_point():
            out[k] = v.clone()
            continue
        acc = torch.zeros_like(v, dtype=torch.float64)
        for state in states:
            acc += state[k].to(torch.float64)
        out[k] = (acc / len(states)).to(v.dtype)
    return out


# --------------------------------------------------------------------------- #
# 5. The loop
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoopConfig:
    """Run-level knobs. Everything model- or objective-shaped lives in the configs.

    🔴 **Every field here is read by this module**, and
    `tests/test_loop.py::test_no_loop_config_field_is_silently_ignored` enforces
    it. A knob that validates and then does nothing is worse than a missing
    knob: it makes an ablation report a difference it never tested
    (`models.model.CONSUMED_ELSEWHERE` is the same guard one layer down).
    """

    out_dir: Path = Path("runs/dev")
    n_buckets: int = 4
    #: 0 disables the EMA entirely. ⚠️ Anything EMA- or checkpoint-related must
    #: skip Replay speed and start at Medium: Replay systematically favours ideas
    #: that help early in training (docs/validation/03 §2).
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    device: str = "cpu"
    #: 0 = checkpoint at pass boundaries only. Any other value also checkpoints
    #: every N optimizer steps, which is what makes a mid-epoch resume testable.
    checkpoint_every: int = 0
    #: Replay speed and the tests. ⚠️ A run with `max_steps` set is a truncated
    #: run -- `StageResult.truncated` says so, and it is not quotable.
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {self.n_buckets}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")


@dataclass
class StageResult:
    """What one stage did. Raw material for the experiment ledger."""

    stage: str
    plan: StagePlan
    steps: int
    passes_done: int
    loss_history: list[dict[str, float]] = field(default_factory=list)
    checkpoints: list[TrainCheckpoint] = field(default_factory=list)
    ema: EMA | None = None
    truncated: bool = False


def _schedule(plan: StagePlan, epochs: int) -> list[tuple[int, int]]:
    """The flat ``(group index, epoch within the group)`` pass list.

    ⚠️ Flat, so `pass_index` alone locates a resume *and* keys the draw. Under S1
    that means five branch passes of `epochs` epochs each, every one drawing its
    own corpus -- rather than the same epoch-0 corpus five times, which would
    make the branches' training sets identical and their comparison a coincidence.
    """
    return [(g, e) for g in range(len(plan.branch_groups)) for e in range(epochs)]


def _render_batch(specs: Sequence[SampleSpec], indices: Sequence[int],
                  index: ManifestIndex, cfg: RenderConfig) -> dict[str, Any]:
    return collate([render(specs[i], index, cfg) for i in indices])


def train_stage(model: DeepVoiceNet, dataset: SpecDataset, *,
                train_cfg: TrainConfig, loop_cfg: LoopConfig | None = None,
                stage: str | None = None,
                resume_from: Path | str | None = None) -> StageResult:
    """Train one stage. Resumable, EMA'd, and checkpointed with its draw state.

    ``dataset`` must be a **training** `SpecDataset` (`from_sampler`): the loop
    calls `set_epoch`, and a frozen eval set refuses that on purpose.

    ⚠️ Rendering happens inline rather than through a `DataLoader` with workers.
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
    # ⚠️ fp16 only. bf16 has fp32's exponent range, so it needs no scaler, and an
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
        # 🔴 A resume that redraws under a different key is a different run under
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

    for pass_index in range(start.pass_index, len(passes)):
        g, _ = passes[pass_index]
        group = plan.branch_groups[g]
        if g != group_index:
            group_index = g
            optimizer = torch.optim.AdamW(
                trainable_parameters(model, plan, group),
                lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                optimizer_state = None
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
                scaler_state = None
        loss_cfg_model = _stage_loss_config(model.cfg, group)

        # 🔴 `pass_index` is the epoch key, so every pass draws its own corpus.
        dataset.set_epoch(pass_index)
        specs = codec_variant_specs(dataset.specs, plan.codec_variants)
        batch_seed = start.batch_seed if pass_index == start.pass_index \
            else train_cfg.seed + pass_index
        batches = bucket_batches(spec_durations(specs), train_cfg.batch_size,
                                 n_buckets=loop_cfg.n_buckets, seed=batch_seed)
        first_batch = start.batch_index if pass_index == start.pass_index else 0

        model.train()
        parts_acc: list[dict[str, float]] = []
        for batch_index in range(first_batch, len(batches)):
            state = SamplerState(pass_index, start.epoch_seed, start.n_specs,
                                 batch_seed, batch_index)
            if loop_cfg.max_steps is not None and result.steps >= loop_cfg.max_steps:
                result.truncated = True
                _checkpoint(result, model, optimizer, ema, stage, state, train_cfg,
                            loop_cfg, scaler=scaler, tag="truncated")
                return result

            batch = _render_batch(specs, batches[batch_index], dataset.index, dataset.cfg)
            # ⚠️ float32 in, and it stays float32: the model casts under autocast.
            wav = prepare_waveform(batch["wav"].to(device), model.cfg.audio)
            lengths = batch["lengths"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            with autocast_for(train_cfg.precision, device):
                out = model(wav, lengths)
                total, parts = multitask_loss(out, targets, loss_cfg_model,
                                              train_cfg.loss)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if loop_cfg.grad_clip:
                # 🔴 Unscale first. Clipping a *scaled* gradient clips at a
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

            if loop_cfg.checkpoint_every and result.steps % loop_cfg.checkpoint_every == 0:
                _checkpoint(result, model, optimizer, ema, stage,
                            SamplerState(pass_index, start.epoch_seed,
                                         start.n_specs, batch_seed, batch_index + 1),
                            train_cfg, loop_cfg, tag=f"step{result.steps}")

        result.passes_done += 1
        result.loss_history.append(_mean_parts(parts_acc, stage, pass_index, group))
        _checkpoint(result, model, optimizer, ema, stage,
                    SamplerState(pass_index + 1, start.epoch_seed, start.n_specs,
                                 train_cfg.seed + pass_index + 1, 0),
                    train_cfg, loop_cfg, scaler=scaler, tag=f"pass{pass_index}")
    return result


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
                train_cfg, loop_cfg, *, scaler=None, tag: str) -> None:
    ck = save_train_checkpoint(
        Path(loop_cfg.out_dir) / f"{stage}-{tag}.pt", model=model,
        optimizer=optimizer, ema=ema, stage=stage, global_step=result.steps,
        sampler=state, train_cfg=train_cfg, scaler=scaler)
    result.checkpoints.append(ck)


def run_schedule(model: DeepVoiceNet, dataset: SpecDataset, *,
                 train_cfg: TrainConfig, loop_cfg: LoopConfig | None = None,
                 stages: Sequence[str] = STAGES) -> list[StageResult]:
    """S1 -> S2 -> S3 on one model, in order, carrying the weights forward.

    🔷 ``codec_aware`` last and never skipped when time is short: it is the
    **best-evidenced stage in the recipe** (★ ArtifactNet hard-negative FPR
    98.7% -> 8.0%) and it is a schedule rather than an architecture, so it is
    adopted regardless of how the frontend question resolves
    (docs/training/04 §3). ⚠️ ``joint`` is kept on different grounds -- its EER
    delta is 0.5 pts, below our local resolution -- so it is cheap, not measured.
    """
    for s in stages:
        stage_plan(s, model.cfg)              # refuse rank_polish before any work
    return [train_stage(model, dataset, train_cfg=train_cfg, loop_cfg=loop_cfg,
                        stage=s) for s in stages]


# --------------------------------------------------------------------------- #
# 6. Validation -- docs/validation/02
# --------------------------------------------------------------------------- #

def generator_key(spec: SampleSpec, index: ManifestIndex) -> str:
    """The per-sample **generator** label the per-family breakdown slices on.

    A composed sample has no single provenance, so this is a rule and it is
    written down: the key is the artifact family of the sample's **fake**
    components (joined, for cell 8, which has two), or ``real:<source>`` when no
    component is generated -- a real component has no `artifact_family` at all
    (`training.manifest`: real pools carry none).

    ⚠️ Both sides are therefore label-determining, which is exactly the case
    `metrics.breakdown.by(..., contrast="auto")` handles by scoring against a
    shared contrast pool. Do not "fix" that by giving REAL rows a fake family.
    """
    fake, real = [], []
    for comp in spec.components:
        row = index[comp.file_id]
        family = row.get("artifact_family")
        source = row.get("source_name")
        if family is not None and not (isinstance(family, float) and math.isnan(family)):
            fake.append(str(family))
        elif source is not None and not (isinstance(source, float) and math.isnan(source)):
            real.append(str(source))
    if fake:
        return "+".join(sorted(set(fake)))
    return "real:" + "+".join(sorted(set(real))) if real else "unknown"


def _pair_id(spec: SampleSpec, index: ManifestIndex) -> Any:
    for comp in spec.components:
        pid = index[comp.file_id].get("pair_id")
        if pid is not None and not (isinstance(pid, float) and math.isnan(pid)):
            return str(pid)
    return None


def predict(model: DeepVoiceNet, dataset: SpecDataset, *,
            batch_size: int = 8, device: str | torch.device = "cpu",
            precision: str = "fp32") -> dict[str, np.ndarray]:
    """The five submission columns over a frozen eval set, in dataset order.

    ⚠️ `eval_batches` -- in order, unbucketed, nothing dropped. Bucketing the
    eval set would destroy the duration-vs-score check on the REAL class that
    docs/pipelines/04 §3 requires be measured on unbucketed batches.

    🔴 ``precision`` defaults to **fp32 and should stay there**, even though
    inference ships fp16. The submitted number is a probability and bf16 carries
    8 mantissa bits, so squashing a bf16 logit **ties files together** — the
    mechanism behind EER 0.0950 -> 0.3017 (docs/validation/02 §7).

    ⚠️ Measured, and 🔴 **the effect is size-dependent**, which is how it would
    have escaped a small fixture. Unit-scale logits, `n_unique` against VG5's
    `> 0.5 n` gate:

    | n | bf16 | fp16 | fp32 |
    |---|---|---|---|
    | 400 (a test fixture) | 241 PASS | 370 PASS | 400 PASS |
    | **1,200 (the VAL floor)** | **399 FAIL** | 996 PASS | 1,200 PASS |

    So a bf16 evaluation passes VG5 on any fixture small enough to be convenient
    and fails at the size we actually validate on. `tests/test_loop.py` asserts
    it at 1,200, not at the fixture size.
    """
    device = torch.device(device)
    model.to(device).eval()
    columns: dict[str, list[np.ndarray]] = {c: [] for c in PREDICTION_COLUMNS}
    with torch.no_grad():
        for indices in eval_batches(dataset, batch_size):
            batch = _render_batch(dataset.specs, indices, dataset.index, dataset.cfg)
            wav = prepare_waveform(batch["wav"].to(device), model.cfg.audio)
            with autocast_for(precision, device):
                out = model(wav, batch["lengths"].to(device))
                probs = model.submission_probs(out)
            for c in PREDICTION_COLUMNS:
                columns[c].append(probs[c].detach().double().cpu().numpy())
    return {c: np.concatenate(v) for c, v in columns.items()}


def prediction_frame(specs: Sequence[SampleSpec], preds: Mapping[str, np.ndarray],
                     index: ManifestIndex, *, fold: int | None = None) -> pd.DataFrame:
    """Ground truth + predictions, one row per eval sample, ready for `metrics/`.

    🔴 The truth columns come from `spec.labels`, i.e. from the **cell**, and
    nowhere else -- the same derivation the metric harness uses. ⚠️ An absent
    component's fake label is written as `0`, not `None`, per `training.spec`:
    `file_fake_label` survives `None` only incidentally, and the masked pools
    drop those rows anyway.
    """
    rows = []
    for spec in specs:
        labels = spec.labels
        rows.append({
            "file_id": f"s{spec.sample_id:08d}",
            "cell": spec.cell,
            "fold": fold,
            "artifact_family": generator_key(spec, index),
            "pair_id": _pair_id(spec, index),
            "duration_s": float(spec.duration_s),
            "voice_present": int(labels["voice_present"]),
            "music_present": int(labels["music_present"]),
            "voice_fake": int(labels["voice_fake"] or 0),
            "music_fake": int(labels["music_fake"] or 0),
            "file_fake": int(labels["file_fake"]),
        })
    frame = pd.DataFrame(rows)
    n = len(frame)
    for c in PREDICTION_COLUMNS:
        col = np.asarray(preds[c], dtype=np.float64)
        if col.shape != (n,):
            raise ValueError(f"{c}: got {col.shape}, expected ({n},) -- one row per spec")
        frame[c] = col
    return frame


@dataclass(frozen=True)
class ValidationReport:
    """One fold's numbers. 🔴 Pooled *and* sliced, because pooled alone is not a result.

    docs/validation/02 §1 puts `per_cell_eer` and `per_family_eer` in Tier 3 and
    the pooled Score in Tier 2, and a good pooled EER routinely hides a collapsed
    cell -- cells 6 and 7 are the entire reason the competition has two fake
    heads. The breakdowns are therefore fields of the report rather than
    something a caller may forget to ask for, and `__str__` prints them.
    """

    fold: int | None
    metrics: MetricSet
    per_cell: pd.DataFrame
    per_generator: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def worst_cell_eer(self) -> float:
        usable = self.per_cell[(~self.per_cell["thin"]) & self.per_cell["eer"].notna()]
        return float(usable["eer"].max()) if len(usable) else float("nan")

    def __str__(self) -> str:
        m = self.metrics
        head = (f"fold {self.fold}: score {m.score:.4f} "
                f"(file {m.eer_file:.4f} / voice {m.eer_voice:.4f} / "
                f"music {m.eer_music:.4f} / vp {m.auc_vp:.4f} / mp {m.auc_mp:.4f}) "
                f"over n_file={m.n_file} n_voice={m.n_voice} n_music={m.n_music}")
        cells = "\n".join(
            f"  cell {int(r.value):>2}  eer {r.eer:.4f}  [{r.contrast}] n={r.n_slice}"
            + ("  ⚠️ thin" if r.thin else "")
            for r in self.per_cell[self.per_cell["head"] == "file"].itertuples())
        gens = "\n".join(
            f"  {str(r.value):<28} eer {r.eer:.4f}  [{r.contrast}] n={r.n_slice}"
            + ("  ⚠️ thin" if r.thin else "")
            for r in self.per_generator[self.per_generator["head"] == "file"].itertuples())
        return f"{head}\nper cell (file head):\n{cells}\nper generator (file head):\n{gens}"


def evaluate(model: DeepVoiceNet, dataset: SpecDataset, *,
             batch_size: int = 8, device: str | torch.device = "cpu",
             precision: str = "fp32", fold: int | None = None) -> ValidationReport:
    """Score one fold. Calls `metrics/`; computes nothing itself.

    🔴 `dacon_score` raises on an empty or single-class masked pool rather than
    returning 0.5. That propagates: a fold whose music pool has one class has no
    music EER, and averaging a plausible-looking 0.5 into the headline is exactly
    the failure `metrics.dacon.EmptyPoolError` exists to prevent.
    """
    if not dataset.is_frozen:
        raise ValueError(
            "evaluate needs the frozen eval set (SpecDataset.frozen): a dataset "
            "that can redraw makes the validation curve measure a different set "
            "every epoch, which no downstream assertion can see")
    preds = predict(model, dataset, batch_size=batch_size, device=device,
                    precision=precision)
    frame = prediction_frame(dataset.specs, preds, dataset.index,
                             fold=dataset.fold if fold is None else fold)
    return ValidationReport(
        fold=fold if fold is not None else dataset.fold,
        metrics=dacon_score(frame),
        per_cell=breakdown_table(frame, keys=("cell",)),
        per_generator=breakdown_table(frame, keys=("artifact_family",)),
        predictions=frame)


# --------------------------------------------------------------------------- #
# 7. Leak tripwires -- docs/architecture/08 §5, docs/pipelines/05 §6
# --------------------------------------------------------------------------- #

#: If local CV shows better than these on an unseen-generator split, suspect a
#: leak rather than success. Published cross-generator music detection is 46.4%
#: EER and ASVspoof 5's best voice system is ~4% (docs/survey/10).
MUSIC_UNSEEN_FLOOR = 0.03
VOICE_UNSEEN_FLOOR = 0.01


def measured_split_kind(train_specs: Sequence[SampleSpec],
                        val_specs: Sequence[SampleSpec],
                        index: ManifestIndex) -> tuple[str, str]:
    """Is VAL generator-disjoint from TRAIN? **Measured, not declared.**

    Returns ``(kind, detail)`` with ``kind`` in
    ``{"generator_disjoint", "generator_overlapping", "undecidable"}``.

    🔴 The alternative -- a `split_kind=` argument the caller passes -- would let
    the tripwires be switched off by the same mistake they exist to catch: a
    builder that thought it had a family-disjoint split is precisely the one
    whose 0.5% music EER needs explaining. This compares the realised generator
    sets of the two drawn streams, in the style `training.registries` uses to
    measure time invariance instead of trusting a declaration field.
    """
    def families(specs):
        out = set()
        for spec in specs:
            key = generator_key(spec, index)
            if not key.startswith("real:") and key != "unknown":
                out.update(key.split("+"))
        return out

    tr, va = families(train_specs), families(val_specs)
    if not tr or not va:
        return "undecidable", (f"TRAIN has {len(tr)} and VAL {len(va)} realised "
                               "generator families; disjointness is vacuous")
    shared = sorted(tr & va)
    if shared:
        return "generator_overlapping", (
            f"{len(shared)}/{len(va)} VAL generator famil(y/ies) also appear in "
            f"TRAIN: {shared[:5]}")
    return "generator_disjoint", (f"{len(va)} VAL generator famil(y/ies), none of "
                                  f"TRAIN's {len(tr)}")


def leak_tripwires(metrics: MetricSet, split_kind: str, detail: str = "") -> AuditReport:
    """🔴 Numbers so good they are evidence of a leak. Fails the run, loudly.

    | Head | Suspicious if | Because |
    |---|---|---|
    | music fake, unseen generator | **< 3% EER** | published cross-generator is 46.4% |
    | voice fake, unseen generator | **< 1% EER** | ASVspoof 5's best is ~4% |
    | any head | perfect separation on a random split | re-split by generator |

    ⚠️ Takes the **`MetricSet` the official harness already produced**, not a
    prediction frame: re-deriving these EERs here would be a second EER
    implementation in the one repo that forbids them, and the tripwire would
    then be able to disagree with the number it is guarding.

    ⚠️ Which rows run depends on what `measured_split_kind` found, and the others
    **SKIP** -- they never pass. A tripwire that reports PASS on a split it
    cannot speak about is worse than no tripwire, because the run then carries a
    green gate it did not earn (`AuditReport.SKIP`, `training.audit`).
    """
    r: dict[str, tuple[bool, str]] = {}
    disjoint = split_kind == "generator_disjoint"

    for head, value, floor, key in (
            ("music", metrics.eer_music, MUSIC_UNSEEN_FLOOR, "L1_music_unseen_generator"),
            ("voice", metrics.eer_voice, VOICE_UNSEEN_FLOOR, "L2_voice_unseen_generator")):
        if not disjoint:
            r[key] = (True, AuditReport.SKIP + f"split is {split_kind}: {detail}")
            continue
        r[key] = (
            not (np.isfinite(value) and value < floor),
            f"{head} EER {value:.4f} on an unseen-generator split, floor {floor:.2f} "
            f"-- below it, suspect a leak rather than success ({detail})")

    # ⚠️ EER 0 *and* AUC 1: the presence heads are AUCs, and a presence head that
    # separates perfectly is the same finding (docs/validation/03 §7 stops
    # investing there, which is a different decision from trusting the number).
    separations = {"eer_file": metrics.eer_file <= 0.0,
                   "eer_voice": metrics.eer_voice <= 0.0,
                   "eer_music": metrics.eer_music <= 0.0,
                   "auc_vp": metrics.auc_vp >= 1.0,
                   "auc_mp": metrics.auc_mp >= 1.0}
    perfect = sorted(k for k, v in separations.items() if v)
    if disjoint:
        r["L3_perfect_separation"] = (
            True, AuditReport.SKIP + "only meaningful on a split that is not "
                                     "generator-disjoint; L1/L2 cover this one")
    else:
        r["L3_perfect_separation"] = (
            not perfect,
            (f"head(s) perfectly separated on a {split_kind} split: {perfect} "
             f"-- re-split by generator ({detail})") if perfect else
            (f"no head separates perfectly on a {split_kind} split "
             f"(worst-case margin: file EER {metrics.eer_file:.4f}, "
             f"AUC_vp {metrics.auc_vp:.4f})"))
    return AuditReport(r)


# --------------------------------------------------------------------------- #
# 8. The gates -- docs/validation/04
# --------------------------------------------------------------------------- #

def output_sanity(preds: pd.DataFrame, reference_ids: Sequence[str] | None = None
                  ) -> AuditReport:
    """**VG5** B1-B5: the five columns still carry ranking information.

    🔴 The guard is `n_unique > 0.5 n`, and it is not a formality. Saturating the
    top and bottom 80% of a column took EER 0.0950 -> **0.3017** with no other
    warning, and the usual fix -- rank-normalising the column -- is a cross-file
    statistic forbidden by rule 2.4. The way to satisfy it is float64 logits and
    a float64 squash with no rounding, which is what `models.outputs` does and
    what `LoopConfig.eval_precision` protects.
    """
    r: dict[str, tuple[bool, str]] = {}
    n = len(preds)
    bad = []
    for c in PREDICTION_COLUMNS:
        v = preds[c].to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or v.min() < 0.0 or v.max() > 1.0:
            bad.append(c)
    r["VG5_B1_finite_in_unit_interval"] = (
        not bad, f"{len(bad)} column(s) non-finite or outside [0, 1]: {bad}")

    uniques = {c: int(pd.Series(preds[c]).nunique()) for c in PREDICTION_COLUMNS}
    worst = min(uniques, key=uniques.get)
    r["VG5_B2_ranking_resolution"] = (
        uniques[worst] > 0.5 * n,
        f"worst column {worst!r} has {uniques[worst]} unique values over {n} files "
        f"(gate > {0.5 * n:.0f}); ties near the operating point took EER "
        f"0.0950 -> 0.3017")
    constant = [c for c, u in uniques.items() if u <= 1]
    r["VG5_B3_no_constant_column"] = (
        not constant, f"constant column(s): {constant}")

    if reference_ids is None:
        r["VG5_B4_id_set_matches"] = (
            True, AuditReport.SKIP + "no reference id set given")
    else:
        same = list(preds["file_id"]) == list(reference_ids)
        r["VG5_B4_id_set_matches"] = (
            same, f"{len(preds)} rows against {len(reference_ids)} reference ids, "
                  f"same order: {same}")
    r["VG5_B5_no_fallback_nan"] = (
        bool(preds[list(PREDICTION_COLUMNS)].notna().all().all()),
        "no NaN in any prediction column (the per-file fallback path must emit a "
        "finite value)")
    return AuditReport(r)


def run_gates(report: ValidationReport, *,
              folds: pd.DataFrame | None = None,
              eval_specs: Sequence[SampleSpec] | None = None,
              manifest: pd.DataFrame | None = None,
              slice_: str | None = None, fold: int | None = None,
              scheme_version: str | None = None,
              probe_log: Path | str | None = None,
              opened_probe: bool = False) -> AuditReport:
    """VG1-VG6 for one experiment. 🔴 No number is quotable without this.

    Wires the gates that **exist** and SKIPs the rest by name, so the run record
    says which of the six actually ran:

    | Gate | Here |
    |---|---|
    | VG1 A1-A7, A10 | `training.folds.check_split_integrity` |
    | VG1 A8/A9 | `training.audit.audit_specs(..., eval_floors=True)` |
    | VG1 A1-A6 at draw time | the same audit's **I5**, which needs `slice_`/`fold` |
    | VG2 | the same audit's **I1b**, E-S2 at spec level |
    | VG3 adversarial validation | ⚠️ **SKIP** -- not implemented |
    | VG4 | `metrics.breakdown.t3_gap` |
    | VG5 | `output_sanity` |
    | VG6 | the `probe_openings.log` line count |

    ⚠️ VG3 is the honest gap. It needs a TRAIN-vs-VAL classifier over the
    metadata features VG2 uses, which is real work and would land as a green
    stub if it were faked here. It reports SKIP, and `RunReport.quotable`
    counts SKIPs so the shortfall is visible in the ledger rather than in a
    docstring.
    """
    r: dict[str, tuple[bool, str]] = {}

    if folds is None:
        r["VG1_split_integrity"] = (True, AuditReport.SKIP + "no folds table given")
    else:
        for k, v in check_split_integrity(folds, scheme_version).results.items():
            r[f"VG1_{k}"] = v

    if eval_specs is None:
        for k in ("VG1_A8A9_eval_size_floors", "VG1_I5_split_safety",
                  "VG2_shortcut_audit"):
            r[k] = (True, AuditReport.SKIP + "no eval specs given")
    else:
        # ⚠️ `slice_`/`fold` are what make **I5** run rather than SKIP -- the
        # draw-time form of VG1 A1-A6, which re-derives the allowed `file_id`
        # set from the manifest and reports any drawn component outside it.
        spec_report = audit_specs(list(eval_specs), manifest=manifest,
                                  slice_=slice_, fold=fold, eval_floors=True)
        r["VG1_A8A9_eval_size_floors"] = spec_report.results["I7_eval_size_floors"]
        r["VG1_I5_split_safety"] = spec_report.results["I5_split_safety"]
        r["VG2_shortcut_audit"] = spec_report.results["I1b_metadata_shortcut_auc"]

    r["VG3_adversarial_validation"] = (
        True, AuditReport.SKIP + "not implemented: a TRAIN-vs-VAL classifier over "
        "the VG2 metadata features. ⚠️ A low AUC would be weak evidence of "
        "absence anyway (docs/validation/04 VG3); a green stub would be none")

    gap = t3_gap(report.predictions, head="voice")
    if not np.isfinite(gap["t3_gap"]):
        r["VG4_corpus_identity"] = (
            True, AuditReport.SKIP + f"no scorable T3 pair pool "
                                     f"(n_pairs={gap['n_pairs']}): {gap['note']}")
    else:
        r["VG4_corpus_identity"] = (
            bool(gap["passes_vg4"]),
            f"voice T3-pair EER {gap['t3_pair_eer']:.4f} - pooled "
            f"{gap['pooled_eer']:.4f} = {gap['t3_gap']:+.4f}, gate <= 0.10 "
            f"over {gap['n_pairs']} paired rows")

    r.update(output_sanity(report.predictions).results)

    if not opened_probe:
        r["VG6_probe_budget"] = (
            True, AuditReport.SKIP + "this run did not score against PROBE")
    elif probe_log is None:
        r["VG6_probe_budget"] = (
            False, "PROBE was opened with no probe_openings.log to record it in; "
                   "the budget is enforced mechanically, not by discipline")
    else:
        lines = [ln for ln in Path(probe_log).read_text().splitlines() if ln.strip()]
        r["VG6_probe_budget"] = (
            len(lines) <= 3,
            f"{len(lines)} PROBE opening(s) recorded in {probe_log}, budget 3")
    return AuditReport(r)


# --------------------------------------------------------------------------- #
# 9. Folds, and the run record
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FoldResult:
    """One fold, end to end: what it scored and whether it is allowed to count."""

    fold: int | None
    validation: ValidationReport
    gates: AuditReport
    tripwires: AuditReport

    @property
    def ok(self) -> bool:
        return self.gates.ok and self.tripwires.ok


@dataclass(frozen=True)
class RunReport:
    """The ledger row. 🔴 Mean of per-fold metrics, never pooled OOF.

    ⚠️ **A single-fold run is first-class**, not a degraded mode: the full 5-fold
    sweep is often unaffordable and Replay speed is fold 0 only by definition
    (docs/validation/03 §2). What a single fold does *not* give is `Score_sd`,
    which is a P4 input and the ★ E5 tiebreaker -- so `fold_mean` returns 0.0
    there and this report carries a caveat saying that 0.0 is an absence, not a
    measurement. Reading it as "perfectly stable" is the failure mode.
    """

    aggregate: AggregateMetrics
    folds: tuple[FoldResult, ...]
    caveats: tuple[str, ...] = ()

    @property
    def score_mean(self) -> float:
        return self.aggregate.mean.score

    @property
    def score_sd(self) -> float:
        return self.aggregate.score_sd

    @property
    def sd_is_a_measurement(self) -> bool:
        return self.aggregate.n_folds > 1

    @property
    def quotable(self) -> bool:
        """Every gate green and no tripwire fired. 🔴 A red gate voids the result.

        ⚠️ SKIPs do not block -- they are recorded and printed. VG3 is skipped on
        every run today, so treating a SKIP as a failure would make nothing
        quotable and the distinction would stop being read.
        """
        return all(f.ok for f in self.folds)

    def skipped_gates(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in self.folds:
            out.update(f.gates.skipped)
            out.update(f.tripwires.skipped)
        return out

    def as_ledger_row(self) -> dict[str, Any]:
        """The subset of docs/validation/03 §5's schema this module can fill."""
        m = self.aggregate.mean
        return {
            "score_mean": m.score, "score_sd": self.score_sd,
            "ads": m.ads, "cps": m.cps,
            "eer_file": m.eer_file, "eer_voice": m.eer_voice,
            "eer_music": m.eer_music, "auc_vp": m.auc_vp, "auc_mp": m.auc_mp,
            "n_folds": self.aggregate.n_folds,
            "excluded_folds": json.dumps(list(self.aggregate.excluded_folds)),
            "per_fold_score": json.dumps(
                {str(f.fold): f.validation.metrics.score for f in self.folds}),
            "worst_cell_eer": max(
                (f.validation.worst_cell_eer for f in self.folds), default=float("nan")),
            "quotable": self.quotable,
            "caveats": json.dumps(list(self.caveats)),
        }

    def __str__(self) -> str:
        lines = [f"Score {self.score_mean:.4f} ± {self.score_sd:.4f} "
                 f"over {self.aggregate.n_folds} fold(s) "
                 f"[{'quotable' if self.quotable else '🔴 NOT QUOTABLE'}]"]
        lines += [str(f.validation) for f in self.folds]
        for f in self.folds:
            for key, why in {**f.gates.failures, **f.tripwires.failures}.items():
                lines.append(f"  FAIL {key}: {why}")
        lines += [f"  SKIP {k}" for k in sorted(self.skipped_gates())]
        lines += [f"  ⚠️ {c}" for c in self.caveats]
        return "\n".join(lines)


def validate_fold(model: DeepVoiceNet, eval_dataset: SpecDataset, *,
                  train_specs: Sequence[SampleSpec],
                  fold: int | None = None, batch_size: int = 8,
                  device: str | torch.device = "cpu", precision: str = "fp32",
                  folds: pd.DataFrame | None = None,
                  manifest: pd.DataFrame | None = None,
                  scheme_version: str | None = None,
                  probe_log: Path | str | None = None,
                  opened_probe: bool = False) -> FoldResult:
    """Score one fold **with its gates and its tripwires**, as one object.

    🔴 ``train_specs`` is a required argument, and that is the point of this
    function existing at all. The tripwires need to know whether VAL is
    generator-disjoint from TRAIN, that question is **measured** rather than
    declared (`measured_split_kind`), and measuring it needs both streams. Making
    it required means a `FoldResult` cannot be produced without the tripwires
    having run or having said, by name, why they could not.

    ⚠️ Everything after `precision` is a gate input, and each one that is left
    out makes its gate report **SKIP** rather than PASS -- visible in
    `RunReport.skipped_gates()` and printed by `RunReport.__str__`.
    """
    report = evaluate(model, eval_dataset, batch_size=batch_size, device=device,
                      precision=precision, fold=fold)
    kind, detail = measured_split_kind(train_specs, eval_dataset.specs,
                                       eval_dataset.index)
    return FoldResult(
        fold=report.fold,
        validation=report,
        gates=run_gates(report, folds=folds, eval_specs=eval_dataset.specs,
                        manifest=manifest, slice_=eval_dataset.slice_,
                        fold=eval_dataset.fold, scheme_version=scheme_version,
                        probe_log=probe_log, opened_probe=opened_probe),
        tripwires=leak_tripwires(report.metrics, kind, detail))


def aggregate_folds(results: Sequence[FoldResult],
                    min_pool: int | None = None) -> RunReport:
    """Mean of per-fold metrics. 🔴 Never a pooled OOF score.

    Each fold is scored by a *different model*, so their score scales differ and
    EER is computed on the merged ranking: concatenating raw OOF scores measured
    **0.1705 against a true 0.100** (docs/validation/02 §4). The aggregation is
    `metrics.aggregate.fold_mean`, which also records any fold excluded for
    falling below the masked-pool size floor rather than dropping it silently.
    """
    if not results:
        raise ValueError("aggregate_folds: no folds given")
    agg = fold_mean([r.validation.metrics for r in results],
                    min_pool=min_pool,
                    fold_ids=[r.fold for r in results])
    caveats: list[str] = []
    if agg.n_folds == 1:
        caveats.append(
            "single fold: Score_sd is 0.0 because there is nothing to vary, not "
            "because the run is stable. P4 (fold variance) and the ★ E5 "
            "tiebreaker cannot be evaluated from this run")
    if agg.excluded_folds:
        caveats.append(
            f"fold(s) {list(agg.excluded_folds)} fell below min_pool={min_pool} "
            "and were excluded from the mean (VG1 A8 should have caught it first)")
    for r in results:
        if not r.gates.ok:
            caveats.append(f"fold {r.fold}: {len(r.gates.failures)} gate(s) red — "
                           "the number is not quotable, comparable or promotable")
        if not r.tripwires.ok:
            caveats.append(f"fold {r.fold}: a leak tripwire fired — "
                           "suspect the split, not the model")
    return RunReport(agg, tuple(results), tuple(caveats))
