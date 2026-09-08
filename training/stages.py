"""The stage schedule: what S1 / S2 / S3 train, over what, and in what precision.

Split out of `training.loop` so the schedule can be read and reasoned about
without the trainer: everything here is a pure function of a `ModelConfig`, and
nothing in this module imports `training.loop` or `training.validate`.

Caveat: **S4 is dropped** (docs/training/04 §4). `stage='rank_polish'` raises
here rather than running S2 again under a different name: `models.config` still
accepts the value, and a stage that validates and then quietly does something
else is how an ablation reports a difference it never tested.
"""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from models.config import ModelConfig
from models.model import DeepVoiceNet
from training.spec import SampleSpec

__all__ = [
    "CODEC_VARIANTS", "STAGES", "StagePlan",
    "autocast_for", "codec_variant_specs", "stage_plan", "trainable_parameters",
]


# --------------------------------------------------------------------------- #
# 1. The stage schedule -- docs/training/04 §1
# --------------------------------------------------------------------------- #

#: The three stages that survive. `rank_polish` is deliberately absent: S4 was
#: dropped (docs/training/04 §4) after TFPARN's own ablation (primary source)
#: moved EER 12.91 -> 12.92 and its second justification landed below our ≈1 pt
#: resolution threshold. `models.config` still *accepts* the value because a
#: config file in the wild may carry it; `stage_plan` refuses it.
STAGES = ("independent", "joint", "codec_aware")

#: S3's 4-way codec menu (docs/training/04 §3). ArtifactNet P2->P3 (primary
#: source): hard-negative FPR 98.7% -> 8.0%, cross-codec drift -83%; the
#: best-evidenced stage in the recipe, so this is where the schedule's remaining
#: value is concentrated.
#:
#: Critical: the four are applied to **every** spec, not drawn per spec. That is
#: what keeps `normalize` label-independent by construction: a per-spec draw is a
#: transform parameter, and `container = mp3 if fake else wav` scored AUC 1.000
#: on the I1b metadata probe while every other invariant stayed green
#: (docs/pipelines/05 §1). Expanding uniformly makes the balance structural
#: rather than a property the sampler has to remember.
#:
#: Caveat: short of the full A-S3 menu for the reason `render.CODEC_CONTAINERS`
#: gives: AAC/OPUS/AMR-NB/GSM each need their encoder delay verified the way
#: MP3's is, and an uncancelled encoder delay moves the audio while
#: `frame_intervals` stay put. The 8 kHz telephone leg is included because
#: ASVspoof 5's hardest condition (primary source) is codec-10 (speex, 8 kHz,
#: low bitrate) and that is our telephone slice, named as the worst case by an
#: independent evaluation.
#:
#: Caveat: **`flac` is the candidate for replacement**, once one of AAC / OPUS /
#: AMR-NB / GSM has its encoder delay verified the way MP3's was. It is
#: lossless, so it exercises a different container and decode path but leaves
#: the signal untouched -- the least signal of the four, for a quarter of S3's
#: compute. That quarter is not cheap: S3 is the strongest-evidenced stage in
#: the recipe, so its budget is the one the schedule's remaining value sits in.
CODEC_VARIANTS: tuple[dict[str, Any], ...] = (
    {},                                              # as decoded
    {"container": "mp3", "bitrate": 64},             # 64 kbps cost MusicDET +37 EER pts
    {"container": "flac"},                           # the weak leg -- see below
    {"telephone_hz": 8000, "companding": "ulaw"},    # A-S4, the telephone slice
)


@dataclass(frozen=True)
class StagePlan:
    """What one stage trains, and over what.

    ``branch_groups`` is the whole difference between S1 and S2/S3: S1 is one
    group per branch (each branch alone, in sequence), S2/S3 are a single group
    holding every branch. Caveat: the component losses stay masked in both --
    masking mirrors the metric, not the schedule (docs/architecture/08 §2).
    """

    stage: str
    branch_groups: tuple[tuple[str, ...], ...]
    codec_variants: tuple[Mapping[str, Any], ...]
    train_frontends: bool
    #: Critical: frontends whose ``FrontendConfig.freeze`` disagrees with what
    #: this stage imposes. Empty on every shipped config; non-empty means a
    #: config field is not being honoured, and `caveats` says so out loud.
    freeze_overrides: tuple[str, ...] = ()

    @property
    def caveats(self) -> tuple[str, ...]:
        """What this plan does that the config does not say. Never silent.

        The house pattern is `training.folds.FoldPlan.caveats`: the thing a
        reader skips is the thing that has to travel with the result, so it is a
        field of the plan rather than a log line.
        """
        if not self.freeze_overrides:
            return ()
        names = ", ".join(self.freeze_overrides)
        return (
            f"stage {self.stage!r} freezes the frontend(s) {names}, whose config "
            f"says `freeze: false`. S1 is defined as 'each branch alone on frozen "
            f"frontends', so the override is deliberate -- but the field is not "
            f"being honoured in this stage and a run that reports 'trained with "
            f"an unfrozen encoder' would be wrong. It IS honoured in S2/S3.",)

    def __str__(self) -> str:
        groups = " | ".join("+".join(g) for g in self.branch_groups)
        head = (f"{self.stage}: groups [{groups}], "
                f"{len(self.codec_variants)}-way codec, "
                f"frontends {'trainable' if self.train_frontends else 'frozen'}")
        return "\n".join([head, *(f"  caveat: {c}" for c in self.caveats)])


def stage_plan(stage: str, cfg: ModelConfig) -> StagePlan:
    """The schedule for one stage. Refuses the dropped one.

    | Stage | Branches | Frontends | Codec |
    |---|---|---|---|
    | ``independent`` (S1) | one at a time | **frozen** | 1-way |
    | ``joint`` (S2)       | all together  | per config | 1-way |
    | ``codec_aware`` (S3) | all together  | per config | **4-way** |

    Critical: S1 **overrides** ``FrontendConfig.freeze`` rather than reading it,
    and that is a deliberate asymmetry: "independent, frozen frontends +
    adapters" is what the stage *is*, and reading the field instead would make
    S1 and S2 the same stage on both shipped configs (they already say
    ``freeze: true``), so an S1-vs-S2 comparison would measure nothing.

    Caveat: **the override is announced, not silent.** If a config ever says
    ``freeze: false``, S1 still freezes -- and `StagePlan.caveats` names the
    frontends whose field is not being honoured, `__str__` prints it, and
    `StageResult.caveats` carries it to the run report. Reported rather than
    raised: the field is honoured in S2 and S3, so this is a divergence to
    announce for one stage, not a config the schedule cannot run. A silent
    divergence between a config field and actual behaviour is the defect here;
    the override itself is the schedule.
    """
    branches = tuple(cfg.branches)
    if stage == "independent":
        overrides = tuple(name for name, fe in cfg.frontends.items() if not fe.freeze)
        return StagePlan(stage, tuple((b,) for b in branches), ({},), False,
                         freeze_overrides=overrides)
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

    Critical: uniform expansion, never a per-spec draw -- see ``CODEC_VARIANTS``.
    The resulting stream has ``P(normalize | label) = P(normalize)`` exactly,
    which `audit_specs`' **I1b** measures rather than assumes;
    `tests/test_stages.py` mutates the expansion to a label-conditional draw and
    watches I1b go red.

    Caveat: the variant is *merged into* whatever the spec already carried, so a
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

    Critical: selected by *module*, not by zeroing a loss term. Restricting the
    loss alone still lets weight decay and optimizer momentum move a branch
    nobody is training this pass, which is exactly the silent divergence S1
    exists to avoid -- and it would make "each branch alone" a claim rather than
    a fact. `tests/test_stages.py` asserts the inactive branches come out
    bitwise unchanged.
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

    Caveat: a restricted *config*, not a filtered *output* dict: `multitask_loss`
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

    Critical: the pipeline emits float32 and the *model* casts, under autocast,
    with fp32 master weights. Pre-casting the batch is how this repo produced
    NaN attention: GeM overflowed in fp16 above a feature scale of ~40
    (docs/pipelines/04 §4), and an fp16 batch also arrives at `bandpass`'s rFFT
    and at the loss in a precision neither was measured in.

    Caveat: `fp32` returns a null context rather than an autocast with float32 --
    the two are not the same thing, and the second one still routes ops through
    the autocast dispatcher.
    """
    if precision not in _TORCH_DTYPE:
        raise ValueError(f"precision must be one of {sorted(_TORCH_DTYPE)}, got {precision!r}")
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(torch.device(device).type, dtype=_TORCH_DTYPE[precision])
