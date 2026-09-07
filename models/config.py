"""Typed configuration for the model architecture.

Design constraints, all of which come from the submission contract
(docs/competition/02-submission.md) rather than from taste:

* **`pyyaml` only.** The eval server preinstalls `pyyaml==6.0.1`; Hydra/OmegaConf
  would spend the 10-minute install budget and add an offline failure mode. So a
  config is a plain dict on disk and a frozen dataclass in memory.
* **The config travels inside the checkpoint.** `script.py` rebuilds the model
  from the stored config before `load_state_dict`, which is what makes a strict
  load a real assertion rather than a coincidence.
* **Unknown keys are an error, not a warning.** A typo'd knob that silently does
  nothing is how an ablation ends up measuring the wrong thing.

Candidates A and B are two instances of the same schema, not two schemas:
A names one frontend, B names two and gives the file branch both. See
docs/architecture/03-candidates.md.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from metrics.dacon import PREDICTION_COLUMNS

__all__ = [
    "AdapterConfig", "AggregationConfig", "AudioConfig", "AuxConfig",
    "BranchConfig", "ConfigError", "DistillConfig", "FileHeadConfig",
    "FreqPoolConfig", "FrontendConfig", "LossConfig", "ModelConfig",
    "OutputConfig", "RuntimeConfig", "SEDHeadConfig", "SegmentationConfig",
    "TrainConfig",
    "load_model_config", "load_train_config", "dump_config",
]

#: Frontends we know how to build. `stub` is a randomly-initialised encoder with
#: no weights -- it is what lets every shape/mask/invariance test run before any
#: checkpoint is downloaded and before the corpus exists.
KNOWN_FRONTENDS = (
    "stub",
    "xlsr_300m", "xlsr_1b", "xlsr_2b", "wavlm_large", "w2v_bert2",  # speech
    "beats", "eat", "sslam", "mert", "panns",                        # general audio / music
)

#: Which ground-truth column masks a branch's loss. `None` means "every file".
#: These mirror the official metric: Voice EER is computed only over
#: voice-present files (docs/competition/03-evaluation.md).
MASK_KEYS = ("voice_present", "music_present")


class ConfigError(ValueError):
    """A config that would build a model we do not mean to build."""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000     # the test set is standardised to 16 kHz
    min_seconds: float = 4.0
    max_seconds: float = 60.0
    #: How stereo becomes model input. `mid_side` keeps inter-channel information,
    #: which may be signal -- or may be a shortcut if fake sources skew mono.
    #: Test it as a leak before treating it as a feature (09 B8).
    channels: str = "downmix"     # downmix | left | mid_side
    #: Optional band restriction, (low_hz, high_hz). Deliberately restricting the
    #: model to low-frequency subbands cut EER by up to 25% relative under codec
    #: conditions (D9), and our telephone slice lives there (09 B7). `null` = full band.
    band_hz: tuple[float, float] | None = None


@dataclass(frozen=True)
class AdapterConfig:
    """Parameter-efficient tuning on a frozen frontend (docs/architecture/06 §2).

    ⚠️ Defaults to `none` because nothing implements adapters yet. Defaulting to
    `lora` would make every bare FrontendConfig unbuildable, and -- worse -- would
    read as though adapters were the working default.
    """
    kind: str = "none"                                   # lora | conv | none
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass(frozen=True)
class FreqPoolConfig:
    """Pooling over the frequency axis.

    Only meaningful for patch-grid frontends (BEATs / EAT / SSLAM), which emit a
    (F', T') token grid. wav2vec2-family frontends emit (B, T, D) with no
    frequency axis and must set `kind: none`.
    """
    kind: str = "gem"          # gem | mean | max | none
    p_init: float = 3.0        # 3.0 = sharper than mean, softer than max
    learnable: bool = True     # lets the model settle G5 instead of us guessing
    #: How signed features are made positive before the power mean. GeM comes
    #: from image retrieval on post-ReLU (non-negative) features; SSL hidden
    #: states are signed, and roughly half of them are negative. ⚠️ `clamp`
    #: maps every negative entry to eps -- discarding ~50% of the distribution
    #: with exactly zero gradient -- and is kept only to reproduce that
    #: behaviour deliberately. `softplus` is monotone, positive and
    #: differentiable everywhere, so nothing is thrown away.
    rectifier: str = "softplus"   # softplus | clamp


@dataclass(frozen=True)
class FrontendConfig:
    name: str
    weights: str | None = None      # local dir; None => randomly initialised (tests)
    layers: int | None = None       # truncation depth; None = keep all
    freeze: bool = True
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    freq_pool: FreqPoolConfig = field(default_factory=lambda: FreqPoolConfig(kind="none"))
    output_dim: int = 1024          # required for `stub`; asserted against real weights
    fps: float = 50.0               # frames per second of audio; used for time alignment
    #: Height of the patch grid, for frontends that keep a frequency axis. BEATs /
    #: EAT / SSLAM tokenise a mel spectrogram into a (F', T') grid and the wrapper
    #: needs F' to reshape. Must be set exactly when freq_pool.kind != "none".
    n_freq: int | None = None


@dataclass(frozen=True)
class SEDHeadConfig:
    """One SED head (docs/architecture/04).

    🔴 `attention` matters more than it looks. The source notebook applies
    `tanh` to the attention logits before the softmax, which bounds them to
    [-1, 1] and caps the weight any single frame can receive at
    `1/(1 + (T-1)e^-2)` ~= 7.4/T -- so at T=3000 (a 60 s file under
    `whole_file`) the "attention" pooling is within 7.4x of a uniform mean, and
    `clip_logits` is functionally the mean pool that 04 §1 calls "the whole
    problem". `linear` removes the cap; `tanh` is kept only for parity with the
    source recipe and is documented as capped.
    """
    hidden: int = 512
    dropout_in: float = 0.25
    dropout_out: float = 0.5
    #: 🔴 1.0 = clip only. Committed in docs/training/02 §3: supervising the
    #: utterance and frame levels through ONE shared head measured 0.71-3.63 EER
    #: points WORSE than utterance-only (Zhang et al., ASVspoof 2021 Workshop,
    #: Table 5), and our head is exactly that configuration -- `clip_logits` is a
    #: pooled function of the `frame_logits` the frame loss also touches.
    #: ⚠️ Pending ablation T1; the transfer caveats are in docs/training/02 §3.
    #: ⚠️ At 1.0 the submitted score ignores `frame_max` entirely, which makes the
    #: rule-2.4 frame_max guards vacuous -- the tests force the blend on rather
    #: than inheriting this default. Do not "simplify" them back.
    clip_weight: float = 1.0        # blend of clip vs frame_max, applied in LOGIT space
    attention: str = "linear"       # linear | scaled_tanh | tanh


@dataclass(frozen=True)
class BranchConfig:
    """One output branch.

    `sources` names frontends, not input types -- every file passes through every
    branch (docs/architecture/03). A branch with two sources concatenates them,
    which requires resampling one onto the other's time base via `align_to`.
    """
    sources: tuple[str, ...]
    column: str                      # the submission column this branch produces
    masked_by: str | None = None     # voice_present | music_present | None
    align_to: str | None = None      # required when len(sources) > 1
    head: SEDHeadConfig = field(default_factory=SEDHeadConfig)


@dataclass(frozen=True)
class SegmentationConfig:
    """How the file is fed to the frontends (docs/architecture/04 §6, 09 B5)."""
    mode: str = "whole_file"         # whole_file | tiling
    window_seconds: float = 5.0
    hop_seconds: float = 5.0


@dataclass(frozen=True)
class AggregationConfig:
    """Cross-window pooling. Ignored when segmentation.mode == whole_file.

    `max` is deliberately not the default: it is stochastically larger for longer
    files, which is a duration bias inside a ranking that pools 4-60 s files
    (docs/architecture/04 §6.1).
    """
    kind: str = "topk_mean"          # max | mean | topk_mean | quantile | confidence_gated
    k: int = 3
    quantile: float = 0.9


@dataclass(frozen=True)
class FileHeadConfig:
    """How FILE_FAKE_PROB is formed (G3 -- no prior art; build all three)."""
    mode: str = "learned"            # learned | noisy_or | max


@dataclass(frozen=True)
class OutputConfig:
    """The submission-facing numerics.

    Saturation, not rounding, is the measured risk: saturating the operating
    point took EER 0.0950 -> 0.3017 (PROGRESS.md). Emit float64 and keep the
    decision region resolvable.
    """
    dtype: str = "float64"
    #: `softsign` is preferred: a float64 sigmoid reaches an exact 1.0 by z~=37
    #: and every file beyond that becomes an unbreakable tie, while softsign
    #: holds out to |z| ~ 1e16. Rank normalisation, the usual fix for ties, is
    #: forbidden by rule 2.4.
    squash: str = "softsign"         # softsign | sigmoid
    logit_scale: float = 1.0         # <1 widens the unsaturated region
    clamp_eps: float = 1e-12


@dataclass(frozen=True)
class RuntimeConfig:
    """Inference-time execution. Affects wall clock, and `precision` affects numerics.

    ⚠️ `compile` defaults off: compilation happens inside the 60-minute budget on
    first call, and the corrected margin means we do not need the speedup
    (docs/architecture/06 §4). Turn it on only if measurement says so.
    """
    precision: str = "fp16"       # fp32 | fp16 | bf16 -- the model's compute dtype
    compile: bool = False
    batch_size: int = 8           # ⚠️ must not change any output; asserted in tests
    num_workers: int = 5          # 6 vCPU on the eval server, leave one


@dataclass(frozen=True)
class DistillConfig:
    """Student-side wiring only; teachers live in TrainConfig (they never ship)."""
    enabled: bool = False
    embed_dim: int = 1024
    stop_gradient: bool = True       # docs/architecture/06 §3 -- and see 09 B9


@dataclass(frozen=True)
class AuxConfig:
    """Auxiliary heads discarded before packaging (candidate C-lite)."""
    separation_head: bool = False
    stft_bins: int = 513


@dataclass(frozen=True)
class ModelConfig:
    name: str
    frontends: dict[str, FrontendConfig]
    branches: dict[str, BranchConfig]
    seed: int = 0
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    file_head: FileHeadConfig = field(default_factory=FileHeadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)


@dataclass(frozen=True)
class LossConfig:
    """Per-head weights and the loss blend (docs/architecture/08 §2).

    ✅ `weights` is metric-proportional. 09 B11 is closed: it was rated "probably
    not resolvable individually -- set by argument", a targeted search found no
    study testing loss-weights against metric-weights, and the argument is in
    docs/training/02 §4.
    """
    #: 🔴 Metric-proportional, per docs/training/02 §4. The effective weights are
    #: File .45 / Music .27 / Voice .18 / presence .05 each; an earlier default of
    #: all-1.0 was inherited from PC-Mix, whose metric weighted its components
    #: equally and ours does not. No study tests loss-weights ∝ metric-weights, so
    #: this is set by argument (09 B11 predicted it would be); every adaptive
    #: alternative (GradNorm/PCGrad/DWA/uncertainty) has strong negative results
    #: against well-tuned constants.
    #: ⚠️ The weight in effect is `w_c / p_c`, not `w_c` -- `_masked_mean` divides
    #: by the present-count. Log both before tuning.
    weights: dict[str, float] = field(default_factory=lambda: {
        "voice": 0.18, "music": 0.27, "file": 0.45, "v_pres": 0.05, "m_pres": 0.05})
    ranking_weight: float = 0.0
    distill_weight: float = 1.0
    label_smoothing: float = 0.0

    # ⚠️ There is deliberately no `frame_weight` here. The clip-vs-frame_max blend
    # is `SEDHeadConfig.clip_weight`, and the loss reads that same field, so
    # training and inference cannot disagree about it. A separate loss-side knob
    # existed and was documented as the *frame-supervision* weight of 04 §4 --
    # a different quantity entirely -- so anyone tuning it per that section was
    # tuning the blend instead.
    #
    # ⚠️ There is also no `frame_resolutions_ms`. Multi-resolution frame
    # supervision (04 §4) is not implemented; the field was accepted, defaulted,
    # round-tripped and read nowhere. The plan lives in the doc, not in a config
    # field that does nothing.


@dataclass(frozen=True)
class TrainConfig:
    """Training-only settings. Never shipped, never stored in the checkpoint."""
    stage: str = "joint"             # independent | joint | codec_aware | rank_polish
    loss: LossConfig = field(default_factory=LossConfig)
    #: Frozen distillation teachers. A teacher needs a full spec (weights path,
    #: depth), not just a name -- and none of them ever enter the inference path.
    teachers: dict[str, FrontendConfig] = field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-5
    precision: str = "bf16"
    seed: int = 0


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #

def _hints(cls) -> dict[str, Any]:
    """Resolved type hints.

    `from __future__ import annotations` makes `dataclasses.fields(...).type` a
    *string*, so the naive `is_dataclass(f.type)` check silently never fires and
    nested mappings stay as dicts. Resolve them properly instead.
    """
    return get_type_hints(cls)


#: `X | None` yields `types.UnionType` on 3.10+, while `Optional[X]` yields
#: `typing.Union`. Both appear in this file, and missing one silently skips the
#: nested-dataclass and tuple conversions below.
_UNIONS = (Union, types.UnionType)


def _is_union(hint) -> bool:
    return get_origin(hint) in _UNIONS


def _nested_dataclass(hint) -> type | None:
    """The dataclass inside a hint, unwrapping `X | None`; else None."""
    if is_dataclass(hint):
        return hint
    if _is_union(hint):
        inner = [a for a in get_args(hint) if a is not type(None)]
        if len(inner) == 1 and is_dataclass(inner[0]):
            return inner[0]
    return None


def _is_tuple_hint(hint) -> bool:
    if get_origin(hint) is tuple:
        return True
    if _is_union(hint):
        return any(get_origin(a) is tuple for a in get_args(hint))
    return False


def _build(cls, value: Any, path: str):
    """Recursively build a frozen dataclass, rejecting unknown keys."""
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(known)}")
    hints = _hints(cls)
    kwargs = {}
    for key, raw in value.items():
        hint = hints[key]
        nested = _nested_dataclass(hint)
        if nested is not None and isinstance(raw, dict):
            kwargs[key] = _build(nested, raw, f"{path}.{key}")
        elif _is_tuple_hint(hint) and isinstance(raw, list):
            kwargs[key] = tuple(raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


def _model_from_dict(d: dict) -> ModelConfig:
    d = dict(d)
    frontends = {k: _build(FrontendConfig, v, f"frontends.{k}")
                 for k, v in (d.pop("frontends", None) or {}).items()}
    branches = {}
    for k, v in (d.pop("branches", None) or {}).items():
        v = dict(v)
        if isinstance(v.get("sources"), str):        # `sources: speech` shorthand
            v["sources"] = [v["sources"]]
        if isinstance(v.get("sources"), list):
            v["sources"] = tuple(v["sources"])
        branches[k] = _build(BranchConfig, v, f"branches.{k}")
    rest = _build(ModelConfig, {**d, "frontends": {}, "branches": {}}, "model")
    cfg = dataclasses.replace(rest, frontends=frontends, branches=branches)
    validate_model_config(cfg)
    return cfg


def validate_model_config(cfg: ModelConfig) -> None:
    """Reject configs that would build a model we do not mean to build."""
    if not cfg.frontends:
        raise ConfigError("model: at least one frontend is required")
    for name, fe in cfg.frontends.items():
        if fe.name not in KNOWN_FRONTENDS:
            raise ConfigError(
                f"frontends.{name}: unknown frontend {fe.name!r}; "
                f"known: {sorted(KNOWN_FRONTENDS)}")
        if fe.layers is not None and fe.layers < 1:
            raise ConfigError(f"frontends.{name}.layers must be >= 1, got {fe.layers}")
        if fe.fps <= 0:
            raise ConfigError(f"frontends.{name}.fps must be > 0")
        if fe.freq_pool.rectifier not in ("softplus", "clamp"):
            raise ConfigError(
                f"frontends.{name}.freq_pool.rectifier invalid: {fe.freq_pool.rectifier!r}")
        if fe.freq_pool.kind not in ("gem", "mean", "max", "none"):
            raise ConfigError(f"frontends.{name}.freq_pool.kind invalid: {fe.freq_pool.kind!r}")
        if (fe.freq_pool.kind == "none") != (fe.n_freq is None):
            raise ConfigError(
                f"frontends.{name}: n_freq and freq_pool must agree -- a patch-grid "
                f"frontend needs n_freq to reshape, and a (B,T,D) frontend has no "
                f"frequency axis to pool. Got freq_pool.kind={fe.freq_pool.kind!r}, "
                f"n_freq={fe.n_freq!r}")
        if fe.n_freq is not None and fe.n_freq < 1:
            raise ConfigError(f"frontends.{name}.n_freq must be >= 1")
        if fe.adapter.kind not in ("lora", "conv", "none"):
            raise ConfigError(f"frontends.{name}.adapter.kind invalid: {fe.adapter.kind!r}")
        if not fe.freeze and fe.adapter.kind != "none":
            raise ConfigError(
                f"frontends.{name}: an unfrozen frontend with adapters is almost never "
                "intended -- set adapter.kind: none or freeze: true")

    if not cfg.branches:
        raise ConfigError("model: at least one branch is required")
    for name, br in cfg.branches.items():
        if not br.sources:
            raise ConfigError(f"branches.{name}: needs at least one source")
        for src in br.sources:
            if src not in cfg.frontends:
                raise ConfigError(
                    f"branches.{name}: source {src!r} is not a declared frontend "
                    f"({sorted(cfg.frontends)})")
        if len(br.sources) > 1:
            if br.align_to is None:
                raise ConfigError(
                    f"branches.{name}: multi-source branches must set align_to, because "
                    "the frontends have different frame rates")
            if br.align_to not in br.sources:
                raise ConfigError(
                    f"branches.{name}.align_to={br.align_to!r} must be one of its own "
                    f"sources {list(br.sources)}")
        if br.masked_by is not None and br.masked_by not in MASK_KEYS:
            raise ConfigError(
                f"branches.{name}.masked_by must be one of {list(MASK_KEYS)} or null")
        if not 0.0 <= br.head.clip_weight <= 1.0:
            raise ConfigError(f"branches.{name}.head.clip_weight must be in [0, 1]")
        if br.head.attention not in ("linear", "scaled_tanh", "tanh"):
            raise ConfigError(
                f"branches.{name}.head.attention invalid: {br.head.attention!r}")

    # The five submission columns must be produced exactly once each. This ties the
    # architecture to the already-verified metric contract instead of restating it.
    produced = [br.column for br in cfg.branches.values()]
    if len(produced) != len(set(produced)):
        dupes = sorted({c for c in produced if produced.count(c) > 1})
        raise ConfigError(f"branches: column(s) produced more than once: {dupes}")
    missing = [c for c in PREDICTION_COLUMNS if c not in produced]
    extra = [c for c in produced if c not in PREDICTION_COLUMNS]
    if missing or extra:
        raise ConfigError(
            f"branches must produce exactly the five submission columns; "
            f"missing={missing} unexpected={extra}")

    if cfg.segmentation.mode not in ("whole_file", "tiling"):
        raise ConfigError(f"segmentation.mode invalid: {cfg.segmentation.mode!r}")
    if cfg.segmentation.mode == "tiling":
        if cfg.segmentation.window_seconds <= 0 or cfg.segmentation.hop_seconds <= 0:
            raise ConfigError("segmentation: window_seconds and hop_seconds must be > 0")
        if cfg.segmentation.hop_seconds > cfg.segmentation.window_seconds:
            raise ConfigError(
                "segmentation: hop_seconds > window_seconds would skip audio; "
                "full coverage is a label-semantics requirement (04 §6)")
    if cfg.aggregation.kind not in ("max", "mean", "topk_mean", "quantile", "confidence_gated"):
        raise ConfigError(f"aggregation.kind invalid: {cfg.aggregation.kind!r}")
    if cfg.aggregation.k < 1:
        raise ConfigError("aggregation.k must be >= 1")
    if not 0.0 < cfg.aggregation.quantile <= 1.0:
        raise ConfigError("aggregation.quantile must be in (0, 1]")
    if cfg.file_head.mode not in ("learned", "noisy_or", "max"):
        raise ConfigError(f"file_head.mode invalid: {cfg.file_head.mode!r}")
    if cfg.output.dtype not in ("float64", "float32"):
        raise ConfigError(f"output.dtype invalid: {cfg.output.dtype!r}")
    if cfg.output.dtype != "float64":
        raise ConfigError(
            "output.dtype must be float64: saturation cost EER 0.0950 -> 0.3017 in "
            "measurement, and float32 sigmoid manufactures ties across files")
    if cfg.output.squash not in ("sigmoid", "softsign"):
        raise ConfigError(f"output.squash invalid: {cfg.output.squash!r}")
    if cfg.output.logit_scale <= 0:
        raise ConfigError("output.logit_scale must be > 0")

    if cfg.audio.channels not in ("downmix", "left", "mid_side"):
        raise ConfigError(f"audio.channels invalid: {cfg.audio.channels!r}")
    if cfg.audio.sample_rate != 16_000:
        raise ConfigError(
            f"audio.sample_rate must be 16000 -- the test set is standardised to it "
            f"and resampling is a shortcut risk; got {cfg.audio.sample_rate}")
    if cfg.audio.band_hz is not None:
        lo, hi = cfg.audio.band_hz
        nyquist = cfg.audio.sample_rate / 2
        if not 0 <= lo < hi:
            raise ConfigError(f"audio.band_hz must satisfy 0 <= low < high, got {(lo, hi)}")
        if hi > nyquist:
            raise ConfigError(
                f"audio.band_hz high={hi} exceeds Nyquist {nyquist}; there is no "
                "information above it to restrict to")
    if cfg.audio.min_seconds <= 0 or cfg.audio.max_seconds < cfg.audio.min_seconds:
        raise ConfigError("audio: need 0 < min_seconds <= max_seconds")
    if cfg.runtime.precision not in ("fp32", "fp16", "bf16"):
        raise ConfigError(f"runtime.precision invalid: {cfg.runtime.precision!r}")
    if cfg.runtime.batch_size < 1:
        raise ConfigError("runtime.batch_size must be >= 1")


def load_model_config(path: str | Path) -> ModelConfig:
    return _model_from_dict(yaml.safe_load(Path(path).read_text()) or {})


def load_train_config(path: str | Path) -> TrainConfig:
    d = dict(yaml.safe_load(Path(path).read_text()) or {})
    teachers = {k: _build(FrontendConfig, v, f"train.teachers.{k}")
                for k, v in (d.pop("teachers", None) or {}).items()}
    cfg = dataclasses.replace(_build(TrainConfig, d, "train"), teachers=teachers)
    for key in cfg.loss.weights:
        if key not in {"voice", "music", "file", "v_pres", "m_pres"}:
            raise ConfigError(f"train.loss.weights: unknown head {key!r}")
    for name, fe in cfg.teachers.items():
        if fe.name not in KNOWN_FRONTENDS:
            raise ConfigError(f"train.teachers.{name}: unknown frontend {fe.name!r}")
        if not fe.freeze:
            raise ConfigError(
                f"train.teachers.{name}: a teacher must be frozen -- an unfrozen "
                "teacher is just another trainable branch")
    if cfg.stage not in ("independent", "joint", "codec_aware", "rank_polish"):
        raise ConfigError(f"train.stage invalid: {cfg.stage!r}")
    return cfg


def dump_config(cfg) -> dict:
    """Plain dict, suitable for `yaml.safe_dump` and for the checkpoint."""
    return dataclasses.asdict(cfg)
