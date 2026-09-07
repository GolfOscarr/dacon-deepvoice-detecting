"""The three registries: where a processing idea gets plugged in.

🔴 Which registry a transform belongs to is decided by one question -- *does it
also run at test time?* (docs/data/10 P2). Getting it wrong is silent in both
directions, so the three contracts are deliberately **different shapes** and the
difference is enforced at registration time rather than by convention:

===============  ====================  ==========  =========  ==================
Registry         Signature             Test time   Labels     RNG
===============  ====================  ==========  =========  ==================
``PREPROCESS``   (wav, sample_rate,    ✅ same      ❌ never    ❌ never
                  lengths) -> wav       code path
``AUGMENT``      (wav, rng) -> wav     ❌           ❌ **structurally**  ✅
``FILTER``       (manifest_row,        ❌           ✅          ❌
                  quality_row) -> Verdict
===============  ====================  ==========  =========  ==================

🔴 **An augment cannot receive labels.** Not "must not read them" -- the bound
callable takes exactly two arguments, and registration refuses a function that
could take a third. ``P(T | L) = P(T)`` (docs/data/06) is thereby a property of
the type rather than of a code review.

🔴 **A filter cannot touch audio.** Same mechanism: it is handed two metadata
rows and nothing else, so "verdicts are sidecar annotations" (docs/data/10 §1,
gate **G5**) is not a promise anybody has to keep.

⚠️ This module is the one part of the pipeline vendored into ``submit.zip``, so
it imports ``numpy`` and ``torch`` and nothing else from this repo. Do not add a
``pandas``/``models``/``training`` import here: it costs against the offline
install budget (docs/architecture/01) and drags the sampler into the submission.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "AUGMENT", "FILTER", "PREPROCESS", "VERDICTS",
    "Augment", "Filter", "Preprocess", "Registry", "RegistryError", "Verdict",
    "augment_chain", "preprocess_chain",
]

#: ``(wav, sample_rate, lengths) -> wav``. ``wav`` is ``(rows, samples)``; a row
#: is a *channel* when the renderer calls it and a *file* when a batch does, and
#: the contract is the same either way: each row is transformed over its own
#: valid prefix ``lengths[i]``, length-preserving, with nothing read from any
#: other row.
Preprocess = Callable[[Tensor, int, "Tensor | None"], Tensor]

#: ``(wav, rng) -> wav``. Runs on **one** sample inside ``render``, never on a
#: padded batch -- which is why it needs no ``lengths`` and must not be given
#: any. ``wav`` is ``(channels, samples)``.
Augment = Callable[[Tensor, np.random.Generator], Tensor]

#: ``(manifest_row, quality_row) -> Verdict``. Offline, sidecar only.
Filter = Callable[[Mapping[str, Any], Mapping[str, Any]], "Verdict"]

#: docs/data/10 §1. ``quarantine`` is not ``drop``: the row stays in the corpus
#: and stays shipped to DACON, it is only held out of the training stream.
VERDICTS = ("keep", "quarantine", "drop")

#: Parameter names an augment or a preprocess step may not have. The load-bearing
#: guarantee is the arity -- a bound augment takes exactly ``(wav, rng)``, so a
#: label has no way in. This list is the second lock: it stops a label arriving
#: dressed as a "config" keyword that the spec's ``transforms`` dict could fill.
LABEL_PARAM_NAMES = frozenset({
    "label", "labels", "y", "target", "targets", "spec", "cell", "stratum",
    "fake", "is_fake", "file_fake", "voice_fake", "music_fake",
    "voice_present", "music_present", "row", "manifest_row",
})

#: A preprocess step is deterministic. Anything that smells like entropy is
#: refused at registration rather than discovered when a rerun does not match.
RNG_PARAM_NAMES = frozenset({"rng", "generator", "random_state", "seed", "noise"})

#: A filter never sees audio (docs/data/10 §1 / gate G5).
AUDIO_PARAM_NAMES = frozenset({"wav", "audio", "signal", "samples", "waveform", "x"})


class RegistryError(ValueError):
    """A callable that does not satisfy the contract of its registry."""


@dataclass(frozen=True)
class Verdict:
    """What a filter decides about one file.

    ⚠️ ``threshold_version`` is not decoration. docs/data/10 §6 step 5 requires
    every threshold to carry the version, value and approver that produced it,
    because a verdict whose threshold nobody can name cannot be revisited -- and
    revisiting is the entire reason verdicts are sidecar annotations.
    """

    action: str
    reason: str = ""
    threshold_version: str = ""

    def __post_init__(self) -> None:
        if self.action not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.action!r}")

    @property
    def keeps(self) -> bool:
        return self.action == "keep"


# --------------------------------------------------------------------------- #
# The registry


def _check_signature(fn: Callable[..., Any], *, kind: str,
                     positional: Sequence[str],
                     forbidden: frozenset[str]) -> tuple[str, ...]:
    """Refuse anything whose *shape* could break the registry's contract.

    Returns the names of the keyword-only parameters, which are the only ones a
    spec's drawn params may fill.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:                    # pragma: no cover
        raise RegistryError(f"{kind}: cannot inspect {fn!r}: {exc}") from exc

    params = list(sig.parameters.values())
    for p in params:
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise RegistryError(
                f"{kind} {fn.__name__!r}: *{p.name} is refused -- a variadic "
                f"positional is exactly how an extra argument sneaks in")
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise RegistryError(
                f"{kind} {fn.__name__!r}: **{p.name} is refused -- it would let "
                f"any keyword through, including the ones this registry exists "
                f"to keep out")

    head = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    got = tuple(p.name for p in head)
    if got != tuple(positional):
        raise RegistryError(
            f"{kind} {fn.__name__!r} must take exactly {tuple(positional)} "
            f"positionally, got {got}")

    kwonly = tuple(p.name for p in params
                   if p.kind is inspect.Parameter.KEYWORD_ONLY)
    for name in kwonly:
        if name in forbidden:
            raise RegistryError(
                f"{kind} {fn.__name__!r}: parameter {name!r} is forbidden here")
        if sig.parameters[name].default is inspect.Parameter.empty:
            raise RegistryError(
                f"{kind} {fn.__name__!r}: keyword-only parameter {name!r} needs "
                f"a default -- a step must be callable with no params at all")
    return kwonly


class Registry:
    """A named collection of callables that all satisfy one contract.

    ``build(name, params)`` returns a callable with the registry's *bare*
    signature, with the drawn params already bound. That is what keeps the call
    site honest: ``render`` calls ``fn(wav, rng)`` and has nothing else to pass.
    """

    def __init__(self, kind: str, positional: Sequence[str],
                 forbidden: frozenset[str]):
        self.kind = kind
        self.positional = tuple(positional)
        self.forbidden = forbidden
        self._fns: dict[str, Callable[..., Any]] = {}
        self._params: dict[str, tuple[str, ...]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._fns:
                raise RegistryError(f"{self.kind} {name!r} is already registered")
            self._params[name] = _check_signature(
                fn, kind=self.kind, positional=self.positional,
                forbidden=self.forbidden)
            self._fns[name] = fn
            return fn
        return decorate

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._fns:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {self.names()}")
        return self._fns[name]

    def params_of(self, name: str) -> tuple[str, ...]:
        self.get(name)
        return self._params[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._fns))

    def __contains__(self, name: object) -> bool:
        return name in self._fns

    def __len__(self) -> int:
        return len(self._fns)

    def build(self, name: str, params: Mapping[str, Any] | None = None
              ) -> Callable[..., Any]:
        """Bind drawn params, returning a callable with the bare signature.

        ⚠️ An unknown key is an error, not a warning -- the same rule
        ``models.config`` applies to configs, for the same reason: a typo'd knob
        that silently does nothing is how an ablation measures the wrong thing.
        """
        fn = self.get(name)
        params = dict(params or {})
        unknown = sorted(set(params) - set(self._params[name]))
        if unknown:
            raise RegistryError(
                f"{self.kind} {name!r} has no parameter(s) {unknown}; "
                f"it takes {list(self._params[name])}")
        arity = len(self.positional)
        if arity == 3:
            def bound_preprocess(wav, sample_rate, lengths):        # noqa: ANN001
                return fn(wav, sample_rate, lengths, **params)
            return bound_preprocess
        if arity == 2 and self.kind == "augment":
            def bound_augment(wav, rng):                            # noqa: ANN001
                return fn(wav, rng, **params)
            return bound_augment

        def bound_filter(manifest_row, quality_row):                # noqa: ANN001
            return fn(manifest_row, quality_row, **params)
        return bound_filter


PREPROCESS = Registry("preprocess", ("wav", "sample_rate", "lengths"),
                      LABEL_PARAM_NAMES | RNG_PARAM_NAMES)
AUGMENT = Registry("augment", ("wav", "rng"), LABEL_PARAM_NAMES)
FILTER = Registry("filter", ("manifest_row", "quality_row"), AUDIO_PARAM_NAMES)


def preprocess_chain(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> Preprocess:
    """Compose registered preprocess steps into one ``Preprocess``."""
    fns = [PREPROCESS.build(name, params) for name, params in steps]

    def chain(wav: Tensor, sample_rate: int, lengths: Tensor | None) -> Tensor:
        for fn in fns:
            wav = fn(wav, sample_rate, lengths)
        return wav
    return chain


def augment_chain(transforms: Sequence[tuple[str, Mapping[str, Any]]]) -> Augment:
    """Compose a spec's ``transforms`` into one ``Augment``.

    🔴 The returned callable's signature is ``(wav, rng)``. Whatever the caller
    knows about the sample, it has no argument to put it in.
    """
    fns = [AUGMENT.build(name, params) for name, params in transforms]

    def chain(wav: Tensor, rng: np.random.Generator) -> Tensor:
        for fn in fns:
            wav = fn(wav, rng)
        return wav
    return chain


# --------------------------------------------------------------------------- #
# Preprocess steps -- symmetric, and shipped
#
# 🔴 The rule-2.4 contract (docs/pipelines/03 §1): a row's output must not
# depend on what else is in the batch. `lengths` is required, not optional, and
# every step below works on `wav[i, :lengths[i]]`. No batch statistics: no
# per-batch normalisation, no percentile over the batch, no frequency grid
# derived from the padded width. `models.audio.bandpass` is what happens when a
# step forgets -- one sample of padding moved its output by up to 1.0.
#
# ⚠️ Resampling (P-S2) is deliberately *not* here: it changes the sample count,
# and a registry whose steps may change `lengths` cannot be composed. It happens
# once per file at decode time, in `render.load_audio`, with the resampler
# injected so G1 can swap it.


def _valid_lengths(wav: Tensor, lengths: Tensor | None) -> Tensor:
    if wav.dim() != 2:
        raise ValueError(f"preprocess expects (rows, samples), got {tuple(wav.shape)}")
    if lengths is None:
        return torch.full((wav.shape[0],), wav.shape[-1],
                          dtype=torch.long, device=wav.device)
    lengths = lengths.to(dtype=torch.long, device=wav.device)
    if lengths.shape != (wav.shape[0],):
        raise ValueError(
            f"lengths must be ({wav.shape[0]},), got {tuple(lengths.shape)}")
    if int(lengths.max()) > wav.shape[-1]:
        raise ValueError(
            f"lengths up to {int(lengths.max())} exceed the tensor's "
            f"{wav.shape[-1]} samples")
    return lengths


@PREPROCESS.register("dc_offset")
def dc_offset(wav: Tensor, sample_rate: int, lengths: Tensor | None) -> Tensor:
    """P-S4 -- remove each row's DC offset. Harmless, symmetric, and it kills a
    trivial corpus-identity cue (docs/data/10 §2).

    🔴 The mean is taken over the row's **valid prefix**. Over the padded row it
    would be a function of how much padding the batch happens to carry, which is
    the `bandpass` defect with cheaper arithmetic.
    """
    lengths = _valid_lengths(wav, lengths)
    out = wav.clone()
    for i, n in enumerate(lengths.tolist()):
        if n < 1:
            continue
        out[i, :n] = wav[i, :n] - wav[i, :n].mean()
    return out


@PREPROCESS.register("pre_emphasis")
def pre_emphasis(wav: Tensor, sample_rate: int, lengths: Tensor | None, *,
                 coeff: float = 0.97) -> Tensor:
    """P-B2 -- ``y[t] = x[t] - a·x[t-1]``, a fixed, content-independent tilt.

    The recursion is seeded from ``x[0]`` of the row's own prefix, so a row that
    happens to sit next to a longer one is unaffected.
    """
    if not 0.0 <= coeff < 1.0:
        raise ValueError(f"coeff must be in [0, 1), got {coeff}")
    lengths = _valid_lengths(wav, lengths)
    out = wav.clone()
    for i, n in enumerate(lengths.tolist()):
        if n < 1:
            continue
        seg = wav[i, :n]
        out[i, 0] = seg[0]
        if n > 1:
            out[i, 1:n] = seg[1:] - coeff * seg[:-1]
    return out


# --------------------------------------------------------------------------- #
# Augment steps -- training only, label-independent by construction
#
# ⚠️ Two entries of the docs/data/06 menu are deliberately absent, and both for
# the same reason: they move audio along the timeline, while `frame_intervals`
# report where the components were placed. An augment returns only a waveform,
# so it has no way to tell the renderer that the timeline moved, and I13
# ("frame targets and audio describe the same timeline") would silently break.
#
#   * **A-A8 time shift** belongs in the *placement* -- jitter
#     `ComponentDraw.target_start_s` in the sampler, where the spec records it.
#   * **A-A11 silence edits** likewise: leading/trailing silence is a placement
#     decision (`target_start_s` > 0 already expresses it).
#
# A-A5 RawBoost's convolutive and non-stationary variants, A-A10 RIR and A-A9
# SpecAugment are not implemented yet; SpecAugment in particular acts on a
# spectrogram, so it belongs to the frontend's time base, not here.


def _rms(wav: Tensor) -> Tensor:
    return wav.pow(2).mean().clamp_min(1e-20).sqrt()


@AUGMENT.register("gain_jitter")
def gain_jitter(wav: Tensor, rng: np.random.Generator, *,
                db: float | None = None,
                db_range: tuple[float, float] = (-6.0, 6.0)) -> Tensor:
    """A-A7 -- ±6 dB, "simulates varying recording distances".

    ⚠️ Note how modest this is. The measured recipe does its heavy lifting with
    MixUp, not with signal mangling (docs/kaggle/06 §5).

    ``db`` drawn at spec time wins; otherwise it is drawn here from ``db_range``.
    """
    if db is None:
        db = float(rng.uniform(*db_range))
    return wav * float(10.0 ** (db / 20.0))


@AUGMENT.register("gaussian_noise")
def gaussian_noise(wav: Tensor, rng: np.random.Generator, *,
                   snr_db: float | None = None,
                   snr_db_range: tuple[float, float] = (10.0, 30.0)) -> Tensor:
    """A-A6-shaped additive noise at a per-file SNR of 10-30 dB.

    ⚠️ White noise stands in for the MUSAN *noise* and *speech* partitions until
    the corpus exists. Never MUSAN's **music** partition: adding music flips
    ``MUSIC_PRESENT`` to 1, which makes it a component draw (step 2), not an
    augmentation (docs/data/06, Tier X).

    The SNR reference is the file's own RMS, so the operation is a function of
    one file and nothing else.
    """
    if snr_db is None:
        snr_db = float(rng.uniform(*snr_db_range))
    noise = torch.from_numpy(
        rng.standard_normal(tuple(wav.shape)).astype(np.float32)).to(wav.device)
    scale = _rms(wav) / _rms(noise) * float(10.0 ** (-snr_db / 20.0))
    return wav + noise * scale


@AUGMENT.register("rawboost_ssi")
def rawboost_ssi(wav: Tensor, rng: np.random.Generator, *,
                 snr_db_range: tuple[float, float] = (10.0, 40.0),
                 tilt_db_range: tuple[float, float] = (-12.0, 12.0)) -> Tensor:
    """A-A5, stationary signal-independent variant: coloured additive noise.

    ☆ RawBoost is the most effective single augmentation family in a systematic
    comparison against AWGN / vocoded / RIR (docs/survey/08). This is its
    stationary arm only -- a noise floor with a random spectral tilt, modelling
    transmission and microphone colouration. The convolutive and non-stationary
    arms are not implemented.
    """
    snr_db = float(rng.uniform(*snr_db_range))
    tilt_db = float(rng.uniform(*tilt_db_range))
    n = wav.shape[-1]
    noise = torch.from_numpy(
        rng.standard_normal(tuple(wav.shape)).astype(np.float32)).to(wav.device)
    spec = torch.fft.rfft(noise.double(), dim=-1)
    ramp = torch.linspace(0.0, 1.0, spec.shape[-1], dtype=torch.float64,
                          device=spec.device)
    spec = spec * (10.0 ** (tilt_db * ramp / 20.0))
    coloured = torch.fft.irfft(spec, n=n, dim=-1).to(wav.dtype)
    scale = _rms(wav) / _rms(coloured) * float(10.0 ** (-snr_db / 20.0))
    return wav + coloured * scale


@AUGMENT.register("stereo_imbalance")
def stereo_imbalance(wav: Tensor, rng: np.random.Generator, *,
                     db_range: tuple[float, float] = (-4.0, 4.0)) -> Tensor:
    """A-B3 -- L/R level imbalance. A no-op on mono, by construction.

    ⚠️ This is why ``RenderedSample.wav`` stays ``(C, S)``: downmixing in the
    dataset would disable the channel augmentations and fork the channel policy
    away from ``models.audio.prepare_waveform`` (docs/pipelines/01 §4).
    """
    if wav.shape[0] < 2:
        return wav
    out = wav.clone()
    for c in range(wav.shape[0]):
        out[c] = wav[c] * float(10.0 ** (float(rng.uniform(*db_range)) / 20.0))
    return out


# --------------------------------------------------------------------------- #
# Filters -- offline, sidecar, never audio
#
# 🔴 The criterion is **label-evidence sufficiency, not cleanliness**
# (docs/data/10 P1). Our test set contains 전화채널 audio -- narrowband,
# codec-degraded, noisy by construction. Noise is a property of the target
# domain, not a defect. The question is never "is this clean?" but "is the
# labelled component still discernible enough for its status to be judgeable?"


@FILTER.register("corruption")
def corruption(manifest_row: Mapping[str, Any], quality_row: Mapping[str, Any],
               ) -> Verdict:
    """F-S2 -- decode failure, zero length, all-silent, NaN/inf, DC-only.

    The one filter with no threshold to gate: a file that does not decode
    carries no evidence for any label.
    """
    if not quality_row.get("decode_ok", True):
        return Verdict("drop", "decode failed")
    if not quality_row.get("finite", True):
        return Verdict("drop", "non-finite samples")
    if float(quality_row.get("duration_s", manifest_row.get("duration_s", 0.0))) <= 0.0:
        return Verdict("drop", "zero length")
    if bool(quality_row.get("all_silent", False)):
        return Verdict("drop", "all silent")
    return Verdict("keep")


@FILTER.register("usable_duration")
def usable_duration(manifest_row: Mapping[str, Any], quality_row: Mapping[str, Any],
                    *, min_seconds: float = 4.0) -> Verdict:
    """F-A3 -- after salvage, no contiguous valid region ≥ the test minimum.

    ⚠️ ``min_seconds`` defaults to 4.0 because that is the competition's own
    floor (docs/competition/01), not a threshold anyone chose -- which is why
    this one filter needs no ``threshold_version``.
    """
    longest = quality_row.get("longest_valid_span_s")
    if longest is None:
        longest = quality_row.get("duration_s", manifest_row.get("duration_s"))
    if longest is None:
        return Verdict("quarantine", "no usable-duration evidence")
    if float(longest) < min_seconds:
        return Verdict("drop", f"longest valid span {float(longest):.2f}s "
                               f"< {min_seconds}s")
    return Verdict("keep")


@FILTER.register("label_evidence")
def label_evidence(manifest_row: Mapping[str, Any], quality_row: Mapping[str, Any],
                   *, min_component_snr_db: float | None = None,
                   threshold_version: str = "") -> Verdict:
    """F-A2 -- the real version of "too noisy": is the *labelled* component
    still perceptible?

    🔴 Raises unless a threshold **and** its version are supplied. docs/data/10
    §6: no threshold is ever hardcoded from intuition -- it is measured, expressed
    as a quantile of our own data, put in front of a human (**G3**) and recorded
    with a version. A default here would be exactly the hardcoded intuition the
    protocol exists to forbid, and it would be invisible in a green suite.
    """
    if min_component_snr_db is None or not threshold_version:
        raise RegistryError(
            "label_evidence needs min_component_snr_db and threshold_version: "
            "docs/data/10 §6 forbids an intuition threshold, and gate G3 has to "
            "have approved this one before it runs at scale")
    snr = quality_row.get("component_snr_db")
    if snr is None:
        return Verdict("quarantine", "no component SNR measured",
                       threshold_version)
    if float(snr) < min_component_snr_db:
        return Verdict("quarantine",
                       f"component SNR {float(snr):.1f} dB < "
                       f"{min_component_snr_db} dB",
                       threshold_version)
    return Verdict("keep", "", threshold_version)
