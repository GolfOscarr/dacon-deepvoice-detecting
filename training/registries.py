"""The three registries: where a processing idea gets plugged in.

Critical: which registry a transform belongs to is decided by one question --
*does it also run at test time?* (docs/data/10 P2). Getting it wrong is silent
in both directions, so the three contracts are deliberately **different
shapes** and the difference is enforced at registration time rather than by
convention:

==============  =======================  =========  ===============  =====
Registry        Signature                Test time  Labels           RNG
==============  =======================  =========  ===============  =====
``PREPROCESS``  (wav, sample_rate,       yes, same  never            never
                lengths) -> wav          code path
``AUGMENT``     (wav, rng) -> wav        no         never, and       yes
                                                    structurally so
``FILTER``      (manifest_row,           no         yes              never
                quality_row) -> Verdict
==============  =======================  =========  ===============  =====

Critical: **An augment cannot receive labels.** Not "must not read them" -- the
bound callable takes exactly two arguments, and registration refuses a function
that could take a third. ``P(T | L) = P(T)`` (docs/data/06) is thereby a
property of the type rather than of a code review.

Critical: **A filter cannot touch audio.** Same mechanism: it is handed two
metadata rows and nothing else, so "verdicts are sidecar annotations"
(docs/data/10 §1, gate **G5**) is not a promise anybody has to keep.

Caveat: this module is the one part of the pipeline vendored into
``submit.zip``, so it imports ``numpy`` and ``torch`` and nothing else from
this repo. Do not add a ``pandas``/``models``/``training`` import here: it
costs against the offline install budget (docs/architecture/01) and drags the
sampler into the submission.
"""

from __future__ import annotations

import functools
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
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

    Caveat: ``threshold_version`` is not decoration. docs/data/10 §6 step 5
    requires every threshold to carry the version, value and approver that
    produced it, because a verdict whose threshold nobody can name cannot be
    revisited -- and revisiting is the entire reason verdicts are sidecar
    annotations.
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


# --------------------------------------------------------------------------- #
# Critical: the time-invariance contract
#
# > **Steps 4-5 are time-invariant. Every time-warping decision lives in the
# > draw (steps 0-3), where the spec records it.**
#
# `frame_intervals` are derived from `ComponentDraw.target_start_s`, drawn before
# any file is opened. So anything that moves audio *after* the draw
# desynchronises the labels from the waveform, silently, in absolute seconds
# (docs/pipelines/03 §4, I13). There are four ways to move audio, and only the
# first is one number:
#
#   1. rigid shift        time shift, an encoder's delay, leading silence
#   2. rate change        time stretch, an uncompensated resample
#   3. non-monotonic edit internal trimming, splicing, packet-loss concealment
#   4. group delay        RIR convolution, and any non-linear-phase filter
#
# Caveat: class 4 is the one nobody lists, and it is why this is measured
# rather than reviewed. `models.audio.bandpass` is safe only because it is
# zero-phase (brick wall in rFFT) and `resample_poly` only because it is linear
# phase and self-compensating. Neither is a decision anyone recorded -- they
# are load-bearing accidents, and this repo has been bitten four times by
# exactly that.
#
# The enforcement is a probe at **registration** time, not a docstring and not a
# declaration a caller must trust: the step is run on a chirp and its output is
# correlated back against the input. A step that moves audio cannot be
# registered, so it cannot reach `render`.

_PROBE_SR = 16_000
_PROBE_N = 4096                        # 0.256 s -- long enough for a clean peak
_PROBE_WIDTH = _PROBE_N // 4
#: Windows sit *inside* the probe, not at its edges, so a shift of either sign
#: has somewhere to come from. A window starting at 0 cannot express a positive
#: delay -- its content came from before the signal began -- and reports noise.
_PROBE_WINDOWS = (_PROBE_N // 6, _PROBE_N - _PROBE_N // 6 - _PROBE_WIDTH)


def _probe_signal() -> np.ndarray:
    """Fixed-seed white noise, ``(2, N)``, stereo so channel-wise steps run twice.

    Caveat: noise rather than a chirp, and the reason is the whole point of the
    probe. A chirp's head and tail hold *different frequencies*, so a filter
    with frequency-dependent phase -- `pre_emphasis`, a 2-tap differencer --
    reads as a head and tail that disagree, and a step that moves nothing gets
    refused. Noise is flat in every window, so what the two windows compare is
    the time base and nothing else.
    """
    x = np.random.default_rng(0x7A17).standard_normal((2, _PROBE_N)) * 0.3
    x[1] *= 0.8
    return x.astype(np.float32)


def _window_lag(reference: np.ndarray, signal: np.ndarray,
                start: int, width: int) -> int:
    """How far ``signal[start:start+width]`` sits from where it began, in samples."""
    seg = signal[start:start + width]
    seg = seg - seg.mean()
    ref = reference - reference.mean()
    if not np.any(seg):                       # a silenced window says nothing
        return 0
    size = 1 << int(np.ceil(np.log2(len(ref) + width)))
    corr = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(seg, size)),
                        size)[:len(ref) - width + 1]
    return int(start - int(np.argmax(corr)))


def _measure_time_warp(run: Callable[[Tensor], Tensor]) -> tuple[int, int, int]:
    """``(length change, head lag, tail lag)`` of a step, in samples.

    Critical: two windows, not one. A single lag catches class 1 and 4 but is
    blind to 2 and 3: a 1% stretch and an internal excision both leave the head
    where it was. Comparing the head's lag with the tail's makes the *rigidity*
    of the map observable -- if they disagree, the time base was warped rather
    than moved.
    """
    probe = _probe_signal()
    out = run(torch.from_numpy(probe.copy()))
    if not isinstance(out, Tensor) or out.dim() != 2:
        raise RegistryError(
            f"a step must return a 2-D tensor, got {type(out).__name__}")
    got = out.detach().cpu().numpy()[0].astype(np.float64)
    delta = got.shape[-1] - _PROBE_N
    if delta:
        return delta, 0, 0
    ref = probe[0].astype(np.float64)
    head, tail = (_window_lag(ref, got, start, _PROBE_WIDTH)
                  for start in _PROBE_WINDOWS)
    return 0, head, tail


class Registry:
    """A named collection of callables that all satisfy one contract.

    ``build(name, params)`` returns a callable with the registry's *bare*
    signature, with the drawn params already bound. That is what keeps the call
    site honest: ``render`` calls ``fn(wav, rng)`` and has nothing else to pass.
    """

    def __init__(self, kind: str, positional: Sequence[str],
                 forbidden: frozenset[str], time_invariant: bool = False):
        self.kind = kind
        self.positional = tuple(positional)
        self.forbidden = forbidden
        #: Whether registration probes the step for a time warp.
        self.time_invariant = time_invariant
        self._fns: dict[str, Callable[..., Any]] = {}
        self._params: dict[str, tuple[str, ...]] = {}
        self._group_delay: dict[str, int] = {}

    def register(self, name: str, *, group_delay: int = 0
                 ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a step, **measuring** that it satisfies the contract.

        ``group_delay`` is the shift the step introduces, in samples at 16 kHz.
        Critical: it is a declaration the probe *checks*, not one it believes:
        declare 0 and shift by 137 and registration fails; declare 137 and
        shift by 0 and it fails too. An undeclared shift cannot be registered,
        so it cannot reach `render` and cannot desynchronise `frame_intervals`.

        Caveat: an augment may not declare one at all. A delay it wanted
        would be a *draw* -- `ComponentDraw.target_start_s` -- where the spec
        records it and the frame targets are computed from it.
        """
        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._fns:
                raise RegistryError(f"{self.kind} {name!r} is already registered")
            self._params[name] = _check_signature(
                fn, kind=self.kind, positional=self.positional,
                forbidden=self.forbidden)
            if group_delay and not self.time_invariant:
                raise RegistryError(
                    f"{self.kind} {name!r}: only a preprocess step may declare a "
                    f"group delay")
            if self.kind == "augment" and group_delay:
                raise RegistryError(
                    f"augment {name!r}: an augment cannot move audio in time. A "
                    f"shift is a draw -- jitter ComponentDraw.target_start_s, "
                    f"where the spec records it and the frame targets follow it")
            if self.time_invariant:
                self._check_time_invariance(name, fn, group_delay)
            self._fns[name] = fn
            self._group_delay[name] = int(group_delay)
            return fn
        return decorate

    def _check_time_invariance(self, name: str, fn: Callable[..., Any],
                               group_delay: int) -> None:
        if self.kind == "preprocess":
            def run(wav: Tensor) -> Tensor:
                return fn(wav, _PROBE_SR,
                          torch.tensor([wav.shape[-1]] * wav.shape[0]))
        else:
            def run(wav: Tensor) -> Tensor:
                return fn(wav, np.random.default_rng(0))

        delta, head, tail = _measure_time_warp(run)
        where = f"{self.kind} {name!r}"
        if delta:
            raise RegistryError(
                f"{where} changed the sample count by {delta:+d}. Length is the "
                f"timeline: an edit that inserts or removes audio (A-A11 silence, "
                f"A-C2 stretch, a codec) belongs in the draw, where "
                f"ComponentDraw.duration_s records it")
        if head != tail:
            raise RegistryError(
                f"{where} warped the time base -- its head moved {head:+d} "
                f"samples and its tail {tail:+d}. A rate change or an internal "
                f"edit cannot be described by frame_intervals, which are "
                f"intervals on the drawn timeline")
        if head != group_delay:
            raise RegistryError(
                f"{where} shifts audio by {head:+d} samples but declares "
                f"group_delay={group_delay}. An undeclared shift moves the "
                f"waveform out from under frame_intervals -- the `align_time` "
                f"defect. Declare the measured delay, or make the step "
                f"zero-phase (like models.audio.bandpass) or linear phase and "
                f"self-compensating (like scipy.signal.resample_poly)")

    def group_delay_of(self, name: str) -> int:
        """The step's declared, measured shift in samples at 16 kHz."""
        self.get(name)
        return self._group_delay[name]

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

        Caveat: an unknown key is an error, not a warning -- the same rule
        ``models.config`` applies to configs, for the same reason: a typo'd
        knob that silently does nothing is how an ablation measures the wrong
        thing.
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
                      LABEL_PARAM_NAMES | RNG_PARAM_NAMES, time_invariant=True)
AUGMENT = Registry("augment", ("wav", "rng"), LABEL_PARAM_NAMES,
                   time_invariant=True)
FILTER = Registry("filter", ("manifest_row", "quality_row"), AUDIO_PARAM_NAMES)


def preprocess_chain(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> Preprocess:
    """Compose registered preprocess steps into one ``Preprocess``.

    The composed callable carries ``.group_delay``, the sum of its steps'. It is
    0 for everything registered today; a caller that ever assembles a chain with
    a non-zero total owns compensating for it, because that is the number by
    which its output has moved away from ``frame_intervals``.
    """
    fns = [PREPROCESS.build(name, params) for name, params in steps]

    def chain(wav: Tensor, sample_rate: int, lengths: Tensor | None) -> Tensor:
        for fn in fns:
            wav = fn(wav, sample_rate, lengths)
        return wav

    chain.group_delay = sum(PREPROCESS.group_delay_of(name) for name, _ in steps)
    return chain


def augment_chain(transforms: Sequence[tuple[str, Mapping[str, Any]]]) -> Augment:
    """Compose a spec's ``transforms`` into one ``Augment``.

    Critical: the returned callable's signature is ``(wav, rng)``. Whatever the
    caller knows about the sample, it has no argument to put it in.
    """
    names = [name for name, _ in transforms]
    fns = [AUGMENT.build(name, params) for name, params in transforms]

    def chain(wav: Tensor, rng: np.random.Generator) -> Tensor:
        for name, fn in zip(names, fns):
            before = wav.shape
            wav = fn(wav, rng)
            # Caveat: the registration probe runs with *default* params, so it
            # cannot see a warp that only a drawn parameter turns on. This is
            # the same check on the real audio, on every sample, for a few
            # hundred nanoseconds. Defence in depth for the class of defect
            # that has shipped here four times.
            if wav.shape != before:
                raise RegistryError(
                    f"augment {name!r} returned {tuple(wav.shape)} for "
                    f"{tuple(before)}: step 4 is time-invariant, so a length "
                    f"change belongs in the draw (ComponentDraw.duration_s)")
        return wav
    return chain


# --------------------------------------------------------------------------- #
# Preprocess steps -- symmetric, and shipped
#
# Critical: the rule-2.4 contract (docs/pipelines/03 §1): a row's output must
# not depend on what else is in the batch. `lengths` is required, not optional,
# and every step below works on `wav[i, :lengths[i]]`. No batch statistics: no
# per-batch normalisation, no percentile over the batch, no frequency grid
# derived from the padded width. `models.audio.bandpass` is what happens when a
# step forgets -- one sample of padding moved its output by up to 1.0.
#
# Caveat: resampling (P-S2) is deliberately *not* here: it changes the sample
# count, and a registry whose steps may change `lengths` cannot be composed. It
# happens once per file at decode time, in `render.load_audio`, with the
# resampler injected so G1 can swap it.


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

    Critical: the mean is taken over the row's **valid prefix**. Over the
    padded row it would be a function of how much padding the batch happens to
    carry, which is the `bandpass` defect with cheaper arithmetic.
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
# Caveat: two entries of the docs/data/06 menu are deliberately absent, and
# both for the same reason: they move audio along the timeline, while
# `frame_intervals` report where the components were placed. An augment returns
# only a waveform, so it has no way to tell the renderer that the timeline
# moved, and I13 ("frame targets and audio describe the same timeline") would
# silently break.
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

    Caveat: note how modest this is. The measured recipe does its heavy lifting
    with MixUp, not with signal mangling (docs/kaggle/06 §5).

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

    Caveat: white noise stands in for the MUSAN *noise* and *speech* partitions
    until the corpus exists. Never MUSAN's **music** partition: adding music
    flips ``MUSIC_PRESENT`` to 1, which makes it a component draw (step 2), not
    an augmentation (docs/data/06, Tier X).

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
                 snr_db: float | None = None, tilt_db: float | None = None,
                 snr_db_range: tuple[float, float] = (10.0, 40.0),
                 tilt_db_range: tuple[float, float] = (-12.0, 12.0)) -> Tensor:
    """A-A5, stationary signal-independent variant: coloured additive noise.

    Weak evidence: RawBoost is the most effective single augmentation family in
    a systematic comparison against AWGN / vocoded / RIR (docs/survey/08). This
    is its stationary arm only -- a noise floor with a random spectral tilt,
    modelling transmission and microphone colouration. The convolutive and
    non-stationary arms are not implemented.
    """
    # the scalars are drawn at spec time by the processing sampler (05 B12),
    # so the audit sees them; the ranges are the fallback for a bare call
    if snr_db is None:
        snr_db = float(rng.uniform(*snr_db_range))
    if tilt_db is None:
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
                     db: float | None = None,
                     db_range: tuple[float, float] = (-4.0, 4.0)) -> Tensor:
    """A-B3 -- L/R level imbalance. A no-op on mono, by construction.

    Caveat: this is why ``RenderedSample.wav`` stays ``(C, S)``: downmixing in
    the dataset would disable the channel augmentations and fork the channel
    policy away from ``models.audio.prepare_waveform`` (docs/pipelines/01 §4).
    """
    if wav.shape[0] < 2:
        return wav
    out = wav.clone()
    if db is not None:
        # one spec-time scalar: +db/2 on the left, -db/2 on the right
        out[0] = wav[0] * float(10.0 ** (db / 40.0))
        out[1] = wav[1] * float(10.0 ** (-db / 40.0))
        return out
    for c in range(wav.shape[0]):
        out[c] = wav[c] * float(10.0 ** (float(rng.uniform(*db_range)) / 20.0))
    return out


@AUGMENT.register("pink_noise")
def pink_noise(wav: Tensor, rng: np.random.Generator, *,
               snr_db: float | None = None,
               snr_db_range: tuple[float, float] = (10.0, 30.0)) -> Tensor:
    """A-A6-shaped additive noise with a 1/f power spectrum (docs/processing/03
    DRAW-6), at a per-file SNR against the file's own RMS.

    White noise is shaped in the rFFT domain by ``1 / sqrt(f)`` (the DC bin is
    zeroed), which is the -3 dB/octave slope of pink noise; `gaussian_noise` is
    the flat case. ``snr_db`` drawn at spec time wins.
    """
    if snr_db is None:
        snr_db = float(rng.uniform(*snr_db_range))
    n = wav.shape[-1]
    white = torch.from_numpy(
        rng.standard_normal(tuple(wav.shape)).astype(np.float32)).to(wav.device)
    spec = torch.fft.rfft(white.double(), dim=-1)
    f = torch.arange(spec.shape[-1], dtype=torch.float64, device=spec.device)
    shape = torch.where(f > 0, 1.0 / torch.sqrt(torch.clamp(f, min=1.0)), torch.zeros_like(f))
    pink = torch.fft.irfft(spec * shape, n=n, dim=-1).to(wav.dtype)
    scale = _rms(wav) / _rms(pink) * float(10.0 ** (-snr_db / 20.0))
    return wav + pink * scale


#: A-A10's fallback bank: without a directory of measured responses the
#: augment convolves with a synthetic one -- a unit direct path at 0 followed
#: by exponentially decaying noise (RT60 ~0.4 s) -- so the registry probe and
#: any machine without the corpus can run it. Built once per (seed, length).
_SYNTHETIC_RIR_N = 8_000


def _synthetic_rir(rng: np.random.Generator, n: int = _SYNTHETIC_RIR_N) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / 16_000.0
    h = rng.standard_normal(n) * np.exp(-t / 0.06) * 0.05
    h[0] = 1.0
    return h.astype(np.float32)


@functools.lru_cache(maxsize=8)
def _rir_bank(bank_dir: str, pattern: str) -> tuple[str, ...]:
    files = tuple(sorted(str(p) for p in Path(bank_dir).glob(pattern)))
    if not files:
        raise RegistryError(f"rir: no file matches {pattern!r} under {bank_dir}")
    return files


def _load_rir(path: str, channel: int, max_s: float) -> np.ndarray:
    """One response: the drawn channel, resampled to 16 kHz if needed, the
    direct path moved to sample 0, truncated, L2-normalised."""
    import soundfile as sf

    h, sr = sf.read(path, dtype="float32", always_2d=True)
    h = h[:, channel % h.shape[1]].astype(np.float64)
    if sr != 16_000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 16_000)
        h = resample_poly(h, 16_000 // g, int(sr) // g)
    h = h[int(np.argmax(np.abs(h))):][:int(max_s * 16_000)]
    return (h / max(np.linalg.norm(h), 1e-12)).astype(np.float32)


@AUGMENT.register("rir")
def rir(wav: Tensor, rng: np.random.Generator, *,
        bank_dir: str | None = None, pattern: str = "*.wav",
        wet: float | None = None, wet_range: tuple[float, float] = (0.3, 1.0),
        pick: float | None = None, max_rir_s: float = 1.0) -> Tensor:
    """A-A10 -- room reverberation by convolution with a measured impulse
    response (docs/processing/03 DRAW-6), dry/wet mixed, level preserved.

    Critical: an augment may not move audio, and a raw response does -- the
    RIRS real responses put the direct path ~2,100 samples in. The response is
    therefore aligned so its direct path sits at sample 0 before convolving,
    which the registry's time-warp probe verifies (lag 0 at head and tail).
    ``bank_dir`` unset uses the synthetic response, so registration and the
    probe need no corpus; the v1 config points it at RIRS' 218 real responses.
    The output is rescaled to the input's RMS: reverberation is not a gain.
    """
    if wet is None:
        wet = float(rng.uniform(*wet_range))
    if bank_dir is None:
        h = _synthetic_rir(rng)
    else:
        # `pick` in [0, 1) names the response and its channel at spec time
        # (05 B12); every channel of a multichannel response is reachable.
        bank = _rir_bank(str(bank_dir), pattern)
        if pick is None:
            pick = float(rng.random())
        u = min(max(float(pick), 0.0), 1.0 - 1e-12)
        which = int(u * len(bank))
        chan = int((u * len(bank) - which) * 64)        # sub-index -> channel
        h = _load_rir(bank[which], chan, max_rir_s)
    n = wav.shape[-1]
    size = 1 << int(np.ceil(np.log2(n + len(h))))
    x = wav.detach().cpu().double()
    hk = torch.fft.rfft(torch.from_numpy(h.astype(np.float64)), size)
    wet_sig = torch.fft.irfft(torch.fft.rfft(x, size) * hk, size)[..., :n]
    out = (1.0 - wet) * x + wet * wet_sig
    out = out * (_rms(x) / _rms(out).clamp_min(1e-12))
    return out.to(wav.dtype).to(wav.device)


#: docs/training/15 N4 -- the augments below assume the render rate, as `rir`
#: does (the registry's probe runs at 16 kHz too).
_AUG_SR = 16_000


@AUGMENT.register("packet_loss")
def packet_loss(wav: Tensor, rng: np.random.Generator, *,
                rate: float | None = None, rate_range: tuple[float, float] = (0.01, 0.05),
                frame_ms: float = 20.0, repeat_prob: float = 0.5) -> Tensor:
    """docs/training/15 N4 -- VoIP packet loss: each ``frame_ms`` frame is lost
    with probability ``rate`` and concealed IN PLACE, by zero fill or by
    repeating the previous (already concealed) frame -- one concealment per
    sample, repeat with ``repeat_prob``. A lost first frame is zero-filled.

    Caveat: the class-3 edit the time-invariance contract names is a
    concealment that stretches or splices; this one only replaces a frame's
    samples where they stand, so every sample keeps its time and the frame
    targets stay true (the registry's probe measures lag 0). The loss pattern
    is drawn from ``rng`` -- the spec's render stream -- so it is fixed per
    spec. ``rate`` drawn at spec time wins.
    """
    if rate is None:
        rate = float(rng.uniform(*rate_range))
    repeat = bool(rng.random() < repeat_prob)
    frame = max(1, int(round(frame_ms * _AUG_SR / 1000.0)))
    n = wav.shape[-1]
    n_frames = -(-n // frame)
    lost = np.flatnonzero(rng.random(n_frames) < rate)
    if not len(lost):
        return wav
    out = wav.clone()
    for i in lost.tolist():
        a, b = i * frame, min(n, (i + 1) * frame)
        if repeat and i > 0:
            out[..., a:b] = out[..., a - frame:a - frame + (b - a)]
        else:
            out[..., a:b] = 0.0
    return out


@AUGMENT.register("compression")
def compression(wav: Tensor, rng: np.random.Generator, *,
                threshold_db: float | None = None,
                threshold_db_range: tuple[float, float] = (-12.0, 0.0),
                ratio: float | None = None, ratio_range: tuple[float, float] = (2.0, 8.0),
                release_ms: float | None = None,
                release_ms_range: tuple[float, float] = (50.0, 300.0),
                attack_ms: float = 5.0,
                target_dbfs: float | None = None,
                target_dbfs_range: tuple[float, float] = (-26.0, -14.0)) -> Tensor:
    """docs/training/15 N4 -- dynamic-range compression then loudness
    normalisation, the post-process a phone, a platform or an editor applies
    (REAL under A1: it happens to genuine recordings).

    A feed-forward compressor, channels linked: the level is the RMS of 1 ms
    blocks, the static curve takes ``(level - threshold) * (1 - 1 / ratio)``
    off above ``threshold`` (``threshold_db`` is relative to the file's own
    RMS, so the curve bites the same way whatever the gain before it), and the
    gain is smoothed with a ``attack_ms`` / ``release_ms`` one-pole ballistic.
    Then the file is scaled to ``target_dbfs`` RMS, capped so the peak stays
    at or below -0.1 dBFS (a normaliser does not clip).

    Critical: the gain multiplies the undelayed signal, interpolated from the
    block centres -- a level-dependent gain, never a shift (probe: lag 0).
    Scalars drawn at spec time win; otherwise they are drawn here.
    """
    if threshold_db is None:
        threshold_db = float(rng.uniform(*threshold_db_range))
    if ratio is None:
        ratio = float(rng.uniform(*ratio_range))
    if release_ms is None:
        release_ms = float(rng.uniform(*release_ms_range))
    if target_dbfs is None:
        target_dbfs = float(rng.uniform(*target_dbfs_range))
    x = wav.detach().cpu().double().numpy()
    n = x.shape[-1]
    block = _AUG_SR // 1000
    n_blocks = -(-n // block)
    power = np.zeros(n_blocks * block)
    power[:n] = (x ** 2).mean(axis=0)
    level = 10.0 * np.log10(power.reshape(n_blocks, block).mean(axis=1) + 1e-12)
    thr = 10.0 * np.log10(float((x ** 2).mean()) + 1e-12) + threshold_db
    want = -np.maximum(level - thr, 0.0) * (1.0 - 1.0 / max(float(ratio), 1.0))
    a_att = math.exp(-1.0 / max(attack_ms, 1e-3))          # per 1 ms block
    a_rel = math.exp(-1.0 / max(release_ms, 1e-3))
    gain_db = np.empty(n_blocks)
    g = 0.0
    for i, w in enumerate(want.tolist()):
        coef = a_att if w < g else a_rel                   # falling gain = attack
        g = coef * g + (1.0 - coef) * w
        gain_db[i] = g
    centres = np.arange(n_blocks) * block + (block - 1) / 2.0
    gain = 10.0 ** (np.interp(np.arange(n), centres, gain_db) / 20.0)
    y = x * gain
    rms = math.sqrt(float((y ** 2).mean()))
    peak = float(np.abs(y).max()) if y.size else 0.0
    if rms > 1e-9:
        scale = 10.0 ** (target_dbfs / 20.0) / rms
        if peak * scale > 10.0 ** (-0.1 / 20.0):
            scale = 10.0 ** (-0.1 / 20.0) / peak
        y = y * scale
    return torch.from_numpy(y).to(wav.dtype).to(wav.device)


# --------------------------------------------------------------------------- #
# Filters -- offline, sidecar, never audio
#
# Critical: the criterion is **label-evidence sufficiency, not cleanliness**
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

    Caveat: ``min_seconds`` defaults to 4.0 because that is the competition's
    own floor (docs/competition/01), not a threshold anyone chose -- which is
    why this one filter needs no ``threshold_version``.
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

    Critical: raises unless a threshold **and** its version are supplied.
    docs/data/10 §6: no threshold is ever hardcoded from intuition -- it is
    measured, expressed as a quantile of our own data, put in front of a human
    (**G3**) and recorded with a version. A default here would be exactly the
    hardcoded intuition the protocol exists to forbid, and it would be
    invisible in a green suite.
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
