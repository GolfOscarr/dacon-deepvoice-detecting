"""The shipped chain -- SHIP-3 to SHIP-6 of docs/processing/03 §3: what happens
to a waveform between decoding and the model, **identically in training and
in ``submit.zip``**.

Critical: one function object, one call site. A rendered training sample is
the analogue of a raw test file -- the last thing done to it was what the
organizers did to theirs (REN-4) -- so everything after that point must be
train/test symmetric (docs/data/10 P2) and per file (rule 2.4): the output of
a row over its valid prefix is bitwise identical solo and inside any batch
(I14). ``ship`` is that function; the training loop and the inference script
both call it and nothing else touches the waveform in between.

The stages, in order::

    SHIP-3  channel policy        models.audio.prepare_waveform   (C, S) -> (S,)
    SHIP-4  preprocess registry   training.registries.PREPROCESS  dc_offset (D-10)
    SHIP-5  band limit            models.audio.bandpass           band_hz (D-11, D-12)
    SHIP-6  loudness              -- none (D-9)

Caveat: the preprocess registry was registered but called by nothing before
this module (docs/processing/03 SHIP-4: "not in any config today"). It is
not run by the renderer, deliberately: a preprocess step is symmetric, so its
call site is here, at the model boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from models.audio import CHANNEL_POLICIES, bandpass, prepare_waveform
from models.config import AudioConfig
from training.registries import PREPROCESS, preprocess_chain
from training.render import RenderedSample

__all__ = ["PreprocessStep", "ShipConfig", "ship", "ship_sample"]


@dataclass(frozen=True)
class PreprocessStep:
    """One registered preprocess step and its bound parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in PREPROCESS:
            raise ValueError(
                f"preprocess {self.name!r} is not registered; known: {PREPROCESS.names()}")
        unknown = sorted(set(self.params) - set(PREPROCESS.params_of(self.name)))
        if unknown:
            raise ValueError(
                f"preprocess {self.name!r} has no parameter(s) {unknown}; it takes "
                f"{list(PREPROCESS.params_of(self.name))}")

    @classmethod
    def from_flat(cls, raw: Any, path: str) -> PreprocessStep:
        """``dc_offset`` or ``{name: pre_emphasis, coeff: 0.97}``, as YAML writes it."""
        if isinstance(raw, str):
            return cls(raw)
        if isinstance(raw, Mapping) and "name" in raw:
            return cls(str(raw["name"]), {k: v for k, v in raw.items() if k != "name"})
        raise ValueError(f"{path}: a preprocess entry is a name or {{name, <params>}}, got {raw!r}")

    def to_flat(self) -> str | dict[str, Any]:
        return self.name if not self.params else {"name": self.name, **self.params}


@dataclass(frozen=True)
class ShipConfig:
    """Everything between the decoded waveform and the model."""

    sample_rate: int = 16_000
    #: SHIP-3. ``downmix`` | ``left`` | ``mid_side`` (``models.audio``).
    channels: str = "downmix"
    #: SHIP-4, D-10. DC is a generator id (0.994 on `cfad/gl`) that inverts
    #: across archives (0.392 grouped); removing it is free and symmetric.
    preprocess: tuple[PreprocessStep, ...] = (PreprocessStep("dc_offset"),)
    #: SHIP-5. D-11: no high-pass (``band_hz[0] = 0`` -- 40 Hz removes 25-75 %
    #: of four vocoders' pair difference). D-12, OPEN: 7 200 Hz upper edge
    #: against the resampler shelf and SONICS' 7.3 kHz rolloff, to be ablated
    #: against ``None``.
    band_hz: tuple[float, float] | None = (0.0, 7200.0)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels not in CHANNEL_POLICIES:
            raise ValueError(f"channels must be one of {CHANNEL_POLICIES}, got {self.channels!r}")
        names = [s.name for s in self.preprocess]
        if len(set(names)) != len(names):
            raise ValueError(f"preprocess lists a step twice: {names}")
        if self.band_hz is not None:
            lo, hi = self.band_hz
            if not 0.0 <= lo < hi <= self.sample_rate / 2:
                raise ValueError(
                    f"band_hz must satisfy 0 <= lo < hi <= {self.sample_rate / 2}, "
                    f"got {self.band_hz}")
        # Critical: the chain is built here so a bad step fails at config time,
        # and its declared group delay must be 0 -- a shift would move the
        # audio out from under `frame_intervals` (docs/pipelines/03 §4).
        chain = preprocess_chain([(s.name, s.params) for s in self.preprocess])
        if chain.group_delay != 0:
            raise ValueError(
                f"the shipped preprocess chain declares a group delay of "
                f"{chain.group_delay} samples; only a zero-delay chain may ship")

    @property
    def audio(self) -> AudioConfig:
        """The ``AudioConfig`` view ``models.audio`` reads."""
        return AudioConfig(sample_rate=self.sample_rate, channels=self.channels,
                           band_hz=self.band_hz)


def _as_batch(wav: Tensor) -> Tensor:
    """``(S,)`` / ``(C, S)`` / ``(B, S)`` / ``(B, C, S)`` -> ``(B, C, S)``.

    Caveat: a 2-D tensor is read as ONE file's ``(C, S)`` only when it is
    passed through ``ship_sample``; ``ship`` reads 2-D as a ``(B, S)`` mono
    batch, which is what the collator produces.
    """
    if wav.dim() == 1:
        return wav[None, None, :]
    if wav.dim() == 2:
        return wav[:, None, :]
    if wav.dim() == 3:
        return wav
    raise ValueError(f"expected (S,), (B, S) or (B, C, S), got {tuple(wav.shape)}")


def ship(wav: Tensor, cfg: ShipConfig, lengths: Tensor | None = None) -> Tensor:
    """The shipped chain over a batch: ``(B, C, S)`` or ``(B, S)`` -> ``(B, S)``.

    ``lengths`` is each row's valid prefix; every stage works over it and
    nothing else (I14). Caveat: the padding beyond it is not signal -- the
    band stage returns zeros there, and the model masks it either way.
    """
    batch = _as_batch(wav)
    audio = cfg.audio
    mono = prepare_waveform(batch, audio)                             # SHIP-3
    if lengths is None:
        lengths = torch.full((mono.shape[0],), mono.shape[-1],
                             dtype=torch.long, device=mono.device)
    chain = preprocess_chain([(s.name, s.params) for s in cfg.preprocess])
    mono = chain(mono, cfg.sample_rate, lengths)                      # SHIP-4
    return bandpass(mono, audio, lengths)                             # SHIP-5


def ship_sample(sample: RenderedSample | Tensor, cfg: ShipConfig) -> Tensor:
    """One file, ``(C, S)`` as decoded -> ``(S,)`` as the model sees it."""
    wav = sample.wav if isinstance(sample, RenderedSample) else sample
    if wav.dim() != 2:
        raise ValueError(f"one file is (C, S), got {tuple(wav.shape)}")
    return ship(wav[None], cfg)[0]
