"""Waveform preparation.

Two `AudioConfig` knobs act on the raw signal rather than on the network, and
both are real experiments rather than plumbing:

* ``channels`` -- how stereo becomes model input. ⚠️ `mid_side` keeps
  inter-channel information, which may be signal *or* may be a shortcut if fake
  sources skew mono. Test it as a leak before treating it as a feature
  (docs/architecture/09 B8).
* ``band_hz`` -- deliberately restricting the model to low-frequency subbands
  cut EER by up to 25% relative under codec conditions (D9), and our
  telephone-channel slice lives there (09 B7).
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.config import AudioConfig

__all__ = ["bandpass", "prepare_waveform"]


def prepare_waveform(wav: Tensor, cfg: AudioConfig) -> Tensor:
    """(B, C, S) or (B, S) -> (B, S), applying the channel policy.

    Mono input is returned unchanged whatever the policy, since there is no
    second channel to combine.
    """
    if wav.dim() == 2:
        return wav
    if wav.dim() != 3:
        raise ValueError(f"expected (B, C, S) or (B, S), got {tuple(wav.shape)}")
    if wav.shape[1] == 1:
        return wav.squeeze(1)

    if cfg.channels == "downmix":
        return wav.mean(dim=1)
    if cfg.channels == "left":
        return wav[:, 0]
    if cfg.channels == "mid_side":
        # The *mid* channel is the model input; `side` is what a leak test would
        # look at separately. Returning mid alone keeps the (B, S) contract.
        return 0.5 * (wav[:, 0] + wav[:, 1])
    raise ValueError(f"unknown channel policy {cfg.channels!r}")


def bandpass(wav: Tensor, cfg: AudioConfig) -> Tensor:
    """Restrict a waveform to ``cfg.band_hz``, or return it unchanged if None.

    Implemented as a brick wall in the rFFT domain: exact, differentiable, and
    with no filter state to get wrong. ⚠️ It is applied per file over the whole
    signal, so it introduces no cross-file dependence (rule 2.4).
    """
    if cfg.band_hz is None:
        return wav
    low, high = cfg.band_hz
    n = wav.shape[-1]
    spec = torch.fft.rfft(wav.double(), dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / cfg.sample_rate).to(spec.device)
    keep = (freqs >= low) & (freqs <= high)
    return torch.fft.irfft(spec * keep, n=n, dim=-1).to(wav.dtype)
