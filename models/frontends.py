"""Frontend wrappers.

Two families of pretrained encoder are in play and they do **not** share an
output shape (docs/architecture/02, 04 §2):

* wav2vec2 / XLS-R / WavLM emit ``(B, T, D)`` -- frequency already collapsed.
* BEATs / EAT / SSLAM are ViT-style over mel patches, so their tokens form an
  ``(F', T')`` grid that must be pooled over ``F'`` before the head sees it.

Every wrapper here normalises to one contract, so `models.heads` never needs to
know which family fed it:

    forward(wav, lengths) -> (features: (B, T, D), frame_mask: (B, T))

plus ``output_dim`` and ``fps``. ``fps`` is what lets two frontends with
different frame rates be aligned onto a common time base for the file branch.

`StubFrontend` is a randomly-initialised encoder with no pretrained weights. It
exists so that every shape, mask and invariance test runs *now* -- before any
checkpoint is downloaded, and before the licence questions in
docs/architecture/09 C1-C3 are answered.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.config import AudioConfig, FrontendConfig
from models.heads import FreqPool

__all__ = ["Frontend", "StubFrontend", "build_frontend", "frames_for"]


def frames_for(n_samples: Tensor | int, sample_rate: int, fps: float) -> Tensor | int:
    """How many frames a frontend emits for a given number of samples.

    Defined once, because the frame mask, the time alignment between two
    frontends, and the stub's own pooling must all agree on it.
    """
    if isinstance(n_samples, Tensor):
        return torch.clamp((n_samples.double() * fps / sample_rate).round().long(), min=1)
    return max(1, round(n_samples * fps / sample_rate))


class Frontend(nn.Module):
    """Common contract. Subclasses implement `_encode`."""

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.audio = audio
        self.fps = cfg.fps
        self.output_dim = cfg.output_dim
        self.freq_pool = FreqPool(cfg.freq_pool) if cfg.freq_pool.kind != "none" else None

    def _encode(self, wav: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``wav`` is (B, S) at ``audio.sample_rate``; ``lengths`` is (B,) in samples."""
        if wav.dim() != 2:
            raise ValueError(f"frontend expects (B, S) mono audio, got {tuple(wav.shape)}")
        feats = self._encode(wav)

        if self.freq_pool is not None:
            if feats.dim() != 4:
                raise ValueError(
                    f"{type(self).__name__}: freq_pool is configured but the encoder "
                    f"emitted {feats.dim()}-D output; expected (B, F, T, D)")
            feats = self.freq_pool(feats)
        elif feats.dim() != 3:
            raise ValueError(
                f"{type(self).__name__}: expected (B, T, D) with freq_pool disabled, "
                f"got {tuple(feats.shape)}")

        b, t, _ = feats.shape
        if lengths is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=feats.device)
        else:
            valid = frames_for(lengths, self.audio.sample_rate, self.fps).to(feats.device)
            mask = torch.arange(t, device=feats.device)[None, :] < valid[:, None].clamp(max=t)
        return feats, mask


class StubFrontend(Frontend):
    """A small deterministic encoder with no pretrained weights.

    Deliberately cheap: a strided conv over the waveform, then an exact resample
    to the frame count `frames_for` predicts. Exactness matters more than realism
    -- the whole point is that masks and time alignment can be tested.
    """

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__(cfg, audio)
        self.n_freq = cfg.n_freq or 1
        width = 64
        self.conv = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=25, stride=8, padding=12), nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, stride=4, padding=4), nn.GELU(),
        )
        self.proj = nn.Linear(width, self.n_freq * cfg.output_dim)

    def _encode(self, wav: Tensor) -> Tensor:
        t = frames_for(wav.shape[-1], self.audio.sample_rate, self.fps)
        h = self.conv(wav.unsqueeze(1))                       # (B, width, T')
        h = F.adaptive_avg_pool1d(h, t).transpose(1, 2)       # (B, T, width)
        h = self.proj(h)                                      # (B, T, F*D)
        if self.cfg.n_freq is None:
            return h
        b, t_, _ = h.shape
        # (B, T, F, D) -> (B, F, T, D), matching what a patch-grid frontend gives us
        return h.view(b, t_, self.n_freq, self.output_dim).permute(0, 2, 1, 3).contiguous()


def build_frontend(cfg: FrontendConfig, audio: AudioConfig) -> Frontend:
    """Construct a frontend from config.

    ⚠️ Only `stub` is implemented. Wiring real checkpoints is deliberately not
    done yet: docs/architecture/09 C1-C3 record that the licences for SSLAM, EAT
    and W2V-BERT 2.0 are unverified, and a licence that forbids third-party
    provision makes a checkpoint unusable *at all* here rather than merely
    unshippable. Downloading first and checking later is the wrong order.
    """
    if cfg.name == "stub":
        return StubFrontend(cfg, audio)
    raise NotImplementedError(
        f"frontend {cfg.name!r} is not wired yet. Only 'stub' is implemented; real "
        f"checkpoints are gated on the licence verification in "
        f"docs/architecture/09-open-questions.md (C1-C3). Set `name: stub` with "
        f"`weights: null` to build and test the architecture without weights.")
