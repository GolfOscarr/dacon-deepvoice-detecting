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

__all__ = ["Frontend", "StubFrontend", "build_frontend", "frames_for", "hop_length"]


def hop_length(sample_rate: int, fps: float) -> int:
    """Samples per frame."""
    return max(1, round(sample_rate / fps))


def frames_for(n_samples: Tensor | int, sample_rate: int, fps: float) -> Tensor | int:
    """How many frames a frontend emits for a given number of samples.

    Defined once, because the frame mask, the time alignment between two
    frontends, and the stub's own framing must all agree on it. 🔴 Frame *i*
    must correspond to a fixed sample range regardless of how long the rest of
    the batch is -- otherwise padding shifts a file's content and both the mask
    and rule-2.4 batch invariance become meaningless.
    """
    hop = hop_length(sample_rate, fps)
    if isinstance(n_samples, Tensor):
        return torch.clamp(-(-n_samples // hop), min=1)          # ceil division
    return max(1, -(-int(n_samples) // hop))


class Frontend(nn.Module):
    """Common contract. Subclasses implement `_encode`."""

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.audio = audio
        self.fps = cfg.fps
        self.output_dim = cfg.output_dim
        self.freq_pool = FreqPool(cfg.freq_pool) if cfg.freq_pool.kind != "none" else None

    def _apply_freeze(self) -> None:
        """Freeze the pretrained encoder, but not our own pooling.

        Called at the end of a subclass's __init__, once its modules exist.
        `freq_pool`'s learnable GeM exponent is ours, not the checkpoint's, so
        freezing the frontend must not freeze it too.
        """
        if not self.cfg.freeze:
            return
        pool_params = set(id(p) for p in (self.freq_pool.parameters() if self.freq_pool else []))
        for p in self.parameters():
            if id(p) not in pool_params:
                p.requires_grad_(False)

    def _encode(self, wav: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``wav`` is (B, S) at ``audio.sample_rate``; ``lengths`` is (B,) in samples."""
        if wav.dim() != 2:
            raise ValueError(f"frontend expects (B, S) mono audio, got {tuple(wav.shape)}")

        # 🔴 Zero everything past each row's own `lengths` before encoding.
        # `frames_for` rounds UP, so a row whose length is not a multiple of
        # `hop` has a last valid frame that is *part padding* -- and that frame
        # is masked IN, so whatever fills the pad reaches the score through it.
        # Measured on the stub: two pad fillings moved a submitted probability
        # by 1.35e-3 on rendered audio, and the same rows trimmed to whole
        # frames moved by exactly 0.
        #
        # ⚠️ A no-op on the shipped path: training and inference both pad with
        # zeros, so `wav * keep` changes no number we produce today. What it
        # changes is that "a row's features are a function of its own samples"
        # stops being incidental -- true only because everyone happens to pad
        # with zeros -- and becomes structural. Do not go looking for a metric
        # shift; there isn't one.
        #
        # It could not be seen before because every padding test in the suite
        # used `lengths = SR * 4`, an exact multiple of the 320-sample hop, so
        # no test had ever had a partial boundary frame. The pipeline draws
        # U(4, 60)s and renders arbitrary sample counts.
        if lengths is not None:
            keep = torch.arange(wav.shape[-1], device=wav.device)[None, :] < \
                lengths.to(wav.device)[:, None]
            wav = wav * keep.to(wav.dtype)
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

    🔴 Frames are **absolutely positioned**: the waveform is cut into fixed
    `hop`-sample frames, so frame *i* always covers samples [i*hop, (i+1)*hop)
    no matter how long the rest of the batch is. An earlier version pooled
    adaptively to the frame count, which *stretched* a short file's content
    across the whole padded width -- that makes the frame mask describe the
    wrong frames and quietly breaks rule-2.4 batch invariance. Real SSL
    frontends are strided convolutions and behave the absolute way; the stub
    must too, or it tests a property the real model will not have.
    """

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__(cfg, audio)
        self.n_freq = cfg.n_freq or 1
        self.hop = hop_length(audio.sample_rate, cfg.fps)
        width = 64
        self.frame = nn.Linear(self.hop, width)
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(width, self.n_freq * cfg.output_dim))

        # ⚠️ Refuse to look like we honoured a knob we did not. The stub has no
        # transformer layers to truncate and no attention projections to adapt,
        # so silently accepting these would let an ablation "measure" a setting
        # that never took effect.
        if cfg.layers is not None:
            raise NotImplementedError(
                "frontends: `layers` (truncation depth) has no meaning for the stub "
                "encoder and is not applied. It becomes real when a checkpoint is "
                "wired; until then set `layers: null`.")
        if cfg.adapter.kind != "none":
            raise NotImplementedError(
                f"frontends: adapter.kind={cfg.adapter.kind!r} is not implemented for the "
                "stub encoder -- there are no attention projections to adapt. Set "
                "`adapter: {kind: none}` for stub configs.")
        self._apply_freeze()

    def _encode(self, wav: Tensor) -> Tensor:
        b, n = wav.shape
        t = frames_for(n, self.audio.sample_rate, self.fps)
        pad = t * self.hop - n
        if pad:
            wav = F.pad(wav, (0, pad))
        h = self.frame(wav.view(b, t, self.hop))              # (B, T, width)
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
