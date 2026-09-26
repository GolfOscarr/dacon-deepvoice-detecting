"""Output heads.

One structure serves all five outputs: an SED (sound event detection) head that
makes a per-frame prediction and pools it with learned attention. The choice is
determined by our labels rather than by taste -- `FILE_FAKE = VOICE OR MUSIC`
over present components, and a component is FAKE if *any* generated segment is
present, so a fake component can occupy 3 seconds of a 60-second file and the
correct label is still FAKE. Whole-file mean pooling dilutes that by 20x.

See docs/architecture/04-heads-and-pooling.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.config import FreqPoolConfig, SEDHeadConfig

__all__ = ["FreqPool", "SEDHead", "SEDOutput"]


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) == y, for initialising the exponent."""
    import math
    return math.log(math.expm1(y))


class SEDOutput(dict):
    """What an SED head returns.

    A dict subclass so it survives `torch.jit`/`nn.Module` boundaries and stays
    printable, with attribute access for readability at call sites.

    ``clip_logits``  (B,)    attention-pooled over time
    ``frame_logits`` (B, T)  per-frame evidence, kept for the frame_max term,
                             for multi-resolution supervision, and because it is
                             the interpretability artifact the 2nd-stage report
                             is scored on
    ``attention``    (B, T)  where the model actually looked
    ``mask``         (B, T)  which frames are real
    ``embedding``    (B, hidden)  attention-pooled hidden, the OC-Softmax input,
    ``oc_center``    (hidden,)    and its centre; both present only when the head
                             has ``oc: true`` (training-only, docs/training/13 O1)

    🔴 ``mask`` is carried here rather than left to the caller because
    ``frame_logits`` on padded frames are arbitrary. A ``frame_max`` taken
    without it makes a file's submitted score depend on what was batched with
    it -- which is a rule 2.4 violation, and one that `clip_logits` alone does
    not expose, since attention already excludes padding.
    """

    __getattr__ = dict.__getitem__


class FreqPool(nn.Module):
    """Pooling over a frequency axis, for frontends that have one.

    ⚠️ Only patch-grid frontends (BEATs / EAT / SSLAM) keep a frequency axis.
    The wav2vec2 family emits (B, T, D) with frequency already collapsed, and
    must configure ``kind="none"`` -- in which case this module is not built at
    all. Conflating the two is easy and produces silently wrong shapes.

    GeM with a learnable ``p`` interpolates continuously between mean (p=1) and
    max (p -> inf), which lets the model settle the pooling question per head
    instead of us guessing once for all five (survey/10 G5). The learned values
    are themselves a reportable finding.
    """

    def __init__(self, cfg: FreqPoolConfig):
        super().__init__()
        self.kind = cfg.kind
        self.rectifier = cfg.rectifier
        if cfg.kind == "gem":
            # 🔴 Parameterised as p = 1 + softplus(raw), not as a raw value clamped
            # in forward. `p.clamp(min=1.0)` has exactly zero gradient in the
            # clamped region, so an exponent that ever drifted below 1.0 was
            # frozen there permanently -- measured: grad 0.0 at p_init=0.5.
            raw = torch.tensor(_inv_softplus(max(float(cfg.p_init) - 1.0, 1e-4)))
            self.raw_p = nn.Parameter(raw) if cfg.learnable else nn.Buffer(raw)
        self.eps = 1e-6

    @property
    def p(self) -> Tensor:
        """The effective exponent, always >= 1 and always differentiable."""
        return 1.0 + F.softplus(self.raw_p)

    def forward(self, x: Tensor) -> Tensor:
        """(B, F, T, D) -> (B, T, D)."""
        if x.dim() != 4:
            raise ValueError(f"FreqPool expects (B, F, T, D), got {tuple(x.shape)}")
        if self.kind == "mean":
            return x.mean(dim=1)
        if self.kind == "max":
            return x.amax(dim=1)
        if self.kind == "gem":
            p = self.p
            # A negative base with fractional p is NaN, so x must be made
            # positive first. 🔴 `clamp` does that by mapping every negative
            # entry to eps -- which on SSL hidden states discards ~50% of the
            # distribution and gives it exactly zero gradient. `softplus` is
            # monotone, positive and differentiable everywhere, so it costs no
            # information.
            # 🔴 The power mean runs in fp32 even under fp16 inference. At
            # p=3 any feature above ~40 overflows fp16 (40^3 = 64,000 against a
            # 65,504 max), inf survives the 1/p root, and the attention softmax
            # downstream becomes NaN. Both shipped configs pair
            # `precision: fp16` with `gem, p_init: 3.0`.
            dtype = x.dtype
            x32 = x.float()
            base = (F.softplus(x32) + self.eps if self.rectifier == "softplus"
                    else x32.clamp(min=self.eps))
            return base.pow(p.float()).mean(dim=1).pow(1.0 / p.float()).to(dtype)
        raise ValueError(f"FreqPool: nothing to do for kind={self.kind!r}")


class SEDHead(nn.Module):
    """Per-frame logits plus an attention-pooled clip logit.

    ``clip`` is the "overall character" operator and ``frame_max`` is the
    "any part of this file is fake" operator. Our label semantics decompose the
    same way, which is why both are produced and blended rather than one being
    chosen.

    🔴 The blend happens in **logit space**, in `models.outputs`. Averaging two
    *sigmoids* -- which is what the source recipe does -- is the exact
    construction that saturates to 0.0/1.0 in float32 and manufactures ties
    across files; the measured cost of saturation is EER 0.0950 -> 0.3017.
    This module therefore never applies a sigmoid.
    """

    def __init__(self, in_dim: int, cfg: SEDHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.dense = nn.Sequential(
            nn.Dropout(cfg.dropout_in),
            nn.Linear(in_dim, cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_out),
        )
        self.att = nn.Conv1d(cfg.hidden, 1, kernel_size=1)
        self.cla = nn.Conv1d(cfg.hidden, 1, kernel_size=1)
        # Only `scaled_tanh` needs a parameter: it keeps tanh's bounded shape but
        # lets the magnitude grow, so the softmax can still concentrate.
        self.att_scale = (nn.Parameter(torch.ones(())) if cfg.attention == "scaled_tanh"
                          else None)
        # O1: the one-class centre. Only built when asked for, so a flag-off head
        # has exactly the run-2 parameters and a run-2 checkpoint loads strictly.
        # Its init scale is irrelevant: the loss L2-normalises it.
        if cfg.oc:
            self.oc_center = nn.Parameter(torch.randn(cfg.hidden) / cfg.hidden ** 0.5)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> SEDOutput:
        """(B, T, D) -> clip (B,), frame (B, T), attention (B, T).

        ``mask`` is (B, T) with True on *valid* frames. Padded frames must be
        excluded from the attention softmax, or a batch of unequal-length files
        would let padding compete for attention mass -- which would also make
        the output depend on what else is in the batch, and rule 2.4 forbids
        that (docs/architecture/01 §1.4).
        """
        if x.dim() != 3:
            raise ValueError(f"SEDHead expects (B, T, D), got {tuple(x.shape)}")
        h = self.dense(x).transpose(1, 2)                 # (B, hidden, T)
        raw_att = self.att(h).squeeze(1)                   # (B, T)
        if self.cfg.attention == "linear":
            att_logits = raw_att
        elif self.cfg.attention == "scaled_tanh":
            att_logits = torch.tanh(raw_att) * self.att_scale.abs()
        else:                                              # "tanh" -- capped, legacy
            att_logits = torch.tanh(raw_att)
        frame_logits = self.cla(h).squeeze(1)              # (B, T)

        if mask is not None:
            if mask.shape != frame_logits.shape:
                raise ValueError(
                    f"SEDHead: mask {tuple(mask.shape)} does not match frames "
                    f"{tuple(frame_logits.shape)}")
            att_logits = att_logits.masked_fill(~mask, float("-inf"))

        if mask is None:
            mask = torch.ones_like(frame_logits, dtype=torch.bool)
        attention = torch.softmax(att_logits, dim=-1)
        clip_logits = (attention * frame_logits).sum(dim=-1)
        out = SEDOutput(clip_logits=clip_logits, frame_logits=frame_logits,
                        attention=attention, mask=mask)
        if self.cfg.oc:
            # Masked frames carry exactly 0 attention, so this is the sum over
            # valid frames only. Nothing on the inference path reads it.
            # The centre travels with the output (as the mask does) so the loss
            # needs no handle on the model and the training loop is unchanged.
            out["embedding"] = torch.einsum("bt,bht->bh", attention, h)
            out["oc_center"] = self.oc_center
        return out


def frame_max(frame_logits: Tensor, mask: Tensor | None = None) -> Tensor:
    """The "any part of this file is fake" operator, padding-safe."""
    if mask is not None:
        frame_logits = frame_logits.masked_fill(~mask, float("-inf"))
    return frame_logits.amax(dim=-1)
