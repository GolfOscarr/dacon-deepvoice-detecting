"""Turning branch logits into the five submission numbers.

This module exists because of one measured result: **saturation, not rounding,
is the output risk.** Rounding predictions to 2 decimal places is harmless,
while saturating the operating point took EER 0.0950 -> 0.3017
(PROGRESS.md, scripts/verify_metric.py).

Two consequences, both enforced here rather than left to convention:

* **Blend in logit space.** The source recipe averages two *sigmoids*; that is
  the construction that collapses to 0.0/1.0 and manufactures ties across
  files. We blend the logits and squash once.
* **Squash without saturating.** In float64 a sigmoid reaches exactly 1.0 by
  z ~= 37, and every file above that becomes an unbreakable tie. `softsign`
  holds out to |z| ~ 1e16. Ties cost ranking, and rank normalisation -- the
  usual fix -- is forbidden by rule 2.4.
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.config import AggregationConfig, OutputConfig, SEDHeadConfig
from models.heads import SEDOutput, frame_max

__all__ = ["aggregate_windows", "blend_logits", "branch_logit", "to_probability"]


def blend_logits(clip_logits: Tensor, frame_max_logits: Tensor, clip_weight: float) -> Tensor:
    """`w*clip + (1-w)*frame_max`, in logit space.

    `clip` is the "overall character" operator, `frame_max` the "any part of
    this file is fake" operator. Our labels decompose the same way, which is why
    both are produced and blended rather than one being chosen.
    """
    if not 0.0 <= clip_weight <= 1.0:
        raise ValueError(f"clip_weight must be in [0, 1], got {clip_weight}")
    return clip_weight * clip_logits + (1.0 - clip_weight) * frame_max_logits


def branch_logit(out: SEDOutput, head_cfg: SEDHeadConfig, mask: Tensor | None = None) -> Tensor:
    """One branch's blended file-level logit.

    🔴 Defaults to the mask the head recorded. Passing nothing used to mean
    "frame_max over every frame including padding", which made the submitted
    number depend on batch composition even though `clip_logits` did not.
    """
    if mask is None:
        mask = out.get("mask")
    return blend_logits(out["clip_logits"],
                        frame_max(out["frame_logits"], mask),
                        head_cfg.clip_weight)


def to_probability(logits: Tensor, cfg: OutputConfig) -> Tensor:
    """Monotone map into [0, 1], in float64, chosen not to saturate.

    Monotonicity is all the metric needs -- EER and ROC-AUC read only the
    ordering -- so the only job here is to land inside [0, 1] without losing
    resolution near the decision region.
    """
    z = logits.to(torch.float64) * cfg.logit_scale
    if cfg.squash == "sigmoid":
        p = torch.sigmoid(z)
    elif cfg.squash == "softsign":
        # 0.5 * (1 + z/(1+|z|)): strictly increasing, and it does not reach an
        # exact 0.0/1.0 until |z| ~ 1e16, versus ~37 for a float64 sigmoid.
        p = 0.5 * (1.0 + z / (1.0 + z.abs()))
    else:
        raise ValueError(f"unknown squash {cfg.squash!r}")
    return p.clamp(cfg.clamp_eps, 1.0 - cfg.clamp_eps)


def aggregate_windows(
    window_logits: Tensor,
    cfg: AggregationConfig,
    mask: Tensor | None = None,
) -> Tensor:
    """Pool per-window logits into one file logit. (B, W) -> (B,).

    🔴 `max` is available but is not the default, because it is stochastically
    larger the more windows a file has. Durations span 4-60 s, so a plain max
    makes long files score higher than short ones regardless of content -- a
    duration bias landing directly on a ranking metric
    (docs/architecture/04 §6.1). `topk_mean` with fixed k depends on the *k most
    suspicious* windows rather than on how many windows exist.

    ⚠️ Whichever is used, duration-vs-score correlation on the REAL class must
    be checked explicitly; it is not something this function can guarantee.
    """
    if window_logits.dim() != 2:
        raise ValueError(f"expected (B, W) window logits, got {tuple(window_logits.shape)}")
    if mask is None:
        mask = torch.ones_like(window_logits, dtype=torch.bool)
    if not mask.any(dim=-1).all():
        raise ValueError("aggregate_windows: every file needs at least one valid window")

    neg_inf = torch.finfo(window_logits.dtype).min
    valid = window_logits.masked_fill(~mask, neg_inf)

    if cfg.kind == "max":
        return valid.amax(dim=-1)
    if cfg.kind == "mean":
        return (window_logits * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    if cfg.kind == "topk_mean":
        # k is capped per file so a 4 s file with one window is not averaged
        # against padding -- which would reintroduce the duration dependence.
        n = mask.sum(-1)
        k = int(min(cfg.k, int(n.max().item())))
        top = valid.topk(k, dim=-1).values
        keep = torch.arange(k, device=n.device)[None, :] < n[:, None].clamp(max=k)
        return (top * keep).sum(-1) / keep.sum(-1).clamp(min=1)
    if cfg.kind == "quantile":
        out = []
        for row, m in zip(window_logits, mask):
            out.append(torch.quantile(row[m].double(), cfg.quantile))
        return torch.stack(out).to(window_logits.dtype)
    if cfg.kind == "confidence_gated":
        # ☆ [DFDC 2020, 1st]: mean of the confident tail when there is one,
        # else the plain mean. Operates on logits, so the 0.8/0.2 probability
        # gates of the original become +/- logit(0.8) ~= 1.386.
        hi, lo = 1.3863, -1.3863
        strong = mask & (window_logits > hi)
        weak = mask & (window_logits < lo)
        n = mask.sum(-1).clamp(min=1)
        mean = (window_logits * mask).sum(-1) / n
        strong_mean = (window_logits * strong).sum(-1) / strong.sum(-1).clamp(min=1)
        weak_mean = (window_logits * weak).sum(-1) / weak.sum(-1).clamp(min=1)
        use_strong = strong.sum(-1) > (0.4 * n)
        use_weak = (~use_strong) & (weak.sum(-1) > (0.9 * n))
        return torch.where(use_strong, strong_mean, torch.where(use_weak, weak_mean, mean))
    raise ValueError(f"unknown aggregation {cfg.kind!r}")
