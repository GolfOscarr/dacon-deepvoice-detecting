"""Small shared pieces that more than one module must agree on."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["align_time"]


def align_time(feats: Tensor, mask: Tensor, target_frames: int) -> tuple[Tensor, Tensor]:
    """Resample a frame sequence onto another frontend's time base.

    Two frontends with different `fps` produce different frame counts for the
    same audio, so the file branch cannot simply concatenate them
    (docs/architecture/03 -- "Ts and Ta will not match"). Linear interpolation
    over the time axis is sufficient and is what `align_to` selects.

    ⚠️ The mask is resampled by nearest-neighbour, not linearly: a fractional
    "0.5 valid" frame is meaningless, and rounding it the wrong way would let
    padding back into the attention softmax.
    """
    if feats.dim() != 3:
        raise ValueError(f"align_time expects (B, T, D), got {tuple(feats.shape)}")
    if target_frames < 1:
        raise ValueError(f"align_time: target_frames must be >= 1, got {target_frames}")
    if feats.shape[1] == target_frames:
        return feats, mask

    aligned = F.interpolate(
        feats.transpose(1, 2), size=target_frames, mode="linear", align_corners=False
    ).transpose(1, 2)
    aligned_mask = (
        F.interpolate(mask.unsqueeze(1).float(), size=target_frames, mode="nearest")
        .squeeze(1)
        .bool()
    )
    return aligned, aligned_mask
