"""Small shared pieces that more than one module must agree on."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["align_time"]


def align_time(
    feats: Tensor,
    mask: Tensor,
    target_frames: int,
    fps_src: float,
    fps_tgt: float,
) -> tuple[Tensor, Tensor]:
    """Resample a frame sequence onto another frontend's time base.

    Two frontends with different `fps` produce different frame counts for the
    same audio, so the file branch cannot simply concatenate them
    (docs/architecture/03 -- "Ts and Ta will not match").

    🔴 The mapping is defined in **absolute time**, not as a ratio of the two
    padded frame counts. Target frame *j* is centred at ``(j + 0.5) / fps_tgt``
    seconds, which lands at source index ``(j + 0.5) * fps_src / fps_tgt - 0.5``
    regardless of how much padding the batch happens to carry.

    An earlier version used ``F.interpolate`` over the whole padded axis. That
    made the source→target mapping ``T_src / T_tgt`` -- a ratio of two
    independently-ceiled *padded* counts, which moves with the batch -- and let
    frames near the end of the valid region interpolate into padded source
    frames. Both make the submitted score depend on batch composition, which
    rule 2.4 forbids: measured drift was 3.5e-4 on FILE_FAKE_PROB at 9 s of
    padding, growing monotonically with the padding width.

    Interpolation is clamped per sample to that sample's own valid frames, so no
    padded source frame is ever read.
    """
    if feats.dim() != 3:
        raise ValueError(f"align_time expects (B, T, D), got {tuple(feats.shape)}")
    if target_frames < 1:
        raise ValueError(f"align_time: target_frames must be >= 1, got {target_frames}")
    if fps_src <= 0 or fps_tgt <= 0:
        raise ValueError(f"align_time: fps must be positive, got {fps_src}, {fps_tgt}")
    if feats.shape[1] == target_frames and fps_src == fps_tgt:
        return feats, mask

    b, t_src, d = feats.shape
    device = feats.device
    n_src = mask.sum(dim=-1).clamp(min=1)                          # (B,) valid source frames

    # docs/training/07 §2: an integer down-ratio (XLS-R 50 fps -> BEATs 6.25 fps
    # is 8) averages the source frames each target frame spans, over the
    # sample's own valid frames only, instead of sampling one frame in eight.
    ratio = fps_src / fps_tgt
    r = int(round(ratio))
    if r > 1 and abs(ratio - r) < 1e-9:
        width = target_frames * r
        valid = mask.to(feats.dtype)
        x = feats * valid.unsqueeze(-1)
        if t_src < width:
            x = torch.nn.functional.pad(x, (0, 0, 0, width - t_src))
            valid = torch.nn.functional.pad(valid, (0, width - t_src))
        x, valid = x[:, :width], valid[:, :width]
        num = x.reshape(b, target_frames, r, d).sum(dim=2)
        den = valid.reshape(b, target_frames, r).sum(dim=2)
        pooled = num / den.clamp(min=1.0).unsqueeze(-1)
        valid_tgt = torch.ceil(n_src.to(feats.dtype) / r).long()
        pooled_mask = (torch.arange(target_frames, device=device).unsqueeze(0)
                       < valid_tgt.clamp(max=target_frames).unsqueeze(-1))
        return pooled, pooled_mask

    # Absolute-time centres of the target frames, expressed in source indices.
    pos = (torch.arange(target_frames, device=device, dtype=feats.dtype) + 0.5) \
        * (fps_src / fps_tgt) - 0.5
    pos = pos.unsqueeze(0).expand(b, -1)                            # (B, T_tgt)

    # Never sample outside this sample's own valid region.
    hi = (n_src - 1).to(feats.dtype).unsqueeze(-1)
    pos = pos.clamp(min=0.0).minimum(hi)

    i0 = pos.floor().long().clamp(0, t_src - 1)
    i1 = (i0 + 1).clamp(max=t_src - 1).minimum((n_src - 1).long().unsqueeze(-1))
    w = (pos - i0.to(pos.dtype)).unsqueeze(-1)

    idx0 = i0.unsqueeze(-1).expand(-1, -1, d)
    idx1 = i1.unsqueeze(-1).expand(-1, -1, d)
    aligned = feats.gather(1, idx0) * (1.0 - w) + feats.gather(1, idx1) * w

    # A target frame is valid when its centre falls inside the real audio.
    valid_tgt = torch.ceil(n_src.to(pos.dtype) * fps_tgt / fps_src).long()
    aligned_mask = (torch.arange(target_frames, device=device).unsqueeze(0)
                    < valid_tgt.clamp(max=target_frames).unsqueeze(-1))
    return aligned, aligned_mask
