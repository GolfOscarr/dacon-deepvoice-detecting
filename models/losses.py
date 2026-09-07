"""The training objective.

🔴 The masks are not an optimisation -- they mirror the metric. Voice EER is
computed **only over voice-present files** using the organizers' own presence
labels, so a voice-fake loss on a music-only file trains the model to fit
something that will never be scored (docs/architecture/01 §3.3). PC-Mix does
exactly this masking with its component losses.

Each head is trained on both `clip` and `frame_max`, because `frame_max` is the
"any part of this file is fake" operator and `clip` the "overall character"
one, and our labels decompose the same way. ⚠️ Note this is the *loss* blend;
averaging the two losses and averaging the two scores are different operations,
and only the second one had the saturation defect (see models.outputs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.config import LossConfig, ModelConfig
from models.heads import frame_max

__all__ = ["TARGET_FOR_COLUMN", "multitask_loss", "pairwise_ranking_loss"]

#: Which ground-truth key each submission column is trained against. Keyed by
#: column rather than by branch name, because the column is what config
#: validation already checks against metrics.dacon.PREDICTION_COLUMNS.
TARGET_FOR_COLUMN = {
    "VOICE_FAKE_PROB": "voice_fake",
    "MUSIC_FAKE_PROB": "music_fake",
    "FILE_FAKE_PROB": "file_fake",
    "VOICE_PRESENT_PROB": "voice_present",
    "MUSIC_PRESENT_PROB": "music_present",
}

#: Head short-name used for per-head loss weights, keyed by column.
WEIGHT_KEY_FOR_COLUMN = {
    "VOICE_FAKE_PROB": "voice",
    "MUSIC_FAKE_PROB": "music",
    "FILE_FAKE_PROB": "file",
    "VOICE_PRESENT_PROB": "v_pres",
    "MUSIC_PRESENT_PROB": "m_pres",
}


def _masked_mean(per_sample: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over the *masked* samples.

    🔴 Normalising by `mask.sum()` rather than by batch size is load-bearing.
    Dividing by the batch size would make a head's effective learning rate move
    with how many present-component files happened to land in the batch, which
    turns batch composition into a silent hyperparameter.
    """
    if mask is None:
        return per_sample.mean()
    mask = mask.to(per_sample.dtype)
    denom = mask.sum()
    if denom == 0:
        # No file in this batch carries the component. Contribute nothing, but
        # keep the graph connected so DDP does not deadlock on unused params.
        return per_sample.sum() * 0.0
    return (per_sample * mask).sum() / denom


def pairwise_ranking_loss(scores: Tensor, labels: Tensor, mask: Tensor | None = None) -> Tensor:
    """RankNet-style logistic loss over positive/negative pairs.

    EER is a pure ranking metric, so a pairwise loss optimises it directly --
    independently arrived at by TFPARN and by the LLM-Detect-AI winner
    (docs/papers/09). Returns 0 when a batch has only one class.
    """
    if mask is not None:
        keep = mask.bool()
        scores, labels = scores[keep], labels[keep]
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    return F.softplus(-(pos[:, None] - neg[None, :])).mean()


def multitask_loss(
    outputs: dict,
    targets: dict[str, Tensor],
    cfg: ModelConfig,
    loss_cfg: LossConfig,
    frame_masks: dict[str, Tensor] | None = None,
    teacher_emb: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Total loss and a per-part breakdown for the experiment ledger.

    ``targets`` needs the five keys in TARGET_FOR_COLUMN, each (B,) float/int.
    ``outputs`` is what DeepVoiceNet.forward returned.
    """
    missing = [k for k in TARGET_FOR_COLUMN.values() if k not in targets]
    if missing:
        raise KeyError(f"multitask_loss: missing target(s) {missing}")

    total = None
    parts: dict[str, float] = {}

    for branch, br_cfg in cfg.branches.items():
        target_key = TARGET_FOR_COLUMN[br_cfg.column]
        y = targets[target_key].float()
        if loss_cfg.label_smoothing:
            eps = loss_cfg.label_smoothing
            y = y * (1 - eps) + 0.5 * eps

        out = outputs[branch]
        # Same reason as models.outputs.branch_logit: an unmasked frame_max
        # trains against arbitrary padded-frame logits.
        fmask = (frame_masks or {}).get(branch, out.get("mask"))
        clip = out["clip_logits"]
        fmax = frame_max(out["frame_logits"], fmask)

        clip_bce = F.binary_cross_entropy_with_logits(clip, y, reduction="none")
        frame_bce = F.binary_cross_entropy_with_logits(fmax, y, reduction="none")
        fw = loss_cfg.frame_weight
        per_sample = (1.0 - fw) * clip_bce + fw * frame_bce

        sample_mask = None if br_cfg.masked_by is None else targets[br_cfg.masked_by].bool()
        head_loss = _masked_mean(per_sample, sample_mask)

        if loss_cfg.ranking_weight:
            head_loss = head_loss + loss_cfg.ranking_weight * pairwise_ranking_loss(
                clip, y, sample_mask)

        weight = loss_cfg.weights.get(WEIGHT_KEY_FOR_COLUMN[br_cfg.column], 1.0)
        contribution = weight * head_loss
        parts[branch] = float(head_loss.detach())
        total = contribution if total is None else total + contribution

    if teacher_emb is not None:
        aux = outputs.get("_aux", {})
        if "distill_emb" not in aux:
            raise KeyError(
                "multitask_loss: a teacher embedding was supplied but the model has no "
                "distill head; set distill.enabled: true in the config")
        # 🔴 The stop-gradient is on the *teacher*, which is frozen anyway; the
        # student-side detach that the source recipe applies lives in the model,
        # not here (docs/architecture/06 §3 -- and see 09 B9, which flags that a
        # frozen frontend weakens the mechanism this borrows).
        distill = F.mse_loss(aux["distill_emb"], teacher_emb.detach())
        parts["distill"] = float(distill.detach())
        total = total + loss_cfg.distill_weight * distill

    parts["total"] = float(total.detach())
    return total, parts
