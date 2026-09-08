"""The training objective.

🔴 The masks are not an optimisation -- they mirror the metric. Voice EER is
computed **only over voice-present files** using the organizers' own presence
labels, so a voice-fake loss on a music-only file trains the model to fit
something that will never be scored (docs/architecture/01 §3.3). PC-Mix does
exactly this masking with its component losses.

Each head can be trained on `clip`, on `frame_max`, or on a blend of the two,
via `SEDHeadConfig.clip_weight` -- the same field inference blends with, so the
two cannot diverge. 🔴 The default is **1.0 (clip only)**: supervising the
utterance and frame levels through one shared head measured 0.71-3.63 EER points
worse than utterance-only, and our head is that configuration because
`clip_logits` is a pooled function of the same `frame_logits` (docs/training/02
§3). ⚠️ Pending ablation T1.

⚠️ Note this is the *loss* blend; averaging the two losses and averaging the two
scores are different operations, and only the second one had the saturation
defect (see models.outputs).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.config import LossConfig, ModelConfig
from models.heads import frame_max

__all__ = ["TRAIN_CONSUMED_ELSEWHERE", "TARGET_FOR_COLUMN", "multitask_loss",
           "pairwise_ranking_loss"]

#: TrainConfig fields the loss deliberately does not read, with who owns them.
#: ⚠️ This exists because the ModelConfig-only version of the ignored-field guard
#: missed `LossConfig.frame_resolutions_ms`, which was accepted, defaulted,
#: round-tripped and read nowhere. The guard now walks TrainConfig too.
TRAIN_CONSUMED_ELSEWHERE = {
    "stage":        "the training script -- selects the S1..S3 schedule (S4 dropped)",
    "teachers":     "the training script -- frozen teachers, never shipped",
    "epochs":       "the training script",
    "batch_size":   "the training script",
    "lr":           "the training script",
    "weight_decay": "the training script",
    "precision":    "the training script",
    "seed":         "the training script",
}

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

    🔴 Normalising by `mask.sum()` rather than by batch size is load-bearing --
    but not for the reason an earlier version of this docstring gave. It claimed
    batch-size normalisation "would make a head's effective learning rate move
    with how many present-component files happened to land in the batch". Both
    schemes move with that count, in opposite directions: measured gradient norm
    scales as ~1/sqrt(n) under subset normalisation and ~sqrt(n) under batch
    normalisation (16x apart at n=2 in a batch of 32).

    What subset normalisation actually fixes is the *loss scale*: a head's
    contribution per batch is then independent of how prevalent its component
    is, so `LossConfig.weights` means what it says. Under batch normalisation
    the cell mix -- a sampler knob (docs/pipelines/02) -- would silently reweight
    the heads. The Freesound-2019 winner chose batch normalisation; we do not,
    for that reason.

    ⚠️ The cost is variance: at small `n` the per-sample influence is 1/n, so
    rare-component batches take large, noisy steps. Constraint C2 puts a floor
    on the per-head present-count per batch.
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

    🔴 Kept, but DEFAULT OFF (`LossConfig.ranking_weight = 0.0`), and the
    justification this docstring used to carry was wrong.

    It read: "EER is a pure ranking metric, so a pairwise loss optimises it
    directly -- independently arrived at by TFPARN and by the LLM-Detect-AI
    winner." TFPARN's own ablation refutes that for EER: adding the pairwise
    branch moves EER 12.91 -> 12.92 while lowering minDCF/Cllr/actDCF, and the
    paper says so outright -- "the ranking term acts on the decision cost and
    score ordering rather than on the equal-error point" (arXiv:2606.02980
    Table VI). Every gain it buys is calibration, which a ranking metric cannot
    read.

    A pairwise term is only worth something when it repairs BCE's vanishing
    gradient on negatives under severe class imbalance, and we choose our own
    class balance in the sampler instead (constraint C1, docs/pipelines/02).
    See docs/training/03 for the full reading.

    Returns 0 when a batch has only one class.
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
        # 🔴 The same blend weight inference uses, read from the head's own
        # config. A separate loss-side knob could drift from it silently, and
        # then the model would be trained on one objective and scored on another.
        cw = br_cfg.head.clip_weight
        per_sample = cw * clip_bce + (1.0 - cw) * frame_bce

        sample_mask = None if br_cfg.masked_by is None else targets[br_cfg.masked_by].bool()
        head_loss = _masked_mean(per_sample, sample_mask)

        if loss_cfg.ranking_weight:
            # 🔴 Rank the *blended* logit -- the quantity inference actually ranks
            # (models.outputs.branch_logit). Ranking `clip` alone would optimise
            # the ordering of half the submitted score, which defeats the point
            # of a loss whose entire justification is that EER is pure ranking.
            blended = cw * clip + (1.0 - cw) * fmax
            head_loss = head_loss + loss_cfg.ranking_weight * pairwise_ranking_loss(
                blended, y, sample_mask)

        # ⚠️ Changing the objective's *magnitude* -- these weights, the blend, a
        # new term -- can turn `tests/test_loop.py::
        # test_the_mid_epoch_checkpoint_carries_the_fp16_loss_scale` red for a
        # reason that has nothing to do with checkpointing. That test asserts
        # `_growth_tracker == steps` as its non-vacuity guard, and a single fp16
        # GradScaler backoff resets the tracker. The coupling is unintended and
        # undocumented at the test; if you land here from that failure, check for
        # an fp16 overflow in the loss before suspecting the checkpoint path.
        # 🔴 Indexed, not `.get(..., 1.0)`. The fallback made a partial
        # `weights` dict silently train the absent heads at 1.0 -- 20x the 0.05
        # a presence head is worth. `LossConfig` now requires all five keys, and
        # this reads them as required so the two cannot drift apart.
        weight = loss_cfg.weights[WEIGHT_KEY_FOR_COLUMN[br_cfg.column]]
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
