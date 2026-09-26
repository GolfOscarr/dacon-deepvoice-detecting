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
           "oc_softmax_loss", "pairwise_ranking_loss", "parts_to_floats"]

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


def oc_softmax_loss(embedding: Tensor, center: Tensor, labels: Tensor, alpha: float,
                    m_real: float, m_fake: float) -> Tensor:
    """Per-sample OC-Softmax (Zhang, Jiang & Duan 2021), shape (B,).

    ``s = cos(w, x)`` with both L2-normalised; real (label 0) pays
    ``softplus(alpha * (m_real - s))``, fake pays ``softplus(alpha * (s - m_fake))``.
    Real is the compact class: an unseen fake only has to be "not real"
    (docs/training/13 O1). Computed in fp32 whatever the autocast dtype -- at
    alpha=20 a bf16 cosine is off by up to ~0.08 in the exponent.
    """
    x = F.normalize(embedding.float(), dim=-1)
    w = F.normalize(center.float(), dim=-1)
    s = x @ w
    fake = labels.float() > 0.5
    margin = torch.where(fake, s - m_fake, m_real - s)
    return F.softplus(alpha * margin)


def multitask_loss(
    outputs: dict,
    targets: dict[str, Tensor],
    cfg: ModelConfig,
    loss_cfg: LossConfig,
    frame_masks: dict[str, Tensor] | None = None,
    teacher_emb: Tensor | None = None,
    tensor_parts: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Total loss and a per-part breakdown for the experiment ledger.

    ``targets`` needs the five keys in TARGET_FOR_COLUMN, each (B,) float/int.
    ``outputs`` is what DeepVoiceNet.forward returned.

    ``parts`` carries one **loss** key per branch, plus ``total`` and, when a
    teacher is supplied, ``distill``. Keys containing a ``/`` are **diagnostics,
    not loss terms**: ``<branch>/p_c`` is the fraction of the batch carrying that
    branch's component and ``<branch>/w_eff`` the `w_c / p_c` of docs/training/02
    §4. Averaging them over a pass, as `training.loop._mean_parts` does, is the
    intended reading. ``<branch>/oc`` (heads with ``oc: true`` only) is the raw
    OC-Softmax loss; ``oc_weight`` times it is inside ``total``, not ``<branch>``.

    ``tensor_parts=True`` leaves every part a detached 0-d tensor on the loss's
    device, for `parts_to_floats` to convert later: each ``float(...)`` here is
    a device sync, eleven per training step, and the loop only reads the parts
    every `log_every` steps. The converted values are the same floats.
    """
    missing = [k for k in TARGET_FOR_COLUMN.values() if k not in targets]
    if missing:
        raise KeyError(f"multitask_loss: missing target(s) {missing}")

    total = None
    parts: dict[str, float] = {}

    for branch, br_cfg in cfg.branches.items():
        target_key = TARGET_FOR_COLUMN[br_cfg.column]
        y = targets[target_key].float()
        y_hard = y                      # the OC term takes no label smoothing
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
        parts[branch] = head_loss.detach() if tensor_parts else float(head_loss.detach())
        # 🔴 The standing diagnostic docs/training/02 §4 commits to. `_masked_mean`
        # divides by the present-count, so the per-sample weight a masked head
        # exerts on the shared trunk is `w_c / p_c`, not `w_c` -- and §4 says
        # outright not to tune `w_c` without looking at it. It was recorded
        # nowhere, which made T2 unreadable as specified. `p_c` is 1.0 for an
        # unmasked head, and `w_eff` is 0 when the batch carries the component
        # nowhere, because then the head contributes nothing at all.
        if tensor_parts and sample_mask is not None:
            # float64 on the device: the same IEEE division as the float path.
            p_c64 = sample_mask.float().mean().double()
            parts[f"{branch}/p_c"] = p_c64
            parts[f"{branch}/w_eff"] = torch.where(
                p_c64 > 0, torch.full_like(p_c64, weight) / p_c64, torch.zeros_like(p_c64))
        else:
            p_c = 1.0 if sample_mask is None else float(sample_mask.float().mean())
            parts[f"{branch}/p_c"] = p_c
            parts[f"{branch}/w_eff"] = weight / p_c if p_c else 0.0
        # O1: only for a head built with `oc`, so a flag-off model adds no op to
        # the graph and its total is bitwise run 2's. At oc_weight 0 the term is
        # still added (x 0) so `oc_center` stays in the graph -- DDP deadlocks on
        # unused params. Same mask as the BCE term; logged raw (unweighted).
        if br_cfg.head.oc:
            if "embedding" not in out or "oc_center" not in out:
                raise KeyError(
                    f"multitask_loss: branch {branch!r} has head.oc but its output "
                    "carries no embedding / oc_center")
            oc = _masked_mean(oc_softmax_loss(
                out["embedding"], out["oc_center"], y_hard, loss_cfg.oc_alpha,
                loss_cfg.oc_m_real, loss_cfg.oc_m_fake), sample_mask)
            contribution = contribution + loss_cfg.oc_weight * oc
            parts[f"{branch}/oc"] = oc.detach() if tensor_parts else float(oc.detach())
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
        parts["distill"] = distill.detach() if tensor_parts else float(distill.detach())
        total = total + loss_cfg.distill_weight * distill

    parts["total"] = total.detach() if tensor_parts else float(total.detach())
    return total, parts


def parts_to_floats(parts: list[dict]) -> list[dict[str, float]]:
    """`multitask_loss(..., tensor_parts=True)` parts -> plain floats, with one
    device sync for the whole list rather than one per value."""
    tensors = [v for p in parts for v in p.values() if isinstance(v, Tensor)]
    if not tensors:
        return [dict(p) for p in parts]
    values = iter(torch.stack([t.double() for t in tensors]).tolist())
    return [{k: (next(values) if isinstance(v, Tensor) else v) for k, v in p.items()}
            for p in parts]
