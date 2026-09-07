"""The masking is the part most likely to be subtly wrong, and it is the part
that mirrors the metric, so it gets the most tests."""

import dataclasses

import pytest
import torch

from models.config import LossConfig, load_model_config
from models.losses import multitask_loss, pairwise_ranking_loss
from models.model import DeepVoiceNet

SR = 16_000


def _model(name="b_stub", **over):
    cfg = load_model_config(f"configs/{name}.yaml")
    if over:
        cfg = dataclasses.replace(cfg, **over)
    return DeepVoiceNet(cfg).eval(), cfg


def _targets(voice_present, music_present, voice_fake=None, music_fake=None):
    vp = torch.tensor(voice_present, dtype=torch.float32)
    mp = torch.tensor(music_present, dtype=torch.float32)
    vf = torch.tensor(voice_fake if voice_fake is not None else [0.0] * len(vp))
    mf = torch.tensor(music_fake if music_fake is not None else [0.0] * len(mp))
    return {"voice_present": vp, "music_present": mp, "voice_fake": vf,
            "music_fake": mf, "file_fake": ((vp * vf) + (mp * mf)).clamp(max=1)}


# --------------------------------------------------------------------------- #
# masking mirrors the metric

def test_music_only_file_contributes_nothing_to_the_voice_loss():
    """🔴 Voice EER is computed only over voice-present files."""
    model, cfg = _model()
    torch.manual_seed(0)
    wav = torch.randn(4, SR * 4)
    out = model(wav)

    base = _targets([1, 1, 0, 0], [0, 0, 1, 1], voice_fake=[1, 0, 0, 0])
    flipped = dict(base)
    flipped["voice_fake"] = torch.tensor([1.0, 0.0, 1.0, 1.0])   # only music-only files change

    _, a = multitask_loss(out, base, cfg, LossConfig())
    _, b = multitask_loss(out, flipped, cfg, LossConfig())
    assert a["voice"] == pytest.approx(b["voice"], abs=1e-6)


def test_unmasked_head_does_see_every_file():
    model, cfg = _model()
    out = model(torch.randn(4, SR * 4))
    base = _targets([1, 1, 0, 0], [0, 0, 1, 1])
    flipped = dict(base)
    flipped["file_fake"] = 1.0 - base["file_fake"]
    _, a = multitask_loss(out, base, cfg, LossConfig())
    _, b = multitask_loss(out, flipped, cfg, LossConfig())
    assert a["file"] != pytest.approx(b["file"], abs=1e-6)


def test_masked_mean_normalises_by_mask_not_batch_size():
    """Otherwise a head's effective learning rate moves with batch composition."""
    model, cfg = _model()
    torch.manual_seed(1)
    wav = torch.randn(2, SR * 4)
    out = model(wav)

    both = _targets([1, 1], [0, 0], voice_fake=[1, 0])
    _, full = multitask_loss(out, both, cfg, LossConfig())

    # Same two files, but only the first counts as voice-present.
    one = _targets([1, 0], [0, 1], voice_fake=[1, 0])
    _, half = multitask_loss(out, one, cfg, LossConfig())

    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        out["voice"]["clip_logits"], torch.tensor([1.0, 0.0]), reduction="none")
    fmax = out["voice"]["frame_logits"].amax(-1)
    per_sample = 0.5 * per_sample + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        fmax, torch.tensor([1.0, 0.0]), reduction="none")
    assert half["voice"] == pytest.approx(float(per_sample[0]), abs=1e-5)
    assert full["voice"] == pytest.approx(float(per_sample.mean()), abs=1e-5)


def test_batch_with_no_present_component_is_not_nan():
    model, cfg = _model()
    out = model(torch.randn(2, SR * 4))
    targets = _targets([0, 0], [1, 1])          # no voice anywhere
    total, parts = multitask_loss(out, targets, cfg, LossConfig())
    assert torch.isfinite(total)
    assert parts["voice"] == 0.0


# --------------------------------------------------------------------------- #
# weights and blend

def test_per_head_weights_are_applied():
    """The metric weights File .45 / Music .27 / Voice .18 while the default
    config weights them equally -- so this knob must actually work (09 B11)."""
    model, cfg = _model()
    out = model(torch.randn(3, SR * 4))
    targets = _targets([1, 1, 1], [1, 1, 1], voice_fake=[1, 0, 1], music_fake=[0, 1, 1])

    equal, _ = multitask_loss(out, targets, cfg, LossConfig())
    metric_shaped, _ = multitask_loss(out, targets, cfg, LossConfig(
        weights={"voice": 0.18, "music": 0.27, "file": 0.45, "v_pres": 0.05, "m_pres": 0.05}))
    assert float(equal) != pytest.approx(float(metric_shaped))


def test_frame_weight_moves_the_objective():
    model, cfg = _model()
    out = model(torch.randn(3, SR * 4))
    targets = _targets([1, 1, 1], [0, 0, 0], voice_fake=[1, 0, 1])
    clip_only, _ = multitask_loss(out, targets, cfg, LossConfig(frame_weight=0.0))
    frame_only, _ = multitask_loss(out, targets, cfg, LossConfig(frame_weight=1.0))
    assert float(clip_only) != pytest.approx(float(frame_only))


def test_loss_is_differentiable():
    model, cfg = _model()
    out = model(torch.randn(2, SR * 4))
    targets = _targets([1, 1], [1, 1], voice_fake=[1, 0], music_fake=[0, 1])
    total, _ = multitask_loss(out, targets, cfg, LossConfig())
    total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_missing_target_is_an_error():
    model, cfg = _model()
    out = model(torch.randn(2, SR * 4))
    targets = _targets([1, 1], [1, 1])
    del targets["music_fake"]
    with pytest.raises(KeyError, match="music_fake"):
        multitask_loss(out, targets, cfg, LossConfig())


# --------------------------------------------------------------------------- #
# ranking and distillation

def test_pairwise_ranking_rewards_correct_ordering():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = pairwise_ranking_loss(torch.tensor([5.0, 4.0, -4.0, -5.0]), labels)
    bad = pairwise_ranking_loss(torch.tensor([-5.0, -4.0, 4.0, 5.0]), labels)
    assert good < bad


def test_pairwise_ranking_is_zero_for_a_single_class():
    out = pairwise_ranking_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
    assert float(out) == 0.0


def test_distillation_requires_a_distill_head():
    model, cfg = _model("a_stub")            # distill disabled in A
    out = model(torch.randn(2, SR * 4))
    targets = _targets([1, 1], [1, 1])
    with pytest.raises(KeyError, match="distill head"):
        multitask_loss(out, targets, cfg, LossConfig(), teacher_emb=torch.randn(2, 1024))


def test_distillation_contributes_when_enabled():
    model, cfg = _model("b_stub")            # distill enabled in B
    out = model(torch.randn(2, SR * 4))
    targets = _targets([1, 1], [1, 1])
    _, parts = multitask_loss(out, targets, cfg, LossConfig(),
                              teacher_emb=torch.randn(2, cfg.distill.embed_dim))
    assert parts["distill"] > 0
