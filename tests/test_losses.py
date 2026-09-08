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

    # Derive the blend from the config rather than hardcoding it: clip_weight is
    # a tunable default (1.0 = clip-only, docs/training/02 §3) and this test is
    # about the *normaliser*, not the blend.
    cw = cfg.branches["voice"].head.clip_weight
    y = torch.tensor([1.0, 0.0])
    clip_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        out["voice"]["clip_logits"], y, reduction="none")
    fmax = out["voice"]["frame_logits"].amax(-1)
    frame_bce = torch.nn.functional.binary_cross_entropy_with_logits(fmax, y, reduction="none")
    per_sample = cw * clip_bce + (1.0 - cw) * frame_bce
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
    """The knob must work, and the default must be the metric weights.

    The metric weights File .45 / Music .27 / Voice .18 / presence .05 each. An
    earlier default weighted all five equally, inherited from PC-Mix whose metric
    weighted its components equally and ours does not (docs/training/02 §4).
    """
    model, cfg = _model()
    out = model(torch.randn(3, SR * 4))
    targets = _targets([1, 1, 1], [1, 1, 1], voice_fake=[1, 0, 1], music_fake=[0, 1, 1])

    assert LossConfig().weights == {
        "voice": 0.18, "music": 0.27, "file": 0.45, "v_pres": 0.05, "m_pres": 0.05}

    metric_shaped, _ = multitask_loss(out, targets, cfg, LossConfig())
    equal, _ = multitask_loss(out, targets, cfg, LossConfig(
        weights={"voice": 1.0, "music": 1.0, "file": 1.0, "v_pres": 1.0, "m_pres": 1.0}))
    assert float(equal) != pytest.approx(float(metric_shaped))


#: What the competition metric weights each submission column at
#: (metrics.dacon.SCORE_WEIGHTS, docs/validation/02 §1).
#:
#: 🔴 Written out here, column-keyed and by hand, on purpose. The two tests below
#: must not reach for `losses.WEIGHT_KEY_FOR_COLUMN` or `losses.TARGET_FOR_COLUMN`
#: to say what they expect: those mappings are the thing under test, and a test
#: that looks a weight up through the same mapping the loss uses survives every
#: permutation of it. `test_per_head_weights_are_applied` above checks the weights
#: *dict*; these check the weights *in effect*.
METRIC_WEIGHT_FOR_COLUMN = {
    "FILE_FAKE_PROB": 0.45,
    "MUSIC_FAKE_PROB": 0.27,
    "VOICE_FAKE_PROB": 0.18,
    "VOICE_PRESENT_PROB": 0.05,
    "MUSIC_PRESENT_PROB": 0.05,
}

#: The ground-truth key each column is scored against, likewise by hand.
METRIC_TARGET_FOR_COLUMN = {
    "FILE_FAKE_PROB": "file_fake",
    "MUSIC_FAKE_PROB": "music_fake",
    "VOICE_FAKE_PROB": "voice_fake",
    "VOICE_PRESENT_PROB": "voice_present",
    "MUSIC_PRESENT_PROB": "music_present",
}


def test_each_head_is_weighted_at_its_own_metric_weight():
    """🔴 The weight *in effect* per head, recovered without the mapping.

    `multitask_loss` walks `cfg.branches`, so restricting the config to one
    branch -- exactly what `training.stages._stage_loss_config` does for real --
    makes the returned total that branch's weighted contribution alone, while
    `parts[branch]` is the same head loss unweighted. Their ratio is the weight
    the objective actually applied to that head, and it is compared against the
    metric weight for the branch's own column.

    ⚠️ Every part of the head-to-weight path is then covered: permuting
    `WEIGHT_KEY_FOR_COLUMN` moves the ratio and not the expectation. Mutating it
    so both presence heads read `"file"` trains two 0.05 heads at 0.45; the whole
    suite passed that mutation before this test existed.
    """
    from metrics.dacon import SCORE_WEIGHTS
    assert sorted(METRIC_WEIGHT_FOR_COLUMN.values()) == sorted(SCORE_WEIGHTS.values()), \
        "the metric's head weights moved; this table is stale"

    model, cfg = _model()
    torch.manual_seed(3)
    out = model(torch.randn(4, SR * 4))
    targets = _targets([1, 1, 1, 0], [1, 0, 1, 1],
                       voice_fake=[1, 0, 1, 0], music_fake=[0, 1, 1, 0])

    assert set(br.column for br in cfg.branches.values()) == set(METRIC_WEIGHT_FOR_COLUMN)
    for name, br in cfg.branches.items():
        one_branch = dataclasses.replace(cfg, branches={name: br})
        total, parts = multitask_loss(out, targets, one_branch, LossConfig())
        # Non-vacuity: a zero head loss would make every weight look right.
        assert parts[name] > 1e-3, f"{name}: head loss too small to divide by"
        in_effect = float(total) / parts[name]
        assert in_effect == pytest.approx(METRIC_WEIGHT_FOR_COLUMN[br.column], rel=1e-4), \
            f"{name} ({br.column}) is trained at {in_effect}"


def test_each_head_is_trained_against_its_own_column_label():
    """🔴 The target *in effect* per head, recovered without the mapping.

    Companion to the weight test: permuting `TARGET_FOR_COLUMN` would train each
    head on a label the metric scores a different column against, and no shape,
    mask or weight assertion notices.

    The two presence keys are also the masks, so they are perturbed separately
    and only the two presence heads are checked against them -- flipping
    `voice_present` legitimately moves the voice head through its mask.
    """
    model, cfg = _model()
    torch.manual_seed(4)
    out = model(torch.randn(4, SR * 4))
    base = {"voice_present": torch.ones(4), "music_present": torch.ones(4),
            "voice_fake": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "music_fake": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "file_fake": torch.tensor([1.0, 1.0, 1.0, 0.0])}
    _, ref = multitask_loss(out, base, cfg, LossConfig())

    def moved(perturbed):
        _, got = multitask_loss(out, perturbed, cfg, LossConfig())
        return {n for n in cfg.branches
                if got[n] != pytest.approx(ref[n], abs=1e-6)}

    column_of = {n: br.column for n, br in cfg.branches.items()}

    # All masks stay all-ones here, so a fake label can only reach a head that is
    # trained against it.
    for key in ("voice_fake", "music_fake", "file_fake"):
        flipped = dict(base, **{key: 1.0 - base[key]})
        expected = {n for n, col in column_of.items()
                    if METRIC_TARGET_FOR_COLUMN[col] == key}
        assert moved(flipped) == expected, f"{key} reached the wrong head(s)"

    # Presence keys: check only the presence heads, whose masks are all null.
    presence_heads = {n for n, col in column_of.items()
                      if METRIC_TARGET_FOR_COLUMN[col] in ("voice_present", "music_present")}
    assert len(presence_heads) == 2
    for key in ("voice_present", "music_present"):
        one_absent = dict(base, **{key: torch.tensor([1.0, 1.0, 1.0, 0.0])})
        expected = {n for n, col in column_of.items()
                    if METRIC_TARGET_FOR_COLUMN[col] == key}
        assert moved(one_absent) & presence_heads == expected, \
            f"{key} reached the wrong presence head(s)"


def test_loss_blend_comes_from_the_head_config_not_a_separate_knob():
    """🔴 Training and inference must use the same clip/frame_max blend.

    A separate loss-side `frame_weight` existed and was documented as the
    frame-*supervision* weight of 04 §4 -- a different quantity -- so it could
    silently disagree with the head's `clip_weight`, training the model on one
    objective and scoring it on another.
    """
    model, cfg = _model()
    out = model(torch.randn(3, SR * 4))
    targets = _targets([1, 1, 1], [0, 0, 0], voice_fake=[1, 0, 1])

    def with_clip_weight(w):
        branches = {n: dataclasses.replace(b, head=dataclasses.replace(b.head, clip_weight=w))
                    for n, b in cfg.branches.items()}
        c = dataclasses.replace(cfg, branches=branches)
        return float(multitask_loss(out, targets, c, LossConfig())[0])

    assert with_clip_weight(0.0) != pytest.approx(with_clip_weight(1.0))

    # and it is the *same* number the output stage blends with
    from models.outputs import branch_logit
    br = next(iter(cfg.branches.values()))
    assert branch_logit(out["voice"], dataclasses.replace(br.head, clip_weight=1.0)).allclose(
        out["voice"]["clip_logits"])


def test_ranking_loss_ranks_what_inference_ranks():
    """Ranking `clip` alone would optimise the ordering of half the score."""
    from models.outputs import branch_logit

    model, cfg = _model()
    out = model(torch.randn(6, SR * 4))
    targets = _targets([1] * 6, [0] * 6, voice_fake=[1, 1, 1, 0, 0, 0])

    _, plain = multitask_loss(out, targets, cfg, LossConfig())
    _, ranked = multitask_loss(out, targets, cfg, LossConfig(ranking_weight=1.0))
    assert ranked["voice"] > plain["voice"], "the ranking term must contribute"

    # The term is computed on the blended logit, which is what branch_logit
    # returns. Force the blend on: clip_weight defaults to 1.0 (clip-only), where
    # "blend" and "clip" coincide and the assertion would be vacuous.
    br_head = dataclasses.replace(cfg.branches["voice"].head, clip_weight=0.5)
    blended = branch_logit(out["voice"], br_head)
    assert not torch.allclose(blended, out["voice"]["clip_logits"]), \
        "at clip_weight 0.5 the blend must differ from clip alone"


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


def test_loss_frame_max_ignores_padding_by_default():
    """The training signal must not come from padded-frame logits either."""
    model, cfg = _model()
    # ⚠️ `+ 1`, so the last valid frame is part padding. At `SR * 4` this guard
    # ran on 200 whole frames and never saw the boundary frame the pipeline's
    # U(4, 60)s durations produce on almost every sample.
    n = SR * 4 + 1
    x = torch.randn(1, n)
    lengths = torch.tensor([n])
    targets = _targets([1], [1], voice_fake=[1.0], music_fake=[0.0])

    quiet = model(torch.nn.functional.pad(x, (0, SR * 4)), lengths)
    loud = model(torch.cat([x, torch.randn(1, SR * 4) * 50], dim=-1), lengths)
    _, a = multitask_loss(quiet, targets, cfg, LossConfig())
    _, b = multitask_loss(loud, targets, cfg, LossConfig())
    for head in ("voice", "music", "file"):
        assert a[head] == pytest.approx(b[head], abs=1e-6), head


def test_no_train_config_field_is_silently_ignored():
    """🔴 The ModelConfig-only guard missed LossConfig.frame_resolutions_ms.

    It was accepted, defaulted, round-tripped and read nowhere, so a config
    could request multi-resolution frame supervision that does not exist.
    """
    import pathlib
    import re

    from models.config import load_train_config
    from models.losses import TRAIN_CONSUMED_ELSEWHERE

    src = "\n".join(
        (pathlib.Path("models") / f"{m}.py").read_text()
        for m in ("model", "frontends", "heads", "losses", "outputs", "utils", "audio"))
    cfg = load_train_config("configs/train_joint.yaml")

    ignored = []
    for f in dataclasses.fields(cfg):
        if f.name in TRAIN_CONSUMED_ELSEWHERE:
            continue
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            for sub in dataclasses.fields(value):
                if not re.search(rf"\.{re.escape(sub.name)}\b", src):
                    ignored.append(f"{f.name}.{sub.name}")
        elif not re.search(rf"\.{re.escape(f.name)}\b", src):
            ignored.append(f.name)

    assert not ignored, (
        f"TrainConfig fields neither read nor allowlisted: {ignored}")


def test_gem_power_mean_survives_fp16():
    """Both shipped configs pair `precision: fp16` with `gem, p_init: 3.0`.

    At p=3 any feature above ~40 overflows fp16 (40**3 = 64,000 against a
    65,504 max), inf survives the 1/p root, and the attention softmax downstream
    becomes NaN.
    """
    from models.config import FreqPoolConfig
    from models.heads import FreqPool

    pool = FreqPool(FreqPoolConfig(kind="gem", p_init=3.0)).half()
    for scale in (65, 200, 1000):
        out = pool((torch.rand(1, 4, 3, 4) * scale).half())
        assert torch.isfinite(out).all(), f"overflowed at feature scale {scale}"
        assert out.dtype == torch.float16


def test_gem_exponent_can_recover_from_below_one():
    """`p.clamp(min=1.0)` had exactly zero gradient in the clamped region, so an
    exponent that drifted below 1.0 was frozen there permanently."""
    from models.config import FreqPoolConfig
    from models.heads import FreqPool

    pool = FreqPool(FreqPoolConfig(kind="gem", p_init=0.5, learnable=True))
    assert float(pool.p) >= 1.0, "the effective exponent must stay >= 1"
    pool(torch.rand(2, 4, 5, 3) + 0.1).sum().backward()
    assert float(pool.raw_p.grad) != 0.0, "a frozen exponent can never recover"
