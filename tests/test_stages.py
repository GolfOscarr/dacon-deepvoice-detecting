"""`training.stages` -- the S1/S2/S3 schedule, S3's codec expansion, precision.

The two claims worth reading first are the ones with a mutation attached. S1 is
"each branch alone on frozen frontends", and both halves are asserted on the
*weights* rather than on the loss, because restricting the loss still lets
AdamW's decay and momentum move a branch nobody is training. And the codec
expansion's label-independence is asserted as an exact cross product, with the
known leak shape -- `container = mp3 if fake else wav` -- run through I1b to
prove the audit can see it.

Where a claim is statistical it sweeps seeds: I1b is an estimated AUC and four
earlier rounds of flakiness in this repo all traced to a single hard-coded seed.
"""

import dataclasses
from pathlib import Path

import pytest
import torch

from loop_fixtures import (HAS_FFMPEG, DRAW, _dataset, _flat, _model, _same,
                           _train_cfg, _unfrozen, corpus, model_cfg)
from models.config import load_train_config
from models.model import DeepVoiceNet
from training.audit import audit_specs
from training.loop import LoopConfig, train_stage
from training.render import render
from training.sampler import Sampler
from training.stages import (CODEC_VARIANTS, STAGES, autocast_for,
                             codec_variant_specs, stage_plan,
                             trainable_parameters)


# --------------------------------------------------------------------------- #
# 1. The stage schedule


def test_the_dropped_stage_is_refused_rather_than_aliased(model_cfg):
    """S4 is dropped. Running S2 under its name would report a stage that
    was never trained -- the same class of defect as a config knob that
    validates and then does nothing (docs/training/04 §4)."""
    with pytest.raises(NotImplementedError, match="DROPPED"):
        stage_plan("rank_polish", model_cfg)
    assert "rank_polish" not in STAGES
    # And `models.config` still accepts it, which is why the refusal has to
    # live here rather than being assumed upstream.
    assert load_train_config("configs/train_joint.yaml").stage in (
        "independent", "joint", "codec_aware", "rank_polish")


def test_the_three_stages_differ_in_the_two_ways_the_schedule_says(model_cfg):
    s1, s2, s3 = (stage_plan(s, model_cfg) for s in STAGES)
    # S1: one branch at a time, frontends frozen.
    assert len(s1.branch_groups) == len(model_cfg.branches)
    assert all(len(g) == 1 for g in s1.branch_groups)
    assert not s1.train_frontends
    # S2: everything together.
    assert s2.branch_groups == (tuple(model_cfg.branches),)
    assert s2.train_frontends
    # S3: S2 plus the 4-way codec expansion, and nothing else.
    assert s3.branch_groups == s2.branch_groups
    assert len(s3.codec_variants) == 4 and len(s2.codec_variants) == 1


def test_s1_trains_one_branch_and_leaves_the_others_bitwise_unchanged(corpus, model_cfg):
    """S1 is "each branch alone". Asserted on the *weights*, not on the loss.

    Restricting the loss alone still lets AdamW's weight decay and momentum move
    a branch nobody is training this pass, so "alone" would be a claim about the
    objective rather than a fact about the run.
    """
    model = _model(model_cfg)
    before = _flat(model)
    plan = stage_plan("independent", model_cfg)
    params = trainable_parameters(model, plan, ("voice",))
    ids = {id(p) for p in params}

    assert ids == {id(p) for p in model.heads["voice"].parameters()}
    for other in model_cfg.branches:
        if other == "voice":
            continue
        assert not (ids & {id(p) for p in model.heads[other].parameters()}), other
    assert not (ids & {id(p) for p in model.frontends.parameters()})
    assert _same(before, _flat(model))            # selection alone moved nothing


def test_s1_end_to_end_moves_only_the_active_branches(corpus, model_cfg):
    """The same statement after real optimizer steps, which is where it can break."""
    torch.manual_seed(0)
    model = _model(model_cfg)
    before = _flat(model)
    ds = _dataset(corpus, n=4)
    result = train_stage(model, ds, train_cfg=_train_cfg(stage="independent"),
                         loop_cfg=LoopConfig(out_dir=Path(corpus[2].root) / "s1",
                                             n_buckets=1, ema_decay=0.99))
    after = _flat(model)

    assert result.steps > 0
    assert result.passes_done == len(model_cfg.branches)
    # Every head moved (each got its own pass) ...
    for branch in model_cfg.branches:
        keys = [k for k in after if k.startswith(f"heads.{branch}.")]
        assert keys
        assert any(not torch.equal(before[k], after[k]) for k in keys), branch
    # ... and no frontend parameter did.
    frontend_keys = [k for k in after if k.startswith("frontends.")]
    assert frontend_keys
    assert all(torch.equal(before[k], after[k]) for k in frontend_keys)


def test_the_frozen_frontend_claim_goes_red_if_s1_stops_freezing(corpus, model_cfg):
    """MUTATION for the test above: hand S1 a plan that trains the frontends.

    Evidence that "no frontend parameter moved" is a check and not a tautology
    on a config that already says `freeze: true`.
    """
    torch.manual_seed(0)
    model = _model(model_cfg)
    # Unfreeze the encoder the way an S2 plan would, then run S1's group.
    for p in model.frontends.parameters():
        p.requires_grad_(True)
    mutated = dataclasses.replace(stage_plan("independent", model_cfg),
                                  train_frontends=True)
    params = trainable_parameters(model, mutated, ("voice",))
    assert {id(p) for p in params} & {id(p) for p in model.frontends.parameters()}


def test_s1_announces_the_freeze_override_when_the_config_disagrees(model_cfg):
    """S1 overrides `FrontendConfig.freeze` rather than reading it, and says so.

    The override is right -- reading the field would make S1 and S2 identical on
    both shipped stubs and reduce the ablation to measuring nothing. What is not
    right is a silent divergence between a config field and actual behaviour, so
    the plan names the frontends whose field is not being honoured.
    """
    plan = stage_plan("independent", _unfrozen(model_cfg))
    assert plan.freeze_overrides == tuple(model_cfg.frontends)
    assert plan.caveats and "freeze: false" in plan.caveats[0]
    for name in model_cfg.frontends:
        assert name in plan.caveats[0]
    assert "caveat:" in str(plan)


def test_s1_is_silent_when_the_config_already_agrees(model_cfg):
    """Non-vacuity for the test above: the caveat is a *check*, not a banner.

    Both shipped stubs say `freeze: true`, so the shipped path must be quiet --
    otherwise the warning would be present on every run and stop being read.
    """
    plan = stage_plan("independent", model_cfg)
    assert plan.freeze_overrides == () and plan.caveats == ()
    assert "caveat:" not in str(plan)


def test_only_s1_overrides_the_field_s2_and_s3_honour_it(model_cfg):
    """Caveat: the divergence is one stage's, not the schedule's. S2/S3 read the
    config, so they have nothing to announce and must not claim otherwise."""
    unfrozen = _unfrozen(model_cfg)
    for stage in ("joint", "codec_aware"):
        plan = stage_plan(stage, unfrozen)
        assert plan.train_frontends
        assert plan.freeze_overrides == () and plan.caveats == (), stage


def test_trainable_parameters_refuses_an_empty_set(model_cfg):
    """A fully frozen group is a silently empty run, not a fast one."""
    model = _model(model_cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameter"):
        trainable_parameters(model, stage_plan("independent", model_cfg), ("voice",))


# --------------------------------------------------------------------------- #
# 2. S3's codec expansion


def test_the_codec_expansion_is_an_exact_cross_product(corpus):
    """This is *the* label-independence argument, and it is structural.

    `P(normalize | label) = P(normalize)` holds because every variant is paired
    with every spec -- so it is asserted as set equality on the specs stripped of
    their `normalize` draw, not inferred from an estimated AUC.
    """
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(60))
    out = codec_variant_specs(specs, CODEC_VARIANTS)
    assert len(out) == len(specs) * len(CODEC_VARIANTS)

    def bare(seq):
        return sorted(dataclasses.replace(s, normalize={}).to_dict().__repr__()
                      for s in seq)

    reference = bare(specs)
    for variant in CODEC_VARIANTS:
        matching = [s for s in out
                    if all(s.normalize.get(k) == v for k, v in variant.items())
                    and set(s.normalize) == set(variant)]
        assert bare(matching) == reference, variant
    # The labels are untouched: a codec is a channel, not a class.
    assert sorted(s.cell for s in out) == sorted(
        c for s in specs for c in [s.cell] * len(CODEC_VARIANTS))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_the_stream_the_expansion_is_built_from_passes_i1b(corpus, seed):
    """The measured half: the *drawn* stream carries no metadata shortcut, and
    the expansion adds a feature that is constant across labels by construction.

    Seed-swept because I1b is an estimated AUC and four earlier rounds of
    flakiness in this repo all traced to a single hard-coded seed.
    """
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(400, seed=seed))
    passed, why = audit_specs(specs).results["I1b_metadata_shortcut_auc"]
    assert passed, why


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_label_conditional_codec_draw_is_caught_by_i1b(corpus, seed):
    """MUTATION: the known leak shape. `container = mp3 if fake else wav` scored
    AUC 1.000 on the I1b probe while every other invariant stayed green
    (docs/pipelines/05 §1), so the uniform-expansion argument is falsifiable."""
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(400, seed=seed))
    leaky = [dataclasses.replace(
        s, normalize={"container": "mp3", "bitrate": 64} if s.file_fake else {})
        for s in specs]
    passed, why = audit_specs(leaky).results["I1b_metadata_shortcut_auc"]
    assert not passed, f"I1b did not see a label-conditional codec draw: {why}"


def test_i1b_must_not_be_run_on_the_expanded_stream(corpus):
    """Measured, and the reason the two tests above audit the *unexpanded*
    stream. The expansion emits `k` near-duplicates of every spec, and I1b's
    probe is **cross-validated** -- so a row's twins land in the other folds and
    the classifier memorises rather than generalises.

    The proof that this is duplication and not the codec: expanding four ways
    with **no codec at all** trips I1b just as hard. Anyone who later "improves"
    the S3 audit by pointing it at the expanded stream gets AUC 1.0000 and no
    information.
    """
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(400, seed=0))
    assert audit_specs(specs).results["I1b_metadata_shortcut_auc"][0]

    placebo = list(codec_variant_specs(specs, ({}, {}, {}, {})))
    passed, why = audit_specs(placebo).results["I1b_metadata_shortcut_auc"]
    assert not passed and "1.0000" in why, why


@pytest.mark.skipif(not HAS_FFMPEG, reason="the codec round-trip needs ffmpeg")
def test_each_codec_variant_actually_changes_the_audio(corpus):
    """Measured, not declared. A 4-way expansion whose four legs render
    identical audio is S2 at 4x the cost, and nothing else would notice."""
    manifest, index, rcfg = corpus
    spec = Sampler(manifest, DRAW).sample_spec(0)
    rendered = [render(dataclasses.replace(spec, normalize=dict(v)), index, rcfg).wav
                for v in CODEC_VARIANTS]

    base = rendered[0]
    for variant, wav in zip(CODEC_VARIANTS[1:], rendered[1:]):
        assert wav.shape == base.shape, variant
        assert not torch.equal(wav, base), f"{variant} rendered identically to clean"
    # And they differ from *each other*, not merely from clean -- two legs that
    # collapse onto one another would halve the expansion silently.
    for i in range(1, len(rendered)):
        for j in range(i + 1, len(rendered)):
            assert not torch.equal(rendered[i], rendered[j]), (i, j)


# --------------------------------------------------------------------------- #
# 3. Precision


def test_the_batch_reaches_the_model_as_float32_under_every_precision(
        corpus, model_cfg, tmp_path, monkeypatch):
    """The pipeline emits float32 and the *model* casts, under autocast.

    Pre-casting the batch is how this repo produced NaN attention: GeM overflowed
    in fp16 above a feature scale of ~40 (docs/pipelines/04 §4).
    """
    seen = []
    original = DeepVoiceNet.forward

    def spy(self, wav, lengths=None):
        seen.append(wav.dtype)
        return original(self, wav, lengths)

    monkeypatch.setattr(DeepVoiceNet, "forward", spy)
    for precision in ("fp32", "bf16", "fp16"):
        seen.clear()
        torch.manual_seed(0)
        train_stage(_model(model_cfg), _dataset(corpus, n=2),
                    train_cfg=_train_cfg(stage="joint", precision=precision),
                    loop_cfg=LoopConfig(out_dir=tmp_path / precision, n_buckets=1,
                                        ema_decay=0.0))
        assert seen and set(seen) == {torch.float32}, (precision, set(seen))


def test_autocast_actually_changes_the_compute_dtype(model_cfg):
    """MUTATION-adjacent: if `autocast_for` returned a null context for every
    precision, the test above would pass while the precision knob did nothing."""
    a, b = torch.randn(8, 8), torch.randn(8, 8)
    with autocast_for("fp32", "cpu"):
        assert (a @ b).dtype is torch.float32
    with autocast_for("bf16", "cpu"):
        assert (a @ b).dtype is torch.bfloat16
    with autocast_for("fp16", "cpu"):
        assert (a @ b).dtype is torch.float16
    with pytest.raises(ValueError):
        autocast_for("int8", "cpu")
