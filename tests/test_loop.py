"""`training.loop` -- the stage runner, resume, EMA, souping and the gates.

🔴 The single highest-value test in this file is
`test_a_resumed_run_reproduces_an_uninterrupted_run_bitwise`, and the one that
gives it meaning is the mutation right below it: drop the sampler state from the
checkpoint and watch the same assertion go red. A resume that restores weights
and optimizer but not the *draw* trains on a different corpus, reports it under
the same exp_id, and nothing in a green suite or a loss curve shows it.

Every invariant here is paired with a mutation that has been observed to fail.
Where a claim is statistical it sweeps seeds, and where a claim is
size-dependent (the bf16 tie rate) it is asserted at the size we validate on
rather than at the size that is convenient -- this repo has already shipped a
padding defect that every test missed because every fixture used
`lengths = SR * 4`.
"""

import dataclasses
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from metrics.aggregate import fold_mean
from metrics.dacon import PREDICTION_COLUMNS, MetricSet, eer
from models.config import load_model_config, load_train_config
from models.model import DeepVoiceNet
from training.audit import AuditReport, audit_specs
from training.dataset import SpecDataset, fold_manifest, frozen_eval_specs
from training.folds import FoldConfig, build_folds
from training.loop import (CODEC_VARIANTS, STAGES, EMA, FoldResult, LoopConfig,
                           SamplerState, ValidationReport, aggregate_folds,
                           autocast_for, checkpoint_soup, codec_variant_specs,
                           evaluate, generator_key, leak_tripwires,
                           load_train_checkpoint, measured_split_kind,
                           output_sanity, prediction_frame, run_gates,
                           run_schedule, save_train_checkpoint, stage_plan,
                           train_stage, trainable_parameters, validate_fold)
from training.render import ManifestIndex, RenderConfig, render
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus

CORPUS = dict(n_per_pool=8, n_whole_file=8, seed=0, duration_range=(6.0, 8.0))
DRAW = SamplerConfig(duration_range=(4.0, 5.0))

HAS_FFMPEG = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


# --------------------------------------------------------------------------- #
# Fixtures


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("loop-corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return manifest, ManifestIndex.from_frame(manifest), RenderConfig(root=root)


@pytest.fixture(scope="module")
def model_cfg():
    return load_model_config("configs/a_stub.yaml")


def _train_cfg(**kw):
    base = load_train_config("configs/train_joint.yaml")
    return dataclasses.replace(base, **{"epochs": 1, "batch_size": 2, **kw})


def _dataset(corpus, n=4, seed=0):
    manifest, index, rcfg = corpus
    return SpecDataset.from_sampler(
        Sampler(manifest, DRAW), n, index, rcfg, seed=seed)


def _model(model_cfg):
    return DeepVoiceNet(model_cfg)


def _flat(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _max_abs_diff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


# --------------------------------------------------------------------------- #
# 1. The stage schedule


def test_no_loop_config_field_is_silently_ignored():
    """🔴 The `models.model.CONSUMED_ELSEWHERE` guard, one layer up.

    A knob that validates and then does nothing is worse than a missing knob: it
    makes an ablation report a difference it never tested. `eval_precision`,
    `eval_batch_size` and `log_every` were exactly that here until this caught
    them.
    """
    source = Path("training/loop.py").read_text()
    unread = [f.name for f in dataclasses.fields(LoopConfig)
              if f"loop_cfg.{f.name}" not in source]
    assert not unread, f"LoopConfig field(s) read nowhere: {unread}"


def test_loop_config_rejects_values_that_would_run_nothing():
    with pytest.raises(ValueError, match="n_buckets"):
        LoopConfig(n_buckets=0)
    with pytest.raises(ValueError, match="max_steps"):
        LoopConfig(max_steps=0)


def test_the_dropped_stage_is_refused_rather_than_aliased(model_cfg):
    """🔴 S4 is dropped. Running S2 under its name would report a stage that
    was never trained -- the same class of defect as a config knob that
    validates and then does nothing (docs/training/04 §4)."""
    with pytest.raises(NotImplementedError, match="DROPPED"):
        stage_plan("rank_polish", model_cfg)
    assert "rank_polish" not in STAGES
    # ⚠️ And `models.config` still accepts it, which is why the refusal has to
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
    """S1 is "each branch alone". 🔴 Asserted on the *weights*, not on the loss.

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
    """🔴 This is *the* label-independence argument, and it is structural.

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
    """🔴 Measured, and the reason the two tests above audit the *unexpanded*
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
    """🔴 Measured, not declared. A 4-way expansion whose four legs render
    identical audio is S2 at 4x the cost, and nothing else would notice."""
    manifest, index, rcfg = corpus
    spec = Sampler(manifest, DRAW).sample_spec(0)
    rendered = [render(dataclasses.replace(spec, normalize=dict(v)), index, rcfg).wav
                for v in CODEC_VARIANTS]

    base = rendered[0]
    for variant, wav in zip(CODEC_VARIANTS[1:], rendered[1:]):
        assert wav.shape == base.shape, variant
        assert not torch.equal(wav, base), f"{variant} rendered identically to clean"
    # ⚠️ And they differ from *each other*, not merely from clean -- two legs that
    # collapse onto one another would halve the expansion silently.
    for i in range(1, len(rendered)):
        for j in range(i + 1, len(rendered)):
            assert not torch.equal(rendered[i], rendered[j]), (i, j)


# --------------------------------------------------------------------------- #
# 3. 🔴 Resume


#: 🔴 6 specs at batch_size 2 with one bucket = **3 batches per pass**, so
#: `max_steps=2` stops *inside* a pass rather than on its boundary. At n=4 the
#: interruption landed exactly on a pass boundary and `batch_index` was always 0
#: -- the mid-epoch half of the resume would have gone untested while every
#: assertion below still passed.
def _run(model_cfg, corpus, out, *, max_steps=None, resume_from=None, n=6,
         epochs=2, ckpt_every=0, torch_seed=1234):
    model = _model(model_cfg)
    torch.manual_seed(torch_seed)
    ds = _dataset(corpus, n=n)
    result = train_stage(
        model, ds, train_cfg=_train_cfg(stage="joint", epochs=epochs),
        loop_cfg=LoopConfig(out_dir=out, n_buckets=1, ema_decay=0.99,
                            max_steps=max_steps, checkpoint_every=ckpt_every),
        resume_from=resume_from)
    return model, result


def test_a_resumed_run_reproduces_an_uninterrupted_run_bitwise(corpus, model_cfg,
                                                               tmp_path):
    """🔴 The load-bearing test of this module.

    Uninterrupted: N steps. Interrupted: stop at step 2, checkpoint, resume in a
    fresh process-equivalent (a fresh model, a fresh optimizer, a fresh dataset)
    and finish. The two must agree **bitwise** -- not approximately, because
    every source of divergence here (a re-drawn corpus, a re-seeded dropout
    stream, a rebuilt optimizer moment) is a difference of kind rather than of
    rounding.
    """
    full, res_full = _run(model_cfg, corpus, tmp_path / "full")
    assert res_full.steps >= 4, "the fixture must run past the interruption point"

    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2)
    assert res_part.truncated
    # ⚠️ Assert the interruption is *mid-pass*. On a pass boundary `batch_index`
    # is 0 and the resume would only have to restore the pass counter.
    stopped = res_part.checkpoints[-1].sampler
    assert stopped.pass_index == 0 and stopped.batch_index == 2, stopped
    ck = res_part.checkpoints[-1].path

    resumed, res_resume = _run(model_cfg, corpus, tmp_path / "resume",
                               resume_from=ck)
    assert res_resume.steps == res_full.steps
    assert _same(_flat(full), _flat(resumed)), (
        f"resumed run diverged by {_max_abs_diff(_flat(full), _flat(resumed)):.3e}")


def _rewrite(path: Path, out: Path, **changes) -> Path:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob.update(changes)
    torch.save(blob, out)
    return out


def test_dropping_the_sampler_state_makes_the_resume_diverge(corpus, model_cfg,
                                                             tmp_path):
    """🔴 MUTATION for the test above, and the reason `SamplerState` exists.

    The corpus is *drawn*, so a checkpoint that restores weights, optimizer and
    RNG but restarts the draw at batch 0 silently trains on a different sample
    stream. Nothing raises, the loss curve is unremarkable, and the run is
    reported under the same exp_id as the one it is not.
    """
    full, _ = _run(model_cfg, corpus, tmp_path / "full")
    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2)

    blob = torch.load(res_part.checkpoints[-1].path, map_location="cpu",
                      weights_only=False)
    forgetful = _rewrite(res_part.checkpoints[-1].path, tmp_path / "no_sampler.pt",
                         sampler={**blob["sampler"], "batch_index": 0,
                                  "pass_index": 0})
    resumed, _ = _run(model_cfg, corpus, tmp_path / "resume",
                      resume_from=forgetful)
    assert not _same(_flat(full), _flat(resumed)), (
        "dropping the sampler position changed nothing -- the bitwise-resume "
        "test above is then not testing the draw")


def test_dropping_only_the_batch_position_makes_the_resume_diverge(corpus,
                                                                  model_cfg,
                                                                  tmp_path):
    """The sharper mutation: keep `pass_index`, reset **only** `batch_index`.

    A resume that restores the epoch but restarts it from batch 0 replays the
    first half of the pass and never sees the second -- the shape a
    "resume from the last epoch" implementation has by default.
    """
    full, _ = _run(model_cfg, corpus, tmp_path / "full")
    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2)

    blob = torch.load(res_part.checkpoints[-1].path, map_location="cpu",
                      weights_only=False)
    assert blob["sampler"]["batch_index"] == 2
    rewound = _rewrite(res_part.checkpoints[-1].path, tmp_path / "rewound.pt",
                       sampler={**blob["sampler"], "batch_index": 0})
    resumed, _ = _run(model_cfg, corpus, tmp_path / "resume", resume_from=rewound)
    assert not _same(_flat(full), _flat(resumed))


def test_dropping_the_torch_rng_state_makes_the_resume_diverge(corpus, model_cfg,
                                                               tmp_path):
    """MUTATION: the *other* half of the state. The SED head carries
    `dropout_in=0.25` / `dropout_out=0.5`, so the global torch generator is part
    of the run and a resume that re-seeds it diverges on the first step."""
    full, _ = _run(model_cfg, corpus, tmp_path / "full")
    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2)

    torch.manual_seed(999)
    stale = _rewrite(res_part.checkpoints[-1].path, tmp_path / "no_rng.pt",
                     rng={"cpu": torch.get_rng_state()})
    resumed, _ = _run(model_cfg, corpus, tmp_path / "resume", resume_from=stale)
    assert not _same(_flat(full), _flat(resumed))


def test_dropping_the_optimizer_state_makes_the_resume_diverge(corpus, model_cfg,
                                                               tmp_path):
    """MUTATION: AdamW's moments. The most-remembered piece, kept honest anyway."""
    full, _ = _run(model_cfg, corpus, tmp_path / "full")
    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2)
    bare = _rewrite(res_part.checkpoints[-1].path, tmp_path / "no_opt.pt",
                    optimizer=None)
    resumed, _ = _run(model_cfg, corpus, tmp_path / "resume", resume_from=bare)
    assert not _same(_flat(full), _flat(resumed))


def test_a_checkpoint_without_sampler_state_is_refused_not_guessed(tmp_path):
    torch.save({"state_dict": {}, "rng": {}, "stage": "joint"},
               tmp_path / "old.pt")
    with pytest.raises(ValueError, match="sampler"):
        load_train_checkpoint(tmp_path / "old.pt")


def test_resuming_a_dataset_that_draws_differently_is_refused(corpus, model_cfg,
                                                              tmp_path):
    """🔴 A resume whose dataset draws a different number of specs, or at a
    different seed, is a different run. It raises rather than warning: nothing
    downstream can see the substitution."""
    _, res_part = _run(model_cfg, corpus, tmp_path / "part", max_steps=2, n=4)
    model = _model(model_cfg)
    with pytest.raises(ValueError, match="different corpus"):
        train_stage(model, _dataset(corpus, n=6),
                    train_cfg=_train_cfg(stage="joint", epochs=2),
                    loop_cfg=LoopConfig(out_dir=tmp_path / "x", n_buckets=1),
                    resume_from=res_part.checkpoints[-1].path)


def test_resuming_into_a_different_stage_is_refused(corpus, model_cfg, tmp_path):
    _, res = _run(model_cfg, corpus, tmp_path / "part", max_steps=1)
    model = _model(model_cfg)
    with pytest.raises(ValueError, match="stage"):
        train_stage(model, _dataset(corpus, n=4),
                    train_cfg=_train_cfg(stage="independent"),
                    loop_cfg=LoopConfig(out_dir=tmp_path / "y", n_buckets=1),
                    resume_from=res.checkpoints[-1].path)


def test_train_stage_refuses_the_frozen_eval_set(corpus, model_cfg, tmp_path):
    manifest, index, rcfg = corpus
    frozen = SpecDataset.frozen(
        frozen_eval_specs(Sampler(manifest, DRAW), 4), index, rcfg, fold=None)
    with pytest.raises(ValueError, match="frozen"):
        train_stage(_model(model_cfg), frozen, train_cfg=_train_cfg(),
                    loop_cfg=LoopConfig(out_dir=tmp_path))


def test_each_pass_draws_its_own_corpus(corpus, model_cfg, tmp_path):
    """⚠️ Under S1 there are five branch passes; if they all replayed epoch 0 the
    five branches would see byte-identical training sets and any comparison
    between them would be an artefact of the schedule."""
    ds = _dataset(corpus, n=4)
    ds.set_epoch(0)
    first = [s.to_dict() for s in ds.specs]
    ds.set_epoch(1)
    assert [s.to_dict() for s in ds.specs] != first


# --------------------------------------------------------------------------- #
# 4. EMA


def test_the_ema_after_one_update_is_exactly_the_current_weights(model_cfg):
    """🔴 The bias correction, asserted as an exact identity.

    Without it the "EMA weights" of a short run are mostly the random
    initialisation -- at decay 0.999 a raw EMA is still 63% init after 1,000
    steps -- and the run reports a number for a model it never trained.
    """
    model = _model(model_cfg)
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    ema.update(model)

    got, want = ema.state_dict_for(model), model.state_dict()
    for k in want:
        assert torch.allclose(got[k].double(), want[k].double(), atol=1e-6), k


def test_without_the_bias_correction_the_ema_is_mostly_the_initialisation(model_cfg):
    """MUTATION for the test above, computed rather than asserted: the raw
    shadow after one update is 0.001 of the weights and 0.999 of zero."""
    model = _model(model_cfg)
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    ema.update(model)

    corrected = ema.state_dict_for(model)
    raw = ema.state_dict()["shadow"]
    key = next(k for k in raw if raw[k].numel() > 1)
    assert not torch.allclose(raw[key].double(), corrected[key].double(), atol=1e-6)
    assert torch.allclose(raw[key].double(),
                          0.001 * corrected[key].double(), atol=1e-9)


def test_the_ema_weights_differ_from_the_trained_weights(corpus, model_cfg, tmp_path):
    """Non-vacuity: an EMA that tracked the weights exactly would be free and
    also pointless."""
    torch.manual_seed(0)
    model = _model(model_cfg)
    result = train_stage(model, _dataset(corpus, n=4),
                         train_cfg=_train_cfg(stage="joint", epochs=2),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.9))
    assert result.ema.steps == result.steps
    ema_state = result.ema.state_dict_for(model)
    assert not _same(ema_state, _flat(model))
    # ... and still loads strictly, which is what makes it shippable.
    model.load_state_dict(ema_state, strict=True)


def test_the_ema_survives_a_checkpoint_round_trip_bitwise(corpus, model_cfg, tmp_path):
    torch.manual_seed(0)
    model = _model(model_cfg)
    result = train_stage(model, _dataset(corpus, n=4),
                         train_cfg=_train_cfg(stage="joint"),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.9))
    blob = load_train_checkpoint(result.checkpoints[-1].path)
    restored = EMA(model, decay=0.9)
    restored.load_state_dict(blob["ema"])
    assert restored.steps == result.ema.steps
    assert _same(restored.state_dict_for(model), result.ema.state_dict_for(model))


def test_an_ema_with_no_updates_refuses_to_materialise(model_cfg):
    with pytest.raises(RuntimeError, match="no update"):
        EMA(_model(model_cfg)).state_dict_for(_model(model_cfg))


# --------------------------------------------------------------------------- #
# 5. Checkpoint soup


def _write_ckpt(path, model, scale):
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0).add_(scale)
    return save_train_checkpoint(
        path, model=model, optimizer=None, ema=None, stage="joint",
        global_step=0, sampler=SamplerState(0, 0, 4, 0, 0),
        train_cfg=load_train_config("configs/train_joint.yaml")).path


def test_the_soup_is_the_uniform_average(model_cfg, tmp_path):
    """★ Free ensembling at zero inference cost -- but only if it is an average.
    Asserted on known constants so the answer is exact, not approximately right."""
    a = _write_ckpt(tmp_path / "a.pt", _model(model_cfg), 1.0)
    b = _write_ckpt(tmp_path / "b.pt", _model(model_cfg), 3.0)
    c = _write_ckpt(tmp_path / "c.pt", _model(model_cfg), 8.0)

    soup = checkpoint_soup([a, b, c])
    for k, v in soup.items():
        if v.is_floating_point() and v.numel():
            assert torch.allclose(v, torch.full_like(v, 4.0)), k
    # It loads strictly into the architecture it came from.
    _model(model_cfg).load_state_dict(soup, strict=True)


def test_the_soup_refuses_a_single_checkpoint(model_cfg, tmp_path):
    a = _write_ckpt(tmp_path / "a.pt", _model(model_cfg), 1.0)
    with pytest.raises(ValueError, match="pass two or more"):
        checkpoint_soup([a])


def test_the_soup_refuses_mismatched_architectures(model_cfg, tmp_path):
    """🔴 The average of two architectures is not a model. Refused here rather
    than deferred to a `load_state_dict` failure in whoever ships it."""
    a = _write_ckpt(tmp_path / "a.pt", _model(model_cfg), 1.0)
    other = load_model_config("configs/b_stub.yaml")
    b = _write_ckpt(tmp_path / "b.pt", DeepVoiceNet(other), 1.0)
    with pytest.raises(ValueError):
        checkpoint_soup([a, b])


def test_the_soup_carries_integer_buffers_rather_than_averaging_them(model_cfg,
                                                                     tmp_path):
    a = _write_ckpt(tmp_path / "a.pt", _model(model_cfg), 1.0)
    b = _write_ckpt(tmp_path / "b.pt", _model(model_cfg), 3.0)
    blob = torch.load(a, map_location="cpu", weights_only=False)
    ints = [k for k, v in blob["state_dict"].items() if not v.is_floating_point()]
    soup = checkpoint_soup([a, b])
    for k in ints:
        assert soup[k].dtype == blob["state_dict"][k].dtype


def test_souping_across_training_checkpoints_of_one_run(corpus, model_cfg, tmp_path):
    """The across-epoch soup, on real checkpoints rather than hand-built ones."""
    torch.manual_seed(0)
    model = _model(model_cfg)
    result = train_stage(model, _dataset(corpus, n=4),
                         train_cfg=_train_cfg(stage="joint", epochs=3),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1))
    paths = [c.path for c in result.checkpoints]
    assert len(paths) >= 2
    _model(model_cfg).load_state_dict(checkpoint_soup(paths), strict=True)


# --------------------------------------------------------------------------- #
# 6. Precision


def test_the_batch_reaches_the_model_as_float32_under_every_precision(
        corpus, model_cfg, tmp_path, monkeypatch):
    """🔴 The pipeline emits float32 and the *model* casts, under autocast.

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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_bf16_probability_column_loses_ranking_resolution_at_val_size(seed):
    """🔴 Why `LoopConfig.eval_precision` is fp32, measured at the size that
    matters.

    VG5's gate is `n_unique > 0.5 n`. At n=400 -- a comfortable fixture size --
    bf16 passes; at the 1,200-per-class VAL floor it does not. A test written at
    fixture size would have certified the wrong default, which is the same shape
    as the padding defect where every fixture used `lengths = SR * 4`.
    """
    rng = torch.Generator().manual_seed(seed)
    logits = torch.randn(1200, generator=rng) * 1.5

    def frame(dtype):
        p = torch.sigmoid(logits.to(dtype)).double().numpy()
        df = pd.DataFrame({"file_id": [f"s{i}" for i in range(len(p))]})
        for c in PREDICTION_COLUMNS:
            df[c] = p
        return df

    assert output_sanity(frame(torch.float32)).results["VG5_B2_ranking_resolution"][0]
    assert not output_sanity(frame(torch.bfloat16)).results["VG5_B2_ranking_resolution"][0]


# --------------------------------------------------------------------------- #
# 7. Validation goes through the official harness


@pytest.fixture(scope="module")
def scored(corpus, model_cfg):
    manifest, index, rcfg = corpus
    torch.manual_seed(0)
    model = _model(model_cfg)
    specs = frozen_eval_specs(Sampler(manifest, DRAW), 40, seed=11)
    ds = SpecDataset.frozen(specs, index, rcfg, slice_="train", fold=None)
    return model, ds, evaluate(model, ds, batch_size=8, fold=0)


def test_evaluate_goes_through_the_official_estimator(scored, monkeypatch):
    """🔴 Never recompute EER. Proven by breaking the official one and watching
    `evaluate` fail, rather than by comparing two numbers this module produced."""
    model, ds, _ = scored

    def refuse(*a, **kw):
        raise AssertionError("roc_curve was reached -- good")

    monkeypatch.setattr("metrics.dacon.roc_curve", refuse)
    with pytest.raises(AssertionError, match="roc_curve was reached"):
        evaluate(model, ds, batch_size=8, fold=0)


def test_the_loop_contains_no_roc_code_of_its_own():
    source = Path("training/loop.py").read_text()
    for forbidden in ("roc_curve", "roc_auc_score", "def eer("):
        assert forbidden not in source, forbidden


def test_the_report_carries_per_cell_and_per_generator_not_pooled_only(scored):
    """docs/validation: report per cell and per generator, never pooled only. A
    good pooled EER routinely hides a collapsed cell, and cells 6/7 are the whole
    reason the competition has two fake heads."""
    _, ds, report = scored
    assert isinstance(report, ValidationReport)
    drawn = {s.cell for s in ds.specs}
    assert set(report.per_cell[report.per_cell["head"] == "file"]["value"]) == drawn
    assert len(report.per_generator) > 0
    # ⚠️ Every slice records which contrast it used, so a shared-contrast row and
    # a within-slice row can never be silently averaged together.
    assert set(report.per_cell["contrast"]) <= {"shared", "within"}


def test_evaluate_refuses_a_redrawable_dataset(corpus, model_cfg):
    with pytest.raises(ValueError, match="frozen"):
        evaluate(_model(model_cfg), _dataset(corpus, n=4))


def test_prediction_truth_columns_come_from_the_cell(scored):
    _, ds, report = scored
    frame = report.predictions
    for spec, row in zip(ds.specs, frame.itertuples()):
        assert row.cell == spec.cell
        assert row.file_fake == spec.file_fake
        assert row.voice_present == spec.voice_present
        assert row.music_present == spec.music_present


def test_prediction_frame_refuses_a_mismatched_prediction_length(scored):
    _, ds, report = scored
    preds = {c: report.predictions[c].to_numpy()[:-1] for c in PREDICTION_COLUMNS}
    with pytest.raises(ValueError, match="one row per spec"):
        prediction_frame(ds.specs, preds, ds.index)


def test_generator_key_never_invents_a_family_for_a_real_row(corpus):
    """A real component carries no `artifact_family` at all. The key says
    `real:<source>` rather than borrowing one, because a borrowed family would
    put REAL rows into a generator's slice and make that slice's EER a fiction."""
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(120, seed=3))
    families = set(manifest["artifact_family"].dropna().astype(str))
    for spec in specs:
        key = generator_key(spec, index)
        if spec.file_fake:
            assert not key.startswith("real:"), (spec.cell, key)
            assert set(key.split("+")) <= families
        else:
            assert key.startswith("real:") or key == "unknown", (spec.cell, key)


# --------------------------------------------------------------------------- #
# 8. Aggregation


def _metric_set(eer_file, eer_voice=0.2, eer_music=0.3, auc_vp=0.9, auc_mp=0.9,
                n=2000):
    from metrics.dacon import roll_up
    ads, cps, score = roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp)
    return MetricSet(eer_file, eer_voice, eer_music, auc_vp, auc_mp, ads, cps,
                     score, n, n, n)


def _breakdown(values):
    """A breakdown frame with the columns `metrics.breakdown.by` actually emits."""
    return pd.DataFrame([
        {"value": v, "head": "file", "eer": 0.1 + 0.01 * i, "contrast": "shared",
         "n_slice": 300, "n_slice_fake": 150, "n_pool": 900, "thin": False,
         "note": ""}
        for i, v in enumerate(values)])


def _fold_result(fold, metrics, gates=None, tripwires=None, n=200):
    report = ValidationReport(fold, metrics, _breakdown(range(1, 10)),
                              _breakdown(["hifigan", "suno_v3"]),
                              pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]}))
    green = AuditReport({"x": (True, "fine")})
    return FoldResult(fold, report, gates or green, tripwires or green)


@pytest.mark.parametrize("seed", range(6))
def test_the_headline_is_the_mean_of_folds_and_pooling_raw_oof_is_wrong(seed):
    """🔴 Reproduces docs/validation/02 §4 at small scale.

    Five folds of one model with true EER 0.100, each fold's scores put through
    a *harmless monotone* rescale -- which changes no fold's own EER at all.
    Mean-of-folds recovers 0.100; pooling the raw scores does not, and the error
    is one-sided because the drift is independent of the label.
    """
    rng = np.random.default_rng(seed)
    per_fold, pooled = [], []
    for k in range(5):
        n = 600
        y = np.repeat([0, 1], n)
        s = np.concatenate([rng.normal(0.0, 1.0, n),
                            rng.normal(2.563, 1.0, n)])       # true EER ~= 0.10
        a, b = 1.0 + 0.8 * k, 3.0 * k                          # monotone per fold
        per_fold.append(eer(y, s))
        pooled.append(pd.DataFrame({"y": y, "s": a * s + b}))

    mean_of_folds = float(np.mean(per_fold))
    pooled_frame = pd.concat(pooled, ignore_index=True)
    pooled_eer = eer(pooled_frame["y"].to_numpy(), pooled_frame["s"].to_numpy())

    # The rescale is monotone *within* a fold, so it cannot move a per-fold EER.
    assert abs(mean_of_folds - 0.100) < 0.02, mean_of_folds
    # Pooled is worse, materially, and never better.
    assert pooled_eer > mean_of_folds + 0.03, (pooled_eer, mean_of_folds)


def test_aggregate_folds_uses_fold_mean_and_not_a_reimplementation():
    metrics = [_metric_set(0.10 + 0.01 * k) for k in range(5)]
    results = [_fold_result(k, m) for k, m in enumerate(metrics)]
    report = aggregate_folds(results)
    reference = fold_mean(metrics, fold_ids=list(range(5)))
    assert report.score_mean == reference.mean.score
    assert report.score_sd == reference.score_sd
    assert report.sd_is_a_measurement


def test_a_single_fold_run_is_first_class_but_says_its_sd_is_not_a_measurement():
    """⚠️ The full 5-fold sweep is often unaffordable and Replay is fold 0 by
    definition. What a single fold does not give is `Score_sd`, which is P4's
    input and the ★ E5 tiebreaker -- and `fold_mean` returns 0.0 there, which
    reads as "perfectly stable" unless something says otherwise."""
    report = aggregate_folds([_fold_result(0, _metric_set(0.12))])
    assert report.aggregate.n_folds == 1
    assert report.score_sd == 0.0
    assert not report.sd_is_a_measurement
    assert any("nothing to vary" in c for c in report.caveats)
    assert report.quotable


def test_a_fold_below_the_pool_floor_is_excluded_and_recorded():
    big = [_fold_result(k, _metric_set(0.10, n=2000)) for k in range(4)]
    thin = _fold_result(4, _metric_set(0.40, n=200))
    report = aggregate_folds([*big, thin], min_pool=1200)
    assert report.aggregate.excluded_folds == (4,)
    assert any("min_pool" in c for c in report.caveats)
    assert abs(report.score_mean - aggregate_folds(big).score_mean) < 1e-12


def test_a_run_with_a_red_gate_is_not_quotable():
    red = AuditReport({"VG1_A1": (False, "a family is in two folds")})
    report = aggregate_folds([_fold_result(0, _metric_set(0.10), gates=red)])
    assert not report.quotable
    assert "NOT QUOTABLE" in str(report)
    assert report.as_ledger_row()["quotable"] is False


def test_a_skipped_gate_does_not_block_but_is_reported():
    """⚠️ VG3 is skipped on every run today. Treating a SKIP as a failure would
    make nothing quotable and the distinction would stop being read -- but a
    silent SKIP is the `I7` defect, which printed PASS for a check that existed
    nowhere."""
    skipped = AuditReport({"VG3": (True, AuditReport.SKIP + "not implemented")})
    report = aggregate_folds([_fold_result(0, _metric_set(0.10), gates=skipped)])
    assert report.quotable
    assert "VG3" in report.skipped_gates()
    assert "SKIP VG3" in str(report)
    assert "VG3" not in skipped.ran


# --------------------------------------------------------------------------- #
# 9. Leak tripwires


def test_the_music_tripwire_fires_below_three_percent():
    """🔴 Published cross-generator music detection is 46.4% EER. A local 0.5%
    on an unseen-generator split is evidence of a leak, not of success."""
    report = leak_tripwires(_metric_set(0.10, eer_music=0.005),
                            "generator_disjoint", "8 VAL families")
    passed, why = report.results["L1_music_unseen_generator"]
    assert not passed and "0.0050" in why


def test_the_voice_tripwire_fires_below_one_percent():
    report = leak_tripwires(_metric_set(0.10, eer_voice=0.004, eer_music=0.30),
                            "generator_disjoint", "")
    assert not report.results["L2_voice_unseen_generator"][0]
    assert report.results["L1_music_unseen_generator"][0]      # music is fine


@pytest.mark.parametrize("value,fires", [(0.0299, True), (0.0300, False),
                                         (0.0301, False)])
def test_the_music_threshold_is_asserted_on_the_quantity_it_names(value, fires):
    """⚠️ On the EER itself, at the boundary. A test that asserted `0 < eer`
    would pass at every one of these values."""
    report = leak_tripwires(_metric_set(0.10, eer_music=value),
                            "generator_disjoint", "")
    assert report.results["L1_music_unseen_generator"][0] is not fires


def test_a_healthy_unseen_generator_result_passes():
    report = leak_tripwires(_metric_set(0.20, eer_voice=0.09, eer_music=0.31),
                            "generator_disjoint", "8 VAL families")
    assert report.ok
    assert "L1_music_unseen_generator" in report.ran      # ran, not skipped


def test_the_tripwires_skip_rather_than_pass_on_a_split_they_cannot_speak_about():
    """🔴 The `AuditReport.SKIP` convention. A tripwire reading PASS on a
    generator-overlapping split gives the run a green gate it did not earn."""
    report = leak_tripwires(_metric_set(0.10, eer_voice=0.0, eer_music=0.0),
                            "generator_overlapping", "3 families shared")
    for key in ("L1_music_unseen_generator", "L2_voice_unseen_generator"):
        assert key in report.skipped
        assert key not in report.ran
    # ... and the row that *does* apply to this split fires on those same numbers.
    assert not report.results["L3_perfect_separation"][0]


def test_perfect_separation_on_an_overlapping_split_fires_on_every_head():
    for kw in ({"eer_file": 0.0}, {"eer_file": 0.2, "eer_voice": 0.0},
               {"eer_file": 0.2, "eer_music": 0.0},
               {"eer_file": 0.2, "auc_vp": 1.0}, {"eer_file": 0.2, "auc_mp": 1.0}):
        report = leak_tripwires(_metric_set(**{"eer_voice": 0.2, "eer_music": 0.2,
                                               "auc_vp": 0.9, "auc_mp": 0.9, **kw}),
                                "generator_overlapping", "")
        assert not report.results["L3_perfect_separation"][0], kw


def test_the_split_kind_is_measured_from_the_drawn_streams(tmp_path):
    """🔴 Measured, not declared. The caller who *thinks* the split is
    family-disjoint is exactly the caller whose 0.5% music EER needs explaining,
    so a `split_kind=` argument would be switched off by the same mistake the
    tripwires exist to catch."""
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    index = ManifestIndex.from_frame(manifest)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame
    resolved = fold_manifest(manifest, folds, fold=0)

    train = list(Sampler(resolved, DRAW, slice_="train").epoch_specs(600, seed=0))
    val = list(Sampler(resolved, DRAW, slice_="val").epoch_specs(600, seed=1))
    kind, detail = measured_split_kind(train, val, index)
    assert kind == "generator_disjoint", detail

    # The same corpus without the fold resolution: TRAIN and VAL share families.
    both = list(Sampler(manifest, DRAW).epoch_specs(600, seed=2))
    other = list(Sampler(manifest, DRAW).epoch_specs(600, seed=3))
    overlapping, why = measured_split_kind(both, other, index)
    assert overlapping == "generator_overlapping", why


def test_an_undecidable_split_is_undecidable_not_disjoint(corpus):
    """A stream with no generated component realises no families, and "disjoint
    from nothing" is vacuously true -- which would silently arm L1/L2."""
    manifest, index, rcfg = corpus
    real_only = [s for s in Sampler(manifest, DRAW).epoch_specs(200) if not s.file_fake]
    kind, _ = measured_split_kind(real_only, real_only, index)
    assert kind == "undecidable"
    report = leak_tripwires(_metric_set(0.0, eer_voice=0.0, eer_music=0.0), kind)
    assert "L1_music_unseen_generator" in report.skipped


# --------------------------------------------------------------------------- #
# 10. The gates


def test_vg5_catches_the_saturation_that_took_eer_from_0095_to_0302():
    """The documented failure, reproduced: saturating the top and bottom 40% of
    a column collapses ranking near the operating point with no other warning."""
    n = 1000
    rng = np.random.default_rng(0)
    clean = rng.random(n)
    saturated = np.clip(clean, 0.4, 0.6)
    saturated = np.where(saturated <= 0.4, 0.0, np.where(saturated >= 0.6, 1.0,
                                                         saturated))

    def frame(col):
        df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
        for c in PREDICTION_COLUMNS:
            df[c] = col
        return df

    assert output_sanity(frame(clean)).results["VG5_B2_ranking_resolution"][0]
    assert not output_sanity(frame(saturated)).results["VG5_B2_ranking_resolution"][0]


def test_vg5_catches_a_constant_column_and_a_nan():
    n = 100
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
    for c in PREDICTION_COLUMNS:
        df[c] = np.linspace(0.01, 0.99, n)
    assert output_sanity(df).ok

    constant = df.assign(FILE_FAKE_PROB=0.5)
    assert not output_sanity(constant).results["VG5_B3_no_constant_column"][0]

    nan = df.copy()
    nan.loc[0, "VOICE_FAKE_PROB"] = np.nan
    assert not output_sanity(nan).results["VG5_B5_no_fallback_nan"][0]
    out_of_range = df.assign(MUSIC_FAKE_PROB=df["MUSIC_FAKE_PROB"] * 2)
    assert not output_sanity(out_of_range).results["VG5_B1_finite_in_unit_interval"][0]


def test_vg5_b4_skips_rather_than_passes_without_a_reference():
    n = 20
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
    for c in PREDICTION_COLUMNS:
        df[c] = np.linspace(0.01, 0.99, n)
    report = output_sanity(df)
    assert "VG5_B4_id_set_matches" in report.skipped
    assert output_sanity(df, list(df["file_id"])).results["VG5_B4_id_set_matches"][0]
    assert not output_sanity(df, list(df["file_id"])[::-1]
                             ).results["VG5_B4_id_set_matches"][0]


def test_vg1_is_wired_to_check_split_integrity_and_goes_red_on_a_broken_split(scored):
    """🔴 Not a re-implementation: `training.folds.check_split_integrity` is the
    gate, and this proves the wiring by breaking the table it reads."""
    _, _, report = scored
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame

    green = run_gates(report, folds=folds, scheme_version="synthetic-v1")
    assert green.results["VG1_A1_family_in_one_cell"][0], green

    broken = folds.copy()
    rotating = broken[(broken["slice"] == "train_val")
                      & broken["artifact_family"].notna()]
    fam = rotating["artifact_family"].iloc[0]
    rows = broken.index[broken["artifact_family"] == fam]
    assert len(rows) >= 2, "the fixture must have a family with rows to split"
    broken.loc[rows[: len(rows) // 2], "fold"] = \
        (int(broken.loc[rows[0], "fold"]) + 1) % 5
    red = run_gates(report, folds=broken, scheme_version="synthetic-v1")
    assert not red.results["VG1_A1_family_in_one_cell"][0]
    assert not red.ok


def test_vg1_a10_notices_a_scheme_version_mismatch(scored):
    _, _, report = scored
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame
    wrong = run_gates(report, folds=folds, scheme_version="something-else")
    keys = [k for k in wrong.results if "A10" in k]
    assert keys and not wrong.results[keys[0]][0]


def test_vg3_reports_skip_and_never_pass(scored):
    """⚠️ The honest gap. A green stub would be worse than a SKIP: it would let a
    run claim a gate it never ran."""
    _, _, report = scored
    gates = run_gates(report)
    assert "VG3_adversarial_validation" in gates.skipped
    assert "VG3_adversarial_validation" not in gates.ran


def test_vg1_and_vg2_skip_without_the_inputs_they_need(scored):
    _, _, report = scored
    gates = run_gates(report)
    for key in ("VG1_split_integrity", "VG1_A8A9_eval_size_floors",
                "VG1_I5_split_safety", "VG2_shortcut_audit"):
        assert key in gates.skipped, key


def test_vg6_refuses_a_fourth_probe_opening(scored, tmp_path):
    """🔴 Enforced mechanically, not by discipline. The sealed slice is worthless
    once it has been optimized against."""
    _, _, report = scored
    log = tmp_path / "probe_openings.log"
    log.write_text("e1 first model\ne2 corpus freeze\ne3 final selection\n")
    assert run_gates(report, probe_log=log, opened_probe=True
                     ).results["VG6_probe_budget"][0]

    log.write_text(log.read_text() + "e4 one more look\n")
    assert not run_gates(report, probe_log=log, opened_probe=True
                         ).results["VG6_probe_budget"][0]
    # Opening PROBE with nowhere to record it is itself the failure.
    assert not run_gates(report, opened_probe=True).results["VG6_probe_budget"][0]
    # And a run that did not open PROBE skips rather than passes.
    assert "VG6_probe_budget" in run_gates(report).skipped


def test_vg4_uses_the_shared_t3_gap_implementation(scored):
    _, _, report = scored
    gates = run_gates(report)
    key = "VG4_corpus_identity"
    assert key in gates.results
    passed, why = gates.results[key]
    assert why.startswith(AuditReport.SKIP) or "gate <= 0.10" in why


# --------------------------------------------------------------------------- #
# 11. The schedule end to end


def test_run_schedule_refuses_the_dropped_stage_before_doing_any_work(
        corpus, model_cfg, tmp_path):
    model = _model(model_cfg)
    before = _flat(model)
    with pytest.raises(NotImplementedError):
        run_schedule(model, _dataset(corpus, n=2), train_cfg=_train_cfg(),
                     loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1),
                     stages=("joint", "rank_polish"))
    assert _same(before, _flat(model)), "work was done before the refusal"


@pytest.mark.skipif(not HAS_FFMPEG, reason="S3's codec round-trip needs ffmpeg")
def test_s1_then_s2_then_s3_runs_end_to_end_and_carries_the_weights_forward(
        corpus, model_cfg, tmp_path):
    torch.manual_seed(0)
    model = _model(model_cfg)
    before = _flat(model)
    results = run_schedule(model, _dataset(corpus, n=2),
                           train_cfg=_train_cfg(epochs=1),
                           loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                               ema_decay=0.9))
    assert [r.stage for r in results] == list(STAGES)
    # S3 sees four times the specs of S2, from the codec expansion.
    assert results[2].steps == 4 * results[1].steps
    assert not _same(before, _flat(model))


def test_validate_fold_cannot_produce_a_result_without_running_the_tripwires(
        corpus, model_cfg):
    """🔴 `train_specs` is required, so a `FoldResult` cannot exist without the
    split kind having been *measured* and the tripwires having run or said, by
    name, why they could not."""
    manifest, index, rcfg = corpus
    sampler = Sampler(manifest, DRAW)
    torch.manual_seed(0)
    model = _model(model_cfg)
    # ⚠️ `slice_` must name the slice the specs were DRAWN from, not the role
    # they are being used in. `SpecDataset.frozen` defaults to "val", and these
    # specs come from a `slice_="train"` sampler -- I5 caught exactly that
    # mislabelling here, which is the point of forwarding slice_/fold to it.
    eval_ds = SpecDataset.frozen(frozen_eval_specs(sampler, 24, seed=5), index,
                                 rcfg, slice_=sampler.slice_, fold=None)
    train_specs = list(sampler.epoch_specs(60, seed=0))

    result = validate_fold(model, eval_ds, train_specs=train_specs, fold=0)
    assert result.fold == 0
    assert set(result.tripwires.results) == {
        "L1_music_unseen_generator", "L2_voice_unseen_generator",
        "L3_perfect_separation"}
    # The gates that had no inputs skipped by name rather than passing quietly.
    assert "VG3_adversarial_validation" in result.gates.skipped
    assert "VG1_split_integrity" in result.gates.skipped
    # 🔴 I5 SKIPs without a manifest and RUNS with one -- never a silent pass.
    assert "VG1_I5_split_safety" in result.gates.skipped
    with_manifest = validate_fold(model, eval_ds, train_specs=train_specs,
                                  fold=0, manifest=manifest)
    passed, why = with_manifest.gates.results["VG1_I5_split_safety"]
    assert "VG1_I5_split_safety" not in with_manifest.gates.skipped, why
    assert passed, why
    assert result.validation.per_cell is not None

    with pytest.raises(TypeError):
        validate_fold(model, eval_ds, fold=0)          # no train_specs
