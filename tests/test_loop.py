"""`training.loop` -- the stage runner, the pass schedule and resume.

The single highest-value test in this file is
`test_a_resumed_run_reproduces_an_uninterrupted_run_bitwise`, and the ones that
give it meaning are the mutations right below it: drop the sampler state from
the checkpoint and watch the same assertion go red. A resume that restores
weights and optimizer but not the *draw* trains on a different corpus, reports
it under the same exp_id, and nothing in a green suite or a loss curve shows it.

The second claim here is that `pass_index` is the epoch key, so every pass draws
its own corpus. That lived only in a docstring until a review mutated
`set_epoch(pass_index)` to `set_epoch(0)` and all 97 tests stayed green:
`test_each_pass_trains_on_its_own_corpus` spies on `render` and asserts on which
specs actually reached the model.

Every invariant here is paired with a mutation that has been observed to fail.
"""

import dataclasses
from pathlib import Path

import pytest
import torch

from loop_fixtures import (HAS_FFMPEG, DRAW, _dataset, _flat, _max_abs_diff,
                           _metric_set, _model, _fold_result, _same,
                           _train_cfg, _unfrozen, corpus, model_cfg)
from models.model import DeepVoiceNet
from training.dataset import SpecDataset, frozen_eval_specs
from training.loop import (LoopConfig, run_schedule, spec_digest, train_stage)
from training.sampler import Sampler
from training.spec import SampleSpec
from training.stages import STAGES
from training.validate import aggregate_folds


# --------------------------------------------------------------------------- #
# 1. LoopConfig, and the stage result


def test_no_loop_config_field_is_silently_ignored():
    """The `models.model.CONSUMED_ELSEWHERE` guard, one layer up.

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


def test_the_override_caveat_reaches_the_run_report(corpus, model_cfg, tmp_path):
    """A caveat that stops at the `StageResult` is one nobody reads."""
    torch.manual_seed(0)
    model = DeepVoiceNet(_unfrozen(model_cfg))
    result = train_stage(model, _dataset(corpus, n=2),
                         train_cfg=_train_cfg(stage="independent"),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.0))
    assert result.caveats and "freeze: false" in result.caveats[0]

    report = aggregate_folds([_fold_result(0, _metric_set(0.10))],
                             caveats=result.caveats)
    assert any("freeze: false" in c for c in report.caveats)
    assert "freeze: false" in str(report)
    assert "freeze: false" in report.as_ledger_row()["caveats"]


def test_a_truncated_stage_says_it_is_not_quotable(corpus, model_cfg, tmp_path):
    """`max_steps` is Replay speed and the tests. Same mechanism, so a
    truncated run cannot reach the ledger looking like a full one."""
    torch.manual_seed(0)
    result = train_stage(_model(model_cfg), _dataset(corpus, n=6),
                         train_cfg=_train_cfg(stage="joint", epochs=2),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.0, max_steps=2))
    assert result.truncated
    assert any("truncated" in c and "not quotable" in c for c in result.caveats)


# --------------------------------------------------------------------------- #
# 2. Resume


#: 6 specs at batch_size 2 with one bucket = **3 batches per pass**, so
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
    """The load-bearing test of this module.

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
    # Assert the interruption is *mid-pass*. On a pass boundary `batch_index`
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
    """MUTATION for the test above, and the reason `SamplerState` exists.

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


def test_resuming_a_dataset_that_draws_differently_is_refused(corpus, model_cfg,
                                                              tmp_path):
    """A resume whose dataset draws a different number of specs, or at a
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


def test_set_epoch_redraws_the_dataset(corpus):
    """The dataset half, in isolation. Necessary and **not sufficient** -- it
    says `set_epoch` works, not that `train_stage` calls it with the pass index.
    A review mutated `set_epoch(pass_index)` to `set_epoch(0)` and this test, and
    all 96 others, stayed green. `test_each_pass_trains_on_its_own_corpus` is the
    one that covers the wiring."""
    ds = _dataset(corpus, n=4)
    ds.set_epoch(0)
    first = [s.to_dict() for s in ds.specs]
    ds.set_epoch(1)
    assert [s.to_dict() for s in ds.specs] != first


def _specs_rendered_per_pass(monkeypatch, model_cfg, corpus, out, *, stage, epochs,
                             n=6):
    """Run a stage and capture the spec **actually rendered** on every step.

    Spying on `render` rather than reading a field of the result: the question
    is which samples reached the model, and every field on `StageResult` is
    something this module computed and could compute wrongly in the same way.
    """
    import training.loop as loop_mod

    seen: list[SampleSpec] = []
    real_render = loop_mod.render

    def spy(spec, index, cfg):
        seen.append(spec)
        return real_render(spec, index, cfg)

    monkeypatch.setattr(loop_mod, "render", spy)
    torch.manual_seed(0)
    result = train_stage(_model(model_cfg), _dataset(corpus, n=n),
                         train_cfg=_train_cfg(stage=stage, epochs=epochs),
                         loop_cfg=LoopConfig(out_dir=out, n_buckets=1,
                                             ema_decay=0.0))
    return result, seen


def test_each_pass_trains_on_its_own_corpus(corpus, model_cfg, tmp_path,
                                            monkeypatch):
    """The wiring from `pass_index` to the epoch key, which nothing covered.

    `SamplerState`'s docstring claims `pass_index` "doubles as the epoch key for
    the draw", so every branch pass and every epoch sees its own corpus rather
    than replaying epoch 0. A review mutated `dataset.set_epoch(pass_index)` to
    `dataset.set_epoch(0)` and all 97 tests passed: multi-pass runs were
    exercised, but nothing asserted on *which specs each pass drew*. Under S1
    that mutation gives the five branches byte-identical training sets and makes
    any comparison between them an artefact of the schedule.

    Asserted on the rendered spec **contents** -- a `SampleSpec` is the whole
    decision to build a sample, so two passes that drew the same corpus produce
    equal `to_dict()`s and different ones do not.
    """
    result, seen = _specs_rendered_per_pass(monkeypatch, model_cfg, corpus,
                                            tmp_path, stage="joint", epochs=3)
    assert result.passes_done == 3

    by_epoch: dict[int, list[dict]] = {}
    for spec in seen:
        by_epoch.setdefault(spec.epoch, []).append(spec.to_dict())
    # One distinct epoch key per pass, and they are the pass indices themselves.
    assert sorted(by_epoch) == [0, 1, 2], sorted(by_epoch)

    # ... and the corpora those keys produced genuinely differ in content, not
    # only in the `epoch` field they carry.
    def content(rows):
        return sorted(repr({k: v for k, v in r.items() if k != "epoch"})
                      for r in rows)

    zero, one, two = (content(by_epoch[e]) for e in (0, 1, 2))
    assert zero != one and one != two and zero != two


def test_the_recorded_pass_digests_agree_with_what_was_rendered(
        corpus, model_cfg, tmp_path, monkeypatch):
    """`StageResult.pass_digests` is the permanent record of the above.

    Caveat: cross-checked against the rendered specs rather than trusted: a digest
    that happened to fold in a counter would differ every pass and let the same
    mutation straight back through.
    """
    result, seen = _specs_rendered_per_pass(monkeypatch, model_cfg, corpus,
                                            tmp_path, stage="joint", epochs=2)
    assert len(result.pass_digests) == 2
    assert len(set(result.pass_digests)) == 2, "two passes drew the same corpus"

    # Recompute from what the spy saw, in draw order, per pass.
    for pass_index, digest in enumerate(result.pass_digests):
        drawn = sorted((s for s in seen if s.epoch == pass_index),
                       key=lambda s: s.sample_id)
        assert spec_digest(drawn) == digest, pass_index


def test_spec_digest_is_stable_and_content_sensitive(corpus):
    """The digest's own contract, so the cross-check above means something."""
    manifest, index, rcfg = corpus
    sampler = Sampler(manifest, DRAW)
    a = list(sampler.epoch_specs(8, epoch=0))
    assert spec_digest(a) == spec_digest(list(sampler.epoch_specs(8, epoch=0)))
    assert spec_digest(a) != spec_digest(list(sampler.epoch_specs(8, epoch=1)))
    # S3's codec variants must move it, or the digest could not tell two
    # codec-aware passes apart.
    assert spec_digest(a) != spec_digest(
        [dataclasses.replace(s, normalize={"container": "flac"}) for s in a])


def test_s1_gives_each_branch_pass_its_own_corpus(corpus, model_cfg, tmp_path,
                                                  monkeypatch):
    """The consequence that motivated the design, stated on S1 itself.

    Five branch passes replaying epoch 0 would train every branch on
    byte-identical data, so a "voice head vs music head" reading would be an
    artefact of the schedule rather than a result.
    """
    result, seen = _specs_rendered_per_pass(monkeypatch, model_cfg, corpus,
                                            tmp_path, stage="independent",
                                            epochs=1, n=4)
    assert result.passes_done == len(model_cfg.branches)
    assert len(result.pass_digests) == len(model_cfg.branches)
    assert len(set(result.pass_digests)) == len(model_cfg.branches)
    assert sorted({s.epoch for s in seen}) == list(range(len(model_cfg.branches)))


# --------------------------------------------------------------------------- #
# 3. The schedule end to end


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
