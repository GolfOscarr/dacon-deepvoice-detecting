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


#: The base every knob probe below varies exactly one field of.
_KNOB_BASE = dict(n_buckets=1, ema_decay=0.9, grad_clip=5.0, checkpoint_every=0,
                  max_steps=None)

#: `LoopConfig` field -> (value A, value B, the observable that must move).
#:
#: Critical: this **replaces** a grep. The guard here used to read
#: `training/loop.py` and assert the string `loop_cfg.<name>` appeared in it,
#: which is satisfied by a mention: `grad_clip` and `checkpoint_every` both
#: passed it while deleting their behaviour entirely left the suite green. A
#: knob that validates and then does nothing is worse than a missing knob -- it
#: makes an ablation report a difference it never tested -- so what is asserted
#: is that two values of the knob produce two different runs.
_KNOB_PROBES: dict[str, tuple] = {
    "n_buckets": (1, 4, "weights"),
    "ema_decay": (0.0, 0.9, "ema"),
    "grad_clip": (0.0, 1e-3, "weights"),
    "checkpoint_every": (0, 1, "checkpoints"),
    "max_steps": (None, 1, "steps"),
    "lr_schedule": ("constant", "cosine", "weights"),
    # read only under the cosine schedule, so probed inside it
    "warmup_steps": (0, 3, "weights", {"lr_schedule": "cosine"}),
    "min_lr_ratio": (0.0, 0.9, "weights", {"lr_schedule": "cosine"}),
    "frontend_lr_scale": (1.0, 0.01, "weights"),
    "log_every": (0, 1, "log"),
}

#: `out_dir` is where the probe writes rather than something it can vary against
#: a fixed destination, so it has its own test below.
_OUT_DIR = "out_dir"

#: Fields no probe here can move, each with the reason. Caveat: an entry here is
#: a hole, not an exemption -- keep it short and keep the reason true.
_NOT_VARIABLE_ON_THIS_HARDWARE = {
    "device": "the test hardware has one device; a cuda probe would skip rather "
              "than check, which is the failure mode this table exists to avoid",
    "grad_checkpointing": "must NOT move the run (a recompute); the stub has no "
                          "transformer layers, and tests/test_grad_checkpointing.py "
                          "asserts equal outputs and gradients on real layers",
    "render_workers": "must NOT move the run: tests/test_loop_pool.py asserts pooled "
                      "rendering trains the inline model bitwise",
}


def _probe_run(model_cfg, corpus, out, **overrides):
    """One short stage run, reduced to everything a `LoopConfig` knob can move."""
    torch.manual_seed(0)
    model = _model(model_cfg)
    result = train_stage(
        model, _dataset(corpus, n=6), train_cfg=_train_cfg(stage="joint", epochs=1),
        loop_cfg=LoopConfig(out_dir=out, **{**_KNOB_BASE, **overrides}))
    return {
        "weights": _flat(model),
        "ema": result.ema is not None,
        "checkpoints": tuple(str(c.path) for c in result.checkpoints),
        "steps": (result.steps, result.truncated),
        "log": (out / "train_log.jsonl").exists(),
    }


def _moved(a, b, key):
    return not _same(a[key], b[key]) if key == "weights" else a[key] != b[key]


def test_the_knob_table_names_every_loop_config_field():
    """A new knob has to arrive with a probe or an admitted hole; there is no
    third option, and adding the field alone turns this red."""
    named = set(_KNOB_PROBES) | {_OUT_DIR} | set(_NOT_VARIABLE_ON_THIS_HARDWARE)
    assert named == {f.name for f in dataclasses.fields(LoopConfig)}


@pytest.mark.parametrize("field", sorted(_KNOB_PROBES))
def test_every_loop_config_knob_changes_the_run(field, corpus, model_cfg, tmp_path):
    """Each knob, at two values, against a run that is otherwise identical.

    Both sides write to the same `out_dir` on purpose: the second overwrites the
    first, so the only thing that can differ between the two observables is the
    knob.
    """
    a_value, b_value, key, *ctx = _KNOB_PROBES[field]
    extra = ctx[0] if ctx else {}
    out = tmp_path / field
    a = _probe_run(model_cfg, corpus, out, **{**extra, field: a_value})
    b = _probe_run(model_cfg, corpus, out, **{**extra, field: b_value})
    assert _moved(a, b, key), (
        f"LoopConfig.{field} = {a_value!r} and {b_value!r} produced the same "
        f"{key}: the knob validates and then does nothing")


def test_out_dir_is_where_the_checkpoints_land(corpus, model_cfg, tmp_path):
    """`out_dir`'s own probe: it is the destination, so it cannot be varied
    against a fixed one the way the table's knobs are."""
    a = _probe_run(model_cfg, corpus, tmp_path / "a", checkpoint_every=1)
    b = _probe_run(model_cfg, corpus, tmp_path / "b", checkpoint_every=1)
    assert a["checkpoints"] and b["checkpoints"]
    assert all(p.startswith(str(tmp_path / "a")) for p in a["checkpoints"])
    assert all(p.startswith(str(tmp_path / "b")) for p in b["checkpoints"])
    assert _same(a["weights"], b["weights"]), "out_dir moved something it should not"


#: Captured at import: `_clip_norms` patches `torch.nn.utils.clip_grad_norm_`,
#: and a test that calls it twice would otherwise wrap its own spy.
_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_


def _clip_norms(monkeypatch, model_cfg, corpus, out, *, precision, grad_clip):
    """Every `(pre-clip, post-clip)` gradient norm the run clipped at.

    Spying on `clip_grad_norm_` rather than on the weights: the quantity the
    threshold is compared against is the one both claims below are about, and it
    is not recoverable from the weights afterwards.
    """
    seen: list[tuple[float, float]] = []

    def spy(params, max_norm, *args, **kwargs):
        params = list(params)
        pre = _CLIP_GRAD_NORM(params, max_norm, *args, **kwargs)
        grads = [p.grad.detach() for p in params if p.grad is not None]
        post = torch.norm(torch.stack([g.double().norm() for g in grads]))
        seen.append((float(pre), float(post)))
        return pre

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    torch.manual_seed(0)
    train_stage(_model(model_cfg), _dataset(corpus, n=6),
                train_cfg=_train_cfg(stage="joint", epochs=1, precision=precision),
                loop_cfg=LoopConfig(out_dir=out, n_buckets=1, ema_decay=0.0,
                                    grad_clip=grad_clip))
    return seen


def test_grad_clip_bounds_the_gradient_the_optimizer_sees(corpus, model_cfg,
                                                          tmp_path, monkeypatch):
    """The knob's actual effect, on the gradient rather than on the weights.

    Threshold well below the run's own norms so the clip bites on every step:
    a test whose gradients never reached the threshold would pass with the clip
    removed.
    """
    seen = _clip_norms(monkeypatch, model_cfg, corpus, tmp_path, precision="bf16",
                       grad_clip=1e-3)
    assert seen, "the clip was never reached"
    assert all(pre > 1e-3 for pre, _ in seen), (
        f"the threshold never bit, so nothing here is a check: {seen}")
    # 1e-4 relative, which is `clip_grad_norm_`'s own float32 rounding on a
    # threshold this small. Unclipped these norms are ~2.4, so the slack is four
    # orders of magnitude short of hiding a missing clip.
    assert all(post <= 1e-3 * (1 + 1e-4) for _, post in seen), seen


def test_the_gradient_is_unscaled_before_it_is_clipped(corpus, model_cfg,
                                                       tmp_path, monkeypatch):
    """Critical: `scaler.unscale_` comes first, and this is what says so.

    Clipping a *scaled* gradient clips at a threshold that moves with the
    scaler's own state -- at the initial scale of 65536 a `grad_clip` of 5.0
    would really be 7.6e-05, and it would change again on every backoff. So the
    norm fp16 clips at must be the same quantity bf16 clips at, up to fp16's own
    rounding, rather than 65536 times it.
    """
    fp16 = _clip_norms(monkeypatch, model_cfg, corpus, tmp_path / "fp16",
                       precision="fp16", grad_clip=1e9)
    bf16 = _clip_norms(monkeypatch, model_cfg, corpus, tmp_path / "bf16",
                       precision="bf16", grad_clip=1e9)
    # `grad_clip` above every norm, so nothing is actually clipped and `pre` is
    # the raw norm the threshold would have been compared against.
    assert len(fp16) == len(bf16) and fp16
    for (a, _), (b, _) in zip(fp16, bf16):
        assert 0.5 < a / b < 2.0, (
            f"fp16 clipped at {a:.4g} where bf16 clipped at {b:.4g}: a factor of "
            "the loss scale means the gradient reached the clip still scaled")


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


#: What `_mean_parts` adds around the loss terms, plus `multitask_loss`'s own
#: `total`. Everything else in a row is one branch's head loss.
def test_the_schedules_caveats_are_the_union_of_its_stages(corpus, model_cfg,
                                                           tmp_path):
    """`run_schedule` returns one result per stage, and only S1 has a caveat.

    Critical: a caller that reports the last result -- the obvious thing to do
    with a list whose stages carry the weights forward -- drops it. Nothing in
    this module can wire the union into `aggregate_folds` itself, because
    scoring happens in `training.validate` against a frozen eval set the loop
    never sees, so the union is the caller's obligation and this is what states
    it.
    """
    torch.manual_seed(0)
    model = DeepVoiceNet(_unfrozen(model_cfg))
    results = run_schedule(model, _dataset(corpus, n=2),
                           train_cfg=_train_cfg(epochs=1),
                           loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                               ema_decay=0.0),
                           stages=("independent", "joint"))
    assert [bool(r.caveats) for r in results] == [True, False], (
        "only S1 overrides FrontendConfig.freeze, so only S1 has anything to say")

    union = tuple(c for r in results for c in r.caveats)
    report = aggregate_folds([_fold_result(0, _metric_set(0.10))], caveats=union)
    assert any("freeze: false" in c for c in report.caveats)
    assert "freeze: false" in report.as_ledger_row()["caveats"]


_ROW_META = {"stage", "pass", "group", "n_batches", "total"}


def _loss_terms(row):
    assert isinstance(row["total"], float) and row["total"] == row["total"], row
    # `multitask_loss` also records per-head diagnostics -- `<branch>/p_c` and
    # `<branch>/w_eff`, the effective weight of docs/training/02 §4. They ride in
    # the same row and are not loss terms, so the `/` keeps them out of the sets
    # asserted below. Their own coverage is in `tests/test_losses.py`.
    return {k: v for k, v in row.items() if k not in _ROW_META and "/" not in k}


def test_the_loss_history_records_one_row_per_pass_with_that_pass_group(
        corpus, model_cfg, tmp_path):
    """Nothing asserted on `loss_history` at all: every row could be `{}`.

    It is the stage's only record of what the objective actually was, and under
    S1 it is also the observable that separates "each branch alone" as an
    *objective* from the weight-level statement `tests/test_stages.py` makes.
    `_stage_loss_config` hands `multitask_loss` the group's branches, and
    `multitask_loss` puts one key per branch of the config it is given -- so a
    pass that scored the whole model would show all five branches here.
    """
    torch.manual_seed(0)
    result = train_stage(_model(model_cfg), _dataset(corpus, n=6),
                         train_cfg=_train_cfg(stage="independent", epochs=1),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.0))
    branches = list(model_cfg.branches)
    assert len(result.loss_history) == result.passes_done == len(branches)

    for pass_index, (row, branch) in enumerate(zip(result.loss_history, branches)):
        assert (row["stage"], row["pass"], row["group"]) == ("independent",
                                                             pass_index, branch)
        assert row["n_batches"] == 3, row
        terms = _loss_terms(row)
        assert set(terms) == {branch}, (
            f"pass {pass_index} trained {branch!r} alone but scored {sorted(terms)}")
        assert all(isinstance(v, float) and v == v for v in terms.values()), row


def test_a_joint_pass_scores_every_branch(corpus, model_cfg, tmp_path):
    """Non-vacuity for the test above: the restriction is S1's, not the loop's.

    If `_stage_loss_config` returned one branch regardless, or if `loss_history`
    only ever recorded the first key it saw, the S1 test would still pass.
    """
    torch.manual_seed(0)
    result = train_stage(_model(model_cfg), _dataset(corpus, n=6),
                         train_cfg=_train_cfg(stage="joint", epochs=2),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1,
                                             ema_decay=0.0))
    assert len(result.loss_history) == 2
    for row in result.loss_history:
        assert row["group"] == "+".join(model_cfg.branches)
        assert set(_loss_terms(row)) == set(model_cfg.branches), row


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
         epochs=2, ckpt_every=0, torch_seed=1234, precision=None):
    model = _model(model_cfg)
    torch.manual_seed(torch_seed)
    ds = _dataset(corpus, n=n)
    extra = {} if precision is None else {"precision": precision}
    result = train_stage(
        model, ds, train_cfg=_train_cfg(stage="joint", epochs=epochs, **extra),
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


def test_a_pass_boundary_resume_reproduces_an_uninterrupted_run_bitwise(
        corpus, model_cfg, tmp_path):
    """The **default** interruption, which nothing covered.

    `checkpoint_every` is 0 by default, so a real run that dies mid-training
    resumes from a `pass<N>.pt` file -- and every resume test above starts from a
    `truncated` one, which only `max_steps` ever writes. Two mutants lived here:
    making the in-loop `batch_seed` constant, and off-by-one-ing the seed the
    pass-boundary checkpoint stores.

    Critical: the batch seed's formula is written **twice** -- `train_cfg.seed +
    pass_index` in the loop and `train_cfg.seed + pass_index + 1` in the stored
    `SamplerState` -- and nothing but this test ties the two together.

    The resumed run is given a different global torch seed on purpose: it has to
    reproduce the original bitwise anyway, which it can only do by restoring the
    checkpoint's RNG rather than by happening to start where the first run did.
    """
    full, res_full = _run(model_cfg, corpus, tmp_path / "full", epochs=3)
    assert res_full.passes_done == 3 and len(set(res_full.pass_digests)) == 3

    boundary = _tagged(res_full, "pass0.pt")
    stopped = torch.load(boundary, map_location="cpu", weights_only=False)["sampler"]
    assert (stopped["pass_index"], stopped["batch_index"]) == (1, 0), stopped

    resumed, res_resume = _run(model_cfg, corpus, tmp_path / "resume", epochs=3,
                               resume_from=boundary, torch_seed=4321)
    assert res_resume.steps == res_full.steps
    assert res_resume.passes_done == 2
    # The corpora the remaining passes drew, not only the weights they produced.
    assert res_resume.pass_digests == res_full.pass_digests[1:]
    assert _same(_flat(full), _flat(resumed)), (
        f"resumed run diverged by {_max_abs_diff(_flat(full), _flat(resumed)):.3e}")


def test_a_pass_boundary_resume_at_the_wrong_batch_seed_diverges(corpus, model_cfg,
                                                                 tmp_path):
    """MUTATION for the test above, aimed at the off-by-one specifically.

    The stored seed is one greater than the seed the pass just used, because it
    describes the *next* pass. Storing the pass's own seed instead is a plausible
    mistake, it raises nothing, and it leaves the corpus digests untouched -- only
    the order the specs are batched in moves, which no field of `StageResult`
    records.
    """
    full, res_full = _run(model_cfg, corpus, tmp_path / "full", epochs=3)
    boundary = _tagged(res_full, "pass0.pt")
    stored = torch.load(boundary, map_location="cpu", weights_only=False)["sampler"]

    off_by_one = _rewrite(boundary, tmp_path / "off_by_one.pt",
                          sampler={**stored,
                                   "batch_seed": stored["batch_seed"] - 1})
    resumed, res_resume = _run(model_cfg, corpus, tmp_path / "resume", epochs=3,
                               resume_from=off_by_one)
    # The draw is untouched -- which is the point: nothing else can see this.
    assert res_resume.pass_digests == res_full.pass_digests[1:]
    assert not _same(_flat(full), _flat(resumed)), (
        "resuming at the previous pass's batch seed changed nothing, so the "
        "bitwise test above is not testing the batching order")


def _scaler_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["scaler"]


def _tagged(result, suffix):
    """The checkpoint this run wrote under `<stage>-<suffix>`."""
    return next(c.path for c in result.checkpoints if c.path.name.endswith(suffix))


def test_the_mid_epoch_checkpoint_carries_the_fp16_loss_scale(corpus, model_cfg,
                                                              tmp_path):
    """The `checkpoint_every` call site used to drop `scaler=`, and only it.

    Critical: this needs **both** `checkpoint_every > 0` and `precision="fp16"`,
    and the two holes are independent, which is why nothing caught it:

    * at bf16 the mid-epoch file's `scaler: None` is the *correct* value
      (`save_train_checkpoint` stores nothing unless `scaler.is_enabled()`), and
      `configs/train_joint.yaml` -- the config every test in this suite loads --
      says bf16;
    * at the default `checkpoint_every = 0` the branch never runs at all, so the
      only two call sites exercised are the truncated one and the pass-boundary
      one, and both always passed the scaler.

    Asserted on the **resumed scaler state** rather than on the weights: over a
    6-step run the scale never leaves its initial 65536, so the two runs' weights
    coincide and the growth tracker is the only thing that moves. The test below
    is the one where the divergence reaches the weights.
    """
    full, res_full = _run(model_cfg, corpus, tmp_path / "full", ckpt_every=1,
                          precision="fp16")
    mid = _tagged(res_full, "step2.pt")
    assert _scaler_state(mid) is not None, (
        "the mid-epoch checkpoint wrote `scaler: None` under fp16: a resume from "
        "it rebuilds the loss scaler at its default and replays the warm-up")

    ending = _scaler_state(_tagged(res_full, "pass1.pt"))
    # Non-vacuity: a scaler that never advanced would make the comparison below
    # true for the wrong reason. It advances once per successful step.
    assert ending["_growth_tracker"] == res_full.steps == 6, ending

    resumed, res_resume = _run(model_cfg, corpus, tmp_path / "resume",
                               ckpt_every=1, precision="fp16", resume_from=mid)
    assert _scaler_state(_tagged(res_resume, "pass1.pt")) == ending, (
        "the resumed run's loss scaler is not where the uninterrupted run's is")
    assert _same(_flat(full), _flat(resumed))


def test_a_resume_keeps_a_backed_off_loss_scale_rather_than_restarting_at_65536(
        corpus, model_cfg, tmp_path, monkeypatch):
    """The same hole where it reaches the weights, not only the tracker.

    An inf gradient on step 2 makes `GradScaler` skip that step and halve the
    scale to 32768. A checkpoint written after it that forgot the scaler sends
    the resumed run back to 65536, so from there on the two runs scale their
    gradients by different amounts and round differently in fp16 -- 5.5e-07 of
    max weight divergence when this was broken, against bitwise equality now.

    The inf is injected into the loss rather than faked in the scaler: what is
    under test is that the loop's own scaler state survives the round trip.
    """
    import training.loop as loop_mod

    real_loss = loop_mod.multitask_loss
    calls = {"n": 0}

    def inf_on_step_two(*args, **kwargs):
        total, parts = real_loss(*args, **kwargs)
        calls["n"] += 1
        return (total * float("inf"), parts) if calls["n"] == 2 else (total, parts)

    monkeypatch.setattr(loop_mod, "multitask_loss", inf_on_step_two)

    full, res_full = _run(model_cfg, corpus, tmp_path / "full", ckpt_every=1,
                          precision="fp16")
    mid = _tagged(res_full, "step2.pt")
    backed_off = _scaler_state(mid)
    assert backed_off is not None and backed_off["scale"] == 32768.0, (
        f"the mid-epoch checkpoint should hold the backed-off scale: {backed_off}")

    # The resumed run picks up after the backed-off step, so the injection must
    # not fire again: the counter is already past it.
    calls["n"] = 2
    resumed, _ = _run(model_cfg, corpus, tmp_path / "resume", ckpt_every=1,
                      precision="fp16", resume_from=mid)

    assert _scaler_state(_tagged(res_full, "pass1.pt"))["scale"] == 32768.0
    assert _same(_flat(full), _flat(resumed)), (
        "the resumed run restarted the loss scaler at its default and diverged "
        f"by {_max_abs_diff(_flat(full), _flat(resumed)):.3e}")


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
