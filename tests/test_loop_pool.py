"""The first-run loop knobs (docs/training/07 §3): pooled rendering, the LR
schedule, the frontend learning-rate group and the per-step log.

The load-bearing claim is the first test: rendering in worker processes trains
the SAME model as rendering inline, bitwise. A render is a pure function of its
spec, so the pool may change speed and nothing else; if it ever changed a
sample, every resume and every comparison between runs would inherit it.
"""

import json
import math

import pytest
import torch

from loop_fixtures import _dataset, _flat, _model, _same, _train_cfg, corpus, model_cfg  # noqa: F401
from training.loop import LoopConfig, _param_groups, lr_factor, train_stage


def _train(corpus, model_cfg, tmp_path, **loop):
    torch.manual_seed(0)
    model = _model(model_cfg)
    result = train_stage(model, _dataset(corpus, n=6),
                         train_cfg=_train_cfg(stage="joint", epochs=2),
                         loop_cfg=LoopConfig(out_dir=tmp_path, n_buckets=1, **loop))
    return model, result


def test_pooled_rendering_trains_the_inline_model_bitwise(corpus, model_cfg, tmp_path):
    inline, r1 = _train(corpus, model_cfg, tmp_path / "inline")
    pooled, r2 = _train(corpus, model_cfg, tmp_path / "pooled", render_workers=2)
    assert r1.steps == r2.steps > 0
    assert r1.pass_digests == r2.pass_digests
    assert _same(_flat(inline), _flat(pooled))


def test_the_constant_schedule_is_the_historical_behaviour():
    cfg = LoopConfig()
    assert all(lr_factor(s, 100, cfg) == 1.0 for s in (0, 50, 99))


def test_the_cosine_schedule_warms_up_then_decays_to_the_floor():
    cfg = LoopConfig(lr_schedule="cosine", warmup_steps=10, min_lr_ratio=0.1)
    assert lr_factor(0, 110, cfg) == pytest.approx(0.1)
    assert lr_factor(9, 110, cfg) == pytest.approx(1.0)
    assert lr_factor(10, 110, cfg) == pytest.approx(1.0)
    assert lr_factor(60, 110, cfg) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 2)))
    assert lr_factor(110, 110, cfg) == pytest.approx(0.1)
    xs = [lr_factor(s, 110, cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


def test_the_schedule_moves_the_optimizer_rate(corpus, model_cfg, tmp_path):
    """Non-vacuity: a schedule that computed factors and never wrote them into
    the optimizer would pass the two tests above and change nothing."""
    const, _ = _train(corpus, model_cfg, tmp_path / "c")
    cos, _ = _train(corpus, model_cfg, tmp_path / "k", lr_schedule="cosine",
                    warmup_steps=1, min_lr_ratio=0.0)
    assert not _same(_flat(const), _flat(cos))


def test_frontend_parameters_get_their_own_rate(model_cfg):
    model = _model(model_cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = _param_groups(model, params, 1e-3, LoopConfig(frontend_lr_scale=0.1))
    fe = {id(p) for p in model.frontends.parameters()}
    for g in groups:
        in_fe = {id(p) in fe for p in g["params"]}
        assert len(in_fe) == 1
        assert g["lr"] == pytest.approx(1e-4 if in_fe.pop() else 1e-3)
    assert sum(len(g["params"]) for g in groups) == len(params)


def test_the_step_log_is_written(corpus, model_cfg, tmp_path):
    _, r = _train(corpus, model_cfg, tmp_path, log_every=1)
    lines = (tmp_path / "train_log.jsonl").read_text().splitlines()
    assert len(lines) == r.steps
    rec = json.loads(lines[-1])
    assert rec["step"] == r.steps and "lr_factor" in rec


@pytest.mark.parametrize("bad", [dict(render_workers=-1), dict(lr_schedule="step"),
                                 dict(warmup_steps=-1), dict(min_lr_ratio=1.5),
                                 dict(frontend_lr_scale=0.0)])
def test_bad_loop_knobs_are_refused(bad):
    with pytest.raises(ValueError):
        LoopConfig(**bad)
