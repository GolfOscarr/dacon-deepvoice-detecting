"""`training.checkpoint` -- the EMA, the checkpoint file and the weight soup.

The EMA's bias correction is asserted as an exact identity (after one update the
EMA *is* the current weights) and paired with the mutation that computes what a
raw shadow would have given instead: 0.001 of the weights and 0.999 of zero. On
a short schedule an uncorrected EMA is mostly the random initialisation, and the
run would report a number for a model it never trained.

The soup's refusals are the other half: a mismatched key set or a differently
shaped head is refused here rather than deferred to a `load_state_dict` failure
in whoever ships it.
"""

import pytest
import torch

from loop_fixtures import (_dataset, _flat, _model, _same, _train_cfg, corpus,
                           model_cfg)
from models.config import load_model_config, load_train_config
from models.model import DeepVoiceNet
from training.checkpoint import (EMA, SamplerState, checkpoint_soup,
                                 load_train_checkpoint,
                                 save_train_checkpoint)
from training.loop import LoopConfig, train_stage


# --------------------------------------------------------------------------- #
# 1. The checkpoint file


def test_a_checkpoint_without_sampler_state_is_refused_not_guessed(tmp_path):
    torch.save({"state_dict": {}, "rng": {}, "stage": "joint"},
               tmp_path / "old.pt")
    with pytest.raises(ValueError, match="sampler"):
        load_train_checkpoint(tmp_path / "old.pt")


# --------------------------------------------------------------------------- #
# 2. EMA


def test_the_ema_after_one_update_is_exactly_the_current_weights(model_cfg):
    """The bias correction, asserted as an exact identity.

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
# 3. Checkpoint soup


def _write_ckpt(path, model, scale):
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0).add_(scale)
    return save_train_checkpoint(
        path, model=model, optimizer=None, ema=None, stage="joint",
        global_step=0, sampler=SamplerState(0, 0, 4, 0, 0),
        train_cfg=load_train_config("configs/train_joint.yaml")).path


def test_the_soup_is_the_uniform_average(model_cfg, tmp_path):
    """Free ensembling at zero inference cost (primary source) -- but only if it
    is an average.
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
    """The average of two architectures is not a model. Refused here rather
    than deferred to a `load_state_dict` failure in whoever ships it."""
    a = _write_ckpt(tmp_path / "a.pt", _model(model_cfg), 1.0)
    other = load_model_config("configs/b_stub.yaml")
    b = _write_ckpt(tmp_path / "b.pt", DeepVoiceNet(other), 1.0)
    with pytest.raises(ValueError):
        checkpoint_soup([a, b])


# --- The four refusals, one isolating case each ---------------------------- #
#
# Critical: `a_stub` against `b_stub` differs in its key set, its shapes *and*
# its config at once, so it trips whichever check is left standing -- and each
# of the three survived being deleted individually under it. The blobs below are
# hand-built so exactly one thing differs at a time.
# `test_a_real_checkpoint_has_what_the_soup_reads` ties them back to what
# `save_train_checkpoint` actually writes.

_CONFIG = {"branches": {"voice": {}}, "dim": 8}


def _blob(path, state, config=_CONFIG, *, drop_config=False):
    """A minimal checkpoint. Only `state_dict` and `config` are read here."""
    payload = {"state_dict": state}
    if not drop_config:
        payload["config"] = config
    torch.save(payload, path)
    return path


def test_a_real_checkpoint_has_what_the_soup_reads(model_cfg, tmp_path):
    """Non-vacuity for the hand-built blobs below: a checkpoint this repo
    actually writes carries both keys, so the fixtures are not a shape the soup
    will never meet."""
    real = torch.load(_write_ckpt(tmp_path / "real.pt", _model(model_cfg), 1.0),
                      map_location="cpu", weights_only=False)
    assert {"state_dict", "config"} <= set(real)


def test_the_soup_refuses_a_checkpoint_that_carries_no_config(tmp_path):
    """A missing config is "cannot tell", not "the same".

    This read `blob.get("config")`, so two checkpoints without one both came out
    `None`, compared equal, and were souped with nothing having compared their
    architectures. The two blobs here are otherwise identical, so the key-set
    and shape checks cannot fire.
    """
    state = {"w": torch.ones(2, 3)}
    a = _blob(tmp_path / "a.pt", state, drop_config=True)
    b = _blob(tmp_path / "b.pt", state, drop_config=True)
    with pytest.raises(ValueError, match="no 'config'"):
        checkpoint_soup([a, b])


def test_the_soup_refuses_an_extra_key_rather_than_dropping_it(tmp_path):
    """Isolates the key-set check: same config, same shapes on every shared key.

    Caveat: removing this check is **silent**. The average runs over the first
    checkpoint's keys, so a key only the second one has is dropped without
    comment -- the "silent partial average" the docstring names.
    """
    a = _blob(tmp_path / "a.pt", {"w": torch.ones(2, 3)})
    b = _blob(tmp_path / "b.pt", {"w": torch.ones(2, 3), "extra": torch.ones(4)})
    with pytest.raises(ValueError, match="key set differs"):
        checkpoint_soup([a, b])


def test_the_soup_refuses_a_shape_that_would_broadcast(tmp_path):
    """Isolates the shape check: same config, same key set, one shape apart.

    Caveat: removing this check is silent too, and this is the fixture that
    shows it. `(1, 3)` against `(2, 3)` broadcasts, so the average would come
    out well-shaped and wrong rather than raising -- which is why the check
    cannot be left to `acc += state[k]` to discover.
    """
    a = _blob(tmp_path / "a.pt", {"w": torch.ones(2, 3)})
    b = _blob(tmp_path / "b.pt", {"w": torch.full((1, 3), 3.0)})
    with pytest.raises(ValueError, match=r"w is \(1, 3\)"):
        checkpoint_soup([a, b])


def test_the_soup_refuses_two_configs_that_differ_by_one_field(tmp_path):
    """Isolates the config check: identical key set, identical shapes.

    `test_the_soup_refuses_mismatched_architectures` above cannot do this -- its
    two stubs differ in all three ways, so it stays green with the config check
    deleted.
    """
    state = {"w": torch.ones(2, 3)}
    a = _blob(tmp_path / "a.pt", state)
    b = _blob(tmp_path / "b.pt", state, config={**_CONFIG, "dim": 16})
    with pytest.raises(ValueError, match="different configs"):
        checkpoint_soup([a, b])


def test_the_soup_carries_integer_buffers_rather_than_averaging_them(tmp_path):
    """Asserted on the **value**, which is the whole content of the claim.

    This asserted `soup[k].dtype == ...` and was doubly vacuous: `(acc / n)
    .to(v.dtype)` preserves the dtype while averaging the value, and `a_stub`
    has no non-float entries at all, so the loop ran zero times. Buffers 3 and 9
    must give 3 -- the first checkpoint's -- and not 6.
    """
    counter = torch.tensor(3, dtype=torch.int64)
    a = _blob(tmp_path / "a.pt", {"w": torch.ones(2), "n": counter})
    b = _blob(tmp_path / "b.pt",
              {"w": torch.full((2,), 3.0), "n": torch.tensor(9, dtype=torch.int64)})

    soup = checkpoint_soup([a, b])
    assert soup["n"].dtype == torch.int64
    assert int(soup["n"]) == 3, "the counter was averaged, not carried"
    # ... while the float entry beside it *is* averaged, so "carried" is a
    # decision about non-float entries rather than about this soup.
    assert torch.allclose(soup["w"], torch.full((2,), 2.0))


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
