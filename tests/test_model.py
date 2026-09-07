"""The assembly. Two properties matter most: A and B are the same class, and a
file's score does not depend on what is batched with it."""

import dataclasses

import pytest
import torch

from metrics.dacon import PREDICTION_COLUMNS
from models.config import load_model_config
from models.model import DeepVoiceNet, load_checkpoint, save_checkpoint

SR = 16_000


def _cfg(name):
    """A shipped config, with real frontends swapped for the weightless stub."""
    cfg = load_model_config(f"configs/{name}.yaml")
    return dataclasses.replace(
        cfg, frontends={k: dataclasses.replace(v, name="stub")
                        for k, v in cfg.frontends.items()})


def _model(name):
    return DeepVoiceNet(_cfg(name)).eval()


@pytest.fixture(params=["a_shared_trunk", "b_three_branch"])
def model(request):
    return _model(request.param)


# --------------------------------------------------------------------------- #
# shape and structure

def test_produces_every_submission_column(model):
    out = model(torch.randn(2, SR * 5))
    assert sorted(model.columns.values()) == sorted(PREDICTION_COLUMNS)
    for branch in model.cfg.branches:
        assert out[branch]["clip_logits"].shape == (2,)


def test_a_and_b_share_branch_structure():
    """docs/architecture/03: B strictly contains A."""
    a, b = _model("a_shared_trunk"), _model("b_three_branch")
    assert set(a.heads) == set(b.heads)
    assert len(a.frontends) == 1 and len(b.frontends) == 2
    assert b.n_parameters() > a.n_parameters()


def test_every_file_passes_through_every_branch():
    """The branches are outputs, not input types -- nothing is routed."""
    m = _model("b_three_branch")
    out = m(torch.randn(1, SR * 4))
    for branch in ("voice", "music", "file"):
        assert torch.isfinite(out[branch]["clip_logits"]).all()


@pytest.mark.parametrize("seconds", [4, 7, 60])
def test_handles_the_documented_duration_range(model, seconds):
    out = model(torch.randn(1, SR * seconds))
    assert out["file"]["frame_logits"].shape[1] == seconds * 50


def test_file_branch_aligns_two_frame_rates():
    """B's file branch reads both frontends, which disagree on fps."""
    cfg = _cfg("b_three_branch")
    fes = dict(cfg.frontends)
    fes["speech"] = dataclasses.replace(fes["speech"], fps=25.0)
    m = DeepVoiceNet(dataclasses.replace(cfg, frontends=fes)).eval()
    out = m(torch.randn(1, SR * 4))
    # align_to is `audio` at 50 fps, so the file branch runs on the audio grid.
    assert out["file"]["frame_logits"].shape[1] == 200
    assert out["voice"]["frame_logits"].shape[1] == 100        # speech grid, 25 fps


# --------------------------------------------------------------------------- #
# 🔴 rule 2.4 — per-file independence

def test_score_does_not_depend_on_the_batch(model):
    """A file scored alone and inside a batch of longer files must agree.

    Not bitwise: kernel selection varies with batch shape. Tolerance plus rank
    preservation is the real contract (docs/architecture/01 §1.4).
    """
    torch.manual_seed(0)
    solo = torch.randn(1, SR * 4)
    alone = model(solo, torch.tensor([SR * 4]))

    others = torch.randn(7, SR * 9) * 10                       # longer, louder
    batch = torch.cat([torch.nn.functional.pad(solo, (0, SR * 5)), others])
    lengths = torch.tensor([SR * 4] + [SR * 9] * 7)
    batched = model(batch, lengths)

    for branch in model.cfg.branches:
        a = alone[branch]["clip_logits"][0]
        b = batched[branch]["clip_logits"][0]
        assert torch.allclose(a, b, atol=1e-4), f"{branch}: {a.item()} vs {b.item()}"


def test_ranking_over_a_canned_set_is_batch_invariant(model):
    """Ranking is what EER reads, so it is the property that must hold."""
    torch.manual_seed(1)
    files = [torch.randn(1, SR * (4 + i)) for i in range(6)]
    solo = torch.stack([model(f)["file"]["clip_logits"][0] for f in files])

    width = max(f.shape[-1] for f in files)
    batch = torch.cat([torch.nn.functional.pad(f, (0, width - f.shape[-1])) for f in files])
    lengths = torch.tensor([f.shape[-1] for f in files])
    together = model(batch, lengths)["file"]["clip_logits"]

    assert torch.equal(solo.argsort(), together.argsort())


def test_padding_content_cannot_leak_into_a_score(model):
    """Same file, two different pad fillings -> identical score."""
    torch.manual_seed(2)
    x = torch.randn(1, SR * 4)
    lengths = torch.tensor([SR * 4])
    quiet = model(torch.nn.functional.pad(x, (0, SR * 4)), lengths)
    loud = torch.cat([x, torch.randn(1, SR * 4) * 50], dim=-1)
    noisy = model(loud, lengths)
    for branch in model.cfg.branches:
        assert torch.allclose(quiet[branch]["clip_logits"],
                              noisy[branch]["clip_logits"], atol=1e-4), branch


def test_deterministic_in_eval_mode(model):
    x = torch.randn(2, SR * 4)
    assert torch.equal(model(x)["file"]["clip_logits"], model(x)["file"]["clip_logits"])


# --------------------------------------------------------------------------- #
# checkpoints

def test_checkpoint_round_trip_carries_the_config(tmp_path):
    m = _model("b_three_branch")
    path = tmp_path / "model.pt"
    save_checkpoint(m, path)
    back = load_checkpoint(path)
    assert back.cfg == m.cfg
    x = torch.randn(1, SR * 4)
    assert torch.allclose(m(x)["file"]["clip_logits"], back(x)["file"]["clip_logits"])


def test_checkpoint_with_mismatched_weights_fails_loudly(tmp_path):
    """A strict load is only an assertion if it actually raises."""
    m = _model("b_three_branch")
    path = tmp_path / "model.pt"
    save_checkpoint(m, path)
    blob = torch.load(path, weights_only=False)
    blob["state_dict"].pop(next(iter(blob["state_dict"])))
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path)


def test_non_checkpoint_file_is_rejected(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a DeepVoiceNet checkpoint"):
        load_checkpoint(path)


# --------------------------------------------------------------------------- #
# training-only appendages

def test_distill_head_is_aux_not_a_column():
    m = _model("b_three_branch")
    out = m(torch.randn(1, SR * 4))
    assert "distill_emb" in out["_aux"]
    assert "_aux" not in m.columns


def test_c_lite_separation_head_is_one_config_flip():
    cfg = _cfg("b_three_branch")
    cfg = dataclasses.replace(cfg, aux=dataclasses.replace(cfg.aux, separation_head=True))
    out = DeepVoiceNet(cfg).eval()(torch.randn(1, SR * 4))
    assert out["_aux"]["separation"].shape[-1] == 2 * cfg.aux.stft_bins


def test_a_has_no_aux_by_default():
    assert "_aux" not in _model("a_shared_trunk")(torch.randn(1, SR * 4))
