"""Frontends must normalise two genuinely different output shapes to one
contract, and the frame mask has to be exactly right or every downstream
padding guarantee is void."""

import dataclasses

import pytest
import torch

from models.config import AudioConfig, FreqPoolConfig, FrontendConfig, load_model_config
from models.frontends import build_frontend, frames_for

AUDIO = AudioConfig()
SR = AUDIO.sample_rate


def _stub(**kw):
    base = dict(name="stub", output_dim=32, fps=50.0)
    return build_frontend(FrontendConfig(**{**base, **kw}), AUDIO).eval()


def _patch_stub(n_freq=4, **kw):
    return _stub(n_freq=n_freq, freq_pool=FreqPoolConfig(kind="gem"), **kw)


# --------------------------------------------------------------------------- #
# the contract

def test_both_families_normalise_to_b_t_d():
    wav = torch.randn(2, SR * 3)
    for fe in (_stub(), _patch_stub()):
        feats, mask = fe(wav)
        assert feats.shape == (2, 150, 32)
        assert mask.shape == (2, 150)


def test_frames_for_matches_the_emitted_length():
    for seconds in (4, 7, 60):
        feats, _ = _stub()(torch.randn(1, SR * seconds))
        assert feats.shape[1] == frames_for(SR * seconds, SR, 50.0) == seconds * 50


def test_mask_marks_exactly_the_valid_frames():
    wav = torch.randn(3, SR * 4)
    lengths = torch.tensor([SR * 4, SR * 2, SR // 2])
    _, mask = _stub()(wav, lengths)
    assert mask.sum(-1).tolist() == [200, 100, 25]


def test_short_file_still_gets_one_frame():
    """4 s is the documented minimum, but a truncated file must not yield zero
    frames -- an empty attention softmax is NaN, not an error."""
    assert frames_for(10, SR, 50.0) == 1
    feats, mask = _stub()(torch.randn(1, 200), torch.tensor([200]))
    assert feats.shape[1] >= 1 and mask.any()


def test_frame_rate_difference_is_visible_to_callers():
    """The file branch aligns two frontends by fps; it must be readable."""
    slow, fast = _stub(fps=25.0), _stub(fps=50.0)
    wav = torch.randn(1, SR * 2)
    assert slow(wav)[0].shape[1] == 50
    assert fast(wav)[0].shape[1] == 100
    assert (slow.fps, fast.fps) == (25.0, 50.0)


# --------------------------------------------------------------------------- #
# guards

def test_stereo_input_is_rejected():
    with pytest.raises(ValueError, match="mono"):
        _stub()(torch.randn(2, 2, SR))


def test_unwired_frontend_fails_loudly_and_says_why():
    cfg = FrontendConfig(name="xlsr_300m", output_dim=1024)
    with pytest.raises(NotImplementedError, match="licence"):
        build_frontend(cfg, AUDIO)


def test_patch_grid_and_plain_frontends_are_not_interchangeable():
    """Configuring freq_pool on a (B,T,D) encoder is a silent shape bug if
    unguarded; config validation catches it, and so does the wrapper."""
    fe = _stub()
    fe.freq_pool = _patch_stub().freq_pool          # force the mismatch
    with pytest.raises(ValueError, match=r"B, F, T, D"):
        fe(torch.randn(1, SR))


def test_determinism_in_eval_mode():
    fe, wav = _stub(), torch.randn(2, SR * 2)
    assert torch.equal(fe(wav)[0], fe(wav)[0])


def test_shipped_configs_build_with_the_stub_backend():
    """Both candidates must be constructible before any checkpoint exists."""
    for name in ("a_shared_trunk", "b_three_branch"):
        cfg = load_model_config(f"configs/{name}.yaml")
        for fe_cfg in cfg.frontends.values():
            fe = build_frontend(dataclasses.replace(fe_cfg, name="stub"), cfg.audio).eval()
            feats, mask = fe(torch.randn(1, SR * 5))
            assert feats.shape == (1, 250, fe_cfg.output_dim)
            assert mask.all()
