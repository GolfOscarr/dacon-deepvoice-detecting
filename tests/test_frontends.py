"""Frontends must normalise two genuinely different output shapes to one
contract, and the frame mask has to be exactly right or every downstream
padding guarantee is void."""

import dataclasses

import pytest
import torch

from models.config import AudioConfig, FreqPoolConfig, FrontendConfig, load_model_config
from models.frontends import build_frontend, frames_for, hop_length

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
    # ⚠️ The third row is deliberately NOT a whole number of frames: `frames_for`
    # rounds up, so it gets 26 frames of which the last is part padding. Every
    # length here used to divide the 320-sample hop exactly.
    lengths = torch.tensor([SR * 4, SR * 2, SR // 2 + 1])
    _, mask = _stub()(wav, lengths)
    assert mask.sum(-1).tolist() == [200, 100, 26]


def test_the_boundary_frame_ignores_what_lies_past_lengths():
    """🔴 A row's features are a function of its own samples, full stop.

    `frames_for` rounds up, so a row whose length is not a multiple of `hop` has
    a last valid frame that is part padding -- and it is masked *in*. Before
    `Frontend.forward` zeroed past `lengths`, whatever filled that pad was
    encoded into a frame the model attends to and takes `frame_max` over: two
    pad fillings moved a submitted probability by 1.35e-3 on rendered audio.

    ⚠️ Bitwise, and over the **valid** frames only. The masked-out frames are
    allowed to differ -- nothing reads them.
    """
    fe = _stub()
    n = SR * 2 + 137                       # 137 samples into the 320-sample hop
    assert n % hop_length(SR, 50.0)
    torch.manual_seed(0)
    row = torch.randn(1, n)
    lengths = torch.tensor([n])

    quiet = torch.nn.functional.pad(row, (0, SR))
    loud = torch.cat([row, 50.0 * torch.randn(1, SR)], dim=-1)
    a, mask = fe(quiet, lengths)
    b, _ = fe(loud, lengths)
    assert torch.equal(a[mask], b[mask])

    # And the check is not vacuous: without `lengths` there is nothing to zero
    # past, so the pad is encoded and the very same frames move.
    assert not torch.equal(fe(quiet)[0][mask], fe(loud)[0][mask])


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


def test_shipped_stub_configs_build():
    """Both candidates must be constructible before any checkpoint exists."""
    for name in ("a_stub", "b_stub"):
        cfg = load_model_config(f"configs/{name}.yaml")
        for fe_cfg in cfg.frontends.values():
            fe = build_frontend(fe_cfg, cfg.audio).eval()
            feats, mask = fe(torch.randn(1, SR * 5))
            assert feats.shape == (1, 250, fe_cfg.output_dim)
            assert mask.all()


@pytest.mark.parametrize("field,value", [
    ("layers", 9),
    ("adapter", "LORA"),
])
def test_stub_refuses_knobs_it_cannot_honour(field, value):
    """🔴 A knob that validates and then does nothing is how an ablation ends up
    measuring a setting that never took effect."""
    from models.config import AdapterConfig
    kw = {field: (AdapterConfig(kind="lora") if value == "LORA" else value)}
    with pytest.raises(NotImplementedError):
        build_frontend(FrontendConfig(name="stub", output_dim=32, **kw), AUDIO)


def test_freeze_actually_freezes_the_encoder_but_not_our_pooling():
    fe = _patch_stub(freeze=True)
    enc = [p for n, p in fe.named_parameters() if not n.startswith("freq_pool")]
    assert not any(p.requires_grad for p in enc), "encoder must be frozen"
    assert fe.freq_pool.p.requires_grad, "GeM exponent is ours, not the checkpoint's"

    thawed = _patch_stub(freeze=False)
    assert all(p.requires_grad for p in thawed.parameters())
