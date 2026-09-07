"""align_time is the only place two frontends' clocks are reconciled, and it is
the easiest place to make the model batch-dependent without noticing."""

import pytest
import torch

from models.utils import align_time


def _seq(n_valid, n_pad, d=4, seed=0):
    """A (1, n_valid+n_pad, d) sequence with a known ramp on the valid part."""
    torch.manual_seed(seed)
    total = n_valid + n_pad
    feats = torch.randn(1, total, d)
    feats[0, :n_valid] = torch.arange(n_valid, dtype=torch.float32)[:, None].repeat(1, d)
    feats[0, n_valid:] = 999.0                      # padding is deliberately loud
    mask = torch.zeros(1, total, dtype=torch.bool)
    mask[0, :n_valid] = True
    return feats, mask


def test_identity_when_the_clocks_already_agree():
    feats, mask = _seq(10, 0)
    out, out_mask = align_time(feats, mask, 10, 50.0, 50.0)
    assert torch.equal(out, feats) and torch.equal(out_mask, mask)


def test_upsampling_doubles_the_frame_count():
    feats, mask = _seq(10, 0)
    out, out_mask = align_time(feats, mask, 20, 25.0, 50.0)
    assert out.shape == (1, 20, 4)
    assert out_mask.sum() == 20


@pytest.mark.parametrize("n_pad", [0, 5, 37, 200])
def test_alignment_is_independent_of_padding_width(n_pad):
    """🔴 The regression this file exists for.

    The old implementation interpolated over the padded axis, so the source ->
    target mapping was a ratio of two padded frame counts and moved with the
    batch. Measured drift on the submitted FILE column was 3.5e-4 at 9 s of
    padding, growing monotonically.
    """
    ref, ref_mask = _seq(101, 0)
    ref_out, _ = align_time(ref, ref_mask, 201, 25.0, 50.0)

    feats, mask = _seq(101, n_pad)
    out, _ = align_time(feats, mask, 201, 25.0, 50.0)
    assert torch.allclose(ref_out, out, atol=1e-6), f"padding of {n_pad} moved the alignment"


def test_padded_source_frames_are_never_read():
    """Padding is 999.0 here; nothing that loud may reach the output."""
    feats, mask = _seq(20, 80)
    out, _ = align_time(feats, mask, 40, 25.0, 50.0)
    assert out.max() < 100.0, "a padded source frame leaked into the aligned features"


def test_target_mask_covers_only_real_audio():
    feats, mask = _seq(10, 90)              # 10 valid source frames at 25 fps = 0.4 s
    _, out_mask = align_time(feats, mask, 200, 25.0, 50.0)
    assert out_mask.sum() == 20, "0.4 s at 50 fps is 20 frames"


def test_absolute_time_mapping_is_centred():
    """Target frame j is centred at (j+0.5)/fps_tgt, i.e. source (j+0.5)*r-0.5."""
    feats, mask = _seq(10, 0)               # values are the frame index
    out, _ = align_time(feats, mask, 20, 25.0, 50.0)
    # j=0 -> 0.5*0.5-0.5 = -0.25 -> clamped to 0; j=2 -> 2.5*0.5-0.5 = 0.75
    assert out[0, 0, 0].item() == pytest.approx(0.0)
    assert out[0, 2, 0].item() == pytest.approx(0.75, abs=1e-5)


def test_rejects_bad_arguments():
    feats, mask = _seq(4, 0)
    with pytest.raises(ValueError, match="B, T, D"):
        align_time(feats.squeeze(0), mask, 4, 50.0, 50.0)
    with pytest.raises(ValueError, match="target_frames"):
        align_time(feats, mask, 0, 50.0, 50.0)
    with pytest.raises(ValueError, match="fps"):
        align_time(feats, mask, 4, 0.0, 50.0)
