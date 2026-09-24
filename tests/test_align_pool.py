"""docs/training/07 §2: an integer fps ratio pools rather than samples.

XLS-R (50 fps) feeds heads on BEATs' 6.25 fps grid. Sampling one source frame
per target frame discarded 7 of every 8 speech frames; the pooled path takes
the mean of the frames each target frame spans, over the sample's own valid
frames only, so padding in the batch cannot move a value (rule 2.4).
"""

import torch

from models.utils import align_time


def test_each_target_frame_is_the_mean_of_its_eight_source_frames():
    x = torch.randn(1, 64, 5)
    m = torch.ones(1, 64, dtype=torch.bool)
    y, ym = align_time(x, m, 8, 50.0, 6.25)
    assert torch.allclose(y, x.reshape(1, 8, 8, 5).mean(dim=2), atol=1e-6)
    assert ym.all()


def test_padding_does_not_move_a_pooled_value():
    short = torch.randn(1, 21, 4)
    alone, m_alone = align_time(short, torch.ones(1, 21, dtype=torch.bool), 3, 50.0, 6.25)
    batch = torch.zeros(2, 60, 4)
    batch[0, :21] = short[0]
    batch[1] = torch.randn(60, 4)
    mask = torch.zeros(2, 60, dtype=torch.bool)
    mask[0, :21] = True
    mask[1] = True
    out, om = align_time(batch, mask, 8, 50.0, 6.25)
    assert torch.equal(om[0, :3], m_alone[0]) and not om[0, 3:].any()
    assert torch.allclose(out[0, :3], alone[0], atol=1e-6)
    # the last partial block averages only its 5 valid frames
    assert torch.allclose(out[0, 2], short[0, 16:21].mean(dim=0), atol=1e-6)


def test_a_non_integer_ratio_keeps_the_interpolating_path():
    x = torch.arange(30, dtype=torch.float32).reshape(1, 30, 1)
    y, _ = align_time(x, torch.ones(1, 30, dtype=torch.bool), 20, 30.0, 20.0)
    assert y.shape == (1, 20, 1)
