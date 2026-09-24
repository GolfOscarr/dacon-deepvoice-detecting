"""docs/training/07 §3: BEATs encoded in 62-column windows.

A 60 s row was one 3,000-token sequence, and BEATs' attention was 3.7 of a
3.9 s training step. `window_patches` cuts the patch grid into ~10 s windows
(BEATs' pretraining length) after the filterbank, so frame counts and masks are
unchanged, and batches equal-length windows across rows.
"""

import dataclasses
import os

import pytest
import torch

from models.config import load_model_config
from models.frontends import build_frontend

BEATS = os.environ.get("DACON_BEATS_WEIGHTS", "/data/project/private/dacon-weights/beats")
pytestmark = pytest.mark.skipif(not os.path.isdir(BEATS), reason=f"no BEATs under {BEATS}")
SR = 16000


def _fe(window):
    cfg = load_model_config("configs/c_first_run.yaml")
    fe = dataclasses.replace(cfg.frontends["audio"], weights=BEATS, window_patches=window)
    torch.manual_seed(0)
    return build_frontend(fe, cfg.audio).eval()


@pytest.fixture(scope="module")
def pair():
    return _fe(None), _fe(62)


def test_a_row_shorter_than_one_window_is_unchanged(pair):
    whole, windowed = pair
    windowed.load_state_dict(whole.state_dict())
    wav = torch.randn(1, SR * 6)
    with torch.no_grad():
        a, ma = whole(wav)
        b, mb = windowed(wav)
    assert torch.equal(ma, mb)
    assert torch.allclose(a, b, atol=1e-4)


def test_frame_counts_and_masks_do_not_change(pair):
    whole, windowed = pair
    wav = torch.randn(2, SR * 25)
    lengths = torch.tensor([SR * 25, SR * 13])
    with torch.no_grad():
        a, ma = whole(wav, lengths)
        b, mb = windowed(wav, lengths)
    assert a.shape == b.shape and torch.equal(ma, mb)


def test_a_long_row_does_not_depend_on_its_batch(pair):
    _, windowed = pair
    g = torch.Generator().manual_seed(1)
    row = torch.randn(1, SR * 23, generator=g)
    batch = torch.randn(3, SR * 40, generator=g)
    batch[1, :SR * 23] = row[0]
    lengths = torch.tensor([SR * 40, SR * 23, SR * 31])
    with torch.no_grad():
        alone, m1 = windowed(row)
        inb, m2 = windowed(batch, lengths)
    n = int(m1.sum())
    assert int(m2[1].sum()) == n
    assert torch.allclose(alone[0, :n], inb[1, :n], atol=1e-4)


def test_each_window_is_encoded_on_its_own(pair):
    """The first window of a 25 s row equals the whole-sequence encoding of
    exactly that window's audio span."""
    whole, windowed = pair
    windowed.load_state_dict(whole.state_dict())
    wav = torch.randn(1, SR * 25)
    with torch.no_grad():
        b, _ = windowed(wav)
        # 62 patch columns = 62 * 16 mel frames; the span that yields exactly them
        span = (62 * 16 - 1) * 160 + 400
        a, ma = whole(wav[:, :span])
    assert int(ma.sum()) == 62
    assert torch.allclose(a[0, :62], b[0, :62], atol=1e-4)
