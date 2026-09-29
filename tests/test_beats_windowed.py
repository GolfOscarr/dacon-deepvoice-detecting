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


# --- FrontendConfig.batched_tokens: the loop over rows and windows, batched ---

def _batched_pair():
    loop = _fe(62)
    cfg = load_model_config("configs/c_first_run.yaml")
    fe = dataclasses.replace(cfg.frontends["audio"], weights=BEATS, window_patches=62,
                             batched_tokens=True)
    batched = build_frontend(fe, cfg.audio).eval()
    batched.load_state_dict(loop.state_dict())
    return loop, batched


def test_batched_fbank_is_kaldi_fbank():
    """Every kept frame of every row is `_fbank` over that row's own prefix --
    to float rounding: the mel matmul's BLAS kernel depends on the frame count
    (a 17-frame row differs from its solo self by 1.8e-07 on CPU)."""
    _, fe = _batched_pair()
    g = torch.Generator().manual_seed(2)
    wav = torch.randn(3, SR * 7, generator=g) * 0.1
    lengths = [SR * 7, SR * 3 + 123, 3001]
    for i, n in enumerate(lengths):
        wav[i, n:] = 0.0
    frames = int(fe._mel_frames(SR * 7))
    got = fe._fbank_batch(wav, frames)
    for i, n in enumerate(lengths):
        ref = fe._fbank(wav[i, :n])
        assert torch.allclose(got[i, :ref.shape[0]], ref, atol=1e-5, rtol=0)


def test_batched_tokens_match_the_row_loop():
    """Masks exactly; features to float rounding. Includes a row under one patch
    (the tail-repeat rule) and one under one kaldi window (400 samples)."""
    loop, batched = _batched_pair()
    g = torch.Generator().manual_seed(3)
    wav = torch.randn(5, SR * 23, generator=g) * 0.1
    lengths = torch.tensor([SR * 23, SR * 13 + 77, SR * 10, 2000, 350])
    with torch.no_grad():
        a, ma = loop(wav, lengths)
        b, mb = batched(wav, lengths)
    assert torch.equal(ma, mb)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-5, rtol=0)


def test_batched_tokens_do_not_read_a_neighbour():
    """Rule 2.4: same batch shape, different neighbour content -> bitwise equal."""
    _, batched = _batched_pair()
    g = torch.Generator().manual_seed(4)
    wav = torch.randn(3, SR * 21, generator=g) * 0.1
    other = wav.clone()
    other[0] = torch.randn(SR * 21, generator=g)
    other[2] *= 50.0
    lengths = torch.tensor([SR * 21, SR * 12, SR * 17])
    with torch.no_grad():
        a, _ = batched(wav, lengths)
        b, _ = batched(other, lengths)
    assert torch.equal(a[1], b[1])
