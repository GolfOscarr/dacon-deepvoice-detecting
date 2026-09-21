"""Step 12: the `pink_noise` and `rir` augments -- shape, SNR, alignment,
determinism, and the real bank on this machine when present."""

from pathlib import Path

import numpy as np
import pytest
import torch

from training.registries import AUGMENT

SR = 16_000
BANK = Path("/data/project/private/dacon-corpus/interim/rirs-noises/openslr-28/RIRS_NOISES/"
            "real_rirs_isotropic_noises")


def _tone(hz=440.0, n=2 * SR, amp=0.3):
    t = torch.arange(n, dtype=torch.float64) / SR
    return (amp * torch.sin(2 * np.pi * hz * t)).float()[None]


def _band_db(x, lo, hi):
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / SR)
    p = torch.fft.rfft(x[0].double()).abs() ** 2
    return 10 * np.log10(float(p[(f >= lo) & (f < hi)].mean()))


def test_pink_noise_falls_three_db_per_octave_and_sits_at_its_snr():
    x = torch.zeros(1, 4 * SR) + 1e-6
    out = AUGMENT.build("pink_noise", {"snr_db": 0.0})(x.clone(), np.random.default_rng(0))
    noise = out - x
    slope = _band_db(noise, 200, 400) - _band_db(noise, 1600, 3200)
    assert 8.0 < slope < 10.0, slope             # 3 octaves at ~3 dB each
    sig = _tone()
    out = AUGMENT.build("pink_noise", {"snr_db": 20.0})(sig.clone(), np.random.default_rng(1))
    snr = 20 * np.log10(float(sig.pow(2).mean().sqrt() / (out - sig).pow(2).mean().sqrt()))
    assert snr == pytest.approx(20.0, abs=0.1)


def test_rir_preserves_length_and_level_and_is_identity_when_dry():
    x = _tone()
    out = AUGMENT.build("rir", {"wet": 0.0})(x.clone(), np.random.default_rng(0))
    assert torch.allclose(out, x, atol=1e-6)
    out = AUGMENT.build("rir", {"wet": 1.0})(x.clone(), np.random.default_rng(0))
    assert out.shape == x.shape
    assert float(out.pow(2).mean().sqrt()) == pytest.approx(float(x.pow(2).mean().sqrt()), rel=1e-4)
    assert not torch.allclose(out, x, atol=1e-3)


def test_rir_is_registered_with_no_delay_and_is_reproducible():
    """Registration ran the time-warp probe (lag 0 at head and tail); this
    pins the reproducibility from the rng alone."""
    assert AUGMENT.group_delay_of("rir") == 0
    x = torch.from_numpy(np.random.default_rng(2).standard_normal((2, SR)).astype(np.float32))
    a = AUGMENT.build("rir")(x.clone(), np.random.default_rng(7))
    b = AUGMENT.build("rir")(x.clone(), np.random.default_rng(7))
    assert torch.equal(a, b)
    assert not torch.equal(a, AUGMENT.build("rir")(x.clone(), np.random.default_rng(8)))


@pytest.mark.skipif(not BANK.exists(), reason="the RIRS bank is not on this machine")
def test_the_real_bank_is_filtered_to_responses_and_aligned():
    from training.registries import _load_rir, _rir_bank
    files = _rir_bank(str(BANK), "*_rir_*.wav")
    assert len(files) == 218 and all("_rir_" in f for f in files)
    h = _load_rir(files[0], 0, 1.0)
    assert int(np.argmax(np.abs(h))) == 0 and len(h) <= SR
    assert np.linalg.norm(h) == pytest.approx(1.0, rel=1e-5)
    x = _tone()
    out = AUGMENT.build("rir", {"bank_dir": str(BANK), "pattern": "*_rir_*.wav", "wet": 0.8})(
        x.clone(), np.random.default_rng(0))
    assert out.shape == x.shape and not torch.allclose(out, x, atol=1e-3)
    # the direct path is at 0: a burst's onset does not move
    burst = torch.zeros(1, SR); burst[0, 4000:4400] = 0.5
    wet = AUGMENT.build("rir", {"bank_dir": str(BANK), "pattern": "*_rir_*.wav", "wet": 1.0})(
        burst.clone(), np.random.default_rng(3))
    assert int((wet[0].abs() > 0.05).nonzero()[0]) in range(3995, 4010)


def test_an_empty_bank_is_loud(tmp_path):
    with pytest.raises(Exception, match="no file matches"):
        AUGMENT.build("rir", {"bank_dir": str(tmp_path)})(_tone(), np.random.default_rng(0))
