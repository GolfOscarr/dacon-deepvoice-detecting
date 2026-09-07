"""Waveform preparation. The band filter is the part that broke rule 2.4."""

import pytest
import torch

from models.audio import bandpass, prepare_waveform
from models.config import AudioConfig

SR = 16_000
BAND = AudioConfig(band_hz=(0.0, 4000.0))


# --------------------------------------------------------------------------- #
# 🔴 the band filter must not see padding

@pytest.mark.parametrize("pad", [1, 320, SR, SR * 5])
def test_filter_output_is_independent_of_padding(pad):
    """The rFFT grid is a function of the transform length, so filtering a
    padded batch tensor made the filtered signal depend on the batch.

    Measured before the fix, on unit-variance audio: **1 sample** of padding
    moved the filtered waveform by up to 1.0, and 1 s by ~0.6. That is a
    different waveform reaching the frontends, not a rounding deviation.
    """
    torch.manual_seed(0)
    x = torch.randn(1, SR * 4)
    lengths = torch.tensor([SR * 4])
    ref = bandpass(x, BAND, lengths)[0, :SR * 4]
    got = bandpass(torch.nn.functional.pad(x, (0, pad)), BAND, lengths)[0, :SR * 4]
    assert torch.equal(ref, got), f"{pad} samples of padding changed the filtered signal"


def test_rows_of_different_lengths_are_filtered_independently():
    torch.manual_seed(1)
    a, b = torch.randn(1, SR * 4), torch.randn(1, SR * 7)
    lengths = torch.tensor([SR * 4, SR * 7])
    batch = torch.cat([torch.nn.functional.pad(a, (0, SR * 3)), b])
    out = bandpass(batch, BAND, lengths)
    assert torch.equal(out[0, :SR * 4], bandpass(a, BAND, torch.tensor([SR * 4]))[0])
    assert torch.equal(out[1], bandpass(b, BAND, torch.tensor([SR * 7]))[0])


def test_padding_region_stays_zero():
    x = torch.randn(1, SR * 2)
    out = bandpass(torch.nn.functional.pad(x, (0, SR)), BAND, torch.tensor([SR * 2]))
    assert out[0, SR * 2:].abs().max() == 0.0


def test_band_actually_removes_energy_above_the_cutoff():
    t = torch.arange(SR * 2) / SR
    x = (torch.sin(2 * torch.pi * 500 * t) + torch.sin(2 * torch.pi * 6000 * t))[None]
    out = bandpass(x, BAND, torch.tensor([SR * 2]))
    spec = torch.fft.rfft(out[0].double()).abs()
    freqs = torch.fft.rfftfreq(SR * 2, d=1.0 / SR)
    passband = spec[(freqs > 400) & (freqs < 600)].max()
    stopband = spec[freqs > 4100].max()
    # Relative, not absolute: the residual is float32 round-trip noise from the
    # irfft cast, and an absolute threshold just encodes the input's amplitude.
    assert passband > 100, "the 500 Hz tone must survive"
    assert stopband / passband < 1e-6, f"6 kHz tone not suppressed ({stopband/passband:.2e})"


def test_no_band_is_a_passthrough():
    x = torch.randn(2, 100)
    assert torch.equal(bandpass(x, AudioConfig()), x)


def test_bandpass_rejects_non_2d():
    with pytest.raises(ValueError, match=r"B, S"):
        bandpass(torch.randn(1, 2, 100), BAND)


# --------------------------------------------------------------------------- #
# channel policy

@pytest.mark.parametrize("policy,expected", [
    ("downmix", 1.5), ("left", 1.0), ("mid_side", 1.5),
])
def test_channel_policies(policy, expected):
    stereo = torch.stack([torch.ones(1, 8), torch.full((1, 8), 2.0)], dim=1)
    out = prepare_waveform(stereo, AudioConfig(channels=policy))
    assert out.shape == (1, 8)
    assert out[0, 0].item() == pytest.approx(expected)


def test_mono_passes_through_whatever_the_policy():
    x = torch.randn(2, 100)
    for policy in ("downmix", "left", "mid_side"):
        assert torch.equal(prepare_waveform(x, AudioConfig(channels=policy)), x)
    single = torch.randn(2, 1, 100)
    assert torch.equal(prepare_waveform(single, AudioConfig()), single.squeeze(1))


def test_prepare_rejects_bad_rank():
    with pytest.raises(ValueError, match=r"B, C, S"):
        prepare_waveform(torch.randn(2, 2, 3, 4), AudioConfig())
