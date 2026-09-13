"""S tier: the `[128]`-wide statistics (docs/EDA/00 §1, docs/EDA/09 step 3).

`ltas`, `mel_band_skew`, `mel_band_kurt` -- the TISMIR music features, recorded
because they are **expected to die at 16 kHz** and the only way to know is to
measure the same content on both planes. They are vectors, so they go to
`vectors.npz` keyed by `file_id` rather than into `signal.parquet`: one table
with 128-wide columns would force a choice between 384 mostly-null columns per
row and silently dropping the thing X1 needs.

All three come out of **one** log-mel spectrogram, which is why they are one
module and not three. The scalar half is `spectral.py`; the two are written by
the same pass over the same decode.

🔴 **The mel bank is pinned to absolute Hz, 0-8000, on both planes.** The
obvious alternative -- 0 to each plane's own Nyquist -- makes band 61 mean
5.2 kHz on a 44.1 kHz file and 2.4 kHz on the chain plane, and then
`native - chain` is not a subtraction of anything. `spectral.py` settled the
same question the same way for `hf_ratio_8k`: *"an absolute band, so it means
the same thing at every native rate"*.

⚠️ The consequence, stated rather than hidden: **these vectors cannot see above
8 kHz**, which is where R1's fingerprint is supposed to live. That region is the
scalar half's job -- `hf_ratio_8k`, `effective_bandwidth_hz`,
`near_nyquist_ratio` -- and it is deliberate that the comparable statistic and
the above-Nyquist statistic are different columns. A file whose native rate is
below 16 kHz simply has empty top bands, which is true and visible.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis, skew

from eda.extract import register_vector
from eda.extract.level import DB_FLOOR
from eda.extract.spectral import HOP, N_FFT

__all__ = ["N_MELS", "MEL_HZ_MAX", "mel_vectors"]

#: Bands. The plan's `[128]`.
N_MELS = 128
#: The top of the bank, in Hz. The chain plane's Nyquist -- see the module
#: docstring for why this is absolute rather than per-plane.
MEL_HZ_MAX = 8000.0
MEL_HZ_MIN = 0.0

COLUMNS = ("ltas", "mel_band_skew", "mel_band_kurt", "mel_bands_flat",
           "vector_ok", "vector_error")
#: Every vector column is this wide. Declared so a failure can emit NaNs of the
#: right shape rather than a ragged array that breaks the stack at merge time.
WIDTH = N_MELS


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK mel. Deliberately the plain formula rather than Slaney's, because it
    has one definition and nothing here needs to match a published filterbank."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def _filterbank(sample_rate: int, n_fft: int) -> np.ndarray:
    """`(n_freqs, N_MELS)` triangular bank over `[MEL_HZ_MIN, MEL_HZ_MAX]`.

    Cached on `(sample_rate, n_fft)`: the corpus holds a handful of distinct
    rates and rebuilding this per file is ~90k needless allocations.

    ⚠️ Not area-normalised. These statistics are read as a *shape* -- the skew
    of one band across time, and the same band before and after the chain --
    and a normalisation that divides by band width would make neighbouring
    bands incomparable for no gain. `ltas` is reported in dB relative to its own
    peak for the same reason.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(MEL_HZ_MIN), _hz_to_mel(MEL_HZ_MAX),
                                   N_MELS + 2))
    bank = np.zeros((len(freqs), N_MELS), dtype=np.float64)
    for m in range(N_MELS):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (freqs >= lo) & (freqs <= mid)
        falling = (freqs > mid) & (freqs <= hi)
        if mid > lo:
            bank[rising, m] = (freqs[rising] - lo) / (mid - lo)
        if hi > mid:
            bank[falling, m] = (hi - freqs[falling]) / (hi - mid)
    return bank


def _failed(error: str) -> dict[str, Any]:
    nan = np.full(WIDTH, np.nan, dtype=np.float32)
    return {"ltas": nan, "mel_band_skew": nan.copy(),
            "mel_band_kurt": nan.copy(), "mel_bands_flat": None,
            "vector_ok": False, "vector_error": error}


# ⚠️ Named `vector`, matching its `vector_ok` / `vector_error` columns.
# `Extractor.failed` finds the error column as `f"{name}_error"`, so a
# name that does not match silently drops the error text on failure.
@register_vector("vector", COLUMNS, "vector_ok", width=WIDTH,
                 vectors=("ltas", "mel_band_skew", "mel_band_kurt"))
def mel_vectors(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Log-mel `[128]` statistics over `(C, n)` float32.

    * `ltas` -- the long-term average spectrum: the mean over time of each
      band's power, in dB **relative to the loudest band**. Relative because an
      absolute LTAS ranks files by mastering level, which `level.py` already
      reports and which would drown the spectral shape this exists to show.
    * `mel_band_skew`, `mel_band_kurt` -- the skew and excess kurtosis of each
      band's log-power *across time*. A steady tone and a series of transients
      can share an LTAS and differ entirely here, which is the whole reason the
      pair is recorded beside the mean.

    Channels are downmixed first, exactly as `spectral.py` does: the spectrum
    of a stereo file is the spectrum of what the chain hands the model.
    """
    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return _failed("empty waveform")
    mono = wav.mean(axis=0)
    nperseg = min(N_FFT, mono.size)
    if nperseg < 16:
        return _failed(f"only {mono.size} sample(s); need >= 16")

    _, _, spec = stft(mono, fs=sample_rate, nperseg=nperseg,
                      noverlap=max(0, nperseg - HOP), padded=False,
                      boundary=None)
    power = np.abs(spec) ** 2                        # (n_freqs, n_frames)
    if power.shape[1] < 2:
        # 🔴 One frame is not a time series. skew and kurtosis of a single
        # observation are NaN, and emitting them beside a perfectly good `ltas`
        # would publish two columns of NaN that read as a decode failure.
        return _failed(f"{power.shape[1]} STFT frame(s); skew and kurtosis "
                       f"over time need at least 2")

    bank = _filterbank(int(sample_rate), nperseg)
    mel = bank.T @ power                             # (N_MELS, n_frames)

    peak = float(mel.max())
    if peak <= 0.0:
        # Digital silence below 8 kHz. `spectral.py` makes the same call and
        # for the same reason: say it once rather than emit NaNs that read as a
        # decode failure.
        nan = np.full(WIDTH, np.nan, dtype=np.float32)
        return {"ltas": np.full(WIDTH, DB_FLOOR, dtype=np.float32),
                "mel_band_skew": nan, "mel_band_kurt": nan.copy(),
                "mel_bands_flat": WIDTH, "vector_ok": True,
                "vector_error": "silent: no energy below 8 kHz"}

    # Floored at DB_FLOOR below the peak, the same dynamic range `level.py` and
    # `spectral.py` use. Without it an empty top band -- every file whose native
    # rate is below 16 kHz has some -- takes log(0) and poisons the skew of a
    # band whose real answer is "nothing here".
    floor = peak * 10.0 ** (DB_FLOOR / 10.0)
    log_mel = 10.0 * np.log10(np.maximum(mel, floor))

    # 🔴 A band that never moves has no skew and no kurtosis -- the estimators
    # are 0/0 -- and that is a real state rather than a failure: every file
    # whose native rate is below 16 kHz has empty top bands, and an mp3's
    # bottom two bands sit on the floor too. Measured on pool C: 3.6% of native
    # bands, 0.0% of chain bands.
    #
    # ⚠️ They stay NaN, and the **count is reported** so a reader who finds one
    # can tell "this band is silent" from "something went wrong". Filling them
    # with 0.0 instead would put a perfectly symmetric, perfectly Gaussian band
    # in the table wherever there was no audio at all.
    flat = int((log_mel.max(axis=1) - log_mel.min(axis=1) <= 0.0).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        band_skew = skew(log_mel, axis=1, bias=False)
        band_kurt = kurtosis(log_mel, axis=1, fisher=True, bias=False)

    return {
        "ltas": (log_mel.mean(axis=1) - 10.0 * np.log10(peak)
                 ).astype(np.float32),
        "mel_bands_flat": flat,
        # `bias=False` is the sample estimator, and `fisher=True` makes the
        # kurtosis excess -- 0 for a Gaussian band rather than 3.
        "mel_band_skew": band_skew.astype(np.float32),
        "mel_band_kurt": band_kurt.astype(np.float32),
        "vector_ok": True,
        "vector_error": None,
    }
