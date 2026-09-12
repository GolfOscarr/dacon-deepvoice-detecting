"""S tier: spectral scalars (docs/EDA/00 section 4).

🔴 This is the tier that can answer the question the whole music-head strategy
rests on. Generated audio is expected to carry a fingerprint above 8 kHz --
vocoder artefacts, a resampling shelf, a hard low-pass where the generator's
own rate ended. The competition standardizes every file to 16 kHz, whose
Nyquist is 8 kHz, so **any fingerprint living above it is gone before the model
sees it**. `effective_bandwidth_hz` measured on both planes is the measurement:
a large `native` spread that collapses at `chain` means the shortcut was never
available, and no published work has tested it (survey/10).

`near_nyquist_ratio` is the plan's sharpening of it: the ratio of energy in
[0.94, 0.99]*Nyquist to [0.80, 0.94]*Nyquist, near zero for a file whose content
stops just short of its own Nyquist and near one for a genuine full-band
recording.

🔴 **It only sees a cutoff that falls inside its own two bands, which is a
narrower question than it looks.** Measured here: a 44.1 kHz file band-limited
to 8 kHz -- the shape a 16 kHz generator republished at 44.1 actually has --
reports **0.37**, not ~0. Both bands sit far inside the stopband, so the
statistic is the ratio of two near-zero numbers and carries the filter's skirt
shape rather than the cutoff. `hf_ratio_8k` is the column that answers the
question the music head needs asked, and it exists because of this.

⚠️ The `[128]`-wide statistics of the plan -- `ltas`, `mel_band_skew`,
`mel_band_kurt` -- are **not** here. They are vectors and belong in
`vectors.npz` keyed by `file_id`, not in a scalar table (docs/EDA/00 section 1);
this module is the scalar half and the two are written by the same pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from eda.extract import register_signal
from eda.extract.level import DB_FLOOR as LEVEL_DB_FLOOR

__all__ = ["spectral_row"]

#: The plan's grid: `n_fft=1024, hop=256`. Welch wants the overlap instead.
N_FFT = 1024
HOP = 256
#: Bandwidth is the highest frequency still within this of the peak bin.
BANDWIDTH_DROP_DB = 60.0
#: 8 x 1 kHz bands, so the set covers 0-8 kHz: the chain plane's whole range,
#: and the part of the native plane that survives into it. Bands above 8 kHz
#: are deliberately not tabulated -- `effective_bandwidth_hz` and
#: `near_nyquist_ratio` are what describe that region, and they do it in terms
#: of the file's own Nyquist rather than in absolute bins that only exist for
#: some sample rates.
BAND_EDGES_HZ = tuple(range(0, 9000, 1000))

#: The chain plane's Nyquist. Energy above it is energy the model never sees.
CHAIN_NYQUIST_HZ = 8000.0

COLUMNS = (
    "effective_bandwidth_hz", "near_nyquist_ratio", "hf_ratio_8k",
    "spectral_flatness", "spectral_centroid_hz", "nyquist_hz",
    *(f"band_energy_{i}" for i in range(len(BAND_EDGES_HZ) - 1)),
    "spectral_ok", "spectral_error",
)


@register_signal("spectral", COLUMNS, "spectral_ok")
def spectral_row(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Welch PSD scalars over `(C, n)` float32.

    Channels are downmixed: the spectrum of a stereo file is the spectrum of
    what the chain plane will hand the model, and averaging per-channel spectra
    instead would report a bandwidth no single channel has.
    """
    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return {**{c: None for c in COLUMNS}, "spectral_ok": False,
                "spectral_error": "empty waveform"}
    mono = wav.mean(axis=0)
    nperseg = min(N_FFT, mono.size)
    if nperseg < 16:
        return {**{c: None for c in COLUMNS}, "spectral_ok": False,
                "spectral_error": f"only {mono.size} sample(s); need >= 16"}
    # ⚠️ `welch` detrends each segment (`detrend="constant"`) by default, so a
    # DC offset is invisible here and a pure-DC signal reports as spectrally
    # silent. That is the right split of labour -- DC is a level property and
    # `level.dc_offset` measures it -- but it means "silent" in this module
    # means "no varying content", not "all samples are zero".
    freqs, psd = welch(mono, fs=sample_rate, nperseg=nperseg,
                       noverlap=max(0, nperseg - HOP), scaling="density")
    total = float(psd.sum())
    nyquist = sample_rate / 2.0

    if total <= 0:
        # Digital silence. Every ratio below would be 0/0; say so once rather
        # than emit a row of NaNs that reads like a decode failure.
        return {**{c: 0.0 for c in COLUMNS if c.startswith("band_energy_")},
                "effective_bandwidth_hz": 0.0, "near_nyquist_ratio": 0.0,
                "hf_ratio_8k": 0.0,
                "spectral_flatness": 0.0, "spectral_centroid_hz": 0.0,
                "nyquist_hz": nyquist, "spectral_ok": True,
                "spectral_error": "silent: spectrum is identically zero"}

    peak = float(psd.max())
    above = np.flatnonzero(psd >= peak * 10.0 ** (-BANDWIDTH_DROP_DB / 10.0))
    bandwidth = float(freqs[above[-1]]) if above.size else 0.0

    def _band(lo: float, hi: float, *, closed: bool = False) -> float:
        # ⚠️ Half-open, except at the top. Welch puts a bin at exactly Nyquist,
        # and a half-open top band drops it -- the eight bands then summed to
        # 0.99902 of the spectrum at 16 kHz while claiming to tile it. A
        # fraction that is silently not a fraction is the kind of column a
        # reader normalizes by twice.
        sel = (freqs >= lo) & (freqs <= hi if closed else freqs < hi)
        return float(psd[sel].sum())

    near = _band(0.94 * nyquist, 0.99 * nyquist)
    below = _band(0.80 * nyquist, 0.94 * nyquist)

    # Geometric over arithmetic mean, in the log domain so a long PSD cannot
    # underflow the product to zero -- which it does around 400 bins.
    #
    # ⚠️ Exactly-zero bins are floored rather than dropped, which makes this
    # function total: `log(0)` is -inf, `exp(-inf)` is 0, and a flatness of
    # exactly 0.0 is already this module's way of saying "silent". A computed
    # 0.0 and a gave-up 0.0 would then be the same value.
    #
    # They are rare. Measured while writing this: a 16-bit tone behind a 5 s
    # digital-silence pad -- the TTS shape, and the likeliest real case --
    # produces **none**, because Welch averages over segments. The construction
    # that does is a lone impulse inside an otherwise all-zero file. The floor
    # is cheap and the alternative is a NaN nobody traces back to here.
    #
    # The floor is 120 dB below the peak: the same dynamic range `level.DB_FLOOR`
    # uses, chosen there for the same reason, and far below any real audio
    # content. Being relative to the peak keeps the statistic scale-invariant,
    # which is the only property flatness is worth anything for.
    floor = peak * 10.0 ** (LEVEL_DB_FLOOR / 10.0)
    flatness = float(np.exp(np.mean(np.log(np.maximum(psd, floor)))) / np.mean(psd))

    row: dict[str, Any] = {
        "effective_bandwidth_hz": bandwidth,
        # 0.0 when the octave below is empty too: the file has nothing up
        # there at all, which is a different statement from "the top is
        # relatively quiet" and must not read as one.
        "near_nyquist_ratio": float(near / below) if below > 0 else 0.0,
        # 🔴 The music-head question in one number: what fraction of this file's
        # energy is above the chain's Nyquist, and therefore gone before the
        # model sees anything? Unlike `near_nyquist_ratio` this is an absolute
        # band, so it means the same thing at every native rate -- and it is
        # identically ~0 on the chain plane by construction, which is the point:
        # the native-minus-chain difference *is* the discarded fingerprint.
        "hf_ratio_8k": _band(CHAIN_NYQUIST_HZ, nyquist, closed=True) / total,
        "spectral_flatness": flatness,
        "spectral_centroid_hz": float((freqs * psd).sum() / total),
        "nyquist_hz": nyquist,
        "spectral_ok": True,
        "spectral_error": None,
    }
    top = len(BAND_EDGES_HZ) - 2
    for i, (lo, hi) in enumerate(zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:])):
        # Fraction, not absolute power: absolute bands would rank sources by
        # how loud they were mastered, which `level` already reports and which
        # would drown the spectral shape this column exists to show.
        row[f"band_energy_{i}"] = _band(lo, hi, closed=(i == top)) / total
    return row
