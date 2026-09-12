"""S tier: level and dynamics (docs/EDA/00 section 4).

Why these and not a loudness model: the question the shortcut audit asks of pool
level is *"can a head tell the corpora apart by how loud they are"*, and that is
a question about the distribution of `rms_dbfs` and `crest_factor` across
sources, not about perceived loudness. Level is also the one statistic a
render-time transform can neutralize outright, so measuring it before choosing
the transform is what tells us whether the transform is needed.

⚠️ `lufs_integrated` is in the plan's statistic set and is **not** implemented
here. BS.1770 K-weighting is specified at 48 kHz and has to be redesigned per
sample rate, and the native plane runs at whatever the publisher shipped --
22.05, 24, 32, 44.1, 48 kHz all appear in this corpus. A K-weighting filter
applied at the wrong rate does not fail; it returns a plausible number that is
wrong by a few LU, which is exactly the size of the effect we would be looking
for. It is deferred rather than approximated; `rms_dbfs` and `crest_factor`
answer the level-shortcut question without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eda.extract import register_signal

__all__ = ["level_row"]

#: Below this, `20*log10` is reported as this floor rather than `-inf`. Parquet
#: holds -inf, but every consumer -- quantiles, means, a boxplot -- silently
#: becomes -inf with it, and a single digitally-silent file would erase a
#: source's whole level distribution.
DB_FLOOR = -120.0
#: A sample within this of full scale counts as clipped. 1 LSB at 16-bit, which
#: is what the plan specifies: the corpus is overwhelmingly 16-bit PCM and
#: re-encoded float audio keeps the flat tops.
CLIP_EPS = 1.0 / 32768.0

COLUMNS = ("peak_dbfs", "rms_dbfs", "crest_factor_db", "dc_offset",
           "clipping_ratio", "stat_t_std", "stat_t_var", "stat_t_rms",
           "stat_t_pwr", "level_ok", "level_error")


def _dbfs(x: float) -> float:
    return DB_FLOOR if x <= 0 else max(DB_FLOOR, float(20.0 * np.log10(x)))


@register_signal("level", COLUMNS, "level_ok")
def level_row(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Level and dynamics over `(C, n)` float32.

    Channels are reduced by taking the measurement over every sample of every
    channel rather than over a downmix: a downmix hides a channel that is
    silent, and "one channel is silent" is a real property of scraped stereo
    music that the mid_side channel policy would turn into signal.
    """
    if wav.size == 0:
        return {**{c: None for c in COLUMNS}, "level_ok": False,
                "level_error": "empty waveform"}
    flat = np.asarray(wav, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(flat)))
    power = float(np.mean(flat * flat))
    rms = float(np.sqrt(power))
    return {
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
        # Crest is peak-over-rms *in dB*, so the name carries the unit. A bare
        # `crest_factor` invites a reader to compare it against a linear ratio
        # from another table; the corpus has already paid once for a column
        # whose unit lived only in someone's head.
        "crest_factor_db": _dbfs(peak) - _dbfs(rms),
        "dc_offset": float(np.mean(flat)),
        "clipping_ratio": float(np.mean(np.abs(flat) >= 1.0 - CLIP_EPS)),
        # statistics-T, [BirdCLEF 2024 1st]: four moments of the same signal,
        # kept separately rather than summed. The published feature is their
        # sum; summing here would make the 0.8-quantile threshold in that
        # solution unvalidatable against our own data, which is the whole
        # reason the plan says to record it.
        "stat_t_std": float(np.std(flat)),
        "stat_t_var": float(np.var(flat)),
        "stat_t_rms": rms,
        "stat_t_pwr": power,
        "level_ok": True,
        "level_error": None,
    }
