"""S tier: time structure (docs/EDA/00 section 4).

The competition samples 4-60 s windows, so what matters about a file is not its
duration but **how much of it is usable**: a 30 s TTS clip that is 6 s of speech
between two pads of digital silence offers one valid window, not six. E3 is the
pool-E form of the same question and `rirs-isotropic` is what it cost -- 314 of
417 files below the 4 s floor, so the group "contributed nothing while still
counting toward the file tally".

Segmentation follows the plan: 0.05 s chunks, chunk power in dBFS, and a chunk
is silence below `SILENCE_DBFS`.

⚠️ The threshold is **absolute dBFS, not relative to the file's peak.** A
relative threshold rescales with the content, so a quiet field recording and a
loud one report the same silence ratio -- which is the right answer for a
speech-activity question and the wrong one here. What this feeds is the
sampler's ability to find a window with audio in it, and the sampler does not
renormalize first. `level.rms_dbfs` is what a reader should pair this with when
they want the relative view.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eda.extract import register_signal

__all__ = ["timing_row"]

#: Chunk length for the energy segmentation, seconds. [BC25 separation notebook]
CHUNK_S = 0.05
#: A chunk quieter than this is silence.
SILENCE_DBFS = -50.0
#: The sampler's floor -- `AudioConfig.min_seconds`. A span shorter than this
#: cannot become a training window whatever else is true of it.
VALID_SPAN_S = 4.0

COLUMNS = ("duration_s_decoded", "silence_ratio", "lead_silence_s",
           "tail_silence_s", "n_valid_spans_ge_4s", "longest_valid_span_s",
           "timing_ok", "timing_error")


def _chunk_db(mono: np.ndarray, chunk: int) -> np.ndarray:
    """Per-chunk power in dBFS. The tail shorter than a chunk is dropped."""
    n = (mono.size // chunk) * chunk
    if n == 0:
        return np.empty(0)
    power = np.mean(np.square(mono[:n].reshape(-1, chunk)), axis=1)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(power, 1e-20))


def _spans(loud: np.ndarray) -> list[tuple[int, int]]:
    """`[(start, stop))` chunk indices of each maximal run of True."""
    if not loud.any():
        return []
    edges = np.diff(np.concatenate(([0], loud.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


@register_signal("timing", COLUMNS, "timing_ok")
def timing_row(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Silence structure and usable-span inventory over `(C, n)` float32.

    Channels are downmixed **for this statistic only**: silence is a property of
    the moment, and a stereo file is not audible in a chunk where both channels
    are quiet. Taking the max across channels instead would report a file as
    loud on the strength of one channel's noise floor.
    """
    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return {**{c: None for c in COLUMNS}, "timing_ok": False,
                "timing_error": "empty waveform"}
    mono = wav.mean(axis=0)
    chunk = max(1, int(round(CHUNK_S * sample_rate)))
    db = _chunk_db(mono, chunk)
    duration = mono.size / sample_rate
    if db.size == 0:
        # Shorter than one chunk. Not a failure -- a finding, and the row says
        # so with numbers rather than nulls: nothing is loud, nothing is valid.
        return {"duration_s_decoded": duration, "silence_ratio": 1.0,
                "lead_silence_s": duration, "tail_silence_s": duration,
                "n_valid_spans_ge_4s": 0, "longest_valid_span_s": 0.0,
                "timing_ok": True, "timing_error": None}

    loud = db > SILENCE_DBFS
    chunk_s = chunk / sample_rate
    spans = _spans(loud)
    lengths = [(stop - start) * chunk_s for start, stop in spans]
    lead = spans[0][0] * chunk_s if spans else duration
    tail = (db.size - spans[-1][1]) * chunk_s if spans else duration
    return {
        "duration_s_decoded": duration,
        "silence_ratio": float(np.mean(~loud)),
        "lead_silence_s": float(lead),
        "tail_silence_s": float(tail),
        "n_valid_spans_ge_4s": int(sum(1 for x in lengths if x >= VALID_SPAN_S)),
        "longest_valid_span_s": float(max(lengths)) if lengths else 0.0,
        "timing_ok": True,
        "timing_error": None,
    }
