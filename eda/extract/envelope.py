"""C5: onset and offset envelope morphology -- hard cut, fade, or natural start.

docs/EDA/03 C5. The hypothesis it exists to test is the sharpest one in the
EDA and the only one no spectral analysis could ever surface:

    🔷 If every REAL music sample starts with a hard cut and every FAKE one
    starts from silence, the first 100 ms of the file separates the classes
    perfectly, at 0.27 weight -- because it is an **editing** artifact rather
    than an acoustic one.

FMA's `fma_small` is 30 s excerpts cut from track centres, so it begins and ends
mid-phrase at full level. A pool-D clip is a complete generation with a
beginning and an end. If that asymmetry is real, the music head can be solved by
reading one frame, and every model we train would be doing exactly that without
saying so.

🔴 **Measured on the chain plane, and on every pool, not only on C and D.** The
morphology question is about what the model receives, and the crop policy C5
proposes is a *transform* -- R2 makes it a symmetry obligation, so the rates
have to be known everywhere before one is designed. A statistic computed on the
music pools alone could not tell "pool C is unusual" from "everything except
pool D is unusual".

⚠️ **The thresholds below are a classification, not a measurement, and the
scalars are reported beside the class for that reason.** `onset_level_deficit_db`
and `onset_rise_s` are what was measured; `onset_class` is a reading of them at
one set of cuts. A reader who disagrees with the cuts can re-derive the classes
from the columns without a re-decode, which is the whole reason both are stored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eda.extract.level import DB_FLOOR

__all__ = ["BOUNDARY_S", "COLUMNS", "FADE_MIN_S", "HARD_CUT_DB", "HOP_S",
           "SILENCE_DB", "WINDOW_S", "envelope_row", "classify"]

#: The window each end is judged over. C5 says *"the first and last 0.5 s"*.
BOUNDARY_S = 0.5
#: Envelope framing. 20 ms window, 10 ms hop -- fine enough to resolve the
#: "first 100 ms" the hypothesis turns on, coarse enough not to track pitch.
WINDOW_S = 0.020
HOP_S = 0.010

#: Within this of the clip's own median level counts as **starting at full
#: level**, i.e. a hard cut. 6 dB is a factor of two in amplitude; a clip that
#: opens within 6 dB of its own median did not fade in.
HARD_CUT_DB = 6.0
#: Below this, relative to the clip's median, counts as silence for the
#: lead/tail measurement.
SILENCE_DB = 40.0
#: A rise slower than this is a **fade**; faster is a natural attack. 50 ms is
#: above the slowest instrumental onset and far below any deliberate fade.
FADE_MIN_S = 0.050

COLUMNS = ("onset_level_deficit_db", "onset_rise_s", "onset_lead_silence_s",
           "onset_class", "offset_level_deficit_db", "offset_rise_s",
           "offset_lead_silence_s", "offset_class", "envelope_ref_dbfs",
           "envelope_ok", "envelope_error")


def _envelope_db(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Frame RMS in dBFS, floored. `(n_frames,)`."""
    window = max(1, int(round(WINDOW_S * sample_rate)))
    hop = max(1, int(round(HOP_S * sample_rate)))
    n = 1 + max(0, (mono.size - window) // hop)
    if n < 1:
        return np.empty(0, dtype=np.float64)
    idx = np.arange(window)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # The same floor `level.py` uses, so "silent" means one thing in the EDA.
    return 20.0 * np.log10(np.maximum(rms, 10.0 ** (DB_FLOOR / 20.0)))


def _edge(env: np.ndarray, ref_db: float) -> tuple[float, float, float]:
    """`(level_deficit_db, rise_s, lead_silence_s)` for an envelope read from
    its start. The offset half passes the reversed envelope.

    `level_deficit_db` is **positive when the clip starts below its own level**,
    so a hard cut is near 0 and a fade from silence is large. Signed that way
    because "deficit" reading negative for the common case inverts every
    comparison a reader makes.
    """
    window = min(len(env), max(1, int(round(BOUNDARY_S / HOP_S))))
    head = env[:window]
    deficit = float(ref_db - head[0])

    # Time before the signal first exceeds the silence floor.
    audible = np.flatnonzero(env > ref_db - SILENCE_DB)
    lead = float(audible[0] * HOP_S) if audible.size else float("nan")

    # 🔴 The rise is measured from **first audible**, not from frame zero, and
    # the difference is the whole classification. A generated clip with 0.3 s of
    # leading silence and then an immediate attack has a rise of ~0 and a lead
    # of 0.3; measured from frame zero its rise would be 0.3 and it would be
    # called a *fade* -- collapsing the two classes C5 exists to tell apart.
    # Measured: that is exactly what the first version of this function did, and
    # `test_a_clip_that_starts_from_silence_with_an_attack_reads_as_natural`
    # caught it.
    #
    # Searched over the whole envelope, not just the boundary window: a clip
    # that takes 3 s to arrive has a rise time of 3 s, and truncating at 0.5 s
    # would report the same number as one that never arrives at all.
    reached = np.flatnonzero(env >= ref_db - HARD_CUT_DB)
    if not reached.size or not audible.size:
        return deficit, float("nan"), lead
    rise = float(max(0, reached[0] - audible[0]) * HOP_S)
    return deficit, rise, lead


def classify(deficit_db: float, rise_s: float) -> str:
    """`hard_cut` / `fade` / `natural`, from the two measured scalars.

    ⚠️ One reading of the numbers at one set of cuts. The scalars are stored
    beside it so this can be re-derived without a re-decode.
    """
    if not np.isfinite(deficit_db):
        return "unknown"
    if deficit_db <= HARD_CUT_DB:
        # Opens within 6 dB of its own median: it did not fade in.
        return "hard_cut"
    if not np.isfinite(rise_s):
        # Starts low and never reaches its own median -- not a beginning at all.
        return "unknown"
    return "fade" if rise_s >= FADE_MIN_S else "natural"


def _failed(error: str) -> dict[str, Any]:
    out: dict[str, Any] = {c: None for c in COLUMNS}
    out["envelope_ok"] = False
    out["envelope_error"] = error
    return out


def envelope_row(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Onset and offset morphology for one file. A failure is a row, not a raise."""
    wav = np.asarray(wav, dtype=np.float64)
    if wav.size == 0:
        return _failed("empty waveform")
    mono = wav.mean(axis=0) if wav.ndim > 1 else wav
    need = int(round(WINDOW_S * sample_rate))
    if mono.size < need * 2:
        return _failed(f"{mono.size} sample(s); need at least two {WINDOW_S}s frames")

    env = _envelope_db(mono, sample_rate)
    if env.size < 2:
        return _failed(f"{env.size} envelope frame(s)")

    # 🔴 The reference is the clip's **own** median, not an absolute level.
    # An absolute one would classify every quiet recording as a fade and would
    # re-measure the 22.6 dB of median-RMS spread the level report already
    # found -- turning a mastering difference into an editing finding.
    ref_db = float(np.median(env))
    on_deficit, on_rise, on_lead = _edge(env, ref_db)
    off_deficit, off_rise, off_lead = _edge(env[::-1], ref_db)
    return {
        "onset_level_deficit_db": on_deficit,
        "onset_rise_s": on_rise,
        "onset_lead_silence_s": on_lead,
        "onset_class": classify(on_deficit, on_rise),
        "offset_level_deficit_db": off_deficit,
        "offset_rise_s": off_rise,
        "offset_lead_silence_s": off_lead,
        "offset_class": classify(off_deficit, off_rise),
        "envelope_ref_dbfs": ref_db,
        "envelope_ok": True,
        "envelope_error": None,
    }
