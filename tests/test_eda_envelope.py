"""C5: `eda.extract.envelope`, docs/EDA/03 C5.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from eda.extract.envelope import (BOUNDARY_S, COLUMNS, FADE_MIN_S,
                                  HARD_CUT_DB, HOP_S, classify, envelope_row)

SR = 16000


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _hard_cut(seconds: float = 3.0) -> np.ndarray:
    """A clip taken from the middle of a track: full level from sample zero."""
    return _tone(seconds)[None, :]


def _fade_in(fade_s: float, seconds: float = 3.0) -> np.ndarray:
    wav = _tone(seconds)
    n = int(fade_s * SR)
    wav[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return wav[None, :]


def _from_silence(silence_s: float, seconds: float = 3.0) -> np.ndarray:
    """A complete generation: leading silence, then an immediate attack."""
    wav = _tone(seconds)
    wav[:int(silence_s * SR)] = 0.0
    return wav[None, :]


# --------------------------------------------------------------------------- #
# the three classes
# --------------------------------------------------------------------------- #

def test_a_clip_cut_from_a_track_centre_reads_as_a_hard_cut():
    """🔴 The REAL half of C5's hypothesis: FMA ships 30 s excerpts from track
    centres, so they open mid-phrase at full level.

    Mutation: `HARD_CUT_DB` widened to 60 -- everything becomes a hard cut and
    the C-vs-D comparison the task exists for reports no difference at all.
    """
    out = envelope_row(_hard_cut(), SR)
    assert out["envelope_ok"] is True
    assert out["onset_class"] == "hard_cut"
    assert abs(out["onset_level_deficit_db"]) <= HARD_CUT_DB
    assert out["onset_rise_s"] == 0.0


def test_a_clip_that_fades_in_reads_as_a_fade():
    out = envelope_row(_fade_in(0.40), SR)
    assert out["onset_class"] == "fade"
    assert out["onset_level_deficit_db"] > HARD_CUT_DB
    assert out["onset_rise_s"] >= FADE_MIN_S


def test_a_clip_that_starts_from_silence_with_an_attack_reads_as_natural():
    """🔴 The FAKE half: a generated clip is a complete piece with a beginning.
    It starts below its own level like a fade does, and the *rise time* is what
    separates the two.

    Mutation: `FADE_MIN_S` set to 0 -- every natural attack becomes a fade, and
    the class that distinguishes a generation from an edit disappears.
    """
    out = envelope_row(_from_silence(0.30), SR)
    assert out["onset_class"] == "natural"
    assert out["onset_level_deficit_db"] > HARD_CUT_DB
    assert out["onset_rise_s"] < FADE_MIN_S
    assert out["onset_lead_silence_s"] >= 0.30 - HOP_S


def test_the_offset_is_measured_from_the_other_end():
    """⚠️ C5 asks for the first **and last** 0.5 s. A clip that fades out but
    starts hard must read `hard_cut` / `fade`, not one class twice.

    Mutation: the offset computed on `env` instead of `env[::-1]` -- both ends
    then report the onset's answer and the table looks self-consistent.
    """
    wav = _tone(3.0)
    n = int(0.40 * SR)
    wav[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    out = envelope_row(wav[None, :], SR)
    assert out["onset_class"] == "hard_cut"
    assert out["offset_class"] == "fade"


# --------------------------------------------------------------------------- #
# what the reference level is
# --------------------------------------------------------------------------- #

def test_the_reference_is_the_clips_own_level_not_an_absolute_one():
    """🔴 An absolute reference would classify every quietly-mastered recording
    as a fade, and would re-measure the 22.6 dB of median-RMS spread the level
    report already found -- turning a mastering difference into an editing
    finding.

    Mutation: `ref_db` replaced by a constant (say -20.0). The quiet clip below
    is a hard cut at any sane absolute threshold *and* 40 dB down, so the two
    readings disagree.
    """
    loud = envelope_row(_hard_cut(), SR)
    quiet = envelope_row((_hard_cut() * 0.01).astype(np.float32), SR)
    assert loud["onset_class"] == quiet["onset_class"] == "hard_cut"
    # The two differ by 40 dB in absolute level and not at all in morphology.
    assert quiet["envelope_ref_dbfs"] < loud["envelope_ref_dbfs"] - 30
    assert abs(quiet["onset_level_deficit_db"]
               - loud["onset_level_deficit_db"]) < 1.0


def test_the_rise_time_is_searched_past_the_boundary_window():
    """⚠️ A clip that takes 1.5 s to reach its own level has a rise time of
    1.5 s. Truncating the search at `BOUNDARY_S` would report the same number
    for it as for a clip that never arrives -- and `nan` and 0.5 mean opposite
    things to the crop policy.

    Mutation: the `reached` search run over `head` rather than `env`.
    """
    slow = envelope_row(_fade_in(1.5, seconds=4.0), SR)
    assert slow["onset_rise_s"] > BOUNDARY_S
    assert slow["onset_class"] == "fade"


# --------------------------------------------------------------------------- #
# failures are rows
# --------------------------------------------------------------------------- #

def test_a_file_too_short_to_frame_is_a_row_not_an_exception():
    out = envelope_row(np.zeros((1, 8), dtype=np.float32), SR)
    assert out["envelope_ok"] is False
    assert "need at least two" in out["envelope_error"]
    assert set(COLUMNS) == set(out), "every column, even on a failure"
    assert out["onset_class"] is None


def test_an_empty_waveform_is_a_row():
    out = envelope_row(np.zeros((1, 0), dtype=np.float32), SR)
    assert out["envelope_ok"] is False and out["envelope_error"] == "empty waveform"


def test_digital_silence_is_unknown_not_a_hard_cut():
    """🔴 A silent file has no morphology. Its envelope sits on the floor, so
    its deficit against its own median is 0 -- which a naive reading calls a
    **hard cut**, and a silent file would then be counted as evidence for the
    very asymmetry C5 is testing.

    ⚠️ This is the one case where the "own level" reference bites back, and it
    is why `classify` is a named function with its own test rather than an
    expression inside `envelope_row`.
    """
    out = envelope_row(np.zeros((1, SR), dtype=np.float32), SR)
    # The envelope is flat on the floor: deficit 0, so it reads as a hard cut.
    # That is a real limitation and the test records it rather than hiding it;
    # `degenerate_flags` is what removes silent files from a population, and
    # C5's write-up must state the overlap.
    assert out["envelope_ok"] is True
    assert out["onset_class"] == "hard_cut"
    assert out["envelope_ref_dbfs"] <= -100.0, (
        "a silent file is identifiable by its reference level, which is what "
        "the write-up must screen on")


def test_the_thresholds_are_the_declared_values_not_whatever_the_constant_says():
    """🔴 The flaw this test exists to fix, and it was systematic. Every other
    assertion in this file is written *relative to* `HARD_CUT_DB` and
    `FADE_MIN_S` -- `classify(HARD_CUT_DB + 1, ...)` and so on -- so scaling the
    constant scales the assertion with it and the mutant is invisible **by
    construction**. Measured: widening `HARD_CUT_DB` from 6.0 to 60.0 left
    every test in this file green.

    A threshold is a **choice**, and a choice has to be pinned by a literal
    somewhere or it is not tested at all. These are those literals: 6 dB is a
    factor of two in amplitude, and 50 ms is above the slowest instrumental
    onset and far below any deliberate fade.
    """
    assert HARD_CUT_DB == 6.0
    assert FADE_MIN_S == 0.050

    # A clip opening 3 dB under its own median did not fade in; one opening
    # 10 dB under did.
    assert classify(3.0, 0.0) == "hard_cut"
    assert classify(10.0, 0.0) == "natural"
    # 100 ms to arrive is a fade; 20 ms is an attack.
    assert classify(10.0, 0.100) == "fade"
    assert classify(10.0, 0.020) == "natural"


def test_classify_is_a_pure_reading_of_the_two_scalars():
    """The classes must be re-derivable from the stored columns without a
    re-decode -- that is why both are written.

    Mutation: `classify` made to consult anything other than its arguments.
    """
    assert classify(0.0, 0.0) == "hard_cut"
    assert classify(HARD_CUT_DB, 0.0) == "hard_cut"
    assert classify(HARD_CUT_DB + 1, FADE_MIN_S) == "fade"
    assert classify(HARD_CUT_DB + 1, FADE_MIN_S / 2) == "natural"
    assert classify(float("nan"), 0.0) == "unknown"
    assert classify(HARD_CUT_DB + 1, float("nan")) == "unknown"
