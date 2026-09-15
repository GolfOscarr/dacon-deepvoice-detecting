"""Step 2's four screens: `eda.analyze.screens`, docs/EDA/10 §2.2.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.analyze.screens import (CLIPPED_RATIO, GENERATOR_DEPTH,
                                 LOOPING_NOT_COMPUTED, SCREENS,
                                 SILENT_PEAK_DBFS, cell9_viability,
                                 degenerate_flags, degenerate_screen,
                                 generator_key, separability)
from eda.extract.vectors import N_MELS
from eda.planes import CHAIN


def _signal(n=20, partition="A", **overrides) -> pd.DataFrame:
    """A benign signal table: nothing in it should trip any screen."""
    frame = pd.DataFrame({
        "file_id": [f"s:{i}.wav" for i in range(n)],
        "source_name": "s",
        "partition": partition,
        "group_key": "s/g",
        "path": [f"s/archive/gen/{i}.wav" for i in range(n)],
        f"peak_dbfs_{CHAIN}": -3.0,
        f"rms_dbfs_{CHAIN}": -20.0,
        f"mel_bands_flat_{CHAIN}": 0,
        f"clipping_ratio_{CHAIN}": 0.0,
        f"duration_s_decoded_{CHAIN}": 10.0,
        f"silence_ratio_{CHAIN}": 0.1,
    })
    for column, value in overrides.items():
        frame[column] = value
    return frame


# --------------------------------------------------------------------------- #
# B4 / D6 -- the screen reports rates, and the rates are the finding
# --------------------------------------------------------------------------- #

def test_a_benign_table_trips_nothing():
    """The baseline that makes every other assertion here mean something: if the
    neutral fixture flagged, a screen firing would prove nothing."""
    flags = degenerate_flags(_signal())
    assert not flags["any_flag"].any(), flags.sum().to_dict()


@pytest.mark.parametrize("column,value,screen", [
    (f"peak_dbfs_{CHAIN}", -80.0, "digital_silence"),
    (f"mel_bands_flat_{CHAIN}", N_MELS, "all_bands_flat"),
    (f"clipping_ratio_{CHAIN}", CLIPPED_RATIO * 2, "clipped"),
    (f"duration_s_decoded_{CHAIN}", 0.1, "too_short"),
    (f"silence_ratio_{CHAIN}", 1.0, "all_silence"),
])
def test_each_screen_fires_on_its_own_condition_and_no_other(column, value, screen):
    """🔴 Each screen must fire on exactly its own condition. A screen that also
    fires on a neighbour's makes `any_flag` right and the per-screen rates --
    which are what the asymmetry comparison reads -- wrong.

    Mutation: any one threshold comparison inverted or widened.
    """
    flags = degenerate_flags(_signal(**{column: value}))
    assert flags[screen].all(), f"{screen} did not fire on its own condition"
    others = [s for s in SCREENS if s != screen]
    assert not flags[others].to_numpy().any(), (
        f"{screen}'s condition also tripped {[s for s in others if flags[s].any()]}")


def test_a_duration_outlier_is_relative_to_its_own_group_not_to_the_corpus():
    """🔴 Pool D is uniformly 10.000 s and pool C is 30.003 s. An absolute
    window would flag one of them wholesale and report a publisher convention as
    a generation failure.

    Mutation: the `groupby(group_key).transform("median")` replaced by a global
    median -- the short group then flags entirely against the long one.
    """
    # ⚠️ The two groups must be far enough apart that the **corpus** median
    # flags one of them. Measured: at 30 s against 10 s the global median is
    # 20 s, both ratios sit inside DURATION_OUTLIER, and the mutant survived --
    # the test proved only that the fixture was benign.
    long_group = _signal(n=10)
    long_group["group_key"] = "s/long"
    long_group[f"duration_s_decoded_{CHAIN}"] = 100.0
    short_group = _signal(n=10)
    short_group[f"duration_s_decoded_{CHAIN}"] = 1.0
    both = pd.concat([long_group, short_group], ignore_index=True)

    flags = degenerate_flags(both)
    assert not flags["duration_outlier"].any(), (
        "two internally consistent groups of different lengths are not outliers")

    # One file 10x its own group's median is.
    both.loc[0, f"duration_s_decoded_{CHAIN}"] = 3000.0
    assert degenerate_flags(both)["duration_outlier"].iloc[0]


def test_the_screen_reports_rates_per_partition_not_a_drop_list():
    """🔴 The rule the whole module turns on (docs/EDA/04 D6): a threshold that
    flags pool C at a materially different rate than pool D is an asymmetric
    filter and is manufacturing a cue. The comparison is between **rows of one
    table**, so the screen must return per-partition rates.

    Mutation: `degenerate_screen` made to return the flagged rows instead. The
    caller then has files to drop and no way to see that the two pools were
    flagged at different rates.
    """
    clean = _signal(n=100, partition="D")
    dirty = _signal(n=100, partition="C", **{f"clipping_ratio_{CHAIN}": 1.0})
    out = degenerate_screen(pd.concat([clean, dirty], ignore_index=True))
    assert set(out["partition"]) == {"C", "D"}
    assert list(out.columns)[:2] == ["partition", "n"]
    by = out.set_index("partition")
    assert by.loc["C", "clipped"] == 1.0 and by.loc["D", "clipped"] == 0.0, (
        "the asymmetry must be visible as two rates, not hidden in a row count")


def test_the_uncomputed_half_of_the_screen_travels_with_the_result():
    """⚠️ B4 and D6 both ask for a looping/babble detector that needs a decode.
    Five screens where six were asked for must not read as six.

    Mutation: the `not_computed` attr dropped -- the table then looks complete.
    """
    out = degenerate_screen(_signal())
    assert "decode" in out.attrs["not_computed"]
    assert out.attrs["not_computed"] == LOOPING_NOT_COMPUTED


def test_a_missing_plane_column_is_a_refusal_not_a_silent_pass():
    """The guard exists for its **message**, and the test says so rather than
    claiming more.

    ⚠️ pandas already refuses a missing column, so `signal[full]` raises either
    way and a mutant that deletes the guard is invisible to a test that only
    asserts `KeyError`. Measured: mutating the guard to `if False:` survived,
    and so did mutating the `return` line -- which the guard shadows entirely.

    What the guard actually adds is **which plane, and why**, on a table that
    carries 80-odd suffixed columns and where the natural mistake is asking for
    the native plane of a chain-only column. So that is what is asserted, and
    the mutation that breaks it is deleting the guard.
    """
    frame = _signal().drop(columns=[f"clipping_ratio_{CHAIN}"])
    with pytest.raises(KeyError, match="cannot be.*run on the 'chain' plane"):
        degenerate_flags(frame)


# --------------------------------------------------------------------------- #
# the generator axis
# --------------------------------------------------------------------------- #

def test_a_depth_that_lands_on_a_shared_parent_is_refused():
    """🔴 The defect that actually happened, three times out of four while
    `GENERATOR_DEPTH` was being written: a depth one component short returns the
    CFAD split (`fake_clean`), the MLAAD **language** (`bg` -- 54 of them, which
    looks exactly like a plausible generator count) or WaveFake's archive
    directory. Each produced a confident, plausible, wrong table.

    Mutation: the `distinct <= 1` guard removed.
    """
    frame = pd.DataFrame({
        "source_name": "fakemusiccaps",
        # The generator component is constant; only the clip varies.
        "path": [f"fakemusiccaps/zenodo/OneGen/{i}.wav" for i in range(10)],
    })
    with pytest.raises(ValueError, match="shared parent directory"):
        generator_key(frame)


def test_the_generator_is_the_declared_component_per_source():
    frame = pd.DataFrame({
        "source_name": ["fakemusiccaps", "wavefake", "mlaad", "cfad-fake"],
        "path": [
            "fakemusiccaps/zenodo-15063698/MusicGen_medium/clip.wav",
            "wavefake/zenodo-5642694/generated_audio/ljspeech_melgan/x_gen.wav",
            "mlaad/v9/payload/fake/am/Edge-TTS/x.wav",
            "cfad/zenodo/CFAD/clean_version/dev_clean/fake_clean/gl/x.wav",
        ],
    })
    # One row per source, so the constant-value guard cannot apply.
    keys = generator_key(frame)
    assert list(keys) == ["fakemusiccaps/MusicGen_medium",
                          "wavefake/ljspeech_melgan",
                          "mlaad/Edge-TTS",
                          "cfad/gl"], list(keys)


def test_a_source_with_no_declared_depth_gets_na_not_a_guess():
    """⚠️ Inventing a component for an unregistered source would put an archive
    name or a clip id in a column labelled `generator`.

    Mutation: the loop given an `else` branch that falls back to a fixed depth.
    """
    frame = pd.DataFrame({"source_name": ["ljspeech"],
                          "path": ["ljspeech/mdc-1.1/LJSpeech-1.1/wavs/x.wav"]})
    assert generator_key(frame).isna().all()
    assert "ljspeech" not in GENERATOR_DEPTH


# --------------------------------------------------------------------------- #
# B6 -- separability
# --------------------------------------------------------------------------- #

def test_an_unmeasurable_axis_returns_a_table_not_an_exception():
    """🔴 The defect this test was written after. Keyed on `group_key`, pool D
    has 5,521 groups averaging 5 rows and **no** group clears `min_rows` -- the
    honest answer is "nothing here is measurable on this axis", and it has to
    arrive as an empty table with the declared columns.

    Measured: returning a bare `DataFrame([])` made `sort_values` raise
    `KeyError: 'best_auc'`, which reads as a code fault rather than as the
    finding that the axis is the wrong grain.

    Mutation: the explicit `columns=` list dropped from the constructor.
    """
    frame = _signal(n=10, partition="D")
    frame["group_key"] = [f"clip/{i}" for i in range(10)]
    frame[f"rms_dbfs_{CHAIN}"] = np.linspace(-30, -10, 10)
    out = separability(frame, "D", min_rows=50)
    assert len(out) == 0
    assert list(out.columns) == ["partition", "axis", "key", "n", "best_auc",
                                "top_feature", "median_auc"]
    assert out.attrs["n_groups"] == 10 and out.attrs["n_measurable"] == 0


def test_separability_names_the_column_that_carries_the_identity():
    """Univariate by design: the point is to name the feature a render-time
    transform would have to neutralise.

    Mutation: `max(scores, key=scores.get)` -> `min(...)` -- the table then
    reports the *least* informative column as the carrier.
    """
    n = 120
    frame = _signal(n=n, partition="B")
    frame["group_key"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    # `dc_offset` separates the two groups perfectly; nothing else moves.
    frame[f"dc_offset_{CHAIN}"] = [0.0] * (n // 2) + [1.0] * (n // 2)
    frame[f"rms_dbfs_{CHAIN}"] = np.tile([-20.0, -21.0], n // 2)
    out = separability(frame, "B", min_rows=10)
    assert len(out) == 2
    assert (out["best_auc"] > 0.99).all()
    assert set(out["top_feature"]) == {f"dc_offset_{CHAIN}"}


# --------------------------------------------------------------------------- #
# E5 -- cell 9
# --------------------------------------------------------------------------- #

def _pool_e(n=10, **overrides):
    frame = _signal(n=n, partition="E")
    frame["vad_ok"] = True
    frame["vad_speech_ratio_50"] = 0.0
    frame[f"duration_s_decoded_{CHAIN}"] = 10.0
    for column, value in overrides.items():
        frame[column] = value
    return frame


def test_any_speech_evidence_disqualifies_a_cell_nine_file():
    """🔴 #417333 A3 confirms `PRESENT = 1` at **any** duration, so a tolerance
    here would be us choosing a level of audibility the rules say does not
    exist.

    Mutation: `speech_ratio` defaulted to something above 0 -- a file with a
    single speech chunk then counts as viable cell-9 material.
    """
    frame = _pool_e(n=10)
    frame.loc[0, "vad_speech_ratio_50"] = 0.001
    out = cell9_viability(frame)
    assert int(out["n_viable"].sum()) == 9
    assert int(out["n_voice_evidenced"].sum()) == 1


def test_a_file_whose_vad_did_not_run_is_not_evidence_of_absent_speech():
    """🔴 `na` is not `pass` -- the same tri-state the gates use. An unmeasured
    file counted as viable would assert `VOICE_PRESENT = 0` on the strength of a
    measurement nobody made.

    Mutation: `silent = ok & (ratio <= speech_ratio)` -> `ratio <= speech_ratio`
    alone; the unmeasured row's NaN compares false, so it would need the `ok`
    term removed *and* the NaN filled -- do both and the count goes to 10.
    """
    frame = _pool_e(n=10)
    frame.loc[0, "vad_ok"] = False
    frame.loc[0, "vad_speech_ratio_50"] = np.nan
    out = cell9_viability(frame)
    assert int(out["n_unmeasured"].sum()) == 1
    assert int(out["n_viable"].sum()) == 9


def test_a_file_under_the_sampler_floor_is_not_viable():
    frame = _pool_e(n=10)
    frame.loc[0, f"duration_s_decoded_{CHAIN}"] = 2.0
    out = cell9_viability(frame, min_duration_s=4.0)
    assert int(out["n_ge_min_duration"].sum()) == 9
    assert int(out["n_viable"].sum()) == 9


def test_the_one_sided_nature_of_the_screen_travels_with_the_result():
    """⚠️ Silero is a speech VAD and cannot evidence music, so `music_present`
    goes unchecked and every count is an upper bound. RESULTS §5.1 records the
    misreading that followed from forgetting this once already.

    Mutation: the `upper_bound` attr dropped.
    """
    out = cell9_viability(_pool_e())
    assert "music" in out.attrs["upper_bound"]
