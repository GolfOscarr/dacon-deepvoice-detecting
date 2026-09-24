"""Step 3: `eda.analyze.roles` and `eda.analyze.census`, docs/EDA/10 §2.3.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.analyze.census import (codec_provenance, format_census,
                                generator_census, mlaad_inventory,
                                near_nyquist_by_generator)
from eda.analyze.roles import (ROLES, UnassignedColumn, column_roles,
                               role_summary)
from eda.planes import CHAIN, NATIVE


def _files(n=6) -> pd.DataFrame:
    return pd.DataFrame({
        "file_id": [f"s:{i}.wav" for i in range(n)],
        "source_name": "fakemusiccaps",
        "path": [f"fakemusiccaps/zenodo/Gen{i % 2}/{i}.wav" for i in range(n)],
        "pool": "D",
        "cell": None,
        "row_kind": "component",
        "group_key": "g",
        "group_key_kind": "publisher",
        "sha256": [f"{i:064x}" for i in range(n)],
        "orig_sr": 16000,
        "orig_channels": 1,
        "duration_s": [10.0] * n,
        "container": "wav",
        "codec_name": "pcm_f32le",
        "sample_fmt": "flt",
        "encoder": None,
        "bit_rate": 512000,
        "file_bytes": 320000,
        "bits_per_raw_sample": None,
        "n_streams": 1,
        "xing_tag": False,
        "lame_tag": False,
        "id3_tag": False,
        "lame_version": None,
        "codec_long_name": "PCM float",
        "format_long_name": "WAV",
        "probe_ok": True,
        "probe_error": None,
        "header_ok": True,
        "identity_ok": True,
    })


# --------------------------------------------------------------------------- #
# X6 -- no column unassigned
# --------------------------------------------------------------------------- #

def test_every_column_is_assigned_and_an_unknown_one_is_fatal():
    """🔴 X6's deliverable is *"every column assigned"*, so a column no rule
    claims must stop the run. A default bucket would silently absorb every
    column added later -- which is the drift this artifact exists to prevent.

    Mutation: `_classify`'s final `return None, ""` replaced by a default role.
    The page is then written with a column nobody classified, and a reader takes
    the silence for an assignment.
    """
    files = _files()
    assert len(column_roles(files)) == len(files.columns)

    files["something_new"] = 1.0
    with pytest.raises(UnassignedColumn, match="something_new"):
        column_roles(files)


def test_the_label_columns_are_the_label_not_a_leakage_risk():
    """🔴 `pool` and `cell` do not *risk* leaking the target, they **are** it --
    `POOL_LABELS` and `CELL_TABLE` derive every head from them. Filing them as
    `leakage_risk` would put the target in the same bucket as `bit_rate` and
    make the page useless at the one thing it is for.

    Mutation: the `label` rule deleted -- `pool` then falls through to the
    metadata check or to `UnassignedColumn`.
    """
    roles = column_roles(_files()).set_index("column")["role"]
    assert roles["pool"] == "label"
    assert roles["row_kind"] == "label"
    assert roles["cell"] == "label"


def test_a_split_key_is_a_split_key_and_not_also_a_feature():
    """⚠️ `source_name` is the archive holdout axis *and* a perfect label
    predictor; those are the same fact. A column used to build folds is never
    handed to the model, so `split_key` is the correct single assignment and
    resolves X6's `exactly one`.

    Mutation: the `split_key` rule moved below the metadata check.
    """
    roles = column_roles(_files()).set_index("column")["role"]
    for column in ("source_name", "group_key", "sha256", "file_id", "path"):
        assert roles[column] == "split_key", column


def test_nothing_in_the_metadata_tier_is_a_feature():
    """🔴 The finding the page exists to carry: X2 measured the M tier at AUC
    1.000 on `music_fake` and 0.0000008 under an archive holdout, so **no**
    metadata column is a legitimate model input.

    Mutation: the `metadata` fallback in `_classify` made to return `feature`.
    """
    roles = column_roles(_files())
    assert "feature" not in set(roles["role"]), (
        "files.parquet must contribute no features at all")
    for column in ("bit_rate", "container", "codec_name", "sample_fmt",
                   "orig_sr", "file_bytes", "encoder"):
        assert roles.set_index("column").loc[column, "role"] == "leakage_risk"


def _signal(n=4) -> pd.DataFrame:
    return pd.DataFrame({
        "partition": "D",
        f"rms_dbfs_{CHAIN}": -20.0,
        f"rms_dbfs_{NATIVE}": -20.0,
        f"duration_s_decoded_{CHAIN}": 10.0,
        f"duration_s_decoded_{NATIVE}": 10.0,
        f"nyquist_hz_{CHAIN}": 8000.0,
        f"nyquist_hz_{NATIVE}": 8000.0,
        f"level_ok_{CHAIN}": True,
        f"level_error_{CHAIN}": None,
        "vad_speech_ratio_50": 0.0,
        "vad_ok": True,
    }, index=range(n))


def test_the_native_plane_is_never_a_feature():
    """🔴 The model never sees it (`eda/planes.py`): the chain plane is what
    reaches it. A statistic that separates at `native` and not at `chain` is a
    property of the archive, which is what 06 X1d measured.

    Mutation: the `_native` rule dropped so the generic `_chain` rule is the
    only plane rule -- native columns then fall through to `UnassignedColumn`,
    or to `feature` if the suffix test is loosened.
    """
    roles = column_roles(_files(4), _signal(4)).set_index("column")["role"]
    assert roles[f"rms_dbfs_{NATIVE}"] == "leakage_risk"
    assert roles[f"rms_dbfs_{CHAIN}"] == "feature"


def test_duration_is_a_leakage_risk_on_both_planes():
    """🔴 The single most consequential row in the table. Duration holds AUC
    0.852 after a whole-publisher holdout, survives the render chain, and in
    pool D the exact millisecond value is a generator id (02 B6b). It carries a
    `_chain` suffix, so without its own rule the generic one makes it a feature.

    Mutation: the `duration_s_decoded` rule removed -- the chain copy silently
    becomes a `feature` and the page recommends the corpus's worst shortcut.
    """
    roles = column_roles(_files(4), _signal(4)).set_index("column")["role"]
    assert roles[f"duration_s_decoded_{CHAIN}"] == "leakage_risk"
    assert roles[f"duration_s_decoded_{NATIVE}"] == "leakage_risk"


def test_nyquist_is_metadata_wearing_an_acoustic_name():
    """⚠️ `nyquist_hz_chain` is 8000.0 on every row and `nyquist_hz_native` is
    the published rate over two. `eda.analyze.signal.CHAIN_EXCLUDED` already
    drops them from the chain audit; the page must agree.

    Mutation: the `CHAIN_EXCLUDED` rule moved below the generic `_chain` rule.
    """
    roles = column_roles(_files(4), _signal(4)).set_index("column")["role"]
    assert roles[f"nyquist_hz_{CHAIN}"] == "leakage_risk"
    assert roles[f"nyquist_hz_{NATIVE}"] == "leakage_risk"


def test_ok_and_error_columns_are_diagnostics_on_either_plane():
    roles = column_roles(_files(4), _signal(4)).set_index("column")["role"]
    assert roles[f"level_ok_{CHAIN}"] == "diagnostic"
    assert roles[f"level_error_{CHAIN}"] == "diagnostic"
    assert roles["probe_ok"] == "diagnostic"
    assert roles["vad_ok"] == "diagnostic"
    # ...but the VAD's measurements are features, not diagnostics.
    assert roles["vad_speech_ratio_50"] == "feature"


def test_the_summary_accounts_for_every_column():
    roles = column_roles(_files(4), _signal(4))
    summary = role_summary(roles)
    assert summary["columns"].sum() == len(roles)
    assert set(summary["role"]) <= set(ROLES)


# --------------------------------------------------------------------------- #
# the six censuses
# --------------------------------------------------------------------------- #

def test_a_constant_duration_generator_is_found_by_min_max_not_by_std():
    """🔴 Measured, and it changed a published number. `mustango` and
    `MusicGen_medium` are each a single repeated value (10.242 and 10.180) whose
    `std` comes back as `1.776357e-15`, not `0.0` -- so a `std == 0` test finds
    3 of FakeMusicCaps' 5 constant generators and the write-up says "3 of 231"
    when the answer is 5.

    Mutation: `duration_constant` computed as `std == 0`.
    """
    # ⚠️ 100 rows, not 4, and 10.242 rather than a round number. Measured: at
    # n=4 pandas returns an exact 0.0 and the `std == 0` mutant survives -- the
    # fixture has to be large enough to accumulate the float error that made
    # this a defect in the first place.
    n = 100
    constant = _files(1).loc[[0] * n].reset_index(drop=True)
    constant["path"] = [f"fakemusiccaps/zenodo/Gen0/{i}.wav" for i in range(n)]
    constant["duration_s"] = 10.242
    varying = _files(1).loc[[0] * n].reset_index(drop=True)
    varying["path"] = [f"fakemusiccaps/zenodo/Gen1/{i}.wav" for i in range(n)]
    varying["duration_s"] = [10.0] * (n - 1) + [11.0]

    out = generator_census(pd.concat([constant, varying], ignore_index=True)
                           ).set_index("generator")
    assert out.loc["fakemusiccaps/Gen0", "duration_constant"]
    assert out.loc["fakemusiccaps/Gen0", "duration_s_std"] != 0.0, (
        "the fixture must reproduce the float error, or the `std == 0` mutant "
        "survives and this test guards nothing")
    assert not out.loc["fakemusiccaps/Gen1", "duration_constant"]


def test_the_format_census_reports_encoder_missingness_not_only_its_value():
    """🔴 `encoder` is null for every wav and present for every mp3, so
    *whether it is there* is the archive fingerprint -- which is what
    `build_design`'s `_isna` indicators encode and what X2 found inverts.

    Mutation: `encoder_present` dropped, leaving only `n_encoders` -- a source
    with no encoder and a source whose encoder is uniform both read as a small
    number.
    """
    files = _files(4)
    out = format_census(files)
    assert out["encoder_present"].iloc[0] == 0.0
    files.loc[:1, "encoder"] = "LAME"
    assert format_census(files)["encoder_present"].iloc[0] == 0.5


def test_the_codec_census_splits_a_source_that_mixes_containers():
    """A source that mixes containers must be visible as two rows rather than
    averaged into one -- the mp3 question is per file, not per archive.

    Mutation: the groupby keyed on `source_name` alone.
    """
    files = _files(4)
    files.loc[:1, "container"] = "mp3"
    out = codec_provenance(files)
    assert len(out) == 2
    assert set(out["container"]) == {"wav", "mp3"}
    assert out.set_index("container").loc["mp3", "lossy"]
    assert not out.set_index("container").loc["wav", "lossy"]


def test_the_mlaad_inventory_is_per_language_and_generator():
    files = _files(4)
    files["source_name"] = "mlaad"
    files["path"] = ["mlaad/v9/payload/fake/ko/VoxCPM2/a.wav",
                     "mlaad/v9/payload/fake/ko/Fish-S2-Pro/b.wav",
                     "mlaad/v9/payload/fake/en/VoxCPM2/c.wav",
                     "mlaad/v9/payload/fake/en/VoxCPM2/d.wav"]
    out = mlaad_inventory(files)
    assert len(out) == 3, out
    assert set(out["language"]) == {"ko", "en"}
    assert out.set_index(["language", "generator"]).loc[("en", "VoxCPM2"), "n"] == 2


def test_near_nyquist_is_reported_on_both_planes_as_a_pair():
    """⚠️ D3's point is R1's: a generator distinctive at `native` and not at
    `chain` has a fingerprint the resample destroys, and one distinctive at both
    has one that survives. Reporting either column alone answers neither.

    Mutation: the `for plane in (NATIVE, CHAIN)` loop narrowed to one plane.
    """
    files = _files(4)
    signal = _signal(4)
    signal["source_name"] = "fakemusiccaps"
    signal["path"] = files["path"]
    for plane in (NATIVE, CHAIN):
        signal[f"near_nyquist_ratio_{plane}"] = 0.1
        signal[f"effective_bandwidth_hz_{plane}"] = 8000.0
        signal[f"hf_ratio_8k_{plane}"] = 0.0
    out = near_nyquist_by_generator(signal)
    # ⚠️ Presence is not enough, and asserting only presence let the mutant
    # survive. `NEAR_NYQUIST_COLUMNS` is handed to the DataFrame constructor so
    # an empty result keeps its shape -- which means a plane that was never
    # computed still has its columns, filled with NaN. Measured: narrowing the
    # loop to `(CHAIN,)` passed a presence-only test.
    for plane in (NATIVE, CHAIN):
        for stat in ("near_nyquist_ratio", "effective_bandwidth_hz", "hf_ratio_8k"):
            column = f"{stat}_{plane}"
            assert column in out.columns, column
            assert out[column].notna().all(), (
                f"{column} exists but was never computed")


def test_a_source_with_no_generator_is_absent_from_the_generator_census():
    """Not zero-filled: a source with no registered generator depth has no
    per-generator answer, and inventing one would put an archive name in the
    `generator` column."""
    files = _files(4)
    files["source_name"] = "ljspeech"
    files["path"] = [f"ljspeech/mdc-1.1/LJSpeech-1.1/wavs/{i}.wav" for i in range(4)]
    assert len(generator_census(files)) == 0
