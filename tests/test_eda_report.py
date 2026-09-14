"""Reading the S tier: `eda.analyze.signal`, docs/EDA/09 step 6.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda.analyze import signal as sg
from eda.analyze.shortcut import METADATA_FEATURES
from eda.planes import CHAIN, NATIVE


def _signal(rows: list[dict]) -> pd.DataFrame:
    """A signal table with both planes, defaulting anything not specified."""
    base = {}
    for group in sg.SIGNAL_SCALARS.values():
        for col in group:
            for plane in (NATIVE, CHAIN):
                base[f"{col}_{plane}"] = 0.0
    out = []
    for i, row in enumerate(rows):
        r = {"file_id": row.pop("file_id", f"s{i}:a.wav"),
             "source_name": row.pop("source_name", "s"),
             "partition": row.pop("partition", "A"),
             "signal_ok": True, **base, **row}
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# the plane a feature spec is allowed to see
# --------------------------------------------------------------------------- #

def test_the_chain_feature_spec_excludes_the_published_rate_in_disguise():
    """🔴 `nyquist_hz_chain` is 8000.0 on every row and `nyquist_hz_native` is
    `orig_sr / 2` -- a metadata column wearing a signal column's name. X1b
    already showed `orig_sr` separates the corpus alone, so an audit that asked
    "what does the chain leak?" while including it would rediscover the
    metadata confound and report it as acoustic evidence.

    Mutation: `if c not in CHAIN_EXCLUDED` dropped from `plane_features`.
    """
    for plane in (NATIVE, CHAIN):
        names = sg.plane_features(plane)["numeric"]
        assert f"nyquist_hz_{plane}" not in names
        assert f"effective_bandwidth_hz_{plane}" in names
        assert all(n.endswith(f"_{plane}") for n in names)


def test_a_feature_spec_names_only_its_own_plane():
    """The two planes are the experiment. A spec that leaked the other one would
    make every 'what does the chain see?' number a native number too."""
    native = set(sg.plane_features(NATIVE)["numeric"])
    chain = set(sg.plane_features(CHAIN)["numeric"])
    assert not (native & chain)
    assert len(native) == len(chain)


def test_the_signal_spec_has_the_shape_build_design_expects():
    """It is handed to `build_design`, which reads three keys. A spec missing
    one raises there, three frames away from here."""
    spec = sg.plane_features(CHAIN)
    assert set(spec) == set(METADATA_FEATURES)


def test_an_unknown_plane_is_refused():
    with pytest.raises(ValueError, match="unknown plane"):
        sg.plane_features("mono")


# --------------------------------------------------------------------------- #
# the paired difference
# --------------------------------------------------------------------------- #

def test_the_paired_difference_is_native_minus_chain():
    signal = _signal([{f"effective_bandwidth_hz_{NATIVE}": 18000.0,
                       f"effective_bandwidth_hz_{CHAIN}": 8000.0}])
    got = sg.paired(signal, "effective_bandwidth_hz")
    assert got["delta"].tolist() == [10000.0]


def test_a_row_missing_either_plane_is_dropped_and_counted():
    """⚠️ Keeping it would put a NaN difference beside a real one and make the
    median quietly a median of whatever survived.

    Mutation: the `dropna` removed -- `delta` is then NaN for that row and the
    median silently changes with it.
    """
    signal = _signal([
        {f"hf_ratio_8k_{NATIVE}": 0.5, f"hf_ratio_8k_{CHAIN}": 0.0},
        {f"hf_ratio_8k_{NATIVE}": np.nan, f"hf_ratio_8k_{CHAIN}": 0.0},
    ])
    got = sg.paired(signal, "hf_ratio_8k")
    assert len(got) == 1
    assert got.attrs["dropped"] == 1


def test_an_unpaired_column_raises_and_names_what_is_available():
    signal = _signal([{}])
    with pytest.raises(KeyError, match="paired columns are"):
        sg.paired(signal, "no_such_column")


# --------------------------------------------------------------------------- #
# the three tables
# --------------------------------------------------------------------------- #

def test_the_duration_report_reads_against_the_test_window():
    """🔴 The competition's test set is 4-60 s. A pool whose files are all 10 s
    is a pool the model never sees in that form, and `in_test_window` is that
    gap in one number.

    Mutation: `TEST_MAX_S` raised to infinity -- `over_60s` goes to 0 and a
    partition that is 80% too long reads as perfectly matched.
    """
    rows = ([{f"duration_s_decoded_{CHAIN}": 2.0}] * 1
            + [{f"duration_s_decoded_{CHAIN}": 10.0}] * 2
            + [{f"duration_s_decoded_{CHAIN}": 120.0}] * 1)
    got = sg.duration_report(_signal(rows)).iloc[0]
    assert got["n"] == 4
    assert got["under_4s"] == 0.25
    assert got["over_60s"] == 0.25
    assert got["in_test_window"] == 0.5


def test_the_bandwidth_report_pairs_each_file_with_itself():
    """R1's design: the difference is a property of the transform, not of two
    populations that differ in content too.

    Mutation: `bandwidth_lost_hz` computed as the difference of the two medians
    instead of the median of the differences.

    ⚠️ The chain column has to vary for this to bite. Measured: with every
    chain value at 8000 the two forms agree exactly -- median(14000, 0, 0) = 0
    and 8000 - 8000 = 0 -- and the mutant survived. Three files that each lose
    a different amount separate them: 5000 against 2000.
    """
    signal = _signal([
        {f"effective_bandwidth_hz_{NATIVE}": 20000.0,
         f"effective_bandwidth_hz_{CHAIN}": 8000.0},
        {f"effective_bandwidth_hz_{NATIVE}": 10000.0,
         f"effective_bandwidth_hz_{CHAIN}": 9000.0},
        {f"effective_bandwidth_hz_{NATIVE}": 9000.0,
         f"effective_bandwidth_hz_{CHAIN}": 4000.0},
    ])
    got = sg.bandwidth_report(signal).iloc[0]
    assert got["bandwidth_lost_hz"] == 5000.0       # median of (12000, 1000, 5000)
    assert got["bandwidth_native_hz"] == 10000.0


def test_the_level_report_is_read_on_the_chain_plane():
    """Level after the chain is the level the model sees, and the only one a
    normalisation decision can act on.

    Mutation: `_{CHAIN}` swapped for `_{NATIVE}` -- every number still computes
    and describes audio the model is never handed.
    """
    signal = _signal([{f"rms_dbfs_{NATIVE}": -6.0, f"rms_dbfs_{CHAIN}": -30.0,
                       f"clipping_ratio_{CHAIN}": 0.1}])
    got = sg.level_report(signal).iloc[0]
    assert got["rms_dbfs_median"] == -30.0
    assert got["frac_clipping"] == 1.0


def test_clipping_counts_files_not_samples():
    """`frac_clipping` is the share of *files* with any clipped sample. A mean
    of the ratios would report 0.001 for a source where every file clips."""
    signal = _signal([{f"clipping_ratio_{CHAIN}": 0.0001},
                      {f"clipping_ratio_{CHAIN}": 0.0}])
    assert sg.level_report(signal).iloc[0]["frac_clipping"] == 0.5
