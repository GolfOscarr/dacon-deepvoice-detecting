"""G-EDA6's work list: `eda.analyze.reassign`, docs/EDA/07.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eda.analyze.reassign import (ACTIONS, MIN_JUDGEABLE_S, TARGET_CELL,
                                  contradiction_rows, reassignment_plan)
from eda.analyze.signal import VOICE_EVIDENCE_RATIO as T
from eda.planes import CHAIN
from training.manifest import POOL_LABELS
from training.spec import CELL_TABLE


def _rows(*specs) -> pd.DataFrame:
    """`(pool, source, speech_ratio, duration_s)` tuples to a signal frame."""
    return pd.DataFrame([
        {"file_id": f"s:{i}.wav", "source_name": src, "pool": pool,
         "vad_ok": True, "vad_speech_ratio_50": ratio,
         f"duration_s_decoded_{CHAIN}": dur}
        for i, (pool, src, ratio, dur) in enumerate(specs)])


# --------------------------------------------------------------------------- #
# the target cells must mean what the reassignment claims
# --------------------------------------------------------------------------- #

def test_every_target_cell_asserts_voice_and_keeps_the_music_label():
    """🔴 The only defect this table could introduce, and it would move 1,006
    files into a cell asserting the opposite of what was intended.

    `_check_targets` runs at import, so the mutation is caught by *importing*
    the module — which is why this test restates the property rather than
    relying on the import alone.

    Mutation: `TARGET_CELL = {"C": 5, "D": 8}` -> `{"C": 3, "D": 4}` (the
    voiceless music cells). Import then raises.
    """
    for pool, cell in TARGET_CELL.items():
        voice_present, music_present, _, music_fake = CELL_TABLE[cell]
        assert voice_present == 1, f"cell {cell} does not assert voice"
        assert music_present == 1
        assert music_fake == POOL_LABELS[pool][3], (
            "reassignment must not change what the music is")


def test_the_target_cell_check_actually_refuses_a_bad_table():
    """🔴 The gap the mutation harness found. The test above restates the
    property against the **real** `TARGET_CELL`, which is correct — so deleting
    the guard inside `_check_targets` changes nothing while the table is right,
    and the mutant survived.

    The guard's job is to catch a *future* bad table, so testing it means
    calling it **with** one. Both arms are checked: a cell that does not assert
    voice, and one that silently changes the music label.
    """
    import eda.analyze.reassign as mod

    original = mod.TARGET_CELL
    try:
        # cell 3 is (0, 1, None, 0) -- music, and no voice at all.
        mod.TARGET_CELL = {"C": 3}
        with pytest.raises(ValueError, match="asserts voice_present=0"):
            mod._check_targets()
        # cell 8 is (1, 1, 1, 1): it asserts voice, but pool C's music is real.
        mod.TARGET_CELL = {"C": 8}
        with pytest.raises(ValueError, match="must not change what the music is"):
            mod._check_targets()
    finally:
        mod.TARGET_CELL = original
    mod._check_targets()          # the shipped table still passes


def test_pool_c_goes_to_the_real_music_cell_and_pool_d_to_the_fake_one():
    rows = contradiction_rows(_rows(("C", "fma", 0.7, 30.0),
                                    ("D", "fakemusiccaps", 0.7, 10.0)))
    by_source = rows.set_index("source_name")
    assert by_source.loc["fma", "target_cell"] == 5
    assert by_source.loc["fakemusiccaps", "target_cell"] == 8
    assert set(rows["action"]) == {"reassign_cell"}


# --------------------------------------------------------------------------- #
# pool E is not a reassignment
# --------------------------------------------------------------------------- #

def test_a_noise_file_with_speech_is_restricted_not_reassigned():
    """🔴 Pool E is an **additive layer**, not a standalone sample. A noise clip
    carrying speech, mixed under a music-only composite, makes that composite
    carry voice while asserting `voice_present = 0` — so one contaminated file
    mislabels *every* sample it is mixed into. And reassigning it to cell 1 is
    no better: a field recording with distant speech is a terrible "voice"
    sample.

    Mutation: pool E given an entry in `TARGET_CELL`. 1,032 noise files then
    enter the corpus as voice samples.
    """
    rows = contradiction_rows(_rows(("E", "musan-noise", 0.6, 10.0)))
    assert rows["action"].iloc[0] == "restrict_noise"
    assert pd.isna(rows["target_cell"].iloc[0]), "there is no cell to move it to"
    assert "E" not in TARGET_CELL


# --------------------------------------------------------------------------- #
# the two halves of type B are not the same finding
# --------------------------------------------------------------------------- #

def test_a_file_under_the_four_second_floor_is_unevidenceable_not_degenerate():
    """🔴 `na` is not `fail`. A VAD given less than four seconds has little to
    work with, so "no speech found" is not evidence that there is none — and 4 s
    is the test set's own floor.

    Mutation: the `d < MIN_JUDGEABLE_S` branch removed — 42 rows become
    generation failures on the strength of a measurement nobody could make.
    """
    rows = contradiction_rows(_rows(("B", "mlaad", 0.0, MIN_JUDGEABLE_S - 0.1)))
    assert rows["action"].iloc[0] == "unevidenceable"


def test_a_real_recording_is_never_a_generation_failure():
    """🔴 F-S4 is **generation**-failure detection, and pool A generates
    nothing. Measured: 3 `cfad-real` rows sit just under the threshold at ratio
    0.192, and the first version of this function filed them as degenerate
    output from a corpus that does not generate.

    Mutation: the `POOL_LABELS[...][_VOICE_FAKE] == 1` test removed — real
    recordings are then reported as broken generations.
    """
    real = contradiction_rows(_rows(("A", "cfad-real", 0.19, 10.0)))
    fake = contradiction_rows(_rows(("B", "cfad-fake", 0.19, 10.0)))
    assert real["action"].iloc[0] == "sparse_real"
    assert fake["action"].iloc[0] == "degenerate"
    assert POOL_LABELS["A"][2] == 0 and POOL_LABELS["B"][2] == 1


# --------------------------------------------------------------------------- #
# what does not enter the list at all
# --------------------------------------------------------------------------- #

def test_a_row_whose_vad_did_not_run_cannot_contradict_anything():
    """⚠️ No measurement exists, so there is nothing to contradict — the same
    `na`-is-not-`pass` distinction the gates use.

    Mutation: the `vad_ok` filter removed; unmeasured rows then arrive with a
    NaN ratio and, if the NaN guard is also removed, land in whichever branch
    the comparison happens to take.
    """
    frame = _rows(("C", "fma", 0.7, 30.0))
    frame["vad_ok"] = False
    assert len(contradiction_rows(frame)) == 0


def test_an_agreeing_row_is_not_in_the_list():
    """The baseline that makes the rest mean something."""
    agree = _rows(("C", "fma", 0.0, 30.0),          # says none, has none
                  ("B", "mlaad", 0.9, 10.0))        # says voice, has voice
    assert len(contradiction_rows(agree)) == 0


def test_the_threshold_boundary_is_inclusive_on_the_evidence_side():
    """Exactly at the threshold counts as evidence: `>=`, not `>`. Stated
    because the two differ on real rows and the choice must be deliberate."""
    assert len(contradiction_rows(_rows(("C", "fma", T, 30.0)))) == 1
    assert len(contradiction_rows(_rows(("C", "fma", T - 1e-9, 30.0)))) == 0


def test_every_action_has_a_declared_reason():
    rows = contradiction_rows(_rows(("C", "fma", 0.7, 30.0),
                                    ("E", "musan-noise", 0.6, 10.0),
                                    ("B", "mlaad", 0.0, 1.0),
                                    ("B", "cfad-fake", 0.0, 10.0),
                                    ("A", "cfad-real", 0.0, 10.0)))
    assert set(rows["action"]) <= set(ACTIONS)
    assert (rows["why"] == rows["action"].map(ACTIONS)).all()
    assert len(rows) == 5


def test_the_plan_accounts_for_every_row():
    rows = contradiction_rows(_rows(("C", "fma", 0.7, 30.0),
                                    ("C", "fma", 0.8, 30.0),
                                    ("E", "musan-noise", 0.6, 10.0)))
    plan = reassignment_plan(rows)
    assert plan["n"].sum() == len(rows)
    assert set(plan["action"]) == {"reassign_cell", "restrict_noise"}


def test_an_empty_work_list_is_still_a_table():
    """The same defect that hit `screens.separability` and the censuses: an
    empty result must keep its declared columns."""
    empty = reassignment_plan(contradiction_rows(_rows(("C", "fma", 0.0, 30.0))))
    assert len(empty) == 0
    assert "action" in empty.columns and "n" in empty.columns
