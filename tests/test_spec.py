"""The spec layer: labels come from the cell, and from nowhere else."""

import dataclasses

import pytest

from metrics.dacon import file_fake_label
from training.spec import (CELL_TABLE, ComponentDraw, SampleSpec, cell_labels,
                           cells_in_stratum, is_fake_cell, spec_rng, stratum_of)


def _draw(**over):
    base = dict(file_id="A00001", role="voice", source_offset_s=3.0,
                duration_s=10.0, target_start_s=0.0, gain_db=-6.0)
    return ComponentDraw(**{**base, **over})


def _spec(**over):
    base = dict(sample_id=7, epoch=0, seed=0, scheme_version="v1",
                duration_s=10.0, cell=1, render_mode="composed",
                structure="overlap", components=(_draw(),))
    return SampleSpec(**{**base, **over})


# --------------------------------------------------------------------------- #
# labels

def test_every_cell_matches_the_taxonomy():
    """The nine rows of docs/data/02, asserted rather than trusted."""
    expected = {
        1: (1, 0, 0, None, 0), 2: (1, 0, 1, None, 1),
        3: (0, 1, None, 0, 0), 4: (0, 1, None, 1, 1),
        5: (1, 1, 0, 0, 0), 6: (1, 1, 0, 1, 1),
        7: (1, 1, 1, 0, 1), 8: (1, 1, 1, 1, 1),
        9: (0, 0, None, None, 0),
    }
    for cell, (vp, mp, vf, mf, ff) in expected.items():
        got = cell_labels(cell)
        assert (got["voice_present"], got["music_present"], got["voice_fake"],
                got["music_fake"], got["file_fake"]) == (vp, mp, vf, mf, ff), cell


def test_file_fake_is_computed_not_written(monkeypatch):
    """🔴 One implementation of "OR over present components" in this repo.

    ⚠️ The earlier version asserted
    ``cell_labels(cell)["file_fake"] == file_fake_label(vp, mp, vf or 0, mf or 0)``
    -- which is character-for-character the expression in `spec.py`. A tautology:
    it stays green if `metrics.dacon.file_fake_label` changes meaning, and it
    stays green if `spec.py` stops calling it, because both sides move together.

    The claim in the name is about DELEGATION, so test delegation: replace the
    metric with a sentinel and require `cell_labels` to follow it. A hard-coded
    file_fake column in `CELL_TABLE`, or a second local OR, fails here. The
    *values* are pinned independently against the docs/data/02 table in
    `test_every_cell_matches_the_taxonomy`.
    """
    import training.spec

    calls = []

    def sentinel(vp, mp, vf, mf):
        calls.append((vp, mp, vf, mf))
        return 1 - int(file_fake_label(vp, mp, vf, mf))       # inverted

    monkeypatch.setattr(training.spec, "file_fake_label", sentinel)
    for cell, (vp, mp, vf, mf) in CELL_TABLE.items():
        assert cell_labels(cell)["file_fake"] == 1 - int(
            file_fake_label(vp, mp, vf or 0, mf or 0)), cell
    assert len(calls) == len(CELL_TABLE), "every cell must go through the metric"
    # ⚠️ Absent components arrive as 0, not None -- `file_fake_label` survives
    # None only incidentally (spec.py's own docstring), so pin the call itself.
    assert all(v in (0, 1) for call in calls for v in call), calls


def test_absent_components_never_reach_file_fake():
    """Cells 1-4 are well defined only because the absent side is ignored."""
    assert cell_labels(1)["file_fake"] == 0      # voice real, music absent
    assert cell_labels(4)["file_fake"] == 1      # music fake, voice absent
    assert cell_labels(9)["file_fake"] == 0      # nothing present -> cannot be fake


def test_fake_and_real_cells_partition():
    fake = {c for c in CELL_TABLE if is_fake_cell(c)}
    assert fake == {2, 4, 6, 7, 8}
    assert set(CELL_TABLE) - fake == {1, 3, 5, 9}


def test_strata_partition_the_cells():
    assert cells_in_stratum("voice-only") == (1, 2)
    assert cells_in_stratum("music-only") == (3, 4)
    assert cells_in_stratum("mixed") == (5, 6, 7, 8)
    assert cells_in_stratum("neither") == (9,)
    assert sum(len(cells_in_stratum(s)) for s in
               ("voice-only", "music-only", "mixed", "neither")) == 9


def test_labels_are_not_stored_on_the_spec():
    """A spec carries `cell`; the five labels are derived from it.

    If they were fields, a transform could write to them, and the invariant that
    labels come from steps 1-2 only would be a convention rather than a fact.
    """
    fields = {f.name for f in dataclasses.fields(SampleSpec)}
    assert not fields & {"voice_present", "music_present", "voice_fake",
                         "music_fake", "file_fake"}
    s = _spec(cell=6, components=(_draw(role="voice"), _draw(role="music")))
    assert (s.voice_present, s.music_present, s.file_fake) == (1, 1, 1)


# --------------------------------------------------------------------------- #
# RNG discipline

def test_rng_is_reproducible_across_processes():
    """🔴 Python's hash() is salted per process; blake2b is not.

    A hash()-keyed stream reproduces within a run and differs across runs --
    the exact opposite of what A-S2 asks for.
    """
    import subprocess
    import sys
    got = subprocess.run(
        [sys.executable, "-c",
         "from training.spec import spec_rng; print(spec_rng(11, 2, 5).random())"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert float(got) == spec_rng(11, 2, 5).random()


@pytest.mark.parametrize("a,b", [((1, 0, 0), (2, 0, 0)),
                                 ((1, 0, 0), (1, 1, 0)),
                                 ((1, 0, 0), (1, 0, 1))])
def test_rng_differs_in_every_key_component(a, b):
    assert spec_rng(*a).random() != spec_rng(*b).random()


# --------------------------------------------------------------------------- #
# validation

def test_cells_6_and_7_cannot_be_whole_files():
    """They hold one real and one fake component, so they cannot be scraped."""
    for cell in (6, 7):
        with pytest.raises(ValueError, match="only be composed"):
            _spec(cell=cell, render_mode="whole_file")


def test_a_component_may_not_run_past_the_timeline():
    with pytest.raises(ValueError, match="past the"):
        _spec(duration_s=5.0, components=(_draw(duration_s=10.0),))


def test_whole_file_specs_hold_exactly_one_row():
    with pytest.raises(ValueError, match="exactly one row"):
        _spec(render_mode="whole_file", components=(_draw(), _draw()))


@pytest.mark.parametrize("bad", [{"cell": 0}, {"cell": 10},
                                 {"render_mode": "blend"}, {"structure": "stack"},
                                 {"duration_s": 0.0}, {"components": ()}])
def test_malformed_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        _spec(**bad)


def test_round_trips_through_a_dict():
    s = _spec(cell=7, components=(_draw(role="voice"), _draw(role="music")),
              transforms=(("gain_jitter", {"db": 2.1}),),
              normalize={"container": "mp3", "bitrate": 96})
    assert SampleSpec.from_dict(s.to_dict()) == s
