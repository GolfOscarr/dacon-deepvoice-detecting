"""The publisher's own grouping key: `eda.groupkeys`, docs/EDA/09 step 1.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from eda import groupkeys as gk
from eda.analyze import grouping as grp
from eda.config import EdaConfig, GateConfig, ProbeConfig, SourceSpec, load_eda_config
from eda.driver import consolidate, probe_source
from eda.gates import FAIL, PASS, _g_eda3

SR = 16000
SHIPPED = Path("configs/eda.yaml")


def _write(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, rng.standard_normal(SR // 4) * 0.1, SR, subtype="PCM_16")
    return path


def _census(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """A minimal census frame: `(source_name, relpath)` pairs."""
    return pd.DataFrame([{"file_id": f"{s}:{r}", "source_name": s}
                         for s, r in rows])


# --------------------------------------------------------------------------- #
# coverage is total, or the provider refuses
# --------------------------------------------------------------------------- #

def test_a_key_that_misses_a_row_raises_rather_than_writing_a_null():
    """The defect the module exists to refuse: a partial key produces a fold
    table where some rows are grouped and some are not, and the ungrouped ones
    leak with nothing reporting a problem."""
    files = _census([("s", "a.wav"), ("s", "b.wav")])
    keys = pd.Series({"s:a.wav": "g1"}, dtype="object")
    with pytest.raises(KeyError, match="covers 1 of 2"):
        gk._check_total("s", files, keys, "a partial key")


def test_a_key_present_but_null_counts_as_missing():
    """Mutation: `or pd.isna(keys.get(f))` deleted. A `map` against an
    incomplete lookup produces NaN, not an absent index entry, so dropping this
    clause passes every real join failure straight through."""
    files = _census([("s", "a.wav"), ("s", "b.wav")])
    keys = pd.Series({"s:a.wav": "g1", "s:b.wav": None}, dtype="object")
    with pytest.raises(KeyError, match="1 have no key"):
        gk._check_total("s", files, keys, "a key with a hole in it")


# --------------------------------------------------------------------------- #
# the three kinds
# --------------------------------------------------------------------------- #

def test_a_source_with_no_provider_still_gets_a_key_from_its_path(tmp_path):
    """⚠️ An earlier revision left these null, which read as tidy and made the
    column useless for the case docs/EDA/09 step 1 is *about*: `cfad-fake`'s
    eleven generators live in the path and nowhere else.

    Mutation: the `path_key` branch in `group_keys_for` removed. Every
    publisher-keyed source still reports correctly and the draw loses the only
    stratum that motivated widening it.
    """
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="src-x", pool="A", root="x"),),
                    gates=GateConfig(min_groups_per_role=2))
    files = _census([("src-x", f"{g}/f{i}.wav")
                     for g in ("gen1", "gen2", "gen3") for i in range(2)])
    out = gk.attach_group_keys(cfg, files)
    assert out[gk.KIND_COLUMN].unique().tolist() == ["path"]
    assert out[gk.KEY_COLUMN].notna().all()
    assert sorted(out[gk.KEY_COLUMN].unique()) == [
        "src-x/gen1", "src-x/gen2", "src-x/gen3"]


def test_a_path_key_is_the_prefix_not_the_name_at_that_depth(tmp_path):
    """🔴 `analyze/grouping.py` names the trap and this is where it would be
    walked into: with `train/spk1` and `test/spk1`, keying on `spk1` merges two
    speakers the publisher kept apart.

    Mutation: `"/".join(p[:d + 1])` -> `p[d]`. Reports 1 group where there are 2,
    with `spk1` under `train/` and `spk1` under `test/` merged into one.

    ⚠️ Two files per directory, and a floor of **3**, both on purpose. Two
    files per directory because `path_key` refuses a depth whose distinct count
    equals the file count -- a key giving every file its own group protects
    nothing -- so a one-file-per-directory fixture exercises that guard instead
    of the prefix rule. A floor of 3 because no depth then reaches it, so the
    fallback picks depth **1**; at depth 0 `train` and `test` are their own
    prefixes and the mutation is invisible. Measured: with a floor of 2 the
    mutant survived.
    """
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="src-y", pool="A", root="y"),),
                    gates=GateConfig(min_groups_per_role=3))
    files = _census([("src-y", f"{split}/spk1/{name}.wav")
                     for split in ("train", "test") for name in ("a", "b")])
    out = gk.attach_group_keys(cfg, files)
    assert out[gk.KEY_COLUMN].nunique() == 2
    assert sorted(out[gk.KEY_COLUMN].unique()) == ["src-y/test/spk1",
                                                   "src-y/train/spk1"]


def test_one_speaker_is_declared_single_not_reported_as_a_missing_key(tmp_path):
    """🔴 LJSpeech. Reporting it as *"needs the publisher's own key"* sends the
    reader after a file that does not exist and hides the fact that matters:
    the source cannot be rotated in a fold table at all."""
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="ljspeech", pool="A", root="lj"),))
    files = _census([("ljspeech", f"wavs/LJ-{i}.wav") for i in range(3)])
    out = gk.attach_group_keys(cfg, files)
    assert out[gk.KIND_COLUMN].unique().tolist() == [gk.SINGLE]
    assert out[gk.KEY_COLUMN].nunique() == 1

    report = grp.grouping_report(out, gates=GateConfig(min_groups_per_role=6))
    row = report.iloc[0]
    assert not row["meets_floor"]
    assert "no key can split it" in row["candidate_reason"]
    assert "needs the publisher's own key" not in row["candidate_reason"]


# --------------------------------------------------------------------------- #
# the publisher's key wins, even when it is smaller
# --------------------------------------------------------------------------- #

def test_the_publisher_key_replaces_path_depth_even_when_it_lowers_the_count(tmp_path):
    """🔴 The FakeMusicCaps correction. Path depth finds five generator
    directories; the real atom is the MusicCaps clip, which each of the five
    renders once. Taking the larger number takes the wrong one -- hold out one
    generator and its clips are still in the training fold four times over.

    Mutation: `if kind != groupkeys.PATH` inverted, or the branch made to keep
    whichever count is larger. Both put 5 back where 2 belongs.
    """
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="fakemusiccaps", pool="D", root="f"),))
    files = _census([("fakemusiccaps", f"{gen}/clip{i}.wav")
                     for gen in ("g1", "g2", "g3", "g4", "g5")
                     for i in range(2)])
    keyed = gk.attach_group_keys(cfg, files)
    assert keyed[gk.KEY_COLUMN].nunique() == 2      # two clips, not five models

    bare = grp.grouping_report(files, gates=GateConfig(min_groups_per_role=3))
    assert bare.iloc[0]["candidate_groups"] == 5

    report = grp.grouping_report(keyed, gates=GateConfig(min_groups_per_role=3))
    row = report.iloc[0]
    assert row["candidate_groups"] == 2
    assert row["candidate_depth"] is None
    assert row["group_key_kind"] == gk.PUBLISHER


def test_the_room_key_merges_channels_of_one_recording(tmp_path):
    """Ten microphones in one room record the same air. Mutation: `parts[:4]`
    widened to `parts[:5]`, which makes each channel its own group and reports
    92 atoms where there are 10."""
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="rirs-isotropic-noise", pool="E",
                                        root="r"),))
    files = _census([("rirs-isotropic-noise",
                      f"RVB2014_type1_noise_largeroom1_{i}.wav") for i in range(4)]
                    + [("rirs-isotropic-noise",
                        "RVB2014_type1_noise_smallroom2_1.wav")])
    out = gk.attach_group_keys(cfg, files)
    assert sorted(out[gk.KEY_COLUMN].unique()) == [
        "rirs-isotropic-noise/RVB2014_type1_noise_largeroom1",
        "rirs-isotropic-noise/RVB2014_type1_noise_smallroom2"]


def test_a_filename_without_the_room_shape_raises(tmp_path):
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="rirs-isotropic-noise", pool="E",
                                        root="r"),))
    with pytest.raises(ValueError, match="does not have the"):
        gk.attach_group_keys(cfg, _census([("rirs-isotropic-noise", "odd.wav")]))


# --------------------------------------------------------------------------- #
# the gate tells the two shortfalls apart
# --------------------------------------------------------------------------- #

def _report(rows):
    return pd.DataFrame(rows)


def test_the_gate_separates_a_keyed_shortfall_from_an_unlooked_at_one():
    """🔴 What made SONICS -- 5 real generators reported as 1 -- read exactly
    like LJSpeech, which is 1 and always will be. Mutation: the two lists
    merged back into one, which passes every count assertion and loses the only
    thing the reader needed."""
    cfg = load_eda_config(SHIPPED)
    groups = _report([
        {"source_name": "sonics", "meets_floor": False,
         "group_key_kind": "publisher"},
        {"source_name": "musan-speech", "meets_floor": False,
         "group_key_kind": "path"},
        {"source_name": "mlaad", "meets_floor": True, "group_key_kind": "path"},
    ])
    result = _g_eda3(cfg, groups)
    assert result.verdict == FAIL
    assert "no publisher key" in result.detail and "musan-speech" in result.detail
    assert "genuinely short" in result.detail and "sonics" in result.detail


def test_the_gate_passes_when_every_source_meets_the_floor():
    cfg = load_eda_config(SHIPPED)
    groups = _report([{"source_name": "a", "meets_floor": True,
                       "group_key_kind": "publisher"}])
    assert _g_eda3(cfg, groups).verdict == PASS


def test_the_gate_reads_a_report_written_before_the_kind_column_existed():
    """A saved `grouping_report.parquet` from an earlier revision has no
    `group_key_kind`. It must be read as `path` -- the old behaviour -- not
    raise three modules from where it was written."""
    cfg = load_eda_config(SHIPPED)
    groups = _report([{"source_name": "a", "meets_floor": False}])
    assert _g_eda3(cfg, groups).verdict == FAIL


# --------------------------------------------------------------------------- #
# the census carries it, and the shipped config accounts for every source
# --------------------------------------------------------------------------- #

def test_consolidate_writes_the_key_columns(tmp_path):
    """Mutation: the `attach_group_keys` call removed from `consolidate`. The
    census still has every row and every metadata column, and the key silently
    never reaches `sample.json` or the grouping report."""
    root = tmp_path / "interim"
    for i in range(2):
        _write(root / "ljspeech/wavs" / f"LJ{i}.wav", seed=i)
    cfg = EdaConfig(
        root=root, out=tmp_path / "out",
        sources=(SourceSpec(name="ljspeech", pool="A", root="ljspeech",
                            suffixes=(".wav",)),),
        probe=ProbeConfig(workers=1, shard_size=2))
    probe_source(cfg, cfg.source("ljspeech"))
    files = pd.read_parquet(consolidate(cfg, "A"))
    assert gk.KEY_COLUMN in files.columns and gk.KIND_COLUMN in files.columns
    assert files[gk.KIND_COLUMN].unique().tolist() == [gk.SINGLE]


def test_every_shipped_source_either_has_a_provider_or_says_why_not():
    """🔴 The silence this module exists to remove. A source in neither table
    falls back to path depth with no line anywhere saying that was a decision --
    which is the state all thirteen were in before step 1."""
    cfg = load_eda_config(SHIPPED)
    runnable = [s.name for s in cfg.sources if not s.blocked]
    unaccounted = [n for n in runnable
                   if n not in gk.PROVIDERS and n not in gk.NO_PROVIDER]
    assert unaccounted == [], (
        f"{unaccounted} have neither a provider nor a recorded reason for "
        f"using path depth; add one to eda.groupkeys")


def test_the_shipped_config_still_reads_sonics_key_from_the_raw_tree():
    """The CSV is beside `fake_songs/` in `raw/`, not in the unpacked interim
    tree. Mutation: `cfg.raw` -> `cfg.root`, which is a FileNotFoundError on a
    real run and invisible in any fixture that writes both trees."""
    cfg = load_eda_config(SHIPPED)
    sonics = cfg.source("sonics")
    assert (cfg.raw / sonics.root / "payload" / "fake_songs.csv") != (
        cfg.source_root(sonics) / "payload" / "fake_songs.csv")


# --------------------------------------------------------------------------- #
# the one key that crosses a source boundary
# --------------------------------------------------------------------------- #

def test_wavefake_and_ljspeech_share_a_key_because_they_share_a_speaker(tmp_path):
    """🔴 B1's precondition. WaveFake's seven `ljspeech_*` directories and its
    Common-Voice-prompt directory are LJSpeech's speaker re-vocoded, so they get
    `ljspeech`'s own key rather than one of their own -- a fold builder grouping
    on `group_key` then keeps the real utterance and its vocoded twin on the
    same side of every boundary without being told the pairing exists.

    Mutation: the key made source-local (`f"{source.name}/..."`). Every count in
    this file still holds and `LJ001-0001` starts appearing on both sides.
    """
    cfg = EdaConfig(
        root=tmp_path, out=tmp_path / "out",
        sources=(SourceSpec(name="ljspeech", pool="A", root="lj"),
                 SourceSpec(name="wavefake", pool="B", root="wf")))
    files = _census(
        [("ljspeech", "LJSpeech-1.1/wavs/LJ001-0001.wav")]
        + [("wavefake", f"generated_audio/ljspeech_{v}/LJ001-0001_gen.wav")
           for v in ("melgan", "hifiGAN", "waveglow")]
        + [("wavefake", "generated_audio/"
                        "common_voices_prompts_from_conformer_fastspeech2_pwg_"
                        "ljspeech/gen_0.wav")]
        + [("wavefake", "generated_audio/jsut_parallel_wavegan/"
                        "BASIC5000_0001_gen.wav")])
    out = gk.attach_group_keys(cfg, files)
    by_source = out.groupby("source_name")[gk.KEY_COLUMN].apply(set)
    assert gk.LJ_SPEAKER in by_source["ljspeech"]
    assert by_source["wavefake"] == {gk.LJ_SPEAKER, gk.JSUT_SPEAKER}
    # The real recording and its three vocoded twins are one group, not four.
    assert int((out[gk.KEY_COLUMN] == gk.LJ_SPEAKER).sum()) == 5


def test_an_unrecognised_wavefake_directory_raises_rather_than_guessing(tmp_path):
    """A vocoder directory naming a voice we have not decided about must stop
    the run. Defaulting it to LJSpeech would silently merge a second speaker
    into the pairing group that B1 rests on."""
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="wavefake", pool="B", root="wf"),))
    with pytest.raises(ValueError, match="no recognised vocoder directory"):
        gk.attach_group_keys(
            cfg, _census([("wavefake", "generated_audio/vctk_melgan/x.wav")]))


# --------------------------------------------------------------------------- #
# what the shipped config records about the two decisions of docs/EDA/09 step 1
# --------------------------------------------------------------------------- #

def test_the_shipped_config_spreads_the_draw_over_the_group_key():
    """Decided 2026-09-13. Mutation: `spread_by` emptied. The budget is
    unchanged -- 15,086 sampled files either way -- so no count catches it, and
    `cfad-fake` goes back to 17..108 files per generator."""
    cfg = load_eda_config(SHIPPED)
    assert cfg.sample.spread_by == ("group_key",)
    assert cfg.sample.stratify_by == ("source_name",), (
        "the budget stays per-source: widening this instead multiplies it by "
        "the group count (54,000 from cfad-fake alone)")


def test_the_shipped_config_drops_wavefakes_duplicate_copy_of_its_own_files():
    """🔴 Measured 2026-09-13: WaveFake's Common-Voice-prompt directory holds
    16,283 files and repeats all 16,283 byte-identical under a nested
    `generated/`. Counting both weights those prompts double in pool B and puts
    an identical pair on either side of any fold.

    Mutation: the exclusion removed. The census grows to 134,266 rows, every
    gate still passes, and E1 reports 16,283 duplicate groups that read as an
    ordinary within-source finding.
    """
    cfg = load_eda_config(SHIPPED)
    wavefake = cfg.source("wavefake")
    assert wavefake.exclude == ("generated",)
    assert wavefake.exclude_reason.strip(), "every exclusion must name its reason"
    # ⚠️ Whole path components: the exclusion must not swallow the
    # `generated_audio/` root that every WaveFake file sits under.
    assert not wavefake.excludes_path("generated_audio/ljspeech_melgan/a.wav")
    assert wavefake.excludes_path(
        "generated_audio/common_voices_prompts_from_conformer_fastspeech2_pwg_"
        "ljspeech/generated/gen_0.wav")


def test_compspoof_groups_by_parent_recording_not_by_split(tmp_path):
    """🔴 CompSpoof is segmented: one parent recording becomes many 4 s files,
    and its two splits share parents. Measured on the full census: **292 parent
    recordings appear in both `eval_source` and `test_source`, over 1,060
    files**. The path's own answer -- split x dataset, 6 tidy groups -- puts
    segment 0 of a Helsinki bus in train and segment 1 in validation.

    Mutation: the split put back into the key (`rel.parts[0:-1]` instead of
    `rel.parts[1:-1]`). The group count barely moves and every one of those
    1,060 files starts leaking again.
    """
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="compspoof-env-bonafide", pool="E",
                                        root="c"),))
    files = _census([
        ("compspoof-env-bonafide",
         f"{split}/env_sources/EnvSDD/bonafide/TUTASC2019Dev/"
         f"bus-helsinki-20-789-a_{i}.wav")
        for split, i in (("eval_source", 0), ("test_source", 1))])
    out = gk.attach_group_keys(cfg, files)
    assert out[gk.KEY_COLUMN].nunique() == 1, (
        "two segments of one recording must be one group whatever split they "
        "were filed under")


def test_each_compspoof_subtree_gets_its_publisher_s_own_parent_rule():
    """Every rule was read off the files. Mutation: any one pattern loosened to
    `(.+)`, which makes the parent the whole stem and the segments stop
    grouping."""
    from eda.groupkeys import _COMPSPOOF_PARENT

    cases = {
        "env_sources/AudioCapsEnv/bonafide": ("Y-4B1PkgXOMI_80_seg000",
                                              "Y-4B1PkgXOMI"),
        "env_sources/VGGSoundEnv/bonafide": ("-3z5mFRgbxc_000030.mp4_chunk1",
                                             "-3z5mFRgbxc"),
        "env_sources/EnvSDD/bonafide/UrbanSound8K": ("100263-2-0-137", "100263"),
        "env_sources/EnvSDD/bonafide/TUTASC2019Dev": (
            "airport-barcelona-0-12-a_0", "airport-barcelona-0-12-a"),
        "env_sources/EnvSDD/bonafide/TUTSED2016Dev": ("a001_33", "a001"),
    }
    for subtree, (stem, want) in cases.items():
        for marker, pattern in _COMPSPOOF_PARENT:
            if marker in subtree:
                assert pattern.match(stem).group(1) == want, (subtree, stem)
                break
        else:
            raise AssertionError(f"no rule matched {subtree}")


def test_an_unrecognised_compspoof_stem_becomes_its_own_group(tmp_path):
    """The safe direction: a group of one can never merge two recordings, only
    fail to merge segments of one."""
    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out",
                    sources=(SourceSpec(name="compspoof-env-bonafide", pool="E",
                                        root="c"),))
    files = _census([("compspoof-env-bonafide",
                      "eval_source/env_sources/Unknown/bonafide/odd-name.wav")])
    out = gk.attach_group_keys(cfg, files)
    assert out[gk.KEY_COLUMN].notna().all()
    assert out[gk.KEY_COLUMN].nunique() == 1
