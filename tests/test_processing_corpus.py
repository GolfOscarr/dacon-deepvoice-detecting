"""The corpus manifest builder: §4.1's keys per source from real path shapes,
D-14's actions, duplicate groups, the licence, the verdict sidecar, and every
rule asserted -- with the build shown to REFUSE a corpus that breaks one.
"""

import pandas as pd
import pytest

from processing.corpus import (EXTRA_COLUMNS, BuildInputs, apply_worklist, assign_keys,
                               build_manifest, check_rules, licence_verdicts)
from training.manifest import REQUIRED_COLUMNS



def _row(fid, path, source, pool=None, cell=None, gk=None, dur=12.0, sha=None):
    return {"file_id": fid, "path": path, "source_name": source,
            "row_kind": "component" if pool else "whole_file", "pool": pool, "cell": cell,
            "sha256": sha or f"sha_{fid}", "probe_ok": True, "container": "wav",
            "orig_sr": 16000, "orig_channels": 1, "duration_s": dur, "group_key": gk or source}


def _files():
    rows = [
        # cfad: three fake utterances, two with a real twin
        _row("cf1", "cfad/x/CFAD/clean_version/dev_clean/fake_clean/gl/SSB03540001_gl.wav",
             "cfad-fake", "B", gk="cfad-fake/dev_clean/fake_clean/gl"),
        _row("cf2", "cfad/x/CFAD/clean_version/dev_clean/fake_clean/wavenet/SSB03540002_wavenet.wav",
             "cfad-fake", "B", gk="cfad-fake/dev_clean/fake_clean/wavenet"),
        _row("cf3", "cfad/x/CFAD/clean_version/dev_clean/fake_clean/gl/SSB99990001_gl.wav",
             "cfad-fake", "B", gk="cfad-fake/dev_clean/fake_clean/gl"),
        _row("cr1", "cfad/x/CFAD/clean_version/dev_clean/real_clean/aishell3/SSB03540001.wav",
             "cfad-real", "A", gk="cfad-real/dev_clean/real_clean/aishell3"),
        _row("cr2", "cfad/x/CFAD/clean_version/dev_clean/real_clean/aishell3/SSB03540002.wav",
             "cfad-real", "A", gk="cfad-real/dev_clean/real_clean/aishell3"),
        _row("cr3", "cfad/x/CFAD/clean_version/dev_clean/real_clean/thchs30/A7_191.wav",
             "cfad-real", "A", gk="cfad-real/dev_clean/real_clean/thchs30"),
        # ljspeech + wavefake twins, three families + jsut + cv
        _row("lj1", "ljspeech/mdc-1.1/LJSpeech-1.1/wavs/LJ001-0001.wav", "ljspeech", "A", gk="ljspeech/LJ"),
        _row("wf1", "wavefake/z/generated_audio/ljspeech_melgan/LJ001-0001_gen.wav", "wavefake", "B", gk="ljspeech/LJ"),
        _row("wf2", "wavefake/z/generated_audio/ljspeech_waveglow/LJ001-0001.wav", "wavefake", "B", gk="ljspeech/LJ"),
        _row("wf3", "wavefake/z/generated_audio/ljspeech_multi_band_melgan/LJ001-0001_gen.wav", "wavefake", "B", gk="ljspeech/LJ"),
        _row("wf4", "wavefake/z/generated_audio/jsut_parallel_wavegan/BASIC5000_0001_gen.wav", "wavefake", "B", gk="ljspeech/LJ"),
        _row("wf5", "wavefake/z/generated_audio/common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/gen_0.wav", "wavefake", "B", gk="ljspeech/LJ"),
        _row("ml1", "mlaad/v9/payload/fake/ko/Edge-TTS/book_1_f000001.wav", "mlaad", "B", gk="mlaad/fake/ko"),
        _row("zk1", "zeroth-korean/o/train_data_01/003/104/104_003_0019.flac", "zeroth-korean", "A", gk="zeroth-korean/train_data_01/003/104"),
        # music: fma (licence + artist), musan, fakemusiccaps, sonics
        _row("fm1", "fma/fma_small/fma_small/000/000002.mp3", "fma", "C", gk="fma/fma_small/000", dur=30.0),
        _row("fm2", "fma/fma_small/fma_small/000/000005.mp3", "fma", "C", gk="fma/fma_small/000", dur=30.0),
        _row("fm3", "fma/fma_small/fma_small/000/000010.mp3", "fma", "C", gk="fma/fma_small/000", dur=30.0),
        _row("mm1", "musan/o/musan/music/fma/music-fma-0000.wav", "musan-music", "C", gk="musan-music/fma/Airglow", dur=60.0),
        _row("fc1", "fakemusiccaps/zenodo-15063698/musicldm/SLdVSirZMSI.wav", "fakemusiccaps", "D", gk="fakemusiccaps/SLdVSirZMSI", dur=10.0),
        _row("fc2", "fakemusiccaps/zenodo-15063698/MusicGen_medium/SLdVSirZMSI.wav", "fakemusiccaps", "D", gk="fakemusiccaps/SLdVSirZMSI", dur=10.0),
        _row("so1", "sonics/hf-2025/fake_songs/fake_00001_suno_0.mp3", "sonics", cell=8, gk="sonics/chirp-v3.5", dur=120.0),
        _row("so2", "sonics/hf-2025/fake_songs/fake_27547_udio_1.mp3", "sonics", cell=8, gk="sonics/udio-120s", dur=120.0),
        # noise, with a byte-identical pair and a speech-carrying file
        _row("cs1", "compspoof-v2/E/eval_source/env_sources/AudioCapsEnv/bonafide/Y-47_30_seg000.wav", "compspoof-env-bonafide", "E", gk="compspoof-env-bonafide/env_sources/AudioCapsEnv/bonafide/Y-47", dur=4.0),
        _row("cs2", "compspoof-v2/E/test_source/env_sources/AudioCapsEnv/bonafide/Y-47_30_seg001.wav", "compspoof-env-bonafide", "E", gk="compspoof-env-bonafide/env_sources/AudioCapsEnv/bonafide/Y-47", dur=4.0, sha="same"),
        _row("mn1", "musan/o/musan/noise/free-sound/noise-free-sound-0000.wav", "musan-noise", "E", gk="musan-noise/free-sound", dur=4.0, sha="same"),
        _row("mn2", "musan/o/musan/noise/free-sound/noise-free-sound-0001.wav", "musan-noise", "E", gk="musan-noise/free-sound", dur=3.0),
        # a degenerate fake voice row, and one with a decode failure
        _row("dg1", "cfad/x/CFAD/clean_version/dev_clean/fake_clean/gl/SSB88880001_gl.wav", "cfad-fake", "B", gk="cfad-fake/dev_clean/fake_clean/gl"),
        _row("bad", "cfad/x/CFAD/clean_version/dev_clean/fake_clean/gl/SSB77770001_gl.wav", "cfad-fake", "B", gk="cfad-fake/dev_clean/fake_clean/gl"),
    ]
    return pd.DataFrame(rows)


def _tracks():
    cols = pd.MultiIndex.from_tuples([("artist", "id"), ("track", "license")])
    return pd.DataFrame([[10, "Attribution"], [10, "Attribution-NonCommercial-NoDerivatives 4.0"],
                         [11, "FMA-Limited: Download Only"]], index=[2, 5, 10], columns=cols)


def _inputs(**over):
    files = _files()
    worklist = pd.DataFrame([
        {"file_id": "fm1", "action": "reassign_cell", "target_cell": 5},
        {"file_id": "fc1", "action": "reassign_cell", "target_cell": 8},
        {"file_id": "cs1", "action": "restrict_noise", "target_cell": None},
        {"file_id": "dg1", "action": "degenerate", "target_cell": None},
        {"file_id": "cf3", "action": "unevidenceable", "target_cell": None},
    ])
    duplicates = files[files.sha256 == "same"][["file_id", "sha256"]]
    signal = pd.DataFrame([{"file_id": "bad", "signal_ok": False},
                           {"file_id": "cf1", "signal_ok": True}])
    base = dict(files=files, worklist=worklist, duplicates=duplicates, signal=signal,
                fma_tracks=_tracks(), fma_allow={2}, stages={"mlaad": "raw"},
                component_floor_s=4.0)
    return BuildInputs(**{**base, **over})


@pytest.fixture(scope="module")
def built():
    return build_manifest(_inputs())


# --------------------------------------------------------------------------- #
# §4.1 keys


def test_keys_per_source(built):
    m, _, _ = built
    r = m.set_index("file_id")
    assert r.at["cf1", "artifact_family"] == "cfad/gl" and r.at["cf1", "domain_key"] == "cfad-fake|cfad/gl"
    assert r.at["cf1", "speaker_ref_id"] == "SSB0354" and r.at["cf1", "pair_id"] == "cfad:SSB03540001"
    assert r.at["cr1", "pair_id"] == "cfad:SSB03540001" and pd.isna(r.at["cr1", "artifact_family"])
    assert pd.isna(r.at["cf3", "pair_id"]) and pd.isna(r.at["cr3", "pair_id"]), "one side is no pair"
    assert r.at["lj1", "speaker_ref_id"] == "ljspeech_LJ" and r.at["lj1", "pair_id"] == "lj:LJ001-0001"
    assert r.at["wf1", "artifact_family"] == "wf_melgan" and r.at["wf2", "artifact_family"] == "wf_gan"
    assert r.at["wf3", "artifact_family"] == "wf_mb_melgan"
    assert r.at["wf4", "artifact_family"] == "wf_jsut_parallel_wavegan" and r.at["wf4", "speaker_ref_id"] == "jsut"
    assert r.at["wf5", "artifact_family"] == "wf_cv_fastspeech2_pwg" and pd.isna(r.at["wf5", "pair_id"])
    assert r.at["wf1", "pair_id"] == "lj:LJ001-0001" and r.at["wf1", "domain_key"] == "wavefake|ljspeech_melgan"
    assert r.at["ml1", "artifact_family"] == "mlaad/Edge-TTS" and r.at["ml1", "speaker_ref_id"] == "ko/Edge-TTS"
    assert r.at["fm2", "speaker_ref_id"] == "fma_artist_10"
    assert r.at["fc2", "artifact_family"] == "fakemusiccaps/MusicGen_medium"
    assert r.at["fc2", "speaker_ref_id"] == "fakemusiccaps/MusicGen_medium/SLdVSirZMSI"
    assert r.at["fc1", "speaker_ref_id"] == "fakemusiccaps/musicldm/SLdVSirZMSI"
    assert r.at["so1", "artifact_family"] == "suno_chirp" and r.at["so2", "artifact_family"] == "udio"
    assert r.at["so1", "row_kind"] == "whole_file" and r.at["so1", "cell"] == 8


def test_paths_carry_the_stage_and_labels_come_from_the_tables(built):
    m, _, _ = built
    r = m.set_index("file_id")
    assert r.at["ml1", "path"].startswith("raw/mlaad/") and r.at["ml1", "stage"] == "raw"
    assert r.at["cf1", "path"].startswith("interim/cfad/")
    assert tuple(r.loc["cf1", ["label_voice_present", "label_music_present", "label_voice_fake"]]) == (1, 0, 1)
    assert pd.isna(r.at["cf1", "label_music_fake"])
    assert r.at["cf1", "label_confidence"] == "exact" and r.at["cr1", "label_confidence"] == "reported"
    assert (m["aug_strength"] == 1.0).all() and (m["scheme_version"] == "strategy-v1").all()
    assert list(m.columns) == list(REQUIRED_COLUMNS) + list(EXTRA_COLUMNS)


def test_source_name_is_the_publishers_atom_and_corpus_keeps_the_name(built):
    """docs/validation/01 §1: a fold atom is a family / sub-corpus / speaker /
    artist, never a whole corpus (13 atoms over 278k rows was infeasible)."""
    m, _, _ = built
    r = m.set_index("file_id")
    assert r.at["cf1", "corpus"] == "cfad-fake" and r.at["cf1", "source_name"] == "cfad/gl"
    assert r.at["cr1", "source_name"] == "cfad-real/dev_clean/real_clean/aishell3"
    assert r.at["cr3", "source_name"] == "cfad-real/dev_clean/real_clean/thchs30"
    assert r.at["wf1", "source_name"] == "wf_melgan" and r.at["wf2", "source_name"] == "wf_gan"
    assert r.at["fm2", "source_name"] == "fma_artist_10" and r.at["fm2", "corpus"] == "fma"
    assert r.at["fm1", "source_name"] == "fma_artist_10"          # reassigned, still its artist
    assert r.at["so1", "source_name"] == "suno_chirp"
    assert r.at["fc2", "source_name"] == "fakemusiccaps/MusicGen_medium"
    assert r.at["zk1", "source_name"] == "zeroth-korean/train_data_01/003/104"


# --------------------------------------------------------------------------- #
# D-14, OFF-3, OFF-4


def test_the_worklist_actions(built):
    m, _, _ = built
    r = m.set_index("file_id")
    assert r.at["fm1", "row_kind"] == "whole_file" and r.at["fm1", "cell"] == 5
    assert pd.isna(r.at["fm1", "pool"]) and pd.isna(r.at["fm1", "artifact_family"])
    assert tuple(r.loc["fm1", ["label_voice_present", "label_music_present"]]) == (1, 1)
    assert r.at["fm1", "reassigned_from"] == "C"
    assert r.at["fc1", "cell"] == 8 and r.at["fc1", "artifact_family"] == "fakemusiccaps/musicldm"
    assert r.at["cs1", "noise_has_speech"] and r.at["cs1", "pool"] == "E"
    assert not r.at["cs2", "noise_has_speech"]
    assert r.at["cf3", "label_confidence"] == "reported"
    assert "dg1" not in r.index, "degenerate rows are dropped by the sidecar"


def test_duplicates_share_a_dup_group(built):
    m, _, _ = built
    r = m.set_index("file_id")
    assert r.at["cs2", "dup_group"] == r.at["mn1", "dup_group"] and r.at["cs2", "dup_group"].startswith("sha:")
    assert pd.isna(r.at["cs1", "dup_group"])


def test_the_licence_and_the_verdict_sidecar(built):
    m, side, report = built
    r = m.set_index("file_id")
    assert r.at["fm1", "licence_verdict"] == "allow"
    assert r.at["fm2", "licence_verdict"] == "derivatives_barred"
    assert "fm3" not in r.index                                # deny -> dropped
    s = side.set_index(["file_id", "filter"])
    assert s.at[("fm3", "licence"), "verdict"] == "drop"
    assert s.at[("bad", "corruption"), "verdict"] == "drop" and "bad" not in r.index
    assert s.at[("mn2", "usable_duration"), "verdict"] == "drop" and "mn2" not in r.index
    assert s.at[("cf1", "usable_duration"), "verdict"] == "keep"
    assert s.at[("dg1", "label_evidence"), "verdict"] == "drop"
    assert "min_seconds=4.0" in s.at[("mn2", "usable_duration"), "threshold_version"]
    assert report["dropped"] == {"corruption": 1, "usable_duration": 1, "licence": 1,
                                 "label_evidence": 1}
    assert report["noise_has_speech"] == 1 and report["cell"][5] == 1


def test_licence_unknown_is_deny():
    files = _files()
    v = licence_verdicts(files, None, {2})
    fm = v[files.source_name == "fma"]
    assert list(fm) == ["allow", "deny", "deny"]
    assert v[files.source_name != "fma"].isna().all()


# --------------------------------------------------------------------------- #
# the rules refuse a corpus that breaks them


def test_a_missing_pool_is_refused():
    inp = _inputs()
    files = inp.files[inp.files.pool != "E"]
    with pytest.raises(AssertionError, match="pool E is empty"):
        build_manifest(BuildInputs(**{**inp.__dict__, "files": files}))


def test_a_real_row_with_a_family_is_refused(built):
    m, _, _ = built
    broken = m.copy()
    broken.loc[broken.file_id == "cr1", "artifact_family"] = "cfad/gl"
    with pytest.raises(AssertionError, match="real components carrying an artifact_family"):
        check_rules(broken)


def test_a_one_sided_pair_is_refused(built):
    m, _, _ = built
    broken = m.copy()
    broken.loc[broken.file_id == "cr3", "pair_id"] = "cfad:SSB00000000"
    with pytest.raises(AssertionError, match="one side only"):
        check_rules(broken)


def test_apply_worklist_ignores_files_not_in_the_manifest():
    m = build_manifest(_inputs())[0]
    out = apply_worklist(m.drop(columns=list(EXTRA_COLUMNS)),
                         pd.DataFrame([{"file_id": "ghost", "action": "reassign_cell", "target_cell": 5}]))
    assert len(out) == len(m) and not out["noise_has_speech"].any()


def test_assign_keys_without_fma_metadata_keeps_the_group_key():
    files = _files()
    out = assign_keys(files, None)
    assert out.loc[files.source_name == "fma", "speaker_ref_id"].iloc[0] == "fma/fma_small/000"
