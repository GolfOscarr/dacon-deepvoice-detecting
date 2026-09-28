"""strategy-v6b ACE-Step instrumental reader: kept clips are pool-D rows of one
generator atom, and rows the QC dropped never enter."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

from processing.extend import NEW_CORPORA, _acestep_inst

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_folds_pinned import FAMILY_FOLD  # noqa: E402


def _write(tmp_path):
    d = tmp_path / "interim" / "acestep-inst"
    d.mkdir(parents=True)
    pd.DataFrame({
        "file": ["clips/acestep15inst_00000.wav", "clips/acestep15inst_00001.wav"],
        "file_id": ["acestep-inst:acestep15inst_00000", "acestep-inst:acestep15inst_00001"],
        "kept": [True, False], "drop_reason": ["", "ast-vocal-or-ambiguous"]}).to_csv(d / "metadata.csv", index=False)


def test_kept_clips_are_pool_d_rows_of_one_atom(tmp_path):
    _write(tmp_path)
    out = _acestep_inst(tmp_path, pd.DataFrame())
    assert list(out["file_id"]) == ["acestep-inst:acestep15inst_00000"]      # dropped never enters
    assert list(out["path"]) == ["interim/acestep-inst/clips/acestep15inst_00000.wav"]
    assert set(out["artifact_family"]) == {"acestep15-inst"}
    assert set(out["speaker_ref_id"]) == {"acestep-inst/acestep15-inst"}
    assert set(out["domain_key"]) == {"acestep-inst|acestep15-inst"}
    assert {nc.name: nc.pool for nc in NEW_CORPORA}["acestep-inst"] == "D"
    # pinned folds place a new family by its last path component: must be mapped
    assert FAMILY_FOLD["acestep15-inst"] == FAMILY_FOLD["acestep15"]


def test_missing_metadata_is_empty(tmp_path):
    assert _acestep_inst(tmp_path, pd.DataFrame()).empty


def test_caption_and_qc_rules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "synth3"))
    from music_acestep_inst_gen import caption, qc_reason
    c = caption("genre---instrumentalrock|genre---metal|instrument---electricguitar|mood/theme---dark",
                random.Random(0))
    assert "instrumental instrumental" not in c and "vocals" in c and "electric guitar" in c
    ok = {"ok": True, "zero_run_ms": 0.0, "zero_frac": 0.0, "rms_dbfs": -15.0, "vox_max": 0.01}
    assert qc_reason(ok) == ""
    assert qc_reason({**ok, "zero_run_ms": 25.0}) == "exact-zero"
    assert qc_reason({**ok, "zero_frac": 0.06}) == "exact-zero"
    assert qc_reason({**ok, "rms_dbfs": -45.0}) == "near-silent"
    assert qc_reason({**ok, "vox_max": 0.03}) == "ast-vocal-or-ambiguous"
    assert qc_reason({**ok, "vox_max": float("nan")}) == "untagged"
    assert qc_reason({"ok": False}) == "decode-failed"


def test_clips_are_cut_into_three_10s_rows(tmp_path):
    """Symmetric with the Jamendo side: the music draw's tile share is a row count."""
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "synth3"))
    from music_acestep_inst_gen import segments
    (tmp_path / "clips").mkdir()
    y = (0.1 * np.sin(np.arange(30 * 16000) / 10.0)).astype(np.float32)
    y[20 * 16000:] = 0.0                                   # a silent last third is dropped
    sf.write(tmp_path / "clips/a.wav", y, 16000, subtype="PCM_16")
    kept = pd.DataFrame({"file": ["clips/a.wav"], "id": ["a"], "file_id": ["acestep-inst:a"],
                         "offset_s": [4.0], "duration_s": [30.0]})
    out = segments(kept, tmp_path)
    assert list(out["file_id"]) == ["acestep-inst:a_0", "acestep-inst:a_1"]
    assert list(out["file"]) == ["seg10/a_0.wav", "seg10/a_1.wav"]
    assert list(out["clip"]) == ["clips/a.wav"] * 2 and list(out["offset_s"]) == [4.0, 14.0]
    assert all(sf.info(tmp_path / f).duration == 10.0 for f in out["file"])


def test_second_batch_is_the_same_atom_in_its_own_dir(tmp_path):
    """strategy-v6c: interim/acestep-inst-2 rows keep the family / atom / domain (fold 2
    with the rest of ACE-Step), paths point into their own dir."""
    d = tmp_path / "interim" / "acestep-inst-2"
    d.mkdir(parents=True)
    pd.DataFrame({"file": ["seg10/acestep15inst2_00000_0.wav"],
                  "file_id": ["acestep-inst-2:acestep15inst2_00000_0"], "kept": [True],
                  "drop_reason": [""]}).to_csv(d / "metadata.csv", index=False)
    nc = {nc.name: nc for nc in NEW_CORPORA}["acestep-inst-2"]
    out = nc.reader(tmp_path, pd.DataFrame())
    assert nc.pool == "D"
    assert list(out["file_id"]) == ["acestep-inst-2:acestep15inst2_00000_0"]
    assert list(out["path"]) == ["interim/acestep-inst-2/seg10/acestep15inst2_00000_0.wav"]
    assert set(out["artifact_family"]) == {"acestep15-inst"}
    assert set(out["speaker_ref_id"]) == {"acestep-inst/acestep15-inst"}
    assert set(out["domain_key"]) == {"acestep-inst|acestep15-inst"}
