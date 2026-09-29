"""`_emilia` reads N3's metadata_yodas2.csv beside metadata.csv (kept rows only)."""

from __future__ import annotations

import pandas as pd

from processing.extend import _emilia

COLS = ["file", "part", "speaker_ref_id", "text", "duration_s", "sample_rate", "licence", "kept",
        "drop_reason", "source_tar", "dnsmos", "orig_format", "diar_speaker"]


def _write(path, rows):
    pd.DataFrame([dict(zip(COLS, r)) for r in rows], columns=COLS).to_csv(path, index=False)


def _row(file, spk, kept):
    return [file, "yodas", spk, "안녕", 5.0, 24000, "CC-BY-4.0", kept, "", "t.tar", 3.0, "mp3", spk + "_SPEAKER_00"]


def test_emilia_concatenates_yodas2(tmp_path):
    d = tmp_path / "interim" / "emilia-ko"
    d.mkdir(parents=True)
    _write(d / "metadata.csv", [_row("yodas/A/a.mp3", "emilia-ko/yodas/KO_a", True),
                                _row("yodas/A/b.mp3", "emilia-ko/yodas/KO_b", False)])
    _write(d / "metadata_yodas2.csv", [_row("yodas/B/c.mp3", "emilia-ko/yodas/KO_c", True),
                                       _row("yodas/B/d.mp3", "emilia-ko/yodas/KO_d", False)])
    out = _emilia(tmp_path, "ko")
    assert sorted(out["path"]) == ["interim/emilia-ko/yodas/A/a.mp3", "interim/emilia-ko/yodas/B/c.mp3"]
    assert set(out["speaker_ref_id"]) == {"emilia-ko/yodas/KO_a", "emilia-ko/yodas/KO_c"}
    assert (out["lang"] == "ko").all()


def test_emilia_without_yodas2(tmp_path):
    d = tmp_path / "interim" / "emilia-en"
    d.mkdir(parents=True)
    _write(d / "metadata.csv", [_row("yodas/A/a.mp3", "emilia-en/yodas/EN_a", True)])
    assert list(_emilia(tmp_path, "en")["path"]) == ["interim/emilia-en/yodas/A/a.mp3"]
    assert _emilia(tmp_path, "zz").empty
