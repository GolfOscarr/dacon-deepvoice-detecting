"""`_sing` (N2 fake stems): a SONICS stem must not share the PROBE-sealed `sonics/<version>`
key of SONICS' whole files, and its key (the draw's tile bucket) is cut to ~3 source songs."""

from __future__ import annotations

import pandas as pd

from processing.extend import _sing


def test_sonics_stems_get_their_own_key(tmp_path):
    d = tmp_path / "interim" / "sing-fake"
    d.mkdir(parents=True)
    pd.DataFrame({"file": ["chirp-v3/a.wav", "acestep-en/b.wav", "chirp-v3/c.wav"],
                  "generator_version": ["chirp-v3", "acestep-en", "chirp-v3"],
                  "speaker_ref_id": ["sonics/chirp-v3", "aisong-kr-en/acestep15", "sonics/chirp-v3"],
                  "source_file_id": ["sonics:x.mp3", "aisong-kr-en:en/y.mp3", "sonics:z.mp3"],
                  "lang": ["en", "en", "en"], "kept": [True, True, False]}).to_csv(d / "metadata.csv", index=False)
    out = _sing(tmp_path, pd.DataFrame(), "fake")
    assert list(out["speaker_ref_id"]) == ["sing-fake/chirp-v3/s0000", "aisong-kr-en/acestep15/s0000"]
    assert list(out["artifact_family"]) == ["sing-fake/chirp-v3", "sing-fake/acestep-en"]
