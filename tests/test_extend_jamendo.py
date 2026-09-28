"""strategy-v6 MTG-Jamendo reader: instrumentals are pool-C rows keyed by artist, and
rows the vocal screen did not keep never enter."""

from __future__ import annotations

import pandas as pd

from processing.extend import NEW_CORPORA, _mtg_jamendo


def _write(tmp_path):
    d = tmp_path / "interim" / "mtg-jamendo"
    d.mkdir(parents=True)
    pd.DataFrame({
        "file": ["seg10/00/track_1_0.wav", "seg10/01/track_2_0.wav"],
        "track_id": ["track_1", "track_2"],
        "source_name": ["mtg-jamendo/artist_1", "mtg-jamendo/artist_2"],
        "speaker_ref_id": ["artist:foo", "artist:bar"],
        "licence_verdict": ["allow", "derivatives_barred"],
        "kept": [True, False]}).to_csv(d / "metadata.csv", index=False)


def test_instrumentals_are_pool_c_rows(tmp_path):
    _write(tmp_path)
    out = _mtg_jamendo(tmp_path, pd.DataFrame())
    assert list(out["file_id"]) == ["mtg-jamendo:seg10/00/track_1_0"]            # kept=False never enters
    assert list(out["path"]) == ["interim/mtg-jamendo/seg10/00/track_1_0.wav"]
    assert list(out["source_name"]) == ["mtg-jamendo/artist_1"]
    assert list(out["speaker_ref_id"]) == ["artist:foo"]
    assert out["artifact_family"].isna().all() and out["domain_key"].isna().all()
    assert {nc.name: nc.pool for nc in NEW_CORPORA}["mtg-jamendo"] == "C"


def test_missing_metadata_is_empty(tmp_path):
    assert _mtg_jamendo(tmp_path, pd.DataFrame()).empty


def test_second_batch_reads_its_own_dir_with_the_same_keys(tmp_path):
    """strategy-v6c: interim/mtg-jamendo-2 rows are their own corpus (file_id prefix =
    cache dir) but keep the artist keys, so an artist's new tracks join its atom."""
    d = tmp_path / "interim" / "mtg-jamendo-2"
    d.mkdir(parents=True)
    pd.DataFrame({
        "file": ["seg10/50/track_9_1.wav"], "track_id": ["track_9"],
        "source_name": ["mtg-jamendo/artist_1"], "speaker_ref_id": ["artist:foo"],
        "licence_verdict": ["allow"], "kept": [True]}).to_csv(d / "metadata.csv", index=False)
    nc = {nc.name: nc for nc in NEW_CORPORA}["mtg-jamendo-2"]
    out = nc.reader(tmp_path, pd.DataFrame())
    assert nc.pool == "C"
    assert list(out["file_id"]) == ["mtg-jamendo-2:seg10/50/track_9_1"]
    assert list(out["path"]) == ["interim/mtg-jamendo-2/seg10/50/track_9_1.wav"]
    assert list(out["speaker_ref_id"]) == ["artist:foo"]
