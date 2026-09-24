"""Test time through the one shipped chain (processing/infer.py, script.py):
the P0-a probe scores exactly 0.5000, a decode failure is a counted fallback
row, a constant real prediction is refused, and the model sees exactly
``ship(load_audio(file))``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from metrics.dacon import PREDICTION_COLUMNS, dacon_score
from processing.infer import (ConstantModel, InferenceReport, list_test_files, predict_files,
                              write_submission)
from processing.ship import ShipConfig, ship
from training.render import load_audio
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def test_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("p0a")
    manifest = synthetic_manifest(n_per_pool=6, n_whole_file=4, seed=0,
                                  duration_range=(4.0, 8.0))
    write_synthetic_corpus(manifest, root / "corpus", seed=0)
    test = root / "data" / "test"
    test.mkdir(parents=True)
    for i, p in enumerate(sorted((root / "corpus").rglob("*"))):
        if p.suffix.lower() in (".wav", ".flac", ".mp3"):
            (test / f"TEST_{i:04d}{p.suffix}").write_bytes(p.read_bytes())
    return test


def _truth(ids, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"ID": list(ids)})
    for c in ("voice_present", "music_present", "voice_fake", "music_fake"):
        t[c] = rng.integers(0, 2, len(t))
    t["file_fake"] = ((t.voice_present & t.voice_fake)
                      | (t.music_present & t.music_fake)).astype(int)
    return t


def test_the_contract_probe_scores_exactly_half(test_dir, tmp_path):
    """P0-a (docs/architecture/03): every column constant -> 0.5000, exactly."""
    files = list_test_files(test_dir)
    assert len(files) >= 10
    report = predict_files(ConstantModel(0.5), files, ShipConfig())
    assert report.n == len(files) and report.n_fallback == 0
    out = write_submission(report, tmp_path / "output" / "submission.csv",
                           require_variation=False)
    sub = pd.read_csv(out)
    assert list(sub.columns) == ["ID"] + list(PREDICTION_COLUMNS)
    merged = _truth(sub["ID"]).merge(sub, on="ID")
    assert dacon_score(merged).score == pytest.approx(0.5, abs=1e-9)


def test_script_py_writes_the_probe_submission(test_dir, tmp_path):
    out = tmp_path / "output" / "submission.csv"
    r = subprocess.run([sys.executable, str(REPO / "script.py"), "--probe",
                        "--test-dir", str(test_dir), "--out", str(out)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr[-800:]
    sub = pd.read_csv(out)
    assert len(sub) == len(list_test_files(test_dir))
    assert (sub[list(PREDICTION_COLUMNS)] == 0.5).all().all()


def test_a_file_that_cannot_be_decoded_is_a_counted_fallback_row(test_dir, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    for p in list_test_files(test_dir)[:12]:
        (bad / p.name).write_bytes(p.read_bytes())
    (bad / "TEST_broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    files = list_test_files(bad)
    report = predict_files(ConstantModel(0.5), files, ShipConfig(), max_fallback=0.5)
    assert report.n == len(files) and report.n_fallback == 1
    assert report.failures[0][0] == "TEST_broken.wav"
    assert "TEST_broken" in report.ids
    with pytest.raises(RuntimeError, match="fell back"):
        predict_files(ConstantModel(0.5), files, ShipConfig(), max_fallback=0.01)


def test_a_constant_real_prediction_is_refused(test_dir, tmp_path):
    """02 §guards 5: a silent model failure must not masquerade as a run."""
    report = predict_files(ConstantModel(0.7), list_test_files(test_dir), ShipConfig())
    with pytest.raises(RuntimeError, match="constant"):
        write_submission(report, tmp_path / "s.csv")


class _Recording:
    """A 'model' that reports what it was handed: the mean over the valid prefix."""

    def __init__(self):
        self.seen: list[torch.Tensor] = []

    def probs(self, wav, lengths):
        self.seen.append((wav.clone(), lengths.clone()))
        v = torch.stack([wav[i, : int(lengths[i])].double().abs().mean()
                         for i in range(wav.shape[0])])
        return {c: v for c in PREDICTION_COLUMNS}


def test_the_model_sees_exactly_the_shipped_decode(test_dir):
    """The one chain: what `predict_files` hands the model is
    ``ship(load_audio(file))`` over the file's own length, batch or not."""
    files = list_test_files(test_dir)[:5]
    cfg = ShipConfig()
    rec = _Recording()
    report = predict_files(rec, files, cfg, batch_size=3)
    seen = torch.cat([w[i, : int(n[i])] for w, n in rec.seen for i in range(w.shape[0])])
    solo = []
    for p in files:
        x = torch.from_numpy(load_audio(p, sample_rate=cfg.sample_rate))
        solo.append(ship(x[None], cfg, torch.tensor([x.shape[-1]]))[0])
    assert torch.equal(seen, torch.cat(solo))
    assert report.probs["FILE_FAKE_PROB"] == pytest.approx(
        [float(s.double().abs().mean()) for s in solo], abs=1e-6)


def test_the_report_frame_has_the_submission_columns():
    r = InferenceReport()
    r.ids.append("x")
    for c in PREDICTION_COLUMNS:
        r.probs[c].append(0.25)
    assert list(r.frame().columns) == ["ID"] + list(PREDICTION_COLUMNS)
