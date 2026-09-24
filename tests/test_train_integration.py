"""06 P8: `scripts/train.py` on the processing pipeline, end to end on a
synthetic corpus -- the fold view, the processing sampler, the processing
renderer, the shipped chain once, the frozen eval specs on disk, the chain
written beside the weights, and `script.py` scoring those weights."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from processing.config import load_processing_config
from processing.splits import build_and_check, write_outputs
from training.folds import FoldConfig
from training.spec import SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A written corpus, its manifest and folds on disk, and a processing
    config pointing at them -- the layout `scripts/train.py` reads."""
    root = tmp_path_factory.mktemp("integration")
    manifest = synthetic_manifest(n_per_pool=60, n_whole_file=30, seed=0,
                                  duration_range=(4.5, 9.0))
    # real speakers coarsened (buckets of ~20); fake rows keep their own
    # speaker atoms -- a shared one would union every family into one fold atom
    a = manifest.index[manifest.pool == "A"]
    manifest.loc[a, "speaker_ref_id"] = [f"{s}_spk{i % 3}"
                                         for i, s in enumerate(manifest.loc[a, "source_name"])]
    write_synthetic_corpus(manifest, root / "corpus", seed=0)
    mdir = root / "manifests"
    mdir.mkdir()
    manifest.to_parquet(mdir / "manifest.parquet", index=False)
    fcfg = FoldConfig(n_folds=2, probe_share=0.1, assigned_at="2026-01-01T00:00:00+00:00",
                      probe_min_real_hours=0.0, probe_min_real_atoms=1,
                      caveat_min_role_hours=0.0)
    plan, report, summary = build_and_check(manifest, fcfg)
    write_outputs(mdir, plan, report, summary)

    raw = yaml.safe_load((REPO / "configs" / "processing_v1.yaml").read_text())
    raw["draw"]["duration_range"] = [4.0, 8.0]
    raw["draw"]["augments"] = [a for a in raw["draw"]["augments"] if a["name"] != "rir"]
    raw["render"]["root"] = str(root / "corpus")
    raw["render"]["cache_root"] = None
    raw["folds"]["n_folds"] = 2
    raw["folds"]["probe_min_real_hours"] = 0.0
    raw["folds"]["probe_min_real_atoms"] = 1
    raw["folds"]["assigned_at"] = "2026-01-01T00:00:00+00:00"
    raw["loop"]["device"] = "cpu"
    cfg_path = root / "processing_test.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    load_processing_config(cfg_path)                       # it parses
    return root, mdir, cfg_path


def _train(corpus, out: Path, *extra: str) -> subprocess.CompletedProcess:
    root, mdir, cfg_path = corpus
    cmd = [PY, str(REPO / "scripts" / "train.py"),
           "--manifest-dir", str(mdir), "--processing", str(cfg_path),
           "--model", "configs/a_stub.yaml", "--out", str(out),
           "--folds", "0", "--stages", "joint", "--epochs", "1", "--batch-size", "4",
           "--draws", "12", "--eval-n", "24", "--max-steps", "2",
           "--device", "cpu", "--allow-unquotable", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_dry_run_reads_the_processing_config_and_the_fold_table(corpus, tmp_path):
    r = _train(corpus, tmp_path / "dry", "--dry-run")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "scheme" in r.stdout and "folds    [0]" in r.stdout


def test_a_fold_trains_evaluates_and_writes_the_chain_beside_the_weights(corpus, tmp_path):
    out = tmp_path / "run"
    r = _train(corpus, out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    fold = out / "fold0"
    assert (fold / "scored.pt").exists() and (fold / "processing.json").exists()
    # the frozen eval specs are on disk and round-trip equal (05 C5)
    specs = [SampleSpec.from_dict(d) for d in json.loads((fold / "val_specs.json").read_text())]
    assert specs and all(s.transforms == () for s in specs), "eval mode: no augments"
    assert all(json.loads(json.dumps(s.to_dict())) == s.to_dict() for s in specs)
    assert [SampleSpec.from_dict(json.loads(json.dumps(s.to_dict()))) for s in specs] == specs
    chain = json.loads((fold / "processing.json").read_text())
    assert chain["ship"]["band_hz"] == [0, 7200] and chain["ship"]["channels"] == "downmix"
    assert (fold / "val_predictions.parquet").exists()
    preds = pd.read_parquet(fold / "val_predictions.parquet")
    assert len(preds) == len(specs)

    # script.py scores the weights behind the chain the run wrote
    model_dir = tmp_path / "submit" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "scored.pt").write_bytes((fold / "scored.pt").read_bytes())
    (model_dir / "processing.json").write_text((fold / "processing.json").read_text())
    test_dir = tmp_path / "submit" / "data" / "test"
    test_dir.mkdir(parents=True)
    root = corpus[0]
    for i, p in enumerate(sorted((root / "corpus").rglob("*.wav"))[:12]):
        (test_dir / f"TEST_{i:04d}.wav").write_bytes(p.read_bytes())
    sub_out = tmp_path / "submit" / "output" / "submission.csv"
    s = subprocess.run([PY, str(REPO / "script.py"), "--test-dir", str(test_dir),
                        "--model-dir", str(model_dir), "--out", str(sub_out), "--device", "cpu"],
                       capture_output=True, text=True, cwd=REPO)
    assert s.returncode == 0, s.stderr[-2000:]
    sub = pd.read_csv(sub_out)
    assert len(sub) == 12 and sub.iloc[:, 1:].std(axis=0).min() > 0


def test_the_raw_manifest_is_refused_without_folds(corpus, tmp_path):
    """05 A7: `scripts/train.py` goes through `apply_folds`; a sampler on the
    raw manifest raises, so a wrong call cannot train on PROBE and VAL."""
    from processing.config import load_processing_config
    from processing.sampler import Sampler
    root, mdir, cfg_path = corpus
    cfg = load_processing_config(cfg_path)
    raw = pd.read_parquet(mdir / "manifest.parquet")
    raw["fold"] = pd.array([None] * len(raw), dtype="Int64")
    with pytest.raises(ValueError, match="apply_folds"):
        Sampler(raw, cfg.draw)


def test_all_data_mode_trains_on_every_fold_and_saves_the_weights(corpus, tmp_path):
    """docs/training/07 §3: the shipping runs train on TRAIN + VAL of every fold,
    PROBE sealed, with pooled rendering and the cosine schedule from the config."""
    root, mdir, cfg_path = corpus
    raw = yaml.safe_load(cfg_path.read_text())
    raw["loop"].update(render_workers=2, lr_schedule="cosine", warmup_steps=1,
                       log_every=1)
    cfg2 = tmp_path / "processing_pool.yaml"
    cfg2.write_text(yaml.safe_dump(raw))
    out = tmp_path / "alldata"
    r = _train((root, mdir, cfg2), out, "--all-data", "--seed", "3", "--draws", "48")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    run = out / "all_data_seed3"
    assert (run / "scored.pt").exists() and (run / "processing.json").exists()
    assert not (run / "val_predictions.parquet").exists()
    assert len((run / "train_log.jsonl").read_text().splitlines()) == 2
    assert "all data, seed 3" in r.stdout
