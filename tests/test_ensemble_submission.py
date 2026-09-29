"""The per-file PREDICTION ensemble (rule 2.4): `processing.infer.ensemble_probs`,
`scripts/package_submission.py --member` (members of different architectures,
shipped side by side), and `script.py`'s ensemble path, which must write exactly
the average of what each member scores alone."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from metrics.dacon import PREDICTION_COLUMNS
from models.config import load_model_config
from models.model import DeepVoiceNet, load_checkpoint, save_checkpoint
from processing.config import (dump_processing_config, load_processing_config,
                               processing_config_from_dict)
from processing.infer import ensemble_probs, list_test_files, predict_files
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _members(*rows):
    return [{c: np.asarray(r, dtype=np.float64) for c in PREDICTION_COLUMNS} for r in rows]


# --------------------------------------------------------------------------- #
# ensemble_probs

def test_prob_mean_and_logit_mean():
    a, b = [0.2, 0.9, 0.5], [0.6, 0.1, 0.5]
    pm = ensemble_probs(_members(a, b), "prob_mean")
    np.testing.assert_allclose(pm["FILE_FAKE_PROB"], [0.4, 0.5, 0.5])
    lm = ensemble_probs(_members(a, b), "logit_mean")
    z = (np.log(np.array(a) / (1 - np.array(a))) + np.log(np.array(b) / (1 - np.array(b)))) / 2
    np.testing.assert_allclose(lm["FILE_FAKE_PROB"], 1 / (1 + np.exp(-z)))


def test_member_weights_and_a_single_member_is_the_identity():
    a, b = [0.2, 0.9], [0.6, 0.1]
    w = ensemble_probs(_members(a, b), "prob_mean", [3, 1])
    np.testing.assert_allclose(w["VOICE_FAKE_PROB"], [0.75 * 0.2 + 0.25 * 0.6,
                                                      0.75 * 0.9 + 0.25 * 0.1])
    for rule in ("prob_mean", "logit_mean"):
        np.testing.assert_allclose(ensemble_probs(_members(a), rule)["VOICE_FAKE_PROB"], a)


def test_is_per_file():
    """Rule 2.4: a file's output depends only on its own member rows."""
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.01, 0.99, 7), rng.uniform(0.01, 0.99, 7)
    for rule in ("prob_mean", "logit_mean"):
        together = ensemble_probs(_members(a, b), rule, [0.6, 0.4])["FILE_FAKE_PROB"]
        for i in range(7):
            alone = ensemble_probs(_members([a[i]], [b[i]]), rule, [0.6, 0.4])
            assert alone["FILE_FAKE_PROB"][0] == pytest.approx(together[i], abs=1e-15)


def test_fallback_rows_are_dropped_per_member():
    a, b = [0.2, 0.5, 0.5], [0.6, 0.8, 0.5]
    out = ensemble_probs(_members(a, b), "prob_mean",
                         valid=[[True, False, False], [True, True, False]])
    np.testing.assert_allclose(out["FILE_FAKE_PROB"], [0.4, 0.8, 0.5])


def test_bad_rule_and_weights_are_refused():
    with pytest.raises(ValueError, match="rule"):
        ensemble_probs(_members([0.5]), "median")
    with pytest.raises(ValueError, match="weights"):
        ensemble_probs(_members([0.5], [0.5]), "prob_mean", [1.0])
    with pytest.raises(ValueError, match="weights"):
        ensemble_probs(_members([0.5], [0.5]), "prob_mean", [0.0, 0.0])


# --------------------------------------------------------------------------- #
# packaging + script.py

@pytest.fixture(scope="module")
def test_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ens")
    manifest = synthetic_manifest(n_per_pool=3, n_whole_file=2, seed=1,
                                  duration_range=(3.0, 5.0))
    write_synthetic_corpus(manifest, root / "corpus", seed=1)
    test = root / "data" / "test"
    test.mkdir(parents=True)
    for i, p in enumerate(sorted((root / "corpus").rglob("*.wav"))[:6]):
        (test / f"TEST_{i:04d}.wav").write_bytes(p.read_bytes())
    return test


def _run(tmp_path, name, seed):
    """A finished 'run' dir: scored.pt + processing.json (the real chain)."""
    torch.manual_seed(seed)
    model = DeepVoiceNet(load_model_config(REPO / "configs" / f"{name}.yaml")).eval()
    run = tmp_path / f"run_{name}"
    run.mkdir()
    save_checkpoint(model, run / "scored.pt")
    chain = dump_processing_config(load_processing_config(REPO / "configs" / "processing_v1.yaml"))
    (run / "processing.json").write_text(json.dumps(chain), encoding="utf-8")
    return run


def test_package_and_script_py_ensemble_of_two_architectures(tmp_path, monkeypatch, test_dir):
    pkg = _load_script("package_submission")
    ra, rb = _run(tmp_path, "a_stub", 0), _run(tmp_path, "b_stub", 1)
    out = tmp_path / "sub"
    monkeypatch.setattr(sys, "argv", [
        "package_submission.py", "--out", str(out), "--no-zip", "--file-mode", "max3",
        "--member", f"{ra / 'scored.pt'}::audio={tmp_path}",
        "--member", f"{rb / 'scored.pt'}::speech={tmp_path}::{rb / 'processing.json'}",
        "--ens-weights", "0.7,0.3"])
    assert pkg.main() == 0
    model_dir = out / "model"
    assert not (model_dir / "scored.pt").exists() and (out / "script.py").exists()
    meta = json.loads((model_dir / "members.json").read_text())
    ens = meta["ensemble"]
    assert ens["rule"] == "prob_mean" and ens["weights"] == [0.7, 0.3]
    assert [m["dir"] for m in ens["members"]] == ["members/0", "members/1"]
    for i, run in enumerate((ra, rb)):
        mdir = model_dir / "members" / str(i)
        info = json.loads((mdir / "members.json").read_text())
        assert info["scored"] == [str((run / "scored.pt").resolve())]
        assert info["file_mode"] == {"trained": "learned", "shipped": "max3"}
        assert load_checkpoint(mdir / "scored.pt").cfg.file_head.mode == "max3"
        assert (mdir / "processing.json").read_text() == (run / "processing.json").read_text()

    csv = tmp_path / "submission.csv"
    r = subprocess.run([sys.executable, str(out / "script.py"), "--test-dir", str(test_dir),
                        "--out", str(csv), "--device", "cpu"],
                       capture_output=True, text=True, cwd=out)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    assert "ensemble prob_mean of 2 member(s)" in r.stdout
    assert "0 fallback row(s)" in r.stdout
    sub = pd.read_csv(csv)

    # exactly the weighted per-file average of what each member scores alone
    files = list_test_files(test_dir)
    alone = []
    for i in range(2):
        mdir = model_dir / "members" / str(i)
        ship_cfg = processing_config_from_dict(
            json.loads((mdir / "processing.json").read_text())).ship
        alone.append(predict_files(load_checkpoint(mdir / "scored.pt"), files, ship_cfg).probs)
    want = ensemble_probs(alone, "prob_mean", [0.7, 0.3])
    assert list(sub.columns) == ["ID"] + list(PREDICTION_COLUMNS)
    assert list(sub["ID"]) == [f.stem for f in files]
    for c in PREDICTION_COLUMNS:
        np.testing.assert_allclose(sub[c].to_numpy(), want[c], rtol=0, atol=1e-12)
    # and it is a real blend, not one member
    assert not np.allclose(sub["FILE_FAKE_PROB"], alone[0]["FILE_FAKE_PROB"])


def test_member_excludes_scored_and_file_stack(tmp_path, monkeypatch):
    pkg = _load_script("package_submission")
    for extra in (["--scored", "a.pt"], ["--file-stack", "s.json"]):
        monkeypatch.setattr(sys, "argv", [
            "package_submission.py", "--out", str(tmp_path / "x"),
            "--member", "a.pt::a=b", *extra])
        with pytest.raises(SystemExit, match="--member excludes"):
            pkg.main()
    monkeypatch.setattr(sys, "argv", [
        "package_submission.py", "--out", str(tmp_path / "x"),
        "--member", "a.pt::a=b", "--member", "b.pt::a=b", "--ens-weights", "1"])
    with pytest.raises(SystemExit, match="ens-weights"):
        pkg.main()
    assert not (tmp_path / "x").exists()


def test_identical_frontend_weights_ship_once(tmp_path):
    pkg = _load_script("package_submission")
    src = tmp_path / "beats"
    src.mkdir()
    (src / "BEATs_iter3_plus_AS2M.pt").write_bytes(b"x" * 10)
    other = tmp_path / "xlsr"
    other.mkdir()
    for f in ("config.json", "pytorch_model.bin", "preprocessor_config.json"):
        (other / f).write_bytes(b"{}")

    def fake(fes):
        return SimpleNamespace(cfg=SimpleNamespace(frontends={
            k: SimpleNamespace(name=n, layers=None) for k, n in fes.items()}))

    model_dir = tmp_path / "model"
    shipped: dict = {}
    r0 = pkg._ship_frontends(fake({"audio": "beats", "speech": "xlsr_300m"}),
                             {"audio": src, "speech": other},
                             model_dir / "members" / "0" / "weights", shipped, rel_to=model_dir)
    r1 = pkg._ship_frontends(fake({"audio": "beats"}), {"audio": src},
                             model_dir / "members" / "1" / "weights", shipped, rel_to=model_dir)
    assert r0 == {"audio": "members/0/weights/audio", "speech": "members/0/weights/speech"}
    assert r1 == {"audio": "members/0/weights/audio"}
    assert not (model_dir / "members" / "1").exists()


def test_eval_probe_weights_must_be_one_or_one_per_model(monkeypatch):
    ev = _load_script("eval_probe")
    monkeypatch.setattr(sys, "argv", [
        "eval_probe.py", "--manifest-dir", "x", "--out", "y",
        "--scored", "a.pt", "--scored", "b.pt", "--scored", "c.pt",
        "--weights", "a=b", "--weights", "a=c"])
    with pytest.raises(SystemExit, match="one per --scored"):
        ev.main()
    monkeypatch.setattr(sys, "argv", [
        "eval_probe.py", "--manifest-dir", "x", "--out", "y",
        "--scored", "a.pt", "--scored", "b.pt", "--weights", "a=b", "--ens-weights", "1,2,3"])
    with pytest.raises(SystemExit, match="ens-weights"):
        ev.main()
