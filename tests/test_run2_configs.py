"""docs/training/11: the run-2 control (T5) must draw exactly like run 2 -- the two
processing configs may differ ONLY in draw.scheme_version (the manifest's stamp)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_v3_control_differs_only_in_scheme():
    a, b = _load("processing_run2.yaml"), _load("processing_run2_v3ctl.yaml")
    assert a["draw"]["scheme_version"] == "strategy-v4"
    assert b["draw"]["scheme_version"] == "strategy-v3"
    a["draw"].pop("scheme_version")
    b["draw"].pop("scheme_version")
    assert a == b


def test_run2_model_is_first_run_plus_batched_tokens():
    a, b = _load("c_first_run.yaml"), _load("c_run2.yaml")
    assert b["frontends"]["audio"].pop("batched_tokens") is True
    a.pop("name"), b.pop("name")
    assert a == b


def test_run2_train_configs_differ_only_in_lr():
    a, b = _load("train_run2.yaml"), _load("train_run2_scratch.yaml")
    assert (a.pop("lr"), b.pop("lr")) == (1.0e-4, 2.0e-4)
    assert a == b
