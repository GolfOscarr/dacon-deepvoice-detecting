"""`scripts/train.py --weights`: one path for a one-trunk config, name=DIR per
frontend for configs/c_first_run.yaml, which reads BEATs and XLS-R."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ONE = {"audio": None}
TWO = {"audio": None, "speech": None}


def test_a_bare_path_feeds_a_single_frontend(train):
    assert train.parse_weights("/w/beats", ONE) == {"audio": "/w/beats"}


def test_absent_means_the_config_s_own_values(train):
    assert train.parse_weights(None, TWO) == {}


def test_named_paths_feed_each_frontend(train):
    got = train.parse_weights("audio=/w/beats, speech=/w/xlsr-300m", TWO)
    assert got == {"audio": "/w/beats", "speech": "/w/xlsr-300m"}


def test_a_partial_mapping_leaves_the_rest_alone(train):
    assert train.parse_weights("speech=/w/x", TWO) == {"speech": "/w/x"}


@pytest.mark.parametrize("arg, match", [
    ("/w/beats", "one path but the model has 2"),     # BEATs dir handed to XLS-R
    ("vision=/w/x", "frontend 'vision'"),
    ("audio=/a,audio=/b", "twice"),
    ("audio", "one path"),
    ("audio=", "cannot parse"),
])
def test_ambiguous_or_wrong_names_are_refused(train, arg, match):
    with pytest.raises(SystemExit, match=match):
        train.parse_weights(arg, TWO)


def test_build_model_threads_each_path_to_its_frontend(train, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, cfg):
            seen.update({k: v.weights for k, v in cfg.frontends.items()})

    monkeypatch.setattr(train, "DeepVoiceNet", Fake)
    train.build_model(str(REPO / "configs" / "c_first_run.yaml"), "audio=/a,speech=/s")
    assert seen == {"audio": "/a", "speech": "/s"}
