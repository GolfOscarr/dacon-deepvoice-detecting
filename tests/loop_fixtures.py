"""Fixtures and helpers shared by the four `training.loop` test modules.

One copy, deliberately. `tests/test_loop.py` was split along the module seam
(`stages` / `checkpoint` / `loop` / `validate`) and every one of the four needs
the same synthetic corpus, the same stub config and the same weight-comparison
helpers. Duplicating them across four files is how fixture conventions drift,
and this repo has already been bitten by a padding fixture that was wrong in
every file at once -- `lengths = SR * 4` everywhere, so no test could see it.

Not a `conftest.py`: these names are specific to the training-loop suite and a
root conftest would put `corpus` and `model_cfg` in front of every test module
in the repo. Import what you use instead; pytest picks up a fixture that a test
module imported just as it picks up one the module defined.
"""

import dataclasses
import subprocess

import pandas as pd
import pytest
import torch

from metrics.dacon import MetricSet
from models.config import load_model_config, load_train_config
from models.model import DeepVoiceNet
from training.audit import AuditReport
from training.dataset import SpecDataset
from training.render import ManifestIndex, RenderConfig
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus
from training.validate import FoldResult, ValidationReport

CORPUS = dict(n_per_pool=8, n_whole_file=8, seed=0, duration_range=(6.0, 8.0))
DRAW = SamplerConfig(duration_range=(4.0, 5.0))

HAS_FFMPEG = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("loop-corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return manifest, ManifestIndex.from_frame(manifest), RenderConfig(root=root)


@pytest.fixture(scope="module")
def model_cfg():
    return load_model_config("configs/a_stub.yaml")


def _train_cfg(**kw):
    base = load_train_config("configs/train_joint.yaml")
    return dataclasses.replace(base, **{"epochs": 1, "batch_size": 2, **kw})


def _dataset(corpus, n=4, seed=0):
    manifest, index, rcfg = corpus
    return SpecDataset.from_sampler(
        Sampler(manifest, DRAW), n, index, rcfg, seed=seed)


def _model(model_cfg):
    return DeepVoiceNet(model_cfg)


def _flat(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _max_abs_diff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def _unfrozen(cfg):
    return dataclasses.replace(cfg, frontends={
        k: dataclasses.replace(v, freeze=False) for k, v in cfg.frontends.items()})


def _metric_set(eer_file, eer_voice=0.2, eer_music=0.3, auc_vp=0.9, auc_mp=0.9,
                n=2000):
    from metrics.dacon import roll_up
    ads, cps, score = roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp)
    return MetricSet(eer_file, eer_voice, eer_music, auc_vp, auc_mp, ads, cps,
                     score, n, n, n)


def _breakdown(values):
    """A breakdown frame with the columns `metrics.breakdown.by` actually emits."""
    return pd.DataFrame([
        {"value": v, "head": "file", "eer": 0.1 + 0.01 * i, "contrast": "shared",
         "n_slice": 300, "n_slice_fake": 150, "n_pool": 900, "thin": False,
         "note": ""}
        for i, v in enumerate(values)])


def _fold_result(fold, metrics, gates=None, tripwires=None, n=200):
    report = ValidationReport(fold, metrics, _breakdown(range(1, 10)),
                              _breakdown(["hifigan", "suno_v3"]),
                              pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]}))
    green = AuditReport({"x": (True, "fine")})
    return FoldResult(fold, report, gates or green, tripwires or green)
