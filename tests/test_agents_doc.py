"""Execute every Python snippet in metrics/AGENTS.md and models/AGENTS.md.

A usage guide that has drifted from the code is worse than no guide, and these
snippets are the first thing anyone copies. Running them is cheap; leaving them
unverified is how the column-order defect would have survived.
"""

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from metrics.dacon import PREDICTION_COLUMNS

DOC = pathlib.Path(__file__).resolve().parents[1] / "metrics" / "AGENTS.md"

CELLS = {1: (1, 0, 0, 0), 2: (1, 0, 1, 0), 3: (0, 1, 0, 0), 4: (0, 1, 0, 1),
         5: (1, 1, 0, 0), 6: (1, 1, 0, 1), 7: (1, 1, 1, 0), 8: (1, 1, 1, 1)}


def _frame(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    cell = rng.choice(list(CELLS), n)
    vp, mp, vf, mf = (np.array([CELLS[c][i] for c in cell]) for i in range(4))
    ff = ((vp & vf) | (mp & mf)).astype(int)
    sig = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 2.0 * ff)))
    fam = np.where(ff == 1, rng.choice(["hifigan", "encodec", "diffusion"], n),
                   rng.choice(["libritts", "musdb"], n))
    return pd.DataFrame({
        "file_id": [f"f{i}" for i in range(n)], "cell": cell,
        "voice_present": vp, "music_present": mp,
        "voice_fake": vf, "music_fake": mf, "file_fake": ff,
        "FILE_FAKE_PROB": sig, "VOICE_FAKE_PROB": sig, "MUSIC_FAKE_PROB": sig,
        "VOICE_PRESENT_PROB": 1 / (1 + np.exp(-(3 * vp - 1.5 + rng.normal(0, .3, n)))),
        "MUSIC_PRESENT_PROB": 1 / (1 + np.exp(-(3 * mp - 1.5 + rng.normal(0, .3, n)))),
        "artifact_family": fam, "fold": rng.integers(0, 5, n),
        "pair_id": np.where(rng.random(n) < 0.3,
                            "p" + rng.integers(0, 200, n).astype(str), None),
    })


def _blocks():
    return re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)


def test_the_doc_actually_has_snippets():
    assert len(_blocks()) >= 8


@pytest.mark.parametrize("index", range(len(_blocks())))
def test_agents_md_snippet_runs(index, tmp_path):
    df = _frame()
    ids = df["file_id"].tolist()
    pd.DataFrame({"ID": ids, **{c: 0.0 for c in PREDICTION_COLUMNS}}).to_csv(
        tmp_path / "sample_submission.csv", index=False)

    env = {
        "np": np, "pd": pd, "df": df, "n": len(df),
        "fold_frames": [_frame(seed=k + 1) for k in range(5)],
        "incumbent_df": df,
        "candidate_df": df.assign(FILE_FAKE_PROB=np.clip(
            df.FILE_FAKE_PROB + 0.05 * df.file_fake, 0, 1)),
        "preds": {c: df[c].to_numpy() for c in PREDICTION_COLUMNS},
        "lb_file_probe": 0.62, "lb_voice_probe": 0.55,
        "lb_music_probe": 0.57, "lb_full_run": 0.75,
    }
    code = (_blocks()[index]
            .replace('"data/sample_submission.csv"', f'r"{tmp_path / "sample_submission.csv"}"')
            .replace('"output/submission.csv"', f'r"{tmp_path / "submission.csv"}"'))
    exec(compile(code, f"AGENTS.md[block {index + 1}]", "exec"), env)


# --------------------------------------------------------------------------- #
# models/AGENTS.md

MODELS_DOC = pathlib.Path(__file__).resolve().parents[1] / "models" / "AGENTS.md"


def _model_blocks():
    return re.findall(r"```python\n(.*?)```", MODELS_DOC.read_text(encoding="utf-8"), re.S)


def test_models_doc_has_snippets():
    assert len(_model_blocks()) >= 5


@pytest.mark.parametrize("index", range(len(_model_blocks())))
def test_models_agents_md_snippet_runs(index, tmp_path):
    """The guide is the first thing anyone copies, so it must actually run."""
    import dataclasses

    import torch

    from models.config import load_model_config
    from models.model import DeepVoiceNet

    cfg = load_model_config("configs/b_stub.yaml")
    model = DeepVoiceNet(cfg).eval()
    wav = torch.randn(2, 16_000 * 5)
    out = model(wav, torch.tensor([16_000 * 5, 16_000 * 3]))

    env = {"cfg": cfg, "model": model, "out": out, "torch": torch}
    (tmp_path / "model").mkdir(exist_ok=True)
    code = (_model_blocks()[index]
            .replace('"model/model.pt"', f'r"{tmp_path / "model" / "model.pt"}"'))
    exec(compile(code, f"models/AGENTS.md[block {index + 1}]", "exec"), env)
