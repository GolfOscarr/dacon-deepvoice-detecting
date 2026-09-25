"""An optional per-file stacker for FILE_FAKE_PROB (docs/training/10 F6).

The learned file head under-uses the music head: on first-v3 it missed 0.36 of
fold 0's cell-6 files (real voice + fake music) that the music head ranked
well. A logistic combination of the five outputs' logits, fitted on the folds'
VAL predictions, took mean file EER 0.064 -> 0.053 leave-one-fold-out.

Per file only (rule 2.4: no cross-file statistics). The presence logits are
deliberately NOT features on their own: alone they encode our draw's cell prior
(which cells are fake), which the test need not share.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FEATURES = ("lg_file", "lg_voice", "lg_music", "lg_voice_x_vp", "lg_music_x_mp")
_EPS = 1e-9


def _lg(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def features(probs: Mapping[str, Sequence[float]]) -> np.ndarray:
    f, v, m = (_lg(probs[c]) for c in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB"))
    vp = np.asarray(probs["VOICE_PRESENT_PROB"], dtype=np.float64)
    mp = np.asarray(probs["MUSIC_PRESENT_PROB"], dtype=np.float64)
    return np.stack([f, v, m, v * vp, m * mp], axis=1)


def _softsign_prob(z: np.ndarray) -> np.ndarray:
    # the model's own squash: no float64 saturation, so no ties at the extremes
    return np.clip(0.5 + 0.5 * z / (1.0 + np.abs(z)), 1e-12, 1 - 1e-12)


def apply(probs: Mapping[str, Sequence[float]], stack: Mapping) -> np.ndarray:
    if tuple(stack["features"]) != FEATURES:
        raise ValueError(f"file_stack features {stack['features']} != {FEATURES}")
    z = features(probs) @ np.asarray(stack["coef"], dtype=np.float64) + float(stack["bias"])
    return _softsign_prob(z)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit(frame, C: float = 1.0) -> dict:
    """Fit on a frame with the five prediction columns and `file_fake`."""
    from sklearn.linear_model import LogisticRegression
    X = features({c: frame[c].to_numpy() for c in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")})
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, frame["file_fake"].to_numpy())
    return {"features": list(FEATURES), "coef": [float(c) for c in clf.coef_[0]],
            "bias": float(clf.intercept_[0]), "C": C, "n": int(len(frame))}
