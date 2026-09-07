"""The official DACON 236749 metric.

The single place EER is computed. Transcribed from the competition's published
scorer; see docs/validation/02-metric-harness.md for why each detail matters.

    Score = 0.9 * ADS + 0.1 * CPS
    ADS   = 0.5*(1-EER_file) + 0.2*(1-EER_voice) + 0.3*(1-EER_music)
    CPS   = 0.5*AUC_voice_present + 0.5*AUC_music_present

FAKE is the positive class (1). Voice EER is computed only over voice-present
samples and Music EER only over music-present ones, using the *ground-truth*
presence labels -- our presence predictions cannot corrupt the fake EERs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = [
    "MetricSet",
    "PREDICTION_COLUMNS",
    "auc",
    "dacon_score",
    "eer",
    "file_fake_label",
    "roll_up",
]

#: The five submission columns, in the order the competition lists them
#: (docs/competition/01-overview.md). We have not seen the real
#: sample_submission.csv, so this is a documented default, not ground truth --
#: metrics.submission reads the actual order from the reference file when one
#: is available and never relies on this tuple's ordering.
PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)

#: Weight of each head in the total Score (0.9*ADS + 0.1*CPS expanded).
SCORE_WEIGHTS = {
    "eer_file": 0.45,
    "eer_music": 0.27,
    "eer_voice": 0.18,
    "auc_vp": 0.05,
    "auc_mp": 0.05,
}


class EmptyPoolError(ValueError):
    """A masked pool was empty or single-class, so its metric is undefined.

    Raised rather than returning 0.5, which is a plausible-looking wrong answer
    that would average silently into the headline number.
    """


@dataclass(frozen=True)
class MetricSet:
    """The eight Tier-1 numbers, plus the pool sizes they were computed over.

    Carrying `n_*` means a downstream reader can always see what a metric was
    measured on -- an EER over 180 music-present files is not the same evidence
    as one over 2,400 (docs/validation/01-split-scheme.md#4-size-floors).
    """

    eer_file: float
    eer_voice: float
    eer_music: float
    auc_vp: float
    auc_mp: float
    ads: float
    cps: float
    score: float
    n_file: int
    n_voice: int
    n_music: int

    def as_dict(self) -> dict:
        return asdict(self)


def _check_pool(y_true: np.ndarray, name: str) -> np.ndarray:
    y = np.asarray(y_true)
    if y.size == 0:
        raise EmptyPoolError(f"{name}: pool is empty")
    uniq = np.unique(y)
    if uniq.size < 2:
        raise EmptyPoolError(
            f"{name}: pool has a single class ({uniq.tolist()}); metric undefined"
        )
    if not np.isin(uniq, (0, 1)).all():
        raise ValueError(f"{name}: labels must be 0/1, got {uniq.tolist()}")
    return y


def _check_scores(y_score: np.ndarray, name: str) -> np.ndarray:
    s = np.asarray(y_score, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError(f"{name}: scores contain NaN or inf")
    return s


def eer(y_true, y_score) -> float:
    """Equal Error Rate, exactly as the official scorer computes it.

    Three details are load-bearing and must not be "cleaned up":

    * ``drop_intermediate=False`` -- the default prunes ROC vertices, which
      moves where the argmin lands.
    * ``pos_label=1`` with FAKE encoded as 1 -- inverting the class inverts
      the ordering.
    * ``argmin`` over ``|fpr - fnr|``, *not* interpolation to the fpr == fnr
      crossing. Textbook implementations interpolate and disagree slightly.
    """
    y = _check_pool(y_true, "eer")
    s = _check_scores(y_score, "eer")
    if y.shape != s.shape:
        raise ValueError(f"eer: shape mismatch {y.shape} vs {s.shape}")

    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def auc(y_true, y_score) -> float:
    """ROC-AUC, used for the two presence heads."""
    y = _check_pool(y_true, "auc")
    s = _check_scores(y_score, "auc")
    if y.shape != s.shape:
        raise ValueError(f"auc: shape mismatch {y.shape} vs {s.shape}")
    return float(roc_auc_score(y, s))


def roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp):
    """Combine the five head metrics into (ADS, CPS, Score).

    Score is *linear* in these five arguments, which is why the mean of
    per-fold Scores equals the Score of the per-fold means. That convenience
    does not extend to the EERs themselves -- see metrics.aggregate.
    """
    ads = 0.5 * (1 - eer_file) + 0.2 * (1 - eer_voice) + 0.3 * (1 - eer_music)
    cps = 0.5 * auc_vp + 0.5 * auc_mp
    return float(ads), float(cps), float(0.9 * ads + 0.1 * cps)


def file_fake_label(voice_present, music_present, voice_fake, music_fake):
    """Ground-truth FILE_FAKE: voice_fake OR music_fake over *present* components.

    A component's fake label is ignored where that component is absent, which is
    what makes cells 1-4 well defined (docs/data/02-label-taxonomy.md).
    """
    vp = np.asarray(voice_present).astype(bool)
    mp = np.asarray(music_present).astype(bool)
    return ((vp & np.asarray(voice_fake).astype(bool))
            | (mp & np.asarray(music_fake).astype(bool))).astype(int)


def dacon_score(df) -> MetricSet:
    """Score a frame carrying ground truth and the five prediction columns.

    Required columns:
        voice_present, music_present, voice_fake, music_fake, file_fake
        plus the five names in PREDICTION_COLUMNS.
    """
    required = (
        "voice_present", "music_present", "voice_fake", "music_fake", "file_fake",
        *PREDICTION_COLUMNS,
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"dacon_score: missing columns {missing}")

    v = np.asarray(df["voice_present"]).astype(bool)
    m = np.asarray(df["music_present"]).astype(bool)

    eer_file = eer(df["file_fake"].to_numpy(), df["FILE_FAKE_PROB"].to_numpy())
    eer_voice = eer(df["voice_fake"].to_numpy()[v], df["VOICE_FAKE_PROB"].to_numpy()[v])
    eer_music = eer(df["music_fake"].to_numpy()[m], df["MUSIC_FAKE_PROB"].to_numpy()[m])

    auc_vp = auc(df["voice_present"].to_numpy(), df["VOICE_PRESENT_PROB"].to_numpy())
    auc_mp = auc(df["music_present"].to_numpy(), df["MUSIC_PRESENT_PROB"].to_numpy())

    ads, cps, score = roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp)
    return MetricSet(
        eer_file=eer_file, eer_voice=eer_voice, eer_music=eer_music,
        auc_vp=auc_vp, auc_mp=auc_mp, ads=ads, cps=cps, score=score,
        n_file=int(len(df)), n_voice=int(v.sum()), n_music=int(m.sum()),
    )
