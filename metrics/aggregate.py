"""Fold aggregation and confidence intervals. docs/validation/02 sections 4 and 6.

The two mistakes this module exists to prevent:

1. Pooling raw out-of-fold scores. Each fold is scored by a *different* model,
   EER is computed on the merged ranking, so concatenation measures the score
   scale drift between folds rather than the models. Measured: 0.1705 against
   a true 0.100, where the mean of per-fold EERs gives 0.1040. The magnitude
   depends on how far the scales drift, but the direction does not -- drift is
   independent of the label, so pooling can only add apparent error.

2. Bootstrapping individual files. Forty clips from one Jamendo artist are not
   forty independent observations. The resampling unit is the group column,
   normally `artifact_family`.

Score is linear in its five components, so mean(Score per fold) equals
Score(mean of components). That convenience does *not* extend to the EERs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics.dacon import MetricSet, roll_up

__all__ = ["AggregateMetrics", "fold_mean", "pooled_oof_scores", "group_bootstrap_ci",
           "paired_delta_ci"]

_METRIC_FIELDS = ("eer_file", "eer_voice", "eer_music", "auc_vp", "auc_mp")


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean and sd across folds, plus which folds contributed."""

    mean: MetricSet
    sd: dict[str, float]
    n_folds: int
    excluded_folds: tuple[int, ...]

    @property
    def score_sd(self) -> float:
        return self.sd["score"]


def fold_mean(per_fold, min_pool=None, fold_ids=None) -> AggregateMetrics:
    """Mean of per-fold MetricSets. Never concatenates scores.

    `min_pool` drops any fold whose masked pool falls below the size floor
    (docs/validation/01 section 4): averaging in a fold with 200 music-present
    files adds noise, not information. Exclusions are recorded, not silent.
    """
    per_fold = list(per_fold)
    if not per_fold:
        raise ValueError("fold_mean: no folds given")
    ids = list(fold_ids) if fold_ids is not None else list(range(len(per_fold)))

    kept, excluded = [], []
    for fid, m in zip(ids, per_fold):
        too_small = min_pool is not None and min(m.n_voice, m.n_music, m.n_file) < min_pool
        (excluded if too_small else kept).append((fid, m))
    if not kept:
        raise ValueError(f"fold_mean: every fold fell below min_pool={min_pool}")

    mean_vals = {f: float(np.mean([getattr(m, f) for _, m in kept])) for f in _METRIC_FIELDS}
    ads, cps, score = roll_up(**mean_vals)
    mean = MetricSet(
        **mean_vals, ads=ads, cps=cps, score=score,
        n_file=int(sum(m.n_file for _, m in kept)),
        n_voice=int(sum(m.n_voice for _, m in kept)),
        n_music=int(sum(m.n_music for _, m in kept)),
    )

    def _sd(values):
        return float(statistics.stdev(values)) if len(values) > 1 else 0.0

    sd = {f: _sd([getattr(m, f) for _, m in kept]) for f in _METRIC_FIELDS}
    sd["score"] = _sd([m.score for _, m in kept])
    return AggregateMetrics(mean, sd, len(kept), tuple(fid for fid, _ in excluded))


def pooled_oof_scores(fold_frames, score_col):
    """Concatenate OOF predictions after per-fold rank normalisation.

    For diagnostics that genuinely need one ranked list -- error analysis,
    score histograms. Never for the headline number: use `fold_mean`.

    Rank normalisation is a cross-file statistic. Legal here because this runs
    locally over our own validation folds; forbidden inside script.py by rule
    2.4.
    """
    out = []
    for df in fold_frames:
        v = df[score_col].to_numpy(dtype=np.float64)
        ranks = (np.argsort(np.argsort(v)) + 0.5) / len(v)
        out.append(df.assign(**{score_col: ranks}))
    return pd.concat(out, ignore_index=True)


def _resample_groups(df, group_col, rng):
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    return df.iloc[np.concatenate([idx[g] for g in picked])]


def group_bootstrap_ci(df, metric_fn, group_col="artifact_family", n=10_000,
                       alpha=0.05, seed=0):
    """Percentile CI for one metric, resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        try:
            vals.append(metric_fn(_resample_groups(df, group_col, rng)))
        except Exception:
            continue                      # a resample can leave a pool single-class

    if len(vals) < 0.5 * n:
        raise ValueError(
            f"group_bootstrap_ci: only {len(vals)}/{n} resamples were scorable; "
            "the pool is too small or too group-imbalanced for a reliable interval"
        )
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta_ci(df_a, df_b, metric_fn, group_col="artifact_family", n=10_000,
                    alpha=0.05, seed=0, key="file_id"):
    """CI for (B - A), taking the difference *inside* each resample.

    The pairing is the entire reason a 1-point gain is detectable: at our VAL
    size, unpaired comparison resolves it 85% of the time and paired 95%
    (docs/validation/02 section 7). Comparing two independently-computed
    marginal CIs discards that and would have us throw real gains away.

    Both frames must cover the same files, so the same resample applies to each.
    """
    if list(df_a[key]) != list(df_b[key]):
        raise ValueError("paired_delta_ci: frames must carry the same files in the same order")

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        groups = df_a[group_col].to_numpy()
        uniq = np.unique(groups)
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = {g: np.flatnonzero(groups == g) for g in uniq}
        rows = np.concatenate([idx[g] for g in picked])
        try:
            deltas.append(metric_fn(df_b.iloc[rows]) - metric_fn(df_a.iloc[rows]))
        except Exception:
            continue

    if len(deltas) < 0.5 * n:
        raise ValueError(
            f"paired_delta_ci: only {len(deltas)}/{n} resamples were scorable"
        )
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(deltas)), float(lo), float(hi)
