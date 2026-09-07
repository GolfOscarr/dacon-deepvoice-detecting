"""Fold aggregation and bootstrap tests. T11, T12 from docs/validation/02 section 8."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from metrics.aggregate import fold_mean, group_bootstrap_ci, paired_delta_ci, pooled_oof_scores
from metrics.dacon import MetricSet, eer, roll_up


def make_metricset(eer_file, eer_voice, eer_music, auc_vp, auc_mp, n=2000):
    ads, cps, score = roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp)
    return MetricSet(eer_file, eer_voice, eer_music, auc_vp, auc_mp, ads, cps, score, n, n, n)


# --- T11 ----------------------------------------------------------------
def test_t11_pooling_raw_oof_scores_is_wrong():
    """Per-fold models have different score scales; concatenating measures the drift."""
    rng = np.random.default_rng(11)
    d = 2 * norm.isf(0.10)
    fold_eers, ys, raw, frames = [], [], [], []
    for k in range(5):
        y = np.r_[np.ones(400), np.zeros(400)].astype(int)
        s = np.r_[rng.normal(d, 1, 400), rng.normal(0, 1, 400)]
        a, b = rng.uniform(0.5, 2.0), rng.uniform(-2, 2)   # harmless monotone rescale
        obs = a * s + b
        fold_eers.append(eer(y, obs))
        ys.append(y)
        raw.append(obs)
        frames.append(pd.DataFrame({"y": y, "score": obs, "fold": k}))

    y_all = np.concatenate(ys)
    mean_of_folds = float(np.mean(fold_eers))
    pooled_raw = eer(y_all, np.concatenate(raw))
    pooled_rank = eer(pooled_oof_scores(frames, "score")["y"].to_numpy(),
                      pooled_oof_scores(frames, "score")["score"].to_numpy())

    assert mean_of_folds == pytest.approx(0.10, abs=0.02)
    assert pooled_raw > mean_of_folds + 0.05          # the trap
    assert pooled_rank == pytest.approx(mean_of_folds, abs=0.02)


def test_fold_mean_matches_score_linearity():
    """mean(Score per fold) == Score(mean of components)."""
    rng = np.random.default_rng(5)
    sets = [make_metricset(*rng.uniform(0.05, 0.35, 5)) for _ in range(5)]
    agg = fold_mean(sets)
    assert agg.mean.score == pytest.approx(float(np.mean([s.score for s in sets])), abs=1e-12)
    assert agg.n_folds == 5
    assert agg.score_sd > 0


def test_fold_mean_excludes_undersized_folds_and_records_it():
    big = make_metricset(0.10, 0.10, 0.10, 0.99, 0.99, n=2000)
    small = make_metricset(0.40, 0.40, 0.40, 0.60, 0.60, n=200)
    agg = fold_mean([big, big, small, big, big], min_pool=1200, fold_ids=[0, 1, 2, 3, 4])
    assert agg.excluded_folds == (2,)
    assert agg.n_folds == 4
    assert agg.mean.eer_file == pytest.approx(0.10)


def test_fold_mean_refuses_when_every_fold_is_too_small():
    small = make_metricset(0.2, 0.2, 0.2, 0.9, 0.9, n=100)
    with pytest.raises(ValueError, match="min_pool"):
        fold_mean([small, small], min_pool=1200)


# --- T12 ----------------------------------------------------------------
def test_t12_eer_is_invariant_to_class_prevalence():
    rng = np.random.default_rng(13)
    n = 4000
    d = 2 * norm.isf(0.10)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    s = np.r_[rng.normal(d, 1, n), rng.normal(0, 1, n)]
    base = eer(y, s)
    for keep in (0.5, 0.2):
        idx = np.r_[np.arange(n), n + rng.choice(n, int(n * keep), replace=False)]
        assert eer(y[idx], s[idx]) == pytest.approx(base, abs=0.02)


def test_eer_is_not_invariant_to_within_class_composition():
    """The invariance people over-read. Same class ratio, different cell mix."""
    rng = np.random.default_rng(13)
    neg = rng.normal(0, 1, 3000)
    easy = rng.normal(2 * norm.isf(0.03), 1, 3000)
    hard = rng.normal(2 * norm.isf(0.30), 1, 3000)
    y = np.r_[np.ones(3000), np.zeros(3000)].astype(int)
    out = []
    for f_hard in (0.0, 0.5, 1.0):
        k = int(3000 * f_hard)
        out.append(eer(y, np.r_[np.r_[hard[:k], easy[:3000 - k]], neg]))
    assert out[0] < 0.06 and out[-1] > 0.25          # a ~9x swing, model unchanged


# --- bootstrap ----------------------------------------------------------
def _frame(seed, shift=0.0, n_groups=25, per_group=80):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        y = rng.integers(0, 2, per_group)
        # a per-group offset is what makes file-level bootstrap over-confident
        s = rng.normal(1.2 + shift, 1, per_group) * y + rng.normal(0, 1, per_group) * (1 - y)
        s += rng.normal(0, 0.5)
        rows.append(pd.DataFrame({
            "file_id": [f"g{g}_f{i}" for i in range(per_group)],
            "artifact_family": f"fam{g}", "y": y, "score": s,
        }))
    return pd.concat(rows, ignore_index=True)


def _eer_of(df):
    return eer(df["y"].to_numpy(), df["score"].to_numpy())


def test_group_bootstrap_ci_brackets_the_point_estimate():
    df = _frame(1)
    lo, hi = group_bootstrap_ci(df, _eer_of, n=400, seed=0)
    assert lo < _eer_of(df) < hi


def test_group_bootstrap_is_wider_than_naive_file_bootstrap():
    """Grouping must not be silently optional -- it changes the interval."""
    df = _frame(2)
    lo_g, hi_g = group_bootstrap_ci(df, _eer_of, n=400, seed=0)
    df_files = df.assign(artifact_family=df["file_id"])       # every file its own group
    lo_f, hi_f = group_bootstrap_ci(df_files, _eer_of, n=400, seed=0)
    assert (hi_g - lo_g) > (hi_f - lo_f)


def test_paired_delta_ci_detects_a_real_gain():
    a = _frame(3, shift=0.0)
    b = a.copy()
    b["score"] = b["score"] + 0.35 * b["y"]                   # B genuinely better
    mean, lo, hi = paired_delta_ci(a, b, _eer_of, n=400, seed=0)
    assert mean < 0 and hi < 0                                 # EER dropped, CI excludes 0


def test_paired_delta_ci_requires_aligned_frames():
    a = _frame(4)
    with pytest.raises(ValueError, match="same files"):
        paired_delta_ci(a, a.iloc[::-1].reset_index(drop=True), _eer_of, n=10)
