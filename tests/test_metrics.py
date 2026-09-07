"""Known-answer tests for the official metric. docs/validation/02 section 8.

No fixtures and no data required -- correctness here is establishable in full
before any corpus exists, which is why this module is built first.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from metrics.dacon import (
    PREDICTION_COLUMNS,
    EmptyPoolError,
    auc,
    dacon_score,
    eer,
    file_fake_label,
    roll_up,
)


def make_frame(n=400, seed=0, perfect=False, inverted=False, constant=None):
    """A frame covering all four presence combinations, with tunable predictions."""
    rng = np.random.default_rng(seed)
    voice_present = rng.integers(0, 2, n)
    music_present = rng.integers(0, 2, n)
    # guarantee both masked pools are populated and two-class
    voice_present[:4] = [1, 1, 1, 1]
    music_present[:4] = [1, 1, 1, 1]
    voice_fake = rng.integers(0, 2, n)
    music_fake = rng.integers(0, 2, n)
    voice_fake[:4] = [0, 1, 0, 1]
    music_fake[:4] = [0, 0, 1, 1]
    file_fake = file_fake_label(voice_present, music_present, voice_fake, music_fake)

    df = pd.DataFrame(dict(
        voice_present=voice_present, music_present=music_present,
        voice_fake=voice_fake, music_fake=music_fake, file_fake=file_fake,
    ))
    truth = dict(
        FILE_FAKE_PROB=file_fake, MUSIC_FAKE_PROB=music_fake,
        VOICE_FAKE_PROB=voice_fake, VOICE_PRESENT_PROB=voice_present,
        MUSIC_PRESENT_PROB=music_present,
    )
    for col in PREDICTION_COLUMNS:
        if constant is not None:
            df[col] = float(constant)
        elif perfect:
            df[col] = truth[col].astype(float)
        elif inverted:
            df[col] = 1.0 - truth[col].astype(float)
        else:  # informative but imperfect
            df[col] = np.clip(truth[col] + rng.normal(0, 0.6, n), 0, 1)
    return df


# --- T1 -----------------------------------------------------------------
@pytest.mark.parametrize("c", [0.0, 0.123, 0.5, 1.0])
def test_t1_constant_column_is_neutral(c):
    y = np.random.default_rng(0).integers(0, 2, 1200)
    s = np.full(1200, c)
    assert eer(y, s) == 0.5
    assert auc(y, s) == 0.5


# --- T2 -----------------------------------------------------------------
@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_t2_all_constant_scores_exactly_half(c):
    assert dacon_score(make_frame(constant=c)).score == 0.5


# --- T3 / T4 ------------------------------------------------------------
def test_t3_perfect_separation():
    m = dacon_score(make_frame(perfect=True))
    assert (m.eer_file, m.eer_voice, m.eer_music) == (0.0, 0.0, 0.0)
    assert (m.auc_vp, m.auc_mp) == (1.0, 1.0)
    assert m.score == 1.0


def test_t4_perfectly_inverted():
    m = dacon_score(make_frame(inverted=True))
    assert (m.eer_file, m.eer_voice, m.eer_music) == (1.0, 1.0, 1.0)
    assert (m.auc_vp, m.auc_mp) == (0.0, 0.0)
    assert m.score == 0.0


# --- T5 -----------------------------------------------------------------
@pytest.mark.parametrize("name,fn", [
    ("cube", lambda x: x ** 3),
    ("affine", lambda x: 3 * x + 7),
    ("logit", lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / (1 - np.clip(x, 1e-9, 1 - 1e-9)))),
])
def test_t5_monotone_transforms_are_free(name, fn):
    df = make_frame(seed=1)
    base = dacon_score(df).score
    for col in PREDICTION_COLUMNS:
        df[col] = fn(df[col].to_numpy())
    assert dacon_score(df).score == pytest.approx(base, abs=1e-12)


# --- T6 -----------------------------------------------------------------
def test_t6_masked_pools_ignore_absent_components():
    df = make_frame(seed=2)
    base = dacon_score(df).score
    rng = np.random.default_rng(9)
    absent_v = ~df.voice_present.astype(bool)
    absent_m = ~df.music_present.astype(bool)
    df.loc[absent_v, "VOICE_FAKE_PROB"] = rng.random(int(absent_v.sum()))
    df.loc[absent_m, "MUSIC_FAKE_PROB"] = rng.random(int(absent_m.sum()))
    assert dacon_score(df).score == pytest.approx(base, abs=1e-12)


# --- T7 -----------------------------------------------------------------
def test_t7_row_order_is_irrelevant():
    df = make_frame(seed=3)
    base = dacon_score(df).score
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert dacon_score(shuffled).score == pytest.approx(base, abs=1e-12)


# --- T8 -----------------------------------------------------------------
def test_t8_hand_computed_fixture():
    """Five rows, EER worked out by hand.

    scores 0.9 0.8 0.4 0.3 0.1 with labels 1 1 0 1 0.
    The ROC grid (drop_intermediate=False) walks thresholds high to low; the
    vertex minimising |fpr - fnr| is fpr=1/2, fnr=1/3, giving EER = 5/12.
    """
    y = np.array([1, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    assert (fpr[i], fnr[i]) == (0.5, pytest.approx(1 / 3))
    assert eer(y, s) == pytest.approx(5 / 12)


# --- T9 -----------------------------------------------------------------
def test_t9_drop_intermediate_default_would_differ():
    """Guards against `drop_intermediate=False` being 'cleaned up' away."""
    rng = np.random.default_rng(4)
    y = np.r_[np.ones(300), np.zeros(300)].astype(int)
    s = np.r_[rng.normal(1.0, 1, 300), rng.normal(0, 1, 300)]

    def eer_pruned(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=True)
        fnr = 1 - tpr
        i = np.argmin(np.abs(fpr - fnr))
        return (fpr[i] + fnr[i]) / 2

    assert eer_pruned(y, s) != eer(y, s)


# --- T10 ----------------------------------------------------------------
def test_t10_leaderboard_decomposition_is_exact():
    """4 submissions recover 3 EERs + CPS. docs/validation/05."""
    eF, eV, eM, aV, aM = 0.132, 0.181, 0.294, 0.973, 0.961
    s = lambda *a: roll_up(*a)[2]                                   # noqa: E731
    r_f = 0.5 - (s(eF, .5, .5, .5, .5) - 0.5) / 0.45
    r_v = 0.5 - (s(.5, eV, .5, .5, .5) - 0.5) / 0.18
    r_m = 0.5 - (s(.5, .5, eM, .5, .5) - 0.5) / 0.27
    ads = roll_up(r_f, r_v, r_m, 0.5, 0.5)[0]
    r_cps = (s(eF, eV, eM, aV, aM) - 0.9 * ads) / 0.1

    assert r_f == pytest.approx(eF, abs=1e-12)
    assert r_v == pytest.approx(eV, abs=1e-12)
    assert r_m == pytest.approx(eM, abs=1e-12)
    assert r_cps == pytest.approx(0.5 * (aV + aM), abs=1e-12)


# --- T13 / T14 ----------------------------------------------------------
def test_t13_empty_masked_pool_raises():
    """An empty pool must not silently return 0.5 and average into the mean."""
    df = make_frame(seed=5)
    df["music_present"] = 0
    with pytest.raises(EmptyPoolError):
        dacon_score(df)


def test_t14_single_class_pool_raises():
    df = make_frame(seed=6)
    df["voice_fake"] = 1                       # voice pool now all-positive
    with pytest.raises(EmptyPoolError):
        dacon_score(df)


def test_file_fake_ignores_absent_components():
    # voice absent + voice_fake set must not make the file fake
    assert file_fake_label([0], [1], [1], [0]).tolist() == [0]
    assert file_fake_label([1], [0], [1], [0]).tolist() == [1]
    assert file_fake_label([1], [1], [0], [1]).tolist() == [1]
    assert file_fake_label([1], [1], [0], [0]).tolist() == [0]


# --- submission writer --------------------------------------------------
def test_write_submission_round_trips(tmp_path):
    from metrics.submission import validate_submission, write_submission

    rng = np.random.default_rng(0)
    ids = [f"TEST_{i:04d}" for i in range(1200)]
    preds = {c: rng.random(1200) for c in PREDICTION_COLUMNS}
    write_submission(ids, preds, tmp_path / "submission.csv")
    back = validate_submission(tmp_path / "submission.csv", reference_ids=ids)
    assert len(back) == 1200
    assert np.allclose(back["FILE_FAKE_PROB"], preds["FILE_FAKE_PROB"])


def test_write_submission_preserves_full_float64_precision(tmp_path):
    from metrics.submission import write_submission

    ids = [f"TEST_{i:04d}" for i in range(1200)]
    base = np.linspace(0.1, 0.9, 1200) + 1e-13 * np.arange(1200)
    preds = {c: base for c in PREDICTION_COLUMNS}
    write_submission(ids, preds, tmp_path / "s.csv")
    back = pd.read_csv(tmp_path / "s.csv")
    assert len(np.unique(back["FILE_FAKE_PROB"].to_numpy())) == 1200


@pytest.mark.parametrize("saturate", [0.6, 0.8])
def test_vg5_rejects_saturated_columns(tmp_path, saturate):
    from metrics.submission import VG5Error, write_submission

    n = 1000
    ids = [f"TEST_{i:04d}" for i in range(n)]
    v = np.linspace(0, 1, n)
    k = int(saturate * n / 2)
    v[:k] = 0.0
    v[-k:] = 1.0
    preds = {c: v for c in PREDICTION_COLUMNS}
    with pytest.raises(VG5Error, match="Ranking resolution"):
        write_submission(ids, preds, tmp_path / "s.csv")


def test_vg5_rejects_constant_and_nonfinite(tmp_path):
    from metrics.submission import VG5Error, write_submission

    ids = [f"TEST_{i:04d}" for i in range(100)]
    ok = np.linspace(0, 1, 100)

    preds = {c: ok for c in PREDICTION_COLUMNS}
    preds["FILE_FAKE_PROB"] = np.full(100, 0.5)
    with pytest.raises(VG5Error, match="constant"):
        write_submission(ids, preds, tmp_path / "a.csv")

    preds = {c: ok.copy() for c in PREDICTION_COLUMNS}
    preds["VOICE_FAKE_PROB"][3] = np.nan
    with pytest.raises(VG5Error, match="non-finite"):
        write_submission(ids, preds, tmp_path / "b.csv")


def test_validate_submission_catches_id_mismatch(tmp_path):
    from metrics.submission import VG5Error, validate_submission, write_submission

    ids = [f"TEST_{i:04d}" for i in range(50)]
    preds = {c: np.linspace(0, 1, 50) for c in PREDICTION_COLUMNS}
    write_submission(ids, preds, tmp_path / "s.csv")
    with pytest.raises(VG5Error, match="rows|ID set"):
        validate_submission(tmp_path / "s.csv", reference_ids=ids[:49])


# --- submission contract regressions ------------------------------------
def test_prediction_column_order_matches_the_competition_listing():
    """docs/competition/01-overview.md lists FILE, VOICE, MUSIC, then presence.

    Regression guard: the order was briefly taken from a metric-weight-ordered
    table (FILE, MUSIC, VOICE), which would have written a mis-ordered file.
    """
    assert PREDICTION_COLUMNS == (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    )


def test_column_order_is_taken_from_the_reference_not_assumed(tmp_path):
    from metrics.submission import read_sample_submission, validate_submission, write_submission

    # a sample_submission whose order differs from our default
    odd = ["ID", "MUSIC_PRESENT_PROB", "FILE_FAKE_PROB", "VOICE_PRESENT_PROB",
           "MUSIC_FAKE_PROB", "VOICE_FAKE_PROB"]
    ref = tmp_path / "sample_submission.csv"
    pd.DataFrame({c: ([f"TEST_{i:04d}" for i in range(3)] if c == "ID" else [0.5] * 3)
                  for c in odd})[odd].to_csv(ref, index=False)

    ids, cols = read_sample_submission(ref)
    assert list(cols) == odd[1:]

    preds = {c: np.linspace(0, 1, 3) for c in PREDICTION_COLUMNS}
    write_submission(ids, preds, tmp_path / "submission.csv", columns=cols)
    back = validate_submission(tmp_path / "submission.csv",
                               reference_ids=ids, reference_columns=cols,
                               strict_sanity=False)
    assert list(back.columns) == odd


def test_csv_round_trip_is_bit_exact(tmp_path):
    """pandas' default reader is not correctly rounded; the validator must not
    silently compare against a re-parsed approximation."""
    from metrics.submission import validate_submission, write_submission

    rng = np.random.default_rng(0)
    ids = [f"TEST_{i:04d}" for i in range(1200)]
    v = rng.random(1200)
    v[:3] = [1 / 3, np.nextafter(0.5, 1), 0.1 + 0.2]
    write_submission(ids, {c: v for c in PREDICTION_COLUMNS}, tmp_path / "s.csv")

    naive = pd.read_csv(tmp_path / "s.csv")["FILE_FAKE_PROB"].to_numpy()
    exact = validate_submission(tmp_path / "s.csv")["FILE_FAKE_PROB"].to_numpy()
    assert (exact == v).all(), "round_trip read must be bit-exact"
    assert not (naive == v).all(), "default reader is expected to drift; guard is meaningful"


def test_read_sample_submission_rejects_wrong_columns(tmp_path):
    from metrics.submission import VG5Error, read_sample_submission

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ID": ["a"], "FILE_FAKE_PROB": [0.5], "NOPE": [0.5]}).to_csv(bad, index=False)
    with pytest.raises(VG5Error, match="do not match"):
        read_sample_submission(bad)
