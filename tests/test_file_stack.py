import numpy as np
import pandas as pd
import pytest

from processing import file_stack

COLS = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    d = {c: np.clip(rng.uniform(0, 1, n) * 0.5 + 0.5 * y * (c != "MUSIC_PRESENT_PROB"), 0.01, 0.99)
         for c in COLS}
    d["file_fake"] = y
    return pd.DataFrame(d)


def test_fit_apply_roundtrip_and_range():
    df = _frame()
    st = file_stack.fit(df)
    p = file_stack.apply({c: df[c].to_numpy() for c in COLS}, st)
    assert p.shape == (len(df),) and np.all((p > 0) & (p < 1))
    assert p[df.file_fake == 1].mean() > p[df.file_fake == 0].mean()


def test_per_file_only():
    # rule 2.4: a row's output must not depend on the other rows
    df = _frame()
    st = file_stack.fit(df)
    full = file_stack.apply({c: df[c].to_numpy() for c in COLS}, st)
    one = file_stack.apply({c: df[c].to_numpy()[7:8] for c in COLS}, st)
    assert one[0] == full[7]


def test_stale_features_rejected():
    st = file_stack.fit(_frame())
    st["features"] = ["lg_file"]
    with pytest.raises(ValueError):
        file_stack.apply({c: [0.5] for c in COLS}, st)


def test_no_saturation_ties():
    st = {"features": list(file_stack.FEATURES), "coef": [50.0, 0, 0, 0, 0], "bias": 0.0}
    p = file_stack.apply({c: [0.999999, 0.9999999] if c == "FILE_FAKE_PROB" else [0.5, 0.5]
                          for c in COLS}, st)
    assert p[1] > p[0]
