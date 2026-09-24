"""docs/training/07 D-d: language balance in the voice draw.

The built corpus had Korean on the real side only (Zeroth 52.9 h vs 0.9 h of
Korean fake), so "Korean => real" was a learnable shortcut. `lang_shares`
rescales each voice pool after the DOSS cap so every language holds the same
share of the real and the fake side. These tests build that imbalance on the
synthetic corpus and measure the drawn shares with and without the knob.
"""

import numpy as np
import pytest

from processing.config import (DrawConfig, ProcessingConfig, dump_processing_config,
                               processing_config_from_dict)
from processing.sampler import Sampler
from training.synthetic import synthetic_manifest


def _skewed(ko_real=0.9, ko_fake=0.1):
    """Voice buckets are language-pure (a speaker has one language), with ko
    dominating the real side and rare on the fake side; jsut-like `ja` fakes."""
    m = synthetic_manifest(n_per_pool=400, n_whole_file=200, seed=0).copy()
    m["lang"] = None
    rng = np.random.default_rng(0)
    for pool, p_ko in (("A", ko_real), ("B", ko_fake)):
        idx = m.index[m.pool == pool]
        lang = np.where(rng.random(len(idx)) < p_ko, "ko", "en")
        if pool == "B":
            lang[: len(idx) // 10] = "ja"
        m.loc[idx, "lang"] = lang
        m.loc[idx, "speaker_ref_id"] = [f"{pool}_{lg}_{i % 2}" for i, lg in enumerate(lang)]
    return m


def _drawn_shares(m, cfg, n=1500):
    lang = m.set_index("file_id")["lang"]
    pool = m.set_index("file_id")["pool"]
    counts = {"A": {}, "B": {}}
    for spec in Sampler(m, cfg).epoch_specs(n, seed=1):
        for c in spec.components:
            if c.role == "voice" and c.snr_db is None:
                side = pool[c.file_id]
                if side not in counts:
                    continue            # a whole-file row: no voice pool
                counts[side][lang[c.file_id]] = counts[side].get(lang[c.file_id], 0) + 1
    return {s: {k: v / sum(d.values()) for k, v in d.items()} for s, d in counts.items()}


def test_without_the_knob_language_reads_the_label():
    shares = _drawn_shares(_skewed(), DrawConfig())
    assert shares["A"]["ko"] - shares["B"].get("ko", 0) > 0.4


def test_with_the_knob_both_sides_carry_the_same_language_shares():
    cfg = DrawConfig(lang_shares={"ko": 0.5, "en": 0.5})
    shares = _drawn_shares(_skewed(), cfg)
    for side in ("A", "B"):
        assert shares[side]["ko"] == pytest.approx(0.5, abs=0.06), shares


def test_an_unlisted_language_without_an_other_share_is_never_drawn():
    cfg = DrawConfig(lang_shares={"ko": 0.5, "en": 0.5})
    assert "ja" not in _drawn_shares(_skewed(), cfg)["B"]


def test_other_collects_unlisted_languages():
    cfg = DrawConfig(lang_shares={"ko": 0.45, "en": 0.45, "other": 0.1})
    assert _drawn_shares(_skewed(), cfg)["B"].get("ja", 0) > 0


def test_the_knob_needs_the_lang_column():
    m = _skewed().drop(columns=["lang"])
    with pytest.raises(ValueError, match="lang"):
        Sampler(m, DrawConfig(lang_shares={"ko": 1.0}))


@pytest.mark.parametrize("bad", [{"ko": -0.1}, {"ko": 0.0}, {}])
def test_bad_shares_are_refused(bad):
    with pytest.raises(ValueError):
        DrawConfig(lang_shares=bad)


def test_the_shares_round_trip_through_the_config_file():
    cfg = ProcessingConfig(draw=DrawConfig(lang_shares={"ko": 0.4, "en": 0.6}))
    back = processing_config_from_dict(dump_processing_config(cfg))
    assert back.draw.lang_shares == (("en", 0.6), ("ko", 0.4))
    assert back == cfg
