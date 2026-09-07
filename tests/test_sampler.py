"""The sampler: split safety, DOSS capping, determinism, epoch semantics."""

import numpy as np
import pandas as pd
import pytest

from training.sampler import (REFERENCE_MIX, CellMix, Sampler, SamplerConfig,
                              composed_fractions, head_positive_rates)
from training.spec import CELL_TABLE
from training.synthetic import synthetic_manifest


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)


@pytest.fixture(scope="module")
def sampler(manifest):
    return Sampler(manifest)


# --------------------------------------------------------------------------- #
# the reference mix

def test_reference_mix_sums_to_one():
    assert sum(REFERENCE_MIX.values()) == pytest.approx(1.0, abs=1e-9)


def test_reference_mix_satisfies_c1():
    rates = head_positive_rates(CellMix())
    assert rates == pytest.approx(
        {"file": 0.610, "voice": 0.507, "music": 0.511,
         "v_pres": 0.690, "m_pres": 0.695}, abs=1e-3)
    assert all(0.2 <= v <= 0.8 for v in rates.values())


@pytest.mark.parametrize("bad", [
    {c: 1.0 for c in range(1, 10)},                       # does not sum to 1
    {c: 1 / 8 for c in range(1, 9)},                      # missing cell 9
])
def test_malformed_cell_mixes_are_rejected(bad):
    with pytest.raises(ValueError):
        CellMix(bad)


def test_negative_cell_probability_is_rejected():
    p = dict(REFERENCE_MIX)
    p[9], p[5] = -0.05, p[5] + 0.05
    with pytest.raises(ValueError, match=">= 0"):
        CellMix(p)


# --------------------------------------------------------------------------- #
# determinism

def test_same_key_gives_an_identical_spec(sampler):
    assert sampler.sample_spec(42, epoch=3, seed=7) == sampler.sample_spec(42, epoch=3, seed=7)


@pytest.mark.parametrize("kw", [{"sample_id": 43}, {"epoch": 4}, {"seed": 8}])
def test_changing_any_key_component_changes_the_stream(sampler, kw):
    base = dict(sample_id=42, epoch=3, seed=7)
    assert sampler.sample_spec(**base) != sampler.sample_spec(**{**base, **kw})


def test_an_epoch_is_a_fixed_count_of_specs(sampler):
    """Otherwise sample_id is undefined and reproducibility is nominal."""
    e0 = list(sampler.epoch_specs(50, epoch=0))
    e1 = list(sampler.epoch_specs(50, epoch=1))
    assert len(e0) == len(e1) == 50
    assert [s.sample_id for s in e0] == list(range(50))
    assert [s.sample_id for s in e1] == list(range(50)), "sample_id is epoch-LOCAL"
    assert e0 != e1, "epoch is in the RNG key, so the streams still differ"


def test_changing_steps_per_epoch_does_not_reroll_the_corpus(sampler):
    """🔴 `sample_id` was `epoch * n + i`, so it moved with steps_per_epoch.

    Changing the batch size would silently re-roll every sample in every later
    epoch, which makes a run irreproducible for a reason nobody would look for.
    `epoch` is already in the RNG key, so the index within the epoch suffices.
    """
    short = list(sampler.epoch_specs(50, epoch=1))
    long_ = list(sampler.epoch_specs(200, epoch=1))
    assert short == long_[:50]


# --------------------------------------------------------------------------- #
# 🔴 split safety

def test_only_draws_from_the_active_slice(manifest):
    """artifact_family disjointness is why this must hold at draw time."""
    m = manifest.copy()
    # Partition at random, not by row order: the manifest is grouped by pool, so
    # an index slice would hand one side no fake components at all.
    rng = np.random.default_rng(0)
    m["slice"] = np.where(rng.random(len(m)) < 0.5, "val", "train")
    for slice_ in ("train", "val"):
        allowed = set(m.loc[m["slice"] == slice_, "file_id"])
        specs = list(Sampler(m, slice_=slice_).epoch_specs(500))
        drawn = {c.file_id for s in specs for c in s.components}
        assert drawn <= allowed, f"{slice_}: leaked {sorted(drawn - allowed)[:3]}"


def test_an_empty_slice_fails_loudly(manifest):
    with pytest.raises(ValueError, match="no manifest rows"):
        Sampler(manifest, slice_="probe")


def test_a_fold_filter_is_applied(manifest):
    m = manifest.copy()
    rng = np.random.default_rng(1)
    m["fold"] = rng.integers(0, 2, len(m))
    specs = list(Sampler(m, fold=1).epoch_specs(200))
    allowed = set(m.loc[m.fold == 1, "file_id"])
    assert {c.file_id for s in specs for c in s.components} <= allowed


def test_a_slice_missing_a_component_kind_fails_loudly(manifest):
    """A degenerate slice must raise, not quietly emit a lopsided stream.

    Found by a test that partitioned the manifest by row order: the rows are
    grouped by pool, so one side had no fake voice components and the sampler
    would otherwise have drawn an all-real corpus without complaint.
    """
    m = manifest.copy()
    m.loc[m.pool == "B", "slice"] = "val"          # remove every fake voice row
    with pytest.raises(ValueError, match="no fake voice components"):
        list(Sampler(m).epoch_specs(500))


# --------------------------------------------------------------------------- #
# DOSS capping

@pytest.fixture(scope="module")
def big_manifest():
    """A corpus large enough that the SHIPPED `domain_cap` actually binds.

    🔴 At `n_per_pool=200` the largest domain holds ~75 files, far under the
    shipped `N_c = 500`, so every weight is 1.0 and the default is a no-op --
    the earlier DOSS test had to pass `cap=20` to make anything happen, and
    therefore never exercised the value we ship.
    """
    return synthetic_manifest(n_per_pool=2000, n_whole_file=200, seed=0)


def test_the_shipped_domain_cap_binds_on_a_production_sized_corpus(big_manifest):
    """The premise of every assertion below. Stated, so it cannot rot silently."""
    fake = big_manifest[(big_manifest.row_kind == "component")
                        & big_manifest.pool.isin(["B", "D"])]
    counts = fake.domain_key.value_counts()
    cap = SamplerConfig().domain_cap
    over = counts[counts > cap]
    assert len(over) >= 2, (
        f"nothing exceeds the shipped cap {cap}; largest domain is {counts.max()}")


def test_doss_flattens_over_represented_domains(big_manifest):
    """★ 0.2k h domain-balanced -> 2.77% EER vs 6.4k h naive -> 3.29%.

    A weight, not a corpus edit: nothing is discarded, and N_c is sweepable.

    🔴 Two things the earlier version did not do. It runs at the SHIPPED
    `domain_cap`, on a corpus where that value binds (see the fixture), and it
    asserts an EFFECT SIZE. `capped < uncapped` is met by a rounding error, and
    the shipped default was measured at N_eff 7.46 capped against 7.46 uncapped
    -- identical, because the cap bound on nothing.

    The weights are read straight off the sampler, so this is exact rather than
    a Monte-Carlo estimate: `_doss_weights` is the production implementation.
    """
    cap = SamplerConfig().domain_cap
    dominant = big_manifest.loc[big_manifest.pool == "D",
                                "domain_key"].value_counts().idxmax()

    def domain_weights(domain_cap):
        s = Sampler(big_manifest, SamplerConfig(domain_cap=domain_cap))
        rows, w = s._by_role_fake[("music", True)], s._weights[("music", True)]
        return pd.Series(w, index=rows.domain_key.to_numpy()).groupby(level=0).sum()

    def n_eff(w):
        return float(np.exp(-(w * np.log(w)).sum()))

    uncapped, capped = domain_weights(10**9), domain_weights(cap)
    assert uncapped[dominant] >= 0.25, (
        f"no head to flatten: the dominant domain holds {uncapped[dominant]:.3f}")
    assert capped[dominant] <= 0.85 * uncapped[dominant], (
        f"the shipped cap must take at least 15% off the dominant domain: "
        f"{capped[dominant]:.4f} vs {uncapped[dominant]:.4f}")
    assert n_eff(capped) >= n_eff(uncapped) + 0.3, (
        f"effective domains must rise materially: "
        f"{n_eff(capped):.3f} vs {n_eff(uncapped):.3f}")
    # Nothing is discarded: capping is a reweighting, so every domain survives.
    assert set(capped.index) == set(uncapped.index)
    assert capped.sum() == pytest.approx(1.0)


def test_doss_capping_reaches_the_drawn_stream(big_manifest):
    """The weights are only a claim until the drawn stream moves with them.

    ⚠️ Measured over pool-D draws only. Pooled over every component, real files
    (weight-free, no `domain_key`) and the whole-file rows dilute the effect to
    within the noise of a 6k-spec draw -- an adjacent quantity.
    """
    cap = SamplerConfig().domain_cap
    dom = big_manifest.set_index("file_id").domain_key
    pool_d = set(big_manifest.loc[(big_manifest.row_kind == "component")
                                  & (big_manifest.pool == "D"), "file_id"])
    dominant = big_manifest.loc[big_manifest.pool == "D",
                                "domain_key"].value_counts().idxmax()

    def share(domain_cap):
        # a = b = 1 composes the single-component cells too, so cell 4 also draws
        # fake music: ~5,200 pool-D draws, enough that a 0.06 gap is ~9 sigma.
        cfg = SamplerConfig(domain_cap=domain_cap, single_composed_rate=1.0)
        specs = Sampler(big_manifest, cfg).epoch_specs(20_000)
        ids = [c.file_id for s in specs for c in s.components if c.file_id in pool_d]
        assert len(ids) > 3_000, len(ids)
        return float((dom.reindex(ids) == dominant).mean())

    uncapped, capped = share(10**9), share(cap)
    assert uncapped >= 0.25, f"no head to flatten: dominant domain at {uncapped:.3f}"
    assert capped <= uncapped - 0.04, (
        f"the shipped cap must visibly flatten the drawn stream: "
        f"{capped:.4f} vs {uncapped:.4f}")


def test_domain_weights_are_a_probability_vector(sampler):
    for key, w in sampler._weights.items():
        if len(w):
            assert w.sum() == pytest.approx(1.0), key
            assert (w >= 0).all(), key


def test_domain_cap_must_be_positive():
    with pytest.raises(ValueError, match="domain_cap"):
        SamplerConfig(domain_cap=0)


# --------------------------------------------------------------------------- #
# structure

def test_every_spec_respects_its_cell(sampler):
    """Both render modes. The `if composed` guard used to exempt ~59% of the
    stream, so the whole-file role assignment was checked by nothing."""
    seen = {"composed": 0, "whole_file": 0}
    for s in sampler.epoch_specs(3000):
        vp, mp, _, _ = CELL_TABLE[s.cell]
        roles = {c.role for c in s.components}
        seen[s.render_mode] += 1
        if s.render_mode == "composed":
            assert ("voice" in roles) == bool(vp), s.cell
            assert ("music" in roles) == bool(mp), s.cell
        else:
            # A whole file is one row used as-is; its role must still name the
            # component the cell says is present ("noise" for cell 9).
            assert len(s.components) == 1, s.cell
            expected = "voice" if vp else "music" if mp else "noise"
            assert roles == {expected}, (s.cell, roles)
    assert min(seen.values()) > 100, f"both render modes must be exercised: {seen}"


def test_durations_stay_inside_the_test_range(sampler):
    """🔴 Both bounds. An earlier version asserted `0 < d <= 60.0`.

    The lower bound was 0, not 4 -- the adjacent quantity -- so it passed while
    a corpus of short sources produced 2,580 of 4,000 specs below
    `AudioConfig.min_seconds`, because `take = min(duration, row.duration_s)`
    silently shortens the timeline.
    """
    lo, hi = SamplerConfig().duration_range
    for s in sampler.epoch_specs(2000):
        assert lo <= s.duration_s <= hi, s.duration_s
        for c in s.components:
            assert c.target_start_s + c.duration_s <= s.duration_s + 1e-6


def test_sources_shorter_than_the_minimum_are_dropped(manifest):
    """A source shorter than the floor cannot back a legal sample."""
    m = manifest.copy()
    m.loc[m.row_kind == "whole_file", "duration_s"] = 1.5
    sampler = Sampler(m)
    assert sampler.n_dropped_short > 0
    lo = SamplerConfig().duration_range[0]
    assert all(s.duration_s >= lo for s in sampler.epoch_specs(2000))


def test_a_corpus_of_only_short_sources_fails_loudly(manifest):
    m = manifest.copy()
    m["duration_s"] = 1.0
    with pytest.raises(ValueError, match="shorter than"):
        Sampler(m)


def test_sequential_samples_carry_a_crossfade(sampler):
    seq = [s for s in sampler.epoch_specs(3000) if s.structure == "sequential"]
    assert seq, "the sequential path must be exercised"
    assert all(s.crossfade_ms > 0 for s in seq)
    assert all(len(s.components) > 1 for s in seq)


def test_gain_is_skewed_toward_the_quiet_end(sampler):
    """★ G2Net: models generalise low-SNR -> high-SNR, not the reverse.

    🔴 `mean < 0` was the adjacent quantity: `gain_db_mean = -3.6` is the only
    source of gain in the sampler, so that assertion could only fail if gain were
    unwired entirely. It would have greened at `gain_db_mean = -0.01`, which is
    not a skew. Assert the drawn distribution instead: its location, its spread,
    and the mass below unity gain that *is* the skew.
    """
    cfg = SamplerConfig()
    gains = np.array([c.gain_db for s in sampler.epoch_specs(6000)
                      if s.render_mode == "composed"
                      for c in s.components if c.role == "voice"])
    assert len(gains) > 1_500, len(gains)

    # sd of the mean is ~0.08 here, so 0.35 is >4 sigma of slack and still
    # catches a mean moved to 0 (or to the -1.0 a "mild" tweak would pick).
    assert abs(gains.mean() - cfg.gain_db_mean) < 0.35, gains.mean()
    assert abs(gains.std() - cfg.gain_db_sigma) < 0.5, gains.std()
    # P(N(-3.6, 4) < 0) = 0.816. At gain_db_mean = 0 this is 0.5 and fails.
    assert (gains < 0).mean() > 0.75, (gains < 0).mean()
    lo, hi = cfg.gain_db_range
    assert lo <= gains.min() and gains.max() <= hi


def test_gain_is_the_voice_music_ratio_not_a_level_shift(manifest):
    """⚠️ Filtering to the mixed stratum hid what the sampler used to do.

    `gain_db` is the voice/music level ratio (docs/data/02, A-A3). It was applied
    to every component whose role is "voice", so a composed voice-only sample
    (cells 1/2, reachable whenever `single_composed_rate > 0`) was gained with no
    music to be relative to -- an absolute level shift, which is A-A7 jitter and
    belongs in the augment registry. Nothing renormalises it either
    (`render._normalize` models the test chain; there is no loudness stage), so
    it reached the waveform as a composedness cue inside the voice-only stratum.

    Two exceptions, both deliberate: a whole-file row is used as-is at gain 0,
    and a cell-9 noise draw has no voice component to gain.
    """
    cfg = SamplerConfig(single_composed_rate=1.0)
    by_stratum: dict[str, list[float]] = {}
    whole_file_gains: list[float] = []
    for s in Sampler(manifest, cfg).epoch_specs(6000):
        for c in s.components:
            if c.role != "voice":
                continue
            if s.render_mode == "composed":
                by_stratum.setdefault(s.stratum, []).append(c.gain_db)
            else:
                whole_file_gains.append(c.gain_db)

    assert set(by_stratum) == {"voice-only", "mixed"}, sorted(by_stratum)

    solo = np.array(by_stratum["voice-only"])
    assert len(solo) > 400, len(solo)
    assert not solo.any(), (
        f"a solo voice component has nothing to be relative to, so it carries no "
        f"ratio; {int((solo != 0).sum())} of {len(solo)} were gained")

    mixed = np.array(by_stratum["mixed"])
    assert len(mixed) > 400, len(mixed)
    assert abs(mixed.mean() - cfg.gain_db_mean) < 0.6, mixed.mean()
    assert (mixed < 0).mean() > 0.72, (mixed < 0).mean()

    assert whole_file_gains and not any(whole_file_gains), "whole files are ungained"


def test_restricting_gain_to_the_ratio_does_not_move_the_shipped_stream(manifest):
    """🔴 The change is a no-op at `single_composed_rate = 0.0`.

    That is why it was cheap to make: the shipped config emits no composed
    voice-only samples, so every gain the shipped stream draws is a genuine
    voice/music ratio and the drawn specs are byte-identical either way. Pin the
    premise, so a future change to `single_composed_rate`'s default has to
    confront it.
    """
    specs = list(Sampler(manifest, SamplerConfig()).epoch_specs(4000))
    solo_composed = [s for s in specs
                     if s.render_mode == "composed" and s.stratum == "voice-only"]
    assert not solo_composed, (
        f"{len(solo_composed)} composed voice-only specs at the shipped default; "
        f"the ratio restriction is no longer a no-op")


@pytest.mark.parametrize("bad", [{"duration_range": (60.0, 4.0)},
                                 {"sequential_prob": 1.5}])
def test_malformed_sampler_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        SamplerConfig(**bad)
