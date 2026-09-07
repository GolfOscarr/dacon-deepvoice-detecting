"""The sampler: split safety, DOSS capping, determinism, epoch semantics."""

import numpy as np
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
    assert [s.sample_id for s in e1] == list(range(50, 100))
    assert e0 != e1


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

def test_doss_flattens_over_represented_domains(manifest):
    """★ 0.2k h domain-balanced -> 2.77% EER vs 6.4k h naive -> 3.29%.

    A weight, not a corpus edit: nothing is discarded, and N_c is sweepable.
    """
    fake = manifest[(manifest.row_kind == "component") & manifest.pool.isin(["B", "D"])]
    biggest = fake.domain_key.value_counts().idxmax()

    def share(cap):
        specs = list(Sampler(manifest, SamplerConfig(domain_cap=cap)).epoch_specs(4000))
        ids = [c.file_id for s in specs for c in s.components]
        dom = manifest.set_index("file_id").domain_key
        drawn = dom.reindex(ids).dropna()
        return (drawn == biggest).mean()

    uncapped, capped = share(10**9), share(20)
    assert capped < uncapped, f"capping must reduce the dominant domain: {capped} vs {uncapped}"


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
    for s in sampler.epoch_specs(2000):
        vp, mp, _, _ = CELL_TABLE[s.cell]
        roles = {c.role for c in s.components}
        if s.render_mode == "composed":
            assert ("voice" in roles) == bool(vp), s.cell
            assert ("music" in roles) == bool(mp), s.cell


def test_durations_stay_inside_the_test_range(sampler):
    for s in sampler.epoch_specs(2000):
        assert 0 < s.duration_s <= 60.0
        for c in s.components:
            assert c.target_start_s + c.duration_s <= s.duration_s + 1e-6


def test_sequential_samples_carry_a_crossfade(sampler):
    seq = [s for s in sampler.epoch_specs(3000) if s.structure == "sequential"]
    assert seq, "the sequential path must be exercised"
    assert all(s.crossfade_ms > 0 for s in seq)
    assert all(len(s.components) > 1 for s in seq)


def test_gain_is_skewed_toward_the_quiet_end(sampler):
    """★ G2Net: models generalise low-SNR -> high-SNR, not the reverse."""
    gains = [c.gain_db for s in sampler.epoch_specs(3000)
             for c in s.components if c.role == "voice" and s.stratum == "mixed"]
    assert np.mean(gains) < 0, "voice must sit below music on average"
    lo, hi = SamplerConfig().gain_db_range
    assert all(lo <= g <= hi for g in gains)


@pytest.mark.parametrize("bad", [{"duration_range": (60.0, 4.0)},
                                 {"sequential_prob": 1.5}])
def test_malformed_sampler_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        SamplerConfig(**bad)
