"""The spec audit, and proof that each invariant can actually fail.

🔴 A green audit that cannot go red is worse than no audit. Every check here is
mutation-tested: the constraint is deliberately broken and the corresponding
invariant must fire.
"""

import dataclasses

import pytest

from training.audit import audit_specs, run_audit
from training.sampler import (REFERENCE_MIX, CellMix, Sampler, SamplerConfig,
                              composed_fractions, head_positive_rates,
                              mixedness_balance)
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest

N = 6_000


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)


@pytest.fixture(scope="module")
def specs(manifest):
    return list(Sampler(manifest).epoch_specs(N))


def _spec(cell, composed=True, sample_id=0, transforms=(), file_id="A00001"):
    return SampleSpec(
        sample_id=sample_id, epoch=0, seed=0, scheme_version="v1",
        duration_s=10.0, cell=cell,
        render_mode="composed" if composed else "whole_file",
        structure="overlap", transforms=transforms,
        components=(ComponentDraw(file_id=file_id, role="voice", source_offset_s=0.0,
                                  duration_s=10.0, target_start_s=0.0, gain_db=0.0),))


# --------------------------------------------------------------------------- #
# the reference configuration passes

def test_reference_sampler_passes_every_invariant(manifest, specs):
    report = audit_specs(specs, manifest=manifest, slice_="train")
    assert report.ok, str(report)


def test_measured_rates_match_the_designed_mix(specs):
    """The stream must reproduce the arithmetic, not just agree with itself."""
    designed = head_positive_rates(CellMix())
    report = audit_specs(specs)
    assert report.results["I8_C1_positive_rates"][0]
    measured = {}
    for head, pool, pred in (
            ("file", specs, lambda s: s.file_fake == 1),
            ("voice", [s for s in specs if s.voice_present], lambda s: s.voice_fake == 1),
            ("music", [s for s in specs if s.music_present], lambda s: s.music_fake == 1)):
        measured[head] = sum(1 for s in pool if pred(s)) / len(pool)
    for head, want in ((h, designed[h]) for h in measured):
        assert abs(measured[head] - want) < 0.03, f"{head}: {measured[head]} vs {want}"


# --------------------------------------------------------------------------- #
# 🔴 mutation tests -- each invariant must be able to fail

def test_I8_catches_the_naive_6_7_heavy_mix(manifest):
    """The advice "over-weight cells 6 and 7" violates C1 on both presence heads.

    Cells 6/7/8 all have BOTH components present, so over-weighting them drives
    v_pres and m_pres toward 1. Measured at 0.820, outside [0.2, 0.8].
    """
    naive = CellMix({1: .08, 2: .08, 3: .08, 4: .08, 5: .10,
                     6: .22, 7: .22, 8: .12, 9: .02})
    rates = head_positive_rates(naive)
    assert rates["v_pres"] == pytest.approx(0.820, abs=1e-3)
    assert rates["m_pres"] == pytest.approx(0.820, abs=1e-3)

    sampler = Sampler(manifest, SamplerConfig(cell_mix=naive))
    report = run_audit(sampler, n=N, manifest=manifest)
    assert not report.ok
    assert "I8_C1_positive_rates" in report.failures


def test_I2b_catches_mixedness_predicting_fakeness(manifest):
    """A mix can satisfy C1 and still make "is a mixed file" predict FAKE.

    The earlier reference mix did exactly that: P(mixed|FAKE) 0.667 vs
    P(mixed|REAL) 0.300.
    """
    trapped = CellMix({1: .10, 2: .10, 3: .10, 4: .10, 5: .12,
                       6: .15, 7: .15, 8: .10, 9: .08})
    assert all(0.2 <= v <= 0.8 for v in head_positive_rates(trapped).values()), \
        "this mix passes C1 -- that is the point"
    # Over cells 1-8, per the definition: cell 9 is excluded because a file with
    # no components cannot be fake. The cell-9-inclusive values are 0.667/0.300.
    pm_f, pm_r = mixedness_balance(trapped)
    assert pm_f == pytest.approx(0.667, abs=1e-3)
    assert pm_r == pytest.approx(0.375, abs=1e-3)

    report = run_audit(Sampler(manifest, SamplerConfig(cell_mix=trapped)),
                       n=N, manifest=manifest)
    assert not report.ok
    assert "I2b_mixedness_balance" in report.failures


def test_I2_catches_a_marginally_balanced_within_stratum_shortcut():
    """🔴 The stratified form, not the marginal one.

    f1=1, f2=0 balances marginally while "composed" predicts REAL perfectly
    among voice-only files. I2 must see it.
    """
    specs = ([_spec(1, composed=True, sample_id=i) for i in range(500)] +
             [_spec(2, composed=False, sample_id=500 + i) for i in range(500)])
    report = audit_specs(specs)
    assert not report.results["I2_stratified_composedness"][0]
    assert "1.0" in report.results["I2_stratified_composedness"][1]


def test_I1_catches_a_label_dependent_transform():
    """If only fakes get a transform, the model learns the transform."""
    specs = ([_spec(2, sample_id=i, transforms=(("codec", {}),)) for i in range(500)] +
             [_spec(1, sample_id=500 + i, transforms=()) for i in range(500)])
    report = audit_specs(specs)
    assert not report.results["I1_transform_label_independence"][0]
    assert "codec" in report.results["I1_transform_label_independence"][1]


def test_I5_catches_a_file_drawn_from_outside_the_slice(manifest, specs):
    report = audit_specs(specs[:200] + [_spec(1, file_id="NOT_IN_MANIFEST")],
                         manifest=manifest, slice_="train")
    assert not report.results["I5_split_safety"][0]
    assert "NOT_IN_MANIFEST" in report.results["I5_split_safety"][1]


def test_I9_catches_a_starved_masked_head():
    """C2: rare-component batches make _masked_mean take large, noisy steps."""
    specs = [_spec(3 if i % 32 else 1, sample_id=i) for i in range(320)]
    report = audit_specs(specs, batch_size=32, min_present=2)
    assert not report.results["I9_C2_present_count_floor"][0]


def test_I7_catches_a_scraped_cell_6(monkeypatch):
    """Cells 6/7 cannot be scraped; SampleSpec refuses to build one at all."""
    with pytest.raises(ValueError, match="only be composed"):
        _spec(6, composed=False)


# --------------------------------------------------------------------------- #
# the f8 knob

@pytest.mark.parametrize("f8,f5", [(0.0, 0.7246), (0.25, 0.7935),
                                   (0.5, 0.8623), (1.0, 1.0)])
def test_f_schedule_matches_the_published_table(f8, f5):
    """docs/data/02 and docs/pipelines/02 §4, re-derived from the mix."""
    f = composed_fractions(CellMix(), f8)
    assert f[5] == pytest.approx(f5, abs=1e-3)
    assert f[6] == f[7] == 1.0, "cells 6/7 cannot be scraped"
    assert f[8] == f8


def test_strict_policy_is_the_f8_equals_one_endpoint(manifest):
    """Strict is a value change, not a second code path."""
    strict = Sampler(manifest, SamplerConfig(f8=1.0))
    specs = list(strict.epoch_specs(N))
    mixed = [s for s in specs if s.stratum == "mixed"]
    assert all(s.render_mode == "composed" for s in mixed), \
        "under strict every mixed file is composed"
    assert audit_specs(specs, manifest=manifest, slice_="train").ok


def test_conditional_policy_keeps_genuine_whole_file_audio(manifest):
    """f8 = 0 is primary precisely because it does."""
    specs = list(Sampler(manifest, SamplerConfig(f8=0.0)).epoch_specs(N))
    whole = [s for s in specs if s.render_mode == "whole_file"]
    assert len(whole) / len(specs) > 0.10, "conditional must retain real whole files"
    assert any(s.cell == 8 for s in whole), "AI songs must be usable as-is"
    assert any(s.cell == 5 for s in whole), "natural songs must be usable as-is"


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_f8_out_of_range_is_rejected(bad):
    with pytest.raises(ValueError, match="f8"):
        composed_fractions(CellMix(), bad)


# --------------------------------------------------------------------------- #
# I1b -- the metadata shortcut audit (E-S2 at spec level)

def test_I1b_passes_for_both_policies(manifest):
    """The joint check: each balance can hold while a combination still leaks."""
    for f8 in (0.0, 1.0):
        report = run_audit(Sampler(manifest, SamplerConfig(f8=f8)), n=N, manifest=manifest)
        passed, why = report.results["I1b_metadata_shortcut_auc"]
        assert passed, f"f8={f8}: {why}"


def test_I1b_excludes_cell_9_and_why(manifest):
    """🔴 Cell 9 is always REAL and never composed, so it contributes an
    unfixable "not composed => REAL" correlation.

    Including it, the strict policy scores 0.6003 and fails the 0.60 gate --
    a false alarm blocking a policy we deliberately support.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    specs = list(Sampler(manifest, SamplerConfig(f8=1.0)).epoch_specs(8000))

    def auc(pool):
        y = np.array([s.file_fake for s in pool])
        X = np.array([[s.duration_s, float(s.render_mode == "composed"),
                       float(s.structure == "sequential"), float(len(s.components)),
                       float(np.mean([c.gain_db for c in s.components])), s.crossfade_ms,
                       float(np.mean([c.source_offset_s for c in s.components]))]
                      for s in pool])
        Xs = StandardScaler().fit_transform(X)
        return roc_auc_score(y, LogisticRegression(max_iter=2000).fit(Xs, y)
                             .predict_proba(Xs)[:, 1])

    assert auc(specs) > 0.59, "including cell 9, the strict policy sits at the gate"
    assert auc([s for s in specs if s.cell != 9]) < 0.56, "cells 1-8 must have headroom"


def test_I1b_catches_a_duration_shortcut():
    """Nothing else in the audit looks at duration.

    If fake samples were systematically longer -- e.g. because AI songs are
    used whole while real mixes are cropped -- this is the only check that sees
    it, and it is exactly the E-S2 failure mode.
    """
    def timed(cell, sample_id, seconds):
        return SampleSpec(
            sample_id=sample_id, epoch=0, seed=0, scheme_version="v1",
            duration_s=seconds, cell=cell, render_mode="composed",
            structure="overlap",
            components=(ComponentDraw(file_id="A00001", role="voice",
                                      source_offset_s=0.0, duration_s=seconds,
                                      target_start_s=0.0, gain_db=0.0),))

    specs = ([timed(2, i, 50.0) for i in range(500)] +
             [timed(1, 500 + i, 8.0) for i in range(500)])
    report = audit_specs(specs)
    passed, why = report.results["I1b_metadata_shortcut_auc"]
    assert not passed, why
    assert "AUC = 1.0" in why or "AUC = 0.9" in why, why
