"""The spec audit, and proof that each invariant can actually fail.

Critical: A green audit that cannot go red is worse than no audit. Every check
here is mutation-tested: the constraint is deliberately broken and the
corresponding invariant must fire.
"""

import dataclasses

import numpy as np
import pytest

from training.audit import (SHORTCUT_AUC_GATE, _metadata_shortcut,
                            audit_specs, run_audit)
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

#: Every invariant `audit_specs` emits, mapped to the test **in this module**
#: that proves it can go RED.
#:
#: Critical: `report.ok` on the reference stream is worth exactly as much as
#: the checks behind it can fail. A review found this file greening a bag of
#: invariants of which several could not go red, and a skipped check reads as a
#: pass on the `.ok` property. This map is the fix: adding an invariant without
#: a mutation test, renaming one, or silently dropping one, all fail
#: `test_reference_sampler_passes_every_invariant`.
MUTATION_TESTS = {
    "I1_transform_name_independence": "test_I1_catches_a_label_dependent_transform",
    "I1b_metadata_shortcut_auc": "test_I1b_catches_a_duration_shortcut",
    "I2_stratified_composedness":
        "test_I2_catches_a_marginally_balanced_within_stratum_shortcut",
    "I2b_mixedness_balance": "test_I2b_catches_mixedness_predicting_fakeness",
    "I2c_marginal_composedness": "test_I2c_catches_the_marginal_composedness_residual",
    "I3_real_components_on_both_sides": "test_I3_can_actually_fail",
    "I4_component_pools_imply_the_labels": "test_I4_is_falsifiable",
    "I5_split_safety": "test_I5_catches_a_file_drawn_from_outside_the_slice",
    "I7a_cells_6_7_always_composed": "test_I7a_fires_on_a_scraped_cell_6",
    "I7_eval_size_floors": "test_I7_size_floors_run_when_asked",
    "I8_C1_positive_rates": "test_I8_catches_the_naive_6_7_heavy_mix",
    "I9_C2_present_count_floor": "test_I9_catches_a_starved_masked_head",
    "I21_generator_diversity": "test_I21_catches_generator_monoculture",
}

#: The one invariant that legitimately does not run on a TRAINING stream. VG1
#: A8/A9 bind on an evaluation stream, so it reports SKIP rather than PASS.
EXPECTED_SKIPS = {"I7_eval_size_floors"}


def test_reference_sampler_passes_every_invariant(manifest, specs):
    """Critical: and every invariant is one that could have failed.

    Three things, because "the reference config is green" alone is compatible
    with a suite of unfalsifiable checks and silent skips:
    """
    report = audit_specs(specs, manifest=manifest, slice_="train")
    assert report.ok, str(report)

    # 1. nothing ran that we do not know about, and nothing we expect vanished
    assert set(report.results) == set(MUTATION_TESTS), (
        f"unmapped: {sorted(set(report.results) - set(MUTATION_TESTS))}; "
        f"missing: {sorted(set(MUTATION_TESTS) - set(report.results))}")

    # 2. the green came from checks that RAN. `.ok` counts a skip as a pass, so
    #    a check that quietly stopped running would otherwise be invisible here.
    assert set(report.skipped) == EXPECTED_SKIPS, report.skipped
    assert set(report.ran) == set(MUTATION_TESTS) - EXPECTED_SKIPS

    # 3. each of them is mutation-tested, by a test that exists
    missing = sorted(t for t in MUTATION_TESTS.values() if t not in globals())
    assert not missing, f"mutation test(s) named but not defined: {missing}"


def test_measured_rates_match_the_designed_mix(specs):
    """The stream must reproduce the arithmetic, not just agree with itself.

    Caveat: all FIVE of C1's quantities. The earlier version checked
    `file`/`voice`/`music` and skipped `v_pres`/`m_pres` -- the two the naive
    6/7-heavy mix actually breaks (`test_I8_catches_the_naive_6_7_heavy_mix`),
    so the presence rates were designed, gated, and never measured.
    """
    designed = head_positive_rates(CellMix())
    report = audit_specs(specs)
    assert report.results["I8_C1_positive_rates"][0]
    pools = {
        "file": (specs, lambda s: s.file_fake == 1),
        "voice": ([s for s in specs if s.voice_present], lambda s: s.voice_fake == 1),
        "music": ([s for s in specs if s.music_present], lambda s: s.music_fake == 1),
        "v_pres": (specs, lambda s: s.voice_present == 1),
        "m_pres": (specs, lambda s: s.music_present == 1),
    }
    assert set(pools) == set(designed), "every designed rate must be measured"
    for head, (pool, pred) in pools.items():
        measured = sum(1 for s in pool if pred(s)) / len(pool)
        assert abs(measured - designed[head]) < 0.03, \
            f"{head}: {measured} vs {designed[head]}"


# --------------------------------------------------------------------------- #
# Critical: mutation tests -- each invariant must be able to fail

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
    """Critical: the stratified form, not the marginal one.

    f1=1, f2=0 balances marginally while "composed" predicts REAL perfectly
    among voice-only files. I2 must see it.
    """
    specs = ([_spec(1, composed=True, sample_id=i) for i in range(500)] +
             [_spec(2, composed=False, sample_id=500 + i) for i in range(500)])
    report = audit_specs(specs)
    passed, why = report.results["I2_stratified_composedness"]
    assert not passed
    # Caveat: WHICH stratum, not just the size of the gap. `"1.0" in why`
    # matched the tolerance, the sample count, any float -- so a mutation that
    # tripped the wrong stratum still passed.
    assert "= 1.0000 in 'voice-only'" in why, why


def _voice_only_stream(gap, n=1000, base=0.5):
    """`n` FAKE and `n` REAL voice-only specs whose composed rates differ by `gap`.

    Cells 1 and 2 are the same presence stratum, so this is exactly the
    within-stratum shortcut I2 exists to catch, at a controllable effect size.
    """
    specs, sid = [], 0
    for cell, rate in ((2, base + gap / 2), (1, base - gap / 2)):
        composed = int(round(rate * n))
        for i in range(n):
            specs.append(_spec(cell, composed=i < composed, sample_id=sid))
            sid += 1
    return specs


@pytest.mark.parametrize("gap", [0.25, 0.15])
def test_I2_still_fires_on_a_real_gap_under_the_noise_aware_bound(gap):
    """Critical: the bound must not have bought stability with blindness.

    At n = 1,000 a side and a composed rate near 0.5 the gap's standard error is
    0.022, so the noise-aware tolerance is 0.089 -- and a 0.15 gap is still 6.7
    SE of signal. The tolerance widens to cover the estimator, not to cover a
    leak.
    """
    passed, why = audit_specs(
        _voice_only_stream(gap)).results["I2_stratified_composedness"]
    assert not passed, why
    assert "'voice-only'" in why, why


def test_I2_is_stable_across_seeds_on_a_clean_corpus(manifest):
    """Critical: the flakiness this bound was written for. A guardrail that fails at
    random on a clean corpus gets switched off, and then it does not catch the
    composition trap it exists for.

    Measured before the fix, over 16 seeds on this manifest: at
    `single_composed_rate = 0.5` the flat `tol = 0.02` tripped I2 on 2 of 4 seeds
    at n = 8,000 and again at n = 20,000, and even at the shipped `a = 0.0` the
    `mixed` stratum reached a gap of 0.0471 -- twice the tolerance -- because
    that is the one stratum with any variance at the shipped config. The suite
    passed only because it draws seed 0.

    Caveat: deliberately more than one seed and more than one config. A
    single-seed assertion is how this went unnoticed.
    """
    for a in (0.0, 0.5):
        for seed in range(4):
            specs = list(Sampler(manifest, SamplerConfig(single_composed_rate=a))
                         .epoch_specs(6_000, seed=seed))
            report = audit_specs(specs, manifest=manifest, slice_="train")
            for key in ("I2_stratified_composedness", "I2c_marginal_composedness"):
                passed, why = report.results[key]
                assert passed, f"a={a} seed={seed} {key}: {why}"


def test_the_noise_aware_tolerance_keeps_the_floor_where_the_estimate_is_exact():
    """Caveat: widening must be earned by variance, not applied everywhere.

    The shipped `single_composed_rate = 0.0` leaves the two single-component
    strata with no composed samples on either side: p = 0, so the standard error
    is 0 and the full 0.02 floor still binds. Only strata that actually vary pay
    for their own noise.
    """
    from training.audit import NOISE_K, _gap_tolerance

    exact, noise = _gap_tolerance(0, 500, 0, 500, floor=0.02)
    assert (exact, noise) == (0.02, 0.0)

    # a stratum that varies pays for it, and the price falls as 1/sqrt(n)
    small_tol, small_noise = _gap_tolerance(250, 500, 250, 500, floor=0.02)
    big_tol, big_noise = _gap_tolerance(2_500, 5_000, 2_500, 5_000, floor=0.02)
    assert small_tol > big_tol > 0.02
    assert small_noise == pytest.approx(big_noise * np.sqrt(10), rel=0.02)
    assert small_noise == pytest.approx(NOISE_K * np.sqrt(0.25 * (2 / 500)), rel=1e-6)


def test_an_unresolvable_stratum_reports_a_skip_not_a_pass():
    """Critical: A stream too thin to resolve anything must not read as green.

    The ceiling is derived rather than picked: for a binary feature
    `AUC = 0.5 + gap / 2`, so the E-S2 shortcut gate of 0.60 is a gap of 0.20. A
    stratum whose noise floor exceeds that cannot see a leak the pipeline would
    act on, so the check reports SKIP -- which `AuditReport.skipped` keeps
    distinct from a pass.
    """
    from training.audit import RESOLVABLE_GAP

    assert RESOLVABLE_GAP == pytest.approx(0.20)

    thin = _voice_only_stream(0.0, n=12)          # 4 SE ~ 0.41, far over 0.20
    report = audit_specs(thin)
    assert "I2_stratified_composedness" in report.skipped, report.results
    assert "SKIP  I2_stratified_composedness" in str(report)
    assert "PASS  I2_stratified_composedness" not in str(report)


def test_I1_catches_a_label_dependent_transform():
    """If only fakes get a transform, the model learns the transform."""
    specs = ([_spec(2, sample_id=i, transforms=(("codec", {}),)) for i in range(500)] +
             [_spec(1, sample_id=500 + i, transforms=()) for i in range(500)])
    report = audit_specs(specs)
    assert not report.results["I1_transform_name_independence"][0]
    assert "codec" in report.results["I1_transform_name_independence"][1]


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


@pytest.mark.parametrize("cell", [6, 7])
def test_I7a_fires_on_a_scraped_cell_6(cell):
    """Cells 6/7 cannot be scraped. Two layers, and the test must exercise both.

    Critical: the earlier version was named for I7 and never called the audit:
    it asserted only that `SampleSpec.__post_init__` refuses to construct such
    a spec, which is `test_spec.py`'s job, and carried an unused `monkeypatch`
    fixture. I7a is declared defense-in-depth precisely *because* the type
    refuses first -- so the only way to know it works is to defeat the type and
    hand the audit a spec that should never exist.
    """
    with pytest.raises(ValueError, match="only be composed"):
        _spec(cell, composed=False)               # layer 1: the type refuses

    scraped = _spec(cell, composed=True)          # layer 2: forced past the type
    object.__setattr__(scraped, "render_mode", "whole_file")
    assert scraped.render_mode == "whole_file"

    report = audit_specs([scraped])
    passed, why = report.results["I7a_cells_6_7_always_composed"]
    assert not passed, why
    assert "1 spec(s)" in why, why
    assert audit_specs([_spec(cell, composed=True)]).results[
        "I7a_cells_6_7_always_composed"][0], "a composed cell 6/7 is fine"


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
    """f8 = 0 is primary precisely because it does.

    Critical: the quantity the docs mean by "~13.8% of the corpus genuine
    whole-file audio" is the MIXED-stratum whole-file mass: cells 5 (natural
    songs) and 8 (AI songs), the two `f8` decides. The earlier version measured
    `len(whole) / len(specs)` -- 0.588, because cells 1/2/3/4/9 are whole-file
    at `a = b = 0`, an unrelated knob. It cleared its 0.10 bar six times over
    while saying nothing about the policy, and would have kept clearing it with
    the cell-5/8 mass at zero.
    """
    specs = list(Sampler(manifest, SamplerConfig(f8=0.0)).epoch_specs(N))
    whole = [s for s in specs if s.render_mode == "whole_file"]
    genuine = [s for s in whole if s.cell in (5, 8)]
    mass = len(genuine) / len(specs)
    assert 0.115 <= mass <= 0.160, (
        f"cells 5+8 whole-file mass is {mass:.4f}; docs/data/02 says ~0.138")
    assert any(s.cell == 8 for s in genuine), "AI songs must be usable as-is"
    assert any(s.cell == 5 for s in genuine), "natural songs must be usable as-is"

    # Caveat: and it is `f8` that produces it: the strict endpoint drives the
    # same quantity to exactly zero. Without this, `mass` could come from
    # anywhere.
    strict = list(Sampler(manifest, SamplerConfig(f8=1.0)).epoch_specs(3_000))
    assert not [s for s in strict
                if s.render_mode == "whole_file" and s.cell in (5, 8)]


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
    """Critical: cell 9 is always REAL and never composed, so it contributes an
    unfixable "not composed => REAL" correlation.

    Including it, the strict policy fails the gate -- a false alarm blocking a
    policy we deliberately support.

    Caveat: measured with the audit's OWN feature matrix and AUC estimator. The
    earlier version pasted a copy of `_feature_frame`'s feature list inline and
    fit its own in-sample logistic regression. When the audit dropped
    `source_offset_s` and moved to cross-validated AUC, this test kept passing
    against the stale copy -- it was testing a reimplementation, not the audit.
    """
    from training.audit import _auc_or_none, _feature_frame

    specs = list(Sampler(manifest, SamplerConfig(f8=1.0)).epoch_specs(8000))

    def pooled_auc(pool):
        X, _ = _feature_frame(pool)
        return _auc_or_none(X, np.array([s.file_fake for s in pool]))

    with_9 = pooled_auc(specs)
    without_9 = pooled_auc([s for s in specs if s.cell != 9])
    assert with_9 >= SHORTCUT_AUC_GATE, (
        f"including cell 9 the strict policy must trip the gate, got {with_9:.4f}")
    assert without_9 < SHORTCUT_AUC_GATE - 0.04, (
        f"cells 1-8 must have headroom, got {without_9:.4f}")

    # ...and the audit is on the right side of that line, because it excludes it.
    passed, why = _metadata_shortcut(specs)
    assert passed, why
    assert f"gate < {SHORTCUT_AUC_GATE}" in why, why


def test_I1b_catches_a_duration_shortcut():
    """Nothing else in the audit looks at duration."""
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
    passed, why = audit_specs(specs).results["I1b_metadata_shortcut_auc"]
    assert not passed, why


# --------------------------------------------------------------------------- #
# Critical regression tests: three leaks that passed the whole audit clean

def _retimed(spec, seconds):
    k = seconds / spec.duration_s
    return dataclasses.replace(spec, duration_s=seconds, components=tuple(
        dataclasses.replace(c, duration_s=c.duration_s * k,
                            target_start_s=c.target_start_s * k)
        for c in spec.components))


LEAKS = {
    # A name-only frequency table is blind to this: the NAME is on both sides at
    # exactly equal rate, so I1 reports 0.0000 while the parameter is decisive.
    "transform_parameter": lambda s, f: dataclasses.replace(
        s, transforms=(("rawboost", {"strength": 0.9 if f else 0.1}),)),
    # `normalize` is the A-S1 test-chain draw and was read by nothing at all.
    "normalize_container": lambda s, f: dataclasses.replace(
        s, normalize={"container": "mp3" if f else "wav"}),
    # Cancels marginally, points opposite ways inside two strata. The model gets
    # the interaction for free because it is trained to predict presence.
    "duration_x_stratum": lambda s, f: _retimed(s, float(np.clip(
        s.duration_s + (14 if s.stratum == "mixed" else -14) * (1 if f else -1),
        4.0, 60.0))),
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_audit_catches_each_leak_that_once_passed(manifest, specs, leak):
    """Every one of these passed the entire audit clean before I1b was hardened."""
    trapped = [LEAKS[leak](s, s.file_fake) for s in specs]
    report = audit_specs(trapped, manifest=manifest, slice_="train")
    assert not report.ok, f"{leak} is invisible to the audit"
    assert "I1b_metadata_shortcut_auc" in report.failures, report.failures


def test_audit_catches_all_three_leaks_together(manifest, specs):
    trapped = list(specs)
    for fn in LEAKS.values():
        trapped = [fn(s, s.file_fake) for s in trapped]
    assert not audit_specs(trapped, manifest=manifest, slice_="train").ok


def test_the_name_only_check_is_genuinely_blind_to_a_parameter(manifest, specs):
    """Critical: pins why I1 is not sufficient, so nobody deletes I1b as redundant."""
    trapped = [LEAKS["transform_parameter"](s, s.file_fake) for s in specs]
    report = audit_specs(trapped, manifest=manifest, slice_="train")
    passed, why = report.results["I1_transform_name_independence"]
    assert passed, "the transform NAME is balanced -- that is the trap"
    assert "0.0000" in why
    assert not report.results["I1b_metadata_shortcut_auc"][0], "I1b must catch it"


def test_reference_stream_keeps_headroom_under_the_stricter_probe(manifest):
    """The per-stratum probe must not manufacture false alarms."""
    for f8 in (0.0, 1.0):
        report = run_audit(Sampler(manifest, SamplerConfig(f8=f8)), n=N, manifest=manifest)
        passed, why = report.results["I1b_metadata_shortcut_auc"]
        assert passed, f"f8={f8}: {why}"


def test_audit_is_stable_across_draw_budgets_and_manifests(manifest):
    """Critical: A guardrail that cries wolf gets switched off.

    Two false alarms were found this way: `source_offset_s` as a feature (the
    model cannot observe where in a source file we started reading, and with a
    small whole-file pool its range separated the labels at AUC 0.764), and I3
    counting components drawn only once (a file drawn once cannot appear on
    both sides -- that measures the draw budget, not the sampler).

    Critical: **And it must vary the SEED.** An earlier version swept manifest
    size, policy and draw budget -- every axis except the one that produces
    flakiness. Both `synthetic_manifest(seed=0)` and `run_audit(seed=0)`
    default to the same seed, so "every test draws seed 0" is a property of
    this harness rather than a habit of any one test. The I2 tolerance bug
    tripped 6 of 16 seeds in the shipped config and this guard could not see
    it.

    Caveat: it also asserts on `skipped`, not only on `ok`: `.ok` counts a SKIP
    as a pass, so without this the guard would green on a check's *absence*.
    """
    from training.synthetic import synthetic_manifest
    expected_skips = {"I7_eval_size_floors"}
    for npp, nwf in ((60, 60), (200, 200)):
        for mseed in (0, 1):
            m = synthetic_manifest(n_per_pool=npp, n_whole_file=nwf, seed=mseed)
            for f8 in (0.0, 1.0):
                for n in (1500, 6000):
                    for seed in (0, 1, 2, 3):
                        report = run_audit(Sampler(m, SamplerConfig(f8=f8)), n=n,
                                           manifest=m, seed=seed)
                        where = f"{npp}/{nwf} mseed={mseed} f8={f8} n={n} seed={seed}"
                        assert report.ok, f"{where}: {report.failures}"
                        assert set(report.skipped) <= expected_skips, (
                            f"{where}: unexpected SKIP {set(report.skipped)} -- a "
                            f"guard that greens on a check's absence is not a guard")


def test_source_offset_is_not_a_feature():
    """It is not observable by the model, so it cannot be a shortcut."""
    from training.audit import _feature_frame
    _, names = _feature_frame([_spec(1, sample_id=i) for i in range(50)])
    assert not any("offset" in n for n in names), names


# --------------------------------------------------------------------------- #
# Critical -- review findings 2 and 4: checks that could not fail, and one that
# lied

def test_I4_is_falsifiable(manifest):
    """The old I4 compared `s.file_fake` to the expression that defines it.

    A brute force over all nine cells found 0 constructible specs that could
    trip it. It now cross-checks against the component's POOL, an independent
    source: pool B is fake voice, so it must not back a cell-1 (voice REAL)
    sample.
    """
    real_voice = str(manifest[manifest.pool == "A"].file_id.iloc[0])
    fake_voice = str(manifest[manifest.pool == "B"].file_id.iloc[0])

    def composed_cell1(sample_id, file_id):
        return SampleSpec(
            sample_id=sample_id, epoch=0, seed=0, scheme_version="v1",
            duration_s=10.0, cell=1, render_mode="composed", structure="overlap",
            components=(ComponentDraw(file_id=file_id, role="voice",
                                      source_offset_s=0.0, duration_s=10.0,
                                      target_start_s=0.0, gain_db=0.0),))

    clean = [composed_cell1(i, real_voice) for i in range(300)]
    assert audit_specs(clean, manifest=manifest,
                       slice_="train").results["I4_component_pools_imply_the_labels"][0]

    trapped = [composed_cell1(i, fake_voice) for i in range(300)]
    passed, why = audit_specs(trapped, manifest=manifest,
                              slice_="train").results["I4_component_pools_imply_the_labels"]
    assert not passed, "a fake-voice component backing cell 1 must be caught"
    assert "cell 1" in why, why


def test_I6_is_gone_not_silently_passing():
    """"Labels come from the cell" is a property of the TYPE, not of a stream.

    It is asserted in tests/test_spec.py (SampleSpec has no label fields). A
    per-spec loop over derived properties can only ever pass.
    """
    report = audit_specs([_spec(1, sample_id=i) for i in range(300)])
    assert not any(k.startswith("I6") for k in report.results), report.results.keys()


def test_I7_reports_a_skip_not_a_pass_for_the_size_floors(manifest, specs):
    """Critical: it used to print PASS for a check implemented nowhere."""
    report = audit_specs(specs, manifest=manifest, slice_="train")
    assert "I7_eval_size_floors" in report.skipped
    assert "PASS  I7_eval_size_floors" not in str(report)
    assert "SKIP  I7_eval_size_floors" in str(report)


def test_I7_size_floors_run_when_asked(manifest, specs):
    """VG1 A8/A9: >=1,200 per class per masked pool, >=100 per cell."""
    # 6k specs clear the floors; a thin stream must not.
    ok = audit_specs(specs, manifest=manifest, slice_="train", eval_floors=True)
    assert ok.results["I7_eval_size_floors"][0], ok.results["I7_eval_size_floors"][1]

    thin = audit_specs(specs[:800], manifest=manifest, slice_="train", eval_floors=True)
    passed, why = thin.results["I7_eval_size_floors"]
    assert not passed, "800 specs cannot meet a 1,200-per-class floor"
    assert "VG1 A8/A9" in why


def test_I5_checks_the_fold_not_only_the_slice(manifest):
    """Documented I5 is slice AND the grouping keys; it implemented only slice."""
    import numpy as np
    m = manifest.copy()
    rng = np.random.default_rng(0)
    m["fold"] = rng.integers(0, 2, len(m))
    specs = list(Sampler(m, fold=0).epoch_specs(400))
    # audited against the OTHER fold: every draw is now out of bounds
    report = audit_specs(specs, manifest=m, slice_="train", fold=1)
    assert not report.results["I5_split_safety"][0]
    assert "fold=1" in report.results["I5_split_safety"][1]


def test_I21_catches_generator_monoculture(manifest):
    """Critical: domain_cap is a weight over what is PRESENT.

    A fold split leaving TRAIN generator-poor reproduces the DOSS failure with a
    green audit. Nothing measured realized family diversity before.
    """
    mono = manifest.copy()
    keep = mono.artifact_family.isin(["hifigan", "suno_v3"]) | mono.artifact_family.isna()
    mono.loc[~keep, "slice"] = "val"
    report = run_audit(Sampler(mono), n=4000, manifest=mono)
    assert not report.ok
    assert "I21_generator_diversity" in report.failures
    assert "POOR" in report.results["I21_generator_diversity"][1]


def test_I21_passes_on_a_diverse_slice(manifest):
    report = run_audit(Sampler(manifest), n=4000, manifest=manifest)
    passed, why = report.results["I21_generator_diversity"]
    assert passed, why
    assert "effective" in why, "the measured diversity must be visible in the report"


def test_I3_can_actually_fail(manifest):
    """Critical: I3 measured 400/400 = 1.000 and had no mutation test.

    With a = b = 0 every drawn pool-A/C file necessarily lands in both cell 5
    and cell 6/7, so the check is structurally saturated. Prove it can go red.
    """
    real_a = str(manifest[manifest.pool == "A"].file_id.iloc[0])

    def one_sided(cell, sample_id):
        return SampleSpec(
            sample_id=sample_id, epoch=0, seed=0, scheme_version="v1",
            duration_s=10.0, cell=cell, render_mode="composed", structure="overlap",
            components=(ComponentDraw(file_id=real_a, role="voice",
                                      source_offset_s=0.0, duration_s=10.0,
                                      target_start_s=0.0, gain_db=0.0),))

    # 30 distinct real files, each drawn twice, each only ever in a REAL sample
    ids = [str(x) for x in manifest[manifest.pool == "A"].file_id.head(30)]
    specs = []
    for k, fid in enumerate(ids):
        for rep in range(2):
            specs.append(dataclasses.replace(one_sided(1, 2 * k + rep), components=(
                dataclasses.replace(one_sided(1, 0).components[0], file_id=fid),)))
    passed, why = audit_specs(specs, manifest=manifest,
                              slice_="train").results["I3_real_components_on_both_sides"]
    assert not passed, why
    assert "0.000" in why or "0/1" in why, why


def test_I2c_catches_the_marginal_composedness_residual(manifest):
    """Measured, not asserted in a docstring.

    Without balancing, cell 9 (always REAL, never composed) skews "not composed"
    toward REAL: 0.127 marginally, against 0.0086 over cells 1-8.
    """
    unbalanced = SamplerConfig(balance_marginal_composedness=False)
    report = run_audit(Sampler(manifest, unbalanced), n=8000, manifest=manifest)
    assert not report.results["I2c_marginal_composedness"][0]
    assert report.results["I2_stratified_composedness"][0], \
        "the stratified constraint still holds -- that is the point"

    balanced = run_audit(Sampler(manifest, SamplerConfig()), n=8000, manifest=manifest)
    assert balanced.results["I2c_marginal_composedness"][0]


def test_C2_floor_defaults_to_8_not_the_pathological_value():
    """docs/pipelines/02 §4: n=2 carries 3.9x the gradient norm of n=32.

    A floor of 2 accepts exactly the case C2 exists to prevent.
    """
    import inspect
    from training.audit import audit_specs as fn
    assert inspect.signature(fn).parameters["min_present"].default == 8
