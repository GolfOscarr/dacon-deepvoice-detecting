"""The three registries, and the guarantees that are supposed to be structural.

Critical: every check here is mutation-tested: for each invariant there is a
paired test that breaks the thing deliberately and proves the check fires. A
review of this repo found three checks that could not fail -- a brute force
over all nine cells produced 0 specs able to trip one of them -- so "the test
passes" is not evidence until "the broken version fails" is also evidence.
"""

import inspect

import numpy as np
import pytest
import torch

from training.registries import (AUGMENT, FILTER, PREPROCESS, LABEL_PARAM_NAMES,
                                 Registry, RegistryError, Verdict,
                                 augment_chain, preprocess_chain)

SR = 16_000


# --------------------------------------------------------------------------- #
# I14 -- rule 2.4 over the preprocess registry
#
# "Every registered preprocess step: output over a row's valid prefix is bitwise
# identical solo and inside a batch of longer, louder rows." The existing
# `tests/test_audio.py::test_filter_output_is_independent_of_padding` covers
# `bandpass` alone; docs/pipelines/05 §5 asks for the pattern generalised to the
# registry, which is what `_batch_invariance_gap` is.


def _batch_invariance_gap(step, n: int = 977, total: int = 3_000) -> float:
    """max |solo - in-batch| over the row's valid prefix. Must be exactly 0.

    The batch is deliberately hostile: the row under test is padded with **loud
    non-zero garbage** (two pad fillings must give the same answer), and it sits
    beside rows that are longer and ~30 dB louder.
    """
    rng = np.random.default_rng(0)
    row = torch.from_numpy(rng.standard_normal(n).astype(np.float32))

    solo = step(row[None, :].clone(), SR, torch.tensor([n]))[0, :n]

    batch = torch.from_numpy(
        (30.0 * rng.standard_normal((4, total))).astype(np.float32))
    batch[0, :n] = row
    lengths = torch.tensor([n, total, total - 1, total // 2])
    inside = step(batch, SR, lengths)[0, :n]
    return float((solo - inside).abs().max())


@pytest.mark.parametrize("name", PREPROCESS.names())
def test_preprocess_step_is_batch_invariant(name):
    """I14. A file's output must not depend on what else is in the batch."""
    assert _batch_invariance_gap(PREPROCESS.build(name)) == 0.0


def test_the_batch_invariance_check_can_fail():
    """Critical: mutation test for I14 itself.

    `models/audio.py::bandpass` shipped this defect with a docstring asserting
    the opposite, and a green suite. A batch-dependent step -- the mean taken
    over the *padded* row instead of the valid prefix -- must make the check
    above fail, or the check is decoration.
    """
    broken = Registry("preprocess", ("wav", "sample_rate", "lengths"), frozenset())

    @broken.register("dc_offset_over_the_padded_row")
    def _bad(wav, sample_rate, lengths):
        return wav - wav.mean(dim=-1, keepdim=True)

    assert _batch_invariance_gap(broken.build("dc_offset_over_the_padded_row")) > 0.0


def test_a_step_that_reads_another_row_is_caught():
    """The other rule-2.4 shape: a batch statistic rather than a padding leak."""
    broken = Registry("preprocess", ("wav", "sample_rate", "lengths"), frozenset())

    @broken.register("per_batch_normalisation")
    def _bad(wav, sample_rate, lengths):
        return wav / wav.std()

    assert _batch_invariance_gap(broken.build("per_batch_normalisation")) > 0.0


@pytest.mark.parametrize("name", PREPROCESS.names())
def test_preprocess_is_deterministic_and_length_preserving(name):
    step = PREPROCESS.build(name)
    x = torch.from_numpy(
        np.random.default_rng(1).standard_normal((3, 500)).astype(np.float32))
    lengths = torch.tensor([500, 400, 123])
    first, second = step(x.clone(), SR, lengths), step(x.clone(), SR, lengths)
    assert torch.equal(first, second)
    assert first.shape == x.shape


def test_preprocess_leaves_the_padding_region_alone():
    """A step that wrote into the pad would make `lengths` a lie downstream."""
    x = torch.zeros(1, 100)
    x[0, 50:] = 7.0
    out = PREPROCESS.build("dc_offset")(x, SR, torch.tensor([50]))
    assert torch.equal(out[0, 50:], torch.full((50,), 7.0))


def test_dc_offset_removes_the_offset_it_claims_to():
    """Assert on the quantity that ships, not on an adjacent one."""
    x = torch.ones(1, 128) * 3.0 + torch.sin(torch.arange(128.0))
    out = PREPROCESS.build("dc_offset")(x, SR, torch.tensor([128]))
    assert abs(float(out[0].mean())) < 1e-6
    assert abs(float(x[0].mean())) > 1e-6          # the input really had one


def test_pre_emphasis_matches_its_own_difference_equation():
    x = torch.from_numpy(
        np.random.default_rng(2).standard_normal((1, 64)).astype(np.float32))
    out = PREPROCESS.build("pre_emphasis", {"coeff": 0.5})(x, SR, torch.tensor([64]))
    assert float(out[0, 0]) == float(x[0, 0])
    assert torch.allclose(out[0, 1:], x[0, 1:] - 0.5 * x[0, :-1], atol=1e-6)


def test_preprocess_rejects_lengths_longer_than_the_tensor():
    with pytest.raises(ValueError, match="exceed"):
        PREPROCESS.build("dc_offset")(torch.zeros(1, 10), SR, torch.tensor([11]))


# --------------------------------------------------------------------------- #
# The augment contract -- label independence, enforced by the signature


def test_a_bound_augment_takes_exactly_wav_and_rng():
    """Critical: `P(T | L) = P(T)` is a property of the type, not of a code review.

    Whatever the caller knows about the sample, the composed augment has no
    argument to put it in.
    """
    chain = augment_chain((("gain_jitter", {"db": 3.0}),))
    assert list(inspect.signature(chain).parameters) == ["wav", "rng"]
    with pytest.raises(TypeError):
        chain(torch.zeros(1, 10), np.random.default_rng(0), 1)      # a label


@pytest.mark.parametrize("name", AUGMENT.names())
def test_every_registered_augment_has_the_bare_signature(name):
    params = inspect.signature(AUGMENT.get(name)).parameters
    positional = [p.name for p in params.values()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == ["wav", "rng"]
    assert not set(params) & LABEL_PARAM_NAMES


def test_registering_an_augment_that_takes_a_label_is_refused():
    """Critical mutation test: the enforcement must actually refuse something."""
    reg = Registry("augment", ("wav", "rng"), LABEL_PARAM_NAMES)

    with pytest.raises(RegistryError, match="positionally"):
        @reg.register("peeks")
        def _peeks(wav, rng, file_fake):
            return wav * (2.0 if file_fake else 1.0)

    with pytest.raises(RegistryError, match="forbidden"):
        @reg.register("peeks_by_keyword")
        def _peeks_kw(wav, rng, *, file_fake=0):
            return wav * (2.0 if file_fake else 1.0)

    with pytest.raises(RegistryError, match=r"\*\*"):
        @reg.register("peeks_by_kwargs")
        def _peeks_kwargs(wav, rng, **anything):
            return wav
    assert len(reg) == 0


def test_an_augment_parameter_must_have_a_default():
    reg = Registry("augment", ("wav", "rng"), LABEL_PARAM_NAMES)
    with pytest.raises(RegistryError, match="needs a default"):
        @reg.register("half_built")
        def _half(wav, rng, *, db):
            return wav


def test_building_with_an_unknown_param_is_an_error():
    """A typo'd knob that silently does nothing is how an ablation lies."""
    with pytest.raises(RegistryError, match="no parameter"):
        AUGMENT.build("gain_jitter", {"dB": 3.0})


def test_a_registered_name_cannot_be_shadowed():
    with pytest.raises(RegistryError, match="already registered"):
        AUGMENT.register("gain_jitter")(lambda wav, rng: wav)


def test_gain_jitter_applies_the_gain_it_was_given():
    x = torch.ones(1, 16)
    out = AUGMENT.build("gain_jitter", {"db": -20.0})(x, np.random.default_rng(0))
    assert float(out[0, 0]) == pytest.approx(0.1, abs=1e-6)


def test_augments_are_reproducible_from_the_rng_alone():
    """A-S2: the same key must give byte-identical audio."""
    x = torch.from_numpy(
        np.random.default_rng(3).standard_normal((2, 4000)).astype(np.float32))
    chain = augment_chain((("gaussian_noise", {"snr_db": 15.0}),
                           ("rawboost_ssi", {}),
                           ("stereo_imbalance", {})))
    a = chain(x.clone(), np.random.default_rng(7))
    b = chain(x.clone(), np.random.default_rng(7))
    c = chain(x.clone(), np.random.default_rng(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)                   # the rng is actually consumed


def test_gaussian_noise_lands_on_the_snr_it_was_asked_for():
    x = torch.from_numpy(
        np.random.default_rng(4).standard_normal((1, 40_000)).astype(np.float32))
    out = AUGMENT.build("gaussian_noise", {"snr_db": 20.0})(x, np.random.default_rng(0))
    noise = out - x
    measured = 20 * np.log10(float(x.pow(2).mean().sqrt() / noise.pow(2).mean().sqrt()))
    assert measured == pytest.approx(20.0, abs=0.5)


def test_stereo_imbalance_is_a_no_op_on_mono():
    """It must not invent a second channel: the channel policy is downstream."""
    x = torch.ones(1, 32)
    out = AUGMENT.build("stereo_imbalance")(x, np.random.default_rng(0))
    assert torch.equal(out, x)
    stereo = torch.ones(2, 32)
    out = AUGMENT.build("stereo_imbalance")(stereo, np.random.default_rng(0))
    assert float(out[0, 0]) != float(out[1, 0])


def test_the_menu_entries_that_would_desynchronise_frame_targets_are_absent():
    """Caveat: A-A8 time shift and A-A11 silence edits move audio along the timeline.

    An augment returns only a waveform, so it cannot tell the renderer the
    timeline moved, and I13 -- "frame targets and audio describe the same
    timeline" -- would break silently. They belong in the *placement*
    (`ComponentDraw.target_start_s`), where the spec records them.
    """
    assert "time_shift" not in AUGMENT
    assert "silence_pad" not in AUGMENT


# --------------------------------------------------------------------------- #
# Critical: the time-invariance contract
#
# > Steps 4-5 are time-invariant. Every time-warping decision lives in the draw.
#
# Registration *measures* it. These tests exercise one member of each of the four
# warp classes (docs/pipelines/03 §4) and show it being refused -- a probe that
# passes everything is the "check that cannot fail" this file exists to avoid.


def _fresh(kind="preprocess"):
    if kind == "preprocess":
        return Registry("preprocess", ("wav", "sample_rate", "lengths"),
                        frozenset(), time_invariant=True)
    return Registry("augment", ("wav", "rng"), frozenset(), time_invariant=True)


def test_class_1_an_undeclared_rigid_shift_cannot_be_registered():
    """A-A8's shape. This is the `align_time` defect at the plugin boundary."""
    reg = _fresh()
    with pytest.raises(RegistryError, match="declares group_delay=0"):
        @reg.register("shifts")
        def _shifts(wav, sample_rate, lengths):
            return torch.roll(wav, 37, dims=-1)
    assert "shifts" not in reg


def test_a_declared_group_delay_is_checked_against_the_measurement():
    """Critical: A declaration the probe believes would be worthless. Declare 37 and
    shift by 37 and it registers; declare 37 and shift by 12 and it does not."""
    reg = _fresh()

    @reg.register("honest", group_delay=37)
    def _honest(wav, sample_rate, lengths):
        return torch.roll(wav, 37, dims=-1)

    assert reg.group_delay_of("honest") == 37

    with pytest.raises(RegistryError, match="declares group_delay=37"):
        @reg.register("liar", group_delay=37)
        def _liar(wav, sample_rate, lengths):
            return torch.roll(wav, 12, dims=-1)


def test_class_2_a_rate_change_cannot_be_registered():
    """A-C2 time stretch: the head and the tail move by different amounts, so no
    single delay describes it and `frame_intervals` cannot follow it."""
    reg = _fresh()
    with pytest.raises(RegistryError, match="warped the time base"):
        @reg.register("stretches")
        def _stretches(wav, sample_rate, lengths):
            n = wav.shape[-1]
            src = np.arange(n) * 1.01
            out = np.stack([np.interp(np.arange(n), src, row)
                            for row in wav.numpy()])
            return torch.from_numpy(out.astype(np.float32))


def test_class_3_a_non_monotonic_edit_cannot_be_registered():
    """A-A11's trimming shape: excise 200 samples from the middle and pad the
    tail. The length is unchanged and the head never moves -- only the two-window
    comparison sees it."""
    reg = _fresh()
    with pytest.raises(RegistryError, match="warped the time base"):
        @reg.register("excises")
        def _excises(wav, sample_rate, lengths):
            n = wav.shape[-1]
            keep = torch.cat([wav[:, :n // 2], wav[:, n // 2 + 200:]], dim=-1)
            return torch.cat([keep, torch.zeros(wav.shape[0], 200)], dim=-1)


def test_class_4_group_delay_is_measured_not_assumed():
    """Critical: the class nobody lists. A-A10 RIR convolution shifts by its direct-path
    offset; `bandpass` is safe only because it is zero-phase and `resample_poly`
    only because it is linear phase. Those are load-bearing accidents until
    something measures them."""
    tail = 0.3 * np.random.default_rng(1).standard_normal(200) * np.exp(-np.arange(200) / 50)

    def convolve(wav, rir):
        n = wav.shape[-1]
        out = np.stack([np.convolve(row, rir)[:n] for row in wav.numpy()])
        return torch.from_numpy(out.astype(np.float32))

    late = np.concatenate([np.zeros(20), [1.0], tail])
    reg = _fresh()
    with pytest.raises(RegistryError, match="shifts audio by \\+20"):
        @reg.register("rir_late")
        def _late(wav, sample_rate, lengths):
            return convolve(wav, late)

    # ...and a filter whose direct path is at zero is fine. The contract is
    # "does not move audio", not "is not a filter".
    reg2 = _fresh()

    @reg2.register("rir_aligned")
    def _aligned(wav, sample_rate, lengths):
        return convolve(wav, np.concatenate([[1.0], tail]))

    assert reg2.group_delay_of("rir_aligned") == 0


def test_a_length_change_is_refused_and_named_as_a_draw():
    reg = _fresh()
    with pytest.raises(RegistryError, match="changed the sample count by \\+800"):
        @reg.register("pads")
        def _pads(wav, sample_rate, lengths):
            return torch.cat([torch.zeros(wav.shape[0], 800), wav], dim=-1)


def test_an_augment_may_not_even_declare_a_delay():
    """Caveat: A delay an augment wanted is a *draw*. Letting it declare one would
    reopen the case-by-case tracking the contract exists to close."""
    reg = _fresh("augment")
    with pytest.raises(RegistryError, match="cannot move audio in time"):
        @reg.register("shifts", group_delay=8)
        def _shifts(wav, rng):
            return torch.roll(wav, 8, dims=-1)


@pytest.mark.parametrize("name", AUGMENT.names())
def test_every_registered_augment_declares_no_delay(name):
    assert AUGMENT.group_delay_of(name) == 0


@pytest.mark.parametrize("name", PREPROCESS.names())
def test_every_shipped_preprocess_step_is_zero_delay_today(name):
    """Recorded rather than assumed: `preprocess_chain(...).group_delay` is the
    number a caller would have to compensate, and it is 0."""
    assert PREPROCESS.group_delay_of(name) == 0


def test_a_chain_reports_the_delay_it_would_impose():
    chain = preprocess_chain((("dc_offset", {}), ("pre_emphasis", {})))
    assert chain.group_delay == 0


def test_the_runtime_guard_catches_a_warp_a_default_probe_cannot_see():
    """Caveat: registration probes with *default* params, so a warp that only a drawn
    parameter turns on gets past it. `augment_chain` re-checks on the real audio.
    A step like this is exactly how A-A11 would come back."""
    reg = Registry("augment", ("wav", "rng"), frozenset(), time_invariant=True)

    @AUGMENT.register("__pads_only_when_asked")
    def _sneaky(wav, rng, *, pad=0):
        return torch.cat([wav, torch.zeros(wav.shape[0], pad)], dim=-1)

    try:
        chain = augment_chain((("__pads_only_when_asked", {"pad": 160}),))
        with pytest.raises(RegistryError, match="step 4 is time-invariant"):
            chain(torch.zeros(1, 1000), np.random.default_rng(0))
        # ...and the same step with its default is untouched.
        ok = augment_chain((("__pads_only_when_asked", {}),))
        assert ok(torch.zeros(1, 1000), np.random.default_rng(0)).shape == (1, 1000)
    finally:
        AUGMENT._fns.pop("__pads_only_when_asked", None)
        AUGMENT._params.pop("__pads_only_when_asked", None)
        AUGMENT._group_delay.pop("__pads_only_when_asked", None)
    assert len(reg) == 0


# --------------------------------------------------------------------------- #
# The filter contract -- offline, sidecar, never audio


def test_a_filter_cannot_be_given_audio():
    """Critical, G5: nothing in this registry rewrites or deletes audio. It is not
    handed any, and registration refuses a function that asks for some."""
    reg = Registry("filter", ("manifest_row", "quality_row"),
                   frozenset({"wav", "audio"}))
    with pytest.raises(RegistryError, match="positionally"):
        @reg.register("mutates")
        def _mutates(manifest_row, quality_row, wav):
            return Verdict("keep")

    with pytest.raises(RegistryError, match="forbidden"):
        @reg.register("mutates_by_keyword")
        def _mutates_kw(manifest_row, quality_row, *, wav=None):
            return Verdict("keep")


@pytest.mark.parametrize("name", FILTER.names())
def test_registered_filters_take_two_metadata_rows(name):
    params = inspect.signature(FILTER.get(name)).parameters
    positional = [p.name for p in params.values()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == ["manifest_row", "quality_row"]


def test_a_verdict_outside_the_three_is_refused():
    with pytest.raises(ValueError, match="verdict must be"):
        Verdict("delete")


def test_corruption_drops_only_what_carries_no_evidence():
    row = {"file_id": "x", "duration_s": 10.0}
    assert FILTER.build("corruption")(row, {"decode_ok": True}).action == "keep"
    assert FILTER.build("corruption")(row, {"decode_ok": False}).action == "drop"
    assert FILTER.build("corruption")(row, {"all_silent": True}).action == "drop"


def test_a_noisy_file_is_not_dropped_for_being_noisy():
    """Critical, P1: noise is a property of the target domain, not a defect. Our test
    set contains 전화채널 audio by construction."""
    telephone = {"file_id": "x", "duration_s": 12.0}
    quality = {"decode_ok": True, "snr_db": 2.0, "component_snr_db": 2.0,
               "effective_bandwidth_hz": 3400, "longest_valid_span_s": 12.0}
    for name in FILTER.names():
        if name == "label_evidence":
            verdict = FILTER.build(name, {"min_component_snr_db": -5.0,
                                          "threshold_version": "t1"})(telephone, quality)
        else:
            verdict = FILTER.build(name)(telephone, quality)
        assert verdict.keeps, f"{name} dropped a legitimate telephone file"


def test_label_evidence_refuses_to_run_on_an_unapproved_threshold():
    """Critical, docs/data/10 §6: no threshold is hardcoded from intuition, and G3 has
    to have approved it. A default here would be invisible in a green suite."""
    row, quality = {"file_id": "x"}, {"component_snr_db": -20.0}
    with pytest.raises(RegistryError, match="G3"):
        FILTER.build("label_evidence")(row, quality)
    with pytest.raises(RegistryError, match="G3"):
        FILTER.build("label_evidence", {"min_component_snr_db": 3.0})(row, quality)

    verdict = FILTER.build("label_evidence", {"min_component_snr_db": 3.0,
                                              "threshold_version": "v2026-01-a"})(row, quality)
    assert verdict.action == "quarantine"
    assert verdict.threshold_version == "v2026-01-a"


def test_a_drop_verdict_reports_the_quantity_that_produced_it():
    """A verdict is a sidecar annotation, so it has to carry enough to be
    revisited: which measurement crossed which line (docs/data/10 §1, §6)."""
    verdict = FILTER.build("usable_duration")({"file_id": "x"},
                                              {"longest_valid_span_s": 1.0})
    assert verdict.action == "drop" and not verdict.keeps
    assert "1.00s" in verdict.reason        # the measured quantity, not a flag


# --------------------------------------------------------------------------- #
# Chains


def test_a_preprocess_chain_applies_its_steps_in_order():
    x = torch.from_numpy(
        np.random.default_rng(5).standard_normal((1, 200)).astype(np.float32)) + 4.0
    lengths = torch.tensor([200])
    chained = preprocess_chain((("dc_offset", {}), ("pre_emphasis", {})))(x, SR, lengths)
    by_hand = PREPROCESS.build("pre_emphasis")(
        PREPROCESS.build("dc_offset")(x, SR, lengths), SR, lengths)
    assert torch.equal(chained, by_hand)
    assert _batch_invariance_gap(
        preprocess_chain((("dc_offset", {}), ("pre_emphasis", {})))) == 0.0


def test_an_empty_chain_is_the_identity():
    x = torch.ones(1, 10)
    assert torch.equal(preprocess_chain(())(x, SR, None), x)
    assert torch.equal(augment_chain(())(x, np.random.default_rng(0)), x)


def test_an_unknown_step_name_says_what_is_registered():
    with pytest.raises(KeyError, match="registered"):
        AUGMENT.build("does_not_exist")
