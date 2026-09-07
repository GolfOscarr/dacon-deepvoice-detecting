"""`training.collate`, against real rendered audio.

🔴 Every invariant here is mutation-tested: a deliberately broken collator is
shown failing the check before the check is trusted on the real one. This repo
has shipped nine production defects and ten vacuous tests, *none* caught by a
green suite, and two shapes recurred -- a check that asserted on an **adjacent**
quantity, and a check nobody had ever seen fail.

⚠️ Where the quantity that ships is a probability, the assertion is on the
probability. `frame_max` was padding-sensitive in the submitted probability
while its test asserted on `clip_logits`, and that is exactly how it shipped.
"""

import numpy as np
import pytest
import torch

from models.audio import CHANNEL_POLICIES, prepare_waveform
from models.config import AudioConfig, load_model_config
from models.losses import TARGET_FOR_COLUMN
from models.model import DeepVoiceNet
from training.collate import (BATCH_KEYS, bucket_batches, bucket_edges,
                              bucket_of, collate, padding_fraction,
                              promote_channels, spec_durations)
from training.registries import PREPROCESS, preprocess_chain
from training.render import (ManifestIndex, RenderConfig, RenderedSample,
                             render)
from training.sampler import Sampler, SamplerConfig
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

SR = 16_000
CORPUS = dict(n_per_pool=6, n_whole_file=6, seed=0, duration_range=(7.0, 9.0))
DRAW = SamplerConfig(duration_range=(4.0, 6.0))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("collate-corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return root, manifest, ManifestIndex.from_frame(manifest)


@pytest.fixture(scope="module")
def cfg(corpus):
    return RenderConfig(root=corpus[0])


@pytest.fixture(scope="module")
def samples(corpus, cfg):
    """A handful of genuinely rendered samples -- mixed durations and channels."""
    _, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    return [render(spec, index, cfg)
            for spec in sampler.epoch_specs(8, epoch=0, seed=0)]


def _fake(wav: torch.Tensor, *, cell: int = 5, file_id: str = "X",
          sample_id: int = 0) -> RenderedSample:
    """A `RenderedSample` around a given waveform, with no I/O.

    Used where the point is the *collator*, not the renderer: batches no sampler
    would draw (all mono, one row 30 dB louder, no present component at all).
    """
    spec = SampleSpec(
        sample_id=sample_id, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=wav.shape[-1] / SR, cell=cell, render_mode="composed",
        structure="overlap",
        components=(ComponentDraw(file_id, "voice" if cell in (1, 2, 5, 6, 7, 8)
                                  else "music" if cell in (3, 4) else "noise",
                                  0.0, wav.shape[-1] / SR, 0.0, 0.0),))
    labels = spec.labels
    return RenderedSample(
        wav=wav, sample_rate=SR,
        targets={k: int(labels[k] or 0) for k in TARGET_FOR_COLUMN.values()},
        frame_intervals={"voice": (), "music": (), "file": ()}, spec=spec)


def _noise(n: int, channels: int = 1, scale: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        (scale * rng.standard_normal((channels, n))).astype(np.float32))


# --------------------------------------------------------------------------- #
# The contract of docs/pipelines/04 §2


def test_the_batch_is_exactly_the_five_documented_keys(samples):
    batch = collate(samples)
    assert set(batch) == set(BATCH_KEYS)
    b = len(samples)
    assert batch["wav"].dim() == 3 and batch["wav"].shape[0] == b
    assert batch["wav"].dtype == torch.float32
    assert batch["lengths"].shape == (b,) and batch["lengths"].dtype == torch.int64
    assert len(batch["frame_intervals"]) == b and len(batch["specs"]) == b


def test_the_target_keys_are_exactly_the_loss_keys(samples):
    """I18. A renamed target key must fail here, not at hour three of a run."""
    batch = collate(samples)
    assert set(batch["targets"]) == set(TARGET_FOR_COLUMN.values())
    for key, value in batch["targets"].items():
        assert value.shape == (len(samples),), key
        assert value.dtype == torch.float32, key


def test_lengths_are_each_rows_own_sample_count_not_the_padded_width(samples):
    """`lengths` is mandatory and never inferred (docs/pipelines/04 §2).

    The mutation is built into the assertion: the batch is deliberately ragged,
    so a collator that returned `S_max` for every row -- the plausible wrong
    thing -- disagrees on at least one row.
    """
    batch = collate(samples)
    assert [int(v) for v in batch["lengths"]] == [s.wav.shape[-1] for s in samples]
    s_max = int(batch["wav"].shape[-1])
    assert int(batch["lengths"].min()) < s_max, \
        "the fixture stopped being ragged, so this test stopped testing anything"


def test_each_rows_own_channels_appear_bitwise_at_offset_zero(samples):
    """🔴 The batch must contain each row's audio, unaltered and at offset 0.

    ⚠️ Sliced to `s.wav.shape[0]` -- the **sample's** channel count -- and never
    to `batch["wav"].shape[1]`, so this cannot be satisfied by restating the
    promotion rule. It is the content half of the contract; `lengths` is the
    extent half and the zero-padding test is the position half, and all three
    were needed: a collator that rolled the waveform (0.587), normalised per row
    (0.637), flipped the channels (0.583) or cast to bf16 (4.9e-4) passed every
    other test in this file.
    """
    batch = collate(samples)
    for i, sample in enumerate(samples):
        got = batch["wav"][i, :sample.wav.shape[0], :sample.wav.shape[-1]]
        assert torch.equal(got, sample.wav.to(torch.float32)), \
            f"row {i}: the batch does not contain the sample's own audio"


@pytest.mark.parametrize("corrupt,name", [
    (lambda w: torch.roll(w, 1_000, dims=-1), "rolled"),
    (lambda w: w / w.abs().max().clamp(min=1e-9), "per-row normalised"),
    (lambda w: w.flip(0), "channel-flipped"),
    (lambda w: w.to(torch.bfloat16).to(torch.float32), "bf16 round-tripped"),
])
def test_the_content_check_can_fail(samples, corrupt, name):
    """Mutation for the check above -- the four corruptions the reviewer found
    slipping through a suite that checked only shape, length and padding."""
    sample = samples[0]
    hurt = RenderedSample(wav=corrupt(sample.wav), sample_rate=sample.sample_rate,
                          targets=sample.targets,
                          frame_intervals=sample.frame_intervals, spec=sample.spec)
    batch = collate([hurt])
    got = batch["wav"][0, :sample.wav.shape[0], :sample.wav.shape[-1]]
    assert not torch.equal(got, sample.wav.to(torch.float32)), name


def test_the_padding_region_is_zero(samples):
    batch = collate(samples)
    for i, n in enumerate(batch["lengths"]):
        pad = batch["wav"][i, :, int(n):]
        assert pad.numel() == 0 or pad.abs().max() == 0.0


def test_frame_intervals_stay_absolute_seconds_and_are_not_rasterised(samples):
    """🔴 I13 at the collation boundary: seconds in, seconds out.

    A collator that rasterised to a frame grid would have to invent a frame
    rate, which is the `align_time` defect one layer up. The values must survive
    collation unchanged and stay inside `[0, duration_s]`, where `duration_s` is
    read back from `lengths` -- the quantity the model masks with.
    """
    batch = collate(samples)
    for i, (sample, got) in enumerate(zip(samples, batch["frame_intervals"])):
        assert got == sample.frame_intervals
        duration = int(batch["lengths"][i]) / SR
        for branch, spans in got.items():
            for start, end, label in spans:
                assert 0.0 <= start < end <= duration + 1e-6, (branch, start, end)
                assert label in (0, 1)


def test_the_frame_interval_check_can_fail(samples):
    """Mutation for the check above: an interval past the end must be caught."""
    bad = samples[0]
    over = RenderedSample(
        wav=bad.wav, sample_rate=bad.sample_rate, targets=bad.targets,
        frame_intervals={"voice": ((0.0, bad.duration_s + 1.0, 0),),
                         "music": (), "file": ()},
        spec=bad.spec)
    batch = collate([over])
    duration = int(batch["lengths"][0]) / SR
    start, end, _ = batch["frame_intervals"][0]["voice"][0]
    assert not (0.0 <= start < end <= duration + 1e-6)


# --------------------------------------------------------------------------- #
# I12 -- the length regime, asserted on min_seconds / max_seconds themselves


def test_every_collated_row_sits_in_the_length_regime(samples):
    """I12, on the quantity the model receives: `lengths / sample_rate`.

    ⚠️ Asserted against `AudioConfig.min_seconds` / `max_seconds` and nothing
    adjacent. A version of this test asserting `0 < duration` passed while
    2,580 of 4,000 samples were under the competition's own 4 s floor.
    """
    audio = AudioConfig()
    batch = collate(samples)
    durations = batch["lengths"].numpy() / SR
    assert durations.min() >= audio.min_seconds - 1e-6, durations.min()
    assert durations.max() <= audio.max_seconds + 1e-6, durations.max()


def test_the_length_regime_check_can_fail(corpus, cfg):
    """Mutation for I12: a short sample must be caught by the same assertion.

    Rendered with `check_duration=False` so the *renderer's* I12 does not fire
    first -- the point is that the collated batch is where a caller who disabled
    that check still sees it.
    """
    _, manifest, index = corpus
    voice = manifest[manifest.pool == "A"].file_id.iloc[0]
    audio = AudioConfig()
    short = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=1.5, cell=1, render_mode="composed", structure="overlap",
        components=(ComponentDraw(str(voice), "voice", 0.0, 1.5, 0.0, 0.0),))
    sample = render(short, index, RenderConfig(root=cfg.root, check_duration=False))
    durations = collate([sample])["lengths"].numpy() / SR
    assert not durations.min() >= audio.min_seconds - 1e-6


# --------------------------------------------------------------------------- #
# I15 -- rule 2.4 on the collator itself


def _batch_normalising_collate(samples, *, pad_value: float = 0.0):
    """The plausible wrong collator: peak-normalise the batch tensor.

    Every row still comes back at its own length with its own targets, so it
    passes any check that only looks at shapes -- which is why I15 has to be
    asserted on the samples themselves.
    """
    batch = collate(samples, pad_value=pad_value)
    batch["wav"] = batch["wav"] / batch["wav"].abs().max()
    return batch


def _prepared(batch, i, policy="downmix"):
    """The row as the model sees it: `prepare_waveform`, then its valid prefix.

    🔴 This, not `batch["wav"][i]`, is the quantity I15 protects. The raw row
    carries `C_max` channels, which is a property of the other rows -- see
    `promote_channels`. What must not move is what reaches `DeepVoiceNet`.
    """
    return prepare_waveform(batch["wav"], AudioConfig(channels=policy))[
        i, :int(batch["lengths"][i])]


@pytest.mark.parametrize("collate_fn,invariant", [(collate, True),
                                                  (_batch_normalising_collate, False)])
def test_a_row_is_collated_the_same_whatever_shares_its_batch(samples, collate_fn,
                                                              invariant):
    """I15 and its mutation in one parametrisation.

    The same sample is collated alone, then inside a batch of longer and ~30 dB
    louder rows. `wav` over its valid prefix, `lengths` and `targets` must be
    bitwise identical -- and the batch-normalising collator must make that fail,
    or the check is decoration.
    """
    under_test = samples[0]
    loud = [_fake(_noise(int(under_test.wav.shape[-1] * 1.7), scale=30.0, seed=s),
                  sample_id=100 + s) for s in range(3)]

    solo = collate_fn([under_test])
    inside = collate_fn([under_test, *loud])

    same = torch.equal(_prepared(solo, 0), _prepared(inside, 0))
    assert same is invariant
    if invariant:
        assert int(solo["lengths"][0]) == int(inside["lengths"][0])
        assert {k: float(v[0]) for k, v in solo["targets"].items()} == \
               {k: float(v[0]) for k, v in inside["targets"].items()}
        assert solo["frame_intervals"][0] == inside["frame_intervals"][0]


def test_the_position_in_the_batch_does_not_matter(samples):
    """The other half of I15: order, not just company."""
    a, b, c = samples[0], samples[1], samples[2]
    first = collate([a, b, c])
    last = collate([b, c, a])
    assert torch.equal(_prepared(first, 0), _prepared(last, 2))
    assert int(first["lengths"][0]) == int(last["lengths"][2])


# --------------------------------------------------------------------------- #
# The channel rule -- the rule-2.4 surface of `wav` being (B, C, S_max)


def _zero_filling_collate(samples, *, pad_value: float = 0.0):
    """The rejected channel rule: pad the channel axis with silence."""
    n = len(samples)
    channels = max(int(s.wav.shape[0]) for s in samples)
    s_max = max(int(s.wav.shape[-1]) for s in samples)
    wav = torch.full((n, channels, s_max), float(pad_value), dtype=torch.float32)
    for i, s in enumerate(samples):
        wav[i, :s.wav.shape[0], :s.wav.shape[-1]] = s.wav
        wav[i, s.wav.shape[0]:, :] = 0.0
    out = collate(samples, pad_value=pad_value)
    out["wav"] = wav
    return out


@pytest.mark.parametrize("policy", CHANNEL_POLICIES)
def test_promotion_is_invisible_at_the_model_boundary(policy):
    """🔴 The chosen channel rule, measured against every shipped policy.

    `CHANNEL_POLICIES` is enumerated from `models.config` rather than restated,
    so a new policy that promotion is *not* invariant under fails this test
    instead of shipping silently.
    """
    mono = _fake(_noise(SR * 4, channels=1, seed=1))
    stereo = _fake(_noise(SR * 5, channels=2, seed=2), sample_id=1)

    alone = collate([mono])
    mixed = collate([mono, stereo])
    assert alone["wav"].shape[1] == 1 and mixed["wav"].shape[1] == 2

    cfg = AudioConfig(channels=policy)
    a = prepare_waveform(alone["wav"], cfg)[0, :SR * 4]
    b = prepare_waveform(mixed["wav"], cfg)[0, :SR * 4]
    assert torch.equal(a, b), f"{policy}: promotion moved the model input"


def test_zero_filling_the_channel_axis_is_caught():
    """Mutation for the rule above, with the number that motivates it.

    Zero-filling halves a mono row's amplitude under `downmix`, because it gets
    averaged against a channel of silence that belongs to another file.
    """
    mono = _fake(_noise(SR * 4, channels=1, seed=1))
    stereo = _fake(_noise(SR * 5, channels=2, seed=2), sample_id=1)
    cfg = AudioConfig(channels="downmix")

    alone = prepare_waveform(collate([mono])["wav"], cfg)[0, :SR * 4]
    filled = prepare_waveform(_zero_filling_collate([mono, stereo])["wav"], cfg)[
        0, :SR * 4]
    assert not torch.equal(alone, filled)
    ratio = float((filled.abs().sum() / alone.abs().sum()))
    assert 0.49 < ratio < 0.51, f"expected the half-amplitude signature, got {ratio}"


def test_an_uneven_promotion_is_refused_rather_than_performed():
    """🔴 The one production defect the review found, and its number.

    Cyclic repeat is the identity at the model boundary only when the target is
    an exact multiple of the row's own channel count. 2 -> 3 gives `[L, R, L]`,
    which `downmix` averages to `(2L+R)/3` instead of `(L+R)/2` -- so a stereo
    row's audio changes because a 3-channel row shared its batch. That is the
    half-amplitude violation `promote_channels` rejects zero-fill for, arriving
    through the rule that replaced it.

    ⚠️ Latent today only because no corpus file has more than 2 channels, which
    is a property of the corpus and not of this code: `sf.read` uses
    `always_2d=True`, `render._to_channels` keeps the leading channels of a
    multi-channel source, and the `channels` normalize draw accepts `null`.
    """
    stereo = _noise(SR * 4, channels=2, seed=1)
    with pytest.raises(ValueError, match="not a multiple"):
        promote_channels(stereo, 3)

    # And the collator refuses the batch that would need it, naming both counts.
    with pytest.raises(ValueError, match=r"channel counts \[2\]"):
        collate([_fake(stereo), _fake(_noise(SR * 5, channels=3, seed=2), sample_id=1)])


def test_the_uneven_promotion_would_have_moved_the_audio():
    """Non-vacuity for the refusal: measure what it is refusing.

    Without this the test above would pass against a `promote_channels` that
    refused for no reason. The deviation is asserted on `prepare_waveform`
    output -- the model's input -- not on the raw tensor.
    """
    stereo = _noise(SR * 4, channels=2, seed=1)
    uneven = torch.cat([stereo, stereo[:1]], dim=0)          # [L, R, L]
    cfg = AudioConfig(channels="downmix")

    honest = prepare_waveform(stereo[None], cfg)[0]
    lied = prepare_waveform(uneven[None], cfg)[0]
    assert torch.allclose(lied, (2 * stereo[0] + stereo[1]) / 3, atol=1e-6)
    moved = float((honest - lied).abs().max())
    assert moved > 0.5, moved                                # measured 1.05


def test_promote_channels_repeats_rather_than_pads():
    row = _noise(100, channels=1, seed=3)
    out = promote_channels(row, 2)
    assert out.shape == (2, 100)
    assert torch.equal(out[0], out[1]) and torch.equal(out[0], row[0])


# --------------------------------------------------------------------------- #
# I16 -- two pad fillings, one submitted probability


@pytest.fixture(scope="module")
def model():
    return DeepVoiceNet(load_model_config("configs/b_stub.yaml")).eval()


def _submitted(model, batch, lengths=None):
    cfg = AudioConfig()
    with torch.no_grad():
        out = model(prepare_waveform(batch["wav"], cfg),
                    batch["lengths"] if lengths is None else lengths)
        return model.submission_probs(out)


#: The most a submitted probability may move between two batchings of the same
#: row. 🔴 Derived, not picked. Bitwise equality is unavailable: a wider batch
#: re-associates the frontend's batched GEMM, so a 320-term reduction is summed
#: in a different order. float32 eps is 1.19e-7 and the logits here are O(1), so
#: a few eps of relative error is the floor; 1e-6 is ~8x that. The measured
#: spread over the seed sweep below is 1.7e-9, and the mutation test that
#: follows moves the same number by >1e-3 -- three orders of magnitude of
#: separation, so the bound is not doing the work the check is.
GEMM_NOISE = 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_the_submitted_probability_does_not_move_with_the_batch(samples, model, seed):
    """I16 / rule 2.4, extended from the model to pipeline output.

    🔴 Asserted on `submission_probs` -- the number that reaches `submit.zip` --
    and deliberately **not** on `clip_logits`. `clip_logits` are already
    padding-safe because attention excludes masked frames; `frame_max` was not,
    and a test on the adjacent quantity is exactly how that shipped, moving a
    file's probability from 0.519 to 0.847 depending on its batch.

    The shipped collator pads with zeros, so a row's padded tail is the same
    whatever else is in the batch.

    ⚠️ The sweep draws a random **subset** of the other rows, not a permutation
    of all of them. A permutation keeps the company and therefore `S_max`
    constant, so all four seeds produced the bit-identical deviation
    1.7059283430320704e-09 -- one test run four times. With a varying subset the
    padded width varies, which is the thing that could move a score.
    """
    rng = np.random.default_rng(seed)
    others = list(rng.permutation(np.arange(1, len(samples))))
    keep = others[:int(rng.integers(1, len(others) + 1))]
    order = [0] + [int(i) for i in keep]
    rng.shuffle(order)
    where = order.index(0)

    alone = _submitted(model, collate([samples[0]]))
    together = _submitted(model, collate([samples[i] for i in order]))
    for column in alone:
        moved = abs(float(alone[column][0]) - float(together[column][where]))
        assert moved < GEMM_NOISE, f"{column} moved by {moved}"


@pytest.mark.parametrize("pad_value", [0.0, 7.5, -3.0])
def test_the_pad_value_cannot_reach_the_submitted_probability(samples, model,
                                                              pad_value):
    """I16 in its documented form: the padding **value** must not matter.

    🔴 This is exact, and it was not, until `Frontend.forward` started zeroing
    past `lengths`. `frames_for` rounds up, so a row whose length is not a
    multiple of the 320-sample hop had a last valid frame that was part padding
    and masked *in*; two fillings moved a submitted probability by 1.35e-3 on
    these very samples. Every earlier padding test used `lengths = SR * 4` and
    could not see it.

    ⚠️ The rows must genuinely be hop-unaligned or this is the old vacuous test
    in a new costume.
    """
    assert any(s.wav.shape[-1] % 320 for s in samples), \
        "no sample straddles a frame boundary; this test proves nothing"
    zeros = _submitted(model, collate(samples, pad_value=0.0))
    filled = _submitted(model, collate(samples, pad_value=pad_value))
    for column in zeros:
        assert np.allclose(zeros[column], filled[column], atol=0.0, rtol=0.0), column


def test_the_batch_invariance_check_can_fail(samples, model):
    """Mutation for the check above: lie about `lengths` and the padding gets in.

    `lengths` set to the padded width means every padded frame counts as real
    audio, so the pad *filling* -- which belongs to no file at all -- reaches the
    submitted probability. That is the defect the check exists for, reproduced,
    and it must clear `GEMM_NOISE` by orders of magnitude or the check above is
    measuring float noise.
    """
    zeros = collate(samples, pad_value=0.0)
    garbage = collate(samples, pad_value=7.5)
    full = torch.full_like(zeros["lengths"], int(zeros["wav"].shape[-1]))

    a = _submitted(model, zeros, full)
    b = _submitted(model, garbage, full)
    moved = max(float(np.abs(np.asarray(a[c]) - np.asarray(b[c])).max()) for c in a)
    assert moved > 1e-3, moved
    assert moved > 100 * GEMM_NOISE


# --------------------------------------------------------------------------- #
# I14 at the collation boundary
#
# ⚠️ Not a rebuild of `tests/test_registries.py::test_preprocess_step_is_batch_
# invariant`, which already generalises I14 from `bandpass` to every registered
# step on synthetic noise. This is the same guarantee one layer out: the whole
# `preprocess_chain`, at its real call site (the model boundary), over a real
# collated batch of rendered audio at real durations.


def test_the_whole_preprocess_chain_is_row_invariant_on_a_collated_batch(samples):
    """I14, over the composed chain rather than step by step."""
    chain = preprocess_chain(tuple((name, {}) for name in PREPROCESS.names()))
    assert chain.group_delay == 0, \
        "a chain that moves audio desynchronises frame_intervals; compensate it"
    cfg = AudioConfig()

    solo = collate([samples[0]])
    inside = collate(samples)
    n = int(solo["lengths"][0])

    a = chain(prepare_waveform(solo["wav"], cfg), SR, solo["lengths"])[0, :n]
    b = chain(prepare_waveform(inside["wav"], cfg), SR, inside["lengths"])[0, :n]
    assert torch.equal(a, b)


def test_the_chain_invariance_check_can_fail(samples):
    """Mutation: a batch statistic in the chain must be caught here too."""
    cfg = AudioConfig()
    solo = collate([samples[0]])
    inside = collate(samples)
    n = int(solo["lengths"][0])

    def batch_normalise(wav, sample_rate, lengths):
        return wav / wav.std()

    a = batch_normalise(prepare_waveform(solo["wav"], cfg), SR, solo["lengths"])[0, :n]
    b = batch_normalise(prepare_waveform(inside["wav"], cfg), SR,
                        inside["lengths"])[0, :n]
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Duration bucketing -- docs/pipelines/04 §3


@pytest.fixture(scope="module")
def draw_durations():
    """Durations from a real drawn stream, not from `U(4, 60)` directly.

    Duration and cell are not independent, so the bucketing tests must run on
    the joint distribution the sampler actually produces.
    """
    manifest = synthetic_manifest(n_per_pool=120, n_whole_file=120, seed=0)
    sampler = Sampler(manifest, SamplerConfig(), slice_="train")
    # ⚠️ 2,000 and not 2,048: with four quantile buckets and batch_size 32,
    # 2,048 divides exactly and every bucket comes out a whole number of
    # batches -- which made `all(len(b) == 32)` pass on a `drop_last=False`
    # collator. A size with a remainder is what gives that assertion teeth.
    return list(sampler.epoch_specs(2_000, epoch=0, seed=0))


def test_bucketing_is_deterministic_and_partitions_the_stream(draw_durations):
    d = spec_durations(draw_durations)
    first = bucket_batches(d, 32, n_buckets=4, seed=0)
    again = bucket_batches(d, 32, n_buckets=4, seed=0)
    assert first == again

    flat = [i for b in first for i in b]
    assert len(flat) == len(set(flat)), "an index landed in two batches"
    assert set(flat) <= set(range(len(d)))
    assert all(len(b) == 32 for b in first), "drop_last=True must give full batches"
    # Nothing is dropped beyond the per-bucket remainder.
    assert len(flat) >= len(d) - 4 * 31


def test_the_batch_order_is_shuffled_not_short_files_first(draw_durations):
    """🔴 Deleting `order = rng.permutation(...)` passed every other test.

    Buckets are built shortest-first, so without the final shuffle every epoch
    walks the short files first and the long ones last -- a duration schedule the
    optimiser sees, which is batch composition back in the objective by the side
    door. §3's whole argument is that bucketing is free of that.

    Measured on the mean duration per batch position: unshuffled, the rank
    correlation with batch index is +1.0 by construction. Shuffled it must be
    near zero, and the bound is derived from the null -- Spearman's rho over B
    batches has sd 1/sqrt(B-1) under H0, so 4 sd is the gate.
    """
    d = spec_durations(draw_durations)
    plan = bucket_batches(d, 32, n_buckets=4, seed=0)
    means = np.array([np.mean([d[i] for i in b]) for b in plan])

    order_rank = np.argsort(np.argsort(np.arange(len(means))))
    dur_rank = np.argsort(np.argsort(means))
    rho = float(np.corrcoef(order_rank, dur_rank)[0, 1])
    gate = 4.0 / np.sqrt(len(means) - 1)
    assert abs(rho) < gate, f"batches are ordered by duration: rho={rho:.3f}"

    # Not vacuous: the same statistic on the unshuffled plan is +1.0.
    ordered = sorted(plan, key=lambda b: np.mean([d[i] for i in b]))
    om = np.array([np.mean([d[i] for i in b]) for b in ordered])
    rho_sorted = float(np.corrcoef(np.arange(len(om)), np.argsort(np.argsort(om)))[0, 1])
    assert rho_sorted > 0.99, rho_sorted


@pytest.mark.parametrize("seed", range(4))
def test_quantile_edges_stay_balanced_where_equal_width_edges_do_not(seed):
    """The justification in `bucket_edges`' docstring, tested where it bites.

    ⚠️ **Not** on the drawn stream. `SamplerConfig.duration_range` is `U(4, 60)`,
    so equal-width edges are balanced there too (measured 0.224-0.291 across four
    buckets) -- a test on that stream cannot tell the two rules apart, and a
    first version of this one said so by failing its own non-vacuity guard.

    The claim `bucket_edges` actually makes is about *any* stream it is handed:
    occupancy decides both padding efficiency and how many batches a bucket
    yields, and a thin bucket loses nearly all of its rows to `drop_last`. A
    skewed stream is where the two rules diverge, and durations skew the moment
    a corpus's own lengths reach the draw (a short-clip pool truncates
    `ComponentDraw.duration_s`) or `duration_range` is swept.
    """
    rng = np.random.default_rng(seed)
    skewed = np.clip(4.0 + rng.lognormal(1.0, 1.0, 4_000), 4.0, 60.0)

    q = np.bincount([bucket_of(x, bucket_edges(skewed, 4)) for x in skewed],
                    minlength=4) / len(skewed)
    lo, hi = skewed.min(), skewed.max()
    wide = tuple(lo + (hi - lo) * k / 4 for k in (1, 2, 3))
    w = np.bincount([bucket_of(x, wide) for x in skewed], minlength=4) / len(skewed)

    assert q.min() > 0.20 and q.max() < 0.30, q.tolist()
    # Non-vacuity, on the same stream: equal width leaves a bucket starved.
    assert w.min() < 0.10, w.tolist()


def test_equal_durations_never_split_across_buckets():
    """Bucketing is a function of the duration, not of the row's rank.

    ⚠️ `bucket_of`'s `side="right"` is a tie-break with no observable effect on a
    drawn stream (0 of 2,000 rows assigned differently by `side="left"`, because
    exact ties with a quantile edge have measure zero over `U(4, 60)`). This is
    the property underneath it that does bite: a rank-based bucketer --
    `np.array_split` over `argsort`, the obvious alternative -- splits a tied
    mass across a boundary, so which batch a row lands in depends on the others.
    """
    # ⚠️ Deliberately uneven group sizes. With 400/400/400 the rank thirds land
    # exactly on the value boundaries and the rank-based bucketer agrees by
    # accident -- so the non-vacuity check below would pass for the wrong reason.
    d = [10.0] * 500 + [20.0] * 100 + [30.0] * 600
    edges = bucket_edges(d, 3)
    for value in (10.0, 20.0, 30.0):
        assert len({bucket_of(x, edges) for x in d if x == value}) == 1, value

    # Non-vacuity: the rank-based alternative splits both tied masses.
    ranked = np.argsort(np.argsort(d)) * 3 // len(d)
    for value in (10.0, 30.0):
        assert len({int(b) for b, x in zip(ranked, d) if x == value}) > 1, value


def test_a_batch_never_straddles_a_bucket(draw_durations):
    d = spec_durations(draw_durations)
    edges = bucket_edges(d, 4)
    for batch in bucket_batches(d, 32, n_buckets=4, seed=0):
        assert len({bucket_of(d[i], edges) for i in batch}) == 1


def test_bucketing_is_the_padding_win_it_claims_to_be(draw_durations):
    """The whole reason to bucket, as a number rather than a claim.

    Measured over the drawn stream: unbucketed whole-file batching wastes most
    of the tensor on padding across a 15x duration span.
    """
    d = spec_durations(draw_durations)
    loose = bucket_batches(d, 32, n_buckets=1, seed=0)
    tight = bucket_batches(d, 32, n_buckets=4, seed=0)
    before, after = padding_fraction(d, loose), padding_fraction(d, tight)
    assert before > 0.4, before
    assert after < 0.5 * before, (before, after)


@pytest.mark.parametrize("seed", range(6))
def test_bucketing_does_not_starve_a_masked_head(draw_durations, seed):
    """🔴 The C2 caveat docs/pipelines/04 §3 leaves live, measured not assumed.

    ✅ Duration and cell are independent by construction -- `sample_spec` draws
    the duration first, from U(4, 60), then fits components into it -- so this is
    insurance rather than a live hazard (measured presence per bucket:
    0.676-0.694). It stays because the property is worth a tripwire and the check
    is cheap, and the seed is swept: four rounds of flakiness in this repo traced
    to every test drawing seed 0.

    The floor is `audit_specs`' own `min_present=8` at `batch_size=32`, which is
    where docs/pipelines/02 §4's table puts the gradient-norm inflation at 2.0x.

    🔴 The count is **not** reimplemented here. An earlier version looped over
    the plan inline and then asserted that `audit_specs`' *draw-order* batching
    also passed -- two different batchings, one of which the optimiser never
    sees. That is the adjacent-quantity pattern, in the file written against it.
    `audit_specs(..., batches=)` now takes the plan itself, so the shipped check
    and the shipped batching are the same two objects.
    """
    from training.audit import audit_specs

    specs = list(draw_durations)
    plan = bucket_batches(spec_durations(specs), 32, n_buckets=4, seed=seed)
    report = audit_specs(specs, batches=plan, min_present=8)
    passed, why = report.results["I9_C2_present_count_floor"]
    assert passed, f"seed {seed}: {why}"
    assert "supplied plan" in why, "the report must say which batching it measured"


def test_the_starvation_check_can_fail(draw_durations):
    """Mutation for C2-under-bucketing: bucket by *presence* and it must fire.

    Sorting the stream by whether voice is present is the extreme of what
    bucketing does mildly -- it makes batch composition a function of a label.
    Run through the same `batches=` path the check above uses, so the mutation
    exercises the shipped code rather than a loop written next to it.
    """
    from training.audit import audit_specs

    specs = list(draw_durations)
    by_presence = sorted(range(len(specs)), key=lambda i: specs[i].voice_present)
    plan = [by_presence[k:k + 32] for k in range(0, len(specs) - 31, 32)]
    report = audit_specs(specs, batches=plan, min_present=8)
    passed, why = report.results["I9_C2_present_count_floor"]
    assert not passed, why
    assert "voice" in why


def test_the_audited_plan_must_index_the_stream_it_audits(draw_durations):
    """A plan from another epoch would silently measure the wrong specs."""
    from training.audit import audit_specs

    specs = list(draw_durations)
    with pytest.raises(ValueError, match="outside the spec list"):
        audit_specs(specs, batches=[[0, 1, len(specs) + 5]])
