"""`render`, against real files on disk.

Caveat: nothing here is mocked. A mock decode cannot catch a decode, resample,
codec or alignment defect, and those are exactly where this stage fails -- so
the fixture writes an actual corpus (`training.synthetic.write_synthetic_corpus`)
with mixed containers, sample rates and channel counts.

Critical: each invariant is mutation-tested: the check is shown failing on a
deliberately broken version before it is trusted on the real one.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from models.config import AudioConfig
from models.losses import TARGET_FOR_COLUMN
from training.registries import AUGMENT
from training.render import (DecodeError, ManifestIndex, RenderConfig,
                             frame_intervals_for, load_audio, render,
                             resample_poly_to)
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]
CORPUS = dict(n_per_pool=5, n_whole_file=6, seed=0, duration_range=(7.0, 10.0))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return root, manifest, ManifestIndex.from_frame(manifest)


@pytest.fixture(scope="module")
def corpus_root(corpus):
    return corpus[0]


@pytest.fixture(scope="module")
def cfg(corpus):
    root, _, _ = corpus
    return RenderConfig(root=root)


def _ids(manifest, **where):
    sub = manifest
    for k, v in where.items():
        sub = sub[sub[k] == v]
    return sub.file_id.tolist()


def _spec(manifest, *, sample_id=0, cell=5, duration=5.0, structure="overlap",
          crossfade_ms=0.0, transforms=(), normalize=None, gains=(0.0, 0.0),
          starts=None, durations=None):
    """A composed voice+music spec over two real pool files."""
    voice = _ids(manifest, pool="A")[0]
    music = _ids(manifest, pool="C")[0]
    if starts is None:
        starts = (0.0, 0.0) if structure == "overlap" else (0.0, duration / 2)
    if durations is None:
        durations = (duration, duration) if structure == "overlap" \
            else (duration / 2, duration / 2)
    return SampleSpec(
        sample_id=sample_id, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=duration, cell=cell, render_mode="composed", structure=structure,
        components=(
            ComponentDraw(voice, "voice", 0.5, durations[0], starts[0], gains[0]),
            ComponentDraw(music, "music", 0.5, durations[1], starts[1], gains[1]),
        ),
        crossfade_ms=crossfade_ms, transforms=tuple(transforms),
        normalize=dict(normalize or {}))


# --------------------------------------------------------------------------- #
# I10 -- render(spec) == render(spec), bitwise


def test_rendering_twice_is_bitwise_identical(corpus, cfg):
    """I10. Required for the 2nd-stage submission to reproduce the Private
    score (A-S2 / R9)."""
    _, manifest, index = corpus
    spec = _spec(manifest, transforms=(("gaussian_noise", {}),
                                       ("rawboost_ssi", {}),
                                       ("gain_jitter", {})),
                 normalize={"container": "mp3", "bitrate": 96, "channels": "mono"})
    a, b = render(spec, index, cfg), render(spec, index, cfg)
    assert torch.equal(a.wav, b.wav)
    assert a.targets == b.targets


def test_rendering_in_another_process_is_bitwise_identical(corpus, cfg):
    """Critical: I10 says *across processes*, and that is not pedantry: `hash()` is
    salted per process, so an RNG keyed on it reproduces within a run and
    differs across runs -- the exact failure A-S2 exists to prevent. A
    same-process check cannot see it."""
    root, manifest, index = corpus
    spec = _spec(manifest, sample_id=17,
                 transforms=(("gaussian_noise", {}), ("rawboost_ssi", {})))
    mine = hashlib.sha256(render(spec, index, cfg).wav.numpy().tobytes()).hexdigest()

    code = (
        "import json,sys,hashlib\n"
        "from pathlib import Path\n"
        "from training.render import render, RenderConfig, ManifestIndex\n"
        "from training.spec import SampleSpec\n"
        "from training.synthetic import synthetic_manifest\n"
        f"m = synthetic_manifest(**{CORPUS!r})\n"
        "spec = SampleSpec.from_dict(json.loads(sys.argv[2]))\n"
        "w = render(spec, ManifestIndex.from_frame(m),\n"
        "           RenderConfig(root=Path(sys.argv[1]))).wav\n"
        "print(hashlib.sha256(w.numpy().tobytes()).hexdigest())\n")
    proc = subprocess.run([sys.executable, "-c", code, str(root),
                           json.dumps(spec.to_dict())],
                          cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == mine


def test_the_reproducibility_check_can_fail(corpus, cfg):
    """Mutation test for I10: two different RNG keys must not collide, or the
    check above would pass on a renderer that ignored the spec entirely."""
    _, manifest, index = corpus
    one = _spec(manifest, sample_id=1, transforms=(("gaussian_noise", {}),))
    two = _spec(manifest, sample_id=2, transforms=(("gaussian_noise", {}),))
    assert not torch.equal(render(one, index, cfg).wav, render(two, index, cfg).wav)


# --------------------------------------------------------------------------- #
# I11 -- the RNG key is wired, not defaulted


def test_only_the_sample_id_differs_and_the_audio_does_too(corpus, cfg):
    """I11. `spec.rng` is blake2b-keyed on (sample_id, epoch, seed); a renderer
    that built its own generator would pass every other test here."""
    _, manifest, index = corpus
    a = render(_spec(manifest, sample_id=3, transforms=(("gaussian_noise", {}),)),
               index, cfg).wav
    b = render(_spec(manifest, sample_id=4, transforms=(("gaussian_noise", {}),)),
               index, cfg).wav
    assert a.shape == b.shape
    assert not torch.equal(a, b)


def test_without_a_stochastic_transform_the_sample_id_changes_nothing(corpus, cfg):
    """The other half of I11: the difference above must come from the RNG, not
    from the sample_id leaking into decode or composition."""
    _, manifest, index = corpus
    a = render(_spec(manifest, sample_id=3), index, cfg).wav
    b = render(_spec(manifest, sample_id=4), index, cfg).wav
    assert torch.equal(a, b)


def test_a_drawn_parameter_beats_the_rng(corpus, cfg):
    """A spec-level draw is the ledger. If it is present the transform must use
    it, or the recorded params would not describe the rendered audio."""
    _, manifest, index = corpus
    quiet = render(_spec(manifest, transforms=(("gain_jitter", {"db": -20.0}),)),
                   index, cfg).wav
    plain = render(_spec(manifest), index, cfg).wav
    ratio = float(quiet.pow(2).mean().sqrt() / plain.pow(2).mean().sqrt())
    assert ratio == pytest.approx(0.1, rel=1e-4)


# --------------------------------------------------------------------------- #
# I12 -- the length regime


def test_rendered_duration_sits_in_the_length_regime(corpus, cfg):
    """I12, asserted on the **rendered sample count** -- the quantity that
    reaches the model -- not on `spec.duration_s`, which is the adjacent one.
    The `frame_max` defect survived exactly that substitution."""
    _, manifest, index = corpus
    audio = AudioConfig()
    for duration in (4.0, 5.5, 6.0):
        out = render(_spec(manifest, duration=duration), index, cfg)
        assert out.wav.shape[-1] == round(duration * audio.sample_rate)
        assert audio.min_seconds <= out.duration_s <= audio.max_seconds


def test_a_sample_below_the_floor_is_refused(corpus, cfg):
    """Mutation test for I12: the check must reject something. 2,580 of 4,000
    specs came out under the 4 s floor on a corpus of short clips."""
    _, manifest, index = corpus
    with pytest.raises(ValueError, match="length regime"):
        render(_spec(manifest, duration=2.0), index, cfg)


# --------------------------------------------------------------------------- #
# I13 -- frame targets and audio describe the same timeline


def test_frame_intervals_are_absolute_seconds_inside_the_timeline(corpus, cfg):
    """Critical: never a rasterised frame grid: frame rate belongs to the frontend, and
    hard-coding one reproduces the `align_time` bug (docs/pipelines/01 §4)."""
    _, manifest, index = corpus
    spec = _spec(manifest, cell=6, duration=6.0, structure="sequential",
                 crossfade_ms=50.0)
    out = render(spec, index, cfg)
    for branch, intervals in out.frame_intervals.items():
        for start, end, label in intervals:
            assert isinstance(start, float) and isinstance(end, float)
            assert 0.0 <= start < end <= spec.duration_s + 1e-9
            assert label in (0, 1)

    # The union is the component placement, not an approximation of it.
    placed = sorted((c.target_start_s, c.target_start_s + c.duration_s)
                    for c in spec.components)
    assert sorted((s, e) for s, e, _ in out.frame_intervals["file"]) == placed


def test_frame_labels_follow_the_component_not_the_file(corpus, cfg):
    """Cell 6 is real voice over fake music. If the frame branch copied
    `file_fake` the voice frames would be labelled fake, which is the label
    noise the two fake heads exist to avoid."""
    _, manifest, index = corpus
    out = render(_spec(manifest, cell=6, duration=5.0), index, cfg)
    assert [lab for _, _, lab in out.frame_intervals["voice"]] == [0]
    assert [lab for _, _, lab in out.frame_intervals["music"]] == [1]
    assert out.targets["file_fake"] == 1


def test_a_whole_file_row_carries_no_frame_targets(corpus, cfg):
    """They exist only for composed rows -- there is no composition to describe."""
    _, manifest, index = corpus
    from training.spec import CELL_TABLE
    row = manifest[manifest.row_kind == "whole_file"].iloc[0]
    cell = int(row.cell)
    vp, mp, vf, mf = CELL_TABLE[cell]
    spec = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=5.0, cell=cell, render_mode="whole_file", structure="overlap",
        components=(ComponentDraw(
            str(row.file_id), "voice" if vp else "music" if mp else "noise",
            0.0, 5.0, 0.0, 0.0),))
    out = render(spec, index, cfg)
    assert out.frame_intervals == {"voice": (), "music": (), "file": ()}
    # Caveat: written out rather than read back from `spec.labels`: an
    # assertion against the expression that produced the value can only ever
    # pass. This is the competition's own definition -- OR over *present*
    # components.
    assert out.targets == {
        "voice_present": vp, "music_present": mp,
        "voice_fake": int(vf or 0), "music_fake": int(mf or 0),
        "file_fake": int(bool(vp and vf) or bool(mp and mf))}


def test_frame_intervals_track_a_moved_component(corpus):
    """Mutation test for I13: move the placement and the intervals must move.
    A hard-coded `(0, duration)` would satisfy the containment check above."""
    manifest = synthetic_manifest(**CORPUS)
    early = frame_intervals_for(_spec(manifest, duration=6.0, structure="sequential"))
    late = frame_intervals_for(_spec(manifest, duration=6.0, structure="sequential",
                                     starts=(0.0, 4.0), durations=(2.0, 2.0)))
    assert early["music"] != late["music"]
    assert late["music"] == ((4.0, 6.0, 0),)


# --------------------------------------------------------------------------- #
# Critical: A warp that lives in the draw moves the audio AND the frame targets
#
# A-A8 and A-A11 are not step-4 augments; they are `target_start_s`. That is the
# whole payoff of the contract: the frame targets are *computed from* the
# placement, so they cannot disagree with it.


def _first_overlap_spec(sampler, limit=200):
    for i in range(limit):
        spec = sampler.sample_spec(i)
        if spec.render_mode == "composed" and spec.structure == "overlap":
            return spec
    raise AssertionError("no composed overlap spec drawn")   # pragma: no cover


def test_a_drawn_silence_lead_moves_the_audio_and_the_frame_targets_together(corpus, cfg):
    """A-A11/A-A8 as a draw. The head of the waveform is silent for exactly the
    drawn lead, and `frame_intervals` start there -- not because the renderer
    tracked a warp, but because both come from `target_start_s`."""
    from training.sampler import Sampler, SamplerConfig
    _, manifest, index = corpus
    sr = AudioConfig().sample_rate

    quiet = Sampler(manifest, SamplerConfig(duration_range=(4.0, 6.0),
                                            silence_lead_s=0.5), slice_="train")
    spec = _first_overlap_spec(quiet)
    lead = spec.components[0].target_start_s
    assert lead > 0.05, "this draw should have produced a lead"

    out = render(spec, index, cfg)
    head = out.wav[:, :int(lead * sr) - 8]
    body = out.wav[:, int(lead * sr) + 8:]
    assert float(head.abs().max()) == 0.0        # silence, in the audio
    assert float(body.abs().max()) > 0.0
    assert out.frame_intervals["file"][0][0] == pytest.approx(lead)


def test_without_the_draw_there_is_no_lead_to_find(corpus, cfg):
    """Mutation test for the check above: at the shipped default the head is
    *not* silent, so the assertion is measuring the draw and not a constant."""
    from training.sampler import Sampler, SamplerConfig
    _, manifest, index = corpus
    sr = AudioConfig().sample_rate

    plain = Sampler(manifest, SamplerConfig(duration_range=(4.0, 6.0)),
                    slice_="train")
    spec = _first_overlap_spec(plain)
    assert spec.components[0].target_start_s == 0.0
    out = render(spec, index, cfg)
    assert float(out.wav[:, :int(0.5 * sr)].abs().max()) > 0.0
    assert out.frame_intervals["file"][0][0] == 0.0


# --------------------------------------------------------------------------- #
# The sample contract


def test_targets_are_exactly_the_loss_keys(corpus, cfg):
    """I18 at the render boundary: a renamed target key must not be discovered
    at hour three of a run."""
    _, manifest, index = corpus
    out = render(_spec(manifest), index, cfg)
    assert set(out.targets) == set(TARGET_FOR_COLUMN.values())
    assert all(isinstance(v, int) for v in out.targets.values())


def test_an_absent_component_gets_zero_not_none(corpus, cfg):
    """Caveat: `None` survives `np.asarray(None).astype(bool)` as True and is only
    saved by the presence mask zeroing it. That is incidental, not a contract."""
    _, manifest, index = corpus
    row = manifest[(manifest.row_kind == "whole_file") & (manifest.cell == 1)].iloc[0]
    spec = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=5.0, cell=1, render_mode="whole_file", structure="overlap",
        components=(ComponentDraw(str(row.file_id), "voice", 0.0, 5.0, 0.0, 0.0),))
    out = render(spec, index, cfg)
    assert spec.music_fake is None
    assert out.targets["music_fake"] == 0


def test_wav_is_channels_by_samples_and_not_downmixed(corpus, cfg):
    """Caveat: downmixing here would fork the channel policy away from
    `models.audio.prepare_waveform`, disable the A-B3 channel augmentations and
    make the mid_side leak test impossible (docs/pipelines/01 §4)."""
    root, manifest, index = corpus
    stereo = manifest[(manifest.pool == "A") & (manifest.orig_channels == 2)]
    if stereo.empty:                                          # pragma: no cover
        pytest.skip("no stereo voice row in this corpus draw")
    row = stereo.iloc[0]
    spec = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=5.0, cell=1, render_mode="composed", structure="overlap",
        components=(ComponentDraw(str(row.file_id), "voice", 0.0, 5.0, 0.0, 0.0),))
    out = render(spec, index, cfg)
    assert out.wav.dim() == 2 and out.wav.shape[0] == 2
    assert not torch.equal(out.wav[0], out.wav[1])       # really two channels


def test_a_mono_source_stays_one_channel(corpus, cfg):
    root, manifest, index = corpus
    mono = manifest[(manifest.pool == "A") & (manifest.orig_channels == 1)]
    if mono.empty:                                            # pragma: no cover
        pytest.skip("no mono voice row in this corpus draw")
    row = mono.iloc[0]
    spec = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=5.0, cell=1, render_mode="composed", structure="overlap",
        components=(ComponentDraw(str(row.file_id), "voice", 0.0, 5.0, 0.0, 0.0),))
    assert render(spec, index, cfg).wav.shape[0] == 1


def test_the_model_accepts_a_rendered_sample(corpus, cfg):
    """I17 at the render boundary: `prepare_waveform` is the call site the
    channel policy lives at, in training and inference alike."""
    from models.audio import prepare_waveform
    _, manifest, index = corpus
    out = render(_spec(manifest, duration=4.0), index, cfg)
    prepared = prepare_waveform(out.wav[None], AudioConfig())
    assert prepared.shape == (1, out.wav.shape[-1])


# --------------------------------------------------------------------------- #
# Composition -- step 3


def test_the_component_gain_is_the_gain_that_ships(corpus, cfg):
    """A-A3: the voice/music ratio is logged per sample for the accuracy-vs-SNR
    curve, so it has to be the ratio actually rendered."""
    _, manifest, index = corpus
    loud = render(_spec(manifest, gains=(0.0, -200.0)), index, cfg).wav
    quiet = render(_spec(manifest, gains=(-20.0, -200.0)), index, cfg).wav
    ratio = float(quiet.pow(2).mean().sqrt() / loud.pow(2).mean().sqrt())
    assert ratio == pytest.approx(0.1, rel=1e-3)


def test_overlap_sums_the_components(corpus, cfg):
    _, manifest, index = corpus
    both = render(_spec(manifest, gains=(0.0, 0.0)), index, cfg).wav
    voice = render(_spec(manifest, gains=(0.0, -400.0)), index, cfg).wav
    music = render(_spec(manifest, gains=(-400.0, 0.0)), index, cfg).wav
    assert torch.allclose(both, voice + music, atol=1e-6)


def test_a_sequential_joint_is_faded_rather_than_cut(corpus, cfg):
    """A-A4: a hard cut becomes a splice shortcut. Caveat: the taper is complementary
    at the joint rather than an overlapping crossfade, because the components
    are drawn back-to-back and the renderer does not overrule the spec."""
    _, manifest, index = corpus
    sr = AudioConfig().sample_rate
    faded = render(_spec(manifest, duration=6.0, structure="sequential",
                         crossfade_ms=200.0), index, cfg).wav
    cut = render(_spec(manifest, duration=6.0, structure="sequential",
                       crossfade_ms=0.0), index, cfg).wav
    joint = 3 * sr
    window = slice(joint - 400, joint + 400)
    assert float(faded[:, window].abs().max()) < float(cut[:, window].abs().max())
    # Away from the joint the two are the same audio.
    assert torch.allclose(faded[:, :joint - 4000], cut[:, :joint - 4000], atol=1e-6)


# --------------------------------------------------------------------------- #
# Test-chain normalisation -- step 5, always last


def test_a_codec_round_trip_does_not_move_the_audio(corpus, cfg):
    """Critical: LAME's 1105-sample (69 ms) delay would slide the waveform out from
    under `frame_intervals` while they stayed put -- the `align_time` defect in
    a place the existing tests do not look."""
    _, manifest, index = corpus
    plain = render(_spec(manifest, duration=5.0), index, cfg).wav
    coded = render(_spec(manifest, duration=5.0,
                         normalize={"container": "mp3", "bitrate": 128}),
                   index, cfg).wav
    assert coded.shape == plain.shape

    a = (plain[0] - plain[0].mean()).numpy()
    b = (coded[0] - coded[0].mean()).numpy()
    lag = int(np.argmax(np.correlate(b[:40_000], a[8_000:24_000], "valid"))) - 8_000
    assert lag == 0
    assert not torch.equal(coded, plain)             # it really was encoded


def test_the_alignment_check_can_fail(corpus, cfg):
    """Mutation test for the check above: a deliberately shifted signal must
    produce a non-zero lag, or `lag == 0` is measuring nothing."""
    _, manifest, index = corpus
    plain = render(_spec(manifest, duration=5.0), index, cfg).wav
    shifted = torch.roll(plain, 1105, dims=-1)
    a = (plain[0] - plain[0].mean()).numpy()
    b = (shifted[0] - shifted[0].mean()).numpy()
    lag = int(np.argmax(np.correlate(b[:40_000], a[8_000:24_000], "valid"))) - 8_000
    assert lag == 1105


def test_the_telephone_chain_removes_the_band_it_should(corpus, cfg):
    """A-S4: 16k -> 8k -> 16k. The test set explicitly contains 전화채널 audio."""
    _, manifest, index = corpus
    sr = AudioConfig().sample_rate
    plain = render(_spec(manifest, duration=5.0), index, cfg).wav
    phone = render(_spec(manifest, duration=5.0,
                         normalize={"telephone_hz": 8000, "companding": "ulaw"}),
                   index, cfg).wav
    assert phone.shape == plain.shape

    def high_band_energy(x):
        spec = np.abs(np.fft.rfft(x[0].numpy().astype(np.float64)))
        freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sr)
        return float((spec[freqs > 4200] ** 2).sum() / (spec ** 2).sum())

    assert high_band_energy(phone) < 0.05 * high_band_energy(plain)


def test_the_mono_emit_is_a_downmix_of_the_rendered_channels(corpus, cfg):
    _, manifest, index = corpus
    out = render(_spec(manifest, duration=5.0, normalize={"channels": "mono"}),
                 index, cfg)
    assert out.wav.shape[0] == 1
    stereo = render(_spec(manifest, duration=5.0, normalize={"channels": "stereo"}),
                    index, cfg)
    assert stereo.wav.shape[0] == 2


def test_an_unknown_normalize_key_is_an_error(corpus, cfg):
    """A typo'd knob that silently does nothing is how an ablation measures the
    wrong thing -- `models.config`'s rule, applied to the test-chain draw."""
    _, manifest, index = corpus
    with pytest.raises(ValueError, match="unknown normalize key"):
        render(_spec(manifest, normalize={"bitrates": 96}), index, cfg)
    with pytest.raises(ValueError, match="container must be"):
        render(_spec(manifest, normalize={"container": "aac"}), index, cfg)


def test_an_empty_normalize_draw_is_a_no_op(corpus, cfg):
    """Caveat: G1's `signal_chain.yaml` does not exist yet, so the stage is written
    and not parameterised. It must not invent parameters in the meantime."""
    _, manifest, index = corpus
    assert torch.equal(render(_spec(manifest, normalize={}), index, cfg).wav,
                       render(_spec(manifest, normalize=None), index, cfg).wav)


# --------------------------------------------------------------------------- #
# Rule 2.4 at the render boundary


def test_a_render_does_not_depend_on_the_rest_of_the_manifest(corpus, cfg):
    """Rule 2.4's spec-level shape: a file's rendered output must not depend on
    what else is in the corpus, so an index with extra rows or a different order
    renders the same bytes."""
    _, manifest, index = corpus
    spec = _spec(manifest, transforms=(("gaussian_noise", {}),))
    shuffled = ManifestIndex.from_frame(
        manifest.sample(frac=1.0, random_state=0))
    assert torch.equal(render(spec, index, cfg).wav,
                       render(spec, shuffled, cfg).wav)


def test_the_augment_call_site_can_only_pass_wav_and_rng(corpus, cfg):
    """Critical: `P(T | L) = P(T)` made structural: whatever `render` knows about the
    sample, it has no argument to pass it in. A convention would be a code
    review; this is the call site."""
    seen = {}

    @AUGMENT.register("__probe_for_the_call_site")
    def _probe(wav, rng):
        seen["args"] = (type(wav).__name__, type(rng).__name__)
        return wav

    try:
        _, manifest, index = corpus
        render(_spec(manifest, transforms=(("__probe_for_the_call_site", {}),)),
               index, cfg)
    finally:
        AUGMENT._fns.pop("__probe_for_the_call_site", None)
        AUGMENT._params.pop("__probe_for_the_call_site", None)
    assert seen["args"] == ("Tensor", "Generator")


# --------------------------------------------------------------------------- #
# Decode -- P-S1 and P-S2


def test_every_container_in_the_corpus_decodes(corpus):
    """P-S1: the eval server hands us mixed containers and a decode crash burns
    one of three daily submissions."""
    root, manifest, _ = corpus
    seen = set()
    for row in manifest.to_dict(orient="records"):
        wav = load_audio(root / row["path"], sample_rate=16_000,
                         offset_s=0.0, duration_s=4.0, file_id=row["file_id"])
        assert wav.shape == (int(row["orig_channels"]), 64_000)
        assert np.isfinite(wav).all()
        seen.add(row["container"])
    assert seen == {"wav", "mp3", "flac"}


def test_the_extension_is_never_trusted(corpus, tmp_path):
    """P-S1: "never assume the extension". libsndfile sniffs the container."""
    root, manifest, _ = corpus
    flac = manifest[manifest.container == "flac"].iloc[0]
    liar = tmp_path / "actually_a_flac.wav"
    liar.write_bytes((root / flac["path"]).read_bytes())
    honest = load_audio(root / flac["path"], sample_rate=16_000, duration_s=4.0)
    assert np.array_equal(load_audio(liar, sample_rate=16_000, duration_s=4.0),
                          honest)


def test_a_container_libsndfile_refuses_goes_through_ffmpeg(tmp_path):
    """The second decode path. Caveat: untested, it is a `pass` in a `try`."""
    src = tmp_path / "tone.wav"
    sr, sr_n = 16_000, 16_000 * 5
    import soundfile as sf
    sf.write(str(src), (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr_n) / sr)
                        ).astype(np.float32), sr)
    aac = tmp_path / "tone.m4a"
    proc = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                           "-i", str(src), "-c:a", "aac", str(aac)],
                          capture_output=True)
    if proc.returncode != 0:                                  # pragma: no cover
        pytest.skip("this ffmpeg has no aac encoder")
    with pytest.raises(Exception):
        sf.read(str(aac))                        # libsndfile really refuses it
    wav = load_audio(aac, sample_rate=16_000, duration_s=4.0)
    assert wav.shape[-1] == 64_000
    assert float(np.abs(wav).max()) > 0.05


def test_the_fallback_path_uses_the_injected_resampler_too(tmp_path):
    """Critical: P-S2 is "one fixed resampler applied identically", so which resampler
    a file meets must not depend on its container. Passing `-ar` to ffmpeg would
    put swr on this path and `resample_poly` on the other, and every test that
    only checks the sample count would still pass."""
    import soundfile as sf
    src = tmp_path / "tone48.wav"
    sf.write(str(src), np.zeros(48_000 * 5, dtype=np.float32) + 0.3, 48_000)
    aac = tmp_path / "tone48.m4a"
    proc = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                           "-i", str(src), "-c:a", "aac", str(aac)],
                          capture_output=True)
    if proc.returncode != 0:                                  # pragma: no cover
        pytest.skip("this ffmpeg has no aac encoder")

    seen = []

    def spy(x, orig_sr, target_sr):
        seen.append((orig_sr, target_sr))
        return np.zeros((x.shape[0], int(round(x.shape[-1] * target_sr / orig_sr))),
                        dtype=np.float32)

    wav = load_audio(aac, sample_rate=16_000, duration_s=4.0, resampler=spy)
    assert seen == [(48_000, 16_000)]        # decoded at the native rate
    assert float(np.abs(wav).max()) == 0.0   # and the injected kernel was used


def test_an_undecodable_file_names_itself(tmp_path):
    """A decode crash costs a submission, so it must say which file did it."""
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFFnot really a wav at all")
    with pytest.raises(DecodeError) as exc:
        load_audio(broken, sample_rate=16_000, duration_s=4.0, file_id="A00007")
    assert "broken.wav" in str(exc.value)
    assert "A00007" in str(exc.value)        # the row, not just the path


def test_a_spec_naming_an_absent_file_id_is_loud(corpus, cfg):
    _, manifest, index = corpus
    spec = SampleSpec(
        sample_id=0, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=5.0, cell=1, render_mode="composed", structure="overlap",
        components=(ComponentDraw("nope", "voice", 0.0, 5.0, 0.0, 0.0),))
    with pytest.raises(DecodeError, match="not in the manifest"):
        render(spec, index, cfg)


def test_decoding_returns_exactly_the_requested_span(corpus):
    """The timeline the sampler drew is the timeline that comes back, whatever
    the source rate -- otherwise composition arithmetic is off by a resample."""
    root, manifest, _ = corpus
    for row in manifest.to_dict(orient="records")[:8]:
        wav = load_audio(root / row["path"], sample_rate=16_000,
                         offset_s=1.0, duration_s=3.25)
        assert wav.shape[-1] == 52_000


def test_a_source_shorter_than_the_manifest_says_is_loud(corpus):
    """Critical: padding it with silence would make a sample that is mostly silence
    because a corpus row lies -- and nothing downstream could tell."""
    root, manifest, _ = corpus
    row = manifest.iloc[0]
    with pytest.raises(DecodeError, match="short of the"):
        load_audio(root / row["path"], sample_rate=16_000,
                   duration_s=float(row["duration_s"]) + 5.0,
                   file_id=str(row["file_id"]))


def test_resampler_rounding_is_still_absorbed(corpus):
    """The check above must not fire on the ±few samples a rate conversion
    leaves: 22_050 -> 16_000 does not divide evenly."""
    root, manifest, _ = corpus
    odd = manifest[manifest.orig_sr == 22_050]
    if odd.empty:                                             # pragma: no cover
        pytest.skip("no 22.05 kHz row in this corpus draw")
    row = odd.iloc[0]
    wav = load_audio(root / row["path"], sample_rate=16_000, duration_s=5.0)
    assert wav.shape[-1] == 80_000


def test_the_offset_is_honoured(corpus):
    root, manifest, _ = corpus
    row = manifest[manifest.container == "wav"].iloc[0]
    head = load_audio(root / row["path"], sample_rate=16_000, duration_s=2.0)
    later = load_audio(root / row["path"], sample_rate=16_000, offset_s=2.0,
                       duration_s=2.0)
    assert not np.allclose(head, later)


def test_the_resampler_is_injectable(corpus, cfg):
    """Critical: G1 may reveal which resampler the organizers used, and matching them
    is worth more than kernel quality (docs/data/06 A-S1)."""
    calls = []

    def spy(x, orig_sr, target_sr):
        calls.append((orig_sr, target_sr))
        return resample_poly_to(x, orig_sr, target_sr)

    _, manifest, index = corpus
    spec = _spec(manifest)
    out = render(spec, index, RenderConfig(root=cfg.root, resampler=spy))
    assert calls and all(t == 16_000 for _, t in calls)
    assert torch.equal(out.wav, render(spec, index, cfg).wav)


def test_a_different_resampler_changes_the_audio(corpus, cfg):
    """Mutation test for the injection: if the argument were ignored, the spy
    above would prove nothing."""
    def flat(x, orig_sr, target_sr):
        return np.zeros((x.shape[0], int(round(x.shape[-1] * target_sr / orig_sr))),
                        dtype=np.float32)

    _, manifest, index = corpus
    out = render(_spec(manifest), index, RenderConfig(root=cfg.root, resampler=flat))
    assert float(out.wav.abs().max()) == 0.0


def test_resampling_preserves_a_tone_rather_than_shifting_it(tmp_path):
    """`resample_poly` is linear phase and compensates its own delay -- which
    the frame targets depend on."""
    import soundfile as sf
    sr, target = 48_000, 16_000
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), x, sr)
    got = load_audio(path, sample_rate=target, duration_s=2.0)[0]
    want = (0.3 * np.sin(2 * np.pi * 300 * np.arange(target * 2) / target))
    assert float(np.abs(got[800:-800] - want[800:-800]).max()) < 0.02


# --------------------------------------------------------------------------- #
# I12 -- the length regime, and the guard in its ON state

#: Every `RenderConfig` field, with the value the pipeline ships. Critical: the
#: table is asserted exhaustive below, so a new field arrives with a pinned
#: default or the suite goes red. `resampler` is compared by identity because it
#: is a callable, not a value.
_SHIPPED_RENDER_DEFAULTS = {
    "root": Path("."),
    "audio": AudioConfig(),
    "resampler": resample_poly_to,
    "crossfade_shape": "sigmoid",
    # I12, the length-regime guard. Off only for tests that build a deliberate
    # violation.
    "check_duration": True,
}


def test_the_shipped_render_defaults_are_pinned():
    """A default nothing asserts is a default anyone can flip in a diff review.

    Caveat: `check_duration` was *already* pinned, incidentally --
    `test_a_sample_below_the_floor_is_refused` runs on the module `cfg`, which
    carries the default, so flipping it to `False` turns that test red. This
    table makes the pinning deliberate and exhaustive rather than a side effect
    of one test's fixture choice, and covers the other four fields, which
    nothing pinned at all.
    """
    cfg = RenderConfig()
    assert set(_SHIPPED_RENDER_DEFAULTS) == {f.name for f in fields(RenderConfig)}, \
        "a new RenderConfig field arrived without a pinned default"
    for name, want in _SHIPPED_RENDER_DEFAULTS.items():
        got = getattr(cfg, name)
        if callable(want):
            assert got is want, name
        else:
            assert got == want, name


def test_the_length_regime_guard_fires_on_a_long_render(corpus, corpus_root):
    """The other side of the bound, which nothing covered.

    `test_a_sample_below_the_floor_is_refused` mutation-tests the floor. The
    ceiling had no test at all, and it is the half docs/pipelines/04 §1 is
    actually about: `tanh` attention was correct at T~=250 and had become a mean
    pool at T~=3000, ~12x worse.

    Caveat: driven by narrowing `AudioConfig` rather than by rendering a 61 s
    sample -- the synthetic corpus holds 7-10 s sources, and padding one out to
    61 s would be testing the pad rather than the bound. The quantity compared
    is the same one either way.
    """
    _, manifest, index = corpus
    narrow = RenderConfig(root=corpus_root,
                          audio=AudioConfig(min_seconds=4.0, max_seconds=5.0))
    with pytest.raises(ValueError, match="I12"):
        render(_spec(manifest, duration=6.0), index, narrow)


def test_the_guard_is_the_only_thing_that_rejects_an_out_of_regime_render(
        corpus, corpus_root):
    """The other half of both mutations: with the guard off, both renders succeed.

    Without this, the two `raises` tests would still pass if a 1.5 s or 6 s spec
    were unrenderable for some unrelated reason, and the check they name would
    not be the thing under test.
    """
    _, manifest, index = corpus
    off = RenderConfig(root=corpus_root, check_duration=False)
    short = render(_spec(manifest, duration=1.5), index, off)
    assert short.duration_s == pytest.approx(1.5, abs=1e-6)
    assert short.duration_s < AudioConfig().min_seconds

    narrow_off = RenderConfig(root=corpus_root, check_duration=False,
                              audio=AudioConfig(min_seconds=4.0, max_seconds=5.0))
    long_ = render(_spec(manifest, duration=6.0), index, narrow_off)
    assert long_.duration_s == pytest.approx(6.0, abs=1e-6)
    assert long_.duration_s > narrow_off.audio.max_seconds


# --------------------------------------------------------------------------- #
# every RenderConfig field is a knob, and every knob moves the rendered sample
#
# Same template and same known limit as `tests/test_loop.py`'s `_KNOB_PROBES`:
# it proves a knob does *something*, not that it does the right thing.


def _fourier_resample(x, orig_sr, target_sr):
    """A genuinely different kernel, which is the point of the injection.

    Not a mangled copy of `resample_poly_to`: `scipy.signal.resample` is the
    Fourier method, and swapping the kernel is a live plan -- if G1's dummy-file
    forensics name the organizers' resampler, matching it is worth more than
    kernel quality (A-S1). This is what that swap looks like.
    """
    from scipy.signal import resample as _fourier

    if orig_sr == target_sr:
        return np.ascontiguousarray(x, dtype=np.float32)
    n = int(round(x.shape[-1] * target_sr / orig_sr))
    return np.ascontiguousarray(_fourier(x.astype(np.float64), n, axis=-1),
                                dtype=np.float32)


#: `field -> (value a, value b, observable, spec kwargs)`. The spec differs per
#: probe because two of these knobs are only readable on a particular sample:
#: `crossfade_shape` needs a sequential spec with a crossfade, and
#: `check_duration` needs one outside the length regime.
_RENDER_KNOBS: dict[str, tuple] = {
    "audio": (AudioConfig(), AudioConfig(sample_rate=8_000), "shape", {}),
    "resampler": (resample_poly_to, _fourier_resample, "wav", {}),
    "crossfade_shape": ("sigmoid", "linear", "wav",
                        {"structure": "sequential", "crossfade_ms": 50.0}),
    "check_duration": (True, False, "raises", {"duration": 1.5}),
}

#: Fields no probe here can move, with the reason. An entry is a hole, not an
#: exemption.
_NOT_A_RENDER_KNOB = {
    "root": "where the corpus is read FROM rather than a rendering parameter, so "
            "it cannot be varied against a fixed source the way these are; it "
            "has its own test below",
}


def _render_observables(root, manifest, index, spec_kwargs, **overrides):
    """One rendered sample, reduced to everything a `RenderConfig` knob can move."""
    cfg = RenderConfig(root=root, **overrides)
    try:
        out = render(_spec(manifest, **spec_kwargs), index, cfg)
    except ValueError:
        return {"raises": True, "wav": None, "shape": None}
    return {"raises": False,
            "wav": hashlib.blake2b(out.wav.numpy().tobytes()).hexdigest(),
            "shape": tuple(out.wav.shape)}


def test_the_knob_table_names_every_render_config_field():
    """A new knob arrives with a probe or an admitted hole; there is no third
    option, and adding the field alone turns this red."""
    named = set(_RENDER_KNOBS) | set(_NOT_A_RENDER_KNOB)
    assert named == {f.name for f in fields(RenderConfig)}


@pytest.mark.parametrize("field", sorted(_RENDER_KNOBS))
def test_every_render_config_knob_changes_the_rendered_sample(field, corpus,
                                                              corpus_root):
    _, manifest, index = corpus
    a_value, b_value, key, spec_kwargs = _RENDER_KNOBS[field]
    a = _render_observables(corpus_root, manifest, index, spec_kwargs,
                            **{field: a_value})
    b = _render_observables(corpus_root, manifest, index, spec_kwargs,
                            **{field: b_value})
    assert a[key] != b[key], (
        f"RenderConfig.{field} = {a_value!r} and {b_value!r} produced the same "
        f"{key}: the knob validates and then does nothing")


def test_root_is_where_the_audio_is_read_from(corpus, corpus_root, tmp_path):
    """`root`'s own probe: it is the source, so it cannot be varied against a
    fixed one the way the table's knobs are.

    Two directions, because either alone is weak: a root that resolves
    elsewhere must give byte-identical audio (so `root` is a path prefix and
    nothing more), and a root that resolves nowhere must fail loudly with the
    file named (P-S1 -- a decode crash burns one of three daily submissions).
    """
    _, manifest, index = corpus
    spec = _spec(manifest)
    link = tmp_path / "elsewhere"
    link.symlink_to(corpus_root, target_is_directory=True)
    here = render(spec, index, RenderConfig(root=corpus_root))
    there = render(spec, index, RenderConfig(root=link))
    assert torch.equal(here.wav, there.wav)

    with pytest.raises(DecodeError, match="file_id="):
        render(spec, index, RenderConfig(root=tmp_path / "nothing_here"))
