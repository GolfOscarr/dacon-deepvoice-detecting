"""docs/training/15 N4 -- the phone-channel legs, packet loss and compression.

Every leg must keep the audio where ``frame_intervals`` say it is: the length
comes back exact and the cross-correlation peak sits within 1 ms (16 samples)
of lag 0 -- the rule `training.render.CODEC_CONTAINERS` states for any lossy
round trip. Every leg must be deterministic per spec (I10), and every new
knob must default OFF so the existing configs draw the stream they drew.
"""
import subprocess

import numpy as np
import pytest
import torch

from processing.config import (AUGMENTS_V1, NORMALIZE_MENU_V1, AugmentSpec, DrawConfig,
                               NormalizeMenu)
from training.registries import AUGMENT
from training.render import (NORMALIZE_KEYS, PHONE_CODECS, DecodeError, _normalize,
                             _phone_roundtrip, resample_poly_to)

SR = 16000


@pytest.fixture(scope="module")
def manifest():
    """tests/test_processing_sampler.py's corpus: voice buckets of ~100 files on
    both sides, so the draw does not exhaust a real speaker."""
    from training.synthetic import synthetic_manifest
    m = synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)
    v = m.index[m.pool.isin(["A", "B"])]
    m.loc[v, "speaker_ref_id"] = [f"{p}_spk{i % 2}" for i, p in enumerate(m.loc[v, "pool"])]
    return m


HAS_FFMPEG = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available")

#: 1 ms at 16 kHz -- the tolerance docs/training/15 N4 sets for a cancelled delay.
MAX_LAG = 16

#: One draw per leg, as `processing.sampler` writes it into ``spec.normalize``.
LEGS = {
    "gsm": {"telephone_hz": 8000, "phone_codec": "gsm"},
    "opus_8": {"phone_codec": "opus", "phone_bitrate": 8},
    "opus_12": {"phone_codec": "opus", "phone_bitrate": 12},
    "opus_16": {"phone_codec": "opus", "phone_bitrate": 16},
    "opus_24": {"phone_codec": "opus", "phone_bitrate": 24},
    "mp3_32": {"container": "mp3", "bitrate": 32},
    "mp3_48": {"container": "mp3", "bitrate": 48},
}


def _speechlike(n, seed=0, channels=1):
    """A voiced signal a speech codec can code: harmonics of a gliding 90-220 Hz
    pitch under a 4 Hz syllable envelope, plus a little noise. White noise is
    the wrong probe here -- GSM and 8 kbps Opus model speech, not noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 150.0 + 60.0 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 24))
    env = (0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))) ** 2
    x = x * env + 0.01 * rng.standard_normal(n)
    x = 0.1 * x / np.sqrt(np.mean(x ** 2))
    return np.repeat(x[None, :], channels, axis=0).astype(np.float32)


def _lag(ref, out, maxlag=400):
    """Where ``out`` best matches ``ref``, in samples, and the peak's correlation."""
    ref = ref - ref.mean()
    out = out - out.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(ref))))
    c = np.fft.irfft(np.fft.rfft(out, size) * np.conj(np.fft.rfft(ref, size)), size)
    c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
    k = int(np.argmax(c))
    return k - maxlag, float(c[k] / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-12))


# --------------------------------------------------------------------------- #
# the codec legs


@needs_ffmpeg
@pytest.mark.parametrize("leg", sorted(LEGS))
@pytest.mark.parametrize("n", (64000, 64123, 117512, 16000 * 12 + 7))
def test_every_leg_preserves_the_length_exactly(leg, n):
    out = _normalize(_speechlike(n), LEGS[leg], SR, resample_poly_to)
    assert out.shape == (1, n) and out.dtype == np.float32


@needs_ffmpeg
@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_cancels_its_delay_to_within_one_ms(leg):
    """🔴 MUTATION: drop the Ogg pre-skip (``-f opus`` to a raw stream) or trim
    the GSM padding from the front and the peak leaves the 16-sample window."""
    for n, seed in ((64123, 0), (16000 * 12 + 7, 1)):
        x = _speechlike(n, seed)
        out = _normalize(x, LEGS[leg], SR, resample_poly_to)
        lag, corr = _lag(x[0], out[0])
        assert abs(lag) <= MAX_LAG, f"{leg}: correlation peaks at lag {lag}"
        assert corr > 0.6, f"{leg}: peak correlation only {corr:.3f}"


@needs_ffmpeg
@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_is_bitwise_deterministic(leg):
    x = _speechlike(50011, seed=3, channels=2)
    a = _normalize(x.copy(), LEGS[leg], SR, resample_poly_to)
    b = _normalize(x.copy(), LEGS[leg], SR, resample_poly_to)
    assert np.array_equal(a, b)


@needs_ffmpeg
@pytest.mark.parametrize("leg", ("gsm", "opus_16"))
def test_a_phone_leg_is_dual_mono_and_keeps_the_channel_count(leg):
    x = _speechlike(40000, channels=2)
    x[1] *= 0.5
    out = _normalize(x, LEGS[leg], SR, resample_poly_to)
    assert out.shape == x.shape and np.array_equal(out[0], out[1])


@needs_ffmpeg
def test_the_phone_codecs_actually_degrade_the_signal():
    """The legs are not pass-throughs: GSM removes everything above 4 kHz (the
    8 kHz leg) and every codec leaves a residual."""
    x = _speechlike(64000)
    for leg in ("gsm", "opus_8"):
        out = _normalize(x, LEGS[leg], SR, resample_poly_to)
        assert np.sqrt(np.mean((out - x) ** 2)) > 0.01 * np.sqrt(np.mean(x ** 2)), leg
    gsm = _normalize(x, LEGS["gsm"], SR, resample_poly_to)[0]
    spec = np.abs(np.fft.rfft(gsm)) ** 2
    f = np.fft.rfftfreq(len(gsm), 1 / SR)
    assert spec[f > 4200].sum() < 1e-3 * spec.sum()


def test_a_padding_beyond_the_tolerance_raises(monkeypatch):
    """🔴 The guard can fire: a decode that comes back longer than the codec's
    trailing padding is a delay no longer cancelled, and must not be trimmed."""
    import training.render as tr
    real = tr.sf.read

    def late(*a, **kw):
        out, sr = real(*a, **kw)
        return np.concatenate([np.zeros(700, out.dtype), out]), sr
    monkeypatch.setattr(tr.sf, "read", late)
    with pytest.raises(DecodeError, match="no longer"):
        _phone_roundtrip(_speechlike(16000)[:, ::2].copy(), 8000, "gsm", None)


@pytest.mark.parametrize("draw, match", [
    ({"phone_codec": "amrnb"}, "phone_codec must be"),
    ({"phone_codec": "opus", "phone_bitrate": 7}, "phone_bitrate must be"),
    ({"phone_codec": "opus"}, "phone_bitrate must be"),
    ({"phone_codec": "gsm"}, "runs at 8000"),                 # gsm without the 8 kHz leg
    ({"telephone_hz": 8000, "phone_codec": "gsm", "phone_bitrate": 13}, "fixed-rate"),
    ({"telephone_hz": 8000, "phone_codec": "gsm", "companding": "ulaw"}, "a draw takes one"),
    ({"phone_bitrate": 8}, "needs phone_codec"),
])
def test_malformed_phone_draws_are_refused(draw, match):
    with pytest.raises(ValueError, match=match):
        _normalize(_speechlike(16000), draw, SR, resample_poly_to)


# --------------------------------------------------------------------------- #
# the menu and the draw


def test_the_new_knobs_default_off():
    """Existing configs draw the stream they drew: no phone codec in the v1
    menu, no new augment in the v1 list."""
    assert not any(k.partition("_")[0] in PHONE_CODECS for k in NORMALIZE_MENU_V1.telephone)
    assert not {"packet_loss", "compression"} & {a.name for a in AUGMENTS_V1}
    assert {"phone_codec", "phone_bitrate"} <= NORMALIZE_KEYS


@pytest.mark.parametrize("telephone, hz", [
    ({"none": 0.5, "opus_7": 0.5}, 8000),
    ({"none": 0.5, "opus": 0.5}, 8000),
    ({"none": 0.5, "gsm_13": 0.5}, 8000),
    ({"none": 0.5, "amrnb_12": 0.5}, 8000),
    ({"none": 0.5, "gsm": 0.5}, 16000),
])
def test_malformed_phone_menus_are_rejected(telephone, hz):
    with pytest.raises(ValueError):
        NormalizeMenu(container={"wav": 1.0}, channels={"mono": 1.0},
                      telephone=telephone, telephone_hz=hz)


RUN5_MENU = NormalizeMenu(
    container={"wav": 0.5, "mp3_32": 0.25, "mp3_48": 0.25},
    channels={"mono": 0.5, "stereo": 0.5},
    telephone={"none": 0.4, "ulaw": 0.1, "gsm": 0.2, "opus_8": 0.1, "opus_24": 0.2})


def test_the_draw_writes_the_phone_keys_at_the_menu_rates(manifest):
    from processing.sampler import Sampler
    specs = list(Sampler(manifest, DrawConfig(normalize_menu=RUN5_MENU)).epoch_specs(4000))
    n = len(specs)
    for s in specs:
        assert set(s.normalize) <= NORMALIZE_KEYS
        if s.normalize.get("phone_codec") == "gsm":
            assert s.normalize["telephone_hz"] == 8000 and "phone_bitrate" not in s.normalize
            assert "companding" not in s.normalize
        if s.normalize.get("phone_codec") == "opus":
            assert "telephone_hz" not in s.normalize            # wideband: no 8 kHz leg
            assert s.normalize["phone_bitrate"] in (8, 24)
    gsm = sum(s.normalize.get("phone_codec") == "gsm" for s in specs) / n
    o8 = sum(s.normalize.get("phone_bitrate") == 8 for s in specs) / n
    m32 = sum(s.normalize.get("bitrate") == 32 for s in specs) / n
    assert abs(gsm - 0.2) < 0.03 and abs(o8 - 0.1) < 0.03 and abs(m32 - 0.25) < 0.03
    # label-independent: the codec rate is the same for real and fake files
    for label in (0, 1):
        group = [s for s in specs if s.file_fake == label]
        rate = sum("phone_codec" in s.normalize for s in group) / len(group)
        assert abs(rate - 0.5) < 0.04, (label, rate)


def test_the_phone_draw_is_taken_before_the_cell_and_moves_nothing_else(manifest):
    """Adding the codec options changes WHICH telephone leg a sample gets and
    nothing else: one ``rng.choice`` either way, so every other draw of the
    sample is the same."""
    from processing.sampler import Sampler
    base = NormalizeMenu(container={"wav": 1.0}, channels={"mono": 1.0},
                         telephone={"none": 0.8, "ulaw": 0.2})
    phone = NormalizeMenu(container={"wav": 1.0}, channels={"mono": 1.0},
                          telephone={"none": 0.8, "opus_8": 0.2})    # sorts where ulaw does
    a = Sampler(manifest, DrawConfig(normalize_menu=base))
    b = Sampler(manifest, DrawConfig(normalize_menu=phone))
    for i in range(200):
        sa, sb = a.sample_spec(i), b.sample_spec(i)
        assert sa.components == sb.components and sa.cell == sb.cell
        assert sa.transforms == sb.transforms
        assert ("companding" in sa.normalize) == ("phone_codec" in sb.normalize)


# --------------------------------------------------------------------------- #
# packet loss and compression


def _probe_lag(name, params, n=48000, seed=5):
    x = torch.from_numpy(_speechlike(n, seed, channels=2))
    out = AUGMENT.build(name, params)(x, np.random.default_rng(seed))
    return x, out


@pytest.mark.parametrize("name", ("packet_loss", "compression"))
def test_the_new_augments_are_registered_with_no_delay(name):
    assert name in AUGMENT and AUGMENT.group_delay_of(name) == 0


@pytest.mark.parametrize("name, params", [
    ("packet_loss", {"rate": 0.05}),
    ("packet_loss", {"rate": 0.05, "repeat_prob": 1.0}),
    ("compression", {"threshold_db": -12.0, "ratio": 8.0, "release_ms": 50.0,
                     "target_dbfs": -14.0}),
])
def test_the_new_augments_keep_length_and_time_and_are_deterministic(name, params):
    x, out = _probe_lag(name, params)
    assert out.shape == x.shape and out.dtype == x.dtype
    lag, _ = _lag(x[0].numpy().astype(np.float64), out[0].numpy().astype(np.float64))
    assert lag == 0, f"{name} moved the audio by {lag} samples"
    _, again = _probe_lag(name, params)
    assert torch.equal(out, again)


def test_packet_loss_drops_whole_frames_at_the_rate_with_either_concealment():
    n, frame = 16000 * 30, 320
    x = torch.from_numpy(_speechlike(n) + 0.05)                   # no exact zeros
    zero = AUGMENT.build("packet_loss", {"rate": 0.03, "repeat_prob": 0.0})(
        x, np.random.default_rng(1))
    frames = zero[0, :n - n % frame].reshape(-1, frame)
    lost = (frames == 0).all(dim=1)
    assert torch.equal((frames == 0).any(dim=1), lost)            # whole frames only
    assert 0.02 < float(lost.float().mean()) < 0.04
    rep = AUGMENT.build("packet_loss", {"rate": 0.03, "repeat_prob": 1.0})(
        x, np.random.default_rng(1))
    changed = (rep[0, :n - n % frame].reshape(-1, frame)
               != x[0, :n - n % frame].reshape(-1, frame)).any(dim=1)
    for i in torch.nonzero(changed).flatten().tolist():            # a copy of the frame before
        assert torch.equal(rep[0, i * frame:(i + 1) * frame],
                           rep[0, (i - 1) * frame:i * frame])


def test_compression_reduces_the_crest_and_lands_on_its_target():
    x = _speechlike(16000 * 8)
    x[:, 16000:20000] *= 6.0                                      # a loud burst
    x = torch.from_numpy(x)
    out = AUGMENT.build("compression", {"threshold_db": -6.0, "ratio": 6.0,
                                        "release_ms": 100.0, "target_dbfs": -20.0})(
        x, np.random.default_rng(0))

    def crest(w):
        w = w.double()
        return float(w.abs().max() / w.pow(2).mean().sqrt())
    assert crest(out) < crest(x)
    rms_db = 20 * np.log10(float(out.double().pow(2).mean().sqrt()))
    assert abs(rms_db - -20.0) < 0.1
    assert float(out.abs().max()) <= 10 ** (-0.1 / 20) + 1e-6


def test_augment_specs_accept_the_drawn_ranges():
    AugmentSpec("packet_loss", 0.1, {"rate_range": (0.01, 0.05)})
    AugmentSpec("compression", 0.2, {"threshold_db_range": (-12.0, 0.0),
                                     "ratio_range": (2.0, 8.0),
                                     "release_ms_range": (50.0, 300.0),
                                     "target_dbfs_range": (-26.0, -14.0)})
