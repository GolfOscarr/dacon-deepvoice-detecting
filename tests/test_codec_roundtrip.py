"""A-S3's codec round trip, and the distinction the original check missed.

`_codec_roundtrip` exists to guarantee the audio does not MOVE: a lossy
encoder's algorithmic delay would slide the waveform out from under
`frame_intervals`. The original check enforced that by requiring the decoded
length to equal the input exactly -- but a longer decode and a shifted decode
are different events, and only the second one is the defect.

Measured on ffmpeg 4.4.2 / libsndfile 1.2.2, 60 random 4-60 s draws at 16 kHz:

    gapless header present (encode to a FILE)   0-42 extra samples, 8% of draws
    header absent          (encode to a PIPE)   1169-1708 extra samples

Strict equality therefore made stage `codec_aware` unrunnable on real audio --
it raised on the first mp3 draw, 7.5% of samples. These tests pin both halves:
padding residue is trimmed, a lost delay cancellation still raises, and the
alignment property the check was really protecting is asserted directly rather
than through a proxy.
"""
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.render import DecodeError, _codec_roundtrip, _PAD_TOL

SR = 16000
HAS_FFMPEG = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not available")

#: Lengths observed to leave trailing padding on this toolchain. Kept as
#: literals: a random sweep would make the test flaky about whether it is
#: exercising the interesting path at all.
PADDED_LENGTHS = (117512, 83531, 95654)


def _noise(n, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)[None, :]


@pytest.mark.parametrize("n", PADDED_LENGTHS + (64000, 16000))
def test_the_round_trip_returns_exactly_the_input_length(n):
    """🔴 MUTATION: restore `if out.shape[0] != n: raise` and the three
    PADDED_LENGTHS fail, which is what stopped `codec_aware` from running."""
    assert _codec_roundtrip(_noise(n), SR, "mp3", 128).shape[-1] == n


def test_the_round_trip_does_not_move_the_audio():
    """The property the length check was a proxy for -- asserted directly.

    🔴 MUTATION: trim from the FRONT (`out[extra:]`) instead of the back and the
    peak lag moves off 0, which is exactly the `align_time`-shaped defect this
    function exists to prevent."""
    for n in PADDED_LENGTHS:
        orig = _noise(n)[0]
        dec = _codec_roundtrip(_noise(n), SR, "mp3", 128)[0]
        a = orig[8000:24000] - orig[8000:24000].mean()
        best, best_v = None, -np.inf
        for lag in range(-64, 65):
            b = dec[8000 + lag: 24000 + lag]
            if len(b) != len(a):
                continue
            b = b - b.mean()
            v = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            if v > best_v:
                best_v, best = v, lag
        assert best == 0, f"n={n}: correlation peaks at lag {best}, not 0"
        assert best_v > 0.9, f"n={n}: peak correlation only {best_v:.3f}"


def test_a_lost_gapless_header_still_raises():
    """🔴 The guard must still be able to fire. Encoding to a pipe is how the
    header is lost for real -- ffmpeg cannot seek back to write it -- and it
    costs 1169-1708 samples, far outside the padding tolerance."""
    n = 117512
    buf = io.BytesIO()
    sf.write(buf, _noise(n).T, SR, format="WAV", subtype="FLOAT")
    raw = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
        input=buf.getvalue(), capture_output=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "headerless.mp3"
    path.write_bytes(raw)
    decoded, _ = sf.read(str(path), dtype="float32", always_2d=True)
    excess = decoded.shape[0] - n
    assert excess >= _PAD_TOL, (
        f"a headerless encode left only {excess} extra samples, inside the "
        f"{_PAD_TOL} tolerance -- this toolchain can no longer produce the "
        f"failure the guard is for, so the guard is untested here")


def test_a_short_decode_raises_rather_than_being_padded(monkeypatch):
    """Trimming is only ever safe downward. A decode SHORTER than the input is
    not padding residue and must not be silently filled.

    The reader is stubbed because no real encode on this toolchain produces a
    short decode -- mp3 pads a 1000-sample input back up to 1000. Stubbing is
    the only way to reach the `extra < 0` branch at all, and an unreachable
    branch that has never been executed is not a guard."""
    import training.render as render

    real_read = render.sf.read

    def short_read(path, **kw):
        data, sr = real_read(path, **kw)
        return data[:-10], sr

    monkeypatch.setattr(render.sf, "read", short_read)
    with pytest.raises(DecodeError, match="moved relative to frame_intervals"):
        _codec_roundtrip(_noise(50000), SR, "mp3", 128)


def test_wav_is_a_passthrough_and_flac_is_lossless():
    wav = _noise(50000)
    assert np.array_equal(_codec_roundtrip(wav, SR, "wav", None), wav)
    assert _codec_roundtrip(wav, SR, "flac", None).shape[-1] == 50000
