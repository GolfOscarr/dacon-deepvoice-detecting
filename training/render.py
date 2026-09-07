"""Turning a `SampleSpec` into audio. All the I/O of the pipeline lives here.

The split that everything rests on (docs/pipelines/01 §1)::

    sample_spec(rng, manifest, slice, fold) -> SampleSpec   # pure, microseconds
    render(spec, manifest)                  -> RenderedSample

🔴 **`render(spec) == render(spec)`, bitwise.** The only entropy is
``spec.rng`` -- blake2b-keyed on ``(sample_id, epoch, seed)`` -- and every stage
below is a deterministic function of the spec, the files it names and that one
generator. This is I10, and it is what makes the 2nd-stage submission able to
reproduce the Private score (docs/data/06 A-S2, docs/data/09 R9).

The stages, in the order docs/data/06 fixes::

    decode        P-S1 robust decode + P-S2 resample to 16 kHz, per file
    compose       step 3 -- overlap (gain ratio) or sequential (crossfade)
    augment       step 4 -- the AUGMENT registry, label-independent by signature
    normalize     step 5 -- TEST-CHAIN, always last

⚠️ **The preprocess registry is not run here**, and that is deliberate. A
preprocess step is train/test *symmetric* (docs/data/10 P2), so its call site is
the model boundary -- next to ``models.audio.prepare_waveform`` and
``models.audio.bandpass``, in the training loop and in ``submit.zip`` alike. A
rendered sample is the analogue of a *raw test file*: the last thing done to it
is what the organizers did to theirs.
"""

from __future__ import annotations

import io
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor

from models.config import AudioConfig
from models.losses import TARGET_FOR_COLUMN
from training.registries import augment_chain
from training.spec import SampleSpec

__all__ = [
    "CODEC_CONTAINERS", "NORMALIZE_KEYS",
    "DecodeError", "ManifestIndex", "RenderConfig", "RenderedSample",
    "frame_intervals_for", "load_audio", "render", "resample_poly_to",
]

#: Containers A-S3 can round-trip through. ⚠️ Short of the full menu: AAC, OPUS,
#: AMR-NB, GSM and G.722 are named in A-S3/A-S4 but are not wired, because each
#: needs its encoder delay verified the way MP3's is below and the local ffmpeg
#: does not carry every encoder. The 8 kHz telephone leg is implemented (see
#: `_normalize`); the narrowband *codecs* are not.
CODEC_CONTAINERS = ("wav", "flac", "mp3")

#: The keys ``SampleSpec.normalize`` may carry. Unknown keys are an error, not a
#: warning -- ``models.config``'s rule, for the same reason: a typo'd knob that
#: silently does nothing is how an ablation measures the wrong thing.
NORMALIZE_KEYS = frozenset({
    "resampler", "container", "bitrate", "channels", "telephone_hz", "companding",
})


class DecodeError(RuntimeError):
    """A file the pipeline could not turn into samples, named loudly.

    P-S1: the eval server hands us mixed containers and a decode crash burns one
    of three daily submissions, so every decode failure carries the ``file_id``
    and the path that produced it.
    """


# --------------------------------------------------------------------------- #
# Manifest access


class ManifestIndex(Mapping[str, Mapping[str, Any]]):
    """``file_id -> row``, built once so rendering is not O(manifest) per sample."""

    def __init__(self, rows: Mapping[str, Mapping[str, Any]]):
        self._rows = dict(rows)

    @classmethod
    def from_frame(cls, manifest: pd.DataFrame) -> ManifestIndex:
        return cls({str(r["file_id"]): r
                    for r in manifest.to_dict(orient="records")})

    @classmethod
    def coerce(cls, manifest: pd.DataFrame | ManifestIndex) -> ManifestIndex:
        if isinstance(manifest, ManifestIndex):
            return manifest
        return cls.from_frame(manifest)

    def __getitem__(self, file_id: str) -> Mapping[str, Any]:
        return self._rows[file_id]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


# --------------------------------------------------------------------------- #
# Decode -- P-S1 and P-S2


def resample_poly_to(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """``scipy.signal.resample_poly`` over ``(C, n)``.

    ✅ The resampler is decided (docs/data/06, A-S1): the requirement is *one
    fixed resampler applied identically in train and test*, not a specific
    library, and ``scipy`` is already a dependency while ``soxr`` is not --
    adding a shipped dependency costs against the offline-install budget and
    buys quality we cannot measure.

    🔴 Injectable on purpose. If G1's dummy-file forensics reveal which resampler
    the organizers used, *matching them* is worth more than kernel quality, and
    the swap is then one argument (``RenderConfig.resampler``).

    ``resample_poly`` is linear phase and compensates its own filter delay, so
    this introduces no time shift -- which the frame targets depend on.
    """
    if orig_sr == target_sr:
        return np.ascontiguousarray(x, dtype=np.float32)
    g = math.gcd(int(orig_sr), int(target_sr))
    out = resample_poly(x.astype(np.float64), target_sr // g, orig_sr // g, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _ffmpeg_decode(path: Path, file_id: str = "") -> tuple[np.ndarray, int]:
    """Fallback decode for anything libsndfile refuses (P-S1's second path).

    🔴 Decodes at the file's **native** rate and leaves resampling to the caller's
    injected resampler. Passing ``-ar`` here would put ffmpeg's swr on the
    fallback path and ``resample_poly`` on the main one, so which resampler a
    file met would depend on its container -- the opposite of "one fixed
    resampler applied identically" (P-S2), and invisible in any test that only
    checks the sample count.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-i", str(path), "-f", "wav", "-c:a", "pcm_f32le",
         "-bitexact", "pipe:1"],
        capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise DecodeError(
            f"ffmpeg could not decode {path} (file_id={file_id!r}): "
            f"{proc.stderr.decode('utf-8', 'replace')[-400:]}")
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sr)


def load_audio(path: str | Path, *, sample_rate: int, offset_s: float = 0.0,
               duration_s: float | None = None,
               resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample_poly_to,
               file_id: str = "") -> np.ndarray:
    """P-S1 + P-S2: decode a slice of one file to ``(C, n)`` float32 at 16 kHz.

    ``n`` is exactly ``round(duration_s * sample_rate)`` -- trimmed or
    zero-padded after resampling, so the timeline the sampler drew is the
    timeline that comes back. Never assumes the extension: libsndfile sniffs the
    container, and anything it refuses goes through ffmpeg.
    """
    path = Path(path)
    want = None if duration_s is None else int(round(duration_s * sample_rate))
    try:
        with sf.SoundFile(str(path)) as fh:
            orig_sr = int(fh.samplerate)
            start = int(round(offset_s * orig_sr))
            frames = -1 if duration_s is None else int(round(duration_s * orig_sr))
            fh.seek(min(start, len(fh)))
            data = fh.read(frames=frames, dtype="float32", always_2d=True)
        wav = np.ascontiguousarray(data.T)
    except (sf.LibsndfileError, RuntimeError):
        wav, orig_sr = _ffmpeg_decode(path, file_id)
        start = int(round(offset_s * orig_sr))
        stop = None if duration_s is None else start + int(round(duration_s * orig_sr))
        wav = wav[:, start:stop]
    except OSError as exc:
        raise DecodeError(f"cannot open {path} (file_id={file_id!r}): {exc}") from exc

    if wav.size == 0 or wav.shape[0] == 0:
        raise DecodeError(
            f"{path} (file_id={file_id!r}) decoded to nothing at offset "
            f"{offset_s}s for {duration_s}s")
    if not np.isfinite(wav).all():
        raise DecodeError(f"{path} (file_id={file_id!r}) decoded non-finite samples")

    wav = resampler(wav, orig_sr, sample_rate)
    if want is not None:
        # 🔴 A file shorter than the manifest says would otherwise be padded with
        # silence, and a training sample that is 40% silence because a corpus
        # row lies about its duration is exactly the kind of defect a green
        # suite hides. Only resampler rounding (a few samples) is absorbed.
        slack = max(4, int(0.01 * sample_rate))
        if wav.shape[-1] < want - slack:
            raise DecodeError(
                f"{path} (file_id={file_id!r}) yielded "
                f"{wav.shape[-1] / sample_rate:.3f}s from offset {offset_s:.3f}s, "
                f"short of the {duration_s:.3f}s the manifest promised")
        if wav.shape[-1] > want:
            wav = wav[:, :want]
        elif wav.shape[-1] < want:
            wav = np.pad(wav, ((0, 0), (0, want - wav.shape[-1])))
    return np.ascontiguousarray(wav, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The rendered sample


@dataclass
class RenderedSample:
    """What ``render`` returns, and the only thing the collator sees.

    ⚠️ ``wav`` is ``(C, S)`` **as decoded, not downmixed**. The channel policy is
    ``AudioConfig.channels``, applied by ``models.audio.prepare_waveform`` at the
    *same call site* in training and inference. Downmixing here would fork that
    policy into two places, disable the A-B3 channel augmentations and make the
    ``mid_side`` leak test impossible to run (docs/pipelines/01 §4).

    🔴 ``frame_intervals`` are **absolute times in seconds**, never a rasterised
    frame grid. Frame rate is a property of the frontend; the collator does not
    know it and must not guess. Hard-coding one here reproduces the `align_time`
    bug in a place the existing tests do not look.
    """

    wav: Tensor                          # (C, S) float32
    sample_rate: int                     # always AudioConfig.sample_rate
    targets: dict[str, int]              # the five keys of TARGET_FOR_COLUMN
    frame_intervals: dict[str, tuple[tuple[float, float, int], ...]]
    spec: SampleSpec

    @property
    def duration_s(self) -> float:
        return self.wav.shape[-1] / self.sample_rate


@dataclass(frozen=True)
class RenderConfig:
    """Everything ``render`` needs beyond the spec and the manifest."""

    #: Manifest paths are relative to this.
    root: Path = Path(".")
    audio: AudioConfig = field(default_factory=AudioConfig)
    #: 🔴 Injected so G1 can swap it (see ``resample_poly_to``).
    resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample_poly_to
    #: ⚠️ ``sigmoid`` beats a hard cut, which becomes a splice shortcut (A-A4).
    crossfade_shape: str = "sigmoid"
    #: I12 -- the rendered duration must sit in the length regime the model and
    #: the competition agree on. Off only for tests that build a deliberate
    #: violation.
    check_duration: bool = True

    def __post_init__(self) -> None:
        if self.crossfade_shape not in ("sigmoid", "linear"):
            raise ValueError(
                f"crossfade_shape must be sigmoid|linear, got {self.crossfade_shape!r}")


# --------------------------------------------------------------------------- #
# Composition -- step 3


def _fade(n: int, shape: str) -> np.ndarray:
    """A 0->1 ramp of ``n`` samples, exactly 0 at the start and 1 at the end."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if shape == "linear":
        w = t
    else:
        # ★ `[Freesound 2019, 1st]` SigmoidConcatMixer -- a smooth transition
        # rather than a hard cut. Rescaled so the endpoints are exact.
        s = 1.0 / (1.0 + np.exp(-8.0 * (2.0 * t - 1.0)))
        w = (s - s[0]) / (s[-1] - s[0])
    return w.astype(np.float32)


def _place(canvas: np.ndarray, piece: np.ndarray, start: int) -> None:
    """Sum ``piece`` into ``canvas`` at ``start``, clipped to the timeline."""
    total = canvas.shape[-1]
    if start >= total:
        return
    take = min(piece.shape[-1], total - start)
    canvas[:, start:start + take] += piece[:, :take]


def _to_channels(wav: np.ndarray, channels: int) -> np.ndarray:
    """Broadcast a mono source across ``channels``, or fold extras down.

    ⚠️ Mono-duplication is itself a possible cue (docs/data/07 E-B4). It is done
    here rather than downmixing everything to mono because the alternative --
    collapsing the sample -- would fork the channel policy away from
    ``prepare_waveform``. The test set has both mono and stereo per sample, so
    the composed sample carries whatever its sources had.
    """
    if wav.shape[0] == channels:
        return wav
    if wav.shape[0] == 1:
        return np.repeat(wav, channels, axis=0)
    return np.ascontiguousarray(wav[:channels])


def _compose(spec: SampleSpec, pieces: Sequence[np.ndarray], cfg: RenderConfig,
             sample_rate: int) -> np.ndarray:
    """Place the drawn components on the timeline the spec fixed.

    🔴 The renderer does not invent placement. ``target_start_s`` and
    ``duration_s`` are drawn before any file is opened and the spec *is* the
    ledger, so what lands here is exactly what ``frame_intervals`` reports.

    ⚠️ Consequence for A-A4: sequential components are drawn back-to-back with
    no overlap, so a true constant-power crossfade would have to move one of
    them off the timeline the spec fixed. Instead the joint gets complementary
    sigmoid tapers -- a smooth transition rather than a hard cut, which is the
    property A-A4 is after (a splice edge is a shortcut), without the renderer
    overruling the draw.
    """
    total = int(round(spec.duration_s * sample_rate))
    channels = max(p.shape[0] for p in pieces)
    canvas = np.zeros((channels, total), dtype=np.float32)
    xfade = int(round(spec.crossfade_ms / 1000.0 * sample_rate))

    for draw, piece in zip(spec.components, pieces):
        piece = _to_channels(piece, channels) * np.float32(10.0 ** (draw.gain_db / 20.0))
        start = int(round(draw.target_start_s * sample_rate))
        n = piece.shape[-1]
        if spec.structure == "sequential" and xfade > 0 and n > 2:
            k = min(xfade, n // 2)
            ramp = _fade(k, cfg.crossfade_shape)
            piece = piece.copy()
            if start > 0:                       # not the first piece: fade in
                piece[:, :k] *= ramp
            if start + n < total:               # not the last piece: fade out
                piece[:, n - k:] *= ramp[::-1]
        _place(canvas, piece, start)
    return canvas


# --------------------------------------------------------------------------- #
# Test-chain normalisation -- step 5, always last


def _mu_law(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    y = np.sign(x) * np.log1p(mu * np.abs(np.clip(x, -1.0, 1.0))) / np.log1p(mu)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)).astype(np.float32)


def _a_law(x: np.ndarray, a: float = 87.6) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    ax = np.abs(x)
    lo = a * ax / (1.0 + np.log(a))
    hi = (1.0 + np.log(a * np.maximum(ax, 1e-12))) / (1.0 + np.log(a))
    y = np.sign(x) * np.where(ax < 1.0 / a, lo, hi)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    aq = np.abs(q)
    ilo = aq * (1.0 + np.log(a)) / a
    ihi = np.exp(aq * (1.0 + np.log(a)) - 1.0) / a
    return (np.sign(q) * np.where(aq < 1.0 / (1.0 + np.log(a)), ilo, ihi)).astype(np.float32)


def _codec_roundtrip(wav: np.ndarray, sample_rate: int, container: str,
                     bitrate: int | None) -> np.ndarray:
    """A-S3 -- encode and decode again, at the drawn container and bitrate.

    ⚠️ MP3 at 64 kbps cost MusicDET **+37 EER points** (docs/survey/02) and the
    test set is MP3/WAV/FLAC, so this is not a cosmetic stage.

    🔴 **The round trip must not move the audio.** A lossy encoder has an
    algorithmic delay -- LAME's is 1105 samples, 69 ms -- which would slide the
    waveform out from under ``frame_intervals`` while they stayed put: the
    `align_time` defect, in a place the existing tests do not look. The delay is
    cancelled by the encoder's own gapless (Xing/LAME) header, which ffmpeg can
    only write when it can seek back over its output, i.e. to a **file**, never
    to a pipe. Encoding to a pipe here silently shifts every MP3 sample by 69 ms.

    ⚠️ So the length is checked rather than trusted: if the header is ever lost,
    the decoded frame count stops matching and this raises instead of shifting.
    """
    if container not in CODEC_CONTAINERS:
        raise ValueError(f"container must be one of {CODEC_CONTAINERS}, "
                         f"got {container!r}")
    if container == "wav":
        return wav
    n = wav.shape[-1]
    buf = io.BytesIO()
    sf.write(buf, wav.T, sample_rate, format="WAV", subtype="FLOAT")
    codec = ["-c:a", "libmp3lame", "-b:a", f"{int(bitrate or 128)}k"] \
        if container == "mp3" else ["-c:a", "flac"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"chain.{container}"
        enc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", "pipe:0", *codec, str(out_path)],
            input=buf.getvalue(), capture_output=True, check=False)
        if enc.returncode != 0 or not out_path.exists():
            raise DecodeError(
                f"ffmpeg could not encode to {container}: "
                f"{enc.stderr.decode('utf-8', 'replace')[-400:]}")
        out, sr = sf.read(str(out_path), dtype="float32", always_2d=True)
    if int(sr) != sample_rate:                                # pragma: no cover
        raise DecodeError(f"{container} round-trip returned {sr} Hz, "
                          f"expected {sample_rate}")
    if out.shape[0] != n:
        raise DecodeError(
            f"{container} round-trip returned {out.shape[0]} samples for {n}: "
            f"the encoder delay is no longer being cancelled, so the audio has "
            f"moved relative to frame_intervals")
    return np.ascontiguousarray(out.T, dtype=np.float32)


def _normalize(wav: np.ndarray, draw: Mapping[str, Any], sample_rate: int,
               resampler: Callable[[np.ndarray, int, int], np.ndarray]) -> np.ndarray:
    """A-S1 / A-S3 / A-S4 -- what the organizers did to the test set.

    🔴 Always last, and **not** in the augment registry: this is the one stage
    that models the test chain rather than adding variety, which is why it is
    drawn into ``SampleSpec.normalize`` and applied here (docs/pipelines/03 §2).

    ⚠️ Its parameters come from **G1** ``signal_chain.yaml``, which does not
    exist yet. Until the dummy-file forensics land (E-S1), an empty draw is a
    no-op and every field is optional -- the stage is written, not parameterised.
    """
    unknown = sorted(set(draw) - NORMALIZE_KEYS)
    if unknown:
        raise ValueError(f"unknown normalize key(s) {unknown}; "
                         f"allowed: {sorted(NORMALIZE_KEYS)}")

    telephone_hz = draw.get("telephone_hz")
    if telephone_hz:
        # A-S4: 16k -> 8k -> 16k. The test set explicitly contains 전화채널 audio.
        n = wav.shape[-1]
        wav = resampler(wav, sample_rate, int(telephone_hz))
        companding = draw.get("companding")
        if companding == "ulaw":
            wav = _mu_law(wav)
        elif companding == "alaw":
            wav = _a_law(wav)
        elif companding is not None:
            raise ValueError(f"companding must be ulaw|alaw|null, got {companding!r}")
        wav = resampler(wav, int(telephone_hz), sample_rate)[:, :n]
        if wav.shape[-1] < n:
            wav = np.pad(wav, ((0, 0), (0, n - wav.shape[-1])))
    elif draw.get("companding"):
        raise ValueError("companding needs telephone_hz -- the 8 kHz leg is what "
                         "makes it the A-S4 chain and not a lone quantiser")

    channels = draw.get("channels")
    if channels == "mono":
        wav = wav.mean(axis=0, keepdims=True)
    elif channels == "stereo":
        wav = _to_channels(wav, 2)
    elif channels is not None:
        raise ValueError(f"channels must be mono|stereo|null, got {channels!r}")

    container = draw.get("container")
    if container:
        wav = _codec_roundtrip(wav, sample_rate, container, draw.get("bitrate"))
    return np.ascontiguousarray(wav, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Frame targets


def frame_intervals_for(spec: SampleSpec) -> dict[str, tuple[tuple[float, float, int], ...]]:
    """Where each component sits, in **absolute seconds**, and what it is.

    Keyed by branch (``voice`` / ``music`` / ``file``), values ``(start_s,
    end_s, label)``. Whole-file rows carry empty tuples -- there is no
    composition to describe.

    ⚠️ The default loss does not consume these: ``SEDHeadConfig.clip_weight`` is
    committed at 1.0 (clip-only), so frame targets are produced but unused.
    🔴 Produce them anyway. T1 is the first training experiment and cannot run
    without them, they are near-free here (the placement arithmetic is already
    done), and retrofitting them later means re-rendering the corpus
    (docs/pipelines/01 §4).
    """
    empty: dict[str, tuple[tuple[float, float, int], ...]] = {
        "voice": (), "music": (), "file": ()}
    if spec.render_mode != "composed":
        return empty

    fake_for_role = {"voice": spec.voice_fake, "music": spec.music_fake, "noise": 0}
    out: dict[str, list[tuple[float, float, int]]] = {"voice": [], "music": [], "file": []}
    for draw in spec.components:
        start = max(0.0, float(draw.target_start_s))
        end = min(float(spec.duration_s), start + float(draw.duration_s))
        if end <= start:
            continue
        label = int(fake_for_role.get(draw.role) or 0)
        if draw.role in ("voice", "music"):
            out[draw.role].append((start, end, label))
        # 🔴 The file branch is fake wherever a fake component is: the
        # competition defines FILE_FAKE as OR over *present* components, and a
        # frame is a smaller "present" than a file.
        out["file"].append((start, end, label))
    return {k: tuple(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# render


def render(spec: SampleSpec, manifest: pd.DataFrame | ManifestIndex,
           cfg: RenderConfig | None = None) -> RenderedSample:
    """Decode, compose, augment, normalise. ``render(spec) == render(spec)``.

    🔴 The augment chain is called as ``fn(wav, rng)`` and there is no third
    argument to put a label in -- ``P(T | L) = P(T)`` is enforced by the
    signature, not by this function's good behaviour (docs/pipelines/03 §2).
    """
    cfg = cfg or RenderConfig()
    index = ManifestIndex.coerce(manifest)
    sr = int(cfg.audio.sample_rate)

    pieces: list[np.ndarray] = []
    for draw in spec.components:
        try:
            row = index[draw.file_id]
        except KeyError:
            raise DecodeError(
                f"spec {spec.sample_id} draws file_id {draw.file_id!r}, which is "
                f"not in the manifest") from None
        pieces.append(load_audio(
            Path(cfg.root) / str(row["path"]), sample_rate=sr,
            offset_s=draw.source_offset_s, duration_s=draw.duration_s,
            resampler=cfg.resampler, file_id=draw.file_id))

    wav = _compose(spec, pieces, cfg, sr)

    # step 4 -- label-independent signal augmentation.
    rng = spec.rng
    tensor = torch.from_numpy(wav)
    tensor = augment_chain(spec.transforms)(tensor, rng)
    wav = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)

    # step 5 -- the test chain, always last.
    wav = _normalize(wav, spec.normalize, sr, cfg.resampler)

    duration = wav.shape[-1] / sr
    if cfg.check_duration:
        lo, hi = cfg.audio.min_seconds, cfg.audio.max_seconds
        # ⚠️ Asserted on the rendered length -- the quantity that reaches the
        # model -- not on `spec.duration_s`, which is the adjacent one.
        if not lo - 1e-6 <= duration <= hi + 1e-6:
            raise ValueError(
                f"rendered {duration:.3f}s, outside the [{lo}, {hi}]s length "
                f"regime AudioConfig and the competition agree on (I12)")

    labels = spec.labels
    targets = {key: int(labels[key] or 0) for key in TARGET_FOR_COLUMN.values()}
    return RenderedSample(
        wav=torch.from_numpy(wav), sample_rate=sr, targets=targets,
        frame_intervals=frame_intervals_for(spec), spec=spec)
