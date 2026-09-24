"""R1's two measurement planes, from a single decode.

Every S-tier statistic is computed twice: once on the file as the publisher
shipped it (`native`) and once on what the model actually receives (`chain`).
The derived column that matters is the **difference** -- a statistic that
separates classes at `native` and stops separating them at `chain` is a
fingerprint that does not survive the competition's own 16 kHz standardization,
which is the premise the music-head strategy rests on (docs/EDA/00 section 2).

Critical: **one decode, two planes.** The chain plane is built from the native
plane's samples by the same `resample_poly_to` that `training.render.load_audio`
would have called internally, in the same order, so the two routes are not
merely equivalent by argument -- `test_the_chain_plane_is_byte_identical_to_the
_training_path` decodes both ways and compares the arrays. Decoding twice would
halve the S tier's throughput and, worse, leave two decode paths that can drift.

Critical: **a plane is data, not a flag.** `Plane` carries its name for the
column suffix, but `SIGNAL` extractors are bound to `(wav, sample_rate)` and
registration refuses a third parameter (`eda.extract`), so an extractor
*cannot* branch on which plane it is running -- R1 is a property of the type.

⚠️ **The chain is what the code does, not what the plan wishes it did.**
docs/EDA/00 section 2 describes the chain as `decode -> resample -> channel
policy -> DC removal -> (normalize)`. There is no DC removal and no
normalization anywhere in `training.render` or `models.audio` today, so this
module applies neither: a chain plane carrying a step the model does not have
would measure a signal nothing receives. `level.dc_offset` is measured on both
planes precisely so the question "should there be one?" is answered with the
corpus rather than assumed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from models.audio import prepare_waveform
from models.config import AudioConfig
from training.render import load_audio, resample_poly_to

__all__ = ["NATIVE", "CHAIN", "PLANES", "Plane", "NativeRateError", "load_planes",
           "native_rate"]

NATIVE = "native"
CHAIN = "chain"
#: The order columns are suffixed in. Fixed so a table's column order is stable
#: across runs and two `signal.parquet` files can be diffed.
PLANES = (NATIVE, CHAIN)


class NativeRateError(RuntimeError):
    """The file's real sample rate could not be established, or it disagrees
    with what the M tier recorded.

    Its own type because the two failures it covers are *findings*, not
    plumbing: a file libsndfile and ffprobe both refuse is corruption (F-S2),
    and a rate that disagrees with `files.parquet` means the census and the
    decoder are looking at different files.
    """


@dataclass(frozen=True)
class Plane:
    """One plane's samples. `wav` is `(C, n)` float32, as `load_audio` returns."""

    name: str
    wav: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.wav.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.wav.shape[-1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


def _ffprobe_rate(path: Path, ffprobe: str = "ffprobe") -> int | None:
    """The first audio stream's rate, or None. Used only when libsndfile refuses."""
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    if proc.returncode:
        return None
    head = proc.stdout.strip().splitlines()
    try:
        return int(head[0])
    except (IndexError, ValueError):
        return None


def native_rate(path: str | Path, declared: int | None = None,
                ffprobe: str = "ffprobe") -> int:
    """The file's own sample rate, cross-checked against the M tier's.

    🔴 `declared` is the `orig_sr` already in `files.parquet`, and passing it is
    not belt-and-braces. The native plane is produced by asking `load_audio` for
    the file's own rate, and `resample_poly_to` short-circuits only when the two
    match. Hand it a wrong rate and it *resamples*, silently, and every
    "native" statistic is measured on an upsampled signal -- a whole plane that
    looks fine and answers a question nobody asked.
    """
    path = Path(path)
    try:
        rate: int | None = int(sf.info(str(path)).samplerate)
    except Exception:                                # noqa: BLE001 -- ffmpeg's turn
        rate = _ffprobe_rate(path, ffprobe)
    if not rate or rate <= 0:
        raise NativeRateError(
            f"{path}: neither libsndfile nor ffprobe could read a sample rate. "
            f"The file is unreadable, which is a finding (F-S2), not an error")
    if declared is not None and int(declared) != rate:
        raise NativeRateError(
            f"{path}: the decoder reads {rate} Hz and files.parquet records "
            f"{int(declared)} Hz. The census and the decoder disagree about this "
            f"file; measuring it would put an upsampled signal in the native plane")
    return rate


def load_planes(path: str | Path, *, declared_sr: int | None = None,
                chain_sr: int = 16_000, audio: AudioConfig | None = None,
                file_id: str = "") -> dict[str, Plane]:
    """`{"native": Plane, "chain": Plane}` from one decode of `path`.

    `declared_sr` is the M tier's `orig_sr` for this file; see `native_rate`.
    `audio` supplies the channel policy -- the default `AudioConfig` is the one
    `configs/run_default.yaml` ships (`downmix`).
    """
    audio = audio or AudioConfig(sample_rate=chain_sr)
    rate = native_rate(path, declared_sr)
    wav = load_audio(path, sample_rate=rate, file_id=file_id)
    native = Plane(NATIVE, wav, rate)

    # The same call `load_audio` makes internally, on the same samples.
    resampled = resample_poly_to(wav, rate, chain_sr)
    # `prepare_waveform` is (B, C, S) -> (B, S); the batch axis is borrowed for
    # one file and dropped again. Going through the real function rather than a
    # local `.mean(0)` is the point: `channels: mid_side` is a live option and a
    # second implementation of it would drift from the one the model uses.
    mono = prepare_waveform(torch.from_numpy(resampled)[None], audio)[0]
    chain = Plane(CHAIN, np.ascontiguousarray(mono.numpy(), dtype=np.float32)[None],
                  chain_sr)
    return {NATIVE: native, CHAIN: chain}
