"""OFF-6 -- the decode cache: every manifest file decoded once, resampled to
16 kHz with the one resampler the shipped chain uses, stored as int16 at its
native channel count, and read back as a SLICE (docs/processing/03 §3).

The contract, stated exactly because it is not the one the spec wrote:

* A cached file is ``quantise(resample(full native decode))``, where
  ``resample`` is ``RenderConfig.resampler`` and ``quantise`` rounds to int16
  with clipping. A slice read from it is a slice of that array, zero-padded or
  trimmed to ``round(duration_s * 16000)`` samples exactly as
  ``training.render.load_audio`` does, and it is **bit-identical to the same
  slice of the same array** -- that is what the test asserts.
* It is NOT bit-identical to ``load_audio`` of the same slice, for two reasons
  neither of which is a defect: int16 is a quantisation (< 1 LSB, -90 dBFS),
  and ``load_audio`` resamples the *slice* while the cache slices the
  *resample*, which differ at the slice's edges by the resampler's transient.
  The cache is the better signal (no edge transient) and the faster one (no
  decode, no filter); what matters is that a run uses ONE regime throughout,
  so a renderer with ``cache_root`` set reads the cache and nothing else, and
  a missing file is an error rather than a fallback.
* ``.npy`` rather than the spec's ``.pt``: ``numpy.load(mmap_mode="r")`` reads
  the slice without touching the rest of the file, which is the point of a
  cache the crop reads a slice from.

Size: the five component pools plus cell 5 are ~620 audio-hours -> ~72 GB
mono-equivalent at 16 kHz int16. Cell 8 (SONICS, 1,973 h) is skipped by
default: it is never drawn under D-1.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from training.render import DecodeError, load_audio, resample_poly_to

__all__ = ["CacheReport", "build_cache", "cache_path", "decode_int16", "encode_int16",
           "read_slice"]

SCALE = 32768.0


def cache_path(cache_root: Path, file_id: str) -> Path:
    """``<root>/<corpus>/<path>.npy`` -- the file_id's ``corpus:path`` split."""
    corpus, sep, rel = str(file_id).partition(":")
    if not sep:                            # an id without a corpus prefix
        return Path(cache_root) / (corpus + ".npy")
    return Path(cache_root) / corpus / (rel + ".npy")


def encode_int16(wav: np.ndarray) -> np.ndarray:
    """float32 ``(C, n)`` in [-1, 1] -> int16, rounded half away from zero, clipped."""
    x = np.asarray(wav, dtype=np.float64) * SCALE
    return np.clip(np.round(x), -SCALE, SCALE - 1).astype(np.int16)


def decode_int16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32) / np.float32(SCALE))


def read_slice(path: Path, *, sample_rate: int, offset_s: float = 0.0,
               duration_s: float | None = None, file_id: str = "") -> np.ndarray:
    """A ``(C, n)`` float32 slice of one cached file, ``n = round(duration_s *
    sample_rate)`` -- the same padding / trimming rule and the same short-file
    refusal as ``load_audio``."""
    path = Path(path)
    if not path.exists():
        raise DecodeError(f"{path} (file_id={file_id!r}) is not in the cache; build it "
                          f"(scripts/build_cache.py) or unset render.cache_root")
    arr = np.load(path, mmap_mode="r")
    start = int(round(offset_s * sample_rate))
    want = None if duration_s is None else int(round(duration_s * sample_rate))
    stop = arr.shape[-1] if want is None else start + want
    piece = np.asarray(arr[:, start:stop])
    if piece.size == 0 or piece.shape[0] == 0:
        raise DecodeError(f"{path} (file_id={file_id!r}) has nothing at offset "
                          f"{offset_s}s for {duration_s}s")
    wav = decode_int16(piece)
    if want is not None:
        slack = max(4, int(0.01 * sample_rate))
        if wav.shape[-1] < want - slack:
            raise DecodeError(
                f"{path} (file_id={file_id!r}) yielded {wav.shape[-1] / sample_rate:.3f}s "
                f"from offset {offset_s:.3f}s, short of the {duration_s:.3f}s the "
                f"manifest promised")
        if wav.shape[-1] < want:
            wav = np.pad(wav, ((0, 0), (0, want - wav.shape[-1])))
    return np.ascontiguousarray(wav, dtype=np.float32)


# --------------------------------------------------------------------------- #
# building


def build_one(row: dict[str, Any], corpus_root: Path, cache_root: Path, *, sample_rate: int,
              resampler: Callable[[np.ndarray, int, int], np.ndarray]) -> dict[str, Any]:
    """Decode one manifest row in full, resample, quantise, write. Idempotent:
    an existing file is left alone (resumable)."""
    fid = str(row["file_id"])
    out = cache_path(cache_root, fid)
    if out.exists():
        return {"file_id": fid, "status": "exists", "samples": None}
    try:
        wav = load_audio(Path(corpus_root) / str(row["path"]), sample_rate=sample_rate,
                         resampler=resampler, file_id=fid)
    except DecodeError as exc:
        return {"file_id": fid, "status": "failed", "error": str(exc)[:300]}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")   # np.save appends .npy to any other name
    np.save(tmp, encode_int16(wav))
    tmp.replace(out)                     # a reader never sees a half-written file
    return {"file_id": fid, "status": "written", "samples": int(wav.shape[-1]),
            "channels": int(wav.shape[0])}


class CacheReport(dict):
    """``{written, exists, failed, files, samples, failures: [...]}``."""


def build_cache(manifest: pd.DataFrame, corpus_root: Path, cache_root: Path, *,
                sample_rate: int = 16_000,
                resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample_poly_to,
                skip_cells: Iterable[int] = (8,), workers: int = 32,
                progress: Callable[[int, int], None] | None = None) -> CacheReport:
    """Every manifest row not in ``skip_cells``, in parallel, resumable."""
    skip = set(int(c) for c in skip_cells)
    cell = pd.to_numeric(manifest["cell"], errors="coerce")
    rows = manifest[~cell.isin(skip)].to_dict(orient="records")
    counts = {"written": 0, "exists": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    samples = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(
                lambda r: build_one(r, corpus_root, cache_root, sample_rate=sample_rate,
                                    resampler=resampler), rows)):
            counts[res["status"]] += 1
            if res["status"] == "written":
                samples += res["samples"]
            elif res["status"] == "failed":
                failures.append(res)
            if progress is not None and (i + 1) % 1000 == 0:
                progress(i + 1, len(rows))
    return CacheReport(**counts, files=len(rows), samples=samples, failures=failures,
                       skipped_cells=sorted(skip), sample_rate=sample_rate)


def write_report(cache_root: Path, report: CacheReport) -> None:
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    (Path(cache_root) / "cache_report.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
