"""Turning a ``SampleSpec`` into audio -- REN-1 to REN-4 of docs/processing/03
§3. All the I/O of the processing pipeline lives here.

Critical: **``render(spec) == render(spec)``, bitwise** (I10). The only entropy
is ``spec.rng``, keyed on ``(sample_id, epoch, seed)``, and every stage is a
deterministic function of the spec, the files it names and that generator.

What differs from ``training.render``, and why (docs/processing/03 REN-2):

* **Sample-exact placement.** Every draw's start and end are rounded to the
  sample grid *from the spec*, and the decode is asked for exactly
  ``end - start`` samples -- so consecutive tiles neither overlap nor leave a
  one-sample gap, whatever the float arithmetic of ``start + i * tile`` did.
* **A taper at every joint, not only the sequential one.** A joint is any
  place one piece ends where another begins: the tile joints DRAW-3 creates
  inside a component, and the sequential joint between components. Each gets
  complementary sigmoid tapers over ``spec.crossfade_ms`` (A-A4: a hard cut is
  a splice shortcut). Overlap components share a start and get none.
* **Tiles merge into one frame interval.** A component's tiles are one span
  of one label; ``frame_intervals`` describe spans, not tiles.
* **A noise layer is scaled to its SNR here** (DRAW-5): the composite is
  built first, then each layer is scaled once so its RMS over its span sits
  ``snr_db`` below the composite's. Layers carry no frame target.

Decode (REN-1), the augment chain (REN-3) and the test-chain normalisation
(REN-4) are the training renderer's, imported not copied: they are the parts
docs/processing/03 marks EXISTS. The preprocess registry is *not* run here --
it is train/test symmetric and belongs to the shipped chain (SHIP-4/5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from models.config import AudioConfig
from models.losses import TARGET_FOR_COLUMN
from processing.cache import cache_path, read_slice
from training.registries import augment_chain
from training.render import (DecodeError, ManifestIndex, RenderedSample, _fade,
                             _normalize, _to_channels, load_audio, resample_poly_to)
from training.spec import SampleSpec

__all__ = ["DecodeError", "ManifestIndex", "RenderConfig", "RenderedSample",
           "frame_intervals_for", "placements_for", "render"]


@dataclass(frozen=True)
class RenderConfig:
    """Everything ``render`` needs beyond the spec and the manifest."""

    #: Manifest paths are relative to this.
    root: Path = Path(".")
    #: ``sample_rate`` and the I12 length regime are read here. Caveat:
    #: ``channels`` and ``band_hz`` are the *shipped chain's* (SHIP-3/5) and are
    #: not applied by the renderer -- a rendered sample is the analogue of a
    #: raw test file, ``(C, S)`` as decoded.
    audio: AudioConfig = field(default_factory=AudioConfig)
    #: One fixed resampler, injected so it can be swapped (A-S1 / G1).
    resampler: Callable[[np.ndarray, int, int], np.ndarray] = resample_poly_to
    #: ``sigmoid`` beats a hard cut, which becomes a splice shortcut (A-A4).
    crossfade_shape: str = "sigmoid"
    #: I12 -- the rendered duration must sit in the length regime.
    check_duration: bool = True
    #: OFF-6. Set: every component is read as a slice of its cached 16 kHz
    #: int16 decode (`processing.cache`), and a file missing from the cache is
    #: an error, not a fallback -- a run mixes no decode regimes. None: decode.
    cache_root: Path | None = None

    def __post_init__(self) -> None:
        if self.crossfade_shape not in ("sigmoid", "linear"):
            raise ValueError(
                f"crossfade_shape must be sigmoid|linear, got {self.crossfade_shape!r}")


# --------------------------------------------------------------------------- #
# Placement -- the spec on the sample grid


@dataclass(frozen=True)
class Placement:
    """One ``ComponentDraw`` on the sample grid: ``[start, end)`` samples."""

    index: int            # position in ``spec.components``
    start: int
    end: int

    @property
    def n(self) -> int:
        return self.end - self.start


def placements_for(spec: SampleSpec, sample_rate: int) -> tuple[Placement, ...]:
    """Every draw's ``[start, end)`` in samples, clipped to the timeline.

    Critical: ``end`` is rounded from ``target_start_s + duration_s`` and
    ``start`` from ``target_start_s`` -- both from the spec, never from a
    decoded length -- so two tiles the sampler drew as abutting land on
    exactly adjacent samples. Draws that round to nothing are dropped.
    """
    total = int(round(spec.duration_s * sample_rate))
    out = []
    for i, draw in enumerate(spec.components):
        start = int(round(draw.target_start_s * sample_rate))
        end = min(total, int(round((draw.target_start_s + draw.duration_s) * sample_rate)))
        if end - start > 0:
            out.append(Placement(i, start, end))
    return tuple(out)


def _joints(placements: Sequence[Placement]) -> tuple[set[int], set[int]]:
    """``(fade_in_at, fade_out_at)``: the placement indices whose start meets
    another's end, and whose end meets another's start."""
    starts = {p.start: p.index for p in placements}
    ends = {p.end: p.index for p in placements}
    fade_in = {p.index for p in placements if p.start in ends and ends[p.start] != p.index}
    fade_out = {p.index for p in placements if p.end in starts and starts[p.end] != p.index}
    return fade_in, fade_out


# --------------------------------------------------------------------------- #
# Composition -- REN-2


#: The composite's channel ceiling; a piece with more keeps its leading ones.
MAX_CHANNELS = 2


def _compose(spec: SampleSpec, placements: Sequence[Placement],
             pieces: Sequence[np.ndarray], cfg: RenderConfig, sample_rate: int) -> np.ndarray:
    """Place every decoded piece where the spec put it, tapering every joint.

    Caveat: the tapers are complementary at the joint rather than an
    overlapping crossfade, because the pieces are drawn back-to-back and the
    renderer does not overrule the draw (the spec *is* the ledger).
    """
    total = int(round(spec.duration_s * sample_rate))
    # Critical: the composite has at most MAX_CHANNELS. The test set is mono
    # and stereo; pool E holds 92 microphone-array recordings (8 and 30
    # channels, RIRS isotropic noise) and `_to_channels` can lift mono to N
    # or keep leading channels, never lift 2 to 8 -- so an uncapped canvas
    # met a stereo piece with a broadcast error (found by the shipped-residue
    # run over the built corpus, docs/processing/03 §7 item 3).
    channels = min(MAX_CHANNELS, max(p.shape[0] for p in pieces))
    canvas = np.zeros((channels, total), dtype=np.float32)
    xfade = int(round(spec.crossfade_ms / 1000.0 * sample_rate))
    fade_in, fade_out = _joints(placements)

    def taper(place: Placement, piece: np.ndarray) -> np.ndarray:
        n = place.n
        k = min(xfade, n // 2)
        if k > 1:
            ramp = _fade(k, cfg.crossfade_shape)
            if place.index in fade_in:
                piece[:, :k] *= ramp
            if place.index in fade_out:
                piece[:, n - k:] *= ramp[::-1]
        return piece

    # the composite first: every piece that is not a layer
    layers: list[tuple[Placement, np.ndarray]] = []
    for place, piece in zip(placements, pieces):
        draw = spec.components[place.index]
        piece = _to_channels(piece, channels)
        if draw.snr_db is not None:
            layers.append((place, piece))
            continue
        piece = taper(place, piece * np.float32(10.0 ** (draw.gain_db / 20.0)))
        canvas[:, place.start:place.end] += piece

    # DRAW-5: each layer -- all the tiles of one (file, snr) -- is scaled ONCE
    # so that its RMS over its span sits `snr_db` below the composite's RMS
    # over the same span. A silent composite (cell 9 with a silent file, a
    # span inside the lead) gives no reference and the layer is skipped.
    groups: dict[tuple[str, float], list[tuple[Placement, np.ndarray]]] = {}
    for place, piece in layers:
        d = spec.components[place.index]
        groups.setdefault((d.file_id, float(d.snr_db)), []).append((place, piece))
    for (_, snr_db), tiles in groups.items():
        ref = np.concatenate([canvas[:, p.start:p.end] for p, _ in tiles], axis=-1)
        noise = np.concatenate([x for _, x in tiles], axis=-1)
        rms_ref = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
        rms_noise = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
        if rms_ref <= 0.0 or rms_noise <= 0.0:
            continue
        scale = np.float32(rms_ref / rms_noise * 10.0 ** (-snr_db / 20.0))
        for place, piece in tiles:
            canvas[:, place.start:place.end] += taper(place, piece * scale)
    return canvas


# --------------------------------------------------------------------------- #
# Frame targets


def frame_intervals_for(spec: SampleSpec) -> dict[str, tuple[tuple[float, float, int], ...]]:
    """Where each component sits, in **absolute seconds**, and what it is.

    Keyed by branch (``voice`` / ``music`` / ``file``), values ``(start_s,
    end_s, label)``. Consecutive tiles of one component merge into one
    interval: the tiles are a rendering device, the span is the fact.

    A whole-file spec is one row carrying every component its cell says is
    present, placed under the same lead/tail and tiles as a composed one -- so
    its tiles describe every present role at once, labelled by the cell.
    (``training.render.frame_intervals_for`` returns empty tuples for it: the
    training sampler never places a whole-file row anywhere but 0.)
    """
    fake_for_role = {"voice": spec.voice_fake, "music": spec.music_fake, "noise": 0}
    out: dict[str, list[tuple[float, float, int]]] = {"voice": [], "music": [], "file": []}
    whole_roles = tuple(r for r, present in (("voice", spec.voice_present),
                                             ("music", spec.music_present)) if present)
    for draw in spec.components:
        if draw.snr_db is not None:
            continue                      # a layer is not a component the cell describes
        start = max(0.0, float(draw.target_start_s))
        end = min(float(spec.duration_s), start + float(draw.duration_s))
        if end <= start:
            continue
        roles = whole_roles if spec.render_mode == "whole_file" else (draw.role,)
        for role in roles or ("noise",):
            label = int(fake_for_role.get(role) or 0)
            _merge(out, (("file",) if role == "noise" else (role, "file")), start, end, label)
    return {k: tuple(v) for k, v in out.items()}


def _merge(out: dict[str, list[tuple[float, float, int]]], keys: tuple[str, ...],
           start: float, end: float, label: int) -> None:
    for key in keys:
        spans = out[key]
        if spans and spans[-1][2] == label and abs(spans[-1][1] - start) < 1e-6:
            spans[-1] = (spans[-1][0], end, label)
        else:
            spans.append((start, end, label))


# --------------------------------------------------------------------------- #
# render


def render(spec: SampleSpec, manifest: pd.DataFrame | ManifestIndex,
           cfg: RenderConfig | None = None) -> RenderedSample:
    """Decode, compose, augment, normalise. ``render(spec) == render(spec)``."""
    cfg = cfg or RenderConfig()
    index = ManifestIndex.coerce(manifest)
    sr = int(cfg.audio.sample_rate)

    placements = placements_for(spec, sr)
    pieces: list[np.ndarray] = []
    for place in placements:
        draw = spec.components[place.index]
        try:
            row = index[draw.file_id]
        except KeyError:
            raise DecodeError(
                f"spec {spec.sample_id} draws file_id {draw.file_id!r}, which is "
                f"not in the manifest") from None
        # Critical: exactly `place.n` samples, from the grid, not from
        # `draw.duration_s` -- see `placements_for`.
        if cfg.cache_root is not None:
            piece = read_slice(cache_path(cfg.cache_root, draw.file_id), sample_rate=sr,
                               offset_s=draw.source_offset_s, duration_s=place.n / sr,
                               file_id=draw.file_id)
        else:
            piece = load_audio(
                Path(cfg.root) / str(row["path"]), sample_rate=sr,
                offset_s=draw.source_offset_s, duration_s=place.n / sr,
                resampler=cfg.resampler, file_id=draw.file_id)
        if piece.shape[-1] != place.n:                          # pragma: no cover
            raise DecodeError(
                f"{draw.file_id}: decoded {piece.shape[-1]} samples, wanted {place.n}")
        pieces.append(piece)

    wav = _compose(spec, placements, pieces, cfg, sr)

    # REN-3 -- label-independent signal augmentation, `fn(wav, rng)` only.
    tensor = torch.from_numpy(wav)
    tensor = augment_chain(spec.transforms)(tensor, spec.rng)
    wav = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)

    # REN-4 -- the test chain, always last.
    wav = _normalize(wav, spec.normalize, sr, cfg.resampler)

    duration = wav.shape[-1] / sr
    if cfg.check_duration:
        lo, hi = cfg.audio.min_seconds, cfg.audio.max_seconds
        if not lo - 1e-6 <= duration <= hi + 1e-6:
            raise ValueError(
                f"rendered {duration:.3f}s, outside the [{lo}, {hi}]s length "
                f"regime AudioConfig and the competition agree on (I12)")

    labels = spec.labels
    targets = {key: int(labels[key] or 0) for key in TARGET_FOR_COLUMN.values()}
    return RenderedSample(
        wav=torch.from_numpy(wav), sample_rate=sr, targets=targets,
        frame_intervals=frame_intervals_for(spec), spec=spec)
