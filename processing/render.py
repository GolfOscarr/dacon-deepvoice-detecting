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


def _compose(spec: SampleSpec, placements: Sequence[Placement],
             pieces: Sequence[np.ndarray], cfg: RenderConfig, sample_rate: int) -> np.ndarray:
    """Place every decoded piece where the spec put it, tapering every joint.

    Caveat: the tapers are complementary at the joint rather than an
    overlapping crossfade, because the pieces are drawn back-to-back and the
    renderer does not overrule the draw (the spec *is* the ledger).
    """
    total = int(round(spec.duration_s * sample_rate))
    channels = max(p.shape[0] for p in pieces)
    canvas = np.zeros((channels, total), dtype=np.float32)
    xfade = int(round(spec.crossfade_ms / 1000.0 * sample_rate))
    fade_in, fade_out = _joints(placements)

    for place, piece in zip(placements, pieces):
        draw = spec.components[place.index]
        piece = _to_channels(piece, channels) * np.float32(10.0 ** (draw.gain_db / 20.0))
        n = place.n
        k = min(xfade, n // 2)
        if k > 1:
            ramp = _fade(k, cfg.crossfade_shape)
            if place.index in fade_in:
                piece[:, :k] *= ramp
            if place.index in fade_out:
                piece[:, n - k:] *= ramp[::-1]
        canvas[:, place.start:place.end] += piece
    return canvas


# --------------------------------------------------------------------------- #
# Frame targets


def frame_intervals_for(spec: SampleSpec) -> dict[str, tuple[tuple[float, float, int], ...]]:
    """Where each component sits, in **absolute seconds**, and what it is.

    Keyed by branch (``voice`` / ``music`` / ``file``), values ``(start_s,
    end_s, label)``. Consecutive tiles of one component merge into one
    interval: the tiles are a rendering device, the span is the fact.
    Whole-file specs carry empty tuples (interim -- step 3).
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
        keys = ("file",) if draw.role == "noise" else (draw.role, "file")
        for key in keys:
            spans = out[key]
            if spans and spans[-1][2] == label and abs(spans[-1][1] - start) < 1e-6:
                spans[-1] = (spans[-1][0], end, label)
            else:
                spans.append((start, end, label))
    return {k: tuple(v) for k, v in out.items()}


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
