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
from training.render import RenderConfig as _TrainingRenderConfig
from training.render import (DecodeError, ManifestIndex, RenderedSample, _fade,
                             _normalize, _to_channels, load_audio, resample_poly_to)
from training.spec import SampleSpec, spec_rng

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
    #: 06 D7. The composite (with its layer) is scaled once so its RMS over
    #: the placed span sits here, BEFORE the augments: a mixed sample is the
    #: sum of two components and was louder than a single one by
    #: construction, which the presence heads read at 0.67 / 0.78 after the
    #: shipped chain (04 §3). ``None`` leaves the level alone.
    target_rms_dbfs: float | None = -23.0
    #: 06 D6. Clip to [-1, 1] after the augments and before the test chain, as
    #: an int16 wav on disk would: a third of samples exceeded 1.0 and the
    #: containers disagreed about it -- wav and mp3 passed it through, flac
    #: and the telephone legs clipped -- so real music (louder composites)
    #: carried a cue no test file can (05 A4).
    clip: bool = True
    #: Bucket tiling stitches DIFFERENT files into one slot (06 D5), and files
    #: differ in level: a real-music slot built from several tracks carried
    #: a level variance a fake-music slot (one generator) did not, and the
    #: crest factor read music_fake at 0.78 after the shipped chain. Each
    #: non-layer tile is scaled to ``tile_rms_dbfs`` before placement, so a
    #: slot's dynamics are its tiles' own, not the corpus's level spread.
    #: ``None`` leaves the tiles as decoded.
    tile_rms_dbfs: float | None = -23.0
    #: 06 D11. The augment chain runs on one torch thread: float32 reductions
    #: differ across thread counts and I10 (bitwise across processes) held
    #: only at a fixed count (05 B14). Measured cost: see 04.
    single_thread: bool = True

    def __post_init__(self) -> None:
        if self.crossfade_shape not in ("sigmoid", "linear"):
            raise ValueError(
                f"crossfade_shape must be sigmoid|linear, got {self.crossfade_shape!r}")

    @classmethod
    def coerce(cls, cfg: object | None) -> RenderConfig:
        """This class, or a ``training.render.RenderConfig`` lifted into it with
        the processing defaults (06 P8: the loop, the dataset and the
        validator render through this module; their callers still build the
        older config)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, _TrainingRenderConfig):
            return cls(root=cfg.root, audio=cfg.audio, resampler=cfg.resampler,
                       crossfade_shape=cfg.crossfade_shape, check_duration=cfg.check_duration)
        raise TypeError(f"expected a RenderConfig, got {type(cfg).__name__}")


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


def _crossfades(spec: SampleSpec, placements: Sequence[Placement]
                ) -> list[tuple[int, int, int]]:
    """``(out_index, in_index, overlap_samples)`` for every joint: consecutive
    tiles of one slot, and -- sequential structure -- the last tile of a slot
    with the first of the next. Two tiles overlap by the crossfade the
    sampler drew (06 D8); the earlier fades out over the overlap while the
    later fades in, with equal-power ramps, so the joint is a crossfade over
    real audio rather than a dip to silence (05 A5). An overlap of 0 (the
    sampler's ``crossfade_ms = 0``) is a hard cut, deliberately."""
    by_slot: dict[tuple[int, str, float | None], list[Placement]] = {}
    for p in placements:
        c = spec.components[p.index]
        by_slot.setdefault((c.slot, c.role, c.snr_db), []).append(p)
    chains = [sorted(ps, key=lambda p: p.start) for _, ps in sorted(by_slot.items())]
    out: list[tuple[int, int, int]] = []
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            out.append((a.index, b.index, max(0, min(a.end - b.start, a.n, b.n))))
    if spec.structure == "sequential":
        own = [c for c in chains if spec.components[c[0].index].snr_db is None]
        own.sort(key=lambda c: c[0].start)
        for a_chain, b_chain in zip(own, own[1:]):
            a, b = a_chain[-1], b_chain[0]
            out.append((a.index, b.index, max(0, min(a.end - b.start, a.n, b.n))))
    return out


def _ramps(k: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power ``(fade_out, fade_in)`` over ``k`` samples: ``out**2 + in**2
    == 1`` everywhere, so a crossfade of two equally loud sources keeps the
    level instead of dipping 6 dB at the middle."""
    f = _fade(k, shape).astype(np.float64)
    return np.sqrt(1.0 - f).astype(np.float32), np.sqrt(f).astype(np.float32)


# --------------------------------------------------------------------------- #
# Composition -- REN-2


#: The composite's channel ceiling; a piece with more keeps its leading ones.
MAX_CHANNELS = 2
#: Below this RMS (-120 dBFS) a composite is silent: no layer reference, no
#: level target -- rather than a layer scaled to nothing and then a silent
#: canvas amplified to the target.
SILENT_RMS = 1e-6


def _compose(spec: SampleSpec, placements: Sequence[Placement],
             pieces: Sequence[np.ndarray], cfg: RenderConfig, sample_rate: int) -> np.ndarray:
    """Place every decoded piece where the spec put it, crossfading every
    joint, then the layers at their SNR, then the level (06 D7)."""
    total = int(round(spec.duration_s * sample_rate))
    # Critical: the composite has at most MAX_CHANNELS. The test set is mono
    # and stereo; pool E holds 92 microphone-array recordings (8 and 30
    # channels, RIRS isotropic noise) and `_to_channels` can lift mono to N
    # or keep leading channels, never lift 2 to 8 -- so an uncapped canvas
    # met a stereo piece with a broadcast error (found by the shipped-residue
    # run over the built corpus, docs/processing/03 §7 item 3).
    channels = min(MAX_CHANNELS, max(p.shape[0] for p in pieces))
    canvas = np.zeros((channels, total), dtype=np.float32)
    by_index = {p.index: p for p in placements}
    shaped = {p.index: _to_channels(pieces[i], channels).copy()
              for i, p in enumerate(placements)}
    if cfg.tile_rms_dbfs is not None:
        target = 10.0 ** (cfg.tile_rms_dbfs / 20.0)
        for p in placements:
            if spec.components[p.index].snr_db is not None:
                continue                              # a layer is scaled by its SNR
            rms = float(np.sqrt(np.mean(shaped[p.index].astype(np.float64) ** 2)))
            if rms > SILENT_RMS:
                shaped[p.index] *= np.float32(target / rms)
    for out_i, in_i, k in _crossfades(spec, placements):
        if k > 1:
            fade_out, fade_in = _ramps(k, cfg.crossfade_shape)
            shaped[out_i][:, by_index[out_i].n - k:] *= fade_out
            shaped[in_i][:, :k] *= fade_in

    # the composite first: every piece that is not a layer
    for place in placements:
        draw = spec.components[place.index]
        if draw.snr_db is not None:
            continue
        gain = np.float32(10.0 ** (draw.gain_db / 20.0))
        canvas[:, place.start:place.end] += shaped[place.index] * gain

    # DRAW-5: each layer -- all the tiles of one (slot, snr) -- is scaled ONCE
    # so that its RMS over its span sits `snr_db` below the composite's RMS
    # over the same span. Measured on the TAPERED tiles (05 B5: measured
    # untapered, the layer landed 0.2-0.8 dB quiet). A silent composite
    # (cell 9 with a silent file, a span inside the lead) gives no reference
    # and the layer is skipped.
    groups: dict[tuple[int, float], list[Placement]] = {}
    for place in placements:
        d = spec.components[place.index]
        if d.snr_db is not None:
            groups.setdefault((d.slot, float(d.snr_db)), []).append(place)
    for (_, snr_db), tiles in groups.items():
        # the layer as it will sound: its tapered tiles summed on their own
        # canvas (overlapping tiles crossfade, not double), measured over the
        # union of their spans against the composite over the same samples
        layer = np.zeros_like(canvas)
        mask = np.zeros(total, dtype=bool)
        for place in tiles:
            layer[:, place.start:place.end] += shaped[place.index]
            mask[place.start:place.end] = True
        rms_ref = float(np.sqrt(np.mean(canvas[:, mask].astype(np.float64) ** 2)))
        rms_noise = float(np.sqrt(np.mean(layer[:, mask].astype(np.float64) ** 2)))
        if rms_ref <= SILENT_RMS or rms_noise <= SILENT_RMS:
            continue
        canvas += layer * np.float32(rms_ref / rms_noise * 10.0 ** (-snr_db / 20.0))

    # 06 D7: one level for the whole composite, over the placed span
    if cfg.target_rms_dbfs is not None and placements:
        lo = min(p.start for p in placements)
        hi = max(p.end for p in placements)
        rms = float(np.sqrt(np.mean(canvas[:, lo:hi].astype(np.float64) ** 2)))
        if rms > SILENT_RMS:                # a silent composite stays silent
            canvas *= np.float32(10.0 ** (cfg.target_rms_dbfs / 20.0) / rms)
    return canvas


# --------------------------------------------------------------------------- #
# Frame targets


def frame_intervals_for(spec: SampleSpec) -> dict[str, tuple[tuple[float, float, int], ...]]:
    """Where each component sits, in **absolute seconds**, and what it is.

    Keyed by branch (``voice`` / ``music`` / ``file``), values ``(start_s,
    end_s, label)``. The tiles of one slot -- abutting or overlapping by the
    crossfade (06 D8) -- merge into one interval: the tiles are a rendering
    device, the span is the fact. A role branch is the merge of that role's
    slots; the file branch is the merge of every slot's span with its own
    label (two overlapping slots of different labels stay two entries, which
    is what the frame-level OR downstream needs).

    A whole-file spec is one row carrying every component its cell says is
    present, placed under the same lead/tail and tiles as a composed one -- so
    its tiles describe every present role at once, labelled by the cell.
    (``training.render.frame_intervals_for`` returns empty tuples for it: the
    training sampler never places a whole-file row anywhere but 0.)
    """
    fake_for_role = {"voice": spec.voice_fake, "music": spec.music_fake, "noise": 0}
    whole_roles = tuple(r for r, present in (("voice", spec.voice_present),
                                             ("music", spec.music_present)) if present)
    # one merged span list per (slot, role)
    slots: dict[tuple[int, str], list[tuple[float, float, int]]] = {}
    for draw in sorted(spec.components, key=lambda c: c.target_start_s):
        if draw.snr_db is not None:
            continue                      # a layer is not a component the cell describes
        start = max(0.0, float(draw.target_start_s))
        end = min(float(spec.duration_s), start + float(draw.duration_s))
        if end <= start:
            continue
        roles = whole_roles if spec.render_mode == "whole_file" else (draw.role,)
        for role in roles or ("noise",):
            label = int(fake_for_role.get(role) or 0)
            _merge({"s": slots.setdefault((draw.slot, role), [])}, ("s",), start, end, label)
    out: dict[str, list[tuple[float, float, int]]] = {"voice": [], "music": [], "file": []}
    for (_, role), spans in sorted(slots.items(), key=lambda kv: (kv[1][0][0], kv[0])):
        for start, end, label in spans:
            _merge(out, (("file",) if role == "noise" else (role, "file")), start, end, label)
    return {k: tuple(v) for k, v in out.items()}


def _merge(out: dict[str, list[tuple[float, float, int]]], keys: tuple[str, ...],
           start: float, end: float, label: int) -> None:
    """Consecutive same-label spans that abut or OVERLAP (a crossfade, 06 D8)
    become one; the placement of a slot is the fact, the tiles are how it
    was rendered."""
    for key in keys:
        spans = out[key]
        if spans and spans[-1][2] == label and start <= spans[-1][1] + 1e-6:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), label)
        else:
            spans.append((start, end, label))


# --------------------------------------------------------------------------- #
# render


def render(spec: SampleSpec, manifest: pd.DataFrame | ManifestIndex,
           cfg: RenderConfig | None = None) -> RenderedSample:
    """Decode, compose, augment, normalise. ``render(spec) == render(spec)``.

    Caveat: pass a ``ManifestIndex``; a DataFrame is coerced on EVERY call
    (6 s on the built manifest, 05 C7)."""
    cfg = RenderConfig.coerce(cfg)
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

    # REN-3 -- label-independent signal augmentation, `fn(wav, rng)` only, on
    # the RENDER stream (05 B6) and one torch thread (05 B14, 06 D11).
    tensor = torch.from_numpy(wav)
    rng = spec_rng(spec.sample_id, spec.epoch, spec.seed, "render")
    threads = torch.get_num_threads()
    if cfg.single_thread:
        torch.set_num_threads(1)
    try:
        tensor = augment_chain(spec.transforms)(tensor, rng)
    finally:
        if cfg.single_thread:
            torch.set_num_threads(threads)
    wav = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    # 06 D6 -- what an int16 file on disk does to an over-range sample
    if cfg.clip:
        wav = np.clip(wav, -1.0, 1.0)

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
