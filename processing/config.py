"""The processing configuration -- docs/processing/03 §5, as dataclasses and as YAML.

Critical: every value the sampler reads is a field here, and every field is
read. A knob that validates, round-trips through a config file and is read
nowhere is how an ablation measures the wrong thing, so
``tests/test_processing_sampler.py`` keeps an exhaustive table asserting each
field moves the drawn stream.

The defaults are the spec's v1 values (D-1, D-3, D-5, D-6, D-16), not the
training pipeline's: this pipeline has no shipped stream to keep byte-identical,
and ``configs/processing_v1.yaml`` writes every field out so the ledger row is
the file, not the code. Values docs/processing/03 §8 leaves OPEN are marked.

The YAML rules are ``models.config``'s and are imported rather than restated:
unknown keys are an error, nested hints are resolved, and the cell mix's keys
are coerced to ``int`` (a quoted YAML key arrives as ``"1"``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from models.config import ConfigError, _build
from processing.render import RenderConfig
from processing.ship import PreprocessStep, ShipConfig
from training.config import RESAMPLERS, _cell_mix_from, _loop_from_dict
from training.folds import FoldConfig
from training.loop import LoopConfig
from training.registries import AUGMENT
from training.render import CODEC_CONTAINERS
from training.sampler import CellMix, check_mix, composed_fractions

__all__ = ["AUGMENTS_V1", "NORMALIZE_MENU_V1", "RESAMPLERS", "SECTIONS", "AugmentSpec",
           "ConfigError", "DrawConfig", "FoldConfig", "NormalizeMenu", "ProcessingConfig",
           "RenderConfig", "ShipConfig", "dump_processing_config", "load_processing_config",
           "processing_config_from_dict"]


# --------------------------------------------------------------------------- #
# DRAW-6 / DRAW-7 menus


@dataclass(frozen=True)
class AugmentSpec:
    """One entry of the augment menu: a registered name, its per-sample rate,
    and its parameter ranges.

    Critical: ``p`` is per SAMPLE, never per pool or label (R2), and the
    entry is drawn before the cell. A key ``k_range`` whose augment accepts a
    scalar ``k`` is drawn to that scalar at spec time, so I1b sees the value;
    a range the augment draws from itself (``rawboost_ssi``'s) is passed
    through and drawn inside the augment from ``spec.rng`` -- label-blind by
    the registry's signature.
    """

    name: str
    p: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in AUGMENT:
            raise ValueError(
                f"augment {self.name!r} is not registered; known: {AUGMENT.names()}. "
                f"Register it in training.registries first (docs/processing/03 §6 step 12)")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"augment {self.name!r}: p must be in [0, 1], got {self.p}")
        accepted = set(AUGMENT.params_of(self.name))
        for key, value in self.params.items():
            drawn = key[:-len("_range")] if key.endswith("_range") else None
            if key not in accepted and drawn not in accepted:
                raise ValueError(
                    f"augment {self.name!r} has no parameter {key!r}; it takes "
                    f"{sorted(accepted)}")
            if key.endswith("_range"):
                if not (isinstance(value, (tuple, list)) and len(value) == 2
                        and value[0] <= value[1]):
                    raise ValueError(
                        f"augment {self.name!r}: {key} must be [lo, hi], got {value!r}")

    @classmethod
    def from_flat(cls, raw: Mapping[str, Any], path: str) -> AugmentSpec:
        """``{name, p, <param>: ...}`` as the YAML writes it."""
        if not isinstance(raw, Mapping) or "name" not in raw or "p" not in raw:
            raise ConfigError(f"{path}: an augment entry needs `name` and `p`, got {raw!r}")
        params = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in raw.items() if k not in ("name", "p")}
        try:
            return cls(str(raw["name"]), float(raw["p"]), params)
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from None

    def to_flat(self) -> dict[str, Any]:
        return {"name": self.name, "p": self.p,
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()}}


#: D-8 and data/06 A-A5/A-A6/A-B3, at the spec's v1 rates, plus `pink_noise`
#: (step 12). Caveat: `rir` (A-A10, p 0.2) is in `configs/processing_v1.yaml`
#: and not here: its bank is a directory of measured responses on this
#: machine, and a code default may not name a path.
AUGMENTS_V1: tuple[AugmentSpec, ...] = (
    AugmentSpec("gain_jitter", 1.0, {"db_range": (-12.0, 12.0)}),
    AugmentSpec("rawboost_ssi", 0.5, {"snr_db_range": (10.0, 40.0),
                                      "tilt_db_range": (-12.0, 12.0)}),
    AugmentSpec("gaussian_noise", 0.3, {"snr_db_range": (10.0, 30.0)}),
    AugmentSpec("pink_noise", 0.3, {"snr_db_range": (10.0, 30.0)}),
    AugmentSpec("stereo_imbalance", 0.3, {"db_range": (-4.0, 4.0)}),
)


def _distribution(name: str, d: Mapping[str, float], allowed: set[str] | None) -> None:
    if not d:
        raise ValueError(f"normalize_menu.{name} must not be empty")
    if allowed is not None:
        bad = sorted(set(d) - allowed)
        if bad:
            raise ValueError(f"normalize_menu.{name}: unknown option(s) {bad}; "
                             f"allowed: {sorted(allowed)}")
    if any(v < 0 for v in d.values()):
        raise ValueError(f"normalize_menu.{name}: probabilities must be >= 0")
    total = sum(d.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"normalize_menu.{name} must sum to 1, got {total}")


@dataclass(frozen=True)
class NormalizeMenu:
    """D-13: what the organizers did to the test set, drawn per sample from
    ONE menu for every cell (docs/processing/03 DRAW-7).

    ``container`` options are ``wav``, ``flac`` and ``mp3_<kbps>``;
    ``telephone`` options are ``none``, ``ulaw``, ``alaw`` (8 kHz + companding)
    and ``plain`` (the 8 kHz leg alone). The draw's keys are a subset of
    ``training.render.NORMALIZE_KEYS``.
    """

    container: dict[str, float]
    channels: dict[str, float]
    telephone: dict[str, float]
    telephone_hz: int = 8000

    def __post_init__(self) -> None:
        _distribution("container", self.container, None)
        for key in self.container:
            base, _, rate = key.partition("_")
            if base not in CODEC_CONTAINERS or (base == "mp3") != bool(rate) \
                    or (rate and not rate.isdigit()):
                raise ValueError(
                    f"normalize_menu.container: {key!r} is not wav | flac | mp3_<kbps>")
        _distribution("channels", self.channels, {"mono", "stereo"})
        _distribution("telephone", self.telephone, {"none", "ulaw", "alaw", "plain"})
        if self.telephone_hz <= 0:
            raise ValueError(f"telephone_hz must be > 0, got {self.telephone_hz}")


#: D-13, the v1 menu (docs/processing/03 §5).
NORMALIZE_MENU_V1 = NormalizeMenu(
    container={"wav": 0.30, "flac": 0.15, "mp3_64": 0.15, "mp3_96": 0.15,
               "mp3_128": 0.15, "mp3_192": 0.10},
    channels={"mono": 0.5, "stereo": 0.5},
    telephone={"none": 0.80, "ulaw": 0.10, "alaw": 0.05, "plain": 0.05},
)


@dataclass(frozen=True)
class DrawConfig:
    """Everything the draw needs -- DRAW-1 to DRAW-4 of docs/processing/03 §3."""

    # -- DRAW-2: cell and composition policy ------------------------------- #
    cell_mix: CellMix = field(default_factory=CellMix)          # D-2, OPEN (sweep)
    #: D-1. 1.0 = strict: every mixed sample composed. Only ``f8 = 1`` balances
    #: composedness on the *component* heads, for any mix (02 §4.2); the
    #: whole-file rows then exist for the sweep ``{1.0, 0.5, 0.0}`` alone.
    f8: float = 1.0
    # `single_composed_rate`, `noise_composed_rate` and
    # `balance_marginal_composedness` were deleted on 2026-09-24 (06 D12):
    # the manifest has whole-file rows for cells 5 and 8 only, so every other
    # cell is composed whatever they said, and a sweep ledger recorded values
    # that did nothing (05 B15).
    #: D-16. DOSS per-domain cap as a sampling weight, ``min(count, cap) /
    #: count`` -- and applied to whole-file rows too, which ``training.sampler``
    #: draws uniformly.
    domain_cap: int = 500
    #: docs/training/07 D-d. ``None`` = off. Otherwise ``(lang, share)`` pairs:
    #: after the DOSS cap, each voice pool's weight is rescaled so language
    #: ``lang`` holds ``share`` of it (renormalised over the languages the pool
    #: has), on the real AND the fake side alike, so language alone does not
    #: predict fakeness. A language not listed counts as ``other``; share 0
    #: removes a language. Needs the manifest's ``lang`` column. YAML may give
    #: a mapping; it is stored as sorted pairs so the config stays hashable.
    lang_shares: tuple[tuple[str, float], ...] | None = None
    #: docs/training/10 F1. ``None`` = off. Otherwise ``(prefix, multiplier)``
    #: pairs: after the DOSS cap, a row whose DOSS domain (``domain_key``, else
    #: ``source_name``) starts with ``prefix`` has its weight multiplied (the
    #: longest matching prefix wins), before the language rescale. For a corpus
    #: whose many domains are one speaker (WaveFake: 8 vocoders of LJ). Stored as
    #: sorted pairs, like ``lang_shares``.
    domain_weights: tuple[tuple[str, float], ...] | None = None

    # -- DRAW-1: the timeline ----------------------------------------------- #
    duration_range: tuple[float, float] = (4.0, 60.0)

    # -- DRAW-3: the take/offset/tile rule ---------------------------------- #
    #: D-3 (revised, D-21). ONE take per sample, ``U(lo, hi)`` seconds, shared
    #: by every role and row kind, and never capped by a file: ``take_hi <=
    #: component_floor_s - 2 * edge_margin_s`` is asserted. Both halves are
    #: measured (docs/processing/03 D-21): a take capped by a short file means
    #: more tiles, and pool B is short (voice_fake I1b 0.65); a take drawn per
    #: role makes the larger of two join counts read as "two components"
    #: (voice_present 0.68). OPEN with the floor: {3.0 / U(1.5, 2), 4.0 /
    #: U(2, 3), 6.0 / U(3, 5)} keep 92 / 80 / 63 % of pool B's hours.
    take_range_s: tuple[float, float] = (1.5, 2.5)
    #: D-3. The take lies *strictly inside* the file: offset >= margin and
    #: offset + take <= file - margin. Took onset exposure from 20 % -> 0.8 %
    #: (voice) and 78 % -> 1.9 % (noise; CompSpoof clips are exactly 4.00 s).
    edge_margin_s: float = 0.5
    #: D-5 (revised by 06 D5). Rows shorter than this are dropped at
    #: construction (OFF-4's ``usable_duration``). Under bucket tiling the
    #: floor admits a row that can hold the SHORTEST take (``take_lo + 2 *
    #: margin``); a longer tile is served by the rows that can hold it. 2.5 s
    #: keeps 271 of pool B's 303 h (4.0 s kept 235).
    component_floor_s: float = 2.5

    # -- DRAW-4: placement -------------------------------------------------- #
    gain_db_range: tuple[float, float] = (-15.0, 15.0)
    gain_db_mean: float = -3.6
    gain_db_sigma: float = 4.0
    sequential_prob: float = 0.25
    #: The taper at every join -- the sequential joint *and* every tile joint
    #: (D-4). Drawn for every sample: a tiled overlap sample has joins too.
    crossfade_ms_range: tuple[float, float] = (10.0, 200.0)
    #: D-6. ``lead ~ U(0, silence_lead_s)``, ``tail ~ U(0, silence_tail_s)``,
    #: capped together at half the timeline, drawn for every sample. Lead is
    #: FIXED (0.595 -> 0.553); tail is OPEN -- unmeasured.
    silence_lead_s: float = 3.0
    silence_tail_s: float = 1.0

    # -- DRAW-5: the noise layer ------------------------------------------- #
    #: With this probability a pool-E row is added UNDER the composite (any
    #: cell, both branches) at ``snr_db ~ U(*noise_snr_db_range)``, as a
    #: ``ComponentDraw(role="noise", snr_db=...)`` tiled by DRAW-3. Drawn before
    #: the cell; the row itself after it, because a row flagged
    #: ``noise_has_speech`` may not go under a ``voice_present = 0`` cell.
    p_noise_layer: float = 0.5
    noise_snr_db_range: tuple[float, float] = (10.0, 30.0)

    # -- DRAW-6 / DRAW-7: the augment and normalize draws ------------------- #
    #: The augment menu, drawn per sample before the cell into
    #: ``spec.transforms`` (REN-3 applies it). ``()`` = no augmentation.
    augments: tuple[AugmentSpec, ...] = AUGMENTS_V1
    #: The test-chain menu, drawn per sample before the cell into
    #: ``spec.normalize`` (REN-4 applies it). ``None`` = as rendered.
    normalize_menu: NormalizeMenu | None = NORMALIZE_MENU_V1

    scheme_version: str = "strategy-v2"
    #: The audit's mutation-test hatch, as in ``training.sampler``. A run that
    #: sets it is not quotable.
    allow_unsound_mix: bool = False

    def __post_init__(self) -> None:
        lo, hi = self.duration_range
        if not 0 < lo < hi:
            raise ValueError(f"duration_range must be 0 < lo < hi, got {self.duration_range}")
        t_lo, t_hi = self.take_range_s
        if not 0 < t_lo <= t_hi:
            raise ValueError(f"take_range_s must be 0 < lo <= hi, got {self.take_range_s}")
        if self.edge_margin_s < 0:
            raise ValueError(f"edge_margin_s must be >= 0, got {self.edge_margin_s}")
        # Critical: under bucket tiling (docs/processing/06 D5) a file is used
        # only for tiles it can hold, so the floor's job is to admit a row that
        # can hold at least the shortest take: floor >= take_lo + 2 * margin.
        # No file ever caps the take (D-21); a file that cannot hold a tile is
        # simply not eligible for it.
        floor_min = t_lo + 2.0 * self.edge_margin_s
        if self.component_floor_s + 1e-9 < floor_min:
            raise ValueError(
                f"component_floor_s={self.component_floor_s} admits rows that cannot hold "
                f"the shortest take: need >= take_lo + 2 * edge_margin_s = {floor_min} (D-21 "
                f"under bucket tiling, docs/processing/06 D5)")
        if self.domain_cap < 1:
            raise ValueError(f"domain_cap must be >= 1, got {self.domain_cap}")
        if self.lang_shares is not None:
            pairs = (self.lang_shares.items() if isinstance(self.lang_shares, Mapping)
                     else self.lang_shares)
            pairs = tuple(sorted((str(k), float(v)) for k, v in pairs))
            if not pairs or any(v < 0 or v != v for _, v in pairs) \
                    or sum(v for _, v in pairs) <= 0:
                raise ValueError(f"lang_shares must be >= 0 with a positive sum, got {pairs}")
            object.__setattr__(self, "lang_shares", pairs)
        if self.domain_weights is not None:
            pairs = (self.domain_weights.items() if isinstance(self.domain_weights, Mapping)
                     else self.domain_weights)
            pairs = tuple(sorted((str(k), float(v)) for k, v in pairs))
            if not pairs or any(not k or not v >= 0 or not math.isfinite(v) for k, v in pairs):
                raise ValueError(f"domain_weights must map non-empty prefixes to finite "
                                 f"multipliers >= 0, got {pairs}")
            object.__setattr__(self, "domain_weights", pairs)
        for name in ("silence_lead_s", "silence_tail_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.sequential_prob <= 1.0:
            raise ValueError(f"sequential_prob must be in [0, 1], got {self.sequential_prob}")
        if not 0.0 <= self.p_noise_layer <= 1.0:
            raise ValueError(f"p_noise_layer must be in [0, 1], got {self.p_noise_layer}")
        s_lo, s_hi = self.noise_snr_db_range
        if s_lo > s_hi:
            raise ValueError(f"noise_snr_db_range must be lo <= hi, got {self.noise_snr_db_range}")
        names = [a.name for a in self.augments]
        if len(set(names)) != len(names):
            raise ValueError(f"augments lists a name twice: {names}")
        c_lo, c_hi = self.crossfade_ms_range
        if not 0 <= c_lo <= c_hi:
            raise ValueError(f"crossfade_ms_range must be 0 <= lo <= hi, "
                             f"got {self.crossfade_ms_range}")
        composed_fractions(self.cell_mix, self.f8)      # validates f8
        if not self.allow_unsound_mix:
            check_mix(self.cell_mix)
        g_lo, g_hi = self.gain_db_range
        if not g_lo <= g_hi:
            raise ValueError(f"gain_db_range must be lo <= hi, got {self.gain_db_range}")
        if self.gain_db_sigma < 0:
            raise ValueError(f"gain_db_sigma must be >= 0, got {self.gain_db_sigma}")
        for name in ("gain_db_range", "gain_db_mean", "gain_db_sigma", "noise_snr_db_range",
                     "take_range_s", "duration_range", "crossfade_ms_range"):
            values = getattr(self, name)
            values = values if isinstance(values, tuple) else (values,)
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")

    @property
    def f(self) -> dict[int, float]:
        """The per-cell composed fraction (``training.sampler.composed_fractions``).
        Only cells 5 and 8 have whole-file rows, so 1-4 and 9 are composed
        whatever this says for them."""
        return composed_fractions(self.cell_mix, self.f8)

    def for_eval(self) -> DrawConfig:
        """The evaluation draw (06 P4): no augments -- they are training's --
        and the normalize menu kept, because it models the test chain."""
        return dataclasses.replace(self, augments=())


# --------------------------------------------------------------------------- #
# YAML


#: The sections a processing config is made of. Grows as the stages land.
SECTIONS: dict[str, type] = {"draw": DrawConfig, "render": RenderConfig, "ship": ShipConfig,
                             "folds": FoldConfig, "loop": LoopConfig}


def _draw_from_dict(d: Mapping[str, Any]) -> DrawConfig:
    d = dict(d)
    mix = d.pop("cell_mix", None)
    augments = d.pop("augments", None)
    cfg = _build(DrawConfig, d, "draw")
    if mix is not None:
        cfg = dataclasses.replace(cfg, cell_mix=_cell_mix_from(mix, "draw.cell_mix"))
    if augments is not None:
        if not isinstance(augments, list):
            raise ConfigError("draw.augments: expected a list of {name, p, ...} entries")
        cfg = dataclasses.replace(cfg, augments=tuple(
            AugmentSpec.from_flat(a, f"draw.augments[{i}]") for i, a in enumerate(augments)))
    return cfg


def _render_from_dict(d: Mapping[str, Any]) -> RenderConfig:
    """``resampler`` is a callable: YAML carries its name and
    ``training.config.RESAMPLERS`` resolves it; an unknown name raises."""
    d = dict(d)
    named = d.pop("resampler", None)
    cfg = _build(RenderConfig, d, "render")
    cfg = dataclasses.replace(cfg, root=Path(cfg.root),
                              cache_root=None if cfg.cache_root is None else Path(cfg.cache_root))
    if named is not None:
        if named not in RESAMPLERS:
            raise ConfigError(
                f"render.resampler: {named!r} is not a resampler this build can "
                f"resolve; known: {sorted(RESAMPLERS)}")
        cfg = dataclasses.replace(cfg, resampler=RESAMPLERS[named])
    return cfg


def _ship_from_dict(d: Mapping[str, Any]) -> ShipConfig:
    d = dict(d)
    steps = d.pop("preprocess", None)
    cfg = _build(ShipConfig, d, "ship")
    if steps is not None:
        if not isinstance(steps, list):
            raise ConfigError("ship.preprocess: expected a list of step names or {name, ...}")
        try:
            cfg = dataclasses.replace(cfg, preprocess=tuple(
                PreprocessStep.from_flat(x, f"ship.preprocess[{i}]") for i, x in enumerate(steps)))
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
    return cfg


def _folds_from_dict(d: Mapping[str, Any]) -> FoldConfig:
    return _build(FoldConfig, dict(d), "folds")


_BUILDERS = {"draw": _draw_from_dict, "render": _render_from_dict, "ship": _ship_from_dict,
             "loop": _loop_from_dict,
             "folds": _folds_from_dict}


@dataclass(frozen=True)
class ProcessingConfig:
    draw: DrawConfig = dataclasses.field(default_factory=DrawConfig)
    render: RenderConfig = dataclasses.field(default_factory=RenderConfig)
    ship: ShipConfig = dataclasses.field(default_factory=ShipConfig)
    #: OFF-5. ``training.folds.FoldConfig``; D-22: four folds, the number of
    #: composable fake-music families minus PROBE's one.
    folds: FoldConfig = dataclasses.field(default_factory=lambda: FoldConfig(n_folds=4))
    #: 06 P8. The training loop's own knobs (``training.loop.LoopConfig``), so
    #: one file drives a run: draw, render, ship, folds and loop. The older
    #: ``configs/run_*.yaml`` (sampler / render / folds / loop) is the
    #: training sampler's and is not read by ``scripts/train.py`` any more.
    loop: LoopConfig = dataclasses.field(default_factory=LoopConfig)


def processing_config_from_dict(d: Mapping[str, Any]) -> ProcessingConfig:
    """Build a ``ProcessingConfig``. Unknown *sections* are an error too."""
    d = dict(d or {})
    unknown = sorted(set(d) - set(SECTIONS))
    if unknown:
        raise ConfigError(
            f"processing: unknown section(s) {unknown}; valid sections are "
            f"{sorted(SECTIONS)}")
    return ProcessingConfig(**{name: _BUILDERS[name](d[name] or {})
                               for name in SECTIONS if name in d})


def load_processing_config(path: str | Path) -> ProcessingConfig:
    return processing_config_from_dict(yaml.safe_load(Path(path).read_text()) or {})


def dump_processing_config(cfg: ProcessingConfig) -> dict:
    """A ``yaml.safe_dump``-able dict; ``load(dump(cfg)) == cfg``."""
    out: dict[str, Any] = {}
    for name in SECTIONS:
        section = dataclasses.asdict(getattr(cfg, name))
        for key, value in list(section.items()):
            if isinstance(value, Path):
                section[key] = str(value)
            elif key == "augments":
                section[key] = [a.to_flat() for a in getattr(cfg, name).augments]
            elif key in ("lang_shares", "domain_weights") and value is not None:
                section[key] = {k: v for k, v in value}
            elif key == "preprocess":
                section[key] = [s.to_flat() for s in getattr(cfg, name).preprocess]
            elif isinstance(value, tuple):
                section[key] = list(value)
            elif callable(value):
                section[key] = f"{value.__module__}.{value.__qualname__}"
        out[name] = section
    return out
