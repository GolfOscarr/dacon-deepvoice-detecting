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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from models.config import ConfigError, _build
from processing.render import RenderConfig
from processing.ship import PreprocessStep, ShipConfig
from training.config import RESAMPLERS, _cell_mix_from
from training.registries import AUGMENT
from training.render import CODEC_CONTAINERS
from training.sampler import CellMix, check_mix, composed_fractions

__all__ = ["AUGMENTS_V1", "NORMALIZE_MENU_V1", "RESAMPLERS", "SECTIONS", "AugmentSpec",
           "ConfigError", "DrawConfig", "NormalizeMenu", "ProcessingConfig", "RenderConfig",
           "ShipConfig", "dump_processing_config", "load_processing_config",
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


#: D-8 and data/06 A-A5/A-A6/A-B3, at the spec's v1 rates. Caveat: `pink_noise`
#: (docs/processing/03 DRAW-6, p 0.3) is absent until its registry entry lands
#: (§6 step 12) -- an unregistered name is refused at construction, not at
#: render.
AUGMENTS_V1: tuple[AugmentSpec, ...] = (
    AugmentSpec("gain_jitter", 1.0, {"db_range": (-12.0, 12.0)}),
    AugmentSpec("rawboost_ssi", 0.5, {"snr_db_range": (10.0, 40.0),
                                      "tilt_db_range": (-12.0, 12.0)}),
    AugmentSpec("gaussian_noise", 0.3, {"snr_db_range": (10.0, 30.0)}),
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
    single_composed_rate: float = 0.0
    noise_composed_rate: float = 0.0
    balance_marginal_composedness: bool = True
    #: D-16. DOSS per-domain cap as a sampling weight, ``min(count, cap) /
    #: count`` -- and applied to whole-file rows too, which ``training.sampler``
    #: draws uniformly.
    domain_cap: int = 500

    # -- DRAW-1: the timeline ----------------------------------------------- #
    duration_range: tuple[float, float] = (4.0, 60.0)

    # -- DRAW-3: the take/offset/tile rule ---------------------------------- #
    #: D-3. Each component *take* is ``U(lo, hi)`` seconds -- below pool D's
    #: 10 s so the offset range is never empty, and the audit is flat across
    #: 2-8 s (P5/P7/P8 within 0.03). Music-only draw AUC 0.993 -> 0.497.
    take_range_s: tuple[float, float] = (3.0, 8.0)
    #: D-3. The take lies *strictly inside* the file: offset >= margin and
    #: offset + take <= file - margin. Took onset exposure from 20 % -> 0.8 %
    #: (voice) and 78 % -> 1.9 % (noise; CompSpoof clips are exactly 4.00 s).
    edge_margin_s: float = 0.5
    #: D-5. Rows shorter than this are dropped at construction (OFF-4's
    #: ``usable_duration``). The 4 s floor cost 22.5 % of pool B's hours; 2 s
    #: recovers 63.5 h. Must exceed ``2 * edge_margin_s`` or a row at the floor
    #: has no usable interior.
    component_floor_s: float = 2.0
    #: D-21 (measured in step 3, not in the spec). The take is capped by the
    #: file's usable interior, so a short file yields shorter tiles and MORE
    #: joins -- and pool B is short: under the DOSS weights 39 % of fake-voice
    #: mass has < 3 s of interior against 15 % of real-voice mass, which made
    #: the tile count a voice-fake cue (I1b 0.65 in the mixed stratum on the
    #: S-tier manifest; 10.1 tiles per fake voice component vs 7.7 real).
    #: Fix: within each role, the file weights are re-balanced so both sides
    #: draw the same histogram of usable duration over these bin edges
    #: (``w *= target(bin) / side(bin)``, target = the two sides' mean; a bin
    #: one side lacks gets 0). Nothing is discarded. Bins only matter below
    #: ``take_hi``; above it the cap never binds. ``None`` switches it off.
    duration_match_edges_s: tuple[float, ...] | None = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

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

    # -- DRAW-6 / DRAW-7: the augment and normalize draws ------------------- #
    #: The augment menu, drawn per sample before the cell into
    #: ``spec.transforms`` (REN-3 applies it). ``()`` = no augmentation.
    augments: tuple[AugmentSpec, ...] = AUGMENTS_V1
    #: The test-chain menu, drawn per sample before the cell into
    #: ``spec.normalize`` (REN-4 applies it). ``None`` = as rendered.
    normalize_menu: NormalizeMenu | None = NORMALIZE_MENU_V1

    scheme_version: str = "strategy-v1"
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
        # Critical: the one invariant the tile rule rests on. A row at the floor
        # must still have a positive interior to crop from; at
        # `floor <= 2 * margin` the take collapses to zero and `ComponentDraw`
        # refuses it -- at draw time, long after the config was accepted.
        if self.component_floor_s <= 2.0 * self.edge_margin_s:
            raise ValueError(
                f"component_floor_s ({self.component_floor_s}) must exceed "
                f"2 * edge_margin_s ({2.0 * self.edge_margin_s}): a row at the "
                f"floor needs a positive interior to take from")
        if self.domain_cap < 1:
            raise ValueError(f"domain_cap must be >= 1, got {self.domain_cap}")
        edges = self.duration_match_edges_s
        if edges is not None and (len(edges) == 0 or min(edges) <= 0
                                  or list(edges) != sorted(set(edges))):
            raise ValueError(f"duration_match_edges_s must be strictly increasing "
                             f"positive edges or null, got {edges}")
        for name in ("silence_lead_s", "silence_tail_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.sequential_prob <= 1.0:
            raise ValueError(f"sequential_prob must be in [0, 1], got {self.sequential_prob}")
        names = [a.name for a in self.augments]
        if len(set(names)) != len(names):
            raise ValueError(f"augments lists a name twice: {names}")
        c_lo, c_hi = self.crossfade_ms_range
        if not 0 <= c_lo <= c_hi:
            raise ValueError(f"crossfade_ms_range must be 0 <= lo <= hi, "
                             f"got {self.crossfade_ms_range}")
        composed_fractions(                       # validates the knobs
            self.cell_mix, self.f8,
            a=self.single_composed_rate, b=self.single_composed_rate,
            f9=self.noise_composed_rate,
            balance_marginal=self.balance_marginal_composedness)
        if not self.allow_unsound_mix:
            check_mix(self.cell_mix)

    @property
    def f(self) -> dict[int, float]:
        """The per-cell composed fraction (``training.sampler.composed_fractions``)."""
        return composed_fractions(
            self.cell_mix, self.f8,
            a=self.single_composed_rate, b=self.single_composed_rate,
            f9=self.noise_composed_rate,
            balance_marginal=self.balance_marginal_composedness)


# --------------------------------------------------------------------------- #
# YAML


#: The sections a processing config is made of. Grows as the stages land.
SECTIONS: dict[str, type] = {"draw": DrawConfig, "render": RenderConfig, "ship": ShipConfig}


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
    cfg = dataclasses.replace(cfg, root=Path(cfg.root))
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


_BUILDERS = {"draw": _draw_from_dict, "render": _render_from_dict, "ship": _ship_from_dict}


@dataclass(frozen=True)
class ProcessingConfig:
    draw: DrawConfig = dataclasses.field(default_factory=DrawConfig)
    render: RenderConfig = dataclasses.field(default_factory=RenderConfig)
    ship: ShipConfig = dataclasses.field(default_factory=ShipConfig)


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
            elif key == "preprocess":
                section[key] = [s.to_flat() for s in getattr(cfg, name).preprocess]
            elif isinstance(value, tuple):
                section[key] = list(value)
            elif callable(value):
                section[key] = f"{value.__module__}.{value.__qualname__}"
        out[name] = section
    return out
