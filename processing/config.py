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
from training.config import RESAMPLERS, _cell_mix_from
from training.sampler import CellMix, check_mix, composed_fractions

__all__ = ["RESAMPLERS", "SECTIONS", "ConfigError", "DrawConfig", "ProcessingConfig",
           "RenderConfig", "dump_processing_config", "load_processing_config",
           "processing_config_from_dict"]


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
        for name in ("silence_lead_s", "silence_tail_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.sequential_prob <= 1.0:
            raise ValueError(f"sequential_prob must be in [0, 1], got {self.sequential_prob}")
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
SECTIONS: dict[str, type] = {"draw": DrawConfig, "render": RenderConfig}


def _draw_from_dict(d: Mapping[str, Any]) -> DrawConfig:
    d = dict(d)
    mix = d.pop("cell_mix", None)
    cfg = _build(DrawConfig, d, "draw")
    if mix is not None:
        cfg = dataclasses.replace(cfg, cell_mix=_cell_mix_from(mix, "draw.cell_mix"))
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


_BUILDERS = {"draw": _draw_from_dict, "render": _render_from_dict}


@dataclass(frozen=True)
class ProcessingConfig:
    draw: DrawConfig = dataclasses.field(default_factory=DrawConfig)
    render: RenderConfig = dataclasses.field(default_factory=RenderConfig)


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
            elif isinstance(value, tuple):
                section[key] = list(value)
            elif callable(value):
                section[key] = f"{value.__module__}.{value.__qualname__}"
        out[name] = section
    return out
