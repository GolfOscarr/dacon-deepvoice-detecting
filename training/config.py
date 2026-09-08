"""YAML for the four dataclasses that determine a training run.

`models.config` already loads the model and the objective from YAML. Everything
on the *data* side -- which corpus was drawn, how, split how, and trained with
what loop settings -- lived in Python defaults, reachable only by editing code.
Two consequences, and neither is ergonomic:

* **The experiment ledger stops being an identifier.** docs/validation/03
  specifies a ledger and `RunReport.as_ledger_row()` emits one, but if half a
  run's determinants are Python defaults then two rows can be identical while
  the runs trained on different corpora.
* **T1 is not reachable.** The project chose conditional `f8 = 0` as primary
  with strict `f8 = 1` as the fallback, explicitly so that the fallback would be
  "a one-value change, not a second code path". Until this module, that one
  value was only reachable from Python.

The rules are `models.config`'s, and deliberately not a second set of them:

* **`pyyaml` only.** The eval server preinstalls `pyyaml==6.0.1`;
  Hydra/OmegaConf would spend the offline install budget.
* **Unknown keys are an error.** A typo'd knob that silently does nothing is how
  an ablation ends up measuring the wrong thing.
* **Nested dataclasses are resolved through `get_type_hints`**, not through
  `fields(...).type` -- `from __future__ import annotations` makes the latter a
  string, so `is_dataclass(f.type)` silently never fires.

Critical: those rules are *imported* from `models.config` rather than restated.
`_build` is private there, and reaching across for it is the lesser evil: a copy
would let the two loaders drift, and the `_hints` subtlety above is exactly the
kind of thing that gets re-broken in a copy. The tidy version of this is to
promote `_build` to public API in `models.config`.

Caveat: **`RenderConfig.resampler` is a callable and cannot come from YAML.**
It is not silently defaulted either -- see `RESAMPLERS`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

# Critical: the private names are the point -- one implementation of "unknown
# keys are an error" and of the nested-hint resolution, not two.
from models.config import ConfigError, _build
from training.folds import FoldConfig
from training.loop import LoopConfig
from training.render import RenderConfig, resample_poly_to
from training.sampler import CellMix, SamplerConfig

__all__ = [
    "RESAMPLERS", "SECTIONS", "RunConfig", "ConfigError",
    "dump_run_config", "load_run_config", "run_config_from_dict",
]

#: The four dataclasses a run is made of, and the section each reads.
SECTIONS: dict[str, type] = {
    "sampler": SamplerConfig,
    "render": RenderConfig,
    "folds": FoldConfig,
    "loop": LoopConfig,
}

#: Resamplers a config may name, by import path.
#:
#: Critical: `RenderConfig.resampler` is a *callable*, so YAML cannot carry it.
#: The two wrong answers are both worse than this table. Ignoring the key would
#: be the typo'd-knob failure `NORMALIZE_KEYS` exists to prevent, and omitting
#: the field from the dump would leave the ledger unable to say which resampler
#: a run used -- and A-S1/G1 record swapping it as a live plan: if the
#: dummy-file forensics reveal the organizers' kernel, matching it is worth more
#: than kernel quality. So the round trip carries the *name*, the loader
#: resolves it here, and a name that is not here raises rather than falling back.
#: Adding G1's kernel is one line.
RESAMPLERS: dict[str, Callable[..., Any]] = {
    "training.render.resample_poly_to": resample_poly_to,
}


def _ref(fn: Callable[..., Any]) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


# --------------------------------------------------------------------------- #
# per-section construction


def _cell_mix_from(raw: Any, path: str) -> CellMix:
    """`{p: {1: .06, ...}}` -> `CellMix`, with the cell keys coerced to int.

    Caveat: the coercion is not politeness. `CellMix.__post_init__` compares
    `set(self.p)` against `set(CELL_TABLE)`, so a mix whose keys arrived as the
    strings `"1".."9"` -- which is what a quoted YAML key gives -- fails with
    "a cell mix needs all of 1-9, got ['1', ...]", which reads like a missing
    cell rather than a quoting mistake.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"p"}
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; a cell mix has one field, "
            f"`p` -- the nine cell probabilities")
    p = raw.get("p")
    if p is None:
        return CellMix()
    if not isinstance(p, dict):
        raise ConfigError(f"{path}.p: expected a mapping cell -> probability")
    out: dict[int, float] = {}
    for key, value in p.items():
        try:
            cell = int(key)
        except (TypeError, ValueError):
            raise ConfigError(f"{path}.p: cell keys must be 1-9, got {key!r}") from None
        out[cell] = float(value)
    return CellMix(out)


def _sampler_from_dict(d: Mapping[str, Any]) -> SamplerConfig:
    d = dict(d)
    mix = d.pop("cell_mix", None)
    cfg = _build(SamplerConfig, d, "sampler")
    if mix is not None:
        cfg = dataclasses.replace(cfg, cell_mix=_cell_mix_from(mix, "sampler.cell_mix"))
    return cfg


def _render_from_dict(d: Mapping[str, Any]) -> RenderConfig:
    d = dict(d)
    named = d.pop("resampler", None)
    cfg = _build(RenderConfig, d, "render")
    # `root` arrives as a string; the field is a Path and `Path("x") != "x"`.
    cfg = dataclasses.replace(cfg, root=Path(cfg.root))
    if named is not None:
        if named not in RESAMPLERS:
            raise ConfigError(
                f"render.resampler: {named!r} is not a resampler this build can "
                f"resolve; known: {sorted(RESAMPLERS)}. It is a callable, so YAML "
                f"carries its name and `training.config.RESAMPLERS` resolves it -- "
                f"add the kernel there rather than expecting an import path to "
                f"work (docs/data/06 A-S1).")
        cfg = dataclasses.replace(cfg, resampler=RESAMPLERS[named])
    return cfg


def _loop_from_dict(d: Mapping[str, Any]) -> LoopConfig:
    cfg = _build(LoopConfig, dict(d), "loop")
    return dataclasses.replace(cfg, out_dir=Path(cfg.out_dir))


def _folds_from_dict(d: Mapping[str, Any]) -> FoldConfig:
    return _build(FoldConfig, dict(d), "folds")


_BUILDERS = {"sampler": _sampler_from_dict, "render": _render_from_dict,
             "folds": _folds_from_dict, "loop": _loop_from_dict}


# --------------------------------------------------------------------------- #
# the run


@dataclass(frozen=True)
class RunConfig:
    """The four sections together, because a run is all four or it is nothing.

    Critical: one file, not four. The ledger's job is to make two rows
    distinguishable exactly when the runs differed, and a per-section file lets
    a row cite a sampler config while saying nothing about the split it drew
    from.
    """

    sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig)
    render: RenderConfig = dataclasses.field(default_factory=RenderConfig)
    folds: FoldConfig = dataclasses.field(default_factory=FoldConfig)
    loop: LoopConfig = dataclasses.field(default_factory=LoopConfig)


def run_config_from_dict(d: Mapping[str, Any]) -> RunConfig:
    """Build a `RunConfig`. Unknown *sections* are an error too."""
    d = dict(d or {})
    unknown = sorted(set(d) - set(SECTIONS))
    if unknown:
        raise ConfigError(
            f"run: unknown section(s) {unknown}; valid sections are "
            f"{sorted(SECTIONS)}")
    return RunConfig(**{name: _BUILDERS[name](d[name] or {})
                        for name in SECTIONS if name in d})


def load_run_config(path: str | Path) -> RunConfig:
    return run_config_from_dict(yaml.safe_load(Path(path).read_text()) or {})


def dump_run_config(cfg: RunConfig) -> dict:
    """A plain dict, suitable for `yaml.safe_dump` and for the ledger row.

    Critical: `yaml.safe_dump`-able, which `dataclasses.asdict` alone is not --
    it leaves `Path` objects and a bare function in the tree, and `safe_dump`
    refuses both. Every value here is a str, a number, a bool, `None`, a list or
    a dict, and `load_run_config(dump_run_config(cfg)) == cfg`.
    """
    out: dict[str, Any] = {}
    for name in SECTIONS:
        section = dataclasses.asdict(getattr(cfg, name))
        for key, value in list(section.items()):
            if isinstance(value, Path):
                section[key] = str(value)
            elif isinstance(value, tuple):
                section[key] = list(value)
            elif callable(value):
                section[key] = _ref(value)
        out[name] = section
    # `asdict` recurses, so `cell_mix` is already `{"p": {...}}`; the keys are
    # ints and stay ints, which is what `_cell_mix_from` reads back.
    return out
