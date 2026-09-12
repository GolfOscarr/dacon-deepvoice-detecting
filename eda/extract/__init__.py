"""The extractor registry.

Critical: an extractor **declares the columns it emits**, and the declaration is
enforced on every call rather than in a test. This package exists because of a
defect class this repo keeps paying for -- 19 silently-ignored config fields, a
`dup_group` computed and hardcoded to `None`, a guard satisfied by a mere
textual mention. All three are the same shape: a value that was calculated and
never reached the thing that consumed it. Here, a column that is computed and
not declared cannot reach the table, and a column declared and not computed
raises.

Critical: **a signal extractor cannot see which plane it is running on.** Not
"must not read it" -- the bound callable takes exactly `(wav, sample_rate)`, and
registration refuses a function that could take a third argument. R1 (docs/EDA
README) is thereby a property of the type rather than of a code review. The
mechanism is `training.registries`', for the same reason.

Critical: **failure is a row, not an exception.** An extractor that cannot do
its job returns its full declared column set filled with `None` and sets its
`*_ok` flag false. A file that vanishes from a census makes the census wrong in
the direction that hides problems, and corruption is a finding (F-S2), not an
error.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["Extractor", "ExtractorError", "METADATA", "SIGNAL", "Registry",
           "register_metadata", "register_signal", "run_metadata"]


class ExtractorError(ValueError):
    """An extractor that does not honour the contract it declared."""


@dataclass(frozen=True)
class Extractor:
    name: str
    tier: str
    columns: tuple[str, ...]
    ok_column: str
    fn: Callable[..., Mapping[str, Any]]

    def __call__(self, *args: Any) -> dict[str, Any]:
        out = self.fn(*args)
        got, want = set(out), set(self.columns)
        if got != want:
            missing, extra = sorted(want - got), sorted(got - want)
            raise ExtractorError(
                f"extractor {self.name!r} declared {sorted(want)} but emitted "
                f"{sorted(got)} (missing={missing}, undeclared={extra}). Declare "
                f"every column you compute -- an undeclared one never reaches the "
                f"table, and that is how the dup_group leak happened")
        return dict(out)

    def failed(self, error: str) -> dict[str, Any]:
        """The all-`None` row this extractor emits when it cannot run."""
        row: dict[str, Any] = {c: None for c in self.columns}
        row[self.ok_column] = False
        err = f"{self.name}_error"
        if err in row:
            row[err] = error
        return row


class Registry(dict):
    """`name -> Extractor`, with the arity check at registration."""

    def __init__(self, tier: str, arity: int, param_names: tuple[str, ...]):
        super().__init__()
        self.tier, self.arity, self.param_names = tier, arity, param_names

    def add(self, name: str, columns: tuple[str, ...], ok_column: str,
            fn: Callable[..., Mapping[str, Any]]) -> Extractor:
        if name in self:
            raise ExtractorError(f"{name!r} is already registered in tier {self.tier}")
        if not columns:
            raise ExtractorError(f"{name!r} declares no columns")
        if ok_column not in columns:
            raise ExtractorError(
                f"{name!r} declares ok_column={ok_column!r}, which is not among its "
                f"columns {sorted(columns)}")
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
            raise ExtractorError(
                f"{name!r} takes *args/**kwargs. The contract is the arity: a "
                f"variadic extractor could be handed the plane it must not see")
        if len(params) != self.arity:
            raise ExtractorError(
                f"{name!r} takes {len(params)} argument(s); tier {self.tier} "
                f"extractors take exactly {self.arity}: {self.param_names}")
        ex = Extractor(name=name, tier=self.tier, columns=tuple(columns),
                       ok_column=ok_column, fn=fn)
        self[name] = ex
        return ex


#: `(path, probe_cfg) -> dict`. No decode.
METADATA = Registry("M", 2, ("path", "probe"))
#: `(wav, sample_rate) -> dict`. Runs once per plane; never told which.
SIGNAL = Registry("S", 2, ("wav", "sample_rate"))


def register_metadata(name: str, columns: tuple[str, ...], ok_column: str):
    def deco(fn):
        METADATA.add(name, columns, ok_column, fn)
        return fn
    return deco


def register_signal(name: str, columns: tuple[str, ...], ok_column: str):
    def deco(fn):
        SIGNAL.add(name, columns, ok_column, fn)
        return fn
    return deco


def run_metadata(path: Path, probe: Any, names: tuple[str, ...] | None = None
                 ) -> dict[str, Any]:
    """Every registered M-tier extractor over one file, failures included.

    Caveat: column collisions between extractors are an error rather than a
    last-write-wins merge. Two extractors that both emit `duration_s` is a
    disagreement, and silently keeping one of them is how two crawls end up
    disagreeing about what a column means.
    """
    chosen = METADATA.values() if names is None else [METADATA[n] for n in names]
    row: dict[str, Any] = {}
    for ex in chosen:
        try:
            out = ex(path, probe)
        except Exception as exc:                     # noqa: BLE001 -- R2: a row, not a raise
            out = ex.failed(f"{type(exc).__name__}: {exc}")
        clash = set(out) & set(row)
        if clash:
            raise ExtractorError(
                f"extractor {ex.name!r} emits column(s) {sorted(clash)} already "
                f"written by an earlier extractor")
        row.update(out)
    return row
