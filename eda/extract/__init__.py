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

__all__ = ["CONTENT", "Extractor", "ExtractorError", "METADATA", "SIGNAL",
           "VECTOR", "Registry", "VectorRegistry", "register_content",
           "register_metadata", "register_signal", "register_vector",
           "run_content", "run_metadata", "run_signal", "run_vectors"]


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


@dataclass(frozen=True)
class VectorExtractor(Extractor):
    """An `Extractor` whose non-flag columns are `width`-wide float arrays."""

    width: int = 0
    #: 🔴 Which of `columns` are arrays. Declared rather than inferred, because
    #: the only way to infer it is to look at a value -- and the one call that
    #: has no values is exactly the one that needs the answer: a decode failure.
    #: Inferring it from "not a flag" put a 128-wide NaN array in an integer
    #: column and broke the part writer three frames later.
    vectors: tuple[str, ...] = ()

    def failed(self, error: str) -> dict[str, Any]:
        """NaNs of the declared width for the vector columns, not `None`.

        🔴 The base class fills `None`, which parquet tolerates and `np.stack`
        does not: one `None` among 500 arrays makes the part an object array,
        and the merge fails on dtype rather than reporting the decode error
        that caused it.
        """
        import numpy as np                           # local: the base tier has no numpy

        # Built on the base's row so the `*_ok` / `*_error` convention has one
        # implementation; only the declared vector columns become arrays.
        row = super().failed(error)
        for column in self.vectors:
            row[column] = np.full(self.width, np.nan, dtype=np.float32)
        return row


class VectorRegistry(Registry):
    """`Registry` for extractors whose columns are fixed-width arrays.

    🔴 The width is declared, and it is not decoration. A vector extractor that
    fails must emit arrays of the **right shape**, because the parts are
    stacked at merge time: one ragged row turns a 91,765 x 128 matrix into an
    object array, and the failure surfaces as a dtype three modules away rather
    than as the decode error it is.
    """

    def add(self, name, columns, ok_column, fn, *, width: int = 0,
            vectors: tuple[str, ...] = ()):
        if width < 1:
            raise ExtractorError(
                f"{name!r} declares width={width}; a vector extractor must say "
                f"how wide its arrays are so a failure can emit NaNs of that "
                f"shape rather than a ragged row")
        unknown = sorted(set(vectors) - set(columns))
        if not vectors or unknown:
            raise ExtractorError(
                f"{name!r} declares vectors={sorted(vectors)}, which must be a "
                f"non-empty subset of its columns {sorted(columns)}"
                + (f"; {unknown} are not among them" if unknown else ""))
        super().add(name, columns, ok_column, fn)
        # Re-registered as the vector flavour: `Registry.add` has already run
        # every arity and declaration check, and this replaces the instance
        # with one that knows how wide a failure has to be.
        ex = VectorExtractor(name=name, tier=self.tier, columns=tuple(columns),
                             ok_column=ok_column, fn=fn, width=width,
                             vectors=tuple(vectors))
        self[name] = ex
        self.widths[name] = width
        return ex

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widths: dict[str, int] = {}


#: `(path, probe_cfg) -> dict`. No decode.
METADATA = Registry("M", 2, ("path", "probe"))
#: `(wav, sample_rate) -> dict`. Runs once per plane; never told which.
SIGNAL = Registry("S", 2, ("wav", "sample_rate"))
#: The same contract, emitting fixed-width arrays instead of scalars. Its own
#: registry rather than a flag on `SIGNAL` because the two are written to
#: different artifacts -- `signal.parquet` and `vectors.npz` -- and a single
#: registry would make "which file does this column go to?" a runtime question
#: about the value's type (docs/EDA/00 section 1).
VECTOR = VectorRegistry("V", 2, ("wav", "sample_rate"))
#: 🔴 The content tier: what a model *hears*, as opposed to what the signal
#: measures. Same `(wav, sample_rate)` contract, and run on the **chain plane
#: only** -- see `run_content`. A separate registry rather than more entries in
#: SIGNAL precisely because that one-plane rule is a property of the tier.
CONTENT = Registry("C", 2, ("wav", "sample_rate"))


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


def register_content(name: str, columns: tuple[str, ...], ok_column: str):
    def deco(fn):
        CONTENT.add(name, columns, ok_column, fn)
        return fn
    return deco


def register_vector(name: str, columns: tuple[str, ...], ok_column: str, *,
                    width: int, vectors: tuple[str, ...]):
    def deco(fn):
        VECTOR.add(name, columns, ok_column, fn, width=width, vectors=vectors)
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


def run_signal(planes: Mapping[str, Any], names: tuple[str, ...] | None = None
               ) -> dict[str, Any]:
    """Every registered S-tier extractor over every plane, suffixed by plane.

    `planes` is `eda.planes.load_planes`' output, `{"native": Plane, ...}`.
    A column `rms_dbfs` emitted by the `level` extractor lands as `rms_dbfs_native`
    and `rms_dbfs_chain`, which is what makes the difference columns of R1 a
    subtraction rather than a join.

    Critical: the extractor is handed `(plane.wav, plane.sample_rate)` and
    nothing else -- not the plane, not its name. `SIGNAL`'s arity check refuses
    a function that could accept a third argument, so an extractor that wanted
    to behave differently on the chain plane could not be registered, let alone
    run. R1 asks for the same statistic computed twice; an extractor that knew
    which time it was would quietly make it two statistics.

    Caveat: a failure on one plane does not fail the other. A file whose native
    plane is 8-channel and whose chain plane is mono can legitimately break one
    and not the other, and collapsing both to a single `*_ok` would hide which.
    """
    chosen = SIGNAL.values() if names is None else [SIGNAL[n] for n in names]
    row: dict[str, Any] = {}
    for plane_name, plane in planes.items():
        for ex in chosen:
            try:
                out = ex(plane.wav, plane.sample_rate)
            except Exception as exc:                 # noqa: BLE001 -- a row, not a raise
                out = ex.failed(f"{type(exc).__name__}: {exc}")
            suffixed = {f"{k}_{plane_name}": v for k, v in out.items()}
            clash = set(suffixed) & set(row)
            if clash:
                raise ExtractorError(
                    f"extractor {ex.name!r} emits column(s) {sorted(clash)} on "
                    f"plane {plane_name!r} already written by an earlier extractor")
            row.update(suffixed)
    return row


def run_vectors(planes: Mapping[str, Any], names: tuple[str, ...] | None = None
                ) -> dict[str, Any]:
    """Every registered V-tier extractor over every plane, suffixed by plane.

    The same shape as `run_signal` and deliberately so: `ltas` becomes
    `ltas_native` and `ltas_chain`, so R1's question stays a subtraction. The
    extractor is handed `(plane.wav, plane.sample_rate)` and nothing else, for
    the same reason and enforced by the same arity check.

    ⚠️ The `*_ok` and `*_error` columns come back too, and the caller writes
    them into `signal.parquet` rather than into `vectors.npz`. A row whose
    vectors are NaN and whose scalars are fine is a real state -- an 18-sample
    file has an LTAS and no time series -- and it has to be explicable from the
    table a reader actually opens.
    """
    chosen = VECTOR.values() if names is None else [VECTOR[n] for n in names]
    row: dict[str, Any] = {}
    for plane_name, plane in planes.items():
        for ex in chosen:
            try:
                out = ex(plane.wav, plane.sample_rate)
            except Exception as exc:                 # noqa: BLE001 -- a row, not a raise
                out = ex.failed(f"{type(exc).__name__}: {exc}")
            suffixed = {f"{k}_{plane_name}": v for k, v in out.items()}
            clash = set(suffixed) & set(row)
            if clash:
                raise ExtractorError(
                    f"extractor {ex.name!r} emits column(s) {sorted(clash)} on "
                    f"plane {plane_name!r} already written by an earlier extractor")
            row.update(suffixed)
    return row


def run_content(planes: Mapping[str, Any], names: tuple[str, ...] | None = None
                ) -> dict[str, Any]:
    """Every registered C-tier extractor, on the **chain plane only**.

    🔴 One plane, and unsuffixed columns. Every other tier runs twice and
    suffixes, because R1 asks what the 16 kHz chain removes and the answer is a
    subtraction. The content tier asks a different question -- *what is in this
    audio?* -- whose answer is a property of the recording, not of the transform,
    and whose models are trained at one rate. Silero VAD wants exactly 16 kHz;
    running it on a 44.1 kHz native plane would mean resampling inside an
    extractor, which is the render chain's job, or measuring a different thing
    per source.

    ⚠️ So `vad_speech_ratio` has no `_native` twin, and that asymmetry is
    deliberate. A reader who expects one should read this docstring, not file a
    bug.
    """
    from eda.planes import CHAIN

    if CHAIN not in planes:
        raise KeyError(
            f"the content tier runs on the {CHAIN!r} plane and it is not in "
            f"{sorted(planes)}. See this function's docstring for why it is "
            f"one plane rather than two")
    chosen = CONTENT.values() if names is None else [CONTENT[n] for n in names]
    plane = planes[CHAIN]
    row: dict[str, Any] = {}
    for ex in chosen:
        try:
            out = ex(plane.wav, plane.sample_rate)
        except Exception as exc:                     # noqa: BLE001 -- a row, not a raise
            out = ex.failed(f"{type(exc).__name__}: {exc}")
        clash = set(out) & set(row)
        if clash:
            raise ExtractorError(
                f"extractor {ex.name!r} emits column(s) {sorted(clash)} already "
                f"written by an earlier content extractor")
        row.update(out)
    return row
