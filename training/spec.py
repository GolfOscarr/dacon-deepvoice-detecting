"""What a training sample *is*, as data, before any audio is touched.

Critical: the whole pipeline rests on one split -- deciding what a sample is
(`SampleSpec`) is separate from rendering it. A spec is a small serialisable
value, so the expensive guarantees become cheap -- the shortcut audit is a
frequency table over specs rather than a corpus decode, the evaluation set is
frozen as *specs* rather than a seed, and byte-reproducibility is
``render(spec) == render(spec)`` (docs/pipelines/01 §1).

Critical: the labels are a pure function of ``cell`` and nothing else.
`transforms` and `normalize` cannot reach them. That is ``P(T | L) = P(T)``
made structural rather than conventional.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from metrics.dacon import file_fake_label

__all__ = [
    "CELL_TABLE", "STRATA", "ComponentDraw", "SampleSpec",
    "cell_labels", "cells_in_stratum", "is_fake_cell", "spec_rng", "stratum_of",
]

#: The 8+1 cells (docs/data/02), as (voice_present, music_present, voice_fake,
#: music_fake). ``None`` marks a component that is absent, whose fake label the
#: metric ignores -- that is what makes cells 1-4 well defined.
CELL_TABLE: dict[int, tuple[int, int, int | None, int | None]] = {
    1: (1, 0, 0, None),      # voice-only REAL
    2: (1, 0, 1, None),      # voice-only FAKE
    3: (0, 1, None, 0),      # instrumental-only REAL
    4: (0, 1, None, 1),      # instrumental-only FAKE
    5: (1, 1, 0, 0),         # mixed R/R
    6: (1, 1, 0, 1),         # mixed R/F   <- cannot be scraped
    7: (1, 1, 1, 0),         # mixed F/R   <- cannot be scraped
    8: (1, 1, 1, 1),         # mixed F/F
    9: (0, 0, None, None),   # neither
}

#: Presence stratum per cell. Critical: the composedness balance is enforced
#: *within* each stratum, not marginally: a marginal balance can hold while
#: "composed" predicts the label inside a stratum (docs/pipelines/02 §3).
STRATA: dict[int, str] = {
    c: ("mixed" if v and m else "voice-only" if v else "music-only" if m else "neither")
    for c, (v, m, _, _) in CELL_TABLE.items()
}


def cell_labels(cell: int) -> dict[str, int | None]:
    """The five ground-truth labels implied by a cell.

    ``file_fake`` is computed by ``metrics.dacon.file_fake_label`` rather than
    written down, because the competition defines it as OR over *present*
    components and there is exactly one implementation of that in this repo.

    Caveat: absent components are passed as ``0``, not ``None``.
    ``file_fake_label`` survives ``None`` only because
    ``np.asarray(None).astype(bool)`` is ``True`` and the presence mask happens
    to zero it -- incidental, not a contract.
    """
    if cell not in CELL_TABLE:
        raise ValueError(f"cell must be 1-9, got {cell!r}")
    vp, mp, vf, mf = CELL_TABLE[cell]
    return {
        "voice_present": vp,
        "music_present": mp,
        "voice_fake": vf,
        "music_fake": mf,
        "file_fake": int(file_fake_label(vp, mp, vf or 0, mf or 0)),
    }


def is_fake_cell(cell: int) -> bool:
    return bool(cell_labels(cell)["file_fake"])


def stratum_of(cell: int) -> str:
    if cell not in STRATA:
        raise ValueError(f"cell must be 1-9, got {cell!r}")
    return STRATA[cell]


def cells_in_stratum(stratum: str) -> tuple[int, ...]:
    return tuple(c for c, s in STRATA.items() if s == stratum)


def spec_rng(sample_id: int, epoch: int, seed: int) -> np.random.Generator:
    """The one RNG a sample is allowed to use.

    Critical: keyed on ``(sample_id, epoch, seed)`` and hashed with blake2b,
    **not** Python's ``hash()``: string hashing is salted per process, so a
    ``hash()``-keyed stream would be reproducible within a run and different
    across runs -- the exact opposite of what A-S2 asks for.
    """
    digest = hashlib.blake2b(
        f"{seed}:{epoch}:{sample_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class ComponentDraw:
    """One component placed on the sample timeline."""

    file_id: str
    role: str                      # "voice" | "music" | "noise"
    source_offset_s: float         # where we start reading inside the source
    duration_s: float
    target_start_s: float          # where it lands on the sample timeline
    gain_db: float
    is_mixup_partner: bool = False

    def __post_init__(self) -> None:
        if self.role not in ("voice", "music", "noise"):
            raise ValueError(f"role must be voice|music|noise, got {self.role!r}")
        for name in ("source_offset_s", "duration_s", "target_start_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")


@dataclass(frozen=True)
class SampleSpec:
    """The complete, serialisable decision to build one training sample.

    Every field is drawn before any file is opened. Caveat: ``duration_s`` is
    drawn **first** and components are placed on that timeline -- an earlier
    design cropped last, which renders audio only to discard it (expensive
    given the codec round-trips) and makes frame targets a post-hoc re-slice
    instead of exact by construction.
    """

    sample_id: int
    epoch: int
    seed: int
    scheme_version: str

    duration_s: float
    cell: int
    render_mode: str               # "composed" | "whole_file"
    structure: str                 # "overlap" | "sequential"
    components: tuple[ComponentDraw, ...]
    crossfade_ms: float = 0.0

    transforms: tuple[tuple[str, dict[str, Any]], ...] = ()
    normalize: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cell not in CELL_TABLE:
            raise ValueError(f"cell must be 1-9, got {self.cell!r}")
        if self.render_mode not in ("composed", "whole_file"):
            raise ValueError(f"render_mode must be composed|whole_file, "
                             f"got {self.render_mode!r}")
        if self.structure not in ("overlap", "sequential"):
            raise ValueError(f"structure must be overlap|sequential, "
                             f"got {self.structure!r}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        if not self.components:
            raise ValueError("a spec needs at least one component draw")
        # Critical: a whole_file spec is ONE row used as-is -- but it may be
        # placed as several tiles of that row (docs/processing/03 DRAW-3 applies
        # the take/offset/tile rule to every row kind). What is invariant is
        # that every component names the same file and the same role; a second
        # file would be a composition wearing the whole-file label.
        if self.render_mode == "whole_file":
            files = {c.file_id for c in self.components}
            roles = {c.role for c in self.components}
            if len(files) != 1 or len(roles) != 1:
                raise ValueError(
                    f"a whole_file spec is exactly one row used as-is: every "
                    f"component must name the same file and role, got files "
                    f"{sorted(files)} and roles {sorted(roles)}")
        # Cells 6 and 7 hold one real and one fake component, so they cannot be
        # scraped -- that is the whole reason two fake heads exist.
        if self.cell in (6, 7) and self.render_mode != "composed":
            raise ValueError(f"cell {self.cell} can only be composed")
        end = max(c.target_start_s + c.duration_s for c in self.components)
        if end > self.duration_s + 1e-6:
            raise ValueError(
                f"component ends at {end:.4f}s, past the {self.duration_s:.4f}s timeline")

    # -- labels: derived, never stored independently ----------------------- #

    @property
    def labels(self) -> dict[str, int | None]:
        return cell_labels(self.cell)

    @property
    def voice_present(self) -> int:
        return CELL_TABLE[self.cell][0]

    @property
    def music_present(self) -> int:
        return CELL_TABLE[self.cell][1]

    @property
    def voice_fake(self) -> int | None:
        return CELL_TABLE[self.cell][2]

    @property
    def music_fake(self) -> int | None:
        return CELL_TABLE[self.cell][3]

    @property
    def file_fake(self) -> int:
        return int(cell_labels(self.cell)["file_fake"])

    @property
    def stratum(self) -> str:
        return STRATA[self.cell]

    @property
    def rng(self) -> np.random.Generator:
        return spec_rng(self.sample_id, self.epoch, self.seed)

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["components"] = [asdict(c) for c in self.components]
        d["transforms"] = [[name, dict(params)] for name, params in self.transforms]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SampleSpec:
        d = dict(d)
        d["components"] = tuple(ComponentDraw(**c) for c in d["components"])
        d["transforms"] = tuple((n, dict(p)) for n, p in d.get("transforms", ()))
        d["normalize"] = dict(d.get("normalize", {}))
        return cls(**d)
