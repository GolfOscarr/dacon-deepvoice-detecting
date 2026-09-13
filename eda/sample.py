"""The S tier's stratified draw, and the record of it.

🔴 **The sample is part of the corpus definition, not a convenience**
(docs/EDA/00 section 3). 291 GiB decoded is a multi-day job, so the signal tier
runs on `min(per_stratum, N)` files per stratum -- and which files those are
decides every S-tier number anyone ever quotes. The MLAAD cap is the precedent
(docs/data/12): a selection recorded with its seed is reproducible from
originals plus code, which is what the 2nd-stage submission rests on.

So the draw is written to `eda/out/<partition>/sample.json` **before** anything
decodes, and `draw` refuses to silently replace one that exists. Redrawing is a
deliberate act with a flag, not something that happens because a shard was
re-run on a machine with a different pandas version.

⚠️ Rows are sorted by `file_id` before the RNG is touched. `files.parquet`'s row
order is an artifact of shard scheduling -- thread completion order, and which
parts happened to exist -- so a draw taken in table order is reproducible only
until someone re-probes one source. Sorting first makes the seed mean what it
says.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eda.config import EdaConfig

__all__ = ["SAMPLE_FILE", "Sample", "SampleExists", "draw", "load_sample",
           "sampled_rows"]

SAMPLE_FILE = "sample.json"


class SampleExists(RuntimeError):
    """A draw is already recorded for this partition.

    Its own type because the caller has a real choice to make -- reuse it, which
    is almost always right, or redraw and accept that every S-tier number
    published from the old one is now unreproducible.
    """


@dataclass(frozen=True)
class Sample:
    """One partition's draw, exactly as `sample.json` records it."""

    partition: str
    seed: int
    per_stratum: int
    stratify_by: tuple[str, ...]
    #: How the budget was divided inside a stratum. Recorded because it changes
    #: which files are measured, which is the whole reason this file exists.
    spread_by: tuple[str, ...]
    full: bool
    n_population: int
    file_ids: tuple[str, ...]

    @property
    def n_drawn(self) -> int:
        return len(self.file_ids)

    @property
    def fingerprint(self) -> str:
        """sha256 over the drawn ids, in draw order. Twelve hex digits of it.

        🔴 What makes a signal part *belong* to a draw. Parts are indexed by
        position in the draw, so a redraw that changes the population silently
        re-points every index: admit 92 files to pool E, redraw, and shards
        0..28 still exist and are still marked DONE, so the pass skips all of
        them and the 92 new files are never measured. The table looks complete
        and is 92 rows short. Measured on exactly that change.
        """
        h = hashlib.sha256()
        for fid in self.file_ids:
            h.update(fid.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:12]

    def to_json(self) -> dict:
        return {"partition": self.partition, "seed": self.seed,
                "fingerprint": self.fingerprint,
                "per_stratum": self.per_stratum,
                "stratify_by": list(self.stratify_by),
                "spread_by": list(self.spread_by), "full": self.full,
                "n_population": self.n_population, "n_drawn": self.n_drawn,
                "file_ids": list(self.file_ids)}

    @classmethod
    def from_json(cls, d: dict) -> "Sample":
        return cls(partition=d["partition"], seed=int(d["seed"]),
                   per_stratum=int(d["per_stratum"]),
                   stratify_by=tuple(d["stratify_by"]),
                   spread_by=tuple(d.get("spread_by", ())), full=bool(d["full"]),
                   n_population=int(d["n_population"]),
                   file_ids=tuple(d["file_ids"]))


def _is_full(cfg: EdaConfig, partition: str) -> bool:
    """Is this partition measured in full rather than sampled?

    Pool D because five generator families are too few to sample from and it
    carries 0.27 of the metric; pool E because it is small and its last surprise
    cost a corpus rebuild (docs/EDA/00 section 5).
    """
    return partition in cfg.sample.full_pools


def _take(ids: np.ndarray, n: int, rng: np.random.Generator) -> list[str]:
    """`n` ids without replacement, left in `files.parquet`'s own order."""
    if len(ids) <= n:
        return ids.tolist()
    idx = rng.choice(len(ids), size=n, replace=False)
    return ids[np.sort(idx)].tolist()


def _spread(cfg: EdaConfig, rows: pd.DataFrame,
            rng: np.random.Generator) -> list[str]:
    """One stratum's `per_stratum` files, divided evenly over `spread_by`.

    🔴 `stratify_by` sets the budget; this decides how it is spent inside one.
    With `spread_by` empty the stratum is one uniform draw, which is what every
    S-tier number before docs/EDA/09 was taken from.

    The allocation is water-filling: each group gets
    `ceil(remaining_budget / remaining_groups)` capped by what it holds, walked
    **smallest group first** so that a group too small for its share hands the
    rest to groups that can still absorb it.

    ⚠️ Smallest-first is not cosmetic. Walking in key order instead spends the
    remainder only forwards, and it runs out: measured on `fma`, 2,000 over 156
    shard directories drew **1,953** -- 47 files quietly unmeasured because the
    groups that could have taken them had already been passed.

    ⚠️ Counts are decided before the RNG is touched, and the files are then
    drawn in sorted **key** order. The allocation therefore cannot depend on
    frame order, and the draw is reproducible for the same reason the strata are
    sorted.
    """
    budget = cfg.sample.per_stratum
    if not cfg.sample.spread_by:
        return _take(rows["file_id"].to_numpy(), budget, rng)

    missing = [c for c in cfg.sample.spread_by if c not in rows.columns]
    if missing:
        raise KeyError(
            f"cannot spread the draw over {missing}: not in files.parquet. Its "
            f"columns are {sorted(rows.columns)}. `group_key` is written by "
            f"`consolidate`; re-run it (no re-probe, nothing decodes)")

    groups = {key: g["file_id"].to_numpy()
              for key, g in rows.groupby(list(cfg.sample.spread_by), sort=True)}
    quota: dict[object, int] = {}
    left, remaining = budget, len(groups)
    # Smallest first, then by key so equal sizes break deterministically.
    for key in sorted(groups, key=lambda k: (len(groups[k]), str(k))):
        share = -(-left // remaining)               # ceil, integer-only
        quota[key] = min(len(groups[key]), share)
        left -= quota[key]
        remaining -= 1

    picked: list[str] = []
    for key, ids in groups.items():                 # sorted key order: see above
        picked.extend(_take(ids, quota[key], rng))
    return picked


def draw(cfg: EdaConfig, files: pd.DataFrame, partition: str, *,
         force: bool = False) -> Sample:
    """Draw -- or re-read -- this partition's S-tier sample.

    Raises `SampleExists` when a draw is already recorded and `force` is not
    set. Read it with `load_sample` instead; that is the normal path, and it is
    what makes a resumed signal pass measure the same files as the first one.
    """
    path = cfg.out / partition / SAMPLE_FILE
    if path.exists() and not force:
        raise SampleExists(
            f"{path} already records a draw of {load_sample(cfg, partition).n_drawn} "
            f"file(s). Re-read it rather than redrawing: every S-tier number "
            f"published from the old draw becomes unreproducible. Pass force=True "
            f"if that is what you mean")

    missing = [c for c in cfg.sample.stratify_by if c not in files.columns]
    if missing:
        raise KeyError(
            f"cannot stratify partition {partition!r} by {missing}: not in "
            f"files.parquet. Its columns are {sorted(files.columns)}")
    if files.empty:
        raise ValueError(
            f"partition {partition!r}: no rows to draw from. Probe and "
            f"consolidate it first")

    # ⚠️ Before the RNG, not after. See the module docstring.
    ordered = files.sort_values("file_id", kind="mergesort")
    full = _is_full(cfg, partition)
    if full:
        chosen = ordered["file_id"].tolist()
    else:
        rng = np.random.default_rng(cfg.sample.seed)
        picked: list[str] = []
        # Strata in sorted key order so the RNG is consumed in the same
        # sequence on every machine -- `groupby(sort=True)` is the default, and
        # it is load-bearing here rather than cosmetic.
        for _, rows in ordered.groupby(list(cfg.sample.stratify_by), sort=True):
            picked.extend(_spread(cfg, rows, rng))
        chosen = picked

    sample = Sample(partition=partition, seed=cfg.sample.seed,
                    per_stratum=cfg.sample.per_stratum,
                    stratify_by=tuple(cfg.sample.stratify_by),
                    spread_by=tuple(cfg.sample.spread_by), full=full,
                    n_population=len(files), file_ids=tuple(chosen))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample.to_json(), indent=2), encoding="utf-8")
    return sample


def load_sample(cfg: EdaConfig, partition: str) -> Sample | None:
    """The recorded draw, or None when the partition has never been drawn."""
    path = cfg.out / partition / SAMPLE_FILE
    if not path.exists():
        return None
    return Sample.from_json(json.loads(path.read_text(encoding="utf-8")))


def sampled_rows(files: pd.DataFrame, sample: Sample) -> pd.DataFrame:
    """The `files.parquet` rows this draw selected, in the draw's own order.

    Critical: a recorded id that is no longer in `files.parquet` raises. It
    means the corpus moved under a draw -- a source re-probed with different
    exclusions, or a blocked source's rows dropped -- and silently measuring the
    remainder would publish an S tier that does not match its own sample.json.
    """
    have = files.set_index("file_id")
    missing = [i for i in sample.file_ids if i not in have.index]
    if missing:
        raise KeyError(
            f"partition {sample.partition}: {len(missing)} sampled file_id(s) are "
            f"no longer in files.parquet, e.g. {missing[:3]}. The corpus changed "
            f"under the draw; redraw it deliberately (`draw(..., force=True)`) "
            f"rather than measuring what is left")
    return have.loc[list(sample.file_ids)].reset_index()
