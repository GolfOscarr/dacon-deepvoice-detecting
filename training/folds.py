"""`folds.parquet`: which fold every source file rotates through, and what is sealed.

Built **once** from the manifest and reused everywhere -- strongly evidenced,
docs/validation/01 §3 E4. Regenerating it requires bumping `scheme_version`;
results across versions are never compared.

Critical: the constraint is **artifact-family disjointness, not generator-name
disjointness** (docs/validation/01 §1). Many open TTS systems share a vocoder
or a neural codec, so holding out "XTTS-v2" while training on three other
HiFi-GAN systems is not a generator-disjoint split. `artifact_family` is
resolved from the artifact-producing stage and frozen before the first fold is
built; this module only reads it. It binds on **fake** rows only -- a real
component has no family, which is why the other four keys exist.

## Sealed PROBE, rotating TRAIN/VAL

A family is either **sealed into PROBE** -- never TRAIN, never VAL, in any fold --
or it **rotates**: VAL in exactly one fold, TRAIN in all the others. This is
docs/validation/01 §3's music option 1 read literally ("5 TRAIN / 2 VAL rotating /
1 sealed PROBE"), and it is why the family floors stay at ≥20 voice / ≥8 music.

    slice ∈ {train_val, shadow, probe}          fold ∈ {0..k-1} | null

    fold k:  VAL   = rows with slice == train_val and fold == k
             TRAIN = rows with slice == train_val and fold != k
             PROBE = rows with slice == probe                 (sealed, VG6)
             SHADOW= rows with slice == shadow and fold == k   (VAL's annex)

`apply_folds(manifest, folds, fold=k)` materializes that view as the
`slice(train|val|shadow|probe)` column `training.sampler.Sampler` reads.

Caveat: **The alternative reading, and why it was rejected.** An earlier
version of this module read `slice` as static -- a family assigned to TRAIN or
VAL once and for good -- which is the literal reading of VG1 A1 as it was
originally written ("appears in exactly one of {train, val, probe}"). It is
coherent, but it makes VAL a fixed ~25% of families, so a 5-fold with two
families per validation fold needs **~40 voice and ~16 music families** rather
than the ≥20 / ≥8 that docs/data/08 is budgeted for. The project owner chose
rotation; A1 was reworded to match (docs/validation/04). Recorded here so it is
not rediscovered.

## Two things this module still decides

- **SHADOW is an annex of VAL.** It draws its files from VAL's families by
  construction (docs/validation/01 §2), so it shares its parent's fold and is
  folded into the `train_val` cell for A1-A5. Without that, every S-a channel
  re-render trips A4/A5 against the very VAL file it is paired to.
- **Every grouping-key value resolves to exactly one (slice, fold) cell.** That
  is what makes the rotation safe: a family in two folds would be VAL and TRAIN
  *simultaneously* in each of them. A1-A5 are that statement, per key.

VG1 A8/A9 are **not** here. They are statements about compositions, a component
row has no cell, and they are already implemented in
`training.audit.audit_specs(..., eval_floors=True)` against `val_specs.parquet`
(docs/validation/04 §VG1).

A1-A7 and A10 are not here either, and neither is the emitted schema: they judge
a table rather than build one, so they live in `training.foldcheck` and are
re-exported from here (`check_split_integrity`, `validate_folds`, `load_folds`,
`FOLD_COLUMNS`, `FOLD_SLICES`).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from training.foldcheck import (FOLD_COLUMNS, FOLD_SLICES, check_split_integrity,
                               load_folds, validate_folds)
from training.manifest import POOL_IS_FAKE, validate_manifest

__all__ = ["FOLD_COLUMNS", "FOLD_SLICES", "GROUPING_KEYS", "HEADS",
           "FoldConfig", "FoldInfeasible", "FoldPlan",
           "apply_folds", "build_folds", "check_split_integrity",
           "grouping_atoms", "load_folds", "validate_folds"]

#: docs/validation/01 §1. Every one of them is an equivalence constraint -- "same
#: slice" or "disjoint across slices" -- so the only correct treatment is a
#: union-find over all five at once. Splitting them into a priority order would
#: let a `pair_id` quietly move one member of a `dup_group`.
GROUPING_KEYS: tuple[str, ...] = (
    "artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group")

#: What the emitted `slice` column holds. `train_val` rotates; `probe` is sealed.

HEADS: tuple[str, ...] = ("voice", "music")

#: Roles a fold's TRAIN and VAL sides must each be able to compose from, or the
#: sampler cannot draw. `(role, is_fake)`.
_COVERAGE = (("voice", False), ("voice", True), ("music", False), ("music", True),
             ("noise", False))


#: How many times `probe_share` of the ROWS a family-advancing PROBE seal may
#: reach before the group rotates instead (see `_seal_probe`). 2.0: PROBE is a
#: ~10 % family share and may not, by atom size alone, become a fifth of the
#: corpus that is never trained on.
PROBE_ROW_BUDGET = 2.0


class FoldInfeasible(ValueError):
    """The corpus cannot support the requested split. Never silently downgraded."""


# --------------------------------------------------------------------------- #
# grouping

class _Union:
    def __init__(self) -> None:
        self.parent: dict[object, object] = {}

    def find(self, x: object) -> object:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def grouping_atoms(manifest: pd.DataFrame) -> pd.Series:
    """`file_id -> group id`: the smallest unit that can be given a fold.

    Critical: the transitive closure, not five independent checks. A `pair_id`
    binding a LibriTTS utterance to its HiFi-GAN twin also binds the *whole
    LibriTTS corpus* to the *whole HiFi-GAN family*, because `source_name` and
    `artifact_family` are themselves disjointness keys. That closure is the
    real constraint, and a corpus whose twins were generated before the family
    partition was frozen collapses into one inseparable atom -- which
    `build_folds` reports as infeasible rather than resolving by ignoring a
    key.

    The group id is the lexicographically smallest `file_id` in the group, so it
    is stable across runs and diffable.
    """
    uf = _Union()
    ids = manifest["file_id"].astype(str).to_numpy()
    for fid in ids:
        uf.find(("f", fid))
    for key in GROUPING_KEYS:
        col = manifest[key]
        present = col.notna().to_numpy()
        for fid, value in zip(ids[present], col[present].astype(str).to_numpy()):
            uf.union(("f", fid), (key, value))

    members: dict[object, list[str]] = {}
    for fid in ids:
        members.setdefault(uf.find(("f", fid)), []).append(fid)
    label = {root: min(fids) for root, fids in members.items()}
    return pd.Series([label[uf.find(("f", fid))] for fid in ids],
                     index=manifest["file_id"].astype(str), name="group")


# --------------------------------------------------------------------------- #
# configuration

@dataclass(frozen=True)
class FoldConfig:
    """Everything the builder needs. All of it is reflected in the caveats."""

    #: docs/validation/01 §3. Every rotating family is VAL in exactly one of these.
    n_folds: int = 5
    #: Family share sealed into PROBE (docs/validation/01 §2: ~10%). The TRAIN and
    #: VAL shares are *not* knobs under rotation -- they follow from `n_folds`:
    #: VAL is (1 - probe_share) / n_folds of the families, TRAIN is the rest.
    probe_share: float = 0.10
    scheme_version: str | None = None
    #: Below this many VAL families per head per fold, the per-fold EER is a
    #: one-family estimate. Not fatal -- it is the music head's actual situation
    #: at 8 families (docs/validation/01 §3) -- but it is never left unsaid.
    caveat_families_per_val_fold: int = 2
    #: Critical: A fold whose VAL side has no real voice components cannot
    #: compose cells 1/5/6, and `Sampler` only says so at draw time, long after
    #: the split is frozen. Under rotation this binds per fold, on both the
    #: TRAIN and the VAL side, which is a much stronger requirement than a
    #: static split had.
    require_component_coverage: bool = True
    #: PROBE is what catches "we tuned against VAL until VAL became a training
    #: set". Building without it is option 2 of docs/validation/01 §3 and leaves a
    #: blind spot on the head that matters most, so it must be asked for.
    allow_no_probe: bool = False
    #: Frozen for reproducible tests; `None` stamps the build time.
    assigned_at: str | None = None
    #: Critical: the PROBE row budget counts only rows the sampler DRAWS --
    #: component rows. Whole-file rows are never drawn under docs/processing/03
    #: D-1 (`f8 = 1`), and on the built corpus 25,426 SONICS whole files spent
    #: the whole budget, leaving PROBE's real side with 3 voice files, 1 track
    #: and 8 noise clips (docs/processing/05 A2, 06 D1).
    probe_budget_drawable_only: bool = True
    #: PROBE's real side must be able to estimate a false-positive rate: at
    #: least this many drawable hours and this many grouping atoms of real
    #: voice, real music and noise each (06 D1). Real groups carry no fake
    #: family, so sealing them never starves a head.
    probe_min_real_hours: float = 5.0
    probe_min_real_atoms: int = 5
    #: What the fold rotation balances besides families: `hours` balances the
    #: drawable hours of every (role, fake) pair per fold; `rows` is the
    #: earlier row count. On the built corpus rows balanced (2.9x) while fake
    #: voice VAL hours were 158 / 1.6 / 21 / 13 (05 B4, 06 D4).
    balance_on: str = "hours"
    #: A VAL side with fewer drawable hours than this on any (role, fake) pair
    #: gets a caveat naming the fold; so does a > 3x spread across folds.
    caveat_min_role_hours: float = 5.0

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if not 0.0 <= self.probe_share < 1.0:
            raise ValueError(f"probe_share must be in [0, 1), got {self.probe_share}")
        if self.balance_on not in ("hours", "rows"):
            raise ValueError(f"balance_on must be hours|rows, got {self.balance_on!r}")
        for name in ("probe_min_real_hours", "caveat_min_role_hours"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.probe_min_real_atoms < 0:
            raise ValueError(f"probe_min_real_atoms must be >= 0, got {self.probe_min_real_atoms}")

    @property
    def val_family_share(self) -> float:
        """What fraction of families is VAL in any one fold. Derived, not set."""
        return (1.0 - self.probe_share) / self.n_folds


@dataclass(frozen=True)
class FoldPlan:
    """The emitted table, plus what the builder could not guarantee.

    Caveat: `caveats` are not decoration. The music-head variance caveat exists
    because at 8 music families a 5-fold leaves 1 sealed and 7 rotating -- one
    to two families per validation fold (docs/validation/01 §3) -- and that
    fact has to travel with the table, not with whoever remembers it.
    """

    frame: pd.DataFrame
    caveats: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.caveats

    @property
    def n_folds(self) -> int:
        return int(self.frame["fold"].max()) + 1

    def write(self, path: str | Path) -> Path:
        """Write `folds.parquet`, and the caveats beside it as `.caveats.txt`."""
        path = Path(path)
        self.frame.to_parquet(path, index=False)
        side = path.with_suffix(".caveats.txt")
        side.write_text("\n".join(self.caveats) + ("\n" if self.caveats else ""),
                        encoding="utf-8")
        return path

    def __str__(self) -> str:
        counts = self.frame["slice"].value_counts().to_dict()
        per_fold = self.frame[self.frame["slice"] == "train_val"].groupby(
            "fold").size().to_dict()
        head = (f"folds: {self.n_folds} rotating over "
                + ", ".join(f"{s}={counts.get(s, 0)}" for s in FOLD_SLICES)
                + f"; VAL rows per fold {per_fold}")
        return "\n".join([head] + [f"  CAVEAT: {c}" for c in self.caveats])


# --------------------------------------------------------------------------- #
# building

def _group_facts(manifest: pd.DataFrame, atoms: pd.Series) -> dict[str, dict]:
    """Per group: its families (by head), its row count, and what it can compose."""
    role_of_pool = {"A": "voice", "B": "voice", "C": "music", "D": "music",
                    "E": "noise"}
    df = manifest.assign(_group=manifest["file_id"].astype(str).map(atoms))
    facts: dict[str, dict] = {}
    for group, sub in df.groupby("_group", sort=True):
        fams: dict[str, set[str]] = {h: set() for h in HEADS}
        provides: set[tuple[str, bool]] = set()
        hours: dict[tuple[str, bool], float] = {}
        n_drawable = 0
        for row in sub.to_dict("records"):
            fam = row["artifact_family"]
            if isinstance(fam, str):
                # From the labels, not the pool: a fake whole-file row has no
                # pool, and a cell-8 whole file is fake on both heads at once.
                if row.get("label_voice_fake") == 1:
                    fams["voice"].add(fam)
                if row.get("label_music_fake") == 1:
                    fams["music"].add(fam)
            pool = row["pool"]
            if row["row_kind"] == "component" and isinstance(pool, str):
                key = (role_of_pool[pool], POOL_IS_FAKE[pool])
                provides.add(key)
                hours[key] = hours.get(key, 0.0) + float(row["duration_s"]) / 3600.0
                n_drawable += 1
        facts[group] = {"n_rows": int(len(sub)), "families": fams,
                        "n_families": sum(len(f) for f in fams.values()),
                        "provides": provides,
                        # what the sampler can draw: component rows and their hours
                        "n_drawable": int(n_drawable), "hours": hours,
                        "hours_total": float(sum(hours.values()))}
    return facts


def _strata(manifest: pd.DataFrame, atoms: pd.Series,
            groups: Sequence[str]) -> dict[str, dict]:
    """What docs/validation/01 §3 says to balance folds on, per group.

    `source_name`, duration bucket and channel count.
    """
    df = manifest.assign(_group=manifest["file_id"].astype(str).map(atoms))
    df = df[df["_group"].isin(set(groups))]
    bucket = pd.cut(df["duration_s"], [0, 4, 10, 30, 60, np.inf], right=False)
    out = {
        g: {**{("src", s): c for s, c in sub["source_name"].value_counts().items()},
            **{("dur", str(b)): c for b, c in bucket[sub.index].value_counts().items()},
            **{("ch", int(c)): n for c, n in sub["orig_channels"].value_counts().items()}}
        for g, sub in df.groupby("_group", sort=True)}
    # `value_counts` on a binned column reports empty buckets; a zero-count key
    # is not a balance dimension.
    return {g: {k: int(n) for k, n in keys.items() if n} for g, keys in out.items()}


def _seal_probe(facts: dict[str, dict], cfg: FoldConfig) -> set[str]:
    """Choose the groups sealed into PROBE, before any fold exists.

    Strong evidence, docs/validation/01 §3: "take families in descending size,
    place each into the fold whose current composition is furthest from
    target." Deterministic -- ties break on the group id, never on an RNG,
    because a split that moves between runs cannot be audited.
    """
    if cfg.probe_share <= 0:
        return set()
    totals = {h: len({f for g in facts.values() for f in g["families"][h]})
              for h in HEADS}
    target = {h: cfg.probe_share * totals[h] for h in HEADS}
    # The families a VAL side can actually DRAW from: those some group carries
    # as component rows of the head's fake pool. A family present only as
    # whole-file rows (SONICS under docs/processing/03 D-1) counts toward the
    # sealing target but not toward the rotating floor below -- a fold whose
    # only fake-music family is uncomposable fails coverage all the same.
    composable = {h: {fam for g in facts.values() for fam in g["families"][h]
                      if (h, True) in g["provides"]} for h in HEADS}
    got: dict[str, set] = {h: set() for h in HEADS}
    covered: set[tuple[str, bool]] = set()
    probe: set[str] = set()
    rows_key = "n_drawable" if cfg.probe_budget_drawable_only else "n_rows"
    total_rows = sum(g[rows_key] for g in facts.values())
    probe_rows = 0
    # Critical: among groups with the same family count, the ones that can
    # COMPOSE (`provides` non-empty) come first. A group of whole-file rows
    # only -- SONICS' two families under docs/processing/03 D-1 -- advances a
    # head's family target while giving PROBE nothing it can draw a composed
    # sample from; sealed first by size, it left PROBE with "fake music" it
    # could not compose.
    for group in sorted(facts, key=lambda g: (-facts[g]["n_families"],
                                              -len(facts[g]["provides"]),
                                              -facts[g]["n_rows"], g)):
        f = facts[group]
        # Take the group if it advances the family target on some head, or if
        # it closes a gap in what PROBE can compose. Critical: coverage is not
        # optional: a PROBE with no real voice cannot compose cells 1/5/6, and
        # PROBE is opened at most three times ever (VG6) -- finding out then is
        # too late.
        advances = any(totals[h] and f["families"][h] and len(got[h]) < target[h]
                       for h in HEADS)
        closes = len(covered) < len(_COVERAGE) and bool(f["provides"] - covered)
        # Critical: a group may carry families on BOTH heads (a cell-8 whole
        # file is fake on both; docs/processing/03 D-14 keeps its family), and
        # such groups sort first. Sealing them to advance the larger head
        # can strip the smaller head of every rotating family -- measured on
        # the built corpus: 7 music families, all 7 sealed for the voice
        # target, 0 rotating. So a group is never sealed if that would leave
        # any head below `n_folds` rotating families, which is precisely the
        # condition `_check_feasible` refuses.
        starves = any(
            composable[h] and (f["families"][h] & composable[h])
            and len(composable[h] - (got[h] | f["families"][h])) < cfg.n_folds
            for h in HEADS)
        # PROBE's target is a FAMILY share, but it is sealed forever (VG6), so
        # a family-advancing seal also has a ROW budget: the LJSpeech pair atom
        # (LJSpeech + seven WaveFake families, 89k rows, 32 % of the built
        # corpus) reached PROBE first by family count and 64 % of the corpus
        # was sealed. Over budget, the group rotates instead -- one fold's VAL
        # is then large, which the A8 caveat below says out loud.
        over_budget = (probe_rows + f[rows_key]
                       > PROBE_ROW_BUDGET * cfg.probe_share * total_rows)
        # A seal that closes a coverage gap is taken even if it starves a
        # head: coverage is not optional, and the floor then fails loudly in
        # `_check_feasible` with the message it should. The budget binds on
        # both -- the second pass below closes what the budget left open.
        if not over_budget and (closes or (advances and not starves)):
            probe.add(group)
            probe_rows += f[rows_key]
            for h in HEADS:
                got[h] |= f["families"][h]
            covered |= f["provides"]
    # Coverage the budget left open is closed from the SMALLEST groups that
    # can close it, budget or not: PROBE that cannot compose a cell is unusable,
    # and the cheapest closer costs the rotation the least.
    for group in sorted(facts, key=lambda g: (facts[g]["n_rows"], g)):
        if group in probe or len(covered) >= len(_COVERAGE):
            continue
        f = facts[group]
        if f["provides"] - covered:
            probe.add(group)
            for h in HEADS:
                got[h] |= f["families"][h]
            covered |= f["provides"]
    # Critical: PROBE's REAL side gets a floor in hours and atoms per role.
    # Coverage alone was closed by the smallest closers -- 3 voice files, 1
    # track, 8 noise clips on the built corpus -- and a PROBE that cannot
    # estimate a false-positive rate cannot answer its question (05 A2). Real
    # groups carry no fake family, so sealing them never starves a head;
    # the groups are taken largest first while they fit the remaining need,
    # then smallest first for the atom count.
    for role in ("voice", "music", "noise"):
        key = (role, False)
        pure = [g for g in facts if g not in probe and key in facts[g]["provides"]
                and facts[g]["n_families"] == 0]
        # the floor is capped at the budget's share of the role's real hours,
        # and never seals a group the rotation needs (n_folds providers stay)
        total_h = sum(facts[g]["hours"].get(key, 0.0) for g in facts)
        floor_h = min(cfg.probe_min_real_hours, PROBE_ROW_BUDGET * cfg.probe_share * total_h)
        have_h = sum(facts[g]["hours"].get(key, 0.0) for g in probe)
        have_n = sum(1 for g in probe if key in facts[g]["provides"])

        def rotating_providers() -> int:
            return sum(1 for g in facts if g not in probe and key in facts[g]["provides"])

        for group in sorted(pure, key=lambda g: (-facts[g]["hours"].get(key, 0.0), g)):
            need = floor_h - have_h
            if need <= 1e-9:
                break
            h = facts[group]["hours"].get(key, 0.0)
            if h <= 1.5 * need and rotating_providers() > cfg.n_folds:
                probe.add(group)
                have_h += h
                have_n += 1
        for group in sorted(pure, key=lambda g: (facts[g]["hours"].get(key, 0.0), g)):
            if have_n >= cfg.probe_min_real_atoms or rotating_providers() <= cfg.n_folds:
                break
            if group not in probe:
                probe.add(group)
                have_n += 1
    return probe


def _assign_folds(groups: Sequence[str], facts: dict[str, dict],
                  strata: Mapping[str, dict], cfg: FoldConfig) -> dict[str, int]:
    """Rotate the non-PROBE groups through the folds: each is VAL in exactly one.

    Each dimension is scored against *its own* per-fold target, so a source with
    400 rows and a head with 6 families are comparable quantities. Greedy
    minimisation of the sum of squares is what "furthest from target" means once
    there is more than one thing to be furthest from -- and component coverage
    dominates all of it, because a fold whose VAL side cannot compose a cell is
    not a worse fold, it is an unusable one.
    """
    n_folds = cfg.n_folds
    size_key = "hours_total" if cfg.balance_on == "hours" else "n_rows"
    total_rows = sum(facts[g][size_key] for g in groups) or 1
    # the drawable hours of every (role, fake) pair, balanced per fold (06 D4)
    total_hours = {k: sum(facts[g]["hours"].get(k, 0.0) for g in groups) for k in _COVERAGE}
    total_fam = {h: len({f for g in groups for f in facts[g]["families"][h]})
                 for h in HEADS}
    total_key: dict[object, int] = {}
    for g in groups:
        for key, n in strata.get(g, {}).items():
            total_key[key] = total_key.get(key, 0) + n
    n_keys = max(1, len(total_key))

    tallies: list[dict] = [{} for _ in range(n_folds)]
    fam_tally: list[dict[str, set]] = [{h: set() for h in HEADS}
                                       for _ in range(n_folds)]
    rows_tally = [0.0] * n_folds
    hours_tally: list[dict] = [{k: 0.0 for k in _COVERAGE} for _ in range(n_folds)]
    covered: list[set] = [set() for _ in range(n_folds)]
    out: dict[str, int] = {}
    order = sorted(groups, key=lambda g: (-facts[g]["n_families"],
                                          -facts[g]["n_rows"], g))
    for group in order:
        f, keys = facts[group], strata.get(group, {})
        if cfg.balance_on == "hours":
            # Critical: the duration-bucket and channel strata count ROWS and
            # every group shares them, so the fold holding the LJ atom (101k
            # rows) repelled every other group under them -- fold 0's VAL had
            # 0.0 h of noise. Under hours, only the source strata remain.
            keys = {k: n for k, n in keys.items() if k[0] == "src"}
        best, best_cost = 0, np.inf
        for k in range(n_folds):
            cost = 2.0 * sum(
                (len(fam_tally[k][h] | f["families"][h]) / (total_fam[h] / n_folds)) ** 2
                for h in HEADS if total_fam[h])
            if cfg.balance_on == "hours":
                # Critical: per (role, fake) pair ONLY -- a total-hours term
                # beside it made every other group avoid the fold that holds
                # the LJ atom (182 h of fake voice, indivisible), leaving that
                # fold's VAL with 0.4 h of real music and no noise.
                # ... and only over the pairs THIS group carries: a fold's
                # excess on another pair is the same for every candidate and
                # would make that fold lose every group (fold 0 held 182 h of
                # fake voice and got 0.0 h of noise).
                cost += 2.0 * sum(
                    ((hours_tally[k][key] + f["hours"][key])
                     / (total_hours[key] / n_folds)) ** 2
                    for key in f["hours"] if total_hours.get(key, 0.0) > 0)
            else:
                cost += ((rows_tally[k] + f[size_key]) / (total_rows / n_folds)) ** 2
            cost += sum(
                ((tallies[k].get(key, 0) + n) / (total_key[key] / n_folds)) ** 2
                for key, n in keys.items()) / n_keys
            cost -= 100.0 * len(f["provides"] - covered[k])
            if cost < best_cost:
                best, best_cost = k, cost
        out[group] = best
        for h in HEADS:
            fam_tally[best][h] |= f["families"][h]
        rows_tally[best] += f[size_key]
        for key in _COVERAGE:
            hours_tally[best][key] += f["hours"].get(key, 0.0)
        covered[best] |= f["provides"]
        for key, n in keys.items():
            tallies[best][key] = tallies[best].get(key, 0) + n
    return out


def _check_feasible(facts: dict[str, dict], probe: set[str],
                    fold_of: Mapping[str, int], cfg: FoldConfig) -> list[str]:
    """Raise on what cannot be built; return what merely has to be said out loud."""
    caveats: list[str] = []
    rotating = [g for g in facts if g not in probe]

    fam = {h: {"probe": set(), "rotating": set()} for h in HEADS}
    per_fold_fam = {h: [set() for _ in range(cfg.n_folds)] for h in HEADS}
    for group, f in facts.items():
        for h in HEADS:
            fam[h]["probe" if group in probe else "rotating"] |= f["families"][h]
            if group not in probe:
                per_fold_fam[h][fold_of[group]] |= f["families"][h]

    for head in HEADS:
        total = len(fam[head]["probe"] | fam[head]["rotating"])
        if not total:
            caveats.append(
                f"{head} head: the manifest carries no {head} artifact_family at "
                f"all, so nothing on this head is generator-disjoint")
            continue
        n_rot = len(fam[head]["rotating"])
        if n_rot < cfg.n_folds:
            raise FoldInfeasible(
                f"{head} head: {total} artifact famil{'y' if total == 1 else 'ies'} "
                f"leaves {n_rot} rotating after sealing PROBE, which cannot fill "
                f"{cfg.n_folds} folds -- some fold would validate on no family of "
                f"its own. docs/validation/01 §3 records this for the music head: "
                f"raise the family floor (option 1: >=8 music families) or lower "
                f"n_folds. Do not shrink PROBE to make this pass.")
        realized = [len(s) for s in per_fold_fam[head]]
        if min(realized) < cfg.caveat_families_per_val_fold:
            caveats.append(
                f"{head} head: {n_rot} rotating famil(y/ies) over {cfg.n_folds} "
                f"folds gives {realized} per validation fold (mean "
                f"{n_rot / cfg.n_folds:.1f}). A fold validating on "
                f"{min(realized)} famil{'y' if min(realized) == 1 else 'ies'} is a "
                f"{min(realized)}-family EER estimate -- very high variance "
                f"(docs/validation/01 §3). Every number broken down by fold on "
                f"this head carries that caveat.")
        if not fam[head]["probe"]:
            message = (
                f"{head} head: no artifact_family reached PROBE, so the sealed "
                f"slice cannot answer the question it exists for -- whether VAL "
                f"has become a training set (docs/validation/01 §2).")
            if not cfg.allow_no_probe:
                raise FoldInfeasible(
                    message + " Pass allow_no_probe=True to build anyway; that is "
                    "option 2 of §3 and it is a blind spot, not a shortcut.")
            caveats.append(message)

    if cfg.require_component_coverage:
        def missing(groups: Iterable[str]) -> list[str]:
            have: set = set()
            for g in groups:
                have |= facts[g]["provides"]
            return [f"{'fake' if fake else 'real'} {role}"
                    for role, fake in _COVERAGE if (role, fake) not in have]

        short: list[str] = []
        if probe and (gap := missing(probe)):
            short.append(f"PROBE has no {', '.join(gap)}")
        for k in range(cfg.n_folds):
            val = [g for g in rotating if fold_of[g] == k]
            train = [g for g in rotating if fold_of[g] != k]
            if gap := missing(val):
                short.append(f"fold {k} VAL has no {', '.join(gap)}")
            if gap := missing(train):
                short.append(f"fold {k} TRAIN has no {', '.join(gap)}")
        if short:
            raise FoldInfeasible(
                "the rotation leaves a side the sampler cannot draw from: "
                + "; ".join(short[:6])
                + ". Critical: Under rotation the fold count is bounded by the number of "
                "*real source corpora* per role as well as by the family count: "
                f"every one of the {cfg.n_folds} VAL sides needs its own real "
                "voice, real music and noise source, and PROBE needs one more. "
                "Lower n_folds, add sources, or record source_name at track / "
                "artist / speaker granularity (docs/validation/01 §1 names those "
                "as the examples) so there are more of them.")
    return caveats


def build_folds(manifest: pd.DataFrame, cfg: FoldConfig | None = None,
                shadow_of: Mapping[str, str] | None = None,
                shadow_b: Iterable[str] | None = None) -> FoldPlan:
    """Seal PROBE, then rotate every other family through the `n_folds` folds.

    `shadow_of` maps a re-rendered file to the file it was rendered from (S-a,
    paired); `shadow_b` names files carved out as held-out content slices (S-b,
    unpaired -- sung voice, Korean). Both are inputs because neither is derivable
    from the manifest schema: nothing in it says a row is a telephone re-render
    or that its speaker is singing. Both must name rotating (non-PROBE) files;
    a shadow row inherits its parent's fold, so it is VAL's annex in that fold
    and invisible in the others.

    Caveat: any `slice`/`fold` already on the manifest is ignored. This
    function is the *source* of that assignment, and reading it back would let
    a stale table reproduce itself.
    """
    cfg = cfg or FoldConfig()
    validate_manifest(manifest)
    atoms = grouping_atoms(manifest)
    facts = _group_facts(manifest, atoms)

    shadow_of = dict(shadow_of or {})
    shadow_b = set(shadow_b or ())
    known = set(manifest["file_id"].astype(str))
    unknown = sorted((set(shadow_of) | set(shadow_of.values()) | shadow_b) - known)
    if unknown:
        raise FoldInfeasible(f"shadow file_id(s) not in the manifest: {unknown[:5]}")

    probe = _seal_probe(facts, cfg)
    rotating = sorted(g for g in facts if g not in probe)
    if not rotating:
        raise FoldInfeasible("every group was sealed into PROBE; nothing rotates")
    strata = _strata(manifest, atoms, rotating)
    fold_of = _assign_folds(rotating, facts, strata, cfg)

    caveats = _check_feasible(facts, probe, fold_of, cfg)

    used = sorted({fold_of[g] for g in rotating})
    if used != list(range(cfg.n_folds)):
        raise FoldInfeasible(
            f"fold(s) {sorted(set(range(cfg.n_folds)) - set(used))} came out empty "
            f"over {len(rotating)} rotating group(s); a fold that validates on "
            f"nothing is not a fold")
    per_fold = [sum(facts[g]["n_rows"] for g in rotating if fold_of[g] == k)
                for k in range(cfg.n_folds)]
    # Critical: rows balance while hours do not -- fake voice VAL hours were
    # 158 / 1.6 / 21 / 13 on the built corpus under a silent row caveat (05
    # B4). Every (role, fake) pair's drawable VAL hours are said per fold.
    for key in _COVERAGE:
        hours = [sum(facts[g]["hours"].get(key, 0.0) for g in rotating if fold_of[g] == k)
                 for k in range(cfg.n_folds)]
        name = f"{'fake' if key[1] else 'real'} {key[0]}"
        shown = "[" + ", ".join(f"{h:.1f}" for h in hours) + "]"
        if min(hours) < cfg.caveat_min_role_hours:
            caveats.append(
                f"{name}: VAL hours per fold {shown} -- fold(s) "
                f"{[k for k, h in enumerate(hours) if h < cfg.caveat_min_role_hours]} "
                f"validate on under {cfg.caveat_min_role_hours:g} h; a per-fold number "
                f"on that side is a few-source estimate.")
        elif max(hours) > 3 * max(min(hours), 1e-9):
            caveats.append(
                f"{name}: VAL hours per fold {shown} "
                f"({max(hours) / max(min(hours), 1e-9):.1f}x between the largest and the "
                f"smallest fold). The grouping atoms are indivisible.")
    # Critical: family disjointness dominates row balance: a fold is a whole
    # number of grouping atoms, and those differ in size by an order of
    # magnitude. This is where VG1 A8 bites (docs/validation/04), so it is said
    # out loud here rather than discovered at eval-set materialization.
    if max(per_fold) > 3 * min(per_fold):
        caveats.append(
            f"VAL row counts per fold are uneven: {per_fold} "
            f"({max(per_fold) / min(per_fold):.1f}x between the largest and the "
            f"smallest). The grouping atoms are indivisible, so this cannot be "
            f"fixed by reassignment. Check VG1 A8 (>=1,200 per class per masked "
            f"pool per fold) against val_specs.parquet before quoting a per-fold "
            f"number.")

    # -- SHADOW ------------------------------------------------------------- #
    # Caveat: applied *after* the rotation, never during it. SHADOW is a
    # condition axis (docs/validation/01 §2) and consumes no family budget; a
    # re-render that voted in the family partition would let a channel decision
    # move a generator between folds.
    slice_of_file = {fid: ("probe" if atoms[fid] in probe else "train_val")
                     for fid in known}
    fold_of_file = {fid: fold_of.get(atoms[fid]) for fid in known}
    shadow_kind: dict[str, str] = {}
    bad = [f"{fid} -> {parent}" for fid, parent in sorted(shadow_of.items())
           if slice_of_file[parent] != "train_val"]
    if bad:
        raise FoldInfeasible(
            "every S-a re-render must be paired to a rotating (non-PROBE) file, "
            f"or the VAL->SHADOW drop is not attributable to the channel: {bad[:5]}")
    for fid in sorted(shadow_of):
        shadow_kind[fid] = "a"
    sealed = sorted(f for f in shadow_b if slice_of_file[f] != "train_val")
    if sealed:
        raise FoldInfeasible(
            "S-b held-out content slices are carved out of VAL's families "
            f"(docs/validation/01 §2), not out of sealed PROBE: {sealed[:5]}")
    for fid in sorted(shadow_b):
        shadow_kind[fid] = "b"

    # -- emit ---------------------------------------------------------------- #
    stamp = cfg.assigned_at or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    ids = manifest["file_id"].astype(str).to_numpy()
    out = pd.DataFrame({"file_id": ids, "row_kind": manifest["row_kind"].to_numpy()})
    out["slice"] = [("shadow" if f in shadow_kind else slice_of_file[f]) for f in ids]
    out["shadow_kind"] = [shadow_kind.get(f) for f in ids]
    out["shadow_of"] = [shadow_of.get(f) for f in ids]
    # An S-a row keeps the fold of the file it was rendered from, so the paired
    # VAL->SHADOW delta is read per fold rather than pooled.
    out["fold"] = pd.array(
        [fold_of_file[shadow_of.get(f, f)] for f in ids], dtype="Int64")
    for col in ("artifact_family", "source_name", "speaker_ref_id", "pair_id",
                "dup_group", "cell", "domain_key"):
        out[col] = manifest[col].to_numpy()
    out["cell"] = manifest["cell"].astype("Int64").to_numpy()
    out["assigned_at"] = stamp
    out["scheme_version"] = cfg.scheme_version or manifest["scheme_version"].iloc[0]
    return FoldPlan(validate_folds(out[list(FOLD_COLUMNS)]), tuple(caveats))




def apply_folds(manifest: pd.DataFrame, folds: pd.DataFrame,
                fold: int) -> pd.DataFrame:
    """Materialize the fold-`fold` view: the manifest `Sampler` actually reads.

    docs/pipelines/01 §2: the manifest is the ledger joined to `folds.parquet` on
    `file_id`. Under rotation that join needs a fold to resolve against --
    `train_val` becomes `val` for the families assigned to this fold and `train`
    for all the others.

    Caveat: SHADOW rows whose parent is TRAIN in this fold are **dropped**, not
    carried as `train`. SHADOW is "VAL generators × unseen acoustic
    conditions"; on a fold where those generators are being trained on, the row
    measures nothing and training on it would retire the condition
    (docs/validation/01 §2).
    """
    validate_folds(folds)
    left, right = set(manifest["file_id"].astype(str)), set(folds["file_id"])
    if left != right:
        raise ValueError(
            f"manifest and folds disagree on {len(left ^ right)} file_id(s): "
            f"{sorted(left - right)[:3]} only in manifest, "
            f"{sorted(right - left)[:3]} only in folds")
    if folds["scheme_version"].iloc[0] != manifest["scheme_version"].iloc[0]:
        raise ValueError(
            f"scheme_version mismatch: manifest "
            f"{manifest['scheme_version'].iloc[0]!r} vs folds "
            f"{folds['scheme_version'].iloc[0]!r}")
    n_folds = int(folds.loc[folds["slice"] != "probe", "fold"].max()) + 1
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}, got {fold}")

    keyed = folds.set_index("file_id")
    ids = manifest["file_id"].astype(str)
    slice_ = keyed["slice"].reindex(ids).to_numpy()
    at = keyed["fold"].reindex(ids).to_numpy()
    role = np.where(slice_ == "probe", "probe",
                    np.where(slice_ == "shadow", "shadow",
                             np.where(at == fold, "val", "train")))
    keep = ~((slice_ == "shadow") & (at != fold))

    out = manifest.copy()
    out["slice"] = role
    # Caveat: every kept row carries `fold = fold`. This frame IS the
    # fold-`fold` view, so `Sampler(slice_="train", fold=k)` must see the whole
    # training side -- not a fifth of it.
    out["fold"] = pd.array([fold] * len(out), dtype="Int64")
    return validate_manifest(out[keep].reset_index(drop=True))
