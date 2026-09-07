"""`folds.parquet`: which slice and which validation fold every source file is in.

Built **once** from the manifest and reused everywhere (★ docs/validation/01 §3
E4). Regenerating it requires bumping `scheme_version`; results across versions
are never compared.

🔴 The constraint is **artifact-family disjointness, not generator-name
disjointness** (docs/validation/01 §1). Many open TTS systems share a vocoder or
a neural codec, so holding out "XTTS-v2" while training on three other HiFi-GAN
systems is not a generator-disjoint split. `artifact_family` is resolved from the
artifact-producing stage and frozen before the first fold is built; this module
only reads it. It binds on **fake** rows only -- a real component has no family,
which is why the other four keys exist.

⚠️ Two design decisions this module had to make, because the spec underdetermines
them. Both are recorded here rather than in a commit message:

1. **`slice` is static and `fold` partitions VAL.** VG1 A1 says a family appears
   in exactly one of {train, val, probe} and A6 says PROBE families appear in
   neither -- both are statements about a *fixed* assignment, and the merged
   `training.sampler.Sampler` filters `slice == s AND fold == k`. So a family is
   assigned to one slice for good, and `fold` splits the VAL families into
   `n_folds` family-disjoint validation sets, giving the per-fold mean+sd that
   docs/validation/01 §5 reports on. TRAIN and PROBE rows carry `fold = null`: a
   TRAIN row is used by every fold, so a fold id on it would be an invitation to
   train on a fifth of the corpus.

   ⚠️ docs/validation/01 §3 also describes a *rotating* scheme ("~4 families per
   fold" over ≥20 voice families, "5 TRAIN / 2 VAL rotating / 1 sealed PROBE"),
   under which a family is TRAIN in four folds and VAL in the fifth. That
   contradicts A1 read literally, and it changes the family floors by ~4x --
   under the static reading a 5-fold with two families per validation fold needs
   ≥40 families per head, not ≥20. The discrepancy is real and unresolved; this
   module implements the A1-faithful reading and `build_folds` emits a caveat
   whenever a head is thin, rather than picking a floor silently.

2. **SHADOW is an annex of VAL for the grouping keys.** It draws its files from
   VAL's families by construction (docs/validation/01 §2), so A1-A5 are evaluated
   on an *effective* slice where `shadow` maps to `val`. Without that, every S-a
   channel re-render trips A4/A5 against the very VAL file it is paired to.

VG1 A8/A9 are **not** here. They are statements about compositions, a component
row has no cell, and they are already implemented in
`training.audit.audit_specs(..., eval_floors=True)` against `val_specs.parquet`
(docs/validation/04 §VG1).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from training.audit import AuditReport
from training.manifest import POOL_IS_FAKE, validate_manifest

__all__ = ["FOLD_COLUMNS", "GROUPING_KEYS", "HEADS", "PARTITIONED_SLICES",
           "FoldConfig", "FoldInfeasible", "FoldPlan",
           "apply_folds", "build_folds", "check_split_integrity",
           "grouping_atoms", "load_folds", "validate_folds"]

#: docs/validation/01 §3, in the order the spec lists them.
FOLD_COLUMNS: tuple[str, ...] = (
    "file_id", "row_kind", "slice", "shadow_kind", "shadow_of", "fold",
    "artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group",
    "cell", "domain_key", "assigned_at", "scheme_version",
)

#: docs/validation/01 §1. Every one of them is an equivalence constraint -- "same
#: slice" or "disjoint across slices" -- so the only correct treatment is a
#: union-find over all five at once. Splitting them into a priority order would
#: let a `pair_id` quietly move one member of a `dup_group`.
GROUPING_KEYS: tuple[str, ...] = (
    "artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group")

#: The three generator-partitioned slices, which sum to 100%. SHADOW is a
#: condition axis, not a generator axis, and consumes no budget from them.
PARTITIONED_SLICES: tuple[str, ...] = ("train", "val", "probe")

HEADS: tuple[str, ...] = ("voice", "music")

#: SHADOW draws its files from VAL's families, so it is VAL for A1-A5.
_EFFECTIVE = {"shadow": "val"}

#: Roles a slice must be able to compose from, or the sampler cannot draw in it.
_COVERAGE = (("voice", False), ("voice", True), ("music", False), ("music", True),
             ("noise", False))


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
    """`file_id -> group id`: the smallest unit that can be assigned to a slice.

    🔴 The transitive closure, not five independent checks. A `pair_id` binding a
    LibriTTS utterance to its HiFi-GAN twin also binds the *whole LibriTTS
    corpus* to the *whole HiFi-GAN family*, because `source_name` and
    `artifact_family` are themselves disjointness keys. That closure is the real
    constraint, and a corpus whose twins were generated before the family
    partition was frozen collapses into one inseparable atom -- which
    `build_folds` reports as infeasible rather than resolving by ignoring a key.

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


def _head_of_row(row: pd.Series) -> tuple[str, ...]:
    """Which head(s) a *fake* row's artifact_family belongs to.

    From the labels, not from the pool: a fake whole-file row has no pool, and a
    cell-8 whole file is fake on both heads at once.
    """
    heads = []
    if row.get("label_voice_fake") == 1:
        heads.append("voice")
    if row.get("label_music_fake") == 1:
        heads.append("music")
    return tuple(heads)


# --------------------------------------------------------------------------- #
# configuration

@dataclass(frozen=True)
class FoldConfig:
    """Everything the builder needs. All of it is recorded in the caveats."""

    #: docs/validation/01 §3. `fold(0-4)` in the emitted schema.
    n_folds: int = 5
    #: docs/validation/01 §2, family shares of the three partitioned slices.
    shares: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.65, "val": 0.25, "probe": 0.10})
    #: `None` inherits the manifest's, which is what `apply_folds` requires them
    #: to agree on. Set it explicitly only when deliberately bumping the scheme.
    scheme_version: str | None = None
    #: Below this many VAL families per head per validation fold, the per-fold
    #: EER is a one-family estimate. Not fatal -- it is the music head's actual
    #: situation (docs/validation/01 §3) -- but it is never left unsaid.
    caveat_families_per_val_fold: int = 2
    #: 🔴 A slice with no real voice components cannot compose cells 1/5/6, so
    #: the sampler raises at draw time, long after the split is frozen.
    require_component_coverage: bool = True
    #: PROBE is what catches "we tuned against VAL until VAL became a training
    #: set". Building without it is option 2 of docs/validation/01 §3 and leaves
    #: a blind spot on the head that matters most, so it must be asked for.
    allow_no_probe: bool = False
    #: Frozen for reproducible tests; `None` stamps the build time.
    assigned_at: str | None = None

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if set(self.shares) != set(PARTITIONED_SLICES):
            raise ValueError(f"shares needs {PARTITIONED_SLICES}, got {sorted(self.shares)}")
        if any(v < 0 for v in self.shares.values()):
            raise ValueError(f"shares must be >= 0, got {dict(self.shares)}")
        total = sum(self.shares.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"the three partitioned slices must sum to 1, got {total}")
        if self.shares["val"] <= 0:
            raise ValueError("a zero VAL share leaves nothing to validate on")


@dataclass(frozen=True)
class FoldPlan:
    """The emitted table, plus what the builder could not guarantee.

    ⚠️ `caveats` are not decoration. The music-head variance caveat exists
    because docs/validation/01 §3 says at 5 music families a 5-fold puts one
    family in each validation fold and PROBE cannot be carved out at all -- and
    that fact has to travel with the table, not with whoever remembers it.
    """

    frame: pd.DataFrame
    caveats: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.caveats

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
        head = "folds: " + ", ".join(f"{s}={counts.get(s, 0)}"
                                     for s in ("train", "val", "shadow", "probe"))
        return "\n".join([head] + [f"  CAVEAT: {c}" for c in self.caveats])


# --------------------------------------------------------------------------- #
# building

def _group_facts(manifest: pd.DataFrame, atoms: pd.Series) -> dict[str, dict]:
    """Per group: its families (by head), its row count, and what it can compose."""
    df = manifest.assign(_group=manifest["file_id"].astype(str).map(atoms))
    facts: dict[str, dict] = {}
    for group, sub in df.groupby("_group", sort=True):
        fams: dict[str, set[str]] = {h: set() for h in HEADS}
        provides: set[tuple[str, bool]] = set()
        for row in sub.to_dict("records"):
            fam = row["artifact_family"]
            if isinstance(fam, str):
                for head in _head_of_row(pd.Series(row)):
                    fams[head].add(fam)
            pool = row["pool"]
            if row["row_kind"] == "component" and isinstance(pool, str):
                fake = POOL_IS_FAKE[pool]
                role = {"A": "voice", "B": "voice", "C": "music",
                        "D": "music", "E": "noise"}[pool]
                provides.add((role, fake))
        facts[group] = {"n_rows": int(len(sub)), "families": fams,
                        "n_families": sum(len(f) for f in fams.values()),
                        "provides": provides}
    return facts


def _assign_slices(facts: dict[str, dict], cfg: FoldConfig) -> dict[str, str]:
    """Greedy: biggest group first, into whichever slice is furthest from target.

    ★ docs/validation/01 §3: "take families in descending size, place each into
    the fold whose current composition is furthest from target." Deterministic --
    ties break on the group id, never on an RNG, because a split that moves
    between runs cannot be audited.
    """
    totals = {h: len({f for g in facts.values() for f in g["families"][h]})
              for h in HEADS}
    n_rows = sum(g["n_rows"] for g in facts.values())
    target_fam = {s: {h: cfg.shares[s] * totals[h] for h in HEADS}
                  for s in PARTITIONED_SLICES}
    target_rows = {s: cfg.shares[s] * n_rows for s in PARTITIONED_SLICES}

    got_fam = {s: {h: set() for h in HEADS} for s in PARTITIONED_SLICES}
    got_rows = {s: 0 for s in PARTITIONED_SLICES}
    covered = {s: set() for s in PARTITIONED_SLICES}
    assignment: dict[str, str] = {}

    # ⚠️ A zero-share slice is not a destination. Without this the coverage term
    # below still routes groups into it, and `shares={"probe": 0.0}` silently
    # produced a PROBE slice.
    wanted = [s for s in PARTITIONED_SLICES if cfg.shares[s] > 0]
    order = sorted(facts, key=lambda g: (-facts[g]["n_families"],
                                         -facts[g]["n_rows"], g))
    for group in order:
        f = facts[group]
        best, best_score = None, -np.inf
        for s in wanted:
            score = 0.0
            for h in HEADS:
                if totals[h] and f["families"][h]:
                    deficit = target_fam[s][h] - len(got_fam[s][h])
                    score += 4.0 * deficit / max(1.0, target_fam[s][h])
            if target_rows[s]:
                score += (target_rows[s] - got_rows[s]) / target_rows[s]
            # 🔴 Coverage dominates. A slice the sampler cannot draw from is not
            # a worse split, it is an unusable one.
            score += 10.0 * len(f["provides"] - covered[s])
            if score > best_score:
                best, best_score = s, score
        assignment[group] = best
        for h in HEADS:
            got_fam[best][h] |= f["families"][h]
        got_rows[best] += f["n_rows"]
        covered[best] |= f["provides"]
    return assignment


def _assign_val_folds(groups: Sequence[str], facts: dict[str, dict],
                      manifest: pd.DataFrame, atoms: pd.Series,
                      cfg: FoldConfig) -> dict[str, int]:
    """Split the VAL groups into `n_folds` family-disjoint validation folds.

    Balanced on what docs/validation/01 §3 says to balance on -- `source_name`,
    duration bucket and channel count -- plus the per-head family count, since
    that is what the fold-variance read in §5 is actually over.
    """
    df = manifest.assign(_group=manifest["file_id"].astype(str).map(atoms))
    df = df[df["_group"].isin(set(groups))]
    bucket = pd.cut(df["duration_s"], [0, 4, 10, 30, 60, np.inf], right=False)
    strata = {
        g: {**{("src", s): c for s, c in sub["source_name"].value_counts().items()},
            **{("dur", str(b)): c for b, c in bucket[sub.index].value_counts().items()},
            **{("ch", int(c)): n for c, n in sub["orig_channels"].value_counts().items()}}
        for g, sub in df.groupby("_group", sort=True)}
    # `value_counts` on a binned column reports empty buckets; a zero-count key
    # is not a balance dimension.
    strata = {g: {k: int(n) for k, n in keys.items() if n} for g, keys in strata.items()}

    # Each dimension is scored against *its own* per-fold target, so a source
    # with 400 rows and a head with 6 families are comparable quantities. Greedy
    # minimisation of the sum of squares is what "furthest from target" means
    # once there is more than one thing to be furthest from.
    n_folds = cfg.n_folds
    total_rows = sum(facts[g]["n_rows"] for g in groups) or 1
    total_fam = {h: len({f for g in groups for f in facts[g]["families"][h]})
                 for h in HEADS}
    total_key: dict[object, int] = {}
    for keys in strata.values():
        for key, n in keys.items():
            total_key[key] = total_key.get(key, 0) + n
    n_keys = max(1, len(total_key))

    tallies: list[dict] = [{} for _ in range(n_folds)]
    fam_tally: list[dict[str, set]] = [{h: set() for h in HEADS}
                                       for _ in range(n_folds)]
    rows_tally = [0] * n_folds
    out: dict[str, int] = {}
    order = sorted(groups, key=lambda g: (-facts[g]["n_families"],
                                          -facts[g]["n_rows"], g))
    for group in order:
        f, keys = facts[group], strata.get(group, {})
        best, best_cost = 0, np.inf
        for k in range(n_folds):
            cost = 2.0 * sum(
                (len(fam_tally[k][h] | f["families"][h]) / (total_fam[h] / n_folds)) ** 2
                for h in HEADS if total_fam[h])
            cost += ((rows_tally[k] + f["n_rows"]) / (total_rows / n_folds)) ** 2
            cost += sum(
                ((tallies[k].get(key, 0) + n) / (total_key[key] / n_folds)) ** 2
                for key, n in keys.items()) / n_keys
            if cost < best_cost:
                best, best_cost = k, cost
        out[group] = best
        for h in HEADS:
            fam_tally[best][h] |= f["families"][h]
        rows_tally[best] += f["n_rows"]
        for key, n in keys.items():
            tallies[best][key] = tallies[best].get(key, 0) + n
    return out


def build_folds(manifest: pd.DataFrame, cfg: FoldConfig | None = None,
                shadow_of: Mapping[str, str] | None = None,
                shadow_b: Iterable[str] | None = None) -> FoldPlan:
    """Assign every manifest row a slice and, in VAL, a validation fold.

    `shadow_of` maps a re-rendered file to the VAL file it was rendered from
    (S-a, paired); `shadow_b` names files carved out of VAL as held-out content
    slices (S-b, unpaired -- sung voice, Korean). Both are inputs because
    neither is derivable from the manifest schema: nothing in it says a row is a
    telephone re-render or that its speaker is singing.

    ⚠️ Any `slice`/`fold` already on the manifest is ignored. This function is
    the *source* of that assignment, and reading it back would let a stale table
    reproduce itself.
    """
    cfg = cfg or FoldConfig()
    validate_manifest(manifest)
    atoms = grouping_atoms(manifest)
    facts = _group_facts(manifest, atoms)
    caveats: list[str] = []

    shadow_of = dict(shadow_of or {})
    shadow_b = set(shadow_b or ())
    known = set(manifest["file_id"].astype(str))
    unknown = sorted((set(shadow_of) | shadow_b) - known)
    if unknown:
        raise FoldInfeasible(f"shadow file_id(s) not in the manifest: {unknown[:5]}")

    slice_of_group = _assign_slices(facts, cfg)
    slice_of_file = {fid: slice_of_group[atoms[fid]] for fid in known}

    # -- feasibility, before anything downstream can quietly absorb it -------- #
    fam_slices: dict[str, dict[str, set[str]]] = {
        h: {s: set() for s in PARTITIONED_SLICES} for h in HEADS}
    for group, s in slice_of_group.items():
        for h in HEADS:
            fam_slices[h][s] |= facts[group]["families"][h]

    for head in HEADS:
        total = sum(len(v) for v in fam_slices[head].values())
        if not total:
            caveats.append(
                f"{head} head: the manifest carries no {head} artifact_family at "
                f"all, so nothing on this head is generator-disjoint")
            continue
        n_val = len(fam_slices[head]["val"])
        if n_val < cfg.n_folds:
            raise FoldInfeasible(
                f"{head} head: {total} artifact famil{'y' if total == 1 else 'ies'} "
                f"leaves {n_val} in VAL, which cannot fill {cfg.n_folds} "
                f"family-disjoint validation folds. docs/validation/01 §3 records "
                f"this for the music head at 5 families: a 5-fold puts one family "
                f"in each validation fold and PROBE cannot be carved out at all. "
                f"Raise the family floor (option 1: >=8 music families) or lower "
                f"n_folds -- do not lower the VAL share to make this pass.")
        per_fold = n_val / cfg.n_folds
        if per_fold < cfg.caveat_families_per_val_fold:
            caveats.append(
                f"{head} head: {n_val} VAL famil{'y' if n_val == 1 else 'ies'} over "
                f"{cfg.n_folds} folds = {per_fold:.1f} per validation fold. Per-fold "
                f"EER is a {per_fold:.1f}-family estimate -- very high variance "
                f"(docs/validation/01 §3 option 2). Every number broken down by "
                f"fold on this head carries that caveat.")
        if not fam_slices[head]["probe"]:
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
        short = []
        for s in (x for x in PARTITIONED_SLICES if cfg.shares[x] > 0):
            have = set().union(*(facts[g]["provides"] for g, gs in
                                 slice_of_group.items() if gs == s)) or set()
            missing = [f"{'fake' if fake else 'real'} {role}"
                       for role, fake in _COVERAGE if (role, fake) not in have]
            if missing:
                short.append(f"{s} has no {', '.join(missing)} component(s)")
        if short:
            raise FoldInfeasible(
                "the split leaves a slice the sampler cannot draw from: "
                + "; ".join(short) + ". The grouping keys admit no split with "
                "wider coverage -- add sources or families, do not relax a key.")

    # -- VAL folds ----------------------------------------------------------- #
    val_groups = sorted(g for g, s in slice_of_group.items() if s == "val")
    fold_of_group = _assign_val_folds(val_groups, facts, manifest, atoms, cfg)

    per_fold = [sum(facts[g]["n_rows"] for g in val_groups if fold_of_group[g] == k)
                for k in range(cfg.n_folds)]
    empty = [k for k, n in enumerate(per_fold) if not n]
    if empty:
        raise FoldInfeasible(
            f"validation fold(s) {empty} came out empty over {len(val_groups)} VAL "
            f"group(s); a fold that validates on nothing is not a fold")
    # 🔴 Family disjointness dominates row balance: a VAL fold is a whole number
    # of grouping atoms, and those differ in size by an order of magnitude. This
    # is where VG1 A8 bites (docs/validation/04), so it is said out loud here
    # rather than discovered at eval-set materialization.
    # ⚠️ Per-*fold* component coverage is weaker than per-slice coverage and is
    # reported rather than enforced: with three noise corpora and five folds it
    # cannot hold, and a composed VAL sample draws its voice and music
    # components independently anyway, so "the fold of a composed sample" is not
    # yet a defined quantity (docs/pipelines/02 §5 materializes val_specs over
    # the whole VAL slice). Sampling per VAL fold is what this bites.
    thin_folds = {}
    for k in range(cfg.n_folds):
        have = set().union(*(facts[g]["provides"] for g in val_groups
                             if fold_of_group[g] == k)) or set()
        missing = [f"{'fake' if fake else 'real'} {role}"
                   for role, fake in _COVERAGE if (role, fake) not in have]
        if missing:
            thin_folds[k] = missing
    if thin_folds:
        caveats.append(
            f"VAL fold(s) {sorted(thin_folds)} cannot compose every cell on their "
            f"own: {thin_folds}. Materialize val_specs over the whole VAL slice "
            f"(docs/pipelines/02 §5) rather than per fold, or Sampler raises at "
            f"draw time.")
    if max(per_fold) > 3 * min(per_fold):
        caveats.append(
            f"VAL fold row counts are uneven: {per_fold} (max {max(per_fold) / min(per_fold):.1f}x "
            f"min). The grouping atoms are indivisible, so this cannot be fixed by "
            f"reassignment. Check VG1 A8 (>=1,200 per class per masked pool per "
            f"fold) against val_specs.parquet before quoting a per-fold number.")

    # -- SHADOW overrides ---------------------------------------------------- #
    # ⚠️ Applied *after* the partition, never during it. SHADOW is a condition
    # axis (docs/validation/01 §2) and consumes no family budget; a re-render
    # that voted in the family partition would let a channel decision move a
    # generator across slices.
    shadow_kind: dict[str, str] = {}
    bad_parent = []
    for fid, parent in sorted(shadow_of.items()):
        if slice_of_file.get(parent) != "val":
            bad_parent.append(f"{fid} -> {parent} (slice "
                              f"{slice_of_file.get(parent, 'missing')!r})")
        shadow_kind[fid] = "a"
    if bad_parent:
        raise FoldInfeasible(
            "every S-a re-render must be paired to a VAL file, or the VAL->SHADOW "
            f"drop is not attributable to the channel: {bad_parent[:5]}")
    not_val = sorted(f for f in shadow_b if slice_of_file.get(f) != "val")
    if not_val:
        raise FoldInfeasible(
            "S-b held-out content slices are carved out of VAL's families "
            f"(docs/validation/01 §2); these are not in VAL: {not_val[:5]}")
    for fid in sorted(shadow_b):
        shadow_kind[fid] = "b"

    # -- emit ---------------------------------------------------------------- #
    stamp = cfg.assigned_at or dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    out = pd.DataFrame({
        "file_id": manifest["file_id"].astype(str).to_numpy(),
        "row_kind": manifest["row_kind"].to_numpy(),
    })
    out["slice"] = [shadow_kind.get(f) and "shadow" or slice_of_file[f]
                    for f in out["file_id"]]
    out["shadow_kind"] = [shadow_kind.get(f) for f in out["file_id"]]
    out["shadow_of"] = [shadow_of.get(f) for f in out["file_id"]]
    # A shadow row keeps the fold of the VAL family it came from, so the paired
    # VAL->SHADOW delta is read per fold rather than pooled.
    fold: list[int | None] = []
    for fid, sl in zip(out["file_id"], out["slice"]):
        if sl == "val":
            fold.append(fold_of_group[atoms[fid]])
        elif sl == "shadow":
            parent = shadow_of.get(fid, fid)
            fold.append(fold_of_group.get(atoms[parent]))
        else:
            fold.append(None)
    out["fold"] = pd.array(fold, dtype="Int64")
    for col in ("artifact_family", "source_name", "speaker_ref_id", "pair_id",
                "dup_group", "cell", "domain_key"):
        out[col] = manifest[col].to_numpy()
    out["cell"] = manifest["cell"].astype("Int64").to_numpy()
    out["assigned_at"] = stamp
    out["scheme_version"] = cfg.scheme_version or manifest["scheme_version"].iloc[0]
    out = out[list(FOLD_COLUMNS)]
    return FoldPlan(validate_folds(out), tuple(caveats))


# --------------------------------------------------------------------------- #
# schema and use

def validate_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the emitted schema, or raise with what is wrong. Returns `df`."""
    missing = [c for c in FOLD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"folds table is missing column(s): {missing}")
    if df.empty:
        raise ValueError("folds table is empty")
    if df.file_id.duplicated().any():
        dup = df.file_id[df.file_id.duplicated()].unique()[:5].tolist()
        raise ValueError(f"duplicate file_id(s): {dup}")
    bad = set(df["slice"].unique()) - {"train", "val", "shadow", "probe"}
    if bad:
        raise ValueError(f"unknown slice(s) {sorted(bad)}")
    bad = set(df.shadow_kind.dropna().unique()) - {"a", "b"}
    if bad:
        raise ValueError(f"shadow_kind must be a|b|null, got {sorted(bad)}")
    is_shadow = df["slice"] == "shadow"
    if df.loc[~is_shadow, "shadow_kind"].notna().any():
        raise ValueError("shadow_kind is set on a row outside the shadow slice")
    if df.loc[is_shadow, "shadow_kind"].isna().any():
        raise ValueError("every shadow row needs shadow_kind = a|b")
    if df.loc[df.shadow_kind != "a", "shadow_of"].notna().any():
        raise ValueError("shadow_of belongs to S-a rows only; S-b is unpaired")
    # 🔴 `fold` is non-null exactly on val and shadow. See the module docstring:
    # a TRAIN row is used by every fold, and PROBE has no fold structure.
    folded = df["slice"].isin(["val", "shadow"])
    if df.loc[~folded, "fold"].notna().any():
        raise ValueError("fold must be null outside val/shadow")
    if df.loc[folded, "fold"].isna().any():
        raise ValueError("every val/shadow row needs a fold")
    if len(df.loc[folded]) and int(df.loc[folded, "fold"].min()) < 0:
        raise ValueError("fold must be >= 0")
    # docs/validation/01 §3: cell is non-null exactly when row_kind == whole_file.
    whole = df.row_kind == "whole_file"
    if df.loc[whole, "cell"].isna().any() or df.loc[~whole, "cell"].notna().any():
        raise ValueError("cell must be non-null exactly on whole_file rows")
    if df.scheme_version.nunique() != 1:
        raise ValueError(
            f"folds table mixes scheme_version {sorted(df.scheme_version.unique())}; "
            "regenerating requires bumping it, and versions are never compared")
    return df


def load_folds(path: str | Path) -> pd.DataFrame:
    return validate_folds(pd.read_parquet(path))


def apply_folds(manifest: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Join the split onto the manifest, which is how the pipeline reads it.

    docs/pipelines/01 §2: the manifest is the ledger joined to `folds.parquet` on
    `file_id`. ⚠️ An inner join would silently drop a file the split forgot, so a
    mismatched id set raises here instead.
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
    keyed = folds.set_index("file_id")
    out = manifest.copy()
    ids = out["file_id"].astype(str)
    out["slice"] = keyed["slice"].reindex(ids).to_numpy()
    out["fold"] = pd.array(keyed["fold"].reindex(ids).to_numpy(), dtype="Int64")
    return validate_manifest(out)


# --------------------------------------------------------------------------- #
# VG1

def _effective(df: pd.DataFrame) -> pd.Series:
    return df["slice"].replace(_EFFECTIVE)


def _spanning(df: pd.DataFrame, key: str, slices: Sequence[str]) -> list[str]:
    """Values of `key` present in more than one of `slices`. Sorted, so the
    failure message is stable across runs."""
    sub = df[df["_eff"].isin(slices) & df[key].notna()]
    if sub.empty:
        return []
    seen = sub.groupby(key)["_eff"].nunique()
    return sorted(str(v) for v in seen[seen > 1].index)


def check_split_integrity(folds: pd.DataFrame,
                          run_scheme_version: str | None = None) -> AuditReport:
    """VG1 A1-A7 and A10 over `folds.parquet` (docs/validation/04 §VG1).

    🔴 A8/A9 are **not** here and are not silently passed. They are statements
    about compositions, evaluated against `val_specs.parquet` by
    `training.audit.audit_specs(..., eval_floors=True)`; a component row has no
    cell, so they cannot be computed on this table at all.
    """
    validate_folds(folds)
    df = folds.assign(_eff=_effective(folds))
    r: dict[str, tuple[bool, str]] = {}

    # A1 -- artifact_family appears in exactly one of {train, val, probe}.
    spanning = _spanning(df, "artifact_family", PARTITIONED_SLICES)
    n_fam = int(df["artifact_family"].nunique())
    r["A1_family_in_one_slice"] = (
        not spanning,
        f"{len(spanning)}/{n_fam} artifact_famil(y/ies) span more than one of "
        f"{list(PARTITIONED_SLICES)}" + (f": {spanning[:5]}" if spanning else "")
        + " (SHADOW counts as VAL: it draws from VAL's families by construction)")

    # A2 / A3 -- source and speaker disjoint across train and val.
    for key, name in (("source_name", "A2_source_disjoint"),
                      ("speaker_ref_id", "A3_speaker_disjoint")):
        spanning = _spanning(df, key, ("train", "val"))
        r[name] = (not spanning,
                   f"{len(spanning)} {key}(s) span train and val"
                   + (f": {spanning[:5]}" if spanning else ""))
    # ⚠️ A3 keys on the **bare** speaker_ref_id, not on (source_name, speaker).
    # docs/validation/01 §1 says "disjoint within source_name", but A2 already
    # forbids a source_name from spanning train and val, so the within-source
    # form is entailed by A2 and could never fail on its own. The bare id is the
    # falsifiable statement, and it is the one that matters: a LibriTTS speaker
    # reappearing under a `gen_hifigan` resynthesis is exactly the leak.

    # A4 / A5 -- pair members share a slice; a dup_group is wholly within one.
    for key, name in (("pair_id", "A4_pair_shares_a_slice"),
                      ("dup_group", "A5_dup_group_in_one_slice")):
        spanning = _spanning(df, key, ("train", "val", "probe"))
        r[name] = (not spanning,
                   f"{len(spanning)} {key}(s) split across slices"
                   + (f": {spanning[:5]}" if spanning else ""))

    # A6 -- PROBE families seen in neither train nor val.
    # ⚠️ The second clause is entailed by A1. The first -- that PROBE exists at
    # all -- is not, and it is the one that fails in practice: at 5 music
    # families PROBE cannot be carved out, and A1 would pass vacuously over an
    # empty slice (docs/validation/01 §3).
    probe = set(df.loc[df["slice"] == "probe", "artifact_family"].dropna())
    seen = set(df.loc[df["_eff"].isin(("train", "val")), "artifact_family"].dropna())
    leaked = sorted(probe & seen)
    n_probe_rows = int((df["slice"] == "probe").sum())
    r["A6_probe_sealed"] = (
        bool(probe) and not leaked,
        f"PROBE holds {n_probe_rows} row(s) and {len(probe)} famil(y/ies); "
        + (f"{len(leaked)} also in train/val: {leaked[:5]}" if leaked
           else "none of them seen in train or val"
           if probe else "🔴 PROBE IS EMPTY -- the sealed slice cannot answer "
           "whether VAL has become a training set"))

    # A7 -- every S-a row points at a real VAL file_id.
    sa = df[df["shadow_kind"] == "a"]
    if df["shadow_kind"].isna().all():
        r["A7_shadow_a_points_at_val"] = (
            True, AuditReport.SKIP + "no SHADOW rows in this table")
    else:
        val_ids = set(df.loc[df["slice"] == "val", "file_id"])
        dangling = sorted(str(f) for f in sa["shadow_of"] if f not in val_ids)
        r["A7_shadow_a_points_at_val"] = (
            not dangling,
            f"{len(dangling)}/{len(sa)} S-a re-render(s) whose shadow_of is not a "
            f"VAL file_id" + (f": {dangling[:5]}" if dangling else ""))

    # A8 / A9 -- not computable here, and never reported as a pass.
    r["A8_A9_eval_size_floors"] = (
        True, AuditReport.SKIP + "A8/A9 are statements about compositions and are "
        "evaluated against val_specs.parquet by "
        "training.audit.audit_specs(..., eval_floors=True); a component row has "
        "no cell (docs/validation/04 §VG1)")

    # A10 -- the run's scheme_version matches the table's.
    got = str(df["scheme_version"].iloc[0])
    if run_scheme_version is None:
        r["A10_scheme_version_matches"] = (
            True, AuditReport.SKIP + f"folds.parquet is {got!r}; pass the run's "
            "scheme_version to compare it")
    else:
        r["A10_scheme_version_matches"] = (
            str(run_scheme_version) == got,
            f"run {str(run_scheme_version)!r} vs folds.parquet {got!r}")
    return AuditReport(r)
