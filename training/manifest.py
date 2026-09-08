"""The manifest: one row per source file, and the only thing the pipeline reads
about the corpus.

Critical: two kinds of row, and everything branches on ``row_kind``
(docs/pipelines/01 §2). A **component** row is drawn and composed; a
**whole_file** row is used as-is. ``cell`` is a property of a *composition*, so
it is null on component rows -- a component file drawn from pool A has no cell
until the sampler has chosen one and drawn the other component.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.spec import CELL_TABLE, is_fake_cell

__all__ = ["POOLS", "POOL_LABELS", "REQUIRED_COLUMNS", "ROW_KINDS", "MANIFEST_SLICES",
           "load_manifest", "validate_manifest"]

ROW_KINDS = ("component", "whole_file")
MANIFEST_SLICES = ("train", "val", "shadow", "probe")
POOLS = ("A", "B", "C", "D", "E")          # real/fake voice, real/fake instr., noise

#: docs/pipelines/01 §2. Ordered so a written manifest is diffable.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "file_id", "path", "sha256", "row_kind",
    "pool", "cell",
    "duration_s", "orig_sr", "orig_channels", "container",
    "label_voice_present", "label_music_present",
    "label_voice_fake", "label_music_fake",
    "artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group",
    "domain_key",
    "slice", "fold", "scheme_version",
    "validity_mask_ref", "label_confidence", "aug_strength",
)

#: Which pools can serve which sampler role.
ROLE_POOLS = {"voice": ("A", "B"), "music": ("C", "D"), "noise": ("E",)}

#: A pool's fake status. A real pool has no artifact_family; only a fake pool
#: carries a domain_key, because DOSS caps generator domains.
POOL_IS_FAKE = {"A": False, "B": True, "C": False, "D": True, "E": False}


def _pool_labels() -> dict[str, tuple[int, int, int | None, int | None]]:
    """`pool -> (voice_present, music_present, voice_fake, music_fake)`.

    Derived from ``ROLE_POOLS`` and ``POOL_IS_FAKE`` rather than written out, so
    a pool added to either mapping cannot drift out of this one. Same shape as
    ``training.spec.CELL_TABLE``, and ``None`` means the same thing: the
    component is absent and the metric ignores its fake label.
    """
    out: dict[str, tuple[int, int, int | None, int | None]] = {}
    for role, pools in ROLE_POOLS.items():
        for pool in pools:
            fake = int(POOL_IS_FAKE[pool])
            out[pool] = ((1, 0, fake, None) if role == "voice" else
                         (0, 1, None, fake) if role == "music" else
                         (0, 0, None, None))
    return out


#: What a component row of each pool must carry. Critical: the fake half is
#: load-bearing -- ``folds._group_facts`` builds the per-head artifact-family
#: sets from ``label_voice_fake`` / ``label_music_fake`` and from nothing else.
POOL_LABELS = _pool_labels()

#: The four label columns, in the order ``CELL_TABLE`` and ``POOL_LABELS`` list
#: them.
_LABEL_COLUMNS = ("label_voice_present", "label_music_present",
                  "label_voice_fake", "label_music_fake")


def _disagrees(col: pd.Series, want: int | None) -> pd.Series:
    """Rows of ``col`` that do not carry ``want``. ``None`` means "must be null".

    Caveat: the two-step form is not decoration. ``col != want`` is ``pd.NA``
    wherever ``col`` is null, and a mask carrying NA raises at indexing time --
    so nullness is decided first and the comparison only ever sees real values.
    """
    if want is None:
        return col.notna()
    return (col.isna() | (col != want)).fillna(True).astype(bool)


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the schema, or raise with what is wrong. Returns ``df`` unchanged."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing column(s): {missing}")
    if df.empty:
        raise ValueError("manifest is empty")
    if df.file_id.duplicated().any():
        dup = df.file_id[df.file_id.duplicated()].unique()[:5].tolist()
        raise ValueError(f"duplicate file_id(s): {dup}")

    bad = set(df.row_kind.unique()) - set(ROW_KINDS)
    if bad:
        raise ValueError(f"unknown row_kind(s) {sorted(bad)}; expected {ROW_KINDS}")
    bad = set(df.slice.unique()) - set(MANIFEST_SLICES)
    if bad:
        raise ValueError(f"unknown slice(s) {sorted(bad)}; expected {MANIFEST_SLICES}")

    comp, whole = df.row_kind == "component", df.row_kind == "whole_file"

    # Critical: exactly one of pool/cell is meaningful per row kind. This is
    # the invariant the fold-table redesign turned on (docs/validation/01 §3).
    if df.loc[comp, "pool"].isna().any():
        raise ValueError("component rows must carry a pool (A-E)")
    if df.loc[comp, "cell"].notna().any():
        raise ValueError(
            "component rows must have cell = null -- a component has no cell until "
            "the sampler composes one")
    if df.loc[whole, "cell"].isna().any():
        raise ValueError("whole_file rows must carry a cell (1-9)")
    if df.loc[whole, "pool"].notna().any():
        raise ValueError(
            "whole_file rows must have pool = null -- a whole file is used as-is, "
            "not drawn from a component pool")
    # Critical: cells 6 and 7 hold one real and one fake component, so they
    # cannot be scraped. Without this, a manifest carrying such a row
    # validates, the fold table is built from it, and the failure only surfaces
    # inside SampleSpec.__post_init__ at draw time -- long after the split is
    # frozen.
    bad_cells = sorted({int(c) for c in df.loc[whole, "cell"].dropna()} & {6, 7})
    if bad_cells:
        raise ValueError(
            f"cell(s) {bad_cells} cannot be whole_file rows: they hold one real and "
            f"one fake component, which is why they cannot be scraped")

    pools = set(df.loc[comp, "pool"].unique()) - set(POOLS)
    if pools:
        raise ValueError(f"unknown pool(s) {sorted(pools)}; expected {POOLS}")
    cells = {int(c) for c in df.loc[whole, "cell"].dropna().unique()} - set(CELL_TABLE)
    if cells:
        raise ValueError(f"unknown cell(s) {sorted(cells)}; expected 1-9")

    # Caveat: a whole_file row's stored labels must agree with its cell, or the
    # two label sources disagree and the sampler silently picks one.
    #
    # Critical: all four labels, not just the two presence ones. The fake half
    # was unchecked on both row kinds, and it is not inert:
    # `folds._group_facts` builds the per-head artifact-family sets from
    # `label_voice_fake` / `label_music_fake` and from nothing else. Zeroing
    # `label_voice_fake` across a manifest validated, took the voice family
    # count 24 -> 0, and `build_folds` went on to emit a full table.
    # `audit.py`'s I4 explicitly skips whole_file rows, so nothing downstream
    # closed it either.
    for _, row in df.loc[whole].iterrows():
        want = CELL_TABLE[int(row.cell)]
        got = tuple(row[c] for c in _LABEL_COLUMNS)
        for column, g, w in zip(_LABEL_COLUMNS, got, want):
            null = pd.isna(g)
            if ((not null) if w is None else (null or int(g) != w)):
                raise ValueError(
                    f"{row.file_id}: cell {int(row.cell)} implies "
                    f"{dict(zip(_LABEL_COLUMNS, want))}, manifest says "
                    f"{column} = {None if null else int(g)}"
                    + ("" if column.endswith("present") else
                       " -- the fake labels are the per-head family key that "
                       "folds._group_facts reads"))

    # The same statement for the other row kind: a component row's labels are
    # fixed by its pool, which is the only thing a component has.
    for pool, want in POOL_LABELS.items():
        rows = comp & (df["pool"] == pool)
        if not rows.any():
            continue
        for column, w in zip(_LABEL_COLUMNS, want):
            wrong = df.index[rows & _disagrees(df[column], w)]
            if len(wrong):
                raise ValueError(
                    f"{len(wrong)} pool-{pool} component row(s) whose {column} is "
                    f"not {w}: {df.loc[wrong, 'file_id'].head(3).tolist()}. A "
                    f"component's labels are fixed by its pool "
                    f"(POOL_LABELS[{pool!r}] = {want})")

    # Fake components carry an artifact_family (the split key) and a domain_key
    # (the DOSS capping key).
    fake_pool = df.pool.map(POOL_IS_FAKE)
    is_fake_comp = comp & fake_pool.fillna(False)
    orphan = df[is_fake_comp & df.artifact_family.isna()]
    if len(orphan):
        raise ValueError(
            f"{len(orphan)} fake component row(s) without artifact_family, which is "
            f"the split key: {orphan.file_id.head(3).tolist()}")
    # Critical: a null domain_key does not opt a row out of DOSS, it renames it.
    # `Sampler._doss_weights` does `domain_key.fillna("__real__")`, so every
    # fake component missing one collapses into a single synthetic domain and
    # shares one cap -- the DOSS failure (6.4k h naive 3.29% EER vs 0.2k h
    # balanced 2.77%, docs/papers/05) with a green manifest.
    uncapped = df[is_fake_comp & df.domain_key.isna()]
    if len(uncapped):
        raise ValueError(
            f"{len(uncapped)} fake component row(s) without domain_key, which is "
            f"the DOSS capping key: {uncapped.file_id.head(3).tolist()}. A null "
            f"one is not 'no domain' -- it collapses into the single "
            f"'__real__' bucket and shares one cap with every other null")
    # Critical: and a REAL row must carry no family. `folds.grouping_atoms`
    # unions on `artifact_family` whatever the row's labels say, so a stray one
    # binds that real file into the family's group -- which is a fold
    # assignment, decided before any of the label-aware code runs.
    # Caveat: both row kinds. A real whole_file row has no pool to be judged by,
    # so its fake status comes from its cell.
    real_row = ~(is_fake_comp | (whole & df.cell.map(
        lambda c: pd.notna(c) and is_fake_cell(int(c))).fillna(False)))
    stray = df[real_row & df.artifact_family.notna()]
    if len(stray):
        raise ValueError(
            f"{len(stray)} real row(s) carrying an artifact_family, which is the "
            f"fake-side split key: {stray.file_id.head(3).tolist()}. "
            f"grouping_atoms unions on it regardless of the labels, so this "
            f"binds a real file into a generator family's fold")

    if (df.duration_s <= 0).any():
        raise ValueError("duration_s must be > 0 for every row")
    bad_conf = set(df.label_confidence.unique()) - {"exact", "reported"}
    if bad_conf:
        raise ValueError(f"label_confidence must be exact|reported, got {sorted(bad_conf)}")
    if df.scheme_version.nunique() != 1:
        raise ValueError(
            f"manifest mixes scheme_version {sorted(df.scheme_version.unique())}; "
            "results across versions are never compared")
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read and validate. Parquet keeps the nullable dtypes the schema needs."""
    df = pd.read_parquet(path)
    return validate_manifest(df)
