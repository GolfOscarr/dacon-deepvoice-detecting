"""The manifest: one row per source file, and the only thing the pipeline reads
about the corpus.

🔴 Two kinds of row, and everything branches on ``row_kind``
(docs/pipelines/01 §2). A **component** row is drawn and composed; a
**whole_file** row is used as-is. ``cell`` is a property of a *composition*, so
it is null on component rows -- a component file drawn from pool A has no cell
until the sampler has chosen one and drawn the other component.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.spec import CELL_TABLE

__all__ = ["POOLS", "REQUIRED_COLUMNS", "ROW_KINDS", "SLICES",
           "load_manifest", "validate_manifest"]

ROW_KINDS = ("component", "whole_file")
SLICES = ("train", "val", "shadow", "probe")
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

#: A pool's fake status. Real pools have no artifact_family and no domain_key.
POOL_IS_FAKE = {"A": False, "B": True, "C": False, "D": True, "E": False}


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
    bad = set(df.slice.unique()) - set(SLICES)
    if bad:
        raise ValueError(f"unknown slice(s) {sorted(bad)}; expected {SLICES}")

    comp, whole = df.row_kind == "component", df.row_kind == "whole_file"

    # 🔴 Exactly one of pool/cell is meaningful per row kind. This is the
    # invariant the fold-table redesign turned on (docs/validation/01 §3).
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
    # 🔴 Cells 6 and 7 hold one real and one fake component, so they cannot be
    # scraped. Without this, a manifest carrying such a row validates, the fold
    # table is built from it, and the failure only surfaces inside
    # SampleSpec.__post_init__ at draw time -- long after the split is frozen.
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

    # ⚠️ A whole_file row's stored labels must agree with its cell, or the two
    # label sources disagree and the sampler silently picks one.
    for _, row in df.loc[whole].iterrows():
        vp, mp, vf, mf = CELL_TABLE[int(row.cell)]
        got = (row.label_voice_present, row.label_music_present)
        if (int(got[0]), int(got[1])) != (vp, mp):
            raise ValueError(
                f"{row.file_id}: cell {int(row.cell)} implies presence {(vp, mp)}, "
                f"manifest says {tuple(int(g) for g in got)}")

    # Fake components carry an artifact_family (the split key) and a domain_key
    # (the DOSS capping key); real ones carry neither.
    fake_pool = df.pool.map(POOL_IS_FAKE)
    orphan = df[comp & fake_pool.fillna(False) & df.artifact_family.isna()]
    if len(orphan):
        raise ValueError(
            f"{len(orphan)} fake component row(s) without artifact_family, which is "
            f"the split key: {orphan.file_id.head(3).tolist()}")

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
