"""The `folds.parquet` schema, and the VG1 assertions over a built table.

Split out of `training.folds` so the two directions stay apart: that module
*builds* a table, this one *judges* one. A checker that imports the builder it
is meant to catch out is one refactor away from asserting the builder's own
arithmetic back at itself, and every assertion here is a statement about the
emitted table alone -- it never reads a `FoldConfig`, a group or a corpus.

`training.folds` re-exports all four names, so `from training.folds import
check_split_integrity` keeps working; this is where they live.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.audit import AuditReport

__all__ = ["FOLD_COLUMNS", "FOLD_SLICES",
           "check_split_integrity", "load_folds", "validate_folds"]

#: docs/validation/01 §3, in the order the spec lists them.
FOLD_COLUMNS: tuple[str, ...] = (
    "file_id", "row_kind", "slice", "shadow_kind", "shadow_of", "fold",
    "artifact_family", "source_name", "speaker_ref_id", "pair_id", "dup_group",
    "cell", "domain_key", "assigned_at", "scheme_version",
)

#: What the emitted `slice` column holds. `train_val` rotates; `probe` is sealed.
FOLD_SLICES: tuple[str, ...] = ("train_val", "shadow", "probe")


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
    bad = set(df["slice"].unique()) - set(FOLD_SLICES)
    if bad:
        raise ValueError(f"unknown slice(s) {sorted(bad)}; expected {FOLD_SLICES}")
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
    # Critical: `fold` is non-null exactly off PROBE. A sealed family has no
    # fold: it is never VAL, so "the fold it is validated in" does not exist
    # for it.
    rotating = df["slice"] != "probe"
    if df.loc[~rotating, "fold"].notna().any():
        raise ValueError("fold must be null on probe rows -- PROBE never rotates")
    if df.loc[rotating, "fold"].isna().any():
        raise ValueError("every train_val/shadow row needs a fold")
    if rotating.any():
        used = sorted(int(f) for f in df.loc[rotating, "fold"].unique())
        if used != list(range(len(used))):
            raise ValueError(f"folds must be 0..k-1 with none empty, got {used}")
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


# --------------------------------------------------------------------------- #
# VG1

def _cells(df: pd.DataFrame, key: str) -> dict[str, set]:
    """Each value of `key` -> the set of (slice, fold) cells it appears in.

    SHADOW folds into `train_val`: it draws from VAL's families by construction,
    shares their fold, and is never trained on.
    """
    sub = df[df[key].notna()]
    out: dict[str, set] = {}
    for value, cell in zip(sub[key].astype(str),
                           zip(np.where(sub["slice"] == "probe", "probe", "train_val"),
                               [None if pd.isna(f) else int(f) for f in sub["fold"]])):
        out.setdefault(value, set()).add(cell)
    return out


def _split_across_cells(df: pd.DataFrame, key: str) -> list[str]:
    """Values of `key` that resolve to more than one (slice, fold) cell.

    Critical: that is the whole safety property of the rotation. A family in
    two folds is VAL in one and TRAIN in the other -- which means that in
    *each* of those folds it is on both sides at once.
    """
    return sorted(v for v, cells in _cells(df, key).items() if len(cells) > 1)


def check_split_integrity(folds: pd.DataFrame,
                          run_scheme_version: str | None = None) -> AuditReport:
    """VG1 A1-A7 and A10 over `folds.parquet` (docs/validation/04 §VG1).

    Under the rotating scheme every assertion is the same statement about a
    different key: the value must resolve to exactly one `(slice, fold)` cell.
    Two cells means two roles in the same fold -- TRAIN and VAL simultaneously,
    or sealed and not sealed at once.

    Critical: A8/A9 are **not** here and are not silently passed. They are
    statements about compositions, evaluated against `val_specs.parquet` by
    `training.audit.audit_specs(..., eval_floors=True)`; a component row has no
    cell, so they cannot be computed on this table at all.
    """
    validate_folds(folds)
    df = folds
    r: dict[str, tuple[bool, str]] = {}

    # A1 -- a family is sealed into PROBE or rotates in exactly one fold.
    split = _split_across_cells(df, "artifact_family")
    n_fam = int(df["artifact_family"].nunique())
    r["A1_family_in_one_cell"] = (
        not split,
        f"{len(split)}/{n_fam} artifact_famil(y/ies) appear in more than one of "
        f"{{probe}} ∪ {{fold 0..k-1}}" + (f": {split[:5]}" if split else "")
        + " -- i.e. TRAIN and VAL in the same fold, or sealed and rotating at "
          "once (SHADOW counts as its parent's fold)")

    # A2 / A3 -- source and speaker never straddle the TRAIN/VAL boundary.
    for key, name in (("source_name", "A2_source_disjoint"),
                      ("speaker_ref_id", "A3_speaker_disjoint")):
        split = _split_across_cells(df, key)
        r[name] = (not split,
                   f"{len(split)} {key}(s) span two folds or straddle PROBE, so "
                   f"they are on both sides of some fold's TRAIN/VAL boundary"
                   + (f": {split[:5]}" if split else ""))
    # Caveat: A3 keys on the **bare** speaker_ref_id, not on (source_name,
    # speaker). docs/validation/01 §1 says "disjoint within source_name", but
    # A2 already pins a source_name to one cell, so the within-source form is
    # entailed by A2 and could never fail on its own. The bare id is the
    # falsifiable statement, and it is the one that matters: a LibriTTS speaker
    # reappearing under a `gen_hifigan` resynthesis is exactly the leak.

    # A4 / A5 -- pair members share a cell; a dup_group is wholly within one.
    for key, name in (("pair_id", "A4_pair_shares_a_cell"),
                      ("dup_group", "A5_dup_group_in_one_cell")):
        split = _split_across_cells(df, key)
        r[name] = (not split,
                   f"{len(split)} {key}(s) split across folds or across the PROBE "
                   f"seal" + (f": {split[:5]}" if split else ""))

    # A6 -- PROBE is sealed, and PROBE exists. Caveat: the "seen in neither"
    # clause is entailed by A1. The clause that is not, and the one that fails
    # in practice, is that PROBE exists at all: a builder that carved none
    # would leave A1 green over an empty slice (docs/validation/01 §3). Under
    # rotation this is the main thing A6 carries.
    probe_fams = set(df.loc[df["slice"] == "probe", "artifact_family"].dropna())
    rot_fams = set(df.loc[df["slice"] != "probe", "artifact_family"].dropna())
    leaked = sorted(probe_fams & rot_fams)
    n_probe_rows = int((df["slice"] == "probe").sum())
    r["A6_probe_sealed"] = (
        bool(probe_fams) and not leaked,
        f"PROBE holds {n_probe_rows} row(s) and {len(probe_fams)} famil(y/ies); "
        + (f"{len(leaked)} also rotate: {leaked[:5]}" if leaked
           else "none of them rotate through TRAIN or VAL" if probe_fams
           else "Critical: PROBE IS EMPTY -- the sealed slice cannot answer whether VAL "
                "has become a training set"))

    # A7 -- every S-a row points at a real file in its own fold.
    sa = df[df["shadow_kind"] == "a"]
    if df["shadow_kind"].isna().all():
        r["A7_shadow_a_points_at_its_fold"] = (
            True, AuditReport.SKIP + "no SHADOW rows in this table")
    else:
        fold_of = df.set_index("file_id")["fold"]
        rotating_ids = set(df.loc[df["slice"] == "train_val", "file_id"])
        bad = sorted(
            f"{row.file_id}->{row.shadow_of}" for row in sa.itertuples()
            if row.shadow_of not in rotating_ids
            or fold_of.get(row.shadow_of) is pd.NA
            or fold_of[row.shadow_of] != row.fold)
        r["A7_shadow_a_points_at_its_fold"] = (
            not bad,
            f"{len(bad)}/{len(sa)} S-a re-render(s) whose shadow_of is not a "
            f"rotating file in the same fold" + (f": {bad[:5]}" if bad else ""))

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
