"""Tier-3 diagnostic breakdowns. docs/validation/02 section 1.

Pooled numbers are not reportable on their own: a good pooled EER hides a
collapsed cell, and cells 6 and 7 are the whole reason the competition has two
separate fake heads. These slices never promote an idea -- they decide what to
work on next.

A subtlety that makes this module less trivial than it looks. Some slice keys
are *label-determining*: a cell fixes whether the file is fake (cell 6 is fake
by definition), and so does an artifact family (a generator emits only fakes).
A within-slice EER is then computed on a single class and is undefined. Those
slices are scored against a **shared contrast pool** -- the slice's rows plus
every row of the opposite class in the same masked pool -- which asks the
question we actually care about: how well does this cell's audio rank against
everything of the other label? Label-crossing keys (fold, SNR bucket, duration
bucket) contain both classes and are scored within the slice as usual.

Every row records which contrast was used and the pool size behind it, because
a per-family EER over 40 files is not evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.dacon import EmptyPoolError, eer

__all__ = ["HEAD_POOLS", "by", "breakdown_table", "t3_gap", "worst_cell_eer",
           "worst_family_eer"]

#: head -> (prediction column, truth column, ground-truth presence mask column)
HEAD_POOLS = {
    "file": ("FILE_FAKE_PROB", "file_fake", None),
    "voice": ("VOICE_FAKE_PROB", "voice_fake", "voice_present"),
    "music": ("MUSIC_FAKE_PROB", "music_fake", "music_present"),
}


def _pool(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """The head's masked pool, reduced to score and truth."""
    score_col, truth_col, mask_col = HEAD_POOLS[head]
    sub = df if mask_col is None else df[df[mask_col].astype(bool)]
    return pd.DataFrame({
        "_score": sub[score_col].to_numpy(dtype=np.float64),
        "_truth": sub[truth_col].to_numpy().astype(int),
    }, index=sub.index)


def _safe_eer(pool: pd.DataFrame):
    """EER, or NaN with a reason -- a thin slice must not abort the whole table."""
    if not len(pool):
        return float("nan"), "empty pool"
    try:
        return eer(pool["_truth"].to_numpy(), pool["_score"].to_numpy()), ""
    except EmptyPoolError as exc:
        return float("nan"), str(exc)


def by(df: pd.DataFrame, key: str, head: str = "file", min_n: int = 100,
       contrast: str = "auto") -> pd.DataFrame:
    """EER of one head, sliced by `key`.

    contrast:
        "auto"   -- within-slice where the slice has both classes, shared
                    contrast pool where it does not. The default, and the only
                    one that works across both kinds of key.
        "within" -- always within-slice; raises nothing but yields NaN for
                    label-determining keys.
        "shared" -- always against the full opposite-class pool. Use when
                    comparing slices to each other on a common footing.

    Rows below `min_n` are kept but flagged, so a reader sees the thin slice
    rather than silently trusting or silently losing it.
    """
    if key not in df.columns:
        raise KeyError(f"by: no column {key!r}")
    if contrast not in ("auto", "within", "shared"):
        raise ValueError(f"by: unknown contrast {contrast!r}")

    full = _pool(df, head)
    rows = []
    for value, group in df.groupby(key, dropna=False, sort=True):
        slice_pool = _pool(group, head)
        classes = set(slice_pool["_truth"].unique().tolist())
        use_shared = contrast == "shared" or (contrast == "auto" and len(classes) < 2)

        if use_shared and len(classes) == 1:
            other = full[full["_truth"] != classes.pop()]
            pool = pd.concat([slice_pool, other])
            mode = "shared"
        elif use_shared:                                   # contrast="shared", mixed slice
            pool, mode = full, "shared"
        else:
            pool, mode = slice_pool, "within"

        value_eer, note = _safe_eer(pool)
        rows.append({
            key: value, "head": head, "eer": value_eer, "contrast": mode,
            "n_slice": len(slice_pool),
            "n_slice_fake": int(slice_pool["_truth"].sum()) if len(slice_pool) else 0,
            "n_pool": len(pool),
            "thin": len(slice_pool) < min_n,
            "note": note,
        })
    return pd.DataFrame(rows).sort_values("eer", ascending=False, na_position="first")


def breakdown_table(df: pd.DataFrame, keys=("cell", "artifact_family", "fold"),
                    heads=("file", "voice", "music"), min_n: int = 100) -> pd.DataFrame:
    """Every (key, head) slice in one long frame, ready for the experiment ledger."""
    out = []
    for key in keys:
        if key not in df.columns:
            continue
        for head in heads:
            part = by(df, key, head=head, min_n=min_n).rename(columns={key: "value"})
            out.append(part.assign(key=key))
    if not out:
        raise KeyError(f"breakdown_table: none of {list(keys)} are columns")
    cols = ["key", "value", "head", "eer", "contrast", "n_slice", "n_slice_fake",
            "n_pool", "thin", "note"]
    return pd.concat(out, ignore_index=True)[cols]


def _worst(df, key, head, min_n):
    table = by(df, key, head=head, min_n=min_n)
    usable = table[~table["thin"] & table["eer"].notna()]
    return float(usable["eer"].max()) if len(usable) else float("nan")


def worst_cell_eer(df: pd.DataFrame, head: str = "file", min_n: int = 100) -> float:
    """P2 input: the worst cell, ignoring slices too thin to mean anything."""
    return _worst(df, "cell", head, min_n)


def worst_family_eer(df: pd.DataFrame, head: str = "file", min_n: int = 100) -> float:
    """P3 input: the worst artifact family."""
    return _worst(df, "artifact_family", head, min_n)


def t3_gap(df: pd.DataFrame, head: str = "voice", pair_col: str = "pair_id") -> dict:
    """VG4: EER on T3 matched pairs minus pooled EER on the same fold.

    T3 pairs are the only slice with a matched control -- the same utterance,
    real and resynthesised, so both sides share a recording chain. A pooled EER
    much better than the T3-pair EER means the pooled number is carried by
    corpus identity rather than by artifact detection.

    Gate: gap <= 0.10 (docs/validation/04-audit-gates.md#vg4).
    """
    if pair_col not in df.columns:
        raise KeyError(f"t3_gap: no column {pair_col!r}")

    pooled, pooled_note = _safe_eer(_pool(df, head))
    paired_rows = df[df[pair_col].notna()]
    paired, paired_note = _safe_eer(_pool(paired_rows, head))

    gap = float(paired - pooled) if np.isfinite(paired) and np.isfinite(pooled) else float("nan")
    return {
        "head": head, "pooled_eer": pooled, "t3_pair_eer": paired, "t3_gap": gap,
        "n_pooled": int(len(_pool(df, head))), "n_pairs": int(len(_pool(paired_rows, head))),
        "passes_vg4": bool(np.isfinite(gap) and gap <= 0.10),
        "note": "; ".join(n for n in (pooled_note, paired_note) if n),
    }
