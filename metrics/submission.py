"""Write and validate `submission.csv`.

Vendored into submit.zip unchanged, so it must import nothing beyond numpy and
pandas -- both preinstalled on the evaluation server.

Two rules govern this module, and they pull against each other:

* EER collapses if ranking resolution is destroyed near the operating point.
  Saturating the top and bottom 80% of a column took a measured EER from
  0.0950 to 0.3017 (docs/validation/02 section 7). So scores must stay
  distinct.
* Rule 2.4 requires every test file to be predicted independently, and names
  "rank calibration" across the eval cohort among the forbidden operations.
  So the obvious fix -- rank-normalising each column -- is unavailable.

The reconciliation: emit float64 with no rounding and no clipping, which keeps
ties to genuinely identical inputs, and assert the result is resolved enough
(VG5). Nothing here looks at more than one row at a time except the assertions,
which are diagnostics over our own output and never feed a prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.dacon import PREDICTION_COLUMNS

__all__ = ["VG5Error", "check_output_sanity", "read_sample_submission",
           "write_submission", "validate_submission"]

#: VG5 B2 -- a column must keep at least this fraction of its values distinct.
MIN_UNIQUE_RATIO = 0.5


class VG5Error(AssertionError):
    """Output sanity gate failed. docs/validation/04-audit-gates.md#vg5."""


def check_output_sanity(df: pd.DataFrame, min_unique_ratio: float = MIN_UNIQUE_RATIO) -> None:
    """VG5 B1-B3, B5: finite, in range, non-constant, and resolved enough."""
    n = len(df)
    if n == 0:
        raise VG5Error("submission is empty")

    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            raise VG5Error(f"{col}: missing")
        v = df[col].to_numpy(dtype=np.float64)

        if not np.isfinite(v).all():                                        # B1, B5
            bad = int((~np.isfinite(v)).sum())
            raise VG5Error(f"{col}: {bad} non-finite value(s)")
        if v.min() < 0.0 or v.max() > 1.0:                                  # B1
            raise VG5Error(f"{col}: outside [0,1] (min {v.min()}, max {v.max()})")

        n_unique = len(np.unique(v))
        if n_unique == 1:                                                   # B3
            raise VG5Error(f"{col}: constant at {v[0]}")
        if n_unique <= min_unique_ratio * n:                                # B2
            raise VG5Error(
                f"{col}: only {n_unique} distinct values over {n} files "
                f"({n_unique / n:.1%} < {min_unique_ratio:.0%}). Ranking resolution "
                "has collapsed -- check for float32 sigmoid saturation, rounding, "
                "or clipping in the export path."
            )


def read_sample_submission(path):
    """Return (ids, columns) from DACON's sample_submission.csv.

    The authority on column order. We have not seen the real file, so nothing in
    this module hardcodes an order it cannot check -- pass the result of this
    function to write_submission and validate_submission and the question does
    not arise.
    """
    df = pd.read_csv(path, encoding="utf-8")
    if df.columns[0] != "ID":
        raise VG5Error(f"sample_submission: first column is {df.columns[0]!r}, expected 'ID'")
    cols = tuple(df.columns[1:])
    unknown = set(cols) ^ set(PREDICTION_COLUMNS)
    if unknown:
        raise VG5Error(
            f"sample_submission columns {list(cols)} do not match the five expected "
            f"prediction columns; symmetric difference {sorted(unknown)}"
        )
    return df["ID"].tolist(), cols


def write_submission(ids, preds, path, *, columns=PREDICTION_COLUMNS,
                     validate: bool = True) -> pd.DataFrame:
    """Write `submission.csv`: an ID column plus the five probabilities.

    `preds` maps each prediction column to a sequence of float scores aligned
    with `ids`. `columns` fixes the output order -- pass the tuple from
    read_sample_submission so the order comes from DACON's own file rather than
    from our assumption about it.

    Values are written as float64 and are never rounded or clipped; doing either
    is how ranking resolution gets lost. pandas' default float writer is already
    round-trip safe (verified 1200/1200 bit-exact), so no float_format is set.
    """
    ids = list(ids)
    columns = tuple(columns)
    if set(columns) != set(PREDICTION_COLUMNS):
        raise KeyError(f"write_submission: columns must be the five prediction columns, got {columns}")
    missing = [c for c in columns if c not in preds]
    if missing:
        raise KeyError(f"write_submission: missing prediction columns {missing}")

    df = pd.DataFrame({"ID": ids})
    for col in columns:
        v = np.asarray(preds[col], dtype=np.float64)
        if v.shape != (len(ids),):
            raise ValueError(f"{col}: expected {len(ids)} values, got {v.shape}")
        df[col] = v

    if validate:
        check_output_sanity(df)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def validate_submission(path, reference_ids=None, reference_columns=None, *,
                        strict_sanity: bool = True) -> pd.DataFrame:
    """Re-read a written submission and check it against the contract.

    `reference_ids` and `reference_columns` come from sample_submission.csv.
    Row count, ID set *and* ID order are checked: the scorer joins on ID, but a
    reordered file is a symptom of a bug worth catching here.

    Read with float_precision="round_trip". pandas' default CSV float parser is
    not correctly rounded and silently perturbs about a third of float64 values
    by ~1e-17; without this the validator would be checking a re-parsed
    approximation rather than what we actually wrote.
    """
    df = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")

    expected = ["ID", *(reference_columns or PREDICTION_COLUMNS)]
    if list(df.columns) != expected:
        raise VG5Error(f"columns are {list(df.columns)}, expected {expected}")

    if df["ID"].duplicated().any():
        dupes = df.loc[df["ID"].duplicated(), "ID"].unique()[:5].tolist()
        raise VG5Error(f"duplicate IDs, e.g. {dupes}")

    if reference_ids is not None:
        ref = list(reference_ids)
        if len(df) != len(ref):
            raise VG5Error(f"{len(df)} rows, expected {len(ref)}")
        if set(df["ID"]) != set(ref):
            missing = sorted(set(ref) - set(df["ID"]))[:5]
            extra = sorted(set(df["ID"]) - set(ref))[:5]
            raise VG5Error(f"ID set mismatch (missing e.g. {missing}, extra e.g. {extra})")
        if list(df["ID"]) != ref:
            raise VG5Error("ID order differs from the reference")

    if strict_sanity:
        check_output_sanity(df)
    return df
