"""X6: every column assigned a role, with missingness and cardinality.

docs/EDA/06 X6 asks for *"every column in `files.parquet` and `signal.parquet`
assigned to exactly one of **feature** / **split-key** / **leakage-risk**"*. The
artifact it wanted is `eda/out/_shared/roles.md`, and this module computes it.

🔴 **Three roles do not cover the columns, and forcing them would be false.**
Two groups fit none of them and are not edge cases -- they are 40 of the 118:

* the **label** columns. `pool` and `cell` do not *risk* leaking the label, they
  **are** the label: `training.manifest.POOL_LABELS` and `training.spec`'s
  `CELL_TABLE` derive every head from them. Calling `pool` a leakage risk would
  put the target in the same bucket as `bit_rate` and make the document useless
  at the one thing it is for.
* the **diagnostic** columns -- `probe_ok`, `signal_error`, `vad_ok` and their
  kin. They record whether a measurement happened. They are not model inputs,
  not split keys, and cannot leak because they are not about the audio.

So this reports **five** roles and says so. The extension is deliberate and is
the kind of thing X6 asked to have written down rather than decided silently.

⚠️ **`split_key` already implies "would leak if used as a feature", and that
resolves X6's `exactly one` tension.** `source_name` is the archive holdout axis
*and* a perfect label predictor -- those are the same fact, not two. A column
used to build folds is by construction never handed to the model, so `split_key`
is the correct single assignment and `leakage_risk` is reserved for columns that
have no other job.

🔴 **The native plane is `leakage_risk`, all of it.** The model never sees it
([`eda/planes.py`](../planes.py)): the chain plane is what reaches it. A
statistic that separates classes at `native` and not at `chain` is, by
construction, a property of the archive rather than of the audio -- which is
precisely what [06 X1d](../../docs/EDA/06-cross-pool.md) measured. The native
columns stay in the table because R1 needs the pair; they are not candidate
features.
"""

from __future__ import annotations

import re

import pandas as pd

from eda.analyze.shortcut import METADATA_FEATURES
from eda.analyze.signal import CHAIN_EXCLUDED
from eda.planes import CHAIN, NATIVE

__all__ = ["ROLES", "RULES", "UnassignedColumn", "column_roles", "role_summary"]

#: Role -> what it means and what may be done with it.
ROLES = {
    "label": "the target itself, or what derives it. Never an input",
    "split_key": "builds folds. Never an input -- and would leak if it were",
    "feature": "a legitimate model input: chain-plane acoustics",
    "leakage_risk": "predicts the label with no acoustic content. Not an input",
    "diagnostic": "whether a measurement happened. Not audio, not an input",
}


class UnassignedColumn(RuntimeError):
    """A column no rule claims.

    🔴 Fatal rather than defaulted. X6's deliverable is *"no column
    unassigned"*, and a default bucket would silently absorb every column added
    later -- which is exactly the drift this artifact exists to prevent. A new
    column is a decision, and it has to be made here.
    """


def _metadata_columns() -> frozenset[str]:
    """Every column the M-tier audit treats as a feature, flattened.

    Read from `METADATA_FEATURES` rather than relisted: X2 measured these and
    found them worth **less than nothing** on an unseen publisher
    ([06 X2b](../../docs/EDA/06-cross-pool.md)), and a second copy of the list
    could drift out of agreement with the audit that condemned them.
    """
    return frozenset(c for group in METADATA_FEATURES.values() for c in group)


#: Ordered rules. The **first** match wins, so the specific ones come first --
#: a `_chain` suffix must be tested after `nyquist_hz_chain` has been claimed by
#: `CHAIN_EXCLUDED`, or a metadata column wearing an acoustic name becomes a
#: feature.
RULES: tuple[tuple[str, str, str], ...] = (
    # (role, pattern, why)
    ("label", r"^(pool|cell|row_kind|partition)$",
     "derives every head via POOL_LABELS / CELL_TABLE"),
    ("split_key", r"^(source_name|group_key|group_key_kind|sha256|file_id|path)$",
     "the holdout and grouping axes, and the identity they are read from"),
    ("diagnostic", r"(^|_)(ok|error)(_native|_chain)?$",
     "records whether a measurement happened; tri-state, never a feature"),
    # 🔴 Before the generic native/chain rules. `nyquist_hz_chain` is 8000.0 on
    # every row and `nyquist_hz_native` is the published rate over two -- a
    # metadata column wearing an acoustic name, which `eda.analyze.signal`
    # already excludes from the chain audit for the same reason.
    ("leakage_risk", rf"^({'|'.join(CHAIN_EXCLUDED)})_({NATIVE}|{CHAIN})$",
     "the published sample rate in disguise; see signal.CHAIN_EXCLUDED"),
    # 🔴 Duration, and it is the single most consequential row in this table.
    ("leakage_risk", rf"^duration_s_decoded_({NATIVE}|{CHAIN})$",
     "THE corpus shortcut: AUC 0.852 after a whole-publisher holdout, and in "
     "pool D the exact millisecond value is a generator id (02 B6b)"),
    ("leakage_risk", rf"_{NATIVE}$",
     "the model never sees the native plane; separation there is the archive"),
    ("feature", rf"_{CHAIN}$", "chain-plane acoustics: what the model receives"),
    ("feature", r"^vad_(speech_ratio|spans|speech_prob|chunks)",
     "content evidence, measured on the chain plane"),
)


def column_roles(files: pd.DataFrame,
                 signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per column: role, why, missingness, cardinality.

    `signal` is optional so the artifact can be built from the census alone, but
    X6 asks for both tables and the CLI passes both.
    """
    frames = {"files.parquet": files}
    if signal is not None:
        # The join duplicates every census column into the signal table; report
        # each column once, against the table that owns it.
        extra = [c for c in signal.columns if c not in files.columns]
        frames["signal.parquet"] = signal[extra]

    metadata = _metadata_columns()
    rows = []
    for table, frame in frames.items():
        for column in frame.columns:
            role, why = _classify(column, metadata)
            values = frame[column]
            rows.append({
                "table": table, "column": column, "role": role, "why": why,
                "dtype": str(values.dtype),
                "missing": float(values.isna().mean()),
                # `nunique` on 382k object rows is the expensive part of this
                # function and the reason it is not called per partition.
                "cardinality": int(values.nunique(dropna=True)),
            })
    out = pd.DataFrame(rows)
    unassigned = out[out["role"].isna()]
    if len(unassigned):
        raise UnassignedColumn(
            f"{len(unassigned)} column(s) match no rule: "
            f"{sorted(unassigned['column'])}. X6's deliverable is that no "
            f"column is unassigned -- add a rule rather than a default")
    return out.sort_values(["table", "role", "column"]).reset_index(drop=True)


def _classify(column: str, metadata: frozenset[str]) -> tuple[str | None, str]:
    for role, pattern, why in RULES:
        if re.search(pattern, column):
            return role, why
    # ⚠️ After the patterns, not before: a metadata name that also carries a
    # plane suffix has already been claimed, and this catches the bare M-tier
    # columns plus the `_isna` indicators `build_design` derives from them.
    if column in metadata or column.removesuffix("_isna") in metadata:
        return "leakage_risk", (
            "M-tier metadata: AUC 1.000 on music_fake, and 0.0000008 under an "
            "archive holdout (06 X2b)")
    if column in ("codec_long_name", "format_long_name", "probe_error"):
        return "leakage_risk", "the container's own description of the archive"
    return None, ""


def role_summary(roles: pd.DataFrame) -> pd.DataFrame:
    """Columns per role per table -- the shape a reader checks first."""
    out = (roles.groupby(["table", "role"]).size()
           .rename("columns").reset_index())
    total = out.groupby("table")["columns"].transform("sum")
    out["share"] = out["columns"] / total
    return out.sort_values(["table", "columns"], ascending=[True, False]
                           ).reset_index(drop=True)
