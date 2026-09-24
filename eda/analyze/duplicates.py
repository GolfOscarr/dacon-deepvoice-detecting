"""E1 pass 1: byte-identical files, across sources and across pools.

This check exists because the failure already happened.
`RIRS_NOISES/pointsource_noises/` is MUSAN's `noise/free-sound/` set
redistributed: 88 byte-identical files under the same basenames, carrying
different `file_id` and different `source_name`, so `folds.grouping_atoms`'
transitive closure could not link them and the same recording sat in TRAIN on
the fold where its twin was in VAL. `validate_manifest` never looks at `sha256`,
only at `file_id.duplicated()`, so nothing in the pipeline reported it.

⚠️ **The cascade is the lesson.** Linking them correctly collapsed pool E from 4
groups to 2 and `build_folds` immediately refused -- revealing that the earlier
fold table had been feasible *only because of the leak*. So a duplicate found
here is not a row to delete; it is a grouping fact, and the pool it belongs to
may turn out to be smaller than the fold count requires.

⚠️ Byte identity is pass 1 only. The same recording re-encoded at a different
bitrate is invisible to it, and that is what redistribution across *music*
corpora looks like -- see `cross_source_report`'s note on MUSAN.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["NoHashes", "SUMMARY_KEYS", "assign_dup_groups", "cross_source_report",
           "duplicate_groups", "summary_of"]

#: The scalar half of `cross_source_report`, which is all `eda.gates` reads.
#: Persisted on its own so that a sweep finding **nothing** still leaves
#: evidence that it ran -- otherwise a clean corpus and an un-run sweep are
#: indistinguishable on disk, and the gate would report `na` for a pass. That is
#: the exact failure the tri-state exists to prevent.
SUMMARY_KEYS = ("groups", "rows", "cross_source_groups", "cross_pool_groups")


def summary_of(report: dict) -> dict:
    return {k: int(report[k]) for k in SUMMARY_KEYS}


class NoHashes(RuntimeError):
    """`files.parquet` carries no `sha256` column, so the sweep cannot run.

    Reachable: `ProbeConfig.extractors` can narrow the M tier to `[ffprobe,
    mp3_header]` for a fast first look, and that run writes no hashes. The
    distinction matters because `eda.gates` must then report **`na`** -- "the
    sweep did not run" -- rather than the `pass` that an empty result would
    otherwise produce. A corpus with no duplicates and a corpus nobody hashed
    are not the same finding.
    """


def _hashes(files: pd.DataFrame) -> pd.Series:
    if "sha256" not in files.columns:
        raise NoHashes(
            "no `sha256` column: this table was written by a probe run whose "
            "`extractors` excluded `identity`. E1 needs hashes; re-run "
            "`eda probe` with the full extractor set")
    return files["sha256"]


def duplicate_groups(files: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing a sha256 with at least one other row.

    Caveat: rows whose hash is null (an unreadable file) are excluded rather
    than grouped together. Null is not a value, and grouping every failure into
    one enormous "duplicate" set would bury the real ones.
    """
    have = files[_hashes(files).notna()].copy()
    counts = have.groupby("sha256")["file_id"].transform("size")
    return have[counts > 1].sort_values(["sha256", "file_id"]).reset_index(drop=True)


def assign_dup_groups(files: pd.DataFrame) -> pd.Series:
    """`file_id -> dup_group`, null where a file is unique.

    The column `scripts/build_test_corpus.py` was computing the inputs for and
    hardcoding to `None`. Feeds `folds.grouping_atoms`, whose union-find is what
    actually keeps the twins on the same side of a split.
    """
    dupes = duplicate_groups(files)
    if dupes.empty:
        return pd.Series([None] * len(files), index=files.index, dtype="object")
    codes = {h: f"dup{i:05d}" for i, h in enumerate(sorted(dupes["sha256"].unique()))}
    return _hashes(files).map(codes).astype("object")


def cross_source_report(files: pd.DataFrame) -> dict:
    """Duplicate groups, split by how dangerous they are.

    Three tiers, in increasing order of what they cost:

    * **within one source** -- a publisher shipping the same file twice. Real,
      but `grouping_atoms` already links them by `source_name`.
    * **across sources** -- the RIRS/MUSAN shape. Nothing in the manifest links
      these, and they straddle folds silently.
    * **across pools** -- the same bytes carrying two different labels. This is
      not a split leak, it is a labelling contradiction.

    🔴 The specific thing to look for: MUSAN ships its music partition as
    `music/fma/`, `music/jamendo/` and `music/hd/`, and we separately acquired
    FMA (`fma_small`) and MTG-Jamendo (`raw_30s`). That is the same
    redistribution pattern as the RIRS leak, on the 0.27-weight music head
    instead of on noise. If it does not appear here, it may still appear in
    pass 2 -- MUSAN's copies are re-encoded, and re-encoding defeats a hash.
    """
    dupes = duplicate_groups(files)
    if dupes.empty:
        return {"groups": 0, "rows": 0, "cross_source_groups": 0,
                "cross_pool_groups": 0, "pairs": pd.DataFrame(), "detail": dupes}
    agg = dupes.groupby("sha256").agg(
        rows=("file_id", "size"),
        sources=("source_name", lambda s: tuple(sorted(set(s)))),
        pools=("pool", lambda s: tuple(sorted(set(s)))),
    )
    cross_source = agg[agg["sources"].map(len) > 1]
    cross_pool = agg[agg["pools"].map(len) > 1]
    pairs = (cross_source.groupby("sources")
             .agg(groups=("rows", "size"), rows=("rows", "sum"))
             .reset_index().sort_values("rows", ascending=False))
    return {
        "groups": int(len(agg)),
        "rows": int(len(dupes)),
        "cross_source_groups": int(len(cross_source)),
        "cross_pool_groups": int(len(cross_pool)),
        "pairs": pairs,
        "detail": dupes,
    }
