"""X6 / A3 / B2 / C3 / E4: where the grouping key lives, and how many there are.

docs/data/08 records this as *the one Phase-B decision with no second chance*:
`source_name` at corpus granularity collapses every speaker, artist and
generator into one fold group. The smoke corpus proved the consequence -- a
manifest with corpus-granularity `source_name` **validates clean** and then makes
`build_folds` infeasible at every fold count, and for generated audio the
granularity cannot be retrofitted at all.

The information exists, but it exists in the **publisher's directory layout**:
Common Voice's `client_id`, LibriTTS-R's and Zeroth's speaker directories,
MLAAD's `fake/<language>/<generator>/`, FMA's and Jamendo's artist/album ids. It
is destroyed the moment files are flattened -- which is exactly what the MLAAD
acquisition nearly did.

So rather than hardcode one rule per source, this profiles the **cardinality at
each path depth** and lets the reader see where the key is. A depth whose
distinct count is ~105 in Zeroth-Korean is its 105 speakers; a depth that is 1 is
a wrapper directory; a depth equal to the file count is the leaf.
"""

from __future__ import annotations

import pandas as pd

from eda.config import AnalysisConfig, GateConfig
from eda.ids import relpath_of

__all__ = ["depth_profile", "grouping_report", "path_parts"]


def path_parts(files: pd.DataFrame) -> pd.Series:
    """`file_id` -> the tuple of directory components below the source root."""
    return files["file_id"].map(
        lambda fid: tuple(relpath_of(fid).split("/")[:-1]))


def depth_profile(files: pd.DataFrame, cfg: AnalysisConfig | None = None
                  ) -> pd.DataFrame:
    """Per source and per depth: distinct values, and distinct path prefixes.

    Two columns because they answer different questions. `distinct_values` at
    depth 2 counts how many different names appear there anywhere;
    `distinct_prefixes` counts how many distinct paths reach that depth. They
    differ when a name is reused under several parents -- `train/spk1` and
    `test/spk1` -- and that difference is precisely the trap: grouping on the
    *name* would merge two speakers that the publisher kept apart, or split one
    the publisher kept together.
    """
    max_depth = (cfg or AnalysisConfig()).max_depth
    parts = path_parts(files)
    rows = []
    for source, idx in files.groupby("source_name").groups.items():
        sub = parts.loc[idx]
        n_files = len(sub)
        depth = int(sub.map(len).max() or 0)
        for d in range(min(depth, max_depth)):
            at = sub[sub.map(len) > d]
            rows.append({
                "source_name": source,
                "depth": d,
                "files": n_files,
                "files_at_depth": int(len(at)),
                "distinct_values": int(at.map(lambda p: p[d]).nunique()),
                "distinct_prefixes": int(at.map(lambda p: p[:d + 1]).nunique()),
                "example": at.map(lambda p: p[d]).iloc[0] if len(at) else None,
            })
        if depth == 0:
            rows.append({"source_name": source, "depth": 0, "files": n_files,
                         "files_at_depth": n_files, "distinct_values": 0,
                         "distinct_prefixes": 0, "example": None})
    return pd.DataFrame(rows)


def grouping_report(files: pd.DataFrame, cfg: AnalysisConfig | None = None,
                    gates: GateConfig | None = None) -> pd.DataFrame:
    """Per source, the depth that is the most plausible grouping atom.

    The heuristic: the shallowest depth whose `distinct_prefixes` is at least
    `min_groups` and is not the leaf. It is a **suggestion**, and the column
    `candidate_reason` says so -- a real assignment needs the publisher's own
    key (Common Voice's `client_id` lives in `validated.tsv`, not in the path),
    and docs/EDA/01 A3 and 03 C3 name the per-source source of truth.

    ⚠️ A source reported as having enough groups by path depth may still have
    **one** grouping atom in fact. LJSpeech is 13,100 utterances from a single
    speaker; WaveFake is ~117 k files from that same single speaker re-vocoded.
    Neither fact is visible in a directory tree, and both are decisive in a fold
    table.
    """
    min_groups = (gates or GateConfig()).min_groups_per_role
    prof = depth_profile(files, cfg)
    out = []
    for source, g in prof.groupby("source_name"):
        g = g.sort_values("depth")
        n_files = int(g["files"].iloc[0])
        usable = g[(g["distinct_prefixes"] >= min_groups)
                   & (g["distinct_prefixes"] < n_files)]
        if len(usable):
            pick = usable.iloc[0]
            reason = f"shallowest depth with >= {min_groups} distinct prefixes"
        else:
            pick = g.iloc[-1] if len(g) else None
            reason = "no depth reaches the floor -- needs the publisher's own key"
        out.append({
            "source_name": source,
            "files": n_files,
            "max_depth": int(g["depth"].max()) + 1 if len(g) else 0,
            "candidate_depth": None if pick is None else int(pick["depth"]),
            "candidate_groups": None if pick is None else int(pick["distinct_prefixes"]),
            "meets_floor": bool(pick is not None
                                and pick["distinct_prefixes"] >= min_groups
                                and pick["distinct_prefixes"] < n_files),
            "candidate_reason": reason,
        })
    return pd.DataFrame(out).sort_values("source_name").reset_index(drop=True)
