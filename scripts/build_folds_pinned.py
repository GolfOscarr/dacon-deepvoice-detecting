#!/usr/bin/env python3
"""Build `folds.parquet` for a grown manifest with every base row's (slice, fold)
kept (processing.pinned_folds; docs/training/11 §1.2). Refuses to write a table
that fails VG1, that moves a base row, or that puts a new row into PROBE.

    $V scripts/build_folds_pinned.py --base-dir .../strategy-v3 --manifest-dir .../strategy-v4
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.pinned_folds import extend_folds_pinned   # noqa: E402
from processing.splits import write_outputs                # noqa: E402
from training.foldcheck import check_split_integrity       # noqa: E402
from training.folds import FoldPlan                        # noqa: E402
from training.manifest import load_manifest                # noqa: E402

#: docs/training/11 §1.2: one cloner per fold, in every language
FAMILY_FOLD = {"fishspeech": 0, "maskgct": 1, "seedvc": 2, "cosyvoice": 3, "chatterbox": 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--family-fold", default=json.dumps(FAMILY_FOLD))
    a = ap.parse_args()
    out = Path(a.manifest_dir)
    if (out / "folds.parquet").exists():
        raise SystemExit(f"FATAL: {out / 'folds.parquet'} exists; never overwrite a fold table")
    manifest = load_manifest(out / "manifest.parquet")
    base = pd.read_parquet(Path(a.base_dir) / "folds.parquet")
    frame, report = extend_folds_pinned(
        manifest, base, json.loads(a.family_fold), a.n_folds,
        assigned_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        scheme_version=str(manifest["scheme_version"].iloc[0]))
    # new rows in PROBE can only be members of a base PROBE atom (e.g. a processed
    # copy of a PROBE file): correct, and never a newly placed atom
    vg1 = check_split_integrity(frame)
    if not vg1.ok:
        raise SystemExit(f"FATAL: VG1 fails:\n{vg1}")
    caveats = (f"pinned to {a.base_dir}: base rows keep (slice, fold); "
               f"new families {report['new_family_atoms']}",)
    report["family_fold"] = json.loads(a.family_fold)
    write_outputs(out, FoldPlan(frame, caveats), vg1, report)
    print(json.dumps(report, indent=1))
    print(f"wrote {out / 'folds.parquet'} ({len(frame)} rows); VG1 ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
