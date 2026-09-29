#!/usr/bin/env python3
"""Build strategy-v3: strategy-v2's rows + the first run's corpora + `lang`.

    python scripts/extend_manifest.py \
        --v2 /data/project/private/dacon-corpus/manifests/strategy-v2 \
        --out /data/project/private/dacon-corpus/manifests/strategy-v3

See `processing/extend.py`. Writes manifest.parquet and extend_report.json in
--out; never writes into --v2. Folds are a separate step (scripts/build_folds.py).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from processing.extend import NEW_CORPORA, extend_manifest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2", required=True,
                   help="the BASE manifest directory (strategy-v2 for v3; strategy-v3 for v4); "
                        "its corpora are kept as they are and not re-read")
    p.add_argument("--out", required=True, help="the strategy-v3 directory to write")
    p.add_argument("--corpus-root", default="/data/project/private/dacon-corpus")
    p.add_argument("--floor", type=float, default=2.5, help="component_floor_s")
    p.add_argument("--only", help=f"comma-separated subset of "
                                  f"{[c.name for c in NEW_CORPORA]}; '' adds none")
    p.add_argument("--append", default="",
                   help="comma-separated corpora already in the base whose NEW rows (file_id "
                        "not in the base) are added, e.g. emilia-ko for its second YODAS sample")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--scheme", default="strategy-v3",
                   help="scheme_version for every row (strategy-v4 for round 2, docs/training/09)")
    p.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = p.parse_args()

    v2_dir, out = pathlib.Path(args.v2).resolve(), pathlib.Path(args.out).resolve()
    if out == v2_dir:
        raise SystemExit("--out must not be the v2 directory: v2 is never overwritten")
    only = None if args.only is None else tuple(x for x in args.only.split(",") if x)
    v2 = pd.read_parquet(v2_dir / "manifest.parquet")
    m, report = extend_manifest(v2, pathlib.Path(args.corpus_root),
                                component_floor_s=args.floor, only=only,
                                append=tuple(x for x in args.append.split(",") if x),
                                workers=args.workers, scheme_version=args.scheme)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "manifest.parquet", index=False)
    (out / "extend_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(m)} rows -> {out / 'manifest.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
