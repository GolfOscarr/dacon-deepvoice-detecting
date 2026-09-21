#!/usr/bin/env python3
"""Build `folds.parquet` over the built manifest (docs/processing/03 OFF-5; §6
step 10).

    python -m scripts.build_folds \\
        --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v1

Reads `manifest.parquet` from the directory and the `folds:` section of the
processing config; writes `folds.parquet`, `folds.caveats.txt`, `folds.vg1.txt`
and `folds_report.json` beside it. Refuses to write a table that fails VG1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.config import load_processing_config          # noqa: E402
from processing.splits import build_and_check, write_outputs   # noqa: E402
from training.manifest import load_manifest                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--processing-config", default="configs/processing_v1.yaml")
    ap.add_argument("--manifest-dir", required=True)
    args = ap.parse_args()
    cfg = load_processing_config(args.processing_config).folds
    out = Path(args.manifest_dir)
    manifest = load_manifest(out / "manifest.parquet")
    plan, report, summary = build_and_check(manifest, cfg)
    write_outputs(out, plan, report, summary)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'folds.parquet'} ({len(plan.frame)} rows); "
          f"VG1: {'ok' if report.ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
