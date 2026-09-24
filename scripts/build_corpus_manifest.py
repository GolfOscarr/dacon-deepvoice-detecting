#!/usr/bin/env python3
"""Build the real corpus's manifest from the EDA's tables (docs/processing/03
OFF-1..OFF-4; §6 step 8).

    python -m scripts.build_corpus_manifest \\
        --eda-config configs/eda.yaml --processing-config configs/processing_v1.yaml \\
        --out /data/project/private/dacon-corpus/manifests/strategy-v1

Reads: every partition's `files.parquet` (the M tier), the S tier plus
`vad_extra.parquet` (usable-span evidence), `_shared/reassignment_worklist.parquet`
(D-14), `_shared/duplicates.parquet` (OFF-3), the fma licence ledger and
`tracks.csv` (artist ids). Writes `manifest.parquet` (validated, every §4.1
rule asserted), `verdict.parquet` (one row per file and filter) and
`build_report.json`. Successor of `scripts/build_test_corpus.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda.config import load_eda_config                       # noqa: E402
from processing.config import load_processing_config          # noqa: E402
from processing.corpus import build_manifest, load_inputs, write_outputs   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eda-config", default="configs/eda.yaml")
    ap.add_argument("--processing-config", default="configs/processing_v1.yaml")
    ap.add_argument("--out", required=True, help="directory for manifest.parquet and its sidecars")
    args = ap.parse_args()

    eda_cfg = load_eda_config(args.eda_config)
    draw = load_processing_config(args.processing_config).draw
    corpus_root = Path(eda_cfg.root).parent
    inp = load_inputs(Path(eda_cfg.out), corpus_root, eda_cfg, draw.component_floor_s)
    print(f"M tier: {len(inp.files)} rows over {inp.files['source_name'].nunique()} sources; "
          f"S tier + vad_extra: {len(inp.signal) if inp.signal is not None else 0} rows; "
          f"worklist: {len(inp.worklist)}; duplicates: {len(inp.duplicates)}")
    manifest, side, report = build_manifest(inp)
    write_outputs(Path(args.out), manifest, side, report)
    print(json.dumps(report, indent=2))
    print(f"wrote {Path(args.out) / 'manifest.parquet'} ({len(manifest)} rows), "
          f"verdict.parquet ({len(side)} rows), build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
