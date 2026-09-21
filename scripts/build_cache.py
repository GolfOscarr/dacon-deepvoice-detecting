#!/usr/bin/env python3
"""Build the 16 kHz int16 decode cache over the built manifest (docs/processing/03
OFF-6; §6 step 11).

    python -m scripts.build_cache \\
        --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v1

Reads `manifest.parquet`, the render section of the processing config (corpus
root, resampler, sample rate, cache root) and writes one `.npy` per file under
the cache root, plus `cache_report.json`. Resumable: existing files are kept.
Cell 8 (SONICS, 1,973 h, never drawn under D-1) is skipped unless
`--include-cell-8`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.cache import build_cache, write_report          # noqa: E402
from processing.config import load_processing_config            # noqa: E402
from training.manifest import load_manifest                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--processing-config", default="configs/processing_v1.yaml")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--cache-root", default=None, help="default: render.cache_root")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--include-cell-8", action="store_true")
    args = ap.parse_args()
    render = load_processing_config(args.processing_config).render
    cache_root = Path(args.cache_root or render.cache_root)
    manifest = load_manifest(Path(args.manifest_dir) / "manifest.parquet")
    t = time.time()
    report = build_cache(manifest, Path(render.root), cache_root,
                         sample_rate=int(render.audio.sample_rate), resampler=render.resampler,
                         skip_cells=() if args.include_cell_8 else (8,), workers=args.workers,
                         progress=lambda i, n: print(f"  {i}/{n} files, {time.time() - t:.0f}s",
                                                     flush=True))
    report["seconds"] = round(time.time() - t, 1)
    write_report(cache_root, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print(f"{len(report['failures'])} failure(s); report at {cache_root / 'cache_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
