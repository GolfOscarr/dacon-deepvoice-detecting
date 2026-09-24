"""The submission entry point (docs/competition/02): ``data/test/`` in,
``output/submission.csv`` out, through the ONE shipped chain
(``processing.infer``).

Two modes. With ``model/scored.pt`` and ``model/processing.json`` beside this
file, the trained model is scored behind the chain it was trained behind.
Without a checkpoint -- or with ``--probe`` -- the P0-a contract probe: every
column 0.5, which must score exactly 0.5000 (docs/architecture/03 P0-a). The
probe is how the packaging is verified before any weight exists; it is never
what gets submitted for a score.

Paths resolve relative to this file, with the ``data/test`` -> ``open/test``
fallback the guide's typo makes prudent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from processing.config import processing_config_from_dict          # noqa: E402
from processing.infer import (ConstantModel, list_test_files, predict_files,   # noqa: E402
                              write_submission)
from processing.ship import ShipConfig                              # noqa: E402


def _test_dir(args_dir: str | None) -> Path:
    if args_dir:
        return Path(args_dir)
    for cand in (HERE / "data" / "test", HERE / "open" / "test"):
        if cand.is_dir():
            return cand
    raise SystemExit("no data/test or open/test directory beside script.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--out", default=str(HERE / "output" / "submission.csv"))
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--probe", action="store_true", help="P0-a: every column 0.5")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="default: cuda when torch sees a GPU, else cpu")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.device is None:
        # 🔴 Not CUDA_VISIBLE_DEVICES: a server that does not set it would have
        # run all 1,200 files on CPU and blown the 60-min limit without a word.
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {args.device}", flush=True)

    files = list_test_files(_test_dir(args.test_dir))
    if not files:
        raise SystemExit("no audio files under the test directory")
    model_dir = Path(args.model_dir)
    weights = model_dir / "scored.pt"
    chain = model_dir / "processing.json"

    if args.probe or not weights.exists():
        ship_cfg = ShipConfig()
        model = ConstantModel(0.5, sample_rate=ship_cfg.sample_rate)
        require_variation = False
        print(f"P0-a contract probe: {len(files)} files, every column 0.5", flush=True)
    else:
        import torch
        from models.model import load_checkpoint, shipped_weights
        assert weights.stat().st_size > 1_000_000, "model weights missing/truncated"
        cfg = processing_config_from_dict(json.loads(chain.read_text(encoding="utf-8")))
        ship_cfg = cfg.ship
        # Pretrained frontends read their checkpoint at construction, and the
        # stored paths are the training machine's: model/weights/<frontend>/
        # replaces them (the BEATs .pt; the XLS-R dir with config.json).
        model = load_checkpoint(weights, map_location="cpu",
                                weights=shipped_weights(model_dir))
        if model.cfg.audio.band_hz is not None:
            raise SystemExit("model.cfg.audio.band_hz is set: the band limit is the shipped "
                             "chain's, and would be applied twice")
        require_variation = True
        print(f"{len(files)} files; weights {weights} ({weights.stat().st_size} bytes); "
              f"torch {torch.__version__}", flush=True)

    report = predict_files(model, files, ship_cfg, batch_size=args.batch_size,
                           device=args.device)
    out = write_submission(report, Path(args.out), require_variation=require_variation)
    print(f"wrote {out}: {report.n} rows, {report.n_fallback} fallback row(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
