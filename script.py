"""The submission entry point (docs/competition/02): ``data/test/`` in,
``output/submission.csv`` out, through the ONE shipped chain
(``processing.infer``).

Three modes. With ``model/scored.pt`` and ``model/processing.json`` beside this
file, the trained model is scored behind the chain it was trained behind.
Without ``model/scored.pt`` but with an ``ensemble`` entry in
``model/members.json``, each ``model/members/<i>/`` is loaded ONE AT A TIME
(bounding GPU memory), scored over every file behind its own chain, and freed;
the members' per-file probabilities are then averaged per column
(``processing.infer.ensemble_probs``; per file, so legal under rule 2.4).
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from processing.config import processing_config_from_dict          # noqa: E402
from processing.infer import (ConstantModel, InferenceReport, ensemble_probs,  # noqa: E402
                              list_test_files, predict_files, write_submission)
from processing.ship import ShipConfig                              # noqa: E402


def _test_dir(args_dir: str | None) -> Path:
    if args_dir:
        return Path(args_dir)
    for cand in (HERE / "data" / "test", HERE / "open" / "test"):
        if cand.is_dir():
            return cand
    raise SystemExit("no data/test or open/test directory beside script.py")


def _predict_ensemble(ens: dict, model_dir: Path, files: list[Path], batch_size: int,
                      device: str) -> InferenceReport:
    """Score each member alone over every file, free it, then average per file."""
    import gc

    import torch
    from models.model import load_checkpoint
    reports, valid = [], []
    for i, member in enumerate(ens["members"]):
        mdir = model_dir / member["dir"]
        weights = mdir / "scored.pt"
        assert weights.stat().st_size > 1_000_000, f"member {i} weights missing/truncated"
        cfg = processing_config_from_dict(
            json.loads((mdir / "processing.json").read_text(encoding="utf-8")))
        model = load_checkpoint(weights, map_location="cpu",
                                weights={k: str(model_dir / v)
                                         for k, v in member["weights"].items()} or None)
        if model.cfg.audio.band_hz is not None:
            raise SystemExit(f"member {i}: model.cfg.audio.band_hz is set: the band limit is "
                             "the shipped chain's, and would be applied twice")
        print(f"member {i}: weights {weights} ({weights.stat().st_size} bytes); "
              f"torch {torch.__version__}", flush=True)
        r = predict_files(model, files, cfg.ship, batch_size=batch_size, device=device)
        failed = {Path(name).stem for name, _ in r.failures}
        print(f"member {i}: {r.n} rows, {r.n_fallback} fallback row(s)", flush=True)
        if reports and r.ids != reports[0].ids:
            raise RuntimeError(f"member {i} scored the files in a different order")
        reports.append(r)
        valid.append([x not in failed for x in r.ids])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probs = ensemble_probs([r.probs for r in reports], ens["rule"], ens.get("weights"), valid)
    report = InferenceReport(ids=list(reports[0].ids),
                             probs={c: [float(x) for x in v] for c, v in probs.items()},
                             failures=[f for r in reports for f in r.failures])
    report.n_fallback = sum(not any(col) for col in zip(*valid))
    print(f"ensemble {ens['rule']} of {len(reports)} member(s), weights {ens.get('weights')}",
          flush=True)
    return report


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
    ens = None
    if not args.probe and not weights.exists() and (model_dir / "members.json").exists():
        ens = json.loads((model_dir / "members.json").read_text(encoding="utf-8")).get("ensemble")

    if ens:
        report = _predict_ensemble(ens, model_dir, files, args.batch_size, args.device)
        require_variation = True
    elif args.probe or not weights.exists():
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

    if not ens:
        report = predict_files(model, files, ship_cfg, batch_size=args.batch_size,
                               device=args.device)
    stack_path = model_dir / "file_stack.json"
    if not args.probe and weights.exists() and stack_path.exists():
        # docs/training/10 F6: a per-file logistic over the five outputs' logits
        from processing import file_stack
        report.probs["FILE_FAKE_PROB"] = [
            float(x) for x in file_stack.apply(report.probs, file_stack.load(stack_path))]
        print(f"FILE_FAKE_PROB from {stack_path.name}", flush=True)
    diag_path = model_dir / "diag_constant.json"
    if not args.probe and weights.exists() and diag_path.exists():
        # a DIAGNOSTIC submission (docs/training/11): the named columns are set to 0.5, so
        # their EER/AUC is exactly 0.5 and the leaderboard score isolates the others
        const = json.loads(diag_path.read_text(encoding="utf-8"))["columns"]
        live = [c for c in report.probs if c not in const]
        spread = min(float(np.std(report.probs[c])) for c in live)
        if not spread > 1e-6:
            raise RuntimeError("a live prediction column is constant: the model did not run")
        for c in const:
            report.probs[c] = [0.5] * report.n
        require_variation = False
        print(f"DIAGNOSTIC: columns {const} set to 0.5", flush=True)
    out = write_submission(report, Path(args.out), require_variation=require_variation)
    print(f"wrote {out}: {report.n} rows, {report.n_fallback} fallback row(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
