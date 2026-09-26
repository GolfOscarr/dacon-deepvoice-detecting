#!/usr/bin/env python3
"""Assemble a submission directory and zip (docs/competition/02).

    python scripts/package_submission.py --out submissions/first \
        --scored runs/first/all_data_seed0/scored.pt [--scored ... more to soup] \
        --weights audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m \
        [--file-mode max3]

Layout written (and zipped to <out>.zip):

    script.py  requirements.txt
    models/ processing/ training/ metrics/          the code script.py imports
    model/scored.pt                                  one checkpoint, or the uniform soup
    model/processing.json                            the chain the weights trained behind
    model/weights/<frontend>/                        what each pretrained frontend needs to
                                                     construct offline (models.model.shipped_weights)

Several ``--scored`` are averaged with `training.checkpoint.checkpoint_soup`,
which refuses checkpoints of different architectures; their processing.json
files must be identical. requirements.txt is empty on purpose: every package
the code needs is preinstalled on the server, and pinning a different version
is an install error.

``--file-mode`` overrides ``file_head.mode`` in the shipped checkpoint's config
(docs/training/13 O5): inference-only, no weight changes. members.json records
the trained and the shipped mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.config import FILE_HEAD_MODES, validate_model_config  # noqa: E402
from models.model import DeepVoiceNet, load_checkpoint, save_checkpoint  # noqa: E402
from training.checkpoint import checkpoint_soup  # noqa: E402

CODE = ("models", "processing", "training", "metrics")
#: what each frontend kind needs on disk to construct
NEEDED = {"beats": ("BEATs_iter3_plus_AS2M.pt",),
          "xlsr_300m": ("config.json", "*.bin", "*.safetensors", "preprocessor_config.json")}


def _weights_arg(s: str) -> dict[str, pathlib.Path]:
    out = {}
    for part in s.split(","):
        name, _, path = part.partition("=")
        out[name.strip()] = pathlib.Path(path.strip())
    return out


def with_file_mode(model: DeepVoiceNet, mode: str) -> DeepVoiceNet:
    """The same module with ``cfg.file_head.mode`` replaced (the config is frozen).

    Only `submission_probs` reads the mode, so no parameter or buffer changes.
    """
    cfg = dataclasses.replace(
        model.cfg, file_head=dataclasses.replace(model.cfg.file_head, mode=mode))
    validate_model_config(cfg)
    model.cfg = cfg
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="submission directory (zip beside it)")
    p.add_argument("--scored", action="append", required=True,
                   help="a scored.pt; repeat to ship the uniform soup")
    p.add_argument("--weights", required=True, help="name=DIR[,name=DIR] per frontend")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--file-stack", default=None,
                   help="a processing.file_stack JSON (scripts/diag/fit_file_stack.py); "
                        "shipped as model/file_stack.json, applied by script.py")
    p.add_argument("--file-mode", choices=FILE_HEAD_MODES, default=None,
                   help="override file_head.mode in the shipped config (default: keep the "
                        "checkpoint's own); docs/training/13 O5")
    args = p.parse_args()
    if args.file_stack and args.file_mode not in (None, "learned"):
        raise SystemExit("--file-stack replaces FILE_FAKE_PROB and was fitted on the learned "
                         "FILE head; it cannot be combined with --file-mode "
                         f"{args.file_mode}")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --out (submissions are never overwritten)")
    scored = [pathlib.Path(s).resolve() for s in args.scored]
    chains = {(s.parent / "processing.json").read_text(encoding="utf-8") for s in scored}
    if len(chains) != 1:
        raise SystemExit("the --scored runs trained behind different processing.json files")
    weights = _weights_arg(args.weights)

    model_dir = out / "model"
    model_dir.mkdir(parents=True)
    model = load_checkpoint(scored[0], weights={k: str(v) for k, v in weights.items()})
    if len(scored) > 1:
        model.load_state_dict(checkpoint_soup([str(s) for s in scored]), strict=True)
    trained_mode = model.cfg.file_head.mode
    if args.file_mode is not None:
        model = with_file_mode(model, args.file_mode)
    save_checkpoint(model, model_dir / "scored.pt")
    (model_dir / "processing.json").write_text(chains.pop(), encoding="utf-8")
    if args.file_stack:
        from processing import file_stack
        stack = file_stack.load(pathlib.Path(args.file_stack))
        if tuple(stack["features"]) != file_stack.FEATURES:
            raise SystemExit(f"--file-stack features {stack['features']} are stale")
        shutil.copy2(args.file_stack, model_dir / "file_stack.json")
    (model_dir / "members.json").write_text(json.dumps(
        {"scored": [str(s) for s in scored],
         "file_mode": {"trained": trained_mode, "shipped": model.cfg.file_head.mode}},
        indent=2), encoding="utf-8")

    for name, fe in model.cfg.frontends.items():
        if fe.name == "stub":
            continue
        src = weights.get(name)
        if src is None or not src.is_dir():
            raise SystemExit(f"--weights has no directory for frontend {name!r}")
        dst = model_dir / "weights" / name
        dst.mkdir(parents=True)
        copied = 0
        for pattern in NEEDED.get(fe.name, ("*",)):
            for f in src.glob(pattern):
                shutil.copy2(f, dst / f.name)
                copied += 1
        if not copied:
            raise SystemExit(f"nothing copied for frontend {name!r} from {src}")

    for pkg in CODE:
        shutil.copytree(REPO / pkg, out / pkg,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "AGENTS.md"))
    shutil.copy2(REPO / "script.py", out / "script.py")
    (out / "requirements.txt").write_text("", encoding="utf-8")

    # the shipped copy must construct from its own directory alone
    from models.model import shipped_weights
    again = load_checkpoint(model_dir / "scored.pt", weights=shipped_weights(model_dir))
    probe = torch.zeros(1, 16000 * 4)
    with torch.no_grad():
        a = model.eval()(probe)
        b = again(probe)
    branches = [k for k in a if not k.startswith("_")]
    assert branches and set(branches) == {k for k in b if not k.startswith("_")}
    for k in branches:
        if not torch.equal(a[k]["clip_logits"], b[k]["clip_logits"]):
            raise SystemExit(f"the shipped copy scores branch {k!r} differently")
    # and submits the same columns: both sides carry the (overridden) file mode
    if again.cfg.file_head.mode != model.cfg.file_head.mode:
        raise SystemExit(f"the shipped copy's file mode is {again.cfg.file_head.mode!r}, "
                         f"not {model.cfg.file_head.mode!r}")
    pa, pb = model.submission_probs(a), again.submission_probs(b)
    for col in pa:
        if not torch.equal(pa[col], pb[col]):
            raise SystemExit(f"the shipped copy submits {col} differently")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 1e9:.2f} GB, {len(scored)} checkpoint(s))")
    if not args.no_zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip  {z} ({pathlib.Path(z).stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
