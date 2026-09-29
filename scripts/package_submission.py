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

XLS-R-1B is shipped PRE-TRUNCATED to the config's ``layers`` (docs/training/14 §5.3):
``config.json`` says ``num_hidden_layers: <layers>`` and the ``.bin`` holds only those
blocks -- 24 of 48 is ~1.9 GB instead of 3.9 GB. The frontend's own truncation is then a
no-op, and every weight is overwritten by the strict load of ``scored.pt`` regardless;
the shipped-copy check at the end of `main` proves the result bitwise.

Several ``--scored`` are averaged with `training.checkpoint.checkpoint_soup`,
which refuses checkpoints of different architectures; their processing.json
files must be identical. requirements.txt is empty on purpose: every package
the code needs is preinstalled on the server, and pinning a different version
is an install error.

``--file-mode`` overrides ``file_head.mode`` in the shipped checkpoint's config
(docs/training/13 O5): inference-only, no weight changes. members.json records
the trained and the shipped mode.

ENSEMBLE (architectures may differ, e.g. XLS-R-1B fusion + XLS-R-300M), a per-file
PREDICTION average (rule 2.4), NOT a soup:

    python scripts/package_submission.py --out submissions/ens \
        --member RUN_A/scored.pt::audio=.../beats,speech=.../xlsr-1b[::RUN_A/processing.json] \
        --member RUN_B/scored.pt::audio=.../beats,speech=.../xlsr-300m \
        [--ens-rule prob_mean|logit_mean] [--ens-weights 0.6,0.4] [--file-mode max3]

    model/members.json                               {"ensemble": {rule, weights, members:
                                                     [{dir, weights: {frontend: dir}}]}, ...}
    model/members/<i>/scored.pt processing.json      each member, --file-mode applied to all
    model/members/<i>/weights/<frontend>/            a frontend identical to an earlier
                                                     member's (same source, kind, depth) is
                                                     not copied twice; members.json points at it

There is no model/scored.pt, so script.py takes its ensemble path: members are
loaded one at a time, each scores every file, and the per-file per-column
probabilities are averaged. --file-stack is refused with --member.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.config import FILE_HEAD_MODES  # noqa: E402
from models.model import (DeepVoiceNet, load_checkpoint, save_checkpoint,  # noqa: E402
                          with_file_mode)
from processing.infer import ENSEMBLE_RULES  # noqa: E402
from training.checkpoint import checkpoint_soup  # noqa: E402

CODE = ("models", "processing", "training", "metrics")
#: what each frontend kind needs on disk to construct
NEEDED = {"beats": ("BEATs_iter3_plus_AS2M.pt",),
          "xlsr_300m": ("config.json", "*.bin", "*.safetensors", "preprocessor_config.json"),
          "xlsr_1b": ("config.json", "*.bin", "*.safetensors", "preprocessor_config.json")}
#: frontends whose snapshot ships cut to the config's `layers` (see the docstring)
TRUNCATE_ON_SHIP = ("xlsr_1b",)
_LAYER_KEY = re.compile(r"(?:^|\.)encoder\.layers\.(\d+)\.")


def ship_truncated_xlsr(src: pathlib.Path, dst: pathlib.Path, layers: int) -> int:
    """Copy a wav2vec2 snapshot keeping only encoder blocks ``< layers``; returns
    the number of files written."""
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if layers > cfg["num_hidden_layers"]:
        raise SystemExit(f"layers={layers} > the snapshot's {cfg['num_hidden_layers']}")
    bins = sorted(src.glob("*.bin"))
    if len(bins) != 1 or list(src.glob("*.safetensors")):
        raise SystemExit(f"{src}: expected one pytorch_model.bin and no safetensors")
    state = torch.load(bins[0], map_location="cpu", weights_only=True)
    kept = {k: v for k, v in state.items()
            if not ((m := _LAYER_KEY.search(k)) and int(m.group(1)) >= layers)}
    torch.save(kept, dst / bins[0].name)
    cfg["num_hidden_layers"] = layers
    (dst / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    shutil.copy2(src / "preprocessor_config.json", dst / "preprocessor_config.json")
    print(f"  {dst.name}: {len(kept)} of {len(state)} tensors ({layers} layers)")
    return 3


def _weights_arg(s: str) -> dict[str, pathlib.Path]:
    out = {}
    for part in s.split(","):
        name, _, path = part.partition("=")
        out[name.strip()] = pathlib.Path(path.strip())
    return out


def _ship_frontends(model: DeepVoiceNet, weights: dict[str, pathlib.Path],
                    dst_root: pathlib.Path, shipped: dict | None = None,
                    rel_to: pathlib.Path | None = None) -> dict[str, str]:
    """Copy what each pretrained frontend needs into ``dst_root/<name>/``.

    With ``shipped`` (a dict shared across ensemble members), a frontend whose
    source, kind and shipped depth match one already written reuses that
    directory instead of a second copy. Returns {frontend: its dir relative to
    ``rel_to``} (default: ``dst_root``'s parent).
    """
    rel_to = rel_to or dst_root.parent
    out = {}
    for name, fe in model.cfg.frontends.items():
        if fe.name == "stub":
            continue
        src = weights.get(name)
        if src is None or not src.is_dir():
            raise SystemExit(f"--weights has no directory for frontend {name!r}")
        truncate = fe.name in TRUNCATE_ON_SHIP and fe.layers is not None
        key = (fe.name, str(src.resolve()), fe.layers if truncate else None)
        if shipped is not None and key in shipped:
            out[name] = shipped[key]
            print(f"  {name}: reusing {shipped[key]}")
            continue
        dst = dst_root / name
        dst.mkdir(parents=True)
        copied = 0
        if truncate:
            copied = ship_truncated_xlsr(src, dst, fe.layers)
        for pattern in (() if copied else NEEDED.get(fe.name, ("*",))):
            for f in src.glob(pattern):
                shutil.copy2(f, dst / f.name)
                copied += 1
        if not copied:
            raise SystemExit(f"nothing copied for frontend {name!r} from {src}")
        out[name] = str(dst.relative_to(rel_to))
        if shipped is not None:
            shipped[key] = out[name]
    return out


def _check_shipped(model: DeepVoiceNet, again: DeepVoiceNet, what: str = "the shipped copy"):
    """``again`` (rebuilt from the package alone) must score and submit bitwise as ``model``."""
    probe = torch.zeros(1, 16000 * 4)
    with torch.no_grad():
        a = model.eval()(probe)
        b = again(probe)
    branches = [k for k in a if not k.startswith("_")]
    assert branches and set(branches) == {k for k in b if not k.startswith("_")}
    for k in branches:
        if not torch.equal(a[k]["clip_logits"], b[k]["clip_logits"]):
            raise SystemExit(f"{what} scores branch {k!r} differently")
    # and submits the same columns: both sides carry the (overridden) file mode
    if again.cfg.file_head.mode != model.cfg.file_head.mode:
        raise SystemExit(f"{what}'s file mode is {again.cfg.file_head.mode!r}, "
                         f"not {model.cfg.file_head.mode!r}")
    pa, pb = model.submission_probs(a), again.submission_probs(b)
    for col in pa:
        if not torch.equal(pa[col], pb[col]):
            raise SystemExit(f"{what} submits {col} differently")


def _member_arg(s: str) -> tuple[pathlib.Path, dict[str, pathlib.Path], pathlib.Path]:
    """``SCORED::name=DIR[,name=DIR][::PROCESSING_JSON]`` (processing.json defaults
    to the one beside SCORED)."""
    parts = s.split("::")
    if len(parts) not in (2, 3):
        raise SystemExit(f"--member {s!r}: expected SCORED::WEIGHTS[::PROCESSING_JSON]")
    scored = pathlib.Path(parts[0]).resolve()
    chain = pathlib.Path(parts[2]).resolve() if len(parts) == 3 else scored.parent / "processing.json"
    return scored, _weights_arg(parts[1]), chain


def _copy_code(out: pathlib.Path) -> None:
    for pkg in CODE:
        shutil.copytree(REPO / pkg, out / pkg,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "AGENTS.md"))
    shutil.copy2(REPO / "script.py", out / "script.py")
    (out / "requirements.txt").write_text("", encoding="utf-8")


def _finish(out: pathlib.Path, n: int, no_zip: bool) -> None:
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 1e9:.2f} GB, {n} checkpoint(s))")
    if not no_zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"zip  {z} ({pathlib.Path(z).stat().st_size / 1e9:.2f} GB)")


def _main_ensemble(args, out: pathlib.Path) -> int:
    """model/members/<i>/{scored.pt,processing.json,members.json}; frontend weights
    under the first member that needs them (deduplicated); model/members.json
    carries the ``ensemble`` entry script.py reads."""
    import gc
    members = [_member_arg(m) for m in args.member]
    ens_w = ([float(x) for x in args.ens_weights.split(",")] if args.ens_weights
             else [1.0] * len(members))
    if len(ens_w) != len(members) or min(ens_w) < 0 or not sum(ens_w) > 0:
        raise SystemExit(f"--ens-weights {args.ens_weights!r} does not fit {len(members)} members")
    for scored, _, chain in members:
        if not scored.is_file() or not chain.is_file():
            raise SystemExit(f"missing {scored} or {chain}")
    model_dir = out / "model"
    model_dir.mkdir(parents=True)
    shipped: dict = {}
    entries, meta_members = [], []
    for i, (scored, weights, chain) in enumerate(members):
        print(f"member {i}: {scored}")
        mdir = model_dir / "members" / str(i)
        mdir.mkdir(parents=True)
        model = load_checkpoint(scored, weights={k: str(v) for k, v in weights.items()})
        trained_mode = model.cfg.file_head.mode
        if args.file_mode is not None:
            model = with_file_mode(model, args.file_mode)
        save_checkpoint(model, mdir / "scored.pt")
        (mdir / "processing.json").write_text(chain.read_text(encoding="utf-8"),
                                              encoding="utf-8")
        info = {"scored": [str(scored)], "processing": str(chain),
                "file_mode": {"trained": trained_mode, "shipped": model.cfg.file_head.mode}}
        (mdir / "members.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        rel = _ship_frontends(model, weights, mdir / "weights", shipped, rel_to=model_dir)
        entries.append({"dir": str(mdir.relative_to(model_dir)), "weights": rel})
        meta_members.append(info)
        again = load_checkpoint(mdir / "scored.pt",
                                weights={k: str(model_dir / v) for k, v in rel.items()} or None)
        _check_shipped(model, again, f"member {i}'s shipped copy")
        del model, again
        gc.collect()
    (model_dir / "members.json").write_text(json.dumps(
        {"ensemble": {"rule": args.ens_rule, "weights": ens_w, "members": entries},
         "members": meta_members}, indent=2), encoding="utf-8")
    _copy_code(out)
    _finish(out, len(members), args.no_zip)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="submission directory (zip beside it)")
    p.add_argument("--scored", action="append",
                   help="a scored.pt; repeat to ship the uniform soup")
    p.add_argument("--weights", help="name=DIR[,name=DIR] per frontend")
    p.add_argument("--member", action="append", default=None,
                   help="SCORED::name=DIR[,name=DIR][::PROCESSING_JSON]; repeat to ship a "
                        "per-file PREDICTION ensemble (architectures may differ); "
                        "excludes --scored/--weights")
    p.add_argument("--ens-rule", choices=ENSEMBLE_RULES, default="prob_mean",
                   help="how --member predictions are averaged per file")
    p.add_argument("--ens-weights", default=None,
                   help="comma-separated member weights (default: equal)")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--file-stack", default=None,
                   help="a processing.file_stack JSON (scripts/diag/fit_file_stack.py); "
                        "shipped as model/file_stack.json, applied by script.py")
    p.add_argument("--file-mode", choices=FILE_HEAD_MODES, default=None,
                   help="override file_head.mode in the shipped config (default: keep the "
                        "checkpoint's own); docs/training/13 O5; with --member, every member")
    args = p.parse_args()
    if args.member:
        if args.scored or args.weights or args.file_stack:
            raise SystemExit("--member excludes --scored, --weights and --file-stack")
    elif not (args.scored and args.weights):
        raise SystemExit("give --scored and --weights, or --member")
    if args.file_stack and args.file_mode not in (None, "learned"):
        raise SystemExit("--file-stack replaces FILE_FAKE_PROB and was fitted on the learned "
                         "FILE head; it cannot be combined with --file-mode "
                         f"{args.file_mode}")

    out = pathlib.Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new --out (submissions are never overwritten)")
    if args.member:
        return _main_ensemble(args, out)
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

    _ship_frontends(model, weights, model_dir / "weights")
    _copy_code(out)

    # the shipped copy must construct from its own directory alone
    from models.model import shipped_weights
    again = load_checkpoint(model_dir / "scored.pt", weights=shipped_weights(model_dir))
    _check_shipped(model, again)
    _finish(out, len(scored), args.no_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
