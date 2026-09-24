"""I10 across processes, I13 and I14 on real corpus samples
(docs/processing/03 §7 item 6).

    python scripts/strategy/verify_reproducibility.py --n 64 --fold 0

* I10: ``render(spec) == render(spec)`` bitwise -- the same specs rendered in
  two FRESH interpreters (``subprocess``), compared by the sha256 of the
  waveform bytes and of the frame intervals. Same-process equality would not
  catch a salted ``hash()`` or a process-seeded generator.
* I13: the frame intervals a real render returns are the placement of its
  non-layer components, merged across tiles, labelled from the cell.
* I14: the shipped chain over a batch of real rendered samples, padded to the
  longest, equals the chain over each sample alone -- exactly, over the valid
  prefix.

Writes ``eda/out/_strategy/reproducibility.json``. Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from processing.config import load_processing_config              # noqa: E402
from processing.render import ManifestIndex, render                # noqa: E402
from processing.sampler import Sampler                             # noqa: E402
from processing.ship import ship, ship_sample                      # noqa: E402
from training.folds import apply_folds                             # noqa: E402
from training.spec import CELL_TABLE, SampleSpec                   # noqa: E402

OUT = ROOT / "eda" / "out" / "_strategy"
MANIFEST_DIR = "/data/project/private/dacon-corpus/manifests/strategy-v2"


def _digest(specs_path: str, cfg_path: str, manifest_path: str) -> dict[str, str]:
    """Render every spec in THIS process; sha256 per spec of wav bytes + intervals."""
    cfg = load_processing_config(cfg_path)
    index = ManifestIndex.coerce(pd.read_parquet(manifest_path))
    out = {}
    for d in json.loads(Path(specs_path).read_text()):
        spec = SampleSpec.from_dict(d)
        r = render(spec, index, cfg.render)
        h = hashlib.sha256(r.wav.numpy().tobytes())
        h.update(json.dumps(r.frame_intervals, sort_keys=True).encode())
        out[str(spec.sample_id)] = h.hexdigest()
    return out


def _merged_spans(spec: SampleSpec, role: str | None) -> list[tuple[float, float]]:
    """The non-layer components' spans (one role, or every role), merged."""
    spans = sorted((c.target_start_s, c.target_start_s + c.duration_s)
                   for c in spec.components
                   if c.snr_db is None and (role is None or c.role == role))
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 6), round(e, 6)) for s, e in merged]


def check_i13(spec: SampleSpec, intervals: dict) -> list[str]:
    """The processing contract: a role branch is that role's placement, tiles
    merged, labelled from the cell; the file branch lists every component's
    span with its own label (two overlapping components are two entries,
    which is what the frame-level OR downstream needs). A whole-file spec's
    tiles describe every present role at once."""
    problems = []
    vp, mp, vf, mf = CELL_TABLE[spec.cell]
    present = {"voice": bool(vp), "music": bool(mp)}
    label = {"voice": int(vf or 0), "music": int(mf or 0), "noise": 0}
    whole = spec.render_mode == "whole_file"
    want_file: list[tuple[float, float, int]] = []
    for role in ("voice", "music"):
        got = sorted((round(s, 6), round(e, 6), lab) for s, e, lab in intervals[role])
        if not present[role]:
            want = []
        else:
            spans = _merged_spans(spec, None if whole else role)
            want = [(s, e, label[role]) for s, e in spans]
        if got != sorted(want):
            problems.append(f"{role}: intervals {got} != placement {sorted(want)}")
        want_file += want
    if not whole:
        want_file += [(s, e, 0) for s, e in _merged_spans(spec, "noise")]
    merged: list[tuple[float, float, int]] = []
    for s, e, lab in sorted(want_file):       # abutting or overlapping, same label: one
        if merged and merged[-1][2] == lab and s <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), lab)
        else:
            merged.append((s, e, lab))
    got = sorted((round(s, 6), round(e, 6), lab) for s, e, lab in intervals["file"])
    if got != merged:
        problems.append(f"file: intervals {got} != components {merged}")
    for branch, ivs in intervals.items():
        for s, e, _ in ivs:
            if not (0.0 <= s < e <= spec.duration_s + 1e-6):
                problems.append(f"{branch}: ({s}, {e}) outside the {spec.duration_s}s timeline")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processing-config", default="configs/processing_v1.yaml")
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_processing_config(args.processing_config)
    manifest_path = str(Path(args.manifest_dir) / "manifest.parquet")
    manifest = apply_folds(pd.read_parquet(manifest_path),
                           pd.read_parquet(Path(args.manifest_dir) / "folds.parquet"), args.fold)
    sampler = Sampler(manifest, cfg.draw, slice_="train", fold=args.fold)
    specs = list(sampler.epoch_specs(args.n, epoch=0, seed=args.seed))
    specs_path = OUT / "reproducibility_specs.json"
    specs_path.write_text(json.dumps([s.to_dict() for s in specs]))
    report: dict = {"n": len(specs), "seed": args.seed, "fold": args.fold}

    # I10 -- two fresh interpreters
    code = (f"import json,sys; sys.path.insert(0,{str(ROOT)!r}); "
            f"from scripts.strategy.verify_reproducibility import _digest; "
            f"print(json.dumps(_digest({str(specs_path)!r}, {args.processing_config!r}, "
            f"{manifest_path!r})))")
    digests = []
    for _ in range(2):
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, cwd=ROOT)
        digests.append(json.loads(res.stdout.strip().splitlines()[-1]))
    mismatched = [k for k in digests[0] if digests[0][k] != digests[1][k]]
    report["I10"] = {"pass": not mismatched, "mismatched": mismatched,
                     "detail": f"{len(specs) - len(mismatched)}/{len(specs)} identical across "
                               f"two fresh processes (sha256 of wav bytes + frame intervals)"}
    print("I10:", report["I10"]["detail"], flush=True)

    # I13 / I14 -- in this process
    index = ManifestIndex.coerce(manifest)
    rendered = [render(s, index, cfg.render) for s in specs]
    i13 = {s.sample_id: check_i13(s, r.frame_intervals) for s, r in zip(specs, rendered)}
    bad13 = {k: v for k, v in i13.items() if v}
    n_composed = sum(s.render_mode == "composed" for s in specs)
    n_layer = sum(any(c.snr_db is not None for c in s.components) for s in specs)
    report["I13"] = {"pass": not bad13, "problems": bad13,
                     "detail": f"{len(specs) - len(bad13)}/{len(specs)} specs' frame intervals "
                               f"equal their placement ({n_composed} composed, "
                               f"{len(specs) - n_composed} whole-file, {n_layer} with a layer)"}
    print("I13:", report["I13"]["detail"], flush=True)

    torch.set_num_threads(1)
    solo = [ship_sample(r, cfg.ship) for r in rendered]
    lengths = torch.tensor([r.wav.shape[-1] for r in rendered])
    longest = int(lengths.max())
    batch = torch.zeros(len(rendered), rendered[0].wav.shape[0] if all(
        r.wav.shape[0] == rendered[0].wav.shape[0] for r in rendered) else 1, longest)
    if batch.shape[1] == 1:
        for i, r in enumerate(rendered):
            batch[i, 0, :r.wav.shape[-1]] = r.wav.mean(dim=0)
    else:
        for i, r in enumerate(rendered):
            batch[i, :, :r.wav.shape[-1]] = r.wav
    batch = batch + torch.where(
        torch.arange(longest)[None, :] >= lengths[:, None], 7.0, 0.0)[:, None, :]  # hostile pad
    shipped = ship(batch, cfg.ship, lengths)
    gaps = [float((shipped[i, :int(lengths[i])] - solo[i]).abs().max())
            for i in range(len(rendered))]
    report["I14"] = {"pass": max(gaps) == 0.0, "max_gap": max(gaps),
                     "detail": f"max |solo - in-batch| over the valid prefix = {max(gaps)} "
                               f"across {len(rendered)} real samples, mixed channel counts "
                               f"{sorted({r.wav.shape[0] for r in rendered})}, "
                               f"lengths {int(lengths.min())}..{longest}"}
    print("I14:", report["I14"]["detail"], flush=True)

    (OUT / "reproducibility.json").write_text(json.dumps(report, indent=2, default=str))
    ok = all(report[k]["pass"] for k in ("I10", "I13", "I14"))
    print("all pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
