"""Verify the ko-synth corpus: metadata schema, files present, mono PCM16, ≥ 3 s,
non-silent, prompt files exist, prompt speaker format; print a per-family table.

  python scripts/synth/verify_ko_synth.py [--root /data/project/private/dacon-corpus/interim/ko-synth] [--full]

--full reads every wav (slow on the shared filesystem); default reads the
header of every file and the samples of a seeded 40-file sample per family.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_common import META_COLUMNS, CORPUS_ROOT, OUT_ROOT  # noqa: E402

SPK_RE = re.compile(r"^(zeroth-korean/(train|test)_data_01/\d{3}/\d+|[a-z0-9]+/.+)$")


def check_family(fam_dir: Path, full: bool) -> dict:
    rows = list(csv.DictReader((fam_dir / "metadata.csv").open(newline="", encoding="utf-8")))
    problems: list[str] = []
    if rows and list(rows[0].keys()) != META_COLUMNS:
        problems.append(f"columns {list(rows[0].keys())}")
    files = {r["file"] for r in rows}
    if len(files) != len(rows):
        problems.append("duplicate file rows")
    on_disk = {p.name for p in fam_dir.glob("*.wav")}
    if on_disk - files:
        problems.append(f"{len(on_disk - files)} wavs without metadata")
    if files - on_disk:
        problems.append(f"{len(files - on_disk)} metadata rows without wav")
    dur = np.array([float(r["duration_s"]) for r in rows])
    srs = sorted({int(r["sample_rate"]) for r in rows})
    speakers = {r["prompt_speaker"] for r in rows}
    bad_spk = [s for s in speakers if not SPK_RE.match(s)]
    if bad_spk:
        problems.append(f"bad prompt_speaker format: {bad_spk[:3]}")
    if (dur < 3.0).any():
        problems.append(f"{int((dur < 3.0).sum())} rows under 3 s")
    # prompt files exist (sample)
    rng = random.Random(0)
    for r in rng.sample(rows, min(50, len(rows))):
        for pf in filter(None, r["prompt_files"].split("|")):
            if not (CORPUS_ROOT / "interim" / pf).exists():
                problems.append(f"missing prompt file {pf}")
                break
    # audio checks
    sample = rows if full else rng.sample(rows, min(40, len(rows)))
    n_bad_audio = 0
    for r in (rows if full else rows[:: max(1, len(rows) // 400)]):
        info = sf.info(str(fam_dir / r["file"]))
        if info.channels != 1 or info.subtype != "PCM_16" or info.samplerate != int(r["sample_rate"]) \
                or abs(info.frames / info.samplerate - float(r["duration_s"])) > 0.01:
            n_bad_audio += 1
    for r in sample:
        a, sr = sf.read(str(fam_dir / r["file"]), dtype="float32")
        rms = float(np.sqrt((a ** 2).mean()))
        if rms < 1e-3 or not np.isfinite(a).all() or np.abs(a).max() < 0.01:
            n_bad_audio += 1
    if n_bad_audio:
        problems.append(f"{n_bad_audio} bad audio files (header or content)")
    return {
        "family": fam_dir.name, "files": len(rows), "hours": dur.sum() / 3600,
        "dur_min": dur.min() if len(dur) else 0, "dur_med": float(np.median(dur)) if len(dur) else 0,
        "dur_max": dur.max() if len(dur) else 0, "sr": "/".join(map(str, srs)),
        "speakers": len(speakers), "model": rows[0]["model"] if rows else "", "licence": rows[0]["licence"] if rows else "",
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=OUT_ROOT)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    total_h, total_f = 0.0, 0
    print(f"{'family':10} {'files':>6} {'hours':>7} {'min':>5} {'med':>5} {'max':>5} {'sr':>11} {'spk':>4}  licence")
    for fam_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (fam_dir / "metadata.csv").exists():
            print(f"{fam_dir.name:10} (no metadata.csv)")
            continue
        s = check_family(fam_dir, args.full)
        total_h += s["hours"]; total_f += s["files"]
        print(f"{s['family']:10} {s['files']:6d} {s['hours']:7.2f} {s['dur_min']:5.1f} {s['dur_med']:5.1f} {s['dur_max']:5.1f} "
              f"{s['sr']:>11} {s['speakers']:4d}  {s['licence']}")
        for p in s["problems"]:
            print(f"    PROBLEM: {p}")
    print(f"{'TOTAL':10} {total_f:6d} {total_h:7.2f}")


if __name__ == "__main__":
    main()
