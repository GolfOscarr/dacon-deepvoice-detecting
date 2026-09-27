"""Digital-zero screen (lead's relaxed rule, 2026-09-27): exact-zero samples are a label shortcut (real
recordings rarely hold long runs of exact zeros). Per wav, on the int16 samples after trimming leading and
trailing exact zeros (the render pipeline re-trims), measure
    run_ms_max  the longest run of exact-zero samples inside the span, in ms
    zero_frac   the fraction of exact-zero samples inside the span
A file fails if run_ms_max > 20 or zero_frac > 0.05.

  en_zeros.py scan <family_dir>...      write <family_dir>/zeros.csv only (no metadata change; en-synth2)
  en_zeros.py apply <family_dir>...     scan, then rewrite metadata.csv = synth ⋈ qc as ko_qc.finalize does,
                                        with kept=False and "zeros" added to drop_reason for failing files
CPU only; any python with numpy + soundfile.
"""
from __future__ import annotations

import csv
import fcntl
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

MAX_RUN_MS, MAX_FRAC = 20.0, 0.05
FINAL_COLUMNS = ["file", "family", "model", "model_revision", "licence", "text", "text_source_utts",
                 "prompt_speaker", "prompt_files", "seed", "duration_s", "sample_rate",
                 "qc_cer", "kept", "drop_reason"]


def measure(path: str) -> tuple[float, float]:
    a, sr = sf.read(path, dtype="int16", always_2d=False)
    if a.ndim > 1:
        a = a[:, 0]
    nz = np.flatnonzero(a)
    if len(nz) == 0:
        return float("inf"), 1.0
    z = a[nz[0]: nz[-1] + 1] == 0
    if not z.any():
        return 0.0, 0.0
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return float(runs.max()) / sr * 1000.0, float(z.mean())


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scan(d: Path) -> dict[str, tuple[float, float]]:
    files = [r["file"] for r in read_csv(d / "synth.csv")]
    with ProcessPoolExecutor(16) as ex:
        res = dict(zip(files, ex.map(measure, [str(d / f) for f in files], chunksize=64)))
    tmp = d / "zeros.csv.tmp"
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "run_ms_max", "zero_frac", "zeros_fail"])
        for f, (run, frac) in res.items():
            w.writerow([f, f"{run:.2f}", f"{frac:.5f}", run > MAX_RUN_MS or frac > MAX_FRAC])
    os.replace(tmp, d / "zeros.csv")
    return res


def apply(d: Path, res: dict[str, tuple[float, float]]) -> str:
    lock = open(d / ".finalize.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        qc = {r["file"]: r for r in read_csv(d / "qc.csv")}
        rows, before, extra = [], 0, 0
        for r in read_csv(d / "synth.csv"):
            q = qc.get(r["file"])
            if q is None:
                continue
            kept, why = q["kept"] == "True", [x for x in q["drop_reason"].split("|") if x]
            before += kept
            run, frac = res[r["file"]]
            if run > MAX_RUN_MS or frac > MAX_FRAC:
                extra += kept
                kept = False
                why.append("zeros")
            rows.append({**{k: r[k] for k in FINAL_COLUMNS[:12]}, "qc_cer": q["qc_cer"],
                         "kept": str(kept), "drop_reason": "|".join(why)})
        tmp = d / f"metadata.csv.tmp{os.getpid()}"
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, d / "metadata.csv")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    kept = [r for r in rows if r["kept"] == "True"]
    h = sum(float(r["duration_s"]) for r in kept) / 3600
    return (f"rows={len(rows)} kept_before={before} newly_dropped={extra} ({extra / max(before, 1):.1%}) "
            f"kept={len(kept)} ({len(kept) / max(len(rows), 1):.1%}) = {h:.2f} h")


if __name__ == "__main__":
    cmd, dirs = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    for d in dirs:
        res = scan(d)
        fails = sum(r > MAX_RUN_MS or f > MAX_FRAC for r, f in res.values())
        msg = f"{d}: scanned={len(res)} zeros_fail={fails} ({fails / max(len(res), 1):.1%})"
        if cmd == "apply":
            msg += " | " + apply(d, res)
        print(msg, flush=True)
