"""Exact-zero screen for the round-3 fakes (lead's rule, 2026-09-27, "relaxed"): a file is flagged
when, inside its speech span, it has a run of exact-zero samples longer than 20 ms or more than 5 %
exact-zero samples. (Calibration on 200 files each: real Zeroth/Emilia/YODAS have runs > 20 ms in
0-1.5 % and fractions > 5 % in 0-2 % of files, but fractions > 1 % in 9-24 %: int16 zero crossings.) Digital-silence gaps are a label shortcut (real recordings carry noise), and
dithering only the fakes would be a shortcut too, so flagged files are dropped instead.

Speech span: first to last 10 ms frame above -50 dBFS RMS (ko_qc.py's silence threshold).
Writes <family>/zeros.csv (file, max_zero_run_ms, zero_frac, flagged) for every synth.csv row not
yet in it, then calls ko_qc.finalize, which sets kept=False / drop_reason "zeros" for flagged
files. CPU only; `--loop` repeats until --until (epoch s), then one last pass.

  KO_SYNTH2_ROOT=/data/project/private/dacon-corpus/interim/ko-synth3 \\
    python scripts/synth3/ko_zeros.py --families cosyvoice3 qwen3tts outetts f5tts [--loop --until E]
  ... report   (per-family flag rates)
  ... reflag   (recompute `flagged` in zeros.csv from the stored stats with the current rule, then finalize)
  ... --stats-only (write stats with an empty `flagged` and a `would_flag` column; never finalize:
                    ko_qc.finalize ignores rows whose flagged != "True", so metadata is unchanged)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402
import ko_qc  # noqa: E402

COLS = ["file", "max_zero_run_ms", "zero_frac", "flagged"]
MAX_RUN_S, MAX_FRAC = 0.020, 0.05


def zero_stats(path: Path) -> tuple[float, float]:
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="int16", always_2d=True)
    a = a[:, 0]
    hop = sr // 100
    n = len(a) // hop
    if n == 0:
        return 0.0, 0.0
    fr = np.sqrt((a[: n * hop].astype(np.float64).reshape(n, hop) / 32768.0) ** 2).mean(axis=1)
    loud = np.nonzero(fr > 10 ** (-50 / 20))[0]
    if len(loud) == 0:
        return 0.0, 0.0
    x = a[loud[0] * hop: (loud[-1] + 1) * hop]
    z = (x == 0).astype(np.int8)
    if not z.any():
        return 0.0, 0.0
    d = np.diff(np.concatenate([[0], z, [0]]))
    runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
    return 1000.0 * runs.max() / sr, float(z.mean())


def flag(run_ms: float, frac: float) -> bool:
    return run_ms > MAX_RUN_S * 1000 or frac > MAX_FRAC


def one_pass(family: str, stats_only: bool = False) -> int:
    d = kc.OUT_ROOT / family
    seen = {r["file"] for r in kc.read_csv(d / "zeros.csv")}
    todo = [r["file"] for r in kc.read_csv(d / "synth.csv") if r["file"] not in seen]
    for f in todo:
        run_ms, frac = zero_stats(d / f)
        row = {"file": f, "max_zero_run_ms": f"{run_ms:.1f}", "zero_frac": f"{frac:.5f}",
               "flagged": "" if stats_only else str(flag(run_ms, frac))}
        if stats_only:
            row["would_flag"] = str(flag(run_ms, frac))
        kc.append_row(d / "zeros.csv", COLS + (["would_flag"] if stats_only else []), row)
    if todo and not stats_only and (d / "qc.csv").exists():
        ko_qc.finalize(family)
    return len(todo)


def reflag(family: str) -> None:
    import csv
    import os
    d = kc.OUT_ROOT / family
    rows = kc.read_csv(d / "zeros.csv")
    for r in rows:
        r["flagged"] = str(flag(float(r["max_zero_run_ms"]), float(r["zero_frac"])))
    tmp = d / "zeros.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, d / "zeros.csv")
    ko_qc.finalize(family)


def report(families: list[str]) -> None:
    for fam in families:
        z = kc.read_csv(kc.OUT_ROOT / fam / "zeros.csv")
        m = kc.read_csv(kc.OUT_ROOT / fam / "metadata.csv")
        fl = [r for r in z if "True" in (r["flagged"], r.get("would_flag"))]
        k = [r for r in m if r["kept"] == "True"]
        print(f"{fam:11s} checked={len(z):5d} flagged={len(fl):4d} ({len(fl)/max(len(z),1):.1%}) "
              f"run>20ms={sum(float(r['max_zero_run_ms']) > 20 for r in z)} frac>5%={sum(float(r['zero_frac']) > .05 for r in z)} "
              f"| metadata rows={len(m)} kept={len(k)} ({sum(float(r['duration_s']) for r in k)/3600:.2f} h)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="scan", choices=["scan", "report", "reflag"])
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--until", type=float, default=0.0)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a.families)
    if a.cmd == "reflag":
        for fam in a.families:
            reflag(fam)
        return report(a.families)
    while True:
        last = a.until and time.time() >= a.until
        for fam in a.families:
            n = one_pass(fam, a.stats_only)
            if n:
                kc.log(f"zeros {fam}: +{n} checked")
        if not a.loop or last:
            break
        time.sleep(120)
    report(a.families)


if __name__ == "__main__":
    main()
