"""Train the RVC target voices of the ``rvc`` family (round 3) with Applio (MIT, IAHispano/Applio):
RVC v2, 40 kHz, HiFi-GAN NSF generator fine-tuned from Applio's f0G40k/f0D40k pretrained,
RMVPE f0, ContentVec features, faiss retrieval index ("Auto").

Targets and their training sets come from ko_rvc_targets.py (targets.csv). Epochs scale with the
data: --steps-target optimiser steps per target (batch 8, ~3 s slices), clipped to [60, 300]
epochs. Skips targets whose final weights exist. Never enables Applio's shutdown_check.
Experiment dirs: Applio/logs/<target>/ (the repo is under /data/project/private).

  python scripts/synth3/ko_rvc_train.py [--only t00,t01] [--steps-target 2000]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

A = Path("/data/project/private/dacon-weights/synth3/repos/Applio")
ROOT = Path("/data/project/private/dacon-weights/synth3/rvc")


def final_weights(name: str) -> list[str]:
    return sorted(glob.glob(str(A / "logs" / name / f"{name}_*e_*s.pth")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--steps-target", type=int, default=2000)
    ap.add_argument("--shard", default="0/1")   # ko_run.sbatch passes --shard; i/n over targets
    ap.add_argument("--deadline", type=float, default=0.0)
    a, _ = ap.parse_known_args()
    i, n = (int(x) for x in a.shard.split("@")[0].split("/"))
    os.chdir(A)
    sys.path.insert(0, str(A))
    import core
    rows = list(csv.DictReader((ROOT / "targets.csv").open(encoding="utf-8")))
    only = set(filter(None, a.only.split(",")))
    for k, r in enumerate(rows):
        name = r["target"]
        if (only and name not in only) or k % n != i or final_weights(name):
            continue
        if a.deadline and time.time() > a.deadline:
            break
        t0 = time.time()
        secs = float(r["seconds"])
        epochs = int(min(300, max(60, a.steps_target * 8 * 3.0 / max(secs, 1.0))))
        print(time.strftime("%H:%M:%S"), name, r["speaker"], f"{secs:.0f}s", f"epochs={epochs}", flush=True)
        print(core.run_preprocess_script(name, str(ROOT / "datasets" / name), 40000, 8, "Automatic",
                                         False, False, 0.7, 3.0, 0.3, "none"), flush=True)
        print(core.run_extract_script(name, "rmvpe", 8, 0, 40000, "contentvec", None, 2), flush=True)
        print(core.run_train_script(name, epochs, True, False, epochs, 40000, 8, 0, True, False,
                                    "Auto", True, False, None, None, "HiFi-GAN", False, False), flush=True)
        print(time.strftime("%H:%M:%S"), name, "weights", final_weights(name)[-1:],
              "index", glob.glob(str(A / "logs" / name / "*.index")), f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
