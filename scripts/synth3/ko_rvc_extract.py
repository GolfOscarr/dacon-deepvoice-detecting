"""Recover the inference weights of RVC targets whose training finished but whose final
extraction failed (Applio's extract_model needs assets/config.json, which only the web UI
creates; it is now copied from config_template.json). Loads logs/<t>/G_2333333.pth (the latest
generator, saved with save_only_latest) and writes logs/<t>/<t>_<epoch>e_<step>s.pth exactly as
train.py would. CPU only.

  python scripts/synth3/ko_rvc_extract.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

A = Path("/data/project/private/dacon-weights/synth3/repos/Applio")


def main() -> None:
    os.chdir(A)
    sys.path.insert(0, str(A))
    import torch
    from rvc.train.process.extract_model import extract_model
    from rvc.train.utils import HParams
    for d in sorted((A / "logs").glob("t[0-9][0-9]")):
        name = d.name
        g = d / "G_2333333.pth"
        if not g.exists() or glob.glob(str(d / f"{name}_*e_*s.pth")):
            continue
        ck = torch.load(g, map_location="cpu", weights_only=True)
        hps = HParams(**json.loads((d / "config.json").read_text()))
        epoch = int(ck.get("iteration", 0))
        step = int(ck.get("global_step", epoch))
        extract_model(ckpt=ck["model"], sr=hps.data.sample_rate, name=name,
                      model_path=str(d / f"{name}_{epoch}e_{step}s.pth"), epoch=epoch, step=step,
                      hps=hps, vocoder="HiFi-GAN")


if __name__ == "__main__":
    main()
