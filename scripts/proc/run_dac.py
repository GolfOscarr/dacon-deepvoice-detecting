"""proc-dac: Descript Audio Codec round-trip, 16 kHz or 44.1 kHz model, n_quantizers low/high.
venv: /data/project/private/dacon-venvs/proc-dac   weights: /data/project/private/dacon-weights/proc/dac
  python scripts/proc/run_dac.py --device cuda        (or --workers N on CPU)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C

W = "/data/project/private/dacon-weights/proc/dac"
MODELS = {"16khz": (f"{W}/weights_16khz.pth", "0.0.5/weights_16khz.pth", [2, 3, 4], [8, 12]),
          "44khz": (f"{W}/weights_44khz.pth", "0.0.1/weights.pth", [2, 3, 4], [6, 9])}
LIC = "MIT (descriptinc/descript-audio-codec LICENSE, code and released weights)"


def init(device):
    import dac
    return {k: dac.DAC.load(v[0]).to(device).eval() for k, v in MODELS.items()}


def proc(models, device, row):
    import torch
    r = np.random.default_rng([int(row.seed), 2])
    which = str(r.choice(list(MODELS)))
    _, rel, low, high = MODELS[which]
    band = "low" if r.random() < 0.5 else "high"
    nq = int(r.choice(low if band == "low" else high))
    m = models[which]
    x, sr = C.load(row.path, m.sample_rate)
    with torch.inference_mode():
        t = torch.from_numpy(x)[None, None].to(device)
        t = m.preprocess(t, sr)
        z, *_ = m.encode(t, n_quantizers=nq)
        y = m.decode(z)
    y = y.float().cpu().numpy().reshape(-1)[: len(x)]
    return C.write(row, y, sr, transform=f"dac_{which}_roundtrip",
                   params={"model": which, "n_quantizers": nq, "band": band,
                           "weights": f"github release {rel}"},
                   tool=f"descript-audio-codec ({which})", tool_version=VER, licence=LIC)["duration_s"]


VER = ""
if __name__ == "__main__":
    import dac, torch
    VER = f"descript-audio-codec {dac.__version__}; torch {torch.__version__}"
    a = C.argp(0).parse_args()
    rows = C.plan("proc-dac", a.shard, a.limit)
    C.run("dac", rows, init, proc, workers=a.workers, threads=a.threads, device=a.device, reverse=a.reverse)
