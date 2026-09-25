"""proc-enhance: DeepFilterNet3 speech enhancement at 48 kHz, attenuation limit drawn.
venv: /data/project/private/dacon-venvs/proc-enhance
  python scripts/proc/run_enhance.py --workers 14     (or --device cuda)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import common as C

LIC = "MIT OR Apache-2.0 (Rikorose/DeepFilterNet, code and DeepFilterNet3 weights)"
LIMITS = [6.0, 12.0, 18.0, 24.0, None]


def init(device):
    from df.enhance import init_df
    model, st, _ = init_df(model_base_dir="DeepFilterNet3", log_level="ERROR")
    return model.to(device).eval(), st


def proc(ms, device, row):
    import torch
    from df.enhance import enhance
    model, st = ms
    lim = LIMITS[int(np.random.default_rng([int(row.seed), 3]).integers(len(LIMITS)))]
    x, sr = C.load(row.path, st.sr())
    y = enhance(model, st, torch.from_numpy(x)[None], atten_lim_db=lim)
    y = y.float().cpu().numpy().reshape(-1)[: len(x)]
    return C.write(row, y, sr, transform="deepfilternet3_enhance",
                   params={"atten_lim_db": lim if lim is not None else "unlimited"},
                   tool="DeepFilterNet3", tool_version=VER, licence=LIC)["duration_s"]


VER = ""
if __name__ == "__main__":
    import df, torch
    from importlib.metadata import version
    VER = f"deepfilternet {version('deepfilternet')}; torch {torch.__version__}"
    a = C.argp(14).parse_args()
    rows = C.plan("proc-enhance", a.shard, a.limit)
    C.run("enhance", rows, init, proc, workers=a.workers, threads=a.threads, device=a.device, reverse=a.reverse)
