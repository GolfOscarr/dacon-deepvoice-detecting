"""proc-codec-lm, XCodec2 half (HKUSTAudio/xcodec2, 16 kHz, one 65,536-entry codebook at 50 Hz;
the codec Llasa generates through). The Mimi half is run_hf_codec.py --tool mimi.
venv: /data/project/private/dacon-venvs/proc-xcodec2   (HF_HOME=/data/project/private/dacon-weights/proc/hf)
  python scripts/proc/run_xcodec2.py --device cuda
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/data/project/private/dacon-weights/proc/hf")
import numpy as np
import common as C
from run_hf_codec import codec_lm_tool

REPO = "HKUSTAudio/xcodec2"
LIC = "CC-BY-NC-4.0 (HF card HKUSTAudio/xcodec2)"


def init(device):
    from xcodec2.modeling_xcodec2 import XCodec2Model
    return XCodec2Model.from_pretrained(REPO).to(device).eval()


def proc(model, device, row):
    import torch
    x, sr = C.load(row.path, 16000)
    with torch.inference_mode():
        t = torch.from_numpy(x)[None].to(device)
        codes = model.encode_code(input_waveform=t)
        y = model.decode_code(codes)
    y = y.float().cpu().numpy().reshape(-1)[: len(x)]
    return C.write(row, y, 16000, transform="xcodec2_roundtrip",
                   params={"codebook_size": 65536, "token_rate_hz": 50},
                   tool=REPO, tool_version=VER, licence=LIC)["duration_s"]


VER = ""
if __name__ == "__main__":
    import torch, xcodec2, transformers
    from huggingface_hub import snapshot_download
    rev = os.path.basename(snapshot_download(REPO, local_files_only=True))
    VER = f"xcodec2 0.1.5; transformers {transformers.__version__}; torch {torch.__version__}; rev {rev}"
    a = C.argp(0).parse_args()
    rows = C.plan("proc-codec-lm", a.shard)
    rows = rows[[codec_lm_tool(int(s)) == "xcodec2" for s in rows.seed]]
    if a.limit: rows = rows.head(a.limit)
    C.run("xcodec2", rows, init, proc, workers=a.workers, threads=a.threads, device=a.device, reverse=a.reverse)
