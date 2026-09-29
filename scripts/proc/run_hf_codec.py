"""proc-encodec (EnCodec 24 kHz) and the Mimi half of proc-codec-lm, via transformers.
venv: /data/project/private/dacon-venvs/proc-codec   (HF_HOME=/data/project/private/dacon-weights/proc/hf)
  python scripts/proc/run_hf_codec.py --tool encodec|mimi --workers 14
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("HF_HOME", "/data/project/private/dacon-weights/proc/hf")
import numpy as np
import common as C

SPEC = {
    "encodec": dict(family="proc-encodec", repo="facebook/encodec_24khz",
                    licence="MIT (facebookresearch/encodec code and weights, repo LICENSE); "
                            "HF card facebook/encodec_24khz: licence field empty"),
    "mimi": dict(family="proc-codec-lm", repo="kyutai/mimi",
                 licence="CC-BY-4.0 (HF card kyutai/mimi)"),
}


def codec_lm_tool(seed: int) -> str:
    """proc-codec-lm splits its files between XCodec2 and Mimi by the file seed."""
    return "xcodec2" if np.random.default_rng([seed, 7]).random() < 0.5 else "mimi"


def init(tool):
    def f(device):
        import torch
        from transformers import EncodecModel, MimiModel
        cls = EncodecModel if tool == "encodec" else MimiModel
        m = cls.from_pretrained(SPEC[tool]["repo"]).to(device).eval()
        return m
    return f


def proc(tool, ver):
    def f(model, device, row):
        import torch
        sr = model.config.sampling_rate
        x, _ = C.load(row.path, sr)
        r = np.random.default_rng([int(row.seed), 1])
        with torch.inference_mode():
            t = torch.from_numpy(x)[None, None].to(device)
            if tool == "encodec":
                bw = float(r.choice([1.5, 3.0, 6.0, 12.0]))
                enc = model.encode(t, bandwidth=bw)
                y = model.decode(enc.audio_codes, enc.audio_scales)[0]
                params = {"bandwidth_kbps": bw}
            else:
                nq = int(r.choice([8, 16, 32]))
                codes = model.encode(t, num_quantizers=nq).audio_codes
                y = model.decode(codes).audio_values
                params = {"num_quantizers": nq, "frame_rate_hz": 12.5}
        y = y.float().cpu().numpy().reshape(-1)[: len(x)]
        rec = C.write(row, y, sr, transform=f"{tool}_roundtrip", params=params,
                      tool=SPEC[tool]["repo"], tool_version=ver, licence=SPEC[tool]["licence"])
        return rec["duration_s"]
    return f


if __name__ == "__main__":
    ap = C.argp(14); ap.add_argument("--tool", required=True, choices=list(SPEC))
    a = ap.parse_args()
    import transformers, torch
    from huggingface_hub import snapshot_download
    rev = os.path.basename(snapshot_download(SPEC[a.tool]["repo"], local_files_only=True))
    ver = f"transformers {transformers.__version__}; torch {torch.__version__}; rev {rev}"
    rows = C.plan(SPEC[a.tool]["family"], a.shard)
    if a.tool == "mimi":
        rows = rows[[codec_lm_tool(int(s)) == "mimi" for s in rows.seed]]
    if a.limit: rows = rows.head(a.limit)
    C.run(a.tool, rows, init(a.tool), proc(a.tool, ver), workers=a.workers, threads=a.threads,
          device=a.device, reverse=a.reverse)
