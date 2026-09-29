"""Size benchmark for the speech-encoder choice (docs/training/12; run-3 discussion).

Times, on one GPU, with the shipped inference setting (fp32, batch 8, 60 s clips = the
worst case) and a training-like setting (bf16 autocast, forward + backward):

  * the CURRENT full model (configs/c_run2.yaml: BEATs 12 + XLS-R-300M 12 layers + heads)
  * bare speech encoders: XLS-R-300M at 12 / 24 layers, XLS-R-1B at 24 / 48, XLS-R-2B at 48,
    w2v-BERT 2.0 at 24 (fbank input, 50 fps)

L4 estimate for a candidate = measured server time (3 m 09 s for 1,200 files) x
(current - xlsr300@12 + candidate) / current, i.e. swap the encoder, keep the rest.

    $V scripts/diag/bench_encoders.py --out /data/project/private/dacon-runs/_bench/encoders.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
W = Path("/data/project/private/dacon-weights")
SR = 16000


def _time(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts)


def bench_current(batch, secs):
    sys.argv = ["x"]
    sys.path.insert(0, "scripts")
    from train import build_model
    m = build_model("configs/c_run2.yaml",
                    f"audio={W}/beats,speech={W}/xlsr-300m").cuda().eval()
    wav = torch.randn(batch, secs * SR, device="cuda") * 0.1
    lengths = torch.full((batch,), secs * SR, device="cuda")

    def fwd():
        with torch.no_grad():
            out = m(wav, lengths)
            m.submission_probs(out)
    t = _time(fwd)
    torch.cuda.empty_cache()
    return t


def encoder(kind, layers):
    from transformers import AutoConfig, Wav2Vec2BertModel, Wav2Vec2Model
    path = W / kind
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    cfg.num_hidden_layers = layers
    cls = Wav2Vec2BertModel if kind.startswith("w2v-bert") else Wav2Vec2Model
    return cls.from_pretrained(path, config=cfg, local_files_only=True), cls is Wav2Vec2BertModel


def inputs(is_bert, batch, secs):
    if is_bert:   # SeamlessM4T fbank: 80 mel x stride-2 stacking = 160 dims at 50 fps
        return {"input_features": torch.randn(batch, secs * 50, 160, device="cuda")}
    return {"input_values": torch.randn(batch, secs * SR, device="cuda") * 0.1}


def bench_encoder(kind, layers, batch, secs, train_batch):
    m, is_bert = encoder(kind, layers)
    m = m.cuda()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    m.eval()
    x = inputs(is_bert, batch, secs)

    def fwd():
        with torch.no_grad():
            m(**x)
    t_inf = _time(fwd)
    torch.cuda.reset_peak_memory_stats()
    fwd()
    mem_inf = torch.cuda.max_memory_allocated() / 2**30
    # training-like: bf16, activation grads through every layer (what LoRA needs), full backward
    m.train()
    m.config.apply_spec_augment = False
    xt = inputs(is_bert, train_batch, secs)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(**xt).last_hidden_state
        out.float().mean().backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        t_tr = _time(step, reps=2)
        mem_tr = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        t_tr, mem_tr = float("nan"), float("nan")
    torch.cuda.empty_cache()
    return {"encoder": kind, "layers": layers, "params_M": round(n_params, 1),
            "infer_s_batch": round(t_inf, 3), "infer_peak_GiB": round(mem_inf, 1),
            "train_s_step": round(t_tr, 3), "train_peak_GiB": round(mem_tr, 1),
            "train_batch": train_batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--secs", type=int, default=60)
    ap.add_argument("--train-batch", type=int, default=4)
    ap.add_argument("--server-seconds", type=float, default=189.0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 as shipped
    torch.backends.cudnn.allow_tf32 = False
    cur = bench_current(a.batch, a.secs)
    rows = []
    for kind, layers in [("xlsr-300m", 12), ("xlsr-300m", 24), ("xlsr-1b", 24), ("xlsr-1b", 48),
                         ("xlsr-2b", 48), ("w2v-bert-2.0", 24)]:
        if not (W / kind).exists():
            continue
        r = bench_encoder(kind, layers, a.batch, a.secs, a.train_batch)
        rows.append(r)
        print(json.dumps(r), flush=True)
    base = next(r for r in rows if r["encoder"] == "xlsr-300m" and r["layers"] == 12)
    for r in rows:
        swapped = cur - base["infer_s_batch"] + r["infer_s_batch"]
        r["est_full_model_vs_current"] = round(swapped / cur, 2)
        r["est_L4_minutes_1200"] = round(a.server_seconds * swapped / cur / 60, 1)
        r["train_step_vs_xlsr300_12"] = round(r["train_s_step"] / base["train_s_step"], 2)
    res = {"current_full_model_infer_s_batch": round(cur, 3), "batch": a.batch, "secs": a.secs,
           "server_seconds_measured": a.server_seconds, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
