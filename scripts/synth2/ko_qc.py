"""QC for the round-2 Korean fakes (plan 09 §3.2, identical for every family).

Whisper large-v3 (MIT) Korean CER <= 0.35 against the input text (seedvc: the source
transcript, which is the `text` column), duration 3-20 s, RMS >= -45 dBFS, clipped <= 0.1 %,
silence <= 50 %. CER on Hangul/alnum after dropping punctuation and spaces.

venv: /data/project/private/dacon-venvs/synth2-qc; HF_HOME=/data/project/private/dacon-weights/synth2/hf.

  ko_qc.py qc --family F [--shard i/n] [--wait-pid P]   QC rows of F/synth.csv in this shard not
        yet in F/qc.csv; with --wait-pid keep polling until P exits, then a last pass. After each
        pass F/metadata.csv is rewritten atomically (synth ⋈ qc), under a lock.
  ko_qc.py finalize --family F                          rewrite metadata.csv only (CPU).
  ko_qc.py report                                       per-family totals (CPU).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

WHISPER = "openai/whisper-large-v3"
MAX_CER, MIN_S, MAX_S, MIN_DBFS, MAX_CLIP, MAX_SIL = 0.35, 3.0, 20.0, -45.0, 0.001, 0.5


def norm_ko(s: str) -> str:
    return "".join(ch for ch in s.lower() if ("가" <= ch <= "힣") or ch.isalnum())


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


class ASR:
    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(WHISPER)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=self.dt).to(self.dev).eval()

    def __call__(self, audios: list[np.ndarray]) -> list[str]:
        feats = self.proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.inference_mode():
            ids = self.model.generate(feats.to(self.dev, self.dt), language="ko", task="transcribe",
                                      num_beams=1, do_sample=False)
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


def load16k(path: Path):
    from math import gcd
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    g = gcd(sr, 16000)
    return a, (resample_poly(a, 16000 // g, sr // g).astype(np.float32) if sr != 16000 else a), sr


def finalize(family: str) -> tuple[int, int, float]:
    d = kc.OUT_ROOT / family
    lock = open(d / ".finalize.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        qc = {r["file"]: r for r in kc.read_csv(d / "qc.csv")}
        rows = []
        for r in kc.read_csv(d / "synth.csv"):
            q = qc.get(r["file"])
            if q is None:
                continue
            rows.append({**{k: r[k] for k in kc.META_COLUMNS},
                         "qc_cer": q["qc_cer"], "kept": q["kept"], "drop_reason": q["drop_reason"]})
        tmp = d / f"metadata.csv.tmp{os.getpid()}"
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=kc.FINAL_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, d / "metadata.csv")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    kept = [r for r in rows if r["kept"] == "True"]
    return len(rows), len(kept), sum(float(r["duration_s"]) for r in kept) / 3600


def qc_pass(asr: ASR, family: str, shard: str, bs: int = 16) -> int:
    d = kc.OUT_ROOT / family
    seen = {r["file"] for r in kc.read_csv(d / "qc.csv")}
    todo = [r for r in kc.read_csv(d / "synth.csv") if r["file"] not in seen and kc.in_shard(r["file"], shard)]
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        loaded = [load16k(d / r["file"]) for r in batch]
        hyps = asr([x[1] for x in loaded])
        for r, (a, _, sr), hyp in zip(batch, loaded, hyps):
            dur = len(a) / sr
            rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
            dbfs = 20 * np.log10(max(rms, 1e-9))
            clip = float(np.mean(np.abs(a) >= 0.985)) if len(a) else 1.0
            hop = sr // 100
            n = len(a) // hop
            fr = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) if n else np.zeros(1)
            sil = float(np.mean(fr < 10 ** (-50 / 20)))
            c = cer(norm_ko(r["text"]), norm_ko(hyp))
            why = [k for k, bad in (("cer", c > MAX_CER), ("duration", not MIN_S <= dur <= MAX_S),
                                    ("rms", dbfs < MIN_DBFS), ("clipped", clip > MAX_CLIP),
                                    ("silence", sil > MAX_SIL)) if bad]
            kc.append_row(d / "qc.csv", kc.QC_COLUMNS, {
                "file": r["file"], "qc_cer": f"{c:.4f}", "asr_text": hyp, "duration_s": f"{dur:.3f}",
                "rms_dbfs": f"{dbfs:.2f}", "clip_frac": f"{clip:.5f}", "silence_frac": f"{sil:.3f}",
                "kept": str(not why), "drop_reason": "|".join(why)})
    return len(todo)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def report() -> None:
    for d in sorted(p for p in kc.OUT_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        s = kc.read_csv(d / "synth.csv")
        q = kc.read_csv(d / "qc.csv")
        k = [r for r in q if r["kept"] == "True"]
        from collections import Counter
        why = Counter(x for r in q for x in r["drop_reason"].split("|") if x)
        print(f"{d.name:12s} synth={len(s):5d} ({sum(float(r['duration_s']) for r in s)/3600:.2f} h) "
              f"qc={len(q):5d} kept={len(k):5d} ({sum(float(r['duration_s']) for r in k)/3600:.2f} h) "
              f"reject={1 - len(k)/max(len(q),1):.1%} spk={len({r['prompt_speaker'] for r in s})} {dict(why)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qc", "finalize", "report"])
    ap.add_argument("--family", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--poll", type=float, default=45.0)
    a = ap.parse_args()
    if a.cmd == "report":
        return report()
    if a.cmd == "finalize":
        print(a.family, finalize(a.family))
        return
    asr = ASR()
    while True:
        running = a.wait_pid and alive(a.wait_pid)
        n = qc_pass(asr, a.family, a.shard)
        if n:
            tot, kept, h = finalize(a.family)
            kc.log(f"qc {a.family} {a.shard}: +{n} checked; family metadata {tot} rows, kept {kept} "
                   f"({kept/max(tot,1):.1%}) = {h:.2f} h")
        if not running:
            break
        time.sleep(a.poll)
    kc.log("qc done")


if __name__ == "__main__":
    main()
