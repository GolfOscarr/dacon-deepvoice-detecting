"""Whisper large-v3 for the Mandarin fakes (S4): prompt transcripts, QC and the final join.

venv: /data/project/private/dacon-venvs/synth2-qc (transformers 4.57 + torch 2.14, built by
the Korean S2 track; used read-only) with PYTHONPATH=/data/project/private/dacon-venvs/synth2-zhlib
(zhconv, for traditional -> simplified). Whisper weights: HF openai/whisper-large-v3 (MIT) under
HF_HOME=/data/project/private/dacon-weights/synth2/hf.

  zh_qc.py transcribe                    _index/prompts.csv -> _index/prompt_transcripts.csv
  zh_qc.py qc --families a,b [--wait-pids p1,p2]
        QC every row of <family>/synth.csv not yet in <family>/qc.csv; with --wait-pids keep
        polling until those processes exit, then do a last pass. After each pass
        <family>/metadata.csv is rewritten atomically (synth ⋈ qc).
  zh_qc.py finalize --families a,b       rewrite metadata.csv only (CPU).

QC (plan 09 §3.2, identical for every family): Whisper zh CER <= 0.35 against the input text,
duration 3-20 s, RMS >= -45 dBFS, clipped <= 0.1 %, silence <= 50 %. CER is computed on hanzi
after dropping punctuation/space, with the hypothesis converted to simplified and its Arabic
numerals read out (the input text is already TN-normalised to hanzi).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zh_common as zc  # noqa: E402

WHISPER = "openai/whisper-large-v3"
MAX_CER, MIN_S, MAX_S, MIN_DBFS, MAX_CLIP, MAX_SIL = 0.35, 3.0, 20.0, -45.0, 0.001, 0.5

# --------------------------------------------------------------------------- text normalisation
DIG = "零一二三四五六七八九"


def _int_cn(n: int) -> str:
    if n == 0:
        return "零"
    units = [(10 ** 8, "亿"), (10 ** 4, "万"), (1000, "千"), (100, "百"), (10, "十")]
    out, zero = "", False
    for v, u in units:
        q, n = divmod(n, v)
        if q:
            if zero and out:
                out += "零"
            out += (_int_cn(q) if q >= 10 else DIG[q]) + u
            zero = False
        elif out:
            zero = True
    if n:
        if zero and out:
            out += "零"
        out += DIG[n]
    return out[1:] if out.startswith("一十") else out


def _num(m: re.Match) -> str:
    s, after = m.group(0), m.string[m.end(): m.end() + 1]
    if "." in s:
        a, b = s.split(".", 1)
        return (_int_cn(int(a)) if a else "零") + "点" + "".join(DIG[int(c)] for c in b)
    if after == "年" or len(s) > 9 or s.startswith("0"):
        return "".join(DIG[int(c)] for c in s)
    return _int_cn(int(s))


def norm_zh(s: str, simplify=None) -> str:
    if simplify:
        s = simplify(s)
    s = s.replace("％", "%").replace(",", "")
    s = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: "百分之" + m.group(1), s)
    s = re.sub(r"\d+(?:\.\d+)?", _num, s)
    return "".join(ch for ch in s if "一" <= ch <= "鿿" or ch.isalnum()).lower()


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


# --------------------------------------------------------------------------- whisper
class ASR:
    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(WHISPER)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=dt).to(self.dev).eval()
        self.dt = dt

    def __call__(self, audios: list[np.ndarray]) -> list[str]:
        feats = self.proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.inference_mode():
            ids = self.model.generate(feats.to(self.dev, self.dt), language="zh", task="transcribe",
                                      num_beams=1, do_sample=False)
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


def load16k(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    g = gcd(sr, 16000)
    return a, (resample_poly(a, 16000 // g, sr // g).astype(np.float32) if sr != 16000 else a), sr


def simplifier():
    try:
        import zhconv
        return lambda s: zhconv.convert(s, "zh-cn")
    except ImportError:
        print("WARNING: zhconv missing; traditional output will count as errors", flush=True)
        return None


# --------------------------------------------------------------------------- commands
def transcribe(bs: int = 16) -> None:
    out = zc.IDX / "prompt_transcripts.csv"
    done = {r["prompt_id"] for r in zc.read_csv(out)}
    todo = [p for p in zc.load_prompts() if p.prompt_id not in done]
    zc.log(f"transcribe: {len(todo)} prompts to do")
    asr, simp = ASR(), simplifier()
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        texts = asr([load16k(p.abs_path)[1] for p in batch])
        for p, t in zip(batch, texts):
            t = simp(t) if simp else t
            zc.append_row(out, ["prompt_id", "speaker_ref_id", "text"],
                          {"prompt_id": p.prompt_id, "speaker_ref_id": p.speaker, "text": t})
    zc.log("transcribe done")


def finalize(family: str) -> tuple[int, int]:
    d = zc.OUT_ROOT / family
    if not (d / "synth.csv").exists():   # the synth process has not written its first file yet
        return 0, 0
    qc = {r["file"]: r for r in zc.read_csv(d / "qc.csv")}
    rows = []
    for r in zc.read_csv(d / "synth.csv"):
        q = qc.get(r["file"])
        if q is None:
            continue
        rows.append({**{k: r[k] for k in zc.META_COLUMNS},
                     "qc_cer": q["qc_cer"], "kept": q["kept"], "drop_reason": q["drop_reason"]})
    tmp = d / "metadata.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=zc.FINAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, d / "metadata.csv")
    return len(rows), sum(r["kept"] == "True" for r in rows)


def qc_pass(asr: ASR, simp, family: str, bs: int = 16) -> int:
    d = zc.OUT_ROOT / family
    seen = {r["file"] for r in zc.read_csv(d / "qc.csv")}
    todo = [r for r in zc.read_csv(d / "synth.csv") if r["file"] not in seen]
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
            c = cer(norm_zh(r["text"]), norm_zh(hyp, simp))
            why = [k for k, bad in (("cer", c > MAX_CER), ("duration", not MIN_S <= dur <= MAX_S),
                                    ("rms", dbfs < MIN_DBFS), ("clipped", clip > MAX_CLIP),
                                    ("silence", sil > MAX_SIL)) if bad]
            zc.append_row(d / "qc.csv", zc.QC_COLUMNS, {
                "file": r["file"], "qc_cer": f"{c:.4f}", "asr_text": simp(hyp) if simp else hyp,
                "rms_dbfs": f"{dbfs:.2f}", "clip_frac": f"{clip:.5f}", "silence_frac": f"{sil:.3f}",
                "kept": str(not why), "drop_reason": "|".join(why)})
    return len(todo)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def qc(families: list[str], wait_pids: list[int], poll_s: float = 60.0) -> None:
    asr, simp = ASR(), simplifier()
    while True:
        running = [p for p in wait_pids if alive(p)]
        for f in families:
            try:
                n = qc_pass(asr, simp, f)
                tot, kept = finalize(f)
            except Exception as e:  # never let one bad pass end the follower
                zc.log(f"qc {f}: pass failed: {type(e).__name__}: {e}")
                continue
            if n:
                zc.log(f"qc {f}: +{n} checked, metadata {tot} rows, kept {kept} ({kept/max(tot,1):.1%})")
        if not running:
            break
        time.sleep(poll_s)
    zc.log("qc done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["transcribe", "qc", "finalize"])
    ap.add_argument("--families", default="")
    ap.add_argument("--wait-pids", default="")
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    if a.cmd == "transcribe":
        transcribe()
    elif a.cmd == "qc":
        qc(fams, [int(p) for p in a.wait_pids.split(",") if p])
    else:
        for f in fams:
            print(f, finalize(f))


if __name__ == "__main__":
    main()
