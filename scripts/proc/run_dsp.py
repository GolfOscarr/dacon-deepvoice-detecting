"""proc-dsp: one voice-changer op per file (pitch / formant / tempo), CPU.
venv: /data/project/private/dacon-venvs/proc-dsp
  python scripts/proc/run_dsp.py --workers 14 [--limit N] [--shard i/n]
"""
from __future__ import annotations
import argparse, os, sys, traceback
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import common as C

FAMILY = "proc-dsp"
LIC = ("praat-parselmouth GPL-3.0 (Praat GPL-3.0); librosa ISC; "
       "op code ours; output audio is not covered by the tool licences")


def versions() -> str:
    import librosa, parselmouth
    return f"parselmouth {parselmouth.__version__} (Praat {parselmouth.PRAAT_VERSION}); librosa {librosa.__version__}"


def draw(seed: int) -> dict:
    r = np.random.default_rng(seed)
    op = r.choice(["pitch", "formant", "stretch"], p=[0.4, 0.3, 0.3])
    if op == "pitch":
        st = float(r.uniform(1, 4)) * (1 if r.random() < 0.5 else -1)
        return {"op": "pitch", "semitones": round(st, 3),
                "method": str(r.choice(["praat_psola", "librosa_pv"]))}
    if op == "formant":
        ratio = float(r.uniform(0.85, 0.95) if r.random() < 0.5 else r.uniform(1.05, 1.2))
        return {"op": "formant", "formant_ratio": round(ratio, 4), "method": "praat_change_gender"}
    rate = float(r.uniform(0.9, 0.98) if r.random() < 0.5 else r.uniform(1.02, 1.1))
    return {"op": "stretch", "rate": round(rate, 4),
            "method": str(r.choice(["praat_lengthen_ola", "librosa_pv"]))}


def apply(x: np.ndarray, sr: int, p: dict) -> tuple[np.ndarray, dict]:
    import librosa, parselmouth
    from parselmouth.praat import call
    extra = {}
    if p["method"] == "librosa_pv":
        if p["op"] == "pitch":
            return librosa.effects.pitch_shift(x, sr=sr, n_steps=p["semitones"]), extra
        return librosa.effects.time_stretch(x, rate=p["rate"]), extra
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    if p["op"] == "stretch":
        out = call(snd, "Lengthen (overlap-add)", 75, 600, 1.0 / p["rate"])
    elif p["op"] == "formant":
        out = call(snd, "Change gender", 75, 600, p["formant_ratio"], 0, 1, 1)
    else:
        med = call(snd.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
        if not np.isfinite(med) or med <= 0:        # no voicing found: fall back
            extra["fallback"] = "librosa_pv (praat found no pitch)"
            return librosa.effects.pitch_shift(x, sr=sr, n_steps=p["semitones"]), extra
        extra["src_f0_median_hz"] = round(float(med), 2)
        out = call(snd, "Change gender", 75, 600, 1.0, med * 2 ** (p["semitones"] / 12), 1, 1)
    return out.values[0].astype(np.float32), extra


def work(row):
    try:
        x, sr = C.load(row.path)
        if sr < 16000:
            x, sr = C.load(row.path, 16000)
        p = draw(int(row.seed))
        y, extra = apply(x, sr, p)
        rec = C.write(row, y, sr, transform=f"{p['op']}:{p['method']}", params={**p, **extra},
                      tool="dsp", tool_version=VER, licence=LIC)
        return rec["duration_s"], None
    except Exception:
        return 0.0, f"{row.file_id}: {traceback.format_exc(limit=3)}"


VER = versions()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    d = C.plan(FAMILY, a.shard, a.limit)
    print(f"[{FAMILY}] todo={len(d)} audio_h={d.duration_s.sum()/3600:.2f} {VER}", flush=True)
    sp, errs = C.Speed(FAMILY), 0
    with Pool(a.workers) as pool:
        for dur, err in pool.imap_unordered(work, [r for _, r in d.iterrows()], chunksize=4):
            if err: errs += 1; print("ERR", err, flush=True)
            else: sp.add(dur)
    sp.report(); print(f"[{FAMILY}] done errors={errs}", flush=True)
